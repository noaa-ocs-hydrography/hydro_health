"""Class engine for subtiling raster data into geoparquet files"""

import re
import os
import gc
import shutil
import pathlib
from pathlib import Path

import s3fs
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.warp import reproject, transform, transform_bounds
import dask
import pyarrow.parquet as pq
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs' 


TERRAIN_PRODUCT_ALIASES = {
    "bathy": "bathy",
    "bpi_broad": "bpi_broad",
    "broad_bpi": "bpi_broad",
    "bpi_fine": "bpi_fine",
    "fine_bpi": "bpi_fine",
    "curv_plan": "curv_plan",
    "plan_curvature": "curv_plan",
    "curv_profile": "curv_profile",
    "profile_curvature": "curv_profile",
    "curv_total": "curv_total",
    "total_curvature": "curv_total",
    "flowacc": "flowacc",
    "flow_acc": "flowacc",
    "flowdir": "flowdir",
    "flow_dir": "flowdir",
    "gradmag": "gradmag",
    "gradient_magnitude": "gradmag",
    "rugosity": "rugosity",
    "shearproxy": "shearproxy",
    "shear_proxy": "shearproxy",
    "slope_deg": "slope_deg",
    "slope_degrees": "slope_deg",
    "slope": "slope",
    "tci": "tci",
    "terrain_classification": "terrain_classification",
    "uc": "uc",
    "unc": "uc",
    "uncertainty": "uc",
}

PREDICTION_BT_VARIABLES = [
    "bathy",
    "bpi_broad",
    "bpi_fine",
    "curv_plan",
    "curv_profile",
    "curv_total",
    "flowacc",
    "flowdir",
    "gradmag",
    "rugosity",
    "shearproxy",
    "slope",
    "slope_deg",
    "tci",
    "terrain_classification",
    "uc",
]

TRAINING_TERRAIN_VARIABLES = [
    "bpi_broad",
    "bpi_fine",
    "curv_plan",
    "curv_profile",
    "curv_total",
    "flowacc",
    "flowdir",
    "gradmag",
    "rugosity",
    "shearproxy",
    "slope",
    "slope_deg",
    "tci",
    "terrain_classification",
]

HURRICANE_VARIABLES = [
    "hurr_strength_mean_2004_2006",
    "hurr_strength_mean_2006_2010",
    "hurr_strength_mean_2010_2015",
    "hurr_strength_mean_2015_2022",
]

TSM_VARIABLES = [
    "tsm_mean_2004_2006",
    "tsm_mean_2006_2010",
    "tsm_mean_2010_2015",
    "tsm_mean_2015_2022",
]

STATIC_VARIABLES = [
    "grain_size_layer",
    *HURRICANE_VARIABLES,
    "prim_sed_layer",
    "survey_end_date",
    *TSM_VARIABLES,
]


def _extract_year_pair(name: str) -> tuple[int, int] | None:
    """Return the trailing YYYY_YYYY pair from a standardized column name."""
    match = re.search(r"((?:19|20)\d{2})_((?:19|20)\d{2})$", name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _available_training_bathy_years(df: pd.DataFrame) -> list[int]:
    """Return sorted training bathymetry years containing at least one value."""
    if df is None or df.empty:
        return []

    years = set()
    for column in df.columns:
        match = re.fullmatch(r"bathy_((?:19|20)\d{2})_filled", column)
        if match and df[column].notna().any():
            years.add(int(match.group(1)))
    return sorted(years)


def _available_training_year_pairs(
    bathy_years: list[int], year_ranges: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Return configured year pairs whose two bathymetry years have data."""
    available_years = {int(year) for year in bathy_years}
    available_pairs = []
    seen_pairs = set()

    for year_range in year_ranges:
        if len(year_range) != 2:
            continue
        pair = (int(year_range[0]), int(year_range[1]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        if pair[0] in available_years and pair[1] in available_years:
            available_pairs.append(pair)

    return available_pairs


def _standardize_bluetopo_col_name(
    col_name: str, original_tile: str = ""
) -> str:
    """Normalize any BlueTopo raster to the required yearless bt.* schema."""
    clean_name = re.sub(r"(?i)^bluetopo_?", "", col_name)
    clean_name = re.sub(r"(?i)^bt[._]?", "", clean_name)

    if original_tile:
        clean_name = re.sub(
            rf"(?i)(?:^|_){re.escape(original_tile)}(?=_|$)",
            "_",
            clean_name,
        )

    # Dates identify the BlueTopo vintage but are intentionally not part of
    # prediction feature names.
    clean_name = re.sub(r"(?<!\d)(?:19|20)\d{6}(?!\d)", "_", clean_name)
    clean_name = re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", "_", clean_name)
    clean_name = re.sub(r"(?i)(?:^|_)filled(?=_|$)", "_", clean_name)
    clean_name = re.sub(r"[^A-Za-z0-9]+", "_", clean_name).strip("_").lower()

    # Match longest aliases first so slope_deg is not mistaken for slope.
    for alias in sorted(TERRAIN_PRODUCT_ALIASES, key=len, reverse=True):
        if re.search(rf"(?:^|_){re.escape(alias)}$", clean_name):
            return f"bt.{TERRAIN_PRODUCT_ALIASES[alias]}"

    # The un-suffixed BlueTopo surface is bathymetry.
    return "bt.bathy"


def _standardize_combined_col_name(
    col_name: str, original_tile: str = ""
) -> str:
    """Normalize long combined-LiDAR filenames into training column names."""
    clean_name = re.sub(r"(?i)^combined\d*_?", "", col_name).strip("_")

    if original_tile:
        clean_name = re.sub(
            rf"(?i)(?:^|_){re.escape(original_tile)}(?=_|$)",
            "_",
            clean_name,
        )

    # Survey dates may be encoded as YYYYMMDD. Training columns use only year.
    clean_name = re.sub(
        r"(?<!\d)((?:19|20)\d{2})\d{4}(?!\d)", r"\1", clean_name
    )
    clean_name = re.sub(r"_+", "_", clean_name).strip("_")
    lower_name = clean_name.lower()

    year_pair_match = re.search(r"(?<!\d)((?:19|20)\d{2})_((?:19|20)\d{2})(?!\d)", lower_name)
    if year_pair_match:
        year_token = f"{year_pair_match.group(1)}_{year_pair_match.group(2)}"
    else:
        year_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", lower_name)
        year_token = year_match.group(1) if year_match else ""

    # Terrain products are appended after "filled" by TerrainProductsEngine.
    # Keeping only that suffix removes survey/provider/source identifiers.
    product_match = re.search(r"(?:^|_)filled_(.+)$", lower_name)
    product_name = product_match.group(1) if product_match else ""

    if product_name:
        product_name = re.sub(r"[^a-z0-9]+", "_", product_name).strip("_")
        product_name = TERRAIN_PRODUCT_ALIASES.get(product_name, product_name)
    else:
        # Support derived files that do not contain the optional "filled"
        # marker, while preferring longer suffixes before short ones.
        for suffix in sorted(TERRAIN_PRODUCT_ALIASES, key=len, reverse=True):
            if re.search(rf"(?:^|_){suffix}$", lower_name):
                product_name = TERRAIN_PRODUCT_ALIASES[suffix]
                break

    if product_name:
        return f"{product_name}_{year_token}" if year_token else product_name

    # A combined raster with no terrain-product suffix is the filled bathymetry
    # surface used to calculate training deltas.
    if year_token:
        return f"bathy_{year_token}_filled"

    return lower_name


def _standardize_col_name(col_name: str, original_tile: str = "") -> str:
    """Cleans raster filenames into consistent column names, standardizing years and prefixes."""
    
    clean_name = col_name

    if "survey_end_date" in clean_name.lower():
        return "survey_end_date"

    if re.match(r"(?i)^combined\d*_?", clean_name):
        return _standardize_combined_col_name(clean_name, original_tile)

    is_bluetopo = "bluetopo" in clean_name.lower() or bool(
        re.match(r"(?i)^bt[._]", clean_name)
    )
    if is_bluetopo:
        return _standardize_bluetopo_col_name(clean_name, original_tile)

    clean_name = re.sub(r"(?i)^combined\d*_", "", clean_name)

    if original_tile and original_tile in clean_name:
        clean_name = clean_name.replace(f"_{original_tile}", "").replace(f"{original_tile}_", "").replace(original_tile, "")

    clean_name = re.sub(r"(?<!\d)((?:19|20)\d{2})\d{4}(?!\d)", r"\1", clean_name)
    clean_name = clean_name.strip("_")

    m_pair = re.search(r"(\d{4}_\d{4})", clean_name)
    if m_pair:
        year_pair = m_pair.group(1)
        base = clean_name.replace(year_pair, "").strip("_")
        base = re.sub(r"__+", "_", base)
        final_name = f"{base}_{year_pair}" if base else year_pair
    else:
        m_single = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", clean_name)
        if m_single:
            year = m_single.group(1)
            base = clean_name.replace(year, "").strip("_")
            base = re.sub(r"__+", "_", base)

            base_lower = base.lower()
            if base_lower in ["bathy", "bathy_filled"]:
                final_name = f"bathy_{year}_filled"
            elif base_lower.startswith("bathy_"):
                base = base[6:].strip("_")
                final_name = f"{base}_{year}"
            else:
                final_name = f"{base}_{year}" if base else year
        else:
            final_name = clean_name

    return final_name


def _create_nan_stats_csv(df: pd.DataFrame, tile_id: str) -> pd.DataFrame:
    """Calculates NaN stats for a tile."""
    
    if df.empty:
        return pd.DataFrame()
    
    new_row = {'tile_id': tile_id}
    change_cols = [c for c in df.columns if c.startswith('delta_bathy_')]
    for col in change_cols:
        year_pair = col.replace('delta_bathy_', '')
        new_row[f"{year_pair}_nan_percent"] = round(df[col].isna().mean() * 100, 2)

    return pd.DataFrame([new_row])


def _read_existing_nan_stats(path: UPath, tile_id: str, is_aws: bool) -> pd.DataFrame:
    """Read only delta columns from an existing Parquet output."""

    def _read_from_parquet_file(parquet_file) -> pd.DataFrame:
        if parquet_file.metadata.num_rows == 0:
            return pd.DataFrame()
        delta_cols = [
            name
            for name in parquet_file.schema_arrow.names
            if name.startswith("delta_bathy_")
        ]
        if not delta_cols:
            return pd.DataFrame([{"tile_id": tile_id}])
        delta_df = parquet_file.read(columns=delta_cols).to_pandas()
        return _create_nan_stats_csv(delta_df, tile_id)

    if is_aws and str(path).startswith("s3://"):
        fs = s3fs.S3FileSystem()
        with fs.open(str(path), "rb") as src:
            return _read_from_parquet_file(pq.ParquetFile(src))
    return _read_from_parquet_file(pq.ParquetFile(str(path)))


def _create_multiband_geotiff_from_parquet(
    parquet_path: str,
    raster_crs,
    is_aws: bool,
    local_tmp_dir: str,
    created_sequence: int,
    overwrite_outputs: bool,
) -> str | None:
    """Create a georeferenced float32 multiband GeoTIFF from one Parquet.

    Coordinate and identifier fields are not written as bands. Every remaining
    numeric column is written in the same order in which it occurs in the
    Parquet file. Only one band-sized NumPy array is allocated at a time.
    """
    tmp_tif_path = None
    try:
        parquet_stem = Path(parquet_path).stem
        geotiff_name = f"{parquet_stem}_all_variables.tif"
        final_tif_path = str(UPath(parquet_path).parent / geotiff_name)

        if UPath(final_tif_path).exists():
            if overwrite_outputs:
                Engine.write_message_dask(
                    f" [GEOTIFF OVERWRITE] Existing GeoTIFF will be replaced: "
                    f"{final_tif_path}",
                    OUTPUTS,
                )
            else:
                Engine.write_message_dask(
                    f" [GEOTIFF EXISTS] Keeping existing GeoTIFF: "
                    f"{final_tif_path}",
                    OUTPUTS,
                )
                return final_tif_path

        if raster_crs is None:
            Engine.write_message_dask(
                f" [GEOTIFF SKIP] Cannot create GeoTIFF for {parquet_path}: "
                "the source grid CRS is unavailable.",
                OUTPUTS,
            )
            return None

        if is_aws and parquet_path.startswith("s3://"):
            fs = s3fs.S3FileSystem()
            with fs.open(parquet_path, "rb") as parquet_file:
                df = pd.read_parquet(parquet_file, engine="pyarrow")
        else:
            df = pd.read_parquet(parquet_path, engine="pyarrow")

        if df.empty or "X" not in df.columns or "Y" not in df.columns:
            Engine.write_message_dask(
                f" [GEOTIFF SKIP] Cannot create GeoTIFF for {parquet_path}: "
                "the Parquet is empty or lacks X/Y columns.",
                OUTPUTS,
            )
            return None

        excluded_columns = {"FID", "X", "Y", "tile_id"}
        band_columns = [
            column
            for column in df.columns
            if column not in excluded_columns
            and pd.api.types.is_numeric_dtype(df[column])
        ]
        if not band_columns:
            Engine.write_message_dask(
                f" [GEOTIFF SKIP] No numeric variable columns found in "
                f"{parquet_path}.",
                OUTPUTS,
            )
            return None

        finite_xy = np.isfinite(df["X"].to_numpy()) & np.isfinite(
            df["Y"].to_numpy()
        )
        if not finite_xy.all():
            df = df.loc[finite_xy].copy()
        if df.empty:
            Engine.write_message_dask(
                f" [GEOTIFF SKIP] No finite coordinates found in {parquet_path}.",
                OUTPUTS,
            )
            return None

        x_values = np.sort(df["X"].unique())
        y_values = np.sort(df["Y"].unique())[::-1]
        if len(x_values) < 2 or len(y_values) < 2:
            Engine.write_message_dask(
                f" [GEOTIFF SKIP] Cannot infer pixel size for {parquet_path}; "
                "at least two unique X and Y coordinates are required.",
                OUTPUTS,
            )
            return None

        x_diffs = np.diff(x_values)
        y_diffs = np.abs(np.diff(y_values))
        x_resolution = float(np.median(x_diffs))
        y_resolution = float(np.median(y_diffs))
        tolerance = max(x_resolution, y_resolution) * 1e-5

        if (
            x_resolution <= 0
            or y_resolution <= 0
            or not np.allclose(x_diffs, x_resolution, rtol=1e-5, atol=tolerance)
            or not np.allclose(y_diffs, y_resolution, rtol=1e-5, atol=tolerance)
        ):
            Engine.write_message_dask(
                f" [GEOTIFF SKIP] Coordinates in {parquet_path} do not form "
                "a regular raster grid.",
                OUTPUTS,
            )
            return None

        width = len(x_values)
        height = len(y_values)
        transform_out = rasterio.transform.from_origin(
            float(x_values[0] - x_resolution / 2),
            float(y_values[0] + y_resolution / 2),
            x_resolution,
            y_resolution,
        )

        x_coords = df["X"].to_numpy()
        y_coords = df["Y"].to_numpy()
        col_indices = np.rint(
            (x_coords - x_values[0]) / x_resolution
        ).astype(np.int64)
        row_indices = np.rint(
            (y_values[0] - y_coords) / y_resolution
        ).astype(np.int64)

        indices_valid = (
            (row_indices >= 0)
            & (row_indices < height)
            & (col_indices >= 0)
            & (col_indices < width)
        )
        if not indices_valid.all():
            Engine.write_message_dask(
                f" [GEOTIFF SKIP] Some coordinates in {parquet_path} fall "
                "outside the reconstructed grid.",
                OUTPUTS,
            )
            return None

        tmp_tif_path = str(Path(local_tmp_dir) / geotiff_name)

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": len(band_columns),
            "dtype": "float32",
            "crs": CRS.from_user_input(raster_crs),
            "transform": transform_out,
            "nodata": np.nan,
            "compress": "LZW",
            "predictor": 3,
            "BIGTIFF": "IF_SAFER",
        }

        with rasterio.open(tmp_tif_path, "w", **profile) as dst:
            dst.update_tags(
                source_parquet=parquet_path,
                created_parquet_sequence=created_sequence,
            )
            for band_index, column in enumerate(band_columns, start=1):
                band_data = np.full((height, width), np.nan, dtype=np.float32)
                values = pd.to_numeric(df[column], errors="coerce").to_numpy(
                    dtype=np.float32
                )
                band_data[row_indices, col_indices] = values
                dst.write(band_data, band_index)
                dst.set_band_description(band_index, str(column))
                del band_data, values

        if is_aws and final_tif_path.startswith("s3://"):
            s3fs.S3FileSystem().put(tmp_tif_path, final_tif_path)
        else:
            shutil.copy2(tmp_tif_path, final_tif_path)

        Engine.write_message_dask(
            f" [GEOTIFF SUCCESS] Created sampled multiband GeoTIFF "
            f"#{created_sequence}: {final_tif_path}",
            OUTPUTS,
        )
        return final_tif_path
    except Exception as e:
        Engine.write_message_dask(
            f" [GEOTIFF ERROR] Failed to create a multiband GeoTIFF from "
            f"{parquet_path}: {e}",
            OUTPUTS,
        )
        return None
    finally:
        if tmp_tif_path and Path(tmp_tif_path).exists():
            os.remove(tmp_tif_path)
        if "df" in locals():
            del df
        gc.collect()


def _to_rasterio_path(file: str, is_aws: bool) -> str:
    """Return a path Rasterio/GDAL can open directly."""
    open_path = str(file)
    if is_aws and open_path.startswith("s3://"):
        return open_path.replace("s3://", "/vsis3/", 1)
    return open_path


def _bounds_in_crs(bounds: tuple, source_crs, destination_crs) -> tuple:
    """Transform bounds only when both CRSs exist and differ."""
    if source_crs is None or destination_crs is None:
        return bounds
    source_crs = CRS.from_user_input(source_crs)
    destination_crs = CRS.from_user_input(destination_crs)
    if source_crs == destination_crs:
        return bounds
    return transform_bounds(source_crs, destination_crs, *bounds, densify_pts=21)


def _requested_window(src, bounds: tuple):
    """Return the full pixel-aligned window requested by geographic bounds."""
    requested = src.window(*bounds).round_offsets().round_lengths()
    if requested.width <= 0 or requested.height <= 0:
        return None
    return requested


def _clipped_window(src, bounds: tuple):
    """Return a pixel-aligned window clipped to a raster, or None when disjoint."""
    if disjoint_bounds(bounds, src.bounds):
        return None

    requested = _requested_window(src, bounds)
    if requested is None:
        return None
    full = Window(0, 0, src.width, src.height)
    try:
        clipped = requested.intersection(full)
    except rasterio.errors.WindowError:
        return None

    clipped = clipped.round_offsets().round_lengths()
    if clipped.width <= 0 or clipped.height <= 0:
        return None
    return clipped


def _valid_mask(data: np.ndarray, nodata) -> np.ndarray:
    """Build a validity mask that handles numeric, NaN, and absent NoData values."""
    if np.issubdtype(data.dtype, np.floating):
        valid = np.isfinite(data)
    else:
        valid = np.ones(data.shape, dtype=bool)

    if nodata is not None:
        try:
            if not np.isnan(nodata):
                valid &= data != nodata
        except TypeError:
            valid &= data != nodata
    return valid


def _crs_equivalent(left_crs, right_crs) -> bool:
    """Compare CRSs by full equality, then by canonical EPSG identifier."""
    if left_crs is None or right_crs is None:
        return left_crs is None and right_crs is None
    if left_crs == right_crs:
        return True

    try:
        left_epsg = left_crs.to_epsg()
        right_epsg = right_crs.to_epsg()
    except Exception:
        return False

    return (
        left_epsg is not None
        and right_epsg is not None
        and left_epsg == right_epsg
    )


def _transform_alignment_details(
    current_transform, reference_transform
) -> tuple[bool, float, float, bool]:
    """Compare affine transforms, allowing at most a 0.125-pixel offset.

    Terrain tools sometimes round an otherwise matching raster origin by less
    than one pixel. Rasters within one eighth of a pixel are treated as using
    the same row/column grid. The final return value records whether this
    relaxed allowance, rather than the strict numeric tolerance, was needed.
    """
    coefficient_names = ("a", "b", "c", "d", "e", "f")
    max_difference = max(
        abs(
            float(getattr(current_transform, coefficient))
            - float(getattr(reference_transform, coefficient))
        )
        for coefficient in coefficient_names
    )

    pixel_width = float(
        np.hypot(reference_transform.a, reference_transform.d)
    )
    pixel_height = float(
        np.hypot(reference_transform.b, reference_transform.e)
    )
    pixel_size = max(pixel_width, pixel_height)
    strict_tolerance = max(pixel_size * 0.001, 1e-9)
    tolerance = max(pixel_size * 0.125, 1e-9)
    matches = max_difference <= tolerance
    used_subpixel_tolerance = matches and max_difference > strict_tolerance
    return matches, max_difference, tolerance, used_subpixel_tolerance


def _build_spatial_catalog(raster_files: list, is_aws: bool) -> list:
    """Read raster metadata once so every tile does not reopen every S3 file."""
    catalog = []
    for file in raster_files:
        try:
            with rasterio.open(_to_rasterio_path(file, is_aws)) as src:
                if src.crs is None:
                    Engine.write_message_dask(
                        f"WARNING: Skipping raster without a CRS: {file}", OUTPUTS
                    )
                    continue
                raster_crs = src.crs.to_wkt()
                catalog.append((file, tuple(src.bounds), raster_crs))
        except Exception as e:
            Engine.write_message_dask(
                f"WARNING: Could not catalog raster {file}: {e}", OUTPUTS
            )
    return catalog


def _files_intersecting_tile(
    tile_bounds: tuple, subgrid_crs, raster_catalog: list
) -> list:
    """Select catalogued rasters whose bounds intersect a tile."""
    intersecting = []
    for file, raster_bounds, raster_crs in raster_catalog:
        try:
            bounds = _bounds_in_crs(tile_bounds, subgrid_crs, raster_crs)
            if not disjoint_bounds(bounds, raster_bounds):
                intersecting.append(file)
        except Exception as e:
            Engine.write_message_dask(
                f"WARNING: Could not compare tile bounds with {file}: {e}", OUTPUTS
            )
    return intersecting


def _subtile_process_gridded(
    sub_grid: pd.Series,
    raster_files: list,
    is_aws: bool,
    subgrid_crs,
) -> tuple[pd.DataFrame, object]:
    """Read one tile's gridded rasters sequentially inside its worker.

    The arrays never return to the Dask client. The first intersecting raster
    defines a full-tile destination grid. Rasters with partial coverage or a
    shifted origin are mapped into that grid and uncovered cells remain NaN.
    """
    original_tile = str(sub_grid['original_tile'])
    pattern = re.compile(rf"(?:^|_){re.escape(original_tile)}(?:_|\.)", re.IGNORECASE)

    filtered_files = [f for f in raster_files if pattern.search(Path(f).name)]

    if not filtered_files:
        return pd.DataFrame(), None

    tile_extent = sub_grid.geometry.bounds
    flattened_arrays = {}
    master_valid = None
    common_transform = None
    common_shape = None
    common_crs = None

    for file in filtered_files:
        try:
            with rasterio.open(_to_rasterio_path(file, is_aws)) as src:
                if src.crs is None:
                    Engine.write_message_dask(
                        f"WARNING: Skipping gridded raster without a CRS: {file}",
                        OUTPUTS,
                    )
                    continue
                raster_bounds = _bounds_in_crs(tile_extent, subgrid_crs, src.crs)
                requested_window = _requested_window(src, raster_bounds)
                window = _clipped_window(src, raster_bounds)
                if requested_window is None or window is None:
                    continue

                current_transform = src.window_transform(window)
                current_shape = (int(window.height), int(window.width))

                if common_transform is None:
                    common_transform = src.window_transform(requested_window)
                    common_shape = (
                        int(requested_window.height),
                        int(requested_window.width),
                    )
                    common_crs = src.crs

                if not _crs_equivalent(src.crs, common_crs):
                    Engine.write_message_dask(
                        "WARNING: Skipping gridded raster with a different CRS "
                        f"{file}: expected={common_crs!r}, got={src.crs!r}, "
                        f"expected_epsg={common_crs.to_epsg()}, "
                        f"got_epsg={src.crs.to_epsg()}.",
                        OUTPUTS,
                    )
                    continue

                (
                    transform_matches,
                    _,
                    _,
                    _,
                ) = _transform_alignment_details(
                    current_transform, common_transform
                )
                shape_matches = current_shape == common_shape

                data = src.read(1, window=window)
                valid = _valid_mask(data, src.nodata)
                values = data.astype(np.float32, copy=False)
                values[~valid] = np.nan

                if not shape_matches or not transform_matches:
                    aligned_values = np.full(common_shape, np.nan, dtype=np.float32)
                    reproject(
                        source=values,
                        destination=aligned_values,
                        src_transform=current_transform,
                        src_crs=src.crs,
                        src_nodata=np.nan,
                        dst_transform=common_transform,
                        dst_crs=common_crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.nearest,
                        init_dest_nodata=True,
                        num_threads=1,
                    )
                    del values, valid
                    values = aligned_values
                    valid = np.isfinite(values)

                col_name = _standardize_col_name(Path(file).stem, original_tile)
                values_flat = values.reshape(-1)
                valid_flat = valid.reshape(-1)
                if col_name in flattened_arrays:
                    existing = flattened_arrays[col_name]
                    fill_mask = np.isnan(existing) & valid_flat
                    existing[fill_mask] = values_flat[fill_mask]
                else:
                    flattened_arrays[col_name] = values_flat
                if master_valid is None:
                    master_valid = valid_flat.copy()
                else:
                    master_valid |= valid_flat

                del data, values, values_flat, valid, valid_flat
        except Exception as e:
            Engine.write_message_dask(
                f"WARNING: Error reading gridded file {file}: {e}", OUTPUTS
            )

    if not flattened_arrays or master_valid is None or not master_valid.any():
        return pd.DataFrame(), common_crs

    mask_2d = master_valid.reshape(common_shape)
    rows, cols = np.where(mask_2d)
    xs, ys = rasterio.transform.xy(common_transform, rows, cols, offset='center')

    df_dict = {
        'X': np.round(np.asarray(xs), 3),
        'Y': np.round(np.asarray(ys), 3),
    }

    for col_name, values in flattened_arrays.items():
        df_dict[col_name] = values[master_valid]

    combined_data = pd.DataFrame(df_dict).drop_duplicates(subset=['X', 'Y'])
    del flattened_arrays, master_valid, mask_2d
    return combined_data, common_crs


def _subtile_process_ungridded(
    sub_grid: pd.Series,
    raster_files: list,
    gridded_df: pd.DataFrame,
    static_patterns: list,
    is_aws: bool,
    subgrid_crs,
    point_crs,
    data_type: str,
    training_year_pairs: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """Sample intersecting static rasters sequentially inside the tile worker."""
    
    if gridded_df is None or gridded_df.empty:
        return pd.DataFrame()

    combined_df = gridded_df
    xs = combined_df['X'].to_numpy(copy=False)
    ys = combined_df['Y'].to_numpy(copy=False)
    tile_extent = sub_grid.geometry.bounds
    original_tile = str(sub_grid.get('original_tile', ''))
    allowed_training_year_pairs = set(training_year_pairs or [])

    candidate_files = []
    for pattern in static_patterns:
        candidate_files.extend(
            f for f in raster_files if pattern in Path(f).name.lower()
        )

    # Preserve order while removing duplicates caused by overlapping patterns.
    candidate_files = list(dict.fromkeys(candidate_files))

    for file in candidate_files:
        col_name = _standardize_col_name(Path(file).stem, original_tile)
        bathy_pair_mask = None
        if data_type == "training" and (
            col_name.startswith("hurr_") or col_name.startswith("tsm_")
        ):
            year_pair = _extract_year_pair(col_name)
            if year_pair is None or year_pair not in allowed_training_year_pairs:
                continue

            bathy_pair_mask = (
                combined_df[f"bathy_{year_pair[0]}_filled"].notna().to_numpy()
                & combined_df[f"bathy_{year_pair[1]}_filled"].notna().to_numpy()
            )
            if not bathy_pair_mask.any():
                continue

        try:
            with rasterio.open(_to_rasterio_path(file, is_aws)) as src:
                raster_bounds = _bounds_in_crs(tile_extent, subgrid_crs, src.crs)
                window = _clipped_window(src, raster_bounds)
                if window is None:
                    continue

                sample_xs = xs
                sample_ys = ys
                if point_crs is not None and src.crs is not None:
                    source = CRS.from_user_input(point_crs)
                    destination = CRS.from_user_input(src.crs)
                    if source != destination:
                        tx, ty = transform(source, destination, xs, ys)
                        sample_xs = np.asarray(tx)
                        sample_ys = np.asarray(ty)

                win_data = src.read(1, window=window)
                win_transform = src.window_transform(window)
                win_rows, win_cols = rasterio.transform.rowcol(
                    win_transform, sample_xs, sample_ys
                )
                win_rows = np.asarray(win_rows)
                win_cols = np.asarray(win_cols)

                in_window = (
                    (win_rows >= 0)
                    & (win_rows < win_data.shape[0])
                    & (win_cols >= 0)
                    & (win_cols < win_data.shape[1])
                )
                if bathy_pair_mask is not None:
                    in_window &= bathy_pair_mask
                vals = np.full(len(xs), np.nan, dtype=np.float32)

                if in_window.any():
                    extracted = win_data[
                        win_rows[in_window], win_cols[in_window]
                    ]
                    extracted_valid = _valid_mask(extracted, src.nodata)
                    extracted = extracted.astype(np.float32, copy=False)
                    extracted[~extracted_valid] = np.nan
                    vals[in_window] = extracted

                if col_name in combined_df:
                    existing = combined_df[col_name].to_numpy(copy=False)
                    fill_mask = pd.isna(existing) & ~np.isnan(vals)
                    if fill_mask.any():
                        existing[fill_mask] = vals[fill_mask]
                else:
                    combined_df[col_name] = vals

                del win_data, vals, win_rows, win_cols, in_window
        except Exception as e:
            Engine.write_message_dask(
                f"WARNING: Failed to sample ungridded raster {file}: {e}",
                OUTPUTS,
            )

    return combined_df


def _conform_output_schema(
    combined_df: pd.DataFrame,
    data_type: str,
    bathy_years: list[int],
    training_year_pairs: list[tuple[int, int]],
    tile_id: str,
) -> pd.DataFrame:
    """Add missing expected fields, remove unexpected fields, and order columns."""
    if data_type == "prediction":
        variable_columns = [
            f"bt.{variable}" for variable in PREDICTION_BT_VARIABLES
        ] + STATIC_VARIABLES
    elif data_type == "training":
        valid_pair_set = set(training_year_pairs)
        applicable_hurricane_variables = [
            column
            for column in HURRICANE_VARIABLES
            if _extract_year_pair(column) in valid_pair_set
        ]
        applicable_tsm_variables = [
            column
            for column in TSM_VARIABLES
            if _extract_year_pair(column) in valid_pair_set
        ]
        variable_columns = [f"bathy_{year}_filled" for year in bathy_years]

        for variable in TRAINING_TERRAIN_VARIABLES[:8]:
            variable_columns.extend(
                f"{variable}_{year}" for year in bathy_years
            )

        variable_columns.extend(
            [
                "grain_size_layer",
                *applicable_hurricane_variables,
                "prim_sed_layer",
            ]
        )

        for variable in TRAINING_TERRAIN_VARIABLES[8:12]:
            variable_columns.extend(
                f"{variable}_{year}" for year in bathy_years
            )

        variable_columns.append("survey_end_date")

        for variable in TRAINING_TERRAIN_VARIABLES[12:]:
            variable_columns.extend(
                f"{variable}_{year}" for year in bathy_years
            )

        variable_columns.extend(applicable_tsm_variables)
        variable_columns.extend(
            f"delta_bathy_{year_start}_{year_end}"
            for year_start, year_end in training_year_pairs
        )
    else:
        return combined_df

    expected_columns = ["X", "Y", *variable_columns]
    missing_columns = [
        column for column in expected_columns if column not in combined_df.columns
    ]
    for column in missing_columns:
        combined_df[column] = np.full(
            len(combined_df), np.nan, dtype=np.float32
        )

    protected_columns = set(expected_columns) | {"FID", "tile_id"}
    unexpected_columns = [
        column
        for column in combined_df.columns
        if column not in protected_columns
    ]
    if unexpected_columns:
        Engine.write_message_dask(
            f"WARNING: Tile {tile_id}: Excluding unexpected {data_type} "
            f"columns: {unexpected_columns}",
            OUTPUTS,
        )
        combined_df.drop(columns=unexpected_columns, inplace=True)

    if missing_columns:
        Engine.write_message_dask(
            f"WARNING: Tile {tile_id}: Added missing {data_type} columns as "
            f"NaN: {missing_columns}",
            OUTPUTS,
        )

    return combined_df.loc[:, expected_columns].copy()


def _save_combined_data(combined_df: pd.DataFrame, output_folder: str, data_type: str, tile_id: str, is_aws: bool, local_tmp_dir: str, current_index: int, total_count: int, verbose: bool, training_year_pairs: list[tuple[int, int]] | None = None) -> pd.DataFrame:
    """Combine dataframes, explicitly save to temporary disk, move to output, and aggressively drop memory."""
    
    if combined_df is None or combined_df.empty:
        Engine.write_message_dask(
            f" [{current_index}/{total_count}] [SKIP NO DATA] Tile "
            f"'{tile_id}': No valid data assembled; no Parquet file created.",
            OUTPUTS,
        )
        return pd.DataFrame()

    sorted_years = []
    valid_training_year_pairs = []
    if data_type == "training":
        valid_training_year_pairs = list(training_year_pairs or [])
        sorted_years = sorted(
            {
                year
                for year_pair in valid_training_year_pairs
                for year in year_pair
            }
        )
        valid_pair_names = []

        for y0, y1 in valid_training_year_pairs:
            y0_str, y1_str = str(y0), str(y1)

            c_0 = [c for c in combined_df.columns if re.match(rf"^bathy_{y0_str}_filled$", c, re.IGNORECASE)]
            c_1 = [c for c in combined_df.columns if re.match(rf"^bathy_{y1_str}_filled$", c, re.IGNORECASE)]
            b_y0 = [c for c in c_0 if "filled" in c.lower()][0] if c_0 else None
            b_y1 = [c for c in c_1 if "filled" in c.lower()][0] if c_1 else None

            if b_y0 and b_y1:
                combined_df[f"delta_bathy_{y0_str}_{y1_str}"] = combined_df[b_y1] - combined_df[b_y0]
                valid_pair_names.append(f"{y0_str}_{y1_str}")
            else:
                Engine.write_message_dask(f"WARNING: MISSING BATHY DATA: Cannot calculate delta_bathy for {y0_str}_{y1_str} on tile '{tile_id}'.", OUTPUTS)

        static_pair_columns = set(HURRICANE_VARIABLES + TSM_VARIABLES)
        cols_to_drop = [
            c
            for c in combined_df.columns
            if re.search(r"(\d{4}_\d{4})$", c)
            and not c.startswith("delta_bathy_")
            and c not in static_pair_columns
            and re.search(r"(\d{4}_\d{4})$", c).group(1)
            not in valid_pair_names
        ]
        if cols_to_drop:
            combined_df.drop(columns=cols_to_drop, inplace=True)

    combined_df = _conform_output_schema(
        combined_df,
        data_type,
        sorted_years,
        valid_training_year_pairs,
        tile_id,
    )

    if 'FID' not in combined_df.columns:
        combined_df['FID'] = np.arange(len(combined_df))
    if 'tile_id' not in combined_df.columns:
        combined_df['tile_id'] = tile_id

    leading_columns = [c for c in ('X', 'Y') if c in combined_df.columns]
    trailing_columns = [c for c in ('FID', 'tile_id') if c in combined_df.columns]
    variable_columns = [
        c
        for c in combined_df.columns
        if c not in set(leading_columns + trailing_columns)
    ]
    combined_df = combined_df[
        leading_columns + variable_columns + trailing_columns
    ]

    output_folder_path = UPath(output_folder)
    if not is_aws: 
        output_folder_path.mkdir(parents=True, exist_ok=True)

    final_save_path = str(output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet")
    tmp_dst_path = str(Path(local_tmp_dir) / f"{tile_id}_{data_type}_clipped_data.parquet")

    try:
        combined_df.to_parquet(tmp_dst_path, engine="pyarrow", index=False)

        if is_aws and final_save_path.startswith("s3://"):
            s3fs.S3FileSystem().put(tmp_dst_path, final_save_path)
        else:
            shutil.copy2(tmp_dst_path, final_save_path)

        output_columns = [str(column) for column in combined_df.columns]
        Engine.write_message_dask(
            f" [{current_index}/{total_count}] [SUCCESS] Created Parquet: "
            f"{final_save_path}",
            OUTPUTS,
        )
        Engine.write_message_dask(
            f" [{current_index}/{total_count}] [COLUMNS] {final_save_path} "
            f"({len(output_columns)} columns): {output_columns}",
            OUTPUTS,
        )

        stats_df = _create_nan_stats_csv(combined_df, tile_id)
    finally:
        if tmp_dst_path != final_save_path and Path(tmp_dst_path).exists():
            os.remove(tmp_dst_path)

    return stats_df


def _process_tile(sub_grid: pd.Series, gridded_files: list, ungridded_files: list, static_patterns: list, 
                  is_aws: bool, output_folder: str, data_type: str, tile_name: str, local_tmp_dir: str, 
                  current_index: int, total_count: int, verbose: bool, subgrid_crs,
                  overwrite_outputs: bool, year_ranges: list[tuple[int, int]]) -> pd.DataFrame:
    """Process and save one complete tile inside one Dask worker."""
    
    expected_path = UPath(output_folder) / f"{tile_name}_{data_type}_clipped_data.parquet"

    if verbose:
        Engine.write_message_dask(f"Processing tile {tile_name} ({current_index}/{total_count})...", OUTPUTS)

    # Report an existing output even if its original source raster is no longer
    # present in the current discovery results.
    try:
        if expected_path.exists():
            if overwrite_outputs:
                Engine.write_message_dask(
                    f" [{current_index}/{total_count}] [PARQUET OVERWRITE] "
                    f"Existing Parquet will be replaced: {expected_path}",
                    OUTPUTS,
                )
            else:
                Engine.write_message_dask(
                    f" [{current_index}/{total_count}] [EXISTS] Parquet already "
                    f"exists: {expected_path}. Compiling statistics from existing data.",
                    OUTPUTS,
                )
                return _read_existing_nan_stats(expected_path, tile_name, is_aws)
    except Exception as e:
        Engine.write_message_dask(
            f" [{current_index}/{total_count}] [ERROR CHECKING OUTPUT] "
            f"{expected_path}: {e}",
            OUTPUTS,
        )
        return pd.DataFrame()

    # Prediction must have bluetopo data. Training MUST have combined lidar data.
    has_bluetopo = any(
        "bluetopo" in Path(f).name.lower()
        or Path(f).name.lower().startswith("bt.")
        for f in gridded_files
    )
    has_combined_lidar = any(
        "combined" in Path(f).name.lower() for f in gridded_files
    )

    if data_type == "prediction" and not has_bluetopo:
        Engine.write_message_dask(
            f" [{current_index}/{total_count}] [SKIP MISSING INPUT] Tile "
            f"{tile_name}: Missing required BlueTopo data for prediction.",
            OUTPUTS,
        )
        return pd.DataFrame()

    if data_type == "training" and not has_combined_lidar:
        Engine.write_message_dask(
            f" [{current_index}/{total_count}] [SKIP MISSING INPUT] Tile "
            f"{tile_name}: Missing required combined LiDAR data for training.",
            OUTPUTS,
        )
        return pd.DataFrame()

    try:
        gridded_df, point_crs = _subtile_process_gridded(
            sub_grid, gridded_files, is_aws, subgrid_crs
        )
        valid_training_year_pairs = []
        if data_type == "training":
            available_bathy_years = _available_training_bathy_years(gridded_df)
            valid_training_year_pairs = _available_training_year_pairs(
                available_bathy_years, year_ranges
            )
            if not valid_training_year_pairs:
                Engine.write_message_dask(
                    f" [{current_index}/{total_count}] "
                    f"[SKIP NO CONFIGURED BATHY PAIR] Tile {tile_name}: "
                    f"valid bathymetry years={available_bathy_years}; none "
                    f"form a configured year range from {year_ranges}.",
                    OUTPUTS,
                )
                del gridded_df
                return pd.DataFrame()

            Engine.write_message_dask(
                f" [{current_index}/{total_count}] [TRAINING YEAR PAIRS] Tile "
                f"{tile_name}: valid bathymetry years={available_bathy_years}; "
                f"using configured pairs={valid_training_year_pairs}.",
                OUTPUTS,
            )

        combined_df = _subtile_process_ungridded(
            sub_grid,
            ungridded_files,
            gridded_df,
            static_patterns,
            is_aws,
            subgrid_crs,
            point_crs,
            data_type,
            valid_training_year_pairs,
        )
        
        # Save and calculate final table statistics
        stats = _save_combined_data(combined_df, output_folder, data_type, tile_name, is_aws, local_tmp_dir, current_index, total_count, verbose, valid_training_year_pairs)
        if stats is not None and not stats.empty:
            stats.attrs["created_parquet"] = True
            stats.attrs["parquet_path"] = str(expected_path)
            stats.attrs["raster_crs"] = (
                point_crs.to_wkt()
                if point_crs is not None and hasattr(point_crs, "to_wkt")
                else point_crs
            )
        del gridded_df, combined_df
        return stats
    
    except Exception as e:
        Engine.write_message_dask(f"ERROR: Error processing tile {tile_name}: {e}", OUTPUTS)
        return pd.DataFrame()
    finally:
        gc.collect()


class SubgridTilingEngine(Engine):
    """Process complete raster tiles as memory-isolated Dask tasks."""

    def __init__(
        self,
        param_lookup: dict,
        output_prefix: str | bool = False,
        overwrite_outputs: bool = False,
    ) -> None:
        """Initialize paths, configurations, and environment variables"""
        
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        self.overwrite_outputs = overwrite_outputs

        # Setup local temp dir mapping to ensure EC2 limits aren't exceeded
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "subgrid_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)

        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']
        self.static_patterns = ['sed', 'tsm', 'hurr', 'grain', 'survey']

        self.inputs_dir = INPUTS

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""
        
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f"SubgridTilingEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        # Model output directories 
        prediction_output_dir = get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR')
        self.prediction_out_dir = UPath(f"{s3_dir_base}/{prediction_output_dir}") if self.is_aws else UPath(self.outputs_dir / prediction_output_dir)

        training_out_dir = get_config_item('MODEL', 'TRAINING_OUTPUT_DIR')
        self.training_out_dir = UPath(f"{s3_dir_base}/{training_out_dir}") if self.is_aws else UPath(self.outputs_dir / training_out_dir)

        # Tile directories
        training_tiles_dir = get_config_item('MODEL', 'TRAINING_TILES_DIR')
        self.training_tiles_dir = UPath(f"{s3_dir_base}/{training_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / training_tiles_dir)

        prediction_tiles_dir = get_config_item('MODEL', 'PREDICTION_TILES_DIR')
        self.prediction_tiles_dir = UPath(f"{s3_dir_base}/{prediction_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / prediction_tiles_dir)

        if not self.is_aws:
            self.training_tiles_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_tiles_dir.mkdir(parents=True, exist_ok=True)

        # Subgrid definitions 
        training_subgrid_path = get_config_item('MODEL', 'TRAINING_SUB_GRIDS')
        training_subgrid_layer = get_config_item('MODEL', 'TRAINING_SUB_GRIDS_LAYER')
        
        prediction_subgrid_path = get_config_item('MODEL', 'PREDICTION_SUB_GRIDS')
        prediction_subgrid_layer = get_config_item('MODEL', 'PREDICTION_SUB_GRIDS_LAYER')
        
        self.subgrid_paths = {
            'training': {
                'path': UPath(f"{s3_dir_base}/{training_subgrid_path}") if self.is_aws else UPath(self.outputs_dir / training_subgrid_path),
                'layer': training_subgrid_layer
            },
            'prediction': {
                'path': UPath(f"{s3_dir_base}/{prediction_subgrid_path}") if self.is_aws else UPath(self.outputs_dir / prediction_subgrid_path),
                'layer': prediction_subgrid_layer
            }
        }

        # Terrain defaults
        filled_dir = get_config_item('TERRAIN', 'FILLED_DIR')
        self.filled_out_dir = UPath(f"{s3_dir_base}/{filled_dir}") if self.is_aws else UPath(self.outputs_dir / filled_dir)
        self.filled_folder_name = self.filled_out_dir.name.lower()

        # Config item for combined LiDAR dir for training files 
        combined_lidar_dir = get_config_item('TERRAIN', 'COMBINED_LIDAR_DIR')
        self.combined_lidar_dir = UPath(f"{s3_dir_base}/{combined_lidar_dir}") if self.is_aws else UPath(self.outputs_dir / combined_lidar_dir)


    def _load_subgrids(self, data_type: str) -> gpd.GeoDataFrame:
            """Loads the subgrids definition for the given data type."""
            
            # Get the dictionary containing both 'path' and 'layer'
            subgrid_info = self.subgrid_paths.get(data_type)
            
            if not subgrid_info:
                return None

            # Extract the specific path and layer values
            file_path = subgrid_info['path']
            layer_name = subgrid_info['layer']

            self.write_message(f"Loading subgrids from: {file_path} (Layer: {layer_name})", OUTPUTS)
            try:
                # Pass both the path and the layer argument to GeoPandas
                return gpd.read_file(str(file_path), layer=layer_name)
            except Exception as e:
                self.write_message(f"EXCEPTION: Reading subgrids from {file_path} (Layer: {layer_name}) failed. {e}", OUTPUTS)
                return None


    def _get_filtered_raster_files(self, raster_dirs: list, data_type: str) -> list:
        """Scans and filters raster files based on type rules."""
        
        self.write_message("Scanning directories for raster files...", OUTPUTS)
        all_raster_files = []

        for raster_dir in raster_dirs:
            if not raster_dir.exists():
                self.write_message(f"Directory not found, skipping: {raster_dir}", OUTPUTS)
                continue
                
            for f in raster_dir.rglob("*"):
                if f.suffix.lower() in {'.tif', '.tiff'}:
                    name_lower = f.name.lower()
                    parts_lower = [p.lower() for p in f.parts]

                    # Skip filled tifs ONLY IF they aren't our required combined files for training
                    if (self.filled_folder_name in parts_lower or "filled_lidar" in parts_lower or "filled_tifs" in parts_lower) and "combined" not in name_lower:
                        continue
                    is_bluetopo_file = (
                        "bluetopo" in name_lower or name_lower.startswith("bt.")
                    )
                    if "unc" in name_lower and not (
                        data_type == "prediction" and is_bluetopo_file
                    ):
                        continue
                    
                    exclude_patterns = ["tsm_cumulative", "hurr_count_mean", "hurr_count_cumulative", "hurr_strength_cumulative"]
                    if any(x in name_lower for x in exclude_patterns) or \
                       re.search(r"hurr_count_\d{4}_\d{4}", name_lower) or \
                       re.search(r"hurr_strength_\d{4}_\d{4}", name_lower):
                        continue

                    if data_type == 'training' and ("bluetopo" in name_lower or name_lower.startswith("bt.")) and "survey_end_date" not in name_lower:
                        continue
                    elif data_type == 'prediction' and "bathy" in name_lower and "bluetopo" not in name_lower and not name_lower.startswith("bt."):
                        continue

                    all_raster_files.append(str(f))

        return all_raster_files


    def _partition_raster_files(self, all_raster_files: list, sub_grids: gpd.GeoDataFrame) -> tuple:
        """Pre-partition files into gridded vs ungridded lists."""
        
        valid_tids = [str(tid) for tid in sub_grids['original_tile'].unique() if pd.notna(tid) and str(tid).strip()]
        valid_tid_patterns = [
            re.compile(rf"(?:^|_){re.escape(tid)}(?:_|\.)", re.IGNORECASE)
            for tid in valid_tids
        ]

        gridded_files = [f for f in all_raster_files if any(p.search(Path(f).name) for p in valid_tid_patterns)]
        gridded_set = set(gridded_files)
        ungridded_files = [f for f in all_raster_files if f not in gridded_set]

        return gridded_files, ungridded_files


    def _process_pipeline(self, raster_dirs: list, output_dir: UPath, data_type: str, verbose_workers: bool = False) -> None:
        """Submit whole-tile tasks so large raster arrays remain on workers."""
        
        self.write_message(f"--- Starting {data_type.upper()} pipeline ---", OUTPUTS)
        self.write_message(self.log_system_metrics(), OUTPUTS)

        sub_grids = self._load_subgrids(data_type)
        if sub_grids is None or sub_grids.empty:
            return
        if sub_grids.crs is None:
            self.write_message(
                f"ERROR: {data_type} subgrid has no CRS; refusing to compare "
                "its bounds with raster bounds.",
                OUTPUTS,
            )
            return

        all_raster_files = self._get_filtered_raster_files(raster_dirs, data_type)
        gridded_files, ungridded_files = self._partition_raster_files(all_raster_files, sub_grids)

        ungridded_files = [
            f
            for f in ungridded_files
            if any(pattern in Path(f).name.lower() for pattern in self.static_patterns)
        ]
        ungridded_catalog = _build_spatial_catalog(ungridded_files, self.is_aws)

        total_tiles = len(sub_grids)
        if self.multiband_geotiff_every == 1:
            geotiff_message = (
                "A multiband GeoTIFF will be created for every successful "
                "Parquet write."
            )
        elif self.multiband_geotiff_every > 1:
            geotiff_message = (
                "A multiband GeoTIFF will be created for the first successful "
                "Parquet write and then every "
                f"{self.multiband_geotiff_every} successful writes thereafter."
            )
        else:
            geotiff_message = "Multiband GeoTIFF creation is disabled."
        self.write_message(
            f"Submitting {total_tiles} whole-tile tasks in batches of "
            f"{self.tile_batch_size}. Raster arrays remain inside workers. "
            f"Overwrite outputs: {self.overwrite_outputs}. {geotiff_message}",
            OUTPUTS,
        )
        
        valid_results = []
        pending_tasks = []
        created_parquet_count = 0
        subgrid_crs = sub_grids.crs.to_wkt() if sub_grids.crs is not None else None

        def collect_batch(tasks: list) -> None:
            nonlocal created_parquet_count
            if not tasks:
                return
            for result in dask.compute(*tasks):
                if result is not None and not result.empty:
                    if result.attrs.get("created_parquet", False):
                        created_parquet_count += 1
                        if (
                            self.multiband_geotiff_every > 0
                            and (created_parquet_count - 1)
                            % self.multiband_geotiff_every
                            == 0
                        ):
                            dask.compute(
                                dask.delayed(
                                    _create_multiband_geotiff_from_parquet
                                )(
                                    parquet_path=result.attrs["parquet_path"],
                                    raster_crs=result.attrs["raster_crs"],
                                    is_aws=self.is_aws,
                                    local_tmp_dir=str(self.local_tmp_dir),
                                    created_sequence=created_parquet_count,
                                    overwrite_outputs=self.overwrite_outputs,
                                )
                            )
                    valid_results.append(result)

        for i, (_, sub_grid) in enumerate(sub_grids.iterrows()):
            tile_name = str(sub_grid['tile_id'])
            output_folder = output_dir / tile_name

            original_tile = str(sub_grid['original_tile'])
            tile_pattern = re.compile(
                rf"(?:^|_){re.escape(original_tile)}(?:_|\.)", re.IGNORECASE
            )
            tile_gridded_files = [
                f for f in gridded_files if tile_pattern.search(Path(f).name)
            ]
            tile_ungridded_files = _files_intersecting_tile(
                sub_grid.geometry.bounds, subgrid_crs, ungridded_catalog
            )

            pending_tasks.append(dask.delayed(_process_tile)(
                sub_grid=sub_grid,
                gridded_files=tile_gridded_files,
                ungridded_files=tile_ungridded_files,
                static_patterns=self.static_patterns,
                is_aws=self.is_aws,
                output_folder=str(output_folder),
                data_type=data_type,
                tile_name=tile_name,
                local_tmp_dir=str(self.local_tmp_dir),
                current_index=i + 1,
                total_count=total_tiles,
                verbose=verbose_workers,
                subgrid_crs=subgrid_crs,
                overwrite_outputs=self.overwrite_outputs,
                year_ranges=self.year_ranges,
            ))

            if len(pending_tasks) >= self.tile_batch_size:
                collect_batch(pending_tasks)
                pending_tasks.clear()

        collect_batch(pending_tasks)

        # Concatenate returned dataframes to construct final stats summary
        if valid_results:
            final_results_df = pd.concat(valid_results, ignore_index=True)
            output_csv_path = output_dir.parent / f"year_pair_nan_counts_{data_type}.csv"
            final_results_df.to_csv(str(output_csv_path), index=False, na_rep='NA')
            self.write_message(f"[SUCCESS] Statistics successfully saved to: {output_csv_path}", OUTPUTS)

        self.write_message(self.log_system_metrics(), OUTPUTS)


    def run(self) -> None:
        """Main entry point for evaluating training masks and processing rasters in parallel"""
        
        env = self.param_lookup.get('env', 'local')
        
        try:
            n_workers = max(1, int(os.environ.get("SUBGRID_N_WORKERS", "13")))
            memory_limit = os.environ.get(
                "SUBGRID_WORKER_MEMORY_LIMIT", "2.5GB"
            )
            self.tile_batch_size = max(
                1, int(os.environ.get("SUBGRID_TILE_BATCH_SIZE", str(n_workers)))
            )
            self.multiband_geotiff_every = max(
                0, int(os.environ.get("SUBGRID_GEOTIFF_EVERY", "1"))
            ) 

            self.setup_dask(
                env,
                n_workers=n_workers,
                threads_per_worker=1,
                memory_limit=memory_limit,
            )

            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region) 

                # Process Prediction Data
                self._process_pipeline(
                    raster_dirs=[self.prediction_out_dir], 
                    output_dir=self.prediction_tiles_dir, 
                    data_type="prediction",
                    verbose_workers=False  # Controls task-level logging verbosity
                )

                # Process Training Data
                self._process_pipeline(
                    raster_dirs=[self.training_out_dir, self.combined_lidar_dir], 
                    output_dir=self.training_tiles_dir, 
                    data_type="training",
                    verbose_workers=False  # Controls task-level logging verbosity
                )
        finally:
            self.cleanup_resources(OUTPUTS)
