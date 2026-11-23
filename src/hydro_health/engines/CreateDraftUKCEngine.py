from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory

import fiona
import geopandas as gpd
from hydro_health.engines import CreateHurricaneLayerEngine
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import MergeAlg, Resampling
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import Affine, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from upath import UPath

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


INPUTS = Path(__file__).parents[3] / "inputs"
OUTPUTS = Path(__file__).parents[3] / "outputs"


# NOAA Marine Cadastre VesselType lookup. Each entry is (NOAA group, code meaning).
# Source: https://coast.noaa.gov/data/marinecadastre/ais/VesselTypeCodes2018.pdf
# Use the first item to group vessel-type density rasters and the second for labels.
# The AIS hazard categories X, Y, Z, and OS were formerly labeled A, B, C, and D.
# Codes 60–69 identify passenger ships. Their final digit describes a cargo hazard
# category, a reserved value, or missing detail. It gives no passenger headcount.
# AVIS codes 1012–1015 describe passenger service and inspection status. Passenger
# capacity, passengers aboard, and crew count require separate data.
VESSEL_TYPE_CODES: dict[int, tuple[str, str]] = {
    # Standard AIS ship type codes.
    0: ("Not Available", "Vessel type unavailable or default"),
    **dict.fromkeys(range(1, 20), ("Other", "Reserved AIS vessel type")),
    20: ("Other", "Wing-in-ground craft, general code"),
    # NOAA groups 21 and 22 under Tug Tow. Their raw AIS labels refer to wing-in-ground craft.
    21: ("Tug Tow", "Wing-in-ground craft, hazard category X (formerly A)"),
    22: ("Tug Tow", "Wing-in-ground craft, hazard category Y (formerly B)"),
    23: ("Other", "Wing-in-ground craft, hazard category Z (formerly C)"),
    24: ("Other", "Wing-in-ground craft, hazard category OS (formerly D)"),
    **dict.fromkeys(range(25, 30), ("Other", "Reserved wing-in-ground craft subtype")),
    30: ("Fishing", "Vessel engaged in fishing"),
    31: ("Tug Tow", "Vessel engaged in towing"),
    32: ("Tug Tow", "Towing with tow length over 200 m or breadth over 25 m"),
    33: ("Other", "Dredging or underwater work"),
    34: ("Other", "Diving support operations"),
    35: ("Military", "Vessel engaged in military operations"),
    36: ("Pleasure Craft/Sailing", "Sailing vessel"),
    37: ("Pleasure Craft/Sailing", "Recreational pleasure craft"),
    **dict.fromkeys(range(38, 40), ("Other", "Reserved AIS vessel type")),
    # High-speed craft codes.
    40: ("Other", "High-speed craft, general code"),
    41: ("Other", "High-speed craft, hazard category X (formerly A)"),
    42: ("Other", "High-speed craft, hazard category Y (formerly B)"),
    43: ("Other", "High-speed craft, hazard category Z (formerly C)"),
    44: ("Other", "High-speed craft, hazard category OS (formerly D)"),
    **dict.fromkeys(range(45, 49), ("Other", "High-speed craft, reserved subtype")),
    49: ("Other", "High-speed craft, no further detail reported"),

    # AIS special craft codes.
    50: ("Other", "Pilot boat"),
    51: ("Other", "Search-and-rescue vessel"),
    52: ("Tug Tow", "Tugboat"),
    53: ("Other", "Port tender"),
    54: ("Other", "Vessel equipped for pollution response"),
    55: ("Other", "Law enforcement vessel"),
    56: ("Other", "Spare AIS code for local assignment"),
    57: ("Other", "Spare AIS code for local assignment"),
    58: ("Other", "Medical transport vessel"),
    59: ("Other", "Ship designated under Radio Regulations Resolution No. 18"),
    # Passenger ship codes.
    60: ("Passenger", "Passenger ship, general code"),
    61: ("Passenger", "Passenger ship, hazard category X (formerly A)"),
    62: ("Passenger", "Passenger ship, hazard category Y (formerly B)"),
    63: ("Passenger", "Passenger ship, hazard category Z (formerly C)"),
    64: ("Passenger", "Passenger ship, hazard category OS (formerly D)"),
    **dict.fromkeys(range(65, 69), ("Passenger", "Passenger ship, reserved subtype")),
    69: ("Passenger", "Passenger ship, no further detail reported"),
    # Cargo ship codes.
    70: ("Cargo", "Cargo ship, general code"),
    71: ("Cargo", "Cargo ship, hazard category X (formerly A)"),
    72: ("Cargo", "Cargo ship, hazard category Y (formerly B)"),
    73: ("Cargo", "Cargo ship, hazard category Z (formerly C)"),
    74: ("Cargo", "Cargo ship, hazard category OS (formerly D)"),
    **dict.fromkeys(range(75, 79), ("Cargo", "Cargo ship, reserved subtype")),
    79: ("Cargo", "Cargo ship, no further detail reported"),
    # Tanker codes.
    80: ("Tanker", "Tanker, general code"),
    81: ("Tanker", "Tanker, hazard category X (formerly A)"),
    82: ("Tanker", "Tanker, hazard category Y (formerly B)"),
    83: ("Tanker", "Tanker, hazard category Z (formerly C)"),
    84: ("Tanker", "Tanker, hazard category OS (formerly D)"),
    **dict.fromkeys(range(85, 89), ("Tanker", "Tanker, reserved subtype")),
    89: ("Tanker", "Tanker, no further detail reported"),
    # Other vessel type codes.
    90: ("Other", "Other vessel type, general code"),
    91: ("Other", "Other vessel type, hazard category X (formerly A)"),
    92: ("Other", "Other vessel type, hazard category Y (formerly B)"),
    93: ("Other", "Other vessel type, hazard category Z (formerly C)"),
    94: ("Other", "Other vessel type, hazard category OS (formerly D)"),
    **dict.fromkeys(range(95, 99), ("Other", "Other vessel type, reserved subtype")),
    99: ("Other", "Other vessel type, no further detail reported"),

    # Regional and reserved AIS codes, followed by NOAA's unassigned range.
    **dict.fromkeys(range(100, 200), ("Other", "Reserved for regional AIS use")),
    **dict.fromkeys(range(200, 256), ("Other", "Reserved for future AIS use")),
    **dict.fromkeys(range(256, 1000), ("Other", "Unassigned Marine Cadastre code")),

    # USCG AVIS vessel service codes used by NOAA to fill missing AIS vessel types.
    1001: ("Fishing", "Commercial fishing vessel"),
    1002: ("Fishing", "Vessel that processes fish"),
    1003: ("Cargo", "Barge carrying freight"),
    1004: ("Cargo", "Freight ship"),
    1005: ("Other", "Vessel in industrial service"),
    1006: ("Other", "Vessel in miscellaneous service"),
    1007: ("Other", "Mobile offshore drilling unit"),
    1008: ("Other", "Non-vessel record"),
    1009: ("Other", "Non-vessel record"),
    1010: ("Other", "Offshore supply ship"),
    1011: ("Other", "Vessel in oil recovery service"),
    1012: ("Passenger", "Inspected passenger vessel"),
    1013: ("Passenger", "Uninspected passenger vessel"),
    1014: ("Passenger", "Inspected passenger barge"),
    1015: ("Passenger", "Uninspected passenger barge"),
    1016: ("Cargo", "Public vessel in freight service"),
    1017: ("Tanker", "Public tank ship or tank barge"),
    1018: ("Other", "Unclassified public vessel"),
    1019: ("Pleasure Craft/Sailing", "Recreational vessel"),
    1020: ("Other", "Vessel in research service"),
    1021: ("Military", "Search-and-rescue aircraft"),
    1022: ("Other", "School or training ship"),
    1023: ("Tug Tow", "Tank barge"),
    1024: ("Tanker", "Tank ship"),
    1025: ("Tug Tow", "Vessel in towing service"),
}


@dataclass(frozen=True)
class RegionalPaths:
    mask: UPath
    draft: UPath
    length: UPath
    passenger_capacity: UPath
    bathy: UPath
    ukc: UPath


def _normalize_vessel_names(names: pd.Series) -> pd.Series:
    """Match exact vessel names after normalizing case and whitespace."""
    return names.astype("string").str.strip().str.replace(r"\s+", " ", regex=True).str.casefold()


def _read_passenger_capacities(csv_path: UPath) -> tuple[dict[str, float], int]:
    """Load positive capacities and exclude names with conflicting capacity records."""
    ferry_data = pd.read_csv(str(csv_path), dtype=str, encoding="utf-8-sig")
    ferry_data.columns = [
        "".join(character for character in column.casefold() if character.isalnum())
        for column in ferry_data.columns
    ]
    required = {"vesselname", "passengercapacity"}
    missing = required.difference(ferry_data.columns)
    if missing:
        raise ValueError(f"Missing ferry CSV columns after normalization: {', '.join(sorted(missing))}.")

    names = _normalize_vessel_names(ferry_data["vesselname"])
    raw_capacities = ferry_data["passengercapacity"].str.replace(",", "", regex=False)
    capacities = pd.to_numeric(raw_capacities, errors="coerce")
    valid = names.notna() & names.ne("") & np.isfinite(capacities) & (capacities > 0)
    by_name = capacities[valid].groupby(names[valid])
    conflicting = by_name.nunique().gt(1)
    lookup = by_name.first().loc[~conflicting].to_dict()
    return lookup, int(conflicting.sum())


def _passenger_vessel_names(vessel_names: pd.Series, vessel_types: pd.Series) -> pd.Series:
    """Return normalized names for passenger-group tracks, indexed by AIS feature."""
    passenger_codes = [code for code, (group, _) in VESSEL_TYPE_CODES.items() if group == "Passenger"]
    is_passenger = pd.to_numeric(vessel_types, errors="coerce").isin(passenger_codes)
    return _normalize_vessel_names(vessel_names.where(is_passenger))


def _worker_ais(
    layer: str,
    input_gdb: str,
    orig_bounds: tuple[float, float, float, float],
    orig_crs_wkt: str | None,
    final_crs_wkt: str,
    temp_dir: Path,
    out_meta: dict,
    height: int,
    width: int,
    transform: Affine,
    block_size: int,
    field: str,
    capacity_by_name: dict[str, float] | None = None,
    write_raster: bool = True,
) -> tuple[Path | None, set[str]]:
    """Rasterize one month of an AIS-derived vessel attribute."""
    with fiona.open(input_gdb, layer=layer) as src:
        layer_crs = src.crs

    if orig_crs_wkt and layer_crs:
        orig_box_gdf = gpd.GeoDataFrame({'geometry': [box(*orig_bounds)]}, crs=orig_crs_wkt)
        read_bounds = tuple(orig_box_gdf.to_crs(layer_crs).total_bounds)
    else:
        read_bounds = orig_bounds

    gdf = gpd.read_file(input_gdb, layer=layer, bbox=read_bounds, ignore_geometry=not write_raster)
    observed_names = set()

    if field == "PassengerCapacity":
        missing = {"VesselName", "VesselType"}.difference(gdf.columns)
        if missing:
            raise ValueError(f"Missing AIS columns: {', '.join(sorted(missing))}.")
        names = _passenger_vessel_names(gdf["VesselName"], gdf["VesselType"])
        observed_names = set(names.dropna())
        observed_names.discard("")
        if not write_raster:
            return None, observed_names

        values = names.map(capacity_by_name or {})
    else:
        if field == "Length" and field not in gdf.columns:
            raise ValueError("Missing AIS Length column.")
        if gdf.empty or field not in gdf.columns:
            return None, observed_names

        values = gdf[field]
        if field == "Draft" and (values.dtype == object or pd.api.types.is_string_dtype(values)):
            values = values.astype(str).str.extract(r"(\d+\.?\d*)", expand=False)
        values = pd.to_numeric(values, errors="coerce")

    gdf[field] = values
    gdf = gdf[np.isfinite(values) & (values > 0)]
    if gdf.empty:
        return None, observed_names

    if gdf.crs != final_crs_wkt:
        gdf = gdf.to_crs(final_crs_wkt)

    is_mean = field != "Draft"
    if not is_mean:
        gdf = gdf.sort_values(by=field, ascending=True)
    sindex = gdf.sindex
    monthly_tif_path = temp_dir / f"{layer}_{field}.tiff"
    monthly_profile = {**out_meta, "nodata": 0}
    if is_mean:
        monthly_profile.update(count=2, dtype="float64", BIGTIFF="IF_SAFER")

    with rasterio.open(monthly_tif_path, "w", **monthly_profile) as dest:
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                win_height = min(block_size, height - row)
                win_width = min(block_size, width - col)
                window = Window(col, row, win_width, win_height)
                win_transform = rasterio.windows.transform(window, transform)
                win_bounds = rasterio.windows.bounds(window, transform)

                possible_matches_index = list(sindex.intersection(win_bounds))
                if not possible_matches_index:
                    continue

                possible_matches = gdf.iloc[sorted(possible_matches_index)]
                precise_matches = possible_matches[possible_matches.intersects(box(*win_bounds))]

                if precise_matches.empty:
                    continue

                shapes = zip(precise_matches.geometry, precise_matches[field])
                if is_mean:
                    totals, counts = _rasterize_observations(shapes, win_height, win_width, win_transform)
                    dest.write(totals, 1, window=window)
                    dest.write(counts, 2, window=window)
                    continue

                burned = rasterize(
                    shapes=shapes,
                    out_shape=(win_height, win_width),
                    transform=win_transform,
                    fill=0,
                    all_touched=True,
                    merge_alg=MergeAlg.replace,
                    dtype=np.float32
                )
                dest.write(burned, 1, window=window)

    return monthly_tif_path, observed_names


def _rasterize_observations(
    shapes: Iterator[tuple[BaseGeometry, float]], height: int, width: int, transform: Affine
) -> tuple[np.ndarray, np.ndarray]:
    """Count each AIS feature once per touched cell, even if its track crosses that cell repeatedly."""
    totals = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros_like(totals)

    for geometry, value in shapes:
        bounds = window_from_bounds(*geometry.bounds, transform=transform)
        row_start = max(0, floor(bounds.row_off) - 1)
        col_start = max(0, floor(bounds.col_off) - 1)
        row_end = min(height, ceil(bounds.row_off + bounds.height) + 1)
        col_end = min(width, ceil(bounds.col_off + bounds.width) + 1)
        if row_start >= row_end or col_start >= col_end:
            continue

        hit = rasterize(
            [(geometry, 1)],
            out_shape=(row_end - row_start, col_end - col_start),
            transform=transform * Affine.translation(col_start, row_start),
            all_touched=True,
            dtype="uint8",
        ).astype(bool)
        rows = slice(row_start, row_end)
        cols = slice(col_start, col_end)
        total_view = totals[rows, cols]
        count_view = counts[rows, cols]
        np.add(total_view, value, out=total_view, where=hit)
        np.add(count_view, 1, out=count_view, where=hit)

    return totals, counts


def _mean_observation_window(
    sources: list[rasterio.DatasetReader], window: Window, mask: np.ndarray, nodata: float
) -> np.ndarray:
    """Combine monthly sums and observation counts into one annual mean."""
    totals = np.zeros(mask.shape, dtype=np.float64)
    counts = np.zeros_like(totals)
    for source in sources:
        monthly_totals, monthly_counts = source.read((1, 2), window=window)
        totals += monthly_totals
        counts += monthly_counts

    result = np.full(mask.shape, nodata, dtype=np.float32)
    np.divide(totals, counts, where=mask & (counts > 0), out=result)
    return result


class CreateDraftUKCEngine(Engine):
    """Build annual passenger capacity, length, draft, and UKC rasters."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        self.is_aws = param_lookup.get("env", "local") == "aws"
        self.max_workers = 8

    def _resolve_paths(
        self, eco_regions: list[str]
    ) -> dict[tuple[int, str], tuple[UPath, dict[str, RegionalPaths]]]:
        """Resolve shared inputs and the regional paths for each raster batch."""
        ais_input_dir = UPath(INPUTS / "AIS_data")
        self.input_gdb = ais_input_dir / "AIS_2024.gdb"
        self.ferry_csv_path = ais_input_dir / "Ferry_Data.csv"
        self.master_grids_path = UPath(INPUTS / get_config_item("SHARED", "MASTER_GRIDS"))

        if self.is_aws:
            bucket = get_config_item("SHARED", "OUTPUT_BUCKET")
            output_root = UPath(f"s3://{bucket}", use_listings_cache=False)
        else:
            output_root = UPath(OUTPUTS)

        if self.output_prefix:
            output_root = output_root / self.output_prefix

        low_res_dir = output_root / "low_res"
        drafts_dir = low_res_dir / "drafts"
        lengths_dir = low_res_dir / "lengths"
        passenger_capacity_dir = low_res_dir / "passenger_capacity"
        ukc_dir = low_res_dir / "ukc"
        mosaic_scope = self._mosaic_scope(eco_regions)

        batches = {}
        self.length_batches = {}
        self.passenger_capacity_batches = {}
        for resolution, suffix, bathy_tag, master_bathy in (
            (100, "", "", True),
            (100, "_offshore", "_Offshore", True),
            (20, "_nearshore", "", False),
        ):
            bathy_dir = low_res_dir / f"{resolution}m"
            bathy_name = f"BlueTopo_Bathy_Mosaic{bathy_tag}_{resolution}m.tiff"
            length_name = f"AIS_2024_Annual_Mean_Length_{resolution}m.tiff"
            capacity_name = f"AIS_2024_Annual_Mean_Passenger_Capacity_{resolution}m.tiff"
            regional_paths = {}
            for region in eco_regions:
                bathy_path = (
                    bathy_dir / bathy_name
                    if master_bathy
                    else bathy_dir / region / f"{region}_{bathy_name}"
                )
                regional_paths[region] = RegionalPaths(
                    mask=ais_input_dir / f"{region}_binary_mask_EPSG6350_{resolution}m.tiff",
                    draft=drafts_dir / f"{region}_AIS_2024_Annual_Max_Draft_{resolution}m.tiff",
                    length=lengths_dir / f"{region}_{length_name}",
                    passenger_capacity=passenger_capacity_dir / f"{region}_{capacity_name}",
                    bathy=bathy_path,
                    ukc=ukc_dir / f"{region}_UKC_Mosaic_{resolution}m{suffix}.tiff",
                )
            mosaic_path = ukc_dir / f"{mosaic_scope}UKC_Mosaic_{resolution}m{suffix}.tiff"
            batches[resolution, suffix] = mosaic_path, regional_paths
            length_mosaic = lengths_dir / f"{mosaic_scope}Mosaic_{length_name}"
            self.length_batches.setdefault(resolution, (length_mosaic, regional_paths))
            capacity_mosaic = passenger_capacity_dir / f"{mosaic_scope}Mosaic_{capacity_name}"
            self.passenger_capacity_batches.setdefault(resolution, (capacity_mosaic, regional_paths))

        return batches

    def _mosaic_scope(self, eco_regions: list[str]) -> str:
        """Reserve the shared mosaic filename for the full set of ecoregions."""
        try:
            with fiona.open(str(self.master_grids_path), layer="Enhanced_EcoRegions_50m") as src:
                all_regions = {feature["properties"]["EcoRegion"] for feature in src}
        except (OSError, ValueError, KeyError, fiona.errors.FionaError) as error:
            self.write_message(
                f"Could not determine all ecoregions from {self.master_grids_path}: {error}. "
                "Using a mosaic filename scoped to the selected regions.",
                OUTPUTS,
            )
        else:
            if set(eco_regions) == all_regions:
                return ""

        return f"Selected_{'_'.join(sorted(eco_regions))}_"

    @contextmanager
    def _stage_output(self, output_path: UPath) -> Iterator[Path]:
        """Publish a completed TIFF to S3 or move it into its local destination."""
        if self.is_aws:
            staging_parent = None
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            staging_parent = str(output_path.parent)

        with TemporaryDirectory(dir=staging_parent) as temp_dir:
            staged_path = Path(temp_dir) / output_path.name
            yield staged_path
            if self.is_aws:
                output_path.fs.put_file(str(staged_path), output_path.path)
            else:
                staged_path.replace(str(output_path))

    def generate_annual_max_draft(
        self, eco_region: str, resolution: int, mask_path: UPath, draft_path: UPath
    ) -> bool:
        """Rasterize each AIS month and write the masked annual cell maxima."""
        return self._generate_annual_ais_raster(eco_region, resolution, mask_path, draft_path, "Draft")

    def generate_annual_mean_length(
        self, eco_region: str, resolution: int, mask_path: UPath, length_path: UPath
    ) -> bool:
        """Write the mean AIS vessel length in each cell over all months."""
        return self._generate_annual_ais_raster(eco_region, resolution, mask_path, length_path, "Length")

    def _generate_annual_ais_raster(
        self,
        eco_region: str,
        resolution: int,
        mask_path: UPath,
        output_path: UPath,
        field: str,
        capacity_by_name: dict[str, float] | None = None,
    ) -> bool:
        is_mean = field != "Draft"
        product = {
            "Draft": "maximum draft",
            "Length": "mean vessel length",
            "PassengerCapacity": "mean passenger capacity",
        }[field]
        self.write_message(f"Processing annual {product} for {eco_region} at {resolution} m.", OUTPUTS)
        try:
            use_existing = output_path.is_file()
            if use_existing:
                self.write_message(f"Using existing {product} raster {output_path}.", OUTPUTS)
                if field != "PassengerCapacity":
                    return True
            if not mask_path.is_file():
                self.write_message(f"Skipping {eco_region}. Mask missing at {mask_path}.", OUTPUTS)
                return False
            if not self.input_gdb.is_dir():
                self.write_message(f"AIS geodatabase missing at {self.input_gdb}.", OUTPUTS)
                return False

            ais_layers = [
                layer for layer in fiona.listlayers(str(self.input_gdb))
                if layer.startswith("AIS_2024_")
            ]
            if not ais_layers:
                self.write_message(f"No 2024 AIS layers found in {self.input_gdb}.", OUTPUTS)
                return False

            target_crs = rasterio.crs.CRS.from_user_input(self.target_crs)
            with rasterio.open(str(mask_path)) as mask_src:
                orig_crs = mask_src.crs
                orig_bounds = tuple(mask_src.bounds)
            if orig_crs is None:
                self.write_message(f"Mask CRS is missing for {mask_path}.", OUTPUTS)
                return False

            minx, miny, maxx, maxy = orig_bounds
            if orig_crs != target_crs:
                minx, miny, maxx, maxy = transform_bounds(orig_crs, target_crs, *orig_bounds)

            minx = np.floor(minx / resolution) * resolution
            miny = np.floor(miny / resolution) * resolution
            maxx = np.ceil(maxx / resolution) * resolution
            maxy = np.ceil(maxy / resolution) * resolution
            width = int((maxx - minx) / resolution)
            height = int((maxy - miny) / resolution)
            transform = from_bounds(minx, miny, maxx, maxy, width, height)

            profile = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "transform": transform,
                "crs": target_crs,
                "count": 1,
                "dtype": "float32",
                "nodata": self.nodata_val,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 1024,
                "blockysize": 1024,
            }
            block_size = 1024 if is_mean else 4096
            orig_crs_wkt = orig_crs.to_wkt()
            target_crs_wkt = target_crs.to_wkt()

            with TemporaryDirectory() as temp_dir, ExitStack() as stack:
                action = "checking CSV matches in" if use_existing else "rasterizing"
                self.write_message(f"{eco_region}: {action} {len(ais_layers)} AIS layers for {product}.", OUTPUTS)
                valid_monthly_tifs = []
                failed_layers = []
                with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = {
                        executor.submit(
                            _worker_ais, layer, str(self.input_gdb), orig_bounds,
                            orig_crs_wkt, target_crs_wkt, Path(temp_dir), profile,
                            height, width, transform, block_size, field, capacity_by_name, not use_existing,
                        ): layer
                        for layer in ais_layers
                    }
                    for future in as_completed(futures):
                        layer = futures[future]
                        try:
                            monthly_path, observed_names = future.result()
                        except Exception as error:
                            failed_layers.append(layer)
                            self.write_message(f"{eco_region}: AIS layer {layer} failed: {error}", OUTPUTS)
                        else:
                            if field == "PassengerCapacity":
                                self._observed_passenger_names.update(observed_names)
                            if monthly_path is not None:
                                valid_monthly_tifs.append(monthly_path)

                if failed_layers:
                    self.write_message(f"{eco_region}: annual {product} incomplete. Skipping output.", OUTPUTS)
                    return False
                if use_existing:
                    return True
                if not valid_monthly_tifs:
                    self.write_message(f"No valid {product} observations found for {eco_region}.", OUTPUTS)
                    if field != "PassengerCapacity":
                        return False

                sources = [stack.enter_context(rasterio.open(path)) for path in valid_monthly_tifs]
                self.write_message(f"{eco_region}: aggregating monthly {product} rasters.", OUTPUTS)
                with self._stage_output(output_path) as staged_path:
                    with rasterio.open(staged_path, "w", **profile) as dest:
                        with rasterio.open(str(mask_path)) as mask_src:
                            for _, window in dest.block_windows(1):
                                shape = (int(window.height), int(window.width))
                                mask_chunk = np.zeros(shape, dtype=np.float32)
                                reproject(
                                    source=rasterio.band(mask_src, 1),
                                    destination=mask_chunk,
                                    src_transform=mask_src.transform,
                                    src_crs=mask_src.crs,
                                    dst_transform=dest.window_transform(window),
                                    dst_crs=target_crs,
                                    dst_nodata=0,
                                    resampling=Resampling.nearest,
                                )

                                valid_mask = mask_chunk > 0
                                if is_mean:
                                    result = _mean_observation_window(sources, window, valid_mask, self.nodata_val)
                                else:
                                    result = np.zeros(shape, dtype=np.float32)
                                    for src in sources:
                                        np.maximum(result, src.read(1, window=window), out=result)
                                    result[~valid_mask | (result == 0)] = self.nodata_val
                                dest.write(result, 1, window=window)

            self.write_message(f"Saved annual {product} to {output_path}.", OUTPUTS)
            return True
        except Exception as error:
            self.write_message(f"Annual {product} generation failed for {eco_region}: {error}", OUTPUTS)
            return False

    def calculate_regional_ukc(
        self, eco_region: str, resolution: int, draft_path: UPath, bathy_path: UPath, ukc_path: UPath
    ) -> bool:
        """Align bathymetry to the draft raster and subtract draft from water depth."""
        self.write_message(f"Processing UKC for {eco_region} at {resolution} m.", OUTPUTS)
        try:
            if not bathy_path.is_file():
                self.write_message(f"Skipping {eco_region}. Bathymetry mosaic missing at {bathy_path}.", OUTPUTS)
                return False

            with rasterio.open(str(draft_path)) as draft_src, rasterio.open(str(bathy_path)) as bathy_src:
                vrt_options = {
                    "resampling": Resampling.average,
                    "crs": draft_src.crs,
                    "transform": draft_src.transform,
                    "height": draft_src.height,
                    "width": draft_src.width,
                    "dtype": "float32",
                    "nodata": self.nodata_val,
                }
                out_profile = draft_src.profile.copy()
                out_profile.update(dtype="float32", nodata=self.nodata_val, BIGTIFF="YES")
                with self._stage_output(ukc_path) as staged_path:
                    with WarpedVRT(bathy_src, **vrt_options) as vrt_bathy:
                        with rasterio.open(staged_path, "w", **out_profile) as dest:
                            for _, window in draft_src.block_windows(1):
                                draft = draft_src.read(1, window=window, masked=True)
                                bathy = vrt_bathy.read(1, window=window, masked=True)
                                valid = ~(np.ma.getmaskarray(draft) | np.ma.getmaskarray(bathy))
                                valid &= np.isfinite(draft.data) & np.isfinite(bathy.data)
                                ukc = np.full(draft.shape, self.nodata_val, dtype=np.float32)
                                ukc[valid] = np.abs(bathy.data[valid]) - draft.data[valid]
                                dest.write(ukc, 1, window=window)

            self.write_message(f"Saved regional UKC to {ukc_path}.", OUTPUTS)
            return True
        except Exception as error:
            self.write_message(f"UKC calculation failed for {eco_region}: {error}", OUTPUTS)
            return False

    def merge_regional_ukc(self, regional_paths: list[UPath], mosaic_path: UPath) -> None:
        """Merge regional UKC rasters into a mosaic."""
        self._merge_regional_rasters(regional_paths, mosaic_path)

    def _merge_regional_rasters(self, regional_paths: list[UPath], mosaic_path: UPath) -> None:
        """Mosaic regional rasters on their common output grid."""
        if not regional_paths:
            self.write_message(f"No regional rasters available for {mosaic_path}.", OUTPUTS)
            return

        self.write_message(f"Merging {len(regional_paths)} regional rasters into {mosaic_path}.", OUTPUTS)
        with ExitStack() as stack:
            sources = [stack.enter_context(rasterio.open(str(path))) for path in regional_paths]
            mosaic_data, mosaic_transform = merge(sources, nodata=self.nodata_val)
            profile = sources[0].meta.copy()
            profile.update(
                driver="GTiff",
                height=mosaic_data.shape[1],
                width=mosaic_data.shape[2],
                transform=mosaic_transform,
                nodata=self.nodata_val,
                compress="lzw",
                tiled=True,
                blockxsize=1024,
                blockysize=1024,
                BIGTIFF="YES",
            )
            with self._stage_output(mosaic_path) as staged_path:
                with rasterio.open(staged_path, "w", **profile) as dest:
                    dest.write(mosaic_data)

        self.write_message(f"Saved mosaic to {mosaic_path}.", OUTPUTS)

    def _run_passenger_capacity(self) -> dict[str, float] | None:
        """Generate passenger-capacity rasters and retain the lookup for coverage reporting."""
        self._observed_passenger_names: set[str] = set()
        self._passenger_coverage_incomplete = False
        try:
            capacities, conflicting_names = _read_passenger_capacities(self.ferry_csv_path)
        except (OSError, ValueError) as error:
            self.write_message(f"Skipping passenger capacity. Could not load {self.ferry_csv_path}: {error}", OUTPUTS)
            return None

        self.write_message(
            f"Loaded capacity for {len(capacities)} ferry names. "
            f"Excluded {conflicting_names} names with conflicting capacities.",
            OUTPUTS,
        )
        for resolution, (mosaic_path, regional_paths) in self.passenger_capacity_batches.items():
            completed_paths = []
            for eco_region, paths in regional_paths.items():
                if self._generate_annual_ais_raster(
                    eco_region, resolution, paths.mask, paths.passenger_capacity, "PassengerCapacity", capacities
                ):
                    completed_paths.append(paths.passenger_capacity)

            if len(completed_paths) != len(regional_paths):
                self._passenger_coverage_incomplete = True
                self.write_message(f"Skipping {mosaic_path} because some selected regions failed.", OUTPUTS)
                continue

            try:
                self._merge_regional_rasters(completed_paths, mosaic_path)
            except Exception as error:
                self.write_message(f"Passenger capacity mosaic failed for {mosaic_path}: {error}", OUTPUTS)

        return capacities

    def run(self) -> None:
        """Process passenger capacity, length, draft, and UKC in that order."""
        self.write_message("Starting annual passenger capacity, vessel length, draft, and UKC processing.", OUTPUTS)
        eco_regions = list(dict.fromkeys(self.param_lookup["eco_regions"].value))
        if not eco_regions:
            self.write_message("No ecoregions selected.", OUTPUTS)
            return

        batches = self._resolve_paths(eco_regions)
        passenger_capacities = self._run_passenger_capacity()
        for resolution, (mosaic_path, regional_paths) in self.length_batches.items():
            completed_paths = []
            for eco_region in eco_regions:
                paths = regional_paths[eco_region]
                if self.generate_annual_mean_length(eco_region, resolution, paths.mask, paths.length):
                    completed_paths.append(paths.length)

            if len(completed_paths) == len(eco_regions):
                try:
                    self._merge_regional_rasters(completed_paths, mosaic_path)
                except Exception as error:
                    self.write_message(f"Vessel length mosaic failed for {mosaic_path}: {error}", OUTPUTS)
            else:
                self.write_message(f"Skipping {mosaic_path} because some selected regions failed.", OUTPUTS)

        for (resolution, suffix), (mosaic_path, regional_paths) in batches.items():
            self.write_message(f"Processing {resolution} m{suffix.replace('_', ' ')} rasters.", OUTPUTS)
            completed_paths = []
            for eco_region in eco_regions:
                paths = regional_paths[eco_region]
                if self.generate_annual_max_draft(eco_region, resolution, paths.mask, paths.draft):
                    if self.calculate_regional_ukc(eco_region, resolution, paths.draft, paths.bathy, paths.ukc):
                        completed_paths.append(paths.ukc)

            if len(completed_paths) != len(eco_regions):
                self.write_message(f"Skipping {mosaic_path} because some selected regions failed.", OUTPUTS)
                continue
            self.merge_regional_ukc(completed_paths, mosaic_path)

        if passenger_capacities is not None:
            total = len(self._observed_passenger_names)
            unmatched = len(self._observed_passenger_names - passenger_capacities.keys())
            scope = "scanned AIS tracks"
            if self._passenger_coverage_incomplete:
                scope += " (some inputs failed)"
            self.write_message(
                f"Passenger capacity matches in {scope}: {total:,} unique passenger vessel names, "
                f"{total - unmatched:,} matched, {unmatched:,} without a usable CSV capacity.",
                OUTPUTS,
            )

        self.write_message("Finished processing the selected ecoregions.", OUTPUTS)

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    engine = CreateDraftUKCEngine()
    engine.run()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  