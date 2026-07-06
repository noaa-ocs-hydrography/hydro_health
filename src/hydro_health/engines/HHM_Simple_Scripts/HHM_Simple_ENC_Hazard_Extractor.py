"""
NOAA ENC S-57 Hazard Extraction + Eco Region Summary Workflow
=============================================================

Single-file Python translation of the R-based NOAA native S-57 ENC hazard
workflow.

This script is intentionally kept as ONE FILE for easier sharing, review, and
testing before converting it into a package.

Purpose
-------
Extract charted marine hazards from native NOAA ENC S-57 `.000` files while
preserving important S-57 metadata such as QUAPOS and STATUS. The workflow then
creates Eco Region hazard summaries and raster products for use in HHM.

Main outputs
------------
1. Reusable hazard feature cache:
      - GeoPackage or GeoParquet
      - Classified hazards with QUAPOS logic and methodology weights
      - Neighbourhood hazard scores within 2 nautical miles

2. Per-Eco Region products:
      - P1 true-geometry grid summary
      - P2 buffered-geometry grid summary
      - P1 100 m rasterized hazard score TIFF
      - P2 100 m rasterized hazard score TIFF

Core methodology
----------------
Hazard feature classes:
    OBSTRN = Obstruction
    WRECKS = Wreck
    PILPNT = Pile / post
    UWTROC = Underwater / awash rock

QUAPOS grouping:
    Known hazards:
        QUAPOS 1, 10, 11, or NULL
        Weight = 1

    Unverified / reported / approximate hazards:
        QUAPOS 2 through 9
        OR STATUS / INFORM text suggesting reported, doubtful, approximate,
        unconfirmed, or existence doubtful
        Weight = 3

Neighbourhood scoring:
    Each hazard receives a score equal to the sum of nearby hazard weights
    within 2 nautical miles.

P1 product:
    Uses the true hazard geometry.

P2 product:
    Uses 2 nautical mile buffered geometry for display/coverage, but DOES NOT
    recalculate scores. Each buffered feature keeps the same neighbourhood score
    computed from the true hazard locations.

Recommended environment
-----------------------
Install with conda/mamba:

    mamba create -n enc_hazards_py -c conda-forge \
        python=3.11 geopandas pyogrio fiona shapely pyproj rasterio rioxarray \
        numpy pandas requests lxml pyarrow dask distributed tqdm

    conda activate enc_hazards_py

Important notes
---------------
- Native S-57 reading depends on GDAL/OGR support.
- QUAPOS on primitive/vector records can be difficult to access consistently
  across GDAL versions; this script attempts both normal feature reads and
  primitive reads using OGR_S57_OPTIONS.
- If primitive QUAPOS reads fail, the workflow still runs using feature-level
  QUAPOS where available.
- Raster masks are expected to be EPSG:6350 100 m binary masks, matching the R
  workflow.
- The default output CRS for metric operations is EPSG:6350.
"""


###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import csv
import math
import os
import re
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd
import requests

try:
    import geopandas as gpd
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from shapely.validation import make_valid
except ImportError as exc:
    raise ImportError(
        "Missing geopandas/shapely. Install with: "
        "conda install -c conda-forge geopandas shapely pyogrio fiona"
    ) from exc

try:
    import fiona
except ImportError as exc:
    raise ImportError(
        "Missing fiona. Install with: conda install -c conda-forge fiona"
    ) from exc

try:
    import rasterio
    from rasterio.features import shapes, rasterize
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.transform import Affine
except ImportError as exc:
    raise ImportError(
        "Missing rasterio. Install with: conda install -c conda-forge rasterio"
    ) from exc

try:
    from dask.distributed import Client, LocalCluster, as_completed
except ImportError:
    Client = None
    LocalCluster = None
    as_completed = None


###############################################################################
# User-adjustable defaults
###############################################################################

DEFAULT_WORK_DIR = Path(
    r"N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards"
)

DEFAULT_MASK_DIR = Path(
    r"N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/Eco_Region_Masks/epsg_6350_100m"
)

NOAA_ENC_CATALOG_URL = "https://charts.noaa.gov/ENCs/ENCProdCat.xml"

HAZARD_LAYERS = ["OBSTRN", "WRECKS", "PILPNT", "UWTROC"]

TARGET_METRIC_CRS = "EPSG:6350"
QUAPOS_JOIN_CRS = "EPSG:5070"

NAUTICAL_MILE_M = 1852.0


###############################################################################
# Small data containers
###############################################################################

RegionsArg = Literal["all"] | str | Sequence[str]
CacheFormat = Literal["gpkg", "geoparquet", "parquet"]


@dataclass(frozen=True)
class EncFileRecord:
    """Information about one downloaded/extracted NOAA ENC cell."""

    download_url: str
    zip_path: Path
    extract_dir: Path
    s57_path: Path


@dataclass(frozen=True)
class MaskRecord:
    """Information about one Eco Region raster mask."""

    region_id: str
    mask_path: Path


###############################################################################
# Logging and error handling
###############################################################################

def log_step(text: str) -> None:
    """Print timestamped status messages."""

    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {text}", flush=True)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_error_row(
    error_log: str | Path,
    source_s57: Optional[str],
    layer: Optional[str],
    step: str,
    message: str,
) -> None:
    """Append one row to a CSV error log.

    The goal is to let the workflow continue even if a single ENC layer fails.
    """

    error_log = Path(error_log)
    error_log.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "source_s57": source_s57 or "",
        "layer": layer or "",
        "step": step,
        "message": str(message),
    }

    write_header = not error_log.exists()

    with error_log.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


###############################################################################
# GDAL / S-57 environment helpers
###############################################################################

def set_s57_feature_env() -> None:
    """Set GDAL options for reading normal S-57 feature records.

    This mirrors the R environment:

        SPLIT_MULTIPOINT=ON
        LIST_AS_STRING=ON
        PRESERVE_EMPTY_NUMBERS=ON
        ADD_SOUNDG_DEPTH=ON
        LNAM_REFS=ON

    These options help preserve S-57 attribute fields as strings and keep
    feature-to-feature references where available.
    """

    os.environ["OGR_S57_OPTIONS"] = (
        "SPLIT_MULTIPOINT=ON,"
        "LIST_AS_STRING=ON,"
        "PRESERVE_EMPTY_NUMBERS=ON,"
        "ADD_SOUNDG_DEPTH=ON,"
        "LNAM_REFS=ON"
    )


def set_s57_primitive_env() -> None:
    """Set GDAL options for reading S-57 primitive/vector records.

    Primitive records may contain QUAPOS even when the normal hazard feature
    layer does not expose it directly. This pass attempts to recover those
    primitive QUAPOS values.
    """

    os.environ["OGR_S57_OPTIONS"] = (
        "RETURN_PRIMITIVES=ON,"
        "LIST_AS_STRING=ON,"
        "PRESERVE_EMPTY_NUMBERS=ON"
    )


def set_proj_lib_if_needed() -> None:
    """Best-effort helper for PROJ environments.

    Usually conda handles this automatically. This function is intentionally
    conservative and only warns if PROJ configuration appears problematic.
    """

    if os.environ.get("PROJ_LIB"):
        return

    possible_roots = [
        Path(os.environ.get("CONDA_PREFIX", "")) / "Library" / "share" / "proj",
        Path(os.environ.get("CONDA_PREFIX", "")) / "share" / "proj",
    ]

    for candidate in possible_roots:
        if candidate.exists() and (candidate / "proj.db").exists():
            os.environ["PROJ_LIB"] = str(candidate)
            return


###############################################################################
# Geometry helpers
###############################################################################

def clean_geometries(gdf: gpd.GeoDataFrame, label: str = "GeoDataFrame") -> gpd.GeoDataFrame:
    """Make geometries valid and remove empty/null/bad rows."""

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"{label} is not a GeoDataFrame.")

    if gdf.empty:
        raise ValueError(f"{label} has zero rows.")

    gdf = gdf.copy()

    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    if gdf.empty:
        raise ValueError(f"{label} has zero non-empty geometries.")

    # Shapely 2 make_valid is reliable and fast enough for this workflow.
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: make_valid(geom) if geom is not None and not geom.is_valid else geom
    )

    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[~gdf.geometry.is_empty]

    # Remove geometries with non-finite bounds.
    good = []
    for geom in gdf.geometry:
        try:
            b = geom.bounds
            good.append(all(np.isfinite(b)))
        except Exception:
            good.append(False)

    gdf = gdf.loc[good].copy()

    if gdf.empty:
        raise ValueError(f"{label} has zero valid geometries after cleaning.")

    return gdf


def force_string_attributes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert non-geometry attributes to strings.

    S-57 attributes can be inconsistent across layers/cells. Keeping attributes
    as strings makes concatenation much safer, matching the R workflow.
    """

    gdf = gdf.copy()
    geom_name = gdf.geometry.name

    for col in gdf.columns:
        if col == geom_name:
            continue
        gdf[col] = gdf[col].astype("string").replace({"<NA>": pd.NA})

    return gdf


def representative_points(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Return representative points for any geometry type.

    This is safer than centroids for polygons because representative_point()
    is guaranteed to fall inside the polygon.
    """

    return gdf.geometry.representative_point()


###############################################################################
# Mask helpers
###############################################################################

def normalize_regions(regions: RegionsArg) -> str | list[str]:
    """Normalize region selection.

    Accepts:
      - "all"
      - "ER_6"
      - ["ER_2", "ER_6"]
    """

    if regions == "all":
        return "all"

    if isinstance(regions, str):
        return [regions]

    return [str(r) for r in regions]


def get_mask_table(mask_dir: str | Path, regions: RegionsArg = "all") -> list[MaskRecord]:
    """Find Eco Region mask TIFFs.

    Region IDs are extracted from file names using ER_[0-9]+.
    """

    mask_dir = Path(mask_dir)
    selected = normalize_regions(regions)

    mask_files = sorted(
        p for p in mask_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )

    records: list[MaskRecord] = []

    for path in mask_files:
        match = re.search(r"ER_[0-9]+", path.name)
        if not match:
            continue

        region_id = match.group(0)

        if selected != "all" and region_id not in selected:
            continue

        records.append(MaskRecord(region_id=region_id, mask_path=path))

    if not records:
        raise FileNotFoundError(f"No matching mask TIFFs found in: {mask_dir}")

    return records


def raster_mask_to_polygons(
    mask_path: str | Path,
    aggregate_factor: int = 1,
    dissolve: bool = True,
) -> gpd.GeoDataFrame:
    """Convert binary mask cells to polygons.

    Parameters
    ----------
    mask_path
        Binary mask raster. Cells equal to 1 are treated as the valid AOI.
    aggregate_factor
        Optional integer aggregation factor. For example, a 100 m mask with
        aggregate_factor=5 creates 500 m cells.
    dissolve
        If True, dissolve all mask polygons into one AOI. If False, keep
        individual cells.

    Returns
    -------
    GeoDataFrame
        Polygon representation of mask cells.
    """

    mask_path = Path(mask_path)

    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

        valid = np.where(arr == 1, 1, 0).astype("uint8")

        if aggregate_factor > 1:
            # Aggregate by max using simple block reduction.
            rows, cols = valid.shape
            new_rows = math.ceil(rows / aggregate_factor)
            new_cols = math.ceil(cols / aggregate_factor)

            padded = np.zeros(
                (new_rows * aggregate_factor, new_cols * aggregate_factor),
                dtype="uint8",
            )
            padded[:rows, :cols] = valid

            reshaped = padded.reshape(
                new_rows,
                aggregate_factor,
                new_cols,
                aggregate_factor,
            )
            valid = reshaped.max(axis=(1, 3)).astype("uint8")

            transform = transform * Affine.scale(aggregate_factor, aggregate_factor)

    geom_records = []
    for geom, value in shapes(valid, mask=valid == 1, transform=transform):
        if value == 1:
            geom_records.append(shape(geom))

    if not geom_records:
        raise ValueError(f"Mask has no cells equal to 1: {mask_path}")

    if dissolve:
        geom = unary_union(geom_records)
        gdf = gpd.GeoDataFrame({"mask_value": [1]}, geometry=[geom], crs=crs)
    else:
        gdf = gpd.GeoDataFrame(
            {"mask_value": np.ones(len(geom_records), dtype="uint8")},
            geometry=geom_records,
            crs=crs,
        )

    return clean_geometries(gdf, f"mask polygons {mask_path.name}")


def make_mask_aoi(mask_path: str | Path, aoi_grid_m: float = 500) -> gpd.GeoDataFrame:
    """Create dissolved AOI polygon from a binary mask."""

    with rasterio.open(mask_path) as src:
        res_x = abs(src.transform.a)
        fact = max(1, round(aoi_grid_m / res_x))

    return raster_mask_to_polygons(mask_path, aggregate_factor=fact, dissolve=True)


def make_summary_grid_from_mask(
    mask_path: str | Path,
    grid_m: float = 500,
    region_id: str = "ER",
) -> gpd.GeoDataFrame:
    """Create a polygon grid from a binary mask.

    The R workflow aggregates a 100 m mask to 500 m, keeps cells where any
    source mask cell equals 1, then converts those aggregated cells to polygons.
    """

    with rasterio.open(mask_path) as src:
        res_x = abs(src.transform.a)
        fact = max(1, round(grid_m / res_x))

    grid = raster_mask_to_polygons(mask_path, aggregate_factor=fact, dissolve=False)
    grid = grid.reset_index(drop=True)
    grid["region_id"] = region_id
    grid["grid_id"] = [f"{region_id}_grid_{i + 1}" for i in range(len(grid))]
    return grid


###############################################################################
# NOAA catalog, download, and extraction
###############################################################################

def read_enc_catalog_urls(catalog_url: str = NOAA_ENC_CATALOG_URL) -> pd.DataFrame:
    """Read NOAA ENC product catalog and extract US ENC zip URLs."""

    log_step("Reading NOAA ENC product catalog")

    response = requests.get(catalog_url, timeout=300)
    response.raise_for_status()

    txt = response.text

    urls = sorted(set(re.findall(r"https?://[^\"'< >]+\.zip", txt)))

    rows = []
    for url in urls:
        enc_zip = Path(url).name
        cell_name = re.sub(r"\.zip$", "", enc_zip, flags=re.IGNORECASE)

        if re.match(r"^US", cell_name, flags=re.IGNORECASE):
            rows.append(
                {
                    "download_url": url,
                    "enc_zip": enc_zip,
                    "cell_name": cell_name,
                }
            )

    return pd.DataFrame(rows)


def download_file_if_missing(url: str, dest: str | Path) -> Path:
    """Download a file only if it does not already exist."""

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        return dest

    log_step(f"Downloading {dest.name}")

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()

        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    return dest


def download_and_unzip_cell(
    download_url: str,
    download_dir: str | Path,
    extract_dir: str | Path,
    error_log: str | Path,
) -> list[EncFileRecord]:
    """Download one NOAA ENC zip and extract S-57 `.000` files."""

    download_dir = Path(download_dir)
    extract_dir = Path(extract_dir)

    zip_dest = download_dir / Path(download_url).name
    cell_name = zip_dest.stem
    cell_extract_dir = extract_dir / cell_name

    try:
        download_file_if_missing(download_url, zip_dest)

        if not cell_extract_dir.exists():
            ensure_dir(cell_extract_dir)
            with zipfile.ZipFile(zip_dest, "r") as z:
                z.extractall(cell_extract_dir)

        s57_files = sorted(cell_extract_dir.rglob("*.000"))

        return [
            EncFileRecord(
                download_url=download_url,
                zip_path=zip_dest,
                extract_dir=cell_extract_dir,
                s57_path=s57_path,
            )
            for s57_path in s57_files
        ]

    except Exception as exc:
        log_error_row(
            error_log,
            source_s57=zip_dest.name,
            layer=None,
            step="download_unzip",
            message=str(exc),
        )
        return []


###############################################################################
# S-57 layer readers
###############################################################################

def get_available_layers(s57_path: str | Path, error_log: str | Path) -> pd.DataFrame:
    """List available layers in one S-57 file."""

    s57_path = Path(s57_path)
    set_s57_feature_env()

    try:
        layers = list(fiona.listlayers(s57_path))
        return pd.DataFrame(
            {
                "source_s57": s57_path.name,
                "s57_path": str(s57_path),
                "layer_name": layers,
            }
        )

    except Exception as exc:
        log_error_row(
            error_log,
            source_s57=s57_path.name,
            layer=None,
            step="listlayers",
            message=str(exc),
        )
        return pd.DataFrame(
            {
                "source_s57": [s57_path.name],
                "s57_path": [str(s57_path)],
                "layer_name": [pd.NA],
            }
        )


def read_s57_layer(
    s57_path: str | Path,
    layer_name: str,
    error_log: str | Path,
    primitive: bool = False,
) -> Optional[gpd.GeoDataFrame]:
    """Read one S-57 layer with GeoPandas/Fiona.

    Parameters
    ----------
    primitive
        If True, sets S-57 primitive/vector read options first.
    """

    s57_path = Path(s57_path)

    if primitive:
        set_s57_primitive_env()
    else:
        set_s57_feature_env()

    try:
        gdf = gpd.read_file(s57_path, layer=layer_name)

        if gdf.empty:
            return None

        if gdf.crs is None:
            # Native ENCs are geographic WGS84. Most GDAL builds expose CRS, but
            # we set EPSG:4326 if missing to keep the workflow moving.
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)

        gdf = gdf.to_crs("EPSG:4326")
        gdf = clean_geometries(gdf, f"{s57_path.name} {layer_name}")
        gdf = force_string_attributes(gdf)

        gdf["source_s57"] = s57_path.name
        gdf["source_path"] = str(s57_path)

        return gdf

    except Exception as exc:
        step = "primitive_read_file" if primitive else "read_file"
        log_error_row(
            error_log,
            source_s57=s57_path.name,
            layer=layer_name,
            step=step,
            message=str(exc),
        )
        return None


def read_s57_hazard_layer(
    s57_path: str | Path,
    layer_name: str,
    error_log: str | Path,
) -> Optional[gpd.GeoDataFrame]:
    """Read one hazard layer from one S-57 file."""

    gdf = read_s57_layer(s57_path, layer_name, error_log, primitive=False)

    if gdf is None or gdf.empty:
        return None

    gdf["object_group"] = layer_name

    return gdf


def read_s57_hazards_for_file(
    s57_path: str | Path,
    available_layers: pd.DataFrame,
    hazard_layers: Sequence[str] = HAZARD_LAYERS,
    error_log: str | Path = "s57_hazard_error_log.csv",
) -> Optional[gpd.GeoDataFrame]:
    """Read all requested hazard layers from one S-57 file."""

    s57_path = Path(s57_path)

    layer_names = (
        available_layers.loc[
            available_layers["s57_path"].astype(str) == str(s57_path),
            "layer_name",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    layers_to_read = [lyr for lyr in hazard_layers if lyr in layer_names]

    if not layers_to_read:
        return None

    pieces = []

    for layer in layers_to_read:
        gdf = read_s57_hazard_layer(s57_path, layer, error_log)
        if gdf is not None and not gdf.empty:
            pieces.append(gdf)

    if not pieces:
        return None

    return pd.concat(pieces, ignore_index=True)


def read_s57_quapos_primitives(
    s57_path: str | Path,
    error_log: str | Path,
) -> Optional[gpd.GeoDataFrame]:
    """Read primitive/vector records containing QUAPOS.

    Why this exists
    ---------------
    Some S-57 hazard feature records do not expose QUAPOS directly, but QUAPOS
    may exist on a linked primitive/vector record. This function attempts to
    read primitive layers and keep records where QUAPOS is present.
    """

    s57_path = Path(s57_path)
    set_s57_primitive_env()

    try:
        layers = list(fiona.listlayers(s57_path))
    except Exception as exc:
        log_error_row(
            error_log,
            source_s57=s57_path.name,
            layer=None,
            step="primitive_listlayers",
            message=str(exc),
        )
        return None

    primitive_candidates = [
        lyr for lyr in layers
        if re.search(r"Node|Edge|Face|Primitive|Vector", lyr, flags=re.IGNORECASE)
    ]

    if not primitive_candidates:
        return None

    pieces = []

    for layer_name in primitive_candidates:
        gdf = read_s57_layer(
            s57_path=s57_path,
            layer_name=layer_name,
            error_log=error_log,
            primitive=True,
        )

        if gdf is None or gdf.empty:
            continue

        if "QUAPOS" not in gdf.columns:
            continue

        q = gdf["QUAPOS"].astype("string")
        gdf = gdf[q.notna() & (q.str.strip() != "")].copy()

        if gdf.empty:
            continue

        gdf["primitive_layer"] = layer_name
        gdf["primitive_id"] = [
            f"{s57_path.name}_{layer_name}_{i + 1}" for i in range(len(gdf))
        ]

        pieces.append(gdf)

    if not pieces:
        return None

    return pd.concat(pieces, ignore_index=True)


###############################################################################
# QUAPOS transfer and hazard classification
###############################################################################

def attach_quapos_from_primitives(
    hazards: gpd.GeoDataFrame,
    quapos_primitives: Optional[gpd.GeoDataFrame],
    error_log: str | Path,
    max_join_distance_m: float = 25,
) -> gpd.GeoDataFrame:
    """Transfer nearest primitive QUAPOS onto hazard features.

    This uses representative hazard points and nearest primitive geometries in
    EPSG:5070. QUAPOS is only accepted when the nearest primitive is within
    max_join_distance_m.
    """

    hazards = clean_geometries(hazards, "hazards before primitive QUAPOS join").copy()

    if quapos_primitives is None or quapos_primitives.empty:
        hazards["QUAPOS_primitive"] = pd.NA
        hazards["QUAPOS_source"] = pd.NA
        hazards["QUAPOS_join_distance_m"] = np.nan
        return hazards

    try:
        quapos_primitives = clean_geometries(
            quapos_primitives,
            "QUAPOS primitives before join",
        )

        hazards_m = hazards.to_crs(QUAPOS_JOIN_CRS)
        prim_m = quapos_primitives.to_crs(QUAPOS_JOIN_CRS)

        # Use representative points for hazard polygons/lines.
        hazard_pts = hazards_m.copy()
        hazard_pts["geometry"] = representative_points(hazards_m)

        # sjoin_nearest is efficient and uses spatial indexes when available.
        joined = gpd.sjoin_nearest(
            hazard_pts,
            prim_m[["QUAPOS", "geometry"]],
            how="left",
            distance_col="QUAPOS_join_distance_m",
        )

        joined_quapos = joined["QUAPOS"].astype("string")
        join_dist = joined["QUAPOS_join_distance_m"].astype("float64")

        too_far = join_dist > max_join_distance_m
        joined_quapos[too_far] = pd.NA
        join_dist[too_far] = np.nan

        hazards["QUAPOS_primitive"] = joined_quapos.to_numpy()
        hazards["QUAPOS_source"] = np.where(
            pd.notna(hazards["QUAPOS_primitive"]),
            "S57 primitive/vector record",
            pd.NA,
        )
        hazards["QUAPOS_join_distance_m"] = join_dist.to_numpy()

        return hazards

    except Exception as exc:
        log_error_row(
            error_log,
            source_s57="all",
            layer="QUAPOS primitives",
            step="attach_quapos_from_primitives",
            message=str(exc),
        )

        hazards["QUAPOS_primitive"] = pd.NA
        hazards["QUAPOS_source"] = pd.NA
        hazards["QUAPOS_join_distance_m"] = np.nan
        return hazards


def add_xy_fields(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Add representative lon/lat fields and geometry type."""

    gdf = clean_geometries(gdf, "hazards before XY creation").copy()

    gdf_4326 = gdf.to_crs("EPSG:4326")
    pts = representative_points(gdf_4326)

    gdf_4326["lon_wgs84"] = pts.x
    gdf_4326["lat_wgs84"] = pts.y
    gdf_4326["geometry_type"] = gdf_4326.geometry.geom_type.astype(str)

    return gdf_4326


def ensure_columns(gdf: gpd.GeoDataFrame, columns: Sequence[str]) -> gpd.GeoDataFrame:
    """Add missing columns as NA."""

    gdf = gdf.copy()
    for col in columns:
        if col not in gdf.columns:
            gdf[col] = pd.NA
    return gdf


def _coalesce_string(a: pd.Series, b: pd.Series) -> pd.Series:
    """String coalesce helper."""

    a = a.astype("string")
    b = b.astype("string")
    return a.where(a.notna() & (a.str.strip() != ""), b)


def add_hazard_classification(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Classify hazards into known vs unverified/reported groups.

    This is the core methodology section.

    Known hazard:
        QUAPOS is 1, 10, 11, or missing/null.
        Weight = 1

    Unverified / reported error:
        QUAPOS is 2 through 9, or STATUS/INFORM text indicates uncertainty.
        Weight = 3
    """

    needed = [
        "QUAPOS",
        "QUAPOS_primitive",
        "STATUS",
        "VALSOU",
        "WATLEV",
        "QUASOU",
        "SOUACC",
        "VERACC",
        "HORACC",
        "CATWRK",
        "EXPSOU",
        "SORDAT",
        "SORIND",
        "INFORM",
        "object_group",
        "source_s57",
    ]

    gdf = ensure_columns(gdf, needed).copy()

    object_group = gdf["object_group"].astype("string")
    source_s57 = gdf["source_s57"].astype("string")

    gdf["hazard_id"] = [
        f"{og}_{src}_{i + 1}"
        for i, (og, src) in enumerate(zip(object_group.fillna("UNK"), source_s57.fillna("UNK")))
    ]

    type_map = {
        "OBSTRN": "Obstruction",
        "WRECKS": "Wreck",
        "PILPNT": "Pile",
        "UWTROC": "Underwater / awash rock",
    }

    gdf["hazard_type"] = object_group.map(type_map).fillna("Other")

    gdf["QUAPOS_final"] = _coalesce_string(gdf["QUAPOS"], gdf["QUAPOS_primitive"])

    quapos_code = (
        gdf["QUAPOS_final"]
        .astype("string")
        .str.extract(r"(\d+)", expand=False)
    )

    gdf["QUAPOS_code"] = quapos_code
    gdf["QUAPOS_num"] = pd.to_numeric(quapos_code, errors="coerce").astype("Float64")

    gdf["QUAPOS_method_note"] = np.select(
        [
            gdf["QUAPOS"].astype("string").notna(),
            gdf["QUAPOS"].astype("string").isna()
            & gdf["QUAPOS_primitive"].astype("string").notna(),
        ],
        [
            "Read directly from S-57 feature object",
            "Transferred from nearest S-57 primitive/vector record",
        ],
        default="No QUAPOS available",
    )

    quapos_labels = {
        1: "surveyed",
        2: "unsurveyed",
        3: "inadequately surveyed",
        4: "approximate",
        5: "position doubtful",
        6: "unreliable",
        7: "reported, not surveyed",
        8: "reported, not confirmed",
        9: "estimated",
        10: "precisely known",
        11: "calculated",
    }

    gdf["QUAPOS_label"] = (
        gdf["QUAPOS_num"]
        .astype("float64")
        .map(lambda x: quapos_labels.get(int(x), pd.NA) if not np.isnan(x) else pd.NA)
    )

    status_lc = gdf["STATUS"].astype("string").fillna("").str.lower()
    inform_lc = gdf["INFORM"].astype("string").fillna("").str.lower()
    status_inform = status_lc + " " + inform_lc

    quapos_num = gdf["QUAPOS_num"].astype("float64")

    pos_unverified = (
        quapos_num.between(2, 9, inclusive="both").fillna(False)
        | status_inform.str.contains(
            r"approx|doubt|reported|unconfirmed|existence",
            regex=True,
            na=False,
        )
    )

    pos_known_or_null = quapos_num.isna() | quapos_num.isin([1, 10, 11])

    main_hazard = object_group.isin(HAZARD_LAYERS)

    gdf["pos_unverified"] = pos_unverified
    gdf["pos_known_or_null"] = pos_known_or_null

    gdf["methodology_group"] = np.select(
        [
            main_hazard & pos_unverified,
            main_hazard & pos_known_or_null,
        ],
        [
            "02_unverified_reported_error",
            "01_known_hazard",
        ],
        default="other",
    )

    gdf["hazard_weight"] = np.select(
        [
            gdf["methodology_group"] == "02_unverified_reported_error",
            gdf["methodology_group"] == "01_known_hazard",
        ],
        [3, 1],
        default=0,
    ).astype("float64")

    gdf["known_hazard_weight"] = np.where(
        gdf["methodology_group"] == "01_known_hazard",
        1.0,
        0.0,
    )

    gdf["reported_error_weight"] = np.where(
        gdf["methodology_group"] == "02_unverified_reported_error",
        3.0,
        0.0,
    )

    return gdf


###############################################################################
# Neighbourhood scoring and buffering
###############################################################################

def calculate_neighbourhood_scores(
    hazards: gpd.GeoDataFrame,
    search_radius_nm: float = 2,
) -> gpd.GeoDataFrame:
    """Calculate 2 nautical mile neighbourhood hazard scores.

    Each hazard receives:
      - hazard_score: sum of all nearby hazard weights
      - known_hazard_score: sum of nearby known hazard weights
      - reported_error_score: sum of nearby unverified/reported weights
      - neighbour_hazard_count: number of nearby hazards excluding itself

    This is intentionally calculated on true hazard representative locations
    before any P2 buffering.
    """

    hazards = hazards.copy()

    if hazards.empty:
        hazards["hazard_score"] = []
        hazards["known_hazard_score"] = []
        hazards["reported_error_score"] = []
        hazards["neighbour_hazard_count"] = []
        return hazards

    search_radius_m = search_radius_nm * NAUTICAL_MILE_M

    hazards_m = hazards.to_crs(TARGET_METRIC_CRS)
    pts = hazards_m.copy()
    pts["geometry"] = representative_points(hazards_m)

    # Use spatial index query for efficient within-distance lookup.
    sindex = pts.sindex

    hazard_scores = np.zeros(len(pts), dtype="float64")
    known_scores = np.zeros(len(pts), dtype="float64")
    reported_scores = np.zeros(len(pts), dtype="float64")
    neighbour_counts = np.zeros(len(pts), dtype="int64")

    weights = pd.to_numeric(hazards_m["hazard_weight"], errors="coerce").fillna(0).to_numpy()
    known_weights = pd.to_numeric(hazards_m["known_hazard_weight"], errors="coerce").fillna(0).to_numpy()
    reported_weights = pd.to_numeric(hazards_m["reported_error_weight"], errors="coerce").fillna(0).to_numpy()

    geometries = pts.geometry.to_numpy()

    for i, geom in enumerate(geometries):
        # Query a buffered envelope first, then exact distance.
        possible_idx = list(sindex.query(geom.buffer(search_radius_m), predicate="intersects"))
        if not possible_idx:
            continue

        dists = np.array([geom.distance(geometries[j]) for j in possible_idx])
        within = np.array(possible_idx)[dists <= search_radius_m]

        hazard_scores[i] = weights[within].sum()
        known_scores[i] = known_weights[within].sum()
        reported_scores[i] = reported_weights[within].sum()
        neighbour_counts[i] = max(len(within) - 1, 0)

    hazards_m["hazard_score"] = hazard_scores
    hazards_m["known_hazard_score"] = known_scores
    hazards_m["reported_error_score"] = reported_scores
    hazards_m["neighbour_hazard_count"] = neighbour_counts

    return hazards_m.to_crs("EPSG:4326")


def make_buffered_product_geometry(
    hazards: gpd.GeoDataFrame,
    buffer_nm: float = 2,
    crs_m: str | rasterio.crs.CRS = TARGET_METRIC_CRS,
) -> gpd.GeoDataFrame:
    """Create P2 buffered hazard geometry.

    Important:
    The score fields are not recalculated here. The P2 product is the same
    scored hazard inventory, but displayed/spatially applied as a 2 nm buffer.
    """

    if hazards is None or hazards.empty:
        return hazards

    buffer_m = buffer_nm * NAUTICAL_MILE_M

    h = hazards.to_crs(crs_m).copy()
    h["geometry"] = h.geometry.buffer(buffer_m)
    h = clean_geometries(h, "buffered hazards")
    return h


###############################################################################
# Cache read/write helpers
###############################################################################

def write_hazard_cache(
    hazards: gpd.GeoDataFrame,
    cache_path: str | Path,
    layer: str = "hazard_features_cache",
    cache_format: CacheFormat = "gpkg",
    overwrite: bool = True,
) -> Path:
    """Write hazard cache as GeoPackage or GeoParquet."""

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and overwrite:
        cache_path.unlink()

    log_step(f"Writing hazard cache: {cache_path}")

    if cache_format == "gpkg":
        hazards.to_file(cache_path, layer=layer, driver="GPKG")
    elif cache_format in {"geoparquet", "parquet"}:
        hazards.to_parquet(cache_path, index=False)
    else:
        raise ValueError("cache_format must be 'gpkg' or 'geoparquet'.")

    return cache_path


def read_hazard_cache(
    cache_path: str | Path,
    layer: str = "hazard_features_cache",
) -> gpd.GeoDataFrame:
    """Read hazard cache from GeoPackage or GeoParquet."""

    cache_path = Path(cache_path)

    if cache_path.suffix.lower() in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(cache_path)

    return gpd.read_file(cache_path, layer=layer)


###############################################################################
# Stage 1: build NOAA S-57 hazard cache
###############################################################################

def build_noaa_s57_hazard_cache(
    work_dir: str | Path,
    mask_dir: Optional[str | Path] = None,
    regions: RegionsArg = "all",
    force_rebuild: bool = False,
    use_parallel: bool = True,
    n_workers: int = 6,
    cache_format: CacheFormat = "gpkg",
    max_quapos_join_distance_m: float = 25,
) -> dict[str, str]:
    """Build the reusable NOAA S-57 hazard cache.

    This corresponds to Stage 1 in the R hazard extraction workflow.

    Major steps
    -----------
    1. Read the NOAA ENC product catalog.
    2. Download and unzip ENC cells.
    3. Read hazard feature layers from native S-57 files.
    4. Optionally clip hazards to selected Eco Region mask AOIs.
    5. Read primitive/vector QUAPOS records where available.
    6. Transfer primitive QUAPOS to nearby hazard features.
    7. Add XY fields, hazard classification, and methodology weights.
    8. Calculate 2 nm neighbourhood hazard scores.
    9. Write a reusable cache.
    """

    set_proj_lib_if_needed()

    work_dir = Path(work_dir)
    download_dir = ensure_dir(work_dir / "enc_downloads")
    extract_dir = ensure_dir(work_dir / "enc_extracted")
    log_dir = ensure_dir(work_dir / "logs")
    cache_dir = ensure_dir(work_dir / "cache")

    error_log = log_dir / "s57_hazard_error_log.csv"
    layer_log = log_dir / "s57_available_layers_log.csv"

    region_tag = "all" if regions == "all" else "_".join(normalize_regions(regions))

    if cache_format == "gpkg":
        cache_path = cache_dir / f"noaa_s57_hazard_feature_cache_{region_tag}.gpkg"
    else:
        cache_path = cache_dir / f"noaa_s57_hazard_feature_cache_{region_tag}.parquet"

    cache_layer = "hazard_features_cache"

    if cache_path.exists() and not force_rebuild:
        log_step(f"Using existing hazard cache: {cache_path}")
        return {
            "cache_path": str(cache_path),
            "cache_layer": cache_layer,
            "cache_format": cache_format,
        }

    enc_catalog = read_enc_catalog_urls()
    log_step(f"Catalog cells found: {len(enc_catalog)}")

    # -------------------------------------------------------------------------
    # Download and unzip ENCs
    # -------------------------------------------------------------------------
    if use_parallel and len(enc_catalog) > 1 and Client is not None:
        log_step(f"Downloading/extracting ENCs with Dask: {n_workers} workers")
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True)
        client = Client(cluster)

        try:
            futures = [
                client.submit(
                    download_and_unzip_cell,
                    row.download_url,
                    download_dir,
                    extract_dir,
                    error_log,
                )
                for row in enc_catalog.itertuples()
            ]

            enc_records_nested = []
            for future in as_completed(futures):
                enc_records_nested.append(future.result())

        finally:
            client.close()
            cluster.close()
    else:
        enc_records_nested = [
            download_and_unzip_cell(
                row.download_url,
                download_dir,
                extract_dir,
                error_log,
            )
            for row in enc_catalog.itertuples()
        ]

    enc_records = [rec for sub in enc_records_nested for rec in sub if rec.s57_path.exists()]
    log_step(f"S-57 files available: {len(enc_records)}")

    if not enc_records:
        raise RuntimeError("No S-57 files available after download/extraction.")

    # -------------------------------------------------------------------------
    # List layers
    # -------------------------------------------------------------------------
    available_layers = pd.concat(
        [get_available_layers(rec.s57_path, error_log) for rec in enc_records],
        ignore_index=True,
    )
    available_layers.to_csv(layer_log, index=False)

    # -------------------------------------------------------------------------
    # Read hazard features
    # -------------------------------------------------------------------------
    if use_parallel and len(enc_records) > 1 and Client is not None:
        log_step(f"Reading hazard feature layers with Dask: {n_workers} workers")
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1, processes=True)
        client = Client(cluster)

        try:
            futures = [
                client.submit(
                    read_s57_hazards_for_file,
                    rec.s57_path,
                    available_layers,
                    HAZARD_LAYERS,
                    error_log,
                )
                for rec in enc_records
            ]

            hazard_parts = []
            for future in as_completed(futures):
                result = future.result()
                if result is not None and not result.empty:
                    hazard_parts.append(result)

        finally:
            client.close()
            cluster.close()
    else:
        hazard_parts = []
        for rec in enc_records:
            result = read_s57_hazards_for_file(
                rec.s57_path,
                available_layers,
                HAZARD_LAYERS,
                error_log,
            )
            if result is not None and not result.empty:
                hazard_parts.append(result)

    if not hazard_parts:
        raise RuntimeError("No hazard features read from S-57 files.")

    hazards_raw = pd.concat(hazard_parts, ignore_index=True)
    hazards_raw = clean_geometries(hazards_raw, "hazards_raw full cache")

    # -------------------------------------------------------------------------
    # Optional early clip to selected Eco Region masks.
    # This saves time and disk if only one or a few regions are needed.
    # -------------------------------------------------------------------------
    if mask_dir is not None:
        log_step("Clipping raw hazards to selected mask AOI")

        mask_records = get_mask_table(mask_dir, regions)
        mask_aois = [make_mask_aoi(rec.mask_path, aoi_grid_m=500) for rec in mask_records]

        mask_crs = mask_aois[0].crs
        mask_union_geom = unary_union([aoi.to_crs(mask_crs).geometry.iloc[0] for aoi in mask_aois])
        mask_union = gpd.GeoDataFrame({"id": [1]}, geometry=[mask_union_geom], crs=mask_crs)

        hazards_raw_m = hazards_raw.to_crs(mask_crs)
        hazards_raw_m = gpd.sjoin(
            hazards_raw_m,
            mask_union,
            how="inner",
            predicate="intersects",
        ).drop(columns=["index_right"], errors="ignore")

        hazards_raw = clean_geometries(
            hazards_raw_m.to_crs("EPSG:4326"),
            "hazards_raw mask union clipped",
        )

    # -------------------------------------------------------------------------
    # Read primitive QUAPOS sequentially.
    # This is intentionally sequential because primitive reads can be unstable
    # and memory-heavy in parallel GDAL sessions.
    # -------------------------------------------------------------------------
    source_paths = (
        hazards_raw.drop(columns="geometry")
        .drop_duplicates(subset=["source_path"])["source_path"]
        .astype(str)
        .tolist()
    )

    log_step("Reading QUAPOS primitive/vector records sequentially")

    quapos_parts = []

    for s57_path in source_paths:
        result = read_s57_quapos_primitives(s57_path, error_log)
        if result is not None and not result.empty:
            quapos_parts.append(result)

    quapos_primitives = (
        pd.concat(quapos_parts, ignore_index=True)
        if quapos_parts
        else None
    )

    hazards_with_quapos = attach_quapos_from_primitives(
        hazards=hazards_raw,
        quapos_primitives=quapos_primitives,
        error_log=error_log,
        max_join_distance_m=max_quapos_join_distance_m,
    )

    hazards = add_xy_fields(hazards_with_quapos)
    hazards = add_hazard_classification(hazards)

    log_step("Calculating 2 nm neighbourhood hazard scores")
    hazards = calculate_neighbourhood_scores(hazards, search_radius_nm=2)

    write_hazard_cache(
        hazards=hazards,
        cache_path=cache_path,
        layer=cache_layer,
        cache_format=cache_format,
        overwrite=True,
    )

    log_step(f"Wrote hazard cache: {cache_path}")

    return {
        "cache_path": str(cache_path),
        "cache_layer": cache_layer,
        "cache_format": cache_format,
    }


###############################################################################
# Stage 2: summarize hazards to Eco Region grids
###############################################################################

def safe_max(values: pd.Series | np.ndarray) -> float:
    """Return max, treating empty/all-NA groups as zero."""

    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if arr.empty:
        return 0.0
    return float(arr.max())


def summarize_hazards_to_grid(
    grid: gpd.GeoDataFrame,
    hazards: gpd.GeoDataFrame,
    product_type: Literal["P1_true_geometry", "P2_buffered_geometry"],
) -> gpd.GeoDataFrame:
    """Summarize hazard features onto a polygon grid.

    Important methodology note
    --------------------------
    Both P1 and P2 use MAX for score fields, not SUM.

    Why?
    ----
    Each hazard already carries its full 2 nm neighbourhood score. In the P2
    buffered product, overlapping buffers can cover the same grid cell many
    times. Summing would double/triple count the same neighbourhood score.
    MAX preserves the highest local hazard neighbourhood score.
    """

    grid = grid.copy()
    grid_crs = grid.crs

    if hazards is None or hazards.empty:
        out = grid.copy()
        out["product_type"] = product_type
        for col in [
            "total_hazard_count",
            "known_hazard_count",
            "unverified_reported_error_count",
            "obstruction_count",
            "wreck_count",
            "pile_count",
            "rock_count",
            "total_weighted_hazard_score",
            "known_hazard_score",
            "reported_error_score",
        ]:
            out[col] = 0
        return out

    hazards_m = hazards.to_crs(grid_crs)

    keep_cols = [
        "hazard_id",
        "object_group",
        "methodology_group",
        "hazard_score",
        "known_hazard_score",
        "reported_error_score",
        "geometry",
    ]

    hazards_join = hazards_m[[c for c in keep_cols if c in hazards_m.columns]].copy()

    joined = gpd.sjoin(
        grid[["region_id", "grid_id", "geometry"]],
        hazards_join,
        how="left",
        predicate="intersects",
    )

    def count_eq(series, value):
        return int((series == value).sum())

    grouped = joined.drop(columns="geometry").groupby("grid_id", dropna=False)

    summary = grouped.agg(
        total_hazard_count=("hazard_id", lambda x: int(x.notna().sum())),
        known_hazard_count=(
            "methodology_group",
            lambda x: count_eq(x, "01_known_hazard"),
        ),
        unverified_reported_error_count=(
            "methodology_group",
            lambda x: count_eq(x, "02_unverified_reported_error"),
        ),
        obstruction_count=("object_group", lambda x: count_eq(x, "OBSTRN")),
        wreck_count=("object_group", lambda x: count_eq(x, "WRECKS")),
        pile_count=("object_group", lambda x: count_eq(x, "PILPNT")),
        rock_count=("object_group", lambda x: count_eq(x, "UWTROC")),
        total_weighted_hazard_score=("hazard_score", safe_max),
        known_hazard_score=("known_hazard_score", safe_max),
        reported_error_score=("reported_error_score", safe_max),
    ).reset_index()

    out = grid.merge(summary, on="grid_id", how="left")
    out["product_type"] = product_type

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].fillna(0)

    score_cols = [
        "total_hazard_count",
        "known_hazard_count",
        "unverified_reported_error_count",
        "obstruction_count",
        "wreck_count",
        "pile_count",
        "rock_count",
        "total_weighted_hazard_score",
        "known_hazard_score",
        "reported_error_score",
    ]

    for col in score_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    return out


def rasterize_grid_summary_to_mask(
    grid_summary: gpd.GeoDataFrame,
    mask_path: str | Path,
    out_tif: str | Path,
    value_field: str = "total_weighted_hazard_score",
) -> Path:
    """Rasterize grid summary values back to the 100 m Eco Region mask grid."""

    if value_field not in grid_summary.columns:
        raise ValueError(f"value_field not found in grid_summary: {value_field}")

    mask_path = Path(mask_path)
    out_tif = Path(out_tif)
    out_tif.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(mask_path) as template:
        mask_arr = template.read(1)
        template_profile = template.profile.copy()

        valid_mask = mask_arr == 1

        grid_summary_m = grid_summary.to_crs(template.crs)

        shapes_to_burn = [
            (geom, float(value))
            for geom, value in zip(
                grid_summary_m.geometry,
                grid_summary_m[value_field],
            )
            if geom is not None and not geom.is_empty and pd.notna(value)
        ]

        burned = rasterize(
            shapes_to_burn,
            out_shape=template.shape,
            transform=template.transform,
            fill=np.nan,
            all_touched=True,
            dtype="float32",
        )

        burned[~valid_mask] = np.nan
        burned[burned <= 0] = np.nan

        profile = template_profile
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="LZW",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        log_step(f"Writing raster: {out_tif.name}")

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(burned.astype("float32"), 1)

    return out_tif


def safe_remove_file(path: str | Path) -> Path:
    """Remove an output file if possible; otherwise return a timestamped path."""

    path = Path(path)

    if path.exists():
        try:
            path.unlink()
        except Exception:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
            log_step(f"Existing file locked. Writing to: {new_path}")
            return new_path

    return path


def write_grid_summary(
    gdf: gpd.GeoDataFrame,
    out_path: str | Path,
    layer: str,
    output_format: CacheFormat = "gpkg",
) -> Path:
    """Write grid summary as GeoPackage or GeoParquet."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format == "gpkg":
        out_path = safe_remove_file(out_path)
        gdf.to_file(out_path, layer=layer, driver="GPKG")
        return out_path

    if output_format in {"geoparquet", "parquet"}:
        if out_path.suffix.lower() not in {".parquet", ".geoparquet"}:
            out_path = out_path.with_suffix(".parquet")
        if out_path.exists():
            out_path.unlink()
        gdf.to_parquet(out_path, index=False)
        return out_path

    raise ValueError("output_format must be 'gpkg' or 'geoparquet'.")


def run_hazard_summaries_by_ecoregion(
    work_dir: str | Path,
    mask_dir: str | Path,
    cache_path: str | Path,
    cache_layer: str = "hazard_features_cache",
    regions: RegionsArg = "all",
    grid_m: float = 500,
    value_field: str = "total_weighted_hazard_score",
    buffer_nm: float = 2,
    output_vector_format: CacheFormat = "gpkg",
) -> dict[str, dict[str, object]]:
    """Create P1/P2 Eco Region grid summaries and 100 m TIFFs.

    This corresponds to Stage 2 in the R hazard workflow.
    """

    work_dir = Path(work_dir)
    out_dir = ensure_dir(work_dir / "outputs_by_ecoregion")

    mask_records = get_mask_table(mask_dir, regions)

    hazards_all = read_hazard_cache(cache_path, layer=cache_layer)
    hazards_all = clean_geometries(hazards_all, "cached hazards")

    results: dict[str, dict[str, object]] = {}

    for rec in mask_records:
        region_id = rec.region_id
        mask_path = rec.mask_path

        log_step("===================================")
        log_step(f"Processing {region_id}")
        log_step("===================================")

        with rasterio.open(mask_path) as src:
            mask_crs = src.crs

        mask_aoi = make_mask_aoi(mask_path, aoi_grid_m=grid_m).to_crs(mask_crs)

        hazards_region = hazards_all.to_crs(mask_crs)

        hazards_region = gpd.sjoin(
            hazards_region,
            mask_aoi[["geometry"]],
            how="inner",
            predicate="intersects",
        ).drop(columns=["index_right"], errors="ignore")

        if not hazards_region.empty:
            hazards_region = clean_geometries(hazards_region, f"{region_id} hazards")

        log_step(f"{region_id} true hazards: {len(hazards_region)}")

        summary_grid = make_summary_grid_from_mask(
            mask_path=mask_path,
            grid_m=grid_m,
            region_id=region_id,
        )

        # ---------------------------------------------------------------------
        # P1: true hazard geometry
        # ---------------------------------------------------------------------
        p1_grid_summary = summarize_hazards_to_grid(
            grid=summary_grid,
            hazards=hazards_region,
            product_type="P1_true_geometry",
        )

        # ---------------------------------------------------------------------
        # P2: buffered hazard geometry.
        # Same scores, broader 2 nm geometry footprint.
        # ---------------------------------------------------------------------
        hazards_region_buffered = make_buffered_product_geometry(
            hazards=hazards_region,
            buffer_nm=buffer_nm,
            crs_m=mask_crs,
        )

        p2_grid_summary = summarize_hazards_to_grid(
            grid=summary_grid,
            hazards=hazards_region_buffered,
            product_type="P2_buffered_geometry",
        )

        p1_nonzero = int((p1_grid_summary[value_field] > 0).sum())
        p2_nonzero = int((p2_grid_summary[value_field] > 0).sum())

        log_step(f"{region_id} P1 nonzero cells: {p1_nonzero}")
        log_step(f"{region_id} P2 nonzero cells: {p2_nonzero}")

        if output_vector_format == "gpkg":
            out_vec_p1 = out_dir / f"{region_id}_P1_true_geometry_hazard_grid_{int(grid_m)}m.gpkg"
            out_vec_p2 = out_dir / f"{region_id}_P2_buffered_2nm_hazard_grid_{int(grid_m)}m.gpkg"
        else:
            out_vec_p1 = out_dir / f"{region_id}_P1_true_geometry_hazard_grid_{int(grid_m)}m.parquet"
            out_vec_p2 = out_dir / f"{region_id}_P2_buffered_2nm_hazard_grid_{int(grid_m)}m.parquet"

        out_tif_p1 = out_dir / f"{region_id}_P1_true_geometry_{value_field}_100m.tif"
        out_tif_p2 = out_dir / f"{region_id}_P2_buffered_2nm_{value_field}_100m.tif"

        out_vec_p1 = write_grid_summary(
            p1_grid_summary,
            out_vec_p1,
            layer=f"{region_id}_P1_true_geometry",
            output_format=output_vector_format,
        )

        out_vec_p2 = write_grid_summary(
            p2_grid_summary,
            out_vec_p2,
            layer=f"{region_id}_P2_buffered_2nm",
            output_format=output_vector_format,
        )

        rasterize_grid_summary_to_mask(
            grid_summary=p1_grid_summary,
            mask_path=mask_path,
            out_tif=out_tif_p1,
            value_field=value_field,
        )

        rasterize_grid_summary_to_mask(
            grid_summary=p2_grid_summary,
            mask_path=mask_path,
            out_tif=out_tif_p2,
            value_field=value_field,
        )

        p1_range = (
            float(np.nanmin(p1_grid_summary[value_field])),
            float(np.nanmax(p1_grid_summary[value_field])),
        )

        p2_range = (
            float(np.nanmin(p2_grid_summary[value_field])),
            float(np.nanmax(p2_grid_summary[value_field])),
        )

        log_step(f"{region_id} P1 score range: {p1_range}")
        log_step(f"{region_id} P2 score range: {p2_range}")

        results[region_id] = {
            "region_id": region_id,
            "vector_p1": str(out_vec_p1),
            "vector_p2": str(out_vec_p2),
            "tif_p1": str(out_tif_p1),
            "tif_p2": str(out_tif_p2),
            "n_hazards": int(len(hazards_region)),
            "p1_nonzero_cells": p1_nonzero,
            "p2_nonzero_cells": p2_nonzero,
            "p1_score_range": p1_range,
            "p2_score_range": p2_range,
        }

    return results


###############################################################################
# Full workflow wrapper
###############################################################################

def run_full_enc_hazard_ecoregion_workflow(
    work_dir: str | Path,
    mask_dir: str | Path,
    regions: RegionsArg = "all",
    grid_m: float = 500,
    value_field: str = "total_weighted_hazard_score",
    force_rebuild_cache: bool = False,
    delete_enc_zips_after_cache: bool = False,
    delete_extracted_enc_after_cache: bool = False,
    use_parallel: bool = True,
    n_workers: int = 6,
    buffer_nm: float = 2,
    cache_format: CacheFormat = "gpkg",
    output_vector_format: CacheFormat = "gpkg",
) -> dict[str, object]:
    """One clean user-facing workflow call.

    Parameters mirror the R function as closely as practical.
    """

    work_dir = Path(work_dir)

    cache = build_noaa_s57_hazard_cache(
        work_dir=work_dir,
        mask_dir=mask_dir,
        regions=regions,
        force_rebuild=force_rebuild_cache,
        use_parallel=use_parallel,
        n_workers=n_workers,
        cache_format=cache_format,
    )

    summaries = run_hazard_summaries_by_ecoregion(
        work_dir=work_dir,
        mask_dir=mask_dir,
        cache_path=cache["cache_path"],
        cache_layer=cache["cache_layer"],
        regions=regions,
        grid_m=grid_m,
        value_field=value_field,
        buffer_nm=buffer_nm,
        output_vector_format=output_vector_format,
    )

    if delete_enc_zips_after_cache:
        shutil.rmtree(work_dir / "enc_downloads", ignore_errors=True)

    if delete_extracted_enc_after_cache:
        shutil.rmtree(work_dir / "enc_extracted", ignore_errors=True)

    return {
        "cache": cache,
        "summaries": summaries,
    }


###############################################################################
# Verification helper
###############################################################################

def verify_hazard_cache(
    cache_path: str | Path,
    cache_layer: str = "hazard_features_cache",
    error_log: Optional[str | Path] = None,
) -> gpd.GeoDataFrame:
    """Print simple diagnostics for a hazard cache."""

    hazards = read_hazard_cache(cache_path, layer=cache_layer)

    print("\nFeature counts by object_group:")
    print(hazards.drop(columns="geometry").groupby("object_group").size().sort_values(ascending=False))

    print("\nFeature counts by source ENC and object_group:")
    print(
        hazards.drop(columns="geometry")
        .groupby(["source_s57", "object_group"])
        .size()
        .sort_values(ascending=False)
        .head(50)
    )

    print("\nQUAPOS coverage:")
    print(
        pd.DataFrame(
            {
                "total_features": [len(hazards)],
                "quapos_final_present": [hazards.get("QUAPOS_final", pd.Series(dtype=object)).notna().sum()],
                "quapos_primitive_present": [hazards.get("QUAPOS_primitive", pd.Series(dtype=object)).notna().sum()],
                "quapos_feature_present": [hazards.get("QUAPOS", pd.Series(dtype=object)).notna().sum()],
            }
        )
    )

    print("\nMethodology group counts:")
    print(hazards.drop(columns="geometry").groupby("methodology_group").size().sort_values(ascending=False))

    if error_log is not None and Path(error_log).exists():
        err = pd.read_csv(error_log)
        print("\nLogged read issues by layer/step:")
        print(err.groupby(["step", "layer", "message"]).size().sort_values(ascending=False).head(50))

    return hazards


###############################################################################
# Example calls
###############################################################################

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Example 1:
    # Single Eco Region test run.
    # -------------------------------------------------------------------------
    #
    # result_er6 = run_full_enc_hazard_ecoregion_workflow(
    #     work_dir=DEFAULT_WORK_DIR,
    #     mask_dir=DEFAULT_MASK_DIR,
    #     regions="ER_6",
    #     grid_m=500,
    #     value_field="total_weighted_hazard_score",
    #     force_rebuild_cache=False,
    #     delete_enc_zips_after_cache=False,
    #     delete_extracted_enc_after_cache=False,
    #     use_parallel=True,
    #     n_workers=6,
    #     buffer_nm=2,
    #     cache_format="gpkg",
    #     output_vector_format="gpkg",
    # )
    # print(result_er6)

    # -------------------------------------------------------------------------
    # Example 2:
    # All Eco Regions.
    # -------------------------------------------------------------------------
    #
    # result_all = run_full_enc_hazard_ecoregion_workflow(
    #     work_dir=DEFAULT_WORK_DIR,
    #     mask_dir=DEFAULT_MASK_DIR,
    #     regions="all",
    #     grid_m=500,
    #     value_field="total_weighted_hazard_score",
    #     force_rebuild_cache=True,
    #     delete_enc_zips_after_cache=False,
    #     delete_extracted_enc_after_cache=False,
    #     use_parallel=True,
    #     n_workers=6,
    #     buffer_nm=2,
    #     cache_format="gpkg",
    #     output_vector_format="gpkg",
    # )
    # print(result_all)

    # -------------------------------------------------------------------------
    # Example 3:
    # GeoParquet cache and outputs for faster future workflows.
    # -------------------------------------------------------------------------
    #
    # result_er6_parquet = run_full_enc_hazard_ecoregion_workflow(
    #     work_dir=DEFAULT_WORK_DIR,
    #     mask_dir=DEFAULT_MASK_DIR,
    #     regions="ER_6",
    #     grid_m=500,
    #     value_field="total_weighted_hazard_score",
    #     force_rebuild_cache=False,
    #     use_parallel=True,
    #     n_workers=6,
    #     buffer_nm=2,
    #     cache_format="geoparquet",
    #     output_vector_format="geoparquet",
    # )
    # print(result_er6_parquet)

    # -------------------------------------------------------------------------
    # Example 4:
    # Verify an existing cache.
    # -------------------------------------------------------------------------
    #
    # hazards = verify_hazard_cache(
    #     cache_path=DEFAULT_WORK_DIR / "cache" / "noaa_s57_hazard_feature_cache_all.gpkg",
    #     cache_layer="hazard_features_cache",
    #     error_log=DEFAULT_WORK_DIR / "logs" / "s57_hazard_error_log.csv",
    # )

    print(
        "ENC hazard extraction Python script loaded. Uncomment an example call "
        "at the bottom of this file, or import run_full_enc_hazard_ecoregion_workflow()."
    )