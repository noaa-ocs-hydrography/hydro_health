"""
HHM Simple Analysis Workflow - Python Translation
=================================================

This script is a Python translation and modernization of the R-based HHM
"Simple Analysis" workflow.

The workflow is intentionally kept as a SINGLE SCRIPT FILE for easier sharing,
review, and testing before being converted into a larger Python package.

Main capabilities
-----------------
1. Optional pre-processing mosaic of Eco Region hazard rasters.
2. Stage 1:
   Create a spatially variable decay coefficient from:
      - hurricane exposure
      - tidal current exposure
      - marine hazards / human debris
3. Stage 2:
   Create:
      - survey end year raster
      - Initial Survey Score, ISS
      - Desired Survey Score, DSS
      - Present Survey Score, PSS
      - Hydrographic Gap, HG = DSS - PSS
4. Run at:
      - 100 m national / offshore mosaic scale
      - 20 m nearshore Eco Region folders
5. Optional Dask parallel execution across Eco Regions.
6. Optional Dask-backed raster chunking through rioxarray/xarray.
7. Optional GeoParquet helpers for future vector hazard workflows.

Recommended environment
-----------------------
Install with conda/mamba where possible:

    mamba create -n hhm_py -c conda-forge \
        python=3.11 rasterio rioxarray xarray dask distributed geopandas \
        pyarrow matplotlib numpy pandas shapely fiona pyproj

    conda activate hhm_py

Notes
-----
- The code assumes that rasters are already in a projected CRS suitable for
  meter-based analysis, usually EPSG:6350 in this HHM workflow.
- Nearest-neighbor resampling is used for categorical rasters.
- Bilinear resampling is used for continuous rasters.
- The bathymetry raster is treated as the authoritative spatial reference.
- The scientific scoring logic follows the supplied R script methodology.

Author
------
Converted from Stephanie Watson's R HHM workflow into Python.
"""


###############################################################################
# Imports
###############################################################################

from __future__ import annotations

import math
import re
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge as rio_merge
    from rasterio.transform import from_origin
    from rasterio.warp import reproject
except ImportError as exc:
    raise ImportError(
        "Missing rasterio. Install with: conda install -c conda-forge rasterio"
    ) from exc

try:
    import rioxarray  # noqa: F401 - required to activate .rio accessor
    import xarray as xr
except ImportError as exc:
    raise ImportError(
        "Missing xarray/rioxarray. Install with: conda install -c conda-forge xarray rioxarray"
    ) from exc

try:
    from dask.distributed import Client, LocalCluster, as_completed
except ImportError:
    Client = None
    LocalCluster = None
    as_completed = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


###############################################################################
# User-adjustable defaults
###############################################################################

DEFAULT_ROOT_DIR = Path(
    r"N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/HHM_simple"
)

DEFAULT_HAZARD_ECOREGION_DIR = Path(
    r"N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards/outputs_by_ecoregion"
)

DEFAULT_TARGET_CRS = "EPSG:6350"
DEFAULT_ANALYSIS_YEAR = 2026


###############################################################################
# Small data structures
###############################################################################

Resolution = Literal["100m", "20m"]
StageMode = Literal["both", "stage1", "stage2"]
EcoRegions = Literal["all"] | str | Sequence[str]


@dataclass(frozen=True)
class HHMConfig:
    """Container for input/output directories for one HHM resolution."""

    resolution: Resolution
    input_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class TileRecord:
    """One processing unit.

    For 100 m runs, this is usually one national/offshore raster folder.
    For 20 m runs, this is usually one Eco Region folder such as ER_1 or ER_6.
    """

    tile_id: str
    input_dir: Path
    output_dir: Path


###############################################################################
# Logging
###############################################################################

def log_step(text: str) -> None:
    """Print a timestamped processing message."""

    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {text}", flush=True)


###############################################################################
# Directory and tile discovery
###############################################################################

def get_hhm_config(
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    resolution: Resolution = "100m",
    input_variant_100m: str = "Offshore_tiles_100m",
    output_variant_100m: str = "Offshore_tiles_100m",
) -> HHMConfig:
    """Return the configured HHM input and output folders.

    Parameters
    ----------
    root_dir
        Root HHM simple workflow directory.
    resolution
        Either "100m" or "20m".
    input_variant_100m
        Name of the 100 m input subfolder. Common options are:
          - "Offshore_tiles_100m"
          - "Eco_Region_All_tiles_100m"
          - "Eco_Region_All_tiles_Buffered_100m"
    output_variant_100m
        Name of the 100 m output subfolder.

    Returns
    -------
    HHMConfig
        Input/output locations for the chosen resolution.
    """

    root_dir = Path(root_dir)

    if resolution == "100m":
        return HHMConfig(
            resolution=resolution,
            input_dir=root_dir / "inputs" / input_variant_100m,
            output_dir=root_dir / "outputs" / output_variant_100m,
        )

    if resolution == "20m":
        return HHMConfig(
            resolution=resolution,
            input_dir=root_dir / "inputs" / "Nearshore_tiles_20m",
            output_dir=root_dir / "outputs" / "Nearshore_tiles_20m",
        )

    raise ValueError("resolution must be either '100m' or '20m'.")


def normalize_eco_regions(eco_regions: EcoRegions) -> str | list[str]:
    """Normalize user Eco Region selection.

    Accepts:
      - "all"
      - "ER_6"
      - ["ER_2", "ER_6"]
    """

    if eco_regions == "all":
        return "all"

    if isinstance(eco_regions, str):
        return [eco_regions]

    return [str(x) for x in eco_regions]


def get_tile_table(config: HHMConfig, eco_regions: EcoRegions = "all") -> list[TileRecord]:
    """Build the list of processing tiles/folders.

    Why this exists
    ---------------
    The R workflow has two operating modes:

    - 100 m: process one national/offshore folder.
    - 20 m: process one or more Eco Region folders.

    This function standardizes both cases into a list of TileRecord objects.
    """

    if config.resolution == "100m":
        return [
            TileRecord(
                tile_id="Eco_region_all_tiles_100m",
                input_dir=config.input_dir,
                output_dir=config.output_dir,
            )
        ]

    selected_regions = normalize_eco_regions(eco_regions)

    eco_region_dirs = sorted(
        p for p in config.input_dir.iterdir()
        if p.is_dir() and re.match(r"^ER_[0-9]+$", p.name)
    )

    if selected_regions != "all":
        selected_set = set(selected_regions)
        eco_region_dirs = [p for p in eco_region_dirs if p.name in selected_set]

        if not eco_region_dirs:
            raise FileNotFoundError(
                f"No matching Eco Region folders found for: {', '.join(selected_regions)}"
            )

    return [
        TileRecord(
            tile_id=p.name,
            input_dir=p,
            output_dir=config.output_dir / p.name,
        )
        for p in eco_region_dirs
    ]


###############################################################################
# Raster discovery and IO
###############################################################################

def find_raster(directory: str | Path, pattern: str) -> Path:
    """Find exactly one raster in a directory by regex pattern.

    Parameters
    ----------
    directory
        Directory to search.
    pattern
        Case-insensitive regex pattern applied to file names.

    Returns
    -------
    Path
        Matching raster path.

    Raises
    ------
    FileNotFoundError
        If no matching raster is found.
    RuntimeError
        If more than one matching raster is found.

    Notes
    -----
    This intentionally mirrors the strict behavior of the R function:
    if the workflow expects one bathymetry raster, it should fail loudly
    when zero or multiple candidates exist.
    """

    directory = Path(directory)

    raster_paths = [
        p for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and p.name.lower() != "thumbs.db"
    ]

    matches = [p for p in raster_paths if re.search(pattern, p.name, flags=re.IGNORECASE)]

    if len(matches) != 1:
        listed = "\n".join(f"  - {p.name}" for p in matches[:25])
        raise RuntimeError(
            f"Expected one raster matching pattern '{pattern}' in '{directory}', "
            f"but found {len(matches)}.\n{listed}"
        )

    return matches[0]


def open_raster(
    path: str | Path,
    chunks: Optional[dict[str, int]] = None,
    masked: bool = True,
) -> xr.DataArray:
    """Open a raster with rioxarray.

    Parameters
    ----------
    path
        Raster path.
    chunks
        Optional Dask chunks, for example {"x": 2048, "y": 2048}.
        If None, data can still be processed but may not be lazy/parallel.
    masked
        Whether to convert nodata values to NaN.

    Returns
    -------
    xarray.DataArray
        A single-band raster DataArray.
    """

    da = rioxarray.open_rasterio(path, masked=masked, chunks=chunks)

    # Most HHM rasters are single-band. Keep them as 2D arrays.
    if "band" in da.dims and da.sizes.get("band", 1) == 1:
        da = da.squeeze("band", drop=True)

    return da


def raster_resampling_method(method: Literal["near", "bilinear"]) -> Resampling:
    """Translate HHM method names into rasterio resampling enums."""

    if method == "near":
        return Resampling.nearest
    if method == "bilinear":
        return Resampling.bilinear
    raise ValueError("method must be 'near' or 'bilinear'.")


def align_to_ref(
    raster: xr.DataArray,
    ref: xr.DataArray,
    method: Literal["near", "bilinear"] = "near",
) -> xr.DataArray:
    """Align a raster to a reference raster.

    Why this matters
    ----------------
    HHM raster math is cell-by-cell. That means every input must have the
    same grid, resolution, CRS, transform, and shape before calculations.

    The R workflow used terra::resample / terra::project. Here we use
    rioxarray's reproject_match, which delegates to rasterio/GDAL.

    Important behavior
    ------------------
    If CRS metadata is missing or differs but the rasters appear to be on the
    same grid, this function can force the reference CRS before matching.
    This mirrors the practical fix used in the R workflow for rasters that
    visually aligned in GIS but had inconsistent CRS metadata.
    """

    if not ref.rio.crs:
        raise ValueError("Reference raster is missing CRS.")

    if not raster.rio.crs:
        warnings.warn(
            "Input raster is missing CRS. Assigning reference CRS before alignment."
        )
        raster = raster.rio.write_crs(ref.rio.crs, inplace=False)

    try:
        same_crs = raster.rio.crs == ref.rio.crs
        same_shape = raster.shape == ref.shape
        same_transform = raster.rio.transform() == ref.rio.transform()

        if same_crs and same_shape and same_transform:
            return raster

        return raster.rio.reproject_match(
            ref,
            resampling=raster_resampling_method(method),
        )

    except Exception:
        # Last practical fallback: force CRS to ref and try again.
        warnings.warn(
            "Raster alignment encountered an issue. Forcing reference CRS and retrying."
        )
        raster = raster.rio.write_crs(ref.rio.crs, inplace=False)
        return raster.rio.reproject_match(
            ref,
            resampling=raster_resampling_method(method),
        )


def write_hhm_raster(
    raster: xr.DataArray,
    path: str | Path,
    ref: Optional[xr.DataArray] = None,
    overwrite: bool = True,
    dtype: str = "float32",
    compress: str = "LZW",
) -> Path:
    """Write a GeoTIFF and verify that CRS is preserved.

    Parameters
    ----------
    raster
        Raster to write.
    path
        Output GeoTIFF path.
    ref
        Optional reference raster whose CRS/transform should be copied.
    overwrite
        Whether to overwrite existing file.
    dtype
        Output data type.
    compress
        GeoTIFF compression.

    Returns
    -------
    Path
        Written output path.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not overwrite:
        log_step(f"Skipping existing file: {path.name}")
        return path

    if ref is not None:
        raster = raster.rio.write_crs(ref.rio.crs, inplace=False)

    if not raster.rio.crs:
        raise ValueError(f"Output raster has missing CRS before write: {path.name}")

    log_step(f"Writing {path.name}")

    raster = raster.astype(dtype)

    raster.rio.to_raster(
        path,
        compress=compress,
        tiled=True,
        BIGTIFF="YES",
        dtype=dtype,
    )

    # Verify CRS after write.
    with rasterio.open(path) as src:
        if not src.crs:
            raise ValueError(f"CRS was lost during write: {path.name}")

    return path


###############################################################################
# Chunk-safe scientific helper functions
###############################################################################

def clean_year_values(x: np.ndarray, analysis_year: int) -> np.ndarray:
    """Clean survey year/date values.

    Accepts values stored as:
      - YYYY
      - YYYYMM
      - YYYYMMDD

    Invalid values, future years, and placeholder values become NaN.
    """

    x = np.asarray(x, dtype="float64")
    out = np.rint(x)

    out[np.isnan(out) | (out <= 1)] = np.nan

    yyyymmdd = (~np.isnan(out)) & (out >= 10_000_000)
    out[yyyymmdd] = np.floor(out[yyyymmdd] / 10_000)

    yyyymm = (~np.isnan(out)) & (out >= 10_000) & (out < 10_000_000)
    out[yyyymm] = np.floor(out[yyyymm] / 100)

    invalid = (~np.isnan(out)) & ((out < 1800) | (out > analysis_year))
    out[invalid] = np.nan

    return out


def bin_hurricane_values(x: np.ndarray) -> np.ndarray:
    """Convert hurricane count/exposure to HHM change bins."""

    x = np.asarray(x, dtype="float64")
    out = np.full_like(x, np.nan, dtype="float64")

    out[(~np.isnan(x)) & (x <= 0)] = 1
    out[(~np.isnan(x)) & (x > 0) & (x <= 4)] = 2
    out[(~np.isnan(x)) & (x > 4) & (x <= 10)] = 3
    out[(~np.isnan(x)) & (x > 10) & (x <= 20)] = 4
    out[(~np.isnan(x)) & (x > 20)] = 5

    return out


def bin_tidal_values(x: np.ndarray) -> np.ndarray:
    """Convert tidal exposure to HHM change bins."""

    x = np.asarray(x, dtype="float64")
    out = np.full_like(x, np.nan, dtype="float64")

    out[(~np.isnan(x)) & (x < 0.1)] = 1
    out[(~np.isnan(x)) & (x >= 0.1) & (x < 0.2)] = 2
    out[(~np.isnan(x)) & (x >= 0.2) & (x < 0.5)] = 3
    out[(~np.isnan(x)) & (x >= 0.5) & (x < 1.0)] = 4
    out[(~np.isnan(x)) & (x >= 1.0)] = 5

    return out


def bin_debris_values(x: np.ndarray) -> np.ndarray:
    """Convert marine hazard / human debris score to HHM change bins."""

    x = np.asarray(x, dtype="float64")
    out = np.full_like(x, np.nan, dtype="float64")

    out[(~np.isnan(x)) & (x <= 0)] = 1
    out[(~np.isnan(x)) & (x > 0) & (x <= 1)] = 2
    out[(~np.isnan(x)) & (x > 1) & (x <= 5)] = 3
    out[(~np.isnan(x)) & (x > 5) & (x <= 10)] = 4
    out[(~np.isnan(x)) & (x > 10)] = 5

    return out


def bin_ukc_values(
    ukc: np.ndarray,
    shallow_max: float,
    mid_max: float,
) -> np.ndarray:
    """Bin Under Keel Clearance into Desired Survey Score values."""

    ukc = np.asarray(ukc, dtype="float64")
    out = np.full_like(ukc, np.nan, dtype="float64")

    out[(~np.isnan(ukc)) & (ukc <= 1)] = 100
    out[(~np.isnan(ukc)) & (ukc > 1) & (ukc <= shallow_max)] = 80
    out[(~np.isnan(ukc)) & (ukc > shallow_max) & (ukc <= mid_max)] = 30
    out[(~np.isnan(ukc)) & (ukc > mid_max)] = 10

    return out


def bin_depth_values(
    depth: np.ndarray,
    shallow_max: float,
    mid_max: float,
) -> np.ndarray:
    """Fallback DSS binning based on absolute depth when UKC is unavailable."""

    depth = np.asarray(depth, dtype="float64")
    out = np.full_like(depth, np.nan, dtype="float64")

    out[(~np.isnan(depth)) & (depth < 2)] = 100
    out[(~np.isnan(depth)) & (depth >= 2) & (depth <= shallow_max)] = 80
    out[(~np.isnan(depth)) & (depth > shallow_max) & (depth <= mid_max)] = 30
    out[(~np.isnan(depth)) & (depth > mid_max)] = 10

    return out


def make_dss_values(
    ukc: np.ndarray,
    depth: np.ndarray,
    slope: np.ndarray,
) -> np.ndarray:
    """Create Desired Survey Score, DSS.

    DSS represents how much survey quality is desired based on navigation risk.

    Logic
    -----
    1. Prefer UKC where available.
    2. Fall back to depth where UKC is missing.
    3. Use slope to distinguish simple, moderate, and complex terrain.

    Slope classes
    -------------
    - simple:   slope < 0.5
    - moderate: 0.5 <= slope < 1
    - complex: slope >= 1
    """

    ukc = np.asarray(ukc, dtype="float64")
    depth = np.asarray(depth, dtype="float64")
    slope = np.asarray(slope, dtype="float64")

    ukc = ukc.copy()
    ukc[(~np.isnan(ukc)) & (ukc < 0)] = np.nan

    out = np.full_like(ukc, np.nan, dtype="float64")

    use_ukc = ~np.isnan(ukc)
    use_depth = np.isnan(ukc) & ~np.isnan(depth)

    simple = (~np.isnan(slope)) & (slope < 0.5)
    moderate = (~np.isnan(slope)) & (slope >= 0.5) & (slope < 1.0)
    complex_ = (~np.isnan(slope)) & (slope >= 1.0)

    mask = use_ukc & simple
    out[mask] = bin_ukc_values(ukc[mask], shallow_max=20, mid_max=50)

    mask = use_ukc & moderate
    out[mask] = bin_ukc_values(ukc[mask], shallow_max=30, mid_max=75)

    mask = use_ukc & complex_
    out[mask] = bin_ukc_values(ukc[mask], shallow_max=40, mid_max=100)

    mask = use_depth & simple
    out[mask] = bin_depth_values(depth[mask], shallow_max=20, mid_max=50)

    mask = use_depth & moderate
    out[mask] = bin_depth_values(depth[mask], shallow_max=30, mid_max=75)

    mask = use_depth & complex_
    out[mask] = bin_depth_values(depth[mask], shallow_max=40, mid_max=100)

    return out


###############################################################################
# Stage 1 calculation kernel
###############################################################################

def _stage1_numpy_kernel(
    bathy: np.ndarray,
    survey_date: np.ndarray,
    sand_mud: np.ndarray,
    tidal: np.ndarray,
    hurricanes: np.ndarray,
    debris: np.ndarray,
    analysis_year: int,
    divisor_terms: int,
    decay_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numpy kernel for Stage 1.

    This function is written in vectorized NumPy so it can be called by
    xarray.apply_ufunc over Dask chunks.

    Returns
    -------
    tuple of arrays
        decay_coefficient, hurricane_bin, tidal_bin, debris_bin, tidal_exposure
    """

    survey_year = clean_year_values(survey_date, analysis_year)
    survey_age = analysis_year - survey_year
    survey_age[(~np.isnan(survey_age)) & (survey_age < 0)] = np.nan

    depth = np.abs(bathy)

    valid_domain = (~np.isnan(bathy)) & (~np.isnan(survey_year))
    sand_mud_positive = (~np.isnan(sand_mud)) & (sand_mud > 0)

    # Hurricanes only apply in sand/mud and depths <= 40 m.
    hurricane_applicable = valid_domain & sand_mud_positive & (depth <= 40)
    hurricane_cond = np.where(hurricane_applicable, hurricanes, np.nan)
    hurricane_bin = bin_hurricane_values(hurricane_cond)

    # Tidal exposure only applies in sand/mud and depths <= 20 m.
    tidal_applicable = valid_domain & sand_mud_positive & (depth <= 20)
    tidal_exposure = np.where(tidal_applicable, tidal * survey_age, np.nan)
    tidal_bin = bin_tidal_values(tidal_exposure)

    # Marine debris/hazard term exists across the full valid model domain.
    # NA debris is treated as zero known hazard, not as no data.
    debris_clean = np.where(valid_domain & np.isnan(debris), 0, debris)
    debris_bin = np.where(valid_domain, bin_debris_values(debris_clean), np.nan)

    n_terms = (
        (~np.isnan(hurricane_bin)).astype("float64")
        + (~np.isnan(tidal_bin)).astype("float64")
        + (~np.isnan(debris_bin)).astype("float64")
    )

    change_sum = (
        np.where(np.isnan(hurricane_bin), 0, hurricane_bin)
        + np.where(np.isnan(tidal_bin), 0, tidal_bin)
        + np.where(np.isnan(debris_bin), 0, debris_bin)
    )

    decay_coefficient = ((change_sum - n_terms) * decay_rate) / divisor_terms

    decay_coefficient = np.where(valid_domain & (n_terms == 0), 0, decay_coefficient)
    decay_coefficient = np.where(valid_domain, decay_coefficient, np.nan)

    return (
        decay_coefficient.astype("float32"),
        hurricane_bin.astype("float32"),
        tidal_bin.astype("float32"),
        debris_bin.astype("float32"),
        tidal_exposure.astype("float32"),
    )


def create_decay_coefficient(
    input_dir: str | Path,
    output_dir: str | Path,
    resolution: Resolution,
    tile_id: str,
    analysis_year: int = DEFAULT_ANALYSIS_YEAR,
    divisor_terms: int = 3,
    decay_rate: float = 0.022,
    overwrite: bool = True,
    chunks: Optional[dict[str, int]] = None,
) -> dict[str, Path]:
    """Stage 1: create the HHM decay coefficient and change-agent bins.

    Inputs
    ------
    - bathymetry
    - survey date
    - sand/mud mask
    - tidal current
    - hurricane exposure
    - marine hazards / human debris

    Outputs
    -------
    - decay coefficient
    - hurricane bin
    - tidal bin
    - human debris bin
    - tidal exposure

    Why this step matters
    ---------------------
    The decay coefficient controls how quickly the Initial Survey Score
    degrades into the Present Survey Score. Higher environmental or human
    change exposure means a faster loss of survey confidence.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        chunks = {"x": 2048, "y": 2048}

    log_step("Reading Stage 1 rasters")

    bathy = open_raster(find_raster(input_dir, r"bathy.*mosaic"), chunks=chunks)
    survey_date = open_raster(find_raster(input_dir, r"survey.*date|survey.*end"), chunks=chunks)
    sand_mud = open_raster(find_raster(input_dir, r"sand.*mud.*mask"), chunks=chunks)
    tidal = open_raster(find_raster(input_dir, r"marine.*current|tidal|currents"), chunks=chunks)
    hurricanes = open_raster(find_raster(input_dir, r"hurricane"), chunks=chunks)
    debris = open_raster(find_raster(input_dir, r"marine.*hazard|human.*debris"), chunks=chunks)

    # Bathymetry controls the output grid.
    ref = bathy

    survey_date = align_to_ref(survey_date, ref, "near")
    sand_mud = align_to_ref(sand_mud, ref, "near")
    tidal = align_to_ref(tidal, ref, "bilinear")
    hurricanes = align_to_ref(hurricanes, ref, "near")
    debris = align_to_ref(debris, ref, "near")

    log_step("Computing Stage 1 bins and decay coefficient with Dask/xarray")

    outputs = xr.apply_ufunc(
        _stage1_numpy_kernel,
        bathy,
        survey_date,
        sand_mud,
        tidal,
        hurricanes,
        debris,
        kwargs={
            "analysis_year": analysis_year,
            "divisor_terms": divisor_terms,
            "decay_rate": decay_rate,
        },
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[[], [], [], [], []],
        dask="parallelized",
        vectorize=False,
        output_dtypes=["float32", "float32", "float32", "float32", "float32"],
    )

    output_names = [
        "decay_coefficient",
        "change_hurricanes_bin",
        "change_tidal_bin",
        "change_human_debris_bin",
        "tidal_exposure",
    ]

    output_paths: dict[str, Path] = {}

    for name, da in zip(output_names, outputs):
        da = da.rio.write_crs(ref.rio.crs, inplace=False)
        da.rio.write_transform(ref.rio.transform(), inplace=True)

        out_path = output_dir / f"{name}_{tile_id}.tif"
        output_paths[name] = write_hhm_raster(
            da,
            out_path,
            ref=ref,
            overwrite=overwrite,
            dtype="float32",
        )

    return output_paths


###############################################################################
# Stage 2 calculation kernel
###############################################################################

def _stage2_numpy_kernel(
    survey_year: np.ndarray,
    iss: np.ndarray,
    ukc: np.ndarray,
    bathy: np.ndarray,
    slope: np.ndarray,
    decay_coefficient: np.ndarray,
    analysis_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numpy kernel for Stage 2.

    Returns
    -------
    tuple of arrays
        survey_end_year, ISS, DSS, PSS, HG
    """

    survey_year_clean = clean_year_values(survey_year, analysis_year)
    survey_age = analysis_year - survey_year_clean

    survey_age[(~np.isnan(survey_age)) & (survey_age < 0)] = np.nan
    survey_age[(~np.isnan(survey_age)) & (survey_age > 200)] = 200

    ukc = np.asarray(ukc, dtype="float64")
    ukc = ukc.copy()
    ukc[(~np.isnan(ukc)) & (ukc < 0)] = np.nan

    pss = iss * np.exp(-decay_coefficient * survey_age)
    pss[(~np.isnan(pss)) & (pss < 0)] = 0
    pss[(~np.isnan(pss)) & (pss > 110)] = 110

    depth = np.abs(bathy)
    dss = make_dss_values(ukc, depth, slope)

    hydrographic_gap = dss - pss

    return (
        survey_year_clean.astype("float32"),
        iss.astype("float32"),
        dss.astype("float32"),
        pss.astype("float32"),
        hydrographic_gap.astype("float32"),
    )


def create_survey_scores(
    input_dir: str | Path,
    output_dir: str | Path,
    resolution: Resolution,
    tile_id: str,
    analysis_year: int = DEFAULT_ANALYSIS_YEAR,
    make_plots: bool = True,
    overwrite: bool = True,
    chunks: Optional[dict[str, int]] = None,
) -> dict[str, Path]:
    """Stage 2: create ISS, DSS, PSS, and Hydrographic Gap rasters.

    Inputs
    ------
    - survey year/date
    - ISS / supersession score
    - UKC
    - slope
    - bathymetry
    - Stage 1 decay coefficient

    Outputs
    -------
    - survey_end_year
    - ISS
    - DSS
    - PSS
    - HG = DSS - PSS

    Scientific interpretation
    -------------------------
    ISS is the initial condition.
    PSS is the present survey score after time-decay.
    DSS is the desired survey score based on UKC/depth and slope complexity.
    HG is the hydrographic gap, where larger values indicate a stronger
    potential need for updated survey effort.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        chunks = {"x": 2048, "y": 2048}

    log_step("Reading Stage 2 rasters")

    survey_year = open_raster(find_raster(input_dir, r"survey.*date|survey.*end"), chunks=chunks)
    iss = open_raster(find_raster(input_dir, r"(^|_)iss|supersession.*score"), chunks=chunks)
    ukc = open_raster(find_raster(input_dir, r"ukc|under.*keel.*clearance"), chunks=chunks)
    slope = open_raster(find_raster(input_dir, r"slope"), chunks=chunks)
    bathy = open_raster(find_raster(input_dir, r"bathy.*mosaic"), chunks=chunks)

    decay_path = find_raster(output_dir, r"decay.*coefficient")
    decay_coefficient = open_raster(decay_path, chunks=chunks)

    ref = bathy

    survey_year = align_to_ref(survey_year, ref, "near")
    iss = align_to_ref(iss, ref, "near")
    ukc = align_to_ref(ukc, ref, "bilinear")
    slope = align_to_ref(slope, ref, "bilinear")
    bathy = align_to_ref(bathy, ref, "bilinear")
    decay_coefficient = align_to_ref(decay_coefficient, ref, "bilinear")

    log_step("Computing Stage 2 scores with Dask/xarray")

    outputs = xr.apply_ufunc(
        _stage2_numpy_kernel,
        survey_year,
        iss,
        ukc,
        bathy,
        slope,
        decay_coefficient,
        kwargs={"analysis_year": analysis_year},
        input_core_dims=[[], [], [], [], [], []],
        output_core_dims=[[], [], [], [], []],
        dask="parallelized",
        vectorize=False,
        output_dtypes=["float32", "float32", "float32", "float32", "float32"],
    )

    output_names = ["survey_end_year", "ISS", "DSS", "PSS", "HG"]
    output_paths: dict[str, Path] = {}

    for name, da in zip(output_names, outputs):
        da = da.rio.write_crs(ref.rio.crs, inplace=False)
        da.rio.write_transform(ref.rio.transform(), inplace=True)

        out_path = output_dir / f"{name}_{tile_id}.tif"
        output_paths[name] = write_hhm_raster(
            da,
            out_path,
            ref=ref,
            overwrite=overwrite,
            dtype="float32",
        )

    if make_plots:
        log_step("Creating Stage 2 plots")
        try:
            make_three_panel_plot(
                output_paths["survey_end_year"],
                output_paths["ISS"],
                output_paths["PSS"],
                output_dir / f"survey_end_year_ISS_PSS_{tile_id}.png",
            )
            make_dss_pss_plot(
                output_paths["DSS"],
                output_paths["PSS"],
                output_dir / f"DSS_PSS_{tile_id}.png",
            )
            make_gap_dss_pss_plot(
                output_paths["HG"],
                output_paths["DSS"],
                output_paths["PSS"],
                output_dir / f"HG_DSS_PSS_{tile_id}.png",
            )
        except Exception as exc:
            warnings.warn(f"Plotting failed for {tile_id}: {exc}")

    return output_paths


###############################################################################
# Hazard raster mosaic helper
###############################################################################

def merge_hazard_tifs_resampled(
    input_dir: str | Path,
    pattern: str,
    output_tif: str,
    target_crs: str = DEFAULT_TARGET_CRS,
    res_m: float = 100,
    overwrite: bool = True,
    mosaic_method: Literal["max", "first"] = "max",
) -> Path:
    """Mosaic Eco Region hazard rasters into one common grid.

    This mirrors the R pre-processing function that:
      1. finds matching Eco Region hazard TIFFs,
      2. assigns/forces EPSG:6350,
      3. aligns rasters to a shared snapped 100 m origin,
      4. mosaics overlapping cells using max.

    Parameters
    ----------
    input_dir
        Folder containing ER hazard TIFFs.
    pattern
        Regex pattern for selecting TIFFs.
    output_tif
        Output filename.
    target_crs
        CRS to assign/reproject to, usually EPSG:6350.
    res_m
        Output grid resolution in meters.
    overwrite
        Whether to overwrite an existing mosaic.
    mosaic_method
        "max" is recommended for hazard scores because overlapping values
        should preserve the highest hazard score.

    Returns
    -------
    Path
        Output mosaic path.
    """

    input_dir = Path(input_dir)
    out_path = input_dir / output_tif

    if out_path.exists() and not overwrite:
        log_step(f"Skipping existing mosaic: {out_path}")
        return out_path

    tif_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in {".tif", ".tiff"}
        and re.search(pattern, p.name, flags=re.IGNORECASE)
    )

    if not tif_files:
        raise FileNotFoundError(f"No matching TIFF files found in {input_dir}")

    log_step(f"Found {len(tif_files)} hazard rasters to mosaic")

    # First pass: determine snapped output bounds.
    bounds = []
    for tif in tif_files:
        with rasterio.open(tif) as src:
            bounds.append(src.bounds)

    xmin_all = math.floor(min(b.left for b in bounds) / res_m) * res_m
    xmax_all = math.ceil(max(b.right for b in bounds) / res_m) * res_m
    ymin_all = math.floor(min(b.bottom for b in bounds) / res_m) * res_m
    ymax_all = math.ceil(max(b.top for b in bounds) / res_m) * res_m

    width = int(round((xmax_all - xmin_all) / res_m))
    height = int(round((ymax_all - ymin_all) / res_m))
    transform = from_origin(xmin_all, ymax_all, res_m, res_m)

    if mosaic_method == "max":
        mosaic = np.full((height, width), np.nan, dtype="float32")
    else:
        mosaic = np.full((height, width), np.nan, dtype="float32")

    dst_crs = rasterio.crs.CRS.from_string(target_crs)

    for tif in tif_files:
        log_step(f"Aligning and adding {tif.name}")

        aligned = np.full((height, width), np.nan, dtype="float32")

        with rasterio.open(tif) as src:
            src_crs = src.crs or dst_crs

            reproject(
                source=rasterio.band(src, 1),
                destination=aligned,
                src_transform=src.transform,
                src_crs=src_crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )

        if mosaic_method == "max":
            mosaic = np.fmax(mosaic, aligned)
        else:
            mosaic = np.where(np.isnan(mosaic), aligned, mosaic)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": transform,
        "compress": "LZW",
        "tiled": True,
        "BIGTIFF": "YES",
        "nodata": np.nan,
    }

    log_step(f"Writing mosaic: {out_path}")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mosaic, 1)

    return out_path


def merge_p2_buffered_hazards(
    input_dir: str | Path = DEFAULT_HAZARD_ECOREGION_DIR,
) -> Path:
    """Convenience wrapper for P2 buffered hazard mosaics."""

    return merge_hazard_tifs_resampled(
        input_dir=input_dir,
        pattern=r"P2_buffered_2nm_total_weighted_hazard_score_100m\.tif$",
        output_tif="P2_merged_hazard_score_2nm_buffer_100m_EPSG6350.tif",
        target_crs=DEFAULT_TARGET_CRS,
        res_m=100,
        overwrite=True,
        mosaic_method="max",
    )


def merge_p1_true_geometry_hazards(
    input_dir: str | Path = DEFAULT_HAZARD_ECOREGION_DIR,
) -> Path:
    """Convenience wrapper for P1 true-geometry hazard mosaics."""

    return merge_hazard_tifs_resampled(
        input_dir=input_dir,
        pattern=r"P1_true_geometry_total_weighted_hazard_score_100m\.tif$",
        output_tif="P1_merged_hazard_score_2nm_geometry_100m_EPSG6350.tif",
        target_crs=DEFAULT_TARGET_CRS,
        res_m=100,
        overwrite=True,
        mosaic_method="max",
    )


###############################################################################
# Optional GeoParquet helpers for future vector hazard workflows
###############################################################################

def read_vector_any(path: str | Path):
    """Read a vector dataset from GeoPackage, Shapefile, or GeoParquet.

    This helper is not required for the raster-only HHM simple workflow,
    but is included because the broader marine hazard workflow can benefit
    from GeoParquet for faster cache reads/writes.
    """

    if gpd is None:
        raise ImportError("geopandas is required for vector/GeoParquet support.")

    path = Path(path)

    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        return gpd.read_parquet(path)

    return gpd.read_file(path)


def write_geoparquet(gdf, path: str | Path, overwrite: bool = True) -> Path:
    """Write a GeoDataFrame to GeoParquet.

    Why GeoParquet?
    ---------------
    GeoPackage is convenient, but large NOAA hazard caches can become slow.
    GeoParquet is faster, smaller, and better suited for parallel/cloud use.
    """

    if gpd is None:
        raise ImportError("geopandas is required for vector/GeoParquet support.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and overwrite:
        path.unlink()

    gdf.to_parquet(path, index=False)

    return path


###############################################################################
# Plotting
###############################################################################

def _read_for_plot(path: str | Path) -> tuple[np.ndarray, rasterio.coords.BoundingBox]:
    """Read raster data and bounds for simple matplotlib plotting."""

    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
        bounds = src.bounds
    return arr, bounds


def _imshow_raster(ax, raster_path: str | Path, title: str, cmap: str, vmin=None, vmax=None):
    """Shared raster plotting helper."""

    arr, bounds = _read_for_plot(raster_path)
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    img = ax.imshow(
        arr,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return img


def make_three_panel_plot(
    survey_year_path: str | Path,
    iss_path: str | Path,
    pss_path: str | Path,
    output_file: str | Path,
) -> Path:
    """Create survey year / ISS / PSS overview plot."""

    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    output_file = Path(output_file)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    img = _imshow_raster(axes[0], survey_year_path, "Survey end year", "YlGnBu")
    fig.colorbar(img, ax=axes[0], shrink=0.75)

    img = _imshow_raster(axes[1], iss_path, "Initial Survey Score (ISS)", "Greens", 0, 110)
    fig.colorbar(img, ax=axes[1], shrink=0.75)

    img = _imshow_raster(axes[2], pss_path, "Present Survey Score (PSS)", "Greens", 0, 110)
    fig.colorbar(img, ax=axes[2], shrink=0.75)

    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return output_file


def make_dss_pss_plot(
    dss_path: str | Path,
    pss_path: str | Path,
    output_file: str | Path,
) -> Path:
    """Create DSS / PSS comparison plot."""

    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    output_file = Path(output_file)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    img = _imshow_raster(axes[0], dss_path, "Desired Survey Score (DSS)", "Greens", 0, 110)
    fig.colorbar(img, ax=axes[0], shrink=0.75)

    img = _imshow_raster(axes[1], pss_path, "Present Survey Score (PSS)", "Greens", 0, 110)
    fig.colorbar(img, ax=axes[1], shrink=0.75)

    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return output_file


def make_gap_dss_pss_plot(
    hg_path: str | Path,
    dss_path: str | Path,
    pss_path: str | Path,
    output_file: str | Path,
) -> Path:
    """Create HG / DSS / PSS comparison plot."""

    if plt is None:
        raise ImportError("matplotlib is required for plotting.")

    output_file = Path(output_file)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

    img = _imshow_raster(axes[0], hg_path, "Hydrographic Gap (DSS - PSS)", "RdYlBu")
    fig.colorbar(img, ax=axes[0], shrink=0.75)

    img = _imshow_raster(axes[1], dss_path, "Desired Survey Score (DSS)", "Greens", 0, 110)
    fig.colorbar(img, ax=axes[1], shrink=0.75)

    img = _imshow_raster(axes[2], pss_path, "Present Survey Score (PSS)", "Greens", 0, 110)
    fig.colorbar(img, ax=axes[2], shrink=0.75)

    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return output_file


###############################################################################
# Tile-level and workflow-level orchestration
###############################################################################

def run_one_tile(
    tile: TileRecord,
    resolution: Resolution,
    stages: StageMode = "both",
    analysis_year: int = DEFAULT_ANALYSIS_YEAR,
    make_plots: bool = True,
    overwrite: bool = True,
    chunks: Optional[dict[str, int]] = None,
) -> dict[str, object]:
    """Run one tile or Eco Region.

    This is the unit of work submitted to Dask when parallel execution is used.
    """

    log_step(f"Processing {tile.tile_id} at {resolution}")

    tile.output_dir.mkdir(parents=True, exist_ok=True)

    stage1_complete = False
    stage2_complete = False

    if stages in {"both", "stage1"}:
        create_decay_coefficient(
            input_dir=tile.input_dir,
            output_dir=tile.output_dir,
            resolution=resolution,
            tile_id=tile.tile_id,
            analysis_year=analysis_year,
            overwrite=overwrite,
            chunks=chunks,
        )
        stage1_complete = True

    if stages in {"both", "stage2"}:
        create_survey_scores(
            input_dir=tile.input_dir,
            output_dir=tile.output_dir,
            resolution=resolution,
            tile_id=tile.tile_id,
            analysis_year=analysis_year,
            make_plots=make_plots,
            overwrite=overwrite,
            chunks=chunks,
        )
        stage2_complete = True

    return {
        "tile_id": tile.tile_id,
        "input_dir": str(tile.input_dir),
        "output_dir": str(tile.output_dir),
        "stage1_complete": stage1_complete,
        "stage2_complete": stage2_complete,
    }


def run_hhm_simple_workflow(
    resolution: Resolution = "100m",
    stages: StageMode = "both",
    eco_regions: EcoRegions = "all",
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    analysis_year: int = DEFAULT_ANALYSIS_YEAR,
    make_plots: bool = True,
    overwrite: bool = True,
    use_dask: bool = False,
    n_workers: int = 4,
    threads_per_worker: int = 1,
    chunks: Optional[dict[str, int]] = None,
    input_variant_100m: str = "Offshore_tiles_100m",
    output_variant_100m: str = "Offshore_tiles_100m",
) -> pd.DataFrame:
    """User-facing HHM simple workflow.

    Parameters
    ----------
    resolution
        "100m" or "20m".
    stages
        "both", "stage1", or "stage2".
    eco_regions
        For 20 m processing:
          - "all"
          - "ER_6"
          - ["ER_2", "ER_6"]
        Ignored for 100 m processing.
    root_dir
        Root workflow folder.
    analysis_year
        Year used to compute survey age.
    make_plots
        Whether to create PNG plots for Stage 2.
    overwrite
        Whether to overwrite outputs.
    use_dask
        If True, process multiple tiles/Eco Regions in parallel using
        dask.distributed. Raster chunks are also Dask-backed through xarray.
    n_workers
        Number of Dask workers for tile-level parallelism.
    threads_per_worker
        Threads per Dask worker.
    chunks
        Raster chunk size, for example {"x": 2048, "y": 2048}.
    input_variant_100m, output_variant_100m
        100 m input/output folder variants.

    Returns
    -------
    pandas.DataFrame
        Processing summary table.
    """

    if chunks is None:
        chunks = {"x": 2048, "y": 2048}

    config = get_hhm_config(
        root_dir=root_dir,
        resolution=resolution,
        input_variant_100m=input_variant_100m,
        output_variant_100m=output_variant_100m,
    )

    tile_table = get_tile_table(config, eco_regions=eco_regions)

    log_step(f"Prepared {len(tile_table)} tile(s) for processing")

    if not use_dask or len(tile_table) == 1:
        results = [
            run_one_tile(
                tile=tile,
                resolution=resolution,
                stages=stages,
                analysis_year=analysis_year,
                make_plots=make_plots,
                overwrite=overwrite,
                chunks=chunks,
            )
            for tile in tile_table
        ]
        return pd.DataFrame(results)

    if Client is None or LocalCluster is None:
        raise ImportError(
            "dask.distributed is not installed. Install with: "
            "conda install -c conda-forge dask distributed"
        )

    log_step(f"Starting Dask LocalCluster with {n_workers} workers")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
    )

    client = Client(cluster)

    try:
        futures = [
            client.submit(
                run_one_tile,
                tile,
                resolution,
                stages,
                analysis_year,
                make_plots,
                overwrite,
                chunks,
            )
            for tile in tile_table
        ]

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            log_step(f"Completed {result['tile_id']}")

        return pd.DataFrame(results)

    finally:
        client.close()
        cluster.close()


###############################################################################
# Example calls
###############################################################################

if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Example 1:
    # Mosaic Eco Region P2 buffered hazards into one national EPSG:6350 raster.
    # -------------------------------------------------------------------------
    #
    # merge_p2_buffered_hazards(
    #     input_dir=DEFAULT_HAZARD_ECOREGION_DIR
    # )

    # -------------------------------------------------------------------------
    # Example 2:
    # Mosaic Eco Region P1 true-geometry hazards into one national EPSG:6350 raster.
    # -------------------------------------------------------------------------
    #
    # merge_p1_true_geometry_hazards(
    #     input_dir=DEFAULT_HAZARD_ECOREGION_DIR
    # )

    # -------------------------------------------------------------------------
    # Example 3:
    # Run both stages for the single 100 m offshore/national folder.
    # -------------------------------------------------------------------------
    #
    # results_100m = run_hhm_simple_workflow(
    #     resolution="100m",
    #     stages="both",
    #     root_dir=DEFAULT_ROOT_DIR,
    #     input_variant_100m="Offshore_tiles_100m",
    #     output_variant_100m="Offshore_tiles_100m",
    #     use_dask=False,
    #     make_plots=True,
    # )
    # print(results_100m)

    # -------------------------------------------------------------------------
    # Example 4:
    # Run only Stage 2 after Stage 1 decay coefficient already exists.
    # -------------------------------------------------------------------------
    #
    # results_stage2 = run_hhm_simple_workflow(
    #     resolution="100m",
    #     stages="stage2",
    #     root_dir=DEFAULT_ROOT_DIR,
    #     input_variant_100m="Offshore_tiles_100m",
    #     output_variant_100m="Offshore_tiles_100m",
    #     use_dask=False,
    #     make_plots=True,
    # )
    # print(results_stage2)

    # -------------------------------------------------------------------------
    # Example 5:
    # Run both stages for all 20 m Eco Region folders in parallel.
    # -------------------------------------------------------------------------
    #
    # results_20m_all = run_hhm_simple_workflow(
    #     resolution="20m",
    #     stages="both",
    #     eco_regions="all",
    #     root_dir=DEFAULT_ROOT_DIR,
    #     use_dask=True,
    #     n_workers=6,
    #     threads_per_worker=1,
    #     make_plots=True,
    # )
    # print(results_20m_all)

    # -------------------------------------------------------------------------
    # Example 6:
    # Run both stages only for ER_6.
    # -------------------------------------------------------------------------
    #
    # results_er6 = run_hhm_simple_workflow(
    #     resolution="20m",
    #     stages="both",
    #     eco_regions="ER_6",
    #     root_dir=DEFAULT_ROOT_DIR,
    #     use_dask=False,
    #     make_plots=True,
    # )
    # print(results_er6)

    # -------------------------------------------------------------------------
    # Example 7:
    # Run both stages for multiple selected Eco Regions.
    # -------------------------------------------------------------------------
    #
    # results_selected = run_hhm_simple_workflow(
    #     resolution="20m",
    #     stages="both",
    #     eco_regions=["ER_2", "ER_6"],
    #     root_dir=DEFAULT_ROOT_DIR,
    #     use_dask=True,
    #     n_workers=2,
    #     threads_per_worker=1,
    #     make_plots=True,
    # )
    # print(results_selected)

    print(
        "HHM Python script loaded. Uncomment an example call at the bottom of "
        "this file, or import run_hhm_simple_workflow() from another script."
    )