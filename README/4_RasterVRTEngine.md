# RasterVRTEngine

A distributed and local virtual mosaic pipeline designed to generate GDAL Virtual Rasters (VRTs) over spatial datasets hosted on AWS S3 or local filesystems without duplicating or reprojecting source imagery.

## 💡 Overview

`RasterVRTEngine` reads Cloud Optimized GeoTIFFs (COGs) and GeoTIFFs natively from S3 buckets via GDAL's Virtual File System (`/vsis3/`) using Dask workers, or directly from local directories. It optimizes cloud byte-range read patterns and dynamically organizes spatial datasets into grouped master VRT catalog mosaics (`mosaic_<file_type>_<bin_key>.vrt` or `mosaic_<provider>_DTS.vrt`).

The engine exposes two distinct aggregation pipelines depending on the dataset specification:
1. **BlueTopo Pipeline:** Preserves native UTM projections without warping. It inspects each tile's spatial reference header, strictly bins tiles by their native UTM EPSG code (e.g., `EPSG_32617`, `EPSG_32618`), and builds separate per-zone master VRTs using bilinear interpolation (`allowProjectionDifference=False`).
2. **DigitalCoast & DistanceToShore Pipeline:** Handles `DigitalCoast`, `Digital_Coast_Manual_Downloads`, and `DistanceToShore` datasets. It groups tiles by provider folder, skips archived `/original/` source subdirectories, preserves each provider's native NoData value, and assumes tiles are pre-projected to **`EPSG:32617`**—logging a non-blocking warning via `self.write_message()` if any tile deviates from `EPSG:32617` before compiling per-provider VRTs with nearest-neighbor resampling.

---

## ⚙️ Core Architecture & State

- **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
- **Execution Modes:** Controlled via `param_lookup['env']` (`'aws'` for Dask + `/vsis3/` S3 streaming, or `'local'` for local filesystem traversal).
- **Spatial Reference Buffers:** Pre-loads projected CRS metadata (`pyproj.database.query_crs_info`) on initialization.
- **Target Variable Glob Routing:** Automatically maps `file_type` keys to filename glob patterns (with additional suffix exclusions on BlueTopo `elevation` to filter out uncertainty, slope, ISS, survey end date, and decay rasters):

  | Variable Target (`file_type`) | Glob Signature Matching |
  | :--- | :--- |
  | `elevation` | `*[0-9].tiff` |
  | `uncertainty` | `*_unc.tiff` |
  | `survey_end_date` | `*survey_end_date.tiff` |
  | `slope` | `*_slope.tiff` |
  | `shore` | `*_shore.tif` |
  | `catzoc_decay_all` | `*ISS_all*.tiff` |
  | `catzoc_decay_latest` | `*ISS_latest*.tiff` |
  | `NCMP` | `*.tif` |

---

## 🛠️ Module-Level Functions

### `_set_gdal_s3_options()`
- **What it does:** Injects optimized GDAL configuration options to minimize latency and HTTP overhead over `/vsis3/`.
- **The Specifics:** Disables directory listing scans on open (`GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`), merges consecutive HTTP byte-range requests (`GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES`), enables HTTP multiplexing, and configures an in-memory per-thread byte-range cache:
  $$\text{VSI Cache Size} = \text{25,000,000 Bytes (25 MB)}$$

### `_read_geotiff_metadata(raw_prefix)`
- **What it does:** Extracts header metadata from S3 GeoTIFFs across Dask workers without downloading full raster arrays.
- **The Specifics:** Opens the raster over `/vsis3/`, reads Band 1's NoData value, enforces traditional GIS axis mapping (`OAMS_TRADITIONAL_GIS_ORDER`), auto-identifies the EPSG authority code (with WKT fallback), and parses the provider folder name from the S3 key.
- **Inputs:** `raw_prefix` (`str`): S3 URI or bucket-relative key for the target GeoTIFF.
- **Outputs:** `dict | None`: Dictionary containing `bin_key`, `vsi_path`, `relative_s3_key`, `nodata`, `epsg`, and `wkt`.

---

## 🛠️ Method Reference

### `run(output_folder, ecoregion, data_type, file_type="", output_prefix="", manual_downloads=False)`
- **What it does:** Main entry point orchestrating tile discovery, metadata binning, and master VRT generation locally or on AWS.
- **The Specifics:**
  - In `'aws'` mode, initializes Dask workers, discovers tiles via `s3fs`, excludes `/original/` (and `/distance_to_shore/` when processing standard variables), builds VRTs in a temporary directory, and uploads them to S3.
  - In `'local'` mode, traverses local ecoregion/provider folders, skips `mask_*` files, `unused_providers`, and `original` directories, and writes VRTs directly to disk.
- **Inputs:**
  - `output_folder` (`str`): Root local directory for outputs or temporary scratch space.
  - `ecoregion` (`str`): Target ecoregion identifier (e.g., `'ER_3'`).
  - `data_type` (`str`): Dataset pipeline (`'BlueTopo'`, `'DigitalCoast'`, or `'DistanceToShore'`).
  - `file_type` (`str`, optional): Variable key from `glob_lookup` (required unless `data_type == 'DistanceToShore'`).
  - `output_prefix` (`str`, optional): Optional leading S3 or local path prefix.
  - `manual_downloads` (`bool`, optional): When `True`, also processes `Digital_Coast_Manual_Downloads` provider folders.

### `get_s3_bluetopo_geotiffs(geotiffs)` & `get_local_bluetopo_geotiffs(geotiffs)`
- **What it does:** Groups BlueTopo tiles strictly by their native UTM EPSG code.
- **The Specifics:** Reads spatial references in parallel via Dask (S3) or sequentially via GDAL (local) and bins tile paths under `EPSG_<code>` keys with `nodata_val = -9999.0`.
- **Inputs:** `geotiffs` (`list`): List of S3 paths or `pathlib.Path` objects.
- **Outputs:** `dict`: Mapping of `EPSG_<code>` to `{'tiles': [...], 'nodata_val': -9999.0}`.

### `get_s3_digitalcoast_geotiffs(geotiffs)` & `get_local_digitalcoast_geotiffs(geotiffs)`
- **What it does:** Bins `DigitalCoast`, `Digital_Coast_Manual_Downloads`, and `DistanceToShore` tiles by provider and extracts native NoData values.
- **The Specifics:** Filters out any `/original/` paths, assigns `'primary_crs': 'EPSG:32617'`, and calls `self.write_message()` to log a warning if any tile's detected EPSG code is not `32617` (without triggering `gdal.Warp`).
- **Inputs:** `geotiffs` (`list`): List of S3 paths or `pathlib.Path` objects.
- **Outputs:** `dict`: Mapping of provider names to `{'tiles': [...], 'nodata_val': <float>, 'primary_crs': 'EPSG:32617'}`.

### `build_output_vrts(outputs, output_geotiffs, data_type, file_type="", s3_output_path="")`
- **What it does:** Compiles master VRT files using `gdal.BuildVRT` and uploads them to S3 when running in AWS.
- **The Specifics:**
  - Names outputs `mosaic_<provider>_DTS.vrt` for `DistanceToShore` and `mosaic_<file_type>_<bin_key>.vrt` for all other types.
  - Uses `resampleAlg='bilinear'` and `allowProjectionDifference=False` for `BlueTopo`; uses `resampleAlg='near'` and `allowProjectionDifference=True` for `DigitalCoast`, `Digital_Coast_Manual_Downloads`, and `DistanceToShore`.
- **Inputs:**
  - `outputs` (`pathlib.Path`): Local directory where `.vrt` files are initially written.
  - `output_geotiffs` (`dict`): Grouped tile dictionary returned by the `get_*_geotiffs` methods.
  - `data_type` (`str`): Dataset type controlling resampling and projection options.
  - `file_type` (`str`, optional): Variable name embedded in the output VRT filename.
  - `s3_output_path` (`str`, optional): Destination S3 key prefix when `env == 'aws'`.

---

## 🚀 Quick Start / Usage Example

```python
from hydro_health.engines.RasterVRTEngine import RasterVRTEngine

# 1. Configure execution environment ('aws' or 'local')
param_lookup = {'env': 'aws'}

# 2. Instantiate the VRT generation engine
vrt_engine = RasterVRTEngine(param_lookup=param_lookup)

# 3. Build per-provider DigitalCoast VRTs (including manual downloads)
vrt_engine.run(
    output_folder="/tmp/vrt_runs",
    ecoregion="ER_3",
    data_type="DigitalCoast",
    file_type="elevation",
    manual_downloads=True
)
```