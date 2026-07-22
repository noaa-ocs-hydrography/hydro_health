# DigitalCoastS3Engine

A specialized spatial ingestion pipeline designed to scrape, spatial-filter, optimize, and mirror NOAA NOAA/Digital Coast bathymetric elevation datasets (DEMs) to private storage.

## 💡 Overview

The `DigitalCoastS3Engine` targets NOAA's data distribution networks. It operates in two major parallel phases:
1. **Index Syncing:** Dispatches asynchronous workers to download regional spatial index shapefiles mapping out individual data providers.
2. **Intersection Streaming:** Evaluates spatial intersections against active AOI bounding frames, directly stream-downloads intersected datasets to local disk scratch spaces, reformats the rasters to Cloud-Optimized GeoTIFFs (COGs), and pushes the optimized imagery directly back into S3.

---

## ⚙️ Core Architecture & State

* **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
* **Environment Overrides:** Configures aggressive UTF-8 console encodings for RedHat compatibility and mutes verbose `pyproj` internal logs.
* **Disk Optimization:** Forces an explicit configuration fallback routing heavy download streams and GDAL temporary caches out of ephemeral RAM structures into a concrete physical user directory (`~/temp_scratch`).

---

## 🛠️ Module-Level Orchestrators

### `_download_tile_index(param_inputs)`
* **What it does:** Scans the target public NOAA S3 storage bucket to pull data provider index files.
* **The Specifics:** Checks for data availability paths, matches `dem` data markers, downloads index zip files safely to local paths, and ensures existing structural indicators are not redundantly processed.
* **Inputs:** `param_inputs` (list) - `[download_link, provider_folder, param_lookup, outputs]`.

### `_download_intersected_datasets(param_inputs)`
* **What it does:** The primary processing engine for downloading spatial grid sections that cut across input criteria.
* **The Specifics:** 1. Opens local shapefile layers with `pyogrio` and fixes broken geometries via `make_valid()`.
  2. Runs a spatial join (`sjoin`) using the query tile parameters to filter out non-intersecting grids.
  3. Uses a robust `urllib3.util.Retry` session pool to download the source files in chunk segments (`1MB`).
  4. Converts the file into an optimized, tiled GeoTIFF using **DEFLATE** compression and a type-3 floating-point predictor.
  5. Computes dynamic internal structural overviews (resampling pyramids) if the target layout dimensions align, then uploads the final file to the target output bucket.
* **Inputs:** `param_inputs` (list) - `[ecoregion, tile_gdf, shp_path, outputs, param_lookup]`.
* **Outputs:** `str` - Stem name of the processed index track.

---

## 🛠️ Method Reference

### `run(tile_gdf)`
* **What it does:** Core entry method controlling pipeline workflow loops.
* **The Specifics:** Creates an iterative tracker file (`processed_providers.log`) to provide resume-on-failure safety across executions. Loops across unique ecoregions, isolates local work folders inside temporary environments, groups data queries into parallel block sizes of 5, gathers worker executions, tracks progress markers to disk, and pushes files back to storage.
* **Inputs:** `tile_gdf` (`gpd.GeoDataFrame`) - Geospatial targets tracking expected regional elements.

### `check_tile_index_areas(tile_index_shapefiles, outputs)`
* **What it does:** Validates survey dimensions before triggering intensive down-streams.
* **The Specifics:** Re-projects data frames using an **Albers Equal Area (`EPSG:9822`)** framework to perform accurate physical size evaluations, filtering out tiny patches that fall below engine size allowances.
* **Inputs:** `tile_index_shapefiles` (`list[pathlib.Path]`) - Paths to candidate index references.
* **Outputs:** `list` - Validated index paths that meet the required size constraints.

### `unzip_all_files(output_folder)`
* **What it does:** Internal workspace decompression utility.
* **The Specifics:** Recursively extracts nested `.zip` structures using `zipfile.ZipFile` directly to their local parent directory and deletes the archive file immediately to free up scratch space.

### `upload_file_to_s3(temp_file_path, s3_prefix)` / `upload_files_to_s3(provider_folder)`
* **What it does:** Direct storage sync hooks.
* **The Specifics:** Explicitly maps multi-thread storage uploads pointing directly to target regions (`us-east-2`), resolving path extensions automatically.

---

## 🚀 Quick Start / Usage Example

```python
import geopandas as gpd
from hydro_health.engines.DigitalCoastEngine import DigitalCoastS3Engine

# 1. Prepare operational context metadata configurations
param_lookup = {
    'env': 'production',
    'output_directory': MockField('/mnt/fast_scratch/digital_coast')
}

# 2. Instantiate pipeline processing system
engine = DigitalCoastS3Engine(param_lookup=param_lookup)

# 3. Import tracking geometry index frame 
study_area_gdf = gpd.read_file("coastal_aoi_bounds.geojson")

# 4. Trigger localized synchronization sweeps
engine.run(tile_gdf=study_area_gdf)