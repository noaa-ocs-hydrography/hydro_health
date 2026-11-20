# DigitalCoastEngine

A spatial ingestion and preprocessing pipeline designed to discover, spatially filter, optimize, and mirror NOAA Digital Coast bathymetric digital elevation models (DEMs). The engine supports both local filesystem storage and high-throughput mirroring to AWS S3, featuring native GDAL C++ raster reprojection, Cloud-Optimized GeoTIFF (COG) generation, and unmanaged memory management for large-scale distributed runs.

---

## 💡 Overview

`DigitalCoastEngine` interfaces with NOAA's public distribution endpoints and S3 buckets through two parallelized phases managed by Dask:

1. **Index Syncing & Spatial Filtering:** Discovers and retrieves provider tile index shapefiles across specified ecoregion boundaries, reprojects geometries, verifies survey footprint size thresholds, and computes spatial intersections against the target Area of Interest (AOI).
2. **Streaming, Reprojection & Ingestion:** Downloads intersecting raw DEM tiles, archives the unmodified source GeoTIFFs into an `original/` directory, streams and reprojects rasters into `EPSG:32617` (UTM Zone 17N) as tiled GeoTIFFs with internal overview pyramids, and writes outputs locally or directly to AWS S3 (`us-east-2`).

Depending on `param_lookup['env']`, execution automatically routes to either a local disk pipeline (`_run_local`) or a distributed AWS S3 workflow (`_run_s3`).

---

## ⚙️ Core Architecture & Enhancements

* **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
* **Dual Pipeline Routing:**
* **AWS Pipeline (`env == 'aws'`):** Employs isolated ephemeral scratch directories, batches provider processing into chunks of 5, maintains an idempotent resume ledger (`processed_providers.log`), uploads raw GeoTIFFs to an `original/` S3 prefix, uploads reprojected COGs to the primary prefix, and synchronizes accompanying index shapefiles.
* **Local Pipeline (`env != 'aws'`):** Downloads raw rasters into a local `original/` subfolder, reprojects them directly to the provider parent folder, and cleans up empty or unneeded provider directories.


* **Native GDAL C++ Reprojection & COG Generation:**
* Uses `gdal.Warp` for block-by-block streaming reprojection directly to `EPSG:32617`.
* Configures a strict 128 MB GDAL warp memory limit (`warpMemoryLimit=128 * 1024 * 1024`) and 256 MB global cache (`gdal.SetCacheMax(268435456)`).
* Employs `DEFLATE` compression with 512×512 block tiling, `BIGTIFF=IF_SAFER`, and automatic predictor selection (Predictor `3` for floating-point DEMs, Predictor `2` for integers).
* Dynamically builds internal overviews (`[2, 4, 8, 16]`) using `AVERAGE` resampling when raster dimensions permit.


* **Low-Level Memory Optimization:**
* Invokes `libc.malloc_trim(0)` via `ctypes` alongside Python garbage collection (`gc.collect()`) after intensive operations, forcing Linux glibc to release unmanaged C/GDAL memory arenas back to the operating system.


* **Environment Overrides:** Enforces UTF-8 locale encodings (`PYTHONIOENCODING`, `LC_ALL`, `LANG`), sets `SHAPE_RESTORE_SHX=YES`, and suppresses verbose `pyproj` internal logs.
* **Resilient Network Sessions:** Uses dedicated `requests.Session` pools with `urllib3.util.Retry` exponential backoff (retrying statuses `404/429, 500, 502, 503, 504`) and 1 MB streaming chunks.

---

## 🛠️ Module-Level Helpers & Orchestrators

### `_trim_memory()`

Forces the Linux glibc allocator to release unmanaged C/GDAL memory arenas back to the OS via `ctypes.CDLL("libc.so.6").malloc_trim(0)` following garbage collection.

### `_reproject_raster_gdal(src_path, dst_path, target_crs="EPSG:32617")`

Performs native C++ streaming raster reprojection using `gdal.Warp`.

* **Key Parameters:**
* `src_path` (`str | pathlib.Path`) — Source GeoTIFF.
* `dst_path` (`str | pathlib.Path`) — Destination GeoTIFF path.
* `target_crs` (`str`) — Target projection system (default: `"EPSG:32617"`).


* **Behavior:** Detects whether band 1 is floating point or integer to choose Predictor `3` or `2`, enforces tiled 512×512 blocks, applies DEFLATE compression, and calculates internal overviews (`2, 4, 8, 16`) where dimensions are at least 4× the decimation factor.

### `_download_tile_index(param_inputs)`

Dispatches parallel workers to scan NOAA's public S3 bucket for tile index `.zip` files matching DEM criteria, skips existing shapefiles, and downloads missing archives to local scratch space.

* **Inputs:** `list` — `[download_link, provider_folder, param_lookup, outputs]`.

### `_download_intersected_datasets(param_inputs)`

Router orchestrator that routes tasks based on parameter signature:

* **6 parameters:** Routes to `_download_intersected_datasets_s3`.
* **4 parameters:** Routes to `_download_intersected_datasets_local`.

### `_download_intersected_datasets_local(param_inputs)`

Local pipeline worker for spatial intersection, raster download, and reprojection:

1. Loads the index shapefile, reprojects to `EPSG:4326`, and spatial-joins (`sjoin`) with the target tile GeoDataFrame.
2. Identifies intersecting `.tif` URLs.
3. Downloads the raw raster into an `original/` subfolder.
4. Reprojects the raw GeoTIFF directly into the parent folder as `EPSG:32617` using `_reproject_raster_gdal`.
5. Calls `_trim_memory()` in a `finally` block.

* **Inputs:** `list` — `[tile_gdf, shp_path, param_lookup, outputs]`.

### `_download_intersected_datasets_s3(param_inputs)`

AWS S3 pipeline worker for scratch-buffered download, reprojection, and cloud upload:

1. Configures GDAL scratch space (`~/temp_scratch`) and sets `GDAL_DISABLE_READDIR_ON_OPEN=TRUE`.
2. Sanitizes shapefile columns (drops conflicting indexes and laundered `ecoregio_*` columns) and performs spatial intersection.
3. Generates collision-resistant UUID temp file paths (`raw_<uuid>.tif`, `rep_<uuid>.tif`).
4. Streams the raw raster to scratch disk and uploads it to S3 under `<relative_path>/original/<dataset_name>`.
5. Streams the reprojected COG (`EPSG:32617`) to scratch disk and uploads it to S3 under `<relative_path>/<dataset_name>`.
6. Unlinks scratch files and calls `_trim_memory()` in a `finally` block.

* **Inputs:** `list` — `[ecoregion, tile_gdf, shp_path, outputs, param_lookup, temp_root]`.

---

## 🛠️ Class: `DigitalCoastEngine`

### `__init__(param_lookup)`

Initializes the engine with the runtime configuration dictionary.

* **`param_lookup`** (`dict`): Must contain environment indicators (`'env'`: `'aws'` or local) and output settings (`output_directory`).

### `run(tile_gdf, output_prefix=False)`

Primary execution entry point.

* **Parameters:**
* `tile_gdf` (`gpd.GeoDataFrame`) — AOI boundaries and ecoregion definitions.
* `output_prefix` (`str | bool`, optional) — Subfolder or S3 prefix to prepend to output paths. Defaults to `False`.


* **Routing:** Directs execution to `_run_s3()` when `param_lookup['env'] == 'aws'`, otherwise executes `_run_local()`.

### `_run_local(tile_gdf, output_prefix)`

Executes the local download and conversion lifecycle:

* Configures Dask resources.
* Iterates over ecoregions, downloads tile indexes via `download_support_files()`, and filters small surveys via `check_tile_index_areas()`.
* Processes and reprojects intersected datasets to local storage.
* Removes provider directories without DEM/LAZ content via `delete_unused_folder()`.
* Writes run manifests recording provider counts.

### `_run_s3(tile_gdf, output_prefix)`

Executes the cloud S3 mirroring lifecycle:

* Launches Dask workers with `threads_per_worker=1`.
* Initializes an idempotent tracking file (`processed_providers.log`) to record finished providers and skip them on restart.
* Downloads support files into an isolated `tempfile.TemporaryDirectory()`.
* Chunks approved index shapefiles into batches of 5 providers per iteration.
* Dispatches S3 download, reprojection, and upload tasks.
* Uploads index shapefiles and vector metadata to S3 preserving the source directory tree.
* Updates provider logs and writes the run manifest before terminating the Dask cluster.

### `breakup_cudem(digital_coast_folder)`

Utility for reorganizing raw NOAA NCEI CUDEM collections:

* Scans `NOAA_NCEI_0*` folders, groups `.tif` rasters by year extracted from filename suffixes (e.g., `-6:-2`), moves files into dedicated `NOAA_NCEI_<year>_<id>/dem/<project>` directories, copies associated tile indexes and `feature.json` metadata, and deletes the unorganized source folders.

### `check_tile_index_areas(digital_coast_folder, outputs)`

Filters out survey footprints falling below `self.approved_size`:

* Reprojects index footprints to Albers Equal Area (`EPSG:9822`) and calculates total survey area.
* In directory mode (`digital_coast_folder` is a `Path`), deletes undersized provider directories from disk.
* In list mode (`digital_coast_folder` is a `list`), returns a filtered list of valid `pathlib.Path` objects.

### `delete_unused_folder(digital_coast_folder, outputs)`

Deletes empty provider directories or folders that contain neither `dem` nor `laz` subfolders.

### `download_support_files(digital_coast_folder, tile_gdf, ecoregion, outputs)`

Queries NOAA Digital Coast bulk endpoints across ecoregion bounding geometries, schedules `_download_tile_index` across Dask workers, and unpacks downloaded `.zip` files.

### `get_bucket()`

Establishes an unsigned `boto3.resource('s3')` connection to NOAA's public Open Common Services (OCS) S3 bucket.

### `print_async_results(results, output_folder=None)`

Consolidates and writes non-empty task status strings to the engine logger/output stream.

### `unzip_all_files(digital_coast_folder)`

Recursively searches for `.zip` files within `digital_coast_folder`, extracts contents into the archive's parent folder, and deletes the original zip file.

### `upload_file_to_s3(temp_file_path, s3_prefix)`

Uploads a single file to the S3 bucket defined by `get_config_item('SHARED', 'OUTPUT_BUCKET')` in `us-east-2`.

### `upload_files_to_s3(provider_folder, temp_root)`

Recursively walks `provider_folder`, derives relative POSIX paths against `temp_root`, and uploads all index shapefiles and vector metadata files to maintain remote directory structure integrity.

---

## 🚀 Usage Examples

### Local Ingestion Pipeline

```python
import geopandas as gpd
from hydro_health.engines.DigitalCoastEngine import DigitalCoastEngine


class DirectoryParam:
    valueAsText = "/data/gis/digital_coast_outputs"


# 1. Configure for local execution
param_lookup = {
    "env": "local",
    "output_directory": DirectoryParam()
}

# 2. Initialize engine and load target area
engine = DigitalCoastEngine(param_lookup=param_lookup)
study_area = gpd.read_file("coastal_study_aoi.geojson")

# 3. Execute local download, extraction, and EPSG:32617 reprojection
engine.run(tile_gdf=study_area, output_prefix="regional_dem")

```

### AWS S3 Distributed Ingestion Pipeline

```python
import geopandas as gpd
from hydro_health.engines.DigitalCoastEngine import DigitalCoastEngine


class DirectoryParam:
    valueAsText = "/home/ec2-user/job_logs"


# 1. Configure for AWS EC2 execution
param_lookup = {
    "env": "aws",
    "output_directory": DirectoryParam()
}

# 2. Initialize engine
engine = DigitalCoastEngine(param_lookup=param_lookup)
study_area = gpd.read_file("coastal_study_aoi.geojson")

# 3. Execute cloud ingestion
# Streams to ~/temp_scratch, uploads raw to s3://<bucket>/original/,
# reprojects to EPSG:32617, and uploads optimized COG to s3://<bucket>/
engine.run(tile_gdf=study_area, output_prefix="survey_v1")

```