# DigitalCoastEngine (AWS / S3 Pipeline)

A specialized spatial ingestion pipeline designed to scrape, spatially filter, optimize, and mirror NOAA Digital Coast bathymetric elevation datasets (DEMs) directly into private AWS S3 storage from Linux EC2 instances.

## 💡 Overview

When running in an AWS environment (`param_lookup['env'] == 'aws'`), `DigitalCoastEngine` targets NOAA's public distribution networks through two major parallel phases:

1. **Index Syncing:** Dispatches asynchronous Dask workers to discover and download regional tile index shapefiles into ephemeral local scratch storage.
2. **Intersection Streaming & COG Ingestion:** Identifies datasets intersecting the input AOI, streams tiles directly to physical disk scratch storage to conserve memory, converts raw rasters into Cloud-Optimized GeoTIFFs (COGs) with internal overview pyramids, and uploads the optimized files and metadata directly to S3 (`us-east-2`).

---

## ⚙️ Core Architecture & State

* **Base Class:** Inherits directly from `Engine` (`hydro_health.engines.Engine`).
* **Environment Overrides:** Configures UTF-8 encodings (`PYTHONIOENCODING`, `LC_ALL`, `LANG`) for Red Hat Enterprise Linux / Amazon Linux compatibility and suppresses verbose `pyproj` internal logging.
* **Disk Scratch Optimization:** Prevents EC2 out-of-memory errors by routing heavy raster download streams and GDAL temporary caches (`GDAL_TMPDIR`, `GDAL_CACHEMAX=512`) to physical disk at `~/temp_scratch`.
* **Fault-Tolerant State Logging:** Maintains an idempotent transaction ledger (`processed_providers.log`) inside the output directory, ensuring interrupted runs safely skip already-processed providers upon resumption.

---

## 🛠️ Module-Level Orchestrators

### `_download_tile_index(param_inputs)`

* **What it does:** Scans the target public NOAA S3 storage bucket to pull data provider index files.
* **The Specifics:** Connects via unsigned S3 credentials, locates `.zip` tile index packages matching the DEM criteria, checks whether the uncompressed index shapefile already exists, downloads missing zip archives to provider scratch folders, and logs operations.
* **Inputs:** `param_inputs` (`list`) — `[download_link, provider_folder, param_lookup, outputs]`.

### `_download_intersected_datasets(param_inputs)`

* **What it does:** Orchestrates spatial intersection filtering, raster streaming, Cloud-Optimized GeoTIFF (COG) generation, and immediate S3 upload.
* **The Specifics:**
1. Opens local shapefiles using the high-performance `pyogrio` engine, fixes geometry issues via `.make_valid()`, and normalizes column headers.
2. Executes a spatial join (`sjoin`) with the region's tile geometries to identify intersecting rasters.
3. Uses a persistent `requests.Session` with an exponential backoff retry strategy (`429, 500, 502, 503, 504`) to stream remote `.tif` files into `~/temp_scratch` in 1 MB chunks.
4. Converts each raster to a tiled GeoTIFF (512×512 blocks, DEFLATE compression, floating-point Predictor 3, pixel interleave).
5. Dynamically builds internal overview pyramids (`[2, 4, 8, 16]`) using average resampling where image dimensions allow.
6. Computes the target S3 key from the relative path against `temp_root` and uploads the COG directly to the output S3 bucket.


* **Inputs:** `param_inputs` (`list`) — `[ecoregion, tile_gdf, shp_path, outputs, param_lookup, temp_root]`.
* **Outputs:** `str` — Stem name of the processed index file.

---

## 🛠️ Method Reference

### `DigitalCoastEngine(param_lookup)`

* **What it does:** Initializes the engine with runtime configuration.
* **The Specifics:** Stores `param_lookup`. Setting `'env': 'aws'` within this lookup automatically routes execution to the cloud S3 mirroring and COG conversion pipeline.

### `run(tile_gdf, output_prefix="")`

* **What it does:** Main entry point executing the end-to-end cloud ingestion lifecycle.
* **The Specifics:**
1. Detects `self.param_lookup['env'] == 'aws'` and launches Dask cluster workers via `setup_dask()`.
2. Initializes `processed_providers.log` in the configured outputs directory.
3. Iterates through unique ecoregions in `tile_gdf`, generating an isolated workspace using `tempfile.TemporaryDirectory()`.
4. Downloads and unzips index shapefiles for the region, then validates candidate survey areas.
5. Chunks approved index shapefiles into batches of 5, cross-references `processed_providers.log` to skip completed work, and parallel-dispatches dataset processing to Dask workers.
6. Uploads the accompanying provider index shapefiles and support files into S3 to match directory structure integrity.
7. Writes a run manifest containing provider counts before closing Dask resources.


* **Inputs:**
* `tile_gdf` (`gpd.GeoDataFrame`) — Target spatial boundary definitions.
* `output_prefix` (`str | bool`, optional) — Subfolder or prefix appended to the S3 path structure.



### `check_tile_index_areas(digital_coast_folder, outputs)`

* **What it does:** Filters out survey footprints below minimum threshold requirements.
* **The Specifics:** Re-projects index layers to Albers Equal Area (`EPSG:9822`), aggregates total coverage area, and excludes small patches failing `self.approved_size`. When passed a list of file paths, returns the approved candidate paths for downstream chunking.
* **Inputs:**
* `digital_coast_folder` (`pathlib.Path | list[pathlib.Path]`) — Root folder or list of index shapefiles to inspect.
* `outputs` (`str`) — Path to engine output log directory.


* **Outputs:** `list[pathlib.Path] | None` — List of approved shapefile paths meeting area thresholds.

### `download_support_files(digital_coast_folder, tile_gdf, ecoregion, outputs)`

* **What it does:** Retrieves all remote NOAA index shapefiles covering the regional boundary.
* **The Specifics:** Gathers boundary coordinates, parses bulk download endpoints from the Digital Coast service, dispatches `_download_tile_index` across Dask workers, and triggers unzipping into the active scratch workspace.

### `upload_file_to_s3(temp_file_path, s3_prefix)`

* **What it does:** Uploads an individual file from scratch disk to AWS S3.
* **The Specifics:** Uses `boto3.client('s3', region_name='us-east-2')` to push the designated file to the bucket defined under configuration setting `('SHARED', 'OUTPUT_BUCKET')`.

### `upload_files_to_s3(provider_folder, temp_root)`

* **What it does:** Mirrors an entire local provider folder to S3.
* **The Specifics:** Recursively walks `provider_folder`, extracts relative POSIX paths against `temp_root`, and uploads all index and vector metadata files while preserving folder structure.

---

## 🚀 Quick Start / Usage Example

```python
import geopandas as gpd
from hydro_health.engines.DigitalCoastEngine import DigitalCoastEngine

# 1. Prepare operational context metadata targeting AWS
param_lookup = {
    'env': 'aws',
    'output_directory': MockField('/home/ec2-user/outputs/digital_coast')
}

# 2. Instantiate the unified engine
engine = DigitalCoastEngine(param_lookup=param_lookup)

# 3. Load tracking target boundaries
study_area_gdf = gpd.read_file("coastal_aoi_bounds.geojson")

# 4. Trigger cloud ingestion and S3 mirroring
engine.run(tile_gdf=study_area_gdf, output_prefix="testing")

```