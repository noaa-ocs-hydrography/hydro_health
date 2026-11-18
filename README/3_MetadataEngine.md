# MetadataEngine

A unified metadata extraction pipeline designed to scrape, parse, and catalog chronological survey temporal bounds (start and end date-time pairings) from public NOAA data providers across both local filesystems and AWS S3.

## 💡 Overview

`MetadataEngine` processes `feature.json` catalog configurations generated during DigitalCoast data ingestion phases. It parses multiple JSON schema formats to extract administrative NOAA InPort metadata links, fetches those XML endpoints concurrently via a thread pool, and writes standardized `metadata.txt` chronological registries directly to local storage or S3 before recording a run manifest.

---

## ⚙️ Core Architecture & State

- **Base Class:** Inherits from `Engine`, utilizing shared operational logging (`write_message`) and run tracking (`write_run_manifest`).
- **Dual-Environment Storage Abstraction:** Controlled by `param_lookup['env']` (`'aws'` vs. `'local'`). Internal `_open()` and `_find_files()` helpers transparently route file streams and glob searches between `s3fs.S3FileSystem` and native `pathlib` / UTF-8 local file I/O.
- **Concurrency Model:** Uses `concurrent.futures.ThreadPoolExecutor` rather than Dask clusters, optimizing throughput for lightweight, I/O-bound HTTP requests.

---

## 🛠️ Method Reference

### `__init__(param_lookup)`
- **What it does:** Initializes the engine state, storage mode, and S3 client connection.
- **Inputs:**
  * `param_lookup` (`dict`): Configuration dictionary containing `'env'` (`'aws'` or `'local'`) and `'output_directory'` (an object exposing `.valueAsText`).

### `run(tile_gdf, output_prefix)`
- **What it does:** The primary driver loop cycling through unique EcoRegions in the input spatial index.
- **The Specifics:** Resolves the target DigitalCoast path per `EcoRegion` (prefixing with `s3://{OUTPUT_BUCKET}` in AWS mode or `output_prefix` / `outputs` locally), triggers JSON discovery and metadata downloads, counts all resulting `metadata.txt` files, and records totals via `write_run_manifest()`.
- **Inputs:**
  * `tile_gdf` (`gpd.GeoDataFrame`): Spatial boundary indices containing an `'EcoRegion'` column.
  * `output_prefix` (`str | bool`): Optional subfolder prefix prepended to the output path (`False` or empty to use the default output directory directly).

### `read_json_files(digital_coast_path, outputs)`
- **What it does:** Scans local or S3 directories to compile valid metadata endpoint tasks and executes them concurrently.
- **The Specifics:**
  1. Discovers all nested `*.json` files via `_find_files()`, excluding any paths containing `unused_providers`.
  2. Parses each file stream with `json.load` (skipping corrupt or unreadable files).
  3. Extracts `(label, uri)` pairs via `_extract_metadata_link()`, supporting both `ExternalProviderLink` + `Metalink` and `links` (`linkTypeName == 'Metadata'`) schemas.
  4. Dispatches tasks to a `ThreadPoolExecutor` allocated across:
     $$\text{Workers} = \max(1, \text{CPU Count} - 2)$$
- **Inputs:**
  * `digital_coast_path` (`str | pathlib.Path`): Parent DigitalCoast directory path or S3 URI.
  * `outputs` (`str`): Destination path for operational log output (`log_prints.txt`).

### `download_metadata(param_inputs)`
- **What it does:** Fetches InPort XML metadata for a single provider and writes formatted date ranges to `metadata.txt`.
- **The Specifics:** Normalizes trailing slashes on `download_link`, appends `/inport-xml`, and calls `requests.get(..., timeout=10)` with `raise_for_status()`. Parses `<time-frame>` tags using `BeautifulSoup(..., features='xml')` and writes `Description: <desc>` followed by `<start-date-time>, <end-date-time>` via `_open()`.
- **Inputs:**
  * `param_inputs` (`list`): Packed list containing `[label, download_link, provider_folder, outputs]`, where `provider_folder` is a `pathlib.Path`.
- **Gotchas / Notes:**
  * **Fallback Description:** If a `<description>` tag is missing from a `<time-frame>` block, `provider_folder.name` is used as the description.
  * **Fault Tolerance:** Catches `requests.RequestException` (timeouts, HTTP errors, connection failures) and logs the failure via `write_message()` without interrupting sibling worker threads. Records with labels outside `{'Metadata', 'ISO metadata'}` are skipped and logged.

### `get_ecoregion_geometry_strings(tile_gdf, ecoregion)`
- **What it does:** Compiles dissolved regional footprints in NAD83 geographic coordinates.
- **The Specifics:** Filters `tile_gdf` for the target `ecoregion`, reprojects to **`EPSG:4269`**, dissolves tile polygons via GeoPandas `.dissolve()`, and returns the resulting dissolved Shapely geometry objects.
- **Inputs:**
  * `tile_gdf` (`gpd.GeoDataFrame`): Source vector tile GeoDataFrame.
  * `ecoregion` (`str`): Target EcoRegion identifier.
- **Outputs:** `list` of dissolved Shapely geometry objects matching the requested EcoRegion.

---

## 🚀 Quick Start / Usage Example

```python
from types import SimpleNamespace
import geopandas as gpd
from hydro_health.engines.MetadataEngine import MetadataEngine

# 1. Configure environment and output directory parameters
params = {
    "env": "aws",  # Use "local" for local filesystem execution
    "output_directory": SimpleNamespace(valueAsText="s3://my-hydro-bucket/logs"),
}

# 2. Instantiate the metadata pipeline engine
metadata_engine = MetadataEngine(param_lookup=params)

# 3. Load the operational tile index GeoDataFrame
index_tiles_gdf = gpd.read_file("active_hydro_tiles.geojson")

# 4. Process metadata and write run manifests
metadata_engine.run(
    tile_gdf=index_tiles_gdf,
    output_prefix="run_2026",
)
```