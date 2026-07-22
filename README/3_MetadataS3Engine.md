# MetadataS3Engine

An asynchronous metadata extraction pipeline designed to scrape, parse, and catalog chronological survey temporal bounds (start and end date-time pairings) from public NOAA data providers.

## 💡 Overview

The `MetadataS3Engine` operates exclusively over cloud-native files by reading structural `feature.json` configurations generated during preceding data ingestion phases. It parses these documents to extract administrative NOAA InPort metadata links, hits those API endpoints over a concurrent pool thread layout, and translates XML payloads down into highly readable chronological tracker registries stored directly inside S3.

---

## ⚙️ Core Architecture & State

* **Base Class:** Operates independently of the rigid base spatial wrapper, acting as a lightweight cloud text processing service.
* **Concurrency Model:** Uses a high-speed multithreaded framework (`concurrent.futures.ThreadPoolExecutor`) rather than resource-heavy Dask worker cluster clusters, optimizing it for light I/O HTTP requests.
* **Storage Interactions:** Relies entirely on stream buffers over an active `s3fs.S3FileSystem` connection, skipping the need for local temporary storage caching altogether.

---

## 🛠️ Method Reference

### `run(tile_gdf, outputs)`
* **What it does:** The primary driver loop cycling through valid administrative geographical units.
* **The Specifics:** Loops sequentially across unique regional elements inside an active index boundary, maps physical target keys directly to the shared target S3 project bucket paths, and kicks off underlying data catalog routines.
* **Inputs:** * `tile_gdf` (`gpd.GeoDataFrame`): Spatial boundary indices identifying active areas.
  * `outputs` (`str`): Base filepath string targeting the primary operational logs.

### `read_json_files(digital_coast_path, outputs)`
* **What it does:** Scans cloud storage folders to compile explicit targeted metadata endpoint arrays.
* **The Specifics:** 1. Performs a comprehensive directory sweep (`glob`) identifying nested catalog index descriptors (`*.json`).
  2. Parses file streams using the standard `json.load` protocol.
  3. Evaluates structural keys (`ExternalProviderLink`) to find valid internal text markers (`Metadata` or `ISO metadata`).
  4. Automatically groups target elements and pushes compilation matrices straight out to a dynamic local worker thread pool running across an explicit structural allocation layout:
     $$\text{Workers} = \text{CPU Count} - 2$$
* **Inputs:** * `digital_coast_path` (`pathlib.Path`): Remote parent folder pathway string.
  * `outputs` (`str`): Path targeting the tracking print text file.

### `upload_metadata_to_s3(param_inputs)`
* **What it does:** Fetches structural XML metadata and writes tracking chronologies directly back to cloud storage.
* **The Specifics:** Appends standard tracking endpoints (`/inport-xml`) to target parameters and executes a streaming call via `requests.get`. It drops data payloads directly into a native `BeautifulSoup` parsing tree configured for standard XML. It harvests embedded tracking arrays (`<time-frame>`), extracts individual contextual details (`description`, `start-date-time`, `end-date-time`), and writes formatted lines directly into a cloud path wrapper (`/metadata.txt`).
* **Inputs:** `param_inputs` (`list[list]`): Packed parameters tracking `[label, download_link, provider_folder, outputs]`.
* **Gotchas / Notes:** Features soft safety controls checking connection failures explicitly (`requests.exceptions.ConnectionError`). If an endpoint fails, it documents the broken track in the main configuration log without killing the remaining active execution loops.

### `get_ecoregion_geometry_strings(tile_gdf, ecoregion)`
* **What it does:** Compiles dissolved contextual geographical elements into standardized target query tracking strings.
* **The Specifics:** Groups inputs, changes projections to a strict geographic database standard (**`EPSG:4269`**), dissolves standalone components into cohesive multipolygon footprints via Geopandas `.dissolve()`, and returns the underlying well-known text (WKT) geometries.
* **Inputs:** * `tile_gdf` (`gpd.GeoDataFrame`): Source vector database coordinates.
  * `ecoregion` (`str`): Explicit targeting name filter string.
* **Outputs:** `list` - Tracking geometry shapes matching geographical constraints.

---

## 🚀 Quick Start / Usage Example

```python
import geopandas as gpd
from hydro_health.engines.MetadataEngine import MetadataS3Engine

# 1. Instantiate the metadata pipeline engine
metadata_compiler = MetadataS3Engine()

# 2. Grab your operational reference dataset index
index_tiles_gdf = gpd.read_file("active_hydro_tiles.geojson")

# 3. Direct log pointer output configurations
log_directory = "/var/log/hydro_health"

# 4. Process and catalog historical temporal logs
metadata_compiler.run(
    tile_gdf=index_tiles_gdf,
    outputs=log_directory
)