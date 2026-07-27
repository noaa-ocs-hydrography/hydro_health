# BlueTopoS3Engine

A high-performance, asynchronous geospatial data pipeline designed to ingest, process, and derive bathymetric layers from NOAA's National Bathymetry Source (NBS) **BlueTopo** dataset.

## 💡 Overview

The `BlueTopoS3Engine` manages an optimized **S3-to-S3 orchestration layer** utilizing **Dask** to process large-scale hydrographic datasets in parallel. Each distributed worker acts dynamically within a localized ephemeral workspace to:
1. Stream raw data components from NOAA's public data repositories.
2. Carefully reproject mixed-type continuous/discrete rasters to standardized spatial references.
3. Mine embedded XML Attribute Tables to score data confidence.
4. Export Cloud-Optimized GeoTIFFs (COGs) back to private storage.

---

## ⚙️ Core Architecture & State

* **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
* **Target CRS:** Forced globally to **`EPSG:6350`** (NAD83 / Conus Albers).
* **Execution Paradigm:** Distributed asynchronous mapping using a Dask client connection.
* **Storage Interface:** Ephemeral local file caching with permanent read/write links pointing straight to AWS S3.

---

## 🛠️ Module-Level Orchestrators

### `_process_tile(param_inputs)`
* **What it does:** The runtime logic executed by individual parallel Dask workers.
* **The Specifics:** 1. Spins up an isolated `TemporaryDirectory` tracking a specific tile.
  2. Pulls downstream assets from S3 (`download_nbs_tile`).
  3. Executes explicit dual-algorithm raster reprojections (`warp_bluetopo_tile`).
  4. Generates thematic derivatives (Survey Ends, ISS All, ISS Latest, Rugosity, Slope).
  5. Cleans up edge land boundaries, splits structural bands, builds optimized layout structures, and pushes variables straight back to S3.
* **Inputs:** `param_inputs` (list) - `[param_lookup, tile_id, ecoregion_id, output_prefix, target_res]`.
* **Outputs:** `str` - Status message outlining worker completion or structured warning.

### `_parse_survey_date(date_str)`
* **What it does:** Robustly handles variations in data formatting inside the raster attribute records.
* **The Specifics:** Loops structural parses through common timestamp formats (`%Y-%m-%d`, `%Y/%m/%d`, etc.). If a structure fails, it uses a fallback regular expression matching the closest 4-digit year value (e.g., converting `1995.5` to a concrete chronological start: `1996-01-01`).

---

## 🛠️ Method Reference

### `run(tile_gdf, output_prefix, resolution)`
* **What it does:** Main entry point method to execute large-scale tile batches.
* **The Specifics:** Group-sorts active targets sequentially by numeric EcoRegion bounds, starts up Dask worker parameters according to environment configurations, pushes payloads to workers, blocks until calculations finish, and outputs run metrics.
* **Inputs:**
  * `tile_gdf` (`gpd.GeoDataFrame`): Spatial boundaries matching target regions.
  * `output_prefix` (`str` | `bool`): Active path string modifiers.
  * `resolution` (`list[int]`): Targeted spatial grids in meters (e.g., `[4, 8]`).

### `warp_bluetopo_tile(tiff_file_path, target_res)`
* **What it does:** Safely warps multi-characteristic source tiles without losing data integrity.
* **Gotchas / Notes:** **Critical Operation.** To prevent interpolation corruption across categorical parameters, this method splits the file. It isolates Bands 1 & 2 (Elevation, Uncertainty) and processes them via a **Bilinear** algorithm. Band 3 (Contributor Index) is processed via **Nearest Neighbor**. The bands are then stitched back together, and the original XML metadata table structure is restored.

### `create_catzoc_all(tiff_file_path, increased_scale)` / `create_catzoc_latest(tiff_file_path, increased_scale)`
* **What it does:** Builds explicit Initial Survey Score (ISS) confidence layers.
* **The Specifics:** Uses `lxml.etree` to scrape internal GDAL attribute metadata strings, normalizes qualitative records into flat Pandas data matrices, processes records using external math functions (`supersession` / `catzoc`), and maps numerical indices back into a newly generated single-band GeoTIFF.

### `create_survey_end_date_tiff(tiff_file_path)`
* **What it does:** Extracts chronological survey markers into a dedicated spatial image.
* **The Specifics:** Translates spatial pixels to explicitly represent the integer year value when the specific sector measurement took place.

### `create_rugosity(tiff_file_path)` / `create_slope(tiff_file_path)`
* **What it does:** Generates physical surface derivatives.
* **The Specifics:** Wraps internal high-performance C++ `gdal.DEMProcessing` configurations to build `Roughness` and `slope` images from active values.

### `set_ground_to_nodata(tiff_file_path)`
* **What it does:** Masks out above-water geography features.
* **The Specifics:** Modifies data matrices in place using a fast NumPy overlay step, immediately turning values $\ge 0$ into the standard missing index value (`-9999`).

### `finalize_cog(tiff_path)`
* **What it does:** Converts basic raster files into strict Cloud-Optimized GeoTIFF layout formats.
* **The Specifics:** Creates bilinear internal dataset lookups (`[2, 4, 8, 16]`) and passes variables through a structured `gdal.Translate` block using efficient **DEFLATE** algorithms, floating-point predictors, and structural `512x512` tile alignments.

---

## 🚀 Quick Start / Usage Example

```python
import geopandas as gpd
from hydro_health.engines.BlueTopoEngine import BlueTopoS3Engine

# 1. Define operational configs
param_lookup = {
    'env': 'production',
    'eco_regions': MockField(['Ecoregion 4']),
    'output_directory': MockField('/local/scratch/space')
}

# 2. Instantiate pipeline engine
engine = BlueTopoS3Engine(param_lookup=param_lookup)

# 3. Read input boundaries index 
boundaries_gdf = gpd.read_file("tiles_selection.geojson")

# 4. Fire pipeline tasks sequentially through target metrics
engine.run(
    tile_gdf=boundaries_gdf,
    output_prefix="regional_model",
    resolution=[4, 8, 16]
)