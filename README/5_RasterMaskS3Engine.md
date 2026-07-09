# RasterMaskS3Engine

A high-performance geospatial raster mask generation engine designed to rasterize ecoregion boundaries and combine multi-source virtual mosaics into classification training images.

## 💡 Overview

The `RasterMaskS3Engine` creates targeted binary and categorical masks used to structure machine learning workflows. It uses an **asynchronous worker swarm** via `ProcessPoolExecutor` to stream cloud-hosted master VRTs, check geographic presence, and map data parameters across three categorical classifications:
* **Value 0:** NoData (Exterior zones or unmeasured footprints).
* **Value 1:** Active Area (Cells located inside the specified Ecoregion perimeter but containing no measured data).
* **Value 2:** Valid Hydro-Data (Cells within the ecoregion boundary that possess verified bathymetric measurements).

---

## ⚙️ Core Architecture & State

* **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
* **Spatial Reference Constraints:** Enforces standard coordinate metrics via **`EPSG:32617`** (WGS 84 / UTM Zone 17N) across calculations.
* **Process Isolation Overrides:** Mutes internal PROJ library cache allocations (`PROJ_LIB_CACHE=OFF`, `PROJ_NETWORK=OFF`) across the multiprocessing framework to eliminate deadlock conditions.

---

## 🛠️ Module-Level Orchestrators

### `_set_gdal_s3_options()`
* **What it does:** Configures low-level operational environment variables optimized for multi-process virtual cloud read stability.
* **The Specifics:** Sets network thread limits, enforces long timeouts (`60s`), activates thread-safe coordinate handling, and configures an internal thread-safe project workspace layout pointing to `/tmp/proj_cache`.

### `_vrt_to_mask_worker(vrt_path, scratch_dir, geo_t, cols, rows, target_srs_wkt)`
* **What it does:** Isolated multiprocess worker function responsible for converting a singular S3 master VRT into a detailed localized binary validation footprint.
* **The Specifics:** 1. Opens the remote VRT via `/vsis3/` and extracts the original source NoData signature.
  2. Projects and crops the target cell resolution footprint via `gdal.Warp` using a strict **Nearest Neighbor** approach.
  3. Uses the `dstAlpha=True` parameter during reprojection. This tells GDAL to generate an implicit transparency band (Band 2) that distinguishes cells containing actual data values from blank spaces.
  4. Isolates this alpha mask tracking array via `gdal.Translate` and saves it locally using high-efficiency **DEFLATE** compression and space-saving `SPARSE_OK=YES` parameters.
* **Inputs:**
  * `vrt_path` (`str`): Target remote virtual cloud path string.
  * `scratch_dir` (`str`): Path targeting the tracking worker's local scratch storage disk.
  * `geo_t` (`list[float]`): The standard 6-element affine transform matrix extracted from the base ecoregion file.

---

## 🛠️ Method Reference

### `run(outputs, manual_downloads)`
* **What it does:** Core orchestration method pulling ecoregion geodatabase configurations to step through mask generation.
* **The Specifics:** Evaluates ecoregion footprints directly from an internal Master Geopackage database (`Master_Grids.gpkg`). For each valid zone, it scans storage vectors for corresponding provider VRT paths, constructs structural prediction footprints, and triggers child training builders.
* **Inputs:**
  * `outputs` (`str`): Local system file pointer logging tracking message streams.
  * `manual_downloads` (`bool`): Toggle including custom fallback directory structures.

### `create_prediction_mask(ecoregion, wkt_geom)`
* **What it does:** Rasterizes vector polygons into a standard baseline classification image.
* **The Specifics:** Builds an in-memory vector database footprint using an `ogr.wkbPolygon` model. Calculates the explicit bounding frame coordinates (`GetExtent()`), determines structural cell width/height based on a uniform $8\text{m}$ grid specification, and updates data dimensions. It burns the baseline classification signature (`Value 1`) across cell features via `gdal.RasterizeLayer` and computes nearest-neighbor internal pyramid overviews up to a `64x` downscaling ratio before pushing the image to S3.

### `create_training_mask(ecoregion, s3_vrt_paths, outputs)`
* **What it does:** Spins up multiprocessing workers to perform cell-by-cell classification assignments.
* **The Specifics:** Pulls the baseline prediction mask from S3 to establish spatial extents. Distributes the target VRT arrays across a 6-worker `ProcessPoolExecutor` group to generate partial alpha masks locally. Once complete, it initializes an exhaustive nested block processing loop over a $4096 \times 4096$ pixel array size to perform bitwise optimization:
  
  $$\text{Target Mask Pixel} = \begin{cases} 
  2 & \text{if } \text{Prediction Pixel} = 1 \text{ AND } \text{Combined Worker Masks} > 0 \\ 
  1 & \text{if } \text{Prediction Pixel} = 1 \text{ AND } \text{Combined Worker Masks} = 0 \\
  0 & \text{otherwise} 
  \end{cases}$$

  It writes these updated evaluations directly to an uncompressed output file array, applies a final deflation step, generates downsampled internal pyramids, and uploads the dataset.

### `remerge_training_mask(ecoregion, outputs)`
* **What it does:** Standalone fallback rebuilding function.
* **The Specifics:** Skips intensive remote network fetching phases. It harvests pre-existing `part_*.tif` files already written to the local scratch environment and safely runs bitwise array merges to rebuild the training mask image. Useful for manual QA adjustments or resuming interrupted workflows.

---

## 🚀 Quick Start / Usage Example

```python
from hydro_health.engines.MaskEngine import RasterMaskS3Engine

# 1. Establish internal context tracker parameters
param_lookup = {'env': 'production'}

# 2. Instantiate mask compiling processor
mask_engine = RasterMaskS3Engine(param_lookup=param_lookup)

# 3. Process ecoregions and compile machine learning target training grids
mask_engine.run(
    outputs="/var/log/hydro_mask_run.log",
    manual_downloads=True
)