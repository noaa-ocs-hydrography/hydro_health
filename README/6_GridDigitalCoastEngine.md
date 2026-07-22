# GridDigitalCoastEngine

An asynchronous spatial cookie-cutter engine that clips large-scale DigitalCoast Virtual Rasters (VRTs) down into standardized grid footprints matching the underlying BlueTopo tile coordinate system.

## 💡 Overview

The `GridDigitalCoastEngine` acts as an extraction bridge between two separate hydrographic index conventions. It scans regional project directories for high-resolution elevation models structured as unified virtual indices. It then reads dissolved spatial reference files (`*_dis.shp`), aligns coordinate spaces, and calculates exact intersections against a primary grid index map. 

The tool exposes flexible execution loops based on execution flags:
* **Cloud Mode (S3-to-S3):** Deploys a parallel Dask strategy to stream VRTs using `/vsis3/`, creates short-lived geometric boundaries directly in-memory using `/vsimem/` buffers, warps the target fragments locally, and streams outputs straight back to cloud buckets.
* **Local Mode:** Performs direct local disk sweeps across your storage array paths via localized `pathlib` tracking handles.

---

## ⚙️ Core Architecture & State

* **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
* **Memory Protection Bounds:** Strictly caps internal cache systems (`gdal.SetCacheMax(536870912)`) at $512\text{ MB}$ per process to mitigate memory allocation spikes across complex cluster workers.
* **Network Tuning Hooks:** Activates custom cloud configuration modes to enable efficient partial block updates over cloud files:
  * `CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES`
  * `CPL_VSIL_S3_WRITE_SUPPORT=YES`

---

## 🛠️ Module-Level Orchestrators

### `_grid_single_vrt_s3(params)`
* **What it does:** Clips a singular remote VRT file down into individual spatial tiles over an S3 pipeline.
* **The Specifics:** 1. Opens remote links using virtual network pathways and harvests matching dissolved boundary indicators (`*_dis.shp`).
  2. Compiles shapes inside an ephemeral workspace using Geopandas and standardizes projection coordinates.
  3. Identifies spatial tiles that overlap the input data boundaries.
  4. Drops vector geometries into an in-memory virtual JSON container (`/vsimem/`).
  5. Runs `gdal.Warp` targeting specific metadata masks, applies internal pixel predictors (`PREDICTOR=3`) for optimal floating-point terrain compression, and pushes outputs straight back to cloud paths.
* **Inputs:** `params` (`list`): Packed task context properties passing `[vrt_s3_path, ecoregion_prefix, bluetopo_grids, blue_topo_gdf, param_lookup]`.
* **Outputs:** `str`: Log status tracker message confirming success or file omission.

### `_grid_single_vrt_local(params)`
* **What it does:** Clips a singular local VRT file down into matching tile segments on disk.
* **The Specifics:** Mimics the core spatial validation logic of the cloud component but utilizes direct disk paths and passes raw **Well-Known Text (WKT)** coordinate vectors directly to the `cutlineDSName` parameter inside `gdal.Warp`.
* **Inputs:** `params` (`list`): Packed array tracking `[vrt_path, ecoregion, bluetopo_grids, blue_topo_gdf, param_lookup]`.
* **Outputs:** `str`: Operation string status identifier.

---

## 🛠️ Method Reference

### `run(manual_download)`
* **What it does:** Core initialization supervisor mapping environment architectures.
* **The Specifics:** Opens structural index files (`Master_Grids.gpkg`) to track core boundary frames. **Optimization Note:** Instead of redundantly loading huge spatial indices across multiple computing steps, it executes `self.client.scatter` to broadcast the data array as a high-speed pointer (Dask Future object) out to memory targets. It inspects configuration tags to delegate executions across either local paths or remote cloud pathways.
* **Inputs:** `manual_download` (`bool`): Toggle routing path structures to custom tracking configurations.

### `process_s3_vrt_gridding(blue_topo_gdf_future, outputs, manual_download)`
* **What it does:** Runs automated directory queries and handles distributed Dask maps across S3 paths.
* **The Specifics:** Automatically builds path queries tracking unique EcoRegion elements, matches active spatial variables across configuration subfolders, identifies present `.vrt` descriptors, and launches the `_grid_single_vrt_s3` distributed worker map.

### `process_local_vrt_gridding(blue_topo_gdf_future, outputs)`
* **What it does:** Handles file parsing tasks for environments running on local physical machines.
* **The Specifics:** Leverages native `pathlib.Path.glob` queries to index and trace tracking files across attached storage sectors, passing localized tasks out to multiprocessing groups.

---

## 🚀 Quick Start / Usage Example

```python
from hydro_health.engines.GridEngine import GridDigitalCoastEngine

# 1. Establish structural runtime settings context
param_lookup = {
    'env': 'production_s3',
    'output_directory': MockField('/var/outputs/tiled_grid')
}

# 2. Instantiate tile footprint grid processing engine
grid_engine = GridDigitalCoastEngine(param_lookup=param_lookup)

# 3. Process, cross-intersect, and cookie-cut target datasets
grid_engine.run(manual_download=False)