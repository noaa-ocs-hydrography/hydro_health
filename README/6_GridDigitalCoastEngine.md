# GridDigitalCoastEngine

An asynchronous spatial cookie-cutter engine that clips large-scale DigitalCoast Virtual Rasters (VRTs) down into standardized grid footprints matching the underlying BlueTopo tile coordinate system.

## 💡 Overview

The `GridDigitalCoastEngine` acts as an extraction bridge between DigitalCoast survey mosaics and BlueTopo grid schemes. It scans regional ecoregion directories for approved provider VRTs across three supported data types:

- **`DigitalCoast`**: Standard elevation/bathymetric VRTs (`mosaic_<YEAR>_<PROVIDER>.vrt`).
- **`Digital_Coast_Manual_Downloads`**: Manually acquired survey VRTs stored in `Digital_Coast_Manual_Downloads/`.
- **`DistanceToShore`**: Distance-to-shore rasters (`*_DTS.vrt`) located across both `DigitalCoast/` and `Digital_Coast_Manual_Downloads/`, sharing the spatial footprint and shapefile index of their parent DEM survey.

For each approved VRT, the engine locates the survey's `tileindex*.shp` (falling back to any `.shp` in the survey folder), dissolves geometries on the fly, aligns coordinate reference systems, and clips intersecting BlueTopo tiles into compressed, overview-enabled GeoTIFFs (`<base_stem>_<tile_id>.tiff` or `<base_stem>_<tile_id>_DTS.tiff`).

The tool supports two execution environments based on `param_lookup['env']`:
- **Cloud Mode (S3-to-S3):** Deploys a parallel Dask strategy to stream VRTs via `/vsis3/`, builds cutline boundaries in memory via `/vsimem/`, warps tiles locally, validates against empty NoData warps, and uploads valid outputs back to S3.
- **Local Mode (`local` / `remote`):** Performs direct local disk sweeps across attached storage paths via `pathlib` and Dask workers.

---

## ⚙️ Core Architecture & State

- **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
- **Provider & Dataset Filtering:** Validates VRT providers against `get_approved_providers(ecoregion_stem)` (stripping any `_DTS` suffix before matching) and isolates `*_DTS.vrt` files exclusively for the `DistanceToShore` target.
- **Incremental Execution:** Checks whether each target `.tiff` already exists in S3 or on disk before warping, skipping completed tiles automatically.
- **Memory Protection Bounds:** Caps internal GDAL cache (`gdal.SetCacheMax(536870912)`) at $512\text{ MB}$ per worker process to prevent memory spikes across Dask workers.
- **Network Tuning Hooks:** Configures GDAL virtual filesystem options for S3 streaming and writes:
  - `CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES`
  - `CPL_VSIL_S3_WRITE_SUPPORT=YES`

---

## 🛠️ Module-Level Functions

### `_grid_single_vrt_s3(params)`
- **What it does:** Clips a single remote S3 VRT into intersecting BlueTopo tiles and uploads valid GeoTIFFs back to S3.
- **The Specifics:**
  1. Opens the VRT over `/vsis3/` and resolves its shapefile directory (`<YEAR>_<PROVIDER>`) with any `_DTS` suffix stripped.
  2. Downloads `tileindex` shapefile sidecars (`.shp`, `.shx`, `.dbf`, `.prj`) into a temporary directory, reprojects to the BlueTopo CRS if needed, and dissolves the survey footprint on the fly.
  3. Writes each intersecting BlueTopo polygon to an in-memory GeoJSON cutline (`/vsimem/`) and runs `gdal.Warp` with `512x512` tiling, `DEFLATE` compression, and band-appropriate predictors (`PREDICTOR=3` for float, `PREDICTOR=2` for integer).
  4. Runs `_is_raster_empty` post-warp to discard `1x1` placeholders or pure-NoData edge tiles before building `[2, 4, 8]` `BILINEAR` overviews and uploading to S3.
- **Inputs:** `params` (`list`): `[vrt_s3_path, ecoregion_prefix, bluetopo_grids, blue_topo_gdf, param_lookup]`.
- **Outputs:** `str`: Status message confirming VRT completion, missing shapefile skip, or error details.

### `_grid_single_vrt_local(params)`
- **What it does:** Clips a single local VRT into matching BlueTopo tile GeoTIFFs on disk.
- **The Specifics:** Mirrors `_grid_single_vrt_s3` using local `pathlib` paths for shapefile discovery and output creation, `/vsimem/` GeoJSON cutlines, post-warp empty raster filtering, and `[2, 4, 8]` `BILINEAR` overview generation.
- **Inputs:** `params` (`list`): `[vrt_path, ecoregion, bluetopo_grids, blue_topo_gdf, param_lookup]`.
- **Outputs:** `str`: Status message confirming VRT completion, missing shapefile skip, or error details.

### Helper Functions
- **`_is_dts_vrt(stem)` / `_strip_dts_suffix(stem)`:** Detects and removes a trailing `_DTS` suffix (case-insensitive) when resolving provider names and parent survey shapefile folders.
- **`_build_output_filename(vrt_stem, tile_id)`:** Formats output filenames as `<base_stem>_<tile_id>.tiff` for standard VRTs and `<base_stem>_<tile_id>_DTS.tiff` for DistanceToShore VRTs.
- **`_get_warp_params(ds)`:** Inspects band 1 of the VRT to preserve its native NoData value (defaulting to `-9999`) and select `PREDICTOR=3` (floating-point) or `PREDICTOR=2` (integer) `DEFLATE` creation options.
- **`_is_raster_empty(ds, nodata_val=-9999)`:** Verifies that a warped dataset is larger than `1x1` and contains non-NoData values before saving or uploading.

---

## 🛠️ Method Reference

### `run(output_prefix, data_type='DigitalCoast')`
- **What it does:** Main execution entry point that validates the target dataset, initializes Dask, broadcasts the BlueTopo master grid, and routes execution to S3 or local processing.
- **The Specifics:** Loads `Master_Grids.gpkg` and uses `self.client.scatter(..., broadcast=True)` to share the `GeoDataFrame` across Dask workers without redundant disk/network reads.
- **Inputs:**
  - `output_prefix` (`str`): Optional sub-prefix under the bucket or output directory containing `ER_*` ecoregion folders.
  - `data_type` (`str`): Target dataset to process. Must be one of `'DigitalCoast'`, `'Digital_Coast_Manual_Downloads'`, or `'DistanceToShore'`.

### `process_s3_vrt_gridding(blue_topo_gdf_future, outputs, data_type, output_prefix)`
- **What it does:** Discovers ecoregions, BlueTopo grid folders, and approved VRTs in S3 and dispatches `_grid_single_vrt_s3` tasks across Dask.
- **The Specifics:** Searches `DigitalCoast/` and `Digital_Coast_Manual_Downloads/` when `data_type == 'DistanceToShore'`, or the single matching folder otherwise, filtering VRTs by both `_DTS` suffix and `get_approved_providers()`.

### `process_local_vrt_gridding(blue_topo_gdf_future, outputs, data_type, output_prefix)`
- **What it does:** Discovers local ecoregion directories and approved VRTs on disk and dispatches `_grid_single_vrt_local` tasks across Dask.

---

## 🚀 Quick Start / Usage Example

```python
from hydro_health.engines.GridDigitalCoastEngine import GridDigitalCoastEngine

# 1. Establish runtime settings context
param_lookup = {
    'env': 'aws',
    'output_directory': MockField('/var/outputs/tiled_grid')
}

# 2. Run all three datasets sequentially
for data_type in ['DigitalCoast', 'Digital_Coast_Manual_Downloads', 'DistanceToShore']:
    grid_engine = GridDigitalCoastEngine(param_lookup=param_lookup)
    grid_engine.run(output_prefix='model_variables/Prediction', data_type=data_type)

# Or run a single target dataset on demand:
dts_engine = GridDigitalCoastEngine(param_lookup=param_lookup)
dts_engine.run(output_prefix='model_variables/Prediction', data_type='DistanceToShore')
```