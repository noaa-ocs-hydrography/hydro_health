# BlueTopoEngine

A high-performance, cross-platform geospatial data pipeline designed to ingest, process, resample, derive, and format bathymetric layers from NOAA's National Bathymetry Source (NBS) **BlueTopo** dataset.

This engine provides unified execution across both **local desktop environments (Windows)** and **distributed AWS cloud architectures (EC2/S3)**.

---

## 💡 Overview

The updated `BlueTopoEngine` manages parallel raster processing using **Dask** across dual operational environments:

* **Local Mode:** Streams data directly to localized output directories while leveraging local hardware for parallel Dask execution.
* **AWS Mode:** Orchestrates an isolated S3-to-S3 workflow using ephemeral local NVMe/scratch storage (`~/scratch_tmp/BlueTopoEngine`) for temp processing, streaming public NOAA bathymetry directly into private AWS S3 destinations.

---

## ⚙️ Core Architecture & Updates

* **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
* **Environment Support:** Handles unified execution paths via the `env` runtime parameter (`'aws'` vs. `'local'`).
* **Execution Paradigm:** Parallel worker distribution via Dask (`_process_tile` orchestrator).
* **Storage Interface:** Hybrid memory-safe disk-streaming using isolated `TemporaryDirectory` blocks, GDAL VRTs, and memory-mapped block windows.
* **Cloud Integration:** Anonymous access to NOAA's public `noaa-ocs-nationalbathymetry-pds` bucket, with automated multi-band export to private target AWS S3 buckets.

---

## 🛠️ Key Technical Features

1. **Smart Resampling & Reprojection (`resample_and_reproject`)**
* Dynamically checks pixel grid resolutions to skip redundant processing.
* Isolates Elevation ($B1$) & Uncertainty ($B2$) via **Bilinear** interpolation while forcing Contributor Metadata ($B3$) through **Nearest Neighbor** resampling.
* Stitches bands back using zero-RAM GDAL Virtual Format (VRT) streaming.
* Safely preserves and restores PAM XML Raster Attribute Tables (RAT) across thread-safe execution contexts.


2. **Derivatives & Quality Analysis (`create_catzoc_*`, `create_survey_end_date_tiff`, `create_slope`)**
* **ISS All / Latest:** Scrapes XML GDAL attribute tables using `lxml.etree` to score initial survey quality across all or the most recent survey parameters.
* **Survey End Date:** Maps contributor attributes to output integer temporal grids representing survey end years.
* **Slope:** Computes surface slopes utilizing GDAL's DEM processing API.


3. **Memory-Safe Land Masking (`set_ground_to_nodata`)**
* Uses Rasterio block-window iteration (`block_windows`) to convert positive above-water elevation ($\ge 0$) to NoData (`-9999`) without loading full raster grids into RAM.


4. **Band Splitting & COG Optimization (`multiband_to_singleband`, `finalize_cog`)**
* Splits multi-band rasters into separate single-band products (`_unc` for uncertainty).
* Generates strict Cloud-Optimized GeoTIFFs (COGs) with internal DEFLATE compression, floating-point predictors (`PREDICTOR=3`), bilinear overviews, and 512x512 tiling.



---

## 🛠️ Method Reference

### Entry Point

* `_process_tile(param_inputs)`: Static Dask worker dispatching execution for individual tiles. Runs downloads, resampling, quality scoring, band extraction, masking, COG finalization, and S3 uploads.

### Core Engine Methods

* `run(tile_gdf, output_prefix, resolution)`: Main pipeline entry point. Parses boundary targets, configures environment-specific Dask clusters, submits tile processing tasks, writes run manifests, and emits metric telemetry via `hibase_logging`.
* `download_nbs_tile(tile_id, ecoregion_id, output_prefix, target_res, temp_folder)`: Downloads source TIFF/XML assets directly from NOAA's S3 repository to local or temporary paths.
* `resample_and_reproject(tiff_file_path, target_res)`: Warps 3-band source tiles to target spatial grid scales while maintaining data integrity across discrete and continuous bands.
* `create_catzoc_all(tiff_file_path, increased_scale)`: Calculates Initial Survey Scores (ISS) for all combined survey areas via GDAL RAT parsing.
* `create_catzoc_latest(tiff_file_path, increased_scale)`: Calculates Initial Survey Scores (ISS) specifically filtered by the most recent survey date.
* `create_survey_end_date_tiff(tiff_file_path)`: Produces a GeoTIFF mapping pixel coverage to the year the survey was finalized.
* `create_slope(tiff_file_path)`: Derives physical slope rasters from elevation bands.
* `set_ground_to_nodata(tiff_file_path)`: Masks above-water pixels ($\ge 0$) to `-9999` using chunked window streaming.
* `multiband_to_singleband(tiff_file_path, band)`: Extracts individual bands into distinct raster files.
* `finalize_cog(tiff_path)`: Converts intermediate raster files into strict Cloud-Optimized GeoTIFF layout formats.
* `upload_current_tiles_to_s3(tile_folder, temp_folder)`: Syncs output rasters to the designated destination S3 bucket.

---

## 🚀 Usage Example

# 1. Define configuration dictionary (AWS or Local)
param_lookup = {
    'env': 'aws',  # Options: 'aws' or 'local'
    'eco_regions': ['ER_3'],
    'output_directory': OUTPUTS
}

# 2. Instantiate unified engine
engine = BlueTopoEngine(param_lookup=param_lookup)

# 4. Execute pipeline across target resolutions
engine.run(
    tile_gdf=boundaries_gdf,
    output_prefix="regional_model",
    resolution=[8]
)

```