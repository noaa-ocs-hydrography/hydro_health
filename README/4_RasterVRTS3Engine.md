# RasterVRTS3Engine

An asynchronous, cloud-optimized virtual mosaic pipeline designed to generate GDAL Virtual Rasters (VRTs) directly over spatial datasets hosted on AWS S3 without downloading the underlying raster imagery locally.

## 💡 Overview

The `RasterVRTS3Engine` reads Cloud Optimized GeoTIFFs (COGs) natively from S3 buckets via GDAL's Virtual File System (`/vsis3/`). It optimizes cloud read patterns and dynamically maps spatial datasets into grouped master VRT catalog index mosaics. 

The engine exposes two separate structural aggregation patterns depending on the provider specifications:
1. **BlueTopo Pipeline:** Explicitly warps every individual source tile to a geographic coordinate standard (**`EPSG:4326`**) over a bilinear interpolation approach before assembling a uniform master mosaic layout.
2. **DigitalCoast Pipeline:** Extracts individual structural Coordinate Reference System (CRS) data elements from native headers, groups spatial files into uniform target spatial buckets based on database properties, and forces master VRT generation to inherit projection metrics directly from matching tile indices.

---

## ⚙️ Core Architecture & State

* **Base Class:** Inherits from `Engine` (`hydro_health.engines.Engine`).
* **Spatial Reference Buffers:** Pre-loads comprehensive spatial criteria lookup profiles (`pyproj.database.query_crs_info`) identifying projected target coordinate systems across global `EPSG` configurations.
* **Target Mapping Mosaics:** Automatically routes file extension glob strings across multi-characteristic data properties:
  
  | Variable Target | Glob Signature Matching |
  | :--- | :--- |
  | `elevation` | `*[0-9].tiff` |
  | `uncertainty` | `*_unc.tiff` |
  | `slope` | `*_slope.tiff` |
  | `rugosity` | `*_rugosity.tiff` |
  | `NCMP` | `*.tif` |

---

## 🛠️ Module-Level Orchestrators

### `_clean(s)`
* **What it does:** Normalizes internal projection string identifiers for precise logic matching.
* **The Specifics:** Strips away non-breaking space escape flags (`\xa0`) and uniform space divisions, flattening characters down to lower-case string sequences.
* **Inputs:** `s` (`str`): Target raw spatial name properties.
* **Outputs:** `str`: Cleansed and uniform string token sequence.

### `_set_gdal_s3_options()`
* **What it does:** Inject optimized environment parameters tuning low-level GDAL execution speeds over AWS cloud files.
* **The Specifics:** Prevents network directory catalog timeouts by settings `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR`. It aggregates chunk arrays via network range grouping policies and initiates an active structural caching wrapper:
  $$\text{VSI Cache Size} = \text{10,000,000 Bytes (10MB)}$$

### `_process_single_bluetopo(params)`
* **What it does:** Re-projects and tracks standalone raw spatial tiles directly into an S3 VRT object layout.
* **The Specifics:** Generates localized working pointers using a fast `NamedTemporaryFile` system. Opens cloud links dynamically, triggers an instance of `gdal.Warp` targeting a flat geographic datum (`EPSG:4326`), extracts the horizontal authority code, uploads the VRT back to S3, and frees up local system allocations.
* **Inputs:** `params` (`list`): Array mapping out `[geotiff_prefix, s3_bucket, _]`.
* **Outputs:** `tuple[str, str, str]`: Tracking dataset values returning `(datum_code, final_s3_vrt_path, projection_wkt)`.

### `_read_geotiff_metadata(params)`
* **What it does:** Mines internal geospatial spatial reference arrays without streaming data frames.
* **The Specifics:** Connects to cloud streams, reads targeted NoData value boundaries, enforces traditional GIS coordinate ordering, and pulls matching code definitions. If structural codes are absent, it exports components down into a structured `PROJJSON` string to evaluate name matches against predefined database indexes.
* **Inputs:** `params` (`list`): Values passing `[geotiff_prefix, all_crs_info, data_folder]`.
* **Outputs:** `dict[str]`: Asset property data tracking `bin_key`, `vsi_path`, `nodata`, and `wkt` structures.

---

## 🛠️ Method Reference

### `run(outputs, file_type, ecoregion, data_type, data_folder, skip_existing)`
* **What it does:** Main execution module managing the distributed virtual mosaic aggregation loops.
* **The Specifics:** Configures active environment variables, registers parallel Dask context boundaries, queries global dataset indices from targeted S3 workspaces, loops across target providers, and builds localized target outputs using temporary scratch spaces.
* **Inputs:**
  * `outputs` (`str`): Path targeting master log directories.
  * `file_type` (`str`): Variable matching configuration keys (e.g., `'elevation'`, `'slope'`).
  * `ecoregion` (`str`): Target region tracking boundary (e.g., `'ER_3'`).
  * `data_type` (`str`): Pipeline engine target definition (`'BlueTopo'` or `'DigitalCoast'`).

### `build_output_vrts(output_prefix, file_type, output_geotiffs, temp_output_path, data_type)`
* **What it does:** Master builder module executing native GDAL virtual catalog compilation calls.
* **The Specifics:** Automatically checks data engine properties. If set to `DigitalCoast`, it builds configurations using a strict nearest-neighbor strategy (`near`), forces absolute NoData preservation rules, and sets properties to inherit spatial definitions from the primary source image. For alternative types, it defaults to a smooth bilinear interpolation configuration. It maps elements through `gdal.BuildVRT` and pushes finalized catalogs directly back onto storage clusters.
* **Inputs:**
  * `output_geotiffs` (`dict`): Properties grouping sorted tile arrays.
  * `temp_output_path` (`pathlib.Path`): Local working temporary folder routing paths.

---

## 🚀 Quick Start / Usage Example

```python
import pathlib
from hydro_health.engines.VRTBuilderEngine import RasterVRTS3Engine

# 1. Set up parameter reference tracking context
param_lookup = {'env': 'production'}

# 2. Instantiate the cloud VRT generation engine
vrt_engine = RasterVRTS3Engine(param_lookup=param_lookup)

# 3. Process structural variables across target properties 
vrt_engine.run(
    outputs="/var/log/vrt_runs",
    file_type="elevation",
    ecoregion="ER_2",
    data_type="BlueTopo"
)