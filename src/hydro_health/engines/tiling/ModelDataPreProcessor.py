"""Class for data acquisition and preprocessing of model data"""

import os
import re
import pathlib
import yaml
import warnings
import tempfile
import shutil
import logging 
import psutil 
import gc
import platform
import ctypes
from logging.handlers import RotatingFileHandler
from typing import List, Tuple, Literal
from pathlib import Path

# Rasterio imports for array masking and vectorization
import rasterio
from rasterio.features import shapes, rasterize 
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from shapely.geometry import shape, Point, box, Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union 
import s3fs
from scipy.ndimage import convolve, uniform_filter 

# Tell glibc to return freed memory to the OS immediately
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

import geopandas as gpd
import numpy as np
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster, performance_report, as_completed
from osgeo import gdal
from upath import UPath 

from hydro_health.helpers.tools import get_config_item, get_environment
from hydro_health.engines.CreateSeabedTerrainLayerEngine import CreateSeabedTerrainLayerEngine
from hydro_health.engines.Engine import Engine

# Maximizing worker memory usage limits
dask.config.set({"distributed.worker.memory.terminate": 0.98})
dask.config.set({"distributed.worker.memory.pause": 0.95})
dask.config.set({"distributed.worker.memory.spill": 0.92})

# =========================================================================
# GDAL CONFIGURATION & S3 NETWORK OPTIMIZATIONS
# =========================================================================
GDAL_ENV_VARS = {
    "GDAL_CACHEMAX": "128",                       # Lowered to 128 MB Cache
    "GDAL_HTTP_MAX_RETRY": "10",                  # Increased to 10 retries for transient S3 errors
    "GDAL_HTTP_RETRY_DELAY": "5",                 # Delay between retries
    "AWS_MAX_CONNECTIONS": "16",                  # Reduced to 16 to avoid exhausting S3 connection limits per worker
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": "67108864",                 # 64 MB VSI Cache
    "CHECK_DISK_FREE_SPACE": "FALSE",    
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", 
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.vrt,.gpkg,.parquet", 
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",  # Avoids tiny contiguous read requests
    "CPL_VSIL_CURL_CHUNK_SIZE": "1048576",         # 1MB blocks to significantly cut down request round-trips
    "GDAL_HTTP_MULTIPLEX": "YES",                  # Enables HTTP/2 multiplexing over single TCP connection
    "GDAL_HTTP_TIMEOUT": "30",                     # Prevent silent hanging connections
    "GDAL_HTTP_CONNECTTIMEOUT": "10",              # Fail-fast on stale connections
    "CPL_VSIL_CURL_USE_HEAD": "NO",               # Drastically reduces rate-limiting HEAD requests to S3
    "GDAL_INGested_BYTES_AT_OPEN": "32768"         # Caches metadata header bytes to minimize initial range requests
}

# Apply env configurations globally to the master process
for key, val in GDAL_ENV_VARS.items():
    os.environ[key] = val

# ==========================================
# LOGGING CONFIGURATION
# ==========================================
LOG_FILE_PATH = Path.home() / "hydro_health_preprocessing.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(processName)-15s | %(message)s",
    handlers=[
        RotatingFileHandler(
            LOG_FILE_PATH, maxBytes=10*1024*1024, backupCount=3
        ), 
        logging.StreamHandler()            
    ]
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party logs
logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('boto3').setLevel(logging.WARNING)
logging.getLogger('s3fs').setLevel(logging.WARNING)


class ModelDataPreProcessor(Engine):
    """Class for parallel preprocessing all model data"""

    def __init__(self, overwrite: bool = False, pilot_mode: bool=False):
        super().__init__()
        self.pilot_mode = pilot_mode
        self.overwrite = overwrite

        self.fs = s3fs.S3FileSystem(anon=False)

        self.static_patterns = ['sed_size_raster', 'sed_type_raster', 'tsm_mean', 'hurr']
        self.re_year_suffix = re.compile(r"_\d{4}$")
        self.re_year_extract = re.compile(r"_(\d{4})")
        self.re_bt_prefix = re.compile(r"^bt\.")
        self.re_flowdir = re.compile(r"flowdir")

        self.excluded_keys = self._load_exclusion_config()
        self.is_aws = (get_environment() == 'aws')

    @staticmethod
    def _trim_memory() -> None:
        """
        Aggressively forces garbage collection and tells the OS to reclaim freed memory.
        This resolves Dask's 'Unmanaged memory' warnings caused by glibc hoarding memory
        from pandas DataFrames and numpy arrays.
        """
        gc.collect()
        if platform.system() == "Linux":
            try:
                libc = ctypes.CDLL("libc.so.6")
                libc.malloc_trim(0)
            except Exception:
                pass

    def create_file_paths(self):
        """Creates unified UPath objects that work both locally and on S3."""
        prefix = f"s3://{get_config_item('S3', 'BUCKET_NAME', pilot_mode=self.pilot_mode)}/" if self.is_aws else ""
        logger.info(f"Environment detected: {'AWS' if self.is_aws else 'Local/Remote'}")
        logger.info(f"Mode detected: {'Pilot' if self.pilot_mode else 'Full'}")
        
        self.mask_prediction_pq = UPath(f"{prefix}{get_config_item('MASK', 'PREDICTION_MASK_PQ', pilot_mode=self.pilot_mode)}")
        self.mask_training_pq = UPath(f"{prefix}{get_config_item('MASK', 'TRAINING_MASK_PQ', pilot_mode=self.pilot_mode)}")
        self.grid_gpkg = UPath(f"{prefix}{get_config_item('MODEL', 'SUBGRIDS', pilot_mode=self.pilot_mode)}")
        self.pred_mask_path = UPath(f"{prefix}{get_config_item('MASK', 'MASK_PRED_PATH', pilot_mode=self.pilot_mode)}")
        self.train_mask_path = UPath(f"{prefix}{get_config_item('MASK', 'MASK_TRAINING_PATH', pilot_mode=self.pilot_mode)}")
        self.preprocessed_dir = UPath(f"{prefix}{get_config_item('MODEL', 'PREPROCESSED_DIR', pilot_mode=self.pilot_mode)}")
        self.prediction_out_dir = UPath(f"{prefix}{get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR', pilot_mode=self.pilot_mode)}")
        self.training_out_dir = UPath(f"{prefix}{get_config_item('MODEL', 'TRAINING_OUTPUT_DIR', pilot_mode=self.pilot_mode)}")
        self.training_tiles_dir = UPath(f"{prefix}{get_config_item('MODEL', 'TRAINING_TILES_DIR', pilot_mode=self.pilot_mode)}")
        self.prediction_tiles_dir = UPath(f"{prefix}{get_config_item('MODEL', 'PREDICTION_TILES_DIR', pilot_mode=self.pilot_mode)}")
        self.uncombined_lidar_dir = UPath(f"{prefix}{get_config_item('MODEL', 'UNCOMBINED_LIDAR_DIR', pilot_mode=self.pilot_mode)}")
        self.subgrid_paths = {
            'training': UPath(f"{prefix}{get_config_item('MODEL', 'TRAINING_SUB_GRIDS', pilot_mode=self.pilot_mode)}"),
            'prediction': UPath(f"{prefix}{get_config_item('MODEL', 'PREDICTION_SUB_GRIDS', pilot_mode=self.pilot_mode)}")
        }

        self.preprocessed_subdirs = {
            'bluetopo': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'BLUETOPO', pilot_mode=self.pilot_mode)}"),
            'hurricane': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'HURRICANE', pilot_mode=self.pilot_mode)}"),
            'lidar': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'LIDAR', pilot_mode=self.pilot_mode)}"),
            'sediment': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'SEDIMENT', pilot_mode=self.pilot_mode)}"),
            'tsm': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'TSM', pilot_mode=self.pilot_mode)}")
        }
        
        self.local_tmp_dir = Path.home() / "hydro_health_local_tmp"

    def _clean_local_tmp(self) -> None:
        """Empties the local temporary directory to prevent disk space exhaustion from previous failed runs."""
        if self.local_tmp_dir.exists():
            logger.info(f"Cleaning up existing local temporary directory: {self.local_tmp_dir}")
            shutil.rmtree(self.local_tmp_dir, ignore_errors=True)
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)

    def _load_exclusion_config(self) -> set:
        """Loads dataset exclusion keys from YAML config."""
        try:
            inputs_root = pathlib.Path(__file__).parents[4] / 'inputs'
            config_path = inputs_root / 'lookups' / 'ER_3_lidar_data_config.yaml'
            
            if not config_path.exists():
                logger.warning(f"Exclusion config path not found: {config_path}")
                return set()

            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)

            excluded = {
                key for key, data in config_data.get('EcoRegion-3', {}).items()
                if data.get('use') is False
            }
            
            if excluded:
                logger.info(f"Loaded {len(excluded)} exclusion keys from config.")
            
            return excluded
        except Exception as e:
            logger.exception(f"Loading exclusion config failed: {e}")
            return set()

    def process(self) -> None:
        """Main function to process model data."""   
        logger.info(f"Starting ModelDataPreProcessor. Logs are being saved to: {LOG_FILE_PATH}")
        self.create_file_paths()
        self._clean_local_tmp()

        # NOTE: Phase 1 is for Raster operations which are block-based and use less working memory
        logger.info("Initializing Phase 1 Cluster: Heavy Raster Processing (8 workers, standard memory)")
        cluster = LocalCluster(
            n_workers=8,            
            threads_per_worker=1,  
            memory_limit='3.5GB',
            env=GDAL_ENV_VARS,
            local_directory=str(self.local_tmp_dir) # Route Dask spills to cleanable local tmp
        )
        client = Client(cluster)
        
        logger.info(f"Phase 1 Dask Dashboard: {client.dashboard_link}")

        try:        
            report_file_raster = "dask_performance_report_rasters.html"
            logger.info(f"Saving Phase 1 Dask performance report to: {report_file_raster}")
            
            with performance_report(filename=report_file_raster):
                mask_pred_gdf = gpd.read_parquet(str(self.mask_prediction_pq))
                mask_train_gdf = gpd.read_parquet(str(self.mask_training_pq))

                # Replaced .union_all() with direct bounding box calculation and raw shape export.
                logger.info("Extracting bounds and exporting geometries...")
                mask_pred_bounds = mask_pred_gdf.total_bounds
                mask_train_bounds = mask_train_gdf.total_bounds

                logger.info("Generating cutline files (using GeoPackage for fast spatial indexing)...")
                pred_cutline_path = str(self.local_tmp_dir / "pred_cutline.gpkg")
                train_cutline_path = str(self.local_tmp_dir / "train_cutline.gpkg")
                
                # Using GPKG natively builds a spatial R-Tree index
                mask_pred_gdf.to_file(pred_cutline_path, driver='GPKG')
                mask_train_gdf.to_file(train_cutline_path, driver='GPKG')

                self.parallel_processing_rasters(
                    self.preprocessed_dir, 
                    mask_pred_bounds, 
                    mask_train_bounds,
                    pred_cutline_path,
                    train_cutline_path
                )

            # --- PHASE 2 CLUSTER TRANSITION ---
            logger.info("Phase 1 Complete. Shutting down raster cluster and re-initializing for Parquet Subtiling...")
            client.close()
            cluster.close()
            
            logger.info("Initializing Phase 2 Cluster: Parquet Subtiling & Transforms (Balanced workers, standard memory)")
            cluster = LocalCluster(
                n_workers=8,            
                threads_per_worker=1,  
                memory_limit='3.5GB',
                env=GDAL_ENV_VARS,
                local_directory=str(self.local_tmp_dir) # Route Dask spills to cleanable local tmp
            )
            client = Client(cluster)
            logger.info(f"Phase 2 Dask Dashboard: {client.dashboard_link}")
            
            report_file_tiling = "dask_performance_report_tiling.html"
            logger.info(f"Saving Phase 2 Dask performance report to: {report_file_tiling}")

            with performance_report(filename=report_file_tiling):
                # self.clip_rasters_by_tile(
                #     raster_dir=self.prediction_out_dir, 
                #     output_dir=self.prediction_tiles_dir, 
                #     data_type="prediction"
                # )

                # self.clip_rasters_by_tile(
                #     raster_dir=self.training_out_dir, 
                #     output_dir=self.training_tiles_dir, 
                #     data_type="training"
                # )
                
                self.batch_long_format_transformation(base_dir=self.prediction_tiles_dir, mode="prediction")
                self.batch_long_format_transformation(base_dir=self.training_tiles_dir, mode="training")

        except Exception as e:
            logger.exception("A critical error occurred in the main process loop.")
        finally:
            try:
                client.close()
                cluster.close()
            except:
                pass
            
    def _create_binary_mask_raster(self, cutline_path, bounds, output_path) -> str:
        """
        Creates a global binary raster mask (1 for valid, 0 for invalid) 
        from a vector layer before running Dask distributed processes.
        """
        logger.info(f"Burning vector mask into binary raster: {output_path}")
        gdf = gpd.read_file(cutline_path)
        
        minx, miny, maxx, maxy = bounds
        res = self.target_res
        
        # Snap bounds to target resolution grid
        minx = np.floor(minx / res) * res
        maxy = np.ceil(maxy / res) * res
        maxx = np.ceil(maxx / res) * res
        miny = np.floor(miny / res) * res
        
        width = int((maxx - minx) / res)
        height = int((maxy - miny) / res)
        
        # Generate the affine transform
        transform = from_origin(minx, maxy, res, res)
        
        # Rasterize shapes (burn value 1)
        shapes_gen = ((geom, 1) for geom in gdf.geometry)
        mask_arr = rasterize(
            shapes=shapes_gen,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )
        
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'uint8',
            'crs': self.target_crs,
            'transform': transform,
            'compress': 'lzw',
            'tiled': True,
            'nodata': 0
        }
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask_arr, 1)
            
        return output_path

    def parallel_processing_rasters(self, input_directory, mask_pred_bounds, mask_train_bounds, pred_cutline_path, train_cutline_path) -> None:
        """Process prediction and training rasters in parallel using Dask."""
        input_directory = UPath(input_directory)
        
        if not self.is_aws:
            self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)
            self.training_out_dir.mkdir(parents=True, exist_ok=True)

        existing_pred_outputs = {
            f.name for f in self.prediction_out_dir.rglob("*")
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        existing_uncombined_outputs = {
            f.name for f in self.uncombined_lidar_dir.rglob("*")
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        all_existing_pred_outputs = existing_pred_outputs.union(existing_uncombined_outputs)

        existing_train_outputs = {
            f.name for f in self.training_out_dir.rglob("*")
            if f.suffix.lower() in {'.tif', '.tiff'}
        }

        potential_files = []
        logger.info(f"Scanning for preprocessed input rasters in: {self.preprocessed_dir}")

        for data_type, directory in self.preprocessed_subdirs.items():
            found_files = [
                f for f in directory.rglob("*") 
                if f.suffix.lower() in {'.tif', '.tiff'}
            ]
            logger.info(f" -> {data_type.capitalize()} directory: Found {len(found_files)} files.")
            
            if not found_files:
                raise RuntimeError(
                    f"CRITICAL ERROR: Missing data for '{data_type}'. "
                    f"No .tif files were found in {directory}."
                )
                
            potential_files.extend(found_files)

        logger.info(f"Found {len(potential_files)} total potential source files in input directories.")

        excluded_folders = {'filled_tifs', 'combined_lidar_tifs'}
        valid_source_files = []
        removed_folders = 0
        files_removed_by_keys = []

        for f in potential_files:
            if any(key in f.name for key in self.excluded_keys):
                files_removed_by_keys.append(f.name) 
                continue
                
            if any(folder in f.parts for folder in excluded_folders):
                removed_folders += 1
                continue
                
            valid_source_files.append(f)

        logger.info(f"--- File Filtering Summary ---")
        logger.info(f" -> Total potential files: {len(potential_files)}")
        logger.info(f" -> Removed (excluded keys): {len(files_removed_by_keys)}") 
        logger.info(f" -> Removed (excluded folders): {removed_folders}")
        logger.info(f" -> Valid source files for processing: {len(valid_source_files)}")
        logger.info(f"------------------------------------")

        prediction_files = []
        removed_existing_pred = 0
        
        for f in valid_source_files:
            if not self.overwrite and f.name in all_existing_pred_outputs:
                removed_existing_pred += 1
                continue
            prediction_files.append(f)

        skip_pred_msg = f" (Skipping {removed_existing_pred} existing)" if not self.overwrite else " (Overwrite enabled)"
        
        logger.info(f"Outputting uncombined lidar to: {self.uncombined_lidar_dir}")
        logger.info(f"Outputting prediction rasters to: {self.prediction_out_dir}")
        logger.info(f"Queuing {len(prediction_files)} prediction files{skip_pred_msg}...")
        
        # -------------------------------------------------------------
        # DYNAMIC DASK TASK STREAM (PREDICTION)
        # Prevents idle workers by instantly replacing completed tasks
        # -------------------------------------------------------------
        client = dask.distributed.client.default_client()
        max_concurrent = 200 # Optimal buffer to keep scheduler fast while workers stay saturated
        total_pred = len(prediction_files)
        prediction_iterator = iter(enumerate(prediction_files))
        seq = as_completed()
        
        def submit_pred_task(item):
            i, file_path = item
            base_out = self.uncombined_lidar_dir if "mosaic" in file_path.name.lower() else self.prediction_out_dir
            output_path = base_out / file_path.name
            return client.submit(
                self.process_prediction_raster,
                str(file_path), 
                mask_pred_bounds, 
                str(output_path),
                pred_cutline_path
            )

        # Initial queue fill
        for _ in range(min(max_concurrent, total_pred)):
            try:
                seq.add(submit_pred_task(next(prediction_iterator)))
            except StopIteration:
                break

        # Process stream
        for future in seq:
            future.result() # Raise exceptions if any occurred
            try:
                seq.add(submit_pred_task(next(prediction_iterator)))
            except StopIteration:
                pass

        if total_pred > 0:
            logger.info("[SUCCESS] Prediction raster processing complete.")
        else:
            logger.info("No new prediction rasters to process.")

        # HARD FLUSH: Clear out any accumulated VSI Cache / memory blocks before handing off to the Engine
        logger.info("Restarting Dask client to aggressively flush unmanaged memory before starting Seabed Engine...")
        try:
            client = dask.distributed.client.default_client()
            client.restart()
        except Exception as e:
            logger.warning(f"Could not restart client before Seabed Engine: {e}")

        logger.info("Running Seabed Terrain Layer Engine...")
        engine = CreateSeabedTerrainLayerEngine()
        engine.process()

        # HARD FLUSH #2: Clean up after the Seabed Engine completes to keep training isolated
        logger.info("Restarting Dask client to flush memory after Seabed Engine completion...")
        try:
            client = dask.distributed.client.default_client()
            client.restart()
        except Exception as e:
            logger.warning(f"Could not restart client after Seabed Engine: {e}")

        # GENERATE BINARY TRAINING MASK PRIOR TO DASK WORKERS
        global_mask_path = str(self.local_tmp_dir / "global_train_mask.tif")
        self._create_binary_mask_raster(train_cutline_path, mask_train_bounds, global_mask_path)

        potential_train_inputs = list(self.prediction_out_dir.rglob("*"))
        training_candidates = [
            f for f in potential_train_inputs
            if f.suffix.lower() in {'.tif', '.tiff'}
        ]

        training_files = []
        removed_existing_train = 0
        
        for f in training_candidates:
            if 'mosaic' in f.name.lower() and 'filled' not in f.name.lower():
                continue
                
            if not self.overwrite and f.name in existing_train_outputs:
                removed_existing_train += 1
                continue
                
            training_files.append(f)

        skip_train_msg = f" (Skipping {removed_existing_train} existing)" if not self.overwrite else " (Overwrite enabled)"
        
        logger.info(f"Outputting training rasters to: {self.training_out_dir}")
        logger.info(f"Queuing {len(training_files)} training files{skip_train_msg}...")

        # -------------------------------------------------------------
        # DYNAMIC DASK TASK STREAM (TRAINING)
        # Prevents idle workers by instantly replacing completed tasks
        # -------------------------------------------------------------
        client = dask.distributed.client.default_client()
        total_train = len(training_files)
        training_iterator = iter(enumerate(training_files))
        seq_train = as_completed()
        
        def submit_train_task(item):
            i, file_path = item
            output_path = self.training_out_dir / file_path.name
            return client.submit(
                self.process_training_raster,
                str(file_path), 
                mask_train_bounds, 
                str(output_path),
                global_mask_path,
                current_index=i + 1,
                total_count=total_train
            )

        # Initial queue fill
        for _ in range(min(max_concurrent, total_train)):
            try:
                seq_train.add(submit_train_task(next(training_iterator)))
            except StopIteration:
                break

        # Process stream
        for future in seq_train:
            future.result() 
            try:
                seq_train.add(submit_train_task(next(training_iterator)))
            except StopIteration:
                pass

        if total_train > 0:
            logger.info("[SUCCESS] Training raster processing complete.")
        else:
            logger.info("No new training rasters to process.")

    def process_prediction_raster(self, raster_path, mask_bounds, output_path, cutline_path) -> None:
        """Reprojects, resamples, and crops a raster for prediction."""
        try:
            raster_name = pathlib.Path(raster_path).name.lower()
            open_path = str(raster_path)
            
            if self.is_aws and open_path.startswith('s3://'):
                open_path = open_path.replace('s3://', '/vsis3/')
                
            logger.info(f"-> [STARTING] Worker executing prediction on: {raster_name}")

            try:
                with rasterio.open(open_path) as src:
                    src_nodata = src.nodata
                    raster_crs = src.crs
                    raster_bounds = src.bounds
            except Exception as e:
                logger.exception(f"Could not open {raster_name} with rasterio. File might be corrupted.")
                return

            if raster_crs is not None:
                try:
                    target_crs_obj = rasterio.crs.CRS.from_string(self.target_crs)
                    if raster_crs != target_crs_obj:
                        left, bottom, right, top = transform_bounds(raster_crs, target_crs_obj, *raster_bounds)
                        bounds_geom = box(left, bottom, right, top)
                    else:
                        bounds_geom = box(*raster_bounds)
                except Exception as e:
                    logger.warning(f"Failed to transform bounds for {raster_name}: {e}. Falling back to native bounds.")
                    bounds_geom = box(*raster_bounds)
            else:
                bounds_geom = box(*raster_bounds)

            try:
                mask_box = box(*mask_bounds)
                if not mask_box.intersects(bounds_geom):
                    logger.info(f"- [SKIP] Bounding box does not intersect prediction raster {raster_name}.")
                    return
            except Exception as e:
                logger.exception(f"Bounding box check failed for {raster_name}.")
                return

            logger.info(f" [PROCESSING] Starting warp on prediction file {raster_name}...")
            should_crop = any(k in raster_name for k in ["tsm", "sed", "hurr"])
            is_tsm = "tsm" in raster_name or "strength" in raster_name

            try:
                self._warp_to_cutline(
                    raster_path, 
                    output_path, 
                    cutline_path, 
                    dst_crs=self.target_crs, 
                    x_res=self.target_res, 
                    y_res=self.target_res,
                    crop_to_cutline=should_crop,
                    src_nodata=src_nodata,
                    apply_tsm_smoothing=is_tsm,
                    resample_alg='bilinear' 
                )
            except Exception as e:
                logger.exception(f"Unexpected failure during _warp_to_cutline for {raster_name}.")
        finally:
            self._trim_memory()

    def process_training_raster(self, raster_path, mask_bounds, output_path, global_mask_path, current_index=None, total_count=None) -> None:
        """Process a training raster by extracting array blocks and masking them mathematically."""
        try:
            raster_name = pathlib.Path(raster_path).name.lower()
            open_path = str(raster_path)
            
            progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
            
            if self.is_aws and open_path.startswith('s3://'):
                open_path = open_path.replace('s3://', '/vsis3/')

            logger.info(f"-> [STARTING]{progress_str} Worker executing training array mask on: {raster_name}")

            try:
                with rasterio.open(open_path) as src_pred:
                    src_nodata = src_pred.nodata if src_pred.nodata is not None else np.nan
                    
                    # Check bounding box intersections quickly
                    raster_bounds_geom = box(*src_pred.bounds)
                    mask_box = box(*mask_bounds)
                    if not mask_box.intersects(raster_bounds_geom):
                        logger.info(f"- [SKIP]{progress_str} Bounding box does not intersect raster {raster_name}. Skipping.")
                        return
                    
                    meta = src_pred.meta.copy()
                    meta.update({
                        'nodata': np.nan if np.isnan(src_nodata) else src_nodata,
                        'compress': 'lzw',
                        'tiled': True
                    })

                    # Setup temporary local path if interacting with S3 to avoid streaming writes
                    tmp_dst_path = str(output_path)
                    if self.is_aws:
                        tmp_out = tempfile.NamedTemporaryFile(dir=self.local_tmp_dir, suffix='.tif', delete=False)
                        tmp_dst_path = tmp_out.name
                        tmp_out.close()

                    with rasterio.open(global_mask_path) as src_mask:
                        # Virtual re-alignment ensures the mask array is perfectly registered 
                        # to the incoming prediction raster (even if it was cropped/offset slightly)
                        with WarpedVRT(src_mask, crs=src_pred.crs, transform=src_pred.transform, 
                                       height=src_pred.height, width=src_pred.width, 
                                       resampling=Resampling.nearest) as vrt_mask:
                            
                            with rasterio.Env(CHECK_DISK_FREE_SPACE="FALSE"):
                                with rasterio.open(tmp_dst_path, 'w', **meta) as dest:
                                    
                                    # Evaluate the arrays safely in memory chunks to prevent Dask limits from being exceeded
                                    for ji, window in src_pred.block_windows(1):
                                        pred_arr = src_pred.read(1, window=window)
                                        mask_arr = vrt_mask.read(1, window=window)

                                        # Cast integers to floats if the nodata value is NaN
                                        if np.isnan(meta['nodata']) and pred_arr.dtype not in (np.float32, np.float64):
                                            pred_arr = pred_arr.astype(np.float32)

                                        # Apply mask logic via numpy 
                                        masked_data = np.where(mask_arr == 1, pred_arr, meta['nodata'])
                                        dest.write(masked_data, 1, window=window)
                    
                    # If on AWS, push complete file from fast local disk to S3 bucket
                    if self.is_aws:
                        self.fs.put(tmp_dst_path, str(output_path))
                        os.remove(tmp_dst_path)

                logger.info(f" - [✓ SUCCESS]{progress_str} Processed training raster via array masking: {raster_name}")
                
            except Exception as e:
                logger.exception(f"Unexpected failure during array masking for {raster_name}.")
                # Cleanup temp files if exception occurred
                if self.is_aws and 'tmp_dst_path' in locals() and os.path.exists(tmp_dst_path):
                    os.remove(tmp_dst_path)
        finally:
            self._trim_memory()

    def _warp_to_cutline(self, src_path, dst_path, cutline_path, **kwargs):
        """Helper to handle GDAL Warp boilerplate."""
        src_str = str(src_path)
        dst_str = str(dst_path)

        if self.is_aws and src_str.startswith('s3://'):
            src_str = src_str.replace('s3://', '/vsis3/')

        if self.is_aws:
            with tempfile.NamedTemporaryFile(dir=self.local_tmp_dir, suffix='.tif', delete=False) as tmp_out:
                gdal_dst_str = tmp_out.name
        else:
            gdal_dst_str = dst_str

        resample_alg = kwargs.pop('resample_alg', None) 

        warp_opts = {
            'cutlineDSName': cutline_path,
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE'],
            'creationOptions': [
                'TILED=YES', 
                'BLOCKXSIZE=512',
                'BLOCKYSIZE=512',
                'COMPRESS=LZW',
                'BIGTIFF=YES',
                'NUM_THREADS=1'
            ],
            'multithread': False,
            'warpMemoryLimit': 1024, # 512 MB
            'outputType': gdal.GDT_Float32 
        }
        
        if resample_alg:
            warp_opts['resampleAlg'] = resample_alg 
            
        if 'dst_crs' in kwargs: warp_opts['dstSRS'] = kwargs.pop('dst_crs')
        if 'x_res' in kwargs: warp_opts['xRes'] = kwargs.pop('x_res')
        if 'y_res' in kwargs: warp_opts['yRes'] = kwargs.pop('y_res')
        if 'crop_to_cutline' in kwargs: warp_opts['cropToCutline'] = kwargs.pop('crop_to_cutline')
        if 'src_nodata' in kwargs: warp_opts['srcNodata'] = kwargs.pop('src_nodata')
        if 'dst_nodata' in kwargs: warp_opts['dstNodata'] = kwargs.pop('dst_nodata')
        
        apply_tsm_smoothing = kwargs.pop('apply_tsm_smoothing', False)
        
        try:
            ds = gdal.Warp(gdal_dst_str, src_str, **warp_opts)

            if ds is None:
                raise RuntimeError(f"gdal.Warp returned None for {os.path.basename(src_str)}")

            # Release dataset handle explicitly if it matches a bathymetry layer so that
            # we can run the positive-elevation land filtering block in read+write mode safely.
            src_basename = os.path.basename(src_str).lower()
            apply_bathy_filter = any(k in src_basename for k in ["lidar", "bathy", "bluetopo"])

            if apply_bathy_filter:
                ds = None  # Release the dataset lock
                logger.info(f" [BATHY FILTER] Applying elevation filter (values >= 0 -> nodata) on: {os.path.basename(src_str)}")
                with rasterio.Env(CHECK_DISK_FREE_SPACE="FALSE"):
                    with rasterio.open(gdal_dst_str, 'r+') as dst:
                        nodata_val = dst.nodata if dst.nodata is not None else -9999.0
                        for ji, window in dst.block_windows(1):
                            arr = dst.read(1, window=window)
                            mask_invalid = (arr >= 0.0)
                            if mask_invalid.any():
                                arr[mask_invalid] = nodata_val
                                dst.write(arr, 1, window=window)

            if apply_tsm_smoothing:
                if ds is None:
                    ds = gdal.Open(gdal_dst_str)
                mem = psutil.virtual_memory()
                logger.info(f" [SMOOTHING INIT] {os.path.basename(src_str)} | Sys RAM: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")

                pixel_size = ds.GetGeoTransform()[1]
                radius_pixels = int(2000 / abs(pixel_size))
                size = radius_pixels * 2 + 1
                
                ds = None
                
                smoothed_tmp = gdal_dst_str.replace('.tif', '_smoothed.tif')

                with rasterio.open(gdal_dst_str) as src:
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'dtype': 'float32', 
                        'tiled': True,
                        'blockxsize': 512,
                        'blockysize': 512,
                        'compress': 'lzw', 
                        'bigtiff': 'yes' 
                    })
                    nodata = src.nodata if src.nodata is not None else warp_opts.get('dstNodata', warp_opts.get('srcNodata', -9999.0))
                    
                    block_size = 1024
                    total_chunks_x = (src.width + block_size - 1) // block_size
                    total_chunks_y = (src.height + block_size - 1) // block_size
                    total_chunks = total_chunks_x * total_chunks_y
                    current_chunk = 0
                    
                    with rasterio.Env(CHECK_DISK_FREE_SPACE="FALSE"): 
                        with rasterio.open(smoothed_tmp, 'w', **kwargs) as dst:
                            for y in range(0, src.height, block_size):
                                for x in range(0, src.width, block_size):
                                    
                                    current_chunk += 1
                                    if current_chunk % max(1, total_chunks // 10) == 0 or current_chunk == total_chunks:
                                        mem = psutil.virtual_memory()
                                        logger.info(f"   -> [PROGRESS] {os.path.basename(src_str)} Smoothing: Chunk {current_chunk}/{total_chunks} | Sys RAM: {mem.percent}%")

                                    core_width = min(block_size, src.width - x)
                                    core_height = min(block_size, src.height - y)
                                    window = rasterio.windows.Window(x, y, core_width, core_height)
                                    
                                    read_xoff = max(0, x - radius_pixels)
                                    read_yoff = max(0, y - radius_pixels)
                                    read_right = min(src.width, x + core_width + radius_pixels)
                                    read_bottom = min(src.height, y + core_height + radius_pixels)
                                    
                                    read_window = rasterio.windows.Window(
                                        read_xoff, 
                                        read_yoff, 
                                        read_right - read_xoff, 
                                        read_bottom - read_yoff
                                    )
                                    
                                    array = src.read(1, window=read_window).astype(np.float32)
                                    
                                    if pd.isna(nodata):
                                        valid_mask = (~np.isnan(array)).astype(np.float32)
                                        array[np.isnan(array)] = 0
                                    else:
                                        valid_mask = (array != nodata).astype(np.float32)
                                        array[array == nodata] = 0

                                    smoothed = uniform_filter(array, size=size, mode='constant', cval=0.0)
                                    weights = uniform_filter(valid_mask, size=size, mode='constant', cval=0.0)
                                    
                                    with np.errstate(divide='ignore', invalid='ignore'):
                                        final_array = np.where(weights > 0, smoothed / weights, nodata)
                                    
                                    row_offset = int(y - read_yoff)
                                    col_offset = int(x - read_xoff)
                                    core_array = final_array[row_offset:row_offset + core_height, col_offset:col_offset + core_width]
                                    
                                    original_core = array[row_offset:row_offset + core_height, col_offset:col_offset + core_width]
                                    
                                    if pd.isna(nodata):
                                        out_of_bounds_mask = np.isnan(original_core)
                                    else:
                                        out_of_bounds_mask = (original_core == nodata)
                                    
                                    core_array[out_of_bounds_mask] = nodata
                                    
                                    dst.write(core_array.astype(kwargs['dtype']), 1, window=window)

                if os.path.exists(gdal_dst_str):
                    os.remove(gdal_dst_str)
                shutil.move(smoothed_tmp, gdal_dst_str)
            
            if ds is not None:
                del ds
            
            if self.is_aws:
                self.fs.put(gdal_dst_str, dst_str)
                logger.info(f" - [✓ SUCCESS] Wrote to S3 successfully: {os.path.basename(dst_str)}")
            else:
                logger.info(f" - [✓ SUCCESS] Wrote locally successfully: {os.path.basename(dst_str)}")

        except Exception as e:
            logger.exception(f" - [✗ ERROR] GDAL Warp/Upload failed for {os.path.basename(src_str)}!")
            raise e
        finally:
            try:
                if self.is_aws and os.path.exists(gdal_dst_str):
                    os.remove(gdal_dst_str)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp files: {e}")
            
            if hasattr(gdal, 'VSICurlClearCache'):
                gdal.VSICurlClearCache() 

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type) -> None:
        """Clip raster files by tile and save data in memory-managed batches."""
        logger.info(f"Clipping {data_type} rasters by tile...")
        
        sub_grid_path = self.subgrid_paths.get(data_type)
        if not sub_grid_path:
            logger.error(f"No subgrid path defined for {data_type}")
            return
        
        logger.info(f"Loading subgrids from: {sub_grid_path}")
        try:
             sub_grids = gpd.read_file(str(sub_grid_path))
             logger.info("Successfully loaded subgrids.")
        except Exception as e:
             logger.exception(f"Reading subgrids from {sub_grid_path} failed.")
             return
        
        logger.info(f"Number of tiles to process: {sub_grids.shape[0]}")
        logger.info(f"Raster directory: {raster_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        logger.info(f"Scanning directory for raster files... (This will only happen once)")
        raster_dir_upath = UPath(raster_dir)
        all_raster_files = [
            str(f) for f in raster_dir_upath.rglob("*") 
            if f.suffix.lower() in {'.tif', '.tiff'}
        ]
        logger.info(f"Found {len(all_raster_files)} raster files.")

        # ---------------------------------------------------------------------
        # PARQUET COUNTING & PRE-CALCULATIONS
        # ---------------------------------------------------------------------
        all_tasks = []
        write_counter = 0
        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id']
            output_folder = output_dir / tile_name
            expected_output_path = output_folder / f"{tile_name}_{data_type}_clipped_data.parquet"
            
            should_write = self.overwrite or not expected_output_path.exists()
            if should_write:
                write_counter += 1
                all_tasks.append({
                    'sub_grid': sub_grid,
                    'output_folder': output_folder,
                    'tile_name': tile_name,
                    'should_write': True,
                    'write_index': write_counter
                })
            else:
                all_tasks.append({
                    'sub_grid': sub_grid,
                    'output_folder': output_folder,
                    'tile_name': tile_name,
                    'should_write': False,
                    'expected_output_path': expected_output_path
                })
        
        total_to_write = write_counter
        logger.info(f"--- Subtiling Wide Parquet Summary ({data_type}) ---")
        logger.info(f" -> Total subgrid tiles: {len(sub_grids)}")
        logger.info(f" -> Tiles needing Wide Parquet generation: {total_to_write}")
        logger.info(f" -> Existing tiles (skipped): {len(sub_grids) - total_to_write}")
        logger.info(f"--------------------------------------------------")

        # -------------------------------------------------------------
        # DYNAMIC DASK TASK STREAM (TILE CLIPPING)
        # -------------------------------------------------------------
        client = dask.distributed.client.default_client()
        max_concurrent = 25 
        logger.info(f"Building dynamic Dask task stream (max {max_concurrent} concurrent)...")
        
        sub_grid_iterator = iter(all_tasks)
        total_grids = len(all_tasks)
        seq = as_completed()
        results_list = []
        
        def submit_next_tile(task_item):
            tile_name = task_item['tile_name']
            if not task_item['should_write']:
                expected_output_path = task_item['expected_output_path']
                logger.info(f" [SKIP] Tile already clipped: {tile_name}. Queuing stats generation only.")
                stats_task = dask.delayed(self._generate_stats_from_existing)(str(expected_output_path), tile_name)
                return client.compute(stats_task)
            else:
                sub_grid = task_item['sub_grid']
                output_folder = task_item['output_folder']
                write_idx = task_item['write_index']
                
                gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, all_raster_files)
                combined_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, all_raster_files, gridded_task)
                return client.compute(
                    self.save_combined_data(
                        combined_task, 
                        output_folder, 
                        data_type, 
                        tile_id=tile_name,
                        current_index=write_idx,
                        total_count=total_to_write
                    )
                )

        # Initial queue fill
        for _ in range(min(max_concurrent, total_grids)):
            try:
                seq.add(submit_next_tile(next(sub_grid_iterator)))
            except StopIteration:
                break
                
        # Process stream
        for future in seq:
            results_list.append(future.result())
            try:
                seq.add(submit_next_tile(next(sub_grid_iterator)))
            except StopIteration:
                pass

        logger.info("Dask computation across dynamic task stream finished successfully.")

        logger.info(f"Combining {data_type} tile results and calculating statistics...")
        logger.info(f"Concatenating {len(results_list)} tile result dataframes...")
        final_results_df = pd.concat(results_list, ignore_index=True)
        logger.info(f"Final combined dataframe shape: {final_results_df.shape}")
        
        output_csv_path = output_dir.parent / f"year_pair_nan_counts_{data_type}.csv"
        
        logger.info(f"Generating statistics and saving to CSV format...")
        final_results_df.to_csv(str(output_csv_path), index=False, na_rep='NA')
        logger.info(f"[SUCCESS] Statistics successfully saved to: {output_csv_path}")
        logger.info(f"Finished clipping {data_type} rasters by tile.")

    def subtile_process_gridded(self, sub_grid, raster_files) -> pd.DataFrame:
        """Process gridded rasters for a single tile dynamically and avoid sequential merging."""
        original_tile = sub_grid['original_tile']
                
        filtered_files = [
            f for f in raster_files
            if original_tile in Path(f).name
        ]
        
        if not filtered_files:
            return pd.DataFrame()

        tile_extent = sub_grid.geometry.bounds
        
        data_arrays = {}
        common_window = None
        common_transform = None
        
        # Read all aligned band arrays in a single open/read pass
        for file in filtered_files:
            try:
                with rasterio.open(file) as src:
                    if common_window is None:
                        common_window = src.window(*tile_extent)
                        common_transform = src.window_transform(common_window)
                    
                    # FIX: Add boundless=True to pad dimensions when the window crosses the raster edges,
                    # ensuring that all array dimensions match exactly for the master_mask |= mask bitwise operator.
                    data = src.read(1, window=common_window, boundless=True, fill_value=src.nodata)
                    
                    col_name = pathlib.Path(file).stem
                    data_arrays[col_name] = (data, src.nodata)
            except Exception as e:
                logger.warning(f"Error reading gridded file {file}: {e}")
                
        if not data_arrays:
            return pd.DataFrame()
            
        # Create a unified master mask across all bands (implements Outer Join fast)
        master_mask = None
        for col_name, (data, nodata) in data_arrays.items():
            if nodata is not None and not np.isnan(nodata):
                mask = data != nodata
            else:
                mask = ~np.isnan(data)
            if master_mask is None:
                master_mask = mask.copy()
            else:
                master_mask |= mask
                
        if master_mask is None or not master_mask.any():
            return pd.DataFrame()
            
        # Compute spatial coordinates only ONCE for the whole tile
        rows, cols = np.where(master_mask)
        xs, ys = rasterio.transform.xy(common_transform, rows, cols, offset='center')
        
        # Instantiate the DataFrame in one go to bypass intermediate allocation
        df_dict = {
            'X': np.round(xs, 3),
            'Y': np.round(ys, 3)
        }
        
        for col_name, (data, nodata) in data_arrays.items():
            vals = data[master_mask].astype(np.float32)
            if nodata is not None and not np.isnan(nodata):
                vals[vals == nodata] = np.nan
            df_dict[col_name] = vals
            
        combined_data = pd.DataFrame(df_dict)
        combined_data = combined_data.drop_duplicates(subset=['X', 'Y'])
        return combined_data

    def subtile_process_ungridded(self, sub_grid, raster_files, gridded_df) -> pd.DataFrame:
        """Process ungridded rasters by translating spatial locations directly to pixel indices instead of merging."""
        if gridded_df is None or gridded_df.empty:
            return pd.DataFrame()

        # Copy dataframe structure to insert matching ungridded bands directly
        combined_df = gridded_df.copy()
        
        xs = combined_df['X'].values
        ys = combined_df['Y'].values
        tile_extent = sub_grid.geometry.bounds

        for pattern in self.static_patterns:
            current_files = [f for f in raster_files if pattern in Path(f).name]

            for file in current_files:
                col_name = pathlib.Path(file).stem
                try:
                    with rasterio.open(file) as src:
                        window = src.window(*tile_extent)
                        
                        # Guard against non-intersecting / empty coordinate windows
                        if window.width <= 0 or window.height <= 0:
                            combined_df[col_name] = np.full(len(xs), np.nan, dtype=np.float32)
                            continue

                        win_data = src.read(1, window=window)
                        win_transform = src.window_transform(window)
                        
                        # Translate spatial coordinates directly into row-column positions on this band
                        win_rows, win_cols = rasterio.transform.rowcol(win_transform, xs, ys)
                        win_rows = np.array(win_rows)
                        win_cols = np.array(win_cols)
                        
                        # Check inside-image boundary conditions
                        win_valid = (win_rows >= 0) & (win_rows < win_data.shape[0]) & \
                                    (win_cols >= 0) & (win_cols < win_data.shape[1])
                        
                        vals = np.full(len(xs), np.nan, dtype=np.float32)
                        
                        if win_valid.any():
                            extracted_vals = win_data[win_rows[win_valid], win_cols[win_valid]].astype(np.float32)
                            if src.nodata is not None:
                                nodata_val = src.nodata
                                if not np.isnan(nodata_val):
                                    extracted_vals[extracted_vals == nodata_val] = np.nan
                            vals[win_valid] = extracted_vals
                            
                        combined_df[col_name] = vals
                except Exception as e:
                    logger.warning(f"Failed to sample ungridded raster {file}: {e}")
                    combined_df[col_name] = np.full(len(xs), np.nan, dtype=np.float32)

        return combined_df

    def _extract_raster_to_df(self, raster_path, tile_extent) -> pd.DataFrame:
        """Helper to read a window of a raster and convert to DataFrame."""
        try:
            with rasterio.open(raster_path) as src:
                window = src.window(*tile_extent)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                mask = data != src.nodata
                
                if not mask.any():
                    return pd.DataFrame()

                rows, cols = np.where(mask)
                values = data[mask]
                xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
                
                return pd.DataFrame({
                    'X': xs, 'Y': ys, 'Value': values, 'Raster': pathlib.Path(raster_path).stem
                })
        except Exception as e:
            logger.exception(f"Reading raster window from {raster_path} failed.")
            return pd.DataFrame()

    @dask.delayed
    def save_combined_data(self, combined_df, output_folder, data_type, tile_id, current_index=None, total_count=None) -> pd.DataFrame:
        """Combine dataframes and save to parquet."""
        try:
            if combined_df is None or combined_df.empty:
                return pd.DataFrame()

            output_folder_path = UPath(output_folder)
            
            if not self.is_aws: 
                output_folder_path.mkdir(parents=True, exist_ok=True)
                
            output_path = output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet"
            save_path = str(output_path)

            combined_df.to_parquet(save_path, engine="pyarrow", index=False)
            
            progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
            logger.info(f"{progress_str} [SUCCESS] Saved combined tile data to: {save_path}")
            return self.create_nan_stats_csv(combined_df, tile_id)
        finally:
            self._trim_memory()

    def create_nan_stats_csv(self, df, tile_id) -> pd.DataFrame:
        """Calculates NaN stats for a tile."""
        if df.empty:
            return pd.DataFrame()
        new_row = {'tile_id': tile_id}
        change_cols = [c for c in df.columns if c.startswith('b.change.')]
        for col in change_cols:
            year_pair = col.replace('b.change.', '')
            new_row[f"{year_pair}_nan_percent"] = round(df[col].isna().mean() * 100, 2)
        return pd.DataFrame([new_row])

    def _generate_stats_from_existing(self, filepath: str, tile_id: str) -> pd.DataFrame:
        """Reads an existing parquet file to generate nan stats without reprocessing."""
        try:
            df = pd.read_parquet(filepath)
            return self.create_nan_stats_csv(df, tile_id)
        except Exception as e:
            logger.exception(f"Failed to read existing tile {filepath} for stats.")
            return pd.DataFrame()

    def batch_long_format_transformation(self, base_dir, mode: Literal["training", "prediction"]):
        """Orchestrator for transforming wide tiles to long year-pair format."""
        logger.info(f"Starting Long Format Transformation (Batch: {mode})...")

        file_suffix = f"_{mode}_clipped_data.parquet"

        base_dir_upath = UPath(base_dir)
        files_to_process = list(base_dir_upath.rglob(f"*{file_suffix}"))

        if not files_to_process:
            logger.warning(f"No files found for {mode} transformation in {base_dir}")
            return

        logger.info(f"Outputting transformed {mode} long-format tiles to: {base_dir}")
        logger.info(f"Queueing {len(files_to_process)} tiles...")

        # -------------------------------------------------------------
        # DYNAMIC DASK TASK STREAM (LONG FORMAT TRANSFORMATION)
        # Reduced max_concurrent to prevent memory pileups in scheduler and Dask workers
        # -------------------------------------------------------------
        client = dask.distributed.client.default_client()
        max_concurrent = 100 
        total_files = len(files_to_process)
        tasks_iterator = iter(enumerate(files_to_process))
        seq = as_completed()
        results = []
        
        def submit_long_format_task(item):
            i, fp = item
            return client.submit(
                self._transform_tile_task, 
                str(fp), 
                mode, 
                current_index=i + 1, 
                total_count=total_files
            )

        # Initial queue fill
        for _ in range(min(max_concurrent, total_files)):
            try:
                seq.add(submit_long_format_task(next(tasks_iterator)))
            except StopIteration:
                break

        # Process stream
        for future in seq:
            results.append(future.result())
            future.release() # Release future to prevent metadata accumulation in scheduler
            try:
                seq.add(submit_long_format_task(next(tasks_iterator)))
            except StopIteration:
                pass

        success = sum(1 for r in results if r.startswith("Success"))
        failed = len(results) - success
        
        logger.info(f"[SUCCESS] Transformation Complete. Success: {success}, Failed: {failed}")
        if failed > 0:
            logger.error("Transformation Errors:\n" + "\n".join([r for r in results if not r.startswith("Success")]))

    def _transform_tile_task(self, f_path: str, mode: Literal["training", "prediction"], current_index=None, total_count=None) -> str:
        """Dask Worker: Reads file -> Calls specific processor -> Returns status."""
        gdf = None
        try:
            tile_name = os.path.basename(f_path).split("_")[0]
            output_dir = os.path.dirname(f_path)

            try:
                # Engine 'pyarrow' explicitly set to map parquet files far more memory-efficiently
                gdf = gpd.read_parquet(f_path, engine="pyarrow")
            except Exception:
                df = pd.read_parquet(f_path, engine="pyarrow")
                geometry_col = 'geometry' if 'geometry' in df.columns else None
                gdf = gpd.GeoDataFrame(df, geometry=geometry_col)

            if mode == "training":
                saved = self._process_and_save_training_tile(gdf, output_dir, tile_name, current_index, total_count)
            else:
                saved = self._process_and_save_prediction_tile(gdf, output_dir, tile_name, current_index, total_count)
            
            return f"Success: {tile_name} ({len(saved)} pairs)"

        except Exception as e:
            return f"Failed: {os.path.basename(f_path)} - {str(e)}"
        finally:
            if gdf is not None:
                del gdf
            self._trim_memory()

    def _process_and_save_training_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, current_index=None, total_count=None) -> List[str]:
        """Processes a training tile and writes outputs immediately to disk."""
        self._transform_flowdir_cols_inplace(gdf)
        col_meta = self._get_column_metadata(gdf.columns.tolist())
        
        saved_files = []
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""

        for y0, y1 in getattr(self, 'year_ranges', []): # Using getattr to avoid undeclared issues
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"

            out_name = f"{tile_name}_{pair_name}_long.parquet"
            out_path = str(UPath(output_dir) / out_name)

            if not self.overwrite and UPath(out_path).exists():
                logger.info(f"{progress_str} [SKIP] Long-format training tile already exists: {out_path}")
                saved_files.append(out_name)
                continue

            cols_t_meta = col_meta[col_meta["year"] == y0]
            cols_t1_meta = col_meta[col_meta["year"] == y1]

            # Use union instead of intersection so we keep columns even if missing in one year
            union_vars = set(cols_t_meta["var_base"]).union(cols_t1_meta["var_base"])
            
            if "bathy_filled" not in union_vars and "bathy" not in union_vars:
                continue

            t_map = dict(zip(cols_t_meta["var_base"], cols_t_meta["colname"]))
            t1_map = dict(zip(cols_t1_meta["var_base"], cols_t1_meta["colname"]))

            sorted_union = sorted(union_vars)
            cols_t_exist = [t_map[v] for v in sorted_union if v in t_map]
            cols_t1_exist = [t1_map[v] for v in sorted_union if v in t1_map]

            forcing_pattern = f"{y0_str}_{y1_str}$"
            forcing_cols = [c for c in gdf.columns if re.search(forcing_pattern, c)]
            
            delta_bathy_col = f"delta_bathy_{pair_name}"
            forcing_cols = [c for c in forcing_cols if c != delta_bathy_col]

            static_vars = ["grain_size_layer", "prim_sed_layer", "survey_end_date"]
            static_cols = [c for c in static_vars if c in gdf.columns]
            
            id_cols = [c for c in ["X", "Y", "FID", "tile_id", "geometry"] if c in gdf.columns]

            cols_to_grab = id_cols + cols_t_exist + cols_t1_exist + forcing_cols + static_cols
            if delta_bathy_col in gdf.columns:
                cols_to_grab.append(delta_bathy_col)

            # Ensure we only grab columns that actually exist in the dataframe to prevent KeyError
            cols_to_grab = [c for c in cols_to_grab if c in gdf.columns]

            if not cols_to_grab:
                continue

            pair_gdf = gdf[cols_to_grab].drop_duplicates()

            rename_map = {}
            for v in sorted_union:
                if v in t_map and t_map[v] in pair_gdf.columns:
                    rename_map[t_map[v]] = f"{v}_t"
                if v in t1_map and t1_map[v] in pair_gdf.columns:
                    rename_map[t1_map[v]] = f"{v}_t1"
            
            pair_gdf.rename(columns=rename_map, inplace=True)
            
            if delta_bathy_col in pair_gdf.columns:
                pair_gdf.rename(columns={delta_bathy_col: "delta_bathy"}, inplace=True)

            pair_gdf["year_t"] = y0
            pair_gdf["year_t1"] = y1

            # Strip the _filled suffix before executing the delta logic
            filled_cols = [c for c in pair_gdf.columns if "_filled" in c]
            if filled_cols:
                pair_gdf.rename(columns={c: c.replace("_filled", "") for c in filled_cols}, inplace=True)

            # Calculate delta if bathy exists for both years
            if "delta_bathy" not in pair_gdf.columns and "bathy_t" in pair_gdf.columns and "bathy_t1" in pair_gdf.columns:
                pair_gdf["delta_bathy"] = pair_gdf["bathy_t1"] - pair_gdf["bathy_t"]

            # Pad missing critical columns with NaNs to ensure consistent schema across tiles
            for critical_col in ["bathy_t", "bathy_t1", "delta_bathy"]:
                if critical_col not in pair_gdf.columns:
                    pair_gdf[critical_col] = np.nan

            target_col = "bathy_t1"
            predictor_cols_t = [c for c in pair_gdf.columns if c.endswith("_t")]
            predictor_cols_delta = [c for c in pair_gdf.columns if c.startswith("delta_")]
            
            final_cols = id_cols + ["year_t", "year_t1", target_col] + predictor_cols_t + predictor_cols_delta + forcing_cols + static_cols
            final_cols = [c for c in final_cols if c in pair_gdf.columns]
            
            pair_gdf[final_cols].to_parquet(out_path, index=None, engine="pyarrow")
            logger.info(f"{progress_str} [SUCCESS] Saved training long-format tile to: {out_path}")
            saved_files.append(out_name)
            
            # Critical Cleanup: delete temporary dataframe view immediately before iterating the next year range.
            del pair_gdf
            gc.collect()

        return saved_files

    def _process_and_save_prediction_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, current_index=None, total_count=None) -> List[str]:
        """Processes a prediction tile and writes outputs immediately."""
        self._transform_flowdir_cols_inplace(gdf)
        
        bt_cols = [c for c in gdf.columns if self.re_bt_prefix.match(c)]
        new_t_names = [self.re_bt_prefix.sub("", c) + "_t" for c in bt_cols]
        
        gdf.rename(columns=dict(zip(bt_cols, new_t_names)), inplace=True)
        
        saved_files = []
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""

        for y0, y1 in getattr(self, 'year_ranges', []): # Using getattr to avoid undeclared issues
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"
            
            out_name = f"{tile_name}_{pair_name}_prediction_long.parquet"
            out_path = str(UPath(output_dir) / out_name)

            if not self.overwrite and UPath(out_path).exists():
                logger.info(f"{progress_str} [SKIP] Long-format prediction tile already exists: {out_path}")
                saved_files.append(out_name)
                continue
                
            forcing_pattern = f"{y0_str}_{y1_str}$"
            forcing_cols = [c for c in gdf.columns if re.search(forcing_pattern, c)]

            static_vars = ["grain_size_layer", "prim_sed_layer", "survey_end_date"]
            static_cols = [c for c in static_vars if c in gdf.columns]
            id_cols = [c for c in ["X", "Y", "FID", "tile_id", "geometry"] if c in gdf.columns]

            final_cols = id_cols + new_t_names + forcing_cols + static_cols
            final_cols = [c for c in final_cols if c in gdf.columns]

            pair_gdf = gdf[final_cols].drop_duplicates()
            
            # Strip _filled from names exactly like the training tile for consistency
            filled_cols = [c for c in pair_gdf.columns if "_filled" in c]
            if filled_cols:
                pair_gdf.rename(columns={c: c.replace("_filled", "") for c in filled_cols}, inplace=True)
            
            pair_gdf.to_parquet(out_path, index=None, engine="pyarrow")
            logger.info(f"{progress_str} [SUCCESS] Saved prediction long-format tile to: {out_path}")
            saved_files.append(out_name)
            
            # Critical Cleanup: explicit memory free immediately.
            del pair_gdf
            gc.collect()

        return saved_files

    def _transform_flowdir_cols_inplace(self, df: pd.DataFrame) -> None:
        """Modifies DataFrame in-place to replace flow direction angles."""
        flow_cols = [c for c in df.columns if self.re_flowdir.search(c)]
        if not flow_cols:
            return

        # Explicitly enforce float32 to prevent automatic float64 casting from eating extra memory
        radians = np.deg2rad(df[flow_cols].astype(np.float32))
        for col in flow_cols:
            df[f"{col}_sin"] = np.sin(radians[col]).astype(np.float32)
            df[f"{col}_cos"] = np.cos(radians[col]).astype(np.float32)

        df.drop(columns=flow_cols, inplace=True)
        del radians

    def _get_column_metadata(self, columns: List[str]) -> pd.DataFrame:
        """Efficiently parses column names to extract variables and years."""
        potential_cols = [
            c for c in columns 
            if "_" in c and c[-4:].isdigit() and not (c[-9] == "_" and c[-4:].isdigit() and c[-9:-5].isdigit())
        ]
        
        if not potential_cols:
            return pd.DataFrame(columns=["colname", "year", "var_base"])

        meta = pd.DataFrame({"colname": potential_cols})
        meta["year"] = meta["colname"].str.slice(-4).astype(int)
        meta["var_base"] = meta["colname"].str.replace(self.re_year_suffix, "", regex=True)
        return meta
    
    def raster_to_spatial_df(self, raster_path, process_type) -> gpd.GeoDataFrame:
        """ Convert a raster file to a GeoDataFrame by extracting shapes and their geometries in memory-safe chunks."""   

        logger.info(f"Creating {process_type} mask GeoDataFrame from: {raster_path}")

        open_path = str(raster_path)
        
        if self.is_aws and open_path.startswith("s3://"):
            open_path = open_path.replace("s3://", "/vsis3/")

        geometries = []
        
        with rasterio.open(open_path) as src:
            block_size = 4096
            
            for y in range(0, src.height, block_size):
                for x in range(0, src.width, block_size):
                    window = rasterio.windows.Window(
                        x, y, 
                        min(block_size, src.width - x), 
                        min(block_size, src.height - y)
                    )
                    
                    mask_chunk = src.read(1, window=window, out_dtype='uint8')

                    if process_type == 'prediction':
                        valid_mask = mask_chunk == 1
                    elif process_type == 'training':
                        if self.pilot_mode:
                            valid_mask = mask_chunk == 1
                        else:
                            valid_mask = mask_chunk == 2

                    if not valid_mask.any():
                        continue

                    win_transform = src.window_transform(window)
                    shapes_gen = shapes(mask_chunk, mask=valid_mask, transform=win_transform)  

                    chunk_geoms = []
                    for geom, _ in shapes_gen:
                        chunk_geoms.append(shape(geom))
                        
                    if chunk_geoms:
                        merged_chunk = unary_union(chunk_geoms)
                        geometries.append(merged_chunk)
                        
            crs = src.crs

        logger.info(f" -> Extracted {len(geometries)} unified geometries. Building GeoDataFrame...")
        
        gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=crs)
        gdf = gdf.to_crs(self.target_crs)   

        if process_type == 'prediction':
            mask_path = self.mask_prediction_pq
        else:
            mask_path = self.mask_training_pq
        
        if not self.is_aws:
            mask_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving {process_type} mask GeoDataFrame to: {mask_path}")   

        gdf.to_parquet(str(mask_path))

        return gdf
        
    def create_subgrids(self, mask_gdf, output_path, process_type) -> None:
        """ Create subgrids layer by intersecting grid tiles with the mask geometries"""        
        
        mask_gdf_path = str(mask_gdf)
        logger.info(f"Preparing {process_type} sub-grids...")
        logger.info(f" -> Reading mask GeoDataFrame from: {mask_gdf_path}")

        mask_gdf_df = gpd.read_parquet(mask_gdf_path, filesystem=self.fs if self.is_aws else None)
        
        # We can still union the small number of subgrids here because this is tiny compared to raster data
        combined_geometry = mask_gdf_df.union_all()
        mask_gdf_df = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf_df.crs)

        grid_gpkg_str = str(self.grid_gpkg)
        if self.is_aws and grid_gpkg_str.startswith("s3://"):
            grid_gpkg_str = grid_gpkg_str.replace("s3://", "/vsis3/")

        sub_grids = gpd.read_file(grid_gpkg_str, layer='prediction_subgrid').to_crs(mask_gdf_df.crs)

        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf_df, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
        
        if self.is_aws:
            with tempfile.NamedTemporaryFile(dir=self.local_tmp_dir, suffix=".gpkg", delete=False) as tmp:
                local_tmp_path = tmp.name
                
            logger.info(f" -> Writing GPKG locally to {local_tmp_path} before uploading...")
            intersecting_sub_grids.to_file(local_tmp_path, driver="GPKG") 
            
            logger.info(f" -> Uploading subgrids to S3: {output_path}")
            self.fs.put(local_tmp_path, str(output_path))
            os.remove(local_tmp_path)
        else:
            output_upath = UPath(output_path)
            output_upath.parent.mkdir(parents=True, exist_ok=True)
            intersecting_sub_grids.to_file(str(output_upath), driver="GPKG") 

        logger.info(f"[SUCCESS] Successfully saved {process_type} subgrids to: {output_path}")
        return