"""Class for data acquisition and preprocessing of model data"""
import os
import re
import pathlib
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
    "GDAL_INGESTED_BYTES_AT_OPEN": "32768"         # Caches metadata header bytes to minimize initial range requests
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

        # Extended static patterns to ensure ungridded data like 'grain', 'survey', and 'sed' are captured 
        self.static_patterns = ['sed', 'tsm', 'hurr', 'grain', 'survey']
        self.re_bt_prefix = re.compile(r"^bt\.")

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
        
        self.uncombined_lidar_dir = UPath(f"{prefix}{get_config_item('MODEL', 'TILED_LIDAR_PROC', pilot_mode=self.pilot_mode)}")
        
        # Dynamically retrieve the filled terrain directory from config to ensure accurate exclusion
        try:
            filled_dir_path = get_config_item('TERRAIN', 'FILLED_DIR', pilot_mode=self.pilot_mode)
            self.filled_folder_name = UPath(filled_dir_path).name.lower()
        except Exception:
            logger.warning("Could not load TERRAIN/FILLED_DIR from config. Falling back to default 'filled_tifs'.")
            self.filled_folder_name = "filled_tifs"

        self.subgrid_paths = {
            'training': UPath(f"{prefix}{get_config_item('MODEL', 'TRAINING_SUB_GRIDS', pilot_mode=self.pilot_mode)}"),
            'prediction': UPath(f"{prefix}{get_config_item('MODEL', 'PREDICTION_SUB_GRIDS', pilot_mode=self.pilot_mode)}")
        }

        self.preprocessed_subdirs = {
            'bluetopo': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'BLUETOPO', pilot_mode=self.pilot_mode)}"),
            'hurricane': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'HURRICANE', pilot_mode=self.pilot_mode)}"),
            # Read from the original input directory
            'lidar': UPath(f"{prefix}{get_config_item('MODEL', 'TILED_LIDAR_DIR', pilot_mode=self.pilot_mode)}"),
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
        
        # --- FIX: Force GDAL to use our larger directory for cache/spill instead of /tmp ---
        os.environ["CPL_TMPDIR"] = str(self.local_tmp_dir)
        GDAL_ENV_VARS["CPL_TMPDIR"] = str(self.local_tmp_dir)

    def process(self) -> None:
        """Main function to process model data."""   
        logger.info(f"Starting ModelDataPreProcessor. Logs are being saved to: {LOG_FILE_PATH}")
        self.create_file_paths()
        self._clean_local_tmp()

        # NOTE: Phase 1 is for Raster operations which are block-based and use less working memory
        logger.info("Initializing Phase 1 Cluster: Heavy Raster Processing (8 workers, standard memory)")
        cluster = LocalCluster(
            n_workers=4,            
            threads_per_worker=1,   
            memory_limit='7GB',
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

                # --- FIX: Enforce uniform Target CRS before bounding box evaluation ---
                if hasattr(self, 'target_crs'):
                    logger.info(f"Aligning Mask GDFs to target CRS: {self.target_crs}")
                    if mask_pred_gdf.crs is None: mask_pred_gdf = mask_pred_gdf.set_crs(self.target_crs)
                    elif mask_pred_gdf.crs != self.target_crs: mask_pred_gdf = mask_pred_gdf.to_crs(self.target_crs)
                    
                    if mask_train_gdf.crs is None: mask_train_gdf = mask_train_gdf.set_crs(self.target_crs)
                    elif mask_train_gdf.crs != self.target_crs: mask_train_gdf = mask_train_gdf.to_crs(self.target_crs)

                # --- FIX: Prevent TopologyException side location conflicts ---
                logger.info("Validating geometries to prevent GDAL TopologyExceptions...")
                mask_pred_gdf['geometry'] = mask_pred_gdf.geometry.make_valid().buffer(0)
                mask_train_gdf['geometry'] = mask_train_gdf.geometry.make_valid().buffer(0)
                
                # Clean up any empty geometries resulting from buffer(0)
                mask_pred_gdf = mask_pred_gdf[~mask_pred_gdf.is_empty & mask_pred_gdf.geometry.notnull()]
                mask_train_gdf = mask_train_gdf[~mask_train_gdf.is_empty & mask_train_gdf.geometry.notnull()]
                
                if mask_pred_gdf.empty or mask_train_gdf.empty:
                    raise ValueError("Mask GeoDataFrames are empty after validation. Check your input geometries.")
                # --------------------------------------------------------------

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
                n_workers=4,            
                threads_per_worker=1,  
                memory_limit='7GB',
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

                self.clip_rasters_by_tile(
                    raster_dir=self.training_out_dir, 
                    output_dir=self.training_tiles_dir, 
                    data_type="training"
                )
                
                # self.batch_format_transformation(base_dir=self.prediction_tiles_dir, mode="prediction")
                self.batch_format_transformation(base_dir=self.training_tiles_dir, mode="training")

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

        # Updated to explicitly exclude the filled lidar directory using config
        excluded_folders = {self.filled_folder_name, 'filled_tifs', 'filled_lidar'}
        valid_source_files = []
        removed_folders = 0
        removed_masks = 0

        for f in potential_files:
            if "sand_mud_mask" in f.name:
                removed_masks += 1
                continue
                
            if any(folder in f.parts for folder in excluded_folders):
                removed_folders += 1
                continue
                
            valid_source_files.append(f)

        logger.info(f"--- File Filtering Summary ---")
        logger.info(f" -> Total potential files: {len(potential_files)}")
        logger.info(f" -> Removed (excluded folders): {removed_folders}")
        logger.info(f" -> Removed (sand_mud_mask): {removed_masks}")
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
            name_lower = f.name.lower()
            parts_lower = [p.lower() for p in f.parts]

            # Ensure filled lidar directory is excluded for training 
            if self.filled_folder_name in parts_lower or 'filled_lidar' in parts_lower or 'filled_tifs' in parts_lower:
                continue
            
            # Ensure we use combined lidar rather than filled lidar for mosaics
            if 'mosaic' in name_lower and 'combined' not in name_lower and 'combined_lidar' not in parts_lower:
                continue
            
            # Exclude BlueTopo files strictly from being processed into training datasets
            # EXCEPTION: Keep it if it is the survey_end_date layer
            if ('bluetopo' in name_lower or name_lower.startswith('bt.')) and 'survey_end_date' not in name_lower:
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
                    # Use robust bounding box construction strictly handling minimums and maximums
                    # preventing empty polygons caused by arrays with negative affine orientations.
                    if raster_crs != target_crs_obj:
                        left, bottom, right, top = transform_bounds(raster_crs, target_crs_obj, *raster_bounds)
                        bounds_geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
                    else:
                        bounds_geom = box(min(raster_bounds[0], raster_bounds[2]), min(raster_bounds[1], raster_bounds[3]), max(raster_bounds[0], raster_bounds[2]), max(raster_bounds[1], raster_bounds[3]))
                except Exception as e:
                    # FIX: Do NOT fallback to native bounds if transform fails; that mathematically breaks intersections!
                    logger.warning(f"Failed to transform bounds for {raster_name}: {e}. Bypassing intersection check for safety.")
                    bounds_geom = None
            else:
                bounds_geom = box(min(raster_bounds[0], raster_bounds[2]), min(raster_bounds[1], raster_bounds[3]), max(raster_bounds[0], raster_bounds[2]), max(raster_bounds[1], raster_bounds[3]))

            if bounds_geom is not None:
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

            with tempfile.TemporaryDirectory(dir=self.local_tmp_dir) as task_tmp_dir:
                try:
                    self._warp_to_cutline(
                        raster_path, 
                        output_path, 
                        cutline_path, 
                        task_tmp_dir=task_tmp_dir,
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

            with tempfile.TemporaryDirectory(dir=self.local_tmp_dir) as task_tmp_dir:
                try:
                    with rasterio.open(open_path) as src_pred:
                        src_nodata = src_pred.nodata if src_pred.nodata is not None else np.nan
                        
                        # Check bounding box intersections quickly using safe bounds
                        rb = src_pred.bounds
                        raster_bounds_geom = box(min(rb[0], rb[2]), min(rb[1], rb[3]), max(rb[0], rb[2]), max(rb[1], rb[3]))
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

                        # Setup temporary local path inside quarantined task directory
                        tmp_dst_path = str(output_path)
                        if self.is_aws:
                            tmp_dst_path = str(Path(task_tmp_dir) / "train_mask_tmp.tif")

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

                    logger.info(f" - [✓ SUCCESS]{progress_str} Processed training raster via array masking: {raster_name}")
                    
                except Exception as e:
                    logger.exception(f"Unexpected failure during array masking for {raster_name}.")
        finally:
            self._trim_memory()

    def _warp_to_cutline(self, src_path, dst_path, cutline_path, task_tmp_dir=None, **kwargs):
        """Helper to handle GDAL Warp boilerplate."""
        src_str = str(src_path)
        dst_str = str(dst_path)

        if self.is_aws and src_str.startswith('s3://'):
            src_str = src_str.replace('s3://', '/vsis3/')

        if self.is_aws:
            gdal_dst_str = str(Path(task_tmp_dir) / "warp_tmp.tif")
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

            # FIX: Force close and flush the dataset to disk immediately to prevent locking/bloating
            ds = None

            if apply_tsm_smoothing:
                # Temporarily open just to get the pixel size, then close
                tmp_ds = gdal.Open(gdal_dst_str)
                pixel_size = tmp_ds.GetGeoTransform()[1]
                tmp_ds = None 
                
                mem = psutil.virtual_memory()
                logger.info(f" [SMOOTHING INIT] {os.path.basename(src_str)} | Sys RAM: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")

                radius_pixels = int(2000 / abs(pixel_size))
                size = radius_pixels * 2 + 1
                
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
            
            if self.is_aws:
                self.fs.put(gdal_dst_str, dst_str)
                logger.info(f" - [✓ SUCCESS] Wrote to S3 successfully: {os.path.basename(dst_str)}")
            else:
                logger.info(f" - [✓ SUCCESS] Wrote locally successfully: {os.path.basename(dst_str)}")

        except Exception as e:
            logger.exception(f" - [✗ ERROR] GDAL Warp/Upload failed for {os.path.basename(src_str)}!")
            raise e
        finally:
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
        
        all_raster_files = []
        for f in raster_dir_upath.rglob("*"):
            if f.suffix.lower() in {'.tif', '.tiff'}:
                name_lower = f.name.lower()
                parts_lower = [p.lower() for p in f.parts]
                
                # Double-check exclusion of filled lidar directory in clipping step
                if self.filled_folder_name in parts_lower or "filled_lidar" in parts_lower or "filled_tifs" in parts_lower:
                    continue
                
                # Exclude 'unc', specific hurricane, and tsm_cumulative files globally for BOTH modes
                if "unc" in name_lower:
                    continue
                    
                if "tsm_cumulative" in name_lower or \
                   "hurr_count_mean" in name_lower or \
                   "hurr_count_cumulative" in name_lower or \
                   "hurr_strength_cumulative" in name_lower or \
                   re.search(r"hurr_count_\d{4}_\d{4}", name_lower) or \
                   re.search(r"hurr_strength_\d{4}_\d{4}", name_lower):
                    continue
                    
                if data_type == 'training':
                    # Exclude BlueTopo files strictly from being processed into training datasets
                    # EXCEPTION: Keep it if it is the survey_end_date layer
                    if ("bluetopo" in name_lower or name_lower.startswith("bt.")) and "survey_end_date" not in name_lower:
                        continue
                elif data_type == 'prediction':
                    # Prediction parquet uses the bluetopo files and not the filled lidar bathy
                    if "bathy" in name_lower and "bluetopo" not in name_lower and not name_lower.startswith("bt."):
                        continue
                        
                all_raster_files.append(str(f))
        
        logger.info(f"Filtered out TIFF files containing 'unc' and specific hurricane/tsm derivatives from the {data_type} parquet files.")
        if data_type == 'training':
            logger.info(" -> Specifically excluded BlueTopo files for training (except survey_end_date).")
        elif data_type == 'prediction':
            logger.info(" -> Specifically excluded standard Lidar (bathy) files for prediction (using BlueTopo instead).")

        logger.info(f"Found {len(all_raster_files)} raster files.")
        
        # --- PRE-PARTITION FILES TO PREVENT S3 API CONNECTION EXHAUSTION ---
        # Separates files into strictly tiled (gridded) vs global mosaics (ungridded)
        # This drastically prevents Dask workers from checking thousands of irrelevant tiled S3 files
        valid_tids = [str(tid) for tid in sub_grids['original_tile'].unique() if pd.notna(tid) and str(tid).strip()]
        
        gridded_files = []
        ungridded_files = []
        for f in all_raster_files:
            fname = Path(f).name
            if any(tid in fname for tid in valid_tids):
                gridded_files.append(f)
            else:
                ungridded_files.append(f)
                
        logger.info(f" -> Pre-partitioned into {len(gridded_files)} gridded files (tiled) and {len(ungridded_files)} ungridded files (global).")

        # ---------------------------------------------------------------------
        # PARQUET COUNTING & PRE-CALCULATIONS
        # ---------------------------------------------------------------------
        all_tasks = []
        write_counter = 0
        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id']
            output_folder = output_dir / tile_name
            # Look for the final formatted file to determine if we need to process this tile
            expected_final_path = output_folder / f"{tile_name}_{data_type}_formatted.parquet"
            
            should_write = self.overwrite or not expected_final_path.exists()
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
                    'expected_output_path': expected_final_path
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
                logger.info(f" [SKIP] Tile already processed: {tile_name}. Queuing stats generation only.")
                stats_task = dask.delayed(self._generate_stats_from_existing)(str(expected_output_path), tile_name)
                return client.compute(stats_task)
            else:
                sub_grid = task_item['sub_grid']
                output_folder = task_item['output_folder']
                write_idx = task_item['write_index']
                
                # We specifically pass the pre-partitioned list
                gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, gridded_files)
                combined_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, ungridded_files, gridded_task)
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

    def _standardize_col_name(self, col_name: str, original_tile: str = "") -> str:
        """Cleans raster filenames into consistent column names, standardizing years and prefixes."""
        clean_name = col_name
        
        # Explicit override for survey_end_date to ensure it gets exactly this column name
        if "survey_end_date" in clean_name.lower():
            return "survey_end_date"
        
        # 1. Clean bluetopo prefix and tags
        is_bluetopo = "bluetopo" in clean_name.lower() or clean_name.startswith("bt.")
        if is_bluetopo:
            clean_name = re.sub(r"(?i)^bluetopo_?", "", clean_name)
            if clean_name.startswith("bt."):
                clean_name = clean_name[3:]
                
        # 2. Strip 'combined_' legacy prefixes
        clean_name = re.sub(r"(?i)^combined\d*_", "", clean_name)
        
        # 3. Strip original tile
        if original_tile and original_tile in clean_name:
            clean_name = clean_name.replace(f"_{original_tile}", "").replace(f"{original_tile}_", "").replace(original_tile, "")
            
        # 4. Truncate 8-digit YYYYMMDD dates to 4-digit YYYY
        clean_name = re.sub(r"(?<!\d)((?:19|20)\d{2})\d{4}(?!\d)", r"\1", clean_name)
        clean_name = clean_name.strip("_")
        
        # 5. Extract and shift year / year-pairs dynamically
        m_pair = re.search(r"(\d{4}_\d{4})", clean_name)
        if m_pair:
            year_pair = m_pair.group(1)
            base = clean_name.replace(year_pair, "").strip("_")
            base = re.sub(r"__+", "_", base)
            if base:
                final_name = f"{base}_{year_pair}"
            else:
                final_name = year_pair
        else:
            m_single = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", clean_name)
            if m_single:
                year = m_single.group(1)
                base = clean_name.replace(year, "").strip("_")
                base = re.sub(r"__+", "_", base)
                
                base_lower = base.lower()
                # Enforce '_filled' for root bathy models
                if base_lower == "bathy" or base_lower == "bathy_filled":
                    final_name = f"bathy_{year}_filled"
                # Strip root bathy prefixes from dependent derivatives to keep them succinct 
                elif base_lower.startswith("bathy_"):
                    base = base[6:].strip("_")
                    final_name = f"{base}_{year}"
                elif base:
                    final_name = f"{base}_{year}"
                else:
                    final_name = year
            else:
                final_name = clean_name
                
        if is_bluetopo:
            return f"bt.{final_name}"
            
        return final_name

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
            open_path = str(file)
            if self.is_aws and open_path.startswith("s3://"):
                open_path = open_path.replace("s3://", "/vsis3/")
                
            try:
                with rasterio.open(open_path) as src:
                    if common_window is None:
                        common_window = src.window(*tile_extent)
                        common_transform = src.window_transform(common_window)
                    
                    # Add boundless=True to pad dimensions when the window crosses the raster edges,
                    # ensuring that all array dimensions match exactly for the master_mask |= mask bitwise operator.
                    data = src.read(1, window=common_window, boundless=True, fill_value=src.nodata)
                    
                    col_name = pathlib.Path(file).stem
                    col_name = self._standardize_col_name(col_name, original_tile)
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
        original_tile = sub_grid.get('original_tile', '')

        for pattern in self.static_patterns:
            current_files = [f for f in raster_files if pattern in Path(f).name]

            for file in current_files:
                col_name = pathlib.Path(file).stem
                col_name = self._standardize_col_name(col_name, original_tile)
                
                open_path = str(file)
                if self.is_aws and open_path.startswith("s3://"):
                    open_path = open_path.replace("s3://", "/vsis3/")
                    
                try:
                    with rasterio.open(open_path) as src:
                        window = src.window(*tile_extent)
                        
                        # Guard against non-intersecting / empty coordinate windows
                        if window.width <= 0 or window.height <= 0:
                            # Only initialize to NaN if the column doesn't exist yet to prevent overwriting valid 
                            # data when looping through static mosaicked/tiled layers (like survey_end_date tiles)
                            if col_name not in combined_df:
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
                            
                        # Safely insert without destroying data from other tiles mapped to the same col_name
                        if col_name in combined_df:
                            existing = combined_df[col_name].values
                            existing_nan = np.isnan(existing)
                            existing[existing_nan] = vals[existing_nan]
                            combined_df[col_name] = existing
                        else:
                            combined_df[col_name] = vals

                except Exception as e:
                    logger.warning(f"Failed to sample ungridded raster {file}: {e}")
                    if col_name not in combined_df:
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

            # Dynamically calculate delta_bathy for the wide/tall format before saving
            if data_type in ["training", "prediction"]:
                bathy_years = set()
                # Find all base bathy columns natively
                for c in combined_df.columns:
                    if data_type == "training":
                        if c.startswith('bathy_'):
                            m = re.search(r'_(\d{4})(?:_filled)?$', c, re.IGNORECASE)
                            if m:
                                bathy_years.add(int(m.group(1)))
                    else:  # prediction
                        if c.startswith('bt.'):
                            m = re.search(r'\.(\d{4})$', c)
                            if m:
                                bathy_years.add(int(m.group(1)))
                            
                sorted_years = sorted(list(bathy_years))
                # Create dynamic sequential pairs: e.g., (2004, 2006), (2006, 2010), etc.
                dynamic_year_ranges = [(sorted_years[i], sorted_years[i+1]) for i in range(len(sorted_years)-1)]
                
                valid_pairs = []
                for y0, y1 in dynamic_year_ranges:
                    y0_str, y1_str = str(y0), str(y1)
                    
                    if data_type == "training":
                        pattern_0 = re.compile(rf"^bathy_{y0_str}_filled$", re.IGNORECASE)
                        pattern_1 = re.compile(rf"^bathy_{y1_str}_filled$", re.IGNORECASE)
                    else:
                        pattern_0 = re.compile(rf"^bt\.(?:bluetopo_)?{y0_str}$", re.IGNORECASE)
                        pattern_1 = re.compile(rf"^bt\.(?:bluetopo_)?{y1_str}$", re.IGNORECASE)
                    
                    c_0 = [c for c in combined_df.columns if pattern_0.match(c)]
                    c_1 = [c for c in combined_df.columns if pattern_1.match(c)]
                    
                    if data_type == "training":
                        f_0 = [c for c in c_0 if "filled" in c.lower()]
                        f_1 = [c for c in c_1 if "filled" in c.lower()]
                        b_y0 = f_0[0] if f_0 else (c_0[0] if c_0 else None)
                        b_y1 = f_1[0] if f_1 else (c_1[0] if c_1 else None)
                    else:
                        b_y0 = c_0[0] if c_0 else None
                        b_y1 = c_1[0] if c_1 else None
                    
                    if b_y0 and b_y1:
                        delta_name = f"delta_bathy_{y0_str}_{y1_str}"
                        combined_df[delta_name] = combined_df[b_y1] - combined_df[b_y0]
                        valid_pairs.append(f"{y0_str}_{y1_str}")
                
                # Drop year-pair variables that do not have a matching delta_bathy
                cols_to_drop = []
                for c in combined_df.columns:
                    m = re.search(r"(\d{4}_\d{4})$", c)
                    if m and not c.startswith("delta_bathy_"):
                        if m.group(1) not in valid_pairs:
                            cols_to_drop.append(c)
                if cols_to_drop:
                    combined_df.drop(columns=cols_to_drop, inplace=True)

            # Ensure tile_id is included!
            if 'tile_id' not in combined_df.columns:
                combined_df['tile_id'] = tile_id

            # Ensure FID is included!
            if 'FID' not in combined_df.columns:
                combined_df.insert(0, 'FID', np.arange(len(combined_df)))

            output_folder_path = UPath(output_folder)
            
            if not self.is_aws: 
                output_folder_path.mkdir(parents=True, exist_ok=True)
                
            output_path = output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet"
            save_path = str(output_path)

            combined_df.to_parquet(save_path, engine="pyarrow", index=False)
            
            progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
            
            # Print columns directly to terminal the moment Dask finishes saving!
            logger.info(f"{progress_str} [SUCCESS] Saved combined tile data to: {save_path}")
            
            # Print all parquet columns generated
            cols_str = ", ".join(combined_df.columns.tolist())
            logger.info(f"{progress_str}   -> CREATED PARQUET COLUMNS: {cols_str}")
            
            return self.create_nan_stats_csv(combined_df, tile_id)
        finally:
            self._trim_memory()

    def create_nan_stats_csv(self, df, tile_id) -> pd.DataFrame:
        """Calculates NaN stats for a tile."""
        if df.empty:
            return pd.DataFrame()
        new_row = {'tile_id': tile_id}
        
        # Now searches directly for the dynamically built delta columns!
        change_cols = [c for c in df.columns if c.startswith('delta_bathy_')]
        for col in change_cols:
            year_pair = col.replace('delta_bathy_', '')
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

    def batch_format_transformation(self, base_dir, mode: Literal["training", "prediction"]):
        """Orchestrator for finalizing formatting on wide tiles."""
        logger.info(f"Starting Wide & Batch Format Transformation (Mode: {mode})...")

        year_ranges_val = getattr(self, 'year_ranges', None)
        logger.info(f"-> Validating 'year_ranges' config: {year_ranges_val}")
        if not year_ranges_val:
            logger.error("!!! CRITICAL WARNING: 'self.year_ranges' is empty or not defined. No files will be processed !!!")

        file_suffix = f"_{mode}_clipped_data.parquet"

        base_dir_upath = UPath(base_dir)
        files_to_process = list(base_dir_upath.rglob(f"*{file_suffix}"))

        if not files_to_process:
            logger.warning(f"No files found for {mode} transformation in {base_dir}")
            return

        logger.info(f"Outputting transformed {mode} formatted tiles to: {base_dir}")
        logger.info(f"Queueing {len(files_to_process)} tiles...")

        # -------------------------------------------------------------
        # DYNAMIC DASK TASK STREAM (FORMAT TRANSFORMATION)
        # -------------------------------------------------------------
        client = dask.distributed.client.default_client()
        max_concurrent = 100 
        total_files = len(files_to_process)
        tasks_iterator = iter(enumerate(files_to_process))
        seq = as_completed()
        results = []
        
        def submit_format_task(item):
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
                seq.add(submit_format_task(next(tasks_iterator)))
            except StopIteration:
                break

        success_count = 0
        failed_msgs = []

        # Process stream
        for future in seq:
            res = future.result()
            
            # Print immediately as tasks complete instead of waiting for the end
            if res.startswith("Success"):
                success_count += 1
                logger.info(res) 
            else:
                failed_msgs.append(res)

            future.release() # Release future to prevent metadata accumulation in scheduler
            try:
                seq.add(submit_format_task(next(tasks_iterator)))
            except StopIteration:
                pass

        logger.info(f"--------------------------------------------------")
        logger.info(f"[TRANSFORMATION SUMMARY] Mode: {mode.upper()}")
        logger.info(f" -> Total Attempted Tasks: {len(results)}")
        logger.info(f" -> Successful Tasks: {success_count}")
        logger.info(f" -> Failed/Error Tasks: {len(failed_msgs)}")
        logger.info(f"--------------------------------------------------")
            
        if failed_msgs:
            logger.error("Transformation Errors:\n" + "\n".join(failed_msgs))


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
                saved, cols_str = self._process_and_save_training_tile(gdf, output_dir, tile_name, current_index, total_count)
            else:
                saved, cols_str = self._process_and_save_prediction_tile(gdf, output_dir, tile_name, current_index, total_count)
            
            # --- CLEAN UP INTERMEDIATE RAW WIDE FILE ---
            try:
                f_path_str = str(f_path)
                if self.is_aws and f_path_str.startswith("s3://"):
                    self.fs.rm(f_path_str)
                else:
                    if os.path.exists(f_path_str):
                        os.remove(f_path_str)
            except Exception as e:
                logger.warning(f"Could not delete intermediate file {f_path}: {e}")

            return f"Success: {tile_name} (Generated: {len(saved)} files)\n   -> {cols_str}"

        except Exception as e:
            return f"Failed: {os.path.basename(f_path)} - {str(e)}"
        finally:
            if gdf is not None:
                del gdf
            self._trim_memory()

    def _transform_flowdir_cols_inplace(self, df: pd.DataFrame) -> None:
        """Modifies DataFrame in-place to replace flow direction angles."""
        flow_cols = [c for c in df.columns if self.re_flowdir.search(c)]
        if not flow_cols:
            return

        # Explicitly enforce float32 to prevent automatic float64 casting from eating extra memory
        radians = np.deg2rad(df[flow_cols].astype(np.float32))
        for col in flow_cols:
            # We inject _sin and _cos before the year to match the _t parsing logic later
            # e.g., flowdir_2004 -> flowdir_sin_2004
            match = re.search(r"_(\d{4})", col)
            if match:
                base = col[:match.start()]
                suffix = col[match.start():]
                sin_col = f"{base}_sin{suffix}"
                cos_col = f"{base}_cos{suffix}"
            else:
                sin_col = f"{col}_sin"
                cos_col = f"{col}_cos"

            df[sin_col] = np.sin(radians[col]).astype(np.float32)
            df[cos_col] = np.cos(radians[col]).astype(np.float32)

        df.drop(columns=flow_cols, inplace=True)
        del radians

    def _get_column_metadata(self, columns: List[str]) -> pd.DataFrame:
        """Efficiently parses column names to extract variables and years, handling _filled suffixes."""
        records = []
        # Looks for _YYYY possibly followed by _filled (e.g. bathy_2004, bathy_2004_filled, flowdir_sin_2004)
        year_re = re.compile(r"_(\d{4})(?:_filled)?$")
        
        for c in columns:
            # Safely skip year pair forcing columns and standalone files (e.g. 1998_2004_tsm_mean)
            if re.search(r"\d{4}_\d{4}", c):
                continue
                
            match = year_re.search(c)
            if match:
                year = int(match.group(1))
                var_base = c[:match.start()]
                records.append({"colname": c, "year": year, "var_base": var_base})
                
        if not records:
            return pd.DataFrame(columns=["colname", "year", "var_base"])
            
        return pd.DataFrame(records)

    def _process_and_save_training_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, current_index=None, total_count=None) -> Tuple[List[str], str]:
        """Processes a training tile and writes out BOTH a wide format and batch format data files."""
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        saved_files = []
        year_ranges = getattr(self, 'year_ranges', [])
        
        if not year_ranges:
             logger.warning(f"{progress_str} [WARNING] 'year_ranges' is empty or missing! No pairs will be processed for {tile_name}.")

        rename_dict_global = {}
        for c in gdf.columns:
            new_c = self._standardize_col_name(c)
            if new_c != c:
                rename_dict_global[c] = new_c

        if rename_dict_global:
            gdf.rename(columns=rename_dict_global, inplace=True)

        # ==========================================
        # 1. WIDE FORMAT GENERATION
        # ==========================================
        wide_gdf = gdf.copy()
        
        rename_dict_wide = {}
        if 'x' in wide_gdf.columns: rename_dict_wide['x'] = 'X'
        if 'y' in wide_gdf.columns: rename_dict_wide['y'] = 'Y'

        wide_gdf.rename(columns=rename_dict_wide, inplace=True)

        valid_pairs = []
        for y0, y1 in year_ranges: 
            y0_str, y1_str = str(y0), str(y1)
            
            def get_bathy_col(year_str):
                pattern = re.compile(rf"^bathy_{year_str}_filled$", re.IGNORECASE)
                cols = [c for c in wide_gdf.columns if pattern.match(c)]
                return cols[0] if cols else None

            b_y0 = get_bathy_col(y0_str)
            b_y1 = get_bathy_col(y1_str)

            if b_y0 and b_y1:
                delta_name = f"delta_bathy_{y0_str}_{y1_str}"
                wide_gdf[delta_name] = wide_gdf[b_y1] - wide_gdf[b_y0]
                valid_pairs.append((y0, y1))
                
        # Drop year-pair columns without a matching delta
        valid_pair_strs = [f"{y0}_{y1}" for y0, y1 in valid_pairs]
        cols_to_drop = []
        for c in wide_gdf.columns:
            m = re.search(r"(\d{4}_\d{4})$", c)
            if m and not c.startswith("delta_bathy_"):
                if m.group(1) not in valid_pair_strs:
                    cols_to_drop.append(c)
        if cols_to_drop:
            wide_gdf.drop(columns=cols_to_drop, inplace=True)
        
        cols_created_wide = []
        out_name_wide = f"{tile_name}_training_formatted.parquet"
        out_path_wide = str(UPath(output_dir) / out_name_wide)
        
        if not self.overwrite and UPath(out_path_wide).exists():
            logger.info(f"{progress_str} [SKIP] Saved training WIDE tile already exists: {out_path_wide}")
            saved_files.append(out_name_wide)
        else:
            try:
                wide_gdf.to_parquet(out_path_wide, index=None, engine="pyarrow")
                cols_created_wide = wide_gdf.columns.tolist()
                logger.info(f"{progress_str} [SUCCESS] Saved training WIDE tile to: {out_path_wide}")
                saved_files.append(out_name_wide)
            except Exception as e:
                logger.error(f"{progress_str} [ERROR] Failed to save parquet file {out_path_wide}: {str(e)}")
                raise e
            
        # ==========================================
        # 2. BATCH FORMAT GENERATION 
        # ==========================================
        cols_created_batch = []
        
        for y0, y1 in valid_pairs:
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"
            
            pair_df = pd.DataFrame()
            if 'X' in wide_gdf.columns: pair_df['X'] = wide_gdf['X']
            if 'Y' in wide_gdf.columns: pair_df['Y'] = wide_gdf['Y']
            if 'FID' in wide_gdf.columns: pair_df['FID'] = wide_gdf['FID']
            if 'tile_id' in wide_gdf.columns: pair_df['tile_id'] = wide_gdf['tile_id']
            
            pair_df['year_t'] = y1
            pair_df['year_ti'] = y0
            
            b_y0 = get_bathy_col(y0_str)
            b_y1 = get_bathy_col(y1_str)
            if b_y0: pair_df['bathy_ti'] = wide_gdf[b_y0]
            if b_y1: pair_df['bathy_t'] = wide_gdf[b_y1]
            
            # Map derivatives corresponding to the t year (second year in pair) to _t
            for c in wide_gdf.columns:
                if c.endswith(f"_{y1_str}") and c != b_y1:
                    base = c.replace(f"_{y1_str}", "").lower()
                    if "bpi_broad" in base: pair_df['bpi_broad_t'] = wide_gdf[c]
                    elif "bpi_fine" in base: pair_df['bpi_fine_t'] = wide_gdf[c]
                    elif "curv_plan" in base: pair_df['curv_plan_t'] = wide_gdf[c]
                    elif "curv_profile" in base: pair_df['curv_profile_t'] = wide_gdf[c]
                    elif "curv_total" in base: pair_df['curv_total_t'] = wide_gdf[c]
                    elif "flowacc" in base: pair_df['flowacc_t'] = wide_gdf[c]
                    elif "flowdir" in base:
                        rad = np.deg2rad(wide_gdf[c].astype(np.float32))
                        pair_df['flowdir_cos_t'] = np.cos(rad)
                        pair_df['flowdir_sin_t'] = np.sin(rad)
                    elif "gradmag" in base: pair_df['gradmag_t'] = wide_gdf[c]
                    elif "rugosity" in base: pair_df['rugosity_t'] = wide_gdf[c]
                    elif "shearproxy" in base: pair_df['shearproxy_t'] = wide_gdf[c]
                    elif "slope_deg" in base: pair_df['slope_deg_t'] = wide_gdf[c]
                    elif "slope" in base: pair_df['slope_t'] = wide_gdf[c]
                    elif "tci" in base: pair_df['tci_t'] = wide_gdf[c]
                    elif "terrain_classification" in base: pair_df['terrain_classification_t'] = wide_gdf[c]
                    
            delta_name = f"delta_bathy_{y0_str}_{y1_str}"
            if delta_name in wide_gdf.columns:
                pair_df['delta_bathy'] = wide_gdf[delta_name]
                
            hurr_col = f"hurr_strength_mean_{y0_str}_{y1_str}"
            if hurr_col in wide_gdf.columns: pair_df[hurr_col] = wide_gdf[hurr_col]
            
            tsm_col = f"tsm_mean_{y0_str}_{y1_str}"
            if tsm_col in wide_gdf.columns: pair_df[tsm_col] = wide_gdf[tsm_col]
            
            grain_cols = [c for c in wide_gdf.columns if "grain" in c.lower() or "sed_size" in c.lower()]
            if grain_cols: pair_df['grain_size_layer'] = wide_gdf[grain_cols[0]]
            
            sed_cols = [c for c in wide_gdf.columns if "prim_sed" in c.lower() or "sed_type" in c.lower()]
            if sed_cols: pair_df['prim_sed_layer'] = wide_gdf[sed_cols[0]]
            
            survey_cols = [c for c in wide_gdf.columns if "survey" in c.lower()]
            if survey_cols: pair_df['survey_end_date'] = wide_gdf[survey_cols[0]]

            ordered_cols = [
                'X', 'Y', 'FID', 'tile_id', 'year_ti', 'year_t', 
                'bathy_ti', 'bathy_t', 'bpi_broad_t', 'bpi_fine_t', 
                'curv_plan_t', 'curv_profile_t', 'curv_total_t', 'flowacc_t', 
                'flowdir_cos_t', 'flowdir_sin_t', 'gradmag_t', 'rugosity_t', 
                'shearproxy_t', 'slope_t', 'slope_deg_t', 'tci_t', 
                'terrain_classification_t', 'delta_bathy', 
                f'hurr_strength_mean_{y0_str}_{y1_str}', f'tsm_mean_{y0_str}_{y1_str}', 
                'grain_size_layer', 'prim_sed_layer', 'survey_end_date'
            ]
            
            final_cols = [c for c in ordered_cols if c in pair_df.columns]
            pair_df = pair_df[final_cols].drop_duplicates()
            
            out_name_batch = f"{tile_name}_{pair_name}_training_batch.parquet"
            out_path_batch = str(UPath(output_dir) / out_name_batch)
            
            if not self.overwrite and UPath(out_path_batch).exists():
                logger.info(f"{progress_str} [SKIP] Saved training BATCH tile already exists: {out_path_batch}")
                saved_files.append(out_name_batch)
            else:
                try:
                    pair_df.to_parquet(out_path_batch, index=None, engine="pyarrow")
                    if not cols_created_batch:
                        cols_created_batch = pair_df.columns.tolist()
                    logger.info(f"{progress_str} [SUCCESS] Saved training BATCH tile to: {out_path_batch}")
                    saved_files.append(out_name_batch)
                except Exception as e:
                    logger.error(f"{progress_str} [ERROR] Failed to save parquet file {out_path_batch}: {str(e)}")
                    raise e
                
            del pair_df
            gc.collect()

        del wide_gdf
        gc.collect()

        summary = []
        if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

        return saved_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"

    def _process_and_save_prediction_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, current_index=None, total_count=None) -> Tuple[List[str], str]:
        """Processes a prediction tile and writes out BOTH a wide format and batch format data files."""
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        saved_files = []
        year_ranges = getattr(self, 'year_ranges', [])

        rename_dict_global = {}
        for c in gdf.columns:
            new_c = self._standardize_col_name(c)
            if new_c != c:
                rename_dict_global[c] = new_c

        if rename_dict_global:
            gdf.rename(columns=rename_dict_global, inplace=True)

        # --- STRICT PREDICTION COLUMN FILTERING ---
        # Prediction datasets must ONLY use BlueTopo ('bt.') features for terrain, along with static and forcing variables.
        # This strips out all the underlying LiDAR survey data (e.g. bathy_2004, slope_2015_filled) that was used for training.
        id_cols = [c for c in ["X", "Y", "FID", "tile_id", "geometry"] if c in gdf.columns]
        bt_cols = [c for c in gdf.columns if c.startswith("bt.")]
        
        # Safely fetch standalone cols (tsm, hurr, sed, grain, survey) ignoring where the year chunk is
        other_cols = [c for c in gdf.columns if re.search(r"\d{4}_\d{4}", c) or any(p in c.lower() for p in ["grain", "sed", "survey", "tsm", "hurr"])]
        
        valid_cols = id_cols + bt_cols + other_cols
        valid_cols = list(dict.fromkeys([c for c in valid_cols if c in gdf.columns]))
        gdf = gdf[valid_cols].copy()

        # ==========================================
        # 1. WIDE FORMAT GENERATION
        # ==========================================
        wide_gdf = gdf.copy()
        
        rename_dict_wide = {}
        if 'x' in wide_gdf.columns: rename_dict_wide['x'] = 'X'
        if 'y' in wide_gdf.columns: rename_dict_wide['y'] = 'Y'

        wide_gdf.rename(columns=rename_dict_wide, inplace=True)

        # Strip _filled from wide prediction columns if present, leaving standalone cols completely untouched
        filled_cols = [c for c in wide_gdf.columns if "_filled" in c and c not in other_cols]
        if filled_cols:
            wide_gdf.rename(columns={c: c.replace("_filled", "") for c in filled_cols}, inplace=True)

        valid_pairs = []
        for y0, y1 in year_ranges: 
            y0_str, y1_str = str(y0), str(y1)
            
            def get_bt_col(year_str):
                pattern = re.compile(rf"^bt\.(?:bluetopo_)?{year_str}$", re.IGNORECASE)
                cols = [c for c in wide_gdf.columns if pattern.match(c)]
                return cols[0] if cols else None

            b_y0 = get_bt_col(y0_str)
            b_y1 = get_bt_col(y1_str)

            if b_y0 and b_y1:
                delta_name = f"delta_bathy_{y0_str}_{y1_str}"
                wide_gdf[delta_name] = wide_gdf[b_y1] - wide_gdf[b_y0]
                valid_pairs.append((y0, y1))
                
        # Drop year-pair columns without a matching delta
        valid_pair_strs = [f"{y0}_{y1}" for y0, y1 in valid_pairs]
        cols_to_drop = []
        for c in wide_gdf.columns:
            m = re.search(r"(\d{4}_\d{4})$", c)
            if m and not c.startswith("delta_bathy_"):
                if m.group(1) not in valid_pair_strs:
                    cols_to_drop.append(c)
        if cols_to_drop:
            wide_gdf.drop(columns=cols_to_drop, inplace=True)

        cols_created_wide = []
        out_name_wide = f"{tile_name}_prediction_formatted.parquet"
        out_path_wide = str(UPath(output_dir) / out_name_wide)
        
        if not self.overwrite and UPath(out_path_wide).exists():
            logger.info(f"{progress_str} [SKIP] Saved prediction WIDE tile already exists: {out_path_wide}")
            saved_files.append(out_name_wide)
        else:
            try:
                wide_gdf.to_parquet(out_path_wide, index=None, engine="pyarrow")
                cols_created_wide = wide_gdf.columns.tolist()
                logger.info(f"{progress_str} [SUCCESS] Saved prediction WIDE tile to: {out_path_wide}")
                saved_files.append(out_name_wide)
            except Exception as e:
                logger.error(f"{progress_str} [ERROR] Failed to save prediction parquet file {out_path_wide}: {str(e)}")
                raise e
            
        # ==========================================
        # 2. BATCH FORMAT GENERATION
        # ==========================================
        cols_created_batch = []
        
        for y0, y1 in valid_pairs:
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"
            
            pair_df = pd.DataFrame()
            if 'X' in wide_gdf.columns: pair_df['X'] = wide_gdf['X']
            if 'Y' in wide_gdf.columns: pair_df['Y'] = wide_gdf['Y']
            if 'FID' in wide_gdf.columns: pair_df['FID'] = wide_gdf['FID']
            if 'tile_id' in wide_gdf.columns: pair_df['tile_id'] = wide_gdf['tile_id']
            
            def get_bt_col(year_str):
                pattern = re.compile(rf"^bt\.(?:bluetopo_)?{year_str}$", re.IGNORECASE)
                cols = [c for c in wide_gdf.columns if pattern.match(c)]
                return cols[0] if cols else None
            
            # Map target 't' year (y1) to the features. BlueTopo uses 'bt.' prefix.
            b_y1 = get_bt_col(y1_str)
            if b_y1: pair_df['bathy_t'] = wide_gdf[b_y1]
            
            # Extract bt.*_YYYY variables for the target year (y1)
            for c in wide_gdf.columns:
                if c.endswith(f"_{y1_str}") and c != b_y1 and c.startswith("bt."):
                    base = c.replace(f"_{y1_str}", "").replace("bt.", "").lower()
                    if "bpi_broad" in base: pair_df['bpi_broad_t'] = wide_gdf[c]
                    elif "bpi_fine" in base: pair_df['bpi_fine_t'] = wide_gdf[c]
                    elif "curv_plan" in base: pair_df['curv_plan_t'] = wide_gdf[c]
                    elif "curv_profile" in base: pair_df['curv_profile_t'] = wide_gdf[c]
                    elif "curv_total" in base: pair_df['curv_total_t'] = wide_gdf[c]
                    elif "flowacc" in base: pair_df['flowacc_t'] = wide_gdf[c]
                    elif "flowdir" in base:
                        rad = np.deg2rad(wide_gdf[c].astype(np.float32))
                        pair_df['flowdir_sin_t'] = np.sin(rad)
                        pair_df['flowdir_cos_t'] = np.cos(rad)
                    elif "gradmag" in base: pair_df['gradmag_t'] = wide_gdf[c]
                    elif "rugosity" in base: pair_df['rugosity_t'] = wide_gdf[c]
                    elif "shearproxy" in base: pair_df['shearproxy_t'] = wide_gdf[c]
                    elif "slope_deg" in base: pair_df['slope_deg_t'] = wide_gdf[c]
                    elif "slope" in base: pair_df['slope_t'] = wide_gdf[c]
                    elif "tci" in base: pair_df['tci_t'] = wide_gdf[c]
                    elif "terrain_classification" in base: pair_df['terrain_classification_t'] = wide_gdf[c]
                    elif "unc" in base or "uncertainty" in base: pair_df['uc_t'] = wide_gdf[c]
                    
            hurr_col = f"hurr_strength_mean_{y0_str}_{y1_str}"
            if hurr_col in wide_gdf.columns: pair_df[hurr_col] = wide_gdf[hurr_col]
            
            tsm_col = f"tsm_mean_{y0_str}_{y1_str}"
            if tsm_col in wide_gdf.columns: pair_df[tsm_col] = wide_gdf[tsm_col]
            
            grain_cols = [c for c in wide_gdf.columns if "grain" in c.lower() or "sed_size" in c.lower()]
            if grain_cols: pair_df['grain_size_layer'] = wide_gdf[grain_cols[0]]
            
            sed_cols = [c for c in wide_gdf.columns if "prim_sed" in c.lower() or "sed_type" in c.lower()]
            if sed_cols: pair_df['prim_sed_layer'] = wide_gdf[sed_cols[0]]
            
            survey_cols = [c for c in wide_gdf.columns if "survey" in c.lower()]
            if survey_cols: pair_df['survey_end_date'] = wide_gdf[survey_cols[0]]

            ordered_cols = [
                'X', 'Y', 'FID', 'tile_id', 'bathy_t', 'bpi_broad_t', 'bpi_fine_t', 
                'curv_plan_t', 'curv_profile_t', 'curv_total_t', 'flowacc_t', 
                'gradmag_t', 'rugosity_t', 'shearproxy_t', 'slope_t', 'slope_deg_t', 
                'tci_t', 'terrain_classification_t', 'uc_t', 'flowdir_sin_t', 'flowdir_cos_t', 
                f'hurr_strength_mean_{y0_str}_{y1_str}', f'tsm_mean_{y0_str}_{y1_str}', 
                'grain_size_layer', 'prim_sed_layer', 'survey_end_date' 
            ]
            
            final_cols = [c for c in ordered_cols if c in pair_df.columns]
            pair_df = pair_df[final_cols].drop_duplicates()
            
            out_name_batch = f"{tile_name}_{pair_name}_prediction_batch.parquet"
            out_path_batch = str(UPath(output_dir) / out_name_batch)
            
            if not self.overwrite and UPath(out_path_batch).exists():
                logger.info(f"{progress_str} [SKIP] Saved prediction BATCH tile already exists: {out_path_batch}")
                saved_files.append(out_name_batch)
            else:
                try:
                    pair_df.to_parquet(out_path_batch, index=None, engine="pyarrow")
                    if not cols_created_batch:
                        cols_created_batch = pair_df.columns.tolist()
                    logger.info(f"{progress_str} [SUCCESS] Saved prediction BATCH tile to: {out_path_batch}")
                    saved_files.append(out_name_batch)
                except Exception as e:
                    logger.error(f"{progress_str} [ERROR] Failed to save parquet file {out_path_batch}: {str(e)}")
                    raise e
                
            del pair_df
            gc.collect()

        del wide_gdf
        gc.collect()

        summary = []
        if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

        return saved_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"
        
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
            with tempfile.TemporaryDirectory(dir=self.local_tmp_dir) as task_tmp_dir:
                local_tmp_path = str(Path(task_tmp_dir) / "subgrids_tmp.gpkg")
                
                logger.info(f" -> Writing GPKG locally to {local_tmp_path} before uploading...")
                intersecting_sub_grids.to_file(local_tmp_path, driver="GPKG") 
                
                logger.info(f" -> Uploading subgrids to S3: {output_path}")
                self.fs.put(local_tmp_path, str(output_path))
        else:
            output_upath = UPath(output_path)
            output_upath.parent.mkdir(parents=True, exist_ok=True)
            intersecting_sub_grids.to_file(str(output_upath), driver="GPKG") 

        logger.info(f"[SUCCESS] Successfully saved {process_type} subgrids to: {output_path}")
        return