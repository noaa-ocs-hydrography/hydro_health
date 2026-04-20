"""Class for data acquisition and preprocessing of model data"""

import os
import re
import pathlib
import yaml
import warnings
import tempfile
import shutil
import gc # <--- Added for manual memory management
import logging # <--- Added for standard logging
from logging.handlers import RotatingFileHandler
from typing import List, Tuple, Literal
from pathlib import Path
from rasterio.features import shapes, geometry_mask # <--- geometry_mask added for TSM smoothing
from rasterio.warp import transform_bounds
from shapely.geometry import shape
import s3fs
from scipy.ndimage import convolve, uniform_filter # <--- Added for TSM smoothing

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import dask
from dask.distributed import Client, LocalCluster
from osgeo import gdal
from shapely.geometry import Point, box, Polygon, MultiPolygon, GeometryCollection
from upath import UPath 

from hydro_health.helpers.tools import get_config_item, get_environment
from hydro_health.engines.CreateSeabedTerrainLayerEngine import CreateSeabedTerrainLayerEngine

dask.config.set({"distributed.worker.memory.terminate": 0.95})
dask.config.set({"distributed.worker.memory.pause": 0.85})
dask.config.set({"distributed.worker.memory.spill": 0.80})

# Must be set before starting the Dask cluster
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"

# GDAL Configuration and S3 Network Optimizations
os.environ["GDAL_CACHEMAX"] = "512"             # 512 MB Cache
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"
os.environ["AWS_MAX_CONNECTIONS"] = "32"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "268435456"       # 256 MB VSI Cache (Standard for large TIFFs)


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
        ), # Rotates file when it hits 10MB, keeping the last 3 backups
        logging.StreamHandler()             # Prints to console
    ]
)
logger = logging.getLogger(__name__)


class ModelDataPreProcessor:
    """Class for parallel preprocessing all model data"""

    def __init__(self, overwrite: bool = False, pilot_mode: bool=False):
        self.target_crs = "EPSG:32617"
        self.target_res = 8
        self.pilot_mode = pilot_mode
        self.overwrite = overwrite

        self.fs = s3fs.S3FileSystem(anon=False)

        self.static_patterns = ['sed_size_raster', 'sed_type_raster', 'tsm_mean', 'hurr']
        
        self.year_ranges = [
            (1998, 2004), (2004, 2006), (2006, 2007), (2007, 2010),
            (2010, 2015), (2014, 2022), (2016, 2017), (2017, 2018),
            (2018, 2019), (2020, 2022), (2022, 2024)
        ]

        self.re_year_suffix = re.compile(r"_\d{4}$")
        self.re_year_extract = re.compile(r"_(\d{4})")
        self.re_bt_prefix = re.compile(r"^bt\.")
        self.re_flowdir = re.compile(r"flowdir")

        self.excluded_keys = self._load_exclusion_config()
        self.is_aws = (get_environment() == 'aws')

    def create_file_paths(self):
        """Creates unified UPath objects that work both locally and on S3."""
        # Determine the base prefix depending on the environment
        prefix = f"s3://{get_config_item('S3', 'BUCKET_NAME', pilot_mode=self.pilot_mode)}/" if self.is_aws else ""
        logger.info(f"Using base prefix: '{prefix}' for file paths.")
        
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
        
        # Use the user's home directory to guarantee write permissions and use the large EBS volume
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

    def _clean_geometry(self, geom):
        """Fix geometric artifacts caused by union_all() and enforce MultiPolygon."""
        # Removed clean_geom = geom.buffer(0) as it is computationally ruinous on large rasters. 
        # Modern union_all() typically yields valid geometries.
        
        if geom.geom_type == 'GeometryCollection':
            polys = [g for g in geom.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
            if not polys:
                raise ValueError("No polygons found in the mask geometry.")
            return MultiPolygon(polys)
        elif geom.geom_type == 'Polygon':
            return MultiPolygon([geom])
        
        return geom

    def process(self) -> None:
        """Main function to process model data."""   
        logger.info(f"Starting ModelDataPreProcessor. Logs are being saved to: {LOG_FILE_PATH}")
        self.create_file_paths()
        self._clean_local_tmp()

        # This tells Dask: "Use the whole machine, but stay under 26GB total."
        cluster = LocalCluster(
            n_workers=16,          # Manually restricting workers is still safer for heavy Tiff warps
            threads_per_worker=1,  # GDAL internally uses threads, so we give each worker a couple threads to work with
            memory_limit='2GB'    # 6GB * 5 workers = 30GB total. Safe for 32GB RAM.
        )
        client = Client(cluster)
        logger.info(f"Dask Dashboard: {client.dashboard_link}")

        try:        
            mask_pred_gdf = gpd.read_parquet(str(self.mask_prediction_pq))
            mask_train_gdf = gpd.read_parquet(str(self.mask_training_pq))

            logger.info("Cleaning and preparing mask geometries...")
            mask_pred_clean = self._clean_geometry(mask_pred_gdf.union_all())
            mask_train_clean = self._clean_geometry(mask_train_gdf.union_all())

            logger.info("Generating single shared cutline files...")
            pred_cutline_path = str(self.local_tmp_dir / "pred_cutline.geojson")
            train_cutline_path = str(self.local_tmp_dir / "train_cutline.geojson")
            
            # Save the cutline files ONCE before distributing tasks to workers to prevent memory crashes
            gpd.GeoDataFrame(geometry=[mask_pred_clean], crs=self.target_crs).to_file(pred_cutline_path, driver='GeoJSON')
            gpd.GeoDataFrame(geometry=[mask_train_clean], crs=self.target_crs).to_file(train_cutline_path, driver='GeoJSON')

            # INSTEAD of scattering the massive geometries to the workers, extract the 4-coordinate bounding box.
            # Passing a tiny tuple eliminates the massive deserialization memory spikes.
            mask_pred_bounds = mask_pred_clean.bounds
            mask_train_bounds = mask_train_clean.bounds

            self.parallel_processing_rasters(
                self.preprocessed_dir, 
                mask_pred_bounds, 
                mask_train_bounds,
                pred_cutline_path,
                train_cutline_path
            )

            self.clip_rasters_by_tile(
                raster_dir=self.prediction_out_dir, 
                output_dir=self.prediction_tiles_dir, 
                data_type="prediction"
            )

            self.clip_rasters_by_tile(
                raster_dir=self.training_out_dir, 
                output_dir=self.training_tiles_dir, 
                data_type="training"
            )
            
            self.batch_long_format_transformation(base_dir=self.training_tiles_dir, mode="training")

        except Exception as e:
            logger.exception("A critical error occurred in the main process loop.")
        finally:
            client.close()

    def parallel_processing_rasters(self, input_directory, mask_pred_bounds, mask_train_bounds, pred_cutline_path, train_cutline_path) -> None:
        """Process prediction and training rasters in parallel using Dask."""
        input_directory = UPath(input_directory)
        
        if not self.is_aws:
            self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Identify Existing Outputs ---
        potential_files = list(self.prediction_out_dir.rglob("*"))
        existing_pred_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        logger.info(f"Checking existing outputs to prevent redundant processing...")
        logger.info(f" -> Found {len(existing_pred_outputs)} files in prediction output directory.")

        potential_files = list(self.uncombined_lidar_dir.rglob("*"))
        existing_uncombined_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        logger.info(f" -> Found {len(existing_uncombined_outputs)} files in uncombined lidar directory.")

        all_existing_outputs = existing_pred_outputs.union(existing_uncombined_outputs)

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

        logger.info(f"Found {len(potential_files)} total potential prediction files in input directories.")

        excluded_folders = {'filled_tifs', 'combined_lidar_tifs'}
        prediction_files = [] 

        removed_existing = 0
        removed_folders = 0
        files_removed_by_keys = []
        kept_mosaic_files = []

        for f in potential_files:
            if not self.overwrite and f.name in all_existing_outputs:
                removed_existing += 1
                continue
                
            if any(key in f.name for key in self.excluded_keys):
                files_removed_by_keys.append(f.name) 
                continue
                
            if any(folder in f.parts for folder in excluded_folders):
                removed_folders += 1
                continue
                
            prediction_files.append(f)
            
            if "mosaic" in f.name.lower():
                kept_mosaic_files.append(f.name)

        logger.info(f"--- File Filtering Summary ---")
        logger.info(f" -> Total potential files: {len(potential_files)}")
        if self.overwrite:
            logger.info(f" - Kept existing target files (Overwrite Enabled)")
        else:
            logger.info(f" -> Removed (already existing): {removed_existing}")
        logger.info(f" -> Removed (excluded keys): {len(files_removed_by_keys)}") 
        logger.info(f" -> Removed (excluded folders): {removed_folders}")
        logger.info(f" -> Files kept for processing: {len(prediction_files)}")
        logger.info(f"------------------------------------")

        skip_msg = f" (Skipping {removed_existing} existing)" if not self.overwrite else " (Overwrite enabled)"

        logger.info(f"Found {len(kept_mosaic_files)} lidar mosaic files to process.")
        
        if self.overwrite:
            logger.info(f"Processing {len(prediction_files)} prediction files (Overwrite enabled)...")
        else:
            logger.info(f"Outputting uncombined lidar to: {self.uncombined_lidar_dir}")
        logger.info(f"Outputting prediction rasters to: {self.prediction_out_dir}")
        logger.info(f"Queuing {len(prediction_files)} prediction files{skip_msg}...")
        
        # --- 3. Queue Prediction Tasks ---
        prediction_tasks = []
        for file_path in prediction_files:
            base_out = self.uncombined_lidar_dir if "mosaic" in file_path.name.lower() else self.prediction_out_dir
            output_path = base_out / file_path.name
            
            task = dask.delayed(self.process_prediction_raster)(
                str(file_path), 
                mask_pred_bounds, 
                str(output_path),
                pred_cutline_path
            )
            prediction_tasks.append(task)

        if prediction_tasks:
            dask.compute(*prediction_tasks)
            logger.info("[SUCCESS] Prediction raster processing complete.")
        else:
            logger.info("No new prediction rasters to process.")

        logger.info("Running Seabed Terrain Layer Engine...")
        engine = CreateSeabedTerrainLayerEngine()
        engine.process()

        if not self.is_aws:
            self.training_out_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. Gather Training Inputs (from Prediction Outputs) ---
        potential_files = list(self.training_out_dir.rglob("*"))
        existing_train_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }

        potential_files = list(self.prediction_out_dir.rglob("*"))
        training_candidates = [
            f for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        ]

        training_files = [
            f for f in training_candidates
            if (self.overwrite or f.name not in existing_train_outputs)
            and not ('mosaic' in f.name and 'filled' not in f.name)
        ]

        skip_train_msg = f" (Skipping {len(existing_train_outputs)} existing)" if not self.overwrite else " (Overwrite enabled)"
        
        logger.info(f"Outputting training rasters to: {self.training_out_dir}")
        logger.info(f"Queuing {len(training_files)} training files{skip_train_msg}...")

        # --- 5. Queue Training Tasks ---
        training_tasks = []
        for file_path in training_files:
            output_path = self.training_out_dir / file_path.name
            
            task = dask.delayed(self.process_training_raster)(
                str(file_path), 
                mask_train_bounds, 
                str(output_path),
                train_cutline_path
            )
            training_tasks.append(task)

        if training_tasks:
            dask.compute(*training_tasks)
            logger.info("[SUCCESS] Training raster processing complete.")
        else:
            logger.info("No new training rasters to process.")

    def process_prediction_raster(self, raster_path, mask_bounds, output_path, cutline_path) -> None:
        """Reprojects, resamples, and crops a raster for prediction."""
        raster_name = pathlib.Path(raster_path).name.lower()
        
        logger.info(f"-> [STARTING] Worker executing prediction on: {raster_name}")

        try:
            with rasterio.open(raster_path) as src:
                src_nodata = src.nodata
                raster_crs = src.crs
                raster_bounds = src.bounds
        except Exception as e:
            # Using logger.exception gives us the full traceback automatically
            logger.exception(f"Could not open {raster_name} with rasterio. File might be corrupted.")
            return

        # CRS Reprojection Fix: Transform raster bounds to the target CRS before checking for intersection
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
                apply_tsm_smoothing=is_tsm
            )
        except Exception as e:
            logger.exception(f"Unexpected failure during _warp_to_cutline for {raster_name}.")
        finally:
            # Force garbage collection to prevent memory bloat over hundreds of processed files
            gc.collect()

    def process_training_raster(self, raster_path, mask_bounds, output_path, cutline_path) -> None:
        """Process a training raster by clipping it with a mask."""
        raster_name = pathlib.Path(raster_path).name.lower()

        logger.info(f"-> [STARTING] Worker executing training on: {raster_name}")

        try:
            with rasterio.open(raster_path) as src:
                src_nodata = src.nodata
                raster_crs = src.crs
                raster_bounds = src.bounds
        except Exception as e:
            logger.exception(f"Could not open training raster {raster_name}.")
            return

        # CRS Reprojection Fix: Transform raster bounds to the target CRS before checking for intersection
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
                logger.info(f"- [SKIP] Bounding box does not intersect raster {raster_name}. Skipping.")
                return
        except Exception as e:
            logger.exception(f"Bounding box check failed for {raster_name}.")
            return

        logger.info(f'- [PROCESSING] Starting warp on training file {raster_name}...')
        
        # FIXED: Brought over the TSM/cropping logic to the training side
        should_crop = any(k in raster_name for k in ["tsm", "sed", "hurr"])
        is_tsm = "tsm" in raster_name or "strength" in raster_name

        try:
            self._warp_to_cutline(
                raster_path,
                output_path,
                cutline_path,
                src_nodata=src_nodata,
                dst_nodata=np.nan,
                crop_to_cutline=should_crop,    # Added
                apply_tsm_smoothing=is_tsm      # Added
            )
        except Exception as e:
            logger.exception(f"Unexpected failure during _warp_to_cutline for {raster_name}.")
        finally:
            # Force garbage collection to prevent memory bloat over hundreds of processed files
            gc.collect()

    def _warp_to_cutline(self, src_path, dst_path, cutline_path, **kwargs):
        """Helper to handle GDAL Warp boilerplate."""
        
        # --- 1. PREPARE PATHS FOR GDAL ---
        src_str = str(src_path)
        dst_str = str(dst_path)

        # GDAL uses /vsis3/ instead of s3:// for READING
        if self.is_aws and src_str.startswith('s3://'):
            src_str = src_str.replace('s3://', '/vsis3/')

        # --- 2. SETUP TEMP OUTPUT PATH FOR AWS ---
        if self.is_aws:
            with tempfile.NamedTemporaryFile(dir=self.local_tmp_dir, suffix='.tif', delete=False) as tmp_out:
                gdal_dst_str = tmp_out.name
        else:
            gdal_dst_str = dst_str

        # --- 3. CONFIGURE WARP ---
        warp_opts = {
            'cutlineDSName': cutline_path,
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE'],
            'creationOptions': [
                'TILED=YES', 
                'BLOCKXSIZE=512',       # NEW: Sets tile width to 512 pixels
                'BLOCKYSIZE=512',       # NEW: Sets tile height to 512 pixels
                'COMPRESS=LZW',         # (You can also try 'COMPRESS=ZSTD' here)
                'BIGTIFF=YES',
                'NUM_THREADS=1'         # CRITICAL FIX: Stops GDAL from spawning 32 threads inside EACH dask worker
            ],
            'multithread': False,       # CRITICAL FIX: Let Dask handle file-level parallelism instead of GDAL.
            'warpMemoryLimit': 512,    # Reduced memory footprint per warp to safely accommodate multi-worker processing.
            'resampleAlg': 'bilinear'
        }
        
        if 'dst_crs' in kwargs: warp_opts['dstSRS'] = kwargs.pop('dst_crs')
        if 'x_res' in kwargs: warp_opts['xRes'] = kwargs.pop('x_res')
        if 'y_res' in kwargs: warp_opts['yRes'] = kwargs.pop('y_res')
        if 'crop_to_cutline' in kwargs: warp_opts['cropToCutline'] = kwargs.pop('crop_to_cutline')
        if 'src_nodata' in kwargs: warp_opts['srcNodata'] = kwargs.pop('src_nodata')
        if 'dst_nodata' in kwargs: warp_opts['dstNodata'] = kwargs.pop('dst_nodata')
        
        # FIXED: Removed the duplicate `.pop()` that was overwriting this to False
        apply_tsm_smoothing = kwargs.pop('apply_tsm_smoothing', False)
        
        # --- 4. EXECUTE WARP ---
        try:
            # Output directly to our safe local file
            ds = gdal.Warp(gdal_dst_str, src_str, **warp_opts)

            if ds is None:
                raise RuntimeError(f"gdal.Warp returned None for {os.path.basename(src_str)}")

            # --- 6. OPTIONAL TSM SMOOTHING PASS ---
            if apply_tsm_smoothing:
                logger.info(f" [SMOOTHING] Applying memory-safe chunked focal mean smoothing to {src_str}...")

                # Calculate smoothing pixel radius based on warp dataset transform
                pixel_size = ds.GetGeoTransform()[1]
                radius_pixels = int(2000 / abs(pixel_size))
                size = radius_pixels * 2 + 1
                
                # IMPORTANT: Fully close the GDAL dataset to release file locks 
                # before we reopen it with rasterio
                ds = None
                
                smoothed_tmp = gdal_dst_str.replace('.tif', '_smoothed.tif')
                cutline_gdf = gpd.read_file(cutline_path)
                mask_geometry = cutline_gdf.geometry.iloc[0]

                # Reopen with rasterio to enable chunked window processing
                with rasterio.open(gdal_dst_str) as src:
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'tiled': True,
                        'blockxsize': 512,
                        'blockysize': 512
                    })
                    nodata = src.nodata if src.nodata is not None else warp_opts.get('dstNodata', warp_opts.get('srcNodata', -9999.0))
                    
                    with rasterio.open(smoothed_tmp, 'w', **kwargs) as dst:
                        # Process the file in manageable 4096x4096 chunks (approx 67MB each)
                        block_size = 1024
                        
                        for y in range(0, src.height, block_size):
                            for x in range(0, src.width, block_size):
                                core_width = min(block_size, src.width - x)
                                core_height = min(block_size, src.height - y)
                                window = rasterio.windows.Window(x, y, core_width, core_height)
                                
                                # Read with overlap to prevent edge artifacts in smoothing
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
                                
                                # Safe NoData mask creation
                                if pd.isna(nodata):
                                    valid_mask = (~np.isnan(array)).astype(np.float32)
                                    array[np.isnan(array)] = 0
                                else:
                                    valid_mask = (array != nodata).astype(np.float32)
                                    array[array == nodata] = 0

                                # Run scipy's uniform_filter (fast O(1) sliding window)
                                smoothed = uniform_filter(array, size=size, mode='constant', cval=0.0)
                                weights = uniform_filter(valid_mask, size=size, mode='constant', cval=0.0)
                                
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    final_array = np.where(weights > 0, smoothed / weights, nodata)
                                
                                # Crop the expanded smoothed array back to the core window size
                                row_offset = int(y - read_yoff)
                                col_offset = int(x - read_xoff)
                                core_array = final_array[row_offset:row_offset + core_height, col_offset:col_offset + core_width]
                                
                                # Re-mask strictly to the cutline geometry boundary for this specific chunk
                                window_transform = src.window_transform(window)
                                out_of_bounds_mask = geometry_mask(
                                    [mask_geometry],
                                    out_shape=core_array.shape,
                                    transform=window_transform,
                                    invert=False
                                )
                                core_array[out_of_bounds_mask] = nodata
                                
                                # Write completed chunk directly into the new temp file
                                dst.write(core_array.astype(kwargs['dtype']), 1, window=window)

                # Swap the files: remove the unsmoothed original and replace with smoothed
                if os.path.exists(gdal_dst_str):
                    os.remove(gdal_dst_str)
                shutil.move(smoothed_tmp, gdal_dst_str)
            
            # Explicitly force garbage collection to flush and close the file
            if ds is not None:
                del ds
            
            if self.is_aws:
                # Bypass /vsis3/ entirely and use our robust s3fs client to upload the file
                self.fs.put(gdal_dst_str, dst_str)
                logger.info(f" - [✓ SUCCESS] Wrote to S3 successfully: {os.path.basename(dst_str)}")
            else:
                logger.info(f" - [✓ SUCCESS] Wrote locally successfully: {os.path.basename(dst_str)}")

        except Exception as e:
            logger.exception(f" - [✗ ERROR] GDAL Warp/Upload failed for {os.path.basename(src_str)}!")
            raise e
        finally:
            try:
                # Clean up temporary raster outputs ONLY. 
                # (Never delete the shared cutline_path here, as other workers need it!)
                if self.is_aws and os.path.exists(gdal_dst_str):
                    os.remove(gdal_dst_str)
            except Exception as e:
                # Failing to unlink shouldn't kill the distributed worker
                logger.warning(f"Failed to cleanup temp files: {e}")

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
        
        # FIX 4: Glob once before passing to workers
        logger.info(f"Scanning directory for raster files... (This will only happen once)")
        raster_dir_upath = UPath(raster_dir)
        all_raster_files = [
            str(f) for f in raster_dir_upath.rglob("*") 
            if f.suffix.lower() in {'.tif', '.tiff'}
        ]
        logger.info(f"Found {len(all_raster_files)} raster files.")

        tasks = []
        results_list = []
        batch_size = 25 # FIX 3: Batch size for memory management
        
        logger.info(f"Building Dask task graph in batches of {batch_size}...")
        for i, (_, sub_grid) in enumerate(sub_grids.iterrows()):
            tile_name = sub_grid['tile_id']
            
            output_folder = output_dir / tile_name
            expected_output_path = output_folder / f"{tile_name}_{data_type}_clipped_data.parquet"

            if not self.overwrite and expected_output_path.exists():
                logger.info(f" [SKIP] Tile already clipped: {tile_name}. Queuing stats generation only.")
                stats_task = dask.delayed(self._generate_stats_from_existing)(str(expected_output_path), tile_name)
                tasks.append(stats_task)
            else:
                # Pass the pre-globbed list of strings instead of the directory
                gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, all_raster_files)
                ungridded_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, all_raster_files)
                
                tasks.append(
                    self.save_combined_data(gridded_task, ungridded_task, output_folder, data_type, tile_id=tile_name)
                )

            # FIX 3: Process in batches and aggressively clean memory
            if len(tasks) >= batch_size or i == (len(sub_grids) - 1):
                logger.info(f"Executing batch of {len(tasks)} tile tasks via Dask...")
                batch_results = dask.compute(*tasks)
                results_list.extend(batch_results)
                
                # Reset for next batch and flush memory
                tasks = []
                gc.collect()

        logger.info("Dask computation across all batches finished successfully.")

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
        """Process gridded rasters for a single tile."""
        original_tile = sub_grid['original_tile']
                
        # Filter files passed from the parent function
        filtered_files = [
            f for f in raster_files
            if original_tile in Path(f).name
        ]

        tile_extent = sub_grid.geometry.bounds
        dfs = []
        for file in filtered_files:
            df = self._extract_raster_to_df(file, tile_extent)
            if not df.empty:
                # Memory opt: Rename 'Value' to the raster's name and drop 'Raster' immediately
                col_name = df['Raster'].iloc[0]
                df = df.rename(columns={'Value': col_name}).drop(columns=['Raster'])
                df['X'] = df['X'].round(3)
                df['Y'] = df['Y'].round(3)
                df = df.drop_duplicates(subset=['X', 'Y'])
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()

        # FIX 2: Iterative memory-efficient outer merge instead of pd.concat + pivot
        combined_data = dfs[0]
        for next_df in dfs[1:]:
            combined_data = pd.merge(combined_data, next_df, on=['X', 'Y'], how='outer')

        # FIX 1: Completely removed GeoDataFrame Point generation for memory savings.
        
        # Aggressive memory cleanup inside the worker
        del dfs
        gc.collect()
        
        return combined_data

    def subtile_process_ungridded(self, sub_grid, raster_files) -> pd.DataFrame:
        """Process ungridded rasters for a single tile."""
        dfs = []
        tile_extent = sub_grid.geometry.bounds

        for pattern in self.static_patterns:
            # Filter files passed from parent
            current_files = [f for f in raster_files if pattern in Path(f).name]

            for file in current_files:
                df = self._extract_raster_to_df(file, tile_extent)
                if not df.empty:
                    # Memory opt: Rename 'Value' to the raster's name and drop 'Raster' immediately
                    col_name = df['Raster'].iloc[0]
                    df = df.rename(columns={'Value': col_name}).drop(columns=['Raster'])
                    df['X'] = df['X'].round(3)
                    df['Y'] = df['Y'].round(3)
                    df = df.drop_duplicates(subset=['X', 'Y'])
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        # FIX 2: Iterative memory-efficient outer merge instead of pd.concat + pivot
        combined_data = dfs[0]
        for next_df in dfs[1:]:
            combined_data = pd.merge(combined_data, next_df, on=['X', 'Y'], how='outer')

        # Aggressive memory cleanup inside the worker
        del dfs
        gc.collect()
        
        return combined_data

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
    def save_combined_data(self, gridded_df, ungridded_df, output_folder, data_type, tile_id) -> pd.DataFrame:
        """Combine dataframes and save to parquet."""
        if gridded_df is None or gridded_df.empty:
            return pd.DataFrame()

        output_folder_path = UPath(output_folder)
        
        if not self.is_aws: 
            output_folder_path.mkdir(parents=True, exist_ok=True)
            
        output_path = output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet"
        save_path = str(output_path)

        if ungridded_df is not None and not ungridded_df.empty:
            combined = pd.merge(gridded_df, ungridded_df, on=['X', 'Y'], how='left')
        else:
            combined = gridded_df

        combined.to_parquet(save_path, engine="pyarrow", index=False)
        logger.info(f" [SUCCESS] Saved combined tile data to: {save_path}")
        return self.create_nan_stats_csv(combined, tile_id)

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

        tasks = []
        for fp in files_to_process:
            tasks.append(dask.delayed(self._transform_tile_task)(str(fp), mode))

        results = dask.compute(*tasks)

        success = sum(1 for r in results if r.startswith("Success"))
        failed = len(results) - success
        
        logger.info(f"[SUCCESS] Transformation Complete. Success: {success}, Failed: {failed}")
        if failed > 0:
            logger.error("Transformation Errors:\n" + "\n".join([r for r in results if not r.startswith("Success")]))

    def _transform_tile_task(self, f_path: str, mode: Literal["training", "prediction"]) -> str:
        """Dask Worker: Reads file -> Calls specific processor -> Returns status."""
        try:
            tile_name = os.path.basename(f_path).split("_")[0]
            output_dir = os.path.dirname(f_path)

            try:
                gdf = gpd.read_parquet(f_path)
            except Exception:
                df = pd.read_parquet(f_path)
                geometry_col = 'geometry' if 'geometry' in df.columns else None
                gdf = gpd.GeoDataFrame(df, geometry=geometry_col)

            if mode == "training":
                saved = self._process_and_save_training_tile(gdf, output_dir, tile_name)
            else:
                saved = self._process_and_save_prediction_tile(gdf, output_dir, tile_name)
            
            del gdf
            return f"Success: {tile_name} ({len(saved)} pairs)"

        except Exception as e:
            return f"Failed: {os.path.basename(f_path)} - {str(e)}"

    def _process_and_save_training_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str) -> List[str]:
        """Processes a training tile and writes outputs immediately to disk."""
        self._transform_flowdir_cols_inplace(gdf)
        col_meta = self._get_column_metadata(gdf.columns.tolist())
        
        saved_files = []

        for y0, y1 in self.year_ranges:
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"

            out_name = f"{tile_name}_{pair_name}_long.parquet"
            out_path = str(UPath(output_dir) / out_name)

            if not self.overwrite and UPath(out_path).exists():
                logger.info(f" [SKIP] Long-format training tile already exists: {out_path}")
                saved_files.append(out_name)
                continue

            cols_t_meta = col_meta[col_meta["year"] == y0]
            cols_t1_meta = col_meta[col_meta["year"] == y1]

            common_vars = set(cols_t_meta["var_base"]).intersection(cols_t1_meta["var_base"])
            
            if "bathy_filled" not in common_vars:
                continue

            t_map = dict(zip(cols_t_meta["var_base"], cols_t_meta["colname"]))
            t1_map = dict(zip(cols_t1_meta["var_base"], cols_t1_meta["colname"]))

            sorted_common = sorted(common_vars)
            cols_t_exist = [t_map[v] for v in sorted_common]
            cols_t1_exist = [t1_map[v] for v in sorted_common]

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

            if any(c not in gdf.columns for c in cols_to_grab):
                continue

            pair_gdf = gdf[cols_to_grab].drop_duplicates()

            new_names_t = [f"{v}_t" for v in sorted_common]
            new_names_t1 = [f"{v}_t1" for v in sorted_common]
            rename_map = dict(zip(cols_t_exist + cols_t1_exist, new_names_t + new_names_t1))
            
            pair_gdf.rename(columns=rename_map, inplace=True)
            
            if delta_bathy_col in pair_gdf.columns:
                pair_gdf.rename(columns={delta_bathy_col: "delta_bathy"}, inplace=True)

            pair_gdf["year_t"] = y0
            pair_gdf["year_t1"] = y1

            if "delta_bathy" not in pair_gdf.columns and "bathy_t" in pair_gdf.columns and "bathy_t1" in pair_gdf.columns:
                pair_gdf["delta_bathy"] = pair_gdf["bathy_t1"] - pair_gdf["bathy_t"]

            filled_cols = [c for c in pair_gdf.columns if "_filled" in c]
            if filled_cols:
                pair_gdf.rename(columns={c: c.replace("_filled", "") for c in filled_cols}, inplace=True)

            target_col = "bathy_t1"
            predictor_cols_t = [c for c in pair_gdf.columns if c.endswith("_t")]
            predictor_cols_delta = [c for c in pair_gdf.columns if c.startswith("delta_")]
            
            final_cols = id_cols + ["year_t", "year_t1", target_col] + predictor_cols_t + predictor_cols_delta + forcing_cols + static_cols
            final_cols = [c for c in final_cols if c in pair_gdf.columns]
            
            pair_gdf[final_cols].to_parquet(out_path, index=None)
            logger.info(f" [SUCCESS] Saved training long-format tile to: {out_path}")
            saved_files.append(out_name)
            del pair_gdf

        return saved_files

    def _process_and_save_prediction_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str) -> List[str]:
        """Processes a prediction tile and writes outputs immediately."""
        self._transform_flowdir_cols_inplace(gdf)
        
        bt_cols = [c for c in gdf.columns if self.re_bt_prefix.match(c)]
        new_t_names = [self.re_bt_prefix.sub("", c) + "_t" for c in bt_cols]
        
        gdf.rename(columns=dict(zip(bt_cols, new_t_names)), inplace=True)
        
        saved_files = []

        for y0, y1 in self.year_ranges:
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"
            
            out_name = f"{tile_name}_{pair_name}_prediction_long.parquet"
            out_path = str(UPath(output_dir) / out_name)

            if not self.overwrite and UPath(out_path).exists():
                logger.info(f" [SKIP] Long-format prediction tile already exists: {out_path}")
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
            
            pair_gdf.to_parquet(out_path, index=None)
            logger.info(f" [SUCCESS] Saved prediction long-format tile to: {out_path}")
            saved_files.append(out_name)
            del pair_gdf

        return saved_files

    def _transform_flowdir_cols_inplace(self, df: pd.DataFrame) -> None:
        """Modifies DataFrame in-place to replace flow direction angles."""
        flow_cols = [c for c in df.columns if self.re_flowdir.search(c)]
        if not flow_cols:
            return

        radians = np.deg2rad(df[flow_cols])
        for col in flow_cols:
            df[f"{col}_sin"] = np.sin(radians[col])
            df[f"{col}_cos"] = np.cos(radians[col])

        df.drop(columns=flow_cols, inplace=True)

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
        
        # Use GDAL's virtual file system for fast AWS reading
        if self.is_aws and open_path.startswith("s3://"):
            open_path = open_path.replace("s3://", "/vsis3/")

        geometries = []
        
        with rasterio.open(open_path) as src:
            # Process in 4096x4096 chunks to prevent loading massive Ecoregion TIFFs into RAM
            block_size = 4096
            
            for y in range(0, src.height, block_size):
                for x in range(0, src.width, block_size):
                    window = rasterio.windows.Window(
                        x, y, 
                        min(block_size, src.width - x), 
                        min(block_size, src.height - y)
                    )
                    
                    # Only read this small block into memory
                    mask_chunk = src.read(1, window=window, out_dtype='uint8')

                    if process_type == 'prediction':
                        valid_mask = mask_chunk == 1
                    elif process_type == 'training':
                        if self.pilot_mode:
                            valid_mask = mask_chunk == 1
                        else:
                            # TODO we need to double check what er3 mask uses
                            valid_mask = mask_chunk == 2

                    # Skip empty blocks entirely
                    if not valid_mask.any():
                        continue

                    # Transform shapes based on the current window offset
                    win_transform = src.window_transform(window)
                    shapes_gen = shapes(mask_chunk, mask=valid_mask, transform=win_transform)  

                    for geom, _ in shapes_gen:
                        geometries.append(shape(geom))
                        
            crs = src.crs

        logger.info(f" -> Extracted {len(geometries)} geometries. Building GeoDataFrame...")
        
        # Geopandas union_all() later in process() will cleanly stitch any polygons that got split by block boundaries
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
        combined_geometry = mask_gdf_df.union_all()
        mask_gdf_df = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf_df.crs)

        # FIX A: Use /vsis3/ for reading GPKG via GeoPandas/Fiona on AWS
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
            # FIX B: Ensure local directory exists before saving
            output_upath = UPath(output_path)
            output_upath.parent.mkdir(parents=True, exist_ok=True)
            intersecting_sub_grids.to_file(str(output_upath), driver="GPKG") 

        logger.info(f"[SUCCESS] Successfully saved {process_type} subgrids to: {output_path}")
        return