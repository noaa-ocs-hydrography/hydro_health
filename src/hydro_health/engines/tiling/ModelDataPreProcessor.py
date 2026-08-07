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

    def process(self) -> None:
        """Main function to process model data."""   
 
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

    def parallel_processing_rasters(self, input_directory, mask_pred_bounds, mask_train_bounds, pred_cutline_path, train_cutline_path) -> None:
        """Process prediction and training rasters in parallel using Dask."""

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
        