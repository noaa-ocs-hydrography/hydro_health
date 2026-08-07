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
 
        # self.batch_format_transformation(base_dir=self.prediction_tiles_dir, mode="prediction")
        self.batch_format_transformation(base_dir=self.training_tiles_dir, mode="training")

    def parallel_processing_rasters(self, input_directory, mask_pred_bounds, mask_train_bounds, pred_cutline_path, train_cutline_path) -> None:
        """Process prediction and training rasters in parallel using Dask."""

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
        