"""Class engine that turns wide parquet files into batch/long format"""

import os
import gc
import re
import logging
import pathlib
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import dask
import dask.distributed
import s3fs

from pathlib import Path
from dask.distributed import as_completed, Client
from typing import Literal, List, Tuple, Optional
from upath import UPath

from hydro_health.helpers.tools import get_config_item, get_environment

logger = logging.getLogger(__name__)


class BatchTilingEngine:
    """Class for transforming wide parquet files in batch/long format"""

    # Class-level compiled regex patterns for static access across workers
    STATIC_PATTERNS = ['sed', 'tsm', 'hurr', 'grain', 'survey']
    RE_BT_PREFIX = re.compile(r"^bt\.")
    RE_FLOWDIR = re.compile(r"flowdir")

    def __init__(self, year_ranges: Optional[List[Tuple[int, int]]] = None, overwrite: bool = False, pilot_mode: bool = False):
        """Initialize the BatchTilingEngine"""
        
        super().__init__()
        self.pilot_mode = pilot_mode
        self.overwrite = overwrite
        self.year_ranges = year_ranges or []

        self.is_aws = (get_environment() == 'aws')

    def create_file_paths(self) -> None:
        """Creates unified UPath objects that work both locally and on S3"""
        
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

    def run(self) -> None:
        """Main entry point for executing the batch format transformations"""
        
        self.create_file_paths()
        self.batch_format_transformation(base_dir=self.prediction_tiles_dir, mode="prediction")
        self.batch_format_transformation(base_dir=self.training_tiles_dir, mode="training")

    def batch_format_transformation(self, base_dir: UPath, mode: Literal["training", "prediction"]) -> None:
        """Orchestrator for finalizing formatting on wide tiles"""
        
        logger.info(f"Starting Wide & Batch Format Transformation (Mode: {mode})...")

        logger.info(f"-> Validating 'year_ranges' config: {self.year_ranges}")
        if not self.year_ranges:
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
        try:
            client = dask.distributed.client.default_client()
        except ValueError:
            logger.info("No global Dask client found. Starting a LocalCluster...")
            client = Client()

        max_concurrent = 100 
        total_files = len(files_to_process)
        tasks_iterator = iter(enumerate(files_to_process))
        seq = as_completed()
        results = []
        
        def submit_format_task(item):
            i, fp = item
            return client.submit(
                BatchTilingEngine._transform_tile_task, 
                str(fp), 
                mode, 
                self.overwrite,
                self.year_ranges,
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
        logger.info(f" -> Total Attempted Tasks: {total_files}")
        logger.info(f" -> Successful Tasks: {success_count}")
        logger.info(f" -> Failed/Error Tasks: {len(failed_msgs)}")
        logger.info(f"--------------------------------------------------")
            
        if failed_msgs:
            logger.error("Transformation Errors:\n" + "\n".join(failed_msgs))

    @staticmethod
    def _trim_memory() -> None:
        """Helper to invoke garbage collector explicitly in workers"""
        gc.collect()

    @staticmethod
    def _standardize_col_name(col: str) -> str:
        """Helper to ensure column names are standardized."""
        return str(col).strip()

    @staticmethod
    def _extract_raster_to_df(raster_path: str, tile_extent: Tuple) -> pd.DataFrame:
        """Helper to read a window of a raster and convert to DataFrame"""
        
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

    @staticmethod
    def _transform_tile_task(f_path: str, mode: Literal["training", "prediction"], overwrite: bool, year_ranges: list, current_index: int = None, total_count: int = None) -> str:
        """Dask Worker: Reads file -> Calls specific processor -> Returns status"""
        
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
                saved, cols_str = BatchTilingEngine._process_and_save_training_tile(gdf, output_dir, tile_name, overwrite, year_ranges, current_index, total_count)
            else:
                saved, cols_str = BatchTilingEngine._process_and_save_prediction_tile(gdf, output_dir, tile_name, overwrite, year_ranges, current_index, total_count)
            
            # --- CLEAN UP INTERMEDIATE RAW WIDE FILE ---
            try:
                upath_obj = UPath(f_path)
                if upath_obj.exists():
                    upath_obj.unlink()
            except Exception as e:
                logger.warning(f"Could not delete intermediate file {f_path}: {e}")

            return f"Success: {tile_name} (Generated: {len(saved)} files)\n   -> {cols_str}"

        except Exception as e:
            return f"Failed: {os.path.basename(f_path)} - {str(e)}"
        finally:
            if gdf is not None:
                del gdf
            BatchTilingEngine._trim_memory()

    @staticmethod
    def _transform_flowdir_cols_inplace(df: pd.DataFrame) -> None:
        """Modifies DataFrame in-place to replace flow direction angles"""
        
        flow_cols = [c for c in df.columns if BatchTilingEngine.RE_FLOWDIR.search(c)]
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

    @staticmethod
    def _get_column_metadata(columns: List[str]) -> pd.DataFrame:
        """Efficiently parses column names to extract variables and years, handling _filled suffixes"""
        
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

    @staticmethod
    def _process_and_save_training_tile(gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, overwrite: bool, year_ranges: list, current_index: int = None, total_count: int = None) -> Tuple[List[str], str]:
        """Processes a training tile and writes out BOTH a wide format and batch format data files"""
        
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        saved_files = []
        
        if not year_ranges:
             logger.warning(f"{progress_str} [WARNING] 'year_ranges' is empty or missing! No pairs will be processed for {tile_name}.")

        rename_dict_global = {}
        for c in gdf.columns:
            new_c = BatchTilingEngine._standardize_col_name(c)
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
        
        if not overwrite and UPath(out_path_wide).exists():
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
            
            if not overwrite and UPath(out_path_batch).exists():
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

        del wide_gdf

        summary = []
        if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

        return saved_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"

    @staticmethod
    def _process_and_save_prediction_tile(gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, overwrite: bool, year_ranges: list, current_index: int = None, total_count: int = None) -> Tuple[List[str], str]:
        """Processes a prediction tile and writes out BOTH a wide format and batch format data files"""
        
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        saved_files = []

        rename_dict_global = {}
        for c in gdf.columns:
            new_c = BatchTilingEngine._standardize_col_name(c)
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
        
        if not overwrite and UPath(out_path_wide).exists():
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
            
            if not overwrite and UPath(out_path_batch).exists():
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

        del wide_gdf

        summary = []
        if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

        return saved_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"
    