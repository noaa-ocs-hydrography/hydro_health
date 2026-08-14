"""Class engine that turns wide parquet files into batch/long format"""

import os
import gc
import re
import logging
import pathlib
import pandas as pd
import geopandas as gpd
import numpy as np
import dask.distributed

from pathlib import Path
from dask.distributed import as_completed, Client
from typing import Literal, List, Tuple, Optional
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)


class BatchTilingEngine(Engine):
    """Class for transforming wide parquet files in batch/long format"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False, year_ranges: Optional[List[Tuple[int, int]]] = None) -> None:
        """Initialize the BatchTilingEngine"""
        super().__init__()
        self.param_lookup = param_lookup
        
        # Helper to safely extract values whether they are plain types or Param objects
        def _get_val(key, default=None):
            val = self.param_lookup.get(key, default)
            if hasattr(val, 'valueAsText') and val.valueAsText is not None:
                return val.valueAsText
            if hasattr(val, 'value') and val.value is not None:
                return val.value
            return val
            
        env = _get_val('env', 'local')
        self.is_aws = env in ['remote', 'aws']
        self.overwrite = _get_val('overwrite', False)
        
        # Pull year ranges from arguments, fallback to param lookup, or fallback to the inherited Engine value!
        inherited_yr = getattr(self, 'year_ranges', [])
        yr_val = year_ranges if year_ranges is not None else _get_val('year_ranges', inherited_yr)
        self.year_ranges = yr_val if isinstance(yr_val, list) else inherited_yr

        logger.info(f"Environment detected: {'AWS/Remote' if self.is_aws else 'Local'}")

        # ---------------------------------------------------------
        # Dynamically determine Repo Root and base folders
        # __file__ = src/hydro_health/engines/tiling/BatchTilingEngine.py
        # parents[4] points to the hydro_health/ repository root
        # ---------------------------------------------------------
        self.repo_root = pathlib.Path(__file__).resolve().parents[4]
        
        in_dir = _get_val('input_directory')
        out_dir = _get_val('output_directory')
        
        # Use param_lookup paths if provided (and not empty strings), else default to repo root
        base_in_dir = pathlib.Path(in_dir) if in_dir else self.repo_root / 'inputs'
        base_out_dir = pathlib.Path(out_dir) if out_dir else self.repo_root / 'outputs'
        
        # Mimic RasterMaskEngine logic: append output_prefix to output folder if it exists
        self.inputs_dir = base_in_dir
        self.outputs_dir = base_out_dir / output_prefix if output_prefix and isinstance(output_prefix, str) else base_out_dir
        
        # Dynamically determine the ecoregion (e.g., 'ER_3') to append to output paths
        eco_val = _get_val('eco_regions')
        self.ecoregion = ''
        
        if isinstance(eco_val, list) and eco_val:
            self.ecoregion = eco_val[0]
        elif isinstance(eco_val, str) and eco_val:
            import ast
            try:
                parsed = ast.literal_eval(eco_val)
                if isinstance(parsed, list) and parsed:
                    self.ecoregion = parsed[0]
                else:
                    self.ecoregion = eco_val.strip("[]'\" ")
            except Exception:
                self.ecoregion = eco_val.strip("[]'\" ")

        # If it wasn't explicitly provided in param_lookup, scan the directory
        if not self.ecoregion:
            er_dirs = [d.name for d in self.outputs_dir.glob('ER_*') if d.is_dir()]
            if er_dirs:
                self.ecoregion = er_dirs[0]

        # Finally, append the ecoregion into the path!
        if self.ecoregion:
            self.outputs_dir = self.outputs_dir / self.ecoregion

        def _resolve_path(config_value: str, is_output: bool = False) -> UPath:
            """Helper to safely build paths for AWS or Local execution"""
            if self.is_aws:
                bucket = get_config_item('S3', 'BUCKET_NAME')
                return UPath(f"s3://{bucket}/{config_value}")
            else:
                p = pathlib.Path(config_value)
                # 1. If the config provides an absolute path, just use it
                if p.is_absolute():
                    return UPath(p)
                # 2. If the config value already starts with 'inputs' or 'outputs', 
                #    strip the first folder and attach it directly to the correctly built 
                #    inputs_dir or outputs_dir so that ecoregion prefixes are respected!
                if p.parts:
                    if p.parts[0] == 'inputs':
                        return UPath(self.inputs_dir / pathlib.Path(*p.parts[1:]))
                    elif p.parts[0] == 'outputs':
                        return UPath(self.outputs_dir / pathlib.Path(*p.parts[1:]))
                # 3. Otherwise, map flat filenames/paths to the correct base directory
                base_dir = self.outputs_dir if is_output else self.inputs_dir
                return UPath(base_dir / p)

        # Apply the path resolver to the attributes actually needed by this engine
        self.training_tiles_dir = _resolve_path(get_config_item('MODEL', 'TRAINING_TILES_DIR'), is_output=True)
        self.prediction_tiles_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_TILES_DIR'), is_output=True)
        
        self.local_tmp_dir = pathlib.Path(_get_val('local_tmp_dir', str(Path.home() / "hydro_health_local_tmp")))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the local output directories exist before workers try to write to them
        if not self.is_aws:
            pathlib.Path(self.training_tiles_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.prediction_tiles_dir).mkdir(parents=True, exist_ok=True)

    def __getstate__(self):
        """
        Exclude unpicklable attributes (like Dask Client/Cluster and raw Params)
        when serializing this instance to send to Dask worker nodes.
        """
        state = self.__dict__.copy()
        
        # Drop the active connection handlers which crash the serializer
        state.pop('client', None)
        state.pop('cluster', None)
        
        # Drop raw parameter lookups in case they contain unpicklable COM objects
        state.pop('param_lookup', None)
        
        return state

    def run(self) -> None:
        """Main entry point for executing the batch format transformations"""
        
        # Use param_lookup and the base Engine class to initialize Dask
        env_val = self.param_lookup.get('env', 'local')
        env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        
        self.setup_dask(env)

        try:
            self.batch_format_transformation(base_dir=self.prediction_tiles_dir, mode="prediction")
            self.batch_format_transformation(base_dir=self.training_tiles_dir, mode="training")
        finally:
            try:
                self.close_dask()
            except Exception as e:
                logger.error(f"Could not cleanly close client/cluster: {e}")

    def batch_format_transformation(self, base_dir: UPath, mode: Literal["training", "prediction"]) -> None:
        """Orchestrator for finalizing formatting on wide tiles"""
        
        logger.info(f"Starting Wide & Batch Format Transformation (Mode: {mode})...")

        logger.info(f"-> Validating 'year_ranges' config: {self.year_ranges}")
        if not self.year_ranges:
            logger.error("!!! CRITICAL WARNING: 'self.year_ranges' is empty or not defined. No files will be processed !!!")

        base_dir_upath = UPath(base_dir)
        
        # Search all subdirectories for parquet files, ignoring the final batch files.
        # We allow processing of _formatted files in case the pipeline was interrupted and 
        # the raw wide files were already deleted.
        all_parquets = base_dir_upath.rglob("*.parquet")
        files_to_process = [
            fp for fp in all_parquets 
            if not fp.name.endswith("_batch.parquet")
        ]

        if not files_to_process:
            logger.warning(f"No files found for {mode} transformation in {base_dir}")
            return

        logger.info(f"Outputting transformed {mode} formatted tiles to: {base_dir}")
        logger.info(f"Queueing {len(files_to_process)} tiles...")

        # -------------------------------------------------------------
        # DYNAMIC DASK TASK STREAM (FORMAT TRANSFORMATION)
        # -------------------------------------------------------------
        try:
            client = getattr(self, 'client', None)
            if not client:
                client = Client()
        except ValueError:
            logger.info("No global Dask client found. Starting a LocalCluster...")
            client = Client()

        max_concurrent = 100 
        total_files = len(files_to_process)
        tasks_iterator = iter(enumerate(files_to_process))
        seq = as_completed()
        
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
            # We do not want to delete the file if it is already our formatted output!
            if not f_path.endswith("_formatted.parquet"):
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