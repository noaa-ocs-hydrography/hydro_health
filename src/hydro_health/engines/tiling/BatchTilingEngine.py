"""Class engine that turns wide parquet files into batch/long format"""

import os
import gc
import re
import uuid
import shutil
import logging
import pathlib
import pandas as pd
import geopandas as gpd
import numpy as np
import s3fs

from pathlib import Path
from typing import Literal, List, Tuple, Optional
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


def _standardize_col_name(col: str) -> str:
    """Helper to ensure column names are standardized."""
    return str(col).strip()


def _save_parquet_file(df: pd.DataFrame, output_dir: str, file_name: str, is_aws: bool, local_tmp_dir: str, verbose_prefix: str, verbose: bool) -> None:
    """Save dataframe to local temporary disk, push to final destination, and immediately delete the temp file."""
    # Use a UUID to ensure multiple Dask workers don't collide when writing the temporary file
    unique_tmp_name = f"{uuid.uuid4().hex}_{file_name}"
    tmp_path = str(Path(local_tmp_dir) / unique_tmp_name)
    final_path = str(UPath(output_dir) / file_name)
    
    df.to_parquet(tmp_path, index=False, engine="pyarrow")
    
    if is_aws and final_path.startswith("s3://"):
        s3fs.S3FileSystem().put(tmp_path, final_path)
    else:
        shutil.copy(tmp_path, final_path)
        
    if verbose:
        Engine.write_message_dask(f"{verbose_prefix} [SUCCESS] Saved tile to: {final_path}", OUTPUTS)
        
    if Path(tmp_path).exists():
        os.remove(tmp_path)


def _process_training_tile(gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, year_ranges: list, is_aws: bool, local_tmp_dir: str, current_index: int, total_count: int, verbose: bool) -> Tuple[List[str], str]:
    """Processes a training tile and writes out BOTH a wide format and batch format data files."""
    progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
    saved_files = []
    
    if not year_ranges:
        if verbose:
            Engine.write_message_dask(f"{progress_str} [WARNING] 'year_ranges' is empty. No pairs processed for {tile_name}.", OUTPUTS)
        return saved_files, "NO PARQUET FILES GENERATED"

    rename_dict_global = {}
    for c in gdf.columns:
        new_c = _standardize_col_name(c)
        if new_c != c:
            rename_dict_global[c] = new_c

    if rename_dict_global:
        gdf.rename(columns=rename_dict_global, inplace=True)

    # 1. WIDE FORMAT GENERATION
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
    
    # 2. BATCH FORMAT GENERATION 
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
        if delta_name in wide_gdf.columns: pair_df['delta_bathy'] = wide_gdf[delta_name]
            
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
        _save_parquet_file(pair_df, output_dir, out_name_batch, is_aws, local_tmp_dir, progress_str, verbose)
        saved_files.append(out_name_batch)
            
        if not cols_created_batch:
            cols_created_batch = pair_df.columns.tolist()
            
        del pair_df

    del wide_gdf
    
    summary = []
    if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

    return saved_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"


def _process_prediction_tile(gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str, year_ranges: list, is_aws: bool, local_tmp_dir: str, current_index: int, total_count: int, verbose: bool) -> Tuple[List[str], str]:
    """Processes a prediction tile and writes out BOTH a wide format and batch format data files."""
    progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
    saved_files = []

    rename_dict_global = {}
    for c in gdf.columns:
        new_c = _standardize_col_name(c)
        if new_c != c:
            rename_dict_global[c] = new_c

    if rename_dict_global:
        gdf.rename(columns=rename_dict_global, inplace=True)

    # STRICT PREDICTION COLUMN FILTERING
    id_cols = [c for c in ["X", "Y", "FID", "tile_id", "geometry"] if c in gdf.columns]
    bt_cols = [c for c in gdf.columns if c.startswith("bt.")]
    other_cols = [c for c in gdf.columns if re.search(r"\d{4}_\d{4}", c) or any(p in c.lower() for p in ["grain", "sed", "survey", "tsm", "hurr"])]
    
    valid_cols = id_cols + bt_cols + other_cols
    valid_cols = list(dict.fromkeys([c for c in valid_cols if c in gdf.columns]))
    gdf = gdf[valid_cols].copy()

    # 1. WIDE FORMAT GENERATION
    wide_gdf = gdf.copy()
    rename_dict_wide = {}
    if 'x' in wide_gdf.columns: rename_dict_wide['x'] = 'X'
    if 'y' in wide_gdf.columns: rename_dict_wide['y'] = 'Y'
    wide_gdf.rename(columns=rename_dict_wide, inplace=True)

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
            
    valid_pair_strs = [f"{y0}_{y1}" for y0, y1 in valid_pairs]
    cols_to_drop = []
    for c in wide_gdf.columns:
        m = re.search(r"(\d{4}_\d{4})$", c)
        if m and not c.startswith("delta_bathy_"):
            if m.group(1) not in valid_pair_strs:
                cols_to_drop.append(c)
    if cols_to_drop:
        wide_gdf.drop(columns=cols_to_drop, inplace=True)

    # 2. BATCH FORMAT GENERATION
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
        
        b_y1 = get_bt_col(y1_str)
        if b_y1: pair_df['bathy_t'] = wide_gdf[b_y1]
        
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
        if delta_name in wide_gdf.columns: pair_df['delta_bathy'] = wide_gdf[delta_name]
            
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
        _save_parquet_file(pair_df, output_dir, out_name_batch, is_aws, local_tmp_dir, progress_str, verbose)
        saved_files.append(out_name_batch)
            
        if not cols_created_batch:
            cols_created_batch = pair_df.columns.tolist()
            
        del pair_df

    del wide_gdf
    
    summary = []
    if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

    return saved_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"


def _transform_tile_task(params: list) -> str:
    """Dask Worker: Reads file -> Calls specific processor -> Cleans up temp -> Returns status. Designed for top-level pickling."""
    f_path, mode, year_ranges, output_dir, tile_name, is_aws, local_tmp_dir, current_index, total_count, verbose = params
    
    # Implicit skip logic based on output presence
    # Since we generate multiple pairs, check if any batch files exist for this tile
    try:
        existing_batches = list(UPath(output_dir).glob(f"{tile_name}_*_{mode}_batch.parquet"))
        if existing_batches:
            if verbose:
                Engine.write_message_dask(f" [SKIP] Tile already processed: {tile_name} ({mode}).", OUTPUTS)
            return f"Skipped: {tile_name}"
    except Exception:
        pass
        
    gdf = None
    try:
        try:
            gdf = gpd.read_parquet(f_path, engine="pyarrow")
        except Exception:
            df = pd.read_parquet(f_path, engine="pyarrow")
            geometry_col = 'geometry' if 'geometry' in df.columns else None
            gdf = gpd.GeoDataFrame(df, geometry=geometry_col)

        if mode == "training":
            saved, cols_str = _process_training_tile(gdf, output_dir, tile_name, year_ranges, is_aws, local_tmp_dir, current_index, total_count, verbose)
        else:
            saved, cols_str = _process_prediction_tile(gdf, output_dir, tile_name, year_ranges, is_aws, local_tmp_dir, current_index, total_count, verbose)
        
        # Clean up intermediate input file
        tmp_dst_path = str(f_path)
        is_batch = tmp_dst_path.endswith("_batch.parquet")
        
        if not is_batch:
            try:
                upath_obj = UPath(f_path)
                if upath_obj.exists():
                    upath_obj.unlink()
            except Exception as e:
                Engine.write_message_dask(f"WARNING: Could not cleanly unlink intermediate UPath {f_path}: {e}", OUTPUTS)
                
            if Path(tmp_dst_path).exists():
                try:
                    os.remove(tmp_dst_path)
                except Exception as e:
                    Engine.write_message_dask(f"WARNING: Failed to explicitly delete temp file {tmp_dst_path}: {e}", OUTPUTS)

        return f"Success: {tile_name} (Generated: {len(saved)} files)\n   -> {cols_str}"

    except Exception as e:
        Engine.write_message_dask(f"ERROR: Failed transforming {os.path.basename(f_path)}: {str(e)}", OUTPUTS)
        return f"Failed: {os.path.basename(f_path)} - {str(e)}"
        
    finally:
        if gdf is not None:
            del gdf
        gc.collect()


class BatchTilingEngine(Engine):
    """Class for transforming wide parquet files in batch/long format"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False, year_ranges: Optional[List[Tuple[int, int]]] = None) -> None:
        """Initialize the BatchTilingEngine configurations and environment variables"""
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        
        # Flat variable assignment
        env_val = param_lookup.get('env', 'local')
        self.env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        self.is_aws = self.env in ['remote', 'aws']
        
        inherited_yr = getattr(self, 'year_ranges', [])
        yr_val = year_ranges if year_ranges is not None else param_lookup.get('year_ranges', inherited_yr)
        self.year_ranges = yr_val.value if hasattr(yr_val, 'value') else (yr_val if isinstance(yr_val, list) else inherited_yr)
        
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "batch_tiling_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        self.inputs_dir = INPUTS

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix and isinstance(self.output_prefix, str) else OUTPUTS / region
        self.write_message(f"BatchTilingEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        training_tiles_dir = get_config_item('MODEL', 'TRAINING_TILES_DIR')
        self.training_tiles_dir = UPath(f"{s3_dir_base}/{training_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / training_tiles_dir)

        prediction_tiles_dir = get_config_item('MODEL', 'PREDICTION_TILES_DIR')
        self.prediction_tiles_dir = UPath(f"{s3_dir_base}/{prediction_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / prediction_tiles_dir)

        if not self.is_aws:
            self.training_tiles_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_tiles_dir.mkdir(parents=True, exist_ok=True)

    def _process_pipeline(self, base_dir: UPath, mode: Literal["training", "prediction"], verbose_workers: bool = False) -> None:
        """Orchestrates the tile format transformation pipeline via Dask mapping."""
        self.write_message(f"--- Starting {mode.upper()} format pipeline ---", OUTPUTS)
        self.write_message(self.log_system_metrics(), OUTPUTS)
        
        if not self.year_ranges:
            self.write_message("CRITICAL WARNING: 'self.year_ranges' is empty or not defined. No files will be processed.", OUTPUTS)
            return

        base_dir_upath = UPath(base_dir)
        all_parquets = base_dir_upath.rglob("*.parquet")
        files_to_process = [
            fp for fp in all_parquets 
            if not fp.name.endswith("_batch.parquet")
        ]

        if not files_to_process:
            self.write_message(f"No files found for {mode} transformation in {base_dir}", OUTPUTS)
            return

        self.write_message(f"Queueing {len(files_to_process)} tiles for transformation...", OUTPUTS)

        params_list = []
        total_files = len(files_to_process)
        for i, fp in enumerate(files_to_process):
            # Extract the actual tile_name, preserving subtile indices like _1, _2
            filename = fp.name
            tile_name = filename.split(f"_{mode}")[0]
            
            # Map the output directly to the subtile directory where the input was located
            output_folder = str(fp.parent)
            
            params_list.append([
                str(fp),
                mode,
                self.year_ranges,
                output_folder,
                tile_name,
                self.is_aws,
                str(self.local_tmp_dir),
                i + 1,
                total_files,
                verbose_workers
            ])

        self.write_message(f"Submitting {total_files} task(s) to Dask client map...", OUTPUTS)
        futures = self.client.map(_transform_tile_task, params_list)
        results = self.client.gather(futures)

        success_count = sum(1 for r in results if r and r.startswith("Success"))
        failed_msgs = [r for r in results if r and r.startswith("Failed")]

        self.write_message(f"[TRANSFORMATION SUMMARY] Mode: {mode.upper()}", OUTPUTS)
        self.write_message(f" -> Total Attempted Tasks: {total_files}", OUTPUTS)
        self.write_message(f" -> Successful Tasks: {success_count}", OUTPUTS)
        self.write_message(f" -> Failed/Error Tasks: {len(failed_msgs)}", OUTPUTS)
            
        if failed_msgs:
            self.write_message("Transformation Errors:\n" + "\n".join(failed_msgs), OUTPUTS)
            
        self.write_message(self.log_system_metrics(), OUTPUTS)

    def run(self) -> None:
        """Main entry point for executing the batch format transformations"""
        try:
            self.setup_dask(self.env, n_workers=4, threads_per_worker=1, memory_limit="6GB")
            
            eco_val = self.param_lookup.get('eco_regions')
            eco_regions = eco_val.value if hasattr(eco_val, 'value') else eco_val
            if isinstance(eco_regions, str):
                eco_regions = [eco_regions.strip("[]'\" ")]
            
            for eco_region in eco_regions:
                self._resolve_paths(eco_region)
                
                self._process_pipeline(
                    base_dir=self.prediction_tiles_dir, 
                    mode="prediction",
                    verbose_workers=False
                )
                
                self._process_pipeline(
                    base_dir=self.training_tiles_dir, 
                    mode="training",
                    verbose_workers=False
                )
                
        finally:
            self.cleanup_resources(OUTPUTS)