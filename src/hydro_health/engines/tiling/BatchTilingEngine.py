"""Class engine that turns wide parquet files into batch/long format"""

import os
import re
import uuid
import shutil
import logging
import pathlib
import tempfile
from functools import lru_cache
import pandas as pd
import numpy as np
import s3fs
import pyarrow.parquet as pq

from pathlib import Path
from typing import Literal, List, Tuple, Optional
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)


INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


@lru_cache(maxsize=1)
def _get_s3_filesystem() -> s3fs.S3FileSystem:
    """Reuse one S3 client per worker process."""

    return s3fs.S3FileSystem()


def _standardize_col_name(col: str) -> str:
    """Helper to ensure column names are standardized."""

    return str(col).strip()


def _save_parquet_file(df: pd.DataFrame, output_dir: str, file_name: str, is_aws: bool, local_tmp_dir: str, verbose_prefix: str, verbose: bool) -> None:
    """Save through a temporary file and always remove temporary artifacts."""

    # Use a UUID to ensure multiple Dask workers don't collide when writing the temporary file
    unique_tmp_name = f"{uuid.uuid4().hex}_{file_name}"
    tmp_path = str(Path(local_tmp_dir) / unique_tmp_name)
    final_path = str(UPath(output_dir) / file_name)
    
    Path(local_tmp_dir).mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow")

        if is_aws and final_path.startswith("s3://"):
            _get_s3_filesystem().put(tmp_path, final_path)
        else:
            destination = Path(final_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            partial = destination.with_name(
                f".{destination.name}.{uuid.uuid4().hex}.partial"
            )
            try:
                shutil.copy2(tmp_path, partial)
                os.replace(partial, destination)
            finally:
                partial.unlink(missing_ok=True)

        Engine.write_message_dask(
            f"{verbose_prefix} [SUCCESS] Saved tile to: {final_path}", OUTPUTS
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _output_exists(output_dir: str, file_name: str) -> bool:
    """Return whether one local or S3 output already exists."""

    return UPath(output_dir).joinpath(file_name).exists()


def _deduplicate_pixels(df: pd.DataFrame) -> None:
    """Deduplicate using the smallest available stable pixel key."""

    if (
        "tile_id" in df.columns
        and "FID" in df.columns
        and df[["tile_id", "FID"]].notna().all(axis=1).all()
    ):
        subset = ["tile_id", "FID"]
    elif "FID" in df.columns and df["FID"].notna().all():
        subset = ["FID"]
    elif (
        "X" in df.columns
        and "Y" in df.columns
        and df[["X", "Y"]].notna().all(axis=1).all()
    ):
        subset = ["X", "Y"]
    else:
        subset = None
    df.drop_duplicates(subset=subset, inplace=True, ignore_index=True)


def _read_required_columns(f_path: str, mode: str, year_ranges: list) -> pd.DataFrame:
    """Decode only columns the batch transformation can actually use."""

    requested_years = {str(year) for pair in year_ranges for year in pair}
    end_years = {str(y1) for _, y1 in year_ranges}
    pair_names = {f"{y0}_{y1}" for y0, y1 in year_ranges}
    feature_tokens = (
        "bpi_broad", "bpi_fine", "curv_plan", "curv_profile",
        "curv_total", "flowacc", "flowdir", "gradmag", "rugosity",
        "shearproxy", "slope", "tci", "terrain_classification",
        "unc", "uncertainty",
    )
    shared_tokens = ("grain", "sed_size", "prim_sed", "sed_type", "survey")

    path = UPath(f_path)
    with path.open("rb") as source:
        schema_columns = pq.read_schema(source).names
        selected = []
        for original in schema_columns:
            column = _standardize_col_name(original)
            lower = column.lower()
            include = lower in {"x", "y", "fid", "tile_id"}

            if mode == "prediction":
                include = include or lower.startswith("bt.")
            else:
                bathy_match = re.fullmatch(
                    r"bathy_(\d{4})_filled", lower
                )
                include = include or bool(
                    bathy_match and bathy_match.group(1) in requested_years
                )
                include = include or (
                    any(lower.endswith(f"_{year}") for year in end_years)
                    and any(token in lower for token in feature_tokens)
                )

            include = include or any(token in lower for token in shared_tokens)
            include = include or (
                lower.startswith(("hurr_strength_mean_", "tsm_mean_"))
                and any(lower.endswith(pair) for pair in pair_names)
            )
            if include:
                selected.append(original)

        if not selected:
            raise ValueError("No usable columns were found in the Parquet schema")
        source.seek(0)
        return pd.read_parquet(source, engine="pyarrow", columns=selected)


def _process_training_tile(gdf: pd.DataFrame, output_dir: str, tile_name: str, year_ranges: list, 
                           is_aws: bool, local_tmp_dir: str, current_index: int, total_count: int, verbose: bool,
                           overwrite_files: bool = False,
                           ) -> Tuple[List[str], List[str], str]:
    """Processes a training tile and writes out BOTH a wide format and batch format data files."""

    progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
    saved_files = []
    existing_files = []
    
    if not year_ranges:
        if verbose:
            Engine.write_message_dask(f"{progress_str} [WARNING] 'year_ranges' is empty. No pairs processed for {tile_name}.", OUTPUTS)
        return saved_files, existing_files, "NO PARQUET FILES GENERATED"

    rename_dict_global = {}
    for c in gdf.columns:
        new_c = _standardize_col_name(c)
        if new_c != c:
            rename_dict_global[c] = new_c

    if rename_dict_global:
        gdf.rename(columns=rename_dict_global, inplace=True)

    # 1. WIDE FORMAT GENERATION
    # Work on the already-loaded tile rather than holding a second full copy.
    wide_gdf = gdf
    rename_dict_wide = {}
    if 'x' in wide_gdf.columns: rename_dict_wide['x'] = 'X'
    if 'y' in wide_gdf.columns: rename_dict_wide['y'] = 'Y'
    wide_gdf.rename(columns=rename_dict_wide, inplace=True)

    def get_bathy_col(year_str):
        pattern = re.compile(rf"^bathy_{year_str}_filled$", re.IGNORECASE)
        cols = [c for c in wide_gdf.columns if pattern.match(c)]
        return cols[0] if cols else None

    valid_pairs = list(year_ranges)

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
        out_name_batch = f"{tile_name}_{pair_name}_training_batch.parquet"
        if not overwrite_files and _output_exists(output_dir, out_name_batch):
            existing_files.append(out_name_batch)
            continue
        
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
                
        if b_y0 and b_y1:
            pair_df['delta_bathy'] = wide_gdf[b_y1] - wide_gdf[b_y0]
            
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
            'X', 'Y', 'FID', 'tile_id', 'year_t', 'year_ti', 
            'bathy_ti', 'bathy_t', 'bpi_broad_t', 'bpi_fine_t', 
            'curv_plan_t', 'curv_profile_t', 'curv_total_t', 'flowacc_t', 
            'flowdir_cos_t', 'flowdir_sin_t', 'gradmag_t', 'rugosity_t', 
            'shearproxy_t', 'slope_t', 'slope_deg_t', 'tci_t', 
            'terrain_classification_t', 'delta_bathy', 
            f'hurr_strength_mean_{y0_str}_{y1_str}', f'tsm_mean_{y0_str}_{y1_str}', 
            'grain_size_layer', 'prim_sed_layer', 'survey_end_date'
        ]

        missing_cols = [c for c in ordered_cols if c not in pair_df.columns]
        if missing_cols:
            Engine.write_message_dask(
                f"{progress_str} [WARNING] Training batch {tile_name} "
                f"{pair_name} is missing columns: {missing_cols}. "
                "The batch file will be written without them.",
                OUTPUTS,
            )
        final_cols = [c for c in ordered_cols if c in pair_df.columns]
        pair_df = pair_df[final_cols]
        _deduplicate_pixels(pair_df)
        
        _save_parquet_file(pair_df, output_dir, out_name_batch, is_aws, local_tmp_dir, progress_str, verbose)
        saved_files.append(out_name_batch)
            
        if not cols_created_batch:
            cols_created_batch = pair_df.columns.tolist()
            
        del pair_df

    del wide_gdf
    
    summary = []
    if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

    if existing_files:
        summary.append(f"EXISTING FILES: {len(existing_files)}")

    return saved_files, existing_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"


def _process_prediction_tile(gdf: pd.DataFrame, output_dir: str, tile_name: str, year_ranges: list, 
                             is_aws: bool, local_tmp_dir: str, current_index: int, total_count: int, verbose: bool,
                             overwrite_files: bool = False,
                             ) -> Tuple[List[str], List[str], str]:
    """Processes a prediction tile and writes out BOTH a wide format and batch format data files."""

    progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
    saved_files = []
    existing_files = []

    rename_dict_global = {}
    for c in gdf.columns:
        new_c = _standardize_col_name(c)
        if new_c != c:
            rename_dict_global[c] = new_c

    if rename_dict_global:
        gdf.rename(columns=rename_dict_global, inplace=True)

    # STRICT PREDICTION COLUMN FILTERING
    id_cols = [c for c in ["X", "Y", "x", "y", "FID", "tile_id"] if c in gdf.columns]
    bt_cols = [c for c in gdf.columns if c.lower().startswith("bt.")]
    other_cols = [c for c in gdf.columns if re.search(r"\d{4}_\d{4}", c) or any(p in c.lower() for p in ["grain", "sed", "survey", "tsm", "hurr"])]
    
    valid_cols = id_cols + bt_cols + other_cols
    valid_cols = list(dict.fromkeys([c for c in valid_cols if c in gdf.columns]))
    # _read_required_columns already performed this filtering without geometry.

    # 1. WIDE FORMAT GENERATION
    # Work on the filtered tile rather than holding another full copy.
    wide_gdf = gdf
    rename_dict_wide = {}
    if 'x' in wide_gdf.columns: rename_dict_wide['x'] = 'X'
    if 'y' in wide_gdf.columns: rename_dict_wide['y'] = 'Y'
    wide_gdf.rename(columns=rename_dict_wide, inplace=True)

    filled_cols = [c for c in wide_gdf.columns if "_filled" in c and c not in other_cols]
    if filled_cols:
        wide_gdf.rename(columns={c: c.replace("_filled", "") for c in filled_cols}, inplace=True)

    def get_bt_col(year_str):
        pattern = re.compile(rf"^bt\.(?:bluetopo_)?{year_str}$", re.IGNORECASE)
        cols = [c for c in wide_gdf.columns if pattern.match(c)]
        return cols[0] if cols else None

    valid_pairs = list(year_ranges)

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
        out_name_batch = f"{tile_name}_{pair_name}_prediction_batch.parquet"
        if not overwrite_files and _output_exists(output_dir, out_name_batch):
            existing_files.append(out_name_batch)
            continue
        
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
            if c.endswith(f"_{y1_str}") and c != b_y1 and c.lower().startswith("bt."):
                base = re.sub(r"^bt\.", "", c, flags=re.IGNORECASE)
                base = base.replace(f"_{y1_str}", "").lower()
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

        missing_cols = [c for c in ordered_cols if c not in pair_df.columns]
        if missing_cols:
            Engine.write_message_dask(
                f"{progress_str} [WARNING] Prediction batch {tile_name} "
                f"{pair_name} is missing columns: {missing_cols}. "
                "The batch file will be written without them.",
                OUTPUTS,
            )
        final_cols = [c for c in ordered_cols if c in pair_df.columns]
        pair_df = pair_df[final_cols]
        _deduplicate_pixels(pair_df)
        
        _save_parquet_file(pair_df, output_dir, out_name_batch, is_aws, local_tmp_dir, progress_str, verbose)
        saved_files.append(out_name_batch)
            
        if not cols_created_batch:
            cols_created_batch = pair_df.columns.tolist()
            
        del pair_df

    del wide_gdf
    
    summary = []
    if cols_created_batch: summary.append(f"BATCH COLS: {cols_created_batch}")

    if existing_files:
        summary.append(f"EXISTING FILES: {len(existing_files)}")

    return saved_files, existing_files, "  ||  ".join(summary) if summary else "NO PARQUET FILES GENERATED"


def _transform_tile_task(params: list) -> str:
    """Dask Worker: Reads file -> Calls specific processor -> Cleans up temp -> Returns status. Designed for top-level pickling."""

    (
        f_path, mode, year_ranges, output_dir, tile_name, is_aws,
        local_tmp_dir, current_index, total_count, verbose, overwrite_files,
    ) = params
    
    try:
        # Geometry is not used in any output, so avoid GeoPandas/Shapely objects.
        gdf = _read_required_columns(f_path, mode, year_ranges)

        if mode == "training":
            saved, existing, cols_str = _process_training_tile(
                gdf, output_dir, tile_name, year_ranges, is_aws,
                local_tmp_dir, current_index, total_count, verbose,
                overwrite_files,
            )
        elif mode == "prediction":
            saved, existing, cols_str = _process_prediction_tile(
                gdf, output_dir, tile_name, year_ranges, is_aws,
                local_tmp_dir, current_index, total_count, verbose,
                overwrite_files,
            )
        else:
            raise ValueError(f"Unsupported transformation mode: {mode}")

        if saved:
            for s in saved:
                Engine.write_message_dask(f"   -> [OUTPUT VERIFIED] Batch file generated successfully at: {UPath(output_dir) / s}", OUTPUTS)

        if not saved and existing:
            return f"Skipped: {tile_name} ({len(existing)} valid output files already exist)"

        return (
            f"Success: {tile_name} (Generated: {len(saved)}; "
            f"already existed: {len(existing)}; directory: {output_dir})\n"
            f"   -> {cols_str}"
        )

    except Exception as e:
        message = (
            f"ERROR: Failed transforming {os.path.basename(f_path)}: "
            f"{type(e).__name__}: {e}"
        )
        Engine.write_message_dask(message, OUTPUTS)
        logger.exception(message)
        return f"Failed: {os.path.basename(f_path)} - {type(e).__name__}: {e}"


def _available_memory_bytes() -> int:
    """Return the smallest available host or container memory limit."""

    candidates = []
    try:
        candidates.append(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        )
    except (ValueError, OSError, AttributeError):
        pass

    for limit_path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            raw_value = limit_path.read_text(encoding="utf-8").strip()
            if raw_value != "max":
                value = int(raw_value)
                if 0 < value < 2**60:
                    candidates.append(value)
        except (OSError, ValueError):
            continue

    return min(candidates) if candidates else 16 * 1024**3


def _safe_worker_plan() -> Tuple[int, str]:
    """Keep aggregate worker limits near 65 percent of available RAM."""

    total_gib = _available_memory_bytes() / 1024**3
    worker_budget_gib = max(2.0, total_gib * 0.65)
    cpu_count = max(1, os.cpu_count() or 1)
    worker_count = min(4, cpu_count, max(1, int(worker_budget_gib // 4.0)))
    per_worker_gib = min(5.0, worker_budget_gib / worker_count)
    return worker_count, f"{per_worker_gib:.2f}GB"


class BatchTilingEngine(Engine):
    """Class for transforming wide parquet files in batch/long format"""

    def __init__(
        self,
        param_lookup: dict,
        output_prefix: str | bool = False,
        year_ranges: Optional[List[Tuple[int, int]]] = None,
        overwrite_files: bool = True,
    ) -> None:
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

        self.overwrite_files = overwrite_files
        
        self.local_tmp_dir = pathlib.Path(
            tempfile.gettempdir()
        ) / "hydro_health" / "batch_tiling_tmp" / uuid.uuid4().hex
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        self.inputs_dir = INPUTS
        self._worker_count = 1

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""

        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix and isinstance(self.output_prefix, str) else OUTPUTS / region
        self.write_message(f"BatchTilingEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
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
                verbose_workers,
                self.overwrite_files,
            ])

        batch_size = max(4, self._worker_count * 4)
        self.write_message(
            f"Submitting {total_files} task(s) in batches of at most "
            f"{batch_size}...",
            OUTPUTS,
        )
        results = []
        for start in range(0, total_files, batch_size):
            batch = params_list[start:start + batch_size]
            futures = self.client.map(_transform_tile_task, batch)
            results.extend(self.client.gather(futures))
            del futures

        success_count = sum(1 for r in results if r and r.startswith("Success"))
        skipped_count = sum(1 for r in results if r and r.startswith("Skipped"))
        failed_msgs = [r for r in results if r and r.startswith("Failed")]

        self.write_message(f"[TRANSFORMATION SUMMARY] Mode: {mode.upper()}", OUTPUTS)
        self.write_message(f" -> Total Attempted Tasks: {total_files}", OUTPUTS)
        self.write_message(f" -> Successful Tasks: {success_count}", OUTPUTS)
        self.write_message(f" -> Already Complete Tasks: {skipped_count}", OUTPUTS)
        self.write_message(f" -> Failed/Error Tasks: {len(failed_msgs)}", OUTPUTS)
            
        if failed_msgs:
            self.write_message("Transformation Errors:\n" + "\n".join(failed_msgs), OUTPUTS)
            
        self.write_message(self.log_system_metrics(), OUTPUTS)

    def run(self) -> None:
        """Main entry point for executing the batch format transformations"""

        try:
            self._worker_count, worker_memory = _safe_worker_plan()
            self.write_message(
                f"Starting Dask with {self._worker_count} worker(s), "
                f"1 thread per worker, and {worker_memory} per worker.",
                OUTPUTS,
            )
            self.write_message(
                f"Overwrite existing batch files: {self.overwrite_files}",
                OUTPUTS,
            )
            self.setup_dask(
                self.env,
                n_workers=self._worker_count,
                threads_per_worker=1,
                memory_limit=worker_memory,
            )
            
            eco_val = self.param_lookup.get('eco_regions')
            eco_regions = eco_val.value if hasattr(eco_val, 'value') else eco_val
            if isinstance(eco_regions, str):
                eco_regions = [eco_regions.strip("[]'\" ")]
            
            for eco_region in eco_regions:
                self._resolve_paths(eco_region)

                self._process_pipeline(
                    base_dir=self.training_tiles_dir, 
                    mode="training",
                    verbose_workers=False
                )
                
                
                # self._process_pipeline(
                #     base_dir=self.prediction_tiles_dir, 
                #     mode="prediction",
                #     verbose_workers=False
                # )
                

        finally:
            try:
                self.cleanup_resources(OUTPUTS)
            finally:
                shutil.rmtree(self.local_tmp_dir, ignore_errors=True)
