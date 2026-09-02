"""Class engine for preparing and validating ML predictor data before training."""

import re
import os
import gc
import shutil
import tempfile
import pathlib
from pathlib import Path
from typing import Sequence, Iterable

import s3fs
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'

DYNAMIC_BASES = (
    "hurr_count",
    "hurr_strength",
    "tsm",
    "hurr_count_cumulative",
    "hurr_strength_cumulative",
    "tsm_cumulative",
)

NON_PREDICTORS = {
    "X", "Y", "FID", "tile_id", "source", "dataset_role",
    "year_t", "year_t1", "interval_years", "pair_id",
    "bathy_t1", "delta_bathy", "delta_rate", "survey_end_date",
    "geometry", "prediction_row_usable", "sample_weight"
}


def _pair_duration_years(pair: str) -> float:
    match = re.fullmatch(r"(\d{4})_(\d{4})", pair)
    if not match:
        raise ValueError(f"Invalid year-pair label: {pair}")
    y0, y1 = map(int, match.groups())
    if y1 <= y0:
        raise ValueError(f"Year pair must increase: {pair}")
    return float(y1 - y0)


def _safe_numeric_cols(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in set(columns).intersection(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _ensure_training_targets(df: pd.DataFrame, pair: str, tolerance: float = 1e-8) -> pd.DataFrame:
    out = df.copy()
    missing = {"bathy_t", "bathy_t1"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing required training fields: {sorted(missing)}")
    out = _safe_numeric_cols(
        out,
        ["year_t", "year_t1", "bathy_t", "bathy_t1", "delta_bathy", "delta_rate", "interval_years"],
    )
    if {"year_t", "year_t1"}.issubset(out.columns):
        out["interval_years"] = out["year_t1"] - out["year_t"]
    elif "interval_years" not in out.columns:
        out["interval_years"] = np.nan
        
    fallback = _pair_duration_years(pair)
    invalid_interval = ~np.isfinite(out["interval_years"]) | (out["interval_years"] <= 0)
    out.loc[invalid_interval, "interval_years"] = fallback
    
    recalculated = out["bathy_t1"] - out["bathy_t"]
    if "delta_bathy" in out:
        mismatch = (
            np.isfinite(out["delta_bathy"]) & np.isfinite(recalculated)
            & ((out["delta_bathy"] - recalculated).abs() > tolerance)
        )
        if mismatch.any():
            # In Dask workers, print to general worker logs or use Engine.write_message_dask
            pass 
            
    out["delta_bathy"] = recalculated
    out["delta_rate"] = out["delta_bathy"] / out["interval_years"]
    return out


def _build_pair_specific_predictors(master_predictors: Sequence[str], pair: str) -> list[str]:
    return [f"{p}_{pair}" if p in DYNAMIC_BASES else p for p in master_predictors]


def _resolve_predictors(
    pair: str,
    master_predictors: Sequence[str],
    train_columns: Sequence[str],
    prediction_columns: Sequence[str] | None,
    mode: str,
) -> tuple[list[str], list[str], list[str]]:
    candidates = list(dict.fromkeys(["bathy_t", *_build_pair_specific_predictors(master_predictors, pair)]))
    candidates = [c for c in candidates if c not in NON_PREDICTORS]
    
    in_training = [c for c in candidates if c in train_columns]
    missing_training = [c for c in candidates if c not in train_columns]
    missing_prediction: list[str] = []
    
    if prediction_columns is not None:
        missing_prediction = [c for c in in_training if c not in prediction_columns]
        
    if mode == "deployment_intersection" and prediction_columns is not None:
        resolved = [c for c in in_training if c in prediction_columns]
    else:
        resolved = in_training
        
    if "bathy_t" not in resolved:
        raise ValueError("bathy_t must exist in the resolved predictor schema")
    if mode == "og_training_first" and missing_prediction:
        raise ValueError(f"Training-first schema cannot deploy because prediction data are missing: {missing_prediction}")
        
    return resolved, missing_training, missing_prediction


def _calculate_sample_weights(
    change: np.ndarray,
    alpha: float,
    method: str,
    cap_quantile: float,
    max_weight: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    magnitude = np.abs(change.astype(float))
    if method == "power":
        weights = np.power(magnitude + epsilon, alpha)
        finite_sum = np.nansum(weights[np.isfinite(weights)])
        if not np.isfinite(finite_sum) or finite_sum <= 0:
            return np.ones_like(magnitude)
        weights = weights / finite_sum * len(weights)
    elif method == "capped_linear":
        cap = np.nanquantile(magnitude, cap_quantile)
        if not np.isfinite(cap) or cap <= 0:
            return np.ones_like(magnitude)
        scaled = np.minimum(magnitude, cap) / cap
        weights = 1.0 + (max_weight - 1.0) * np.power(scaled, alpha)
    else:
        raise ValueError(f"Unknown weight method: {method}")
        
    weights[~np.isfinite(weights)] = 1.0
    return weights


def _summarize_missingness(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "predictor": predictors,
        "n_rows": len(df),
        "n_na": [int(df[p].isna().sum()) for p in predictors],
        "proportion_na": [float(df[p].isna().mean()) for p in predictors],
        "n_finite": [int(np.isfinite(df[p].to_numeric(errors="coerce")).sum()) if p in df else 0 for p in predictors],
    })


def _predictor_range_table(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        if predictor not in df:
            continue
        values = pd.to_numeric(df[predictor], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            "predictor": predictor,
            "min_training": float(np.min(finite)) if finite.size else np.nan,
            "max_training": float(np.max(finite)) if finite.size else np.nan,
            "mean_training": float(np.mean(finite)) if finite.size else np.nan,
        })
    return pd.DataFrame(rows)


def _save_geoparquet(df: pd.DataFrame, final_save_path: str, local_tmp_dir: str, filename: str, is_aws: bool, crs: str = "EPSG:32617") -> None:
    """Helper to convert DF to GeoDataFrame, write to EC2 Temp Storage, and upload to S3."""
    out = df.copy()
    valid_xy = np.isfinite(pd.to_numeric(out['X'], errors="coerce")) & np.isfinite(pd.to_numeric(out['Y'], errors="coerce"))
    geometry = gpd.GeoSeries([Point(x, y) if ok else None for x, y, ok in zip(out['X'], out['Y'], valid_xy)], crs=crs)
    gdf = gpd.GeoDataFrame(out, geometry=geometry, crs=crs)
    
    tmp_path = str(Path(local_tmp_dir) / filename)
    gdf.to_parquet(tmp_path, index=False, compression="zstd")
    
    if is_aws and final_save_path.startswith("s3://"):
        s3fs.S3FileSystem().put(tmp_path, final_save_path)
    else:
        UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path, final_save_path)
        
    if Path(tmp_path).exists():
        os.remove(tmp_path)


def _save_parquet(df: pd.DataFrame, final_save_path: str, local_tmp_dir: str, filename: str, is_aws: bool) -> None:
    """Helper to write standard parquet to EC2 Temp Storage, and upload to S3."""
    tmp_path = str(Path(local_tmp_dir) / filename)
    df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
    
    if is_aws and final_save_path.startswith("s3://"):
        s3fs.S3FileSystem().put(tmp_path, final_save_path)
    else:
        UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path, final_save_path)
        
    if Path(tmp_path).exists():
        os.remove(tmp_path)


def _prep_tile_task(params: list) -> dict:
    """Core worker task for preparing a single ML tile. Designed for dask pickling."""
    
    tile_id, pair, raw_train_uri, raw_pred_uri, master_predictors, model_cfg, crs, is_aws, out_train_dir, out_pred_dir, local_tmp_dir, current_index, total_count, verbose = params

    if verbose:
        Engine.write_message_dask(f"Prepping ML data for tile {tile_id} - Pair {pair} ({current_index}/{total_count})...", OUTPUTS)

    try:
        # Load raw data
        training_raw = pd.read_parquet(raw_train_uri, storage_options={"anon": False} if is_aws else None)
        training_raw = _ensure_training_targets(training_raw, pair)
        training_raw["pair_id"] = pair
        training_raw["tile_id"] = tile_id
        
        prediction_raw = None
        if raw_pred_uri:
            prediction_raw = pd.read_parquet(raw_pred_uri, storage_options={"anon": False} if is_aws else None)
            prediction_raw["interval_years"] = _pair_duration_years(pair)
            prediction_raw["pair_id"] = pair
            prediction_raw["tile_id"] = tile_id

        # Resolve Predictor Schema
        predictors, missing_train, missing_pred = _resolve_predictors(
            pair, master_predictors, training_raw.columns,
            prediction_raw.columns if prediction_raw is not None else None,
            model_cfg.get("predictor_resolution_mode", "og_training_first")
        )
        
        # Safe Coercion
        numeric = set(predictors) | {"X", "Y", "FID", "bathy_t", "bathy_t1", "delta_bathy", "delta_rate", "interval_years"}
        training_raw = _safe_numeric_cols(training_raw, numeric)
        if prediction_raw is not None:
            prediction_raw = _safe_numeric_cols(prediction_raw, set(predictors) | {"X", "Y", "FID", "bathy_t", "interval_years"})

        # Summarize Missingness
        train_missing = _summarize_missingness(training_raw, predictors)
        _save_parquet(train_missing, str(UPath(out_train_dir) / f"predictor_missingness_training_{pair}.parquet"), local_tmp_dir, f"{tile_id}_train_missing_{pair}.parquet", is_aws)
        
        if prediction_raw is not None:
            pred_missing = _summarize_missingness(prediction_raw, predictors)
            _save_parquet(pred_missing, str(UPath(out_pred_dir) / f"predictor_missingness_prediction_{pair}.parquet"), local_tmp_dir, f"{tile_id}_pred_missing_{pair}.parquet", is_aws)

        # Essential Rows Filter
        essential = (
            np.isfinite(training_raw["X"]) & np.isfinite(training_raw["Y"])
            & np.isfinite(training_raw["bathy_t"]) & np.isfinite(training_raw["bathy_t1"])
            & np.isfinite(training_raw["delta_bathy"]) & np.isfinite(training_raw["interval_years"])
            & (training_raw["interval_years"] > 0)
        )
        
        if model_cfg.get("pre_xgb_data_mode") == "legacy_complete":
            predictor_complete = np.isfinite(training_raw[predictors].to_numpy(dtype=float)).all(axis=1)
            model_keep = essential & predictor_complete
        else:
            model_keep = essential
            
        subgrid_data = training_raw.loc[model_keep].copy().reset_index(drop=True)
        
        if len(subgrid_data) < model_cfg.get("minimum_training_rows", 100):
            if verbose:
                Engine.write_message_dask(f" [SKIP] Tile '{tile_id}': Insufficient valid training rows ({len(subgrid_data)}).", OUTPUTS)
            return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": "insufficient_training_rows", "n": len(subgrid_data)}

        # Sample Weights
        weights = np.ones(len(subgrid_data))
        if model_cfg.get("use_weighted_loss", True):
            weights = _calculate_sample_weights(
                subgrid_data["delta_bathy"].to_numpy(dtype=float),
                model_cfg.get("weight_alpha", 1.0),
                model_cfg.get("weight_method", "power"),
                model_cfg.get("weight_cap_quantile", 0.95),
                model_cfg.get("max_weight", 2.0)
            )
        subgrid_data["sample_weight"] = weights

        # Predictor Ranges
        ranges = _predictor_range_table(subgrid_data, predictors)
        _save_parquet(ranges, str(UPath(out_train_dir) / f"predictor_ranges_training_{pair}.parquet"), local_tmp_dir, f"{tile_id}_ranges_{pair}.parquet", is_aws)

        # Final GeoParquet Exports
        _save_geoparquet(subgrid_data, str(UPath(out_train_dir) / f"ML_ready_training_{pair}.geoparquet"), local_tmp_dir, f"{tile_id}_ready_train_{pair}.geoparquet", is_aws, crs)
        
        if prediction_raw is not None:
            # Drop unneeded rows to save memory in prediction
            pred_essential = np.isfinite(prediction_raw["X"]) & np.isfinite(prediction_raw["Y"]) & np.isfinite(prediction_raw["bathy_t"])
            prediction_ready = prediction_raw.loc[pred_essential].copy().reset_index(drop=True)
            _save_geoparquet(prediction_ready, str(UPath(out_pred_dir) / f"ML_ready_prediction_{pair}.geoparquet"), local_tmp_dir, f"{tile_id}_ready_pred_{pair}.geoparquet", is_aws, crs)

        if verbose:
            Engine.write_message_dask(f" [{current_index}/{total_count}] [SUCCESS] Tile '{tile_id}' data prepped for ML.", OUTPUTS)
            
        return {"ok": True, "tile_id": tile_id, "pair": pair, "n_fit": len(subgrid_data), "schema_locked": True}

    except Exception as e:
        Engine.write_message_dask(f"ERROR: DataPrep failed for tile {tile_id} - pair {pair}: {e}", OUTPUTS)
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": str(e)}
    finally:
        gc.collect()


class DataPrepEngine(Engine):
    """Class for validating and preparing GeoParquet tables before Machine Learning."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix

        # --- HARDCODED MASTER PREDICTORS ---
        # Edit this list to match the exact contents of what used to be in master_predictors.txt
        self.master_predictors = [
            "bathy_t",
            "hurr_count",
            "hurr_strength",
            "tsm",
            "hurr_count_cumulative",
            "hurr_strength_cumulative",
            "tsm_cumulative",
            # Add your other features here! (e.g., "slope", "current_speed", etc.)
        ]

        # EC2 Temp Storage configuration mapping
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "data_prep_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']
        self.inputs_dir = INPUTS

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments."""
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f"DataPrepEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        # Output Directories from SubgridTilingEngine
        training_tiles_dir = get_config_item('MODEL', 'TRAINING_TILES_DIR')
        self.training_tiles_dir = UPath(f"{s3_dir_base}/{training_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / training_tiles_dir)

        prediction_tiles_dir = get_config_item('MODEL', 'PREDICTION_TILES_DIR')
        self.prediction_tiles_dir = UPath(f"{s3_dir_base}/{prediction_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / prediction_tiles_dir)

        # ML Prep Final Directories
        ml_prep_train_dir = get_config_item('MODEL', 'ML_PREP_TRAIN_DIR')
        self.ml_prep_train_dir = UPath(f"{s3_dir_base}/{ml_prep_train_dir}") if self.is_aws else UPath(self.outputs_dir / ml_prep_train_dir)
        
        ml_prep_pred_dir = get_config_item('MODEL', 'ML_PREP_PRED_DIR')
        self.ml_prep_pred_dir = UPath(f"{s3_dir_base}/{ml_prep_pred_dir}") if self.is_aws else UPath(self.outputs_dir / ml_prep_pred_dir)

    def run(self) -> None:
        """Main execution method pulling config rules and distributing Prep tasks."""
        env = self.param_lookup.get('env', 'local')
        model_cfg = self.param_lookup.get('model_config', {})
        crs = model_cfg.get('crs', "EPSG:32617")
        year_pairs = model_cfg.get('year_pairs', [])
        verbose_workers = model_cfg.get('verbose_logging', False)

        try:
            self.setup_dask(env, n_workers=4, threads_per_worker=1, memory_limit="6GB")
            
            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)
                
                # We now pull directly from the hardcoded list set in __init__
                master_predictors = self.master_predictors
                
                # Fetch available tiles from training directory
                if self.is_aws:
                    fs = s3fs.S3FileSystem(anon=False)
                    train_tile_dirs = [d.split('/')[-1] for d in fs.ls(str(self.training_tiles_dir)) if fs.isdir(d)]
                else:
                    train_tile_dirs = [d.name for d in self.training_tiles_dir.iterdir() if d.is_dir()]

                params_list = []
                total_tasks = len(train_tile_dirs) * len(year_pairs)
                idx = 1
                
                for tile_id in train_tile_dirs:
                    for pair in year_pairs:
                        raw_train_uri = str(self.training_tiles_dir / tile_id / f"{tile_id}_training_clipped_data.parquet")
                        raw_pred_uri = str(self.prediction_tiles_dir / tile_id / f"{tile_id}_prediction_clipped_data.parquet")
                        
                        # Validate raw files exist before queuing
                        fs_check = s3fs.S3FileSystem(anon=False) if self.is_aws else None
                        train_exists = fs_check.exists(raw_train_uri) if self.is_aws else Path(raw_train_uri).exists()
                        pred_exists = fs_check.exists(raw_pred_uri) if self.is_aws else Path(raw_pred_uri).exists()
                        
                        if train_exists:
                            out_train = str(self.ml_prep_train_dir / tile_id)
                            out_pred = str(self.ml_prep_pred_dir / tile_id)
                            
                            params_list.append([
                                tile_id, pair, raw_train_uri, raw_pred_uri if pred_exists else None, 
                                master_predictors, model_cfg, crs, self.is_aws, 
                                out_train, out_pred, str(self.local_tmp_dir), 
                                idx, total_tasks, verbose_workers
                            ])
                            idx += 1

                self.write_message(f"Submitting {len(params_list)} ML Data Prep tasks to Dask client map...", OUTPUTS)
                futures = self.client.map(_prep_tile_task, params_list)
                results = self.client.gather(futures)

                # Process results summary
                valid_results = [r for r in results if r.get('ok')]
                self.write_message(f"Successfully prepared {len(valid_results)} / {len(params_list)} tile-pairs.", OUTPUTS)

        finally:
            self.cleanup_resources(OUTPUTS)