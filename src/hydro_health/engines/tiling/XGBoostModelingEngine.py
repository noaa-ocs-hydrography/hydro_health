"""Class engine for strictly handling XGBoost algorithm training and prediction."""

import os
import gc
import json
import shutil
import pathlib
import tempfile
from pathlib import Path
from typing import Any, Sequence, Dict

import s3fs
import numpy as np
import pandas as pd
import xgboost as xgb
from upath import UPath
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'

def _stable_seed(global_seed: int, *parts: object) -> int:
    """Helper to generate a stable, reproducible random seed based on inputs."""
    import hashlib
    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int((global_seed + int.from_bytes(digest, "little")) % (2**31 - 1))

def _xgb_params(model_cfg: dict, nthread: int = 1) -> dict:
    """Returns standardized XGBoost hyperparameters."""
    return {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "max_depth": model_cfg.get("max_depth", 4),
        "eta": model_cfg.get("eta", 0.01),
        "gamma": model_cfg.get("gamma", 0.5),
        "lambda": model_cfg.get("reg_lambda", 0.0),
        "alpha": model_cfg.get("reg_alpha", 0.0),
        "subsample": model_cfg.get("subsample", 0.7),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "nthread": nthread,
        "tree_method": "hist",
    }

def _spatial_fold_ids(df: pd.DataFrame, block_size_m: float, n_folds: int, seed: int) -> np.ndarray:
    """Creates spatially separated cross-validation folds based on X/Y coordinates."""
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    gx = np.floor((x - np.nanmin(x)) / block_size_m).astype(np.int64)
    gy = np.floor((y - np.nanmin(y)) / block_size_m).astype(np.int64)
    block = pd.Series(gx.astype(str) + ":" + gy.astype(str))
    
    unique_blocks = block.drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    
    mapping = {b: i % min(n_folds, len(unique_blocks)) for i, b in enumerate(unique_blocks)}
    return block.map(mapping).to_numpy(dtype=int)

def _choose_rounds_spatial_cv(
    df: pd.DataFrame, predictors: Sequence[str], weights: np.ndarray, model_cfg: dict, seed: int
) -> int:
    """Runs spatial CV to find the optimal number of boosting rounds."""
    if model_cfg.get("round_selection_mode") == "fixed":
        return model_cfg.get("fixed_nrounds", 1000)

    n_folds = model_cfg.get("n_folds", 5)
    folds = _spatial_fold_ids(df, model_cfg.get("block_size_m", 200.0), n_folds, seed)
    unique_folds = np.unique(folds)
    
    cv_fallback = model_cfg.get("cv_fallback_nrounds", 1000)
    min_rounds = model_cfg.get("minimum_nrounds", 1000)
    
    if unique_folds.size < 2:
        return max(min_rounds, cv_fallback)

    best_iters = []
    features = df[list(predictors)].to_numpy(dtype=np.float32)
    labels = df["bathy_t1"].to_numpy(dtype=np.float32)

    for fold in unique_folds:
        test = folds == fold
        train = ~test
        if train.sum() < 20 or test.sum() < 10:
            continue
            
        dtrain = xgb.DMatrix(features[train], label=labels[train], weight=weights[train], missing=np.nan)
        dtest = xgb.DMatrix(features[test], label=labels[test], missing=np.nan)
        
        model = xgb.train(
            _xgb_params(model_cfg, nthread=1), dtrain,
            num_boost_round=model_cfg.get("cv_nrounds", 2000),
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=model_cfg.get("early_stopping_rounds", 15),
            verbose_eval=False,
        )
        best_iteration = int(getattr(model, "best_iteration", -1)) + 1
        if best_iteration <= 0:
            best_iteration = cv_fallback
        best_iters.append(best_iteration)
        
    if not best_iters:
        return max(min_rounds, cv_fallback)

    selected = int(round(float(np.mean(best_iters))))
    return max(min_rounds, selected)

def _translate_predictions(
    pred_bathy_t1: np.ndarray, bathy_t: np.ndarray, interval_years: np.ndarray, horizon: float
) -> dict:
    """Calculates standardized elevation changes and rates."""
    delta = pred_bathy_t1 - bathy_t
    rate = delta / interval_years
    standard_delta = rate * horizon
    return {
        "mean_predicted_bathy_t1": pred_bathy_t1,
        "mean_predicted_change": delta,
        "mean_predicted_rate": rate,
        "mean_predicted_standard_change": standard_delta,
        "mean_predicted_standard_bathy": bathy_t + standard_delta,
    }

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

def _save_json(payload: dict, final_save_path: str, local_tmp_dir: str, filename: str, is_aws: bool) -> None:
    """Helper to write JSON metadata to EC2 Temp Storage, and upload to S3."""
    tmp_path = str(Path(local_tmp_dir) / filename)
    with open(tmp_path, "wt") as stream:
        json.dump(payload, stream, indent=2, default=str)
        
    if is_aws and final_save_path.startswith("s3://"):
        s3fs.S3FileSystem().put(tmp_path, final_save_path)
    else:
        UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path, final_save_path)
        
    if Path(tmp_path).exists():
        os.remove(tmp_path)

def _train_local_tile_task(params: list) -> dict:
    """Core worker task for training local XGBoost models on a tile. Picklable for Dask."""
    
    tile_id, pair, train_uri, pred_uri, out_dir, model_cfg, is_aws, local_tmp_dir, current_index, total_count, verbose = params

    if verbose:
        Engine.write_message_dask(f"Training XGBoost for tile {tile_id} - Pair {pair} ({current_index}/{total_count})...", OUTPUTS)

    try:
        # Load ML Ready data (prepped by DataPrepEngine)
        train_df = pd.read_parquet(train_uri, storage_options={"anon": False} if is_aws else None)
        pred_df = None
        if pred_uri:
            pred_df = pd.read_parquet(pred_uri, storage_options={"anon": False} if is_aws else None)

        # Extract locked predictors (excluding known non-predictor columns)
        non_predictors = {"X", "Y", "FID", "tile_id", "source", "dataset_role", "year_t", "year_t1", 
                          "interval_years", "pair_id", "bathy_t1", "delta_bathy", "delta_rate", 
                          "survey_end_date", "geometry", "prediction_row_usable", "sample_weight"}
        predictors = [c for c in train_df.columns if c not in non_predictors]
        
        # Spatial CV to find optimal boosting rounds
        seed = _stable_seed(model_cfg.get("global_seed", 12345), "local", tile_id, pair)
        weights = train_df["sample_weight"].to_numpy() if "sample_weight" in train_df else np.ones(len(train_df))
        
        best_rounds = _choose_rounds_spatial_cv(train_df, predictors, weights, model_cfg, _stable_seed(seed, "cv"))

        n_boot = model_cfg.get("n_boot", 5)
        x_fit = train_df[predictors].to_numpy(dtype=np.float32)
        y_fit = train_df["bathy_t1"].to_numpy(dtype=np.float32)
        
        boot_train = np.full((len(train_df), n_boot), np.nan, dtype=np.float32)
        boot_pred = np.full((len(pred_df), n_boot), np.nan, dtype=np.float32) if pred_df is not None else None
        
        d_full = xgb.DMatrix(x_fit, feature_names=predictors, missing=np.nan)
        d_pred = xgb.DMatrix(pred_df[predictors].to_numpy(dtype=np.float32), feature_names=predictors, missing=np.nan) if pred_df is not None else None
        
        rng = np.random.default_rng(seed)
        model_uris = []
        importance_rows = []
        shap_rows = []

        for b in range(n_boot):
            indices = rng.choice(len(train_df), size=len(train_df), replace=True)
            dtrain = xgb.DMatrix(
                x_fit[indices], label=y_fit[indices], weight=weights[indices],
                feature_names=predictors, missing=np.nan,
            )
            
            # Train model using 1 thread per worker to avoid CPU oversubscription
            model = xgb.train(_xgb_params(model_cfg, nthread=1), dtrain, num_boost_round=best_rounds, verbose_eval=False)
            
            # Make Predictions
            boot_train[:, b] = model.predict(d_full)
            if d_pred is not None and boot_pred is not None:
                boot_pred[:, b] = model.predict(d_pred)
                
            # Feature Importance
            score = model.get_score(importance_type="gain")
            importance_rows.extend({"predictor": p, "gain": float(score.get(p, 0.0)), "bootstrap_iteration": b + 1} for p in predictors)
            
            # SHAP Values (Sampled)
            shap_n = min(model_cfg.get("shap_sample_rows", 1000), len(train_df))
            shap_idx = rng.choice(len(train_df), size=shap_n, replace=False)
            dshap = xgb.DMatrix(x_fit[shap_idx], feature_names=predictors, missing=np.nan)
            contributions = model.predict(dshap, pred_contribs=True)[:, :-1]
            shap_rows.extend({"predictor": p, "mean_abs_shap": float(np.nanmean(np.abs(contributions[:, j]))), "bootstrap_iteration": b + 1} for j, p in enumerate(predictors))
            
            # Save Model (EC2 -> S3)
            tmp_model_path = str(Path(local_tmp_dir) / f"model_boot_{b + 1:03d}_{pair}.json")
            model.save_model(tmp_model_path)
            
            final_model_uri = str(UPath(out_dir) / "models" / f"model_boot_{b + 1:03d}_{pair}.json")
            if is_aws and final_model_uri.startswith("s3://"):
                s3fs.S3FileSystem().put(tmp_model_path, final_model_uri)
            else:
                UPath(final_model_uri).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(tmp_model_path, final_model_uri)
                
            model_uris.append(final_model_uri)
            if Path(tmp_model_path).exists():
                os.remove(tmp_model_path)

        train_summary = train_df.copy()
        train_mean = np.nanmean(boot_train, axis=1)
        train_sd = np.nanstd(boot_train, axis=1, ddof=1 if n_boot > 1 else 0)
        
        translated_train = _translate_predictions(
            train_mean, train_summary["bathy_t"].to_numpy(dtype=float), 
            train_summary["interval_years"].to_numpy(dtype=float), model_cfg.get("standard_horizon_years", 2.0)
        )
        for k, v in translated_train.items(): train_summary[k] = v
        train_summary["uncertainty_sd_bathy_t1"] = train_sd
        
        _save_parquet(train_summary, str(UPath(out_dir) / f"LOCAL_training_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_train_sum_{pair}.parquet", is_aws)
        _save_parquet(pd.DataFrame(importance_rows), str(UPath(out_dir) / f"LOCAL_importance_{pair}.parquet"), local_tmp_dir, f"{tile_id}_imp_{pair}.parquet", is_aws)
        _save_parquet(pd.DataFrame(shap_rows), str(UPath(out_dir) / f"LOCAL_shap_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_shap_{pair}.parquet", is_aws)

        # Prediction Summary
        if pred_df is not None and boot_pred is not None:
            pred_summary = pred_df.copy()
            pred_mean = np.nanmean(boot_pred, axis=1)
            pred_sd = np.nanstd(boot_pred, axis=1, ddof=1 if n_boot > 1 else 0)
            
            translated_pred = _translate_predictions(
                pred_mean, pred_summary["bathy_t"].to_numpy(dtype=float), 
                pred_summary["interval_years"].to_numpy(dtype=float), model_cfg.get("standard_horizon_years", 2.0)
            )
            for k, v in translated_pred.items(): pred_summary[k] = v
            pred_summary["uncertainty_sd_bathy_t1"] = pred_sd
            
            _save_parquet(pred_summary, str(UPath(out_dir) / f"LOCAL_prediction_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_pred_sum_{pair}.parquet", is_aws)

        # Metadata JSON
        meta = {
            "tile_id": tile_id,
            "pair": pair,
            "predictors": predictors,
            "best_iteration": best_rounds,
            "model_uris": model_uris
        }
        _save_json(meta, str(UPath(out_dir) / f"LOCAL_model_metadata_{pair}.json"), local_tmp_dir, f"{tile_id}_meta_{pair}.json", is_aws)

        if verbose:
            Engine.write_message_dask(f" [{current_index}/{total_count}] [SUCCESS] Tile '{tile_id}' XGBoost training complete.", OUTPUTS)
            
        return {"ok": True, "tile_id": tile_id, "pair": pair, "n_fit": len(train_df), "best_iteration": best_rounds}

    except Exception as e:
        Engine.write_message_dask(f"ERROR: XGBoost failed for tile {tile_id} - pair {pair}: {e}", OUTPUTS)
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": str(e)}
    finally:
        gc.collect()

class XGBoostModelingEngine(Engine):
    """Class for training and predicting native local XGBoost models in parallel."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix

        # EC2 Temp Storage mapping
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "xgb_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for ML ready data and Output directories."""
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        
        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        # ML Prep Inputs (from DataPrepEngine)
        ml_prep_train_dir = get_config_item('MODEL', 'ML_PREP_TRAIN_DIR')
        self.ml_prep_train_dir = UPath(f"{s3_dir_base}/{ml_prep_train_dir}") if self.is_aws else UPath(self.outputs_dir / ml_prep_train_dir)
        
        ml_prep_pred_dir = get_config_item('MODEL', 'ML_PREP_PRED_DIR')
        self.ml_prep_pred_dir = UPath(f"{s3_dir_base}/{ml_prep_pred_dir}") if self.is_aws else UPath(self.outputs_dir / ml_prep_pred_dir)

        # XGBoost Output Directories
        xgb_output_dir = get_config_item('MODEL', 'XGB_OUTPUT_DIR')
        self.xgb_out_dir = UPath(f"{s3_dir_base}/{xgb_output_dir}") if self.is_aws else UPath(self.outputs_dir / xgb_output_dir)

    def run(self) -> None:
        """Main execution method pulling config rules and distributing XGBoost tasks."""
        env = self.param_lookup.get('env', 'local')
        model_cfg = self.param_lookup.get('model_config', {})
        year_pairs = model_cfg.get('year_pairs', [])
        verbose_workers = model_cfg.get('verbose_logging', False)

        try:
            # XGBoost needs strictly 1 thread per worker in Dask to avoid thrashing
            self.setup_dask(env, n_workers=4, threads_per_worker=1, memory_limit="8GB")
            
            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)
                
                if self.is_aws:
                    fs = s3fs.S3FileSystem(anon=False)
                    train_tile_dirs = [d.split('/')[-1] for d in fs.ls(str(self.ml_prep_train_dir)) if fs.isdir(d)]
                else:
                    train_tile_dirs = [d.name for d in self.ml_prep_train_dir.iterdir() if d.is_dir()]

                params_list = []
                total_tasks = len(train_tile_dirs) * len(year_pairs)
                idx = 1
                
                for tile_id in train_tile_dirs:
                    for pair in year_pairs:
                        train_uri = str(self.ml_prep_train_dir / tile_id / f"ML_ready_training_{pair}.geoparquet")
                        pred_uri = str(self.ml_prep_pred_dir / tile_id / f"ML_ready_prediction_{pair}.geoparquet")
                        out_dir = str(self.xgb_out_dir / tile_id)
                        
                        fs_check = s3fs.S3FileSystem(anon=False) if self.is_aws else None
                        train_exists = fs_check.exists(train_uri) if self.is_aws else Path(train_uri).exists()
                        pred_exists = fs_check.exists(pred_uri) if self.is_aws else Path(pred_uri).exists()
                        
                        if train_exists:
                            params_list.append([
                                tile_id, pair, train_uri, pred_uri if pred_exists else None, 
                                out_dir, model_cfg, self.is_aws, str(self.local_tmp_dir), 
                                idx, total_tasks, verbose_workers
                            ])
                            idx += 1

                self.write_message(f"Submitting {len(params_list)} XGBoost modeling tasks to Dask...", OUTPUTS)
                futures = self.client.map(_train_local_tile_task, params_list)
                results = self.client.gather(futures)

                valid_results = [r for r in results if r.get('ok')]
                self.write_message(f"Successfully trained {len(valid_results)} / {len(params_list)} XGBoost ensembles.", OUTPUTS)

        finally:
            self.cleanup_resources(OUTPUTS)