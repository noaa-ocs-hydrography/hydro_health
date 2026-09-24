"""Class engine for strictly handling XGBoost algorithm training and prediction."""

import gc
import json
import shutil
import pathlib
import re
from functools import lru_cache
from pathlib import Path

import s3fs
import numpy as np
import pandas as pd
import xgboost as xgb
from upath import UPath
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'

NON_PREDICTORS = {
    "X", "Y", "FID", "tile_id", "source", "dataset_role",
    "year_t", "year_ti", "interval_years", "pair_id",
    "bathy_t", "bathy_ti", "delta_bathy", "delta_rate",
    "survey_end_date", "geometry", "prediction_row_usable", "sample_weight",
}


@lru_cache(maxsize=1)
def _get_s3_filesystem() -> s3fs.S3FileSystem:
    """Reuse one S3 client within each worker process."""
    return s3fs.S3FileSystem(anon=False)

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
    if block_size_m <= 0:
        raise ValueError("block_size_m must be greater than zero")
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    gx = np.floor((x - np.nanmin(x)) / block_size_m).astype(np.int64)
    gy = np.floor((y - np.nanmin(y)) / block_size_m).astype(np.int64)
    block = pd.Series(gx).astype(str).str.cat(pd.Series(gy).astype(str), sep=":")
    
    unique_blocks = block.drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    
    mapping = {b: i % min(n_folds, len(unique_blocks)) for i, b in enumerate(unique_blocks)}
    return block.map(mapping).to_numpy(dtype=int)

def _choose_rounds_spatial_cv(
    coordinates: pd.DataFrame,
    features: pd.DataFrame,
    labels: np.ndarray,
    weights: np.ndarray,
    model_cfg: dict,
    seed: int,
) -> int:
    """Runs spatial CV to find the optimal number of boosting rounds."""
    if model_cfg.get("round_selection_mode") == "fixed":
        return model_cfg.get("fixed_nrounds", 1000)

    n_folds = model_cfg.get("n_folds", 5)
    folds = _spatial_fold_ids(
        coordinates,
        model_cfg.get("block_size_m", 200.0),
        n_folds,
        seed,
    )
    unique_folds = np.unique(folds)
    
    cv_fallback = model_cfg.get("cv_fallback_nrounds", 500)
    min_rounds = model_cfg.get("minimum_nrounds", 50)
    
    if unique_folds.size < 2:
        return max(min_rounds, cv_fallback)

    best_iters = []
    feature_names = features.columns.tolist()
    feature_values = features.to_numpy(dtype=np.float32, copy=False)

    for fold in unique_folds:
        test = folds == fold
        train = ~test
        if train.sum() < 20 or test.sum() < 10:
            continue
            
        dtrain = xgb.DMatrix(
            feature_values[train],
            label=labels[train],
            weight=weights[train],
            feature_names=feature_names,
            missing=np.nan,
        )
        dtest = xgb.DMatrix(
            feature_values[test],
            label=labels[test],
            feature_names=feature_names,
            missing=np.nan,
        )
        
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
    predicted_bathy: np.ndarray,
    baseline_bathy: np.ndarray,
    interval_years: np.ndarray,
    horizon: float,
) -> dict:
    """Calculates standardized elevation changes and rates."""
    delta = predicted_bathy - baseline_bathy
    rate = delta / interval_years
    standard_delta = rate * horizon
    return {
        "mean_predicted_bathy_next": predicted_bathy,
        "mean_predicted_change": delta,
        "mean_predicted_rate": rate,
        "mean_predicted_standard_change": standard_delta,
        "mean_predicted_standard_bathy": baseline_bathy + standard_delta,
    }


def _prepare_model_data(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, list[str], list[str]]:
    """Validate ML-ready files and build matching numeric feature frames."""
    required_train = {"X", "Y", "bathy_ti", "bathy_t", "interval_years"}
    missing_train = required_train - set(train_df.columns)
    if missing_train:
        raise ValueError(f"Missing required ML-ready training columns: {sorted(missing_train)}")

    train_df = train_df.copy()
    for column in required_train | {"sample_weight"}:
        if column in train_df:
            train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
    valid_train = (
        np.isfinite(train_df["X"])
        & np.isfinite(train_df["Y"])
        & np.isfinite(train_df["bathy_ti"])
        & np.isfinite(train_df["bathy_t"])
        & np.isfinite(train_df["interval_years"])
        & (train_df["interval_years"] > 0)
    )
    train_df = train_df.loc[valid_train].reset_index(drop=True)

    train_features = pd.DataFrame(index=train_df.index)
    train_features["bathy_baseline"] = train_df["bathy_ti"]
    for column in train_df.columns:
        if column not in NON_PREDICTORS:
            train_features[column] = pd.to_numeric(train_df[column], errors="coerce")

    pred_features = None
    missing_prediction: list[str] = []
    if pred_df is not None:
        required_pred = {"X", "Y", "bathy_t", "interval_years"}
        missing_pred = required_pred - set(pred_df.columns)
        if missing_pred:
            raise ValueError(f"Missing required ML-ready prediction columns: {sorted(missing_pred)}")

        pred_df = pred_df.copy()
        for column in required_pred:
            pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")
        valid_pred = (
            np.isfinite(pred_df["X"])
            & np.isfinite(pred_df["Y"])
            & np.isfinite(pred_df["bathy_t"])
            & np.isfinite(pred_df["interval_years"])
            & (pred_df["interval_years"] > 0)
        )
        pred_df = pred_df.loc[valid_pred].reset_index(drop=True)

        pred_features = pd.DataFrame(index=pred_df.index)
        pred_features["bathy_baseline"] = pred_df["bathy_t"]
        for column in pred_df.columns:
            if column not in NON_PREDICTORS:
                pred_features[column] = pd.to_numeric(pred_df[column], errors="coerce")

        missing_prediction = [
            column for column in train_features if column not in pred_features
        ]
        predictors = [column for column in train_features if column in pred_features]
    else:
        predictors = train_features.columns.tolist()

    predictors = [
        column
        for column in predictors
        if np.isfinite(train_features[column].to_numpy(dtype=float)).any()
    ]
    if "bathy_baseline" not in predictors:
        raise ValueError("No valid baseline bathymetry was available")

    train_features = train_features[predictors].astype(np.float32)
    if pred_features is not None:
        pred_features = pred_features[predictors].astype(np.float32)
    return train_df, pred_df, train_features, pred_features, predictors, missing_prediction


def _update_running_stats(
    mean: np.ndarray,
    squared_difference: np.ndarray,
    values: np.ndarray,
    count: int,
) -> None:
    """Update per-row statistics without storing every bootstrap result."""
    delta = values - mean
    mean += delta / count
    squared_difference += delta * (values - mean)

def _save_parquet(df: pd.DataFrame, final_save_path: str, local_tmp_dir: str, filename: str, is_aws: bool) -> None:
    """Helper to write standard parquet to EC2 Temp Storage, and upload to S3."""
    tmp_path = str(Path(local_tmp_dir) / filename)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
        if is_aws and final_save_path.startswith("s3://"):
            _get_s3_filesystem().put(tmp_path, final_save_path)
        else:
            UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path, final_save_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def _save_json(payload: dict, final_save_path: str, local_tmp_dir: str, filename: str, is_aws: bool) -> None:
    """Helper to write JSON metadata to EC2 Temp Storage, and upload to S3."""
    tmp_path = str(Path(local_tmp_dir) / filename)
    try:
        with open(tmp_path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
        if is_aws and final_save_path.startswith("s3://"):
            _get_s3_filesystem().put(tmp_path, final_save_path)
        else:
            UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path, final_save_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def _train_local_tile_task(params: list) -> dict:
    """Core worker task for training local XGBoost models on a tile. Picklable for Dask."""

    (
        tile_id, pair, train_uri, pred_uri, out_dir, model_cfg, is_aws,
        local_tmp_dir, current_index, total_count, verbose,
    ) = params

    if verbose:
        Engine.write_message_dask(f"Training XGBoost for tile {tile_id} - Pair {pair} ({current_index}/{total_count})...", OUTPUTS)

    try:
        train_df = pd.read_parquet(train_uri, storage_options={"anon": False} if is_aws else None)
        pred_df = None
        if pred_uri:
            pred_df = pd.read_parquet(pred_uri, storage_options={"anon": False} if is_aws else None)

        (
            train_df,
            pred_df,
            train_features,
            pred_features,
            predictors,
            missing_prediction,
        ) = _prepare_model_data(train_df, pred_df)

        minimum_rows = int(model_cfg.get("minimum_training_rows", 100))
        if len(train_df) < minimum_rows:
            return {
                "ok": False,
                "tile_id": tile_id,
                "pair": pair,
                "reason": "insufficient_training_rows",
                "n": len(train_df),
            }

        if missing_prediction and verbose:
            Engine.write_message_dask(
                f"Prediction data for {tile_id} {pair} are missing predictors "
                f"{missing_prediction}; using shared predictors only.",
                OUTPUTS,
            )

        seed = _stable_seed(model_cfg.get("global_seed", 12345), "local", tile_id, pair)
        weights = (
            train_df["sample_weight"].to_numpy(dtype=float)
            if "sample_weight" in train_df
            else np.ones(len(train_df), dtype=float)
        )
        weights[~np.isfinite(weights) | (weights < 0)] = 1.0
        labels = train_df["bathy_t"].to_numpy(dtype=np.float32)
        best_rounds = _choose_rounds_spatial_cv(
            train_df[["X", "Y"]],
            train_features,
            labels,
            weights,
            model_cfg,
            _stable_seed(seed, "cv"),
        )

        n_boot = int(model_cfg.get("n_boot", 5))
        if n_boot < 1:
            raise ValueError("n_boot must be at least 1")

        x_fit = train_features.to_numpy(dtype=np.float32, copy=False)
        d_full = xgb.DMatrix(x_fit, feature_names=predictors, missing=np.nan)
        d_pred = None
        if pred_features is not None and not pred_features.empty:
            d_pred = xgb.DMatrix(
                pred_features.to_numpy(dtype=np.float32, copy=False),
                feature_names=predictors,
                missing=np.nan,
            )

        train_mean = np.zeros(len(train_df), dtype=np.float64)
        train_m2 = np.zeros(len(train_df), dtype=np.float64)
        pred_mean = np.zeros(len(pred_df), dtype=np.float64) if d_pred is not None else None
        pred_m2 = np.zeros(len(pred_df), dtype=np.float64) if d_pred is not None else None
        
        rng = np.random.default_rng(seed)
        model_uris = []
        importance_rows = []
        shap_rows = []

        for bootstrap_index in range(1, n_boot + 1):
            indices = rng.choice(len(train_df), size=len(train_df), replace=True)
            dtrain = xgb.DMatrix(
                x_fit[indices], label=labels[indices], weight=weights[indices],
                feature_names=predictors, missing=np.nan,
            )
            model = xgb.train(_xgb_params(model_cfg, nthread=1), dtrain, num_boost_round=best_rounds, verbose_eval=False)

            train_prediction = model.predict(d_full).astype(np.float64, copy=False)
            _update_running_stats(
                train_mean, train_m2, train_prediction, bootstrap_index
            )
            if d_pred is not None and pred_mean is not None and pred_m2 is not None:
                prediction = model.predict(d_pred).astype(np.float64, copy=False)
                _update_running_stats(
                    pred_mean, pred_m2, prediction, bootstrap_index
                )

            score = model.get_score(importance_type="gain")
            importance_rows.extend(
                {
                    "predictor": predictor,
                    "gain": float(score.get(predictor, 0.0)),
                    "bootstrap_iteration": bootstrap_index,
                }
                for predictor in predictors
            )

            shap_n = min(model_cfg.get("shap_sample_rows", 1000), len(train_df))
            if shap_n > 0:
                shap_idx = rng.choice(len(train_df), size=shap_n, replace=False)
                dshap = xgb.DMatrix(x_fit[shap_idx], feature_names=predictors, missing=np.nan)
                contributions = model.predict(dshap, pred_contribs=True)[:, :-1]
                shap_rows.extend(
                    {
                        "predictor": predictor,
                        "mean_abs_shap": float(np.nanmean(np.abs(contributions[:, j]))),
                        "bootstrap_iteration": bootstrap_index,
                    }
                    for j, predictor in enumerate(predictors)
                )

            tmp_model_path = str(
                Path(local_tmp_dir)
                / f"{tile_id}_model_boot_{bootstrap_index:03d}_{pair}.json"
            )
            final_model_uri = str(
                UPath(out_dir)
                / "models"
                / f"model_boot_{bootstrap_index:03d}_{pair}.json"
            )
            try:
                model.save_model(tmp_model_path)
                if is_aws and final_model_uri.startswith("s3://"):
                    _get_s3_filesystem().put(tmp_model_path, final_model_uri)
                else:
                    UPath(final_model_uri).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(tmp_model_path, final_model_uri)
            finally:
                Path(tmp_model_path).unlink(missing_ok=True)

            model_uris.append(final_model_uri)

        train_summary = train_df.copy()
        train_sd = np.sqrt(train_m2 / (n_boot - 1)) if n_boot > 1 else np.zeros(len(train_df))
        translated_train = _translate_predictions(
            train_mean,
            train_summary["bathy_ti"].to_numpy(dtype=float),
            train_summary["interval_years"].to_numpy(dtype=float), model_cfg.get("standard_horizon_years", 2.0)
        )
        for column, values in translated_train.items():
            train_summary[column] = values
        train_summary["uncertainty_sd_bathy_next"] = train_sd
        train_summary["prediction_residual"] = (
            train_summary["bathy_t"] - train_summary["mean_predicted_bathy_next"]
        )

        _save_parquet(train_summary, str(UPath(out_dir) / f"LOCAL_training_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_train_sum_{pair}.parquet", is_aws)
        _save_parquet(pd.DataFrame(importance_rows), str(UPath(out_dir) / f"LOCAL_importance_{pair}.parquet"), local_tmp_dir, f"{tile_id}_imp_{pair}.parquet", is_aws)
        _save_parquet(pd.DataFrame(shap_rows), str(UPath(out_dir) / f"LOCAL_shap_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_shap_{pair}.parquet", is_aws)

        if pred_df is not None and pred_mean is not None and pred_m2 is not None:
            pred_summary = pred_df.copy()
            pred_sd = np.sqrt(pred_m2 / (n_boot - 1)) if n_boot > 1 else np.zeros(len(pred_df))
            translated_pred = _translate_predictions(
                pred_mean, pred_summary["bathy_t"].to_numpy(dtype=float),
                pred_summary["interval_years"].to_numpy(dtype=float), model_cfg.get("standard_horizon_years", 2.0)
            )
            for column, values in translated_pred.items():
                pred_summary[column] = values
            pred_summary["uncertainty_sd_bathy_next"] = pred_sd

            _save_parquet(pred_summary, str(UPath(out_dir) / f"LOCAL_prediction_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_pred_sum_{pair}.parquet", is_aws)

        meta = {
            "tile_id": tile_id,
            "pair": pair,
            "target": "bathy_t",
            "training_baseline_source": "bathy_ti",
            "prediction_baseline_source": "bathy_t",
            "model_baseline_feature": "bathy_baseline",
            "predictors": predictors,
            "missing_prediction_predictors": missing_prediction,
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
        
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
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
        verbose_workers = model_cfg.get('verbose_logging', False)
        worker_count = int(model_cfg.get('xgb_workers', 2))
        worker_memory = str(model_cfg.get('xgb_memory_limit', '4GB'))

        try:
            self.setup_dask(
                env,
                n_workers=worker_count,
                threads_per_worker=1,
                memory_limit=worker_memory,
            )
            
            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)
                
                params_list = []
                training_files = sorted(
                    self.ml_prep_train_dir.rglob("ML_ready_training_*.geoparquet"),
                    key=str,
                )
                total_tasks = len(training_files)

                for idx, training_path in enumerate(training_files, start=1):
                    match = re.fullmatch(
                        r"ML_ready_training_(?P<pair>\d{4}_\d{4})\.geoparquet",
                        training_path.name,
                    )
                    if not match:
                        self.write_message(
                            f"Skipping unrecognized ML-ready filename: {training_path.name}",
                            OUTPUTS,
                        )
                        continue

                    tile_id = training_path.parent.name
                    pair = match.group("pair")
                    prediction_path = (
                        self.ml_prep_pred_dir
                        / tile_id
                        / f"ML_ready_prediction_{pair}.geoparquet"
                    )
                    params_list.append([
                        tile_id,
                        pair,
                        str(training_path),
                        str(prediction_path) if prediction_path.exists() else None,
                        str(self.xgb_out_dir / tile_id),
                        model_cfg,
                        self.is_aws,
                        str(self.local_tmp_dir),
                        idx,
                        total_tasks,
                        verbose_workers,
                    ])

                task_batch_size = max(
                    1,
                    int(model_cfg.get('xgb_task_batch_size', worker_count * 2)),
                )
                self.write_message(
                    f"Submitting {len(params_list)} XGBoost tasks in batches of at most "
                    f"{task_batch_size}...",
                    OUTPUTS,
                )
                results = []
                for start in range(0, len(params_list), task_batch_size):
                    task_batch = params_list[start:start + task_batch_size]
                    futures = self.client.map(_train_local_tile_task, task_batch)
                    results.extend(self.client.gather(futures))
                    del futures
                    gc.collect()

                valid_results = [r for r in results if r.get('ok')]
                self.write_message(f"Successfully trained {len(valid_results)} / {len(params_list)} XGBoost ensembles.", OUTPUTS)

        finally:
            self.cleanup_resources(OUTPUTS)
