"""Train the tested tile/year-pair XGBoost ensemble within the project repository.

Retain locked feature order, native predictor NAs, prepared sample weights,
spatial CV controls, bootstrap streams and saved output naming. Temporary local
IO reads the latest repository DataPrep outputs while S3 inputs are unavailable.
The real project Engine is used; no standalone support/config module is needed.
Search TEMPORARY REPOSITORY TEST before restoring the cloud production path.
"""
import gc
import tempfile
import traceback
from importlib.metadata import version
from importlib import import_module
import json
import hashlib
import shutil
import pathlib
import re
from functools import lru_cache
from pathlib import Path

# TEMPORARY REPOSITORY TEST: reinstate cloud imports/config lookup before production merge.
# import s3fs
# from hydro_health.helpers.tools import get_config_item
import numpy as np
import pandas as pd
import xgboost as xgb
import geopandas as gpd
from upath import UPath
from hydro_health.engines.Engine import Engine

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
INPUTS = REPOSITORY_ROOT / 'inputs'
OUTPUTS = REPOSITORY_ROOT / 'outputs'

NON_PREDICTORS = {
    "X", "Y", "FID", "tile_id", "source", "dataset_role",
    "year_t", "year_t1", "year_ti", "interval_years", "pair_id",
    "bathy_t", "bathy_t1", "bathy_ti", "delta_bathy", "delta_rate",
    "survey_end_date", "geometry", "prediction_row_usable", "sample_weight",
}


# @lru_cache(maxsize=1)
# def _get_s3_filesystem() -> s3fs.S3FileSystem:
#     """Reuse one S3 client within each worker process."""
#     return s3fs.S3FileSystem(anon=False)
# 

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
        "gamma": model_cfg.get("gamma", 1.0),
        "lambda": model_cfg.get("reg_lambda", 1.0),
        "alpha": model_cfg.get("reg_alpha", 1.0),
        "subsample": model_cfg.get("subsample", 0.7),
        "colsample_bytree": model_cfg.get("colsample_bytree", 0.8),
        "nthread": nthread,
        "tree_method": model_cfg.get("tree_method", "hist"),
        "max_bin": model_cfg.get("max_bin", 256),
        "seed": int(model_cfg.get("xgb_seed", 0)),
    }

def _spatial_fold_ids(df: pd.DataFrame, block_size_m: float, n_folds: int, seed: int) -> np.ndarray:
    """Assign occupied square blocks to folds so a block never spans train/test.

    Neighbouring blocks may be in different folds; no buffer or minimum separation
    is imposed. This retains the tested Python scheme, not R blockCV assignment.
    """
    if block_size_m <= 0:
        raise ValueError("block_size_m must be greater than zero")
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1")
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    gx = np.floor((x - np.nanmin(x)) / block_size_m).astype(np.int64)
    gy = np.floor((y - np.nanmin(y)) / block_size_m).astype(np.int64)
    block = pd.Series(gx).astype(str).str.cat(pd.Series(gy).astype(str), sep=":")
    
    unique_blocks = block.drop_duplicates().to_numpy(copy=True)  # Own writable storage for pandas 3 Copy-on-Write.
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)
    
    mapping = {b: i % min(n_folds, len(unique_blocks)) for i, b in enumerate(unique_blocks)}
    return block.map(mapping).to_numpy(dtype=int)

def _model_metrics(observed, predicted, baseline, interval) -> dict:
    """Compute unweighted state/change/rate metrics; apparent fitting metrics are not held-out accuracy."""
    obs = np.asarray(observed, dtype=float); pred = np.asarray(predicted, dtype=float)
    baseline = np.asarray(baseline, dtype=float); interval = np.asarray(interval, dtype=float)
    result = {}
    for name, left, right in (
        ("bathy_t1", obs, pred),
        ("change", obs - baseline, pred - baseline),
        ("rate", (obs - baseline) / interval, (pred - baseline) / interval),
    ):
        valid = np.isfinite(left) & np.isfinite(right)
        a = left[valid]; b = right[valid]; error = b - a
        result[f"n_{name}"] = int(len(a))
        result[f"rmse_{name}"] = float(np.sqrt(np.mean(error**2))) if len(a) else None
        result[f"mae_{name}"] = float(np.mean(np.abs(error))) if len(a) else None
        result[f"bias_{name}"] = float(np.mean(error)) if len(a) else None
        denominator = float(np.sum((a - np.mean(a))**2)) if len(a) else 0.0
        result[f"r2_predictive_{name}"] = float(1 - np.sum(error**2) / denominator) if denominator > 0 else None
        result[f"correlation_{name}"] = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0 else None
        if name == "change":
            result["sign_accuracy_change"] = float(np.mean(np.sign(a) == np.sign(b))) if len(a) else None
    return result


def _choose_rounds_spatial_cv(
    coordinates: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray,
    weights: np.ndarray, model_cfg: dict, seed: int, audit: dict | None = None,
) -> int:
    """Select rounds with the R local controls; save block folds, held-out predictions, and CV metrics.

    The supplied Python grid-fold scheme is retained. It is not R blockCV's
    optimized assignment, so its selected rounds need not equal the R run.
    """
    audit = audit if audit is not None else {}
    mode = model_cfg.get("round_selection_mode", "cv_with_floor")
    if mode not in ("fixed", "cv_with_floor", "cv_only"):
        raise ValueError(f"Unknown round selection mode: {mode}")
    audit.update(status="fixed" if mode == "fixed" else "not_run", mode=mode,
        folds=pd.DataFrame(columns=["row_index", "fold", "X", "Y"]),
        metrics=pd.DataFrame(columns=["fold", "status", "n_train", "n_test", "best_nrounds", "xgb_seed", "rmse_bathy_t1", "mae_bathy_t1"]),
        oof=pd.DataFrame(columns=["row_index", "fold", "observed_bathy_t1", "predicted_bathy_t1", "best_nrounds"]))
    if mode == "fixed":
        return int(model_cfg.get("fixed_nrounds", 1000))
    folds = _spatial_fold_ids(coordinates, model_cfg.get("block_size_m", 200.0), model_cfg.get("n_folds", 5), seed)
    audit["folds"] = pd.DataFrame({"row_index": np.arange(len(folds)), "fold": folds,
                                   "X": coordinates.X.to_numpy(), "Y": coordinates.Y.to_numpy()})
    fallback = int(model_cfg.get("cv_fallback_nrounds", 1000))
    floor = int(model_cfg.get("minimum_nrounds", 1000))
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        audit["status"] = "fewer_than_two_folds_using_fallback"
        return max(floor, fallback) if mode == "cv_with_floor" else fallback
    best_iters = []; rows = []; oof = []
    values = features.to_numpy(dtype=np.float32, copy=False)
    for fold in unique_folds:
        test = folds == fold; train = ~test
        if train.sum() < 20 or test.sum() < 10:
            rows.append({"fold": int(fold), "status": "insufficient_rows", "n_train": int(train.sum()), "n_test": int(test.sum())})
            continue
        dtrain = xgb.DMatrix(values[train], label=labels[train], weight=weights[train], feature_names=features.columns.tolist(), missing=np.nan)
        dtest = xgb.DMatrix(values[test], label=labels[test], feature_names=features.columns.tolist(), missing=np.nan)
        fold_cfg = dict(model_cfg, xgb_seed=_stable_seed(seed, "cv_model", int(fold)))
        model = xgb.train(_xgb_params(fold_cfg, nthread=1), dtrain,
            num_boost_round=int(model_cfg.get("cv_nrounds", 2000)), evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=int(model_cfg.get("early_stopping_rounds", 15)), verbose_eval=False)
        rounds = int(model.best_iteration) + 1
        prediction = model.predict(dtest, iteration_range=(0, rounds)).astype(float)
        error = prediction - labels[test]
        rows.append({"fold": int(fold), "status": "ok", "n_train": int(train.sum()), "n_test": int(test.sum()),
                     "best_nrounds": rounds, "xgb_seed": fold_cfg["xgb_seed"],
                     "rmse_bathy_t1": float(np.sqrt(np.mean(error**2))), "mae_bathy_t1": float(np.mean(np.abs(error)))})
        oof.append(pd.DataFrame({"row_index": np.flatnonzero(test), "fold": int(fold), "observed_bathy_t1": labels[test],
                                "predicted_bathy_t1": prediction, "best_nrounds": rounds}))
        best_iters.append(rounds)
    audit["metrics"] = pd.DataFrame(rows)
    audit["oof"] = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame(columns=["row_index", "fold", "observed_bathy_t1", "predicted_bathy_t1", "best_nrounds"])
    if not best_iters:
        audit["status"] = "no_valid_folds_using_fallback"
        return max(floor, fallback) if mode == "cv_with_floor" else fallback
    selected = int(round(float(np.mean(best_iters))))
    audit.update(status="cv_success", raw_cv_selected_rounds=selected, fold_best_rounds=best_iters)
    return max(floor if mode == "cv_with_floor" else 50, selected)


def _translate_predictions(
    predicted_bathy: np.ndarray,
    baseline_bathy: np.ndarray,
    interval_years: np.ndarray,
    horizon: float,
) -> dict:
    """Derive later-minus-earlier change, annualized rate and standardized horizon values."""
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
    locked_predictors: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, list[str], list[str]]:
    """Use DataPrep's ordered schema: bathy_t plus drivers predicts bathy_t1.

    Keep predictor NAs and all prediction rows. No silent feature inference,
    target leakage, schema intersection, all-NA removal, or baseline renaming.
    The prepared training table is already fitting-eligible, so invalid essential
    fields cause an error rather than silently changing the verified row set.
    """
    if not locked_predictors or "bathy_t" not in locked_predictors:
        raise ValueError("Supply DataPrep's locked_predictors; bathy_t must be included")
    predictors = list(locked_predictors)
    if len(predictors) != len(set(predictors)):
        raise ValueError("Duplicate names in locked predictor schema")
    forbidden = (set(predictors) & NON_PREDICTORS) - {"bathy_t"}
    if forbidden:
        raise ValueError(f"Target/metadata leakage in locked schema: {sorted(forbidden)}")
    required_train = {"X", "Y", "FID", "bathy_t1", "bathy_t", "interval_years", "sample_weight", "delta_bathy", "delta_rate"}
    if missing := (required_train | set(predictors)) - set(train_df.columns):
        raise ValueError(f"Missing required ML-ready training columns: {sorted(missing)}")
    train_df = train_df.copy()
    for column in required_train | set(predictors):
        train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
    essential = np.isfinite(train_df[["X", "Y", "bathy_t", "bathy_t1", "interval_years"]].to_numpy(dtype=float)).all(axis=1)
    essential &= train_df["interval_years"].to_numpy(dtype=float) > 0
    if not essential.all():
        raise ValueError("ML-ready training has invalid essential fields; rerun DataPrep instead of dropping rows")
    weights = train_df.sample_weight.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Prepared sample weights must be finite and positive")
    train_df = train_df.reset_index(drop=True)
    train_features = train_df[predictors].astype(np.float32)
    # NA is native missing data; infinity is not supported by XGBoost.
    if np.isinf(train_features.to_numpy()).any():
        raise ValueError("Predictors contain infinity; diagnose source data rather than impute it")
    pred_features = None
    if pred_df is not None:
        required_pred = {"X", "Y", "FID", "bathy_t", "interval_years"}
        if missing := (required_pred | set(predictors)) - set(pred_df.columns):
            raise ValueError(f"Training-first schema cannot deploy; missing prediction fields: {sorted(missing)}")
        pred_df = pred_df.copy().reset_index(drop=True)
        for column in required_pred | set(predictors):
            pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")
        usable = np.isfinite(pred_df[["X", "Y", "bathy_t", "interval_years"]].to_numpy(dtype=float)).all(axis=1)
        usable &= pred_df.interval_years.to_numpy(dtype=float) > 0
        if "prediction_row_usable" in pred_df:
            if pred_df.prediction_row_usable.isna().any():
                raise ValueError("prediction_row_usable contains NA")
            usable &= pred_df.prediction_row_usable.to_numpy(dtype=bool)
        pred_df["prediction_row_usable"] = usable
        pred_features = pred_df[predictors].astype(np.float32)
        if np.isinf(pred_features.loc[usable].to_numpy()).any():
            raise ValueError("Usable prediction rows contain infinite predictors")
    return train_df, pred_df, train_features, pred_features, predictors, []


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
    """Stage a diagnostic table and copy it locally; retain the disabled S3 upload hook."""
    tmp_path = str(Path(local_tmp_dir) / filename)
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
        # TEMPORARY REPOSITORY TEST: delete local copy and reinstate S3 before merge:
        # if is_aws and final_save_path.startswith("s3://"):
        #     _get_s3_filesystem().put(tmp_path, final_save_path)
        if is_aws or final_save_path.startswith("s3://"):
            raise ValueError("Cloud writes are disabled")
        UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path, final_save_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def _save_json(payload: dict, final_save_path: str, local_tmp_dir: str, filename: str, is_aws: bool) -> None:
    """Stage auditable model metadata and publish it through the temporary local path."""
    tmp_path = str(Path(local_tmp_dir) / filename)
    try:
        with open(tmp_path, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
        # TEMPORARY REPOSITORY TEST: delete local copy and reinstate S3 before merge:
        # if is_aws and final_save_path.startswith("s3://"):
        #     _get_s3_filesystem().put(tmp_path, final_save_path)
        if is_aws or final_save_path.startswith("s3://"):
            raise ValueError("Cloud writes are disabled")
        UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(tmp_path, final_save_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

def _save_geoparquet(df: pd.DataFrame, path: str, crs: str) -> None:
    """Save a companion spatial summary without discarding rows with invalid coordinates."""
    valid = np.isfinite(df.X) & np.isfinite(df.Y)
    geometry = gpd.points_from_xy(df.X.fillna(0), df.Y.fillna(0), crs=crs)
    geometry[~valid] = None
    gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs).to_parquet(path, index=False, compression="zstd", schema_version="1.0.0")


def _train_local_tile_task(params: list) -> dict:
    """Core worker task for training local XGBoost models on a tile. Picklable for Dask."""

    (
        tile_id, pair, train_uri, pred_uri, out_dir, model_cfg, is_aws,
        local_tmp_dir, current_index, total_count, verbose,
    ) = params

    worker_log_dir = Path(out_dir)
    Path(model_cfg["prediction_output_dir"]).mkdir(parents=True, exist_ok=True)

    if verbose:
        Engine.write_message_dask(f"Training XGBoost for tile {tile_id} - Pair {pair} ({current_index}/{total_count})...", worker_log_dir)

    try:
        # TEMPORARY REPOSITORY TEST: delete local load and reinstate S3 before production merge.
        # train_df = pd.read_parquet(train_uri, storage_options={"anon": False})
        train_df = pd.DataFrame(gpd.read_parquet(train_uri).drop(columns="geometry"))
        pred_df = None
        if pred_uri:
            # TEMPORARY REPOSITORY TEST: delete local load and reinstate S3 before production merge.
            # pred_df = pd.read_parquet(pred_uri, storage_options={"anon": False})
            pred_df = pd.DataFrame(gpd.read_parquet(pred_uri).drop(columns="geometry"))

        (
            train_df,
            pred_df,
            train_features,
            pred_features,
            predictors,
            missing_prediction,
        ) = _prepare_model_data(train_df, pred_df, model_cfg["locked_predictors"])

        minimum_rows = int(model_cfg.get("minimum_training_rows", 100))
        if len(train_df) < minimum_rows:
            return {
                "ok": False,
                "tile_id": tile_id,
                "pair": pair,
                "reason": "insufficient_training_rows",
                "n": len(train_df),
            }

        seed = _stable_seed(model_cfg.get("global_seed", 12345), "local", tile_id, pair)
        # DataPrep already verified these weights; carry them through unchanged.
        weights = train_df["sample_weight"].to_numpy(dtype=float, copy=True)
        labels = train_df["bathy_t1"].to_numpy(dtype=np.float32)
        cv_audit = {}
        best_rounds = _choose_rounds_spatial_cv(
            train_df[["X", "Y"]],
            train_features,
            labels,
            weights,
            model_cfg,
            _stable_seed(seed, "cv"), cv_audit,
        )
        for name in ("folds", "metrics", "oof"):
            table = cv_audit[name]
            if isinstance(table, pd.DataFrame):
                if "row_index" in table:
                    table["FID"] = train_df.FID.to_numpy()[table.row_index.to_numpy(dtype=int)]
                _save_parquet(table, str(UPath(out_dir) / f"LOCAL_spatial_cv_{name}_{pair}.parquet"), local_tmp_dir, f"{tile_id}_cv_{name}_{pair}.parquet", False)

        n_boot = int(model_cfg.get("n_boot", 5))
        if n_boot < 1:
            raise ValueError("n_boot must be at least 1")

        x_fit = train_features.to_numpy(dtype=np.float32, copy=False)
        d_full = xgb.DMatrix(x_fit, feature_names=predictors, missing=np.nan)
        d_pred = None
        pred_usable = pred_df.prediction_row_usable.to_numpy(dtype=bool) if pred_df is not None else None
        if pred_features is not None and pred_usable.any():
            d_pred = xgb.DMatrix(
                pred_features.loc[pred_usable].to_numpy(dtype=np.float32, copy=False),
                feature_names=predictors,
                missing=np.nan,
            )

        train_mean = np.zeros(len(train_df), dtype=np.float64)
        train_m2 = np.zeros(len(train_df), dtype=np.float64)
        pred_mean = np.where(pred_usable, 0.0, np.nan) if pred_df is not None else None
        pred_m2 = np.where(pred_usable, 0.0, np.nan) if pred_df is not None else None
        
        # Bootstrap and SHAP use separate streams so diagnostics cannot change the fitted ensemble.
        bootstrap_audit = []
        model_uris = []
        importance_rows = []
        shap_rows = []

        for bootstrap_index in range(1, n_boot + 1):
            bootstrap_seed = _stable_seed(seed, "bootstrap", bootstrap_index)
            model_seed = _stable_seed(seed, "xgb_model", bootstrap_index)
            rng = np.random.default_rng(bootstrap_seed)
            indices = rng.choice(len(train_df), size=len(train_df), replace=True)
            bootstrap_audit.append({"bootstrap_iteration": bootstrap_index, "bootstrap_seed": bootstrap_seed, "xgb_seed": model_seed})
            if model_cfg.get("save_bootstrap_indices", True):
                _save_parquet(pd.DataFrame({"draw_position": np.arange(len(indices)), "training_row_index": indices,
                    "FID": train_df.FID.to_numpy()[indices]}),
                    str(UPath(out_dir) / "bootstrap_indices" / f"bootstrap_{bootstrap_index:03d}_{pair}.parquet"),
                    local_tmp_dir, f"{tile_id}_indices_{bootstrap_index}_{pair}.parquet", False)
            if verbose:
                Engine.write_message_dask(f"Bootstrap {bootstrap_index}/{n_boot}: {best_rounds} rounds, {len(train_df)} fitting rows", worker_log_dir)
            dtrain = xgb.DMatrix(
                x_fit[indices], label=labels[indices], weight=weights[indices],
                feature_names=predictors, missing=np.nan,
            )
            model = xgb.train(_xgb_params(dict(model_cfg, xgb_seed=model_seed), nthread=1), dtrain, num_boost_round=best_rounds, verbose_eval=False)

            train_prediction = model.predict(d_full).astype(np.float64, copy=False)
            _update_running_stats(
                train_mean, train_m2, train_prediction, bootstrap_index
            )
            if d_pred is not None and pred_mean is not None and pred_m2 is not None:
                prediction = np.full(len(pred_df), np.nan)
                prediction[pred_usable] = model.predict(d_pred).astype(np.float64, copy=False)
                _update_running_stats(
                    pred_mean, pred_m2, prediction, bootstrap_index
                )

            score = model.get_score(importance_type="gain")
            total_gain = model.get_score(importance_type="total_gain")
            gain_sum = sum(total_gain.values())
            importance_rows.extend(
                {
                    "predictor": predictor,
                    "gain": float(score.get(predictor, 0.0)),
                    "total_gain": float(total_gain.get(predictor, 0.0)),
                    "gain_fraction": float(total_gain.get(predictor, 0.0) / gain_sum) if gain_sum else 0.0,
                    "bootstrap_iteration": bootstrap_index,
                }
                for predictor in predictors
            )

            shap_n = min(model_cfg.get("shap_sample_rows", 1000), len(train_df))
            if shap_n > 0:
                shap_rng = np.random.default_rng(_stable_seed(seed, "shap", bootstrap_index))
                shap_idx = shap_rng.choice(len(train_df), size=shap_n, replace=False)
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
                # TEMPORARY REPOSITORY TEST: delete local copy and reinstate S3 before merge:
                # if is_aws and final_model_uri.startswith("s3://"):
                #     _get_s3_filesystem().put(tmp_model_path, final_model_uri)
                UPath(final_model_uri).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp_model_path, final_model_uri)
            finally:
                Path(tmp_model_path).unlink(missing_ok=True)

            model_uris.append(final_model_uri)

        train_summary = train_df.copy()
        train_sd = np.sqrt(np.maximum(train_m2, 0) / (n_boot - 1)) if n_boot > 1 else np.zeros(len(train_df))
        translated_train = _translate_predictions(
            train_mean,
            train_summary["bathy_t"].to_numpy(dtype=float),
            train_summary["interval_years"].to_numpy(dtype=float), model_cfg.get("standard_horizon_years", 2.0)
        )
        for column, values in translated_train.items():
            train_summary[column] = values
        train_summary["uncertainty_sd_bathy_next"] = train_sd
        train_summary["mean_predicted_bathy_t1"] = train_mean
        train_summary["uncertainty_sd_bathy_t1"] = train_sd
        train_summary["uncertainty_sd_change"] = train_sd
        train_summary["uncertainty_sd_rate"] = train_sd / train_summary.interval_years
        train_summary["delta_residual"] = train_summary.delta_bathy - train_summary.mean_predicted_change
        train_summary["prediction_residual"] = (
            train_summary["bathy_t1"] - train_summary["mean_predicted_bathy_next"]
        )

        _save_parquet(train_summary, str(UPath(out_dir) / f"LOCAL_training_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_train_sum_{pair}.parquet", is_aws)
        _save_parquet(pd.DataFrame(importance_rows), str(UPath(out_dir) / f"LOCAL_importance_{pair}.parquet"), local_tmp_dir, f"{tile_id}_imp_{pair}.parquet", is_aws)
        _save_parquet(pd.DataFrame(shap_rows), str(UPath(out_dir) / f"LOCAL_shap_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_shap_{pair}.parquet", is_aws)

        if pred_df is not None and pred_mean is not None and pred_m2 is not None:
            pred_summary = pred_df.copy()
            pred_sd = np.sqrt(np.maximum(pred_m2, 0) / (n_boot - 1)) if n_boot > 1 else np.where(pred_usable, 0.0, np.nan)
            translated_pred = _translate_predictions(
                pred_mean, pred_summary["bathy_t"].to_numpy(dtype=float),
                pred_summary["interval_years"].to_numpy(dtype=float), model_cfg.get("standard_horizon_years", 2.0)
            )
            for column, values in translated_pred.items():
                pred_summary[column] = values
            pred_summary["uncertainty_sd_bathy_next"] = pred_sd
            pred_summary["mean_predicted_bathy_t1"] = pred_mean
            pred_summary["uncertainty_sd_bathy_t1"] = pred_sd
            pred_summary["uncertainty_sd_change"] = pred_sd
            pred_summary["uncertainty_sd_rate"] = pred_sd / pred_summary.interval_years

            _save_parquet(pred_summary, str(UPath(model_cfg["prediction_output_dir"]) / f"LOCAL_prediction_summary_{pair}.parquet"), local_tmp_dir, f"{tile_id}_pred_sum_{pair}.parquet", is_aws)

        _save_geoparquet(train_summary, str(UPath(out_dir) / f"LOCAL_training_summary_{pair}.geoparquet"), model_cfg["crs"])
        if pred_df is not None:
            _save_geoparquet(pred_summary, str(UPath(model_cfg["prediction_output_dir"]) / f"LOCAL_prediction_summary_{pair}.geoparquet"), model_cfg["crs"])
        _save_json(_model_metrics(train_summary.bathy_t1, train_mean, train_summary.bathy_t, train_summary.interval_years),
            str(UPath(out_dir) / f"LOCAL_apparent_training_metrics_{pair}.json"), local_tmp_dir, f"{tile_id}_metrics_{pair}.json", False)
        meta = {
            "tile_id": tile_id,
            "pair": pair,
            "target": "bathy_t1",
            "training_baseline_source": "bathy_t",
            "prediction_baseline_source": "bathy_t",
            "model_baseline_feature": "bathy_t",
            "predictors": predictors,
            "missing_prediction_predictors": missing_prediction,
            "best_iteration": best_rounds,
            "model_uris": model_uris,
            "crs": model_cfg["crs"], "n_boot": n_boot, "n_fit": len(train_df),
            "n_prediction_rows": len(pred_df) if pred_df is not None else 0,
            "n_prediction_usable": int(pred_usable.sum()) if pred_df is not None else 0,
            "bootstrap_seeds": bootstrap_audit, "global_seed": model_cfg.get("global_seed", 12345),
            "round_selection_mode": model_cfg["round_selection_mode"], "cv_status": cv_audit["status"],
            "cv_raw_selected_rounds": cv_audit.get("raw_cv_selected_rounds"),
            "cv_fold_best_rounds": cv_audit.get("fold_best_rounds", []),
            "cv_method": "Python randomized occupied square blocks; not R blockCV", "early_stopping_metric": "mae",
            "xgb_params": _xgb_params(model_cfg), "model_config": model_cfg,
            "xgboost_version": xgb.__version__,
            "versions": {p: version(p) for p in ("numpy", "pandas", "geopandas", "pyarrow", "dask", "distributed")},
            "scope": "local modeling and output-contract validation; R numeric parity requires separate reference outputs"
        }
        _save_json(meta, str(UPath(out_dir) / f"LOCAL_model_metadata_{pair}.json"), local_tmp_dir, f"{tile_id}_meta_{pair}.json", is_aws)

        if verbose:
            Engine.write_message_dask(f" [{current_index}/{total_count}] [SUCCESS] Tile '{tile_id}' XGBoost training complete.", worker_log_dir)
            
        return {"ok": True, "tile_id": tile_id, "pair": pair, "n_fit": len(train_df), "best_iteration": best_rounds}

    except Exception as e:
        Engine.write_message_dask(traceback.format_exc(), worker_log_dir)
        Engine.write_message_dask(f"ERROR: XGBoost failed for tile {tile_id} - pair {pair}: {e}", worker_log_dir)
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": str(e)}
    finally:
        gc.collect()


def _validate_settings(cfg):
    """Reject invalid model controls without changing scientific settings silently."""
    mc = cfg["model_config"]
    if mc["round_selection_mode"] not in ("cv_with_floor", "cv_only", "fixed"):
        raise ValueError("Invalid round_selection_mode")
    for key in ("n_boot", "minimum_training_rows", "minimum_nrounds", "fixed_nrounds", "cv_nrounds",
                "early_stopping_rounds", "cv_fallback_nrounds", "max_depth", "max_bin", "xgb_workers"):
        if int(mc[key]) != mc[key] or int(mc[key]) < 1:
            raise ValueError(f"{key} must be a positive integer")
    if int(mc["n_folds"]) != mc["n_folds"] or mc["n_folds"] < 2:
        raise ValueError("n_folds must be an integer >= 2")
    if mc["xgb_workers"] != 1:
        raise ValueError("Use one local worker for this initial single-tile test")
    for key in ("eta", "block_size_m", "standard_horizon_years"):
        if not np.isfinite(mc[key]) or mc[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("gamma", "reg_lambda", "reg_alpha"):
        if not np.isfinite(mc[key]) or mc[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    for key in ("subsample", "colsample_bytree"):
        if not 0 < mc[key] <= 1:
            raise ValueError(f"{key} must be in (0,1]")
    if int(mc["shap_sample_rows"]) != mc["shap_sample_rows"] or mc["shap_sample_rows"] < 0:
        raise ValueError("shap_sample_rows must be a nonnegative integer")
    if mc["tree_method"] not in ("hist", "exact", "approx"):
        raise ValueError("Unsupported tree_method")
    if int(cfg["verification_chunk_rows"]) < 1:
        raise ValueError("verification_chunk_rows must be positive")

    if int(cfg["verification_chunk_rows"]) != cfg["verification_chunk_rows"]:
        raise ValueError("verification_chunk_rows must be an integer")
    match = re.fullmatch(r"(\d{4})_(\d{4})", cfg["pair"])
    if not match or int(match[2]) <= int(match[1]):
        raise ValueError("pair must be an increasing YYYY_YYYY label")
    if cfg["crs"] != "EPSG:32617":
        raise ValueError("Repository modeling test requires EPSG:32617")


def _file_hash(path: str) -> str:
    """Fingerprint immutable DataPrep inputs to detect stale validation or mid-run changes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




class XGBoostModelingEngine(Engine):
    """Production-shaped model engine with explicit temporary repository-local IO."""

    def __init__(self, param_lookup: dict | None = None, output_prefix: str | bool = False) -> None:
        """Group model/test settings here and initialize the actual repository base class.

        The default consumes the just-validated DataPrep outputs, not the immutable
        input fixtures. Optional settings overrides are used by the self-test.
        """
        super().__init__()
        self.param_lookup = param_lookup or {}
        self.output_prefix = output_prefix
        self.repository_root = Path(self.param_lookup.get("repository_root", REPOSITORY_ROOT)).resolve()
        # --- perhaps move to config file? ---
        self.settings = {
            "tile_id": "BH4S556X_3",
            "pair": "2004_2006",
            "region": "ER_3",
            "crs": "EPSG:32617",
            "training_geoparquet": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}/ML_ready_training_{pair}.geoparquet",
            "prediction_geoparquet": "outputs/local_test_outputs/Prediction_data_grid_tiles/{tile_id}/ML_ready_prediction_{pair}.geoparquet",
            "locked_predictors_parquet": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}/locked_predictors_{pair}.parquet",
            "data_prep_report": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}/Data_Prep_Engine_Unit_Test_report_{pair}.json",
            "training_output_dir": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}",
            "prediction_output_dir": "outputs/local_test_outputs/Prediction_data_grid_tiles/{tile_id}",
            "model_config": {
                "max_depth": 4, "eta": 0.01, "gamma": 1.0,
                "reg_lambda": 1.0, "reg_alpha": 1.0,
                "subsample": 0.7, "colsample_bytree": 0.8,
                "tree_method": "hist", "max_bin": 256,
                "round_selection_mode": "cv_with_floor", "minimum_nrounds": 1000,
                "fixed_nrounds": 1000, "cv_nrounds": 2000,
                "early_stopping_rounds": 15, "cv_fallback_nrounds": 1000,
                "n_folds": 5, "block_size_m": 200.0,
                "n_boot": 5, "standard_horizon_years": 2.0,
                "minimum_training_rows": 100, "global_seed": 12345,
                "shap_sample_rows": 1000, "save_bootstrap_indices": True,
                "xgb_workers": 1, "xgb_memory_limit": "4GB",
                "xgb_processes": True, "verbose_logging": True,
            },
            "verification_chunk_rows": 50000,
            "r_references": {"training": None, "prediction": None},
            "r_comparison_tolerances": {
                "max_abs_prediction_difference_m": None,
                "rmse_prediction_difference_m": None,
                "max_abs_sd_difference_m": None,
            },
            "smoke_model_overrides": {
                "round_selection_mode": "fixed", "fixed_nrounds": 100,
                "n_boot": 2, "shap_sample_rows": 100,
            },
            "smoke_output_subfolder": "xgb_smoke",
        }
        # --- end perhaps move to config file? ---
        import copy
        overrides = copy.deepcopy(self.param_lookup.get("modeling_settings", {}))
        for key in ("model_config", "r_references", "r_comparison_tolerances", "smoke_model_overrides"):
            self.settings[key].update(overrides.pop(key, {}))
        self.settings.update(overrides)
        # TEMPORARY REPOSITORY TEST: remove guard and restore cloud routing before production merge.
        # self.is_aws = self.param_lookup.get('env', 'local') in ['remote', 'aws']
        if self.param_lookup.get("env", "local") != "local":
            raise ValueError("S3 execution is temporarily disabled for repository testing")
        self.is_aws = False
        self.outputs_dir = self.repository_root / "outputs/local_test_outputs"

    def _repository_path(self, value: str) -> str:
        """Anchor configured tile/pair paths to the checkout rather than the terminal cwd."""
        if value.startswith("s3://"):
            raise ValueError("Only local repository paths are enabled in this integration stage")
        path = Path(value.format(tile_id=self.settings["tile_id"], pair=self.settings["pair"]))
        return str(path if path.is_absolute() else self.repository_root / path)

    def configuration(self, profile: str = "full") -> dict:
        """Build a fresh resolved configuration without an external JSON or duplicate defaults.

        Smoke changes only explicit round/bootstrap/SHAP controls and output
        locations; it retains the complete input row sets and locked schema.
        """
        import copy
        cfg = copy.deepcopy(self.settings)
        for key in ("training_geoparquet", "prediction_geoparquet", "locked_predictors_parquet",
                    "data_prep_report", "training_output_dir", "prediction_output_dir"):
            cfg[key] = self._repository_path(cfg[key])
        for role, reference in cfg["r_references"].items():
            if reference:
                cfg["r_references"][role] = self._repository_path(reference)
        cfg["profile"] = profile
        if profile == "smoke":
            cfg["model_config"].update(cfg["smoke_model_overrides"])
            for key in ("training_output_dir", "prediction_output_dir"):
                cfg[key] = str(Path(cfg[key]) / cfg["smoke_output_subfolder"])
        elif profile != "full":
            raise ValueError("profile must be full or smoke")
        _validate_settings(cfg)
        return cfg

    def _resolve_paths(self, region: str, cfg: dict) -> None:
        """Temporarily use repository DataPrep/output folders instead of S3 region lookup.

        DELETE THIS TEMPORARY LOCAL ROUTING BEFORE MERGING TO PRODUCTION AND
        REINSTATE THE S3 CONFIGURATION/DISCOVERY BELOW. Keep scientific fixes.
        """
        self.outputs_dir = Path(cfg["training_output_dir"])
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        Path(cfg["prediction_output_dir"]).mkdir(parents=True, exist_ok=True)
        self.ml_prep_train_dir = UPath(Path(cfg["training_geoparquet"]).parent)
        self.ml_prep_pred_dir = UPath(Path(cfg["prediction_geoparquet"]).parent)
        self.xgb_out_dir = UPath(self.outputs_dir)
        # END TEMPORARY LOCAL ROUTING. Restore resolver and discovery together:
        # self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        # bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        # s3_dir_base = f"s3://{bucket}/{region}"
        # train_dir = get_config_item('MODEL', 'ML_PREP_TRAIN_DIR')
        # pred_dir = get_config_item('MODEL', 'ML_PREP_PRED_DIR')
        # xgb_dir = get_config_item('MODEL', 'XGB_OUTPUT_DIR')
        # self.ml_prep_train_dir = UPath(f"{s3_dir_base}/{train_dir}")
        # self.ml_prep_pred_dir = UPath(f"{s3_dir_base}/{pred_dir}")
        # self.xgb_out_dir = UPath(f"{s3_dir_base}/{xgb_dir}")

    def load_inputs(self, cfg: dict):
        """Validate the new repository DataPrep handoff and return exact fitting/prediction rows.

        A PASS report must match tile, pair, CRS, row counts and ordered schema.
        Fingerprint current input bytes; no imputation, predictor re-selection,
        weight recalculation or row dropping belongs in this modeling stage.
        """
        _validate_settings(cfg)
        prep = json.loads(Path(cfg["data_prep_report"]).read_text())
        if prep.get("status") != "PASS" or any(prep.get(k) != cfg[k] for k in ("tile_id", "pair", "crs")):
            raise ValueError("DataPrep report must be PASS for this tile, pair and CRS")
        table = pd.read_parquet(cfg["locked_predictors_parquet"])
        if "predictor" not in table:
            raise ValueError("DataPrep schema needs a predictor column")
        predictors = table.predictor.tolist()
        if not predictors or predictors[0] != "bathy_t" or "distance_from_shore" not in predictors:
            raise ValueError("Use the successful ordered schema: bathy_t first and distance_from_shore included")
        if predictors != prep.get("locked_predictors"):
            raise ValueError("Current locked predictors differ from the DataPrep PASS report")
        frames = {}
        for role in ("training", "prediction"):
            spatial = gpd.read_parquet(cfg[f"{role}_geoparquet"])
            if spatial.crs is None or spatial.crs.to_epsg() != 32617:
                raise ValueError(f"{role} CRS must already be EPSG:32617")
            df = pd.DataFrame(spatial.drop(columns=spatial.geometry.name))
            if df.empty or not df.columns.is_unique:
                raise ValueError(f"{role} table is empty or has duplicate columns")
            for column, value in (("tile_id", cfg["tile_id"]), ("pair_id", cfg["pair"])):
                if column not in df or df[column].isna().any() or not df[column].eq(value).all():
                    raise ValueError(f"{role} {column} does not match configuration")
            if "FID" not in df or df.FID.isna().any() or df.FID.duplicated().any():
                raise ValueError(f"{role} FID must be nonmissing and unique within the tile/pair")
            if len(df) != prep.get("counts", {}).get(role, {}).get("output_rows"):
                raise ValueError(f"{role} row count differs from the DataPrep report")
            x = pd.to_numeric(df.X, errors="coerce"); y = pd.to_numeric(df.Y, errors="coerce")
            finite_xy = np.isfinite(x) & np.isfinite(y)
            np.testing.assert_array_equal(spatial.geometry.isna(), ~finite_xy)
            points = spatial.loc[finite_xy]
            if not points.geometry.geom_type.eq("Point").all():
                raise ValueError("Model inputs must contain point geometry")
            np.testing.assert_allclose(points.geometry.x,x[finite_xy],atol=1e-7,rtol=0)
            np.testing.assert_allclose(points.geometry.y,y[finite_xy],atol=1e-7,rtol=0)
            frames[role] = df
        train, pred, tx, px, locked, _ = _prepare_model_data(frames["training"], frames["prediction"], predictors)
        if len(train) < cfg["model_config"]["minimum_training_rows"]:
            raise ValueError("Fewer eligible fitting rows than the configured minimum")
        if not pred.prediction_row_usable.any():
            raise ValueError("No usable prediction rows in this tile")
        np.testing.assert_allclose(train.delta_bathy, train.bathy_t1-train.bathy_t, rtol=0, atol=1e-8)
        np.testing.assert_allclose(train.delta_rate, train.delta_bathy/train.interval_years, rtol=0, atol=1e-8)
        years = [int(y) for y in cfg["pair"].split("_")]
        # Preserve DataPrep's per-row intervals: valid survey years may vary from nominal pair bounds.
        if {"year_t", "year_t1"}.issubset(train.columns):
            durations = pd.to_numeric(train.year_t1)-pd.to_numeric(train.year_t)
            valid = np.isfinite(durations) & (durations > 0)
            expected = durations.where(valid, float(years[1]-years[0]))
            np.testing.assert_allclose(train.interval_years,expected,rtol=0,atol=1e-8)
        identity = {key: _file_hash(cfg[key]) for key in
                    ("training_geoparquet", "prediction_geoparquet", "locked_predictors_parquet", "data_prep_report")}
        return train, pred, tx, px, locked, identity

    def _generated_output_paths(self, cfg: dict) -> list[Path]:
        """List only artifacts owned by this model stage and year pair for overwrite checks."""
        pair = cfg["pair"]; train_out = Path(cfg["training_output_dir"])
        paths = []
        for role in ("training", "prediction"):
            folder = Path(cfg[f"{role}_output_dir"])
            paths += [folder / f"LOCAL_{role}_summary_{pair}{suffix}" for suffix in (".parquet", ".geoparquet")]
        paths += [train_out / f"LOCAL_{stem}_{pair}{suffix}" for stem,suffix in (
            ("model_metadata",".json"), ("apparent_training_metrics",".json"),
            ("importance",".parquet"), ("shap_summary",".parquet"),
            ("spatial_cv_folds",".parquet"), ("spatial_cv_metrics",".parquet"), ("spatial_cv_oof",".parquet"))]
        paths += sorted((train_out / "models").glob(f"model_boot_*_{pair}.json"))
        paths += sorted((train_out / "bootstrap_indices").glob(f"bootstrap_*_{pair}.parquet"))
        paths += [train_out / f"LOCAL_test_run_manifest_{pair}.json"]
        return paths

    def close_dask(self) -> None:
        """Retire allocated workers before scheduler shutdown and tolerate partial setup.

        This retains the tested local cleanup improvement without changing the
        shared base Engine. Review a common base fix with the other engines.
        """
        try:
            if self.client is not None and self.client.status != "closed":
                try:
                    workers = list(self.client.scheduler_info().get("workers", {}))
                    if workers:
                        self.client.retire_workers(workers=workers, close_workers=True)
                finally:
                    self.client.close()
        finally:
            if self.cluster is not None:
                self.cluster.close()

    def run(self, overwrite: bool = False, execution: str = "dask", profile: str = "full") -> list[dict]:
        """Validate, execute one model task through Dask and bind outputs to input hashes.

        Explicit overwrite removes only this stage's matching pair artifacts so
        old bootstrap/CV files cannot masquerade as outputs of the new run.
        Debug inline execution is explicit; no silent Dask fallback is used.
        """
        cfg = self.configuration(profile)
        if execution not in ("dask", "inline"):
            raise ValueError("execution must be dask or inline")
        self._resolve_paths(cfg["region"], cfg)
        _, _, _, _, predictors, identity = self.load_inputs(cfg)
        outputs = self._generated_output_paths(cfg)
        if not overwrite and any(p.exists() for p in outputs):
            raise FileExistsError("Model outputs exist: use --validate-only or --overwrite deliberately")
        protected = {Path(cfg[key]).resolve() for key in
                     ("training_geoparquet", "prediction_geoparquet", "locked_predictors_parquet", "data_prep_report")}
        if any(path.resolve() in protected for path in outputs):
            raise ValueError("A model output would overwrite a protected DataPrep input")
        if overwrite:
            for path in outputs:
                path.unlink(missing_ok=True)
        contract = {"identity": identity, "model_config": cfg["model_config"], "schema": predictors, "profile": profile}
        manifest = Path(cfg["training_output_dir"]) / f"LOCAL_test_run_manifest_{cfg['pair']}.json"
        manifest.write_text(json.dumps({"contract": contract,"model_execution_status":"STARTED"}, indent=2))
        self.local_tmp_dir = Path(tempfile.mkdtemp(prefix="hhm_xgb_repository_"))
        model_cfg = dict(cfg["model_config"], locked_predictors=predictors,
                         prediction_output_dir=cfg["prediction_output_dir"], crs=cfg["crs"])
        params = [cfg["tile_id"],cfg["pair"],cfg["training_geoparquet"],cfg["prediction_geoparquet"],
                  cfg["training_output_dir"],model_cfg,False,str(self.local_tmp_dir),1,1,model_cfg["verbose_logging"]]
        # CLOUD: restore region/tile discovery and _get_s3_filesystem() with the production resolver.
        # training_files = self.ml_prep_train_dir.rglob("ML_ready_training_*.geoparquet")
        try:
            if execution == "dask":
                self.setup_dask("local", processes=model_cfg["xgb_processes"], n_workers=model_cfg["xgb_workers"],
                                threads_per_worker=1, memory_limit=model_cfg["xgb_memory_limit"])
                self.write_message("Submitting one repository XGBoost modeling task...", self.outputs_dir)
                results = self.client.gather(self.client.map(_train_local_tile_task,[params]))
            else:
                results = [_train_local_tile_task(params)]
            if len(results) != 1 or not results[0].get("ok"):
                raise RuntimeError(f"XGBoost modeling task failed: {results}")
            for key, digest in identity.items():
                if _file_hash(cfg[key]) != digest:
                    raise ValueError(f"DataPrep input changed during model execution: {cfg[key]}")
            manifest.write_text(json.dumps({"contract":contract,"model_execution_status":"COMPLETED","results":results},indent=2))
            return results
        except Exception as exc:
            manifest.write_text(json.dumps({"contract":contract,"model_execution_status":"FAILED","error":str(exc)},indent=2))
            raise
        finally:
            self.cleanup_resources(self.outputs_dir)
