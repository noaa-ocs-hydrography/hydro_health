import copy
import gc
import hashlib
import json
import re
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item
from upath import UPath

OUTPUTS = Path(__file__).parents[3] / "outputs"

NON_PREDICTORS = {
    "X", "Y", "FID", "tile_id", "source", "dataset_role", "year_t", "year_t1",
    "interval_years", "pair_id", "bathy_t", "bathy_t1", "delta_bathy", "delta_rate",
    "survey_end_date", "geometry", "prediction_row_usable", "sample_weight",
}

def _stable_seed(global_seed: int, *parts: object) -> int:
    """Helper to generate a stable, reproducible random seed based on inputs."""

    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int((global_seed + int.from_bytes(digest, "little")) % (2**31 - 1))

def _xgb_params(model_cfg: dict) -> dict:
    """Use the same XGBoost parameter mapping for CV and ensemble fitting."""

    return {
        "objective": "reg:squarederror", "eval_metric": ["rmse", "mae"],
        "max_depth": model_cfg["max_depth"], "eta": model_cfg["eta"],
        "gamma": model_cfg["gamma"], "lambda": model_cfg["reg_lambda"],
        "alpha": model_cfg["reg_alpha"], "subsample": model_cfg["subsample"],
        "colsample_bytree": model_cfg["colsample_bytree"], "nthread": 1,
        "tree_method": model_cfg["tree_method"], "max_bin": model_cfg["max_bin"],
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

    unique_blocks = block.drop_duplicates().to_numpy(
        copy=True
    )  # Own writable storage for pandas 3 Copy-on-Write.
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_blocks)

    mapping = {b: i % min(n_folds, len(unique_blocks)) for i, b in enumerate(unique_blocks)}
    return block.map(mapping).to_numpy(dtype=int)

def _model_metrics(observed: pd.Series | np.ndarray, predicted: np.ndarray, baseline: pd.Series | np.ndarray, interval: pd.Series | np.ndarray) -> dict:
    """Compute unweighted state/change/rate metrics; apparent fitting metrics are not held-out accuracy."""

    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    interval = np.asarray(interval, dtype=float)
    result = {}
    for name, left, right in (
        ("bathy_t1", obs, pred),
        ("change", obs - baseline, pred - baseline),
        ("rate", (obs - baseline) / interval, (pred - baseline) / interval),
    ):
        valid = np.isfinite(left) & np.isfinite(right)
        a = left[valid]
        b = right[valid]
        error = b - a
        result[f"n_{name}"] = int(len(a))
        result[f"rmse_{name}"] = float(np.sqrt(np.mean(error**2))) if len(a) else None
        result[f"mae_{name}"] = float(np.mean(np.abs(error))) if len(a) else None
        result[f"bias_{name}"] = float(np.mean(error)) if len(a) else None
        denominator = float(np.sum((a - np.mean(a)) ** 2)) if len(a) else 0.0
        result[f"r2_predictive_{name}"] = (
            float(1 - np.sum(error**2) / denominator) if denominator > 0 else None
        )
        result[f"correlation_{name}"] = (
            float(np.corrcoef(a, b)[0, 1])
            if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0
            else None
        )
        if name == "change":
            result["sign_accuracy_change"] = (
                float(np.mean(np.sign(a) == np.sign(b))) if len(a) else None
            )
    return result

def _fit_cv_fold(fold: int, folds: np.ndarray, features: pd.DataFrame, labels: np.ndarray, weights: np.ndarray, model_cfg: dict, seed: int) -> tuple[dict, pd.DataFrame | None]:
    """Fit one spatial holdout fold and return its metrics and predictions."""

    test = folds == fold
    train = ~test
    row = {"fold": int(fold), "status": "insufficient_rows",
           "n_train": int(train.sum()), "n_test": int(test.sum())}
    if train.sum() < 20 or test.sum() < 10:
        return row, None
    values = features.to_numpy(dtype=np.float32, copy=False)
    names = features.columns.tolist()
    dtrain = xgb.DMatrix(values[train], label=labels[train], weight=weights[train],
                        feature_names=names, missing=np.nan)
    dtest = xgb.DMatrix(values[test], label=labels[test], feature_names=names, missing=np.nan)
    model_seed = _stable_seed(seed, "cv_model", int(fold))
    model = xgb.train(
        _xgb_params(dict(model_cfg, xgb_seed=model_seed)), dtrain,
        num_boost_round=model_cfg["cv_nrounds"], evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=model_cfg["early_stopping_rounds"], verbose_eval=False)
    rounds = int(model.best_iteration) + 1
    prediction = model.predict(dtest, iteration_range=(0, rounds)).astype(float)
    error = prediction - labels[test]
    row.update(status="ok", best_nrounds=rounds, xgb_seed=model_seed,
               rmse_bathy_t1=float(np.sqrt(np.mean(error**2))),
               mae_bathy_t1=float(np.mean(np.abs(error))))
    predictions = pd.DataFrame({
        "row_index": np.flatnonzero(test), "fold": int(fold),
        "observed_bathy_t1": labels[test], "predicted_bathy_t1": prediction,
        "best_nrounds": rounds,
    })
    return row, predictions

def _choose_rounds_spatial_cv(coordinates: pd.DataFrame, features: pd.DataFrame, labels: np.ndarray, weights: np.ndarray, model_cfg: dict, seed: int, audit: dict) -> int:
    """Select boosting rounds with the existing randomized spatial-block CV."""

    mode = model_cfg["round_selection_mode"]
    if mode not in ("fixed", "cv_with_floor", "cv_only"):
        raise ValueError(f"Unknown round selection mode: {mode}")
    oof_columns = ["row_index", "fold", "observed_bathy_t1",
                   "predicted_bathy_t1", "best_nrounds"]
    audit.update(
        status="fixed" if mode == "fixed" else "not_run", mode=mode,
        folds=pd.DataFrame(columns=["row_index", "fold", "X", "Y"]),
        metrics=pd.DataFrame(columns=["fold", "status", "n_train", "n_test",
                                     "best_nrounds", "xgb_seed", "rmse_bathy_t1", "mae_bathy_t1"]),
        oof=pd.DataFrame(columns=oof_columns))
    if mode == "fixed":
        return int(model_cfg["fixed_nrounds"])
    folds = _spatial_fold_ids(coordinates, model_cfg["block_size_m"], model_cfg["n_folds"], seed)
    audit["folds"] = pd.DataFrame({
        "row_index": np.arange(len(folds)), "fold": folds,
        "X": coordinates.X.to_numpy(), "Y": coordinates.Y.to_numpy(),
    })
    fallback, floor = int(model_cfg["cv_fallback_nrounds"]), int(model_cfg["minimum_nrounds"])
    unique_folds = np.unique(folds)
    if len(unique_folds) < 2:
        audit["status"] = "fewer_than_two_folds_using_fallback"
        return max(floor, fallback) if mode == "cv_with_floor" else fallback
    best_iters, rows, oof = [], [], []
    for fold in unique_folds:
        row, predictions = _fit_cv_fold(fold, folds, features, labels, weights, model_cfg, seed)
        rows.append(row)
        if predictions is not None:
            oof.append(predictions)
            best_iters.append(row["best_nrounds"])
    audit["metrics"] = pd.DataFrame(rows)
    audit["oof"] = pd.concat(oof, ignore_index=True) if oof else pd.DataFrame(columns=oof_columns)
    if not best_iters:
        audit["status"] = "no_valid_folds_using_fallback"
        return max(floor, fallback) if mode == "cv_with_floor" else fallback
    selected = int(round(float(np.mean(best_iters))))
    audit.update(status="cv_success", raw_cv_selected_rounds=selected, fold_best_rounds=best_iters)
    return max(floor if mode == "cv_with_floor" else 50, selected)

def _translate_predictions(predicted_bathy: np.ndarray, baseline_bathy: np.ndarray, interval_years: np.ndarray, horizon: float) -> dict:
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

def _prepare_model_data(train_df: pd.DataFrame, pred_df: pd.DataFrame | None, locked_predictors: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.DataFrame | None, list[str]]:
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
    required_train = {
        "X",
        "Y",
        "FID",
        "bathy_t1",
        "bathy_t",
        "interval_years",
        "sample_weight",
        "delta_bathy",
        "delta_rate",
    }
    if missing := (required_train | set(predictors)) - set(train_df.columns):
        raise ValueError(f"Missing required ML-ready training columns: {sorted(missing)}")
    train_df = train_df.copy()
    for column in required_train | set(predictors):
        train_df[column] = pd.to_numeric(train_df[column], errors="coerce")
    essential = np.isfinite(
        train_df[["X", "Y", "bathy_t", "bathy_t1", "interval_years"]].to_numpy(dtype=float)
    ).all(axis=1)
    essential &= train_df["interval_years"].to_numpy(dtype=float) > 0
    if not essential.all():
        raise ValueError(
            "ML-ready training has invalid essential fields; rerun DataPrep instead of dropping rows"
        )
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
            raise ValueError(
                f"Training-first schema cannot deploy; missing prediction fields: {sorted(missing)}"
            )
        pred_df = pred_df.copy().reset_index(drop=True)
        for column in required_pred | set(predictors):
            pred_df[column] = pd.to_numeric(pred_df[column], errors="coerce")
        usable = np.isfinite(
            pred_df[["X", "Y", "bathy_t", "interval_years"]].to_numpy(dtype=float)
        ).all(axis=1)
        usable &= pred_df.interval_years.to_numpy(dtype=float) > 0
        if "prediction_row_usable" in pred_df:
            if not pred_df.prediction_row_usable.isin([True, False]).all():
                raise ValueError("prediction_row_usable must contain only boolean values")
            usable &= pred_df.prediction_row_usable.to_numpy(dtype=bool)
        pred_df["prediction_row_usable"] = usable
        pred_features = pred_df[predictors].astype(np.float32)
        if np.isinf(pred_features.loc[usable].to_numpy()).any():
            raise ValueError("Usable prediction rows contain infinite predictors")
    return train_df, pred_df, train_features, pred_features, predictors

def _update_running_stats(mean: np.ndarray, squared_difference: np.ndarray, values: np.ndarray, count: int) -> None:
    """Update per-row statistics without storing every bootstrap result."""

    delta = values - mean
    mean += delta / count
    squared_difference += delta * (values - mean)

def _publish_file(local_path: str, destination: str) -> None:
    """Copy a staged file to a local or authenticated S3 destination."""

    target = UPath(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(local_path).open("rb") as source, target.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)

def _save_parquet(df: pd.DataFrame, final_save_path: str, local_tmp_dir: str) -> None:
    """Stage Parquet locally and publish through the destination's filesystem."""

    tmp_path = Path(local_tmp_dir) / UPath(final_save_path).name
    try:
        df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
        _publish_file(str(tmp_path), final_save_path)
    finally:
        tmp_path.unlink(missing_ok=True)

def _save_json(payload: dict, path: str) -> None:
    """Write metadata or metrics to the resolved local/S3 destination."""

    with UPath(path).open("wt", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)

def _read_geoparquet(path: str) -> gpd.GeoDataFrame:
    """Use the same file-object read for local files and authenticated S3."""

    with UPath(path).open("rb") as stream:
        return gpd.read_parquet(stream)

def _save_geoparquet(df: pd.DataFrame, path: str, crs: str, local_tmp_dir: str) -> None:
    """Preserve invalid-coordinate rows and stage GeoParquet before publishing."""

    valid = np.isfinite(df.X) & np.isfinite(df.Y)
    geometry = gpd.points_from_xy(df.X.fillna(0), df.Y.fillna(0), crs=crs)
    geometry[~valid] = None
    tmp_path = Path(local_tmp_dir) / UPath(path).name
    try:
        gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs).to_parquet(
            tmp_path, index=False, compression="zstd", schema_version="1.0.0"
        )
        _publish_file(str(tmp_path), path)
    finally:
        tmp_path.unlink(missing_ok=True)


@dataclass
class ModelInputs:
    train: pd.DataFrame
    prediction: pd.DataFrame
    train_features: pd.DataFrame
    prediction_features: pd.DataFrame
    predictors: list[str]

def _bootstrap_diagnostics(model: xgb.Booster, x_fit: np.ndarray, predictors: list[str], model_cfg: dict, seed: int, bootstrap_index: int) -> tuple[list[dict], list[dict]]:
    """Compute gain and sampled SHAP diagnostics on their own random stream."""

    importance_rows, shap_rows = [], []
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

    shap_n = min(model_cfg["shap_sample_rows"], len(x_fit))
    if shap_n > 0:
        shap_rng = np.random.default_rng(_stable_seed(seed, "shap", bootstrap_index))
        shap_idx = shap_rng.choice(len(x_fit), size=shap_n, replace=False)
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

    return importance_rows, shap_rows

def _fit_bootstrap_model(inputs: ModelInputs, cfg: dict, seed: int, bootstrap_index: int, best_rounds: int, local_tmp_dir: str) -> tuple[xgb.Booster, dict]:
    """Sample rows with prepared weights, fit one model and save its artifacts."""

    model_cfg, paths = cfg["model_config"], cfg["output_paths"]
    tile_id, pair = cfg["tile_id"], cfg["pair"]
    train_df, predictors = inputs.train, inputs.predictors
    x_fit = inputs.train_features.to_numpy(dtype=np.float32, copy=False)
    labels = train_df.bathy_t1.to_numpy(dtype=np.float32)
    weights = train_df.sample_weight.to_numpy(dtype=float, copy=False)
    bootstrap_seed = _stable_seed(seed, "bootstrap", bootstrap_index)
    model_seed = _stable_seed(seed, "xgb_model", bootstrap_index)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.choice(len(train_df), size=len(train_df), replace=True)
    if model_cfg["verbose_logging"]:
        Engine.write_message_dask(
            f"Bootstrap {bootstrap_index}/{model_cfg['n_boot']}: {best_rounds} rounds, {len(train_df)} fitting rows",
            Path(cfg["log_dir"]),
        )
    dtrain = xgb.DMatrix(
        x_fit[indices],
        label=labels[indices],
        weight=weights[indices],
        feature_names=predictors,
        missing=np.nan,
    )
    model = xgb.train(
        _xgb_params(dict(model_cfg, xgb_seed=model_seed)),
        dtrain,
        num_boost_round=best_rounds,
        verbose_eval=False,
    )

    tmp_model_path = str(
        Path(local_tmp_dir) / f"{tile_id}_model_boot_{bootstrap_index:03d}_{pair}.json"
    )
    final_model_uri = paths["models"][bootstrap_index - 1]
    try:
        model.save_model(tmp_model_path)
        _publish_file(tmp_model_path, final_model_uri)
    finally:
        Path(tmp_model_path).unlink(missing_ok=True)

    audit = {
        "bootstrap_iteration": bootstrap_index,
        "bootstrap_seed": bootstrap_seed,
        "xgb_seed": model_seed,
    }
    return model, audit

def _fit_bootstrap_ensemble(inputs: ModelInputs, cfg: dict, best_rounds: int, seed: int, local_tmp_dir: str ) -> dict:
    """Fit reproducible bootstraps and accumulate per-row means and variance."""

    train_df, pred_df = inputs.train, inputs.prediction
    train_features, pred_features = inputs.train_features, inputs.prediction_features
    predictors = inputs.predictors
    model_cfg, paths = cfg["model_config"], cfg["output_paths"]
    n_boot = int(model_cfg["n_boot"])
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")

    x_fit = train_features.to_numpy(dtype=np.float32, copy=False)
    d_full = xgb.DMatrix(x_fit, feature_names=predictors, missing=np.nan)
    d_pred = None
    pred_usable = pred_df.prediction_row_usable.to_numpy(dtype=bool)
    if pred_usable.any():
        d_pred = xgb.DMatrix(
            pred_features.loc[pred_usable].to_numpy(dtype=np.float32, copy=False),
            feature_names=predictors,
            missing=np.nan,
        )

    train_mean = np.zeros(len(train_df), dtype=np.float64)
    train_m2 = np.zeros(len(train_df), dtype=np.float64)
    pred_mean = np.where(pred_usable, 0.0, np.nan)
    pred_m2 = np.where(pred_usable, 0.0, np.nan)

    # Bootstrap and SHAP use separate streams so diagnostics cannot change the fitted ensemble.
    bootstrap_audit = []
    model_uris = []
    importance_rows = []
    shap_rows = []

    for bootstrap_index in range(1, n_boot + 1):
        model, audit = _fit_bootstrap_model(
            inputs, cfg, seed, bootstrap_index, best_rounds, local_tmp_dir
        )
        bootstrap_audit.append(audit)
        train_prediction = model.predict(d_full).astype(np.float64, copy=False)
        _update_running_stats(train_mean, train_m2, train_prediction, bootstrap_index)
        if d_pred is not None:
            prediction = np.full(len(pred_df), np.nan)
            prediction[pred_usable] = model.predict(d_pred).astype(np.float64, copy=False)
            _update_running_stats(pred_mean, pred_m2, prediction, bootstrap_index)

        importance, shap = _bootstrap_diagnostics(
            model, x_fit, predictors, model_cfg, seed, bootstrap_index
        )
        importance_rows.extend(importance)
        shap_rows.extend(shap)

        model_uris.append(paths["models"][bootstrap_index - 1])

    return {
        "train_mean": train_mean,
        "train_m2": train_m2,
        "pred_mean": pred_mean,
        "pred_m2": pred_m2,
        "bootstrap_audit": bootstrap_audit,
        "model_uris": model_uris,
        "importance": importance_rows,
        "shap": shap_rows,
    }

def _prediction_summary(frame: pd.DataFrame, mean: np.ndarray, m2: np.ndarray, model_cfg: dict, training: bool = False) -> pd.DataFrame:
    """Build shared prediction/change/rate columns for either dataset role."""

    summary = frame.copy()
    n_boot = model_cfg["n_boot"]
    sd = (
        np.sqrt(np.maximum(m2, 0) / (n_boot - 1))
        if n_boot > 1
        else np.where(np.isfinite(mean), 0.0, np.nan)
    )
    for column, values in _translate_predictions(
        mean,
        summary.bathy_t.to_numpy(dtype=float),
        summary.interval_years.to_numpy(dtype=float),
        model_cfg["standard_horizon_years"],
    ).items():
        summary[column] = values
    summary["uncertainty_sd_bathy_next"] = sd
    summary["mean_predicted_bathy_t1"] = mean
    summary["uncertainty_sd_bathy_t1"] = sd
    summary["uncertainty_sd_change"] = sd
    summary["uncertainty_sd_rate"] = sd / summary.interval_years
    if training:
        summary["delta_residual"] = summary.delta_bathy - summary.mean_predicted_change
        summary["prediction_residual"] = summary.bathy_t1 - summary.mean_predicted_bathy_next
    return summary

def _model_metadata(inputs: ModelInputs, ensemble: dict, cfg: dict, best_rounds: int, cv_audit: dict) -> dict:
    """Describe the fitted ensemble, predictors and actual random seeds."""

    model_cfg = cfg["model_config"]
    return {
        "tile_id": cfg["tile_id"], "pair": cfg["pair"], "target": "bathy_t1",
        "training_baseline_source": "bathy_t", "prediction_baseline_source": "bathy_t",
        "model_baseline_feature": "bathy_t", "predictors": inputs.predictors,
        "best_iteration": best_rounds, "model_uris": ensemble["model_uris"],
        "crs": cfg["crs"], "n_boot": model_cfg["n_boot"], "n_fit": len(inputs.train),
        "n_prediction_rows": len(inputs.prediction),
        "n_prediction_usable": int(inputs.prediction.prediction_row_usable.sum()),
        "bootstrap_seeds": ensemble["bootstrap_audit"], "global_seed": model_cfg["global_seed"],
        "round_selection_mode": model_cfg["round_selection_mode"],
        "cv_status": cv_audit["status"], "cv_fold_best_rounds": cv_audit.get("fold_best_rounds", []),
        "cv_raw_selected_rounds": cv_audit.get("raw_cv_selected_rounds"),
        "early_stopping_metric": "mae", "model_config": model_cfg,
        "xgboost_version": xgb.__version__,
    }

def _save_model_outputs(inputs: ModelInputs, ensemble: dict, cfg: dict, best_rounds: int, cv_audit: dict, tmp_dir: str) -> None:
    """Write summaries, diagnostics and metadata using the resolved paths."""

    paths, model_cfg = cfg["output_paths"], cfg["model_config"]
    for role, frame, prefix in (
        ("training", inputs.train, "train"),
        ("prediction", inputs.prediction, "pred"),
    ):
        summary = _prediction_summary(
            frame,
            ensemble[f"{prefix}_mean"],
            ensemble[f"{prefix}_m2"],
            model_cfg,
            training=role == "training",
        )
        _save_parquet(summary, paths[f"{role}_summary"], tmp_dir)
        _save_geoparquet(summary, paths[f"{role}_summary_geo"], cfg["crs"], tmp_dir)
        if role == "training":
            metrics = _model_metrics(
                summary.bathy_t1, ensemble["train_mean"], summary.bathy_t, summary.interval_years
            )
            _save_json(metrics, paths["apparent_training_metrics"])
    for key, rows in (("importance", ensemble["importance"]), ("shap_summary", ensemble["shap"])):
        _save_parquet(pd.DataFrame(rows), paths[key], tmp_dir)
    for name in ("folds", "metrics", "oof"):
        table = cv_audit[name]
        if "row_index" in table:
            table["FID"] = inputs.train.FID.to_numpy()[table.row_index.to_numpy(dtype=int)]
        _save_parquet(table, paths[f"spatial_cv_{name}"], tmp_dir)
    _save_json(_model_metadata(inputs, ensemble, cfg, best_rounds, cv_audit), paths["model_metadata"])

def _load_model_inputs(cfg: dict) -> ModelInputs:
    """Load prepared tables and their locked feature order on the worker."""

    with UPath(cfg["locked_predictors_parquet"]).open("rb") as stream:
        schema = pd.read_parquet(stream)
    if "predictor" not in schema:
        raise ValueError("Locked predictor schema needs a predictor column")
    frames = {}
    for role in ("training", "prediction"):
        spatial = _read_geoparquet(cfg[f"{role}_geoparquet"])
        if spatial.crs is None or spatial.crs != cfg["crs"]:
            raise ValueError(f"{role} CRS must already be {cfg['crs']}")
        frame = pd.DataFrame(spatial.drop(columns=spatial.geometry.name))
        if frame.empty or not frame.columns.is_unique:
            raise ValueError(f"{role} table is empty or has duplicate columns")
        if "FID" not in frame or frame.FID.isna().any() or frame.FID.duplicated().any():
            raise ValueError(f"{role} FID must be nonmissing and unique")
        for column, expected in (("tile_id", cfg["tile_id"]), ("pair_id", cfg["pair"])):
            if column in frame and not frame[column].eq(expected).all():
                raise ValueError(f"{role} {column} does not match {expected}")
        frames[role] = frame
    inputs = ModelInputs(*_prepare_model_data(
        frames["training"], frames["prediction"], schema.predictor.tolist()))
    if len(inputs.train) < cfg["model_config"]["minimum_training_rows"]:
        raise ValueError("Fewer eligible fitting rows than the configured minimum")
    if not inputs.prediction.prediction_row_usable.any():
        raise ValueError("No usable prediction rows in this tile")
    train = inputs.train
    np.testing.assert_allclose(train.delta_bathy, train.bathy_t1 - train.bathy_t, atol=1e-8, rtol=0)
    np.testing.assert_allclose(train.delta_rate, train.delta_bathy / train.interval_years, atol=1e-8, rtol=0)
    return inputs

def _prepare_outputs(cfg: dict) -> None:
    """Check overwrite behavior and create this task's output folders."""

    existing = [UPath(path) for path in cfg["owned_outputs"] if UPath(path).exists()]
    if existing and not cfg["overwrite"]:
        raise FileExistsError(f"Model outputs exist for {cfg['tile_id']}, {cfg['pair']}")
    for path in existing:
        path.unlink()
    for role in ("training", "prediction"):
        UPath(cfg[f"{role}_output_dir"]).mkdir(parents=True, exist_ok=True)

def _train_tile_task(cfg: dict) -> dict:
    """Load, fit and save one tile/year pair through a Dask worker."""

    log_dir = Path(cfg["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        inputs = _load_model_inputs(cfg)
        _prepare_outputs(cfg)
        seed = _stable_seed(cfg["model_config"]["global_seed"], "local", cfg["tile_id"], cfg["pair"])
        audit = {}
        best_rounds = _choose_rounds_spatial_cv(
            inputs.train[["X", "Y"]], inputs.train_features,
            inputs.train.bathy_t1.to_numpy(dtype=np.float32),
            inputs.train.sample_weight.to_numpy(dtype=float),
            cfg["model_config"], _stable_seed(seed, "cv"), audit)
        with tempfile.TemporaryDirectory(prefix="hhm_xgb_") as tmp_dir:
            ensemble = _fit_bootstrap_ensemble(inputs, cfg, best_rounds, seed, tmp_dir)
            _save_model_outputs(inputs, ensemble, cfg, best_rounds, audit, tmp_dir)
        return {"ok": True, "region": cfg["region"], "tile_id": cfg["tile_id"],
                "pair": cfg["pair"], "n_fit": len(inputs.train), "best_iteration": best_rounds}
    except Exception:
        Engine.write_message_dask(traceback.format_exc(), log_dir)
        raise
    finally:
        gc.collect()


class XGBoostModelingEngine(Engine):
    """Train prepared tile/year-pair ensembles for each configured ecoregion."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize model settings and the shared engine configuration."""

        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        if param_lookup["env"] not in ("local", "aws"):
            raise ValueError("env must be local or aws")
        self.is_aws = param_lookup["env"] == "aws"
        overrides = copy.deepcopy(param_lookup.get("modeling_settings", {}))
        self.tile_id = overrides.get("tile_id")
        self.pair = overrides.get("pair")
        self.model_config = {
            "max_depth": 4, "eta": 0.01, "gamma": 1.0,
            "reg_lambda": 1.0, "reg_alpha": 1.0, "subsample": 0.7,
            "colsample_bytree": 0.8, "tree_method": "hist", "max_bin": 256,
            "round_selection_mode": "cv_with_floor", "minimum_nrounds": 1000,
            "fixed_nrounds": 1000, "cv_nrounds": 2000, "early_stopping_rounds": 15,
            "cv_fallback_nrounds": 1000, "n_folds": 5, "block_size_m": 200.0,
            "n_boot": 5, "standard_horizon_years": 2.0, "minimum_training_rows": 100,
            "global_seed": 12345, "shap_sample_rows": 1000,
            "xgb_workers": 1, "xgb_memory_limit": "4GB",
            "xgb_processes": True, "verbose_logging": True,
        }
        self.model_config.update(overrides.get("model_config", {}))
        for key in ("n_boot", "xgb_workers"):
            value = self.model_config[key]
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            self.model_config[key] = int(value)

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths for the environment and the given ecoregion."""

        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.write_message(f"XGBoostModelingEngine resolved outputs_dir for {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item("SHARED", "OUTPUT_BUCKET")
        s3_dir_base = f"s3://{bucket}/{self.output_prefix}/{region}" if self.output_prefix else f"s3://{bucket}/{region}"

        train_dir = get_config_item("MODEL", "ML_PREP_TRAIN_DIR")
        self.ml_prep_train_dir = UPath(f"{s3_dir_base}/{train_dir}") if self.is_aws else UPath(self.outputs_dir / train_dir)

        pred_dir = get_config_item("MODEL", "ML_PREP_PRED_DIR")
        self.ml_prep_pred_dir = UPath(f"{s3_dir_base}/{pred_dir}") if self.is_aws else UPath(self.outputs_dir / pred_dir)

        model_dir = get_config_item("MODEL", "XGB_OUTPUT_DIR")
        self.xgb_out_dir = UPath(f"{s3_dir_base}/{model_dir}") if self.is_aws else UPath(self.outputs_dir / model_dir)

    def collect_training_files(self) -> list[UPath]:
        """Collect prepared training files under the resolved regional directory."""

        return sorted(self.ml_prep_train_dir.rglob("ML_ready_training_*.geoparquet"), key=str)

    def prepare_tasks(self, region: str) -> None:
        """Build tile/year-pair task settings from the prepared training files."""

        self.task_configs = []
        seen = set()
        for training_file in self.collect_training_files():
            match = re.fullmatch(r"ML_ready_training_(\d{4}_\d{4})\.geoparquet", training_file.name)
            if match is None:
                continue
            tile_id, pair = training_file.parent.name, match[1]
            if (self.tile_id and tile_id != self.tile_id) or (self.pair and pair != self.pair):
                continue
            if (tile_id, pair) in seen:
                raise ValueError(f"Duplicate prepared input for {tile_id}, {pair}")
            if int(pair[:4]) >= int(pair[5:]):
                raise ValueError(f"Year pair must be increasing: {pair}")
            seen.add((tile_id, pair))
            train_out, pred_out = self.xgb_out_dir / tile_id / "Training", self.xgb_out_dir / tile_id / "Prediction"
            paths = {}
            for role, folder in (("training", train_out), ("prediction", pred_out)):
                paths[f"{role}_summary"] = str(folder / f"LOCAL_{role}_summary_{pair}.parquet")
                paths[f"{role}_summary_geo"] = str(folder / f"LOCAL_{role}_summary_{pair}.geoparquet")
            for stem in ("importance", "shap_summary", "spatial_cv_folds", "spatial_cv_metrics", "spatial_cv_oof"):
                paths[stem] = str(train_out / f"LOCAL_{stem}_{pair}.parquet")
            for stem in ("model_metadata", "apparent_training_metrics"):
                paths[stem] = str(train_out / f"LOCAL_{stem}_{pair}.json")
            paths["models"] = [str(train_out / "models" / f"model_boot_{i:03d}_{pair}.json")
                               for i in range(1, self.model_config["n_boot"] + 1)]
            owned = [path for path in paths.values() if isinstance(path, str)] + paths["models"]
            owned += [str(path) for path in (train_out / "models").glob(f"model_boot_*_{pair}.json")]
            self.task_configs.append({
                "region": region, "tile_id": tile_id, "pair": pair, "crs": str(self.target_crs),
                "training_geoparquet": str(training_file),
                "prediction_geoparquet": str(self.ml_prep_pred_dir / tile_id / f"ML_ready_prediction_{pair}.geoparquet"),
                "locked_predictors_parquet": str(training_file.parent / f"locked_predictors_{pair}.parquet"),
                "training_output_dir": str(train_out), "prediction_output_dir": str(pred_out),
                "model_config": self.model_config, "output_paths": paths,
                "owned_outputs": sorted(set(owned)), "log_dir": str(self.outputs_dir),
            })

    def execute_tasks(self, overwrite: bool = False) -> list[dict]:
        """Submit bounded batches so configured Dask workers can fit tiles in parallel."""

        results = []
        batch_size = self.model_config["xgb_workers"]
        for start in range(0, len(self.task_configs), batch_size):
            jobs = [dict(cfg, overwrite=overwrite) for cfg in self.task_configs[start:start + batch_size]]
            futures = self.client.map(_train_tile_task, jobs, pure=False)
            try:
                results.extend(self.client.gather(futures))
            except Exception:
                self.client.cancel(futures)
                raise
            finally:
                for future in futures:
                    future.release()
        return results

    def run(self, overwrite: bool = False) -> list[dict]:
        """Set up Dask, process ecoregions and always close the worker pool."""

        results = []
        try:
            cfg = self.model_config
            self.setup_dask(self.param_lookup["env"], processes=cfg["xgb_processes"],
                            n_workers=cfg["xgb_workers"], threads_per_worker=1,
                            memory_limit=cfg["xgb_memory_limit"])
            for eco_region in self.param_lookup["eco_regions"].value:
                self._resolve_paths(eco_region)
                self.prepare_tasks(eco_region)
                self.write_message(f"Queuing {len(self.task_configs)} XGBoost tasks for {eco_region}.", OUTPUTS)
                results.extend(self.execute_tasks(overwrite))
            return results
        finally:
            self.cleanup_resources(OUTPUTS)
