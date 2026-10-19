"""Prepare tile/year-pair tables using the tested scientific preparation rules.

This repository integration candidate keeps the production Engine inheritance,
worker-task boundary, Dask map/gather and output naming. Temporary local IO uses
repository GeoParquet fixtures while upstream S3 inputs are unavailable. It does
not convert R files or fit a model. Search TEMPORARY REPOSITORY TEST before merge.
Already-prepared fixtures test reproducibility, not recovery of discarded rows.
"""
import re
import tempfile
import warnings
import os
import gc
import shutil
import pathlib
from pathlib import Path
from typing import Sequence, Iterable

# TEMPORARY REPOSITORY TEST: reinstate the cloud import before production merge.
# import s3fs
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from upath import UPath

# Production base class retained; no standalone LocalEngineSupport is used.
from hydro_health.engines.Engine import Engine
# TEMPORARY REPOSITORY TEST: reinstate YAML/S3 lookup before production merge.
# from hydro_health.helpers.tools import get_config_item

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[4]
INPUTS = REPOSITORY_ROOT / 'inputs'
OUTPUTS = REPOSITORY_ROOT / 'outputs'

DYNAMIC_BASES = (
    "hurr_strength_mean",
    "tsm_mean",
)

NON_PREDICTORS = {
    "X", "Y", "FID", "tile_id", "source", "dataset_role",
    "year_t", "year_t1", "year_ti", "interval_years", "pair_id",
    "bathy_t1", "bathy_ti", "delta_bathy", "delta_rate", "survey_end_date",
    "geometry", "prediction_row_usable", "sample_weight"
}


def _pair_duration_years(pair: str) -> float:
    """Validate YYYY_YYYY and return the positive elapsed time used for annualization."""
    match = re.fullmatch(r"(\d{4})_(\d{4})", pair)
    if not match:
        raise ValueError(f"Invalid year-pair label: {pair}")
    y0, y1 = map(int, match.groups())
    if y1 <= y0:
        raise ValueError(f"Year pair must increase: {pair}")
    return float(y1 - y0)


def _safe_numeric_cols(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce only the requested existing fields to numeric; conversion failures become NA without dropping rows."""
    out = df.copy()
    for col in set(columns).intersection(out.columns):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _ensure_training_targets(df: pd.DataFrame, pair: str, tolerance: float = 1e-8) -> pd.DataFrame:
    """Recompute newer minus older elevation and annualized change; repair invalid intervals from the pair label."""
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
            warnings.warn(f"{int(mismatch.sum())} delta_bathy values differed from bathy_t1 - bathy_t and were replaced.", stacklevel=2)
            
    out["delta_bathy"] = recalculated
    out["delta_rate"] = out["delta_bathy"] / out["interval_years"]
    return out


def _build_pair_specific_predictors(master_predictors: Sequence[str], pair: str) -> list[str]:
    """Append the year pair only to the two dynamic predictor bases used by the supplied engine."""
    return [f"{p}_{pair}" if p in DYNAMIC_BASES else p for p in master_predictors]


def _resolve_predictors(
    pair: str,
    master_predictors: Sequence[str],
    train_columns: Sequence[str],
    prediction_columns: Sequence[str] | None,
    mode: str,
) -> tuple[list[str], list[str], list[str]]:
    """Lock an ordered training schema, prevent target leakage, and enforce deployment compatibility."""
    if mode not in {"og_training_first", "deployment_intersection"}:
        raise ValueError(f"Unknown predictor resolution mode: {mode}")
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
    """Match the R power or capped-linear weighting formula; these weights apply only to training."""
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
    """Report NA and finite counts before row filtering, including partially missing predictors."""
    return pd.DataFrame({
        "predictor": predictors,
        "n_rows": len(df),
        "n_na": [int(df[p].isna().sum()) for p in predictors],
        "proportion_na": [float(df[p].isna().mean()) for p in predictors],
        "n_finite": [int(np.isfinite(pd.to_numeric(df[p], errors="coerce")).sum()) if p in df else 0 for p in predictors],
    })


def _predictor_range_table(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    """Describe finite predictor values among fitting-eligible rows without filling their NAs."""
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
    """Build EPSG:32617 point geometry and export without dropping invalid-XY rows."""
    out = df.copy()
    valid_xy = np.isfinite(pd.to_numeric(out['X'], errors="coerce")) & np.isfinite(pd.to_numeric(out['Y'], errors="coerce"))
    geometry = gpd.GeoSeries([Point(float(x), float(y)) if ok else None for x, y, ok in zip(out['X'], out['Y'], valid_xy)], index=out.index, crs=crs)
    gdf = gpd.GeoDataFrame(out, geometry=geometry, crs=crs)
    
    tmp_path = str(Path(local_tmp_dir) / filename)
    gdf.to_parquet(tmp_path, index=False, compression="zstd")
    
    # TEMPORARY REPOSITORY TEST: delete local write branch and reinstate S3 before merge:
    # if is_aws and final_save_path.startswith("s3://"):
    #     s3fs.S3FileSystem().put(tmp_path, final_save_path)
    if is_aws or final_save_path.startswith("s3://"):
        raise ValueError("AWS writes are disabled in the local testing copy")
    UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp_path, final_save_path)
        
    if Path(tmp_path).exists():
        os.remove(tmp_path)


def _save_parquet(df: pd.DataFrame, final_save_path: str, local_tmp_dir: str, filename: str, is_aws: bool) -> None:
    """Stage a nonspatial diagnostic table and copy it to the local output folder."""
    tmp_path = str(Path(local_tmp_dir) / filename)
    df.to_parquet(tmp_path, index=False, engine="pyarrow", compression="zstd")
    
    # TEMPORARY REPOSITORY TEST: delete local write branch and reinstate S3 before merge:
    # if is_aws and final_save_path.startswith("s3://"):
    #     s3fs.S3FileSystem().put(tmp_path, final_save_path)
    if is_aws or final_save_path.startswith("s3://"):
        raise ValueError("AWS writes are disabled in the local testing copy")
    UPath(final_save_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp_path, final_save_path)
        
    if Path(tmp_path).exists():
        os.remove(tmp_path)


def _read_repository_geoparquet(uri: str, crs: str, tile_id: str, pair: str) -> pd.DataFrame:
    """Validate fixture CRS/identity/point coordinates before dropping geometry.

    Assigning a CRS cannot reproject numeric X/Y. Reject inconsistent fixtures
    rather than silently writing mislocated output or mixing tiles/year pairs.
    """
    spatial = gpd.read_parquet(uri)
    if spatial.crs is None or spatial.crs.to_epsg() != 32617:
        raise ValueError(f"Input must already use EPSG:32617: {uri}")
    if crs != "EPSG:32617":
        raise ValueError("Repository preparation test requires crs='EPSG:32617'")
    if not {"X", "Y"}.issubset(spatial.columns):
        raise ValueError(f"Missing input X/Y columns: {uri}")
    for col, expected in (("tile_id", tile_id), ("pair_id", pair)):
        if col in spatial and not spatial[col].dropna().astype(str).eq(expected).all():
            raise ValueError(f"Input {col} does not match {expected}: {uri}")
    finite_xy = np.isfinite(pd.to_numeric(spatial.X, errors="coerce")) & np.isfinite(pd.to_numeric(spatial.Y, errors="coerce"))
    if not spatial.geometry.isna().equals(~finite_xy):
        raise ValueError(f"Point geometry/null positions do not match finite X/Y: {uri}")
    points = spatial.loc[finite_xy]
    if not points.geometry.geom_type.eq("Point").all():
        raise ValueError("Preparation fixtures must contain point geometry")
    np.testing.assert_allclose(points.geometry.x, pd.to_numeric(points.X), rtol=0, atol=1e-7)
    np.testing.assert_allclose(points.geometry.y, pd.to_numeric(points.Y), rtol=0, atol=1e-7)
    return pd.DataFrame(spatial.drop(columns=spatial.geometry.name))


def _prep_tile_task(params: list) -> dict:
    """Prepare one tile/pair using the unchanged load/schema/filter/weight/export stages.

    Native missing predictor values remain available to XGBoost; only essential
    fitting fields control training eligibility. Prediction retains every row.
    """
    
    tile_id, pair, raw_train_uri, raw_pred_uri, master_predictors, model_cfg, crs, is_aws, out_train_dir, out_pred_dir, local_tmp_dir, current_index, total_count, verbose = params

    worker_log_dir = Path(out_train_dir)

    if verbose:
        Engine.write_message_dask(f"Prepping ML data for tile {tile_id} - Pair {pair} ({current_index}/{total_count})...", worker_log_dir)

    try:
        # Load raw data
        # TEMPORARY REPOSITORY TEST: delete local fixture load and reinstate S3 before merge.
        # training_raw = pd.read_parquet(raw_train_uri, storage_options={"anon": False})
        training_raw = _read_repository_geoparquet(raw_train_uri, crs, tile_id, pair)
        training_raw = _ensure_training_targets(training_raw, pair)
        training_raw["pair_id"] = pair
        training_raw["tile_id"] = tile_id
        
        prediction_raw = None
        if raw_pred_uri:
            # TEMPORARY REPOSITORY TEST: delete local fixture load and reinstate S3 before merge.
            # prediction_raw = pd.read_parquet(raw_pred_uri, storage_options={"anon": False})
            prediction_raw = _read_repository_geoparquet(raw_pred_uri, crs, tile_id, pair)
            prediction_raw["interval_years"] = _pair_duration_years(pair)
            prediction_raw["pair_id"] = pair
            prediction_raw["tile_id"] = tile_id

        # Resolve Predictor Schema
        predictors, missing_train, missing_pred = _resolve_predictors(
            pair, master_predictors, training_raw.columns,
            prediction_raw.columns if prediction_raw is not None else None,
            model_cfg.get("predictor_resolution_mode", "og_training_first")
        )
        
        if model_cfg.get("strict_requested_predictors", True) and missing_train:
            raise ValueError(f"Requested predictors missing from training: {missing_train}")

        _save_parquet(
            pd.DataFrame({"predictor": predictors}),
            str(UPath(out_train_dir) / f"locked_predictors_{pair}.parquet"),
            local_tmp_dir, f"{tile_id}_schema_{pair}.parquet", is_aws,
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
                Engine.write_message_dask(f" [SKIP] Tile '{tile_id}': Insufficient valid training rows ({len(subgrid_data)}).", worker_log_dir)
            return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": "insufficient_training_rows", "n": len(subgrid_data)}

        # Sample Weights
        weights = np.ones(len(subgrid_data))
        if model_cfg.get("use_weighted_loss", True):
            weights = _calculate_sample_weights(
                subgrid_data["delta_bathy"].to_numpy(dtype=float),
                model_cfg.get("weight_alpha", 0.5),
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
            # Match R section 3B: retain every row, including unusable rows.
            # The later ML engine must use this flag when deciding what to predict.
            pred_essential = np.isfinite(prediction_raw["X"]) & np.isfinite(prediction_raw["Y"]) & np.isfinite(prediction_raw["bathy_t"])
            prediction_ready = prediction_raw.copy().reset_index(drop=True)
            prediction_ready["prediction_row_usable"] = pred_essential.to_numpy()
            _save_geoparquet(prediction_ready, str(UPath(out_pred_dir) / f"ML_ready_prediction_{pair}.geoparquet"), local_tmp_dir, f"{tile_id}_ready_pred_{pair}.geoparquet", is_aws, crs)

        if verbose:
            Engine.write_message_dask(f" [{current_index}/{total_count}] [SUCCESS] Tile '{tile_id}' data prepped for ML.", worker_log_dir)
            
        return {"ok": True, "tile_id": tile_id, "pair": pair, "n_fit": len(subgrid_data), "schema_locked": True, "missing_training": missing_train, "missing_prediction": missing_pred}

    except Exception as e:
        Engine.write_message_dask(f"ERROR: DataPrep failed for tile {tile_id} - pair {pair}: {e}", worker_log_dir)
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": str(e)}
    finally:
        gc.collect()


class DataPrepEngine(Engine):
    """Production-shaped preparation engine with a temporary repository-local IO adapter."""

    def __init__(self, param_lookup: dict | None = None, output_prefix: str | bool = False) -> None:
        """Group test controls here, then retain production base Engine initialization.

        Optional `data_prep_settings`/`model_config` overrides let the test runner
        supply synthetic fixtures without another support module or JSON config.
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
            "input_stage": "prepared_fixture",  # OR raw_batch: use upstream GeoParquet, not filtered fixtures.
            "training_input": "inputs/local_test_inputs/Training_data_grid_tiles/ML_ready_training_{pair}.geoparquet",
            "prediction_input": "inputs/local_test_inputs/Prediction_data_grid_tiles/ML_ready_prediction_{pair}.geoparquet",
            "raw_training_input": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}/raw/{tile_id}/{tile_id}_{pair}_training_batch.parquet",
            "raw_prediction_input": "outputs/local_test_outputs/Prediction_data_grid_tiles/{tile_id}/raw/{tile_id}/{tile_id}_{pair}_prediction_batch.parquet",
            "training_output": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}",
            "prediction_output": "outputs/local_test_outputs/Prediction_data_grid_tiles/{tile_id}",
            "predictor_schema": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}/locked_predictors_{pair}.parquet",
            "master_predictors": None,  # "/tests/boruta_master_predictor_list_new.txt"  #Optional explicit Boruta names IN ORDER; otherwise read saved schema.
            "pre_xgb_data_mode": "native_missing",
            "predictor_resolution_mode": "og_training_first",
            "strict_requested_predictors": True,
            "use_weighted_loss": True,
            "weight_alpha": 0.5,
            "weight_method": "power",
            "weight_cap_quantile": 0.95,
            "max_weight": 2.0,
            "minimum_training_rows": 100,
            "data_prep_workers": 1,
            "data_prep_memory_limit": "4GB",
            "data_prep_processes": True,
            "verbose_logging": True,
        }
        # --- end perhaps move to config file? ---
        self.settings.update(self.param_lookup.get("data_prep_settings", {}))
        self.settings.update(self.param_lookup.get("model_config", {}))
        # TEMPORARY REPOSITORY TEST: remove local-only guard when restoring S3 before merge.
        # self.is_aws = self.param_lookup.get('env', 'local') in ['remote', 'aws']
        if self.param_lookup.get("env", "local") != "local":
            raise ValueError("S3 execution is temporarily disabled for repository testing")
        self.is_aws = False
        self.inputs_dir = self.repository_root / "inputs"
        self.outputs_dir = self.repository_root / "outputs" / "local_test_outputs"

    def _repository_path(self, pattern: str) -> Path:
        """Expand only configured tile/pair placeholders and anchor paths to the repository."""
        path = Path(pattern.format(tile_id=self.settings["tile_id"], pair=self.settings["pair"]))
        return path if path.is_absolute() else self.repository_root / path

    def _load_master_predictors(self) -> list[str]:
        """Preserve the successful Boruta schema exactly instead of guessing its order.

        Temporary default reads the saved locked table before any output writes.
        Set master_predictors to an explicit ordered list for the permanent
        class/config choice; do not fall back to the old broad variable list.
        """
        predictors = self.settings["master_predictors"]
        if predictors is None:
            path = self._repository_path(self.settings["predictor_schema"])
            if not path.is_file():
                raise FileNotFoundError(f"Copy the successful locked schema here, or set master_predictors explicitly: {path}")
            table = pd.read_parquet(path)
            if "predictor" not in table:
                raise ValueError("Locked schema must have a predictor column")
            predictors = table.predictor.tolist()
        if not isinstance(predictors, list) or not predictors or any(not isinstance(p, str) or not p.strip() for p in predictors):
            raise ValueError("Predictors must be a nonempty ordered list of nonblank names")
        if len(set(predictors)) != len(predictors) or predictors[0] != "bathy_t":
            raise ValueError("Schema must contain unique names, with bathy_t first")
        if "distance_from_shore" not in predictors:
            raise ValueError("The tested predictor schema must include distance_from_shore")
        if any(p in NON_PREDICTORS for p in predictors):
            raise ValueError("The chosen predictor list contains target/metadata columns")
        return list(predictors)

    def _resolve_paths(self, region: str) -> None:
        """Temporarily replace S3 region lookup with the screenshot's repository layout.

        DELETE THIS TEMPORARY LOCAL BLOCK BEFORE MERGING TO PRODUCTION AND
        REINSTATE THE S3 CONFIGURATION BELOW. The scientific helpers stay.
        """
        self.ml_prep_train_dir = self._repository_path(self.settings["training_output"])
        self.ml_prep_pred_dir = self._repository_path(self.settings["prediction_output"])
        stage = self.settings["input_stage"]
        if stage not in {"prepared_fixture", "raw_batch"}:
            raise ValueError("input_stage must be prepared_fixture or raw_batch")
        train_key, pred_key = ("training_input", "prediction_input") if stage == "prepared_fixture" else ("raw_training_input", "raw_prediction_input")
        self.raw_train_uri = self._repository_path(self.settings[train_key])
        self.raw_pred_uri = self._repository_path(self.settings[pred_key])
        self.outputs_dir = self.ml_prep_train_dir
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.ml_prep_pred_dir.mkdir(parents=True, exist_ok=True)
        if self.raw_train_uri.resolve() == (self.ml_prep_train_dir / f"ML_ready_training_{self.settings['pair']}.geoparquet").resolve():
            raise ValueError("Training input and output must be different files")
        if self.raw_pred_uri.resolve() == (self.ml_prep_pred_dir / f"ML_ready_prediction_{self.settings['pair']}.geoparquet").resolve():
            raise ValueError("Prediction input and output must be different files")
        for path in (self.raw_train_uri, self.raw_pred_uri):
            if not path.is_file():
                raise FileNotFoundError(f"Configured {stage} GeoParquet input not found: {path}")
        # END TEMPORARY LOCAL BLOCK.
        # Restore the original production resolver/discovery together:
        # self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        # bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        # s3_dir_base = f"s3://{bucket}/{region}"
        # training_tiles_dir = get_config_item('MODEL', 'TRAINING_TILES_DIR')
        # prediction_tiles_dir = get_config_item('MODEL', 'PREDICTION_TILES_DIR')
        # self.training_tiles_dir = UPath(f"{s3_dir_base}/{training_tiles_dir}")
        # self.prediction_tiles_dir = UPath(f"{s3_dir_base}/{prediction_tiles_dir}")
        # ml_prep_train_dir = get_config_item('MODEL', 'ML_PREP_TRAIN_DIR')
        # ml_prep_pred_dir = get_config_item('MODEL', 'ML_PREP_PRED_DIR')
        # self.ml_prep_train_dir = UPath(f"{s3_dir_base}/{ml_prep_train_dir}")
        # self.ml_prep_pred_dir = UPath(f"{s3_dir_base}/{ml_prep_pred_dir}")

    def _build_task_params(self, model_cfg: dict, crs: str, verbose_workers: bool) -> list[list]:
        """Select exactly the configured local tile/pair, retaining the worker argument layout.

        Temporary fixture routing replaces S3 batch discovery; restore discovery
        with the production resolver, while preserving explicit year-pair control.
        """
        # CLOUD discovery to reinstate before production merge:
        # fs = s3fs.S3FileSystem(anon=False)
        # training_batch_files = [f"s3://{p}" for p in fs.find(str(self.training_tiles_dir))
        #                         if p.endswith('_training_batch.parquet')]
        return [[self.settings["tile_id"], self.settings["pair"], str(self.raw_train_uri),
                 str(self.raw_pred_uri), self.master_predictors, model_cfg, crs, self.is_aws,
                 str(self.ml_prep_train_dir), str(self.ml_prep_pred_dir),
                 str(self.local_tmp_dir), 1, 1, verbose_workers]]

    def close_dask(self) -> None:
        """Close only allocated resources, in order, so cleanup cannot mask setup failures.

        The supplied base Engine closes None unconditionally. Keep this small
        override until the same defensive change can be reviewed in the base.
        """
        try:
            if self.client is not None:
                self.client.close()
        finally:
            if self.cluster is not None:
                self.cluster.close()

    def run(self, overwrite: bool = False, execution: str = "dask") -> list[dict]:
        """Run the production task through Dask and surface empty/failed preparation.

        Inline execution is an explicit debugging/self-test choice, not a silent
        Dask fallback. Output replacement is deliberate and limited to prep files.
        All temporary files are private to this run and cleaned even on failure.
        """
        cfg = self.settings
        _pair_duration_years(cfg["pair"])
        if cfg["crs"] != "EPSG:32617":
            raise ValueError("This integration test requires EPSG:32617")
        if cfg["pre_xgb_data_mode"] not in {"native_missing", "legacy_complete"}:
            raise ValueError("Unknown pre_xgb_data_mode")
        if execution not in {"dask", "inline"}:
            raise ValueError("execution must be dask or inline")
        self.local_tmp_dir = Path(tempfile.mkdtemp(prefix="hhm_data_prep_"))
        try:
            self._resolve_paths(cfg["region"])
            self.master_predictors = self._load_master_predictors()
            params_list = self._build_task_params(cfg, cfg["crs"], cfg["verbose_logging"])
            for role, directory in (("training", self.ml_prep_train_dir), ("prediction", self.ml_prep_pred_dir)):
                names = [f"ML_ready_{role}_{cfg['pair']}.geoparquet", f"predictor_missingness_{role}_{cfg['pair']}.parquet"]
                if role == "training":
                    names += [f"locked_predictors_{cfg['pair']}.parquet", f"predictor_ranges_training_{cfg['pair']}.parquet"]
                if not overwrite and any((directory/name).exists() for name in names):
                    raise FileExistsError("Preparation outputs already exist; use --overwrite for a deliberate rerun")
            self.write_message(f"Submitting {len(params_list)} ML Data Prep tasks ({execution})...", self.outputs_dir)
            if execution == "dask":
                self.setup_dask("local", processes=cfg["data_prep_processes"],
                                n_workers=cfg["data_prep_workers"], threads_per_worker=1,
                                memory_limit=cfg["data_prep_memory_limit"])
                results = self.client.gather(self.client.map(_prep_tile_task, params_list))
            else:
                results = [_prep_tile_task(params) for params in params_list]
            if not results or any(not item.get("ok") for item in results):
                raise RuntimeError(f"Preparation failed or produced no tasks: {results}")
            self.write_message(f"Successfully prepared {len(results)} / {len(params_list)} tile-pairs.", self.outputs_dir)
            return results
        finally:
            self.cleanup_resources(self.outputs_dir)
