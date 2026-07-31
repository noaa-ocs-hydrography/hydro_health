#!/usr/bin/env python3
"""
MACHINE-LEARNING SEABED ELEVATION-CHANGE MODEL
===============================================

Purpose
-------
This workflow trains machine-learning models that estimate future seafloor
bathymetry (``bathy_t1``) from starting bathymetry (``bathy_t``) and a set of
environmental predictors. Elevation change is calculated after prediction:

    predicted_delta_bathy = predicted_bathy_t1 - bathy_t
    predicted_delta_rate  = predicted_delta_bathy / interval_years

The script is both a production workflow and a training guide. Comments marked
``ML CONCEPT`` explain machine-learning ideas. Comments marked ``PARITY NOTE``
identify choices that intentionally reproduce the established R workflow.

SECTION GUIDE
-------------
PART A. CONFIGURATION, FILESYSTEM, LOGGING, AND SHARED HELPERS
    Reads YAML settings; defines model, raster, Dask, and S3 configuration;
    provides reproducible random seeds; validates targets and predictors;
    preserves missing predictor values; creates weights, spatial folds,
    performance metrics, extrapolation diagnostics, and cloud I/O helpers.

PART B. CLOUD-OPTIMIZED RASTER WRITING AND GDAL OVERVIEWS
    Converts point-based prediction tables back to the supplied raster grid,
    writes tiled/compressed GeoTIFFs, creates reduced-resolution overviews for
    fast GIS display, translates outputs to Cloud Optimized GeoTIFFs (COGs),
    validates them, and uploads only completed files to S3.

PART C. PRIMARY LOCAL TILE TRAINING + IMMEDIATE PREDICTION
    Processes each tile/year pair independently. It loads training and prediction
    GeoParquet data, locks the predictor schema, preserves predictor NaNs, builds
    spatial cross-validation folds, selects the number of boosting rounds,
    applies optional extreme-change weights, trains bootstrap XGBoost models,
    calculates uncertainty/SHAP/importance/metrics, predicts the full tile, and
    writes tables, model files, COG rasters, and global-training shards.

PART D. FULL-EXTENT DEPLOYMENT WITH NEAREST TRAINED LOCAL ENSEMBLE
    Extends local modelling to prediction-only tiles. Each prediction tile uses
    the geographically nearest successfully trained local model ensemble. This
    is model assignment by location, not interpolation between raster cells.

PART E. BALANCED GLOBAL TRAINING BY YEAR PAIR
    Samples standardized rows from many local training shards so large tiles do
    not dominate, trains one bootstrap ensemble per year pair, and stores the
    global models and their locked predictor schema.

PART F. DASK ORCHESTRATION
    Starts or connects to a Dask cluster, submits independent tile/year-pair
    jobs, gathers results, and writes run summaries. XGBoost uses one thread per
    Dask worker to prevent nested CPU oversubscription.

PART G. OPTIONAL ONE-TIME FST -> GEOPARQUET MIGRATION
    Converts legacy local R FST files into interoperable GeoParquet files. This
    migration is separate from production because FST is not cloud-native.

PART H. COMMAND-LINE ENTRY POINT
    Parses command-line arguments and runs local modelling (including standard
    full-extent deployment), global training, global prediction, VRT creation,
    or all stages in sequence.

KEY SCIENTIFIC AND DATA-HANDLING PARITY
---------------------------------------
* Predictor NaNs are retained and passed to XGBoost as missing values.
* Rows are excluded from fitting only when essential response, coordinate,
  starting-bathymetry, delta, or interval fields are unusable.
* ``full_tile_data`` and fitting-only ``subgrid_data`` remain separate.
* Predictor schema defaults to training-first; incompatible deployment data
  raise an explicit error rather than silently changing the model definition.
* Spatial cross-validation selects boosting rounds with a configurable floor.
* Extreme-change weighting affects model fitting, not prediction rows.
* Local training, local deployment, global training, and global prediction are
  distinct stages so they can be validated independently.
* Tables are written as GeoParquet/Parquet and rasters as COGs with overviews.
* S3 is the standard filesystem through s3fs/fsspec.

Dask provides process-level tile parallelism. Each XGBoost model deliberately
uses one thread during parallel execution to avoid CPU oversubscription.
"""

from __future__ import annotations

import argparse
import hashlib
import glob as globlib
import json
import logging
import math
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import dask
import dask.dataframe as dd
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
import s3fs
import xgboost as xgb
import yaml
from dask.distributed import Client, LocalCluster, as_completed
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.transform import Affine
from rasterio.windows import Window, from_bounds
from shapely.geometry import Point, mapping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from osgeo import gdal  # type: ignore
except Exception:  # pragma: no cover - optional Python GDAL bindings
    gdal = None


# =============================================================================
# PART A. CONFIGURATION, FILESYSTEM, LOGGING, AND SHARED HELPERS
# This section defines every user-controlled setting and the reusable building
# blocks used by all later stages. New users should begin by reading the YAML
# file rather than changing defaults inside this script.
# =============================================================================

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
    "geometry", "prediction_row_usable",
}


@dataclass(frozen=True)
class S3Config:
    anon: bool = False
    profile: str | None = None
    endpoint_url: str | None = None
    requester_pays: bool = False
    client_kwargs: dict[str, Any] = field(default_factory=dict)
    config_kwargs: dict[str, Any] = field(default_factory=dict)

    def storage_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "anon": self.anon,
            "requester_pays": self.requester_pays,
        }
        if self.profile:
            options["profile"] = self.profile
        client_kwargs = dict(self.client_kwargs)
        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url
        if client_kwargs:
            options["client_kwargs"] = client_kwargs
        if self.config_kwargs:
            options["config_kwargs"] = self.config_kwargs
        return options


@dataclass(frozen=True)
class PathConfig:
    grid_uri: str
    template_raster_uri: str
    master_predictor_uri: str
    training_root: str
    prediction_root: str
    global_shard_root: str
    global_model_root: str
    log_root: str


@dataclass(frozen=True)
class ModelConfig:
    year_pairs: tuple[str, ...]
    tile_ids: tuple[str, ...] | Literal["all"] = "all"
    tile_field: str = "tile_id"
    source_field: str = "source"
    allowed_training_sources: tuple[str, ...] = ("both", "training", "prediction")
    allowed_prediction_sources: tuple[str, ...] = ("both", "prediction")
    crs: str = "EPSG:32617"
    block_size_m: float = 200.0
    n_boot: int = 5
    n_folds: int = 5
    use_weighted_loss: bool = True
    weight_alpha: float = 1.0
    weight_method: Literal["power", "capped_linear"] = "power"
    weight_cap_quantile: float = 0.95
    max_weight: float = 2.0
    max_depth: int = 4
    eta: float = 0.01
    gamma: float = 0.5
    reg_lambda: float = 0.0
    reg_alpha: float = 0.0
    subsample: float = 0.7
    colsample_bytree: float = 0.8
    pre_xgb_data_mode: Literal["native_missing", "legacy_complete"] = "native_missing"
    predictor_resolution_mode: Literal["og_training_first", "deployment_intersection"] = "og_training_first"
    round_selection_mode: Literal["cv_with_floor", "fixed", "cv_only"] = "cv_with_floor"
    minimum_nrounds: int = 1000
    fixed_nrounds: int = 1000
    cv_nrounds: int = 2000
    early_stopping_rounds: int = 15
    cv_fallback_nrounds: int = 1000
    standard_horizon_years: float = 2.0
    minimum_training_rows: int = 100
    global_seed: int = 12345
    shap_sample_rows: int = 1000
    max_rows_per_global_tile: int = 100_000
    max_global_rows: int = 2_000_000
    deploy_local_full_extent: bool = True
    overwrite_existing_local: bool = False
    write_full_extent_spatial_outputs: bool = True
    build_vrts_after_local: bool = True


@dataclass(frozen=True)
class RasterConfig:
    x_col: str = "X"
    y_col: str = "Y"
    nodata: float = -9999.0
    dtype: str = "float32"
    compress: str = "DEFLATE"
    predictor: int = 3
    blocksize: int = 512
    overview_levels: tuple[int, ...] = (2, 4, 8, 16, 32)
    overview_resampling: str = "average"
    use_gdaladdo: bool = True
    validate_cog: bool = True


@dataclass(frozen=True)
class DaskConfig:
    scheduler_address: str | None = None
    n_workers: int = max(1, (os.cpu_count() or 2) - 1)
    threads_per_worker: int = 1
    memory_limit: str = "auto"
    local_directory: str | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    paths: PathConfig
    model: ModelConfig
    raster: RasterConfig = field(default_factory=RasterConfig)
    dask: DaskConfig = field(default_factory=DaskConfig)
    s3: S3Config = field(default_factory=S3Config)

    @classmethod
    def from_yaml(cls, path: str) -> "WorkflowConfig":
        with open(path, "r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        return cls(
            paths=PathConfig(**raw["paths"]),
            model=ModelConfig(
                **{
                    **raw["model"],
                    "year_pairs": tuple(raw["model"]["year_pairs"]),
                    "tile_ids": (
                        "all" if raw["model"].get("tile_ids", "all") == "all"
                        else tuple(raw["model"]["tile_ids"])
                    ),
                    "allowed_training_sources": tuple(raw["model"].get(
                        "allowed_training_sources", ("both", "training", "prediction")
                    )),
                    "allowed_prediction_sources": tuple(raw["model"].get(
                        "allowed_prediction_sources", ("both", "prediction")
                    )),
                }
            ),
            raster=RasterConfig(**{
                **raw.get("raster", {}),
                "overview_levels": tuple(raw.get("raster", {}).get(
                    "overview_levels", (2, 4, 8, 16, 32)
                )),
            }),
            dask=DaskConfig(**raw.get("dask", {})),
            s3=S3Config(**raw.get("s3", {})),
        )


class CloudFS:
    """Small path abstraction that uses s3fs for S3 and pathlib locally."""

    def __init__(self, config: S3Config):
        self.storage_options = config.storage_options()
        self.s3 = s3fs.S3FileSystem(**self.storage_options)

    @staticmethod
    def is_s3(uri: str) -> bool:
        return uri.startswith("s3://")

    @staticmethod
    def strip_protocol(uri: str) -> str:
        return uri[5:] if uri.startswith("s3://") else uri

    @staticmethod
    def join(root: str, *parts: str) -> str:
        if root.startswith("s3://"):
            return "s3://" + posixpath.join(root[5:].rstrip("/"), *parts)
        return str(Path(root).joinpath(*parts))

    def exists(self, uri: str) -> bool:
        return self.s3.exists(self.strip_protocol(uri)) if self.is_s3(uri) else Path(uri).exists()

    def glob(self, pattern: str) -> list[str]:
        if self.is_s3(pattern):
            return [f"s3://{p}" for p in self.s3.glob(self.strip_protocol(pattern))]
        return [str(p) for p in globlib.glob(pattern, recursive=True)]

    def makedirs(self, uri: str) -> None:
        if not self.is_s3(uri):
            Path(uri).mkdir(parents=True, exist_ok=True)

    def open(self, uri: str, mode: str = "rb"):
        if self.is_s3(uri):
            return self.s3.open(self.strip_protocol(uri), mode)
        Path(uri).parent.mkdir(parents=True, exist_ok=True)
        return open(uri, mode)

    def upload(self, local_path: str, uri: str) -> None:
        if self.is_s3(uri):
            self.s3.put(local_path, self.strip_protocol(uri))
        else:
            Path(uri).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, uri)

    def download(self, uri: str, local_path: str) -> None:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        if self.is_s3(uri):
            self.s3.get(self.strip_protocol(uri), local_path)
        else:
            shutil.copy2(uri, local_path)

    def rm(self, uri: str, recursive: bool = False) -> None:
        if self.is_s3(uri):
            self.s3.rm(self.strip_protocol(uri), recursive=recursive)
        else:
            p = Path(uri)
            if p.is_dir() and recursive:
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()


class LocalizedFile:
    """Download an S3 object to a temporary local file when a library needs a path."""

    def __init__(self, fs: CloudFS, uri: str, suffix: str = ""):
        self.fs = fs
        self.uri = uri
        self.suffix = suffix or Path(uri).suffix
        self.tempdir: tempfile.TemporaryDirectory[str] | None = None
        self.path: str | None = None

    def __enter__(self) -> str:
        if not self.fs.is_s3(self.uri):
            self.path = self.uri
            return self.uri
        self.tempdir = tempfile.TemporaryDirectory(prefix="seabed_localize_")
        self.path = str(Path(self.tempdir.name) / f"source{self.suffix}")
        self.fs.download(self.uri, self.path)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.tempdir:
            self.tempdir.cleanup()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(processName)s | %(message)s",
    )


def stable_seed(global_seed: int, *parts: object) -> int:
    key = "|".join(map(str, parts)).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int((global_seed + int.from_bytes(digest, "little")) % (2**31 - 1))


def pair_duration_years(pair: str) -> float:
    match = re.fullmatch(r"(\d{4})_(\d{4})", pair)
    if not match:
        raise ValueError(f"Invalid year-pair label: {pair}")
    y0, y1 = map(int, match.groups())
    if y1 <= y0:
        raise ValueError(f"Year pair must increase: {pair}")
    return float(y1 - y0)


def safe_numeric_cols(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in set(columns).intersection(out.columns):
        # errors='coerce' mirrors suppressWarnings(as.numeric(as.character(x))).
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_training_targets(df: pd.DataFrame, pair: str, tolerance: float = 1e-8) -> pd.DataFrame:
    """Validate and reconstruct the observed response variables.

    PARITY NOTE: ``delta_bathy`` is always recalculated from bathy_t1 - bathy_t.
    This prevents an old or rounded delta column from becoming a second,
    inconsistent definition of the response.
    """
    out = df.copy()
    missing = {"bathy_t", "bathy_t1"} - set(out.columns)
    if missing:
        raise ValueError(f"Missing required training fields: {sorted(missing)}")
    out = safe_numeric_cols(
        out,
        ["year_t", "year_t1", "bathy_t", "bathy_t1", "delta_bathy", "delta_rate", "interval_years"],
    )
    if {"year_t", "year_t1"}.issubset(out.columns):
        out["interval_years"] = out["year_t1"] - out["year_t"]
    elif "interval_years" not in out.columns:
        out["interval_years"] = np.nan
    fallback = pair_duration_years(pair)
    invalid_interval = ~np.isfinite(out["interval_years"]) | (out["interval_years"] <= 0)
    out.loc[invalid_interval, "interval_years"] = fallback
    recalculated = out["bathy_t1"] - out["bathy_t"]
    if "delta_bathy" in out:
        mismatch = (
            np.isfinite(out["delta_bathy"]) & np.isfinite(recalculated)
            & ((out["delta_bathy"] - recalculated).abs() > tolerance)
        )
        if mismatch.any():
            logging.warning("Replacing %d inconsistent delta_bathy values", int(mismatch.sum()))
    out["delta_bathy"] = recalculated
    out["delta_rate"] = out["delta_bathy"] / out["interval_years"]
    return out


def translate_predictions(
    pred_bathy_t1: np.ndarray,
    bathy_t: np.ndarray,
    interval_years: np.ndarray,
    standard_horizon_years: float,
) -> dict[str, np.ndarray]:
    delta = pred_bathy_t1 - bathy_t
    rate = delta / interval_years
    standard_delta = rate * standard_horizon_years
    return {
        "mean_predicted_bathy_t1": pred_bathy_t1,
        "mean_predicted_change": delta,
        "mean_predicted_rate": rate,
        "mean_predicted_standard_change": standard_delta,
        "mean_predicted_standard_bathy": bathy_t + standard_delta,
    }


def build_pair_specific_predictors(master_predictors: Sequence[str], pair: str) -> list[str]:
    return [f"{p}_{pair}" if p in DYNAMIC_BASES else p for p in master_predictors]


# ML CONCEPT: A model is only reproducible when predictor names and order are
# locked. Silently dropping a missing deployment predictor would create a
# different model, so training-first mode raises a clear schema error instead.
def resolve_predictors(
    pair: str,
    master_predictors: Sequence[str],
    train_columns: Sequence[str],
    prediction_columns: Sequence[str] | None,
    mode: str,
) -> tuple[list[str], list[str], list[str]]:
    candidates = list(dict.fromkeys(["bathy_t", *build_pair_specific_predictors(master_predictors, pair)]))
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
        raise ValueError(
            "Training-first schema cannot deploy because prediction data are missing: "
            + ", ".join(missing_prediction)
        )
    return resolved, missing_training, missing_prediction


def summarize_missingness(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "predictor": predictors,
        "n_rows": len(df),
        "n_na": [int(df[p].isna().sum()) for p in predictors],
        "proportion_na": [float(df[p].isna().mean()) for p in predictors],
        "n_finite": [int(np.isfinite(df[p].to_numpy(dtype=float)).sum()) for p in predictors],
    })


# ML CONCEPT: Sample weighting changes how strongly each observed row affects
# the fitted loss. Here, larger surveyed changes can receive more influence, but
# weights are never invented for prediction rows because their true change is unknown.
def calculate_sample_weights(
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


def common_metrics(obs_t1: np.ndarray, pred_t1: np.ndarray, bathy_t: np.ndarray, interval: np.ndarray) -> dict[str, float]:
    obs_delta = obs_t1 - bathy_t
    pred_delta = pred_t1 - bathy_t
    obs_rate = obs_delta / interval
    pred_rate = pred_delta / interval

    def finite_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.isfinite(a) & np.isfinite(b)

    state = finite_pair(obs_t1, pred_t1)
    delta = finite_pair(obs_delta, pred_delta)
    rate = finite_pair(obs_rate, pred_rate)

    def metric_or_nan(mask: np.ndarray, fn) -> float:
        return float(fn(mask)) if mask.sum() else float("nan")

    hotspot = np.zeros_like(delta, dtype=bool)
    if delta.sum() > 10:
        cut = np.nanquantile(np.abs(obs_delta[delta]), 0.90)
        hotspot = delta & (np.abs(obs_delta) >= cut)

    return {
        "n_state": int(state.sum()),
        "n_delta": int(delta.sum()),
        "rmse_bathy_t1": metric_or_nan(state, lambda m: math.sqrt(mean_squared_error(obs_t1[m], pred_t1[m]))),
        "mae_bathy_t1": metric_or_nan(state, lambda m: mean_absolute_error(obs_t1[m], pred_t1[m])),
        "bias_bathy_t1": metric_or_nan(state, lambda m: np.mean(pred_t1[m] - obs_t1[m])),
        "r2_predictive_bathy_t1": metric_or_nan(state, lambda m: r2_score(obs_t1[m], pred_t1[m])),
        "r2_correlation_bathy_t1": metric_or_nan(state, lambda m: np.corrcoef(obs_t1[m], pred_t1[m])[0, 1] ** 2),
        "rmse_delta": metric_or_nan(delta, lambda m: math.sqrt(mean_squared_error(obs_delta[m], pred_delta[m]))),
        "mae_delta": metric_or_nan(delta, lambda m: mean_absolute_error(obs_delta[m], pred_delta[m])),
        "bias_delta": metric_or_nan(delta, lambda m: np.mean(pred_delta[m] - obs_delta[m])),
        "delta_correlation": metric_or_nan(delta, lambda m: np.corrcoef(obs_delta[m], pred_delta[m])[0, 1]),
        "rmse_rate": metric_or_nan(rate, lambda m: math.sqrt(mean_squared_error(obs_rate[m], pred_rate[m]))),
        "mae_rate": metric_or_nan(rate, lambda m: mean_absolute_error(obs_rate[m], pred_rate[m])),
        "bias_rate": metric_or_nan(rate, lambda m: np.mean(pred_rate[m] - obs_rate[m])),
        "sign_accuracy_delta": metric_or_nan(delta, lambda m: np.mean(np.sign(pred_delta[m]) == np.sign(obs_delta[m]))),
        "hotspot_rmse_delta": metric_or_nan(hotspot, lambda m: math.sqrt(mean_squared_error(obs_delta[m], pred_delta[m]))),
        "predicted_erosion_proportion": metric_or_nan(delta, lambda m: np.mean(pred_delta[m] < 0)),
        "predicted_deposition_proportion": metric_or_nan(delta, lambda m: np.mean(pred_delta[m] > 0)),
    }


def predictor_range_table(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    rows = []
    for predictor in predictors:
        values = pd.to_numeric(df[predictor], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            "predictor": predictor,
            "min_training": float(np.min(finite)) if finite.size else np.nan,
            "max_training": float(np.max(finite)) if finite.size else np.nan,
            "mean_training": float(np.mean(finite)) if finite.size else np.nan,
        })
    return pd.DataFrame(rows)


def add_extrapolation_flags(df: pd.DataFrame, ranges: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    count = np.zeros(len(out), dtype=np.int16)
    for row in ranges.itertuples(index=False):
        if row.predictor not in out or not np.isfinite(row.min_training) or not np.isfinite(row.max_training):
            continue
        values = pd.to_numeric(out[row.predictor], errors="coerce").to_numpy(dtype=float)
        count += (np.isfinite(values) & ((values < row.min_training) | (values > row.max_training))).astype(np.int16)
    out["predictor_extrapolation_count"] = count
    out["predictor_extrapolation_any"] = count > 0
    return out


def spatial_fold_ids(df: pd.DataFrame, block_size_m: float, n_folds: int, seed: int) -> np.ndarray:
    """Create spatially separated cross-validation folds.

    ML CONCEPT: Random train/test splits can place neighboring seabed cells in
    both sets, making performance look unrealistically strong. Spatial blocks
    test whether the model transfers to nearby-but-unseen areas.
    """
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


def xgb_params(cfg: ModelConfig, nthread: int = 1) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse", "mae"],
        "max_depth": cfg.max_depth,
        "eta": cfg.eta,
        "gamma": cfg.gamma,
        "lambda": cfg.reg_lambda,
        "alpha": cfg.reg_alpha,
        "subsample": cfg.subsample,
        "colsample_bytree": cfg.colsample_bytree,
        "nthread": nthread,
        "tree_method": "hist",
    }


# ML CONCEPT: Boosting rounds are the number of sequential trees. Too few can
# underfit; too many can overfit. Spatial CV estimates a useful value, while the
# configured floor preserves the accuracy-parity safeguard found in R testing.
def choose_rounds_spatial_cv(
    df: pd.DataFrame,
    predictors: Sequence[str],
    weights: np.ndarray,
    cfg: ModelConfig,
    seed: int,
) -> tuple[int, pd.DataFrame]:
    if cfg.round_selection_mode == "fixed":
        return cfg.fixed_nrounds, pd.DataFrame([{"cv_status": "fixed", "best_iteration": cfg.fixed_nrounds}])
    folds = spatial_fold_ids(df, cfg.block_size_m, cfg.n_folds, seed)
    unique_folds = np.unique(folds)
    if unique_folds.size < 2:
        fallback = max(cfg.minimum_nrounds, cfg.cv_fallback_nrounds) if cfg.round_selection_mode == "cv_with_floor" else cfg.cv_fallback_nrounds
        return fallback, pd.DataFrame([{"cv_status": "insufficient_spatial_folds", "best_iteration": fallback}])

    rows: list[dict[str, Any]] = []
    best: list[int] = []
    features = df[list(predictors)].to_numpy(dtype=np.float32)
    labels = df["bathy_t1"].to_numpy(dtype=np.float32)
    for fold in unique_folds:
        test = folds == fold
        train = ~test
        if train.sum() < 20 or test.sum() < 10:
            rows.append({"fold": int(fold), "cv_status": "insufficient_rows"})
            continue
        dtrain = xgb.DMatrix(features[train], label=labels[train], weight=weights[train], missing=np.nan)
        dtest = xgb.DMatrix(features[test], label=labels[test], missing=np.nan)
        model = xgb.train(
            xgb_params(cfg, nthread=1), dtrain,
            num_boost_round=cfg.cv_nrounds,
            evals=[(dtrain, "train"), (dtest, "test")],
            early_stopping_rounds=cfg.early_stopping_rounds,
            verbose_eval=False,
        )
        best_iteration = int(getattr(model, "best_iteration", -1)) + 1
        if best_iteration <= 0:
            best_iteration = cfg.cv_fallback_nrounds
        best.append(best_iteration)
        pred = model.predict(dtest, iteration_range=(0, best_iteration))
        metrics = common_metrics(
            labels[test], pred,
            df.loc[test, "bathy_t"].to_numpy(dtype=float),
            df.loc[test, "interval_years"].to_numpy(dtype=float),
        )
        rows.append({"fold": int(fold), "cv_status": "ok", "best_iteration": best_iteration, **metrics})
    if best:
        selected = int(round(float(np.mean(best))))
        if cfg.round_selection_mode == "cv_with_floor":
            selected = max(cfg.minimum_nrounds, selected)
        else:
            selected = max(50, selected)
    else:
        selected = cfg.cv_fallback_nrounds
        if cfg.round_selection_mode == "cv_with_floor":
            selected = max(cfg.minimum_nrounds, selected)
    return selected, pd.DataFrame(rows)


def read_master_predictors(fs: CloudFS, uri: str) -> list[str]:
    with fs.open(uri, "rt") as stream:
        return list(dict.fromkeys(line.strip() for line in stream if line.strip()))


def read_table(uri: str, fs: CloudFS, columns: Sequence[str] | None = None) -> pd.DataFrame:
    lower = uri.lower()
    if lower.endswith((".parquet", ".geoparquet")):
        return pd.read_parquet(uri, columns=columns, storage_options=fs.storage_options)
    if lower.endswith(".csv"):
        with fs.open(uri, "rb") as stream:
            return pd.read_csv(stream, usecols=columns)
    if lower.endswith(".fst"):
        raise ValueError(
            "FST is R-specific. Convert source FST files once to Parquet/GeoParquet "
            "with the supplied migration stage before cloud production runs."
        )
    raise ValueError(f"Unsupported table format: {uri}")


def dataframe_to_geoparquet(
    df: pd.DataFrame,
    uri: str,
    fs: CloudFS,
    crs: str,
    x_col: str = "X",
    y_col: str = "Y",
    compression: str = "zstd",
) -> None:
    out = df.copy()
    valid_xy = np.isfinite(pd.to_numeric(out[x_col], errors="coerce")) & np.isfinite(pd.to_numeric(out[y_col], errors="coerce"))
    geometry = gpd.GeoSeries([Point(x, y) if ok else None for x, y, ok in zip(out[x_col], out[y_col], valid_xy)], crs=crs)
    gdf = gpd.GeoDataFrame(out, geometry=geometry, crs=crs)
    gdf.to_parquet(uri, index=False, compression=compression, storage_options=fs.storage_options)


def write_parquet(df: pd.DataFrame, uri: str, fs: CloudFS, compression: str = "zstd") -> None:
    df.to_parquet(uri, index=False, compression=compression, storage_options=fs.storage_options)


def write_json(payload: dict[str, Any], uri: str, fs: CloudFS) -> None:
    with fs.open(uri, "wt") as stream:
        json.dump(payload, stream, indent=2, default=str)


def find_tile_file(fs: CloudFS, tile_root: str, tile_id: str, pair: str, kind: Literal["train", "pred"]) -> str | None:
    tile_dir = fs.join(tile_root, tile_id)
    suffixes = (
        [f"*_{pair}_long.parquet", f"*_{pair}_train_long.parquet", f"*_{pair}_long.geoparquet"]
        if kind == "train"
        else [f"*_{pair}_prediction_long.parquet", f"*_{pair}_pred_long.parquet", f"*_{pair}_prediction_long.geoparquet"]
    )
    hits: list[str] = []
    for suffix in suffixes:
        hits.extend(fs.glob(fs.join(tile_dir, suffix)))
    hits = sorted(set(hits), key=lambda p: (len(Path(p).name), Path(p).name))
    return hits[0] if hits else None


def load_grid(cfg: WorkflowConfig, fs: CloudFS) -> gpd.GeoDataFrame:
    with LocalizedFile(fs, cfg.paths.grid_uri, Path(cfg.paths.grid_uri).suffix) as local:
        grid = gpd.read_file(local)
    if grid.crs is None:
        grid = grid.set_crs(cfg.model.crs)
    else:
        grid = grid.to_crs(cfg.model.crs)
    grid[cfg.model.tile_field] = grid[cfg.model.tile_field].astype(str)
    return grid


def select_tiles(grid: gpd.GeoDataFrame, cfg: ModelConfig, prediction: bool = False) -> gpd.GeoDataFrame:
    out = grid.copy()
    allowed = cfg.allowed_prediction_sources if prediction else cfg.allowed_training_sources
    if cfg.source_field in out.columns:
        out = out[out[cfg.source_field].isin(allowed)]
    if cfg.tile_ids != "all":
        out = out[out[cfg.tile_field].isin(cfg.tile_ids)]
    if out.empty:
        raise ValueError("Tile selection returned zero tiles")
    return out


# =============================================================================
# PART B. CLOUD-OPTIMIZED RASTER WRITING AND GDAL OVERVIEWS
# Machine-learning predictions are stored in tables first. This section maps the
# X/Y values back onto the exact template grid and prepares GIS-friendly COGs.
# =============================================================================

def _overview_resampling(name: str) -> Resampling:
    try:
        return Resampling[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported overview resampling: {name}") from exc


def _template_profile(template_uri: str, fs: CloudFS) -> tuple[dict[str, Any], Affine]:
    with rasterio.Env(AWS_REQUEST_PAYER="requester" if fs.storage_options.get("requester_pays") else None):
        with rasterio.open(template_uri) as src:
            return src.profile.copy(), src.transform


def dataframe_to_grid(
    df: pd.DataFrame,
    value_col: str,
    template_uri: str,
    fs: CloudFS,
    raster_cfg: RasterConfig,
    tile_geometry_wkb: bytes | None = None,
    tile_geometry_crs: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rasterize point predictions onto a seam-safe, template-aligned tile.

    PARITY NOTE — authoritative footprint:
    Prediction tables may contain a few points beyond the official model-grid
    polygon. Those points are useful in tables but must never create valid cells
    in an adjacent raster tile. When a tile geometry is supplied, its polygon —
    not the point extent — determines the output window and valid-data mask.
    """
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(df[raster_cfg.x_col], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(df[raster_cfg.y_col], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    if not keep.any():
        raise ValueError(f"No finite X/Y/{value_col} rows available for rasterization")

    with rasterio.open(template_uri) as template:
        tile_geom = None
        if tile_geometry_wkb is not None:
            if not tile_geometry_crs:
                raise ValueError("tile_geometry_crs is required with tile_geometry_wkb")
            tile_geom = gpd.GeoSeries.from_wkb(
                [tile_geometry_wkb], crs=tile_geometry_crs
            ).to_crs(template.crs).iloc[0]
            if tile_geom is None or tile_geom.is_empty:
                raise ValueError("Tile geometry is empty")

            # from_bounds uses the polygon extent, while round_offsets/lengths
            # and intersection preserve the exact origin/cell alignment of the
            # common template raster (equivalent to crop(..., snap='out') in R).
            raw_window = from_bounds(*tile_geom.bounds, transform=template.transform)
            window = raw_window.round_offsets(op="floor").round_lengths(op="ceil")
            window = window.intersection(Window(0, 0, template.width, template.height))
            r0, c0 = int(window.row_off), int(window.col_off)
            height, width = int(window.height), int(window.width)
            if height <= 0 or width <= 0:
                raise ValueError("Tile polygon does not intersect the template raster")
        else:
            # Backward-compatible non-tiled fallback. Production tile outputs
            # should always supply the official grid geometry.
            rows_all, cols_all = rasterio.transform.rowcol(
                template.transform, xs[keep], ys[keep]
            )
            rows_all, cols_all = np.asarray(rows_all), np.asarray(cols_all)
            inside = (rows_all >= 0) & (rows_all < template.height) & (cols_all >= 0) & (cols_all < template.width)
            if not inside.any():
                raise ValueError("All coordinates fall outside the template raster")
            r0, r1 = int(rows_all[inside].min()), int(rows_all[inside].max())
            c0, c1 = int(cols_all[inside].min()), int(cols_all[inside].max())
            height, width = r1 - r0 + 1, c1 - c0 + 1
            window = Window(c0, r0, width, height)

        local_transform = template.window_transform(window)
        rows, cols = rasterio.transform.rowcol(local_transform, xs[keep], ys[keep])
        rows, cols = np.asarray(rows), np.asarray(cols)
        inside_local = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        vals = values[keep][inside_local]
        rows, cols = rows[inside_local], cols[inside_local]

        accum = np.zeros((height, width), dtype=np.float64)
        counts = np.zeros((height, width), dtype=np.uint32)
        if len(vals):
            np.add.at(accum, (rows, cols), vals)
            np.add.at(counts, (rows, cols), 1)
        grid = np.full((height, width), raster_cfg.nodata, dtype=np.float32)
        populated = counts > 0
        grid[populated] = (accum[populated] / counts[populated]).astype(np.float32)

        if tile_geom is not None:
            # Critical seam-safe step: even if prediction points extend beyond
            # the tile, only cells whose centers fall inside the official grid
            # polygon remain valid. Neighboring COGs therefore cannot overlap
            # with competing valid values in a VRT.
            valid_polygon = geometry_mask(
                [mapping(tile_geom)],
                out_shape=(height, width),
                transform=local_transform,
                invert=True,
                all_touched=False,
            )
            grid[~valid_polygon] = raster_cfg.nodata

        profile = template.profile.copy()
        profile.update(
            driver="GTiff", count=1, width=width, height=height,
            transform=local_transform, dtype=raster_cfg.dtype,
            nodata=raster_cfg.nodata,
        )
    return grid, profile


def build_overviews(local_tif: str, cfg: RasterConfig) -> None:
    valid_levels = []
    with rasterio.open(local_tif, "r+") as dst:
        min_dim = min(dst.width, dst.height)
        valid_levels = [level for level in cfg.overview_levels if min_dim // level >= 1]
        if valid_levels:
            dst.build_overviews(valid_levels, _overview_resampling(cfg.overview_resampling))
            dst.update_tags(ns="rio_overview", resampling=cfg.overview_resampling)
    if cfg.use_gdaladdo and valid_levels and shutil.which("gdaladdo"):
        subprocess.run(
            ["gdaladdo", "-r", cfg.overview_resampling, local_tif, *map(str, valid_levels)],
            check=True,
            capture_output=True,
            text=True,
        )


def write_cog_from_dataframe(
    df: pd.DataFrame,
    value_col: str,
    output_uri: str,
    template_uri: str,
    fs: CloudFS,
    cfg: RasterConfig,
    tile_geometry_wkb: bytes | None = None,
    tile_geometry_crs: str | None = None,
) -> str:
    grid, profile = dataframe_to_grid(
        df, value_col, template_uri, fs, cfg,
        tile_geometry_wkb=tile_geometry_wkb,
        tile_geometry_crs=tile_geometry_crs,
    )
    with tempfile.TemporaryDirectory(prefix="seabed_cog_") as tmp:
        raw = str(Path(tmp) / "raw.tif")
        cog = str(Path(tmp) / "cog.tif")
        profile.update(
            tiled=True,
            blockxsize=min(cfg.blocksize, profile["width"]),
            blockysize=min(cfg.blocksize, profile["height"]),
            compress=cfg.compress,
            predictor=cfg.predictor,
            BIGTIFF="IF_SAFER",
        )
        # GTiff block dimensions must be multiples of 16.
        profile["blockxsize"] = max(16, int(math.ceil(profile["blockxsize"] / 16) * 16))
        profile["blockysize"] = max(16, int(math.ceil(profile["blockysize"] / 16) * 16))
        profile["blockxsize"] = min(profile["blockxsize"], max(16, int(math.ceil(profile["width"] / 16) * 16)))
        profile["blockysize"] = min(profile["blockysize"], max(16, int(math.ceil(profile["height"] / 16) * 16)))
        with rasterio.open(raw, "w", **profile) as dst:
            dst.write(grid, 1)
        build_overviews(raw, cfg)
        # Use GDAL COG driver where available; otherwise retain tiled GeoTIFF + overviews.
        if gdal is not None:
            options = gdal.TranslateOptions(
                format="COG",
                creationOptions=[
                    f"COMPRESS={cfg.compress}",
                    f"BLOCKSIZE={cfg.blocksize}",
                    "BIGTIFF=IF_SAFER",
                    "OVERVIEWS=AUTO",
                ],
            )
            result = gdal.Translate(cog, raw, options=options)
            if result is None:
                raise RuntimeError("GDAL COG translation failed")
            result = None
        elif shutil.which("gdal_translate"):
            subprocess.run([
                "gdal_translate", "-of", "COG",
                "-co", f"COMPRESS={cfg.compress}",
                "-co", f"BLOCKSIZE={cfg.blocksize}",
                "-co", "BIGTIFF=IF_SAFER",
                raw, cog,
            ], check=True, capture_output=True, text=True)
        else:
            shutil.copy2(raw, cog)
        fs.upload(cog, output_uri)
    return output_uri


# =============================================================================
# PART C. PRIMARY LOCAL TILE TRAINING + IMMEDIATE PREDICTION
# This is the main modelling section. One independent task is run for each tile
# and year pair, making failures easier to isolate and cloud scaling predictable.
# =============================================================================

def train_local_tile_pair(tile_id: str, pair: str, tile_geometry_wkb: bytes, tile_geometry_crs: str, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = workflow_config_from_dict(cfg_dict)
    fs = CloudFS(cfg.s3)
    interval = pair_duration_years(pair)
    seed = stable_seed(cfg.model.global_seed, "local", tile_id, pair)
    train_uri = find_tile_file(fs, cfg.paths.training_root, tile_id, pair, "train")
    pred_uri = find_tile_file(fs, cfg.paths.prediction_root, tile_id, pair, "pred")
    if train_uri is None:
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": "missing_training_file"}

    master_predictors = read_master_predictors(fs, cfg.paths.master_predictor_uri)
    training_raw = ensure_training_targets(read_table(train_uri, fs), pair)
    training_raw["pair_id"] = pair
    training_raw["tile_id"] = tile_id
    prediction_available = pred_uri is not None
    prediction_raw = read_table(pred_uri, fs) if pred_uri else None
    if prediction_raw is not None:
        prediction_raw["interval_years"] = interval
        prediction_raw["pair_id"] = pair
        prediction_raw["tile_id"] = tile_id

    predictors, missing_train, missing_pred = resolve_predictors(
        pair, master_predictors, training_raw.columns,
        prediction_raw.columns if prediction_raw is not None else None,
        cfg.model.predictor_resolution_mode,
    )
    numeric = set(predictors) | {"X", "Y", "FID", "bathy_t", "bathy_t1", "delta_bathy", "delta_rate", "interval_years"}
    training_raw = safe_numeric_cols(training_raw, numeric)
    if prediction_raw is not None:
        prediction_raw = safe_numeric_cols(prediction_raw, set(predictors) | {"X", "Y", "FID", "bathy_t", "interval_years"})
        prediction_raw["prediction_row_usable"] = (
            np.isfinite(prediction_raw["X"]) & np.isfinite(prediction_raw["Y"]) & np.isfinite(prediction_raw["bathy_t"])
        )

    tile_train_root = fs.join(cfg.paths.training_root, tile_id)
    tile_pred_root = fs.join(cfg.paths.prediction_root, tile_id)
    write_parquet(summarize_missingness(training_raw, predictors), fs.join(tile_train_root, f"predictor_missingness_training_{pair}.parquet"), fs)
    if prediction_raw is not None:
        write_parquet(summarize_missingness(prediction_raw, predictors), fs.join(tile_pred_root, f"predictor_missingness_prediction_{pair}.parquet"), fs)

    # PARITY NOTE: Keep two datasets with different jobs.
    # full_tile_data preserves every selected/coerced row for final reconstruction.
    # subgrid_data below contains only rows eligible to teach the model. Predictor
    # NaNs remain present in native_missing mode and are handled internally by XGBoost.
    full_tile_data = training_raw.copy()
    essential = (
        np.isfinite(full_tile_data["X"]) & np.isfinite(full_tile_data["Y"])
        & np.isfinite(full_tile_data["bathy_t"]) & np.isfinite(full_tile_data["bathy_t1"])
        & np.isfinite(full_tile_data["delta_bathy"]) & np.isfinite(full_tile_data["interval_years"])
        & (full_tile_data["interval_years"] > 0)
    )
    if cfg.model.pre_xgb_data_mode == "legacy_complete":
        predictor_complete = np.isfinite(full_tile_data[predictors].to_numpy(dtype=float)).all(axis=1)
        model_keep = essential & predictor_complete
    else:
        model_keep = essential
    subgrid_data = full_tile_data.loc[model_keep].copy().reset_index(drop=True)
    if len(subgrid_data) < cfg.model.minimum_training_rows:
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": "insufficient_training_rows", "n": len(subgrid_data)}

    # ML CONCEPT: Weighting emphasizes scientifically important large changes.
    # It does not duplicate rows and does not alter the observed target values.
    weights = (
        calculate_sample_weights(
            subgrid_data["delta_bathy"].to_numpy(dtype=float),
            cfg.model.weight_alpha, cfg.model.weight_method,
            cfg.model.weight_cap_quantile, cfg.model.max_weight,
        ) if cfg.model.use_weighted_loss else np.ones(len(subgrid_data))
    )
    best_rounds, cv_metrics = choose_rounds_spatial_cv(subgrid_data, predictors, weights, cfg.model, stable_seed(seed, "cv"))
    cv_metrics["tile_id"] = tile_id
    cv_metrics["year_pair"] = pair
    write_parquet(cv_metrics, fs.join(tile_train_root, f"spatial_cv_metrics_{pair}.parquet"), fs)

    x_fit = subgrid_data[predictors].to_numpy(dtype=np.float32)
    y_fit = subgrid_data["bathy_t1"].to_numpy(dtype=np.float32)
    x_full = full_tile_data[predictors].to_numpy(dtype=np.float32)
    x_pred = prediction_raw[predictors].to_numpy(dtype=np.float32) if prediction_raw is not None else None
    boot_full = np.full((len(full_tile_data), cfg.model.n_boot), np.nan, dtype=np.float32)
    boot_pred = np.full((len(prediction_raw), cfg.model.n_boot), np.nan, dtype=np.float32) if prediction_raw is not None else None
    importance_rows: list[dict[str, Any]] = []
    shap_rows: list[dict[str, Any]] = []
    model_uris: list[str] = []

    d_full = xgb.DMatrix(x_full, feature_names=predictors, missing=np.nan)
    d_pred = xgb.DMatrix(x_pred, feature_names=predictors, missing=np.nan) if x_pred is not None else None
    rng = np.random.default_rng(seed)
    # ML CONCEPT: Bootstrap models are trained on repeated samples of the same
    # training data. Their mean is the ensemble prediction; their spread (SD) is
    # a model-instability indicator, not a complete measure of all uncertainty.
    for b in range(cfg.model.n_boot):
        indices = rng.choice(len(subgrid_data), size=len(subgrid_data), replace=True)
        dtrain = xgb.DMatrix(
            x_fit[indices], label=y_fit[indices], weight=weights[indices],
            feature_names=predictors, missing=np.nan,
        )
        model = xgb.train(xgb_params(cfg.model, nthread=1), dtrain, num_boost_round=best_rounds, verbose_eval=False)
        boot_full[:, b] = model.predict(d_full)
        if d_pred is not None and boot_pred is not None:
            boot_pred[:, b] = model.predict(d_pred)
        score = model.get_score(importance_type="gain")
        importance_rows.extend({"predictor": p, "gain": float(score.get(p, 0.0)), "bootstrap_iteration": b + 1} for p in predictors)
        shap_n = min(cfg.model.shap_sample_rows, len(subgrid_data))
        shap_idx = rng.choice(len(subgrid_data), size=shap_n, replace=False)
        dshap = xgb.DMatrix(x_fit[shap_idx], feature_names=predictors, missing=np.nan)
        # ML CONCEPT: SHAP contributions estimate how strongly each predictor
        # moved a prediction away from the model baseline. We summarize absolute
        # contributions to describe influence, not causation.
        contributions = model.predict(dshap, pred_contribs=True)[:, :-1]
        shap_rows.extend({"predictor": p, "mean_abs_shap": float(np.nanmean(np.abs(contributions[:, j]))), "bootstrap_iteration": b + 1} for j, p in enumerate(predictors))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_model:
            model_path = temp_model.name
        try:
            model.save_model(model_path)
            model_uri = fs.join(tile_train_root, "bootstrap_models", f"model_boot_{b + 1:03d}_{pair}.json")
            fs.upload(model_path, model_uri)
            model_uris.append(model_uri)
        finally:
            Path(model_path).unlink(missing_ok=True)

    full_mean = np.nanmean(boot_full, axis=1)
    full_sd = np.nanstd(boot_full, axis=1, ddof=1 if cfg.model.n_boot > 1 else 0)
    translated = translate_predictions(
        full_mean,
        full_tile_data["bathy_t"].to_numpy(dtype=float),
        full_tile_data["interval_years"].to_numpy(dtype=float),
        cfg.model.standard_horizon_years,
    )
    training_summary = full_tile_data.copy()
    for key, value in translated.items():
        training_summary[key] = value
    training_summary["uncertainty_sd_bathy_t1"] = full_sd
    training_summary["prediction_residual"] = training_summary["bathy_t1"] - training_summary["mean_predicted_bathy_t1"]
    training_summary["delta_residual"] = training_summary["delta_bathy"] - training_summary["mean_predicted_change"]
    ranges = predictor_range_table(subgrid_data, predictors)
    metrics = common_metrics(
        training_summary["bathy_t1"].to_numpy(dtype=float),
        training_summary["mean_predicted_bathy_t1"].to_numpy(dtype=float),
        training_summary["bathy_t"].to_numpy(dtype=float),
        training_summary["interval_years"].to_numpy(dtype=float),
    )
    metrics.update(tile_id=tile_id, year_pair=pair, evaluation="apparent_full_tile", best_iteration=best_rounds)

    train_summary_uri = fs.join(tile_train_root, f"LOCAL_training_summary_{pair}.geoparquet")
    dataframe_to_geoparquet(training_summary, train_summary_uri, fs, cfg.model.crs)
    write_parquet(pd.DataFrame([metrics]), fs.join(tile_train_root, f"LOCAL_training_metrics_{pair}.parquet"), fs)
    write_parquet(ranges, fs.join(tile_train_root, f"LOCAL_predictor_ranges_{pair}.parquet"), fs)
    write_parquet(pd.DataFrame(importance_rows), fs.join(tile_train_root, f"LOCAL_importance_{pair}.parquet"), fs)
    write_parquet(pd.DataFrame(shap_rows), fs.join(tile_train_root, f"LOCAL_shap_summary_{pair}.parquet"), fs)

    prediction_summary_uri = None
    if prediction_raw is not None and boot_pred is not None:
        pred_mean = np.nanmean(boot_pred, axis=1)
        pred_sd = np.nanstd(boot_pred, axis=1, ddof=1 if cfg.model.n_boot > 1 else 0)
        translated_pred = translate_predictions(
            pred_mean,
            prediction_raw["bathy_t"].to_numpy(dtype=float),
            prediction_raw["interval_years"].to_numpy(dtype=float),
            cfg.model.standard_horizon_years,
        )
        prediction_summary = prediction_raw.copy()
        for key, value in translated_pred.items():
            prediction_summary[key] = value
        prediction_summary["uncertainty_sd_bathy_t1"] = pred_sd
        # ML CONCEPT: Extrapolation flags identify predictor values outside the
        # observed training range. A flag is a caution for interpretation; it does
        # not automatically mean the prediction is wrong.
        prediction_summary = add_extrapolation_flags(prediction_summary, ranges)
        prediction_summary_uri = fs.join(tile_pred_root, f"LOCAL_prediction_summary_{pair}.geoparquet")
        dataframe_to_geoparquet(prediction_summary, prediction_summary_uri, fs, cfg.model.crs)
        raster_products = {
            "bathy_t": "LOCAL_Pred_Start_Bathy_t",
            "mean_predicted_bathy_t1": "LOCAL_Pred_Mean_Bathy_t1",
            "mean_predicted_change": "LOCAL_Pred_Mean_Change",
            "uncertainty_sd_bathy_t1": "LOCAL_Pred_SD_Bathy_t1",
            "predictor_extrapolation_count": "LOCAL_Predictor_Extrapolation_Count",
        }
        if "UC_t" in prediction_summary:
            raster_products["UC_t"] = "LOCAL_Pred_BlueTopo_UC_t"
        for value_col, prefix in raster_products.items():
            write_cog_from_dataframe(
                prediction_summary, value_col,
                fs.join(tile_pred_root, f"{prefix}_{pair}.tif"),
                cfg.paths.template_raster_uri, fs, cfg.raster,
                tile_geometry_wkb, tile_geometry_crs,
            )

    train_rasters = {
        "bathy_t": "LOCAL_Train_Start_Bathy_t",
        "mean_predicted_bathy_t1": "LOCAL_Train_Mean_Predicted_Bathy_t1",
        "delta_bathy": "LOCAL_Train_Surveyed_Change",
        "mean_predicted_change": "LOCAL_Train_Mean_Predicted_Change",
        "uncertainty_sd_bathy_t1": "LOCAL_Train_SD_Bathy_t1",
    }
    for value_col, prefix in train_rasters.items():
        write_cog_from_dataframe(
            training_summary, value_col,
            fs.join(tile_train_root, f"{prefix}_{pair}.tif"),
            cfg.paths.template_raster_uri, fs, cfg.raster,
            tile_geometry_wkb, tile_geometry_crs,
        )

    # Standardized global-training shard: include only fitting-eligible rows,
    # while retaining predictor NaNs. Identical schemas make later balanced global
    # sampling transparent and prevent accidental feature-order changes.
    shard = subgrid_data[["tile_id", "pair_id", "X", "Y", "FID", "bathy_t1", "delta_bathy", "delta_rate", "interval_years", *predictors]].copy()
    shard_uri = fs.join(cfg.paths.global_shard_root, pair, f"{tile_id}_{pair}_global_training_shard.parquet")
    write_parquet(shard, shard_uri, fs)
    metadata = {
        "tile_id": tile_id,
        "pair": pair,
        "predictors": predictors,
        "missing_in_training": missing_train,
        "missing_in_prediction": missing_pred,
        "best_iteration": best_rounds,
        "model_uris": model_uris,
        "training_summary_uri": train_summary_uri,
        "prediction_summary_uri": prediction_summary_uri,
        "predictor_ranges_uri": fs.join(tile_train_root, f"LOCAL_predictor_ranges_{pair}.parquet"),
        "crs": cfg.model.crs,
        "pre_xgb_data_mode": cfg.model.pre_xgb_data_mode,
        "predictor_resolution_mode": cfg.model.predictor_resolution_mode,
    }
    write_json(metadata, fs.join(tile_train_root, f"LOCAL_model_metadata_{pair}.json"), fs)
    return {"ok": True, "tile_id": tile_id, "pair": pair, "n_full": len(full_tile_data), "n_fit": len(subgrid_data), "best_iteration": best_rounds}


# =============================================================================
# PART D. FULL-EXTENT DEPLOYMENT WITH NEAREST TRAINED LOCAL ENSEMBLE
# Local models only learn from their own training tile. Prediction-only tiles are
# assigned the nearest trained ensemble so local relationships can be extended.
# =============================================================================

def load_xgb_model(fs: CloudFS, uri: str) -> xgb.Booster:
    with LocalizedFile(fs, uri, ".json") as local:
        model = xgb.Booster()
        model.load_model(local)
        return model


def predict_tile_with_model_ensemble(
    tile_id: str,
    pair: str,
    source_model_tile: str,
    tile_geometry_wkb: bytes,
    tile_geometry_crs: str,
    cfg_dict: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    cfg = workflow_config_from_dict(cfg_dict)
    fs = CloudFS(cfg.s3)
    out_uri = fs.join(cfg.paths.prediction_root, tile_id, f"LOCAL_prediction_summary_{pair}.geoparquet")
    if fs.exists(out_uri) and not overwrite:
        return {"ok": True, "tile_id": tile_id, "pair": pair, "status": "existing_retained"}
    meta_uri = fs.join(cfg.paths.training_root, source_model_tile, f"LOCAL_model_metadata_{pair}.json")
    with fs.open(meta_uri, "rt") as stream:
        metadata = json.load(stream)
    pred_uri = find_tile_file(fs, cfg.paths.prediction_root, tile_id, pair, "pred")
    if pred_uri is None:
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": "missing_prediction_file"}
    pred = read_table(pred_uri, fs)
    predictors = metadata["predictors"]
    missing = [p for p in predictors if p not in pred]
    if missing:
        raise ValueError(f"Prediction tile {tile_id} missing predictors: {missing}")
    pred = safe_numeric_cols(pred, [*predictors, "X", "Y", "bathy_t"])
    pred["interval_years"] = pair_duration_years(pair)
    matrix = xgb.DMatrix(pred[predictors].to_numpy(dtype=np.float32), feature_names=predictors, missing=np.nan)
    boot = np.column_stack([load_xgb_model(fs, uri).predict(matrix) for uri in metadata["model_uris"]])
    mean = np.nanmean(boot, axis=1)
    sd = np.nanstd(boot, axis=1, ddof=1 if boot.shape[1] > 1 else 0)
    translated = translate_predictions(mean, pred["bathy_t"].to_numpy(dtype=float), pred["interval_years"].to_numpy(dtype=float), cfg.model.standard_horizon_years)
    for key, value in translated.items():
        pred[key] = value
    pred["uncertainty_sd_bathy_t1"] = sd
    pred["source_local_model_tile"] = source_model_tile
    ranges = read_table(metadata["predictor_ranges_uri"], fs)
    pred = add_extrapolation_flags(pred, ranges)
    dataframe_to_geoparquet(pred, out_uri, fs, cfg.model.crs)
    for col, prefix in {
        "bathy_t": "LOCAL_Pred_Start_Bathy_t",
        "mean_predicted_bathy_t1": "LOCAL_Pred_Mean_Bathy_t1",
        "mean_predicted_change": "LOCAL_Pred_Mean_Change",
        "uncertainty_sd_bathy_t1": "LOCAL_Pred_SD_Bathy_t1",
        "predictor_extrapolation_count": "LOCAL_Predictor_Extrapolation_Count",
    }.items():
        write_cog_from_dataframe(
            pred, col, fs.join(cfg.paths.prediction_root, tile_id, f"{prefix}_{pair}.tif"),
            cfg.paths.template_raster_uri, fs, cfg.raster,
            tile_geometry_wkb, tile_geometry_crs,
        )
    return {"ok": True, "tile_id": tile_id, "pair": pair, "source_model_tile": source_model_tile}


def nearest_local_assignments(grid: gpd.GeoDataFrame, pair: str, cfg: WorkflowConfig, fs: CloudFS) -> pd.DataFrame:
    tile_field = cfg.model.tile_field
    model_tiles = []
    for tile_id in grid[tile_field].astype(str):
        if fs.exists(fs.join(cfg.paths.training_root, tile_id, f"LOCAL_model_metadata_{pair}.json")):
            model_tiles.append(tile_id)
    if not model_tiles:
        raise ValueError(f"No trained local models found for {pair}")
    model_grid = grid[grid[tile_field].isin(model_tiles)].copy()
    prediction_grid = select_tiles(grid, cfg.model, prediction=True).copy()
    # representative_point() is guaranteed to lie inside each polygon and avoids
    # centroid warnings/edge cases for irregular or multipart grid geometries.
    model_points = model_grid.geometry.representative_point()
    assignments = []
    for tile_id, geom in zip(prediction_grid[tile_field], prediction_grid.geometry.representative_point()):
        distances = model_points.distance(geom)
        source_index = distances.idxmin()
        assignments.append({
            "tile_id": str(tile_id),
            "source_model_tile": str(model_grid.loc[source_index, tile_field]),
            "distance_m": float(distances.loc[source_index]),
        })
    return pd.DataFrame(assignments)


# =============================================================================
# PART E. BALANCED GLOBAL TRAINING BY YEAR PAIR
# The global model learns across tiles. Balanced sampling prevents a tile with
# many more cells from overwhelming the patterns observed in smaller tiles.
# =============================================================================

def train_global_pair(pair: str, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = workflow_config_from_dict(cfg_dict)
    fs = CloudFS(cfg.s3)
    shard_uris = fs.glob(fs.join(cfg.paths.global_shard_root, pair, f"*_{pair}_global_training_shard.parquet"))
    if not shard_uris:
        return {"ok": False, "pair": pair, "reason": "no_shards"}
    if cfg.model.tile_ids != "all":
        shard_uris = [u for u in shard_uris if Path(u).name.split(f"_{pair}_global")[0] in cfg.model.tile_ids]
    schemas = [pq.read_schema(uri, filesystem=fs.s3 if fs.is_s3(uri) else None).names for uri in shard_uris]
    common = set(schemas[0]).intersection(*map(set, schemas[1:]))
    predictors = sorted(common - NON_PREDICTORS - {"tile_id", "pair_id"})
    if "bathy_t" not in predictors:
        raise ValueError("bathy_t missing from common global predictor schema")
    budget = min(cfg.model.max_rows_per_global_tile, max(1, cfg.model.max_global_rows // len(shard_uris)))
    sampled = []
    manifest = []
    for uri in shard_uris:
        tile = Path(uri).name.split(f"_{pair}_global")[0]
        df = read_table(uri, fs, columns=["tile_id", "pair_id", "X", "Y", "bathy_t1", "delta_bathy", "interval_years", *predictors])
        n_take = min(len(df), budget)
        sample = df.sample(n=n_take, replace=False, random_state=stable_seed(cfg.model.global_seed, pair, tile)) if n_take < len(df) else df
        sampled.append(sample)
        manifest.append({"tile_id": tile, "source_rows": len(df), "sampled_rows": len(sample), "uri": uri})
    global_df = pd.concat(sampled, ignore_index=True)
    if len(global_df) > cfg.model.max_global_rows:
        global_df = global_df.sample(cfg.model.max_global_rows, random_state=stable_seed(cfg.model.global_seed, pair, "cap"))
    global_df = safe_numeric_cols(global_df, [*predictors, "X", "Y", "bathy_t1", "delta_bathy", "interval_years"])
    essential = np.isfinite(global_df[["X", "Y", "bathy_t1", "bathy_t", "delta_bathy", "interval_years"]]).all(axis=1) & (global_df["interval_years"] > 0)
    fit = global_df.loc[essential].reset_index(drop=True)
    weights = calculate_sample_weights(fit["delta_bathy"].to_numpy(float), cfg.model.weight_alpha, cfg.model.weight_method, cfg.model.weight_cap_quantile, cfg.model.max_weight) if cfg.model.use_weighted_loss else np.ones(len(fit))
    rounds, cv_metrics = choose_rounds_spatial_cv(fit, predictors, weights, cfg.model, stable_seed(cfg.model.global_seed, "global_cv", pair))
    x_fit = fit[predictors].to_numpy(np.float32)
    y_fit = fit["bathy_t1"].to_numpy(np.float32)
    rng = np.random.default_rng(stable_seed(cfg.model.global_seed, "global_boot", pair))
    model_uris = []
    for b in range(cfg.model.n_boot):
        idx = rng.choice(len(fit), len(fit), replace=True)
        dtrain = xgb.DMatrix(x_fit[idx], label=y_fit[idx], weight=weights[idx], feature_names=predictors, missing=np.nan)
        model = xgb.train(xgb_params(cfg.model, nthread=max(1, cfg.dask.threads_per_worker)), dtrain, num_boost_round=rounds, verbose_eval=False)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = temp.name
        try:
            model.save_model(temp_path)
            uri = fs.join(cfg.paths.global_model_root, pair, f"GLOBAL_model_boot_{b + 1:03d}_{pair}.json")
            fs.upload(temp_path, uri)
            model_uris.append(uri)
        finally:
            Path(temp_path).unlink(missing_ok=True)
    ranges = predictor_range_table(fit, predictors)
    ranges_uri = fs.join(cfg.paths.global_model_root, pair, f"GLOBAL_predictor_ranges_{pair}.parquet")
    write_parquet(ranges, ranges_uri, fs)
    write_parquet(pd.DataFrame(manifest), fs.join(cfg.paths.global_model_root, pair, f"GLOBAL_sampling_manifest_{pair}.parquet"), fs)
    write_parquet(cv_metrics, fs.join(cfg.paths.global_model_root, pair, f"GLOBAL_spatial_cv_metrics_{pair}.parquet"), fs)
    dataframe_to_geoparquet(global_df, fs.join(cfg.paths.global_model_root, pair, f"GLOBAL_sampled_training_cache_{pair}.geoparquet"), fs, cfg.model.crs)
    write_json({
        "pair": pair, "predictors": predictors, "best_iteration": rounds,
        "model_uris": model_uris, "predictor_ranges_uri": ranges_uri,
        "n_rows": len(fit), "crs": cfg.model.crs,
    }, fs.join(cfg.paths.global_model_root, pair, f"GLOBAL_model_metadata_{pair}.json"), fs)
    return {"ok": True, "pair": pair, "n_rows": len(fit), "best_iteration": rounds}


def predict_global_tile(tile_id: str, pair: str, tile_geometry_wkb: bytes, tile_geometry_crs: str, cfg_dict: dict[str, Any]) -> dict[str, Any]:
    cfg = workflow_config_from_dict(cfg_dict)
    fs = CloudFS(cfg.s3)
    meta_uri = fs.join(cfg.paths.global_model_root, pair, f"GLOBAL_model_metadata_{pair}.json")
    if not fs.exists(meta_uri):
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": "missing_global_model"}
    with fs.open(meta_uri, "rt") as stream:
        metadata = json.load(stream)
    pred_uri = find_tile_file(fs, cfg.paths.prediction_root, tile_id, pair, "pred")
    if pred_uri is None:
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": "missing_prediction_file"}
    pred = read_table(pred_uri, fs)
    predictors = metadata["predictors"]
    missing = [p for p in predictors if p not in pred]
    if missing:
        raise ValueError(f"Tile {tile_id} missing global predictors: {missing}")
    pred = safe_numeric_cols(pred, [*predictors, "X", "Y", "bathy_t"])
    pred["interval_years"] = pair_duration_years(pair)
    matrix = xgb.DMatrix(pred[predictors].to_numpy(np.float32), feature_names=predictors, missing=np.nan)
    boot = np.column_stack([load_xgb_model(fs, uri).predict(matrix) for uri in metadata["model_uris"]])
    mean = np.nanmean(boot, axis=1)
    sd = np.nanstd(boot, axis=1, ddof=1 if boot.shape[1] > 1 else 0)
    for key, value in translate_predictions(mean, pred["bathy_t"].to_numpy(float), pred["interval_years"].to_numpy(float), cfg.model.standard_horizon_years).items():
        pred[key] = value
    pred["uncertainty_sd_bathy_t1"] = sd
    pred = add_extrapolation_flags(pred, read_table(metadata["predictor_ranges_uri"], fs))
    root = fs.join(cfg.paths.prediction_root, tile_id)
    dataframe_to_geoparquet(pred, fs.join(root, f"GLOBAL_prediction_summary_{pair}.geoparquet"), fs, cfg.model.crs)
    for col, prefix in {
        "bathy_t": "GLOBAL_Pred_Start_Bathy_t",
        "mean_predicted_bathy_t1": "GLOBAL_Pred_Mean_Bathy_t1",
        "mean_predicted_change": "GLOBAL_Pred_Mean_Change",
        "uncertainty_sd_bathy_t1": "GLOBAL_Pred_SD_Bathy_t1",
        "predictor_extrapolation_count": "GLOBAL_Predictor_Extrapolation_Count",
    }.items():
        write_cog_from_dataframe(
            pred, col, fs.join(root, f"{prefix}_{pair}.tif"),
            cfg.paths.template_raster_uri, fs, cfg.raster,
            tile_geometry_wkb, tile_geometry_crs,
        )
    return {"ok": True, "tile_id": tile_id, "pair": pair}


# =============================================================================
# PART F. DASK ORCHESTRATION
# Dask distributes independent jobs across workers. The scientific calculations
# inside each job are unchanged whether workers are local or cloud-hosted.
# =============================================================================

def workflow_config_to_dict(cfg: WorkflowConfig) -> dict[str, Any]:
    return asdict(cfg)


def workflow_config_from_dict(raw: dict[str, Any]) -> WorkflowConfig:
    return WorkflowConfig(
        paths=PathConfig(**raw["paths"]),
        model=ModelConfig(**{
            **raw["model"],
            "year_pairs": tuple(raw["model"]["year_pairs"]),
            "tile_ids": "all" if raw["model"]["tile_ids"] == "all" else tuple(raw["model"]["tile_ids"]),
            "allowed_training_sources": tuple(raw["model"]["allowed_training_sources"]),
            "allowed_prediction_sources": tuple(raw["model"]["allowed_prediction_sources"]),
        }),
        raster=RasterConfig(**{**raw["raster"], "overview_levels": tuple(raw["raster"]["overview_levels"])}),
        dask=DaskConfig(**raw["dask"]),
        s3=S3Config(**raw["s3"]),
    )


def get_client(cfg: WorkflowConfig) -> tuple[Client, LocalCluster | None]:
    if cfg.dask.scheduler_address:
        return Client(cfg.dask.scheduler_address), None
    cluster = LocalCluster(
        n_workers=cfg.dask.n_workers,
        threads_per_worker=cfg.dask.threads_per_worker,
        processes=True,
        memory_limit=cfg.dask.memory_limit,
        local_directory=cfg.dask.local_directory,
    )
    return Client(cluster), cluster


def run_futures(client: Client, futures: Sequence[Any]) -> pd.DataFrame:
    results = []
    for future in as_completed(futures):
        try:
            result = future.result()
            logging.info("Task result: %s", result)
            results.append(result)
        except Exception as exc:
            logging.exception("Dask task failed")
            results.append({"ok": False, "reason": repr(exc)})
    return pd.DataFrame(results)


def _tile_geometry_payloads(grid: gpd.GeoDataFrame, tile_field: str) -> dict[str, tuple[bytes, str]]:
    if grid.crs is None:
        raise ValueError("Master grid must have a CRS before raster creation")
    crs_text = grid.crs.to_string()
    payloads: dict[str, tuple[bytes, str]] = {}
    for row in grid[[tile_field, "geometry"]].itertuples(index=False):
        tile_id = str(getattr(row, tile_field))
        geom = row.geometry
        if geom is None or geom.is_empty:
            raise ValueError(f"Grid tile {tile_id} has empty geometry")
        if tile_id in payloads:
            raise ValueError(f"Expected exactly one grid geometry for tile {tile_id}")
        payloads[tile_id] = (geom.wkb, crs_text)
    return payloads


def run_local(cfg: WorkflowConfig) -> pd.DataFrame:
    """Train tile-native models, then deploy them across the full extent by default."""
    fs = CloudFS(cfg.s3)
    full_grid = load_grid(cfg, fs)
    grid = select_tiles(full_grid, cfg.model, prediction=False)
    geometry_payloads = _tile_geometry_payloads(full_grid, cfg.model.tile_field)
    cfg_dict = workflow_config_to_dict(cfg)
    client, cluster = get_client(cfg)
    try:
        futures = []
        for tile in grid[cfg.model.tile_field].astype(str):
            geom_wkb, geom_crs = geometry_payloads[tile]
            for pair in cfg.model.year_pairs:
                futures.append(client.submit(
                    train_local_tile_pair, tile, pair, geom_wkb, geom_crs, cfg_dict, pure=False
                ))
        tile_native = run_futures(client, futures)
    finally:
        client.close()
        if cluster:
            cluster.close()

    if not cfg.model.deploy_local_full_extent:
        tile_native["workflow_phase"] = "tile_native"
        return tile_native

    full_extent = run_full_extent_local(
        cfg, overwrite=cfg.model.overwrite_existing_local
    )
    tile_native["workflow_phase"] = "tile_native"
    full_extent["workflow_phase"] = "full_extent"
    combined = pd.concat([tile_native, full_extent], ignore_index=True, sort=False)
    if cfg.model.build_vrts_after_local and cfg.model.write_full_extent_spatial_outputs:
        vrt_results = build_local_and_global_vrts(cfg, include_global=False)
        if not vrt_results.empty:
            vrt_results["workflow_phase"] = "vrt"
            combined = pd.concat([combined, vrt_results], ignore_index=True, sort=False)
    return combined


def run_full_extent_local(cfg: WorkflowConfig, overwrite: bool = False) -> pd.DataFrame:
    fs = CloudFS(cfg.s3)
    grid = load_grid(cfg, fs)
    geometry_payloads = _tile_geometry_payloads(grid, cfg.model.tile_field)
    cfg_dict = workflow_config_to_dict(cfg)
    all_assignments = []
    for pair in cfg.model.year_pairs:
        a = nearest_local_assignments(grid, pair, cfg, fs)
        a["pair"] = pair
        all_assignments.append(a)
    assignments = pd.concat(all_assignments, ignore_index=True)
    write_parquet(assignments, fs.join(cfg.paths.log_root, "nearest_local_assignments.parquet"), fs)
    client, cluster = get_client(cfg)
    try:
        futures = []
        for row in assignments.itertuples(index=False):
            geom_wkb, geom_crs = geometry_payloads[str(row.tile_id)]
            futures.append(client.submit(
                predict_tile_with_model_ensemble, row.tile_id, row.pair,
                row.source_model_tile, geom_wkb, geom_crs, cfg_dict, overwrite, pure=False
            ))
        results = run_futures(client, futures)
    finally:
        client.close()
        if cluster:
            cluster.close()

    failures = results[results.get("ok", False) != True].copy() if not results.empty else results
    if not failures.empty:
        for pair, part in failures.groupby("pair", dropna=False):
            write_parquet(
                part, fs.join(cfg.paths.log_root, f"LOCAL_FULL_EXTENT_FAILURES_{pair}.parquet"), fs
            )
    return results


def run_global_training(cfg: WorkflowConfig) -> pd.DataFrame:
    client, cluster = get_client(cfg)
    try:
        cfg_dict = workflow_config_to_dict(cfg)
        futures = [client.submit(train_global_pair, pair, cfg_dict, pure=False) for pair in cfg.model.year_pairs]
        return run_futures(client, futures)
    finally:
        client.close()
        if cluster:
            cluster.close()


def run_global_prediction(cfg: WorkflowConfig) -> pd.DataFrame:
    fs = CloudFS(cfg.s3)
    full_grid = load_grid(cfg, fs)
    grid = select_tiles(full_grid, cfg.model, prediction=True)
    geometry_payloads = _tile_geometry_payloads(full_grid, cfg.model.tile_field)
    client, cluster = get_client(cfg)
    try:
        cfg_dict = workflow_config_to_dict(cfg)
        futures = []
        for tile in grid[cfg.model.tile_field].astype(str):
            geom_wkb, geom_crs = geometry_payloads[tile]
            for pair in cfg.model.year_pairs:
                futures.append(client.submit(
                    predict_global_tile, tile, pair, geom_wkb, geom_crs, cfg_dict, pure=False
                ))
        results = run_futures(client, futures)
    finally:
        client.close()
        if cluster:
            cluster.close()
    return results


# =============================================================================
# PART G. SEAM-SAFE LOCAL AND GLOBAL VRT CREATION
# VRTs are lightweight mosaics that reference the COG tiles. They do not blend
# overlaps, which is why polygon clipping in Part B is mandatory.
# =============================================================================

LOCAL_TRAIN_VRT_PRODUCTS = (
    "LOCAL_Train_Start_Bathy_t", "LOCAL_Train_Mean_Predicted_Bathy_t1",
    "LOCAL_Train_Surveyed_Change", "LOCAL_Train_Mean_Predicted_Change",
    "LOCAL_Train_SD_Bathy_t1",
)
LOCAL_PRED_VRT_PRODUCTS = (
    "LOCAL_Pred_Start_Bathy_t", "LOCAL_Pred_Mean_Bathy_t1",
    "LOCAL_Pred_Mean_Change", "LOCAL_Pred_SD_Bathy_t1",
    "LOCAL_Predictor_Extrapolation_Count", "LOCAL_Pred_BlueTopo_UC_t",
)
GLOBAL_PRED_VRT_PRODUCTS = (
    "GLOBAL_Pred_Start_Bathy_t", "GLOBAL_Pred_Mean_Bathy_t1",
    "GLOBAL_Pred_Mean_Change", "GLOBAL_Pred_SD_Bathy_t1",
    "GLOBAL_Predictor_Extrapolation_Count",
)


def _gdal_uri(uri: str) -> str:
    return "/vsis3/" + uri[5:] if uri.startswith("s3://") else uri


def build_vrt_from_cogs(
    source_uris: Sequence[str], output_uri: str, fs: CloudFS, nodata: float,
    separate: bool = False,
) -> dict[str, Any]:
    if not source_uris:
        return {"ok": False, "output_uri": output_uri, "reason": "no_source_rasters"}
    ordered = sorted(set(source_uris))
    with tempfile.TemporaryDirectory(prefix="seabed_vrt_") as tmp:
        local_vrt = str(Path(tmp) / "mosaic.vrt")
        sources = [_gdal_uri(uri) for uri in ordered]
        if gdal is not None:
            options = gdal.BuildVRTOptions(
                srcNodata=nodata, VRTNodata=nodata, separate=separate,
                resolution="highest", resampleAlg="nearest",
            )
            dataset = gdal.BuildVRT(local_vrt, sources, options=options)
            if dataset is None:
                raise RuntimeError(f"GDAL BuildVRT failed for {output_uri}")
            dataset.FlushCache()
            dataset = None
        elif shutil.which("gdalbuildvrt"):
            command = ["gdalbuildvrt", "-overwrite", "-srcnodata", str(nodata),
                       "-vrtnodata", str(nodata), "-resolution", "highest"]
            if separate:
                command.append("-separate")
            command.extend([local_vrt, *sources])
            subprocess.run(command, check=True, capture_output=True, text=True)
        else:
            raise RuntimeError("VRT creation requires GDAL Python bindings or gdalbuildvrt")
        fs.upload(local_vrt, output_uri)
    return {"ok": True, "output_uri": output_uri, "n_sources": len(ordered), "separate": separate}


def build_local_and_global_vrts(cfg: WorkflowConfig, include_global: bool = True) -> pd.DataFrame:
    fs = CloudFS(cfg.s3)
    results: list[dict[str, Any]] = []
    groups = [
        (cfg.paths.training_root, fs.join(cfg.paths.training_root, "VRT"), LOCAL_TRAIN_VRT_PRODUCTS),
        (cfg.paths.prediction_root, fs.join(cfg.paths.prediction_root, "VRT"), LOCAL_PRED_VRT_PRODUCTS),
    ]
    if include_global:
        groups.append((cfg.paths.prediction_root, fs.join(cfg.paths.prediction_root, "VRT", "GLOBAL"), GLOBAL_PRED_VRT_PRODUCTS))
    for root, vrt_root, products in groups:
        for product in products:
            yearly_vrts = []
            for pair in cfg.model.year_pairs:
                sources = fs.glob(fs.join(root, "*", f"{product}_{pair}.tif"))
                output = fs.join(vrt_root, f"{product}_{pair}_MOSAIC.vrt")
                result = build_vrt_from_cogs(sources, output, fs, cfg.raster.nodata)
                result.update(product=product, pair=pair, kind="year_mosaic")
                results.append(result)
                if result.get("ok"):
                    yearly_vrts.append(output)
            if len(yearly_vrts) > 1:
                output = fs.join(vrt_root, f"{product}_ALL_YEARS.vrt")
                result = build_vrt_from_cogs(yearly_vrts, output, fs, cfg.raster.nodata, separate=True)
                result.update(product=product, pair="all", kind="multiband_year_stack")
                results.append(result)
    return pd.DataFrame(results)


# =============================================================================
# PART H. OPTIONAL ONE-TIME FST -> GEOPARQUET MIGRATION
# Run this only during data migration. Production modelling should read the
# converted Parquet/GeoParquet files directly from local or S3 storage.
# =============================================================================

def migrate_fst_directory(local_fst_root: str, output_root: str, crs: str, pattern: str = "**/*.fst") -> pd.DataFrame:
    """One-time local migration helper; requires `pip install fstpy` or pyreadr support.

    FST is not a cloud interchange format. Production inputs should be GeoParquet.
    This function intentionally runs outside S3 because the source R FST files are
    usually on a workstation/network share and should be converted once.
    """
    try:
        import fstpy  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install fstpy for one-time FST migration: pip install fstpy") from exc
    rows = []
    for source in Path(local_fst_root).glob(pattern):
        relative = source.relative_to(local_fst_root).with_suffix(".geoparquet")
        destination = Path(output_root) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        df = fstpy.read_fst(str(source))
        if {"X", "Y"}.issubset(df.columns):
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["X"], df["Y"]), crs=crs)
            gdf.to_parquet(destination, compression="zstd", index=False)
        else:
            df.to_parquet(destination.with_suffix(".parquet"), compression="zstd", index=False)
        rows.append({"source": str(source), "destination": str(destination), "n_rows": len(df)})
    return pd.DataFrame(rows)


# =============================================================================
# PART I. COMMAND-LINE ENTRY POINT
# This small section connects terminal commands to the workflow functions above.
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Cloud seabed elevation-change modelling workflow")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--stage", required=True, choices=["local", "full-extent-local", "global-train", "global-predict", "vrt", "all"])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite full-extent local predictions")
    args = parser.parse_args()
    setup_logging()
    cfg = WorkflowConfig.from_yaml(args.config)
    stages = [args.stage] if args.stage != "all" else ["local", "global-train", "global-predict", "vrt"]
    fs = CloudFS(cfg.s3)
    for stage in stages:
        logging.info("Starting stage: %s", stage)
        if stage == "local":
            result = run_local(cfg)
        elif stage == "full-extent-local":
            result = run_full_extent_local(cfg, overwrite=args.overwrite)
        elif stage == "global-train":
            result = run_global_training(cfg)
        elif stage == "global-predict":
            result = run_global_prediction(cfg)
        else:
            result = build_local_and_global_vrts(cfg, include_global=True)
        uri = fs.join(cfg.paths.log_root, f"{stage}_results.parquet")
        write_parquet(result, uri, fs)
        logging.info("Finished %s; results written to %s", stage, uri)


if __name__ == "__main__":
    main()
