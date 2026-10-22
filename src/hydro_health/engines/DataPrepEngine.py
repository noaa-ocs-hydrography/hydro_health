import gc
import json
import os
import pathlib
import re
import shutil
import tempfile
import warnings
from configparser import Error as ConfigError
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import CRS
from shapely.geometry import Point
from upath import UPath

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

DYNAMIC_BASES = ('hurr_strength_mean', 'tsm_mean')
NON_PREDICTORS = {
    'X', 'Y', 'FID', 'tile_id', 'source', 'dataset_role',
    'year_t', 'year_t1', 'year_ti', 'interval_years', 'pair_id',
    'bathy_t1', 'bathy_ti', 'delta_bathy', 'delta_rate', 'survey_end_date',
    'geometry', 'prediction_row_usable', 'sample_weight',
}

def _pair_duration_years(pair: str) -> float:
    """Validate YYYY_YYYY and return its positive elapsed time."""

    match = re.fullmatch(r'(\d{4})_(\d{4})', pair)
    if not match:
        raise ValueError(f'Invalid year-pair label: {pair}')
    start, end = map(int, match.groups())
    if end <= start:
        raise ValueError(f'Year pair must increase: {pair}')
    return float(end - start)

def _safe_numeric_cols(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Coerce numeric fields and treat nonnumeric/infinite values as missing."""

    out = df.copy()
    for column in set(columns).intersection(out.columns):
        values = pd.to_numeric(out[column], errors='coerce').to_numpy(dtype=float, na_value=np.nan, copy=True)
        values[~np.isfinite(values)] = np.nan
        out[column] = values
    return out

def _ensure_pair_dates(df: pd.DataFrame, pair: str) -> pd.DataFrame:
    """Fill genuinely missing years/intervals; reject inconsistent supplied values."""

    duration = _pair_duration_years(pair)
    start, end = map(int, pair.split('_'))
    out = df.copy()
    for column, expected in (('year_t', start), ('year_t1', end), ('interval_years', duration)):
        if column in out:
            raw = out[column]
            absent = raw.isna() | raw.astype('string').str.strip().eq('').fillna(False)
            numeric = pd.to_numeric(raw, errors='coerce').to_numpy(dtype=float, na_value=np.nan)
            invalid = ~absent.to_numpy(dtype=bool) & (
                ~np.isfinite(numeric) | ~np.isclose(numeric, expected, rtol=0, atol=1e-8)
            )
            if invalid.any():
                raise ValueError(
                    f'{int(invalid.sum())} {column} values conflict with pair {pair}; '
                    f'expected {expected}. Reversed or invalid dates are not repaired.'
                )
        out[column] = float(expected)
    return out

def _ensure_training_targets(df: pd.DataFrame, pair: str, tolerance: float = 1e-8) -> pd.DataFrame:
    """Compute newer-minus-older elevation change only for a consistent year pair."""

    missing = {'bathy_t', 'bathy_t1'} - set(df.columns)
    if missing:
        raise ValueError(f'Missing required training fields: {sorted(missing)}')
    out = _ensure_pair_dates(df, pair)
    duration = _pair_duration_years(pair)
    out = _safe_numeric_cols(out, ['bathy_t', 'bathy_t1', 'delta_bathy', 'delta_rate'])
    with np.errstate(over='ignore', invalid='ignore'):
        change = out['bathy_t1'] - out['bathy_t']
    change = change.replace([np.inf, -np.inf], np.nan)
    if 'delta_bathy' in out:
        mismatch = np.isfinite(out['delta_bathy']) & np.isfinite(change) & (
            (out['delta_bathy'] - change).abs() > tolerance
        )
        if mismatch.any():
            warnings.warn(
                f'{int(mismatch.sum())} delta_bathy values replaced with bathy_t1 - bathy_t.',
                stacklevel=2,
            )
    out['delta_bathy'] = change
    out['delta_rate'] = change / duration
    return out

def _normalize_master_predictors(predictors: Sequence[str]) -> list[str]:
    """Normalize old dynamic suffixes to bases without changing feature order."""

    if isinstance(predictors, (str, bytes)) or not isinstance(predictors, (list, tuple)) or not predictors:
        raise ValueError('Master predictors must be a nonempty ordered list of names')
    result = ['bathy_t']
    seen = set()
    for predictor in predictors:
        if not isinstance(predictor, str) or not predictor.strip():
            raise ValueError('Predictor names must be nonblank strings')
        predictor = predictor.strip()
        if predictor in seen:
            raise ValueError(f'Duplicate master predictor: {predictor}')
        seen.add(predictor)
        for base in DYNAMIC_BASES:
            if predictor.startswith(base + '_'):
                suffix = predictor[len(base) + 1:]
                _pair_duration_years(suffix)
                predictor = base
                break
        if predictor in NON_PREDICTORS:
            raise ValueError(f'Target or metadata cannot be a predictor: {predictor}')
        if predictor not in result:
            result.append(predictor)
    return result

def _build_pair_specific_predictors(master_predictors: Sequence[str], pair: str) -> list[str]:
    _pair_duration_years(pair)
    return [f'{name}_{pair}' if name in DYNAMIC_BASES else name
            for name in _normalize_master_predictors(master_predictors)]

def _resolve_predictors(
    pair: str,
    master_predictors: Sequence[str],
    train_columns: Sequence[str],
    prediction_columns: Sequence[str] | None,
    missing_policy: str = 'fill_nan',
) -> tuple[list[str], list[str], list[str]]:
    """Choose one ordered schema for training and prediction; bathy_t is required."""

    if missing_policy not in {'fill_nan', 'error', 'drop'}:
        raise ValueError(f'Unknown missing_predictor_policy: {missing_policy}')
    if 'bathy_t' not in train_columns:
        raise ValueError('Training requires bathy_t')
    if prediction_columns is not None and 'bathy_t' not in prediction_columns:
        raise ValueError('Prediction requires bathy_t')
    candidates = _build_pair_specific_predictors(master_predictors, pair)
    missing_train = [name for name in candidates if name not in train_columns]
    missing_pred = ([name for name in candidates if name not in prediction_columns]
                    if prediction_columns is not None else [])
    if missing_policy == 'error' and (missing_train or missing_pred):
        raise ValueError(f'Missing predictor columns: training={missing_train}, prediction={missing_pred}')
    excluded = set(missing_train) | set(missing_pred) if missing_policy == 'drop' else set()
    return [name for name in candidates if name not in excluded], missing_train, missing_pred

def _align_predictors(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    """Add all-NaN optional columns while retaining row IDs, targets, and metadata."""

    out = df.copy()
    for name in predictors:
        if name not in out:
            out[name] = np.nan
    return _safe_numeric_cols(out, [*predictors, 'X', 'Y'])

def _read_input_table(uri: str, crs: str, tile_id: str, pair: str, input_crs: str | None) -> pd.DataFrame:
    """Validate declared CRS, optional identity columns, and GeoParquet point X/Y."""

    target = CRS.from_user_input(crs)
    with UPath(uri).open('rb') as stream:
        metadata = pq.read_schema(stream).metadata or {}
    is_spatial = b'geo' in metadata
    if is_spatial:
        spatial = gpd.read_parquet(uri)
        if spatial.crs is None:
            raise ValueError(f'GeoParquet has no CRS: {uri}')
        actual_crs = CRS.from_user_input(spatial.crs)
        if input_crs is not None and actual_crs != CRS.from_user_input(input_crs):
            raise ValueError(f'GeoParquet CRS conflicts with configured input_crs: {uri}')
        out = pd.DataFrame(spatial.drop(columns=spatial.geometry.name))
    else:
        if input_crs is None:
            raise ValueError(f'Plain Parquet requires an explicit input_crs setting: {uri}')
        actual_crs = CRS.from_user_input(input_crs)
        out = pd.read_parquet(uri)
        if 'geometry' in out:
            raise ValueError(f'Geometry without GeoParquet CRS metadata is unsupported: {uri}')
    if actual_crs != target:
        raise ValueError(f'Input CRS {actual_crs.to_string()} does not match {target.to_string()}: {uri}')
    missing = {'X', 'Y'} - set(out.columns)
    if missing:
        raise ValueError(f'Missing required coordinate columns {sorted(missing)}: {uri}')
    for column, expected in (('tile_id', tile_id), ('pair_id', pair)):
        if column in out and not out[column].dropna().astype(str).eq(expected).all():
            raise ValueError(f'Input {column} does not match {expected}: {uri}')
    out = _safe_numeric_cols(out, ['X', 'Y'])
    finite_xy = np.isfinite(out['X']) & np.isfinite(out['Y'])
    if is_spatial:
        if not np.array_equal(spatial.geometry.isna().to_numpy(), (~finite_xy).to_numpy()):
            raise ValueError(f'Geometry/null positions do not match finite X/Y: {uri}')
        points = spatial.loc[finite_xy]
        if not points.geometry.geom_type.eq('Point').all() or points.geometry.is_empty.any():
            raise ValueError(f'Input geometry must contain nonempty points: {uri}')
        if not (np.allclose(points.geometry.x, out.loc[finite_xy, 'X'], rtol=0, atol=1e-7)
                and np.allclose(points.geometry.y, out.loc[finite_xy, 'Y'], rtol=0, atol=1e-7)):
            raise ValueError(f'Point geometry does not match numeric X/Y: {uri}')
    out['tile_id'] = tile_id
    out['pair_id'] = pair
    return out

def _calculate_sample_weights(
    change: np.ndarray, alpha: float, method: str, cap_quantile: float,
    max_weight: float, epsilon: float = 1e-6,
) -> np.ndarray:
    """Weight absolute elevation change; normalize power weights to mean one."""

    magnitude = np.abs(change.astype(float))
    if method == 'power':
        # Subtracting the maximum log weight avoids overflow without changing ratios.
        logs = np.log(magnitude + epsilon)
        with np.errstate(over='ignore', under='ignore'):
            weights = np.exp(alpha * (logs - np.max(logs)))
        return weights / weights.sum() * len(weights)
    if method == 'capped_linear':
        cap = np.quantile(magnitude, cap_quantile)
        if cap <= 0:
            return np.ones_like(magnitude)
        return 1.0 + (max_weight - 1.0) * np.power(np.minimum(magnitude, cap) / cap, alpha)
    raise ValueError(f'Unknown weight method: {method}')

def _summarize_missingness(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    """Return a table of missingness counts and proportions for each predictor."""

    return pd.DataFrame({
        'predictor': predictors, 'n_rows': len(df),
        'n_na': [int(df[name].isna().sum()) for name in predictors],
        'proportion_na': [float(df[name].isna().mean()) for name in predictors],
        'n_finite': [int(np.isfinite(df[name]).sum()) for name in predictors],
    })


def _predictor_range_table(df: pd.DataFrame, predictors: Sequence[str]) -> pd.DataFrame:
    """Return a table of min, max, and mean for each predictor in the training set."""

    rows = []
    for name in predictors:
        values = df[name].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        rows.append({
            'predictor': name,
            'min_training': float(finite.min()) if finite.size else np.nan,
            'max_training': float(finite.max()) if finite.size else np.nan,
            'mean_training': float(finite.mean()) if finite.size else np.nan,
        })
    return pd.DataFrame(rows)


def _save_geoparquet(df: pd.DataFrame, path: Path, crs: str) -> None:
    """Write a staged table, retaining rows with invalid XY as null geometry."""

    valid_xy = np.isfinite(df['X']) & np.isfinite(df['Y'])
    geometry = gpd.GeoSeries(
        [Point(float(x), float(y)) if valid else None
         for x, y, valid in zip(df['X'], df['Y'], valid_xy)], index=df.index, crs=crs,
    )
    gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=crs).to_parquet(
        path, index=False, compression='zstd'
    )


def _output_paths(train_dir: str, pred_dir: str, pair: str, has_prediction: bool) -> dict[str, UPath]:
    """Construct output paths for training and prediction files."""

    train, pred = UPath(train_dir), UPath(pred_dir)
    paths = {
        'schema': train / f'locked_predictors_{pair}.parquet',
        'training_missingness': train / f'predictor_missingness_training_{pair}.parquet',
        'ranges': train / f'predictor_ranges_training_{pair}.parquet',
        'training': train / f'ML_ready_training_{pair}.geoparquet',
    }
    if has_prediction:
        paths.update({
            'prediction_missingness': pred / f'predictor_missingness_prediction_{pair}.parquet',
            'prediction': pred / f'ML_ready_prediction_{pair}.geoparquet',
        })
    return paths


def _outputs_complete(marker: UPath, paths: dict[str, UPath]) -> bool:
    """A success marker and all expected files permit a skip."""

    if not marker.is_file():
        return False
    try:
        record = json.loads(marker.read_text())
        return (
            isinstance(record, dict)
            and record['status'] == 'success'
            and set(record['outputs']) == {str(path) for path in paths.values()}
            and all(path.is_file() for path in paths.values())
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _publish_file(staged: Path, destination: UPath, is_aws: bool) -> None:
    """Replace one local file atomically; stream remote files before the marker."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_aws:
        with staged.open('rb') as source, destination.open('wb') as target:
            shutil.copyfileobj(source, target)
    else:
        with tempfile.NamedTemporaryFile(dir=str(destination.parent), delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            shutil.copyfile(staged, temporary_path)
            os.replace(temporary_path, str(destination))
        finally:
            temporary_path.unlink(missing_ok=True)


def _prep_tile_task(params: list) -> dict:
    """Validate, prepare, stage all outputs, then publish the completion marker last."""

    (tile_id, pair, raw_train_uri, raw_pred_uri, master_predictors, cfg, crs, is_aws,
     out_train_dir, out_pred_dir, local_tmp_dir, index, total, verbose, overwrite) = params
    result = {'tile_id': tile_id, 'pair': pair}
    marker = UPath(out_train_dir) / f'prep_complete_{pair}.json'
    try:
        _pair_duration_years(pair)
        paths = _output_paths(out_train_dir, out_pred_dir, pair, raw_pred_uri is not None)
        if not overwrite and _outputs_complete(marker, paths):
            return {**result, 'ok': True, 'status': 'skipped', 'reason': 'already_complete'}
        # Old files without a matching marker are incomplete and can be replaced.
        if marker.exists():
            marker.unlink()
        if verbose:
            Engine.write_message_dask(f'Preparing tile {tile_id}, pair {pair} ({index}/{total})...', OUTPUTS)
        training = _ensure_training_targets(
            _read_input_table(raw_train_uri, crs, tile_id, pair, cfg.get('input_crs')), pair
        )
        prediction = None
        if raw_pred_uri is not None:
            prediction = _read_input_table(raw_pred_uri, crs, tile_id, pair, cfg.get('input_crs'))
            prediction = _ensure_pair_dates(prediction, pair)
        predictors, missing_train, missing_pred = _resolve_predictors(
            pair, master_predictors, training.columns,
            prediction.columns if prediction is not None else None,
            cfg['missing_predictor_policy'],
        )
        training = _align_predictors(training, predictors)
        if prediction is not None:
            prediction = _align_predictors(prediction, predictors)
        essential_fields = ['X', 'Y', 'bathy_t', 'bathy_t1', 'delta_bathy', 'delta_rate', 'interval_years']
        keep = np.isfinite(training[essential_fields].to_numpy(dtype=float)).all(axis=1)
        keep &= training['interval_years'].to_numpy() > 0
        if cfg['pre_xgb_data_mode'] == 'legacy_complete':
            keep &= np.isfinite(training[predictors].to_numpy(dtype=float)).all(axis=1)
        fitting = training.loc[keep].copy().reset_index(drop=True)
        if len(fitting) < cfg['minimum_training_rows']:
            Engine.write_message_dask(f'[SKIP] {tile_id}, {pair}: only {len(fitting)} fitting rows.', OUTPUTS)
            return {**result, 'ok': True, 'status': 'skipped', 'reason': 'insufficient_training_rows', 'n_fit': len(fitting)}
        fitting['sample_weight'] = (_calculate_sample_weights(
            fitting['delta_bathy'].to_numpy(dtype=float), cfg['weight_alpha'],
            cfg['weight_method'], cfg['weight_cap_quantile'], cfg['max_weight'],
        ) if cfg['use_weighted_loss'] else np.ones(len(fitting)))
        diagnostics = {
            'schema': pd.DataFrame({'predictor': predictors}),
            'training_missingness': _summarize_missingness(training, predictors),
            'ranges': _predictor_range_table(fitting, predictors),
        }
        if prediction is not None:
            diagnostics['prediction_missingness'] = _summarize_missingness(prediction, predictors)
            prediction = prediction.reset_index(drop=True)
            usable = np.isfinite(prediction[['X', 'Y', 'bathy_t']].to_numpy(dtype=float)).all(axis=1)
            if cfg['pre_xgb_data_mode'] == 'legacy_complete':
                usable &= np.isfinite(prediction[predictors].to_numpy(dtype=float)).all(axis=1)
            prediction['prediction_row_usable'] = usable
        # Each worker gets its own staging directory, even for retries of the same task.
        with tempfile.TemporaryDirectory(dir=local_tmp_dir) as task_tmp:
            staged = {key: Path(task_tmp) / path.name for key, path in paths.items()}
            for key, table in diagnostics.items():
                table.to_parquet(staged[key], index=False, compression='zstd')
            _save_geoparquet(fitting, staged['training'], crs)
            if prediction is not None:
                _save_geoparquet(prediction, staged['prediction'], crs)
            for key, destination in paths.items():
                _publish_file(staged[key], destination, is_aws)
            # A training-only rerun must not leave obsolete ready prediction files.
            if prediction is None:
                for key, obsolete in _output_paths(out_train_dir, out_pred_dir, pair, True).items():
                    if key not in paths and obsolete.exists():
                        obsolete.unlink()
            record = {
                'status': 'success',
                **result, 'n_fit': len(fitting),
                'outputs': [str(path) for path in paths.values()],
            }
            staged_marker = Path(task_tmp) / marker.name
            staged_marker.write_text(json.dumps(record, indent=2, allow_nan=False))
            _publish_file(staged_marker, marker, is_aws)
        return {**result, 'ok': True, 'status': 'success', 'n_fit': len(fitting),
                'schema_locked': True, 'missing_training': missing_train, 'missing_prediction': missing_pred}
    except Exception as error:
        Engine.write_message_dask(f'ERROR preparing {tile_id}, {pair}: {type(error).__name__}: {error}', OUTPUTS)
        return {**result, 'ok': False, 'status': 'failed', 'reason': f'{type(error).__name__}: {error}'}
    finally:
        gc.collect()


class DataPrepEngine(Engine):
    """Use the same region/path/worker orchestration pattern as TrainingRastersEngine."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        self.is_aws = param_lookup['env'] in ['remote', 'aws']
        self.local_tmp_dir = None
        self.results = []
        self.settings = {
            'master_predictors': None,
            'master_predictors_path': None,
            'input_crs': None,
            'missing_predictor_policy': 'fill_nan',
            'pre_xgb_data_mode': 'native_missing',
            'require_prediction_batch': False,
            'use_weighted_loss': True,
            'weight_alpha': 0.5,
            'weight_method': 'power',
            'weight_cap_quantile': 0.95,
            'max_weight': 2.0,
            'minimum_training_rows': 100,
            'verbose_logging': True,
        }
        overrides = {**param_lookup.get('data_prep_settings', {}), **param_lookup.get('model_config', {})}
        # Retain explicit legacy policy choices; new configurations use one policy key.
        legacy_mode = overrides.pop('predictor_resolution_mode', None)
        legacy_strict = overrides.pop('strict_requested_predictors', None)
        if legacy_mode not in {None, 'og_training_first', 'deployment_intersection'}:
            raise ValueError(f'Unknown predictor_resolution_mode: {legacy_mode}')
        if 'missing_predictor_policy' not in overrides:
            if legacy_strict is True:
                overrides['missing_predictor_policy'] = 'error'
            elif legacy_mode == 'deployment_intersection':
                overrides['missing_predictor_policy'] = 'drop'
        self.settings.update(overrides)
        self._validate_settings()

    def _validate_settings(self) -> None:
        """Check that all settings are valid and consistent."""

        cfg = self.settings
        if cfg['missing_predictor_policy'] not in {'fill_nan', 'drop', 'error'}:
            raise ValueError('missing_predictor_policy must be fill_nan, drop, or error')
        if cfg['pre_xgb_data_mode'] not in {'native_missing', 'legacy_complete'}:
            raise ValueError('pre_xgb_data_mode must be native_missing or legacy_complete')
        if (isinstance(cfg['minimum_training_rows'], bool)
                or not isinstance(cfg['minimum_training_rows'], int) or cfg['minimum_training_rows'] < 1):
            raise ValueError('minimum_training_rows must be a positive integer')
        if cfg['weight_method'] not in {'power', 'capped_linear'}:
            raise ValueError('weight_method must be power or capped_linear')
        for key in ('weight_alpha', 'weight_cap_quantile', 'max_weight'):
            if not np.isfinite(cfg[key]):
                raise ValueError(f'{key} must be finite')
        if cfg['weight_alpha'] < 0 or not 0 < cfg['weight_cap_quantile'] <= 1 or cfg['max_weight'] < 1:
            raise ValueError('Require alpha >= 0, 0 < cap quantile <= 1, and max_weight >= 1')
        if cfg['input_crs'] is not None:
            cfg['input_crs'] = CRS.from_user_input(cfg['input_crs']).to_string()

    def _resolve_paths(self, region: str) -> None:
        """Resolve configured local or S3 directories for the current ecoregion."""
        
        self.region = region
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.write_message(f'DataPrepEngine outputs for {region}: {self.outputs_dir}', OUTPUTS)
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        s3_dir_base = f's3://{bucket}/{self.output_prefix}/{region}' if self.output_prefix else f's3://{bucket}/{region}'

        training_tiles_dir = get_config_item('MODEL', 'TRAINING_TILES_DIR')
        self.training_tiles_dir = UPath(f'{s3_dir_base}/{training_tiles_dir}') if self.is_aws else UPath(self.outputs_dir / training_tiles_dir)

        prediction_tiles_dir = get_config_item('MODEL', 'PREDICTION_TILES_DIR')
        self.prediction_tiles_dir = UPath(f'{s3_dir_base}/{prediction_tiles_dir}') if self.is_aws else UPath(self.outputs_dir / prediction_tiles_dir)

        ml_prep_train_dir = get_config_item('MODEL', 'ML_PREP_TRAIN_DIR')
        self.ml_prep_train_dir = UPath(f'{s3_dir_base}/{ml_prep_train_dir}') if self.is_aws else UPath(self.outputs_dir / ml_prep_train_dir)

        ml_prep_pred_dir = get_config_item('MODEL', 'ML_PREP_PRED_DIR')
        self.ml_prep_pred_dir = UPath(f'{s3_dir_base}/{ml_prep_pred_dir}') if self.is_aws else UPath(self.outputs_dir / ml_prep_pred_dir)

    def _load_master_predictors(self) -> list[str]:
        """Read an independent schema source, never a preparation output."""
        
        predictors = self.settings['master_predictors']
        if predictors is not None:
            return _normalize_master_predictors(predictors)
        source = self.settings['master_predictors_path']
        if not source:
            try:
                source = get_config_item('MODEL', 'MASTER_PREDICTORS_PATH')
            except (KeyError, ConfigError):
                source = None
        if not source:
            raise ValueError('Set master_predictors or master_predictors_path (or MODEL.MASTER_PREDICTORS_PATH)')
        source = str(source).format(region=self.region)
        path = UPath(source)
        if '://' not in source and not Path(source).is_absolute():
            path = UPath(INPUTS / source)
        if not path.is_file():
            raise FileNotFoundError(f'Master predictor list not found: {path}')
        if path.suffix.lower() == '.parquet':
            table = pd.read_parquet(str(path))
            if 'predictor' not in table:
                raise ValueError('Master predictor Parquet must contain a predictor column')
            predictors = table['predictor'].tolist()
        elif path.suffix.lower() == '.json':
            predictors = json.loads(path.read_text())
        else:
            predictors = [line.strip() for line in path.read_text().splitlines()
                          if line.strip() and not line.lstrip().startswith('#')]
        return _normalize_master_predictors(predictors)

    def collect_training_files(self) -> list:
        """Collect all training batch files for the current ecoregion, sorted by name."""

        if not self.training_tiles_dir.is_dir():
            raise FileNotFoundError(f'Training batch directory not found: {self.training_tiles_dir}')
        files = sorted(self.training_tiles_dir.rglob('*_training_batch.parquet'), key=str)
        self.write_message(f'Found {len(files)} training batches.', OUTPUTS)
        return files

    def _build_task_params(self, training_files: list, overwrite: bool = False) -> list[list]:
        """Match raw batches by tile/pair; workers determine whether outputs are complete."""
        
        predictors = self._load_master_predictors()
        prediction_files = {}
        if self.prediction_tiles_dir.is_dir():
            for path in self.prediction_tiles_dir.rglob('*_prediction_batch.parquet'):
                existing = prediction_files.get(path.name)

                if existing is not None:
                    # Ignore the same path if the listing returns it again.
                    if str(existing) == str(path):
                        continue

                    raise ValueError(
                        f'Multiple prediction batches found for {path.name}:\n'
                        f'  {existing}\n'
                        f'  {path}\n'
                        'Check PREDICTION_TILES_DIR for overlapping batch folders.'
                    )

                prediction_files[path.name] = path
        params, seen = [], set()
        for training_path in training_files:
            match = re.fullmatch(r'(.+)_(\d{4}_\d{4})_training_batch\.parquet', training_path.name)
            if match is None:
                raise ValueError(f'Unexpected training batch filename: {training_path.name}')
            tile_id, pair = match.groups()
            _pair_duration_years(pair)
            if (tile_id, pair) in seen:
                raise ValueError(f'Duplicate training batch for {tile_id}, {pair}')
            seen.add((tile_id, pair))
            prediction_path = prediction_files.get(f'{tile_id}_{pair}_prediction_batch.parquet')
            if prediction_path is None:
                if self.settings['require_prediction_batch']:
                    raise FileNotFoundError(f'Missing prediction batch for {tile_id}, {pair}')
                self.write_message(f'{tile_id}, {pair}: no prediction batch; preparing training only.', OUTPUTS)
            params.append([
                tile_id, pair, str(training_path), str(prediction_path) if prediction_path is not None else None,
                predictors, self.settings, str(self.target_crs), self.is_aws,
                str(self.ml_prep_train_dir / tile_id), str(self.ml_prep_pred_dir / tile_id),
                str(self.local_tmp_dir), len(params) + 1, len(training_files),
                self.settings['verbose_logging'], overwrite,
            ])
        return params

    def execute_tasks(self, training_files: list, overwrite: bool = False) -> list[dict]:
        """Submit all tile/pair tasks to Dask and gather results."""

        params = self._build_task_params(training_files, overwrite)
        self.write_message(f'Submitting {len(params)} ML preparation tasks...', OUTPUTS)
        # These tasks write files; repeated submissions must not reuse cached results.
        results = self.client.gather(self.client.map(_prep_tile_task, params, pure=False))
        counts = {status: sum(row['status'] == status for row in results)
                  for status in ('success', 'skipped', 'failed')}
        self.write_message(f'ML preparation results: {counts}', OUTPUTS)
        return results

    def run(self, overwrite: bool = False) -> list[dict]:
        """Initialize Dask once, process ecoregions, and separate skips from failures."""
        
        self.results = []
        try:
            self.local_tmp_dir = Path(tempfile.mkdtemp(prefix='hhm_data_prep_'))
            self.setup_dask(self.param_lookup['env'], n_workers=2, threads_per_worker=1, memory_limit='12GB')
            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)
                training_files = self.collect_training_files()
                if training_files:
                    self.write_message(f'Outputting ML tables for {eco_region} to: {self.ml_prep_train_dir}', OUTPUTS)
                    self.results.extend({**row, 'region': eco_region}
                                        for row in self.execute_tasks(training_files, overwrite))
                else:
                    self.write_message(f'No training batches to process for {eco_region}.', OUTPUTS)
            failures = [row for row in self.results if row['status'] == 'failed']
            if failures:
                raise RuntimeError(f'{len(failures)} ML preparation tasks failed: {failures}')
            return self.results
        finally:
            self.cleanup_resources(OUTPUTS)
