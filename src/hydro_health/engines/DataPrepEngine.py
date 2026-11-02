"""Prepare all training and prediction tile/year pairs independently.

Prediction export is skipped when any expected predictor is at least 20%
missing (prediction_max_missing_fraction). Skipped roles produce no cleaned
or auxiliary files; stale outputs are removed. Training with too few eligible
rows is also skipped. Each cleaned parquet is written before its auxiliaries.
Dask defaults: 16 workers, 2 threads per worker, 6GB per worker.
"""

import gc
import json
import os
import pathlib
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
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
from upath import UPath

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

REPOSITORY_ROOT = next((parent for parent in pathlib.Path(__file__).resolve().parents if (parent / 'src' / 'hydro_health').is_dir()), pathlib.Path(__file__).resolve().parents[3])
INPUTS = REPOSITORY_ROOT / 'inputs'
OUTPUTS = REPOSITORY_ROOT / 'outputs'

DYNAMIC_BASES = ('hurr_strength_mean', 'tsm_mean')
NON_PREDICTORS = {
    'X', 'Y', 'FID', 'tile_id', 'source', 'dataset_role',
    'year_t', 'year_t1', 'year_t1', 'interval_years', 'pair_id',
    'bathy_t1', 'bathy_t1', 'delta_bathy', 'delta_rate', 'survey_end_date',
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

def _resolve_predictors( pair: str, master_predictors: Sequence[str], train_columns: Sequence[str], prediction_columns: Sequence[str] | None, missing_policy: str = 'fill_nan', ) -> tuple[list[str], list[str], list[str]]:
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
        # Plain Parquet batches use the target CRS inherited from Engine
        # unless an input CRS is explicitly configured.
        actual_crs = (
            CRS.from_user_input(input_crs)
            if input_crs is not None
            else target
        )
        out = pd.read_parquet(uri)
        if 'geometry' in out:
            raise ValueError(
                f'Geometry without GeoParquet CRS metadata is unsupported: {uri}'
            )
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

def _calculate_sample_weights( change: np.ndarray, alpha: float, method: str, cap_quantile: float, max_weight: float, epsilon: float = 1e-6, ) -> np.ndarray:
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
    """Build point geometry vectorially while retaining invalid XY as nulls."""
    valid = np.isfinite(df['X']) & np.isfinite(df['Y'])
    geometry = gpd.points_from_xy(df['X'].where(valid, 0), df['Y'].where(valid, 0), crs=crs)
    geometry[~valid.to_numpy()] = None
    gpd.GeoDataFrame(df, geometry=geometry, crs=crs).to_parquet(path, index=False, compression='zstd')


def _role_paths(directory: str, pair: str, role: str) -> dict:
    base = UPath(directory)
    paths = {'schema': base / f'locked_predictors_{pair}.parquet',
             'missingness': base / f'predictor_missingness_{role}_{pair}.parquet',
             'ready': base / f'ML_ready_{role}_{pair}.geoparquet'}
    if role == 'training':
        paths['ranges'] = base / f'predictor_ranges_training_{pair}.parquet'
    return paths


def _publish_file(staged: Path, destination: UPath, is_aws: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if is_aws:
        with staged.open('rb') as source, destination.open('wb') as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        if destination.stat().st_size != staged.stat().st_size:
            raise OSError(f'Upload size mismatch: {destination}')
    else:
        with tempfile.NamedTemporaryFile(dir=str(destination.parent), delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(staged, temporary)
            os.replace(temporary, str(destination))
        finally:
            temporary.unlink(missing_ok=True)


def _write_json(value: dict, destination: UPath, is_aws: bool) -> None:
    with tempfile.TemporaryDirectory(prefix='hhm_prep_json_') as temporary:
        staged = Path(temporary) / 'record.json'
        staged.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding='utf-8')
        _publish_file(staged, destination, is_aws)


def _remove_role_outputs(task: dict, role: str, keep: set | None = None) -> None:
    paths = _role_paths(task['directories'][role], task['pair'], role)
    keep = keep or set()
    for key, path in paths.items():
        if key not in keep and path.exists():
            path.unlink()


def _quality(table: pd.DataFrame, predictors: Sequence[str]) -> dict:
    counts = {name: int(np.isfinite(table[name]).sum()) for name in predictors}
    empty = [name for name, count in counts.items() if count == 0]
    return {'n_rows': len(table), 'finite_predictor_values': counts,
            'all_missing_predictors': empty,
            'missing_fraction': {name: ((len(table) - count) / len(table)) if len(table) else None for name, count in counts.items()}}


def _prepare_role(task: dict, role: str) -> dict:
    """Prepare one role completely, releasing its tables before the other role."""
    cfg, pair, tile = task['cfg'], task['pair'], task['tile_id']
    uri = task['sources'].get(role)
    result = {'role': role, 'source': uri, 'status': 'absent', 'reason': 'no_raw_batch'}
    if not uri and role not in task.get('input_errors', {}):
        return result
    base = UPath(task['directories'][role])
    paths = _role_paths(str(base), pair, role)
    published = False
    try:
        if role in task.get('input_errors', {}):
            raise ValueError(task['input_errors'][role])
        table = _read_input_table(uri, task['crs'], tile, pair, cfg.get('input_crs'))
        table = _ensure_training_targets(table, pair) if role == 'training' else _ensure_pair_dates(table, pair)
        missing_columns = [name for name in task['predictors'] if name not in table]
        if cfg['missing_predictor_policy'] == 'error' and missing_columns:
            raise ValueError(f'Requested predictors missing: {missing_columns}')
        table = _align_predictors(table, task['predictors'])
        quality = _quality(table, task['predictors'])
        result.update(n_input=len(table), quality=quality, missing_predictor_columns=missing_columns,
                      tile_id=tile, pair=pair, role=role)
        messages = []
        if quality['all_missing_predictors']:
            messages.append(f'Entirely missing predictors: {quality["all_missing_predictors"]}')
        if missing_columns:
            messages.append(f'Absent columns filled with NaN: {missing_columns}')
        high = [name for name, fraction in quality['missing_fraction'].items() if fraction is not None and fraction >= cfg['missingness_warning_fraction'] and name not in quality['all_missing_predictors']]
        if high:
            messages.append(f'High missingness predictors: {high}')
        result['warnings'] = messages
        if role == 'training':
            fields = ['X', 'Y', 'bathy_t', 'bathy_t1', 'delta_bathy', 'delta_rate', 'interval_years']
            keep = np.isfinite(table[fields].to_numpy(dtype=float)).all(axis=1)
            keep &= table['interval_years'].to_numpy() > 0
            if cfg['pre_xgb_data_mode'] == 'legacy_complete':
                keep &= np.isfinite(table[task['predictors']].to_numpy(dtype=float)).all(axis=1)
            ready = table.loc[keep].copy().reset_index(drop=True)
            result.update(n_fit=len(ready), n_dropped=int((~keep).sum()))
            if len(ready) < cfg['minimum_training_rows']:
                result.update(status='skipped', reason='insufficient_training_rows')
            else:
                ready['sample_weight'] = _calculate_sample_weights(ready['delta_bathy'].to_numpy(dtype=float), cfg['weight_alpha'], cfg['weight_method'], cfg['weight_cap_quantile'], cfg['max_weight']) if cfg['use_weighted_loss'] else np.ones(len(ready))
                result.update(status='success', reason='prepared')
        else:
            ready = table.reset_index(drop=True)
            usable = np.isfinite(ready[['X', 'Y', 'bathy_t']].to_numpy(dtype=float)).all(axis=1)
            if cfg['pre_xgb_data_mode'] == 'legacy_complete':
                usable &= np.isfinite(ready[task['predictors']].to_numpy(dtype=float)).all(axis=1)
            ready['prediction_row_usable'] = usable
            result.update(n_usable=int(usable.sum()), n_unusable=int((~usable).sum()))
            if usable.any():
                result.update(status='success', reason='prepared')
            else:
                result.update(status='failed', reason='zero_usable_prediction_rows')
            if not usable.all():
                result['warnings'].append(f'{int((~usable).sum())}/{len(ready)} prediction rows unusable')
            limit = cfg['prediction_max_missing_fraction']
            rejected = {name: fraction for name, fraction in quality['missing_fraction'].items() if fraction is not None and fraction > 0 and limit is not None and fraction >= limit}
            if rejected:
                result.update(status='skipped', reason='high_predictor_missingness',
                              rejected_predictors=rejected, prediction_max_missing_fraction=limit)
        write_ready = result['status'] == 'success' or (role == 'prediction' and result['status'] == 'failed' and cfg['write_unusable_prediction'])
        if not write_ready:
            _remove_role_outputs(task, role)
            result['outputs'] = []
            return result
        if result['status'] == 'success' and not task['overwrite'] and all(path.is_file() for path in paths.values()):
            result.update(reason='existing_outputs', outputs=[str(path) for path in paths.values()])
            return result
        temp_parent = cfg.get('worker_temp_dir')
        if temp_parent:
            Path(temp_parent).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='hhm_prep_role_', dir=temp_parent) as temporary:
            stage = Path(temporary)
            # Write the cleaned table first; only then create its auxiliary files.
            staged = {'ready': stage / paths['ready'].name}
            _save_geoparquet(ready, staged['ready'], task['crs'])
            _publish_file(staged['ready'], paths['ready'], task['is_aws'])
            diagnostics = {'schema': pd.DataFrame({'predictor': task['predictors']}),
                           'missingness': _summarize_missingness(table, task['predictors'])}
            if role == 'training':
                diagnostics['ranges'] = _predictor_range_table(ready, task['predictors'])
            del table, ready
            for key, diagnostic in diagnostics.items():
                staged[key] = stage / paths[key].name
                diagnostic.to_parquet(staged[key], index=False, compression='zstd')
                _publish_file(staged[key], paths[key], task['is_aws'])
            _remove_role_outputs(task, role, keep=set(staged))
            result['outputs'] = [str(paths[key]) for key in staged]
            published = True
        return result
    except Exception as exc:
        result.update(status='failed', reason=f'{type(exc).__name__}: {exc}', error_type=type(exc).__name__)
        try:
            if not published:
                _remove_role_outputs(task, role)
        except Exception as cleanup_error:
            result['cleanup_error'] = str(cleanup_error)
        return result
    finally:
        gc.collect()


def _prep_tile_task(task: dict) -> dict:
    """Prepare both roles; a skipped or failed role does not block the other."""
    roles = {role: _prepare_role(task, role) for role in ('training', 'prediction')}
    failures = any(item['status'] == 'failed' for item in roles.values())
    successes = any(item['status'] == 'success' for item in roles.values())
    status = ('partial' if successes else 'failed') if failures else ('success' if successes else 'skipped')
    return {'tile_id': task['tile_id'], 'pair': task['pair'], 'region': task['region'],
            'status': status, 'ok': not failures, 'roles': roles}


class DataPrepEngine(Engine):
    """Prepare the union of raw training/prediction tiles, using independent roles."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        if output_prefix and (not isinstance(output_prefix, str) or output_prefix.startswith('/') or '://' in output_prefix or '..' in Path(output_prefix).parts):
            raise ValueError('output_prefix must be a relative prefix or False')
        self.is_aws = param_lookup['env'] in ['remote', 'aws']
        self.local_tmp_dir = None
        self.results, self.discovery_issues = [], []
        self.settings = {
            'master_predictors': None, 'master_predictors_path': None, 'input_crs': None,
            'missing_predictor_policy': 'fill_nan', 'pre_xgb_data_mode': 'native_missing',
            'require_prediction_batch': False, 'use_weighted_loss': True,
            'weight_alpha': 0.5, 'weight_method': 'power', 'weight_cap_quantile': 0.95,
            'max_weight': 2.0, 'minimum_training_rows': 100, 'verbose_logging': True,
            'write_unusable_prediction': True,
            'missingness_warning_fraction': 0.20, 'worker_temp_dir': None,
            'prediction_max_missing_fraction': 0.20,
            'n_workers': 16, 'threads_per_worker': 2, 'memory_limit': '6GB',
            'raise_on_failure': True,
        }
        overrides = {**param_lookup.get('data_prep_settings', {}), **param_lookup.get('model_config', {})}
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
        cfg = self.settings
        if cfg['missing_predictor_policy'] not in {'fill_nan', 'drop', 'error'}:
            raise ValueError('missing_predictor_policy must be fill_nan, drop, or error')
        if cfg['pre_xgb_data_mode'] not in {'native_missing', 'legacy_complete'}:
            raise ValueError('pre_xgb_data_mode must be native_missing or legacy_complete')
        for name in ('minimum_training_rows', 'n_workers', 'threads_per_worker'):
            if isinstance(cfg[name], bool) or not isinstance(cfg[name], int) or cfg[name] < 1:
                raise ValueError(f'{name} must be a positive integer')
        if cfg['weight_method'] not in {'power', 'capped_linear'}:
            raise ValueError('weight_method must be power or capped_linear')
        for name in ('weight_alpha', 'weight_cap_quantile', 'max_weight', 'missingness_warning_fraction'):
            if not np.isfinite(cfg[name]):
                raise ValueError(f'{name} must be finite')
        if cfg['weight_alpha'] < 0 or not 0 < cfg['weight_cap_quantile'] <= 1 or cfg['max_weight'] < 1:
            raise ValueError('Invalid weighting settings')
        if not 0 <= cfg['missingness_warning_fraction'] <= 1:
            raise ValueError('missingness_warning_fraction must be in [0, 1]')
        limit = cfg['prediction_max_missing_fraction']
        if limit is not None and (isinstance(limit, bool) or not np.isfinite(limit) or not 0 <= limit <= 1):
            raise ValueError('prediction_max_missing_fraction must be in [0, 1], or None to disable the prediction quality gate')
        for name in ('require_prediction_batch', 'use_weighted_loss', 'verbose_logging', 'write_unusable_prediction', 'raise_on_failure'):
            if not isinstance(cfg[name], bool):
                raise ValueError(f'{name} must be boolean')
        if cfg['input_crs'] is not None:
            cfg['input_crs'] = CRS.from_user_input(cfg['input_crs']).to_string()

    def _resolve_paths(self, region: str) -> None:
        self.region = region
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        bucket = str(get_config_item('SHARED', 'OUTPUT_BUCKET')).strip('/').removeprefix('s3://')
        self.s3_dir_base = f's3://{bucket}/{self.output_prefix}/{region}' if self.output_prefix else f's3://{bucket}/{region}'
        for attribute, key in [('training_tiles_dir', 'TRAINING_TILES_DIR'), ('prediction_tiles_dir', 'PREDICTION_TILES_DIR'), ('ml_prep_train_dir', 'ML_PREP_TRAIN_DIR'), ('ml_prep_pred_dir', 'ML_PREP_PRED_DIR')]:
            folder = str(get_config_item('MODEL', key)).strip('/')
            if not folder or '://' in folder or '..' in Path(folder).parts:
                raise ValueError(f'MODEL.{key} must be a relative subfolder')
            setattr(self, attribute, UPath(f'{self.s3_dir_base}/{folder}') if self.is_aws else UPath(self.outputs_dir / folder))
        reports = str(get_config_item('REPORTS', 'SUBFOLDER')).strip('/')
        if not reports or '://' in reports or '..' in Path(reports).parts:
            raise ValueError('REPORTS.SUBFOLDER must be relative')
        self.reports_dir = UPath(f'{self.s3_dir_base}/{reports}') if self.is_aws else UPath(self.outputs_dir / reports)
        self.write_message(f'DataPrepEngine outputs for {region}: {self.outputs_dir}', OUTPUTS)

    def _load_master_predictors(self) -> list[str]:
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
            with path.open('rb') as stream:
                predictors = pd.read_parquet(stream)['predictor'].tolist()
        elif path.suffix.lower() == '.json':
            predictors = json.loads(path.read_text())
        else:
            predictors = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        return _normalize_master_predictors(predictors)

    def _collect_batches(self, directory: UPath, role: str) -> list:
        if not directory.is_dir():
            self.write_message(f'No raw {role} directory: {directory}', OUTPUTS)
            return []
        return sorted(set(directory.rglob(f'*_{role}_batch.parquet')), key=str)

    def collect_training_files(self) -> list:
        return self._collect_batches(self.training_tiles_dir, 'training')

    def collect_prediction_files(self) -> list:
        return self._collect_batches(self.prediction_tiles_dir, 'prediction')

    def _build_task_params(self, training_files: list | None = None, overwrite: bool = True) -> list[dict]:
        master = self._load_master_predictors()
        inventory = {'training': {}, 'prediction': {}}
        for role, directory in [('training', self.training_tiles_dir), ('prediction', self.prediction_tiles_dir)]:
            try:
                files = training_files if role == 'training' and training_files is not None else self._collect_batches(directory, role)
            except Exception as exc:
                self.discovery_issues.append({'region': self.region, 'role': role, 'path': str(directory), 'reason': f'Discovery failed: {type(exc).__name__}: {exc}'})
                files = []
            for path in files:
                match = re.fullmatch(r'(.+)_(\d{4}_\d{4})_' + role + r'_batch\.parquet', path.name)
                if match is None:
                    self.discovery_issues.append({'region': self.region, 'role': role, 'path': str(path), 'reason': 'Unexpected batch filename'})
                    continue
                tile, pair = match.groups()
                if tile in {'.', '..'} or '/' in tile or '\\' in tile:
                    self.discovery_issues.append({'region': self.region, 'role': role, 'path': str(path), 'reason': 'Unsafe tile identifier in batch filename'})
                    continue
                try:
                    _pair_duration_years(pair)
                except ValueError as exc:
                    self.discovery_issues.append({'region': self.region, 'role': role, 'path': str(path), 'reason': str(exc)})
                    continue
                inventory[role].setdefault((tile, pair), []).append(str(path))
        identities = sorted(set(inventory['training']) | set(inventory['prediction']))
        if not identities:
            self.discovery_issues.append({'region': self.region, 'reason': 'No valid raw training or prediction batches were found in the configured directories'})
        schemas = {pair: _build_pair_specific_predictors(master, pair) for _, pair in identities}
        header_errors = {}
        if self.settings['missing_predictor_policy'] == 'drop':
            # Only this explicit legacy policy needs a regional metadata preflight.
            # Every tile for a pair receives the same ordered feature list.
            for role, entries in inventory.items():
                for identity, sources in entries.items():
                    if len(sources) != 1:
                        continue
                    try:
                        with UPath(sources[0]).open('rb') as stream:
                            columns = set(pq.read_schema(stream).names)
                        pair = identity[1]
                        schemas[pair] = [name for name in schemas[pair] if name == 'bathy_t' or name in columns]
                    except Exception as exc:
                        header_errors[(role, identity)] = f'Cannot inspect raw schema: {exc}'
        tasks = []
        for tile, pair in identities:
            task = {'tile_id': tile, 'pair': pair, 'region': self.region, 'sources': {}, 'input_errors': {},
                    'predictors': schemas[pair], 'cfg': self.settings, 'crs': str(self.target_crs),
                    'is_aws': self.is_aws, 'overwrite': overwrite,
                    'directories': {'training': str(self.ml_prep_train_dir / tile), 'prediction': str(self.ml_prep_pred_dir / tile)}}
            for role in ('training', 'prediction'):
                candidates = inventory[role].get((tile, pair), [])
                task['sources'][role] = candidates[0] if len(candidates) == 1 else None
                if len(candidates) > 1:
                    task['input_errors'][role] = f'Ambiguous duplicate {role} batches: {candidates}'
                if (role, (tile, pair)) in header_errors:
                    task['input_errors'][role] = header_errors[(role, (tile, pair))]
            if task['sources']['training'] and not task['sources']['prediction'] and self.settings['require_prediction_batch']:
                task['input_errors'].setdefault('prediction', 'A prediction batch is required but none matched this training tile/year pair')
            tasks.append(task)
        return tasks

    def _write_run_summary(self, path: UPath, region: str, results: list, state: str) -> None:
        role_counts = {role: {status: sum(row.get('roles', {}).get(role, {}).get('status') == status for row in results) for status in ('success', 'skipped', 'failed', 'absent')} for role in ('training', 'prediction')}
        _write_json({'region': region, 'run_id': self.run_id, 'state': state,
                     'updated_at': datetime.now(timezone.utc).isoformat(),
                     'tasks_completed': len(results), 'role_counts': role_counts,
                     'discovery_issues': [item for item in self.discovery_issues if item['region'] == region],
                     'results': results}, path, self.is_aws)

    def execute_tasks(self, training_files: list | None = None, overwrite: bool = True) -> list[dict]:
        from dask.distributed import as_completed
        tasks = self._build_task_params(training_files, overwrite)
        self.write_message(f'Submitting {len(tasks)} union training/prediction preparation tasks for {self.region}...', OUTPUTS)
        results = []
        futures = {future: task for future, task in zip(self.client.map(_prep_tile_task, tasks, pure=False), tasks)}
        try:
            for completed, future in enumerate(as_completed(futures), 1):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {'tile_id': task['tile_id'], 'pair': task['pair'], 'region': task['region'],
                              'status': 'failed', 'ok': False, 'reason': f'Dask worker failure: {type(exc).__name__}: {exc}', 'roles': {}}
                results.append(result)
                self.results.append(result)
                failures = [f'{role}: {item.get("reason")}' for role, item in result.get('roles', {}).items() if item.get('status') == 'failed']
                label = 'FAIL' if not result.get('ok') else 'DONE'
                self.write_message(f'[{label}] {self.region}/{task["tile_id"]}/{task["pair"]} ({completed}/{len(tasks)}): {result["status"]} {"; ".join(failures)}', OUTPUTS)
                for role, item in result.get('roles', {}).items():
                    if item.get('status') == 'skipped':
                        rejected = ', '.join(f'{name}={100 * fraction:.1f}%' for name, fraction in item.get('rejected_predictors', {}).items())
                        self.write_message(f'[SKIP] {self.region}/{task["tile_id"]}/{task["pair"]} {role}: {item.get("reason")}' + (f'; missingness: {rejected}' if rejected else ''), OUTPUTS)
                    for message in item.get('warnings', []):
                        self.write_message(f'[WARN] {self.region}/{task["tile_id"]}/{task["pair"]} {role}: {message}', OUTPUTS)
        finally:
            for future in futures:
                future.release()
        return results

    def run(self, overwrite: bool = True) -> list[dict]:
        self.results, self.discovery_issues = [], []
        regions = self.param_lookup['eco_regions'].value
        if regions is None or isinstance(regions, (str, bytes)):
            raise ValueError("param_lookup['eco_regions'].value must contain selected region names")
        regions = list(regions)
        if not regions or any(not isinstance(region, str) or not region.strip() for region in regions):
            raise ValueError("param_lookup['eco_regions'].value must contain selected region names")
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
        self.local_tmp_dir = Path(tempfile.mkdtemp(prefix='hydro_health_data_prep_'))
        try:
            self.setup_dask(self.param_lookup['env'], n_workers=self.settings['n_workers'], threads_per_worker=self.settings['threads_per_worker'], memory_limit=self.settings['memory_limit'])
            for eco_region in regions:
                summary_path = None
                try:
                    self._resolve_paths(eco_region)
                    summary_path = self.reports_dir / f'data_prep_run_{self.run_id}.json'
                    self._write_run_summary(summary_path, eco_region, [], 'running')
                    self.execute_tasks(overwrite=overwrite)
                except Exception as exc:
                    self.discovery_issues.append({'region': eco_region, 'reason': f'{type(exc).__name__}: {exc}'})
                    self.write_message(f'[FAIL] {eco_region}: {exc}', OUTPUTS)
                finally:
                    region_results = [row for row in self.results if row['region'] == eco_region]
                    if summary_path is not None:
                        state = 'completed_with_errors' if any(not row.get('ok') for row in region_results) or any(item['region'] == eco_region for item in self.discovery_issues) else 'completed'
                        try:
                            self._write_run_summary(summary_path, eco_region, region_results, state)
                        except Exception as exc:
                            self.discovery_issues.append({'region': eco_region, 'reason': f'Run summary write failed: {exc}'})
            failed = [row for row in self.results if not row.get('ok')]
            if self.settings['raise_on_failure'] and (failed or self.discovery_issues):
                examples = [f'{row["region"]}/{row["tile_id"]}/{row["pair"]}: {row.get("reason", row["status"])}' for row in failed[:10]]
                raise RuntimeError(f'{len(failed)} preparation tasks had failures; {len(self.discovery_issues)} discovery/report issues. Successful independent roles were retained. Examples: {examples}. See REPORTS summaries.')
            return self.results
        finally:
            active_error = sys.exc_info()[0] is not None
            try:
                self.cleanup_resources(OUTPUTS)
            except Exception as exc:
                if not active_error:
                    raise
                self.write_message(f'[WARN] Resource cleanup failed without replacing the original error: {exc}', OUTPUTS)
            finally:
                shutil.rmtree(self.local_tmp_dir, ignore_errors=True)
