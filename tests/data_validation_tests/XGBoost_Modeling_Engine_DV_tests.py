"""Validate existing XGBoostModelingEngine outputs; never train or overwrite them.

Install in tests/data_validation_tests and run in the hydro_health environment:
    python ~/Repos/hydro_health/tests/data_validation_tests/XGBoost_Modeling_Engine_DV_tests.py --regions ER_3

Paths use SHARED.OUTPUT_BUCKET, MODEL.ML_PREP_TRAIN_DIR,
MODEL.ML_PREP_PRED_DIR, MODEL.XGB_OUTPUT_DIR and REPORTS.SUBFOLDER.
Default: AWS, 16 standard-library worker processes (no per-worker memory cap).
Optional --execution dask uses 1 thread and --memory-limit 6GB per worker.
Use --env local or --execution processes/serial as needed. All source pairs
are inventoried before validation, including missing outputs and prediction-only
pairs. Replay checks ALL usable rows in chunks, never a sample. CSV and readable
text reports include every tile/pair and issue; no audit JSON is written.
One combined PDF is retained locally and uploaded to the configured S3 reports
folder, even with local inputs; --local-only explicitly disables S3 report writes.
Use run_validation(param_lookup, output_prefix) from the project runner.
PDF generation additionally requires reportlab.

PASS means output-contract consistency, not demonstrated predictive skill.
The engine's CV holdouts select rounds; they are not an untouched final test set.
Bootstrap SD is ensemble spread, not a calibrated prediction interval.
The attached legacy tests supplied useful replay/row/formula/CV checks, but
legacy *_next names, plain parquet summaries, square blocks and saved draw
indices are not requirements of the supplied current engine.
"""
import os

# Set before numerical imports, including in spawned workers.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import csv
import io
import json
import multiprocessing
import re
import shutil
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import pandas as pd
import reportlab
import xgboost as xgb
from pyproj import CRS
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, TableStyle
from upath import UPath

# Dask is optional when using the standard process pool or serial execution.
try:
    from dask.distributed import as_completed
except ImportError:
    as_completed = None

# Make repository imports work when this file is run directly from tests/.
_import_parser = argparse.ArgumentParser(add_help=False)
_import_parser.add_argument('--repo-root', type=Path)
_import_options, _ = _import_parser.parse_known_args()
_import_root = _import_options.repo_root
if _import_root is None:
    _import_root = next((candidate for candidate in (*Path(__file__).resolve().parents, Path.cwd(), *Path.cwd().parents)
                         if (candidate / 'src/hydro_health').is_dir()), None)
if _import_root is not None:
    sys.path.insert(0, str(_import_root.resolve() / 'src'))

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

DERIVED = ['mean_predicted_bathy_t1', 'mean_predicted_change', 'mean_predicted_rate',
           'mean_predicted_standard_change', 'mean_predicted_standard_bathy',
           'uncertainty_sd_bathy_t1', 'uncertainty_sd_change', 'uncertainty_sd_rate']
FORBIDDEN = {'X', 'Y', 'FID', 'tile_id', 'source', 'dataset_role', 'year_t', 'year_t1',
             'interval_years', 'pair_id', 'bathy_t1', 'delta_bathy', 'delta_rate',
             'survey_end_date', 'geometry', 'prediction_row_usable', 'sample_weight'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, rtol=2e-5, atol=2e-5):
    """Float32-aware comparisons, with matching NaNs required."""
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, f'{label}: shapes differ {a.shape} versus {b.shape}')
    good = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    require(good.all(), f'{label}: {int((~good).sum())} mismatched values')


def read_table(path, spatial=False):
    with UPath(path).open('rb') as stream:
        if spatial:
            return gpd.read_parquet(stream)
        return pd.read_parquet(stream)


def read_json(path):
    with UPath(path).open('r') as stream:
        return json.load(stream)


def expected_hex_blocks(frame, width):
    """Reconstruct axial blocks from source XY, rather than trust saved block IDs."""
    xy = frame[['X', 'Y']].to_numpy(dtype=np.float64)
    xy -= xy.min(axis=0) + (xy.max(axis=0)-xy.min(axis=0))/2
    radius = width/np.sqrt(3)
    q = (np.sqrt(3)/3*xy[:, 0]-xy[:, 1]/3)/radius
    r = (2/3*xy[:, 1])/radius
    cubes = np.column_stack([q, -q-r, r])
    rounded = np.rint(cubes)
    adjust = np.argmax(np.abs(cubes-rounded), axis=1)
    for axis in range(3):
        selected = adjust == axis
        rounded[selected, axis] = -rounded[selected][:, [a for a in range(3) if a != axis]].sum(axis=1)
    return rounded[:, [0, 2]].astype(np.int32)


def paths_for(base, tile, pair):
    training = base / tile / 'Training'
    paths = {name: str(training / f'LOCAL_{name}_{pair}.{ext}') for name, ext in (
        ('model_metadata', 'json'), ('apparent_training_metrics', 'json'),
        ('importance', 'parquet'), ('shap_summary', 'parquet'),
        ('spatial_cv_folds', 'parquet'), ('spatial_cv_metrics', 'parquet'), ('spatial_cv_oof', 'parquet'))}
    for role in ('training', 'prediction'):
        paths[role] = str(base / tile / role.title() / f'LOCAL_{role}_summary_{pair}.geoparquet')
    return paths


def inventory(train_dir, pred_dir, model_dir, args):
    """Union inventory prevents successful outputs from defining the denominator."""
    groups = {}
    patterns = ((train_dir, 'ML_ready_training_*.geoparquet', 'training'),
                (pred_dir, 'ML_ready_prediction_*.geoparquet', 'prediction'),
                (model_dir, 'LOCAL_*', 'output'), (model_dir, 'model_boot_*.json', 'output'))
    for root, pattern, role in patterns:
        for path in root.rglob(pattern):
            match = re.search(r'(\d{4}_\d{4})\.(?:geoparquet|parquet|json)$', path.name)
            if not match:
                continue
            pair = match[1]
            tile = path.parent.name if role != 'output' else path.parent.parent.name
            if role == 'output' and path.parent.name == 'models':
                tile = path.parent.parent.parent.name
            if args.tiles and tile not in args.tiles or args.pairs and pair not in args.pairs:
                continue
            record = groups.setdefault((tile, pair), {'training': [], 'prediction': [], 'output': []})
            record[role].append(str(path))
    tasks = []
    for (tile, pair), record in sorted(groups.items()):
        tasks.append({'tile': tile, 'pair': pair, 'region': args.region,
                      'sources': record, 'paths': paths_for(model_dir, tile, pair),
                      'schema': str(train_dir / tile / f'locked_predictors_{pair}.parquet'),
                      'model_dir': str(model_dir), 'chunk_rows': args.chunk_rows,
                      'rtol': args.rtol, 'atol': args.atol, 'expected_bootstrap': args.expected_bootstrap})
    return tasks


def validate_task(task):
    """Each check fails independently; return concrete issues instead of killing workers."""
    result = {key: task[key] for key in ('region', 'tile', 'pair')}
    result.update(status='PASS', checks_passed=0, issues=[], training_rows=0, prediction_rows=0,
                  usable_prediction_rows=0, cv_rmse=None, baseline_cv_rmse=None)
    issues = result['issues']

    def issue(level, check, message):
        issues.append({'level': level, 'check': check, 'message': str(message)})

    def check(name, action):
        try:
            value = action()
            result['checks_passed'] += 1
            return value
        except Exception as exc:
            issue('FAIL', name, f'{type(exc).__name__}: {exc}')
            return None

    def same(a, b, label):
        close(a, b, label, task['rtol'], task['atol'])

    sources, paths = task['sources'], task['paths']
    train = pred = meta = schema = None
    start, end = map(int, task['pair'].split('_'))
    if end <= start:
        issue('FAIL', 'pair', 'Year pair must be increasing')
    for role in ('training', 'prediction'):
        if len(sources[role]) > 1:
            issue('FAIL', 'inventory', f'Duplicate {role} inputs: {sources[role]}')
    if not sources['training']:
        if sources['output']:
            issue('FAIL', 'coverage', 'Model output has no corresponding prepared training input')
        else:
            issue('WARN', 'coverage', 'Prediction-only pair: current engine schedules training inputs only; no model expected')
    if sources['training'] and not sources['prediction']:
        issue('FAIL', 'coverage', 'Missing prepared prediction input; engine skips this training pair')
    expected = bool(sources['training']) or bool(sources['output'])
    if expected:
        for name, path in paths.items():
            if not UPath(path).is_file():
                issue('FAIL', 'bundle', f'Missing {name}: {path}')
        meta = check('metadata readable', lambda: read_json(paths['model_metadata']))
    if sources['training']:
        train = check('training input readable', lambda: read_table(sources['training'][0], True))
        schema_path = UPath(sources['training'][0]).parent / f"locked_predictors_{task['pair']}.parquet"
        schema = check('locked schema readable', lambda: read_table(schema_path).predictor.tolist())
    if sources['prediction']:
        pred = check('prediction input readable', lambda: read_table(sources['prediction'][0], True))

    def source_check(frame, role):
        require(not frame.empty and frame.columns.is_unique, f'{role}: empty or duplicate columns')
        require(frame.crs is not None, f'{role}: no CRS')
        if task.get('target_crs'):
            require(CRS(frame.crs) == CRS(task['target_crs']), f'{role}: CRS differs from project target CRS {task["target_crs"]}')
        require(frame.FID.notna().all() and frame.FID.is_unique, f'{role}: null/duplicate FID')
        for col, value in (('tile_id', task['tile']), ('pair_id', task['pair']),
                           ('year_t', start), ('year_t1', end), ('interval_years', end-start)):
            if col in frame:
                require(frame[col].eq(value).all(), f'{role}: incorrect {col}, expected {value}')
        required = ['X', 'Y', 'bathy_t', 'interval_years'] + (['bathy_t1', 'delta_bathy', 'delta_rate'] if role == 'training' else [])
        require(set(required).issubset(frame), f'{role}: missing {sorted(set(required)-set(frame))}')
        if role == 'training':
            require(np.isfinite(frame[['X', 'Y', 'bathy_t', 'bathy_t1', 'interval_years']]).all().all(), 'Invalid essential training fields')
            same(frame.delta_bathy, frame.bathy_t1-frame.bathy_t, 'Input delta sign')
            same(frame.delta_rate, frame.delta_bathy/frame.interval_years, 'Input delta rate')
        else:
            usable = np.isfinite(frame[['X', 'Y', 'bathy_t', 'interval_years']]).all(axis=1) & frame.interval_years.gt(0)
            if 'prediction_row_usable' in frame:
                require(frame.prediction_row_usable.isin([True, False]).all(), 'Invalid prediction usability flags')
                usable &= frame.prediction_row_usable.astype(bool)
            result['usable_prediction_rows'] = int(usable.sum())
            if not usable.any():
                issue('WARN', 'prediction eligibility', 'No usable prediction rows; the current engine skips this pair')
        result[f'{role}_rows'] = len(frame)

    for role, frame in (('training', train), ('prediction', pred)):
        if frame is not None:
            check(f'{role} identity/inputs', lambda f=frame, r=role: source_check(f, r))
    models = []
    cfg = {}
    if meta is not None and schema is not None:
        cfg = meta.get('model_config', {})
        result['predictors'] = schema

        def metadata_check():
            require(meta['tile_id'] == task['tile'] and meta['pair'] == task['pair'], 'Metadata belongs to different tile/pair')
            require(meta['target'] == 'bathy_t1' and meta['model_baseline_feature'] == 'bathy_t', 'Incorrect target/baseline')
            require(meta['training_baseline_source'] == 'bathy_t' and meta['prediction_baseline_source'] == 'bathy_t', 'Incorrect baseline sources')
            require(schema and len(set(schema)) == len(schema) and 'bathy_t' in schema, 'Invalid locked schema')
            require(not set(schema) & FORBIDDEN, f'Leaked target/metadata predictors: {set(schema) & FORBIDDEN}')
            require(meta['predictors'] == schema, 'Metadata predictor order differs from locked schema')
            n = meta['n_boot']
            require(isinstance(n, int) and n > 0 and n == cfg['n_boot'], 'Invalid bootstrap count')
            if task['expected_bootstrap'] is not None:
                require(n == task['expected_bootstrap'], 'Unexpected bootstrap count')
            for role, frame, count_key in (('training', train, 'n_fit'), ('prediction', pred, 'n_prediction_rows')):
                if frame is not None:
                    require(len(frame) == meta[count_key], f'{role}: metadata row count differs')
                    require(CRS(frame.crs) == CRS(meta['crs']), f'{role}: CRS differs from metadata')
            rounds = meta['best_iteration']
            if train is not None:
                require(len(train) >= cfg['minimum_training_rows'], 'Training count below configured minimum')
            require(isinstance(rounds, int) and rounds > 0, 'Invalid selected round count')
            if cfg['round_selection_mode'] == 'fixed':
                require(rounds == cfg['fixed_nrounds'], 'Fixed rounds differ')
            if cfg['round_selection_mode'] == 'cv_with_floor':
                require(rounds >= cfg['minimum_nrounds'], 'Round floor violated')
        check('metadata/schema', metadata_check)

        def load_models():
            n = meta['n_boot']
            expected_paths = [str(UPath(task['model_dir']) / task['tile'] / 'Training' / 'models' / f"model_boot_{i:03d}_{task['pair']}.json") for i in range(1, n+1)]
            require(meta['model_uris'] == expected_paths, 'Model URIs differ from expected tile/pair paths')
            found = sorted(str(p) for p in UPath(expected_paths[0]).parent.glob(f"model_boot_*_{task['pair']}.json"))
            require(set(found) == set(expected_paths), 'Missing or extra saved bootstrap models')
            loaded = []
            for path in expected_paths:
                with UPath(path).open('rb') as stream:
                    model = xgb.Booster()
                    model.load_model(bytearray(stream.read()))
                model.set_param({'nthread': 1})
                require(model.feature_names == schema, f'Booster predictor order differs: {path}')
                require(model.num_boosted_rounds() == meta['best_iteration'], f'Booster rounds differ: {path}')
                loaded.append(model)
            return loaded
        models = check('saved models', load_models) or []
        if meta.get('xgboost_version') != xgb.__version__:
            issue('WARN', 'version', f"Saved XGBoost {meta.get('xgboost_version')}; validator {xgb.__version__}")
        if 'fallback' in meta.get('cv_status', ''):
            issue('WARN', 'CV', f"Round selection used fallback: {meta['cv_status']}")
        if meta.get('n_boot') == 1:
            issue('WARN', 'uncertainty', 'Single bootstrap: SD is undefined (NaN), as written by this engine')

    summaries = {}
    for role, source in (('training', train), ('prediction', pred)):
        if source is None or not expected:
            continue
        summary = check(f'{role} summary readable', lambda r=role: read_table(paths[r], True))
        if summary is None:
            continue
        summaries[role] = summary

        def summary_check():
            require(summary.crs is not None and summary.crs == source.crs, 'Summary CRS missing/changed')
            require(len(source) == len(summary), 'Dropped/added source rows')
            require(summary.FID.is_unique, 'Duplicate output FID')
            # Geometry is reconstructed from X/Y; sample_weight is recalculated by the engine.
            columns = [c for c in source if c not in {source.geometry.name, 'sample_weight', 'prediction_row_usable'}]
            pd.testing.assert_frame_equal(pd.DataFrame(summary[columns]).reset_index(drop=True), pd.DataFrame(source[columns]).reset_index(drop=True), check_dtype=False, check_exact=True)
            xy_ok = np.isfinite(summary[['X', 'Y']]).all(axis=1).to_numpy()
            require(np.array_equal(summary.geometry.isna(), ~xy_ok), 'Geometry null mask differs from valid coordinates')
            points = summary.loc[xy_ok].geometry
            require(points.geom_type.eq('Point').all(), 'Summary geometry must be Point')
            close(points.x, summary.loc[xy_ok, 'X'], 'Point X', 0, 0)
            close(points.y, summary.loc[xy_ok, 'Y'], 'Point Y', 0, 0)
            require(set(DERIVED).issubset(summary), 'Missing derived prediction fields')
        check(f'{role} row/geometry preservation', summary_check)

        def formula_replay():
            require(meta is not None and schema is not None and models, 'Replay blocked: missing metadata/schema/models')
            require(set(schema).issubset(source), 'Missing source predictors')
            usable = np.ones(len(source), dtype=bool)
            if role == 'prediction':
                usable = np.isfinite(source[['X', 'Y', 'bathy_t', 'interval_years']]).all(axis=1).to_numpy() & (source.interval_years.to_numpy() > 0)
                if 'prediction_row_usable' in source:
                    require(source.prediction_row_usable.isin([True, False]).all(), 'Invalid source usability flags')
                    usable &= source.prediction_row_usable.to_numpy(dtype=bool)
                require(np.array_equal(summary.prediction_row_usable, usable), 'Usable prediction mask changed')
                require(int(usable.sum()) == meta['n_prediction_usable'], 'Metadata usable count differs')
                result['usable_prediction_rows'] = int(usable.sum())
            require(not np.isinf(source.loc[usable, schema].to_numpy()).any(), 'Infinite usable predictors')
            require(summary.loc[~usable, DERIVED].isna().all().all(), 'Unusable rows contain model results')
            positions = np.flatnonzero(usable)
            for offset in range(0, len(positions), task['chunk_rows']):
                rows = positions[offset:offset+task['chunk_rows']]
                matrix = xgb.DMatrix(source.iloc[rows][schema].to_numpy(dtype=np.float32), feature_names=schema, missing=np.nan, nthread=1)
                values = np.stack([m.predict(matrix) for m in models])
                require(np.isfinite(values).all(), 'Saved models produce nonfinite predictions')
                mean = values.mean(axis=0, dtype=np.float32)
                sd = values.std(axis=0, ddof=1, dtype=np.float32) if len(models) > 1 else np.full(len(rows), np.nan, dtype=np.float32)
                got = summary.iloc[rows]
                same(got.mean_predicted_bathy_t1, mean, 'Saved model mean replay')
                # Float32 Welford accumulation loses precision for very small spreads.
                # Add a bound based on model magnitude, independently of stored SD.
                bound = task['atol'] + 16*np.finfo(np.float32).eps*np.max(np.abs(values), axis=0)
                require(np.all(np.isclose(got.uncertainty_sd_bathy_t1, sd, rtol=task['rtol'], atol=bound, equal_nan=True)), 'Saved ensemble sample SD replay mismatch')
            got = summary.iloc[positions]
            require(np.isfinite(got[DERIVED[:5]].to_numpy()).all(), 'Nonfinite usable prediction means')
            if len(models) > 1:
                require(np.isfinite(got[DERIVED[5:]]).all().all() and (got[DERIVED[5:]] >= 0).all().all(), 'Invalid ensemble SD')
            same(got.mean_predicted_change, got.mean_predicted_bathy_t1-got.bathy_t, 'Change sign')
            same(got.mean_predicted_rate, got.mean_predicted_change/got.interval_years, 'Annual rate')
            same(got.mean_predicted_standard_change, got.mean_predicted_rate*np.float32(cfg['standard_horizon_years']), 'Standard horizon change')
            same(got.mean_predicted_standard_bathy, got.bathy_t+got.mean_predicted_standard_change, 'Standard horizon bathy')
            same(got.uncertainty_sd_change, got.uncertainty_sd_bathy_t1, 'Change SD')
            same(got.uncertainty_sd_rate, got.uncertainty_sd_bathy_t1/got.interval_years, 'Rate SD')
            if role == 'training':
                change = np.abs(source.delta_bathy.to_numpy(dtype=np.float64))
                weights = np.ones(len(change), dtype=np.float64)
                if cfg['use_weighted_loss']:
                    if cfg['weight_method'] == 'power':
                        raw = (change+cfg['weight_epsilon'])**cfg['weight_alpha']
                        if np.isfinite(raw.sum()) and raw.sum() > 0:
                            weights = raw/raw.sum()*len(raw)
                    elif cfg['weight_method'] == 'capped_linear':
                        cap = np.quantile(change, cfg['weight_cap_quantile'], method='linear')
                        if cap > 0:
                            weights = 1+(cfg['max_weight']-1)*(np.minimum(change, cap)/cap)**cfg['weight_alpha']
                    else:
                        raise ValueError('Unknown weight method')
                same(summary.sample_weight, weights, 'Recalculated model weights')
                same(got.prediction_residual, got.bathy_t1-got.mean_predicted_bathy_t1, 'State residual')
                same(got.delta_residual, got.delta_bathy-got.mean_predicted_change, 'Change residual')
        check(f'{role} full model replay/formulas', formula_replay)

    def diagnostic_check():
        require(meta is not None and schema is not None, 'Diagnostics blocked: no metadata/schema')
        for name, numeric in (('importance', ['gain', 'total_gain', 'gain_fraction']), ('shap_summary', ['mean_abs_shap'])):
            table = read_table(paths[name])
            if name == 'shap_summary' and cfg['shap_sample_rows'] <= 0:
                require(table.empty, 'SHAP disabled but rows present')
                continue
            expected_keys = {(b, p) for b in range(1, meta['n_boot']+1) for p in schema}
            require(not table.duplicated(['bootstrap_iteration', 'predictor']).any(), f'{name}: duplicate feature/bootstrap')
            require(set(zip(table.bootstrap_iteration, table.predictor)) == expected_keys, f'{name}: missing/extra feature/bootstrap')
            require(np.isfinite(table[numeric]).all().all() and (table[numeric] >= 0).all().all(), f'{name}: invalid values')
            if name == 'importance':
                for _, group in table.groupby('bootstrap_iteration'):
                    total = group.total_gain.sum()
                    same(group.gain_fraction, group.total_gain/total if total > 0 else np.zeros(len(group)), 'Gain fractions')
                # Check saved gain against the actual Boosters too.
                for index, model in enumerate(models, 1):
                    group = table[table.bootstrap_iteration.eq(index)].set_index('predictor')
                    for kind in ('gain', 'total_gain'):
                        scores = model.get_score(importance_type=kind)
                        same(group.loc[schema, kind], [scores.get(p, 0) for p in schema], f'Booster {kind}')
    if expected:
        check('importance/SHAP diagnostics', diagnostic_check)

    def cv_check():
        require(meta is not None and train is not None, 'CV blocked: missing metadata/training')
        folds, metrics, oof = [read_table(paths['spatial_cv_'+name]) for name in ('folds', 'metrics', 'oof')]
        if cfg['round_selection_mode'] == 'fixed':
            require(folds.empty and metrics.empty and oof.empty and meta['cv_status'] == 'fixed', 'Fixed mode has unexpected CV records')
            return
        require(CRS(train.crs).is_projected and all(np.isclose(a.unit_conversion_factor, 1) for a in CRS(train.crs).axis_info[:2]), 'CV CRS must be projected metres')
        require(np.array_equal(folds.row_index, np.arange(len(train))), 'CV row index differs')
        require(np.array_equal(folds.FID, train.FID), 'CV FID differs')
        close(folds.X, train.X, 'CV X', 0, 0)
        close(folds.Y, train.Y, 'CV Y', 0, 0)
        require(np.array_equal(folds[['block_q', 'block_r']], expected_hex_blocks(train, cfg['block_size_m'])), 'Saved hexagons differ from source XY')
        require(folds.groupby(['block_q', 'block_r']).fold.nunique().max() == 1, 'Hexagonal block split across folds')
        require(not metrics.fold.duplicated().any(), 'Duplicate CV metric folds')
        require(set(metrics.status) <= {'ok', 'fit_failed', 'insufficient_rows'}, 'Unknown CV fold status')
        unique = set(folds.fold)
        if len(unique) >= 2:
            require(set(metrics.fold) == unique, 'Missing/extra CV fold diagnostics')
        successful = set(metrics.loc[metrics.status.eq('ok'), 'fold'])
        expected_rows = set(np.flatnonzero(folds.fold.isin(successful)))
        require(not oof.row_index.duplicated().any(), 'Duplicate OOF row')
        require(set(oof.row_index) == expected_rows, 'Missing/extra OOF rows for successful folds')
        for row in metrics.itertuples():
            test = folds.fold.eq(row.fold)
            require(row.n_test == int(test.sum()) and row.n_train == int((~test).sum()), 'CV fold counts differ')
            if row.status != 'ok':
                issue('WARN', 'CV fold', f'Fold {row.fold}: {row.status}; {getattr(row, "error", "")}')
        if not oof.empty:
            idx = oof.row_index.to_numpy(dtype=int)
            require(np.array_equal(oof.FID, train.FID.to_numpy()[idx]), 'OOF FID differs')
            require(np.array_equal(oof.fold, folds.fold.to_numpy()[idx]), 'OOF fold differs')
            same(oof.observed_bathy_t1, train.bathy_t1.to_numpy()[idx], 'OOF observations')
            require(np.isfinite(oof.predicted_bathy_t1).all(), 'Nonfinite OOF predictions')
            for fold, group in oof.groupby('fold'):
                row = metrics[metrics.fold.eq(fold)].iloc[0]
                error = group.predicted_bathy_t1.to_numpy()-group.observed_bathy_t1.to_numpy()
                same(row.rmse_bathy_t1, np.sqrt(np.mean(error**2)), 'OOF RMSE')
                same(row.mae_bathy_t1, np.mean(np.abs(error)), 'OOF MAE')
                require(group.best_nrounds.eq(row.best_nrounds).all(), 'OOF rounds differ')
            error = oof.predicted_bathy_t1.to_numpy()-oof.observed_bathy_t1.to_numpy()
            baseline_error = train.bathy_t.to_numpy()[idx]-oof.observed_bathy_t1.to_numpy()
            result['cv_rmse'] = float(np.sqrt(np.mean(error**2)))
            result['baseline_cv_rmse'] = float(np.sqrt(np.mean(baseline_error**2)))
            if result['cv_rmse'] >= result['baseline_cv_rmse']:
                issue('WARN', 'skill', 'Spatial CV RMSE does not improve on unchanged-bathymetry baseline')
        else:
            issue('WARN', 'skill', 'No held-out CV predictions; predictive skill not assessed')
        if successful:
            raw = int(round(float(metrics.loc[metrics.status.eq('ok'), 'best_nrounds'].mean())))
            selected = max(cfg['minimum_nrounds'], raw) if cfg['round_selection_mode'] == 'cv_with_floor' else max(50, raw)
            require(meta['cv_status'] == 'cv_success' and meta['best_iteration'] == selected, 'CV round selection differs')
        else:
            selected = max(cfg['minimum_nrounds'], cfg['cv_fallback_nrounds']) if cfg['round_selection_mode'] == 'cv_with_floor' else cfg['cv_fallback_nrounds']
            require(meta['best_iteration'] == selected and 'fallback' in meta['cv_status'], 'Fallback round selection differs')

    def apparent_metrics_check():
        require('training' in summaries, 'Training summary unavailable')
        saved = read_json(paths['apparent_training_metrics'])
        frame = summaries['training']
        for name, a, b in (('bathy_t1', frame.bathy_t1.to_numpy(), frame.mean_predicted_bathy_t1.to_numpy()),
                           ('change', (frame.bathy_t1-frame.bathy_t).to_numpy(), frame.mean_predicted_change.to_numpy()),
                           ('rate', ((frame.bathy_t1-frame.bathy_t)/frame.interval_years).to_numpy(), frame.mean_predicted_rate.to_numpy())):
            error = b-a
            require(saved['n_'+name] == len(a), f'{name}: metric row count differs')
            for metric, value in (('rmse', np.sqrt(np.mean(error**2))), ('mae', np.mean(np.abs(error))), ('bias', np.mean(error))):
                same(saved[f'{metric}_{name}'], value, f'Apparent {metric} {name}')
    if expected:
        check('spatial CV alignment/metrics', cv_check)
        check('apparent training metrics', apparent_metrics_check)
    if not expected:
        result['status'] = 'NOT_MODELED'
    if any(i['level'] == 'FAIL' for i in issues):
        result['status'] = 'FAIL'
    elif issues and expected:
        result['status'] = 'WARN'
    return result


def safe_validate(task):
    try:
        return validate_task(task)
    except Exception as exc:
        return {**{k: task[k] for k in ('region', 'tile', 'pair')}, 'status': 'FAIL', 'checks_passed': 0,
                'issues': [{'level': 'FAIL', 'check': 'worker', 'message': f'{type(exc).__name__}: {exc}\n{traceback.format_exc()}'}]}


def run_tasks(tasks, args, client=None):
    """Keep at most 2*workers tasks in flight and print failures as they finish."""
    completed = 0

    def record(result):
        nonlocal completed
        completed += 1
        print(f"{args.region}: completed {completed:,}/{len(tasks):,} ({result['tile']}/{result['pair']}) {result['status']}", flush=True)
        for issue in result['issues']:
            print(f"  [{issue['level']}] {issue['check']}: {issue['message']}", flush=True)
        return result

    if args.execution == 'serial' or args.workers == 1:
        return [record(safe_validate(t)) for t in tasks]
    results = []
    if client is not None:
        require(as_completed is not None, 'Dask execution requires dask[distributed]')
        for start in range(0, len(tasks), args.workers*2):
            futures = client.map(safe_validate, tasks[start:start+args.workers*2], pure=False)
            try:
                for future in as_completed(futures):
                    try:
                        results.append(record(future.result()))
                    except Exception as exc:
                        task = tasks[start+futures.index(future)]
                        results.append(record({**{k: task[k] for k in ('region', 'tile', 'pair')}, 'status': 'FAIL', 'issues': [{'level': 'FAIL', 'check': 'Dask worker', 'message': str(exc)}]}))
                    finally:
                        future.release()
            finally:
                client.cancel(futures)
        return results
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        iterator = iter(tasks)
        pending = {}
        for _ in range(min(len(tasks), args.workers*2)):
            task = next(iterator)
            pending[pool.submit(safe_validate, task)] = task
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in finished:
                task = pending.pop(future)
                try:
                    results.append(record(future.result()))
                except Exception as exc:
                    results.append(record({**{k: task[k] for k in ('region', 'tile', 'pair')}, 'status': 'FAIL', 'issues': [{'level': 'FAIL', 'check': 'process worker', 'message': str(exc)}]}))
                task = next(iterator, None)
                if task is not None:
                    pending[pool.submit(safe_validate, task)] = task
    return results


def write_reports(results, destination, context):
    """Persist full row-level findings and concise readable problems, locally or on S3."""
    destination.mkdir(parents=True, exist_ok=True)
    counts = Counter(r['status'] for r in results)
    lines = ['XGBoost modeling validation', context, f'Tile/year pairs: {len(results)} | ' + ' | '.join(f'{k}: {counts[k]}' for k in ('PASS', 'WARN', 'FAIL', 'NOT_MODELED')),
             'PASS checks saved artifacts; it does not prove generalization or calibrated uncertainty.',
             'Inventory covers prepared inputs and existing model artifacts, not tiles absent from both.',
             'Training metrics are apparent fit. Spatial CV also selected rounds.', '']
    if not results:
        lines.append('FAIL: No matching inputs or outputs discovered. Check paths/region/filters.')
    for result in sorted(results, key=lambda r: (r['status'] != 'FAIL', r['tile'], r['pair'])):
        lines.append(f"{result['status']} {result['region']}/{result['tile']}/{result['pair']} | checks passed {result.get('checks_passed', 0)} | rows train={result.get('training_rows', '?')}, prediction={result.get('prediction_rows', '?')}")
        for issue in result['issues']:
            lines.append(f"  {issue['level']} {issue['check']}: {issue['message']}")
        if result.get('cv_rmse') is not None:
            lines.append(f"  OOF RMSE {result['cv_rmse']:.5g}; unchanged-bathy baseline {result['baseline_cv_rmse']:.5g}")
    for filename, text in (('XGBoost_validation_report.txt', '\n'.join(lines)+'\n'),):
        with (destination / filename).open('w') as stream:
            stream.write(text)
    stream = io.StringIO()
    fields = ['region', 'tile', 'pair', 'status', 'checks_passed', 'training_rows', 'prediction_rows', 'usable_prediction_rows', 'cv_rmse', 'baseline_cv_rmse', 'issues']
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for result in sorted(results, key=lambda r: (r['tile'], r['pair'])):
        writer.writerow({**result, 'issues': ' | '.join(f"{i['level']} {i['check']}: {i['message']}" for i in result['issues'])})
    with (destination / 'XGBoost_validation_results.csv').open('w') as output:
        output.write(stream.getvalue())
    print('\n'.join(lines[:6]), flush=True)
    print(f'Reports saved: {destination}', flush=True)


def cross_tile_schemas(results):
    """Describe exact differences from the most common schema; do not assume it correct."""
    groups = defaultdict(list)
    for result in results:
        if result.get('predictors'):
            groups[(result['region'], result['pair'])].append(result)
    for group in groups.values():
        counts = Counter(tuple(r['predictors']) for r in group)
        if len(counts) < 2:
            continue
        reference = list(counts.most_common(1)[0][0])
        for result in group:
            actual = result['predictors']
            if actual == reference:
                continue
            shared = set(actual) & set(reference)
            order_differs = [p for p in actual if p in shared] != [p for p in reference if p in shared]
            message = (f"Schema differs from most common schema ({counts[tuple(reference)]}/{len(group)} tiles): "
                       f"missing={sorted(set(reference)-set(actual))}; extra={sorted(set(actual)-set(reference))}; "
                       f"shared order differs={order_differs}. Reference order={reference}; actual order={actual}. "
                       'Tile-specific schemas may be intentional; each model is also checked against its own locked schema.')
            result['issues'].append({'level': 'WARN', 'check': 'cross-tile schema', 'message': message})
            if result['status'] == 'PASS':
                result['status'] = 'WARN'
            print(f"[WARN] {result['region']}/{result['tile']}/{result['pair']}: {message}", flush=True)


def write_combined_pdf(results, path, contexts):
    """One readable report for every selected region, with concrete problems first."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    fonts = Path(reportlab.__file__).parent / 'fonts'
    pdfmetrics.registerFont(TTFont('AuditVera', str(fonts / 'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('AuditVeraBold', str(fonts / 'VeraBd.ttf')))
    for name in styles.byName:
        styles[name].fontName = 'AuditVeraBold' if name.startswith('Heading') or name == 'Title' else 'AuditVera'
    styles.add(ParagraphStyle(name='AuditText', parent=styles['BodyText'], fontSize=8, leading=11, spaceAfter=5, splitLongWords=True))
    styles.add(ParagraphStyle(name='AuditCell', parent=styles['AuditText'], fontSize=7, leading=9, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='AuditHeader', parent=styles['AuditCell'], textColor=colors.white))
    width = A4[0]-72
    story = []

    def p(text, style='AuditText'):
        return Paragraph(escape(str(text)).replace('\n', '<br/>'), styles[style])

    def table(headers, rows, fractions):
        if not rows:
            return
        data = [[p(h, 'AuditHeader') for h in headers]]
        data.extend([[p(c, 'AuditCell') for c in row] for row in rows])
        item = LongTable(data, colWidths=[width*f for f in fractions], repeatRows=1, hAlign='LEFT', splitInRow=1)
        item.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#193b50')),
                                 ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f6')]),
                                 ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .25, colors.HexColor('#ccd8df')),
                                 ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        story.extend([item, Spacer(1, 10)])

    counts = Counter(r['status'] for r in results)
    levels = Counter(i['level'] for r in results for i in r['issues'])
    discovery_failures = [c for c in contexts if c.startswith('FAIL ')]
    status = 'ERRORS FOUND' if counts['FAIL'] or discovery_failures or not results else 'WARNINGS FOUND' if levels['WARN'] else 'PASSED'
    story.extend([p('XGBoost Model Validation', 'Title'), p(f'{status} | {len(results):,} tile/year pairs | {levels["FAIL"]:,} errors | {levels["WARN"]:,} warnings')])
    story.append(p('Validation reads existing outputs only. Every usable source row is replayed through every saved bootstrap model.'))
    region_counts = defaultdict(Counter)
    for r in results:
        region_counts[r['region']][r['status']] += 1
    table(['Region', 'PASS', 'WARN', 'FAIL', 'Not modeled'], [[region, *[c[k] for k in ('PASS', 'WARN', 'FAIL', 'NOT_MODELED')]] for region, c in sorted(region_counts.items())], [.28, .18, .18, .18, .18])
    for message in discovery_failures:
        story.append(p(message))
    if not results:
        story.append(p('FAIL: No matching inputs or outputs discovered. Check paths, regions and filters.'))
    story.append(p('Problems and affected tiles', 'Heading2'))
    findings = [(r, i) for r in results for i in r['issues']]
    findings.sort(key=lambda item: (item[1]['level'] != 'FAIL', item[0]['region'], item[0]['tile'], item[0]['pair']))
    table(['Tile / pair', 'Problem'], [[f"{r['region']}/{r['tile']}\n{r['pair']}", f"{i['level']} - {i['check']}\n{i['message']}"] for r, i in findings], [.28, .72])
    if not findings:
        story.append(p('No issues found in the discovered artifacts.'))
    story.append(p('What the results establish', 'Heading2'))
    story.append(p('PASS means artifact consistency: expected tile/pair paths, input identity, rows, CRS, model features and rounds, replayed mean and sample SD, change/rate formulas, spatial CV records, and diagnostics. It does not prove which rows were used during fitting: the supplied engine does not save source fingerprints or bootstrap draws.'))
    story.append(p('Coverage starts from prepared inputs and model artifacts. Tiles absent from both are outside this inventory. Prediction-only pairs are listed as not modeled because the supplied engine schedules training files.'))
    story.append(p('Apparent training errors are fit diagnostics. Spatial CV also selects boosting rounds and is not an untouched final evaluation. Bootstrap SD measures ensemble spread and is not a calibrated prediction interval. A single bootstrap correctly has undefined (NaN) SD.'))
    story.append(p('Tile inventory and spatial CV skill', 'Heading2'))
    def fmt(value):
        return '-' if value is None else f'{value:.5g}'
    table(['Tile / pair', 'Status', 'Rows train / pred', 'OOF RMSE / baseline'],
          [[f"{r['region']}/{r['tile']}\n{r['pair']}", r['status'], f"{r.get('training_rows', '?')} / {r.get('prediction_rows', '?')}", f"{fmt(r.get('cv_rmse'))} / {fmt(r.get('baseline_cv_rmse'))}"] for r in sorted(results, key=lambda r: (r['region'], r['tile'], r['pair']))], [.34, .15, .22, .29])
    story.append(p('Configuration and paths', 'Heading2'))
    for context in contexts:
        story.append(p(context))

    def footer(canvas, document):
        canvas.setFont('AuditVera', 8)
        canvas.drawString(36, 20, 'XGBoost validation | existing artifacts only')
        canvas.drawRightString(A4[0]-36, 20, f'Page {document.page}')
    SimpleDocTemplate(str(path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36).build(story, onFirstPage=footer, onLaterPages=footer)
    return path


def run_validation(param_lookup: dict, output_prefix: str | bool = False):
    """Use the project runner's region object and environment without CLI parsing."""
    values = param_lookup['eco_regions'].value
    require(values is not None and not isinstance(values, (str, bytes)), 'eco_regions.value must be a nonempty iterable')
    regions = list(dict.fromkeys(values))
    require(regions and all(isinstance(r, str) and r.strip() for r in regions), 'Each eco_region must be a nonblank string')
    argv = ['--regions', *[r.strip() for r in regions], '--env', param_lookup['env']]
    if output_prefix:
        argv.extend(['--output-prefix', str(output_prefix)])
    if param_lookup.get('repository_root'):
        argv.extend(['--repo-root', str(param_lookup['repository_root'])])
    return main(argv)


def repository_root(explicit=None):
    if explicit:
        root = Path(explicit).resolve()
        require((root / 'src/hydro_health').is_dir(), 'repo-root must contain src/hydro_health')
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / 'src/hydro_health').is_dir():
            return parent
    raise FileNotFoundError('Install under the hydro_health repository or supply --repo-root')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--regions', nargs='+', required=True)
    parser.add_argument('--env', choices=['aws', 'remote', 'local'], default='aws')
    parser.add_argument('--output-prefix', default='')
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--execution', choices=['dask', 'processes', 'serial'], default='processes')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--memory-limit', default='6GB', help='Dask per-worker limit; process pool has no enforced per-worker limit')
    parser.add_argument('--chunk-rows', type=int, default=10000)
    parser.add_argument('--tiles', nargs='+')
    parser.add_argument('--pairs', nargs='+')
    parser.add_argument('--expected-bootstrap', type=int)
    parser.add_argument('--rtol', type=float, default=2e-5)
    parser.add_argument('--atol', type=float, default=2e-5)
    parser.add_argument('--strict-warnings', action='store_true')
    parser.add_argument('--report', type=Path, help='Local combined PDF path; default outputs/[prefix/]REPORTS.SUBFOLDER/XGBoost_Validation_<timestamp>.pdf')
    parser.add_argument('--local-only', action='store_true', help='Disable S3 report writes, including for local inputs')
    parser.add_argument('--report-s3-root', help='Optional s3://bucket/prefix root above REPORTS.SUBFOLDER')
    args = parser.parse_args(argv)
    require(args.workers > 0 and args.chunk_rows > 0, 'workers and chunk-rows must be positive')
    require(np.isfinite(args.rtol) and np.isfinite(args.atol) and args.rtol >= 0 and args.atol >= 0, 'Tolerances must be finite and nonnegative')
    if args.expected_bootstrap is not None:
        require(args.expected_bootstrap > 0, 'expected-bootstrap must be positive')
    root = repository_root(args.repo_root)
    engine = Engine()
    failed = False
    all_results, contexts = [], []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    reports_folder = str(get_config_item('REPORTS', 'SUBFOLDER')).strip()
    require(reports_folder and reports_folder != 'None' and not Path(reports_folder).is_absolute() and '://' not in reports_folder and '..' not in Path(reports_folder).parts, 'REPORTS.SUBFOLDER must be a relative folder')
    local_root = root / 'outputs' / args.output_prefix if args.output_prefix else root / 'outputs'
    report = (args.report or local_root / reports_folder / f'XGBoost_Validation_{stamp}.pdf').resolve()
    remote_report = None
    if not args.local_only:
        report_root = args.report_s3_root
        if not report_root:
            bucket = str(get_config_item('SHARED', 'OUTPUT_BUCKET')).removeprefix('s3://').strip('/')
            require(bucket and bucket != 'None', 'Missing SHARED.OUTPUT_BUCKET')
            report_root = f's3://{bucket}' + (f'/{args.output_prefix}' if args.output_prefix else '')
        require(report_root.startswith('s3://') and report_root[5:].strip('/'), 'report-s3-root must be s3://bucket/prefix')
        remote_report = UPath(report_root) / reports_folder / report.name

    try:
        if args.execution == 'dask':
            engine.setup_dask(args.env, n_workers=args.workers, threads_per_worker=1, memory_limit=args.memory_limit, processes=True)
        for region in args.regions:
            args.region = region
            base = UPath('s3://' + str(get_config_item('SHARED', 'OUTPUT_BUCKET')).removeprefix('s3://').strip('/')) if args.env in ('aws', 'remote') else UPath(root / 'outputs')
            if args.output_prefix:
                base = base / args.output_prefix
            base = base / region
            train_dir = base / get_config_item('MODEL', 'ML_PREP_TRAIN_DIR')
            pred_dir = base / get_config_item('MODEL', 'ML_PREP_PRED_DIR')
            model_dir = base / get_config_item('MODEL', 'XGB_OUTPUT_DIR')
            destination = UPath(local_root / region / reports_folder / 'xgboost_validation' / stamp)
            print(f'{region}: discovering inputs and model artifacts under {base}', flush=True)
            tasks = inventory(train_dir, pred_dir, model_dir, args)
            print(f'{region}: validating {len(tasks):,} tile/year pairs', flush=True)
            for task in tasks:
                task['target_crs'] = str(engine.target_crs) if getattr(engine, 'target_crs', None) is not None else None
            results = run_tasks(tasks, args, engine.client if args.execution == 'dask' else None)
            cross_tile_schemas(results)
            context = f'UTC {stamp}; env={args.env}; training={train_dir}; prediction={pred_dir}; models={model_dir}; rtol={args.rtol}, atol={args.atol}; replay=all usable rows'
            write_reports(results, destination, context)
            all_results.extend(results)
            contexts.append(context)
            if not results:
                contexts.append(f'FAIL {region}: no matching prepared inputs or model artifacts found')
            failed |= not results or any(r['status'] == 'FAIL' for r in results)
            if args.strict_warnings:
                failed |= any(i['level'] == 'WARN' for r in results for i in r['issues'])
    finally:
        if engine is not None:
            if getattr(engine, 'client', None) is not None:
                engine.client.close()
            if getattr(engine, 'cluster', None) is not None:
                engine.cluster.close()
    write_combined_pdf(all_results, report, contexts)
    print(f'Combined PDF: {report}', flush=True)
    if remote_report is not None:
        try:
            remote_report.parent.mkdir(parents=True, exist_ok=True)
            with report.open('rb') as source, remote_report.open('wb') as target:
                shutil.copyfileobj(source, target, length=1024*1024)
            require(remote_report.stat().st_size == report.stat().st_size, 'Uploaded PDF size differs')
            print(f'S3 PDF: {remote_report}', flush=True)
        except Exception as exc:
            print(f'Could not upload PDF: {exc}. Local report retained: {report}', file=sys.stderr)
            return 2
    return 1 if failed else 0


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        raise SystemExit(main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Fatal validation/reporting error: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(2)
