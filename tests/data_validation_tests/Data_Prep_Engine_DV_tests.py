#!/usr/bin/env python3
"""DataPrepEngine audit: one PDF saved locally and to configured S3 reports folder.

Install:
    pip install numpy pandas pyarrow geopandas universal-pathlib reportlab s3fs

Parallelism:
    --workers 16   # default; standard-library processes, one tile/pair per task
    --workers 1    # sequential debugging, same validation checks
    --workers 4    # reduce memory and network pressure on smaller machines
    Processes have no hard per-worker memory cap. Unlike Dask memory_limit,
    this option controls concurrency only. Each active task holds only the final
    training and prediction ML-ready tables.

Report saving:
    REPORTS.SUBFOLDER is read using get_config_item('REPORTS', 'SUBFOLDER').
    Default S3 destination: configured bucket / output prefix / REPORTS.SUBFOLDER
    / local PDF basename (one combined report covering the selected regions).
    If input --root is local, SHARED.OUTPUT_BUCKET supplies the S3 report root.
    --report-s3-root s3://bucket/prefix overrides the report bucket/prefix.
    --reports-subfolder overrides REPORTS.SUBFOLDER when testing without config.
    --local-only explicitly disables upload. S3 saving is enabled by default.
    Existing reports with the same destination name are replaced.

Run using hydro_health MODEL directory settings:
    python Data_Prep_Engine_DV_tests.py --regions ER_3
    python Data_Prep_Engine_DV_tests.py --output-prefix MY_RUN --regions ER_3
    Normal runs initialize the real project Engine via super().__init__().
    --standalone explicitly skips project Engine initialization for isolated
    tests; it still needs configuration or explicit directory overrides.

Project runner entry point (uses the existing parameter object directly):
    from Data_Prep_Engine_DV_tests import run_validation
    run_validation(param_lookup, output_prefix)
    Engine regions always come from param_lookup['eco_regions'].value.
    A new python process cannot see another process's param_lookup; direct CLI
    usage needs --regions unless the project runner calls the entry point.

Normal S3 input paths match DataPrepEngine:
    s3://SHARED.OUTPUT_BUCKET/[output_prefix/]region/MODEL.<directory>
    --env aws (default) and --env remote use S3; --env local uses
    repository-root/outputs/[output_prefix/]region/MODEL.<directory>.
    --root is an optional explicit override for the root above region folders;
    do not combine it with --output-prefix.

Explicit directories (relative to each region):
    python Data_Prep_Engine_DV_tests.py --root /path/to/outputs --regions ER_3 \
        --train-dir ML_training --pred-dir ML_prediction

Only ML_ready_training_*.geoparquet and ML_ready_prediction_*.geoparquet
are inventoried and validated. Raw batches and auxiliary diagnostic files
are never scanned or required. Predictors come from final file columns,
excluding metadata/targets, or an explicit --master-predictors specification.
Use --training-root and --prediction-root for flat folders ({region} supported).

Native-missing mode allows optional feature NAs. Empty bathy_t and zero usable
prediction rows are errors. Zero-valued or constant predictors are warnings,
not automatically missing. --nodata values are user-declared numeric sentinels.
Ready files are read in full; processing is parallel by tile/year pair.
The audit never runs DataPrepEngine or modifies source/output tables. Only the
report PDF is written locally and uploaded to S3, by the parent process.

Exit: 0 = completed, warnings permitted; 1 = validation errors; 2 = fatal error.
"""

import argparse
import copy
import hashlib
import json
import multiprocessing
import platform
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import CRS
from upath import UPath


DYNAMIC = ('hurr_strength_mean', 'tsm_mean')
TILE_CONSTANT_PREDICTORS = {'grain_size_layer', 'hurr_strength_mean', 'prim_sed_layer'}
META = {'X', 'Y', 'FID', 'tile_id', 'pair_id', 'source', 'dataset_role',
        'year_t', 'year_t1', 'interval_years', 'bathy_t1', 'delta_bathy',
        'delta_rate', 'survey_end_date', 'geometry', 'prediction_row_usable',
        'sample_weight'}
DATES = ('year_t', 'year_t1', 'interval_years')
TRAIN_FIELDS = {'X', 'Y', 'bathy_t', 'bathy_t1', 'delta_bathy', 'delta_rate',
                'interval_years', 'year_t', 'year_t1', 'tile_id', 'pair_id',
                'sample_weight'}
PRED_FIELDS = {'X', 'Y', 'bathy_t', 'interval_years', 'year_t', 'year_t1',
               'tile_id', 'pair_id', 'prediction_row_usable'}
READY = re.compile(r'ML_ready_(training|prediction)_(\d{4}_\d{4})\.geoparquet')
_WORKER_CONTEXT = None


def initialize_worker(args, polygons):
    """Initialize once per spawned worker; do not send full tables between processes."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (args, polygons)


def audit_group_task(paths, found, group, master):
    """Return only report data; all table reads and checks happen in this worker."""
    args, polygons = _WORKER_CONTEXT
    audit = Audit(args)
    # Print in the parent as each result arrives, keeping parallel output readable.
    audit.print_findings = False
    audit.polygons = polygons
    scope = f'{group["region"]}/{group["tile"]}/{group["pair"]}'
    try:
        elapsed(group['pair'])
        audit.group(
            {kind: UPath(path) if path is not None else None for kind, path in paths.items()},
            {kind: [UPath(path) for path in items] for kind, items in found.items()},
            group, master,
        )
    except Exception as exc:
        audit.add('ERROR', scope, 'Tile validation interrupted', f'{type(exc).__name__}: {exc}; remaining checks for this tile/year pair were not completed.')
    return {'group': group, 'issues': audit.issues, 'files': audit.files,
            'provenance': audit.provenance, 'schemas': dict(audit.schema_sets)}


def elapsed(pair):
    if not re.fullmatch(r'\d{4}_\d{4}', pair):
        raise ValueError(f'Invalid year-pair label: {pair}')
    start, end = map(int, pair.split('_'))
    if end <= start:
        raise ValueError(f'Year pair must increase: {pair}')
    return end - start


def tile_constant_allowed(name):
    return name in TILE_CONSTANT_PREDICTORS or name.startswith('hurr_strength_mean_')


def schema_difference(reference, actual):
    """Describe column membership and relative order against an explicit reference."""
    missing = [name for name in reference if name not in actual]
    extra = [name for name in actual if name not in reference]
    expected_order = [name for name in reference if name in actual]
    actual_order = [name for name in actual if name in reference]
    details = [f'Missing predictors: {missing or "none"}.', f'Extra predictors: {extra or "none"}.']
    if expected_order != actual_order:
        details.append('Order-only difference.' if not missing and not extra else 'Shared predictor order also differs.')
        details.extend([f'Expected shared order: {expected_order}.', f'Actual shared order: {actual_order}.'])
    else:
        details.append('Shared predictor order matches.')
    return ' '.join(details)


def numbers(series):
    return pd.to_numeric(series, errors='coerce').to_numpy(dtype=float, na_value=np.nan, copy=True)


def percentage(count, total):
    return 100.0 * count / total if total else None


def fmt_pct(value):
    return 'N/A' if value is None or not np.isfinite(value) else f'{value:.1f}%'


def read_table(path, geo=False):
    with UPath(str(path)).open('rb') as stream:
        schema = pq.read_schema(stream)
    spatial = b'geo' in (schema.metadata or {})
    if geo and not spatial:
        raise ValueError('Missing GeoParquet metadata')
    with UPath(str(path)).open('rb') as stream:
        frame = gpd.read_parquet(stream) if spatial else pd.read_parquet(stream)
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names')
    if not spatial and 'geometry' in frame:
        raise ValueError('Geometry column without GeoParquet CRS metadata')
    return frame


def plain(frame):
    return pd.DataFrame(frame.drop(columns=frame.geometry.name)) if isinstance(frame, gpd.GeoDataFrame) else frame.copy()


def sha256(path):
    digest = hashlib.sha256()
    with UPath(str(path)).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def weights_for(change, args):
    magnitude = np.abs(np.asarray(change, dtype=float))
    if not len(magnitude) or args.weight_method == 'unweighted':
        return np.ones(len(magnitude))
    if args.weight_method == 'power':
        logs = np.log(magnitude + 1e-6)
        values = np.exp(args.weight_alpha * (logs - logs.max()))
        return values / values.sum() * len(values)
    cap = np.quantile(magnitude, args.weight_cap_quantile)
    return np.ones(len(magnitude)) if cap <= 0 else 1 + (args.max_weight - 1) * np.power(np.minimum(magnitude, cap) / cap, args.weight_alpha)


def master_names(path, region):
    source = UPath(str(path).format(region=region))
    if source.suffix == '.json':
        names = json.loads(source.read_text())
    elif source.suffix in {'.parquet', '.geoparquet'}:
        names = read_table(source)['predictor'].tolist()
    else:
        names = [line.strip() for line in source.read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
    if not isinstance(names, list) or not names:
        raise ValueError('Master predictors must be a nonempty list')
    result, seen = ['bathy_t'], set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Invalid predictor name')
        name = name.strip()
        if name in seen:
            raise ValueError(f'Duplicate predictor: {name}')
        seen.add(name)
        for base in DYNAMIC:
            if name.startswith(base + '_'):
                elapsed(name[len(base) + 1:])
                name = base
                break
        if name in META:
            raise ValueError(f'Target/metadata predictor leakage: {name}')
        if name not in result:
            result.append(name)
    return result


class Audit:
    def __init__(self, args):
        self.args = args
        self.issues, self.files, self.groups, self.provenance = [], [], [], []
        self.notes, self.schema_sets = [], defaultdict(set)
        self.polygons = None
        self.path_resolver = None
        self.print_findings = True

    def add(self, level, scope, check, detail):
        self.issues.append({'level': level, 'scope': str(scope), 'check': check, 'detail': str(detail)})
        if self.print_findings:
            label = 'FAIL' if level == 'ERROR' else 'WARN' if level == 'WARNING' else level
            print(f'[{label}] {scope}\n  {check}: {detail}', flush=True)

    def vals(self, frame, name):
        values = numbers(frame[name]) if name in frame else np.full(len(frame), np.nan)
        for value in self.args.nodata:
            values[values == value] = np.nan
        return values

    def directory(self, root, region, kind):
        override = getattr(self.args, kind + '_root')
        if override:
            return UPath(str(override).format(region=region))
        option = {'training': 'train_dir', 'prediction': 'pred_dir'}[kind]
        name = getattr(self.args, option)
        return root / region / name if name else None

    def listing(self, directory, pattern, scope, required=False):
        if directory is None:
            self.add('WARNING', scope, 'Coverage unavailable', f'No directory configured for {pattern}.')
            return []
        try:
            if not directory.is_dir():
                raise FileNotFoundError(f'Directory not found: {directory}')
            return sorted(set(directory.rglob(pattern)), key=str)
        except Exception as exc:
            self.add('ERROR' if required or not isinstance(exc, FileNotFoundError) else 'WARNING', scope, 'Directory unavailable', f'{directory}: {exc}; associated checks were not completed.')
            return []

    def load(self, path, geo=False):
        """Verify source stability around reading; hash fixtures when requested."""
        before = sha256(path) if self.args.hash_inputs else None
        frame = read_table(path, geo=geo)
        if self.args.hash_inputs:
            after = sha256(path)
            self.provenance.append((str(path), before, after))
            if before != after:
                raise ValueError('File content changed while being read')
        return frame

    def identity(self, frame, tile, pair, path, required=False):
        for name, expected in [('tile_id', tile), ('pair_id', pair)]:
            if name not in frame:
                if required:
                    self.add('ERROR', path, 'Identity', f'Missing {name}.')
                continue
            mismatch = ~frame[name].astype(str).eq(expected)
            bad = frame[name].isna() | mismatch if required else frame[name].notna() & mismatch
            if bad.any():
                self.add('ERROR', path, 'Identity', f'{name}: {int(bad.sum()):,} null/mismatched values; expected {expected}.')

    def dates(self, frame, pair, path, allow_missing=False):
        duration = elapsed(pair)
        for name, expected in zip(DATES, [int(pair[:4]), int(pair[5:]), duration]):
            if name not in frame:
                if allow_missing:
                    continue
                self.add('ERROR', path, 'Year fields', f'Missing {name}.')
                continue
            values = self.vals(frame, name)
            supplied = ~(frame[name].isna() | frame[name].astype('string').str.strip().eq('').fillna(False)).to_numpy(dtype=bool)
            bad = ~np.isfinite(values) | ~np.isclose(values, expected, rtol=0, atol=1e-8)
            if allow_missing:
                bad &= supplied
            if bad.any():
                self.add('ERROR', path, 'Year fields', f'{name}: {int(bad.sum()):,} invalid/conflicting values; expected {expected}. Dates are not silently repaired.')

    def spatial(self, frame, path, tile, role, expected_crs=None):
        finite_xy = np.isfinite(self.vals(frame, 'X')) & np.isfinite(self.vals(frame, 'Y'))
        if not finite_xy.all():
            self.add('ERROR' if role == 'training' else 'WARNING', path, 'Coordinates', f'{int((~finite_xy).sum()):,} rows with invalid X/Y.')
        if finite_xy.any():
            coordinates = pd.DataFrame({'X': self.vals(frame, 'X')[finite_xy], 'Y': self.vals(frame, 'Y')[finite_xy]})
            count = int(coordinates.duplicated().sum())
            if count:
                self.add('WARNING', path, 'Duplicate locations', f'{count:,} repeated coordinate rows beyond the first occurrence.')
        if not isinstance(frame, gpd.GeoDataFrame):
            if 'geometry' in frame:
                self.add('ERROR', path, 'Geometry', 'Geometry exists without valid GeoParquet metadata.')
            return
        if frame.crs is None:
            self.add('ERROR', path, 'CRS', 'GeoParquet has no CRS.')
            return
        target = expected_crs or self.args.target_crs
        if target and CRS.from_user_input(frame.crs) != CRS.from_user_input(target):
            self.add('ERROR', path, 'CRS', f'Found {frame.crs}; expected {target}.')
        null = frame.geometry.isna().to_numpy()
        if not np.array_equal(null, ~finite_xy):
            self.add('ERROR', path, 'Geometry null pattern', 'Null geometries do not exactly match invalid coordinate rows.')
        good = ~null & ~frame.geometry.is_empty.to_numpy() & frame.geometry.geom_type.eq('Point').to_numpy() & frame.geometry.is_valid.to_numpy()
        if np.any(finite_xy & ~good):
            self.add('ERROR', path, 'Geometry type', 'Coordinate-valid rows contain invalid, empty, or non-Point geometry.')
        selected = frame.loc[finite_xy & good]
        if len(selected):
            matches = np.isclose(selected.geometry.x, self.vals(selected, 'X'), rtol=0, atol=self.args.geometry_tolerance) & np.isclose(selected.geometry.y, self.vals(selected, 'Y'), rtol=0, atol=self.args.geometry_tolerance)
            if not matches.all():
                self.add('ERROR', path, 'Geometry coordinates', f'{int((~matches).sum()):,} geometries disagree with X/Y.')
            if self.polygons is not None:
                try:
                    polygons = self.polygons[self.polygons[self.args.polygon_tile_column].astype(str).eq(tile)].to_crs(frame.crs)
                    if not len(polygons):
                        self.add('WARNING', path, 'Tile footprint', f'No polygon for {tile}.')
                    elif polygons.geometry.isna().any() or polygons.geometry.is_empty.any() or not polygons.geometry.is_valid.all() or not polygons.geometry.geom_type.isin(['Polygon', 'MultiPolygon']).all():
                        self.add('ERROR', path, 'Tile footprint', 'Official tile geometry must be a valid Polygon/MultiPolygon.')
                    else:
                        footprint = polygons.geometry.union_all() if hasattr(polygons.geometry, 'union_all') else polygons.geometry.unary_union
                        outside = ~selected.geometry.covered_by(footprint)
                        if outside.any():
                            self.add('WARNING', path, 'Tile footprint', f'{int(outside.sum()):,} points outside the official tile; check grid/buffering rules.')
                except Exception as exc:
                    self.add('ERROR', path, 'Tile footprint', exc)

    def ready(self, path, region, tile, pair, role, predictors):
        row = {'path': str(path), 'region': region, 'tile': tile, 'pair': pair, 'role': role, 'rows': None, 'usable': None, 'missing_pct': None, 'rows_missing_pct': None, 'unusable_pct': None, 'stats': {}, 'empty': [], 'readable': False}
        self.files.append(row)
        try:
            frame = self.load(path, geo=True)
        except Exception as exc:
            self.add('ERROR', path, 'File read', exc)
            return row, None
        row['actual_predictors'] = [name for name in frame.columns if name not in META]
        if predictors is None:
            predictors = ['bathy_t'] + [name for name in row['actual_predictors'] if name != 'bathy_t']
        row['predictors'] = list(predictors)
        for name in predictors:
            for base in DYNAMIC:
                if name == base or (name.startswith(base + '_') and name != f'{base}_{pair}'):
                    self.add('ERROR', path, 'Predictor schema', f'Wrong dynamic predictor suffix: {name}.')
        row.update(rows=len(frame), readable=True)
        fields = TRAIN_FIELDS if role == 'training' else PRED_FIELDS
        absent = (fields | set(predictors)) - set(frame)
        if absent:
            self.add('ERROR', path, 'Required columns', f'Missing columns: {sorted(absent)}.')
        if len(frame) == 0:
            self.add('ERROR', path, 'Empty file', 'File has zero rows.')
            return row, frame
        self.identity(frame, tile, pair, path, required=True)
        self.dates(frame, pair, path)
        self.spatial(frame, path, tile, role)
        numeric_fields = set(predictors) | {'X', 'Y', *DATES, 'bathy_t'}
        if role == 'training':
            numeric_fields |= {'bathy_t1', 'delta_bathy', 'delta_rate', 'sample_weight'}
        for name in sorted(numeric_fields & set(frame)):
            values = numbers(frame[name])
            bad_text = frame[name].notna().to_numpy() & np.isnan(values)
            if bad_text.any() or np.isinf(values).any():
                self.add('ERROR', path, 'Numeric values', f'{name}: {int(bad_text.sum()):,} nonnumeric values; {int(np.isinf(values).sum()):,} infinite values.')
            if frame[name].dtype == np.dtype('float64'):
                row.setdefault('float64_fields', []).append(name)
        if row.get('float64_fields') and self.args.warn_float64:
            self.add('WARNING', path, 'Storage dtype', f"float64 fields: {row['float64_fields']}. Precision is not automatically changed; consider float32 where appropriate.")
        gaps = np.zeros(len(frame), dtype=bool)
        missing_cells = 0
        for name in predictors:
            values = self.vals(frame, name)
            finite = np.isfinite(values)
            missing = int((~finite).sum())
            missing_cells += missing
            gaps |= ~finite
            row['stats'][name] = {'missing': missing, 'finite': int(finite.sum()), 'min': float(values[finite].min()) if finite.any() else None, 'max': float(values[finite].max()) if finite.any() else None}
            if not finite.any():
                row['empty'].append(name)
                self.add('ERROR' if name == 'bathy_t' else 'WARNING', path, 'Empty predictor', f'{name}: no finite data in {len(frame):,} rows.')
            elif missing and missing / len(frame) >= self.args.missing_threshold:
                self.add('WARNING', path, 'Predictor missingness', f'{name}: {missing:,}/{len(frame):,} missing ({100 * missing / len(frame):.1f}%).')
            valid = values[finite]
            if valid.size and np.all(valid == valid[0]) and not tile_constant_allowed(name):
                self.add('WARNING', path, 'Constant predictor', f'{name}: all finite values equal {valid[0]:.6g}; may be valid for this tile.')
        # bathy_t must still be evaluated when the feature schema cannot be recovered.
        if not predictors:
            self.add('ERROR', path, 'Unknown feature schema', 'Cannot calculate predictor completeness without final-file/master predictors.')
            if not np.isfinite(self.vals(frame, 'bathy_t')).any():
                row['empty'].append('bathy_t')
                self.add('ERROR', path, 'Empty bathymetry', 'bathy_t contains no finite values.')
        else:
            row['missing_pct'] = percentage(missing_cells, len(frame) * len(predictors))
            row['rows_missing_pct'] = percentage(gaps.sum(), len(frame))
            if len(row['empty']) == len(predictors):
                self.add('ERROR', path, 'No predictor data', 'Every expected predictor is entirely missing.')
        essential = ['X', 'Y', 'bathy_t'] + (['bathy_t1', 'delta_bathy', 'delta_rate', 'interval_years'] if role == 'training' else [])
        usable = np.ones(len(frame), dtype=bool)
        for name in essential + (predictors if self.args.mode == 'legacy_complete' else []):
            usable &= np.isfinite(self.vals(frame, name))
        if role == 'training':
            usable &= self.vals(frame, 'interval_years') > 0
        row.update(usable=int(usable.sum()), unusable_pct=percentage((~usable).sum(), len(frame)))
        if not usable.all():
            self.add('ERROR' if role == 'training' or not usable.any() else 'WARNING', path, 'Unusable rows', f'{int((~usable).sum()):,}/{len(frame):,} unusable ({row["unusable_pct"]:.1f}%); {row["usable"]:,} usable remain under {self.args.mode}.')
        if role == 'training':
            if len(frame) < self.args.minimum_training_rows:
                self.add('ERROR', path, 'Training size', f'{len(frame):,} rows; minimum {self.args.minimum_training_rows:,}.')
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                expected_change = self.vals(frame, 'bathy_t1') - self.vals(frame, 'bathy_t')
                expected_rate = expected_change / elapsed(pair)
            for name, expected in [('delta_bathy', expected_change), ('delta_rate', expected_rate)]:
                actual = self.vals(frame, name)
                valid = np.isfinite(actual) & np.isfinite(expected) & np.isclose(actual, expected, rtol=self.args.rtol, atol=self.args.atol)
                if not valid.all():
                    self.add('ERROR', path, 'Target formula', f'{name}: {int((~valid).sum()):,} invalid/mismatched values.')
            self.check_weights(frame, path)
            if self.args.max_abs_delta_rate is not None:
                extremes = np.abs(self.vals(frame, 'delta_rate')) > self.args.max_abs_delta_rate
                if extremes.any():
                    self.add('WARNING', path, 'Extreme change', f'{int(extremes.sum()):,} rows exceed |delta_rate|={self.args.max_abs_delta_rate:g}.')
        elif 'prediction_row_usable' in frame:
            flags = frame['prediction_row_usable']
            if not pd.api.types.is_bool_dtype(flags.dtype) or flags.isna().any():
                self.add('ERROR', path, 'Usability flag', 'prediction_row_usable must be boolean without nulls.')
            elif not np.array_equal(flags.to_numpy(dtype=bool), usable):
                self.add('ERROR', path, 'Usability flag', f'{int((flags.to_numpy(dtype=bool) != usable).sum()):,} flags disagree with recomputed usability.')
        return row, frame

    def check_weights(self, frame, path):
        values = self.vals(frame, 'sample_weight')
        if not np.isfinite(values).all() or (values < 0).any() or not np.isfinite(values.sum()) or values.sum() <= 0:
            self.add('ERROR', path, 'Sample weights', 'Weights must be finite, nonnegative, and have a finite positive total.')
            return
        changes = self.vals(frame, 'delta_bathy')
        if np.isfinite(changes).all() and not np.allclose(values, weights_for(changes, self.args), rtol=self.args.rtol, atol=self.args.atol):
            self.add('ERROR', path, 'Weight formula', 'Weights differ from the configured preparation formula; verify weighting settings.')
        # Scaling first prevents unnecessary squaring overflow.
        scaled = values / values.max()
        effective = scaled.sum() ** 2 / np.square(scaled).sum()
        if effective < self.args.weight_ess_fraction * len(values):
            self.add('WARNING', path, 'Weight concentration', f'Effective weighted sample size {effective:,.1f}/{len(values):,}.')

    def paired(self, training, prediction, predictors, scope):
        if training is None or prediction is None or not len(training) or not len(prediction):
            return
        if training.crs is not None and prediction.crs is not None and CRS.from_user_input(training.crs) != CRS.from_user_input(prediction.crs):
            self.add('ERROR', scope, 'Paired CRS', 'Training and prediction CRS differ.')
        for name in predictors:
            train, pred = self.vals(training, name), self.vals(prediction, name)
            tf, pf = np.isfinite(train), np.isfinite(pred)
            if tf.any() != pf.any():
                self.add('WARNING', scope, 'Feature availability shift', f'{name}: finite data present in only one of training/prediction.')
            gap = float((~pf).mean() - (~tf).mean())
            if gap > 0 and gap >= self.args.missingness_gap:
                self.add('WARNING', scope, 'Missingness shift', f'{name}: prediction missingness {100 * gap:.1f} percentage points higher than training.')
            if tf.any() and pf.any():
                low, high = train[tf].min(), train[tf].max()
                outside = (pred[pf] < low - self.args.atol) | (pred[pf] > high + self.args.atol)
                if outside.any() and outside.mean() >= self.args.range_threshold:
                    self.add('WARNING', scope, 'Prediction outside training range', f'{name}: {100 * outside.mean():.1f}% of finite prediction values outside [{low:.6g}, {high:.6g}]. This compares this tile, not a pooled model training range.')

    def region(self, root, region):
        resolved = self.path_resolver(region) if self.path_resolver else {kind: self.directory(root, region, kind) for kind in ['training', 'prediction']}
        paths = {kind: resolved[kind] for kind in ['training', 'prediction']}
        inventory = {kind: defaultdict(list) for kind in paths}
        keys = set()
        for role, directory in paths.items():
            pattern = f'ML_ready_{role}_*.geoparquet'
            for path in self.listing(directory, pattern, f'{region}/{role}', required=role == 'prediction' and self.args.require_prediction):
                match = READY.fullmatch(path.name)
                if not match or match[1] != role:
                    self.add('ERROR', path, 'Filename', 'Unexpected file name/role.')
                    continue
                key = (path.parent.name, match[2])
                inventory[role][key].append(path)
                keys.add(key)
        master = None
        if self.args.master_predictors:
            try:
                master = master_names(self.args.master_predictors, region)
            except Exception as exc:
                self.add('ERROR', region, 'Master predictors', exc)
        if not keys:
            self.add('ERROR', region, 'No data', 'No final ML-ready files found.')
        tasks = []
        for tile, pair in sorted(keys):
            scope = f'{region}/{tile}/{pair}'
            group = {'region': region, 'tile': tile, 'pair': pair, 'complete': False}
            found = {kind: inventory[kind].get((tile, pair), []) for kind in inventory}
            group.update({kind: len(items) for kind, items in found.items()})
            self.groups.append(group)
            tasks.append((found, group))
        workers = min(self.args.workers, len(tasks))
        if workers <= 1:
            for found, group in tasks:
                scope = f'{group["region"]}/{group["tile"]}/{group["pair"]}'
                print(f'Checking {scope}', flush=True)
                try:
                    elapsed(group['pair'])
                    self.group(paths, found, group, master)
                except Exception as exc:
                    self.add('ERROR', scope, 'Tile validation interrupted', f'{type(exc).__name__}: {exc}; remaining checks for this tile/year pair were not completed.')
            return
        print(f'{region}: checking {len(tasks):,} tile/year pairs with {workers} processes.', flush=True)
        encoded_paths = {kind: str(path) if path is not None else None for kind, path in paths.items()}
        # Spawn is Windows-safe and avoids inheriting network/fsspec event loops.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=initialize_worker, initargs=(self.args, self.polygons)) as pool:
            pending = {}
            for found, group in tasks:
                encoded_found = {kind: [str(path) for path in items] for kind, items in found.items()}
                future = pool.submit(audit_group_task, encoded_paths, encoded_found, group, master)
                pending[future] = group
            for completed, future in enumerate(as_completed(pending), 1):
                group = pending[future]
                scope = f'{group["region"]}/{group["tile"]}/{group["pair"]}'
                try:
                    result = future.result()
                    group.update(result['group'])
                    for finding in result['issues']:
                        self.add(finding['level'], finding['scope'], finding['check'], finding['detail'])
                    self.files.extend(result['files'])
                    self.provenance.extend(result['provenance'])
                    for identity, schemas in result['schemas'].items():
                        self.schema_sets[identity].update(schemas)
                except Exception as exc:
                    self.add('ERROR', scope, 'Worker failed', f'{type(exc).__name__}: {exc}; this tile/year pair was not completely checked.')
                print(f'{region}: completed {completed:,}/{len(tasks):,} ({scope}).', flush=True)

    def group(self, paths, found, group, master):
        """Validate only discovered final ML-ready files, independently by role."""
        region, tile, pair = group['region'], group['tile'], group['pair']
        scope = f'{region}/{tile}/{pair}'
        tables, schemas = {}, {}
        expected = [f'{name}_{pair}' if name in DYNAMIC else name for name in master] if master else None
        for role in ['training', 'prediction']:
            if len(found[role]) > 1:
                self.add('ERROR', scope, 'Duplicate identity', f'{role}: multiple files {[str(path) for path in found[role]]}.')
            for index, path in enumerate(found[role]):
                record, frame = self.ready(path, region, tile, pair, role, expected)
                if frame is not None:
                    predictors = record['actual_predictors']
                    schemas[role] = predictors
                    self.schema_sets[(region, pair)].add(tuple(predictors))
                    if index == 0:
                        tables[role] = frame
        if found['training'] and not found['prediction']:
            self.add('ERROR' if self.args.require_prediction else 'WARNING', scope, 'Unmatched training', 'Training-ready file has no prediction-ready counterpart.')
        if 'training' in schemas and 'prediction' in schemas and schemas['training'] != schemas['prediction']:
            self.add('ERROR', scope, 'Paired predictor schema', 'Prediction compared with training. ' + schema_difference(schemas['training'], schemas['prediction']))
        self.paired(tables.get('training'), tables.get('prediction'), schemas.get('training', schemas.get('prediction', [])), scope)
        group['complete'] = all(len(found[role]) <= 1 for role in found) and all(role in tables for role in found if found[role])
        if self.args.require_prediction and found['training'] and not found['prediction']:
            group['complete'] = False

    def cross_tile_schemas(self):
        """Compare actual final-file columns; the most common schema is descriptive, not authoritative."""
        groups = defaultdict(lambda: defaultdict(list))
        for row in self.files:
            if row['readable'] and 'actual_predictors' in row:
                groups[(row['region'], row['pair'])][tuple(row['actual_predictors'])].append(row)
        for (region, pair), variants in sorted(groups.items()):
            if len(variants) < 2:
                continue
            ordered = sorted(variants, key=lambda names: (-len(variants[names]), names))
            reference = ordered[0]
            baseline = sorted(variants[reference], key=lambda row: row['path'])
            baseline_roles = dict(Counter(row['role'] for row in baseline))
            for index, actual in enumerate(ordered[1:], 2):
                affected = sorted(variants[actual], key=lambda row: row['path'])
                roles = dict(Counter(row['role'] for row in affected))
                examples = [row['path'] for row in affected[:5]]
                detail = (
                    f'{len(variants)} distinct actual predictor schemas (training and prediction combined). '
                    f'Variant {index}: {len(affected):,} files by role {roles}. '
                    f'Compared with most common schema: {len(baseline):,} files by role {baseline_roles}; '
                    f'reference example: {baseline[0]["path"]}. '
                    'The most common schema is a comparison baseline, not a declaration of correctness. '
                    + schema_difference(reference, actual)
                    + f' Affected file examples (up to 5): {examples}.'
                )
                self.add('WARNING', f'{region}/{pair}', 'Cross-tile schemas', detail)

    def cross_tile_constants(self):
        """Check the three tile-level predictors using merged worker summaries, without rereading files."""
        groups = defaultdict(list)
        for row in self.files:
            groups[(row['region'], row['pair'], row['role'])].append(row)
        for (region, pair, role), rows in sorted(groups.items()):
            tiles = {row['tile'] for row in rows}
            if len(tiles) < 2:
                continue
            names = {name for row in rows for name in row['stats'] if tile_constant_allowed(name)}
            for name in sorted(names):
                stats = [row['stats'].get(name) for row in rows]
                # Missing/unreadable/empty data have their own findings; they cannot
                # establish a shared finite value across every discovered tile.
                if not all(row['readable'] and stat and stat['finite'] > 0 for row, stat in zip(rows, stats)):
                    continue
                low = min(stat['min'] for stat in stats)
                high = max(stat['max'] for stat in stats)
                if low == high:
                    self.add('WARNING', f'{region}/{pair}/{role}', 'Constant predictor across tiles', f'{name}: all finite values equal {low:.6g} across all {len(tiles):,} discovered tiles; no variation between tiles.')

    def run(self):
        root = UPath(self.args.root)
        if self.args.tile_polygons:
            try:
                path = UPath(self.args.tile_polygons)
                self.polygons = read_table(path, geo=True) if path.suffix in {'.parquet', '.geoparquet'} else gpd.read_file(str(path))
                if self.polygons.crs is None or self.args.polygon_tile_column not in self.polygons:
                    raise ValueError('Polygons require CRS and configured tile ID column')
            except Exception as exc:
                self.polygons = None
                self.add('ERROR', self.args.tile_polygons, 'Official polygons unavailable', exc)
        regions = self.args.regions
        if not regions:
            try:
                regions = sorted(path.name for path in root.iterdir() if path.name.startswith('ER_') and path.is_dir())
            except Exception as exc:
                self.add('ERROR', root, 'Region discovery', exc)
                regions = []
        if not regions:
            self.add('ERROR', root, 'No regions', 'No regions selected or discovered.')
        for region in dict.fromkeys(regions):
            try:
                self.region(root, region)
            except Exception as exc:
                self.add('ERROR', region, 'Region validation interrupted', f'{type(exc).__name__}: {exc}; remaining checks were not completed.')
        self.cross_tile_constants()
        self.cross_tile_schemas()
        if not self.files:
            self.add('ERROR', root, 'No ready files', 'No final ML-ready files were found in the selected directories.')
        return self


def metrics(groups, files):
    trains = [row for row in files if row['role'] == 'training']
    preds = [row for row in files if row['role'] == 'prediction']
    matched = sum(bool(row['training'] and row['prediction']) for row in groups)
    training_keys = sum(bool(row['training']) for row in groups)
    return [
        ('Training-ready files found', sum(row['training'] for row in groups)),
        ('Prediction-ready files found', sum(row['prediction'] for row in groups)),
        ('Ready files read successfully', sum(row['readable'] for row in files)),
        ('Unique training tiles (all year pairs)', len({(row['region'], row['tile']) for row in groups if row['training']})),
        ('Unique prediction tiles (all year pairs)', len({(row['region'], row['tile']) for row in groups if row['prediction']})),
        ('Matched training/prediction tile-year pairs', f'{matched:,}/{training_keys:,} ({fmt_pct(percentage(matched, training_keys))})'),
        ('Training tile-year pairs without prediction', sum(bool(row['training'] and not row['prediction']) for row in groups)),
        ('Prediction tile-year pairs without training', sum(bool(row['prediction'] and not row['training']) for row in groups)),
        ('Readable final-output groups', sum(row['complete'] for row in groups)),
        ('Rows in readable training files', sum(row['rows'] or 0 for row in trains)),
        ('Rows in readable prediction files', sum(row['rows'] or 0 for row in preds)),
        ('Usable prediction rows', sum(row['usable'] or 0 for row in preds)),
        ('Unreadable or zero-row ready files', sum(not row['readable'] or row['rows'] == 0 for row in files)),
        ('Readable files with entirely missing bathy_t', sum('bathy_t' in row['empty'] for row in files)),
    ]


def write_pdf(audit):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    args = audit.args
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='AuditBody', parent=styles['BodyText'], fontSize=8, leading=11, spaceAfter=6, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='AuditCell', parent=styles['BodyText'], fontSize=7, leading=9, wordWrap='CJK'))
    styles.add(ParagraphStyle(name='AuditHead', parent=styles['AuditCell'], textColor=colors.white))
    story, width = [], A4[0] - 72

    def p(value, style='AuditBody'):
        return Paragraph(escape(str(value)).replace('\n', '<br/>'), styles[style])

    def heading(value):
        story.append(p(value, 'Heading2'))

    def table(headers, rows, fractions):
        if not rows:
            return
        data = [[p(cell, 'AuditHead') for cell in headers]] + [[p(cell, 'AuditCell') for cell in row] for row in rows]
        output = Table(data, colWidths=[width * fraction for fraction in fractions], repeatRows=1, hAlign='LEFT')
        output.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#193b50')), ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f6')]), ('GRID', (0, 0), (-1, -1), .25, colors.HexColor('#ccd8df')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5), ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
        story.extend([output, Spacer(1, 8)])

    counts = Counter(item['level'] for item in audit.issues)
    story.append(p('ML Ready Data Validation', 'Title'))
    story.append(p(f"{'ERRORS FOUND' if counts['ERROR'] else 'WARNINGS FOUND' if counts['WARNING'] else 'PASSED'} | {counts['ERROR']:,} errors | {counts['WARNING']:,} warnings"))
    story.append(p(f'Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. Root: {args.root}'))
    table(['Inventory', 'Result'], metrics(audit.groups, audit.files), [.74, .26])
    heading('Settings and validation coverage')
    versions = []
    for package in ['numpy', 'pandas', 'geopandas', 'pyarrow', 'universal-pathlib', 'reportlab']:
        try:
            versions.append(f'{package}={version(package)}')
        except PackageNotFoundError:
            versions.append(f'{package}=unknown')
    notes = [
        f'Python {platform.python_version()}; ' + '; '.join(versions),
        'Only final ML_ready_training_*.geoparquet and ML_ready_prediction_*.geoparquet files are discovered and read in full. Raw batches and auxiliary diagnostics are not inspected or required.',
        'Missingness, empty columns, usable rows, dates, targets, weights, geometry, CRS, and paired training/prediction checks use the final tables directly. Prediction-only tiles are included.',
        'Grain size, hurricane strength, and primary sediment may be constant within a tile; other predictors retain per-tile constant warnings. A cross-tile constant warning requires the same finite value in every discovered tile within a region/year pair/role, at least two distinct tiles, and finite data in every file.',
        'Schema warnings compare actual predictor columns, including their stored order, across training and prediction. Each variant lists missing/extra names, shared-order differences, file counts by role, and up to five affected file examples against the most common schema. That baseline is not assumed correct.',
        'Counts describe files currently found. Tiles with no final files are outside this audit; upstream skip reasons and removed-row counts are not inferred.',
        'Predictors are final-file columns excluding targets/metadata; an explicit --master-predictors list also detects missing expected columns. Without that list, a predictor absent from both final files cannot be identified.',
        f'Mode={args.mode}; minimum training rows={args.minimum_training_rows}; missingness warning={args.missing_threshold:.0%}; workers={args.workers}.',
        f'Weight method={args.weight_method}; alpha={args.weight_alpha}; cap quantile={args.weight_cap_quantile}; maximum={args.max_weight}. Match these to the preparation settings.',
        f'Target CRS={args.target_crs or "not supplied; presence and paired CRS checked"}; NoData sentinels={args.nodata}; atol={args.atol}; rtol={args.rtol}.',
        f'SHA-256 checks around final-file reads: {args.hash_inputs}.',
        'Tile polygon containment: ' + (str(args.tile_polygons) if args.tile_polygons else 'not requested.'),
        'Only the report is written. Final input tables are never modified.',
        'Report saving: local only (--local-only).' if args.local_only else f'Report S3 destination: {s3_report_destination(args, Path(args.report))}.',
    ] + audit.notes
    for text in notes:
        story.append(p(text))
    heading('Inventory by region and year pair')
    rows = []
    for region, pair in sorted({(row['region'], row['pair']) for row in audit.groups}):
        groups = [row for row in audit.groups if (row['region'], row['pair']) == (region, pair)]
        rows.append([region, pair, sum(row['training'] for row in groups), sum(row['prediction'] for row in groups), sum(bool(row['training'] and row['prediction']) for row in groups), sum(row['complete'] for row in groups)])
    table(['Region', 'Pair', 'Train files', 'Pred files', 'Matched', 'Complete'], rows, [.17, .19, .16, .16, .16, .16])
    heading('Significant issue summary')
    summary = Counter((row['level'], row['check']) for row in audit.issues)
    table(['Severity', 'Check', 'Findings'], [[level, check, count] for (level, check), count in sorted(summary.items(), key=lambda item: (item[0][0] != 'ERROR', -item[1], item[0][1]))], [.16, .68, .16])
    if not audit.issues:
        story.append(p('No significant issues detected under the configured checks.'))
    story.append(PageBreak())
    heading('Files ranked by missing predictor data')
    story.append(p('Missing cells = missing expected predictor cells / (rows x expected predictors). Rows with gaps = rows missing any predictor. Absent expected columns count as missing. Only affected files are listed, highest missingness first. Empty/unreadable files are listed separately.'))
    for role in ['training', 'prediction']:
        selected = sorted([row for row in audit.files if row['role'] == role and row['missing_pct'] is not None and row['missing_pct'] > 0], key=lambda row: (-row['missing_pct'], row['path']))
        if selected:
            heading(role.capitalize())
            table(['Region / tile / pair', 'Rows', 'Missing cells', 'Rows with gaps', 'Unusable'], [[f"{row['region']} / {row['tile']} / {row['pair']}", f"{row['rows']:,}", fmt_pct(row['missing_pct']), fmt_pct(row['rows_missing_pct']), fmt_pct(row['unusable_pct'])] for row in selected], [.40, .12, .16, .16, .16])
    for key, title in [('unusable_pct', 'Prediction files ranked by unusable rows')]:
        selected = sorted([row for row in audit.files if row.get(key) is not None and row[key] > 0 and (key != 'unusable_pct' or row['role'] == 'prediction')], key=lambda row: (-row[key], row['path']))
        if selected:
            heading(title)
            table(['Region / tile / pair', 'Rows', 'Affected'], [[f"{row['region']} / {row['tile']} / {row['pair']}", row['rows'], fmt_pct(row[key])] for row in selected], [.65, .16, .19])
    broken = [row for row in audit.files if not row['readable'] or row['rows'] == 0]
    if broken:
        heading('Empty or unreadable ready files')
        for row in broken:
            story.append(p(f"{row['path']}: {'unreadable' if not row['readable'] else 'zero rows'}"))
    pooled = defaultdict(lambda: [0, 0, 0, 0])
    for row in audit.files:
        for name, stats in row['stats'].items():
            item = pooled[(row['role'], name)]
            item[0] += row['rows']
            item[1] += stats['missing']
            item[2] += stats['finite'] == 0
            item[3] += 1
    selected = sorted([(role, name, values) for (role, name), values in pooled.items() if values[1]], key=lambda row: -(row[2][1] / row[2][0]))
    if selected:
        heading('Predictors with the most missing data')
        table(['Role', 'Predictor', 'Missing', 'All-empty files'], [[role, name, fmt_pct(percentage(values[1], values[0])), f'{values[2]}/{values[3]}'] for role, name, values in selected], [.15, .47, .17, .21])
    if audit.issues:
        story.append(PageBreak())
        heading('All significant findings')
        scopes = defaultdict(list)
        for item in audit.issues:
            scopes[item['scope']].append(item)
        for scope in sorted(scopes, key=lambda key: (not any(row['level'] == 'ERROR' for row in scopes[key]), key)):
            # Long paths are body paragraphs so keep-with-next cannot create oversized blocks.
            story.append(p(scope))
            for item in sorted(scopes[scope], key=lambda row: (row['level'] != 'ERROR', row['check'])):
                # Keep individual paragraphs bounded even for enormous schema differences.
                text = f"[{item['level']}] {item['check']}: {item['detail']}"
                for start in range(0, len(text), 1800):
                    story.append(p(text[start:start + 1800]))
    if audit.provenance:
        story.append(PageBreak())
        heading('Final-file SHA-256 provenance')
        for path, before, after in audit.provenance:
            story.append(p(f'{path}\nSHA-256: {before}\nStable while read: {before == after}'))
    report = Path(args.report).expanduser().resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(report), pagesize=A4, leftMargin=36, rightMargin=36, topMargin=38, bottomMargin=38, title='ML Ready Data Validation', author='Hydro Health validation')

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#536876'))
        canvas.drawString(36, 21, 'ML Ready Data Validation')
        canvas.drawRightString(A4[0] - 36, 21, f'Page {document.page}')
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return report


def s3_report_destination(args, local_report):
    """Use REPORTS.SUBFOLDER under the configured output bucket/prefix."""
    subfolder = args.reports_subfolder
    if not subfolder:
        raise ValueError("Configure get_config_item('REPORTS', 'SUBFOLDER') or supply --reports-subfolder; use --local-only to disable S3 saving explicitly")
    subfolder = str(subfolder).strip()
    if not subfolder or subfolder.startswith('/') or '://' in subfolder or '..' in Path(subfolder).parts:
        raise ValueError('REPORTS.SUBFOLDER must be a relative folder under the S3 report root')
    base = args.report_s3_root
    if not base:
        if str(args.root).startswith('s3://'):
            base = args.root
        else:
            bucket = config('SHARED', 'OUTPUT_BUCKET')
            if not bucket:
                raise ValueError('Local input root requires SHARED.OUTPUT_BUCKET or --report-s3-root for S3 reports')
            base = bucket if '://' in bucket else 's3://' + bucket.strip('/')
            if args.output_prefix:
                base = base.rstrip('/') + '/' + args.output_prefix
    base = str(base).strip().rstrip('/')
    if not base.startswith('s3://') or not base[5:]:
        raise ValueError('S3 report root must be s3://bucket or s3://bucket/prefix')
    return base + '/' + subfolder.strip('/') + '/' + Path(local_report).name


def upload_report(args, local_report):
    """Stream the completed PDF once; preserve the local copy if upload fails."""
    destination = s3_report_destination(args, local_report)
    target = UPath(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(local_report).open('rb') as source, target.open('wb') as remote:
        shutil.copyfileobj(source, remote, length=1024 * 1024)
    # Verify length without downloading the report again.
    if target.stat().st_size != Path(local_report).stat().st_size:
        raise OSError(f'Uploaded report size differs from local PDF: {destination}')
    return destination


def config(section, key):
    try:
        from hydro_health.helpers.tools import get_config_item
        value = get_config_item(section, key)
        return str(value).strip() if value is not None else None
    except Exception:
        return None


def find_repository_root(explicit=None):
    """Resolve the same repository that owns src/hydro_health and outputs."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    current, script = Path.cwd().resolve(), Path(__file__).resolve().parent
    for candidate in dict.fromkeys([current, *current.parents, script, *script.parents]):
        if (candidate / 'src' / 'hydro_health').is_dir():
            return candidate
    return current


def get_validation_engine_class():
    """Load the actual repository base lazily; never silently substitute a fake Engine."""
    from hydro_health.engines.Engine import Engine
    from hydro_health.helpers.tools import get_config_item

    class MLReadyValidationEngine(Engine):
        """Use the same initialization and region/path pattern as DataPrepEngine."""

        def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
            super().__init__()
            self.param_lookup = param_lookup
            self.output_prefix = output_prefix
            self.is_aws = param_lookup['env'] in ['remote', 'aws']
            self.repository_root = find_repository_root(param_lookup.get('repository_root'))
            self.outputs_root = self.repository_root / 'outputs'
            self.options = None

        def _resolve_paths(self, region: str):
            self.region = region
            self.outputs_dir = self.outputs_root / self.output_prefix / region if self.output_prefix else self.outputs_root / region
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            self.write_message(f'MLReadyValidationEngine outputs for {region}: {self.outputs_dir}', self.outputs_root)
            bucket = str(get_config_item('SHARED', 'OUTPUT_BUCKET')).strip('/')
            bucket = bucket.removeprefix('s3://')
            self.s3_dir_base = f's3://{bucket}/{self.output_prefix}/{region}' if self.output_prefix else f's3://{bucket}/{region}'
            if self.options.root_override:
                self.s3_dir_base = f'{self.options.root.rstrip("/")}/{region}'
            base = UPath(self.s3_dir_base) if self.is_aws else UPath(self.options.root) / region
            configured = {
                'training': get_config_item('MODEL', 'ML_PREP_TRAIN_DIR'),
                'prediction': get_config_item('MODEL', 'ML_PREP_PRED_DIR'),
            }
            option_names = {'training': 'train_dir', 'prediction': 'pred_dir'}
            resolved = {}
            for kind, directory in configured.items():
                override = getattr(self.options, kind + '_root')
                name = getattr(self.options, option_names[kind]) or directory
                resolved[kind] = UPath(str(override).format(region=region)) if override else base / str(name)
            self.ml_prep_train_dir = resolved['training']
            self.ml_prep_pred_dir = resolved['prediction']
            self.reports_subfolder = get_config_item('REPORTS', 'SUBFOLDER')
            return resolved

        def run(self, options=None):
            if 'eco_regions' not in self.param_lookup or not hasattr(self.param_lookup['eco_regions'], 'value'):
                raise ValueError("Expected param_lookup['eco_regions'].value from the project runner. Call run_validation(param_lookup, output_prefix), or supply --regions for a direct CLI run.")
            values = self.param_lookup['eco_regions'].value
            if values is None or isinstance(values, (str, bytes)):
                raise ValueError("param_lookup['eco_regions'].value must be a nonempty iterable of region names")
            regions = []
            for eco_region in self.param_lookup['eco_regions'].value:
                if not isinstance(eco_region, str) or not eco_region.strip():
                    raise ValueError('Each eco_region must be a nonblank string')
                regions.append(eco_region.strip())
            if not regions:
                raise ValueError("No eco_regions selected in param_lookup['eco_regions'].value")
            if options is None:
                argv = ['--standalone', '--repo-root', str(self.repository_root), '--env', self.param_lookup['env']]
                if self.output_prefix:
                    argv.extend(['--output-prefix', str(self.output_prefix)])
                argv.extend(['--regions', *regions])
                options = parse_args(argv)
                options.standalone = False
            # Project parameters are authoritative, including when options were supplied.
            options.regions = list(dict.fromkeys(regions))
            self.options = options
            if options.target_crs is None and getattr(self, 'target_crs', None) is not None:
                options.target_crs = str(self.target_crs)
            return execute_validation(options, path_resolver=self._resolve_paths)

    return MLReadyValidationEngine


def __getattr__(name):
    # Support `from Data_Prep_Engine_DV_tests import MLReadyValidationEngine`
    # without importing the real project base in spawned table-check workers.
    if name == 'MLReadyValidationEngine':
        return get_validation_engine_class()
    raise AttributeError(f'Module {__name__!r} has no attribute {name!r}')


def parse_args(argv=None):
    # Pre-read repo-root so configuration imports work when this file is elsewhere.
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument('--repo-root', type=Path)
    preliminary.add_argument('--standalone', action='store_true')
    preliminary.add_argument('--env', choices=['aws', 'remote', 'local'], default='aws')
    preliminary.add_argument('--output-prefix', default='')
    known, _ = preliminary.parse_known_args(argv)
    repository_root = find_repository_root(known.repo_root)
    sys.path.insert(0, str(repository_root / 'src'))
    project_engine = None
    effective_argv = sys.argv[1:] if argv is None else argv
    if not known.standalone and not any(value in effective_argv for value in ['--help', '-h']):
        try:
            engine_class = get_validation_engine_class()
            project_engine = engine_class({'env': known.env, 'repository_root': str(repository_root)}, known.output_prefix or False)
        except Exception as exc:
            raise RuntimeError(f'Could not initialize the real project Engine: {exc}. Run from the hydro_health environment with --repo-root pointing to the repository. --standalone is available explicitly for isolated validation without the project base.') from exc
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, parents=[preliminary])
    parser.add_argument('--root', help='Optional explicit root above ER_* folders; otherwise use engine configuration.')
    parser.add_argument('--regions', nargs='+')
    parser.add_argument('--report', default='ML_Ready_Validation_Report.pdf')
    parser.add_argument('--reports-subfolder', default=config('REPORTS', 'SUBFOLDER'), help="Default: get_config_item('REPORTS', 'SUBFOLDER').")
    parser.add_argument('--report-s3-root', help='Report bucket/prefix; defaults to S3 input root, otherwise SHARED.OUTPUT_BUCKET.')
    parser.add_argument('--local-only', action='store_true', help='Explicitly save only the local PDF; default also saves to S3.')
    parser.add_argument('--workers', type=int, default=16, help='Maximum worker processes, default 16. Use 1 for sequential execution; reduce for limited RAM.')
    for option, key in [('train', 'ML_PREP_TRAIN_DIR'), ('pred', 'ML_PREP_PRED_DIR')]:
        parser.add_argument(f'--{option}-dir', default=config('MODEL', key))
    for kind in ['training', 'prediction']:
        parser.add_argument(f'--{kind}-root', help='Explicit root; optional {region} template.')
    parser.add_argument('--target-crs')
    parser.add_argument('--master-predictors')
    parser.add_argument('--mode', choices=['native_missing', 'legacy_complete'], default='native_missing')
    parser.add_argument('--missing-predictor-policy', choices=['fill_nan', 'drop', 'error'], default='fill_nan')
    parser.add_argument('--minimum-training-rows', type=int, default=100)
    parser.add_argument('--require-prediction', action='store_true')
    for name, default in [('missing-threshold', .20), ('missingness-gap', .20), ('range-threshold', .20), ('weight-ess-fraction', .10)]:
        parser.add_argument('--' + name, type=float, default=default)
    parser.add_argument('--nodata', action='append', type=float, default=[])
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--rtol', type=float, default=1e-5)
    parser.add_argument('--geometry-tolerance', type=float, default=1e-7)
    parser.add_argument('--max-abs-delta-rate', type=float)
    parser.add_argument('--weight-method', choices=['power', 'capped_linear', 'unweighted'], default='power')
    parser.add_argument('--weight-alpha', type=float, default=.5)
    parser.add_argument('--weight-cap-quantile', type=float, default=.95)
    parser.add_argument('--max-weight', type=float, default=2.)
    parser.add_argument('--tile-polygons')
    parser.add_argument('--polygon-tile-column', default='tile_id')
    parser.add_argument('--hash-inputs', action='store_true')
    parser.add_argument('--warn-float64', action='store_true', help='Optional storage warnings; float64 is allowed by the current engine.')
    args = parser.parse_args(argv)
    args._project_engine = project_engine
    args.root_override = bool(args.root)
    args.repo_root = repository_root
    prefix = str(args.output_prefix).strip().strip('/')
    if '://' in prefix or '..' in Path(prefix).parts or '\\' in prefix:
        parser.error('--output-prefix must be a relative output prefix, not a bucket URL or parent path.')
    args.output_prefix = prefix
    if project_engine is not None:
        project_engine.output_prefix = prefix or False
        if args.regions:
            # CLI compatibility: convert explicit input into the same parameter shape.
            project_engine.param_lookup['eco_regions'] = SimpleNamespace(value=list(args.regions))
    if args.root and args.output_prefix:
        parser.error('Use --output-prefix with configured paths, or --root as an explicit override; do not combine them.')
    if not args.root:
        if args.env in ['remote', 'aws']:
            bucket = config('SHARED', 'OUTPUT_BUCKET')
            if not bucket:
                parser.error("Could not resolve get_config_item('SHARED', 'OUTPUT_BUCKET'). Check --repo-root and project configuration.")
            s3_dir_base = bucket.rstrip('/') if '://' in bucket else f's3://{bucket.strip("/")}'
            args.root = f'{s3_dir_base}/{args.output_prefix}' if args.output_prefix else s3_dir_base
        else:
            outputs = repository_root / 'outputs'
            args.root = str(outputs / args.output_prefix if args.output_prefix else outputs)
    if not args.train_dir and not args.training_root:
        parser.error('Supply --train-dir/--training-root or configure MODEL.ML_PREP_TRAIN_DIR.')
    if not args.pred_dir and not args.prediction_root:
        parser.error('Supply --pred-dir/--prediction-root or configure MODEL.ML_PREP_PRED_DIR.')
    for name in ['train_dir', 'pred_dir']:
        value = getattr(args, name)
        if value and ('://' in value or Path(value).is_absolute() or '..' in Path(value).parts):
            parser.error(f'--{name.replace("_", "-")} must be relative to each region; use a root override for absolute paths.')
    for name in ['missing_threshold', 'missingness_gap', 'range_threshold', 'weight_ess_fraction']:
        value = getattr(args, name)
        if not np.isfinite(value) or not 0 <= value <= 1:
            parser.error(f'{name} must be in [0, 1].')
    for name in ['atol', 'rtol', 'geometry_tolerance', 'weight_alpha']:
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f'{name} must be finite and nonnegative.')
    if args.minimum_training_rows < 1:
        parser.error('minimum-training-rows must be positive.')
    if args.workers < 1:
        parser.error('--workers must be positive.')
    if not np.isfinite(args.weight_cap_quantile) or not 0 < args.weight_cap_quantile <= 1:
        parser.error('weight-cap-quantile must be in (0, 1].')
    if not np.isfinite(args.max_weight) or args.max_weight < 1:
        parser.error('max-weight must be finite and >= 1.')
    if args.max_abs_delta_rate is not None and (not np.isfinite(args.max_abs_delta_rate) or args.max_abs_delta_rate <= 0):
        parser.error('max-abs-delta-rate must be finite and positive.')
    if any(not np.isfinite(value) for value in args.nodata):
        parser.error('NoData sentinels must be finite.')
    if args.target_crs:
        CRS.from_user_input(args.target_crs)
    if not args.report.lower().endswith('.pdf') or '://' in args.report:
        parser.error('--report must be a local .pdf path.')
    return args


def execute_validation(args, path_resolver=None):
    if not args.local_only:
        # Fail before expensive table checks if report destination is misconfigured.
        s3_report_destination(args, Path(args.report))
    audit = Audit(args)
    audit.path_resolver = path_resolver
    audit.run()
    report = write_pdf(audit)
    counts = Counter(row['level'] for row in audit.issues)
    print(f'Report: {report}\nErrors: {counts["ERROR"]:,}; warnings: {counts["WARNING"]:,}')
    if not args.local_only:
        try:
            destination = upload_report(args, report)
            print(f'S3 report: {destination}')
        except Exception as exc:
            print(f'Could not save report to S3: {type(exc).__name__}: {exc}. Local PDF retained at {report}', file=sys.stderr)
            return 2
    return 1 if counts['ERROR'] else 0


def main(argv=None):
    args = parse_args(argv)
    engine = vars(args).pop('_project_engine', None)
    return engine.run(args) if engine is not None else execute_validation(args)


def run_validation(param_lookup: dict, output_prefix: str | bool = False):
    """Entry point for the existing project runner; no CLI region flags required."""
    repository_root = find_repository_root(param_lookup.get('repository_root'))
    sys.path.insert(0, str(repository_root / 'src'))
    engine_class = get_validation_engine_class()
    return engine_class(param_lookup, output_prefix).run()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('Validation interrupted; report may be incomplete.', file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f'Fatal validation/reporting error: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(2)
