"""Batch tiling tests following the BlueTopo pytest structure.

Real Parquet round trips verify values and schemas. Configuration, shared
Engine services, S3 and distributed scheduling are isolated.
"""
import importlib
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))
module = importlib.import_module('hydro_health.engines.BatchTilingEngine')
from hydro_health.engines.BatchTilingEngine import BatchTilingEngine

TILING_PATH = 'hydro_health.engines.BatchTilingEngine'
YEAR_PAIRS = [(2004, 2006), (2006, 2010)]
FEATURES = ['bpi_broad', 'bpi_fine', 'curv_plan', 'curv_profile', 'curv_total',
            'flowacc', 'gradmag', 'rugosity', 'shearproxy', 'slope', 'slope_deg',
            'tci', 'terrain_classification']


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    lookup = {('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
              ('MODEL', 'TRAINING_TILES_DIR'): 'training_tiles',
              ('MODEL', 'PREDICTION_TILES_DIR'): 'prediction_tiles'}
    module._get_s3_filesystem.cache_clear()
    with patch(f'{TILING_PATH}.Engine.__init__', return_value=None), \
         patch(f'{TILING_PATH}.Engine.write_message_dask', create=True) as log, \
         patch(f'{TILING_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{TILING_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{TILING_PATH}.tempfile.gettempdir', return_value=str(tmp_path)), \
         patch(f'{TILING_PATH}.get_config_item', side_effect=lambda *args: lookup[args]), \
         patch(f'{TILING_PATH}.s3fs.S3FileSystem') as s3:
        try:
            yield SimpleNamespace(log=log, s3=s3.return_value, constructor=s3)
        finally:
            module._get_s3_filesystem.cache_clear()


@pytest.fixture
def victim():
    engine = BatchTilingEngine({'env': 'local', 'eco_regions': ['ER_3']},
                               year_ranges=YEAR_PAIRS.copy())
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.cleanup_resources = MagicMock()
    engine.log_system_metrics = MagicMock(return_value='test metrics')
    engine.client = MagicMock()
    engine._resolve_paths('ER_3')
    return engine


def identifiers():
    return pd.DataFrame({'X': [500005., 500015., 500005., 500015.],
                         'Y': [3400015., 3400015., 3400005., 3400005.],
                         'FID': [0, 1, 2, 3], 'tile_id': ['TILE_7'] * 4})


def training_frame():
    df = identifiers()
    df['bathy_2004_filled'] = [0., -10., np.nan, 4.]
    df['bathy_2006_filled'] = [1., -8., 3., np.nan]
    df['bathy_2010_filled'] = [2., -9., 5., 8.]
    for year in [2004, 2006]:
        for index, feature in enumerate(FEATURES):
            df[f'{feature}_{year}'] = np.array([0., 1., np.nan, 3.]) + index + (year - 2004) * 100
        df[f'flowdir_{year}'] = [0., 90., 180., np.nan]
    df['grain_size_layer'] = [0., 2., np.nan, 4.]
    df['prim_sed_layer'] = [1., 2., 3., 4.]
    df['survey_end_date'] = pd.to_datetime(['2020-01-01', None, '2021-06-01', '2022-12-31'])
    for index, (y0, y1) in enumerate(YEAR_PAIRS):
        df[f'hurr_strength_mean_{y0}_{y1}'] = [0., 2. + index, np.nan, 4.]
        df[f'tsm_mean_{y0}_{y1}'] = [1. + index, np.nan, 3., 4.]
    # The batch must recompute its response from bathymetry, not trust this.
    df['delta_bathy_2004_2006'] = 999.
    df['irrelevant_large_feature'] = 'unused'
    return df


def prediction_frame():
    df = identifiers()
    df['bt.bathy'] = [0., -10., np.nan, 4.]
    for index, feature in enumerate(FEATURES):
        df[f'bt.{feature}'] = np.array([0., 1., np.nan, 3.]) + index
    df['bt.flowdir'] = [0., 90., 180., np.nan]
    df['bt.uc'] = [0., .2, np.nan, .4]
    df['grain_size_layer'] = [0., 2., np.nan, 4.]
    df['prim_sed_layer'] = [1., 2., 3., 4.]
    df['survey_end_date'] = pd.to_datetime(['2020-01-01', None, '2021-06-01', '2022-12-31'])
    for index, (y0, y1) in enumerate(YEAR_PAIRS):
        df[f'hurr_strength_mean_{y0}_{y1}'] = [0., 2. + index, np.nan, 4.]
        df[f'tsm_mean_{y0}_{y1}'] = [1. + index, np.nan, 3., 4.]
    return df


def process_frame(victim, frame, mode, pairs=None, overwrite=True, cloud=False):
    output = victim.local_tmp_dir.parent / f'{mode}_outputs'
    processor = module._process_training_tile if mode == 'training' else module._process_prediction_tile
    result = processor(frame, str(output), 'TILE_7', YEAR_PAIRS if pairs is None else pairs,
                       cloud, str(victim.local_tmp_dir), 1, 2, False, overwrite)
    return result, output


def write_input(tmp_path, frame, name='wide.parquet'):
    path = tmp_path / 'TILE_7' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def worker_params(victim, path, mode='training', overwrite=True):
    return [str(path), mode, YEAR_PAIRS, str(path.parent), 'TILE_7', False,
            str(victim.local_tmp_dir), 1, 1, False, overwrite]


# =====================================================================
# Constructor, Configuration & Region Paths
# =====================================================================

@pytest.mark.parametrize('env,expected', [
    ('local', 'local'), ('remote', 'remote'), ('aws', 'aws'),
    (SimpleNamespace(valueAsText='aws', value='local'), 'aws'),
    (SimpleNamespace(valueAsText='', value='remote'), 'remote'),
])
def test_init_environment_wrappers_and_unique_scratch(tmp_path, env, expected):
    engine = BatchTilingEngine({'env': env}, 'low_res', YEAR_PAIRS, False)
    other = BatchTilingEngine({})
    assert engine.env == expected and engine.is_aws is (expected in ['aws', 'remote'])
    assert engine.output_prefix == 'low_res' and engine.overwrite_files is False
    assert engine.local_tmp_dir.is_dir() and other.local_tmp_dir.is_dir()
    assert engine.local_tmp_dir != other.local_tmp_dir
    assert engine.local_tmp_dir.parent == tmp_path / 'hydro_health/batch_tiling_tmp'
    assert engine.inputs_dir == tmp_path / 'inputs'
    assert engine._worker_count == 1


@pytest.mark.parametrize('source', ['explicit', 'lookup', 'wrapped', 'inherited', 'fallback'])
def test_year_ranges_precedence(source, monkeypatch):
    inherited = [(1999, 2000)]
    monkeypatch.setattr(BatchTilingEngine, 'year_ranges', inherited, raising=False)
    params, explicit = {}, None
    if source == 'explicit':
        params['year_ranges'] = [(2015, 2022)]
        explicit = YEAR_PAIRS
    elif source == 'lookup':
        params['year_ranges'] = YEAR_PAIRS
    elif source == 'wrapped':
        params['year_ranges'] = SimpleNamespace(value=YEAR_PAIRS)
    elif source == 'fallback':
        params['year_ranges'] = 'not a list'
    engine = BatchTilingEngine(params, year_ranges=explicit)
    assert engine.year_ranges == (inherited if source in ['inherited', 'fallback'] else YEAR_PAIRS)


@pytest.mark.parametrize('cloud', [False, True])
@pytest.mark.parametrize('prefix', [False, True, 'low_res'])
def test_resolve_paths(victim, tmp_path, cloud, prefix):
    victim.is_aws = cloud
    victim.output_prefix = prefix
    victim._resolve_paths('ER_2')
    root = tmp_path / 'outputs'
    if isinstance(prefix, str):
        root /= prefix
    root /= 'ER_2'
    assert victim.outputs_dir == root
    base = UPath('s3://test-bucket/ER_2') if cloud else UPath(root)
    assert victim.training_tiles_dir == base / 'training_tiles'
    assert victim.prediction_tiles_dir == base / 'prediction_tiles'
    if not cloud:
        assert victim.training_tiles_dir.is_dir() and victim.prediction_tiles_dir.is_dir()


@pytest.mark.parametrize('name,expected', [(' bathy_2004_filled ', 'bathy_2004_filled'),
    ('\tbt.bathy\n', 'bt.bathy'), ('TSM_MEAN', 'TSM_MEAN'), (123, '123')])
def test_standardize_column_whitespace_only(name, expected):
    assert module._standardize_col_name(name) == expected


def test_s3_client_is_cached_per_process(isolated_services):
    assert module._get_s3_filesystem() is module._get_s3_filesystem()
    isolated_services.constructor.assert_called_once_with()
    assert module._get_s3_filesystem.cache_info().hits == 1


# =====================================================================
# Selective Parquet Reading & Pixel Deduplication
# =====================================================================

@pytest.mark.parametrize('mode', ['training', 'prediction'])
def test_read_required_columns_real_schema_projection(tmp_path, mode):
    frame = training_frame() if mode == 'training' else prediction_frame()
    frame['slope_2010'] = 99.
    frame['bathy_2022_filled'] = 99.
    frame['tsm_mean_2015_2022'] = 99.
    frame['hurr_strength_mean_2015_2022'] = 99.
    frame['irrelevant_large_feature'] = 'unused'
    frame.rename(columns={'X': ' X '}, inplace=True)
    path = write_input(tmp_path, frame)
    # Spy on the real reader: excluded columns must not be decoded.
    with patch(f'{TILING_PATH}.pd.read_parquet', wraps=pd.read_parquet) as read:
        result = module._read_required_columns(str(path), mode, YEAR_PAIRS)
    selected = read.call_args.kwargs['columns']
    assert list(result.columns) == selected and ' X ' in selected
    assert 'irrelevant_large_feature' not in selected
    assert 'bathy_2022_filled' not in selected and 'slope_2010' not in selected
    if mode == 'training':
        assert 'bathy_2004_filled' in selected and 'bathy_2010_filled' in selected
        assert 'slope_2006' in selected
        assert 'tsm_mean_2015_2022' not in selected
        assert 'delta_bathy_2004_2006' not in selected
    else:
        assert 'bt.bathy' in selected and 'bt.uc' in selected
        assert 'tsm_mean_2015_2022' in selected
    assert len(result) == len(frame)


def test_read_required_columns_empty_years_shared_only_training(tmp_path):
    frame = training_frame()
    path = write_input(tmp_path, frame)
    result = module._read_required_columns(str(path), 'training', [])
    assert list(result.columns) == ['X', 'Y', 'FID', 'tile_id', 'grain_size_layer', 'prim_sed_layer', 'survey_end_date']


def test_read_required_columns_rejects_unusable_schema(tmp_path):
    path = write_input(tmp_path, pd.DataFrame({'unused': [1]}))
    with pytest.raises(ValueError, match='No usable columns'):
        module._read_required_columns(str(path), 'training', YEAR_PAIRS)


@pytest.mark.parametrize('kind', ['tile_fid', 'fid', 'xy', 'full_row'])
def test_deduplicate_uses_stable_available_key(kind):
    if kind == 'tile_fid':
        df = pd.DataFrame({'tile_id': ['A', 'A', 'B'], 'FID': [1, 1, 1], 'value': [10, 99, 20]})
    elif kind == 'fid':
        df = pd.DataFrame({'FID': [1, 1, 2], 'value': [10, 99, 20]})
    elif kind == 'xy':
        df = pd.DataFrame({'X': [0, 0, 1], 'Y': [1, 1, 2], 'FID': [np.nan] * 3, 'value': [10, 99, 20]})
    else:
        df = pd.DataFrame({'value': [10, 10, 20]})
    module._deduplicate_pixels(df)
    assert df['value'].tolist() == [10, 20]
    assert df.index.tolist() == [0, 1]


def test_dedup_incomplete_coordinates_falls_back_to_whole_rows():
    df = pd.DataFrame({'X': [1, 1, 1], 'Y': [np.nan] * 3, 'value': [2, 2, 3]})
    module._deduplicate_pixels(df)
    assert df['value'].tolist() == [2, 3]


# =====================================================================
# Atomic Local Writes, Cloud Publication & Cleanup
# =====================================================================

@pytest.mark.parametrize('cloud', [False, True])
def test_save_parquet_round_trip_and_temp_cleanup(victim, tmp_path, isolated_services, cloud):
    frame = identifiers()
    frame['value'] = [0., np.nan, 2., 3.]
    output = tmp_path / 'final'
    if cloud:
        output.mkdir()
        isolated_services.s3.put.side_effect = lambda src, dst: pathlib.Path(dst).write_bytes(pathlib.Path(src).read_bytes())
    module._save_parquet_file(frame, str(output), 'batch.parquet', cloud,
                              str(victim.local_tmp_dir), ' [1/1]', False)
    pd.testing.assert_frame_equal(pd.read_parquet(output / 'batch.parquet'), frame)
    assert not list(victim.local_tmp_dir.iterdir())
    assert not list(output.glob('*.partial'))
    assert not list(output.glob('.*.partial'))
    if cloud:
        isolated_services.s3.put.assert_called_once()
    assert module._output_exists(str(output), 'batch.parquet')
    assert not module._output_exists(str(output), 'missing.parquet')


@pytest.mark.parametrize('failure', ['encode', 'copy', 'replace', 'upload'])
def test_save_failure_cleans_artifacts_and_preserves_existing(victim, tmp_path, isolated_services, failure):
    output = tmp_path / 'final'
    output.mkdir()
    destination = output / 'batch.parquet'
    destination.write_bytes(b'previous output')
    target = {'encode': f'{TILING_PATH}.pd.DataFrame.to_parquet',
              'copy': f'{TILING_PATH}.shutil.copy2',
              'replace': f'{TILING_PATH}.os.replace',
              'upload': f'{TILING_PATH}._get_s3_filesystem'}[failure]
    # Fail after a partial artifact exists to exercise cleanup paths.
    def fail_copy(src, dst):
        pathlib.Path(dst).write_bytes(b'partial')
        raise OSError('injected failure')
    if failure == 'upload':
        isolated_services.s3.put.side_effect = OSError('injected failure')
        with pytest.raises(OSError, match='injected failure'):
            module._save_parquet_file(identifiers(), str(output), destination.name, True,
                                      str(victim.local_tmp_dir), '', False)
    else:
        with patch(target, side_effect=fail_copy if failure == 'copy' else OSError('injected failure')):
            with pytest.raises(OSError, match='injected failure'):
                module._save_parquet_file(identifiers(), str(output), destination.name, False,
                                          str(victim.local_tmp_dir), '', False)
    assert destination.read_bytes() == b'previous output'
    assert not list(victim.local_tmp_dir.iterdir())
    assert not list(output.glob('.*.partial'))


def test_save_uses_unique_temporary_names(victim, tmp_path):
    output = tmp_path / 'final'
    seen = []
    real_write = pd.DataFrame.to_parquet
    def record(frame, path, **kwargs):
        seen.append(path)
        return real_write(frame, path, **kwargs)
    with patch.object(pd.DataFrame, 'to_parquet', new=record):
        for _ in range(2):
            module._save_parquet_file(identifiers(), str(output), 'batch.parquet', False,
                                      str(victim.local_tmp_dir), '', False)
    assert len(set(seen)) == 2


# =====================================================================
# Training Batches: Scientific Values & Pair Selection
# =====================================================================

@pytest.mark.parametrize('cow', [False, True], ids=['pandas_default', 'copy_on_write'])
def test_training_real_outputs_features_delta_and_nan_positions(victim, cow):
    source = training_frame()
    expected_source = source.copy(deep=True)
    with pd.option_context('mode.copy_on_write', cow):
        (saved, existing, summary), folder = process_frame(victim, source, 'training')
    assert saved == ['TILE_7_2004_2006_training_batch.parquet', 'TILE_7_2006_2010_training_batch.parquet']
    assert existing == [] and 'BATCH COLS:' in summary
    for y0, y1 in YEAR_PAIRS:
        data = pd.read_parquet(folder / f'TILE_7_{y0}_{y1}_training_batch.parquet')
        assert data.columns[:6].tolist() == ['X', 'Y', 'FID', 'tile_id', 'year_t', 'year_t1']
        assert data['year_t'].tolist() == [y0] * 4
        assert data['year_t1'].tolist() == [y1] * 4
        assert data['tile_id'].tolist() == ['TILE_7'] * 4
        assert len(data) == 4  # No row drop for missing predictors or bathymetry.
        np.testing.assert_allclose(data['bathy_t'], expected_source[f'bathy_{y0}_filled'], equal_nan=True)
        np.testing.assert_allclose(data['bathy_t1'], expected_source[f'bathy_{y1}_filled'], equal_nan=True)
        np.testing.assert_allclose(data['delta_bathy'], expected_source[f'bathy_{y1}_filled'] - expected_source[f'bathy_{y0}_filled'], equal_nan=True)
        for feature in FEATURES:
            np.testing.assert_allclose(data[f'{feature}_t'], expected_source[f'{feature}_{y0}'], equal_nan=True)
        np.testing.assert_allclose(data['flowdir_cos_t'], [1, 0, -1, np.nan], atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(data['flowdir_sin_t'], [0, 1, 0, np.nan], atol=1e-6, equal_nan=True)
        for shared in ['grain_size_layer', 'prim_sed_layer', 'survey_end_date']:
            pd.testing.assert_series_equal(data[shared], expected_source[shared], check_names=False)
        assert f'tsm_mean_{y0}_{y1}' in data and f'hurr_strength_mean_{y0}_{y1}' in data
        assert f'tsm_mean_{2015}_{2022}' not in data
        assert 'irrelevant_large_feature' not in data
    assert not list(victim.local_tmp_dir.iterdir())


@pytest.mark.parametrize('case', ['empty_pairs', 'missing_bathy', 'empty_table'])
def test_training_no_configured_or_available_pairs(victim, case):
    frame = identifiers()
    pairs = [] if case == 'empty_pairs' else YEAR_PAIRS
    if case == 'empty_table':
        frame = frame.iloc[:0]
    (saved, existing, summary), folder = process_frame(victim, frame, 'training', pairs)
    assert saved == existing == []
    assert summary in ['NO VALID YEAR PAIRS', 'NO PARQUET FILES GENERATED']
    assert not folder.exists()


def test_training_only_valid_pair_and_static_pair_isolation(victim):
    frame = training_frame().drop(columns='bathy_2010_filled')
    frame['tsm_mean_2015_2022'] = 99.
    (saved, _, _), folder = process_frame(victim, frame, 'training')
    assert saved == ['TILE_7_2004_2006_training_batch.parquet']
    data = pd.read_parquet(folder / saved[0])
    assert 'tsm_mean_2004_2006' in data
    assert 'tsm_mean_2006_2010' not in data and 'tsm_mean_2015_2022' not in data


def test_training_whitespace_lowercase_xy_and_case_insensitive_bathy(victim):
    frame = training_frame().rename(columns={'X': ' x ', 'Y': ' y ',
        'bathy_2004_filled': ' BATHY_2004_FILLED '})
    (saved, _, _), folder = process_frame(victim, frame, 'training', [(2004, 2006)])
    data = pd.read_parquet(folder / saved[0])
    assert 'X' in data and 'Y' in data
    np.testing.assert_allclose(data['delta_bathy'], [1, 2, np.nan, np.nan], equal_nan=True)


def test_training_missing_optional_predictors_warns_and_preserves_rows(victim, isolated_services):
    frame = identifiers()
    frame['bathy_2004_filled'] = [0, 1, 2, 3]
    frame['bathy_2006_filled'] = [1, 3, 3, 2]
    (saved, _, _), folder = process_frame(victim, frame, 'training', [(2004, 2006)])
    data = pd.read_parquet(folder / saved[0])
    assert 'slope_t' not in data and len(data) == 4
    np.testing.assert_array_equal(data['delta_bathy'], [1, 2, 1, -1])
    assert any('missing columns' in str(c.args[0]) for c in isolated_services.log.call_args_list)


# =====================================================================
# Prediction Batches: Schema, Features & Hurricane/TSM Eligibility
# =====================================================================

@pytest.mark.parametrize('cow', [False, True], ids=['pandas_default', 'copy_on_write'])
def test_prediction_real_outputs_preserve_nan_rows_and_year_pairs(victim, cow):
    frame = prediction_frame()
    with pd.option_context('mode.copy_on_write', cow):
        (saved, existing, summary), folder = process_frame(victim, frame.copy(), 'prediction')
    assert saved == ['TILE_7_2004_2006_prediction_batch.parquet', 'TILE_7_2006_2010_prediction_batch.parquet']
    assert existing == [] and 'BATCH COLS:' in summary
    for y0, y1 in YEAR_PAIRS:
        data = pd.read_parquet(folder / f'TILE_7_{y0}_{y1}_prediction_batch.parquet')
        assert data.columns[:6].tolist() == ['X', 'Y', 'FID', 'tile_id', 'year_t', 'year_t1']
        assert data.columns[-3:].tolist() == ['grain_size_layer', 'prim_sed_layer', 'survey_end_date']
        assert data['year_t'].tolist() == [y0] * 4 and data['year_t1'].tolist() == [y1] * 4
        assert len(data) == 4  # Pair eligibility does not filter individual rows.
        for feature in ['bathy', *FEATURES, 'uc']:
            np.testing.assert_allclose(data[f'{feature}_t'], frame[f'bt.{feature}'], equal_nan=True)
        np.testing.assert_allclose(data['flowdir_cos_t'], [1, 0, -1, np.nan], atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(data['flowdir_sin_t'], [0, 1, 0, np.nan], atol=1e-6, equal_nan=True)
        for shared in ['grain_size_layer', 'prim_sed_layer', 'survey_end_date']:
            pd.testing.assert_series_equal(data[shared], frame[shared])
        for prefix in ['hurr_strength_mean', 'tsm_mean']:
            name = f'{prefix}_{y0}_{y1}'
            pd.testing.assert_series_equal(data[name], frame[name])
        assert 'delta_bathy' not in data and 'bathy_t1' not in data
        other = '2006_2010' if y0 == 2004 else '2004_2006'
        assert f'tsm_mean_{other}' not in data


@pytest.mark.parametrize('kind', ['empty_pairs', 'missing_hurricane', 'missing_tsm',
    'all_nan_hurricane', 'all_nan_tsm', 'no_joint_valid_pixel', 'empty_table'])
def test_prediction_requires_joint_hurricane_and_tsm_data(victim, kind):
    frame = prediction_frame()
    hurr, tsm = 'hurr_strength_mean_2004_2006', 'tsm_mean_2004_2006'
    pairs = [] if kind == 'empty_pairs' else [(2004, 2006)]
    if kind == 'missing_hurricane':
        frame.drop(columns=hurr, inplace=True)
    elif kind == 'missing_tsm':
        frame.drop(columns=tsm, inplace=True)
    elif kind == 'all_nan_hurricane':
        frame[hurr] = np.nan
    elif kind == 'all_nan_tsm':
        frame[tsm] = np.nan
    elif kind == 'no_joint_valid_pixel':
        frame[hurr] = [1, 1, np.nan, np.nan]
        frame[tsm] = [np.nan, np.nan, 2, 2]
    elif kind == 'empty_table':
        frame = frame.iloc[:0]
    (saved, existing, summary), folder = process_frame(victim, frame, 'prediction', pairs)
    assert saved == existing == [] and summary == 'NO PARQUET FILES GENERATED'
    assert not folder.exists()


@pytest.mark.parametrize('mode', ['training', 'prediction'])
def test_zero_hurricane_and_tsm_are_valid(victim, mode):
    frame = training_frame() if mode == 'training' else prediction_frame()
    frame['hurr_strength_mean_2004_2006'] = 0.
    frame['tsm_mean_2004_2006'] = 0.
    (saved, _, _), folder = process_frame(victim, frame, mode, [(2004, 2006)])
    data = pd.read_parquet(folder / saved[0])
    assert (data['hurr_strength_mean_2004_2006'] == 0).all()
    assert (data['tsm_mean_2004_2006'] == 0).all()


def test_prediction_legacy_prefix_and_shared_aliases(victim):
    frame = prediction_frame().rename(columns={
        'X': ' x ', 'Y': ' y ', 'bt.bathy': ' BlueTopo_bathy ',
        'bt.slope': 'BLUETOPO_SLOPE', 'grain_size_layer': 'sed_size',
        'prim_sed_layer': 'sed_type'})
    (saved, _, _), folder = process_frame(victim, frame, 'prediction', [(2004, 2006)])
    data = pd.read_parquet(folder / saved[0])
    np.testing.assert_allclose(data['bathy_t'], [0, -10, np.nan, 4], equal_nan=True)
    assert 'slope_t' in data and 'grain_size_layer' in data and 'prim_sed_layer' in data


@pytest.mark.parametrize('mode', ['training', 'prediction'])
def test_batch_deduplicates_first_pixel_record(victim, mode):
    frame = training_frame() if mode == 'training' else prediction_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    (saved, _, _), folder = process_frame(victim, frame, mode, [(2004, 2006)])
    data = pd.read_parquet(folder / saved[0])
    assert len(data) == 4 and data['FID'].tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize('mode', ['training', 'prediction'])
@pytest.mark.parametrize('overwrite', [False, True])
def test_existing_batches_skip_or_overwrite(victim, mode, overwrite):
    frame = training_frame() if mode == 'training' else prediction_frame()
    folder = victim.local_tmp_dir.parent / f'{mode}_outputs'
    folder.mkdir()
    name = f'TILE_7_2004_2006_{mode}_batch.parquet'
    old = pd.DataFrame({'old': [99]})
    old.to_parquet(folder / name)
    (saved, existing, summary), _ = process_frame(victim, frame, mode, [(2004, 2006)], overwrite)
    if overwrite:
        assert saved == [name] and existing == []
        assert 'old' not in pd.read_parquet(folder / name)
    else:
        assert saved == [] and existing == [name]
        assert summary == 'EXISTING FILES: 1'
        pd.testing.assert_frame_equal(pd.read_parquet(folder / name), old)


# =====================================================================
# Complete Worker Tasks & Failure Reporting
# =====================================================================

@pytest.mark.parametrize('mode', ['training', 'prediction'])
def test_worker_real_parquet_to_batch(victim, tmp_path, mode):
    frame = training_frame() if mode == 'training' else prediction_frame()
    path = write_input(tmp_path, frame)
    result = module._transform_tile_task(worker_params(victim, path, mode))
    assert result.startswith('Success: TILE_7 (Generated: 2; already existed: 0;')
    for y0, y1 in YEAR_PAIRS:
        batch = pd.read_parquet(path.parent / f'TILE_7_{y0}_{y1}_{mode}_batch.parquet')
        assert len(batch) == 4 and batch['year_t'].eq(y0).all()
    assert not list(victim.local_tmp_dir.iterdir())


def test_worker_existing_only_returns_skipped(victim, tmp_path):
    path = write_input(tmp_path, training_frame())
    params = worker_params(victim, path, overwrite=False)
    first = module._transform_tile_task(params)
    second = module._transform_tile_task(params)
    assert first.startswith('Success:')
    assert second == 'Skipped: TILE_7 (2 valid output files already exist)'


@pytest.mark.parametrize('kind', ['bad_input', 'unsupported_mode', 'write_error'])
def test_worker_errors_return_failed_status_and_log(victim, tmp_path, isolated_services, kind):
    path = write_input(tmp_path, training_frame())
    params = worker_params(victim, path)
    if kind == 'bad_input':
        path.write_bytes(b'corrupt parquet')
    elif kind == 'unsupported_mode':
        params[1] = 'unknown'
    with patch(f'{TILING_PATH}.logger.exception') as log:
        if kind == 'write_error':
            with patch(f'{TILING_PATH}._save_parquet_file', side_effect=OSError('disk full')):
                result = module._transform_tile_task(params)
        else:
            result = module._transform_tile_task(params)
    assert result.startswith('Failed: wide.parquet - ')
    if kind == 'unsupported_mode':
        assert 'ValueError: Unsupported transformation mode' in result
    elif kind == 'write_error':
        assert 'OSError: disk full' in result
    log.assert_called_once()
    assert any('ERROR: Failed transforming' in str(c.args[0]) for c in isolated_services.log.call_args_list)


def test_worker_no_eligible_pairs_reports_zero_generated(victim, tmp_path):
    path = write_input(tmp_path, identifiers())
    result = module._transform_tile_task(worker_params(victim, path))
    assert 'Generated: 0; already existed: 0' in result
    assert 'NO VALID YEAR PAIRS' in result
    assert not list(path.parent.glob('*_batch.parquet'))


@pytest.mark.parametrize('mode', ['training', 'prediction'])
def test_worker_cloud_payload_and_save_upload(victim, tmp_path, isolated_services, mode):
    frame = training_frame() if mode == 'training' else prediction_frame()
    path = write_input(tmp_path, frame)
    params = worker_params(victim, path, mode)
    params[3] = 's3://test-bucket/ER_3/tiles/TILE_7'
    params[5] = True
    # Source is a local fixture; output publication is mocked S3, with real encoding.
    result = module._transform_tile_task(params)
    assert result.startswith('Success:')
    assert isolated_services.s3.put.call_count == 2
    for call in isolated_services.s3.put.call_args_list:
        assert call.args[1].startswith('s3://test-bucket/ER_3/tiles/TILE_7/')
        assert not pathlib.Path(call.args[0]).exists()


# =====================================================================
# Pipeline Payloads, Batching & Run Lifecycle
# =====================================================================

@pytest.mark.parametrize('case', ['empty_years', 'no_files', 'batch_only'])
def test_pipeline_skips_without_work(victim, tmp_path, case):
    if case == 'empty_years':
        victim.year_ranges = []
    elif case == 'batch_only':
        write_input(tmp_path, identifiers(), 'already_training_batch.parquet')
    victim._process_pipeline(UPath(tmp_path), 'training')
    victim.client.map.assert_not_called()


@pytest.mark.parametrize('workers,expected_sizes', [(1, [4, 4, 1]), (2, [8, 1])])
def test_pipeline_batches_payloads_and_summary(victim, tmp_path, workers, expected_sizes):
    for i in range(9):
        path = tmp_path / f'BH4S556X_{i}' / 'wide.parquet'
        path.parent.mkdir()
        path.touch()
    (tmp_path / 'BH4S556X_0' / 'existing_training_batch.parquet').touch()
    (tmp_path / 'ignored.txt').touch()
    victim._worker_count = workers
    victim.overwrite_files = False
    statuses = iter(['Success: T', 'Skipped: T', 'Failed: T - injected', None] + ['Success: T'] * 5)
    victim.client.map.side_effect = lambda function, params: [next(statuses) for _ in params]
    victim.client.gather.side_effect = lambda futures: futures
    victim._process_pipeline(UPath(tmp_path), 'training', True)
    calls = victim.client.map.call_args_list
    assert [len(c.args[1]) for c in calls] == expected_sizes
    payloads = [p for c in calls for p in c.args[1]]
    assert {p[4] for p in payloads} == {f'BH4S556X_{i}' for i in range(9)}
    for index, payload in enumerate(payloads, 1):
        assert len(payload) == 11
        assert payload[1:3] == ['training', YEAR_PAIRS]
        assert payload[3] == str(pathlib.Path(payload[0]).parent)
        assert payload[5:] == [False, str(victim.local_tmp_dir), index, 9, True, False]
    messages = [str(c.args[0]) for c in victim.write_message.call_args_list]
    assert ' -> Successful Tasks: 6' in messages
    assert ' -> Already Complete Tasks: 1' in messages
    assert ' -> Failed/Error Tasks: 1' in messages
    assert any('Transformation Errors:' in m and 'injected' in m for m in messages)


@pytest.mark.parametrize('regions', [['ER_1', 'ER_2'], SimpleNamespace(value=['ER_1', 'ER_2']), "['ER_1']"])
def test_run_region_wrappers_order_and_cleanup(victim, regions):
    victim.param_lookup['eco_regions'] = regions
    scratch = victim.local_tmp_dir
    with patch.object(victim, '_process_pipeline') as pipeline:
        victim.run()
    expected_regions = ['ER_1'] if isinstance(regions, str) else ['ER_1', 'ER_2']
    assert [c.kwargs['mode'] for c in pipeline.call_args_list] == ['prediction', 'training'] * len(expected_regions)
    for index, region in enumerate(expected_regions):
        assert region in str(pipeline.call_args_list[index * 2].kwargs['base_dir'])
    victim.setup_dask.assert_called_once_with('local', n_workers=16, threads_per_worker=2, memory_limit='6GB')
    victim.cleanup_resources.assert_called_once_with(module.OUTPUTS)
    assert not scratch.exists()


@pytest.mark.parametrize('failure', ['setup', 'pipeline', 'cleanup'])
def test_run_always_removes_own_scratch_on_failure(victim, failure):
    scratch = victim.local_tmp_dir
    sibling = scratch.parent / 'other_engine_task'
    sibling.mkdir()
    (sibling / 'keep.txt').write_text('preserve')
    if failure == 'setup':
        victim.setup_dask.side_effect = RuntimeError('setup failed')
    elif failure == 'cleanup':
        victim.cleanup_resources.side_effect = RuntimeError('cleanup failed')
    with patch.object(victim, '_process_pipeline',
                      side_effect=RuntimeError('pipeline failed') if failure == 'pipeline' else None):
        with pytest.raises(RuntimeError, match='failed'):
            victim.run()
    victim.cleanup_resources.assert_called_once_with(module.OUTPUTS)
    assert not scratch.exists()
    assert (sibling / 'keep.txt').read_text() == 'preserve'
