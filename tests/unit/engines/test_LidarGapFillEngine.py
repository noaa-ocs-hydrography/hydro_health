"""Lidar gap-fill tests following the supplied BlueTopo pytest structure."""
import pathlib
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import dask.array as da
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.LidarGapFillEngine import (
    LidarGapFillEngine, focal_fill_block, _chunked_fill_holes,
    _process_gap_fill, _process_combine_group,
)

LIDAR_PATH = 'hydro_health.engines.LidarGapFillEngine'


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    lookup = {
        ('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
        ('MODEL', 'TILED_LIDAR_PROC'): 'lidar/uncombined',
        ('TERRAIN', 'FILLED_DIR'): 'lidar/filled',
        ('TERRAIN', 'COMBINED_LIDAR_DIR'): 'lidar/combined',
    }
    with patch(f'{LIDAR_PATH}.Engine.__init__', return_value=None), \
         patch(f'{LIDAR_PATH}.Engine.write_message_dask', create=True) as log, \
         patch(f'{LIDAR_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{LIDAR_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{LIDAR_PATH}.Path.home', return_value=tmp_path / 'home'), \
         patch(f'{LIDAR_PATH}.get_config_item', side_effect=lambda *args: lookup[args]), \
         patch(f'{LIDAR_PATH}.s3fs.S3FileSystem') as s3:
        yield SimpleNamespace(log=log, s3=s3.return_value, lookup=lookup)


@pytest.fixture
def victim():
    params = {'env': 'local', 'eco_regions': SimpleNamespace(value=['ER_3'])}
    engine = LidarGapFillEngine(params)
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.cleanup_resources = MagicMock()
    engine.log_system_metrics = MagicMock(return_value='test metrics')
    engine.client = MagicMock()
    engine._resolve_paths('ER_3')
    return engine


def write_raster(path, data, nodata=-9999.0, left=500000, top=3400320, res=8, crs='EPSG:32617'):
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    with rasterio.open(path, 'w', driver='GTiff', count=1, height=data.shape[0],
        width=data.shape[1], dtype='float32', nodata=nodata, crs=crs,
        transform=from_origin(left, top, res, res)) as dst:
        dst.write(data, 1)
    return path


def gap_params(victim, tmp_path, data, nodata=-9999.0, max_iters=2, chunk_size=20, is_aws=False):
    source = write_raster(tmp_path / 'source.tif', data, nodata=nodata)
    output = tmp_path / 'filled.tif'
    return [str(source), str(output), max_iters, chunk_size, str(victim.local_tmp_dir), is_aws]


def bathy_with_hole(nodata=-9999.0):
    data = np.full((40, 40), -10, dtype=np.float32)
    data[:2, :] = nodata
    data[-2:, :] = nodata
    data[:, :2] = nodata
    data[:, -2:] = nodata
    data[19:22, 19:22] = nodata  # Crosses the 20-pixel Dask chunk boundary.
    return data


# =====================================================================
# Initialization & Region Paths
# =====================================================================

@pytest.mark.parametrize('env,is_aws', [('local', False), ('remote', True), ('aws', True)])
def test_init_defaults_and_scratch(tmp_path, env, is_aws):
    params = {'env': env}
    engine = LidarGapFillEngine(params, output_prefix='low_res')
    assert engine.param_lookup is params and engine.is_aws is is_aws
    assert engine.output_prefix == 'low_res'
    assert engine.local_tmp_dir == tmp_path / 'home/hydro_health_local_tmp/gap_fill_tmp'
    assert engine.local_tmp_dir.is_dir()


def test_init_default_environment():
    assert LidarGapFillEngine({}).is_aws is False


@pytest.mark.parametrize('env', ['local', 'remote', 'aws'])
@pytest.mark.parametrize('prefix', [False, 'low_res'])
def test_resolve_paths(tmp_path, env, prefix):
    engine = LidarGapFillEngine({'env': env}, output_prefix=prefix)
    engine.write_message = MagicMock()
    engine._resolve_paths('ER_2')
    base = tmp_path / 'outputs'
    base = base / prefix if prefix else base
    assert engine.outputs_dir == base / 'ER_2'
    root = UPath('s3://test-bucket/ER_2') if engine.is_aws else UPath(base / 'ER_2')
    for attr, suffix in [('tiled_lidar_dir', 'lidar/uncombined'),
                         ('filled_out_dir', 'lidar/filled'), ('combined_lidar_dir', 'lidar/combined')]:
        assert getattr(engine, attr) == root / suffix
        if not engine.is_aws:
            assert getattr(engine, attr).is_dir()


# =====================================================================
# Focal Mean & Binary Hole-Filling Helpers
# =====================================================================

def test_focal_fill_changes_only_missing_values():
    original = np.array([[-1, -2, -3], [-4, np.nan, -6], [-7, -8, -9]], dtype=np.float64)
    before = original.copy()
    result = focal_fill_block(original, w=3)
    assert result.dtype == np.float32
    assert result[1, 1] == pytest.approx(-5)
    np.testing.assert_array_equal(result[~np.isnan(original)], original[~np.isnan(original)])
    np.testing.assert_allclose(original, before, equal_nan=True)


def test_focal_fill_normalizes_edge_without_zero_padding_bias():
    result = focal_fill_block(np.array([[np.nan, -2], [-4, -6]], dtype=np.float32), w=3)
    assert result[0, 0] == pytest.approx(-4)


def test_focal_fill_all_nan_remains_nan():
    assert np.isnan(focal_fill_block(np.full((5, 5), np.nan))).all()


def test_focal_fill_retains_numeric_zeros_as_valid_neighbors():
    result = focal_fill_block(np.array([[0, 0, 0], [0, np.nan, 0], [0, 0, 0]], dtype=np.float32))
    assert result[1, 1] == 0  # The worker separately masks zero elevations.


def test_focal_fill_no_gaps_is_identity():
    data = np.arange(25).reshape(5, 5)
    np.testing.assert_array_equal(focal_fill_block(data), data)


def test_binary_fill_closes_interior_not_edge_connected_gap():
    data = np.ones((9, 9), dtype=np.uint8)
    data[4, 4] = 0
    data[0:3, 2] = 0
    before = data.copy()
    result = _chunked_fill_holes(data)
    assert result.dtype == bool and result[4, 4]
    assert not result[0:3, 2].any()
    np.testing.assert_array_equal(data, before)


def test_real_dask_focal_overlap_matches_independent_local_means():
    data = np.full((12, 12), -4, dtype=np.float32)
    data[4:8, 4:8] = np.nan
    data[3, 5] = -12
    result = da.from_array(data, chunks=(6, 6)).map_overlap(
        focal_fill_block, depth=1, boundary='reflect', dtype=np.float32, w=3).compute(scheduler='single-threaded')
    for row, col in [(4, 5), (5, 4), (6, 7), (5, 5)]:
        neighbors = data[row-1:row+2, col-1:col+2]
        valid = neighbors[~np.isnan(neighbors)]
        if len(valid):
            assert result[row, col] == pytest.approx(float(valid.mean()))
        else:
            assert np.isnan(result[row, col])


# =====================================================================
# Real Gap-Fill Worker, Nodata, Land & Iteration Behavior
# =====================================================================

@pytest.mark.parametrize('nodata', [-9999.0, np.nan], ids=['sentinel', 'nan'])
def test_gap_worker_fills_interior_across_chunks_preserves_exterior(victim, tmp_path, nodata):
    data = bathy_with_hole(nodata)
    params = gap_params(victim, tmp_path, data, nodata=nodata)
    success, message = _process_gap_fill(params)
    assert success, message
    with rasterio.open(params[1]) as src:
        result = src.read(1)
        assert src.crs.to_epsg() == 32617 and src.res == (8, 8)
        assert src.transform == from_origin(500000, 3400320, 8, 8)
        assert src.count == 1 and src.dtypes == ('float32',)
        assert np.isnan(src.nodata) and src.compression.value.lower() == 'lzw'
    np.testing.assert_allclose(result[19:22, 19:22], -10, atol=1e-5)
    assert np.isnan(result[:2, :]).all() and np.isnan(result[:, :2]).all()
    assert result[5, 5] == -10
    assert not list(victim.local_tmp_dir.iterdir())


@pytest.mark.parametrize('nodata', [-9999.0, np.nan, None], ids=['sentinel', 'nan', 'unset'])
def test_gap_worker_masks_land_and_zero(victim, tmp_path, nodata):
    data = np.full((40, 40), -12, dtype=np.float32)
    data[5, 5] = 3
    data[20, 20] = 0
    params = gap_params(victim, tmp_path, data, nodata=nodata)
    success, message = _process_gap_fill(params)
    assert success, message
    with rasterio.open(params[1]) as src:
        result = src.read(1)
    assert np.isnan(result[5, 5])
    assert result[20, 20] == pytest.approx(-12, abs=1e-5)
    assert result[2, 2] == -12
    assert np.all(result[np.isfinite(result)] < 0)


def test_gap_worker_zero_iterations_retains_gap(victim, tmp_path):
    params = gap_params(victim, tmp_path, bathy_with_hole(), max_iters=0)
    success, message = _process_gap_fill(params)
    assert success, message
    with rasterio.open(params[1]) as src:
        assert np.isnan(src.read(1)[20, 20])


def test_gap_worker_multiple_iterations_reach_deep_gap(victim, tmp_path):
    data = np.full((40, 40), -10, dtype=np.float32)
    data[13:20, 13:20] = -9999
    params = gap_params(victim, tmp_path, data, max_iters=1)
    ok, msg = _process_gap_fill(params)
    assert ok, msg
    with rasterio.open(params[1]) as src:
        assert np.isnan(src.read(1)[16, 16])
    pathlib.Path(params[1]).unlink()
    params[2] = 3
    ok, msg = _process_gap_fill(params)
    assert ok, msg
    with rasterio.open(params[1]) as src:
        assert src.read(1)[16, 16] == pytest.approx(-10, abs=1e-5)


@pytest.mark.parametrize('fill', [-9999.0, 0.0])
def test_gap_worker_empty_returns_false_without_output(victim, tmp_path, fill):
    params = gap_params(victim, tmp_path, np.full((40, 40), fill))
    success, message = _process_gap_fill(params)
    assert not success and 'no valid data' in message
    assert not pathlib.Path(params[1]).exists()
    assert not list(victim.local_tmp_dir.iterdir())


def test_gap_worker_existing_output_skips_input(victim, tmp_path):
    params = [str(tmp_path / 'missing.tif'), str(tmp_path / 'exists.tif'), 2, 20, str(victim.local_tmp_dir), False]
    pathlib.Path(params[1]).write_bytes(b'existing')
    with patch(f'{LIDAR_PATH}.rioxarray.open_rasterio') as read:
        success, _ = _process_gap_fill(params)
    assert success
    read.assert_not_called()
    assert pathlib.Path(params[1]).read_bytes() == b'existing'


def test_gap_worker_read_failure_returns_status_and_cleans(victim, tmp_path):
    params = gap_params(victim, tmp_path, np.full((40, 40), -10))
    with patch(f'{LIDAR_PATH}.rioxarray.open_rasterio', side_effect=OSError('read failed')):
        success, message = _process_gap_fill(params)
    assert not success and 'read failed' in message
    assert not list(victim.local_tmp_dir.iterdir())


@pytest.mark.parametrize('upload_fails', [False, True])
def test_gap_worker_cloud_stages_and_cleans(victim, tmp_path, isolated_services, upload_fails):
    params = gap_params(victim, tmp_path, np.full((40, 40), -10), is_aws=True)
    captured = []
    def put(source, destination):
        captured.append(source)
        assert pathlib.Path(source).exists() and destination == params[1]
        with rasterio.open(source) as src:
            assert np.all(src.read(1) == -10)
        if upload_fails:
            raise OSError('upload failed')
    isolated_services.s3.put.side_effect = put
    success, message = _process_gap_fill(params)
    assert success is (not upload_fails), message
    assert len(captured) == 1 and not pathlib.Path(captured[0]).parent.exists()
    assert not list(victim.local_tmp_dir.iterdir())

# =====================================================================
# Combining Overlaps, Coordinate Alignment & Single-File Copies
# =====================================================================

def combine_params(victim, tmp_path, arrays):
    paths = [str(write_raster(tmp_path / f'input_{i}.tif', data)) for i, data in enumerate(arrays)]
    return [paths, str(tmp_path / 'combined.tif'), 2, str(victim.local_tmp_dir), False]


def test_combine_averages_valid_values_and_masks_nonnegative(victim, tmp_path):
    params = combine_params(victim, tmp_path, [
        [[-2, 0, -9999], [-8, 2, -9999]],
        [[-4, -6, -9999], [0, 4, -9999]],
    ])
    success, message = _process_combine_group(params)
    assert success, message
    with rasterio.open(params[1]) as src:
        # Read with coordinates to make orientation checks explicit.
        expected = { (500004, 3400316): -3, (500012, 3400316): -6,
                     (500004, 3400308): -8 }
        for coordinate, value in expected.items():
            assert next(src.sample([coordinate]))[0] == pytest.approx(value)
        for coordinate in [(500012, 3400308), (500020, 3400316)]:
            assert np.isnan(next(src.sample([coordinate]))[0])
        assert src.crs.to_epsg() == 32617 and src.count == 1
        assert np.isnan(src.nodata) and src.compression.value.lower() == 'lzw'
        assert src.bounds == rasterio.coords.BoundingBox(500000, 3400304, 500024, 3400320)
    assert not list(victim.local_tmp_dir.iterdir())


def test_combine_outer_alignment_partial_overlap(victim, tmp_path):
    a = write_raster(tmp_path / 'a.tif', np.full((2, 2), -2))
    b = write_raster(tmp_path / 'b.tif', np.full((2, 2), -6), left=500008)
    out = tmp_path / 'combined.tif'
    success, message = _process_combine_group([[str(a), str(b)], str(out), 2, str(victim.local_tmp_dir), False])
    assert success, message
    with rasterio.open(out) as src:
        assert src.shape == (2, 3)
        # The union retains nonoverlapping edge cells and averages shared cells.
        for x, value in [(500004, -2), (500012, -4), (500020, -6)]:
            assert next(src.sample([(x, 3400316)]))[0] == pytest.approx(value)


def test_combine_single_local_copies_unchanged(victim, tmp_path):
    params = combine_params(victim, tmp_path, [[[0, 2], [-4, -9999]]])
    success, message = _process_combine_group(params)
    assert success and 'Copied single dataset' in message
    assert pathlib.Path(params[1]).read_bytes() == pathlib.Path(params[0][0]).read_bytes()


def test_combine_single_cloud_uses_server_copy(victim, isolated_services):
    output = 's3://test-bucket/combined.tif'
    with patch.object(UPath, 'exists', return_value=False):
        success, message = _process_combine_group([
            ['s3://test-bucket/filled.tif'], output, 16, str(victim.local_tmp_dir), True])
    assert success, message
    isolated_services.s3.copy.assert_called_once_with('s3://test-bucket/filled.tif', output)
    isolated_services.s3.put.assert_not_called()


def test_combine_existing_output_skips(victim, tmp_path):
    output = tmp_path / 'existing.tif'
    output.write_bytes(b'keep')
    with patch(f'{LIDAR_PATH}.rioxarray.open_rasterio') as read:
        success, message = _process_combine_group([['missing-a', 'missing-b'], str(output), 16, str(victim.local_tmp_dir), False])
    assert success and 'Already exists' in message
    read.assert_not_called()
    assert output.read_bytes() == b'keep'


def test_combine_read_failure_returns_false(victim, tmp_path):
    params = combine_params(victim, tmp_path, [np.full((2, 2), -2), np.full((2, 2), -4)])
    with patch(f'{LIDAR_PATH}.rioxarray.open_rasterio', side_effect=OSError('read failed')):
        success, message = _process_combine_group(params)
    assert not success and 'read failed' in message
    assert not list(victim.local_tmp_dir.iterdir())


def test_combine_grid_size_guard(victim, tmp_path):
    a = write_raster(tmp_path / 'a.tif', np.full((2, 2), -2), left=0, top=8, res=1)
    b = write_raster(tmp_path / 'b.tif', np.full((2, 2), -4), left=100000, top=100008, res=1)
    output = tmp_path / 'too_large.tif'
    success, message = _process_combine_group([[str(a), str(b)], str(output), 2, str(victim.local_tmp_dir), False])
    assert not success and 'Grid is too large' in message
    assert not output.exists()


@pytest.mark.parametrize('upload_fails', [False, True])
def test_combine_cloud_staging_and_cleanup(victim, tmp_path, isolated_services, upload_fails):
    params = combine_params(victim, tmp_path, [np.full((2, 2), -2), np.full((2, 2), -6)])
    params[4] = True
    paths = []
    def put(source, destination):
        paths.append(source)
        with rasterio.open(source) as src:
            assert np.all(src.read(1) == -4)
        assert destination == params[1]
        if upload_fails:
            raise OSError('upload failed')
    isolated_services.s3.put.side_effect = put
    success, message = _process_combine_group(params)
    assert success is (not upload_fails), message
    assert len(paths) == 1 and not pathlib.Path(paths[0]).parent.exists()


# =====================================================================
# Task Discovery, Year/Tile Grouping & Dask Payloads
# =====================================================================

def test_execute_gap_fill_discovers_extensions_and_builds_tasks(victim):
    sources = [victim.tiled_lidar_dir / 'a.tif', victim.tiled_lidar_dir / 'nested/b.tiff']
    for path in sources:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (victim.tiled_lidar_dir / 'ignore.txt').touch()
    victim.client.gather.return_value = [(True, 'filled'), (False, 'skipped')]
    victim._execute_gap_fill(max_iters=4, chunk_size=128)
    args = victim.client.map.call_args
    assert args.args[0] is _process_gap_fill
    expected = {
        (str(sources[0]), str(victim.filled_out_dir / 'a_filled.tif')),
        (str(sources[1]), str(victim.filled_out_dir / 'b_filled.tif')),
    }
    assert {(p[0], p[1]) for p in args.args[1]} == expected
    assert all(p[2:] == [4, 128, str(victim.local_tmp_dir), False] for p in args.args[1])
    assert any('[ERROR/SKIP]' in c.args[0] for c in victim.write_message.call_args_list)
    victim.client.gather.assert_called_once_with(victim.client.map.return_value)


def test_execute_combine_groups_by_year_and_tile(victim):
    names = ['survey_2004_BH4S556X_3_a_filled.tif',
             'survey_2004_BH4S556X_3_b_filled.tiff',
             'survey_2006_BH4S556X_3_c_filled.tif',
             'survey_2004_BH4S2574_3_d_filled.tif', 'other_filled.tif']
    for name in names:
        (victim.filled_out_dir / name).touch()
    victim.client.gather.return_value = [(True, 'combined'), (False, 'skipped')]
    victim._execute_combine_lidar(chunk_size=64)
    function, batches = victim.client.map.call_args.args
    assert function is _process_combine_group and len(batches) == 4
    groups = {pathlib.Path(p[1]).name: p for p in batches}
    pair = groups['combined2_2004_BH4S556X_lidar.tif']
    assert {pathlib.Path(p).name for p in pair[0]} == set(names[:2])
    assert all(p[2:] == [64, str(victim.local_tmp_dir), False] for p in batches)
    assert {name for name in groups if name.startswith('combined1_')} == {f'combined1_{name}' for name in names[2:]}


@pytest.mark.parametrize('method,args', [('_execute_gap_fill', (5, 128)), ('_execute_combine_lidar', (128,))])
def test_execute_empty_directory_no_tasks(victim, method, args):
    getattr(victim, method)(*args)
    victim.client.map.assert_not_called()
    victim.client.gather.assert_not_called()


# =====================================================================
# Multi-Region Run Order & Resource Cleanup
# =====================================================================

def test_run_region_order_and_settings(victim):
    victim.param_lookup['eco_regions'].value = ['ER_1', 'ER_2']
    manager = MagicMock()
    manager.attach_mock(victim.setup_dask, 'setup')
    manager.attach_mock(victim.cleanup_resources, 'cleanup')
    with patch.object(victim, '_resolve_paths') as resolve, \
         patch.object(victim, '_execute_gap_fill') as gap, \
         patch.object(victim, '_execute_combine_lidar') as combine:
        manager.attach_mock(resolve, 'resolve')
        manager.attach_mock(gap, 'gap')
        manager.attach_mock(combine, 'combine')
        victim.run(max_iters=7, chunk_size=256)
    assert manager.mock_calls == [
        call.setup('local', n_workers=3, threads_per_worker=1, memory_limit='10GB'),
        call.resolve('ER_1'), call.gap(7, 256), call.combine(256),
        call.resolve('ER_2'), call.gap(7, 256), call.combine(256),
        call.cleanup(victim.outputs_dir.parents[0]),
    ]


@pytest.mark.parametrize('stage', ['setup', 'gap', 'combine'])
def test_run_exception_still_cleans(victim, stage):
    if stage == 'setup':
        victim.setup_dask.side_effect = RuntimeError('test failure')
    with patch.object(victim, '_resolve_paths'), \
         patch.object(victim, '_execute_gap_fill') as gap, \
         patch.object(victim, '_execute_combine_lidar') as combine:
        if stage == 'gap':
            gap.side_effect = RuntimeError('test failure')
        if stage == 'combine':
            combine.side_effect = RuntimeError('test failure')
        with pytest.raises(RuntimeError, match='test failure'):
            victim.run()
    victim.cleanup_resources.assert_called_once()


def test_run_default_arguments(victim):
    with patch.object(victim, '_resolve_paths'), patch.object(victim, '_execute_gap_fill') as gap, \
         patch.object(victim, '_execute_combine_lidar') as combine:
        victim.run()
    gap.assert_called_once_with(5, 1024)
    combine.assert_called_once_with(1024)
    victim.cleanup_resources.assert_called_once()
