"""Training raster tests following the BlueTopo pytest structure."""
import pathlib
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin, from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

import importlib
module = importlib.import_module('hydro_health.engines.TrainingRastersEngine')
from hydro_health.engines.TrainingRastersEngine import TrainingRastersEngine

TRAINING_PATH = 'hydro_health.engines.TrainingRastersEngine'


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    lookup = {('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
        ('MASK', 'MASK_TRAINING_PATH'): 'masks/training.tif',
        ('MODEL', 'PREDICTION_OUTPUT_DIR'): 'prediction',
        ('MODEL', 'TRAINING_OUTPUT_DIR'): 'training',
        ('MODEL', 'SUBGRIDS'): 'subgrids/training.gpkg'}
    with patch(f'{TRAINING_PATH}.Engine.__init__', return_value=None), \
         patch(f'{TRAINING_PATH}.Engine.write_message_dask', create=True) as log, \
         patch(f'{TRAINING_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{TRAINING_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{TRAINING_PATH}.Path.home', return_value=tmp_path / 'home'), \
         patch(f'{TRAINING_PATH}.get_config_item', side_effect=lambda *args: lookup[args]), \
         patch(f'{TRAINING_PATH}.s3fs.S3FileSystem') as s3:
        yield SimpleNamespace(log=log, s3=s3.return_value, lookup=lookup)


@pytest.fixture
def victim():
    engine = TrainingRastersEngine({'env': 'local', 'eco_regions': SimpleNamespace(value=['ER_3'])})
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.client = MagicMock()
    engine._resolve_paths('ER_3')
    return engine


def write_raster(path, data, nodata=-9999, dtype='float32', transform=None, crs='EPSG:32617'):
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=dtype)
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[0], width=data.shape[1],
        count=1, dtype=dtype, crs=crs, nodata=nodata,
        transform=transform or from_origin(500000, 3400032, 8, 8)) as dst:
        dst.write(data, 1)
    return path


def worker_params(victim, tmp_path, values=None, mask=None, nodata=-9999):
    values = [[0, 1, 2, 3], [4, -9999, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]] if values is None else values
    mask = [[2, 1, 2, 0], [2, 2, 1, 0], [0, 2, 1, 2], [1, 1, 2, 2]] if mask is None else mask
    source = write_raster(tmp_path / 'prediction.tif', values, nodata=nodata)
    mask_path = write_raster(tmp_path / 'mask.tif', mask, nodata=255, dtype='uint8')
    with rasterio.open(mask_path) as src:
        bounds = tuple(src.bounds)
    return [str(source), bounds, str(tmp_path / 'training.tif'), str(mask_path), 1, 1, False, str(victim.local_tmp_dir)]


def aligned_mask(source, mask):
    return WarpedVRT(mask, crs=source.crs, transform=source.transform,
        height=source.height, width=source.width, resampling=Resampling.nearest,
        src_nodata=mask.nodata, nodata=mask.nodata)


# =====================================================================
# Constructor & Region Paths
# =====================================================================

@pytest.mark.parametrize('env,cloud', [('local', False), ('remote', True), ('aws', True)])
def test_init_environment_and_scratch(tmp_path, env, cloud):
    engine = TrainingRastersEngine({'env': env}, 'low_res')
    assert engine.is_aws is cloud and engine.output_prefix == 'low_res'
    assert engine.local_tmp_dir == tmp_path / 'home/hydro_health_local_tmp'
    assert engine.local_tmp_dir.is_dir()


@pytest.mark.parametrize('env', ['local', 'remote', 'aws'])
@pytest.mark.parametrize('prefix', [False, 'low_res'])
def test_resolve_paths(tmp_path, env, prefix):
    engine = TrainingRastersEngine({'env': env}, prefix)
    engine.write_message = MagicMock()
    base = tmp_path / 'outputs'
    base = base / prefix if prefix else base
    # Cloud path resolution creates the output folder; prevent an actual S3 call.
    with ExitStack() as stack:
        if engine.is_aws:
            stack.enter_context(patch.object(UPath, 'mkdir'))
        engine._resolve_paths('ER_2')
    assert engine.outputs_dir == base / 'ER_2'
    cloud_root = 's3://test-bucket/low_res/ER_2' if prefix else 's3://test-bucket/ER_2'
    root = UPath(cloud_root) if engine.is_aws else UPath(base / 'ER_2')
    assert engine.train_mask_path == root / 'masks/training.tif'
    assert engine.prediction_out_dir == root / 'prediction'
    assert engine.training_out_dir == root / 'training'
    assert engine.subgrid_paths == {'training': root / 'subgrids/training.gpkg'}
    assert engine.filled_folder_name == 'filled_tifs'
    if not engine.is_aws: assert engine.training_out_dir.is_dir()


# =====================================================================
# Bounds Intersection, Metadata & Valid-Pixel Precheck
# =====================================================================

@pytest.mark.parametrize('bounds,expected', [((1, 1, 3, 3), True), ((4, 0, 6, 4), True), ((10, 10, 20, 20), False)])
def test_check_intersection(victim, bounds, expected):
    source = SimpleNamespace(bounds=(0, 0, 4, 4))
    assert module._check_intersection(source, bounds, 'test.tif', '') is expected


@pytest.mark.parametrize('nodata', [-9999.0, np.nan, None], ids=['sentinel', 'nan', 'unset'])
def test_prepare_metadata_preserves_grid_and_source(tmp_path, nodata):
    path = write_raster(tmp_path / 'source.tif', [[0, 1], [2, 3]], nodata=nodata)
    with rasterio.open(path) as src:
        original = src.meta.copy()
        meta, source_nodata = module._prepare_metadata(src)
        assert meta['transform'] == src.transform and meta['crs'] == src.crs
        assert meta['width'] == 2 and meta['height'] == 2
        assert meta['compress'] == 'lzw' and meta['tiled'] is True
        assert src.meta == original
    if nodata is None or np.isnan(nodata):
        assert np.isnan(meta['nodata']) and np.isnan(source_nodata)
    else:
        assert meta['nodata'] == source_nodata == nodata


def test_precheck_reads_source_only_for_training_mask():
    source, mask = MagicMock(), MagicMock()
    source.block_windows.return_value = [(0, 'window')]
    mask.read.return_value = np.ones((2, 2), dtype=np.uint8)
    assert not module._pre_check_valid_data(source, mask, {'nodata': -9999})
    source.read.assert_not_called()


@pytest.mark.parametrize('values,nodata,expected', [
    ([[0, -9999], [-9999, -9999]], -9999, True),
    ([[-9999, 20], [-9999, -9999]], -9999, False),
    ([[np.nan, 20], [np.nan, np.nan]], np.nan, False),
    ([[0, 20], [np.nan, np.nan]], np.nan, True),
])
def test_precheck_considers_only_class_two(values, nodata, expected):
    source, mask = MagicMock(), MagicMock()
    source.block_windows.return_value = [(0, 'window')]
    source.read.return_value = np.array(values, dtype=np.float32)
    mask.read.return_value = np.array([[2, 1], [2, 0]], dtype=np.uint8)
    assert module._pre_check_valid_data(source, mask, {'nodata': nodata}) is expected



def test_precheck_nan_is_missing_even_with_numeric_nodata():
    source, mask = MagicMock(), MagicMock()
    source.block_windows.return_value = [(0, 'window')]
    source.read.return_value = np.full((2, 2), np.nan, dtype=np.float32)
    mask.read.return_value = np.full((2, 2), 2, dtype=np.uint8)
    assert not module._pre_check_valid_data(source, mask, {'nodata': -9999})


# =====================================================================
# Actual Block Masking & Worker Output
# =====================================================================

@pytest.mark.parametrize('nodata', [-9999.0, np.nan, None], ids=['sentinel', 'nan', 'unset'])
def test_write_masked_data_real_values_and_metadata(victim, tmp_path, nodata):
    params = worker_params(victim, tmp_path, nodata=nodata)
    with rasterio.open(params[0]) as source, rasterio.open(params[3]) as mask:
        meta, _ = module._prepare_metadata(source)
        with aligned_mask(source, mask) as vrt:
            module._write_masked_data(source, vrt, params[2], meta)
        expected = np.where(mask.read(1) == 2, source.read(1), meta['nodata'])
        transform = source.transform
    with rasterio.open(params[2]) as src:
        np.testing.assert_allclose(src.read(1), expected, equal_nan=True)
        assert src.transform == transform and src.crs.to_epsg() == 32617
        assert src.profile['tiled'] and src.compression.value.lower() == 'lzw'
        assert src.read(1)[0, 0] == 0  # Zero is valid training data.



def test_integer_without_nodata_can_be_masked(victim, tmp_path):
    source_path = write_raster(tmp_path / 'integer.tif', [[0, 1], [2, 3]], nodata=None, dtype='int16')
    mask_path = write_raster(tmp_path / 'mask.tif', [[2, 1], [2, 1]], nodata=255, dtype='uint8')
    out = tmp_path / 'training.tif'
    with rasterio.open(source_path) as source, rasterio.open(mask_path) as mask:
        meta, _ = module._prepare_metadata(source)
        with aligned_mask(source, mask) as vrt:
            module._write_masked_data(source, vrt, str(out), meta)
    with rasterio.open(out) as src:
        np.testing.assert_allclose(src.read(1), [[0, np.nan], [2, np.nan]], equal_nan=True)


def test_worker_complete_local_and_cleanup(victim, tmp_path):
    params = worker_params(victim, tmp_path)
    module._process_training_raster(params)
    with rasterio.open(params[2]) as src:
        np.testing.assert_array_equal(src.read(1), [[0, -9999, 2, -9999],
            [4, -9999, -9999, -9999], [-9999, 9, -9999, 11], [-9999, -9999, 14, 15]])
    assert not list(victim.local_tmp_dir.iterdir())


def test_worker_aligns_coarser_mask_with_nearest_resampling(victim, tmp_path):
    params = worker_params(victim, tmp_path, values=np.arange(16).reshape(4, 4))
    write_raster(params[3], [[2, 1], [1, 2]], nodata=255, dtype='uint8',
        transform=from_origin(500000, 3400032, 16, 16))
    module._process_training_raster(params)
    expected = np.full((4, 4), -9999, dtype=np.float32)
    values = np.arange(16).reshape(4, 4)
    expected[:2, :2] = values[:2, :2]
    expected[2:, 2:] = values[2:, 2:]
    with rasterio.open(params[2]) as src:
        np.testing.assert_array_equal(src.read(1), expected)



def test_worker_overlapping_mask_in_different_crs(victim, tmp_path):
    params = worker_params(victim, tmp_path)
    geographic_bounds = transform_bounds('EPSG:32617', 'EPSG:4326', *params[1])
    write_raster(params[3], np.full((4, 4), 2), nodata=255, dtype='uint8',
        transform=from_bounds(*geographic_bounds, 4, 4), crs='EPSG:4326')
    params[1] = geographic_bounds
    module._process_training_raster(params)
    assert pathlib.Path(params[2]).exists()


@pytest.mark.parametrize('case', ['outside', 'no_training', 'all_nodata'])
def test_worker_skips_unusable_inputs(victim, tmp_path, case):
    params = worker_params(victim, tmp_path)
    if case == 'outside': params[1] = (0, 0, 10, 10)
    if case == 'no_training': write_raster(params[3], np.ones((4, 4)), nodata=255, dtype='uint8')
    if case == 'all_nodata': write_raster(params[0], np.full((4, 4), -9999))
    module._process_training_raster(params)
    assert not pathlib.Path(params[2]).exists()
    assert not list(victim.local_tmp_dir.iterdir())


def test_worker_existing_output_skips(victim, tmp_path):
    params = worker_params(victim, tmp_path)
    pathlib.Path(params[2]).write_bytes(b'keep')
    with patch(f'{TRAINING_PATH}.rasterio.open') as read:
        module._process_training_raster(params)
    read.assert_not_called()
    assert pathlib.Path(params[2]).read_bytes() == b'keep'


def test_worker_read_error_logs_and_cleans(victim, tmp_path, isolated_services):
    params = worker_params(victim, tmp_path)
    with patch(f'{TRAINING_PATH}.rasterio.open', side_effect=OSError('read failed')):
        assert module._process_training_raster(params) is None
    assert any('read failed' in c.args[0] for c in isolated_services.log.call_args_list)
    assert not list(victim.local_tmp_dir.iterdir())


@pytest.mark.parametrize('fails', [False, True])
def test_worker_cloud_stages_and_cleans(victim, tmp_path, isolated_services, fails):
    params = worker_params(victim, tmp_path)
    params[6] = True
    paths = []
    def put(source, destination):
        paths.append(source)
        with rasterio.open(source) as src:
            assert src.read(1)[0, 0] == 0 and src.read(1)[0, 1] == -9999
        assert destination == params[2]
        if fails: raise OSError('upload failed')
    isolated_services.s3.put.side_effect = put
    module._process_training_raster(params)
    assert len(paths) == 1 and not pathlib.Path(paths[0]).parent.exists()
    assert not list(victim.local_tmp_dir.iterdir())
    if fails: assert any('upload failed' in c.args[0] for c in isolated_services.log.call_args_list)

# =====================================================================
# Mask Bounds, Source Filtering & Task Payloads
# =====================================================================

def test_prepare_mask_bounds_local(victim):
    write_raster(victim.train_mask_path, np.full((4, 4), 2), nodata=255, dtype='uint8')
    path, bounds = victim.prepare_mask_bounds()
    assert path == str(victim.train_mask_path)
    assert tuple(bounds) == (500000, 3400000, 500032, 3400032)


def test_prepare_mask_bounds_cloud_virtual_path(victim):
    victim.is_aws = True
    victim.train_mask_path = UPath('s3://bucket/mask.tif')
    with patch(f'{TRAINING_PATH}.rasterio.open') as read:
        read.return_value.__enter__.return_value.bounds = (1, 2, 3, 4)
        path, bounds = victim.prepare_mask_bounds()
    read.assert_called_once_with('/vsis3/bucket/mask.tif')
    assert path == 's3://bucket/mask.tif' and bounds == (1, 2, 3, 4)


def test_collect_training_files_filter_and_size_order(victim):
    victim.prediction_out_dir.mkdir(parents=True)
    names = ['normal.tif', 'nested/other.tiff', 'filled_tifs/a.tif', 'FILLED_LIDAR/b.tif',
        'raw_mosaic.tif', 'combined_mosaic.tif', 'combined_lidar/mosaic.tif', 'notes.txt']
    for index, name in enumerate(names):
        path = victim.prediction_out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'x' * (index+1))
    result = victim.collect_training_files()
    assert [path.relative_to(victim.prediction_out_dir).as_posix() for path in result] == [
        'normal.tif', 'nested/other.tiff', 'combined_mosaic.tif', 'combined_lidar/mosaic.tif']


def test_execute_tasks_payload_and_gather(victim):
    files = [UPath('a.tif'), UPath('nested/b.tif')]
    with patch.object(victim, 'log_system_metrics', return_value='metrics'):
        victim.execute_tasks(files, (1, 2, 3, 4), 'mask.tif')
    function, batches = victim.client.map.call_args.args
    assert function is module._process_training_raster
    assert batches == [
        ['a.tif', (1, 2, 3, 4), str(victim.training_out_dir / 'a.tif'), 'mask.tif', 1, 2, False, str(victim.local_tmp_dir)],
        [str(files[1]), (1, 2, 3, 4), str(victim.training_out_dir / 'b.tif'), 'mask.tif', 2, 2, False, str(victim.local_tmp_dir)],
    ]
    victim.client.gather.assert_called_once_with(victim.client.map.return_value)


# =====================================================================
# System Metrics, Scratch Cleanup & Run Order
# =====================================================================

def test_log_system_metrics_disk_ram_and_temp_size(victim):
    (victim.local_tmp_dir / 'file.bin').write_bytes(b'x' * 1024**2)
    vm = SimpleNamespace(available=8 * 1024**3, total=16 * 1024**3, percent=50)
    with patch(f'{TRAINING_PATH}.shutil.disk_usage', return_value=(100 * 1024**3, 40 * 1024**3, 60 * 1024**3)), \
         patch.dict(sys.modules, {'psutil': SimpleNamespace(virtual_memory=lambda: vm)}):
        message = victim.log_system_metrics()
    assert 'Free: 8.0GB / 16.0GB (Used: 50%)' in message
    assert '60.0GB / 100.0GB' in message and '1.0MB' in message


def test_log_system_metrics_errors_are_reported(victim):
    with patch(f'{TRAINING_PATH}.shutil.disk_usage', side_effect=OSError('disk unavailable')):
        assert 'Error collecting system metrics: disk unavailable' in victim.log_system_metrics()


def test_cleanup_closes_cluster_and_removes_scratch(victim):
    (victim.local_tmp_dir / 'scratch.bin').write_bytes(b'test')
    victim.cleanup_resources()
    victim.close_dask.assert_called_once()
    assert not victim.local_tmp_dir.exists()


def test_cleanup_delete_failure_is_logged(victim):
    with patch(f'{TRAINING_PATH}.shutil.rmtree', side_effect=OSError('locked')):
        victim.cleanup_resources()
    victim.close_dask.assert_called_once()
    assert any('Failed to wipe' in c.args[0] for c in victim.write_message.call_args_list)


def test_run_region_order_and_empty_region_skip(victim):
    victim.param_lookup['eco_regions'].value = ['ER_1', 'ER_2']
    with ExitStack() as stack:
        methods = ['_resolve_paths', 'prepare_mask_bounds', 'collect_training_files', 'execute_tasks', 'cleanup_resources']
        mocks = {name: stack.enter_context(patch.object(victim, name)) for name in methods}
        mocks['prepare_mask_bounds'].return_value = ('mask.tif', (1, 2, 3, 4))
        mocks['collect_training_files'].side_effect = [[UPath('source.tif')], []]
        manager = MagicMock()
        for name, mock in mocks.items(): manager.attach_mock(mock, name)
        victim.run()
    assert manager.mock_calls == [call._resolve_paths('ER_1'), call.prepare_mask_bounds(),
        call.collect_training_files(), call.execute_tasks([UPath('source.tif')], (1, 2, 3, 4), 'mask.tif'),
        call._resolve_paths('ER_2'), call.prepare_mask_bounds(), call.collect_training_files(), call.cleanup_resources()]
    victim.setup_dask.assert_called_once_with('local', n_workers=2, threads_per_worker=1, memory_limit='12GB')


@pytest.mark.parametrize('stage', ['setup', 'bounds', 'gather'])
def test_run_failure_still_cleans(victim, stage):
    if stage == 'setup': victim.setup_dask.side_effect = RuntimeError('test failure')
    if stage == 'gather': victim.client.gather.side_effect = RuntimeError('test failure')
    with patch.object(victim, '_resolve_paths'), \
         patch.object(victim, 'prepare_mask_bounds', return_value=('mask.tif', (1, 2, 3, 4))) as bounds, \
         patch.object(victim, 'collect_training_files', return_value=[UPath('source.tif')]), \
         patch.object(victim, 'cleanup_resources') as cleanup:
        if stage == 'bounds': bounds.side_effect = RuntimeError('test failure')
        with pytest.raises(RuntimeError, match='test failure'):
            victim.run()
    cleanup.assert_called_once()
