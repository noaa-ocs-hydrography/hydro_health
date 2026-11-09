"""Prediction raster tests following the BlueTopo fixture/function pattern."""
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.PredictionRastersEngine import (
    PredictionRastersEngine, _apply_tsm_smoothing, _warp_to_cutline,
    _process_prediction_raster,
)

PREDICTION_PATH = 'hydro_health.engines.PredictionRastersEngine'


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    lookup = {
        ('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
        ('MASK', 'PREDICTION_MASK_PQ'): 'masks/prediction.parquet',
        ('MODEL', 'PREPROCESSED_DIR'): 'preprocessed',
        ('MODEL', 'PREDICTION_OUTPUT_DIR'): 'prediction',
        ('MODEL', 'TILED_LIDAR_PROC'): 'lidar/uncombined',
        ('MODEL', 'TILED_LIDAR_DIR'): 'lidar/source',
        ('TERRAIN', 'FILLED_DIR'): 'terrain/custom_filled',
        ('PREPROCESSED', 'BLUETOPO'): 'bluetopo',
        ('PREPROCESSED', 'HURRICANE'): 'hurricane',
        ('PREPROCESSED', 'SEDIMENT'): 'sediment',
        ('PREPROCESSED', 'TSM'): 'tsm',
    }
    with patch(f'{PREDICTION_PATH}.Engine.__init__', return_value=None), \
         patch(f'{PREDICTION_PATH}.Engine.write_message_dask', create=True) as worker_log, \
         patch(f'{PREDICTION_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{PREDICTION_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{PREDICTION_PATH}.Path.home', return_value=tmp_path / 'home'), \
         patch(f'{PREDICTION_PATH}.get_config_item', side_effect=lambda *args: lookup[args]), \
         patch(f'{PREDICTION_PATH}.s3fs.S3FileSystem') as s3, \
         patch(f'{PREDICTION_PATH}.gdal.Open') as gdal_open, \
         patch(f'{PREDICTION_PATH}.gdal.Warp') as warp, \
         patch(f'{PREDICTION_PATH}.gdal.VSICurlClearCache', create=True) as cache:
        yield SimpleNamespace(lookup=lookup, log=worker_log, s3=s3.return_value,
                              gdal_open=gdal_open, warp=warp, cache=cache)


@pytest.fixture
def victim():
    engine = PredictionRastersEngine({'env': 'local', 'eco_regions': ['ER_3']})
    engine.target_crs = 'EPSG:32617'
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.cleanup_resources = MagicMock()
    engine.log_system_metrics = MagicMock(return_value='test metrics')
    engine.client = MagicMock()
    engine._resolve_paths('ER_3')
    return engine


def write_raster(path, data, pixel_size=2000, nodata=-9999.0, crs='EPSG:32617'):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    with rasterio.open(path, 'w', driver='GTiff', count=1, width=data.shape[1], height=data.shape[0],
        dtype='float32', nodata=nodata, crs=crs,
        transform=from_origin(500000, 3400000, pixel_size, pixel_size)) as dst:
        dst.write(data, 1)
    return path


def configure_pixel_size(services, pixel_size=2000):
    services.gdal_open.return_value.GetGeoTransform.return_value = (500000, pixel_size, 0, 3400000, 0, -pixel_size)


def worker_params(victim, tmp_path, name='bathy.tif'):
    source = write_raster(tmp_path / name, [[1, 2], [3, 4]], pixel_size=100)
    return [str(source), (500000, 3399800, 500200, 3400000), str(tmp_path / 'result.tif'),
            str(tmp_path / 'cutline.gpkg'), False, victim.target_crs, 8.0, str(victim.local_tmp_dir)]


# =====================================================================
# Constructor Normalization & Paths
# =====================================================================

@pytest.mark.parametrize('env,is_aws', [('local', False), ('remote', True), ('aws', True)])
def test_init_environment_and_scratch(tmp_path, env, is_aws):
    engine = PredictionRastersEngine({'env': env}, output_prefix='low_res')
    assert engine.env == env and engine.is_aws is is_aws
    assert engine.target_res == 8.0 and engine.output_prefix == 'low_res'
    assert engine.local_tmp_dir == tmp_path / 'home/hydro_health_local_tmp/prediction_tmp'
    assert engine.local_tmp_dir.is_dir()


def test_init_flattens_parameter_wrappers():
    engine = PredictionRastersEngine({
        'env': SimpleNamespace(valueAsText='aws', value='local'),
        'eco_regions': SimpleNamespace(valueAsText=None, value=['ER_1', 'ER_2']),
        'plain': 12, 'gdal_env_vars': {'GDAL_CACHEMAX': '64'},
    })
    assert engine.param_lookup['env'] == 'aws'
    assert engine.param_lookup['eco_regions'] == ['ER_1', 'ER_2']
    assert engine.param_lookup['plain'] == 12
    assert engine.gdal_env_vars == {'GDAL_CACHEMAX': '64'}


def test_init_none_parameters():
    engine = PredictionRastersEngine(None)
    assert engine.param_lookup == {} and engine.env == 'local'


@pytest.mark.parametrize('env', ['local', 'remote', 'aws'])
@pytest.mark.parametrize('prefix', [False, 'low_res'])
def test_resolve_paths(tmp_path, env, prefix):
    engine = PredictionRastersEngine({'env': env}, prefix)
    engine.write_message = MagicMock()
    engine._resolve_paths('ER_2')
    base = tmp_path / 'outputs'
    local_base = base / prefix if prefix else base
    assert engine.outputs_dir == local_base / 'ER_2'
    root = UPath('s3://test-bucket/ER_2') if engine.is_aws else UPath(engine.outputs_dir)
    for attr, suffix in [('mask_prediction_pq', 'masks/prediction.parquet'),
        ('preprocessed_dir', 'preprocessed'), ('prediction_out_dir', 'prediction'),
        ('uncombined_lidar_dir', 'lidar/uncombined')]:
        assert getattr(engine, attr) == root / suffix
    assert set(engine.preprocessed_subdirs) == {'bluetopo', 'hurricane', 'lidar', 'sediment', 'tsm'}
    assert engine.preprocessed_subdirs['lidar'] == root / 'lidar/source'
    assert engine.filled_folder_name == 'custom_filled'
    if not engine.is_aws:
        assert engine.prediction_out_dir.is_dir() and engine.uncombined_lidar_dir.is_dir()


def test_resolve_paths_default_filled_name(victim, isolated_services):
    isolated_services.lookup[('TERRAIN', 'FILLED_DIR')] = None
    victim._resolve_paths('ER_1')
    assert victim.filled_folder_name == 'filled_tifs'


# =====================================================================
# Spatial Mask Repair & Cutline Output
# =====================================================================

@pytest.mark.parametrize('source_crs', ['EPSG:32617', 'EPSG:4326', None])
def test_prepare_spatial_masks_crs_and_cutline(victim, source_crs):
    frame = gpd.GeoDataFrame(geometry=[box(500000, 3399800, 500200, 3400000)], crs='EPSG:32617')
    if source_crs == 'EPSG:4326':
        frame = frame.to_crs(source_crs)
    elif source_crs is None:
        frame = frame.set_crs(None, allow_override=True)
    with patch(f'{PREDICTION_PATH}.gpd.read_parquet', return_value=frame) as read:
        bounds, path = victim.prepare_spatial_masks('ER_3')
    read.assert_called_once_with(str(victim.mask_prediction_pq))
    np.testing.assert_allclose(bounds, [500000, 3399800, 500200, 3400000], atol=0.001)
    saved = gpd.read_file(path)
    assert saved.crs.to_epsg() == 32617 and saved.geometry.is_valid.all()
    assert pathlib.Path(path).name == 'pred_cutline_ER_3.gpkg'


def test_prepare_spatial_masks_repairs_invalid_geometry(victim):
    bowtie = Polygon([(500000, 3400000), (500200, 3399800),
                       (500000, 3399800), (500200, 3400000), (500000, 3400000)])
    frame = gpd.GeoDataFrame(geometry=[bowtie, None, Polygon()], crs=victim.target_crs)
    with patch(f'{PREDICTION_PATH}.gpd.read_parquet', return_value=frame):
        _, path = victim.prepare_spatial_masks('ER_3')
    saved = gpd.read_file(path)
    assert len(saved) == 1 and saved.geometry.is_valid.all()


def test_prepare_spatial_masks_empty_after_repair(victim):
    frame = gpd.GeoDataFrame(geometry=[None, Polygon()], crs=victim.target_crs)
    with patch(f'{PREDICTION_PATH}.gpd.read_parquet', return_value=frame):
        with pytest.raises(ValueError, match='empty after validation'):
            victim.prepare_spatial_masks('ER_3')


# =====================================================================
# Input Discovery & Exclusion Rules
# =====================================================================

def test_get_valid_source_files_filters_exact_folder_components(victim, tmp_path):
    root = tmp_path / 'sources'
    names = ['a.tif', 'nested/B.TIFF', 'notes.txt', 'sand_mud_mask.tif',
        'custom_filled/x.tif', 'filled_tifs/y.tif', 'filled_lidar/z.tif',
        'almost_filled_tifs/keep.tif']
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    victim.preprocessed_subdirs = {'tsm': UPath(root), 'missing': UPath(tmp_path / 'missing')}
    result = victim.get_valid_source_files()
    assert {path.relative_to(root).as_posix() for path in result} == {'a.tif', 'nested/B.TIFF', 'almost_filled_tifs/keep.tif'}


def test_get_valid_source_files_cloud_scan(victim):
    victim.is_aws = True
    directory = MagicMock()
    directory.exists.return_value = False
    directory.rglob.return_value = [pathlib.Path('remote/source.tif')]
    victim.preprocessed_subdirs = {'bluetopo': directory}
    assert victim.get_valid_source_files() == [pathlib.Path('remote/source.tif')]
    directory.rglob.assert_called_once_with('*')


# =====================================================================
# Real TSM Smoothing Arithmetic, Metadata & Block Boundaries
# =====================================================================

def test_smoothing_valid_values_edge_normalization(tmp_path, isolated_services):
    path = write_raster(tmp_path / 'tsm.tif', [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    configure_pixel_size(isolated_services)
    _apply_tsm_smoothing(str(path), 'source_tsm.tif', {})
    with rasterio.open(path) as src:
        np.testing.assert_allclose(src.read(1), [[3, 3.5, 4], [4.5, 5, 5.5], [6, 6.5, 7]], rtol=1e-6)
        assert src.dtypes == ('float32',) and src.crs.to_epsg() == 32617
        assert src.nodata == -9999 and src.compression.value.lower() == 'lzw'
        assert src.profile['tiled'] is True
    assert not path.with_name('tsm_smoothed.tif').exists()


@pytest.mark.parametrize(
    'nodata', [-9999.0, np.nan], ids=['sentinel', 'nan']
)
def test_smoothing_fills_nodata_from_valid_neighbors(
    tmp_path, isolated_services, nodata
):
    path = write_raster(
        tmp_path / 'tsm.tif',
        [[1, 2, 3], [4, nodata, 6], [7, 8, 9]],
        nodata=nodata,
    )
    configure_pixel_size(isolated_services)

    _apply_tsm_smoothing(str(path), 'source_tsm.tif', {})

    with rasterio.open(path) as src:
        result = src.read(1)

    # The eight valid neighbors average to 5; nodata contributes no weight.
    assert result[1, 1] == pytest.approx(5.0)


def test_smoothing_all_nodata_stays_nodata(tmp_path, isolated_services):
    path = write_raster(tmp_path / 'tsm.tif', np.full((3, 3), -9999))
    configure_pixel_size(isolated_services)
    _apply_tsm_smoothing(str(path), 'source.tif', {})
    with rasterio.open(path) as src:
        np.testing.assert_array_equal(src.read(1), np.full((3, 3), -9999))


def test_smoothing_reads_halo_across_1024_pixel_chunk(tmp_path, isolated_services):
    data = np.tile(np.arange(1030, dtype=np.float32) ** 2, (3, 1))
    path = write_raster(tmp_path / 'tsm.tif', data)
    configure_pixel_size(isolated_services)
    _apply_tsm_smoothing(str(path), 'source.tif', {})
    with rasterio.open(path) as src:
        result = src.read(1)
    for col in [0, 1022, 1023, 1024, 1025, 1029]:
        expected = data[:, max(0, col-1):min(1030, col+2)].mean()
        assert result[1, col] == pytest.approx(float(expected), rel=1e-6)


# =====================================================================
# GDAL Warp Options, Upload Routing & Error Propagation
# =====================================================================

@pytest.mark.parametrize('name,crop', [('tsm_mean.tif', True), ('grain_sed.tif', True),
    ('hurr_mean.tif', True), ('bathy.tif', False)])
def test_warp_local_options(tmp_path, isolated_services, name, crop):
    destination = tmp_path / 'result.tif'
    _warp_to_cutline(name, str(destination), 'cutline.gpkg', str(tmp_path), False, 'EPSG:32617', 8, -9999, False)
    args, opts = isolated_services.warp.call_args
    assert args == (str(destination), name)
    assert opts['cutlineDSName'] == 'cutline.gpkg' and opts['cropToCutline'] is crop
    assert opts['dstSRS'] == 'EPSG:32617' and opts['xRes'] == opts['yRes'] == 8
    assert opts['resampleAlg'] == 'bilinear' and opts['srcNodata'] == -9999
    assert opts['multithread'] is False and opts['warpMemoryLimit'] == 1024
    assert 'NUM_THREADS=1' in opts['creationOptions'] and 'COMPRESS=LZW' in opts['creationOptions']
    isolated_services.cache.assert_called_once()


def test_warp_no_source_nodata_omits_option(tmp_path, isolated_services):
    _warp_to_cutline('source.tif', 'out.tif', 'cut.gpkg', str(tmp_path), False, 'EPSG:32617', 8, None, False)
    assert 'srcNodata' not in isolated_services.warp.call_args.kwargs


def test_warp_cloud_upload_and_smoothing_order(tmp_path, isolated_services):
    def warp(destination, source, **opts):
        pathlib.Path(destination).write_bytes(b'warped')
        return MagicMock()
    isolated_services.warp.side_effect = warp
    events = []
    def smooth(destination, source, opts):
        assert pathlib.Path(destination).exists()
        events.append('smooth')
    def upload(source, destination):
        assert pathlib.Path(source).read_bytes() == b'warped'
        events.append('upload')
    isolated_services.s3.put.side_effect = upload
    with patch(f'{PREDICTION_PATH}._apply_tsm_smoothing', side_effect=smooth):
        _warp_to_cutline('s3://bucket/tsm.tif', 's3://bucket/out.tif', 'cut.gpkg', str(tmp_path), True, 'EPSG:32617', 8, np.nan, True)
    assert isolated_services.warp.call_args.args[1] == '/vsis3/bucket/tsm.tif'
    assert events == ['smooth', 'upload']
    assert not (tmp_path / 'warp_tmp.tif').exists()
    isolated_services.cache.assert_called_once()


@pytest.mark.parametrize('failure', ['none', 'warp', 'upload'])
def test_warp_failure_propagates_and_clears_cache(tmp_path, isolated_services, failure):
    if failure == 'none':
        isolated_services.warp.return_value = None
    elif failure == 'warp':
        isolated_services.warp.side_effect = RuntimeError('test warp failure')
    else:
        isolated_services.s3.put.side_effect = RuntimeError('test upload failure')
    with pytest.raises(RuntimeError):
        _warp_to_cutline('source.tif', 's3://bucket/out.tif', 'cut.gpkg', str(tmp_path), True, 'EPSG:32617', 8, None, False)
    isolated_services.cache.assert_called_once()
    assert any('[ERROR]' in c.args[0] for c in isolated_services.log.call_args_list)

# =====================================================================
# Worker Dispatch, Intersection Checks & Task Cleanup
# =====================================================================

def test_worker_existing_output_skips(victim, tmp_path):
    params = worker_params(victim, tmp_path)
    pathlib.Path(params[2]).touch()
    with patch(f'{PREDICTION_PATH}._warp_to_cutline') as warp, \
         patch(f'{PREDICTION_PATH}.rasterio.open') as read:
        _process_prediction_raster(params)
    read.assert_not_called()
    warp.assert_not_called()


@pytest.mark.parametrize('name,is_tsm', [('bathy.tif', False), ('TSM_mean.tif', True),
    ('hurr_strength_mean.tif', True), ('hurr_count.tif', False)])
def test_worker_dispatches_and_removes_task_directory(victim, tmp_path, name, is_tsm):
    params = worker_params(victim, tmp_path, name)
    tasks = []
    def warp(**kwargs):
        tasks.append(kwargs['task_tmp_dir'])
        assert pathlib.Path(kwargs['task_tmp_dir']).is_dir()
    with patch(f'{PREDICTION_PATH}._warp_to_cutline', side_effect=warp) as mock:
        _process_prediction_raster(params)
    mock.assert_called_once()
    kwargs = mock.call_args.kwargs
    assert kwargs['src_path'] == params[0] and kwargs['dst_path'] == params[2]
    assert kwargs['target_crs'] == victim.target_crs and kwargs['target_res'] == 8
    assert kwargs['src_nodata'] == -9999 and kwargs['is_tsm'] is is_tsm
    assert not pathlib.Path(tasks[0]).exists()


def test_worker_outside_mask_skips(victim, tmp_path):
    params = worker_params(victim, tmp_path)
    params[1] = (0, 0, 10, 10)
    with patch(f'{PREDICTION_PATH}._warp_to_cutline') as warp:
        _process_prediction_raster(params)
    warp.assert_not_called()


def test_worker_corrupt_source_logs_and_returns(victim, tmp_path, isolated_services):
    params = worker_params(victim, tmp_path)
    with patch(f'{PREDICTION_PATH}.rasterio.open', side_effect=OSError('corrupt file')), \
         patch(f'{PREDICTION_PATH}._warp_to_cutline') as warp:
        _process_prediction_raster(params)
    warp.assert_not_called()
    assert any('Could not open' in c.args[0] for c in isolated_services.log.call_args_list)


def test_worker_transforms_bounds_before_intersection(victim, tmp_path):
    params = worker_params(victim, tmp_path)
    write_raster(params[0], [[1, 2], [3, 4]], pixel_size=0.01, crs='EPSG:4326')
    with patch(f'{PREDICTION_PATH}.transform_bounds', return_value=params[1]) as transform, \
         patch(f'{PREDICTION_PATH}._warp_to_cutline') as warp:
        _process_prediction_raster(params)
    transform.assert_called_once()
    warp.assert_called_once()


def test_worker_failed_bounds_transform_still_attempts_warp(victim, tmp_path, isolated_services):
    params = worker_params(victim, tmp_path)
    write_raster(params[0], [[1]], crs='EPSG:4326')
    with patch(f'{PREDICTION_PATH}.transform_bounds', side_effect=ValueError('bad bounds')), \
         patch(f'{PREDICTION_PATH}._warp_to_cutline') as warp:
        _process_prediction_raster(params)
    warp.assert_called_once()
    assert any('Bypassing intersection check' in c.args[0] for c in isolated_services.log.call_args_list)


def test_worker_cloud_source_uses_vsis3(victim, tmp_path):
    params = worker_params(victim, tmp_path)
    params[0], params[4] = 's3://bucket/source.tif', True
    source = MagicMock()
    source.nodata = -9999
    source.crs = rasterio.crs.CRS.from_epsg(32617)
    source.bounds = params[1]
    with patch(f'{PREDICTION_PATH}.rasterio.open') as read, \
         patch(f'{PREDICTION_PATH}._warp_to_cutline') as warp:
        read.return_value.__enter__.return_value = source
        _process_prediction_raster(params)
    read.assert_called_once_with('/vsis3/bucket/source.tif')
    warp.assert_called_once()


def test_worker_warp_failure_logs_returns_and_cleans(victim, tmp_path, isolated_services):
    params = worker_params(victim, tmp_path)
    task_directories = []
    def fail(**kwargs):
        task_directories.append(kwargs['task_tmp_dir'])
        raise RuntimeError('task failed')
    with patch(f'{PREDICTION_PATH}._warp_to_cutline', side_effect=fail), \
         patch(f'{PREDICTION_PATH}.gc.collect') as collect:
        assert _process_prediction_raster(params) is None
    assert not pathlib.Path(task_directories[0]).exists()
    collect.assert_called_once()
    assert any('Unexpected failure' in c.args[0] for c in isolated_services.log.call_args_list)


# =====================================================================
# Dask Orchestration, Output Routing & Finally Cleanup
# =====================================================================

@pytest.mark.parametrize('regions', [['ER_1', 'ER_2'], 'ER_3'])
def test_run_submits_tasks_and_routes_mosaics(victim, tmp_path, regions):
    victim.param_lookup['eco_regions'] = regions
    bounds = np.array([500000, 3399800, 500200, 3400000])
    files = [UPath(tmp_path / 'bathy.tif'), UPath(tmp_path / 'LIDAR_MOSAIC.tif')]
    batches = []
    victim.client.map.side_effect = lambda function, params: batches.append(params) or ['future']
    with patch.object(victim, 'prepare_spatial_masks', return_value=(bounds, 'cutline.gpkg')) as masks, \
         patch.object(victim, 'get_valid_source_files', return_value=files):
        victim.run()
    count = len(regions) if isinstance(regions, list) else 1
    assert len(batches) == count and victim.client.gather.call_count == count
    victim.setup_dask.assert_called_once_with('local', n_workers=4, threads_per_worker=1, memory_limit='6GB')
    for batch, region in zip(batches, regions if isinstance(regions, list) else [regions]):
        assert len(batch) == 2
        assert batch[0][2] == str(UPath(victim.outputs_dir.parent / region / 'prediction/bathy.tif'))
        assert batch[1][2] == str(UPath(victim.outputs_dir.parent / region / 'lidar/uncombined/LIDAR_MOSAIC.tif'))
        np.testing.assert_array_equal(batch[0][1], bounds)
        assert batch[0][3:] == ['cutline.gpkg', False, 'EPSG:32617', 8.0, str(victim.local_tmp_dir)]
    assert all(c.args[0] is _process_prediction_raster for c in victim.client.map.call_args_list)
    assert masks.call_count == count
    victim.cleanup_resources.assert_called_once()


def test_run_no_files_no_map(victim):
    with patch.object(victim, 'prepare_spatial_masks', return_value=([0, 0, 1, 1], 'cutline.gpkg')), \
         patch.object(victim, 'get_valid_source_files', return_value=[]):
        victim.run()
    victim.client.map.assert_not_called()
    victim.cleanup_resources.assert_called_once()


@pytest.mark.parametrize('failure_stage', ['setup', 'mask', 'gather'])
def test_run_failure_still_cleans_resources(victim, tmp_path, failure_stage):
    if failure_stage == 'setup':
        victim.setup_dask.side_effect = RuntimeError('test failure')
    elif failure_stage == 'gather':
        victim.client.gather.side_effect = RuntimeError('test failure')
    with patch.object(victim, 'prepare_spatial_masks', return_value=([0, 0, 1, 1], 'cut.gpkg')) as masks, \
         patch.object(victim, 'get_valid_source_files', return_value=[UPath(tmp_path / 'source.tif')]):
        if failure_stage == 'mask':
            masks.side_effect = RuntimeError('test failure')
        with pytest.raises(RuntimeError, match='test failure'):
            victim.run()
    victim.cleanup_resources.assert_called_once()
