"""Terrain engine tests using the shared BlueTopo pytest structure."""
import os
import pathlib
import sys
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

# Prevent interactive plotting; report orchestration tests mock the plotting call.
import matplotlib
matplotlib.use('Agg')
import importlib
module = importlib.import_module('hydro_health.engines.TerrainProductsEngine')
from hydro_health.engines.TerrainProductsEngine import TerrainProductsEngine

TERRAIN_PATH = 'hydro_health.engines.TerrainProductsEngine'
RADII = {'fine': (0, 8), 'broad': (8, 16)}


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    lookup = {('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
        ('TERRAIN', 'COMBINED_LIDAR_DIR'): 'lidar/combined',
        ('TERRAIN', 'OUTPUTS'): 'terrain',
        ('TERRAIN', 'DICTIONARIES_DIR'): 'dictionaries',
        ('MODEL', 'PREDICTION_OUTPUT_DIR'): 'prediction'}
    with patch(f'{TERRAIN_PATH}.Engine.__init__', return_value=None), \
         patch(f'{TERRAIN_PATH}.Engine.write_message_dask', create=True) as log, \
         patch(f'{TERRAIN_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{TERRAIN_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{TERRAIN_PATH}.Path.home', return_value=tmp_path / 'home'), \
         patch(f'{TERRAIN_PATH}.get_config_item', side_effect=lambda *args: lookup[args]):
        yield SimpleNamespace(log=log, lookup=lookup)


@pytest.fixture
def victim():
    engine = TerrainProductsEngine({'env': 'local', 'eco_regions': SimpleNamespace(value=['ER_3'])})
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.cleanup_resources = MagicMock()
    engine.log_system_metrics = MagicMock(return_value='test metrics')
    engine.client = MagicMock()
    engine._resolve_paths('ER_3')
    return engine


def write_raster(path, data, nodata=-9999, crs='EPSG:32617', transform=None, tag=None):
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
        count=1, dtype='float32', crs=crs, nodata=nodata,
        transform=transform or from_origin(500000, 3400000, 8, 8)) as dst:
        dst.write(data, 1)
        if tag:
            dst.update_tags(hydro_health_product_version=tag)
    return path


def profile_for(path):
    with rasterio.open(path) as src:
        return src.profile.copy()


def reference_bpi(data, inner, outer):
    """Independent small-array annulus calculation, without convolution."""
    result = np.full(data.shape, np.nan, dtype=np.float32)
    for row, col in np.ndindex(data.shape):
        if np.isnan(data[row, col]):
            continue
        neighbors = []
        for y, x in np.ndindex(data.shape):
            distance2 = (row-y)**2 + (col-x)**2
            if inner**2 < distance2 <= outer**2 and np.isfinite(data[y, x]):
                neighbors.append(data[y, x])
        if neighbors:
            result[row, col] = data[row, col] - np.mean(neighbors)
    return result


def reference_tci(data):
    """Independent population standard deviation of valid neighbor differences."""
    result = np.full(data.shape, np.nan, dtype=np.float32)
    for row, col in np.ndindex(data.shape):
        if np.isnan(data[row, col]):
            continue
        differences = []
        for y in range(max(0, row-1), min(data.shape[0], row+2)):
            for x in range(max(0, col-1), min(data.shape[1], col+2)):
                if (y, x) != (row, col) and np.isfinite(data[y, x]):
                    differences.append(data[y, x]-data[row, col])
        if len(differences) > 1:
            result[row, col] = np.std(differences)
    return result


# =====================================================================
# Initialization, Paths & Bathymetry Discovery
# =====================================================================

@pytest.mark.parametrize('env,cloud', [('local', False), ('remote', True), ('aws', True)])
def test_init_and_scratch(tmp_path, env, cloud):
    engine = TerrainProductsEngine({'env': env}, 'low_res')
    assert engine.is_aws is cloud and engine.output_prefix == 'low_res'
    assert engine.local_tmp_dir == tmp_path / 'home/hydro_health_local_tmp/terrain_tmp'
    assert engine.local_tmp_dir.is_dir()


@pytest.mark.parametrize('env', ['local', 'remote', 'aws'])
@pytest.mark.parametrize('prefix', [False, 'low_res'])
def test_resolve_paths(tmp_path, env, prefix):
    engine = TerrainProductsEngine({'env': env}, prefix)
    engine.write_message = MagicMock()
    engine._resolve_paths('ER_2')
    base = tmp_path / 'outputs'
    base = base / prefix if prefix else base
    assert engine.outputs_dir == base / 'ER_2'
    root = UPath('s3://test-bucket/ER_2') if engine.is_aws else UPath(base / 'ER_2')
    for attr, suffix in [('combined_bathy_dir', 'lidar/combined'), ('terrain_outputs_dir', 'terrain'),
        ('dictionary_dir', 'dictionaries'), ('prediction_output_dir', 'prediction')]:
        assert getattr(engine, attr) == root / suffix
        if not engine.is_aws:
            assert getattr(engine, attr).is_dir()


def test_get_files_filters_and_deduplicates(victim):
    names = ['nested/combined_2004_BH4S556X.tif', 'combined_2006.TIFF', 'hurricane.tif',
        'combined_ISS.tif', 'BlueTopo_BH4S556X_20200101.tiff',
        'BlueTopo_BH4S556X_20200101_slope.tif', 'notes.txt']
    for name in names:
        path = victim.combined_bathy_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (victim.prediction_output_dir / 'BlueTopo_BH4S2574_20200101.TIF').touch()
    nested = victim.prediction_output_dir / 'nested'
    nested.mkdir()
    (nested / 'combined_2005.tif').touch()
    result = {UPath(p).name for p in victim._get_files_to_process()}
    assert result == {'combined_2004_BH4S556X.tif', 'combined_2006.TIFF',
        'BlueTopo_BH4S556X_20200101.tiff', 'BlueTopo_BH4S2574_20200101.TIF'}


@pytest.mark.parametrize('path,expected', [('s3://bucket/a.tif', True), ('local.tif', False)])
def test_is_s3_path(path, expected):
    assert module._is_s3_path(path) is expected


@pytest.mark.parametrize('protocol', [('s3', 's3a'), ['s3'], ('file', 'local')])
def test_is_s3_path_protocol_variants(protocol):
    with patch(f'{TERRAIN_PATH}.UPath', return_value=SimpleNamespace(protocol=protocol)):
        assert module._is_s3_path('unused') is ('s3' in protocol)


# =====================================================================
# Raster Validation, Grid Restrictions & Product Versions
# =====================================================================

@pytest.mark.parametrize('kind,reason', [('missing', 'missing'), ('empty', 'empty file'), ('corrupt', None)])
def test_validate_raster_bad_files(tmp_path, kind, reason):
    path = tmp_path / 'bad.tif'
    if kind == 'empty': path.touch()
    if kind == 'corrupt': path.write_text('not a TIFF')
    valid, message = module._validate_raster(path)
    assert not valid and message
    if reason: assert reason in message


@pytest.mark.parametrize('full_read', [False, True])
def test_validate_raster_valid_and_grid(tmp_path, full_read):
    path = write_raster(tmp_path / 'source.tif', np.zeros((4, 5)))
    with rasterio.open(path) as src:
        expected = dict(expected_shape=src.shape, expected_crs=src.crs, expected_transform=src.transform)
    assert module._validate_raster(path, full_read=full_read, **expected) == (True, '')


@pytest.mark.parametrize('kwargs,reason', [({'expected_shape': (2, 2)}, 'shape'),
    ({'expected_crs': rasterio.crs.CRS.from_epsg(6350)}, 'CRS'),
    ({'expected_transform': from_origin(0, 0, 8, 8)}, 'transform mismatch')])
def test_validate_raster_grid_mismatch(tmp_path, kwargs, reason):
    path = write_raster(tmp_path / 'source.tif', np.zeros((4, 5)))
    valid, message = module._validate_raster(path, **kwargs)
    assert not valid and reason in message


@pytest.mark.parametrize('tag,valid', [(None, False), ('old_version', False), ('d8_pointer_v1', True)])
def test_product_version_rejects_legacy(tmp_path, tag, valid):
    path = write_raster(tmp_path / 'flow.tif', [[1, 2], [3, 4]], tag=tag)
    assert module._validate_product_raster(path, '_flowdir.tif')[0] is valid
    assert module._validate_product_raster(path, '_curv_plan.tif') == (True, '')


@pytest.mark.parametrize('crs,transform,valid', [
    ('EPSG:32617', from_origin(500000, 3400000, 8, 8), True),
    ('EPSG:4326', from_origin(-80, 30, .01, .01), False),
    (None, from_origin(500000, 3400000, 8, 8), False),
    ('EPSG:2263', from_origin(1000000, 200000, 8, 8), False),
    ('EPSG:32617', from_origin(500000, 3400000, 8, 16), False),
])
def test_metric_cell_size(tmp_path, crs, transform, valid):
    path = write_raster(tmp_path / 'grid.tif', np.ones((3, 3)), crs=crs, transform=transform)
    with rasterio.open(path) as src:
        if valid:
            assert module._metric_cell_size(src) == 8
        else:
            with pytest.raises(ValueError): module._metric_cell_size(src)


# =====================================================================
# Safe Publication, Materialization & Disk-Backed Writers
# =====================================================================

def test_publish_local_and_materialize_round_trip(tmp_path):
    source = write_raster(tmp_path / 'source.tif', [[1, np.nan], [3, 4]], nodata=np.nan)
    output = tmp_path / 'new/output.tif'
    module._publish_local_raster(str(source), str(output))
    assert output.read_bytes() == source.read_bytes()
    assert not (output.parent / '.hh.tmp').exists()
    local = tmp_path / 'materialized.tif'
    assert module._materialize_raster(source, str(local)) == str(local)
    assert local.read_bytes() == source.read_bytes()


def test_publish_invalid_preserves_destination(tmp_path):
    source = tmp_path / 'bad.tif'
    source.write_bytes(b'broken')
    output = tmp_path / 'existing.tif'
    output.write_bytes(b'keep')
    with pytest.raises(RuntimeError, match='Refusing to publish'):
        module._publish_local_raster(str(source), str(output))
    assert output.read_bytes() == b'keep'


def test_publish_replace_failure_cleans_partial(tmp_path):
    source = write_raster(tmp_path / 'source.tif', [[1, 2], [3, 4]])
    output = tmp_path / 'existing.tif'
    output.write_bytes(b'keep')
    with patch(f'{TERRAIN_PATH}.os.replace', side_effect=OSError('replace failed')):
        with pytest.raises(OSError): module._publish_local_raster(str(source), str(output))
    assert output.read_bytes() == b'keep' and not (tmp_path / '.hh.tmp').exists()


@pytest.mark.parametrize('writer', ['memmap', 'dask'])
def test_disk_backed_writer_values_tags_and_cleanup(tmp_path, writer):
    source = write_raster(tmp_path / 'source.tif', np.ones((6, 6)))
    profile = profile_for(source)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    output = tmp_path / 'saved.tif'
    values = np.arange(36, dtype=np.float32).reshape(6, 6)
    if writer == 'dask':
        module._save_dask_to_raster(da.from_array(values, chunks=(3, 3)), str(output), profile,
            str(scratch), product_suffix='_tci.tif')
    else:
        mmap = np.memmap(tmp_path / 'data.dat', dtype='float32', mode='w+', shape=values.shape)
        mmap[:] = values
        try:
            module._save_memmap_to_raster(mmap, str(output), profile, str(scratch), product_suffix='_tci.tif')
        finally:
            module._close_memmap(mmap)
    with rasterio.open(output) as src:
        np.testing.assert_array_equal(src.read(1), values)
        assert src.tags()['hydro_health_product_version'] == module.PRODUCT_VERSIONS['_tci.tif']
    assert not list(scratch.iterdir())


# =====================================================================
# Independent Numerical References: Slope, BPI & TCI
# =====================================================================

@pytest.mark.parametrize('gradient', [0, 0.5, 1])
def test_slope_plane_matches_analytic_angle(gradient):
    data = np.tile(np.arange(8, dtype=np.float32) * 8 * gradient, (8, 1)) - 100
    result = module._calculate_slope(data, 8)
    np.testing.assert_allclose(result, np.degrees(np.arctan(gradient)), atol=1e-5)


def test_slope_dask_halo_and_nodata_edges():
    data = np.tile(np.arange(12, dtype=np.float32) * 8, (12, 1)) - 200
    result = module._calculate_slope_dask(da.from_array(data, chunks=(6, 6)), 8).compute(scheduler='single-threaded')
    np.testing.assert_allclose(result[1:-1, 1:-1], 45, atol=1e-5)
    assert np.isnan(result[0]).all() and np.isnan(result[:, 0]).all()


@pytest.mark.parametrize('inner,outer', [(0, 1), (1, 2)])
def test_bpi_dask_matches_independent_annulus(inner, outer):
    data = np.arange(64, dtype=np.float32).reshape(8, 8) - 100
    data[2, 3] = np.nan
    expected = reference_bpi(data, inner, outer)
    result = module._calculate_bpi_dask(da.from_array(data, chunks=(4, 4)), 8, inner*8, outer*8).compute(scheduler='single-threaded')
    np.testing.assert_allclose(result, expected, atol=1e-4, equal_nan=True)
    np.testing.assert_allclose(module._calculate_bpi(data, 8, inner*8, outer*8), expected, atol=1e-4, equal_nan=True)


def test_bpi_no_neighbors_remains_nodata():
    data = np.full((8, 8), np.nan, dtype=np.float32)
    data[4, 4] = -10
    result = module._calculate_bpi(data, 8, 0, 8)
    assert np.isnan(result).all()


@pytest.mark.parametrize('cell,inner,outer', [(0, 0, 8), (8, -8, 8), (8, 8, 8), (1, 0, 513)])
def test_bpi_invalid_radii_and_safety_limit(cell, inner, outer):
    with pytest.raises(ValueError):
        module._calculate_bpi_dask(da.ones((8, 8), chunks=(4, 4)), cell, inner, outer)


def test_tci_core_and_dask_match_independent_neighbor_variance():
    data = np.arange(64, dtype=np.float32).reshape(8, 8) - 100
    data[2, 3] = np.nan
    expected = reference_tci(data)
    np.testing.assert_allclose(module._calculate_tci(data), expected, atol=1e-5, equal_nan=True)
    lazy = module._calculate_tci_dask(da.from_array(data, chunks=(4, 4)))
    np.testing.assert_allclose(lazy.compute(scheduler='single-threaded'), expected, atol=1e-5, equal_nan=True)


def test_tci_flat_and_isolated_cells():
    np.testing.assert_allclose(module._calculate_tci(np.full((5, 5), -10, dtype=np.float32)), 0)
    data = np.full((5, 5), np.nan, dtype=np.float32)
    data[2, 2] = -10
    assert np.isnan(module._calculate_tci(data)).all()

# =====================================================================
# Classification Dictionaries, Masks & Derived Products
# =====================================================================

def test_classification_dictionary_quantiles_and_empty_fallback():
    sample = np.array([-10, 0, 10, np.nan], dtype=np.float32)
    frame = module._create_classification_dictionary(sample, sample, np.array([0, 10, 20, np.nan]))
    assert frame.Class_ID.tolist() == list(range(1, 9))
    assert frame.loc[0, 'BroadBPI_Lower'] == pytest.approx(-7)
    assert frame.loc[0, 'BroadBPI_Upper'] == pytest.approx(7)
    assert frame.loc[7, 'Slope_Lower'] == pytest.approx(17)
    empty = module._create_classification_dictionary(np.array([np.nan]), np.array([np.nan]), np.array([np.nan]))
    assert not empty.isna().any().any()
    assert empty.loc[0, 'BroadBPI_Lower'] == -9999


def test_load_dictionary_by_year_and_missing(victim):
    expected = pd.DataFrame({'Class_ID': [1], 'Zone_Name': ['test']})
    expected.to_csv(str(victim.dictionary_dir / 'dictionary_2004.csv'), index=False)
    frame, error = module._load_classification_dictionary(True, False, 'combined_2004_tile', str(victim.dictionary_dir))
    pd.testing.assert_frame_equal(frame, expected)
    assert error is None
    assert module._load_classification_dictionary(False, False, 'unused', 'unused') == (None, None)
    assert 'Dictionary missing for BlueTopo' in module._load_classification_dictionary(True, True, 'unused', str(victim.dictionary_dir))[1]


def test_regional_dictionary_real_sampling(victim, tmp_path):
    data = np.full((16, 16), -20, dtype=np.float32)
    data[8, 8] = -18
    source = write_raster(tmp_path / 'combined_2004.tif', data)
    ok, msg = module._create_regional_dictionary_worker('2004', [str(source)], str(victim.dictionary_dir), RADII, str(victim.local_tmp_dir))
    assert ok, msg
    saved = pd.read_csv(str(victim.dictionary_dir / 'dictionary_2004.csv'))
    assert len(saved) == 8 and not saved.isna().any().any()


def test_regional_dictionary_no_readable_files(victim):
    ok, msg = module._create_regional_dictionary_worker('2004', ['missing.tif'], str(victim.dictionary_dir), RADII, str(victim.local_tmp_dir))
    assert not ok and 'no readable input files' in msg


def test_prepare_bathy_masks_nodata_land_and_shallow_values(tmp_path):
    source = write_raster(tmp_path / 'input.tif', [[-10, 0, 2], [-9999, np.nan, -0.005]])
    local = module._terrain_local_paths(str(tmp_path))
    grid = module._prepare_terrain_bathy(str(source), local, 'tile')
    with rasterio.open(local['bathy']) as src:
        np.testing.assert_array_equal(src.read(1), [[-10, -9999, -9999], [-9999, -9999, -9999]])
        assert src.nodata == -9999
    assert grid[0] == (2, 3) and not pathlib.Path(local['bathy_raw']).exists()


def test_load_bathy_memmap_masks_sentinel_and_closes(tmp_path):
    source = write_raster(tmp_path / 'input.tif', [[-10, -9999], [-12, -14]])
    values, profile, cell, shape = module._load_bathy_memmap(str(source), str(tmp_path))
    try:
        assert isinstance(values, np.memmap) and np.isnan(values[0, 1])
        assert cell == 8 and shape == (2, 2)
    finally:
        module._close_memmap(values)
    assert values._mmap.closed


def test_gradmag_radians_keeps_flat_zero(tmp_path):
    source = write_raster(tmp_path / 'slope.tif', [[0, 30], [45, -9999]])
    out = tmp_path / 'gradmag.tif'
    module._generate_gradmag_from_slope(str(source), str(tmp_path), (2, 2), profile_for(source), str(out), str(tmp_path), '')
    with rasterio.open(out) as src:
        np.testing.assert_allclose(src.read(1), [[0, np.pi/6], [np.pi/4, np.nan]], atol=1e-6, equal_nan=True)
        assert src.tags()['hydro_health_product_version'] == module.PRODUCT_VERSIONS['_gradmag.tif']


def test_bluetopo_mask_and_gradmag_thresholds(tmp_path):
    base = 'BlueTopo_BH4S556X_20200101'
    slope = write_raster(tmp_path / f'{base}_slope.tiff', [[0, 75, 76], [-1, -9999, 45]])
    profile = profile_for(slope)
    values = np.full((2, 3), -10, dtype=np.float32)
    out = tmp_path / 'gradmag.tif'
    module._mask_bluetopo_bathy(values, (2, 3), profile['crs'], profile['transform'],
        str(tmp_path / f'{base}.tiff'), base, str(tmp_path), True, str(tmp_path), profile,
        str(out), str(tmp_path), '')
    np.testing.assert_allclose(values, [[-10, -10, np.nan], [np.nan, np.nan, -10]], equal_nan=True)
    with rasterio.open(out) as src:
        np.testing.assert_allclose(src.read(1), [[0, np.radians(75), np.nan], [np.nan, np.nan, np.pi/4]], atol=1e-6, equal_nan=True)


def test_bluetopo_missing_external_slope_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match='external BlueTopo slope'):
        module._mask_bluetopo_bathy(np.ones((2, 2)), (2, 2), 'EPSG:32617',
            from_origin(0, 16, 8, 8), 'BlueTopo_tile_20200101.tif', 'BlueTopo_tile_20200101',
            str(tmp_path), False, str(tmp_path), {}, 'unused', str(tmp_path), '')


def test_shear_proxy_values_and_nodata(tmp_path):
    slope = write_raster(tmp_path / 'slope.tif', [[0, 10], [20, -9999]])
    plan = write_raster(tmp_path / 'plan.tif', [[-2, -3], [np.nan, 2]])
    out = tmp_path / 'shear.tif'
    errors = []
    module._generate_shear_proxy(True, str(out), str(slope), str(plan), str(slope), str(plan), str(tmp_path), 'tile', errors)
    assert not errors
    with rasterio.open(out) as src:
        np.testing.assert_array_equal(src.read(1), [[0, 30], [-9999, -9999]])


def test_classification_first_match_and_nodata(tmp_path):
    slope = write_raster(tmp_path / 'slope.tif', [[1, 20], [1, -9999]])
    broad = write_raster(tmp_path / 'broad.tif', [[0, 0], [2, 0]])
    fine = write_raster(tmp_path / 'fine.tif', [[0, 0], [2, 0]])
    dictionary = module._create_classification_dictionary(np.array([-1, 0, 1]), np.array([-1, 0, 1]), np.array([0, 10, 20]))
    out = tmp_path / 'class.tif'
    profile = profile_for(slope)
    with patch(f'{TERRAIN_PATH}._materialize_or_rebuild_bpi', side_effect=[str(broad), str(fine)]):
        module._classify_terrain(False, 'unused', 'combined_2004', '', str(slope), str(slope),
            str(tmp_path), (2, 2), str(broad), str(fine), None, 8, RADII, profile,
            str(tmp_path), '', profile['crs'], profile['transform'], dictionary, str(out))
    with rasterio.open(out) as src:
        # Class 3 precedes the more specific broad-crest classes in current rules.
        np.testing.assert_allclose(src.read(1), [[1, 8], [3, np.nan]], equal_nan=True)


# =====================================================================
# Whitebox Calls, Dependency Scheduling & Worker Status
# =====================================================================

def test_missing_products_uses_d8_and_surface_rugosity(tmp_path):
    outputs = module._terrain_output_paths(str(tmp_path / 'out'), 'tile')
    local = module._terrain_local_paths(str(tmp_path))
    wbt = MagicMock()
    with patch(f'{TERRAIN_PATH}._validate_product_raster', return_value=(False, 'missing')), \
         patch(f'{TERRAIN_PATH}._validate_raster', return_value=(False, 'missing')):
        missing, shear, numpy_missing = module._missing_terrain_products(outputs, local, wbt, False)
    commands = {suffix: command for _, command, _, suffix in missing}
    commands['_flowdir.tif']('in', 'out')
    wbt.d8_pointer.assert_called_once_with('in', 'out', esri_pntr=False)
    commands['_rugosity.tif']('in', 'out')
    wbt.surface_area_ratio.assert_called_once_with('in', 'out')
    assert shear and all(numpy_missing.values())


@pytest.mark.parametrize('mode', ['ok', 'exit_code', 'missing_output', 'exception'])
def test_generate_whitebox_product_status(tmp_path, mode):
    local = tmp_path / 'local.tif'
    out = tmp_path / 'published.tif'
    def command(source, destination):
        if mode == 'exception': raise RuntimeError('tool failed')
        if mode == 'exit_code': return 1
        if mode == 'ok': write_raster(destination, [[1, 2], [3, 4]])
        return 0
    errors = []
    module._generate_whitebox_products([(str(out), command, str(local), '_flowdir.tif')],
        MagicMock(), 'bathy', 'slope', 'plan', 'tile', errors)
    if mode == 'ok':
        assert not errors and not local.exists()
        assert module._validate_product_raster(out, '_flowdir.tif') == (True, '')
    else:
        assert errors and not out.exists()


@pytest.mark.parametrize('bluetopo', [False, True])
def test_check_unprocessed_complete_and_missing(tmp_path, bluetopo):
    name = 'BlueTopo_BH4S556X_20200101' if bluetopo else 'combined_2004_tile'
    with patch(f'{TERRAIN_PATH}._validate_product_raster', return_value=(True, '')) as product, \
         patch(f'{TERRAIN_PATH}._validate_raster', return_value=(True, '')):
        assert not module._check_unprocessed_worker(name+'.tif', str(tmp_path), '')
        suffixes = [c.args[1] for c in product.call_args_list]
        assert ('_slope.tif' in suffixes) is (not bluetopo)
    assert module._check_unprocessed_worker(name+'.tif', str(tmp_path), '')


@pytest.mark.parametrize('mode', ['success', 'skip', 'config_error', 'import_error', 'bad_grid', 'product_error'])
def test_terrain_worker_control_flow_and_temp_cleanup(victim, tmp_path, mode):
    wbt = MagicMock()
    methods = ['_configure_terrain_whitebox', '_missing_terrain_products', '_prepare_terrain_bathy',
        '_generate_whitebox_products', '_generate_shear_proxy', '_generate_numpy_products', '_validate_terrain_outputs']
    with ExitStack() as stack:
        mocks = {name: stack.enter_context(patch(f'{TERRAIN_PATH}.{name}')) for name in methods}
        mocks['_configure_terrain_whitebox'].return_value = (wbt, None)
        missing = {'_terrain_classification.tif': True}
        mocks['_missing_terrain_products'].return_value = ([], False, missing)
        mocks['_prepare_terrain_bathy'].return_value = ((2, 2), rasterio.crs.CRS.from_epsg(32617), from_origin(500000, 3400000, 8, 8))
        mocks['_generate_numpy_products'].return_value = None
        if mode == 'skip': mocks['_missing_terrain_products'].return_value = ([], False, {'x': False})
        if mode == 'config_error': mocks['_configure_terrain_whitebox'].return_value = (None, 'configuration failed')
        if mode == 'import_error': mocks['_configure_terrain_whitebox'].side_effect = ImportError('not installed')
        if mode == 'bad_grid': mocks['_prepare_terrain_bathy'].side_effect = ValueError('invalid grid')
        if mode == 'product_error':
            mocks['_validate_terrain_outputs'].side_effect = lambda outputs, bt, grid, errors: errors.append('invalid product')
        success, message = module._process_terrain_raster_worker('combined_2004_tile.tif', 1, 1,
            RADII, str(victim.terrain_outputs_dir), str(victim.prediction_output_dir),
            str(victim.dictionary_dir), str(victim.local_tmp_dir))
    assert success is (mode in ['success', 'skip']), message
    assert not list(victim.local_tmp_dir.iterdir())
    if mode == 'product_error': assert 'invalid product' in message


# =====================================================================
# Cluster Orchestration, Grouping, Remaining Work & Phase Order
# =====================================================================

def test_execute_regional_limits_groups_years_and_bluetopo(victim):
    files = ['combined_2004_a.tif', 'combined_2004_b.tif', 'combined_2006_c.tif', 'BlueTopo_tile_20200101.tif']
    victim.client.scatter.return_value = ['group1', 'group2', 'group3']
    victim.client.gather.return_value = [(True, 'done')] * 3
    victim._execute_regional_limits(files, RADII)
    function, years, futures = victim.client.map.call_args.args
    assert function is module._create_regional_dictionary_worker
    assert years == ['2004', '2006', 'BlueTopo']
    assert victim.client.scatter.call_args.args[0] == [files[:2], [files[2]], [files[3]]]
    assert futures == ['group1', 'group2', 'group3']


def test_execute_terrain_generation_skips_and_sorts(victim, tmp_path):
    paths = [tmp_path / name for name in ['large.tif', 'done.tif', 'small.tif']]
    for path, size in zip(paths, [10, 5, 1]): path.write_bytes(b'x' * size)
    victim.client.gather.side_effect = [[True, False, True], [(True, 'ok'), (False, 'failed')]]
    victim._execute_terrain_generation([str(p) for p in paths], RADII)
    check, generate = victim.client.map.call_args_list
    assert check.args[0] is module._check_unprocessed_worker
    assert generate.args[0] is module._process_terrain_raster_worker
    assert generate.args[1] == [str(paths[2]), str(paths[0])]
    assert generate.args[2] == [1, 2] and generate.kwargs['total_count'] == 2
    assert any('1 failed tile' in c.args[0] for c in victim.write_message.call_args_list)


def test_execute_terrain_generation_all_current(victim):
    victim.client.gather.return_value = [False]
    victim._execute_terrain_generation(['done.tif'], RADII)
    assert victim.client.map.call_count == 1


def test_report_empty_skips_pdf(victim):
    with patch(f'{TERRAIN_PATH}.PdfPages') as pdf:
        victim._generate_pdf_report([])
    pdf.assert_not_called()


def test_run_phase_order(victim):
    victim.param_lookup['eco_regions'].value = ['ER_1', 'ER_2']
    with ExitStack() as stack:
        names = ['_resolve_paths', '_get_files_to_process', '_execute_regional_limits', '_execute_terrain_generation', '_generate_pdf_report']
        mocks = {name: stack.enter_context(patch.object(victim, name)) for name in names}
        mocks['_get_files_to_process'].side_effect = [['combined_2004.tif'], []]
        manager = MagicMock()
        for name, mock in mocks.items(): manager.attach_mock(mock, name)
        victim.run()
    assert manager.mock_calls == [call._resolve_paths('ER_1'), call._get_files_to_process(),
        call._execute_regional_limits(['combined_2004.tif'], {'fine': (8, 32), 'broad': (80, 240)}),
        call._execute_terrain_generation(['combined_2004.tif'], {'fine': (8, 32), 'broad': (80, 240)}),
        call._generate_pdf_report(['combined_2004.tif']), call._resolve_paths('ER_2'), call._get_files_to_process()]
    victim.setup_dask.assert_called_once_with('local', n_workers=2, threads_per_worker=1, memory_limit='13GB')
    victim.cleanup_resources.assert_called_once()


def test_run_exception_cleans_cluster(victim):
    with patch.object(victim, '_resolve_paths', side_effect=RuntimeError('failure')):
        with pytest.raises(RuntimeError): victim.run()
    victim.cleanup_resources.assert_called_once()
