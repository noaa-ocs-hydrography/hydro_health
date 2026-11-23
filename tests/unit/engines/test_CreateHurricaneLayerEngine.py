"""Hurricane engine tests following the supplied BlueTopo pytest template."""
import io
import os
import sys
import pathlib
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.CreateHurricaneLayerEngine import CreateHurricaneLayerEngine

HURRICANE_PATH = 'hydro_health.engines.CreateHurricaneLayerEngine'


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    lookup = {
        ('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
        ('SHARED', 'MASTER_GRIDS'): 'Master_Grids.gpkg',
        ('SHARED', 'ENHANCED_ECOREGIONS'): 'Enhanced_EcoRegions',
        ('SHARED', 'ECOREGION_COLUMN'): 'region_id',
        ('HURRICANE', 'DATA_PATH'): 'hurricane/text',
        ('HURRICANE', 'GPKG_PATH'): 'hurricane/shared.gpkg',
        ('HURRICANE', 'YEAR_PAIR_RASTER_PATH'): 'hurricane/pairs',
        ('HURRICANE', 'RASTER_PATH'): 'hurricane/storms',
        ('HURRICANE', 'COUNT_RASTER_PATH'): 'hurricane/count',
        ('HURRICANE', 'CUMULATIVE_RASTER_PATH'): 'hurricane/strength',
        ('MASK', 'MASK_PRED_PATH'): 'masks/prediction.tif',
    }
    with patch(f'{HURRICANE_PATH}.Engine.__init__', return_value=None), \
         patch(f'{HURRICANE_PATH}.Engine.write_message', create=True), \
         patch(f'{HURRICANE_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{HURRICANE_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{HURRICANE_PATH}.get_config_item', side_effect=lambda *args: lookup[args]), \
         patch(f'{HURRICANE_PATH}.requests.get') as http, \
         patch.object(UPath, 'open', autospec=True) as upath_open:
        # Only local opens are permitted by default. AWS I/O tests supply fakes.
        def local_open(path, *args, **kwargs):
            if str(path).startswith('s3://'):
                raise AssertionError('Real S3 access is forbidden in unit tests')
            return open(str(path), *args, **kwargs)
        upath_open.side_effect = local_open
        yield SimpleNamespace(lookup=lookup, http=http, upath_open=upath_open)


@pytest.fixture
def victim():
    engine = CreateHurricaneLayerEngine({'env': 'local', 'eco_regions': SimpleNamespace(value=['ER_3'])})
    engine.write_message = MagicMock()
    # Shared Engine was not supplied; isolate inherited attributes explicitly.
    engine.target_crs = 'EPSG:32617'
    engine.year_ranges = [(2004, 2006)]
    engine._resolve_paths('ER_3')
    return engine


def write_raster(path, data, transform=None, crs='EPSG:32617', nodata=np.nan):
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    with rasterio.open(path, 'w', driver='GTiff', width=data.shape[1], height=data.shape[0],
        count=1, dtype='float32', transform=transform or from_origin(500000, 3400400, 100, 100),
        crs=crs, nodata=nodata) as dst:
        dst.write(data, 1)
    return path


def eco_frame():
    return gpd.GeoDataFrame({'region_id': ['ER_3']},
        geometry=[box(500000, 3400000, 500200, 3400400)], crs='EPSG:32617')


def hurdat_row(wind=70, radii=None, date='20040801', storm='AL012004'):
    radii = radii if radii is not None else [-999] * 12
    # Three header fields plus trailing blank, followed by 21 observation fields.
    return [storm, ' TEST ', '1', '', date, '0000', '', 'HU', '30.0N', '80.0W',
            str(wind), '980'] + [str(v) for v in radii] + ['20']


def ring_frame():
    return gpd.GeoDataFrame({
        'area_date': ['AL012004', 'AL012004', 'AL022004'],
        'wind_speed': [34, 64, 34], 'year': [2004] * 3,
        'name': ['UNNAMED'] * 3, 'id': ['low', 'high', 'other'],
        'dissolve_id': ['34_A', '64_A', '34_B'],
    }, geometry=[box(0, 0, 400, 400), box(100, 100, 300, 300), box(0, 0, 400, 400)], crs='EPSG:5070')


# =====================================================================
# Constructor, Paths & Region Selection
# =====================================================================

@pytest.mark.parametrize('env,is_aws', [('local', False), ('remote', False), ('aws', True), (None, False)])
def test_init(env, is_aws):
    params = {} if env is None else {'env': env}
    engine = CreateHurricaneLayerEngine(params)
    assert engine.param_lookup is params
    assert engine.is_aws is is_aws
    assert engine.output_prefix is False and engine.overwrite is True


@pytest.mark.parametrize('env', ['local', 'remote', 'aws'])
@pytest.mark.parametrize('prefix', [False, 'low_res'])
def test_resolve_paths(tmp_path, env, prefix):
    engine = CreateHurricaneLayerEngine({'env': env}, prefix)
    engine._resolve_paths('ER_2')
    base = tmp_path / 'outputs'
    local_base = base / prefix if prefix else base
    assert engine.region == 'ER_2' and engine.outputs_dir == local_base / 'ER_2'
    if env == 'aws':
        assert str(engine.txt_data_path) == 's3://test-bucket/hurricane/text'
        assert str(engine.hurricane_data_path) == 's3://test-bucket/hurricane/shared.gpkg'
        root = 's3://test-bucket/ER_2'
    else:
        assert str(engine.txt_data_path) == str(local_base / 'hurricane/text')
        assert str(engine.hurricane_data_path) == str(local_base / 'hurricane/shared.gpkg')
        root = str(local_base / 'ER_2')
        assert engine.txt_data_path.is_dir() and engine.raster_path.is_dir()
    for attr, suffix in [('raster_path', 'hurricane/storms'), ('year_pair_raster_path', 'hurricane/pairs'),
                         ('count_raster_path', 'hurricane/count'), ('cumulative_raster_path', 'hurricane/strength'),
                         ('mask_pred_path', 'masks/prediction.tif')]:
        assert getattr(engine, attr) == UPath(root) / suffix
    assert engine.coast_boundary_path == UPath(tmp_path / 'inputs/Master_Grids.gpkg')


def test_master_grid_uri(victim, isolated_services):
    isolated_services.lookup[('SHARED', 'MASTER_GRIDS')] = 's3://test-bucket/master.gpkg'
    victim._resolve_paths('ER_1')
    assert str(victim.coast_boundary_path) == 's3://test-bucket/master.gpkg'


@pytest.mark.parametrize('path,expected', [('s3://bucket/folder/a.gpkg', '/vsis3/bucket/folder/a.gpkg'),
    ('relative/a.gpkg', 'relative/a.gpkg')])
def test_get_gdal_path(victim, path, expected):
    assert victim._get_gdal_path(path) == expected


def test_read_ecoregion_selects_identifier_and_reprojects(victim):
    frame = gpd.GeoDataFrame({'region_id': [' ER_3 ', 'ER_4']},
        geometry=[box(-81, 30, -80, 31), box(-79, 30, -78, 31)], crs='EPSG:4326')
    with patch.object(victim, '_read_gpkg_layer', return_value=frame) as read:
        result = victim._read_ecoregion()
    assert len(result) == 1 and result.crs.to_epsg() == 32617
    assert frame.crs.to_epsg() == 4326 and len(frame) == 2
    read.assert_called_once_with('Enhanced_EcoRegions', victim.coast_boundary_path)


def test_read_ecoregion_missing(victim):
    victim.region = 'ER_99'
    with patch.object(victim, '_read_gpkg_layer', return_value=eco_frame()):
        with pytest.raises(ValueError, match='was not found'):
            victim._read_ecoregion()


# =====================================================================
# GeoPackage Staging, Layer Preservation & Failed Write Protection
# =====================================================================

def test_gpkg_save_read_preserves_other_layers(victim):
    frame = gpd.GeoDataFrame({'value': [1]}, geometry=[Point(0, 0)], crs='EPSG:5070')
    victim._save_gpkg_layer(frame, 'points')
    victim._save_gpkg_layer(frame.assign(value=2), 'other')
    victim._save_gpkg_layer(frame.assign(value=3), 'points')
    assert set(pyogrio.list_layers(str(victim.hurricane_data_path))[:, 0]) == {'points', 'other'}
    assert victim._read_gpkg_layer('points')['value'].tolist() == [3]
    assert victim._read_gpkg_layer('other')['value'].tolist() == [2]


def test_gpkg_no_overwrite_skips_existing(victim):
    frame = gpd.GeoDataFrame({'value': [1]}, geometry=[Point(0, 0)], crs='EPSG:5070')
    victim._save_gpkg_layer(frame, 'points')
    victim.overwrite = False
    with patch.object(frame, 'to_file') as write:
        victim._save_gpkg_layer(frame, 'points')
    write.assert_not_called()


def test_gpkg_failed_write_preserves_original(victim):
    frame = gpd.GeoDataFrame({'value': [1]}, geometry=[Point(0, 0)], crs='EPSG:5070')
    victim._save_gpkg_layer(frame, 'points')
    original = pathlib.Path(str(victim.hurricane_data_path)).read_bytes()
    staged = []
    def fail(path, **kwargs):
        staged.append(path)
        raise OSError('write failed')
    with patch.object(frame, 'to_file', side_effect=fail):
        with pytest.raises(OSError, match='write failed'):
            victim._save_gpkg_layer(frame, 'points')
    assert pathlib.Path(str(victim.hurricane_data_path)).read_bytes() == original
    assert not pathlib.Path(staged[0]).parent.exists()


def test_gpkg_cloud_read_uses_local_temp(victim):
    victim.is_aws = True
    remote = MagicMock()
    remote.__str__.return_value = 's3://test-bucket/shared.gpkg'
    remote.open.return_value = io.BytesIO(b'test bytes')
    captured = []
    def read(path, layer):
        captured.append(path)
        assert pathlib.Path(path).read_bytes() == b'test bytes'
        return 'read result'
    with patch(f'{HURRICANE_PATH}.gpd.read_file', side_effect=read):
        assert victim._read_gpkg_layer('points', remote) == 'read result'
    assert not pathlib.Path(captured[0]).parent.exists()


def test_gpkg_aws_mode_local_target_avoids_download(victim):
    victim.is_aws = True
    with patch(f'{HURRICANE_PATH}.gpd.read_file') as read:
        victim._read_gpkg_layer('points', pathlib.Path('local.gpkg'))
    read.assert_called_once_with('local.gpkg', layer='points')


def test_gpkg_cloud_save_stages_then_uploads(victim):
    victim.is_aws = True
    remote = MagicMock()
    remote.exists.return_value = False
    victim.hurricane_data_path = remote
    frame = gpd.GeoDataFrame({'value': [1]}, geometry=[Point(0, 0)], crs='EPSG:5070')
    uploaded = []
    class Writer(io.BytesIO):
        def close(self):
            uploaded.append(self.getvalue())
            super().close()
    remote.open.return_value = Writer()
    victim._save_gpkg_layer(frame, 'points')
    remote.open.assert_called_once_with('wb')
    assert uploaded[0].startswith(b'SQLite format 3')


# =====================================================================
# Text Parsing, Coordinate Conversion & Historical Radius Proxies
# =====================================================================

@pytest.mark.parametrize('coord,expected', [('30.5N', 30.5), ('30.5S', -30.5), ('80.0W', -80), ('80.0E', 80)])
def test_convert_coordinates(victim, coord, expected):
    assert victim.convert_coordinates(coord) == expected


@pytest.mark.parametrize('coord', ['30.0', 'BAD', ''])
def test_convert_coordinates_invalid(victim, coord):
    with pytest.raises(ValueError):
        victim.convert_coordinates(coord)


def test_read_text_data_groups_storm_headers(victim):
    path = victim.txt_data_path / 'test.txt'
    path.write_text('AL012004, TEST, 1,\n20040801,0000\n\nAL022004, OTHER, 1,\n20040802,0600\n')
    rows = victim.read_text_data(path)
    assert len(rows) == 2
    assert rows[0][0] == 'AL012004' and rows[1][0] == 'AL022004'
    assert rows[1][-2:] == ['20040802', '0600']


@pytest.mark.parametrize('wind,proxies', [(33, [-999, -999, -999]), (34, [100, -999, -999]),
    (50, [100, 50, -999]), (64, [100, 50, 25])])
def test_convert_text_radius_thresholds(victim, wind, proxies):
    with patch.object(victim, 'read_text_data', return_value=[hurdat_row(wind)]), \
         patch.object(victim, '_save_gpkg_layer') as save:
        frame = victim.convert_text_to_gpkg()
    for speed, expected in zip([34, 50, 64], proxies):
        for quadrant in ['ne', 'se', 'sw', 'nw']:
            assert frame.iloc[0][f'r_{quadrant}{speed}'] == expected
    assert frame.iloc[0]['r_ne0'] == (150 if wind >= 34 else 0)
    assert frame.iloc[0]['cyclone_num'] == '01' and frame.iloc[0]['year'] == 2004
    assert frame.iloc[0]['name'] == 'TEST'
    assert frame.crs.to_epsg() == 4326
    assert frame.geometry.iloc[0].equals(Point(-80, 30))
    assert save.call_args.args[1] == 'atlantic_hurricane_points'


def test_convert_text_preserves_partially_available_radii(victim):
    radii = [20, -999, 0, 0] + [-999] * 8
    with patch.object(victim, 'read_text_data', return_value=[hurdat_row(70, radii)]), \
         patch.object(victim, '_save_gpkg_layer'):
        frame = victim.convert_text_to_gpkg()
    assert frame.iloc[0]['r_ne34'] == 20 and frame.iloc[0]['r_se34'] == -999
    assert frame.iloc[0]['r_ne0'] == 30 and frame.iloc[0]['r_se0'] == 0


def test_convert_text_wrong_column_count(victim):
    with patch.object(victim, 'read_text_data', return_value=[['bad', 'row']]):
        with pytest.raises(ValueError, match='column names'):
            victim.convert_text_to_gpkg()


def test_download_both_basins(victim, isolated_services):
    isolated_services.http.return_value.text = 'mock hurricane text'
    victim.download_hurricane_data()
    assert isolated_services.http.call_count == 2
    assert all(c.kwargs == {'timeout': 120} for c in isolated_services.http.call_args_list)
    assert isolated_services.http.return_value.raise_for_status.call_count == 2
    for name in ['atlantic_hurricane_data.txt', 'pacific_hurricane_data.txt']:
        assert (victim.txt_data_path / name).read_text() == 'mock hurricane text'


def test_download_http_error_propagates(victim, isolated_services):
    isolated_services.http.return_value.raise_for_status.side_effect = OSError('HTTP failure')
    with pytest.raises(OSError, match='HTTP failure'):
        victim.download_hurricane_data()
    assert not (victim.txt_data_path / 'atlantic_hurricane_data.txt').exists()

# =====================================================================
# Quadrant Geometry, Storm Swaths & Distinct Wind Rings
# =====================================================================

@pytest.mark.parametrize('quadrant,bounds', [('ne', (0, 0, 100, 100)), ('nw', (-100, 0, 0, 100)),
    ('se', (0, -100, 100, 0)), ('sw', (-100, -100, 0, 0))])
def test_quarter_circle_and_corners(victim, quadrant, bounds):
    polygon = victim.create_quarter_circle(Point(0, 0), 100, quadrant)
    assert polygon.is_valid
    np.testing.assert_allclose(polygon.bounds, bounds, atol=1e-10)
    assert polygon.area == pytest.approx(np.pi * 10000 / 4, rel=0.01)
    corners = victim.get_corner_points(polygon, quadrant)
    assert len(corners) == 3 and all(len(point) == 2 for point in corners)


def test_mega_polygon_connects_separated_points(victim):
    first = victim.create_quarter_circle(Point(0, 0), 100, 'ne')
    second = victim.create_quarter_circle(Point(300, 0), 100, 'ne')
    result = victim.create_mega_polygon({'geometry': first, 'label': 'ne_34'}, {'geometry': second, 'label': 'ne_34'})
    assert result.is_valid and result.geom_type == 'Polygon'
    assert result.covers(first) and result.covers(second)
    assert result.area > first.area + second.area


@pytest.mark.parametrize('records', [1, 2])
def test_buffers_keep_single_points_and_use_nm_conversion(victim, records):
    data = {'name': ['TEST'] * records, 'year': [2004] * records,
        'area_date': ['AL012004'] * records, 'longitude': [-80] * records, 'latitude': [30] * records}
    for q in ['ne', 'nw', 'se', 'sw']:
        for speed in [0, 34, 50, 64]:
            data[f'r_{q}{speed}'] = [1 if speed == 34 else 0] * records
    frame = gpd.GeoDataFrame(data, geometry=[Point(0, i * 3000) for i in range(records)], crs='EPSG:5070')
    with patch.object(victim, '_save_gpkg_layer') as save, \
         patch.object(victim, 'create_quarter_circle', wraps=victim.create_quarter_circle) as quarter:
        victim.create_overlapping_buffers(frame)
    assert quarter.call_count == 4 * records
    assert all(c.args[1] == 1852 for c in quarter.call_args_list)
    result = save.call_args.args[0]
    assert len(result) == 1 and result.crs.to_epsg() == 5070
    assert result.geometry.iloc[0].area > 0
    assert save.call_args.args[1] == 'atlantic_polygon_buffer'


def test_clip_polygons_rings_and_same_name_storm_independence(victim):
    with patch.object(victim, '_read_gpkg_layer', return_value=ring_frame()), \
         patch.object(victim, '_save_gpkg_layer') as save:
        victim.clip_polygons()
    frame = save.call_args.args[0]
    a = frame[frame.area_date == 'AL012004'].set_index('wind_speed')
    b = frame[frame.area_date == 'AL022004']
    assert a.loc[64].geometry.area == 40000
    assert a.loc[34].geometry.area == 120000
    assert a.loc[64].geometry.intersection(a.loc[34].geometry).area == 0
    assert b.geometry.iloc[0].area == 160000
    assert str(frame['wind_speed'].dtype) == 'Int32'
    assert save.call_args.args[1] == 'trimmed_polygons'


# =====================================================================
# Prediction Bounds, Grid-Preserving Crops & Raster Saving
# =====================================================================

def test_prediction_bounds_real_mask(victim):
    write_raster(victim.mask_pred_path, np.zeros((4, 4)))
    assert victim._prediction_bounds('EPSG:32617') == (500000, 3400000, 500400, 3400400)


def test_prediction_bounds_reprojects(victim):
    transform = from_origin(-81, 31, 0.01, 0.01)
    write_raster(victim.mask_pred_path, [[0, 0], [0, 0]], transform, 'EPSG:4326')
    expected = rasterio.warp.transform_bounds('EPSG:4326', 'EPSG:32617', -81, 30.98, -80.98, 31, densify_pts=21)
    np.testing.assert_allclose(victim._prediction_bounds('EPSG:32617'), expected)


@pytest.mark.parametrize('output_crs,mask_crs', [(None, 'EPSG:32617'), ('EPSG:32617', None)])
def test_prediction_bounds_rejects_missing_crs(victim, output_crs, mask_crs):
    write_raster(victim.mask_pred_path, [[0]], crs=mask_crs)
    with pytest.raises(ValueError, match='both have a CRS'):
        victim._prediction_bounds(output_crs)


@pytest.mark.parametrize('bounds,expected_slice', [
    ((500100, 3400100, 500300, 3400300), (slice(1, 3), slice(1, 3))),
    ((500150, 3400150, 500250, 3400250), (slice(1, 3), slice(1, 3))),
    ((499000, 3399000, 501000, 3401000), (slice(0, 4), slice(0, 4))),
])
def test_clip_prediction_bounds_preserves_grid(victim, bounds, expected_slice):
    data = np.arange(16).reshape(4, 4)
    transform = from_origin(500000, 3400400, 100, 100)
    with patch.object(victim, '_prediction_bounds', return_value=bounds):
        result, output_transform = victim._clip_to_prediction_bounds(data, transform, 'EPSG:32617')
    np.testing.assert_array_equal(result, data[expected_slice])
    assert output_transform.a == 100 and output_transform.e == -100
    assert output_transform.c == 500000 + expected_slice[1].start * 100
    assert output_transform.f == 3400400 - expected_slice[0].start * 100


def test_clip_prediction_bounds_no_overlap(victim):
    with patch.object(victim, '_prediction_bounds', return_value=(0, 0, 10, 10)):
        with pytest.raises(ValueError, match='does not overlap'):
            victim._clip_to_prediction_bounds(np.zeros((4, 4)), from_origin(500000, 3400400, 100, 100), 'EPSG:32617')


def test_save_raster_local_round_trip(victim, tmp_path):
    data = np.array([[0, 34], [np.nan, 64]], dtype=np.float32)
    transform = from_origin(500000, 3400400, 100, 100)
    output = UPath(tmp_path / 'saved.tif')
    victim.save_raster(data, output, 2, 2, transform, 'EPSG:32617')
    with rasterio.open(str(output)) as src:
        np.testing.assert_allclose(src.read(1), data, equal_nan=True)
        assert src.count == 1 and src.dtypes == ('float32',)
        assert src.crs.to_epsg() == 32617 and src.transform == transform
        assert np.isnan(src.nodata) and src.compression.value.lower() == 'lzw'


def test_save_raster_cloud_upload_and_cleanup(victim):
    victim.is_aws = True
    remote = MagicMock()
    uploaded, sources = [], []
    class Writer(io.BytesIO):
        def close(self):
            uploaded.append(self.getvalue())
            super().close()
    remote.open.return_value = Writer()
    original_open = rasterio.open
    def record(path, *args, **kwargs):
        sources.append(path)
        return original_open(path, *args, **kwargs)
    with patch(f'{HURRICANE_PATH}.rasterio.open', side_effect=record):
        victim.save_raster(np.ones((2, 2), dtype=np.float32), remote, 2, 2, from_origin(500000, 3400400, 100, 100), 'EPSG:32617')
    with rasterio.io.MemoryFile(uploaded[0]) as mem:
        with mem.open() as src:
            np.testing.assert_array_equal(src.read(1), np.ones((2, 2)))
    assert not pathlib.Path(sources[0]).parent.exists()


# =====================================================================
# Individual Storm Rasterization & Annual Exposure
# =====================================================================

def test_polygons_to_raster_storm_values_and_bounds(victim):
    frame = gpd.GeoDataFrame({'area_date': ['AL012004'], 'name': [' TEST STORM '],
        'year': [2004], 'wind_speed': [34]}, geometry=[box(500000, 3400000, 500400, 3400400)], crs=victim.target_crs)
    write_raster(victim.mask_pred_path, np.zeros((4, 4)))
    with patch.object(victim, '_read_gpkg_layer', return_value=frame), \
         patch.object(victim, '_read_ecoregion', return_value=eco_frame()):
        victim.polygons_to_raster()
    output = victim.raster_path / '2004/TESTSTORM_AL012004.tif'
    with rasterio.open(str(output)) as src:
        assert src.shape == (4, 2)
        np.testing.assert_array_equal(src.read(1), np.full((4, 2), 34))


def test_polygons_read_failure_has_context(victim):
    with patch.object(victim, '_read_gpkg_layer', side_effect=OSError('broken')):
        with pytest.raises(RuntimeError, match='Could not read the trimmed') as error:
            victim.polygons_to_raster()
    assert isinstance(error.value.__cause__, OSError)


@pytest.mark.parametrize('mode,expected', [('cumulative_count', [[2, 1], [1, 0]]),
    ('cumulative_windspeed', [[34, 64], [50, 0]])])
def test_cumulative_values_zero_years_and_region_mask(victim, mode, expected):
    # Two storms in 2004, no storms in 2005, plus an ecoregion hole.
    eco = gpd.GeoDataFrame(geometry=[box(500000, 3400000, 500200, 3400200).union(
        box(500300, 3400000, 500400, 3400200))], crs=victim.target_crs)
    transform = from_origin(500000, 3400200, 100, 100)
    write_raster(victim.raster_path / '2004/a.tif', [[0, 64], [np.nan, np.nan]], transform)
    write_raster(victim.raster_path / '2004/b.tif', [[34, np.nan], [50, np.nan]], transform)
    with patch.object(victim, '_read_ecoregion', return_value=eco), \
         patch.object(victim, '_read_gpkg_layer', return_value=pd.DataFrame({'year': [2004, 2005]})), \
         patch.object(victim, '_clip_to_prediction_bounds', side_effect=lambda data, t, c: (data, t)):
        victim.generate_cumulative_rasters(victim.count_raster_path, mode)
    with rasterio.open(str(victim.count_raster_path / f'{mode}_2004.tif')) as src:
        data = src.read(1)
        np.testing.assert_allclose(data[:, :2], expected)
        assert np.isnan(data[:, 2]).all()
        np.testing.assert_array_equal(data[:, 3], [0, 0])
        assert src.res == (100, 100)
    with rasterio.open(str(victim.count_raster_path / f'{mode}_2005.tif')) as src:
        data = src.read(1)
        assert np.all(data[:, :2] == 0) and np.isnan(data[:, 2]).all()


def test_cumulative_unknown_type(victim):
    with pytest.raises(ValueError, match='Unknown cumulative raster type'):
        victim.generate_cumulative_rasters(victim.count_raster_path, 'invalid')


# =====================================================================
# Year-Pair Means, Zero Values, Nodata & Grid Alignment
# =====================================================================

def test_year_pair_means_include_zeros_and_ignore_nodata(victim):
    arrays = [ [[0, 10, np.nan], [np.nan, 4, np.nan]],
               [[2, np.nan, np.nan], [6, 8, np.nan]],
               [[4, 20, np.nan], [10, 0, np.nan]] ]
    for year, data in zip(range(2004, 2007), arrays):
        write_raster(victim.cumulative_raster_path / f'cumulative_windspeed_{year}.tif', data, nodata=-9999)
    with patch.object(victim, '_clip_to_prediction_bounds', side_effect=lambda data, t, c: (data, t)):
        victim.year_pair_rasters(2004, 2006)
    with rasterio.open(str(victim.year_pair_raster_path / 'hurr_strength_mean_2004_2006.tif')) as src:
        np.testing.assert_allclose(src.read(1), [[2, 15, np.nan], [8, 4, np.nan]], equal_nan=True)


def test_year_pair_reprojects_coarser_grid(victim):
    ref = from_origin(500000, 3400200, 100, 100)
    write_raster(victim.cumulative_raster_path / 'cumulative_windspeed_2004.tif', [[0, 2], [4, 6]], ref)
    write_raster(victim.cumulative_raster_path / 'cumulative_windspeed_2005.tif', [[10]], from_origin(500000, 3400200, 200, 200))
    with patch.object(victim, '_clip_to_prediction_bounds', side_effect=lambda data, t, c: (data, t)):
        victim.year_pair_rasters(2004, 2005)
    with rasterio.open(str(victim.year_pair_raster_path / 'hurr_strength_mean_2004_2005.tif')) as src:
        np.testing.assert_allclose(src.read(1), [[5, 6], [7, 8]])
        assert src.transform == ref


def test_year_pair_missing_annual_skips(victim):
    with patch.object(victim, 'save_raster') as save:
        victim.year_pair_rasters(2004, 2006)
    save.assert_not_called()
    assert any('missing annual rasters' in c.args[0] for c in victim.write_message.call_args_list)


def test_year_pair_reversed_range(victim):
    with pytest.raises(ValueError, match='end_year'):
        victim.year_pair_rasters(2006, 2004)


def test_year_pair_no_overwrite(victim):
    path = victim.year_pair_raster_path / 'hurr_strength_mean_2004_2006.tif'
    path.write_bytes(b'existing')
    victim.overwrite = False
    with patch.object(victim, 'save_raster') as save:
        victim.year_pair_rasters(2004, 2006)
    save.assert_not_called()
    assert path.read_bytes() == b'existing'


def test_year_pair_misaligned_missing_crs(victim):
    write_raster(victim.cumulative_raster_path / 'cumulative_windspeed_2004.tif', [[1, 1], [1, 1]])
    write_raster(victim.cumulative_raster_path / 'cumulative_windspeed_2005.tif', [[2]], crs=None)
    with pytest.raises(ValueError, match='missing CRS'):
        victim.year_pair_rasters(2004, 2005)


# =====================================================================
# Shared Preparation Once & Multi-Region Workflow
# =====================================================================

@pytest.mark.parametrize('reuse', [False, True])
def test_run_shared_preparation_and_region_order(victim, reuse):
    victim.param_lookup['eco_regions'].value = ['ER_1', 'ER_2']
    victim.overwrite = not reuse
    if reuse:
        victim.hurricane_data_path.write_bytes(b'placeholder')
    methods = ['_resolve_paths', 'download_hurricane_data', 'convert_text_to_gpkg',
        'create_overlapping_buffers', 'clip_polygons', '_read_gpkg_layer',
        'polygons_to_raster', 'generate_cumulative_rasters', 'year_pair_rasters']
    from contextlib import ExitStack
    manager = MagicMock()
    with ExitStack() as stack:
        mocks = {name: stack.enter_context(patch.object(victim, name)) for name in methods}
        for name, mock in mocks.items():
            manager.attach_mock(mock, name)
        victim.run()
    expected = [call._resolve_paths('ER_1'), call.download_hurricane_data()]
    if reuse:
        expected += [call._read_gpkg_layer('atlantic_hurricane_points'), call._read_gpkg_layer('trimmed_polygons')]
    else:
        expected += [call.convert_text_to_gpkg(), call.create_overlapping_buffers(mocks['convert_text_to_gpkg'].return_value), call.clip_polygons()]
    for region in ['ER_1', 'ER_2']:
        expected += [call._resolve_paths(region), call.polygons_to_raster(),
            call.generate_cumulative_rasters(victim.count_raster_path, 'cumulative_count'),
            call.generate_cumulative_rasters(victim.cumulative_raster_path, 'cumulative_windspeed'),
            call.year_pair_rasters(2004, 2006)]
    assert manager.mock_calls == expected


def test_run_no_regions(victim):
    victim.param_lookup['eco_regions'].value = []
    with patch.object(victim, '_resolve_paths') as resolve, patch.object(victim, 'download_hurricane_data') as download:
        victim.run()
    resolve.assert_not_called()
    download.assert_not_called()
