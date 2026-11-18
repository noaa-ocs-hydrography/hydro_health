"""Tests for the merged sediment engine, following the BlueTopo pytest pattern.

External services and shared initialization are mocked. Small geospatial outputs
are real. See the README for two demonstrated defects and raster test isolation.
"""
import os
import sys
import pathlib
import sqlite3
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from shapely.geometry import Point, Polygon, box
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.CreateSedimentLayerEngine import CreateSedimentLayerEngine

SEDIMENT_PATH = 'hydro_health.engines.CreateSedimentLayerEngine'


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    """Block external HTTP/storage access and isolate shared configuration."""
    lookup = {
        ('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
        ('SHARED', 'MASTER_GRIDS'): 'Master_Grids.gpkg',
        ('SEDIMENT', 'SUBFOLDER'): 'sediment/data',
        ('SEDIMENT', 'GPKG_PATH'): 'sediment/sediment_data.gpkg',
        ('SEDIMENT', 'RASTER_PATH'): 'sediment/rasters',
        ('SEDIMENT', 'DATA_URL'): 'https://example.invalid/sediment.csv',
        ('MASK', 'MASK_PRED_PATH'): 'masks/prediction.tif',
    }
    with patch(f'{SEDIMENT_PATH}.Engine.__init__', return_value=None), \
         patch(f'{SEDIMENT_PATH}.Engine.write_message', create=True), \
         patch(f'{SEDIMENT_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{SEDIMENT_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{SEDIMENT_PATH}.get_config_item', side_effect=lambda *args: lookup[args]), \
         patch(f'{SEDIMENT_PATH}.requests.get') as http, \
         patch(f'{SEDIMENT_PATH}.fsspec.filesystem') as filesystem:
        yield {'lookup': lookup, 'http': http, 'factory': filesystem,
               'fs': filesystem.return_value}


@pytest.fixture
def victim(tmp_path):
    """Real engine; shared base attributes set explicitly because base is mocked."""
    params = {'env': 'local', 'eco_regions': SimpleNamespace(value=['ER_3'])}
    engine = CreateSedimentLayerEngine(params)
    engine.write_message = MagicMock()
    engine.target_crs = 'EPSG:6350'
    engine._resolve_paths('ER_3')
    engine.working_gpkg_path = tmp_path / 'working.gpkg'
    return engine


@pytest.fixture
def raster_numpy():
    """Explicit downstream isolation only; missing-import regression omits this."""
    with patch(f'{SEDIMENT_PATH}.np', np, create=True):
        yield


def sediment_frame():
    return pd.DataFrame({
        'Latitude': [30.0, 31.0, 32.0, 33.0],
        'Longitude': [-80.0, -81.0, -82.0, -83.0],
        'Gravel': [80, 5, 5, 5], 'Sand': [10, 80, 10, 10],
        'Mud': [7, 10, 80, 15], 'Clay': [3, 5, 5, 70],
        'Grainsze': [-1.0, 2.0, 5.0, 9.0],
    })


def point_frame(center_code=2):
    return gpd.GeoDataFrame({
        'sed_int': [1, 1, 1, 1, center_code],
        'sed_size': [3.0, 3.0, 3.0, 3.0, 0.25],
    }, geometry=[Point(0, 0), Point(400, 0), Point(400, 400),
                 Point(0, 400), Point(200, 200)], crs='EPSG:6350')


def create_prediction_mask(path, width=4, height=4, resolution=100,
                           left=0, top=400, crs='EPSG:6350'):
    """Real raster; zero pixels verify that cropping uses bounds, not mask values."""
    path = pathlib.Path(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype='uint8', crs=crs,
                       transform=rasterio.transform.from_origin(left, top, resolution, resolution)) as dst:
        dst.write(np.zeros((height, width), dtype=np.uint8), 1)
    return path


@pytest.fixture
def polygon_inputs(victim):
    create_prediction_mask(victim.prediction_mask_path)
    polygons = gpd.GeoDataFrame({
        'sed_type': [2, 4, np.nan], 'sed_size': [0.25, 0.001, np.nan],
        'sand_mud_mask': [1, np.nan, np.nan], 'custom': [7, 8, np.nan],
    }, geometry=[box(0, 200, 400, 400), box(0, 0, 100, 200),
                 box(100, 0, 200, 200)], crs=victim.target_crs)
    eco = gpd.GeoDataFrame(geometry=[box(0, 0, 200, 400)], crs=victim.target_crs)
    def read(path, layer, engine):
        assert engine == 'pyogrio'
        if layer == 'sediment_polygons':
            assert path == str(victim.working_gpkg_path)
            return polygons
        assert layer == 'Enhanced_EcoRegions'
        assert path == str(victim.master_grids_path)
        return eco
    with patch(f'{SEDIMENT_PATH}.gpd.read_file', side_effect=read) as reader:
        yield SimpleNamespace(polygons=polygons, eco=eco, reader=reader)


def capture_raster_transfer(victim):
    """Inspect real bytes before the engine deletes its temporary directory."""
    captured = []
    def put(source, destination):
        with rasterio.open(source) as src:
            captured.append((src.read(1), src.profile.copy(), source, destination))
    victim.filesystem.put_file.side_effect = put
    return captured


# =====================================================================
# Constructor & Per-Region Path Resolution
# =====================================================================

@pytest.mark.parametrize('environment,is_aws', [('local', False), ('remote', True), ('aws', True), (None, False)])
def test_init_defaults(environment, is_aws):
    params = {'eco_regions': SimpleNamespace(value=['ER_3'])}
    if environment is not None:
        params['env'] = environment
    engine = CreateSedimentLayerEngine(params)
    assert engine.param_lookup is params
    assert engine.is_aws is is_aws
    assert engine.output_prefix is False
    assert engine.sediment_types == ['Gravel', 'Sand', 'Mud', 'Clay']
    assert engine.sediment_data is None
    assert not hasattr(engine, 'gpkg_path')  # Deferred until a region is resolved.


@pytest.mark.parametrize('environment', ['local', 'remote', 'aws'])
@pytest.mark.parametrize('prefix', [False, 'low_res'])
def test_resolve_paths(tmp_path, isolated_services, environment, prefix):
    engine = CreateSedimentLayerEngine({'env': environment}, output_prefix=prefix)
    engine._resolve_paths('ER_2')
    base = tmp_path / 'outputs'
    local_base = base / prefix if prefix else base
    assert engine.outputs_dir == local_base / 'ER_2'
    assert engine.data_url == 'https://example.invalid/sediment.csv'
    assert str(engine.master_grids_path) == str(tmp_path / 'inputs/Master_Grids.gpkg')
    if environment == 'local':
        assert str(engine.data_path) == str(local_base / 'sediment/data')
        assert str(engine.gpkg_path) == str(local_base / 'ER_2/sediment/sediment_data.gpkg')
        assert str(engine.raster_path) == str(local_base / 'ER_2/sediment/rasters')
        assert str(engine.prediction_mask_path) == str(local_base / 'ER_2/masks/prediction.tif')
        assert engine.data_path.is_dir() and engine.gpkg_path.parent.is_dir()
        assert engine.raster_path.is_dir()
        isolated_services['factory'].assert_called_once_with('file', auto_mkdir=True)
    else:
        # Current cloud paths do not include output_prefix.
        assert str(engine.data_path) == 's3://test-bucket/sediment/data'
        assert str(engine.gpkg_path) == 's3://test-bucket/ER_2/sediment/sediment_data.gpkg'
        assert str(engine.raster_path) == 's3://test-bucket/ER_2/sediment/rasters'
        assert str(engine.prediction_mask_path) == 's3://test-bucket/ER_2/masks/prediction.tif'
        isolated_services['factory'].assert_called_once_with('s3')
    assert engine.csv_path == str(engine.data_path / 'US9_ONE.csv')


@pytest.mark.parametrize('master', ['s3://test-bucket/grids/master.gpkg', 'file:///tmp/master.gpkg'])
def test_resolve_paths_master_grid_uri(isolated_services, master):
    isolated_services['lookup'][('SHARED', 'MASTER_GRIDS')] = master
    engine = CreateSedimentLayerEngine({'env': 'local'})
    engine._resolve_paths('ER_1')
    assert engine.master_grids_path == UPath(master)


def test_resolve_paths_shared_csv_and_separate_regions(victim):
    first_csv = victim.csv_path
    first_gpkg = victim.gpkg_path
    victim._resolve_paths('ER_6')
    assert victim.csv_path == first_csv
    assert victim.gpkg_path != first_gpkg
    assert 'ER_6' in str(victim.gpkg_path)


# =====================================================================
# Downloads & CSV Filtering
# =====================================================================

def test_download_existing_file(victim, isolated_services):
    victim.filesystem.exists.return_value = True
    victim.download_sediment_data()
    victim.filesystem.exists.assert_called_once_with(victim.csv_path)
    isolated_services['http'].assert_not_called()
    victim.filesystem.open.assert_not_called()


@pytest.mark.parametrize('cloud', [False, True])
def test_download_streams_chunks(victim, isolated_services, cloud):
    if cloud:
        victim.data_path = UPath('s3://test-bucket/sediment/data')
        victim.csv_path = str(victim.data_path / 'US9_ONE.csv')
    victim.filesystem.exists.return_value = False
    response = isolated_services['http'].return_value.__enter__.return_value
    response.iter_content.return_value = [b'Latitude,Longitude\n', b'30,-80\n']
    victim.download_sediment_data()
    isolated_services['http'].assert_called_once_with(victim.data_url, stream=True)
    response.raise_for_status.assert_called_once_with()
    response.iter_content.assert_called_once_with(chunk_size=8192)
    victim.filesystem.makedirs.assert_called_once_with(str(victim.data_path), exist_ok=True)
    victim.filesystem.open.assert_called_once_with(victim.csv_path, 'wb')
    assert victim.filesystem.open.return_value.__enter__.return_value.write.call_args_list == [
        call(b'Latitude,Longitude\n'), call(b'30,-80\n')]


@pytest.mark.parametrize('stage', ['http', 'write'])
def test_download_failure_exits(victim, isolated_services, stage):
    victim.filesystem.exists.return_value = False
    response = isolated_services['http'].return_value.__enter__.return_value
    if stage == 'http':
        response.raise_for_status.side_effect = RuntimeError('test failure')
    else:
        response.iter_content.return_value = [b'data']
        victim.filesystem.open.return_value.__enter__.return_value.write.side_effect = OSError('test failure')
    with pytest.raises(SystemExit) as exc:
        victim.download_sediment_data()
    assert exc.value.code == 1
    assert any('Error downloading sediment CSV: test failure' in c.args[0]
               for c in victim.write_message.call_args_list)


def test_read_filters_sentinels_and_resets_index(victim):
    valid = sediment_frame().iloc[[0]].copy()
    rows = []
    for percentages, phi in [([0, 0, 0, 0], 2), ([-99, -99, -99, -99], 2),
                             ([0, -99, 0, -99], 2), ([10, 80, 5, 5], -99),
                             ([0, 80, -99, 20], 2)]:
        row = valid.iloc[0].copy()
        row[['Gravel', 'Sand', 'Mud', 'Clay']] = percentages
        row['Grainsze'] = phi
        rows.append(row)
    frame = pd.concat([valid, pd.DataFrame(rows)], ignore_index=True)
    frame['unused'] = 'not needed'
    frame.to_csv(victim.csv_path, index=False)
    # Real file handle, while retaining mocked filesystem/service boundaries.
    victim.filesystem.open.side_effect = lambda path, mode: open(path, mode)
    victim.read_sediment_data()
    victim.filesystem.open.assert_called_once_with(victim.csv_path, 'rb')
    assert victim.sediment_data.index.tolist() == [0, 1]
    assert victim.sediment_data['Sand'].tolist() == [10, 80]
    assert victim.sediment_data['Grainsze'].tolist() == [-1.0, 2.0]
    assert 'unused' not in victim.sediment_data.columns


def test_read_missing_required_column(victim):
    sediment_frame().drop(columns='Grainsze').to_csv(victim.csv_path, index=False)
    victim.filesystem.open.side_effect = lambda path, mode: open(path, mode)
    with pytest.raises(ValueError, match='Grainsze'):
        victim.read_sediment_data()


# =====================================================================
# Grain Sizes & Sediment Classification
# =====================================================================
def test_add_sed_size_column(victim):
    victim.sediment_data = sediment_frame()
    victim.add_sed_size_column()
    np.testing.assert_allclose(victim.sediment_data['sed_size'], [2, 0.25, 0.03125, 0.001953125])
    assert 'Grainsze' not in victim.sediment_data.columns
    np.testing.assert_allclose(victim.sediment_data['Size_phi'], [-1, 2, 5, 9])
    assert len(victim.sediment_data) == 4


def test_add_sed_size_column_accepts_integer_phi(victim):
    """CSV inference can produce integer dtype for whole-number phi values."""
    victim.sediment_data = pd.DataFrame({'Grainsze': [0, 1, 2]})
    victim.add_sed_size_column()
    np.testing.assert_allclose(victim.sediment_data['sed_size'], [1, 0.5, 0.25])


@pytest.mark.parametrize('size,expected', [
    (0.001, 'Clay'), (0.0038999, 'Clay'), (0.0039, 'Mud'),
    (0.0624999, 'Mud'), (0.0625, 'Sand'), (1.999999, 'Sand'),
    (2.0, 'Gravel'), (10.0, 'Gravel'),
    (0.0, 'Original'), (-1.0, 'Original'), (np.nan, 'Original'),
])
def test_correct_sed_type_boundaries(victim, size, expected):
    row = pd.Series({'sed_size': size, 'sed_type': 'Original'})
    assert victim.correct_sed_type(row) == expected


def test_determine_sed_types_percentage_ranking_then_size_correction(victim):
    victim.sediment_data = sediment_frame()
    victim.add_sed_size_column()
    # First point ranks Gravel then Sand, but grain size overrides primary to Clay.
    victim.sediment_data.loc[0, 'sed_size'] = 0.001
    victim.determine_sed_types()
    assert victim.sediment_data['sed_type'].tolist() == ['Clay', 'Sand', 'Mud', 'Clay']
    assert victim.sediment_data['sec_sed'].tolist() == ['Sand', 'Mud', 'Sand', 'Mud']


def test_add_sediment_mapping_column(victim):
    victim.sediment_data = pd.DataFrame({'sed_type': ['Gravel', 'Sand', 'Mud', 'Clay', 'Unknown', None]})
    victim.add_sediment_mapping_column()
    np.testing.assert_allclose(victim.sediment_data['sed_int'], [1, 2, 3, 4, np.nan, np.nan], equal_nan=True)


# =====================================================================
# Working GeoPackage Writes & SQLite Validation
# =====================================================================

@pytest.mark.parametrize('mode', ['w', 'a'])
def test_modify_gpkg_arguments(victim, mode):
    frame = MagicMock()
    victim._modify_gpkg(frame, 'sediment_points', mode=mode)
    frame.to_file.assert_called_once_with(victim.working_gpkg_path,
        layer='sediment_points', driver='GPKG', engine='pyogrio', mode=mode)
    victim.filesystem.put_file.assert_not_called()


def test_working_gpkg_real_layers_and_validation(victim):
    frame = point_frame()
    victim._modify_gpkg(frame, 'sediment_points')
    victim.transform_points_to_polygons()
    victim._validate_gpkg()
    points = gpd.read_file(victim.working_gpkg_path, layer='sediment_points', engine='pyogrio')
    polygons = gpd.read_file(victim.working_gpkg_path, layer='sediment_polygons', engine='pyogrio')
    assert len(points) == 5 and len(polygons) == 1
    assert points.crs.to_epsg() == 6350
    assert polygons.iloc[0]['sed_type'] == 2
    victim.filesystem.put_file.assert_not_called()


def test_validate_gpkg_readonly_uri(victim):
    with patch(f'{SEDIMENT_PATH}.sqlite3.connect') as connect:
        connection = connect.return_value.__enter__.return_value
        connection.execute.return_value.fetchall.return_value = [('ok',)]
        victim._validate_gpkg()
    connect.assert_called_once_with(victim.working_gpkg_path.resolve().as_uri() + '?mode=ro', uri=True)
    connection.execute.assert_called_once_with('PRAGMA quick_check')


def test_validate_gpkg_reports_integrity_errors(victim):
    with patch(f'{SEDIMENT_PATH}.sqlite3.connect') as connect:
        connect.return_value.__enter__.return_value.execute.return_value.fetchall.return_value = [('bad page',), ('bad index',)]
        with pytest.raises(RuntimeError, match='bad page; bad index'):
            victim._validate_gpkg()


def test_validate_missing_database_does_not_create_it(victim):
    with pytest.raises(sqlite3.OperationalError):
        victim._validate_gpkg()
    assert not victim.working_gpkg_path.exists()


def test_validate_corrupt_database(victim):
    victim.working_gpkg_path.write_bytes(b'not a SQLite database')
    with pytest.raises(sqlite3.DatabaseError):
        victim._validate_gpkg()


# =====================================================================
# Point Reprojection & Finite Voronoi Regions
# =====================================================================

@pytest.mark.parametrize('crs', ['EPSG:6350', 'EPSG:32617'])
def test_create_point_layer_uses_target_crs(victim, crs):
    victim.target_crs = crs
    victim.sediment_data = sediment_frame()
    with patch.object(victim, '_modify_gpkg') as modify:
        victim.create_point_layer()
    frame = modify.call_args.args[0]
    expected = gpd.GeoSeries(gpd.points_from_xy([-80, -81, -82, -83], [30, 31, 32, 33]), crs='EPSG:4326').to_crs(crs)
    assert frame.crs.to_string() == crs
    np.testing.assert_allclose(frame.geometry.x, expected.x)
    np.testing.assert_allclose(frame.geometry.y, expected.y)
    pd.testing.assert_series_equal(frame['Grainsze'], victim.sediment_data['Grainsze'])
    assert modify.call_args.args[1:] == ('sediment_points',)
    assert modify.call_args.kwargs == {'mode': 'w'}


@pytest.mark.parametrize('code,mask', [(1, None), (2, 1), (3, 1), (4, None)])
def test_transform_points_to_polygons(victim, code, mask):
    with patch(f'{SEDIMENT_PATH}.gpd.read_file', return_value=point_frame(code)) as read, \
         patch.object(victim, '_modify_gpkg') as modify:
        victim.transform_points_to_polygons()
    read.assert_called_once_with(str(victim.working_gpkg_path), layer='sediment_points', engine='pyogrio')
    frame = modify.call_args.args[0]
    assert len(frame) == 1  # Unbounded hull regions are still skipped.
    assert frame.crs.to_string() == victim.target_crs
    row = frame.iloc[0]
    assert row['sed_type'] == code and row['sed_size'] == 0.25
    assert row['sand_mud_mask'] == mask
    assert row.geometry.is_valid and row.geometry.contains(Point(200, 200))
    assert row.geometry.area == pytest.approx(80000)
    assert modify.call_args.args[1:] == ('sediment_polygons',)
    assert modify.call_args.kwargs == {'mode': 'a'}


# =====================================================================
# Raster Defect Regression & Downstream Numerical Tests
# =====================================================================


def test_rasterization_has_numpy_available(victim, polygon_inputs):
    """No injected NumPy: exercise the attached production namespace unchanged."""
    victim.convert_polys_to_raster('sed_size')
    victim.filesystem.put_file.assert_called_once()


@pytest.mark.parametrize('field,filename,top,bottom', [
    ('sed_type', 'prim_sed_layer.tif', 2, 4),
    ('sed_size', 'grain_size_layer.tif', 0.25, 0.001),
    ('sand_mud_mask', 'sand_mud_mask.tif', 1, -9999),
    ('custom', 'custom_raster_100m.tif', 7, 8),
])
@pytest.mark.parametrize('cloud', [False, True])
def test_raster_pixels_metadata_and_destination(victim, polygon_inputs, raster_numpy, field, filename, top, bottom, cloud):
    if cloud:
        victim.raster_path = UPath('s3://test-bucket/ER_3/sediment/rasters')
    captured = capture_raster_transfer(victim)
    victim.convert_polys_to_raster(field)
    assert len(captured) == 1
    data, profile, source, destination = captured[0]
    expected = np.array([[top, top, -9999, -9999], [top, top, -9999, -9999],
                         [bottom, -9999, -9999, -9999], [bottom, -9999, -9999, -9999]], dtype=np.float32)
    np.testing.assert_allclose(data, expected)
    assert profile['dtype'] == 'float32' and profile['count'] == 1
    assert profile['crs'].to_epsg() == 6350
    assert profile['transform'] == rasterio.transform.from_origin(0, 400, 100, 100)
    assert profile['nodata'] == -9999 and profile['compress'].lower() == 'lzw'
    assert destination == f'{victim.raster_path}/{filename}'
    assert not os.path.exists(source) and not pathlib.Path(source).parent.exists()
    victim.filesystem.makedirs.assert_called_once_with(str(victim.raster_path), exist_ok=True)
    # All three products now use the region's configured destination.
    victim.filesystem.put_file.assert_called_once()
    assert polygon_inputs.reader.call_count == 2


@pytest.mark.parametrize('resolution', [0, -100])
def test_raster_rejects_nonpositive_resolution(victim, resolution):
    with patch(f'{SEDIMENT_PATH}.gpd.read_file') as read:
        with pytest.raises(ValueError, match='resolution must be positive'):
            victim.convert_polys_to_raster('sed_size', resolution)
    read.assert_not_called()


def test_raster_rejects_mask_without_crs(victim, polygon_inputs):
    create_prediction_mask(victim.prediction_mask_path, crs=None)
    with pytest.raises(ValueError, match='Prediction mask must have a CRS'):
        victim.convert_polys_to_raster('sed_size')
    victim.filesystem.put_file.assert_not_called()


@pytest.mark.parametrize('bounds', [(0, 0, np.inf, 10), (0, 0, 0, 10), (0, 10, 10, 0)])
def test_raster_rejects_invalid_transformed_bounds(victim, polygon_inputs, raster_numpy, bounds):
    with patch(f'{SEDIMENT_PATH}.transform_bounds', return_value=bounds):
        with pytest.raises(ValueError, match='bounds could not be transformed'):
            victim.convert_polys_to_raster('sed_size')
    victim.filesystem.put_file.assert_not_called()


def test_raster_rounds_outward_at_requested_resolution(victim, polygon_inputs, raster_numpy):
    # Bounds are 350 x 350 m, output must cover them with 4 x 4 cells of 100 m.
    create_prediction_mask(victim.prediction_mask_path, width=7, height=7, resolution=50, top=350)
    captured = capture_raster_transfer(victim)
    victim.convert_polys_to_raster('sed_size', resolution=100)
    data, profile, _, _ = captured[0]
    assert data.shape == (4, 4)
    assert profile['transform'] == rasterio.transform.from_origin(0, 350, 100, 100)
    assert rasterio.transform.array_bounds(4, 4, profile['transform']) == (0, -50, 400, 350)


def test_raster_reprojects_mask_bounds_and_vectors(victim, polygon_inputs, raster_numpy):
    create_prediction_mask(victim.prediction_mask_path, width=2, height=2,
        resolution=0.001, left=-80, top=30, crs='EPSG:4326')
    bounds = rasterio.warp.transform_bounds('EPSG:4326', victim.target_crs,
        -80, 29.998, -79.998, 30, densify_pts=101)
    polygon_inputs.polygons.geometry = [box(bounds[0]-1000, bounds[1]-1000, bounds[2]+1000, bounds[3]+1000), None, Polygon()]
    polygon_inputs.eco.geometry = [polygon_inputs.polygons.geometry.iloc[0]]
    # Replace fixture data in place with actual geographic versions.
    polygons = polygon_inputs.polygons.to_crs('EPSG:4326')
    eco = polygon_inputs.eco.to_crs('EPSG:4326')
    polygon_inputs.reader.side_effect = lambda path, layer, engine: polygons if layer == 'sediment_polygons' else eco
    captured = capture_raster_transfer(victim)
    victim.convert_polys_to_raster('sed_type')
    data, profile, _, _ = captured[0]
    assert profile['crs'].to_epsg() == 6350
    assert profile['transform'].c == pytest.approx(bounds[0])
    assert profile['transform'].f == pytest.approx(bounds[3])
    assert data.shape == (int(np.ceil((bounds[3]-bounds[1])/100)), int(np.ceil((bounds[2]-bounds[0])/100)))
    assert np.all(data == 2)


@pytest.mark.parametrize('empty', ['eco', 'polygons', 'both'])
def test_raster_empty_or_null_geometries_become_nodata(victim, polygon_inputs, raster_numpy, empty):
    if empty in ['eco', 'both']:
        polygon_inputs.eco.geometry = [None]
    if empty in ['polygons', 'both']:
        polygon_inputs.polygons.geometry = [None, Polygon(), None]
    captured = capture_raster_transfer(victim)
    victim.convert_polys_to_raster('sed_size')
    assert np.all(captured[0][0] == -9999)


def test_raster_failed_transfer_cleans_temp(victim, polygon_inputs, raster_numpy):
    sources = []
    def fail(source, destination):
        sources.append(source)
        assert os.path.exists(source)
        raise OSError('transfer failed')
    victim.filesystem.put_file.side_effect = fail
    with pytest.raises(OSError, match='transfer failed'):
        victim.convert_polys_to_raster('sed_size')
    assert len(sources) == 1 and not pathlib.Path(sources[0]).parent.exists()


# =====================================================================
# Multi-Region Workflow, Validation Ordering & Temporary Database Cleanup
# =====================================================================

@pytest.mark.parametrize('fail_validation', [False, True])
def test_run_regions_order_and_cleanup(victim, fail_validation):
    victim.param_lookup['eco_regions'].value = ['ER_1', 'ER_2']
    steps = ['_resolve_paths', 'download_sediment_data', 'read_sediment_data',
        'add_sed_size_column', 'determine_sed_types', 'add_sediment_mapping_column',
        'create_point_layer', 'transform_points_to_polygons', '_validate_gpkg',
        'convert_polys_to_raster']
    manager, paths = MagicMock(), []
    def create_points():
        paths.append(victim.working_gpkg_path)
        assert victim.working_gpkg_path.parent.is_dir()
        victim.working_gpkg_path.write_bytes(b'unit-test database placeholder')
    with ExitStack() as stack:
        mocks = {name: stack.enter_context(patch.object(victim, name)) for name in steps}
        for name, mock in mocks.items():
            manager.attach_mock(mock, name)
        manager.attach_mock(victim.filesystem.put_file, 'publish')
        mocks['create_point_layer'].side_effect = create_points
        if fail_validation:
            mocks['_validate_gpkg'].side_effect = RuntimeError('validation failed')
            with pytest.raises(RuntimeError, match='validation failed'):
                victim.run()
        else:
            victim.run()
    expected = []
    for idx, region in enumerate(['ER_1'] if fail_validation else ['ER_1', 'ER_2']):
        expected += [call._resolve_paths(region), call.download_sediment_data(),
            call.read_sediment_data(), call.add_sed_size_column(), call.determine_sed_types(),
            call.add_sediment_mapping_column(), call.create_point_layer(),
            call.transform_points_to_polygons(), call._validate_gpkg()]
        if not fail_validation:
            expected += [call.convert_polys_to_raster('sed_type'), call.convert_polys_to_raster('sed_size'),
                call.convert_polys_to_raster('sand_mud_mask'), call.publish(str(paths[idx]), str(victim.gpkg_path))]
    assert manager.mock_calls == expected
    assert all(not path.parent.exists() for path in paths)
    if not fail_validation:
        assert paths[0] != paths[1]  # Each region gets a fresh database.


def test_run_no_regions(victim):
    victim.param_lookup['eco_regions'].value = []
    with patch.object(victim, '_resolve_paths') as resolve:
        victim.run()
    resolve.assert_not_called()
    victim.filesystem.put_file.assert_not_called()
