"""Subgrid tiling tests following the BlueTopo pytest structure.

Real small rasters/Parquet files exercise numerical and spatial behavior.
Shared Engine services, configuration, and S3 access are isolated.
"""
import importlib
import pathlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box
from upath import UPath

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))
module = importlib.import_module('hydro_health.engines.SubgridTilingEngine')
from hydro_health.engines.SubgridTilingEngine import SubgridTilingEngine

TILING_PATH = 'hydro_health.engines.SubgridTilingEngine'
GRID_CRS = CRS.from_epsg(32617)
GRID_TRANSFORM = from_origin(500000, 3400020, 10, 10)
YEAR_PAIRS = [(2004, 2006), (2006, 2010)]


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path, monkeypatch):
    lookup = {
        ('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
        ('MODEL', 'PREDICTION_OUTPUT_DIR'): 'prediction',
        ('MODEL', 'TRAINING_OUTPUT_DIR'): 'training',
        ('MODEL', 'TRAINING_TILES_DIR'): 'training_tiles',
        ('MODEL', 'PREDICTION_TILES_DIR'): 'prediction_tiles',
        ('MODEL', 'SUBGRIDS'): 'subgrids.gpkg',
        ('MODEL', 'TRAINING_SUB_GRIDS_LAYER'): 'training_grid',
        ('MODEL', 'PREDICTION_SUB_GRIDS_LAYER'): 'prediction_grid',
        ('TERRAIN', 'FILLED_DIR'): 'filled_tifs',
        ('TERRAIN', 'COMBINED_LIDAR_DIR'): 'combined_lidar',
    }
    for name in ('SUBGRID_N_WORKERS', 'SUBGRID_WORKER_MEMORY_LIMIT',
                 'SUBGRID_TILE_BATCH_SIZE', 'SUBGRID_GEOTIFF_EVERY'):
        monkeypatch.delenv(name, raising=False)
    with patch(f'{TILING_PATH}.Engine.__init__', return_value=None), \
         patch(f'{TILING_PATH}.Engine.write_message_dask', create=True) as log, \
         patch(f'{TILING_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{TILING_PATH}.Path.home', return_value=tmp_path / 'home'), \
         patch(f'{TILING_PATH}.get_config_item', side_effect=lambda *a: lookup[a]), \
         patch(f'{TILING_PATH}.s3fs.S3FileSystem') as s3:
        yield SimpleNamespace(log=log, s3=s3.return_value, lookup=lookup)


@pytest.fixture
def victim():
    engine = SubgridTilingEngine({
        'env': 'local', 'eco_regions': SimpleNamespace(value=['ER_3'])
    })
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.cleanup_resources = MagicMock()
    engine.log_system_metrics = MagicMock(return_value='test metrics')
    engine.year_ranges = YEAR_PAIRS.copy()
    engine.tile_batch_size = 2
    engine.multiband_geotiff_every = 0
    engine._resolve_paths('ER_3')
    return engine


def write_raster(path, values, nodata=-9999, crs=GRID_CRS,
                 transform=GRID_TRANSFORM, dtype='float32'):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values, dtype=dtype)
    with rasterio.open(path, 'w', driver='GTiff', count=1,
                       height=values.shape[0], width=values.shape[1],
                       dtype=dtype, nodata=nodata, crs=crs,
                       transform=transform) as dst:
        dst.write(values, 1)
    return str(path)


def make_subgrids():
    return gpd.GeoDataFrame({
        'original_tile': ['TILE'], 'tile_id': ['TILE_1'],
        'geometry': [box(500000, 3400000, 500020, 3400020)]
    }, crs=GRID_CRS)


@pytest.fixture
def sub_grid():
    return make_subgrids().iloc[0]


def points_frame():
    # Pixel centers, in top-left, top-right, bottom-left, bottom-right order.
    return pd.DataFrame({'X': [500005., 500015., 500005., 500015.],
                         'Y': [3400015., 3400015., 3400005., 3400005.]})


def tile_kwargs(victim, sub_grid, files, data_type='training'):
    return dict(sub_grid=sub_grid, gridded_files=files, ungridded_files=[],
                static_patterns=victim.static_patterns, is_aws=False,
                output_folder=str(victim.training_tiles_dir / 'TILE_1'),
                data_type=data_type, tile_name='TILE_1',
                local_tmp_dir=str(victim.local_tmp_dir), current_index=1,
                total_count=1, verbose=False, subgrid_crs=GRID_CRS,
                overwrite_outputs=True, year_ranges=YEAR_PAIRS)


# =====================================================================
# Constructor, Paths & Input Discovery
# =====================================================================

@pytest.mark.parametrize('env,cloud', [('local', False), ('remote', True), ('aws', True)])
def test_init_environment_and_scratch(tmp_path, env, cloud):
    engine = SubgridTilingEngine({'env': env}, 'low_res', False)
    assert engine.is_aws is cloud
    assert engine.output_prefix == 'low_res'
    assert engine.overwrite_outputs is False
    assert engine.local_tmp_dir == tmp_path / 'home/hydro_health_local_tmp/subgrid_tmp'
    assert engine.local_tmp_dir.is_dir()
    assert set(engine.static_patterns) == {'sed', 'tsm', 'hurr', 'grain', 'survey'}


@pytest.mark.parametrize('cloud', [False, True])
@pytest.mark.parametrize('prefix', [False, 'low_res'])
def test_resolve_paths(victim, tmp_path, cloud, prefix):
    victim.is_aws = cloud
    victim.output_prefix = prefix
    victim._resolve_paths('ER_2')
    local_root = tmp_path / 'outputs'
    if prefix:
        local_root /= prefix
    local_root /= 'ER_2'
    assert victim.outputs_dir == local_root
    base = 's3://test-bucket/ER_2' if cloud else str(local_root)
    for attr, suffix in [('prediction_out_dir', 'prediction'),
                         ('training_out_dir', 'training'),
                         ('training_tiles_dir', 'training_tiles'),
                         ('prediction_tiles_dir', 'prediction_tiles'),
                         ('filled_out_dir', 'filled_tifs'),
                         ('combined_lidar_dir', 'combined_lidar')]:
        assert str(getattr(victim, attr)) == str(UPath(base) / suffix)
    assert victim.subgrid_paths['training']['layer'] == 'training_grid'
    assert str(victim.subgrid_paths['prediction']['path']) == str(UPath(base) / 'subgrids.gpkg')
    if not cloud:
        assert victim.training_tiles_dir.is_dir()


@pytest.mark.parametrize('fail', [False, True])
def test_load_subgrids_layer_and_failure(victim, fail):
    with patch(f'{TILING_PATH}.gpd.read_file',
               side_effect=OSError('unreadable') if fail else None,
               return_value=make_subgrids()) as read:
        result = victim._load_subgrids('training')
    read.assert_called_once_with(str(victim.subgrid_paths['training']['path']),
                                 layer='training_grid')
    assert (result is None) is fail
    assert victim._load_subgrids('unknown') is None


@pytest.mark.parametrize('data_type', ['training', 'prediction'])
def test_discovery_exact_folders_extensions_and_exclusions(victim, tmp_path, data_type):
    names = ['BlueTopo_TILE_bathy.TIFF', 'bt.uc.tif', 'combined_TILE_2004_filled.tif',
             'bathy_2004.tif', 'grain_size_layer.tif', 'hurr_strength_mean_2004_2006.tif',
             'tsm_mean_2004_2006.tif', 'tsm_cumulative_2004_2006.tif',
             'hurr_count_2004_2006.tif', 'hurr_count_mean_2004_2006.tif',
             'hurr_strength_2004_2006.tif', 'hurr_strength_cumulative_2004_2006.tif',
             'survey_end_date_BlueTopo_TILE.tif', 'unc_other.tif', 'notes.txt',
             'filled_tifs/raw.tif', 'filled_lidar/raw.tif',
             'filled_tifs/combined_TILE_2006_filled.tif', 'almost_filled_tifs/keep.tif']
    root = tmp_path / 'discovery'
    for name in names:
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    found = {pathlib.Path(p).relative_to(root).as_posix() for p in
             victim._get_filtered_raster_files([UPath(root), UPath(root / 'missing')], data_type)}
    common = {'combined_TILE_2004_filled.tif', 'grain_size_layer.tif',
              'hurr_strength_mean_2004_2006.tif', 'tsm_mean_2004_2006.tif',
              'survey_end_date_BlueTopo_TILE.tif',
              'filled_tifs/combined_TILE_2006_filled.tif', 'almost_filled_tifs/keep.tif'}
    expected = common | ({'bathy_2004.tif'} if data_type == 'training' else
                         {'BlueTopo_TILE_bathy.TIFF', 'bt.uc.tif'})
    assert found == expected


def test_partition_uses_complete_tile_tokens(victim):
    grids = pd.DataFrame({'original_tile': ['TILE', 'A.1', None, '']})
    files = ['combined_tile_2004.tif', 'BlueTopo_TILE.tiff', 'x_A.1_slope.tif',
             'combined_TILE2_2004.tif', 'prefixTILE.tif', 'grain.tif']
    gridded, static = victim._partition_raster_files(files, grids)
    assert gridded == files[:3]
    assert static == files[3:]


# =====================================================================
# Column Names & Configured Year Pairs
# =====================================================================

@pytest.mark.parametrize('name,expected', [
    ('BlueTopo_TILE_20240101.tif', 'bt.bathy'),
    ('BlueTopo_TILE_20240101_filled_slope_deg', 'bt.slope_deg'),
    ('bt.uncertainty', 'bt.uc'), ('BlueTopo_TILE_broad_bpi', 'bt.bpi_broad'),
    ('BlueTopo_TILE_flow_dir', 'bt.flowdir'),
    ('combined_TILE_20040101_filled', 'bathy_2004_filled'),
    ('combined2_TILE_2004_filled_plan_curvature', 'curv_plan_2004'),
    ('combined_TILE_2004_2006_filled_gradient_magnitude', 'gradmag_2004_2006'),
    ('combined_TILE_2006_shear_proxy', 'shearproxy_2006'),
    ('bathy_2004_filled', 'bathy_2004_filled'),
    ('bathy_TILE_20040101_slope', 'slope_2004'),
    ('tsm_mean_2004_2006', 'tsm_mean_2004_2006'),
    ('BlueTopo_TILE_survey_end_date', 'survey_end_date'),
    ('grain_size_layer', 'grain_size_layer'),
])
def test_standardize_column_names(name, expected):
    assert module._standardize_col_name(name, 'TILE') == expected


@pytest.mark.parametrize('alias,canonical', list(module.TERRAIN_PRODUCT_ALIASES.items()))
def test_terrain_aliases_have_yearless_prediction_and_yearly_training_names(alias, canonical):
    assert module._standardize_bluetopo_col_name(f'BlueTopo_TILE_2024_{alias}', 'TILE') == f'bt.{canonical}'
    assert module._standardize_combined_col_name(f'combined_TILE_2004_filled_{alias}', 'TILE') == f'{canonical}_2004'


@pytest.mark.parametrize('name,expected', [('tsm_mean_2004_2006', (2004, 2006)),
    ('hurr_1999_2001', (1999, 2001)), ('tsm_2004_2006_extra', None),
    ('grain_size_layer', None), ('x_1800_2006', None)])
def test_extract_year_pair(name, expected):
    assert module._extract_year_pair(name) == expected


def test_available_years_and_pairs_preserve_configured_order():
    df = pd.DataFrame({'bathy_2010_filled': [0, np.nan],
                       'bathy_2004_filled': [1, 2], 'bathy_2006_filled': [np.nan, np.nan],
                       'slope_2022': [1, 2], 'bathy_2022': [1, 2]})
    assert module._available_training_bathy_years(df) == [2004, 2010]
    assert module._available_training_bathy_years(None) == []
    assert module._available_training_bathy_years(pd.DataFrame()) == []
    assert module._available_training_year_pairs([2004, 2006, 2010],
        [(2006, 2010), (2004, 2006), (2006, 2010), (2010, 2022), (2004,)]) == [(2006, 2010), (2004, 2006)]
    assert module._paired_static_variables(YEAR_PAIRS, 'tsm_mean') == ['tsm_mean_2004_2006', 'tsm_mean_2006_2010']


# =====================================================================
# Validity, Spatial Catalogs & Grid Alignment
# =====================================================================

@pytest.mark.parametrize('nodata,expected', [(-9999, [True, True, False, False, False]),
    (np.nan, [True, True, True, False, False]), (None, [True, True, True, False, False])])
def test_valid_mask_preserves_zero_excludes_nonfinite(nodata, expected):
    np.testing.assert_array_equal(module._valid_mask(np.array([0., 2., -9999., np.nan, np.inf]), nodata), expected)


def test_valid_mask_integer_sentinel():
    np.testing.assert_array_equal(module._valid_mask(np.array([0, 1, -9999], dtype='int16'), -9999), [True, True, False])


@pytest.mark.parametrize('cloud,expected', [(False, 's3://bucket/a.tif'), (True, '/vsis3/bucket/a.tif')])
def test_rasterio_path(cloud, expected):
    assert module._to_rasterio_path('s3://bucket/a.tif', cloud) == expected


def test_bounds_transform_and_missing_crs():
    bounds = (-1., -1., 1., 1.)
    assert module._bounds_in_crs(bounds, None, GRID_CRS) == bounds
    assert module._bounds_in_crs(bounds, 'EPSG:4326', 'EPSG:4326') == bounds
    result = module._bounds_in_crs(bounds, 'EPSG:4326', 'EPSG:3857')
    np.testing.assert_allclose(result, [-111319.4908, -111325.1429, 111319.4908, 111325.1429], atol=.01)


@pytest.mark.parametrize('left,right,expected', [(None, None, True), (None, GRID_CRS, False),
    (GRID_CRS, GRID_CRS, True), (GRID_CRS, CRS.from_epsg(4326), False)])
def test_crs_equivalence(left, right, expected):
    assert module._crs_equivalent(left, right) is expected


def test_crs_epsg_fallback_and_unusable_objects():
    left, right = MagicMock(), MagicMock()
    left.to_epsg.return_value = right.to_epsg.return_value = 32617
    assert module._crs_equivalent(left, right)
    left.to_epsg.side_effect = ValueError('bad crs')
    assert not module._crs_equivalent(left, right)


@pytest.mark.parametrize('offset,matches,relaxed', [(0, True, False), (.005, True, False),
    (1.25, True, True), (1.251, False, False)])
def test_transform_alignment_eighth_pixel_boundary(offset, matches, relaxed):
    current = Affine(10, 0, 500000 + offset, 0, -10, 3400020)
    result = module._transform_alignment_details(current, GRID_TRANSFORM)
    assert result[0] is matches and result[3] is relaxed
    assert result[1] == pytest.approx(offset)
    assert result[2] == pytest.approx(1.25)


def test_windows_clip_partial_and_reject_disjoint_or_zero(tmp_path):
    file = write_raster(tmp_path / 'window.tif', [[1, 2], [3, 4]])
    with rasterio.open(file) as src:
        bounds = (499990, 3400000, 500010, 3400020)
        assert module._requested_window(src, bounds) == Window(-1, 0, 2, 2)
        assert module._clipped_window(src, bounds) == Window(0, 0, 1, 2)
        assert module._clipped_window(src, (0, 0, 1, 1)) is None
        assert module._requested_window(src, (500000, 3400000, 500000, 3400020)) is None


def test_spatial_catalog_skips_missing_crs_and_unreadable(tmp_path, isolated_services):
    good = write_raster(tmp_path / 'good.tif', [[1, 2], [3, 4]])
    no_crs = write_raster(tmp_path / 'unknown.tif', [[1]], crs=None)
    catalog = module._build_spatial_catalog([good, no_crs, str(tmp_path / 'missing.tif')], False)
    assert len(catalog) == 1 and catalog[0][0] == good
    assert CRS.from_wkt(catalog[0][2]) == GRID_CRS
    assert module._files_intersecting_tile((500000, 3400000, 500020, 3400020), GRID_CRS, catalog) == [good]
    assert module._files_intersecting_tile((0, 0, 1, 1), GRID_CRS, catalog) == []
    assert isolated_services.log.call_count == 2


def test_spatial_selection_transforms_tile_bounds():
    catalog = [('world.tif', (-120000, -120000, 120000, 120000), 'EPSG:3857'),
               ('invalid.tif', (0, 0, 1, 1), 'invalid-crs')]
    assert module._files_intersecting_tile((-1, -1, 1, 1), 'EPSG:4326', catalog) == ['world.tif']


# =====================================================================
# Real Gridded Raster Reads & Static Sampling
# =====================================================================

@pytest.mark.parametrize('nodata', [-9999, np.nan, None], ids=['sentinel', 'nan', 'unset'])
def test_gridded_union_preserves_zero_and_missing_predictors(tmp_path, sub_grid, nodata):
    missing = np.nan if nodata is None else nodata
    files = [write_raster(tmp_path / 'BlueTopo_TILE_bathy.tif', [[0, missing], [missing, 3]], nodata=nodata),
             write_raster(tmp_path / 'BlueTopo_TILE_slope.tif', [[missing, 2], [missing, 4]], nodata=nodata)]
    df, crs = module._subtile_process_gridded(sub_grid, files, False, GRID_CRS)
    assert crs == GRID_CRS
    np.testing.assert_allclose(df[['X', 'Y']], [[500005, 3400015], [500015, 3400015], [500015, 3400005]])
    np.testing.assert_allclose(df['bt.bathy'], [0, np.nan, 3], equal_nan=True)
    np.testing.assert_allclose(df['bt.slope'], [np.nan, 2, 4], equal_nan=True)


def test_gridded_duplicate_columns_fill_gaps_without_overwriting(tmp_path, sub_grid):
    files = [write_raster(tmp_path / 'a/BlueTopo_TILE_bathy.tif', [[1, -9999], [-9999, 4]]),
             write_raster(tmp_path / 'b/BlueTopo_TILE_bathy.tif', [[99, 2], [3, 99]])]
    df, _ = module._subtile_process_gridded(sub_grid, files, False, GRID_CRS)
    np.testing.assert_allclose(df['bt.bathy'], [1, 2, 3, 4])


def test_gridded_partial_first_raster_keeps_full_tile_coordinates(tmp_path, sub_grid):
    files = [write_raster(tmp_path / 'BlueTopo_TILE_bathy.tif', [[1], [3]]),
             write_raster(tmp_path / 'BlueTopo_TILE_slope.tif', [[10, 20], [30, 40]])]
    df, _ = module._subtile_process_gridded(sub_grid, files, False, GRID_CRS)
    np.testing.assert_allclose(df['bt.bathy'], [1, np.nan, 3, np.nan], equal_nan=True)
    np.testing.assert_allclose(df['bt.slope'], [10, 20, 30, 40])
    assert len(df) == 4


def test_gridded_coarser_raster_resampled_nearest(tmp_path, sub_grid):
    files = [write_raster(tmp_path / 'BlueTopo_TILE_bathy.tif', [[1, 2], [3, 4]]),
             write_raster(tmp_path / 'BlueTopo_TILE_slope.tif', [[7]],
                          transform=from_origin(500000, 3400020, 20, 20))]
    df, _ = module._subtile_process_gridded(sub_grid, files, False, GRID_CRS)
    np.testing.assert_allclose(df['bt.slope'], [7, 7, 7, 7])


def test_gridded_skips_bad_sources_and_other_tiles(tmp_path, sub_grid):
    files = [str(tmp_path / 'bad_TILE.tif'),
             write_raster(tmp_path / 'BlueTopo_TILE2_bathy.tif', [[9, 9], [9, 9]]),
             write_raster(tmp_path / 'BlueTopo_TILE_bathy.tif', [[1, 2], [3, 4]]),
             write_raster(tmp_path / 'BlueTopo_TILE_slope.tif', [[9, 9], [9, 9]], crs=None),
             write_raster(tmp_path / 'BlueTopo_TILE_tci.tif', [[9, 9], [9, 9]], crs='EPSG:3857')]
    df, _ = module._subtile_process_gridded(sub_grid, files, False, GRID_CRS)
    assert list(df.columns) == ['X', 'Y', 'bt.bathy']
    np.testing.assert_allclose(df['bt.bathy'], [1, 2, 3, 4])


@pytest.mark.parametrize('files_kind', ['empty', 'all_nodata', 'disjoint'])
def test_gridded_no_usable_data(tmp_path, sub_grid, files_kind):
    files = [] if files_kind == 'empty' else [write_raster(
        tmp_path / 'BlueTopo_TILE_bathy.tif', [[-9999]],
        transform=GRID_TRANSFORM if files_kind == 'all_nodata' else from_origin(0, 1, 1, 1))]
    df, _ = module._subtile_process_gridded(sub_grid, files, False, GRID_CRS)
    assert df.empty


def test_static_sampling_nodata_pair_mask_and_outside_points(tmp_path, sub_grid):
    file = write_raster(tmp_path / 'grain.tif', [[0, 2], [3, -9999]])
    with rasterio.open(file) as src:
        values = module._sample_ungridded_raster(src, sub_grid.geometry.bounds,
            GRID_CRS, GRID_CRS, np.array([500005, 500015, 500005, 500015, 600000]),
            np.array([3400015, 3400015, 3400005, 3400005, 3400015]),
            np.array([True, False, True, True, True]))
    np.testing.assert_allclose(values, [0, np.nan, 3, np.nan, np.nan], equal_nan=True)


def test_static_sampling_transforms_point_crs(tmp_path):
    file = write_raster(tmp_path / 'global.tif', [[1, 2], [3, 4]],
                        crs='EPSG:3857', transform=from_origin(-200000, 200000, 200000, 200000))
    with rasterio.open(file) as src:
        values = module._sample_ungridded_raster(src, tuple(src.bounds),
            'EPSG:3857', 'EPSG:4326', np.array([-.5, .5]), np.array([.5, -.5]), None)
        assert module._sample_ungridded_raster(src, (100, 50, 101, 51),
            'EPSG:4326', 'EPSG:4326', np.array([100]), np.array([50]), None) is None
    np.testing.assert_allclose(values, [1, 4])


def test_ungridded_training_pair_filter_and_missing_bathy(tmp_path, sub_grid):
    df = points_frame()
    df['bathy_2004_filled'] = [0, np.nan, 2, 3]
    df['bathy_2006_filled'] = [1, 2, np.nan, 4]
    files = [write_raster(tmp_path / 'tsm_mean_2004_2006.tif', [[10, 20], [30, 40]]),
             write_raster(tmp_path / 'tsm_mean_2006_2010.tif', [[99, 99], [99, 99]]),
             write_raster(tmp_path / 'grain_size_layer.tif', [[0, 2], [3, 4]])]
    out = module._subtile_process_ungridded(sub_grid, files, df, ['tsm', 'mean', 'grain'],
        False, GRID_CRS, GRID_CRS, 'training', [(2004, 2006)])
    np.testing.assert_allclose(out['tsm_mean_2004_2006'], [10, np.nan, np.nan, 40], equal_nan=True)
    assert 'tsm_mean_2006_2010' not in out
    np.testing.assert_allclose(out['grain_size_layer'], [0, 2, 3, 4])
    assert module._subtile_process_ungridded(sub_grid, files, pd.DataFrame(), ['tsm'],
        False, GRID_CRS, GRID_CRS, 'training').empty


def test_ungridded_duplicate_fills_only_missing_and_tolerates_errors(tmp_path, sub_grid):
    df = points_frame()
    files = [str(tmp_path / 'bad_grain.tif'),
             write_raster(tmp_path / 'a/grain_size_layer.tif', [[1, -9999], [-9999, 4]]),
             write_raster(tmp_path / 'b/grain_size_layer.tif', [[99, 2], [3, 99]])]
    out = module._subtile_process_ungridded(sub_grid, files, df, ['grain'], False,
        GRID_CRS, GRID_CRS, 'prediction')
    np.testing.assert_allclose(out['grain_size_layer'], [1, 2, 3, 4])


# =====================================================================
# Deltas, Output Schema & NaN Statistics
# =====================================================================


def test_training_delta_newer_minus_older_and_nan_propagation():
    df = points_frame()
    df['bathy_2004_filled'] = [0, -10, np.nan, 4]
    df['bathy_2006_filled'] = [1, -8, 3, np.nan]
    df['unexpected_2010_2022'] = 99
    years = module._add_training_deltas(df, [(2004, 2006), (2010, 2022)], 'TILE_1')
    assert years == [2004, 2006, 2010, 2022]
    np.testing.assert_allclose(df['delta_bathy_2004_2006'], [1, 2, np.nan, np.nan], equal_nan=True)
    assert 'delta_bathy_2010_2022' not in df and 'unexpected_2010_2022' not in df


def test_prediction_schema_order_optional_columns_and_identifiers():
    df = points_frame()
    df['bt.bathy'] = [0, 1, np.nan, 3]
    df['tsm_mean_2004_2006'] = [1, np.nan, 3, 4]
    df['hurr_strength_mean_2004_2006'] = np.nan
    df['unexpected'] = 99
    out = module._prepare_combined_output(df, 'prediction', 'TILE_1', [], YEAR_PAIRS)
    expected_variables = [f'bt.{v}' for v in module.PREDICTION_BT_VARIABLES]
    assert list(out.columns) == ['X', 'Y', *expected_variables, 'grain_size_layer',
        'prim_sed_layer', 'survey_end_date', 'tsm_mean_2004_2006', 'FID', 'tile_id']
    assert out['bt.uc'].isna().all()
    assert out['bt.bathy'].iloc[0] == 0 and len(out) == 4
    assert out['tile_id'].tolist() == ['TILE_1'] * 4
    assert out['FID'].tolist() == [0, 1, 2, 3]


def test_training_schema_retains_missing_predictor_rows_and_configured_pairs():
    df = points_frame()
    df['bathy_2004_filled'] = [0, 1, 2, np.nan]
    df['bathy_2006_filled'] = [1, 3, np.nan, 5]
    df['tsm_mean_2004_2006'] = [2, 2, np.nan, np.nan]
    out = module._prepare_combined_output(df, 'training', 'TILE_1', [(2004, 2006)], YEAR_PAIRS)
    assert list(out.columns[:4]) == ['X', 'Y', 'bathy_2004_filled', 'bathy_2006_filled']
    assert list(out.columns[-3:]) == ['delta_bathy_2004_2006', 'FID', 'tile_id']
    assert 'bathy_2010_filled' not in out and 'tsm_mean_2006_2010' not in out
    assert out['slope_2004'].isna().all() and len(out) == 4
    np.testing.assert_allclose(out['delta_bathy_2004_2006'], [1, 2, np.nan, np.nan], equal_nan=True)


def test_nan_statistics_empty_and_multiple_pairs():
    assert module._create_nan_stats_csv(pd.DataFrame(), 'T').empty
    df = pd.DataFrame({'delta_bathy_2004_2006': [0, np.nan, 2],
                       'delta_bathy_2006_2010': [np.nan] * 3, 'other': [1, 2, 3]})
    assert module._create_nan_stats_csv(df, 'T').to_dict('records') == [{
        'tile_id': 'T', '2004_2006_nan_percent': 33.33, '2006_2010_nan_percent': 100.0}]


@pytest.mark.parametrize('kind', ['delta', 'no_delta', 'empty'])
@pytest.mark.parametrize('cloud', [False, True])
def test_existing_parquet_nan_statistics(tmp_path, isolated_services, kind, cloud):
    path = tmp_path / 'existing.parquet'
    df = pd.DataFrame({'delta_bathy_2004_2006': [0, np.nan], 'other': [1, 2]})
    if kind == 'no_delta':
        df = df.drop(columns='delta_bathy_2004_2006')
    elif kind == 'empty':
        df = df.iloc[:0]
    df.to_parquet(path, index=False)
    if cloud:
        isolated_services.s3.open.side_effect = lambda *a, **k: path.open('rb')
    result = module._read_existing_nan_stats(UPath(path), 'T', cloud)
    if kind == 'empty':
        assert result.empty
    else:
        expected = {'tile_id': 'T'}
        if kind == 'delta':
            expected['2004_2006_nan_percent'] = 50.0
        assert result.to_dict('records') == [expected]


# =====================================================================
# Parquet Writes & Complete Worker Outputs
# =====================================================================

@pytest.mark.parametrize('cloud', [False, True])
def test_save_combined_data_real_parquet_and_cleanup(victim, tmp_path, isolated_services, cloud):
    df = points_frame()
    df['bathy_2004_filled'] = [0, 1, np.nan, 3]
    df['bathy_2006_filled'] = [1, 3, 4, np.nan]
    output = tmp_path / 'saved'
    # Cloud upload is represented by copying bytes to a local destination.
    if cloud:
        output.mkdir()
        isolated_services.s3.put.side_effect = lambda src, dst: pathlib.Path(dst).write_bytes(pathlib.Path(src).read_bytes())
    stats = module._save_combined_data(df, str(output), 'training', 'T', cloud,
        str(victim.local_tmp_dir), 1, 1, False, [(2004, 2006)], YEAR_PAIRS)
    stored = pd.read_parquet(output / 'T_training_clipped_data.parquet')
    np.testing.assert_allclose(stored['delta_bathy_2004_2006'], [1, 2, np.nan, np.nan], equal_nan=True)
    assert len(stored) == 4 and stored['bathy_2004_filled'].iloc[0] == 0
    assert stats['2004_2006_nan_percent'].iloc[0] == 50
    assert not list(victim.local_tmp_dir.glob('*.parquet'))
    if cloud:
        isolated_services.s3.put.assert_called_once()


def test_save_empty_skips_and_write_failure_cleans_temp(victim, tmp_path):
    assert module._save_combined_data(pd.DataFrame(), str(tmp_path / 'out'),
        'prediction', 'T', False, str(victim.local_tmp_dir), 1, 1, False).empty
    with patch(f'{TILING_PATH}.shutil.copy2', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            module._save_combined_data(points_frame(), str(tmp_path / 'out'),
                'prediction', 'T', False, str(victim.local_tmp_dir), 1, 1, False)
    assert not list(victim.local_tmp_dir.glob('*.parquet'))


@pytest.mark.parametrize('kind', ['training', 'prediction'])
def test_process_tile_real_end_to_end(victim, tmp_path, sub_grid, kind):
    if kind == 'training':
        files = [write_raster(tmp_path / 'combined_TILE_2004_filled.tif', [[0, 1], [2, -9999]]),
                 write_raster(tmp_path / 'combined_TILE_2006_filled.tif', [[1, 3], [-9999, 5]])]
    else:
        files = [write_raster(tmp_path / 'BlueTopo_TILE_bathy.tif', [[0, 1], [2, -9999]])]
    kwargs = tile_kwargs(victim, sub_grid, files, kind)
    kwargs['ungridded_files'] = [write_raster(tmp_path / 'tsm_mean_2004_2006.tif', [[10, 20], [30, 40]])]
    stats = module._process_tile(**kwargs)
    out_path = pathlib.Path(kwargs['output_folder']) / f'TILE_1_{kind}_clipped_data.parquet'
    data = pd.read_parquet(out_path)
    assert stats.attrs['created_parquet'] is True
    assert stats.attrs['parquet_path'] == str(out_path)
    assert CRS.from_wkt(stats.attrs['raster_crs']) == GRID_CRS
    assert not list(victim.local_tmp_dir.glob('*.parquet'))
    if kind == 'training':
        np.testing.assert_allclose(data['delta_bathy_2004_2006'], [1, 2, np.nan, np.nan], equal_nan=True)
        np.testing.assert_allclose(data['tsm_mean_2004_2006'], [10, 20, np.nan, np.nan], equal_nan=True)
        assert stats['2004_2006_nan_percent'].iloc[0] == 50
    else:
        np.testing.assert_allclose(data['bt.bathy'], [0, 1, 2])
        np.testing.assert_allclose(data['tsm_mean_2004_2006'], [10, 20, 30])


@pytest.mark.parametrize('data_type,files,expected', [('prediction', [], False),
    ('training', ['BlueTopo_TILE.tif'], False), ('training', ['combined_TILE.tif'], True),
    ('prediction', ['bt.bathy.tif'], True)])
def test_required_inputs(data_type, files, expected):
    assert module._has_required_tile_input(files, data_type, 'T', 1, 1) is expected


def test_worker_existing_output_skips_source_reads(victim, sub_grid):
    kwargs = tile_kwargs(victim, sub_grid, [])
    path = pathlib.Path(kwargs['output_folder']) / 'TILE_1_training_clipped_data.parquet'
    path.parent.mkdir(parents=True)
    pd.DataFrame({'delta_bathy_2004_2006': [0, np.nan]}).to_parquet(path)
    kwargs['overwrite_outputs'] = False
    with patch(f'{TILING_PATH}._subtile_process_gridded') as read:
        result = module._process_tile(**kwargs)
    read.assert_not_called()
    assert result['2004_2006_nan_percent'].iloc[0] == 50
    assert not result.attrs.get('created_parquet', False)


@pytest.mark.parametrize('failure', ['no_input', 'no_pair', 'read_error', 'corrupt_existing'])
def test_worker_skips_or_logs_unusable_inputs(victim, tmp_path, sub_grid, failure):
    files = [] if failure == 'no_input' else [write_raster(tmp_path / 'combined_TILE_2004_filled.tif', [[1, 2], [3, 4]])]
    kwargs = tile_kwargs(victim, sub_grid, files)
    if failure == 'corrupt_existing':
        path = pathlib.Path(kwargs['output_folder']) / 'TILE_1_training_clipped_data.parquet'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'not parquet')
        kwargs['overwrite_outputs'] = False
    if failure == 'read_error':
        with patch(f'{TILING_PATH}._subtile_process_gridded', side_effect=OSError('broken')):
            result = module._process_tile(**kwargs)
    else:
        result = module._process_tile(**kwargs)
    assert result.empty


# =====================================================================
# Multiband GeoTIFF Reconstruction & Export
# =====================================================================

@pytest.mark.parametrize('kind', ['empty', 'missing_xy', 'no_bands', 'nonfinite_xy'])
def test_prepare_geotiff_rejects_unusable_tables(tmp_path, kind):
    df = points_frame()
    df['value'] = [0, 1, 2, 3]
    if kind == 'empty':
        df = df.iloc[:0]
    elif kind == 'missing_xy':
        df = df.drop(columns='X')
    elif kind == 'no_bands':
        df = df.drop(columns='value')
        df['tile_id'] = 'T'
    else:
        df['X'] = np.nan
    path = tmp_path / 'invalid.parquet'
    df.to_parquet(path)
    assert module._prepare_geotiff_dataframe(str(path), False) is None


def test_prepare_geotiff_selects_numeric_bands_and_finite_coordinates(tmp_path, isolated_services):
    df = points_frame()
    df['value'] = [0, 1, 2, np.nan]
    df['FID'] = [0, 1, 2, 3]
    df['tile_id'] = 'T'
    df['label'] = 'text'
    df.loc[1, 'X'] = np.inf
    path = tmp_path / 'data.parquet'
    df.to_parquet(path)
    isolated_services.s3.open.side_effect = lambda *a, **k: path.open('rb')
    out, columns = module._prepare_geotiff_dataframe(str(path), True)
    assert columns == ['value'] and len(out) == 3
    assert 1 not in out.index


def test_grid_reconstruction_unsorted_sparse_rows_and_pixel_centers():
    df = points_frame().iloc[[3, 0, 1]].copy()
    height, width, affine, rows, cols = module._reconstruct_geotiff_grid(df, 'tile.parquet')
    assert (height, width) == (2, 2)
    assert affine == GRID_TRANSFORM
    np.testing.assert_array_equal(rows, [1, 0, 0])
    np.testing.assert_array_equal(cols, [1, 0, 1])


@pytest.mark.parametrize('kind', ['one_x', 'one_y', 'irregular'])
def test_grid_reconstruction_rejects_uninferable_grid(kind):
    if kind == 'irregular':
        df = pd.DataFrame({'X': [0, 10, 25], 'Y': [20, 10, 0]})
    else:
        df = points_frame()
        df['X' if kind == 'one_x' else 'Y'] = 1
    assert module._reconstruct_geotiff_grid(df, 'bad.parquet') is None


def test_multiband_geotiff_round_trip_sparse_values_metadata_and_cleanup(victim, tmp_path):
    df = points_frame().iloc[[3, 0, 1]].copy()
    df['bathy'] = [4, 0, np.nan]
    df['tsm'] = [40, 10, 20]
    df['FID'] = [3, 0, 1]
    df['tile_id'] = 'T'
    df['label'] = 'ignored'
    path = tmp_path / 'tile.parquet'
    df.to_parquet(path, index=False)
    result = module._create_multiband_geotiff_from_parquet(str(path), GRID_CRS,
        False, str(victim.local_tmp_dir), 7, True)
    assert result == str(tmp_path / 'tile_all_variables.tif')
    with rasterio.open(result) as src:
        assert src.count == 2 and src.descriptions == ('bathy', 'tsm')
        assert src.crs == GRID_CRS and src.transform == GRID_TRANSFORM
        assert src.dtypes == ('float32', 'float32') and np.isnan(src.nodata)
        assert src.compression.value.lower() == 'lzw'
        assert src.tags()['created_parquet_sequence'] == '7'
        assert src.tags()['source_parquet'] == str(path)
        np.testing.assert_allclose(src.read(1), [[0, np.nan], [np.nan, 4]], equal_nan=True)
        np.testing.assert_allclose(src.read(2), [[10, 20], [np.nan, 40]], equal_nan=True)
    assert not list(victim.local_tmp_dir.glob('*.tif'))


def test_geotiff_existing_skip_then_overwrite(victim, tmp_path):
    path = tmp_path / 'tile.parquet'
    output = tmp_path / 'tile_all_variables.tif'
    output.write_bytes(b'existing')
    assert module._create_multiband_geotiff_from_parquet(str(path), None, False,
        str(victim.local_tmp_dir), 1, False) == str(output)
    assert output.read_bytes() == b'existing'
    df = points_frame()
    df['value'] = [0, 1, 2, 3]
    df.to_parquet(path)
    result = module._create_multiband_geotiff_from_parquet(str(path), GRID_CRS,
        False, str(victim.local_tmp_dir), 1, True)
    with rasterio.open(result) as src:
        np.testing.assert_array_equal(src.read(1), [[0, 1], [2, 3]])


def test_geotiff_missing_crs_and_publish_failure_cleanup(victim, tmp_path):
    df = points_frame()
    df['value'] = [1, 2, 3, 4]
    path = tmp_path / 'tile.parquet'
    df.to_parquet(path)
    args = (str(path), GRID_CRS, False, str(victim.local_tmp_dir), 1, True)
    assert module._create_multiband_geotiff_from_parquet(str(path), None, False,
        str(victim.local_tmp_dir), 1, True) is None
    with patch(f'{TILING_PATH}._publish_multiband_geotiff', side_effect=OSError('failed upload')):
        assert module._create_multiband_geotiff_from_parquet(*args) is None
    assert not list(victim.local_tmp_dir.glob('*.tif'))
    assert not (tmp_path / 'tile_all_variables.tif').exists()


def test_publish_geotiff_cloud_delegates_upload(isolated_services):
    module._publish_multiband_geotiff('local.tif', 's3://test-bucket/out.tif', True)
    isolated_services.s3.put.assert_called_once_with('local.tif', 's3://test-bucket/out.tif')


# =====================================================================
# Dask Batching, Pipeline & Run Lifecycle
# =====================================================================

@pytest.mark.parametrize('every,expected', [(0, []), (1, [1, 2, 3, 4, 5]), (2, [1, 3, 5])])
def test_collect_batches_real_dask_sampling_counts_new_outputs_only(victim, every, expected):
    victim.multiband_geotiff_every = every
    tasks = []
    for created in [True, False, True, True, True, True]:
        result = pd.DataFrame({'tile_id': ['T']})
        result.attrs.update(created_parquet=created, parquet_path='test.parquet', raster_crs='EPSG:32617')
        tasks.append(dask.delayed(lambda value: value)(result))
    tasks.extend([dask.delayed(lambda: None)(), dask.delayed(pd.DataFrame)()])
    collected = []
    # Real delayed tasks and compute; synchronous execution avoids starting workers.
    with dask.config.set(scheduler='synchronous'), \
         patch(f'{TILING_PATH}._create_multiband_geotiff_from_parquet') as export:
        count = victim._collect_tile_batch(tasks[:3], collected, 0)
        count = victim._collect_tile_batch(tasks[3:], collected, count)
    assert count == 5 and len(collected) == 6
    assert [c.kwargs['created_sequence'] for c in export.call_args_list] == expected
    assert victim._collect_tile_batch([], collected, count) == 5


def test_submit_batches_tile_selection_payload_and_remainder(victim, tmp_path):
    grids = pd.concat([make_subgrids()] * 3, ignore_index=True)
    grids = gpd.GeoDataFrame(grids, geometry='geometry', crs=GRID_CRS)
    grids['tile_id'] = ['T1', 'T2', 'T3']
    grids.loc[2, 'original_tile'] = 'OTHER'
    files = ['BlueTopo_TILE.tif', 'BlueTopo_OTHER.tif', 'BlueTopo_TILE2.tif']
    result = pd.DataFrame({'tile_id': ['T']})
    with patch(f'{TILING_PATH}._files_intersecting_tile', return_value=['grain.tif']), \
         patch(f'{TILING_PATH}._process_tile', return_value=result) as worker, \
         patch.object(victim, '_collect_tile_batch', wraps=victim._collect_tile_batch) as collect, \
         dask.config.set(scheduler='synchronous'):
        out = victim._submit_tile_batches(grids, files, [], UPath(tmp_path), 'prediction', False)
    assert len(out) == 3 and collect.call_count == 2
    by_tile = {c.kwargs['tile_name']: c.kwargs for c in worker.call_args_list}
    assert by_tile['T1']['gridded_files'] == ['BlueTopo_TILE.tif']
    assert by_tile['T3']['gridded_files'] == ['BlueTopo_OTHER.tif']
    assert by_tile['T2']['current_index'] == 2 and by_tile['T2']['total_count'] == 3
    assert by_tile['T2']['ungridded_files'] == ['grain.tif']
    assert by_tile['T2']['year_ranges'] == YEAR_PAIRS


@pytest.mark.parametrize('kind', ['none', 'empty', 'missing_crs'])
def test_pipeline_rejects_unusable_subgrids(victim, kind):
    grids = None if kind == 'none' else make_subgrids()
    if kind == 'empty':
        grids = grids.iloc[:0]
    elif kind == 'missing_crs':
        grids = gpd.GeoDataFrame(grids.drop(columns='geometry'), geometry=list(grids.geometry))
    with patch.object(victim, '_load_subgrids', return_value=grids), \
         patch.object(victim, '_get_filtered_raster_files') as discover:
        victim._process_pipeline([], victim.training_tiles_dir, 'training')
    discover.assert_not_called()


def test_pipeline_catalog_static_only_and_statistics_csv(victim):
    stats = pd.DataFrame({'tile_id': ['T'], '2004_2006_nan_percent': [50.]})
    with patch.object(victim, '_load_subgrids', return_value=make_subgrids()), \
         patch.object(victim, '_get_filtered_raster_files', return_value=['BlueTopo_TILE.tif', 'grain.tif', 'unrelated.tif']), \
         patch(f'{TILING_PATH}._build_spatial_catalog', return_value=['catalog']) as catalog, \
         patch.object(victim, '_submit_tile_batches', return_value=[stats]) as submit:
        victim._process_pipeline([], victim.training_tiles_dir, 'training')
    catalog.assert_called_once_with(['grain.tif'], False)
    assert submit.call_args.args[2] == ['catalog']
    path = victim.training_tiles_dir.parent / 'year_pair_nan_counts_training.csv'
    pd.testing.assert_frame_equal(pd.read_csv(path), stats)


@pytest.mark.parametrize('failure', [None, 'setup', 'pipeline'])
def test_run_region_pipeline_order_and_always_cleanup(victim, failure):
    victim.param_lookup['eco_regions'].value = ['ER_1', 'ER_2']
    if failure == 'setup':
        victim.setup_dask.side_effect = RuntimeError('setup failed')
    with patch.object(victim, '_process_pipeline',
                      side_effect=RuntimeError('pipeline failed') if failure == 'pipeline' else None) as pipeline:
        if failure:
            with pytest.raises(RuntimeError, match='failed'):
                victim.run()
        else:
            victim.run()
            assert [c.kwargs['data_type'] for c in pipeline.call_args_list] == ['prediction', 'training', 'prediction', 'training']
            assert 'ER_1' in str(pipeline.call_args_list[0].kwargs['output_dir'])
            assert 'ER_2' in str(pipeline.call_args_list[2].kwargs['output_dir'])
    victim.cleanup_resources.assert_called_once_with(module.OUTPUTS)
    victim.setup_dask.assert_called_once_with('local', n_workers=16, threads_per_worker=2, memory_limit='7GB')


def test_run_environment_limits_and_batch_clamping(victim, monkeypatch):
    monkeypatch.setenv('SUBGRID_N_WORKERS', '0')
    monkeypatch.setenv('SUBGRID_TILE_BATCH_SIZE', '0')
    monkeypatch.setenv('SUBGRID_GEOTIFF_EVERY', '-1')
    monkeypatch.setenv('SUBGRID_WORKER_MEMORY_LIMIT', '2GB')
    with patch.object(victim, '_process_pipeline'):
        victim.run()
    assert victim.tile_batch_size == 1 and victim.multiband_geotiff_every == 0
    victim.setup_dask.assert_called_once_with('local', n_workers=1, threads_per_worker=2, memory_limit='2GB')


# =====================================================================
# Additional Regression Checks & Known Engine Issue
# =====================================================================


def test_static_sampling_preserves_centers_inside_nonaligned_bounds(tmp_path):
    """Both centers lie in the AOI; window rounding must not drop either."""
    file = write_raster(tmp_path / 'coarse_static.tif', [[1, 2], [3, 4]],
                        transform=from_origin(0, 20, 10, 10))
    with rasterio.open(file) as src:
        values = module._sample_ungridded_raster(
            src, (4, 4, 16, 16), GRID_CRS, GRID_CRS,
            np.array([5., 15.]), np.array([15., 5.]), None,
        )
    np.testing.assert_allclose(values, [1, 4])


def test_training_pairs_selection_and_prediction_no_pairs():
    df = pd.DataFrame({'bathy_2004_filled': [0], 'bathy_2006_filled': [1]})
    assert module._training_year_pairs_for_tile(df, 'training', YEAR_PAIRS, 'T', 1, 1) == [(2004, 2006)]
    assert module._training_year_pairs_for_tile(df, 'prediction', YEAR_PAIRS, 'T', 1, 1) == []


def test_gridded_intersecting_different_crs_skipped(tmp_path, sub_grid, isolated_services):
    bounds = module._bounds_in_crs(sub_grid.geometry.bounds, GRID_CRS, 'EPSG:3857')
    left, bottom, right, top = bounds
    files = [write_raster(tmp_path / 'BlueTopo_TILE_bathy.tif', [[1, 2], [3, 4]]),
             write_raster(tmp_path / 'BlueTopo_TILE_slope.tif', [[9, 9], [9, 9]],
                          crs='EPSG:3857',
                          transform=from_origin(left, top, (right-left)/2, (top-bottom)/2))]
    df, _ = module._subtile_process_gridded(sub_grid, files, False, GRID_CRS)
    assert list(df.columns) == ['X', 'Y', 'bt.bathy']
    assert any('different CRS' in str(c.args[0]) for c in isolated_services.log.call_args_list)


def test_worker_overwrite_replaces_existing_parquet(victim, tmp_path, sub_grid):
    files = [write_raster(tmp_path / 'BlueTopo_TILE_bathy.tif', [[0, 1], [2, 3]])]
    kwargs = tile_kwargs(victim, sub_grid, files, 'prediction')
    path = pathlib.Path(kwargs['output_folder']) / 'TILE_1_prediction_clipped_data.parquet'
    path.parent.mkdir(parents=True)
    pd.DataFrame({'old': [99]}).to_parquet(path)
    result = module._process_tile(**kwargs)
    assert result.attrs['created_parquet'] is True
    data = pd.read_parquet(path)
    assert 'old' not in data
    np.testing.assert_array_equal(data['bt.bathy'], [0, 1, 2, 3])


def test_ungridded_skips_pair_with_no_overlapping_bathy(tmp_path, sub_grid):
    df = points_frame()
    df['bathy_2004_filled'] = [1, 1, np.nan, np.nan]
    df['bathy_2006_filled'] = [np.nan, np.nan, 2, 2]
    file = write_raster(tmp_path / 'tsm_mean_2004_2006.tif', [[9, 9], [9, 9]])
    with patch(f'{TILING_PATH}._sample_ungridded_raster') as sample:
        out = module._subtile_process_ungridded(sub_grid, [file], df, ['tsm'],
            False, GRID_CRS, GRID_CRS, 'training', [(2004, 2006)])
    sample.assert_not_called()
    assert 'tsm_mean_2004_2006' not in out


def test_geotiff_unusable_parquet_skips_without_output(victim, tmp_path):
    path = tmp_path / 'bad.parquet'
    pd.DataFrame({'X': [1], 'Y': [1], 'value': [2]}).to_parquet(path)
    assert module._create_multiband_geotiff_from_parquet(str(path), GRID_CRS,
        False, str(victim.local_tmp_dir), 1, True) is None
    pd.DataFrame({'no_coordinates': [1]}).to_parquet(path)
    assert module._create_multiband_geotiff_from_parquet(str(path), GRID_CRS,
        False, str(victim.local_tmp_dir), 1, True) is None
    assert not list(victim.local_tmp_dir.glob('*.tif'))


@pytest.mark.parametrize('every', [1, 3])
def test_pipeline_without_results_does_not_write_statistics(victim, tmp_path, every):
    victim.multiband_geotiff_every = every
    with patch.object(victim, '_load_subgrids', return_value=make_subgrids()), \
         patch.object(victim, '_get_filtered_raster_files', return_value=[]), \
         patch.object(victim, '_submit_tile_batches', return_value=[]):
        victim._process_pipeline([], UPath(tmp_path / 'tiles'), 'prediction')
    assert not (tmp_path / 'year_pair_nan_counts_prediction.csv').exists()
