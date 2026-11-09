"""Unit tests for CreateTSMLayerEngine; follows test_BlueTopoEngine.py.

Place beside the existing BlueTopo test. External FTP/S3/configuration are
mocked; numerical calculations and small GeoTIFF round trips are real.
"""
import os
import sys
import pathlib
from contextlib import contextmanager
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pytest
import rasterio
import xarray as xr

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.CreateTSMLayerEngine import (
    CreateTSMLayerEngine, HydroHealthConfig, HydroHealthCredentialsError,
)

TSM_PATH = 'hydro_health.engines.CreateTSMLayerEngine'


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture(autouse=True)
def isolated_services(tmp_path):
    """Prevent real credentials, base initialization, FTP, or S3 access."""
    lookup = {
        ('TSM', 'DATA_PATH'): 'tsm_nc',
        ('TSM', 'SUBFOLDER'): 'tsm',
        ('SHARED', 'OUTPUT_BUCKET'): 'test-bucket',
    }
    with patch(f'{TSM_PATH}.Engine.__init__', return_value=None), \
         patch(f'{TSM_PATH}.INPUTS', tmp_path / 'inputs'), \
         patch(f'{TSM_PATH}.OUTPUTS', tmp_path / 'outputs'), \
         patch(f'{TSM_PATH}.get_config_item', side_effect=lambda *args: lookup[args]), \
         patch(f'{TSM_PATH}.get_environment', return_value='local'), \
         patch(f'{TSM_PATH}.FTP') as ftp, \
         patch(f'{TSM_PATH}.s3fs.S3FileSystem') as s3:
        yield ftp, s3


@pytest.fixture
def victim():
    """Provide a real engine with fake credentials and isolated paths."""
    with patch(f'{TSM_PATH}.HydroHealthConfig') as creds:
        creds.return_value.username = 'test-user'
        creds.return_value.password = 'test-password'
        return CreateTSMLayerEngine()


def create_credentials(tmp_path, text):
    """Write dummy credentials under the patched INPUTS directory."""
    path = tmp_path / 'inputs' / 'lookups' / 'hydro_health.config'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


def tsm_name(date='20040101', sensor='TEST'):
    return f'{sensor}_L3m_{date}-20040108_TSM_8D_4.nc'


def create_test_raster(path, data):
    """Create a tiny georeferenced raster for year-pair arithmetic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(data, dtype=np.float32)
    with rasterio.open(
        path, 'w', driver='GTiff', height=array.shape[0], width=array.shape[1],
        count=1, dtype='float32', crs='EPSG:4326',
        transform=rasterio.transform.from_origin(-80, 30, 0.1, 0.1),
        nodata=np.nan,
    ) as dst:
        dst.write(array, 1)
    return path


@contextmanager
def dataset_for(data):
    """Use real xarray indexing without depending on a NetCDF backend."""
    with xr.Dataset({'TSM_mean': (('lat', 'lon'), np.asarray(data))}) as ds:
        yield ds


# =====================================================================
# Credentials & Engine Initialization
# =====================================================================

def test_credentials_load_comments_whitespace_and_equals(tmp_path):
    create_credentials(tmp_path, '# Dummy values only\n\n username = tester \npassword = a=b=c\nignored\n')
    config = HydroHealthConfig()
    assert config.username == 'tester'
    assert config.password == 'a=b=c'


def test_credentials_missing_file(tmp_path):
    with pytest.raises(HydroHealthCredentialsError, match='Config file not found'):
        HydroHealthConfig()


@pytest.mark.parametrize('text,missing', [
    ('username=tester\n', 'password'), ('password=fake\n', 'username'),
])
def test_credentials_missing_key(tmp_path, text, missing):
    create_credentials(tmp_path, text)
    with pytest.raises(HydroHealthCredentialsError, match=missing):
        HydroHealthConfig()


@pytest.mark.parametrize('environment', ['local', 'remote', 'aws'])
def test_init_paths_and_defaults(tmp_path, environment):
    with patch(f'{TSM_PATH}.get_environment', return_value=environment), \
         patch(f'{TSM_PATH}.HydroHealthConfig') as creds:
        creds.return_value.username = 'tester'
        creds.return_value.password = 'fake'
        engine = CreateTSMLayerEngine()
    assert engine.username == 'tester'
    assert engine.password == 'fake'
    assert engine.server == 'ftp.hermes.acri.fr'
    assert engine.force_download is False
    assert engine.downloaded_files == []
    if environment == 'aws':
        assert engine.raster_path == 'test-bucket/ER_3/tsm/mean_rasters'
        assert engine.year_pair_path == 'test-bucket/ER_3/tsm/TSM_year_pair_rasters'
        assert engine.nc_files_path == 'test-bucket/tsm_nc'
    else:
        base = tmp_path / 'outputs'
        assert engine.raster_path == base / 'ER_3/tsm/mean_rasters'
        assert engine.year_pair_path == base / 'ER_3/tsm/TSM_year_pair_rasters'
        assert engine.nc_files_path == base / 'tsm_nc'


# =====================================================================
# Filename Parsing & Download Logs
# =====================================================================

@pytest.mark.parametrize('filename,expected', [
    (tsm_name(), '20040101'), ('prefix_19981231-19990107.nc', '19981231'),
    ('no_date.nc', '99999999'), ('', '99999999'),
])
def test_extract_start_date(victim, filename, expected):
    assert victim.extract_start_date(filename) == expected


def test_write_download_log_sorted(victim):
    names = [tsm_name('20060101'), 'unknown.nc', tsm_name('19980101')]
    victim.downloaded_files = names.copy()
    victim.write_download_log()
    assert (victim.output_folder / 'downloaded_files.txt').read_text().splitlines() == [
        names[2], names[0], names[1],
    ]
    assert victim.downloaded_files == names  # Sorting must not mutate engine state.


def test_write_download_log_empty(victim):
    victim.write_download_log()
    assert (victim.output_folder / 'downloaded_files.txt').read_text() == ''


# =====================================================================
# FTP Filtering, Downloads & Connection Cleanup
# =====================================================================

@pytest.mark.parametrize('directory', [
    '/GLOB/modis/8day', '/GLOB/meris/8day', '/GLOB/product/month/files',
    '/GLOB/product/day/files', '/GLOB/olcia/8day/2023/files',
])
def test_download_recursive_skips_excluded_directories(victim, directory):
    ftp = MagicMock()
    victim.download_recursive(ftp, directory)
    ftp.mlsd.assert_not_called()
    ftp.retrbinary.assert_not_called()


def test_download_recursive_filters_and_downloads_local(victim):
    ftp = MagicMock()
    accepted = [tsm_name(), tsm_name('20180501', 'AV-OLA')]
    rejected = [
        'notes.txt', 'TEST_L3m_20040101-20040108_OTHER_8D_4.nc',
        'TEST_L3m_20040101-20040108_TSM_8D_9.nc',
        'TEST_20040101-20040108_TSM_8D_4.nc',
        tsm_name('20180502', 'AV-OLA'),
    ]
    ftp.mlsd.return_value = [(name, {'type': 'file'}) for name in accepted + rejected]
    ftp.retrbinary.side_effect = lambda command, callback: callback(b'netcdf-test-bytes')
    victim.download_recursive(ftp, '/GLOB/test')
    assert victim.downloaded_files == accepted
    assert ftp.retrbinary.call_args_list == [call(f'RETR {n}', ANY) for n in accepted]
    for name in accepted:
        assert (victim.nc_files_path / name).read_bytes() == b'netcdf-test-bytes'
    for name in rejected:
        assert not (victim.nc_files_path / name).exists()



@pytest.mark.parametrize('force', [False, True])
def test_download_recursive_existing_local_file(victim, force):
    name = tsm_name()
    victim.nc_files_path.mkdir(parents=True)
    path = victim.nc_files_path / name
    path.write_bytes(b'original')
    victim.force_download = force
    ftp = MagicMock()
    ftp.mlsd.return_value = [(name, {'type': 'file'})]
    ftp.retrbinary.side_effect = lambda command, callback: callback(b'replaced')
    victim.download_recursive(ftp, '/GLOB/test')
    assert path.read_bytes() == (b'replaced' if force else b'original')
    assert ftp.retrbinary.call_count == int(force)
    assert victim.downloaded_files == ([name] if force else [])


@pytest.mark.parametrize('exists,force,download', [
    (False, False, True), (True, False, False), (True, True, True),
])
def test_download_recursive_s3(victim, isolated_services, exists, force, download):
    _, s3_class = isolated_services
    s3 = s3_class.return_value
    s3.exists.return_value = exists
    victim.nc_files_path = 'test-bucket/tsm_nc'
    victim.force_download = force
    ftp = MagicMock()
    name = tsm_name()
    ftp.mlsd.return_value = [(name, {'type': 'file'})]
    ftp.retrbinary.side_effect = lambda command, callback: callback(b'fake-data')
    with patch(f'{TSM_PATH}.get_environment', return_value='aws'):
        victim.download_recursive(ftp, '/GLOB/test')
    if download:
        s3.open.assert_called_once_with(f'test-bucket/tsm_nc/{name}', 'wb')
        s3.open.return_value.__enter__.return_value.write.assert_called_once_with(b'fake-data')
    else:
        s3.open.assert_not_called()
    assert victim.downloaded_files == ([name] if download else [])


def test_download_recursive_directory_error_is_reported(victim, capsys):
    ftp = MagicMock()
    ftp.cwd.side_effect = OSError('directory unavailable')
    victim.download_recursive(ftp, '/GLOB/test')
    assert 'Error accessing /GLOB/test: directory unavailable' in capsys.readouterr().out
    assert victim.downloaded_files == []


@pytest.mark.xfail(strict=True, reason='Engine does not restore FTP parent directory after recursion')
def test_download_recursive_restores_parent_for_sibling_file(victim):
    """A stateful fake detects downloads attempted in the wrong directory."""
    name = tsm_name()
    ftp = MagicMock()
    location = {'cwd': None}
    ftp.cwd.side_effect = lambda path: location.update(cwd=path)
    ftp.mlsd.side_effect = lambda: (
        [('child', {'type': 'dir'}), (name, {'type': 'file'})]
        if location['cwd'] == '/GLOB/test' else []
    )
    def retrieve(command, callback):
        if location['cwd'] != '/GLOB/test':
            raise FileNotFoundError('Sibling file is in the parent directory')
        callback(b'correct-file')
    ftp.retrbinary.side_effect = retrieve
    victim.download_recursive(ftp, '/GLOB/test')
    assert victim.downloaded_files == [name]
    assert (victim.nc_files_path / name).read_bytes() == b'correct-file'


def test_download_tsm_data_login_download_quit_and_log(victim, isolated_services):
    ftp_class, _ = isolated_services
    ftp = ftp_class.return_value
    manager = MagicMock()
    manager.attach_mock(ftp.login, 'login')
    manager.attach_mock(ftp.quit, 'quit')
    with patch.object(victim, 'download_recursive') as recursive, \
         patch.object(victim, 'write_download_log') as log:
        manager.attach_mock(recursive, 'recursive')
        manager.attach_mock(log, 'log')
        victim.download_tsm_data()
    ftp_class.assert_called_once_with(victim.server)
    assert manager.mock_calls == [
        call.login(user='test-user', passwd='test-password'),
        call.recursive(ftp, '/GLOB'), call.quit(), call.log(),
    ]


def test_download_tsm_data_quits_if_recursion_raises(victim, isolated_services):
    ftp_class, _ = isolated_services
    with patch.object(victim, 'download_recursive', side_effect=RuntimeError('failed')), \
         patch.object(victim, 'write_download_log') as log:
        with pytest.raises(RuntimeError, match='failed'):
            victim.download_tsm_data()
    ftp_class.return_value.quit.assert_called_once()
    log.assert_not_called()


# =====================================================================
# Raster Writing & Annual Means
# =====================================================================

def test_save_raster_data_round_trip(victim, tmp_path):
    data = np.array([[1, np.nan], [3, 4]], dtype=np.float32)
    path = tmp_path / 'saved.tif'
    transform = rasterio.transform.from_origin(-80, 30, 0.1, 0.1)
    victim._save_raster_data(data, path, transform, 'EPSG:4326', np.nan)
    with rasterio.open(path) as src:
        np.testing.assert_allclose(src.read(1), data, equal_nan=True)
        assert src.shape == (2, 2)
        assert src.count == 1
        assert src.dtypes == ('float32',)
        assert src.crs.to_epsg() == 4326
        assert src.transform == transform
        assert np.isnan(src.nodata)
        assert src.profile['tiled'] is True
        assert src.compression.value.lower() == 'deflate'


@pytest.mark.parametrize('environment', ['local', 'remote', 'aws'])
def test_create_mean_year_rasters_values_and_destination(victim, isolated_services, environment):
    """Check per-pixel counts, zero exclusion, NaNs, year filtering and paths."""
    _, s3_class = isolated_services
    s3 = s3_class.return_value
    names = [tsm_name(), tsm_name('20040109'), 'irrelevant.txt']
    if environment == 'aws':
        victim.nc_files_path = 'test-bucket/tsm_nc'
        victim.raster_path = 'test-bucket/mean_rasters'
        s3.ls.return_value = [f'{victim.nc_files_path}/{n}' for n in names]
    else:
        victim.nc_files_path.mkdir(parents=True)
        for name in names:
            (victim.nc_files_path / name).touch()
    arrays = [
        [[2, 0, np.nan], [4, 0, 8]],
        [[4, 6, np.nan], [np.nan, 0, 12]],
    ]
    with patch(f'{TSM_PATH}.get_environment', return_value=environment), \
         patch(f'{TSM_PATH}.xr.open_dataset', side_effect=[dataset_for(a) for a in arrays]) as open_ds, \
         patch.object(victim, '_save_raster_data') as save, np.errstate(invalid='ignore', divide='ignore'):
        victim.create_mean_year_rasters()
    assert open_ds.call_count == 2
    save.assert_called_once()
    data, path, transform, crs, nodata = save.call_args.args
    np.testing.assert_allclose(data, [[3, 6, np.nan], [4, np.nan, 10]], equal_nan=True)
    assert data.dtype == np.float32
    assert crs == 'EPSG:4326'
    assert np.isnan(nodata)
    # Production transform is based on the specified global 4320 x 8640 grid.
    assert transform == rasterio.transform.from_bounds(-180, -90, 180, 90, 8640, 4320)
    if environment == 'aws':
        assert all(str(c.args[0]).startswith('s3://test-bucket/') for c in open_ds.call_args_list)
        s3.put.assert_called_once_with(path, 's3://test-bucket/mean_rasters/TSM_mean_2004.tif')
        assert not os.path.exists(path)
    else:
        assert path == victim.raster_path / 'TSM_mean_2004.tif'
        open_ds.assert_has_calls([
            call(victim.nc_files_path / n, engine='h5netcdf') for n in names[:2]
        ], any_order=True)
        s3.put.assert_not_called()


def test_create_mean_year_rasters_no_directory(victim):
    with patch.object(victim, '_save_raster_data') as save, \
         patch(f'{TSM_PATH}.xr.open_dataset') as open_ds:
        victim.create_mean_year_rasters()
    open_ds.assert_not_called()
    save.assert_not_called()


@pytest.mark.parametrize('all_fail', [False, True])
def test_create_mean_year_rasters_unreadable_input(victim, all_fail, capsys):
    victim.nc_files_path.mkdir(parents=True)
    for date in ['20040101', '20040109']:
        (victim.nc_files_path / tsm_name(date)).touch()
    outcomes = [OSError('bad NetCDF'), OSError('bad NetCDF') if all_fail else dataset_for([[2, 4], [6, 8]])]
    with patch(f'{TSM_PATH}.xr.open_dataset', side_effect=outcomes), \
         patch.object(victim, '_save_raster_data') as save:
        victim.create_mean_year_rasters()
    assert 'bad NetCDF' in capsys.readouterr().out
    assert save.call_count == (0 if all_fail else 1)
    if not all_fail:
        np.testing.assert_allclose(save.call_args.args[0], [[2, 4], [6, 8]])


def test_create_mean_year_rasters_s3_upload_failure_cleans_temp(victim, isolated_services):
    _, s3_class = isolated_services
    s3 = s3_class.return_value
    victim.nc_files_path = 'test-bucket/nc'
    victim.raster_path = 'test-bucket/means'
    s3.ls.return_value = [f'test-bucket/nc/{tsm_name()}']
    s3.put.side_effect = OSError('upload failed')
    with patch(f'{TSM_PATH}.get_environment', return_value='aws'), \
         patch(f'{TSM_PATH}.xr.open_dataset', return_value=dataset_for([[1, 2], [3, 4]])), \
         patch.object(victim, '_save_raster_data') as save:
        with pytest.raises(OSError, match='upload failed'):
            victim.create_mean_year_rasters()
    assert not os.path.exists(save.call_args.args[1])


# =====================================================================
# Inclusive Year-Pair Means & Workflow Orchestration
# =====================================================================

@pytest.mark.parametrize('environment', ['local', 'remote', 'aws'])
def test_year_pair_rasters_mean_and_metadata(victim, isolated_services, tmp_path, environment):
    _, s3_class = isolated_services
    s3 = s3_class.return_value
    source = tmp_path / 'annual'
    # Include both endpoints, an intermediate year, and an out-of-range distractor.
    arrays = {
        2004: [[2, np.nan, np.nan], [0, 4, 2]],
        2005: [[4, 6, np.nan], [3, np.nan, 4]],
        2006: [[6, 8, np.nan], [6, 8, 6]],
        2007: [[999, 999, 999], [999, 999, 999]],
    }
    paths = {year: create_test_raster(source / f'TSM_mean_{year}.tif', data) for year, data in arrays.items()}
    victim.raster_path = source
    original_open = rasterio.open
    reads = []
    def open_source(path, *args, **kwargs):
        reads.append(str(path))
        return original_open(source / str(path).split('/')[-1], *args, **kwargs)
    if environment == 'aws':
        victim.raster_path = 'test-bucket/means'
        victim.year_pair_path = 'test-bucket/pairs'
        s3.ls.return_value = [f'test-bucket/means/{p.name}' for p in paths.values()]
    with patch(f'{TSM_PATH}.get_environment', return_value=environment), \
         patch(f'{TSM_PATH}.rasterio.open', side_effect=open_source), \
         patch.object(victim, '_save_raster_data') as save, np.errstate(invalid='ignore', divide='ignore'):
        victim.year_pair_rasters(2004, 2006)
    save.assert_called_once()
    data, path, transform, crs, nodata = save.call_args.args
    # Year-pair means include real zero values (annual processing already excluded zeros).
    np.testing.assert_allclose(data, [[4, 7, np.nan], [3, 6, 4]], equal_nan=True)
    assert crs.to_epsg() == 4326
    assert transform == rasterio.transform.from_origin(-80, 30, 0.1, 0.1)
    assert np.isnan(nodata)
    assert not any('2007' in p for p in reads)
    assert set(pathlib.Path(p).name for p in reads) == {paths[y].name for y in [2004, 2005, 2006]}
    if environment == 'aws':
        assert all(p.startswith('s3://') for p in reads)
        s3.put.assert_called_once_with(path, 's3://test-bucket/pairs/tsm_mean_2004_2006.tif')
        assert not os.path.exists(path)
    else:
        assert pathlib.Path(path) == victim.year_pair_path / 'tsm_mean_2004_2006.tif'
        s3.put.assert_not_called()


@pytest.mark.parametrize('environment', ['local', 'aws'])
def test_year_pair_rasters_no_matching_inputs(victim, isolated_services, environment):
    _, s3_class = isolated_services
    victim.raster_path.mkdir(parents=True)
    s3_class.return_value.ls.return_value = []
    with patch(f'{TSM_PATH}.get_environment', return_value=environment), \
         patch.object(victim, '_save_raster_data') as save, \
         patch(f'{TSM_PATH}.rasterio.open') as open_raster:
        victim.year_pair_rasters(2004, 2006)
    save.assert_not_called()
    open_raster.assert_not_called()


def test_run_calls_steps_in_order(victim):
    victim.year_ranges = [(2004, 2006), (2006, 2010)]
    manager = MagicMock()
    with patch.object(victim, 'download_tsm_data') as download, \
         patch.object(victim, 'create_mean_year_rasters') as annual, \
         patch.object(victim, 'year_pair_rasters') as pairs:
        manager.attach_mock(download, 'download')
        manager.attach_mock(annual, 'annual')
        manager.attach_mock(pairs, 'pairs')
        victim.run()
    assert manager.mock_calls == [
        call.download(), call.annual(), call.pairs(2004, 2006), call.pairs(2006, 2010),
    ]
