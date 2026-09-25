import pathlib
import pytest
from unittest.mock import MagicMock, patch
from osgeo import gdal, osr

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
print(HYDRO_HEALTH_MODULE)
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.helpers.tools import Param
from hydro_health.engines.RasterVRTS3Engine import (
    RasterVRTS3Engine,
    _clean,
    _process_single_bluetopo,
    _read_geotiff_metadata,
)

TILING_PATH = 'hydro_health.engines.RasterVRTS3Engine'


@pytest.fixture
def victim(tmp_path):
    """Fixture returning an instance of RasterVRTS3Engine with Dask and client mocked."""
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'aws'
    }
    with patch(f'{TILING_PATH}.query_crs_info', return_value=[]):
        engine = RasterVRTS3Engine(param_lookup)

    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.client = MagicMock()

    return engine


# Standard EPSG:4326 WKT string for GDAL OSR initialization
VALID_4326_WKT = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'


def test_process_single_bluetopo_success(tmp_path):
    """Tests warping an S3 BlueTopo tile to EPSG:4326 and uploading the VRT."""

    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create('', 10, 10, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32618)
    ds.SetProjection(srs.ExportToWkt())

    # Create warped memory dataset and import standard EPSG:4326 WKT
    warped_ds = driver.Create('', 10, 10, 1, gdal.GDT_Float32)
    srs_4326 = osr.SpatialReference()
    srs_4326.ImportFromEPSG(4326)
    warped_ds.SetProjection(srs_4326.ExportToWkt())

    params = ['bucket/data/tile.tiff', 'target-bucket', None]

    with patch(f'{TILING_PATH}.gdal.Open', return_value=ds), \
         patch(f'{TILING_PATH}.gdal.Warp', return_value=warped_ds), \
         patch('boto3.client') as mock_boto:

        datum_code, final_s3_vrt_path, projection_wkt = _process_single_bluetopo(params)

        assert final_s3_vrt_path == "/vsis3/target-bucket/data/tile.vrt"
        assert datum_code == "6326"
        mock_boto.return_value.upload_file.assert_called_once()


def test_read_geotiff_metadata_success():
    """Tests reading EPSG and provider bin key from an S3 path using GDAL."""

    driver = gdal.GetDriverByName('MEM')
    ds = driver.Create('', 10, 10, 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).SetNoDataValue(-9999.0)

    raw_prefix = "s3://my-bucket/ER1/subfolder/DigitalCoast/NOAA_Provider/tile.tiff"

    with patch(f'{TILING_PATH}.get_config_item', return_value="my-bucket"), \
         patch(f'{TILING_PATH}.gdal.Open', return_value=ds):

        meta = _read_geotiff_metadata(raw_prefix)

        assert meta is not None
        assert meta['bin_key'] == "NOAA_Provider"
        assert meta['epsg'] == 4326
        assert meta['nodata'] == -9999.0


def test_build_output_vrts(victim, tmp_path):
    """Tests building master VRTs locally and uploading them to S3."""

    output_geotiffs = {
        'NOAA_Provider': {
            'tiles': ['/vsis3/my-bucket/tile1.tif'],
            'nodata_val': -9999
        }
    }

    # Pre-create the VRT file in temp_path so .exists() evaluates to True
    vrt_file = tmp_path / "mosaic_elevation_NOAA_Provider.vrt"
    vrt_file.touch()

    with patch(f'{TILING_PATH}.get_config_item', return_value="my-bucket"), \
         patch(f'{TILING_PATH}.gdal.BuildVRT') as mock_build_vrt, \
         patch('boto3.client') as mock_boto:

        victim.build_output_vrts("prefix/ER1/subfolder/DigitalCoast", "elevation", output_geotiffs, tmp_path, "DigitalCoast")

        mock_build_vrt.assert_called_once()
        mock_boto.return_value.upload_file.assert_called_once_with(
            str(vrt_file),
            "my-bucket",
            "prefix/ER1/subfolder/DigitalCoast/mosaic_elevation_NOAA_Provider.vrt"
        )


def test_get_bluetopo_tifs(victim):
    """Tests mapping BlueTopo processing across Dask cluster workers."""

    geotiffs = ['s3://bucket/tile1.tiff']
    
    # Use valid WKT string instead of 'WKT_STRING'
    mock_results = [('6318', '/vsis3/bucket/tile1.vrt', VALID_4326_WKT)]
    victim.client.gather.return_value = mock_results

    with patch(f'{TILING_PATH}.get_config_item', return_value="my-bucket"):
        result = victim.get_bluetopo_tifs(geotiffs)

        assert '6318' in result
        assert result['6318']['tiles'] == ['/vsis3/bucket/tile1.vrt']


def test_get_digitalcoast_geotiffs_majority_crs(victim, tmp_path):
    """Tests grouping DigitalCoast tiles and skipping reprojection when CRS matches majority."""

    geotiffs = ['s3://bucket/provider/tile1.tiff']
    
    mock_meta = [{
        'bin_key': 'ProviderA',
        'vsi_path': '/vsis3/bucket/provider/tile1.tiff',
        'relative_s3_key': 'provider/tile1.tiff',
        'nodata': -9999,
        'epsg': 4326,
        'wkt': 'EPSG:4326'
    }]
    
    victim.client.gather.return_value = mock_meta

    with patch(f'{TILING_PATH}.get_config_item', return_value="my-bucket"), \
         patch('boto3.client'):

        result = victim.get_digitalcoast_geotiffs(geotiffs, tmp_path, "out")

        assert 'ProviderA' in result
        assert result['ProviderA']['tiles'] == ['/vsis3/bucket/provider/tile1.tiff']
        assert result['ProviderA']['primary_crs'] == 'EPSG:4326'


def test_get_digitalcoast_geotiffs_reproject_minority(victim, tmp_path):
    """Tests reprojecting a minority CRS tile to match the provider majority CRS."""

    geotiffs = [
        's3://bucket/provider/tile1.tiff',
        's3://bucket/provider/tile2.tiff'
    ]
    
    mock_meta = [
        {
            'bin_key': 'ProviderA',
            'vsi_path': '/vsis3/bucket/provider/tile1.tiff',
            'relative_s3_key': 'provider/tile1.tiff',
            'nodata': -9999,
            'epsg': 4326,
            'wkt': 'EPSG:4326'
        },
        {
            'bin_key': 'ProviderA',
            'vsi_path': '/vsis3/bucket/provider/tile2.tiff',
            'relative_s3_key': 'provider/tile2.tiff',
            'nodata': -9999,
            'epsg': 32618,  # Minority tile requires reprojection to 4326
            'wkt': 'EPSG:32618'
        }
    ]
    
    victim.client.gather.return_value = mock_meta

    mock_s3 = MagicMock()
    # Trigger head_object exception to force gdal.Warp reprojection
    mock_s3.head_object.side_effect = Exception("Not Found")

    with patch(f'{TILING_PATH}.get_config_item', return_value="my-bucket"), \
         patch('boto3.client', return_value=mock_s3), \
         patch(f'{TILING_PATH}.gdal.Warp'):

        result = victim.get_digitalcoast_geotiffs(geotiffs, tmp_path, "out")

        tiles = result['ProviderA']['tiles']
        assert '/vsis3/bucket/provider/tile1.tiff' in tiles
        assert '/vsis3/my-bucket/provider/reprojected/tile2.tiff' in tiles


def test_run_bluetopo(victim):
    """Tests BlueTopo execution flow through Dask setup, S3 globbing, and cleanup."""

    with patch(f'{TILING_PATH}.get_config_item', return_value="my-bucket"), \
         patch(f'{TILING_PATH}.s3fs.S3FileSystem') as mock_s3fs, \
         patch.object(victim, 'get_bluetopo_tifs', return_value={}) as mock_get_bt, \
         patch.object(victim, 'build_output_vrts') as mock_build_vrt, \
         patch('shutil.rmtree'):

        mock_s3fs.return_value.glob.return_value = ['s3://my-bucket/ER1/subfolder/BlueTopo/tile.tiff']

        victim.run(
            outputs="/tmp/out",
            file_type="elevation",
            ecoregion="ER1",
            data_type="BlueTopo"
        )

        victim.setup_dask.assert_called_once_with('aws')
        mock_get_bt.assert_called_once()
        mock_build_vrt.assert_called_once()
        victim.close_dask.assert_called_once()


def test_run_digitalcoast_manual_downloads(victim):
    """Tests DigitalCoast execution with manual_downloads flag set to True."""

    with patch(f'{TILING_PATH}.get_config_item', return_value="my-bucket"), \
         patch(f'{TILING_PATH}.s3fs.S3FileSystem') as mock_s3fs, \
         patch.object(victim, 'get_digitalcoast_geotiffs', return_value={}) as mock_get_dc, \
         patch.object(victim, 'build_output_vrts'), \
         patch('shutil.rmtree'):

        # Mock folder globs
        mock_s3fs.return_value.glob.side_effect = [
            ['s3://my-bucket/ER1/sub/DigitalCoast/ProviderA'],  # standard folders
            ['s3://my-bucket/ER1/sub/Digital_Coast_Manual_Downloads/ProviderB'],  # manual folders
            ['s3://my-bucket/ER1/sub/DigitalCoast/ProviderA/tile1.tiff'],  # standard geotiffs
            ['s3://my-bucket/ER1/sub/Digital_Coast_Manual_Downloads/ProviderB/tile2.tiff']  # manual geotiffs
        ]

        victim.run(
            outputs="/tmp/out",
            file_type="elevation",
            ecoregion="ER1",
            data_type="DigitalCoast",
            manual_downloads=True
        )

        assert mock_get_dc.call_count == 2
        victim.close_dask.assert_called_once()