import pathlib
import pytest
from unittest.mock import MagicMock, patch
from osgeo import gdal, osr

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.RasterVRTEngine import RasterVRTEngine

ENGINE_PATH = 'hydro_health.engines.RasterVRTEngine'


@pytest.fixture
def victim(tmp_path):
    """Fixture returning an instance of RasterVRTEngine."""
    param_lookup = {'output_directory': str(tmp_path), 'env': 'local'}
    with patch(f'{ENGINE_PATH}.query_crs_info', return_value=[]):
        engine = RasterVRTEngine(param_lookup)
    engine.write_message = MagicMock()
    return engine


@pytest.fixture
def create_dummy_geotiff(tmp_path):
    """Utility fixture to create a valid small GeoTIFF with GDAL."""

    def _create(filename="test.tiff", epsg=32617, nodata=-9999.0):
        file_path = tmp_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(file_path), 10, 10, 1, gdal.GDT_Float32)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform([500000, 1, 0, 4000000, 0, -1])

        band = ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.Fill(nodata)
        ds.FlushCache()
        ds = None

        return file_path

    return _create


def test_get_local_bluetopo_geotiffs(victim, create_dummy_geotiff):
    """Tests grouping BlueTopo tiles strictly by native EPSG UTM CRS."""
    gtif_18n = create_dummy_geotiff("tile_18n.tiff", epsg=32618)
    gtif_19n = create_dummy_geotiff("tile_19n.tiff", epsg=32619)

    result = victim.get_local_bluetopo_geotiffs([gtif_18n, gtif_19n])

    assert "EPSG_32618" in result
    assert "EPSG_32619" in result
    assert result["EPSG_32618"]["tiles"] == [str(gtif_18n)]
    assert result["EPSG_32619"]["tiles"] == [str(gtif_19n)]
    assert result["EPSG_32618"]["nodata_val"] == -9999.0


def test_get_local_digitalcoast_geotiffs(victim, create_dummy_geotiff):
    """Tests provider binning and nodata extraction for EPSG:32617 DigitalCoast rasters."""
    gtif = create_dummy_geotiff("DigitalCoast/NOAA_Provider/data.tif", epsg=32617, nodata=-32767.0)

    result = victim.get_local_digitalcoast_geotiffs([gtif])

    assert "NOAA_Provider" in result
    assert result["NOAA_Provider"]["tiles"] == [str(gtif)]
    assert result["NOAA_Provider"]["primary_crs"] == "EPSG:32617"
    assert result["NOAA_Provider"]["nodata_val"] == -32767.0
    victim.write_message.assert_not_called()


def test_get_local_digitalcoast_geotiffs_warns_non_32617(victim, create_dummy_geotiff):
    """Tests that non-32617 local tiles log a warning via write_message without reprojecting."""
    gtif = create_dummy_geotiff("DigitalCoast/NOAA_Provider/data_6350.tif", epsg=6350)

    result = victim.get_local_digitalcoast_geotiffs([gtif])

    assert result["NOAA_Provider"]["tiles"] == [str(gtif)]
    assert result["NOAA_Provider"]["primary_crs"] == "EPSG:32617"
    victim.write_message.assert_called_once()
    warning_msg = victim.write_message.call_args[0][0]
    assert "EPSG:6350" in warning_msg
    assert "expected EPSG:32617" in warning_msg


def test_get_local_digitalcoast_geotiffs_manual_downloads(victim, create_dummy_geotiff):
    """Tests provider extraction using the Digital_Coast_Manual_Downloads path variant."""
    gtif = create_dummy_geotiff("Digital_Coast_Manual_Downloads/USACE_Provider/data.tif", epsg=32617)

    result = victim.get_local_digitalcoast_geotiffs([gtif])

    assert "USACE_Provider" in result
    assert result["USACE_Provider"]["tiles"] == [str(gtif)]
    assert result["USACE_Provider"]["primary_crs"] == "EPSG:32617"
    victim.write_message.assert_not_called()


def test_get_s3_digitalcoast_geotiffs(victim):
    """Tests S3 DigitalCoast binning, /original/ filtering, and EPSG:32617 assumption."""
    victim.client = MagicMock()
    victim.client.map.side_effect = lambda fn, items: [fn(i) for i in items]
    victim.client.gather.side_effect = lambda items: items

    s3_tiles = [
        "s3://bucket/ER_3/DigitalCoast/NOAA_Provider/tile_1.tiff",
        "s3://bucket/ER_3/DigitalCoast/NOAA_Provider/original/tile_1.tiff",
    ]

    mock_meta = {
        'bin_key': 'NOAA_Provider',
        'vsi_path': '/vsis3/bucket/ER_3/DigitalCoast/NOAA_Provider/tile_1.tiff',
        'relative_s3_key': 'ER_3/DigitalCoast/NOAA_Provider/tile_1.tiff',
        'nodata': -32767.0,
        'epsg': 32617,
        'wkt': 'PROJCS[...]'
    }

    with patch(f'{ENGINE_PATH}._read_geotiff_metadata', return_value=mock_meta) as mock_reader:
        result = victim.get_s3_digitalcoast_geotiffs(s3_tiles)

    # Only the non-original tile should be read
    mock_reader.assert_called_once_with("s3://bucket/ER_3/DigitalCoast/NOAA_Provider/tile_1.tiff")
    assert result == {
        'NOAA_Provider': {
            'tiles': ['/vsis3/bucket/ER_3/DigitalCoast/NOAA_Provider/tile_1.tiff'],
            'nodata_val': -32767.0,
            'primary_crs': 'EPSG:32617'
        }
    }
    victim.write_message.assert_not_called()


def test_get_s3_digitalcoast_geotiffs_warns_non_32617(victim):
    """Tests that non-32617 S3 tiles trigger write_message warning without reprojecting."""
    victim.client = MagicMock()
    victim.client.map.side_effect = lambda fn, items: [fn(i) for i in items]
    victim.client.gather.side_effect = lambda items: items

    s3_tiles = ["s3://bucket/ER_3/DigitalCoast/NOAA_Provider/tile_6350.tiff"]

    mock_meta = {
        'bin_key': 'NOAA_Provider',
        'vsi_path': '/vsis3/bucket/ER_3/DigitalCoast/NOAA_Provider/tile_6350.tiff',
        'relative_s3_key': 'ER_3/DigitalCoast/NOAA_Provider/tile_6350.tiff',
        'nodata': -9999.0,
        'epsg': 6350,
        'wkt': 'PROJCS[...]'
    }

    with patch(f'{ENGINE_PATH}._read_geotiff_metadata', return_value=mock_meta):
        result = victim.get_s3_digitalcoast_geotiffs(s3_tiles)

    assert result['NOAA_Provider']['primary_crs'] == 'EPSG:32617'
    victim.write_message.assert_called_once()
    warning_msg = victim.write_message.call_args[0][0]
    assert "EPSG:6350" in warning_msg
    assert "expected EPSG:32617" in warning_msg


def test_build_output_vrts_bluetopo(victim, tmp_path):
    """Tests VRT creation using BlueTopo options (disallow projection difference)."""
    tile_a = tmp_path / "tile_a.tif"
    tile_a.touch()

    output_geotiffs = {
        'EPSG_32618': {
            'tiles': [str(tile_a)],
            'nodata_val': -9999.0
        }
    }

    with patch(f'{ENGINE_PATH}.gdal.BuildVRT') as mock_build, \
         patch(f'{ENGINE_PATH}.gdal.BuildVRTOptions') as mock_options:

        victim.build_output_vrts(tmp_path, output_geotiffs, 'BlueTopo', file_type='elevation')

        mock_options.assert_called_once_with(
            resampleAlg='bilinear',
            allowProjectionDifference=False,
            srcNodata=-9999.0,
            VRTNodata=-9999.0
        )
        expected_vrt = str(tmp_path / 'mosaic_elevation_EPSG_32618.vrt')
        mock_build.assert_called_once_with(expected_vrt, [str(tile_a)], options=mock_options.return_value)


def test_build_output_vrts_digitalcoast(victim, tmp_path):
    """Tests VRT creation using DigitalCoast options (near resampling, allow difference)."""
    tile_a = tmp_path / "tile_a.tif"
    tile_a.touch()

    output_geotiffs = {
        'NOAA': {
            'tiles': [str(tile_a)],
            'nodata_val': -32767.0
        }
    }

    with patch(f'{ENGINE_PATH}.gdal.BuildVRT') as mock_build, \
         patch(f'{ENGINE_PATH}.gdal.BuildVRTOptions') as mock_options:

        victim.build_output_vrts(tmp_path, output_geotiffs, 'DigitalCoast', file_type='elevation')

        mock_options.assert_called_once_with(
            resampleAlg='near',
            allowProjectionDifference=True,
            srcNodata=-32767.0,
            VRTNodata=-32767.0
        )
        expected_vrt = str(tmp_path / 'mosaic_elevation_NOAA.vrt')
        mock_build.assert_called_once_with(expected_vrt, [str(tile_a)], options=mock_options.return_value)


def test_build_output_vrts_distancetoshore(victim, tmp_path):
    """Tests VRT filename convention for DistanceToShore."""
    tile_a = tmp_path / "tile_a.tif"
    tile_a.touch()

    output_geotiffs = {
        'NOAA': {
            'tiles': [str(tile_a)],
            'nodata_val': -9999.0
        }
    }

    with patch(f'{ENGINE_PATH}.gdal.BuildVRT') as mock_build, \
         patch(f'{ENGINE_PATH}.gdal.BuildVRTOptions') as mock_options:

        victim.build_output_vrts(tmp_path, output_geotiffs, 'DistanceToShore')

        expected_vrt = str(tmp_path / 'mosaic_NOAA_DTS.vrt')
        mock_build.assert_called_once_with(expected_vrt, [str(tile_a)], options=mock_options.return_value)


def test_run_local_bluetopo(victim, tmp_path):
    """Tests the BlueTopo local branch in run()."""
    with patch(f'{ENGINE_PATH}.get_config_item', return_value="subfolder"), \
         patch.object(victim, 'get_local_bluetopo_geotiffs', return_value={}) as mock_get_bt, \
         patch.object(victim, 'build_output_vrts') as mock_build_vrt:

        target_dir = tmp_path / "ER1" / "subfolder" / "BlueTopo"
        target_dir.mkdir(parents=True)
        (target_dir / "tile_1.tiff").touch()

        victim.run(
            output_folder=str(tmp_path),
            file_type="elevation",
            ecoregion="ER1",
            data_type="BlueTopo"
        )

        mock_get_bt.assert_called_once()
        mock_build_vrt.assert_called_once()


def test_run_local_digitalcoast_excludes_original_and_unused(victim, tmp_path):
    """Tests that local DigitalCoast discovery excludes original, distance_to_shore, and unused_providers."""
    with patch(f'{ENGINE_PATH}.get_config_item', return_value="subfolder"), \
         patch.object(victim, 'get_local_digitalcoast_geotiffs', return_value={}) as mock_get_dc, \
         patch.object(victim, 'build_output_vrts') as mock_build_vrt:

        target_dir = tmp_path / "prefix" / "ER_3" / "subfolder" / "DigitalCoast"
        provider_a = target_dir / "ProviderA"
        provider_a.mkdir(parents=True)
        valid_tile = provider_a / "tile_1.tiff"
        valid_tile.touch()

        # Files that must be excluded
        orig_dir = provider_a / "original"
        orig_dir.mkdir()
        (orig_dir / "tile_orig_1.tiff").touch()

        dts_dir = provider_a / "distance_to_shore"
        dts_dir.mkdir()
        (dts_dir / "tile_dts_1.tiff").touch()

        unused_provider = target_dir / "unused_providers"
        unused_provider.mkdir(parents=True)
        (unused_provider / "tile_2.tiff").touch()

        victim.run(
            output_folder=str(tmp_path),
            file_type="elevation",
            ecoregion="ER_3",
            data_type="DigitalCoast",
            output_prefix="prefix"
        )

        mock_get_dc.assert_called_once_with([valid_tile])
        mock_build_vrt.assert_called_once()


def test_run_aws_digitalcoast_excludes_original(victim, tmp_path):
    """Tests that AWS DigitalCoast discovery excludes /original/ and /distance_to_shore/ prefixes."""
    victim.param_lookup['env'] = 'aws'
    victim.setup_dask = MagicMock()
    victim.close_dask = MagicMock()

    mock_s3fs = MagicMock()
    mock_s3fs.glob.side_effect = [
        # Providers under DigitalCoast
        ["bucket/ER_3/subfolder/DigitalCoast/ProviderA"],
        # Files matched under ProviderA
        [
            "bucket/ER_3/subfolder/DigitalCoast/ProviderA/dem/tile_1.tiff",
            "bucket/ER_3/subfolder/DigitalCoast/ProviderA/dem/original/tile_1.tiff",
            "bucket/ER_3/subfolder/DigitalCoast/ProviderA/dem/distance_to_shore/tile_1.tiff",
        ],
    ]

    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem', return_value=mock_s3fs), \
         patch(f'{ENGINE_PATH}.get_config_item', side_effect=["bucket", "subfolder"]), \
         patch.object(victim, 'get_s3_digitalcoast_geotiffs', return_value={}) as mock_get_s3_dc, \
         patch.object(victim, 'build_output_vrts') as mock_build_vrt:

        victim.run(
            output_folder=str(tmp_path),
            file_type="elevation",
            ecoregion="ER_3",
            data_type="DigitalCoast",
        )

        mock_get_s3_dc.assert_called_once_with(
            ["bucket/ER_3/subfolder/DigitalCoast/ProviderA/dem/tile_1.tiff"]
        )
        mock_build_vrt.assert_called_once()
        victim.close_dask.assert_called_once()