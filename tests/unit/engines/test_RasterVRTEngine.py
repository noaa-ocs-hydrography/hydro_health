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
    return engine


@pytest.fixture
def create_dummy_geotiff(tmp_path):
    """Utility fixture to create a valid small GeoTIFF with GDAL."""

    def _create(filename="test.tiff", epsg=32618, nodata=-9999.0):
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


def test_get_local_digitalcoast_geotiffs(victim, create_dummy_geotiff, tmp_path):
    """Tests extracting CRS and provider info from local DigitalCoast rasters."""
    gtif = create_dummy_geotiff("DigitalCoast/NOAA_Provider/data.tif", epsg=32618)

    result = victim.get_local_digitalcoast_geotiffs([gtif])

    assert "NOAA_Provider" in result
    assert result["NOAA_Provider"]["tiles"] == [str(gtif)]
    assert result["NOAA_Provider"]["primary_crs"] == "EPSG:32618"
    assert result["NOAA_Provider"]["nodata_val"] == -9999.0


def test_get_local_digitalcoast_geotiffs_manual_downloads(victim, create_dummy_geotiff):
    """Tests provider extraction using the Digital_Coast_Manual_Downloads path variant."""
    gtif = create_dummy_geotiff("Digital_Coast_Manual_Downloads/USACE_Provider/data.tif", epsg=32618)

    result = victim.get_local_digitalcoast_geotiffs([gtif])

    assert "USACE_Provider" in result
    assert result["USACE_Provider"]["tiles"] == [str(gtif)]


def test_build_output_vrts_bluetopo(victim, tmp_path):
    """Tests VRT creation using BlueTopo options (disallow projection difference)."""
    tile_a = tmp_path / "tile_a.tif"
    tile_a.touch()

    output_geotiffs = {
        '32618': {
            'tiles': [str(tile_a)],
            'nodata_val': -9999.0
        }
    }

    with patch(f'{ENGINE_PATH}.gdal.BuildVRT') as mock_build, \
         patch(f'{ENGINE_PATH}.gdal.BuildVRTOptions') as mock_options:

        victim.build_output_vrts(tmp_path, 'elevation', output_geotiffs, 'BlueTopo')

        mock_options.assert_called_once_with(
            resampleAlg='bilinear',
            allowProjectionDifference=False,
            srcNodata=-9999.0,
            VRTNodata=-9999.0
        )
        mock_build.assert_called_once()


def test_build_output_vrts_digitalcoast(victim, tmp_path):
    """Tests VRT creation using DigitalCoast options (near resampling, allow difference)."""
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

        victim.build_output_vrts(tmp_path, 'elevation', output_geotiffs, 'DigitalCoast')

        mock_options.assert_called_once_with(
            resampleAlg='near',
            allowProjectionDifference=True,
            srcNodata=-9999.0,
            VRTNodata=-9999.0
        )
        mock_build.assert_called_once()


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


def test_run_local_digitalcoast(victim, tmp_path):
    """Tests the DigitalCoast local branch in run() filtering provider folders."""
    with patch(f'{ENGINE_PATH}.get_config_item', return_value="subfolder"), \
         patch.object(victim, 'get_local_digitalcoast_geotiffs', return_value={}) as mock_get_dc, \
         patch.object(victim, 'build_output_vrts') as mock_build_vrt:

        target_dir = tmp_path / "prefix" / "ER_3" / "subfolder" / "DigitalCoast"
        provider_a = target_dir / "ProviderA"
        provider_a.mkdir(parents=True)
        (provider_a / "tile_1.tiff").touch()

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

        assert mock_get_dc.call_count == 1
        assert mock_build_vrt.call_count == 1