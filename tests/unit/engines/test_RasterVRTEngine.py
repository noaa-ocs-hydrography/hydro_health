import pathlib
import pytest
from unittest.mock import MagicMock, patch
from osgeo import gdal, osr

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
print(HYDRO_HEALTH_MODULE)
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.RasterVRTEngine import (
    RasterVRTEngine,
    _clean,
    _local_process_single_bluetopo,
)

ENGINE_PATH = 'hydro_health.engines.RasterVRTEngine'


@pytest.fixture
def victim(tmp_path):
    """Fixture returning an instance of RasterVRTEngine."""
    param_lookup = {'output_directory': str(tmp_path)}
    with patch(f'{ENGINE_PATH}.query_crs_info', return_value=[]):
        engine = RasterVRTEngine(param_lookup)
    return engine


@pytest.fixture
def create_dummy_geotiff(tmp_path):
    """Utility fixture to create a valid small GeoTIFF with GDAL."""
    
    def _create(filename="test.tiff", epsg=4326, nodata=-999999):
        file_path = tmp_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)

        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(str(file_path), 10, 10, 1, gdal.GDT_Float32)
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform([0, 1, 0, 0, 0, -1])

        band = ds.GetRasterBand(1)
        band.SetNoDataValue(nodata)
        band.Fill(nodata)
        ds.FlushCache()
        ds = None

        return file_path

    return _create


def test_clean_helper():
    """Tests the _clean normalization string utility."""
    assert _clean("  EPSG:4326 \n ") == "epsg:4326"
    assert _clean("NAD83  /   UTM zone 18N") == "nad83 / utm zone 18n"
    assert _clean("") == ""
    assert _clean(None) == ""


def test_local_process_single_bluetopo_success(create_dummy_geotiff, tmp_path):
    """Tests warping a GeoTIFF locally into a VRT with EPSG:4326."""
    geotiff = create_dummy_geotiff("blue_01.tiff", epsg=32618)
    output_dir = tmp_path / "warped"
    output_dir.mkdir()

    datum_code, vrt_path, wkt = _local_process_single_bluetopo(geotiff, output_dir)

    assert pathlib.Path(vrt_path).exists()
    assert pathlib.Path(vrt_path).name == "blue_01_warped.vrt"
    assert "GEOGCS[\"WGS 84\"" in wkt or "GEOGCRS[\"WGS 84\"" in wkt


def test_local_process_single_bluetopo_invalid_file(tmp_path):
    """Tests exception handling when processing a non-existent file."""
    fake_path = tmp_path / "non_existent.tiff"
    output_dir = tmp_path / "warped"
    output_dir.mkdir()

    with pytest.raises(RuntimeError) as exc_info:
        _local_process_single_bluetopo(fake_path, output_dir)

    assert "_local_process_single_bluetopo failed" in str(exc_info.value)


def test_build_output_vrts_digitalcoast(victim, tmp_path):
    """Tests VRT creation using DigitalCoast options branch."""

    tile_a = tmp_path / "tile_a.tif"
    tile_a.touch()

    output_geotiffs = {
        '4326': {
            'tiles': [tile_a],
            'nodata_val': -9999,
            'wkt': 'EPSG:4326'
        }
    }

    with patch(f'{ENGINE_PATH}.gdal.BuildVRT') as mock_build, \
         patch(f'{ENGINE_PATH}.gdal.BuildVRTOptions') as mock_options:

        victim.build_output_vrts(tmp_path, 'elevation', output_geotiffs, 'DigitalCoast')

        mock_options.assert_called_once_with(
            resampleAlg='near',
            srcNodata=-9999,
            VRTNodata=-9999,
            addAlpha=True,
            allowProjectionDifference=True,
            outputSRS='EPSG:4326'
        )
        mock_build.assert_called_once()


def test_build_output_vrts_default(victim, tmp_path):
    """Tests VRT creation using generic/BlueTopo bilinear options branch."""
    tile_a = tmp_path / "tile_a.tif"
    tile_a.touch()

    output_geotiffs = {
        '6318': {
            'tiles': [tile_a],
            'nodata_val': -9999,
            'wkt': 'EPSG:6318'
        }
    }

    with patch(f'{ENGINE_PATH}.gdal.BuildVRT') as mock_build, \
         patch(f'{ENGINE_PATH}.gdal.BuildVRTOptions') as mock_options:

        victim.build_output_vrts(tmp_path, 'elevation', output_geotiffs, 'BlueTopo')

        mock_options.assert_called_once_with(
            resampleAlg='bilinear',
            allowProjectionDifference=True
        )
        mock_build.assert_called_once()


def test_get_local_bluetopo_tifs(victim, create_dummy_geotiff, tmp_path):
    """Tests aggregation of warped BlueTopo tiles into output dictionaries."""
    gtif = create_dummy_geotiff("tile_1.tiff", epsg=4326)
    temp_output_path = tmp_path / "temp"
    temp_output_path.mkdir()

    result = victim.get_local_bluetopo_tifs([gtif], temp_output_path)

    assert len(result) > 0
    first_key = list(result.keys())[0]
    assert len(result[first_key]['tiles']) == 1
    assert result[first_key]['nodata_val'] == -999999


def test_get_local_digitalcoast_geotiffs(victim, create_dummy_geotiff, tmp_path):
    """Tests extracting CRS and provider info from local DigitalCoast rasters."""
    dc_folder = tmp_path / "DigitalCoast" / "NOAA_Provider"
    dc_folder.mkdir(parents=True)
    gtif = create_dummy_geotiff("DigitalCoast/NOAA_Provider/data.tif", epsg=4326)

    result = victim.get_local_digitalcoast_geotiffs([gtif])

    assert "NOAA_Provider" in result
    assert result["NOAA_Provider"]["tiles"] == [str(gtif)]
    assert result["NOAA_Provider"]["nodata_val"] == -999999


def test_get_local_digitalcoast_geotiffs_manual_downloads(victim, create_dummy_geotiff, tmp_path):
    """Tests provider extraction using the Digital_Coast_Manual_Downloads path variant."""
    gtif = create_dummy_geotiff("Digital_Coast_Manual_Downloads/USACE_Provider/data.tif", epsg=4326)

    result = victim.get_local_digitalcoast_geotiffs([gtif])

    assert "USACE_Provider" in result
    assert result["USACE_Provider"]["tiles"] == [str(gtif)]


def test_run_bluetopo(victim, tmp_path):
    """Tests the BlueTopo branch in run() using temporary directories."""
    with patch(f'{ENGINE_PATH}.get_config_item', return_value="subfolder"), \
         patch.object(victim, 'get_local_bluetopo_tifs', return_value={}) as mock_get_bt, \
         patch.object(victim, 'build_output_vrts') as mock_build_vrt:

        # Setup expected file tree
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


def test_run_digitalcoast(victim, tmp_path):
    """Tests the DigitalCoast branch in run() filtering provider folders."""
    with patch(f'{ENGINE_PATH}.get_config_item', return_value="subfolder"), \
         patch.object(victim, 'get_local_digitalcoast_geotiffs', return_value={}) as mock_get_dc, \
         patch.object(victim, 'build_output_vrts') as mock_build_vrt:

        # Setup expected provider folders
        target_dir = tmp_path / "prefix" / "ER1" / "subfolder" / "DigitalCoast"
        provider_a = target_dir / "ProviderA"
        provider_a.mkdir(parents=True)
        (provider_a / "tile_1.tiff").touch()

        # Ignore unused_providers
        unused_provider = target_dir / "unused_providers"
        unused_provider.mkdir(parents=True)
        (unused_provider / "tile_2.tiff").touch()

        victim.run(
            output_folder=str(tmp_path),
            file_type="elevation",
            ecoregion="ER1",
            data_type="DigitalCoast",
            output_prefix="prefix"
        )

        # Should only execute once for ProviderA
        assert mock_get_dc.call_count == 1
        assert mock_build_vrt.call_count == 1