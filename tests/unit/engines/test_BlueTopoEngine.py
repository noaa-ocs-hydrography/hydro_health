import os
import sys
import pathlib
import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from unittest.mock import MagicMock, patch
from botocore import UNSIGNED

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.BlueTopoEngine import BlueTopoEngine, _process_tile
from hydro_health.helpers.tools import Param

TILING_PATH = 'hydro_health.engines.BlueTopoEngine'


@pytest.fixture
def victim(tmp_path):
    """Fixture to provide an instance of BlueTopoEngine with mocked internals."""
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'aws',
        'eco_regions': Param(['ER1'])
    }
    engine = BlueTopoEngine(param_lookup)
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.client = MagicMock()
    return engine


def test_get_bucket(victim):
    """Verify S3 connection uses UNSIGNED config for the specific NOAA bucket."""
    with patch('boto3.resource') as mock_boto_res:
        victim.get_bucket()
        
        _, kwargs = mock_boto_res.call_args
        assert kwargs['config'].signature_version is UNSIGNED
        mock_boto_res.return_value.Bucket.assert_called_with("noaa-ocs-nationalbathymetry-pds")


def test_download_nbs_tile(victim, tmp_path):
    """Test filtering logic and local/temp destination path resolution during download."""
    mock_bucket = MagicMock()
    
    mock_obj_tiff = MagicMock(key="BlueTopo/B1234567/tile.tiff")
    mock_obj_xml = MagicMock(key="BlueTopo/B1234567/tile.xml")
    mock_bucket.objects.filter.return_value = [mock_obj_tiff, mock_obj_xml]
    
    with patch.object(victim, 'get_bucket', return_value=mock_bucket), \
         patch(f'{TILING_PATH}.get_config_item', return_value="sub"), \
         patch('pathlib.Path.exists', return_value=False):
        
        result = victim.download_nbs_tile(
            tile_id="B1234567",
            ecoregion_id="ER1",
            output_prefix=False,
            target_res=8,
            temp_folder=tmp_path
        )
        
        # Verify both tiff and xml downloads were executed
        assert mock_bucket.download_file.call_count == 2
        # Verify returned tile path is a GeoTIFF inside the target ecoregion path
        assert result.suffix == '.tiff'
        assert "ER1" in str(result)


def test_upload_current_tiles_to_s3(victim, tmp_path):
    """Test that the uploader maps local generated paths to target S3 keys."""
    tile_dir = tmp_path / "ER1"
    tile_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = tile_dir / "test_tile.tiff"
    dummy_file.write_text("data")

    expected_s3_key = str(dummy_file.relative_to(tmp_path))

    with patch(f'{TILING_PATH}.get_config_item', return_value="ocs-dev-csdl-hydrohealth") as mock_cfg, \
         patch('boto3.client') as mock_client:
        
        victim.upload_current_tiles_to_s3(tile_dir, tmp_path)
        
        mock_client.return_value.upload_file.assert_called_once_with(
            str(dummy_file), 
            "ocs-dev-csdl-hydrohealth", 
            expected_s3_key
        )


def test_create_slope(victim):
    """Verify GDAL DEMProcessing is invoked with 'slope' algorithm."""
    test_path = pathlib.Path("/tmp/tile.tiff")
    expected_out = str(test_path.parent / "tile_slope.tiff")

    with patch('osgeo.gdal.DEMProcessing') as mock_dem:
        victim.create_slope(test_path)
        mock_dem.assert_called_once_with(expected_out, str(test_path), 'slope')


def test_set_ground_to_nodata(victim, tmp_path):
    """Verify that elevation values >= 0 are masked to -9999 via rasterio streaming windows."""
    test_path = tmp_path / "test_tile.tiff"

    # Create test 2x2 raster
    input_data = np.array([[-10, 5], [0, -5]], dtype=np.float32)
    
    with rasterio.open(
        test_path,
        'w',
        driver='GTiff',
        height=2,
        width=2,
        count=1,
        dtype=input_data.dtype,
        nodata=-9999
    ) as dst:
        dst.write(input_data, 1)

    # Run masking process
    victim.set_ground_to_nodata(test_path)

    # Verify updated values
    with rasterio.open(test_path, 'r') as src:
        result = src.read(1)
        assert result[0, 1] == -9999  # 5 becomes nodata
        assert result[1, 0] == -9999  # 0 becomes nodata
        assert result[0, 0] == -10    # -10 remains unchanged
        assert result[1, 1] == -5     # -5 remains unchanged


def test_process_tile_wrapper():
    """Test execution workflow order inside static _process_tile Dask worker entrypoint."""
    param_lookup = {
        'env': 'aws',
        'output_directory': Param('/tmp/out')
    }
    param_inputs = [param_lookup, 'B1234567', 'ER1', 'regional_model', 8]
    fake_tiff = pathlib.Path("/tmp/fake/tile.tiff")
    fake_mb_tiff = pathlib.Path("/tmp/fake/tile_mb.tiff")

    with patch(f'{TILING_PATH}.BlueTopoEngine') as MockEngine, \
         patch('tempfile.TemporaryDirectory') as mock_temp:
        
        mock_temp.return_value.__enter__.return_value = "/tmp/fake"
        instance = MockEngine.return_value
        instance.scratch_dir = pathlib.Path("/tmp/scratch")
        instance.download_nbs_tile.return_value = fake_tiff
        instance.rename_multiband.return_value = fake_mb_tiff

        # Execute tile processing pipeline
        status = _process_tile(param_inputs)

        # Assert full workflow completion sequence
        instance.download_nbs_tile.assert_called_once()
        instance.resample_and_reproject.assert_called_once_with(fake_tiff, 8)
        instance.create_survey_end_date_tiff.assert_called_once_with(fake_tiff)
        instance.create_catzoc_all.assert_called_once_with(fake_tiff, increased_scale=True)
        instance.create_catzoc_latest.assert_called_once_with(fake_tiff, increased_scale=True)
        instance.create_slope.assert_called_once_with(fake_tiff)
        instance.rename_multiband.assert_called_once_with(fake_tiff)
        assert instance.multiband_to_singleband.call_count == 2
        instance.set_ground_to_nodata.assert_called_once_with(fake_tiff)
        instance.finalize_cog.assert_called_once_with(fake_tiff)
        instance.upload_current_tiles_to_s3.assert_called_once()
        
        assert "successfully completed" in status