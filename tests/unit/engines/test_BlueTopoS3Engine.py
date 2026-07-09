import pytest
import pathlib
import pandas as pd
import geopandas as gpd
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from botocore import UNSIGNED

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[1]
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.BlueTopoS3Engine import BlueTopoS3Engine, _process_tile
from hydro_health.helpers.tools import Param


TILING_PATH = 'hydro_health.engines.BlueTopoS3Engine'


@pytest.fixture
def victim(tmp_path):
    """Fixture to provide an instance of BlueTopoS3Engine with mocked internals."""

    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'aws'
    }
    engine = BlueTopoS3Engine(param_lookup)
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.client = MagicMock()
    return engine


def test_get_bucket(victim):
    """Verify S3 connection uses UNSIGNED config for the specific NOAA bucket."""

    with patch(f'boto3.resource') as mock_boto_res:
        victim.get_bucket()
        
        _, kwargs = mock_boto_res.call_args
        assert kwargs['config'].signature_version is UNSIGNED
        mock_boto_res.return_value.Bucket.assert_called_with("noaa-ocs-nationalbathymetry-pds")


def test_download_nbs_tile(victim, tmp_path):
    """Test the filtering logic and local path creation during download."""

    mock_bucket = MagicMock()
    
    mock_obj_tiff = MagicMock(key="BlueTopo/T1/tile.tiff")
    mock_obj_xml = MagicMock(key="BlueTopo/T1/tile.xml")
    mock_bucket.objects.filter.return_value = [mock_obj_tiff, mock_obj_xml]
    
    with patch.object(victim, 'get_bucket', return_value=mock_bucket), \
         patch(f'{TILING_PATH}.get_config_item', return_value="sub"), \
         patch('pathlib.Path.exists', return_value=False):
        
        result = victim.download_nbs_tile(tmp_path, "T1", "ER1")
        
        # Verify the download was called
        assert mock_bucket.download_file.call_count == 2
        # Verify it returns the path to the .tiff specifically
        assert result.suffix == '.tiff'
        assert "ER1" in str(result)


def test_upload_current_tiles_to_s3(victim, tmp_path):
    """Test that the uploader correctly maps local paths to S3 keys based on ecoregion."""

    eco_dir = tmp_path / "ER1"
    eco_dir.mkdir()
    dummy_file = eco_dir / "test_tile.tiff"
    dummy_file.write_text("data")

    with patch('boto3.client') as mock_client:
        victim.upload_current_tiles_to_s3(eco_dir, "my-bucket", "ER1")
        
        # Ensure upload_file was called with the relative path starting at ER1
        mock_client.return_value.upload_file.assert_called_once_with(
            fr'{tmp_path}\ER1\test_tile.tiff',  # cast pathlib object caused failure
            "my-bucket", 
            r"ER1\test_tile.tiff"
        )


def test_create_slope(victim):
    """Verify GDAL DEMProcessing is called with 'slope'."""

    test_path = pathlib.Path("/tmp/tile.tiff")
    with patch('osgeo.gdal.DEMProcessing') as mock_dem:
        victim.create_slope(test_path)
        expected_out = "\\tmp\\tile_slope.tiff"
        mock_dem.assert_called_once_with(expected_out, str(test_path), 'slope')


def test_set_ground_to_nodata(victim):
    """Verify that values >= 0 are masked to -999999."""

    test_path = pathlib.Path("/tmp/tile.tiff")
    mock_ds = MagicMock()
    # Mock a 2x2 array: two negative (keep), two positive (mask)
    mock_array = np.array([[-10, 5], [0, -5]])
    mock_ds.ReadAsArray.return_value = mock_array
    
    with patch('osgeo.gdal.Open', return_value=mock_ds):
        victim.set_ground_to_nodata(test_path)
        
        # Check that WriteArray was called with the masked data
        written_array = mock_ds.GetRasterBand.return_value.WriteArray.call_args[0][0]
        assert written_array[0, 1] == -999999  # 5 becomes nodata
        assert written_array[1, 0] == -999999  # 0 becomes nodata
        assert written_array[0, 0] == -10      # -10 stays


def test_process_tile_wrapper():
    """Tests the static _process_tile function's sequence of events."""

    param_inputs = [{'env': 'aws'}, "bucket", "tile123", "eco456"]
    
    with patch(f'{TILING_PATH}.BlueTopoS3Engine') as MockEngine, \
         patch('tempfile.TemporaryDirectory') as mock_temp:
        
        mock_temp.return_value.__enter__.return_value = "/tmp/fake"
        instance = MockEngine.return_value
        instance.download_nbs_tile.return_value = pathlib.Path("/tmp/fake/tile.tiff")
        
        _process_tile(param_inputs)
        
        # Verify the sequence of engine calls
        instance.download_nbs_tile.assert_called_once()
        instance.create_slope.assert_called_once()
        instance.finalize_cog.assert_called_once()
        instance.upload_current_tiles_to_s3.assert_called_once()