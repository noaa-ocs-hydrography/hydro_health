import pytest
import pathlib
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
from unittest.mock import MagicMock, patch, mock_open
from osgeo import gdal
from botocore import UNSIGNED

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
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
        
        result = victim.download_nbs_tile(tmp_path, "T1", "ER1", False, 8)
        
        # Verify the download was called
        assert mock_bucket.download_file.call_count == 2
        # Verify it returns the path to the .tiff specifically
        assert result.suffix == '.tiff'
        assert "ER1" in str(result)


import os

def test_upload_current_tiles_to_s3(victim, tmp_path):
    """Test that the uploader correctly maps local paths to S3 keys based on ecoregion."""

    eco_dir = tmp_path / "ER1"
    eco_dir.mkdir()
    dummy_file = eco_dir / "test_tile.tiff"
    dummy_file.write_text("data")

    # Match OS-specific path separators for local path evaluation
    expected_s3_key = os.path.join("ER1", "test_tile.tiff")

    with patch(f'{TILING_PATH}.get_config_item') as mock_cfg, \
         patch('boto3.client') as mock_client:
        
        mock_cfg.return_value = "ocs-dev-csdl-hydrohealth"

        victim.upload_current_tiles_to_s3(eco_dir, tmp_path)
        
        mock_client.return_value.upload_file.assert_called_once_with(
            str(dummy_file), 
            "ocs-dev-csdl-hydrohealth", 
            expected_s3_key
        )


def test_create_slope(victim):
    """Verify GDAL DEMProcessing is called with 'slope'."""

    test_path = pathlib.Path("/tmp/tile.tiff")
    with patch('osgeo.gdal.DEMProcessing') as mock_dem:
        victim.create_slope(test_path)
        expected_out = "\\tmp\\tile_slope.tiff"
        mock_dem.assert_called_once_with(expected_out, str(test_path), 'slope')


def test_set_ground_to_nodata(victim, tmp_path):
    """Verify that values >= 0 are masked to -9999 using rasterio block processing."""
    test_path = tmp_path / "test_tile.tiff"

    # 1. Create a real 2x2 GeoTIFF on disk
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

    # 2. Run the function
    victim.set_ground_to_nodata(test_path)

    # 3. Read back and verify output values
    with rasterio.open(test_path, 'r') as src:
        result = src.read(1)
        assert result[0, 1] == -9999  # 5 becomes -9999
        assert result[1, 0] == -9999  # 0 becomes -9999
        assert result[0, 0] == -10    # -10 stays unchanged
        assert result[1, 1] == -5     # -5 stays unchanged


def test_process_tile_wrapper():
    """Tests the static _process_tile function's sequence of events."""

    param_inputs = [{'env': 'aws'}, 'tile.tiff', 'ER_3', '', '']
    
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