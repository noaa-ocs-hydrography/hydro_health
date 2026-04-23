import pytest
import pathlib
import pandas as pd
import geopandas as gpd
import numpy as np

from unittest.mock import MagicMock, patch, call, mock_open
from shapely.geometry import Polygon
from botocore import UNSIGNED

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[1]
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.helpers.tools import Param
from hydro_health.engines.tiling.DigitalCoastS3Engine import DigitalCoastS3Engine, _download_tile_index


TILING_PATH = 'hydro_health.engines.tiling.DigitalCoastS3Engine'


@pytest.fixture
def victim(tmp_path):
    """
    Base victim fixture to return instance of test Class.
    :param pathlib.Path tmp_path: pytest built-in TemporaryFile object magic
    """
    
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'aws'
    }
    engine = DigitalCoastS3Engine(param_lookup)
    engine.write_message = MagicMock()
    engine.client = MagicMock()
    
    # NEW: Mock Dask setup/close inherited from Engine 
    # so it doesn't try to start a real local cluster
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    
    return engine


def test_get_bucket(victim):
    """Tests the anonymous S3 bucket connection logic."""
    with patch(f'{TILING_PATH}.boto3.resource') as mock_boto_res, \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg:
        
        mock_cfg.return_value = "target-bucket"
        victim.get_bucket()
        
        args, kwargs = mock_boto_res.call_args
        # Use 'is' for sentinel objects like UNSIGNED
        assert kwargs['config'].signature_version is UNSIGNED
        mock_boto_res.return_value.Bucket.assert_called_with("target-bucket")


def test_run_orchestration(victim, tmp_path):
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1'],
        'geometry': [Polygon([(0, 0), (1, 1), (0, 1)])]
    }, crs="EPSG:4326")

    with patch.object(victim, 'download_support_files'), \
         patch.object(victim, 'check_tile_index_areas', return_value=[]), \
         patch.object(victim, 'upload_files_to_s3'), \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg:
        
        # CRITICAL: get_config_item MUST return a string. 
        # If it returns a MagicMock, the path logic: path / mock / path 
        # will fail or create objects that rglob cannot search.
        mock_cfg.return_value = "subfolder"
        
        victim.run(mock_gdf)
        
        expected_path = tmp_path / "processed_providers.log"
        assert expected_path.exists()
        # Verify Dask was "set up" and "closed"
        victim.setup_dask.assert_called_once()
        victim.close_dask.assert_called_once()


def test_download_tile_index_logic():
    """Tests the standalone _download_tile_index function."""
    mock_engine = MagicMock()
    mock_bucket = MagicMock()
    mock_engine.get_bucket.return_value = mock_bucket
    
    mock_obj = MagicMock()
    mock_obj.key = "data/tileindex.zip"
    mock_bucket.objects.filter.return_value = [mock_obj]
    
    param_lookup = {'env': 'aws'}
    # Note: the link contains 'dem'
    param_inputs = ["http://dem-data.com/index.html", pathlib.Path("/tmp"), param_lookup, "out"]

    with patch(f'{TILING_PATH}.DigitalCoastS3Engine', return_value=mock_engine), \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg, \
         patch(f'{TILING_PATH}.os.path.exists', return_value=False), \
         patch('builtins.open', mock_open()):
        
        # CRITICAL: The function has a check: if get_config_item(...) in download_link
        # We must make sure this condition is True, or the S3 code is skipped!
        mock_cfg.return_value = "dem-data" 
        
        _download_tile_index(param_inputs)
        
        mock_bucket.download_fileobj.assert_called_once()