import pytest
import pathlib
import pandas as pd
import geopandas as gpd
import numpy as np

from unittest.mock import MagicMock, patch, call, mock_open
from shapely.geometry import Polygon
from botocore import UNSIGNED

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.helpers.tools import Param
from hydro_health.engines.DigitalCoastEngine import DigitalCoastEngine, _download_tile_index


TILING_PATH = 'hydro_health.engines.DigitalCoastEngine'


@pytest.fixture
def victim(tmp_path):
    """
    Base victim fixture to return an AWS-configured instance of DigitalCoastEngine.
    :param pathlib.Path tmp_path: pytest built-in TemporaryDirectory object
    """
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'aws'
    }
    engine = DigitalCoastEngine(param_lookup)
    engine.write_message = MagicMock()
    engine.client = MagicMock()
    
    # Mock Dask setup/close inherited from Engine so a real cluster is not started
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.write_run_manifest = MagicMock()
    
    return engine


@pytest.fixture
def victim_local(tmp_path):
    """
    Victim fixture to return a local-configured instance of DigitalCoastEngine.
    :param pathlib.Path tmp_path: pytest built-in TemporaryDirectory object
    """
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'local'
    }
    engine = DigitalCoastEngine(param_lookup)
    engine.write_message = MagicMock()
    engine.client = MagicMock()
    
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.write_run_manifest = MagicMock()
    
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


def test_run_orchestration_aws(victim, tmp_path):
    """Tests the run pipeline execution under AWS environment configuration."""
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1'],
        'geometry': [Polygon([(0, 0), (1, 1), (0, 1)])]
    }, crs="EPSG:4326")

    with patch.object(victim, 'download_support_files'), \
         patch.object(victim, 'check_tile_index_areas', return_value=[]), \
         patch.object(victim, 'upload_files_to_s3'), \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg:
        
        mock_cfg.return_value = "subfolder"
        
        victim.run(mock_gdf, output_prefix=False)
        
        expected_path = tmp_path / "processed_providers.log"
        assert expected_path.exists()
        victim.setup_dask.assert_called_once_with('aws')
        victim.close_dask.assert_called_once()
        victim.write_run_manifest.assert_called()


def test_run_orchestration_local(victim_local, tmp_path):
    """Tests the run pipeline execution under local environment configuration."""
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1'],
        'geometry': [Polygon([(0, 0), (1, 1), (0, 1)])]
    }, crs="EPSG:4326")

    with patch.object(victim_local, 'download_support_files'), \
         patch.object(victim_local, 'check_tile_index_areas'), \
         patch.object(victim_local, 'process_intersected_datasets'), \
         patch.object(victim_local, 'delete_unused_folder'), \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg:
        
        mock_cfg.return_value = "subfolder"
        
        victim_local.run(mock_gdf, output_prefix=False)
        
        # S3 processed_providers.log should NOT be created in local mode
        assert not (tmp_path / "processed_providers.log").exists()
        victim_local.setup_dask.assert_called_once_with('local')
        victim_local.close_dask.assert_called_once()
        victim_local.download_support_files.assert_called_once()
        victim_local.check_tile_index_areas.assert_called_once()
        victim_local.process_intersected_datasets.assert_called_once()


def test_download_tile_index_logic():
    """Tests the standalone _download_tile_index function using DigitalCoastEngine."""
    mock_engine = MagicMock()
    mock_bucket = MagicMock()
    mock_engine.get_bucket.return_value = mock_bucket
    
    mock_obj = MagicMock()
    mock_obj.key = "data/tileindex.zip"
    mock_bucket.objects.filter.return_value = [mock_obj]
    
    param_lookup = {'env': 'aws'}
    param_inputs = ["http://dem-data.com/index.html", pathlib.Path("/tmp"), param_lookup, "out"]

    with patch(f'{TILING_PATH}.DigitalCoastEngine', return_value=mock_engine), \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg, \
         patch(f'{TILING_PATH}.os.path.exists', return_value=False), \
         patch('builtins.open', mock_open()):
        
        mock_cfg.return_value = "dem-data" 
        
        _download_tile_index(param_inputs)
        
        mock_bucket.download_fileobj.assert_called_once()