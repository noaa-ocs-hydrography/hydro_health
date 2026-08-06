import pytest
import pathlib
import pathlib
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[1]

import sys
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.helpers.tools import Param
from hydro_health.engines.tiling.RasterMaskS3Engine import RasterMaskS3Engine


OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'
TILING_PATH = 'hydro_health.engines.tiling.RasterMaskS3Engine'


@pytest.fixture
def victim():
    param_lookup = {
        'output_directory': Param(str(OUTPUTS)),
        'env': 'aws'
    }
    engine = RasterMaskS3Engine(param_lookup)
    engine.write_message = MagicMock()
    return engine


def test_create_prediction_mask(victim):
    ecoregion = "ER1"
    wkt = "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0))"

    # Create Mocks
    mock_ds = MagicMock()
    mock_layer = MagicMock()
    mock_s3 = MagicMock()

    # Patch import from the class to catch correct gdal object
    with patch(f'{TILING_PATH}.gdal') as mock_gdal, \
         patch(f'{TILING_PATH}.ogr') as mock_ogr, \
         patch(f'{TILING_PATH}.osr') as mock_osr, \
         patch(f'{TILING_PATH}.boto3') as mock_boto, \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg:

        # Setup Config
        mock_cfg.return_value = "test-folder"
        
        mock_ogr.GetDriverByName.return_value.CreateDataSource.return_value.CreateLayer.return_value = mock_layer
        mock_layer.GetExtent.return_value = (0, 8, 0, 8) 
        
        mock_gdal.GetDriverByName.return_value.Create.return_value = mock_ds
        
        mock_boto.client.return_value = mock_s3

        victim.create_prediction_mask(ecoregion, wkt)

        mock_gdal.RasterizeLayer.assert_called_once()
        mock_s3.upload_file.assert_called_once()


@patch('hydro_health.engines.tiling.RasterMaskS3Engine._set_gdal_s3_options')
def test_create_training_mask(mock_gdal_opts, victim):
    """Tests the multiprocessing orchestration and tile-based merging."""

    TILING_PATH = 'hydro_health.engines.tiling.RasterMaskS3Engine'
    ecoregion = "ER1"
    vrt_paths = ["s3://path/v1.vrt"]
    
    # Small chunk to satisfy logic: win_y, win_x
    dummy_data = np.ones((10, 10), dtype=np.uint8)

    mock_train_ds = MagicMock()
    mock_train_ds.RasterXSize = 10
    mock_train_ds.RasterYSize = 10
    
    mock_train_band = MagicMock()
    # ReadAsArray must return a real array
    mock_train_band.ReadAsArray.return_value = dummy_data.copy()
    mock_train_ds.GetRasterBand.return_value = mock_train_band

    mock_part_ds = MagicMock()
    mock_part_band = MagicMock()
    # ReadAsArray must return a real array
    mock_part_band.ReadAsArray.return_value = dummy_data.copy()
    mock_part_ds.GetRasterBand.return_value = mock_part_band

    # Patch imports from class 
    with patch(f'{TILING_PATH}.boto3.client') as mock_boto, \
         patch(f'{TILING_PATH}.gdal') as mock_gdal, \
         patch(f'{TILING_PATH}.ProcessPoolExecutor') as mock_exec, \
         patch(f'{TILING_PATH}.get_config_item') as mock_cfg:
        
        mock_cfg.return_value = "test-value"
        
        # gdal.Open is called multiple times
        mock_gdal.Open.side_effect = [mock_train_ds, mock_train_ds, mock_part_ds]
        
        # Mock the multiprocessing return
        mock_executor_instance = mock_exec.return_value.__enter__.return_value
        mock_executor_instance.map.return_value = ["/tmp/part_1.tif"]

        result = victim.create_training_mask(ecoregion, vrt_paths, "out_stream")

        assert "completed" in result
        # Check that the training band actually received data
        mock_train_band.WriteArray.assert_called() 
        # Verify the final compression call
        mock_gdal.Translate.assert_called() 
        # Verify S3 interaction
        mock_boto.return_value.upload_file.assert_called()


def test_find_provider_vrts(victim):
    """Tests S3 path globbing logic."""

    with patch('s3fs.S3FileSystem') as mock_s3, \
         patch('hydro_health.engines.tiling.RasterMaskS3Engine.get_config_item') as mock_cfg:
        
        # Setup mocks
        mock_cfg.side_effect = lambda section, key: "bucket" if key == "OUTPUT_BUCKET" else "dc"
        mock_fs = mock_s3.return_value
        mock_fs.glob.return_value = ["s3://bucket/ER1/dc/DigitalCoast/mosaic_1.vrt"]

        results = victim.find_provider_vrts("ER1", manual_downloads=False)

        assert len(results) == 1
        assert "mosaic_1.vrt" in results[0]
        mock_fs.glob.assert_called_once()


def test_remerge_training_mask_error_handling(victim):
    """Tests that remerge fails gracefully if base_pred is missing."""

    with patch('pathlib.Path.exists') as mock_exists:
        mock_exists.return_value = False
        
        result = victim.remerge_training_mask("ER1", "out")
        
        assert "Error" in result
        assert "base_pred.tif not found" in result


@patch('hydro_health.engines.tiling.RasterMaskS3Engine.RasterMaskS3Engine.create_prediction_mask')
@patch('hydro_health.engines.tiling.RasterMaskS3Engine.RasterMaskS3Engine.create_training_mask')
@patch('hydro_health.engines.tiling.RasterMaskS3Engine.RasterMaskS3Engine.find_provider_vrts')
@patch('geopandas.read_file')
def test_run(mock_gpd, mock_find, mock_train, mock_pred, victim):
    """Tests the main loop logic over the GeoDataFrame."""

    mock_df = pd.DataFrame({
        'EcoRegion': ['ER1'],
        'geometry': [MagicMock(wkt="POLYGON(...)")]
    })
    mock_gpd.return_value.to_crs.return_value = mock_df
    
    mock_find.return_value = ["s3://some.vrt"]
    
    with patch('hydro_health.engines.tiling.RasterMaskS3Engine.get_config_item') as mock_cfg, \
         patch('s3fs.S3FileSystem') as mock_s3:
        
        mock_cfg.return_value = "bucket"
        mock_s3.return_value.glob.return_value = ["s3://bucket/ER1"]
        
        victim.run("out_stream")

        mock_pred.assert_called_once()
        mock_train.assert_called_once()
        mock_find.assert_called_once_with('ER1', False)