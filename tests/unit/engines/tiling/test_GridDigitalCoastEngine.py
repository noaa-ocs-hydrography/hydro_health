import pytest
import pathlib
import json
import os
import pandas as pd
import geopandas as gpd
from unittest.mock import MagicMock, patch, call
from shapely.geometry import Polygon

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[1]
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.helpers.tools import Param
from hydro_health.engines.tiling.GridDigitalCoastEngine import GridDigitalCoastEngine, _grid_single_vrt_s3


ENGINE_PATH = 'hydro_health.engines.tiling.GridDigitalCoastEngine'


@pytest.fixture
def victim():
    """Provides the Engine instance with mocked parameters and dask client."""
    mock_param = MagicMock()
    mock_param.valueAsText = "/tmp/output"
    
    param_lookup = {
        'output_directory': mock_param,
        'env': 'aws'
    }
    
    # Patch Client to prevent Dask from starting a real local cluster
    with patch(f'{ENGINE_PATH}.Client'):
        engine = GridDigitalCoastEngine(param_lookup)
        engine.write_message = MagicMock()
        engine.client = MagicMock() # Mocked Dask Client
        return engine


def test_grid_single_vrt_s3_success(victim):
    """
    Tests the standalone S3 gridding logic.
    Ensures that if the input exists and output is missing, GDAL Warp is called.
    """
    
    mock_geom = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    blue_topo_gdf = gpd.GeoDataFrame({
        'tile': ['tile_123'],
        'geometry': [mock_geom]
    }, crs="EPSG:4326")

    params = [
        "s3://bucket/test_vrt_name.vrt",
        "s3://bucket/ER_1",
        ["tile_123"],
        blue_topo_gdf,
        victim.param_lookup
    ]

    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem') as mock_s3_cls, \
         patch(f'{ENGINE_PATH}.gdal') as mock_gdal, \
         patch(f'{ENGINE_PATH}.gpd.read_file') as mock_read_file, \
         patch(f'{ENGINE_PATH}.get_config_item') as mock_cfg, \
         patch(f'{ENGINE_PATH}.GridDigitalCoastEngine') as mock_engine_cls, \
         patch(f'{ENGINE_PATH}.os.remove'), \
         patch(f'{ENGINE_PATH}.pathlib.Path.exists', return_value=True):

        mock_engine_inst = mock_engine_cls.return_value
        mock_engine_inst.write_message = MagicMock()

        # Configure S3 Mocks
        mock_fs = mock_s3_cls.return_value
        mock_fs.glob.return_value = ["s3://bucket/data/test_dis.shp"]
        
        def exists_side_effect(path):
            return not str(path).endswith(".tiff")
        mock_fs.exists.side_effect = exists_side_effect
        
        # Setup Config and GDAL
        mock_cfg.return_value = "tiled_output"
        mock_ds = MagicMock()
        mock_gdal.Open.return_value = mock_ds
        mock_ds.GetProjection.return_value = "EPSG:4326"
        
        # Mock the shapefile reading
        mock_dissolve_gdf = MagicMock()
        mock_dissolve_gdf.crs = blue_topo_gdf.crs
        mock_dissolve_gdf.union_all.return_value = mock_geom
        mock_read_file.return_value = mock_dissolve_gdf

        result = _grid_single_vrt_s3(params)

        assert "Processed S3" in result
        mock_gdal.Warp.assert_called_once()
        mock_fs.put.assert_called_once()
        mock_engine_inst.write_message.assert_called()


def test_process_s3_vrt_gridding_orchestration(victim):
    """Tests that the engine finds ecoregions and maps them via Dask."""
    
    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem') as mock_s3_cls, \
         patch(f'{ENGINE_PATH}.get_config_item') as mock_cfg:
        
        mock_fs = mock_s3_cls.return_value
        mock_fs.glob.side_effect = [
            ["s3://bucket/ER_01"],           # ecoregion_paths
            ["s3://bucket/ER_01/dc/v1.vrt"]  # vrt_files
        ]
        mock_fs.ls.return_value = ["s3://bucket/ER_01/bt/BlueTopo/tile_A"]
        mock_fs.isdir.return_value = True
        mock_cfg.return_value = "subfolder"

        victim.process_s3_vrt_gridding(MagicMock(), "/tmp/out", False)

        assert victim.client.map.called
        assert victim.client.gather.called


@patch(f'{ENGINE_PATH}.gpd.read_file')
def test_run_local_branch(mock_gpd_read, victim):
    """Tests the run() method switches to local processing when env='local'."""
    
    victim.param_lookup['env'] = 'local'
    victim.process_local_vrt_gridding = MagicMock()
    
    mock_df = MagicMock()
    mock_gpd_read.return_value = mock_df
    
    victim.client.scatter.return_value = [MagicMock(name="future_obj")]
    
    with patch(f'{ENGINE_PATH}.get_config_item') as mock_cfg, \
         patch(f'{ENGINE_PATH}.INPUTS', new=pathlib.Path("/fake/path")), \
         patch.object(victim, 'setup_dask'), \
         patch.object(victim, 'close_dask'):

        mock_cfg.return_value = "fake_val"

        victim.run()

        victim.process_local_vrt_gridding.assert_called_once()
        victim.client.scatter.assert_called_once()