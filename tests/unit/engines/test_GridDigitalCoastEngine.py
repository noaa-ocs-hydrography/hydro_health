import pytest
import pathlib
import sys
import geopandas as gpd
from unittest.mock import MagicMock, patch
from shapely.geometry import Polygon

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / "src"
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.GridDigitalCoastEngine import (
    GridDigitalCoastEngine,
    _grid_single_vrt_s3,
)

ENGINE_PATH = 'hydro_health.engines.GridDigitalCoastEngine'


@pytest.fixture
def victim():
    """Provides the Engine instance with mocked parameters and dask client."""
    mock_param = MagicMock()
    mock_param.valueAsText = "/tmp/output"

    param_lookup = {
        'output_directory': mock_param,
        'env': 'aws'
    }

    with patch(f'{ENGINE_PATH}.Client', create=True):
        engine = GridDigitalCoastEngine(param_lookup)
        engine.write_message = MagicMock()
        engine.client = MagicMock()
        return engine


def _configure_mock_gdal(mock_gdal, min_max=(1.0, 50.0), nodata=-9999):
    """Helper to configure valid raster dimensions and band stats on mocked GDAL datasets."""
    mock_ds = MagicMock()
    mock_ds.GetProjection.return_value = "EPSG:4326"
    mock_ds.RasterXSize = 100
    mock_ds.RasterYSize = 100
    mock_ds.RasterCount = 1

    mock_band = MagicMock()
    mock_band.GetNoDataValue.return_value = nodata
    mock_band.DataType = mock_gdal.GDT_Float32
    mock_band.ComputeRasterMinMax.return_value = min_max

    mock_ds.GetRasterBand.return_value = mock_band
    mock_gdal.Open.return_value = mock_ds
    return mock_ds


@pytest.mark.parametrize(
    "vrt_s3_path, expected_shp_glob, expected_s3_out",
    [
        (
            "s3:\\bucket\\ER_1\\dc\\DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372.vrt",
            "s3:\\bucket\\ER_1\\dc\\DigitalCoast/2016_DEM_NOAA_NGS_48372/**/*.shp",
            "s3://bucket/ER_1/tiled_output/tile_123/mosaic_2016_DEM_NOAA_NGS_48372_tile_123.tiff",
        ),
        (
            "s3:\\bucket\\ER_1\\dc\\DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372_DTS.vrt",
            "s3:\\bucket\\ER_1\\dc\\DigitalCoast/2016_DEM_NOAA_NGS_48372/**/*.shp",
            "s3://bucket/ER_1/tiled_output/tile_123/mosaic_2016_DEM_NOAA_NGS_48372_tile_123_DTS.tiff",
        ),
    ],
)
def test_grid_single_vrt_s3_success(victim, vrt_s3_path, expected_shp_glob, expected_s3_out):
    """Tests S3 gridding for both standard and _DTS VRTs, verifying shapefile lookup and output naming."""
    mock_geom = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    blue_topo_gdf = gpd.GeoDataFrame(
        {'tile': ['tile_123'], 'geometry': [mock_geom]},
        crs="EPSG:4326",
    )

    params = [
        vrt_s3_path,
        "s3://bucket/ER_1",
        ["tile_123"],
        blue_topo_gdf,
        victim.param_lookup,
    ]

    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem') as mock_s3_cls, \
         patch(f'{ENGINE_PATH}.gdal') as mock_gdal, \
         patch(f'{ENGINE_PATH}.gpd.read_file') as mock_read_file, \
         patch(f'{ENGINE_PATH}.get_config_item') as mock_cfg, \
         patch(f'{ENGINE_PATH}.GridDigitalCoastEngine') as mock_engine_cls, \
         patch(f'{ENGINE_PATH}.os.remove'):

        mock_engine_inst = mock_engine_cls.return_value
        mock_engine_inst.write_message = MagicMock()

        mock_fs = mock_s3_cls.return_value
        mock_fs.glob.return_value = ["s3://bucket/ER_1/dc/DigitalCoast/2016_DEM_NOAA_NGS_48372/tileindex.shp"]
        mock_fs.exists.side_effect = lambda path: not str(path).endswith(".tiff")

        mock_cfg.return_value = "tiled_output"
        _configure_mock_gdal(mock_gdal, min_max=(1.0, 50.0), nodata=-9999)

        mock_dissolve_gdf = MagicMock()
        mock_dissolve_gdf.crs = blue_topo_gdf.crs
        mock_dissolve_gdf.geometry.union_all.return_value = mock_geom
        mock_read_file.return_value = mock_dissolve_gdf

        result = _grid_single_vrt_s3(params)

        assert "Processed S3" in result
        mock_fs.glob.assert_called_once_with(expected_shp_glob)
        mock_gdal.Warp.assert_called_once()
        mock_fs.put.assert_called_once()
        assert mock_fs.put.call_args[0][1] == expected_s3_out
        mock_engine_inst.write_message.assert_called()


def test_grid_single_vrt_s3_skips_empty_warp(victim):
    """Tests that warped tiles containing only NoData are removed locally and not uploaded to S3."""
    mock_geom = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    blue_topo_gdf = gpd.GeoDataFrame(
        {'tile': ['tile_123'], 'geometry': [mock_geom]},
        crs="EPSG:4326",
    )

    params = [
        "s3://bucket/ER_1/dc/DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372_DTS.vrt",
        "s3://bucket/ER_1",
        ["tile_123"],
        blue_topo_gdf,
        victim.param_lookup,
    ]

    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem') as mock_s3_cls, \
         patch(f'{ENGINE_PATH}.gdal') as mock_gdal, \
         patch(f'{ENGINE_PATH}.gpd.read_file') as mock_read_file, \
         patch(f'{ENGINE_PATH}.get_config_item') as mock_cfg, \
         patch(f'{ENGINE_PATH}.GridDigitalCoastEngine') as mock_engine_cls, \
         patch(f'{ENGINE_PATH}.os.path.exists', return_value=True), \
         patch(f'{ENGINE_PATH}.os.remove') as mock_remove:

        mock_fs = mock_s3_cls.return_value
        mock_fs.glob.return_value = ["s3://bucket/ER_1/dc/DigitalCoast/2016_DEM_NOAA_NGS_48372/tileindex.shp"]
        mock_fs.exists.side_effect = lambda path: not str(path).endswith(".tiff")

        mock_cfg.return_value = "tiled_output"
        _configure_mock_gdal(mock_gdal, min_max=(-9999, -9999), nodata=-9999)

        mock_dissolve_gdf = MagicMock()
        mock_dissolve_gdf.crs = blue_topo_gdf.crs
        mock_dissolve_gdf.geometry.union_all.return_value = mock_geom
        mock_read_file.return_value = mock_dissolve_gdf

        result = _grid_single_vrt_s3(params)

        assert "Processed S3" in result
        mock_gdal.Warp.assert_called_once()
        mock_remove.assert_called_once()
        mock_fs.put.assert_not_called()


def test_process_s3_vrt_gridding_digital_coast(victim):
    """Tests that DigitalCoast filters out _DTS VRTs and unapproved providers."""
    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem') as mock_s3_cls, \
         patch(f'{ENGINE_PATH}.get_config_item', return_value="subfolder"), \
         patch(f'{ENGINE_PATH}.get_approved_providers', return_value=["DEM_NOAA_NGS_48372"]):

        mock_fs = mock_s3_cls.return_value
        mock_fs.glob.side_effect = [
            ["s3://bucket/ER_01"],
            [
                "s3://bucket/ER_01/subfolder/DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372.vrt",
                "s3://bucket/ER_01/subfolder/DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372_DTS.vrt",
                "s3://bucket/ER_01/subfolder/DigitalCoast/mosaic_2016_DEM_UNAPPROVED_99999.vrt",
            ],
        ]
        mock_fs.ls.return_value = ["s3://bucket/ER_01/subfolder/BlueTopo/tile_A"]
        mock_fs.isdir.return_value = True

        victim.process_s3_vrt_gridding(MagicMock(), "/tmp/out", "DigitalCoast", "")

        victim.client.map.assert_called_once()
        mapped_params = victim.client.map.call_args[0][1]
        assert len(mapped_params) == 1
        assert mapped_params[0][0] == "s3://bucket/ER_01/subfolder/DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372.vrt"
        victim.client.gather.assert_called_once()


def test_process_s3_vrt_gridding_distance_to_shore(victim):
    """Tests that DistanceToShore scans both VRT folders and maps only approved _DTS VRTs."""
    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem') as mock_s3_cls, \
         patch(f'{ENGINE_PATH}.get_config_item', return_value="subfolder"), \
         patch(f'{ENGINE_PATH}.get_approved_providers', return_value=["DEM_NOAA_NGS_48372", "Lidar_USGS_48152"]):

        mock_fs = mock_s3_cls.return_value
        mock_fs.glob.side_effect = [
            ["s3://bucket/ER_01"],
            [
                "s3://bucket/ER_01/subfolder/DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372.vrt",
                "s3://bucket/ER_01/subfolder/DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372_DTS.vrt",
            ],
            [
                "s3://bucket/ER_01/subfolder/Digital_Coast_Manual_Downloads/mosaic_1998_Lidar_USGS_48152_DTS.vrt",
                "s3://bucket/ER_01/subfolder/Digital_Coast_Manual_Downloads/mosaic_2004_Lidar_UNAPPROVED_50100_DTS.vrt",
            ],
        ]
        mock_fs.ls.return_value = ["s3://bucket/ER_01/subfolder/BlueTopo/tile_A"]
        mock_fs.isdir.return_value = True

        victim.process_s3_vrt_gridding(MagicMock(), "/tmp/out", "DistanceToShore", "my_prefix")

        victim.client.map.assert_called_once()
        mapped_params = victim.client.map.call_args[0][1]
        mapped_vrts = [p[0] for p in mapped_params]
        assert mapped_vrts == [
            "s3://bucket/ER_01/subfolder/DigitalCoast/mosaic_2016_DEM_NOAA_NGS_48372_DTS.vrt",
            "s3://bucket/ER_01/subfolder/Digital_Coast_Manual_Downloads/mosaic_1998_Lidar_USGS_48152_DTS.vrt",
        ]
        victim.client.gather.assert_called_once()


@patch(f'{ENGINE_PATH}.gpd.read_file')
def test_run_local_branch(mock_gpd_read, victim):
    """Tests the run() method switches to local processing when env='local'."""
    victim.param_lookup['env'] = 'local'
    victim.process_local_vrt_gridding = MagicMock()

    mock_df = MagicMock()
    mock_gpd_read.return_value = mock_df
    victim.client.scatter.return_value = [MagicMock(name="future_obj")]

    with patch(f'{ENGINE_PATH}.get_config_item', return_value="fake_val"), \
         patch(f'{ENGINE_PATH}.INPUTS', new=pathlib.Path("/fake/path")), \
         patch.object(victim, 'setup_dask'), \
         patch.object(victim, 'close_dask'):

        victim.run("test_prefix", data_type="DistanceToShore")

        victim.process_local_vrt_gridding.assert_called_once()
        assert victim.process_local_vrt_gridding.call_args[0][2] == "DistanceToShore"
        assert victim.process_local_vrt_gridding.call_args[0][3] == "test_prefix"
        victim.client.scatter.assert_called_once()


def test_run_invalid_data_type_raises(victim):
    """Tests that run() raises ValueError when given an unsupported data_type."""
    with pytest.raises(ValueError, match="Invalid data_type"):
        victim.run("test_prefix", data_type="InvalidType")