import os
import sys
import pathlib
import pytest
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from shapely.geometry import box
from unittest.mock import MagicMock, patch

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.DistanceToShoreEngine import DistanceToShoreEngine, _process_tile
from hydro_health.helpers.tools import Param

ENGINE_PATH = 'hydro_health.engines.DistanceToShoreEngine'


@pytest.fixture
def victim(tmp_path):
    """Fixture to provide an instance of DistanceToShoreEngine with mocked internals."""
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'local',
        'eco_regions': Param(['ER_3'])
    }
    engine = DistanceToShoreEngine(param_lookup)
    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.client = MagicMock()
    return engine


@pytest.fixture
def sample_raster(tmp_path):
    """Creates a temporary 256x256 GeoTIFF raster with projected CRS for testing."""
    raster_path = tmp_path / "sample_tile.tiff"
    data = np.full((256, 256), -5.0, dtype=np.float32)
    data[0:64, :] = 5.0

    transform = from_origin(0, 256, 1, 1)
    crs = CRS.from_epsg(32619)

    with rasterio.open(
        raster_path,
        'w',
        driver='GTiff',
        height=256,
        width=256,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(data, 1)

    return raster_path


@pytest.fixture
def sample_ecoregion_gdf():
    """Creates a GeoDataFrame boundary polygon covering the sample raster region."""
    poly = box(-10, -10, 300, 300)
    return gpd.GeoDataFrame({'EcoRegion': ['ER_3']}, geometry=[poly], crs="EPSG:32619")


def test_get_shoreline_vertices_local(victim, sample_raster, tmp_path):
    """Test loading shoreline vertices from local input directory matching EPSG code."""
    victim.param_lookup['env'] = 'local'
    support_dir = tmp_path / "inputs" / "support_data"
    support_dir.mkdir(parents=True, exist_ok=True)
    npy_path = support_dir / "ER_3_shoreline_EPSG_32619.npy"

    dummy_vertices = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    np.save(npy_path, dummy_vertices)

    with patch(f'{ENGINE_PATH}.INPUTS', tmp_path / "inputs"):
        vertices = victim.get_shoreline_vertices(sample_raster, "ER_3")
        np.testing.assert_array_equal(vertices, dummy_vertices)


def test_get_shoreline_vertices_s3(victim, sample_raster):
    """Test loading shoreline vertices from S3 bucket when env is 'aws'."""
    victim.param_lookup['env'] = 'aws'
    dummy_vertices = np.array([[500.0, 600.0], [700.0, 800.0]], dtype=np.float32)

    mock_s3 = MagicMock()
    mock_file = MagicMock()
    mock_s3.open.return_value.__enter__.return_value = mock_file

    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem', return_value=mock_s3), \
         patch(f'{ENGINE_PATH}.get_config_item', return_value="test-bucket"), \
         patch('numpy.load', return_value=dummy_vertices):

        vertices = victim.get_shoreline_vertices(sample_raster, "ER_3")
        np.testing.assert_array_equal(vertices, dummy_vertices)


def test_build_ocean_mask(victim, sample_raster, sample_ecoregion_gdf, tmp_path):
    """Verify binary water mask creation and valid water cell counting."""
    output_file, valid_cells = victim.build_ocean_mask(sample_raster, sample_ecoregion_gdf, tmp_path)

    assert pathlib.Path(output_file).exists()
    assert valid_cells == 49152

    with rasterio.open(output_file) as src:
        mask_data = src.read(1)
        assert src.nodata == 255
        assert np.sum(mask_data == 1) == 49152


def test_build_distance_from_shoreline(victim, tmp_path):
    """Test cKDTree distance raster creation, spatial math, and statistical output dict."""
    mask_path = tmp_path / "test_ocean_mask.tif"
    mask_data = np.ones((256, 256), dtype=np.uint8)

    transform = from_origin(0, 256, 1, 1)
    with rasterio.open(
        mask_path, 'w', driver='GTiff', height=256, width=256, count=1,
        dtype=np.uint8, crs="EPSG:32619", transform=transform, nodata=255
    ) as dst:
        dst.write(mask_data, 1)

    shoreline_vertices = np.array([[0.5, 0.5], [255.5, 0.5]], dtype=np.float32)
    tile_dir = tmp_path / "tile_output"
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Intercept rasterio write call to prevent Windows GDAL C-level libtiff crashes
    mock_writer = MagicMock()
    original_open = rasterio.open

    def side_effect_open(path, mode='r', **kwargs):
        if 'w' in mode:
            return mock_writer
        return original_open(path, mode, **kwargs)

    with patch('rasterio.open', side_effect=side_effect_open):
        dist_path, stats = victim.build_distance_from_shoreline(
            shoreline_vertices, mask_path, tile_dir, "test_tile"
        )

    assert dist_path.name == "test_tile_distance_from_shore.tif"
    assert stats["min"] is not None
    assert stats["max"] >= stats["min"]
    assert isinstance(stats["mean"], float)
    # Confirm the output distance array was generated and passed to rasterio.write()
    mock_writer.__enter__.return_value.write.assert_called_once()


def test_get_s3_bluetopo_files(victim):
    """Verify S3 file discovery excludes unneeded derivative rasters like _unc and _slope."""
    mock_s3 = MagicMock()
    mock_s3.glob.return_value = [
        "bucket/ER_3/sub/BlueTopo/tile1/tile1.tiff",
        "bucket/ER_3/sub/BlueTopo/tile1/tile1_unc.tiff",
        "bucket/ER_3/sub/BlueTopo/tile1/tile1_slope.tiff",
        "bucket/ER_3/sub/BlueTopo/tile1/tile1_iss.tiff"
    ]

    with patch(f'{ENGINE_PATH}._set_gdal_s3_options'), \
         patch(f'{ENGINE_PATH}.s3fs.S3FileSystem', return_value=mock_s3), \
         patch(f'{ENGINE_PATH}.get_config_item', return_value="bucket"):

        paths = victim.get_s3_bluetopo_files("ER_3")
        assert len(paths) == 1
        assert paths[0] == "/vsis3/bucket/ER_3/sub/BlueTopo/tile1/tile1.tiff"


def test_upload_file_to_s3(victim, tmp_path):
    """Test uploading output distance raster to target S3 path."""
    raster_file = tmp_path / "distance.tif"
    raster_file.write_text("dummy raster content")

    mock_s3 = MagicMock()
    with patch(f'{ENGINE_PATH}.s3fs.S3FileSystem', return_value=mock_s3), \
         patch(f'{ENGINE_PATH}.get_config_item', side_effect=["bucket", "sub"]):

        victim.upload_file_to_s3(raster_file, "ER_3", "tile_123", output_prefix="test_prefix")
        mock_s3.put.assert_called_once_with(
            str(raster_file),
            "bucket/test_prefix/ER_3/sub/BlueTopo/tile_123",
            recursive=True
        )


def test_process_tile_success(tmp_path):
    """Test successful workflow sequence in static _process_tile worker entrypoint."""
    param_lookup = {'env': 'local'}
    bluetopo_path = tmp_path / "tile_123" / "BlueTopo_123.tiff"
    bluetopo_path.parent.mkdir(parents=True, exist_ok=True)
    bluetopo_path.write_text("data")

    ecoregion_gdf = MagicMock()
    param_inputs = [str(bluetopo_path), "ER_3", ecoregion_gdf, str(tmp_path), param_lookup, ""]

    with patch(f'{ENGINE_PATH}.DistanceToShoreEngine') as MockEngine, \
         patch(f'{ENGINE_PATH}._set_gdal_s3_options'), \
         patch(f'{ENGINE_PATH}._trim_memory'):

        instance = MockEngine.return_value
        instance.build_ocean_mask.return_value = ("/tmp/ocean.tif", 100)
        instance.get_shoreline_vertices.return_value = np.array([[0, 0]], dtype=np.float32)
        instance.build_distance_from_shoreline.return_value = (
            pathlib.Path("/tmp/dist.tif"), {"min": 0.0, "mean": 10.0, "max": 20.0}
        )

        status = _process_tile(param_inputs)

        assert status["status"] == "success"
        assert status["stats"]["min"] == 0.0
        instance.build_ocean_mask.assert_called_once()
        instance.get_shoreline_vertices.assert_called_once()
        instance.build_distance_from_shoreline.assert_called_once()


def test_process_tile_skipped(tmp_path):
    """Test skipping processing when build_ocean_mask returns 0 valid water cells."""
    param_lookup = {'env': 'local'}
    bluetopo_path = tmp_path / "tile_456" / "BlueTopo_456.tiff"
    bluetopo_path.parent.mkdir(parents=True, exist_ok=True)
    bluetopo_path.write_text("data")

    ecoregion_gdf = MagicMock()
    param_inputs = [str(bluetopo_path), "ER_3", ecoregion_gdf, str(tmp_path), param_lookup, ""]

    with patch(f'{ENGINE_PATH}.DistanceToShoreEngine') as MockEngine, \
         patch(f'{ENGINE_PATH}._set_gdal_s3_options'), \
         patch(f'{ENGINE_PATH}._trim_memory'):

        instance = MockEngine.return_value
        instance.build_ocean_mask.return_value = ("/tmp/ocean.tif", 0)

        status = _process_tile(param_inputs)

        assert status["status"] == "skipped"
        assert status["stats"] is None
        instance.get_shoreline_vertices.assert_not_called()