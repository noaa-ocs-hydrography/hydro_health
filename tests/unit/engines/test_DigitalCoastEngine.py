import pathlib
import sys
from unittest.mock import MagicMock, call, mock_open, patch

import geopandas as gpd
import pandas as pd
import pytest
from botocore import UNSIGNED
from shapely.geometry import Polygon

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / "src"
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.helpers.tools import Param
from hydro_health.engines.DigitalCoastEngine import (
    DigitalCoastEngine,
    _download_intersected_datasets,
    _download_intersected_datasets_local,
    _download_intersected_datasets_s3,
    _download_tile_index,
    _reproject_raster_gdal,
    _trim_memory,
)

TILING_PATH = "hydro_health.engines.DigitalCoastEngine"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def victim(tmp_path):
    """Returns an AWS-configured instance of DigitalCoastEngine with mocked Dask."""
    param_lookup = {
        "output_directory": Param(str(tmp_path)),
        "env": "aws",
    }
    engine = DigitalCoastEngine(param_lookup)
    engine.write_message = MagicMock()
    engine.client = MagicMock()
    engine.approved_size = 5000.0

    # Mock Dask / base class Engine methods
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.write_run_manifest = MagicMock()

    return engine


@pytest.fixture
def victim_local(tmp_path):
    """Returns a local-configured instance of DigitalCoastEngine with mocked Dask."""
    param_lookup = {
        "output_directory": Param(str(tmp_path)),
        "env": "local",
    }
    engine = DigitalCoastEngine(param_lookup)
    engine.write_message = MagicMock()
    engine.client = MagicMock()
    engine.approved_size = 5000.0

    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.write_run_manifest = MagicMock()

    return engine


@pytest.fixture
def sample_gdf():
    """Provides a minimal GeoDataFrame for pipeline tests."""
    return gpd.GeoDataFrame(
        {
            "EcoRegion": ["ER1"],
            "geometry": [Polygon([(0, 0), (1, 1), (0, 1)])],
        },
        crs="EPSG:4326",
    )


# ---------------------------------------------------------------------------
# Tests for Module-Level Helper Functions
# ---------------------------------------------------------------------------

def test_trim_memory_invokes_glibc_malloc_trim():
    """Tests _trim_memory triggers gc and calls glibc malloc_trim."""
    with patch(f"{TILING_PATH}.gc.collect") as mock_gc, \
         patch(f"{TILING_PATH}.ctypes.CDLL") as mock_cdll:
        mock_libc = MagicMock()
        mock_cdll.return_value = mock_libc

        _trim_memory()

        mock_gc.assert_called_once()
        mock_cdll.assert_called_once_with("libc.so.6")
        mock_libc.malloc_trim.assert_called_once_with(0)


def test_trim_memory_catches_exceptions_silently():
    """Tests _trim_memory does not raise when libc is missing or fails (e.g., non-Linux)."""
    with patch(f"{TILING_PATH}.gc.collect"), \
         patch(f"{TILING_PATH}.ctypes.CDLL", side_effect=OSError("libc not found")):
        try:
            _trim_memory()
        except Exception as e:
            pytest.fail(f"_trim_memory raised an unexpected exception: {e}")


@pytest.mark.parametrize(
    "datatype_name, expected_predictor",
    [
        ("Float32", "3"),
        ("Float64", "3"),
        ("Int16", "2"),
        ("Byte", "2"),
    ],
)
def test_reproject_raster_gdal_predictor_selection(datatype_name, expected_predictor):
    """Tests predictor selection (3 for Floats, 2 for Ints) and overview building."""
    mock_src_ds = MagicMock()
    mock_band = MagicMock()
    mock_band.DataType = 1
    mock_src_ds.GetRasterBand.return_value = mock_band

    mock_dst_ds = MagicMock()
    mock_dst_ds.RasterXSize = 1024
    mock_dst_ds.RasterYSize = 1024

    with patch(f"{TILING_PATH}.gdal.Open", return_value=mock_src_ds), \
         patch(f"{TILING_PATH}.gdal.GetDataTypeName", return_value=datatype_name), \
         patch(f"{TILING_PATH}.gdal.WarpOptions") as mock_warp_options, \
         patch(f"{TILING_PATH}.gdal.Warp", return_value=mock_dst_ds):

        _reproject_raster_gdal("source.tif", "destination.tif")

        _, kwargs = mock_warp_options.call_args
        assert f"PREDICTOR={expected_predictor}" in kwargs["creationOptions"]
        mock_dst_ds.BuildOverviews.assert_called_once_with("AVERAGE", [2, 4, 8, 16])


def test_reproject_raster_gdal_source_open_failure():
    """Tests RuntimeError is raised when source raster cannot be opened."""
    with patch(f"{TILING_PATH}.gdal.Open", return_value=None):
        with pytest.raises(RuntimeError, match="Could not open source raster"):
            _reproject_raster_gdal("missing.tif", "dst.tif")


def test_reproject_raster_gdal_warp_failure():
    """Tests RuntimeError is raised when gdal.Warp returns None."""
    mock_src_ds = MagicMock()
    with patch(f"{TILING_PATH}.gdal.Open", return_value=mock_src_ds), \
         patch(f"{TILING_PATH}.gdal.GetDataTypeName", return_value="Float32"), \
         patch(f"{TILING_PATH}.gdal.WarpOptions"), \
         patch(f"{TILING_PATH}.gdal.Warp", return_value=None):

        with pytest.raises(RuntimeError, match="gdal.Warp failed"):
            _reproject_raster_gdal("src.tif", "dst.tif")


def test_download_tile_index_skips_existing(tmp_path):
    """Tests _download_tile_index skips download if the .shp file already exists."""
    mock_engine = MagicMock()
    mock_bucket = MagicMock()
    mock_engine.get_bucket.return_value = mock_bucket

    mock_obj = MagicMock()
    mock_obj.key = "dem/survey1/tileindex.zip"
    mock_bucket.objects.filter.return_value = [mock_obj]

    param_inputs = ["https://digitalcoast.com/dem/index.html", tmp_path, {}, "outputs"]

    with patch(f"{TILING_PATH}.DigitalCoastEngine", return_value=mock_engine), \
         patch(f"{TILING_PATH}.get_config_item", return_value="digitalcoast"), \
         patch(f"{TILING_PATH}.os.path.exists", return_value=True):

        _download_tile_index(param_inputs)

        mock_bucket.download_fileobj.assert_not_called()
        assert "Skipping index" in mock_engine.write_message.call_args[0][0]


def test_download_tile_index_downloads_when_missing(tmp_path):
    """Tests _download_tile_index downloads file when target .shp is not found."""
    mock_engine = MagicMock()
    mock_bucket = MagicMock()
    mock_engine.get_bucket.return_value = mock_bucket

    mock_obj = MagicMock()
    mock_obj.key = "dem/survey1/tileindex.zip"
    mock_bucket.objects.filter.return_value = [mock_obj]

    param_inputs = ["https://digitalcoast.com/dem/index.html", tmp_path, {}, "outputs"]

    with patch(f"{TILING_PATH}.DigitalCoastEngine", return_value=mock_engine), \
         patch(f"{TILING_PATH}.get_config_item", return_value="digitalcoast"), \
         patch(f"{TILING_PATH}.os.path.exists", return_value=False), \
         patch("builtins.open", mock_open()):

        _download_tile_index(param_inputs)

        mock_bucket.download_fileobj.assert_called_once()
        assert "Downloading index" in mock_engine.write_message.call_args[0][0]


def test_download_intersected_datasets_router():
    """Tests dispatcher routes by argument list length."""
    with patch(f"{TILING_PATH}._download_intersected_datasets_s3", return_value="s3") as mock_s3, \
         patch(f"{TILING_PATH}._download_intersected_datasets_local", return_value="local") as mock_local:

        assert _download_intersected_datasets([1, 2, 3, 4, 5, 6]) == "s3"
        mock_s3.assert_called_once()

        assert _download_intersected_datasets([1, 2, 3, 4]) == "local"
        mock_local.assert_called_once()


def test_download_intersected_datasets_local_no_intersect(tmp_path):
    shp_file = tmp_path / "test.shp"
    shp_file.touch()

    mock_gdf = MagicMock()
    # Ensure method chaining returns mock_gdf
    mock_gdf.to_crs.return_value = mock_gdf
    mock_gdf.columns = pd.Index(["geometry"])

    # Mock joined output so df_joined['url'].any() evaluates to False
    mock_joined = MagicMock()
    mock_joined.__getitem__.return_value.any.return_value = False
    mock_gdf.sjoin.return_value = mock_joined

    with patch(f"{TILING_PATH}.DigitalCoastEngine"), \
         patch(f"{TILING_PATH}.gpd.read_file", return_value=mock_gdf):

        res = _download_intersected_datasets_local([mock_gdf, shp_file, {}, "outputs"])
        assert res == "- No intersect: test"


def test_download_intersected_datasets_local_download_and_reproject(tmp_path):
    """Tests local download flow: HTTP download, reprojection, and memory trimming."""
    from shapely.geometry import Point

    shp_file = tmp_path / "survey" / "index.shp"
    shp_file.parent.mkdir(parents=True)
    shp_file.touch()

    # Real minimal GeoDataFrames that geometrically intersect at (0, 0)
    shp_df = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]},
        crs="EPSG:4326"
    )
    tile_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)], "tile": [1], "url": ["http://data.com/dem.tif"]},
        crs="EPSG:4326"
    )

    mock_response = MagicMock(status_code=200, content=b"fake_tiff_data")
    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    mock_engine = MagicMock()
    mock_engine.cleansed_url.side_effect = lambda u: u

    with patch(f"{TILING_PATH}.DigitalCoastEngine", return_value=mock_engine), \
         patch(f"{TILING_PATH}.gpd.read_file", return_value=shp_df), \
         patch(f"{TILING_PATH}.gpd.GeoDataFrame.to_file"), \
         patch(f"{TILING_PATH}.requests.Session", return_value=mock_session), \
         patch(f"{TILING_PATH}._reproject_raster_gdal") as mock_reproject, \
         patch(f"{TILING_PATH}._trim_memory") as mock_trim, \
         patch(f"{TILING_PATH}.os.path.exists", return_value=False), \
         patch("builtins.open", mock_open()):

        res = _download_intersected_datasets_local([tile_gdf, shp_file, {}, "outputs"])

        assert res == "- index"
        mock_session.get.assert_called_once_with("http://data.com/dem.tif", timeout=10)
        mock_reproject.assert_called_once()
        mock_trim.assert_called_once()


def test_download_intersected_datasets_s3_pipeline(tmp_path):
    """Tests S3 flow: streaming download, raw upload, reprojection, and final upload."""
    from shapely.geometry import Point

    temp_root = tmp_path / "temp_root"
    shp_file = temp_root / "DigitalCoast" / "ProviderA" / "index.shp"
    shp_file.parent.mkdir(parents=True)
    shp_file.touch()

    shp_df = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]},
        crs="EPSG:4326"
    )
    tile_gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)], "tile": [101], "url": ["https://noaa.gov/elevation.tif"]},
        crs="EPSG:4326"
    )

    mock_engine = MagicMock()
    mock_engine.cleansed_url.side_effect = lambda u: u

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_session = MagicMock()
    mock_session.get.return_value.__enter__.return_value = mock_response

    with patch(f"{TILING_PATH}.DigitalCoastEngine", return_value=mock_engine), \
         patch(f"{TILING_PATH}.gpd.read_file", return_value=shp_df), \
         patch(f"{TILING_PATH}.gpd.GeoDataFrame.to_file"), \
         patch(f"{TILING_PATH}.requests.Session", return_value=mock_session), \
         patch(f"{TILING_PATH}._reproject_raster_gdal") as mock_reproject, \
         patch(f"{TILING_PATH}._trim_memory") as mock_trim, \
         patch("builtins.open", mock_open()):

        params = ["ER1", tile_gdf, shp_file, "outputs", {}, temp_root]
        result = _download_intersected_datasets_s3(params)

        assert result == "- index"
        assert mock_engine.upload_file_to_s3.call_count == 2
        upload_calls = mock_engine.upload_file_to_s3.call_args_list
        assert upload_calls[0][0][1] == "DigitalCoast/ProviderA/original/elevation.tif"
        assert upload_calls[1][0][1] == "DigitalCoast/ProviderA/elevation.tif"
        mock_reproject.assert_called_once()
        assert mock_trim.call_count >= 1


# ---------------------------------------------------------------------------
# Tests for DigitalCoastEngine Class Methods
# ---------------------------------------------------------------------------

def test_get_bucket(victim):
    """Tests anonymous S3 connection configuration."""
    with patch(f"{TILING_PATH}.boto3.resource") as mock_boto_res, \
         patch(f"{TILING_PATH}.get_config_item", return_value="target-bucket"):

        victim.get_bucket()

        _, kwargs = mock_boto_res.call_args
        assert kwargs["config"].signature_version is UNSIGNED
        mock_boto_res.return_value.Bucket.assert_called_with("target-bucket")


def test_breakup_cudem(victim, tmp_path):
    """Tests organizing NOAA CUDEM datasets into distinct year directories."""
    cudem_folder = tmp_path / "NOAA_NCEI_0001"
    sub_dem = cudem_folder / "dem" / "proj_dem"
    sub_dem.mkdir(parents=True)

    # 1999 DEM file
    tif_file = sub_dem / "sample_1999.tif"
    tif_file.touch()

    # Feature JSON at parents[1]
    feature_json = cudem_folder / "feature.json"
    feature_json.touch()

    with patch(f"{TILING_PATH}.gpd.read_file"), \
         patch(f"{TILING_PATH}.shutil.rmtree") as mock_rmtree:

        victim.breakup_cudem(tmp_path)

        mock_rmtree.assert_called_once_with(cudem_folder)


def test_check_tile_index_areas_folder_mode(victim, tmp_path):
    """Tests folder mode: removes folders with survey area under threshold."""
    shp_path = tmp_path / "prov" / "dem" / "sub" / "index.shp"
    shp_path.parent.mkdir(parents=True)
    shp_path.touch()

    mock_gdf = MagicMock()
    # Ensure .to_crs(9822) returns the configured mock
    mock_gdf.to_crs.return_value = mock_gdf
    # Total area 100 < approved_size (5000.0)
    mock_gdf.__getitem__.return_value.sum.return_value = 100.0

    with patch(f"{TILING_PATH}.gpd.read_file", return_value=mock_gdf), \
         patch(f"{TILING_PATH}.shutil.rmtree") as mock_rmtree:

        result = victim.check_tile_index_areas(tmp_path, "outputs")

        assert result is None
        mock_rmtree.assert_called_once_with(shp_path.parents[2])


def test_check_tile_index_areas_list_mode(victim):
    """Tests list mode: filters and returns approved shapefile paths without rmtree."""
    shp_path_small = pathlib.Path("/tmp/prov1/index.shp")
    shp_path_large = pathlib.Path("/tmp/prov2/index.shp")

    mock_gdf_small = MagicMock()
    mock_gdf_small.to_crs.return_value = mock_gdf_small
    mock_gdf_small.__getitem__.return_value.sum.return_value = 100.0

    mock_gdf_large = MagicMock()
    mock_gdf_large.to_crs.return_value = mock_gdf_large
    mock_gdf_large.__getitem__.return_value.sum.return_value = 10000.0

    with patch(f"{TILING_PATH}.gpd.read_file", side_effect=[mock_gdf_small, mock_gdf_large]), \
         patch(f"{TILING_PATH}.shutil.rmtree") as mock_rmtree:

        approved = victim.check_tile_index_areas([shp_path_small, shp_path_large], "outputs")

        assert approved == [shp_path_large]
        mock_rmtree.assert_not_called()


def test_delete_unused_folder(victim, tmp_path):
    """Tests purging empty provider folders not containing dem or laz."""
    empty_prov = tmp_path / "empty_provider"
    empty_prov.mkdir()

    valid_prov = tmp_path / "valid_provider"
    valid_prov.mkdir()
    (valid_prov / "dem").mkdir()

    victim.delete_unused_folder(tmp_path, "outputs")

    assert not empty_prov.exists()
    assert valid_prov.exists()


def test_download_support_files_argument_swapping(victim):
    """Tests argument permutation swap when tile_gdf is passed as the first parameter."""
    mock_gdf = gpd.GeoDataFrame()
    folder_path = pathlib.Path("/tmp/coastal")

    # Return a 1-item list so the loop executes once
    victim.get_ecoregion_geometry_strings = MagicMock(return_value=["POLYGON(...)"])
    victim.get_available_datasets = MagicMock(return_value=[])
    victim.unzip_all_files = MagicMock()

    # Pass in the alternate signature: (tile_gdf, ecoregion, digital_coast_folder, outputs)
    victim.download_support_files(mock_gdf, "ER1", folder_path, "/tmp/out")

    victim.get_ecoregion_geometry_strings.assert_called_once_with(mock_gdf, "ER1")
    victim.unzip_all_files.assert_called_once_with(folder_path)


def test_download_support_files_execution(victim, tmp_path):
    """Tests mapping of bulk download links through Dask."""
    victim.get_ecoregion_geometry_strings = MagicMock(return_value=["POLYGON(...)"])
    victim.get_available_datasets = MagicMock(return_value=[
        {"label": "Bulk Download", "link": "http://link.com", "provider_path": "/tmp"}
    ])
    victim.unzip_all_files = MagicMock()

    victim.download_support_files(tmp_path, gpd.GeoDataFrame(), "ER1", str(tmp_path))

    victim.client.map.assert_called_once()
    victim.client.gather.assert_called_once()
    victim.unzip_all_files.assert_called_once_with(tmp_path)


def test_unzip_all_files(victim, tmp_path):
    """Tests unzipping and removing .zip archives."""
    zip_path = tmp_path / "archive.zip"
    zip_path.touch()

    with patch(f"{TILING_PATH}.zipfile.ZipFile") as mock_zip:
        victim.unzip_all_files(tmp_path)

        mock_zip.assert_called_once_with(zip_path, "r")
        mock_zip.return_value.__enter__.return_value.extractall.assert_called_once()
        assert not zip_path.exists()


def test_upload_file_to_s3(victim):
    """Tests single S3 file upload call."""
    with patch(f"{TILING_PATH}.boto3.client") as mock_boto_client, \
         patch(f"{TILING_PATH}.get_config_item", return_value="target-bucket"):

        victim.upload_file_to_s3("/tmp/test.tif", "/raw/test.tif")

        mock_boto_client.return_value.upload_file.assert_called_once_with(
            "/tmp/test.tif", "target-bucket", "raw/test.tif"
        )


def test_upload_files_to_s3(victim, tmp_path):
    """Tests recursive directory traversal and upload preserving relative paths."""
    temp_root = tmp_path
    provider_dir = temp_root / "DigitalCoast" / "ProviderA"
    provider_dir.mkdir(parents=True)
    file1 = provider_dir / "index.shp"
    file1.touch()

    with patch(f"{TILING_PATH}.boto3.client") as mock_boto_client, \
         patch(f"{TILING_PATH}.get_config_item", return_value="out-bucket"):

        victim.upload_files_to_s3(provider_dir, temp_root)

        mock_boto_client.return_value.upload_file.assert_called_once_with(
            str(file1), "out-bucket", "DigitalCoast/ProviderA/index.shp"
        )


def test_process_intersected_datasets(victim, tmp_path):
    """Tests scanning index shapefiles and dispatching Dask tasks."""
    shp_file = tmp_path / "index.shp"
    shp_file.touch()

    victim.print_async_results = MagicMock()
    victim.client.gather.return_value = ["- index"]

    victim.process_intersected_datasets(tmp_path, gpd.GeoDataFrame(), "outputs")

    victim.client.map.assert_called_once()
    victim.client.gather.assert_called_once()
    victim.print_async_results.assert_called_once_with(["- index"], "outputs")


# ---------------------------------------------------------------------------
# Tests for Orchestration Pipelines
# ---------------------------------------------------------------------------

def test_run_orchestration_aws(victim, sample_gdf, tmp_path):
    """Tests the AWS orchestration pipeline (single-thread dask worker, chunking, logging)."""
    with patch.object(victim, "download_support_files"), \
         patch.object(victim, "check_tile_index_areas", return_value=[]), \
         patch.object(victim, "upload_files_to_s3"), \
         patch(f"{TILING_PATH}.get_config_item", return_value="subfolder"):

        victim.run(sample_gdf, output_prefix="custom_run")

        expected_log = tmp_path / "processed_providers.log"
        assert expected_log.exists()
        victim.setup_dask.assert_called_once_with("aws", threads_per_worker=1)
        victim.close_dask.assert_called_once()
        victim.write_run_manifest.assert_called()


def test_run_orchestration_aws_chunking_and_logging(victim, sample_gdf, tmp_path):
    """Tests AWS loop executes in chunks, writes to provider log, and uploads files."""
    dummy_shp = (
        tmp_path / "temp" / "ER1" / "sub" / "DigitalCoast" / "ProviderA" / "index.shp"
    )

    with patch.object(victim, "download_support_files"), \
         patch.object(victim, "check_tile_index_areas", return_value=[dummy_shp]), \
         patch.object(victim, "upload_files_to_s3") as mock_upload, \
         patch(f"{TILING_PATH}.get_config_item", return_value="subfolder"):

        victim.run(sample_gdf)

        provider_log = tmp_path / "processed_providers.log"
        assert provider_log.exists()
        assert "ProviderA" in provider_log.read_text()
        victim.client.map.assert_called_once()
        mock_upload.assert_called_once()


def test_run_orchestration_local(victim_local, sample_gdf, tmp_path):
    """Tests the local orchestration pipeline execution and lifecycle."""
    with patch.object(victim_local, "download_support_files"), \
         patch.object(victim_local, "check_tile_index_areas"), \
         patch.object(victim_local, "process_intersected_datasets"), \
         patch.object(victim_local, "delete_unused_folder"), \
         patch(f"{TILING_PATH}.get_config_item", return_value="subfolder"):

        victim_local.run(sample_gdf, output_prefix=False)

        assert not (tmp_path / "processed_providers.log").exists()
        victim_local.setup_dask.assert_called_once_with("local")
        victim_local.close_dask.assert_called_once()
        victim_local.download_support_files.assert_called_once()
        victim_local.check_tile_index_areas.assert_called_once()
        victim_local.process_intersected_datasets.assert_called_once()
        victim_local.write_run_manifest.assert_called()