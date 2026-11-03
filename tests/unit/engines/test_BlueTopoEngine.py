import os
import sys
import pathlib
import pytest
from datetime import datetime, date
from unittest.mock import MagicMock, patch
from botocore import UNSIGNED
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.BlueTopoEngine import BlueTopoEngine, _process_tile
from hydro_health.helpers.tools import Param

TILING_PATH = 'hydro_health.engines.BlueTopoEngine'


# =====================================================================
# Fixtures & Test Helpers
# =====================================================================

@pytest.fixture
def victim(tmp_path):
    """Fixture to provide an instance of BlueTopoEngine with mocked internals."""
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'aws',
        'eco_regions': Param(['ER1'])
    }
    with patch('pathlib.Path.mkdir'):
        engine = BlueTopoEngine(param_lookup)

    engine.write_message = MagicMock()
    engine.setup_dask = MagicMock()
    engine.close_dask = MagicMock()
    engine.client = MagicMock()
    return engine


def create_mock_raster_and_xml(folder: pathlib.Path, stem: str = "tile") -> pathlib.Path:
    """
    Creates a temporary 3-band GeoTIFF and companion PAM aux XML containing a Contributor RAT.
    Uses 64x64 dimensions so decimation factors [2, 4, 8, 16] remain > 1x1 for GDAL overview creation.
    """
    tiff_path = folder / f"{stem}.tiff"
    xml_path = folder / f"{stem}.tiff.aux.xml"

    # Band 1: Elevation, Band 2: Uncertainty, Band 3: Contributor ID
    # Initialize with -9999 (nodata)
    data = np.full((3, 64, 64), -9999.0, dtype=np.float32)
    
    # Populate top-left test pixels for assertions
    data[0, 0, 0] = -10.0
    data[0, 0, 1] = 5.0
    data[0, 1, 0] = 0.0
    data[0, 1, 1] = -5.0

    data[1, :, :] = 0.5

    data[2, 0, 0] = 1.0      # Survey 1
    data[2, 0, 1] = 2.0      # Survey 2
    data[2, 1, 0] = 1.0      # Survey 1
    data[2, 1, 1] = -9999.0  # Nodata

    transform = rasterio.transform.from_origin(0, 64, 1, 1)
    with rasterio.open(
        tiff_path,
        'w',
        driver='GTiff',
        height=64,
        width=64,
        count=3,
        dtype=np.float32,
        crs='EPSG:32619',
        transform=transform,
        nodata=-9999.0
    ) as dst:
        dst.write(data)

    xml_content = """<PAMDataset>
  <PAMRasterBand band="3">
    <Description>Contributor</Description>
    <GDALRasterAttributeTable>
      <FieldDefn index="0"><Name>value</Name></FieldDefn>
      <FieldDefn index="1"><Name>survey_date_start</Name></FieldDefn>
      <FieldDefn index="2"><Name>survey_date_end</Name></FieldDefn>
      <FieldDefn index="3"><Name>source_survey_id</Name></FieldDefn>
      <FieldDefn index="4"><Name>significant_features</Name></FieldDefn>
      <FieldDefn index="5"><Name>feature_least_depth</Name></FieldDefn>
      <FieldDefn index="6"><Name>bathy_coverage</Name></FieldDefn>
      <FieldDefn index="7"><Name>horizontal_uncert_fixed</Name></FieldDefn>
      <FieldDefn index="8"><Name>horizontal_uncert_var</Name></FieldDefn>
      <FieldDefn index="9"><Name>vertical_uncert_fixed</Name></FieldDefn>
      <FieldDefn index="10"><Name>vertical_uncert_var</Name></FieldDefn>
      <Row index="0">
        <F>1</F>
        <F>2018-05-01</F>
        <F>2018-10-01</F>
        <F>SURVEY_HISTORIC</F>
        <F>1</F>
        <F>1</F>
        <F>1</F>
        <F>0.5</F>
        <F>0.1</F>
        <F>0.2</F>
        <F>0.05</F>
      </Row>
      <Row index="1">
        <F>2</F>
        <F>2021-01-01</F>
        <F>2021-06-01</F>
        <F>SURVEY_RECENT</F>
        <F>1</F>
        <F>1</F>
        <F>1</F>
        <F>0.3</F>
        <F>0.05</F>
        <F>0.1</F>
        <F>0.02</F>
      </Row>
    </GDALRasterAttributeTable>
  </PAMRasterBand>
</PAMDataset>"""
    xml_path.write_text(xml_content.strip())
    return tiff_path


# =====================================================================
# Engine Initialization & S3 Configuration
# =====================================================================

def test_init_scratch_dir_aws(tmp_path):
    """Verify AWS environment initializes and provisions scratch directories."""
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'aws',
        'eco_regions': Param(['ER1'])
    }
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        engine = BlueTopoEngine(param_lookup)
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
        assert engine.scratch_dir.name == "BlueTopoEngine"


def test_init_scratch_dir_local(tmp_path):
    """Verify Local environment skips scratch directory provisioning."""
    param_lookup = {
        'output_directory': Param(str(tmp_path)),
        'env': 'local',
        'eco_regions': Param(['ER1'])
    }
    with patch('pathlib.Path.mkdir') as mock_mkdir:
        BlueTopoEngine(param_lookup)
        mock_mkdir.assert_not_called()


def test_get_bucket(victim):
    """Verify S3 connection uses UNSIGNED config for NOAA bathymetry bucket."""
    with patch('boto3.resource') as mock_boto_res:
        victim.get_bucket()
        _, kwargs = mock_boto_res.call_args
        assert kwargs['config'].signature_version is UNSIGNED
        mock_boto_res.return_value.Bucket.assert_called_with("noaa-ocs-nationalbathymetry-pds")


# =====================================================================
# S3 Downloads and Staging
# =====================================================================

def test_download_nbs_tile_standard(victim, tmp_path):
    """Test standard destination resolution and cleanup of existing partial files."""
    mock_bucket = MagicMock()
    mock_obj_tiff = MagicMock(key="BlueTopo/B1234567/tile.tiff")
    mock_obj_xml = MagicMock(key="BlueTopo/B1234567/tile.xml")
    mock_bucket.objects.filter.return_value = [mock_obj_tiff, mock_obj_xml]

    with patch.object(victim, 'get_bucket', return_value=mock_bucket), \
         patch(f'{TILING_PATH}.get_config_item', return_value="sub"), \
         patch('pathlib.Path.unlink') as mock_unlink, \
         patch('pathlib.Path.exists', return_value=True):

        result = victim.download_nbs_tile(
            tile_id="B1234567",
            ecoregion_id="ER1",
            output_prefix=False,
            target_res=8,
            temp_folder=tmp_path
        )

        assert mock_unlink.call_count == 2
        assert mock_bucket.download_file.call_count == 2
        assert result.suffix == '.tiff'
        assert "ER1" in str(result)


def test_download_nbs_tile_low_res(victim, tmp_path):
    """Test resolution subfolder mapping when output_prefix is 'low_res'."""
    mock_bucket = MagicMock()
    mock_obj_tiff = MagicMock(key="BlueTopo/B1234567/tile.tiff")
    mock_bucket.objects.filter.return_value = [mock_obj_tiff]

    with patch.object(victim, 'get_bucket', return_value=mock_bucket), \
         patch(f'{TILING_PATH}.get_config_item', return_value="sub"), \
         patch('pathlib.Path.exists', return_value=False):

        result = victim.download_nbs_tile(
            tile_id="B1234567",
            ecoregion_id="ER1",
            output_prefix='low_res',
            target_res=16,
            temp_folder=tmp_path
        )

        assert "low_res" in str(result)
        assert "16m" in str(result)


def test_download_nbs_tile_not_found(victim, tmp_path):
    """Verify download returns None and logs when S3 bucket prefix yields no objects."""
    mock_bucket = MagicMock()
    mock_bucket.objects.filter.return_value = []

    with patch.object(victim, 'get_bucket', return_value=mock_bucket):
        result = victim.download_nbs_tile(
            tile_id="B9999999",
            ecoregion_id="ER1",
            output_prefix=False,
            target_res=8,
            temp_folder=tmp_path
        )
        assert result is None
        victim.write_message.assert_called_once()
        assert "CRITICAL: No source files found" in victim.write_message.call_args[0][0]


# =====================================================================
# Resampling and Reprojection
# =====================================================================

def test_resample_and_reproject_skipped_matching_res(victim):
    """Verify resampling is bypassed when tile is already within 0.1m of target resolution."""
    fake_tiff = pathlib.Path("/tmp/tile.tiff")
    mock_ds = MagicMock()
    mock_ds.GetGeoTransform.return_value = (0, 8.05, 0, 0, 0, -8.05)

    with patch('osgeo.gdal.Open', return_value=mock_ds), \
         patch('osgeo.gdal.Warp') as mock_warp:

        victim.resample_and_reproject(fake_tiff, target_res=8)
        mock_warp.assert_not_called()


def test_resample_and_reproject_open_none(victim):
    """Verify resample_and_reproject gracefully returns when GDAL dataset is unreadable."""
    with patch('osgeo.gdal.Open', return_value=None), \
         patch('osgeo.gdal.Warp') as mock_warp:
        victim.resample_and_reproject(pathlib.Path("/tmp/invalid.tiff"), target_res=8)
        mock_warp.assert_not_called()


def test_resample_and_reproject_execution(victim, tmp_path):
    """Verify GDAL dual-algorithm warps (Bilinear for B1/2, Nearest for B3), VRT build, and XML retention."""
    fake_tiff = tmp_path / "tile.tiff"
    fake_tiff.write_text("orig")
    xml_file = tmp_path / "tile.tiff.aux.xml"
    xml_file.write_text("<xml/>")

    mock_ds = MagicMock()
    mock_ds.GetGeoTransform.return_value = (0, 4.0, 0, 0, 0, -4.0)

    def fake_translate(dest, src, **kwargs):
        pathlib.Path(dest).write_text("translated")

    def fake_warp(dest, src, **kwargs):
        pathlib.Path(dest).write_text("warped")

    with patch('osgeo.gdal.Open', return_value=mock_ds), \
         patch('osgeo.gdal.Translate', side_effect=fake_translate) as mock_translate, \
         patch('osgeo.gdal.Warp', side_effect=fake_warp) as mock_warp, \
         patch('osgeo.gdal.BuildVRT') as mock_vrt:

        victim.resample_and_reproject(fake_tiff, target_res=8)

        # 1. Translate B1/2 and B3, 2. Translate VRT to final GeoTIFF
        assert mock_translate.call_count == 3
        # 1. Warp B1/2 (Bilinear), 2. Warp B3 (Nearest Neighbor)
        assert mock_warp.call_count == 2
        mock_vrt.assert_called_once()
        assert xml_file.exists()


# =====================================================================
# CATZOC, ISS, and Survey End Date Derivatives
# =====================================================================

def test_create_catzoc_all(victim, tmp_path):
    """Verify reclassification of all contributor surveys and increased scale naming."""
    tiff_path = create_mock_raster_and_xml(tmp_path, stem="tile")
    victim.parse_survey_date = MagicMock(side_effect=lambda x: datetime.strptime(x, "%Y-%m-%d").date() if x else None)

    def mock_supersession(meta):
        return 75.0 if meta['value'] == 1.0 else 90.0

    with patch(f'{TILING_PATH}.supersession', side_effect=mock_supersession), \
         patch(f'{TILING_PATH}.catzoc', return_value=1):

        victim.create_catzoc_all(tiff_path, increased_scale=True)

        out_path = tmp_path / "tile_ISS_all_110.tiff"
        assert out_path.exists()

        with rasterio.open(out_path) as src:
            band = src.read(1)
            assert band[0, 0] == 75.0
            assert band[0, 1] == 90.0
            assert band[1, 1] == -9999.0


def test_create_catzoc_latest(victim, tmp_path):
    """Verify latest non-interpolated survey selection and ISS raster reclassification."""
    tiff_path = create_mock_raster_and_xml(tmp_path, stem="tile")
    victim.parse_survey_date = MagicMock(side_effect=lambda x: datetime.strptime(x, "%Y-%m-%d").date() if x else None)

    with patch(f'{TILING_PATH}.supersession', return_value=95.0) as mock_ss, \
         patch(f'{TILING_PATH}.catzoc', return_value=1):

        victim.create_catzoc_latest(tiff_path, increased_scale=False)

        out_path = tmp_path / "tile_ISS_latest.tiff"
        assert out_path.exists()

        with rasterio.open(out_path) as src:
            band = src.read(1)
            assert band[0, 0] == 95.0
            assert band[0, 1] == 95.0
            assert band[1, 1] == -9999.0

        called_meta = mock_ss.call_args[0][0]
        assert called_meta['end_date'] == date(2021, 6, 1)


def test_create_survey_end_date_tiff(victim, tmp_path):
    """Verify extraction of survey end years into single-band GeoTIFF."""
    tiff_path = create_mock_raster_and_xml(tmp_path, stem="tile")
    victim.parse_survey_date = MagicMock(side_effect=lambda x: datetime.strptime(x, "%Y-%m-%d").date() if x else None)

    victim.create_survey_end_date_tiff(tiff_path)

    out_path = tmp_path / "tile_survey_end_date.tiff"
    assert out_path.exists()

    with rasterio.open(out_path) as src:
        band = src.read(1)
        assert band[0, 0] == 2018
        assert band[0, 1] == 2021
        assert band[1, 1] == -9999


# =====================================================================
# Raster Processing & Single-Band Transformations
# =====================================================================

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

    victim.set_ground_to_nodata(test_path)

    with rasterio.open(test_path, 'r') as src:
        result = src.read(1)
        assert result[0, 1] == -9999
        assert result[1, 0] == -9999
        assert result[0, 0] == -10
        assert result[1, 1] == -5


def test_finalize_cog(victim, tmp_path):
    """Test Cloud-Optimized GeoTIFF layout creation and temporary file replacement."""
    source_tiff = tmp_path / "tile.tiff"
    source_tiff.write_text("original")

    def mock_translate_side_effect(dest, src, **kwargs):
        pathlib.Path(dest).write_text("cog_data")

    with patch('osgeo.gdal.Translate', side_effect=mock_translate_side_effect) as mock_translate:
        victim.finalize_cog(source_tiff)

        temp_cog = tmp_path / "temp_tile.tiff"
        mock_translate.assert_called_once_with(
            str(temp_cog),
            str(source_tiff),
            format="COG",
            creationOptions=[
                "COMPRESS=DEFLATE",
                "PREDICTOR=3",
                "BLOCKSIZE=512",
                "OVERVIEW_RESAMPLING=BILINEAR"
            ]
        )
        assert source_tiff.exists()
        assert source_tiff.read_text() == "cog_data"


def test_rename_multiband(victim, tmp_path):
    """Verify renaming original file to intermediate '_mb.tiff' pattern."""
    source_tiff = tmp_path / "tile.tiff"
    source_tiff.write_text("data")

    renamed_path = victim.rename_multiband(source_tiff)
    assert renamed_path.name == "tile_mb.tiff"
    assert renamed_path.exists()
    assert not source_tiff.exists()


def test_multiband_to_singleband(victim, tmp_path):
    """Verify splitting multiband into singleband 1 (elevation) and 2 (uncertainty)."""
    mb_path = tmp_path / "tile_mb.tiff"

    with patch('osgeo.gdal.Translate') as mock_translate:
        # Band 1 test (Elevation: replaces '_mb' with '')
        victim.multiband_to_singleband(mb_path, band=1)
        expected_b1 = tmp_path / "tile.tiff"
        mock_translate.assert_called_with(
            str(expected_b1),
            str(mb_path),
            bandList=[1],
            creationOptions=["COMPRESS=DEFLATE"]
        )

        # Band 2 test (Uncertainty: replaces '_mb' with '_unc')
        victim.multiband_to_singleband(mb_path, band=2)
        expected_b2 = tmp_path / "tile_unc.tiff"
        mock_translate.assert_called_with(
            str(expected_b2),
            str(mb_path),
            bandList=[2],
            creationOptions=["COMPRESS=DEFLATE"]
        )


# =====================================================================
# S3 Upload & Logging
# =====================================================================

def test_upload_current_tiles_to_s3(victim, tmp_path):
    """Verify local files in the tile folder are mapped to S3 keys relative to temp_folder."""
    tile_dir = tmp_path / "ER1"
    tile_dir.mkdir(parents=True, exist_ok=True)
    dummy_file = tile_dir / "test_tile.tiff"
    dummy_file.write_text("data")

    expected_s3_key = str(dummy_file.relative_to(tmp_path))

    with patch(f'{TILING_PATH}.get_config_item', return_value="ocs-dev-csdl-hydrohealth"), \
         patch('boto3.client') as mock_client:

        victim.upload_current_tiles_to_s3(tile_dir, tmp_path)

        mock_client.return_value.upload_file.assert_called_once_with(
            str(dummy_file),
            "ocs-dev-csdl-hydrohealth",
            expected_s3_key
        )


def test_print_async_results(victim):
    """Verify asynchronous results are logged to the output folder message stream."""
    victim.print_async_results(["[B1] Complete", None, "[B2] Complete"], "/tmp/out")
    assert victim.write_message.call_count == 2
    victim.write_message.assert_any_call("[B1] Complete", "/tmp/out")
    victim.write_message.assert_any_call("[B2] Complete", "/tmp/out")


# =====================================================================
# Parallel Worker Entrypoint (_process_tile)
# =====================================================================

def test_process_tile_aws_success():
    """Verify complete sequence inside _process_tile under AWS mode."""
    param_lookup = {
        'env': 'aws',
        'output_directory': Param('/tmp/out')
    }
    param_inputs = [param_lookup, 'B1234567', 'ER1', 'regional_model', 8]
    fake_tiff = pathlib.Path("/tmp/fake/tile.tiff")
    fake_mb_tiff = MagicMock(spec=pathlib.Path)
    fake_mb_tiff.exists.return_value = True

    with patch(f'{TILING_PATH}.BlueTopoEngine') as MockEngine, \
         patch('tempfile.TemporaryDirectory') as mock_temp:

        mock_temp.return_value.__enter__.return_value = "/tmp/fake"
        instance = MockEngine.return_value
        instance.scratch_dir = pathlib.Path("/tmp/scratch")
        instance.download_nbs_tile.return_value = fake_tiff
        instance.rename_multiband.return_value = fake_mb_tiff

        status = _process_tile(param_inputs)

        instance.download_nbs_tile.assert_called_once_with(
            'B1234567', 'ER1', 'regional_model', 8, temp_folder=pathlib.Path("/tmp/fake")
        )
        instance.resample_and_reproject.assert_called_once_with(fake_tiff, 8)
        instance.create_survey_end_date_tiff.assert_called_once_with(fake_tiff)
        instance.create_catzoc_all.assert_called_once_with(fake_tiff, increased_scale=True)
        instance.create_catzoc_latest.assert_called_once_with(fake_tiff, increased_scale=True)
        instance.create_slope.assert_called_once_with(fake_tiff)
        instance.rename_multiband.assert_called_once_with(fake_tiff)
        assert instance.multiband_to_singleband.call_count == 2
        fake_mb_tiff.unlink.assert_called_once()

        # set_ground_to_nodata is intentionally commented out in _process_tile
        instance.set_ground_to_nodata.assert_not_called()

        instance.finalize_cog.assert_called_once_with(fake_tiff)
        instance.upload_current_tiles_to_s3.assert_called_once()
        assert "successfully completed" in status


def test_process_tile_skipped():
    """Verify _process_tile short-circuits gracefully when source tile download fails."""
    param_lookup = {
        'env': 'aws',
        'output_directory': Param('/tmp/out')
    }
    param_inputs = [param_lookup, 'B1234567', 'ER1', 'regional_model', 8]

    with patch(f'{TILING_PATH}.BlueTopoEngine') as MockEngine, \
         patch('tempfile.TemporaryDirectory'):

        instance = MockEngine.return_value
        instance.scratch_dir = pathlib.Path("/tmp/scratch")
        instance.download_nbs_tile.return_value = None

        status = _process_tile(param_inputs)

        assert "Skipped (Source tile download or resolution failed)" in status
        instance.resample_and_reproject.assert_not_called()


def test_process_tile_local_env():
    """Verify _process_tile executes using nullcontext and skips S3 upload in local mode."""
    param_lookup = {
        'env': 'local',
        'output_directory': Param('/tmp/out')
    }
    param_inputs = [param_lookup, 'B1234567', 'ER1', 'regional_model', 8]
    fake_tiff = pathlib.Path("/tmp/fake/tile.tiff")
    fake_mb_tiff = MagicMock(spec=pathlib.Path)
    fake_mb_tiff.exists.return_value = False

    with patch(f'{TILING_PATH}.BlueTopoEngine') as MockEngine:
        instance = MockEngine.return_value
        instance.download_nbs_tile.return_value = fake_tiff
        instance.rename_multiband.return_value = fake_mb_tiff

        status = _process_tile(param_inputs)

        instance.upload_current_tiles_to_s3.assert_not_called()
        assert "successfully completed" in status


# =====================================================================
# Engine Orchestration (run)
# =====================================================================

def test_run(victim):
    """Verify main run method regex tile parsing, ecoregion filtering, Dask dispatch, and logging."""
    tile_gdf = gpd.GeoDataFrame({
        'Tile': ['BlueTopo_BC123456', 'BlueTopo_BC999999', 'INVALID_PATTERN'],
        'EcoRegion': ['ER1', 'ER2', 'ER1']
    })

    victim.write_run_manifest = MagicMock()
    victim.print_async_results = MagicMock()
    victim.client.map.return_value = ['future_obj']
    victim.client.gather.return_value = ['[BC123456] Processing successfully completed.']

    with patch(f'{TILING_PATH}.hibase_logging.send_record') as mock_logging, \
         patch(f'{TILING_PATH}.get_config_item', return_value="sub"), \
         patch('os.getlogin', return_value="test_user"):

        victim.run(tile_gdf, output_prefix="low_res", resolution=[8])

        # Verify Dask initialization for AWS
        victim.setup_dask.assert_called_once_with('aws', n_workers=1, threads_per_worker=1)
        victim.close_dask.assert_called_once()

        # Only BC123456 in ER1 satisfies both regex and ecoregion whitelist
        assert victim.client.map.call_count == 1
        mapped_inputs = victim.client.map.call_args[0][1]
        assert len(mapped_inputs) == 1
        assert mapped_inputs[0][1] == 'BC123456'

        victim.client.gather.assert_called_once_with(['future_obj'])
        victim.print_async_results.assert_called_once()
        victim.write_run_manifest.assert_called_once()
        mock_logging.assert_called_once()

        record = mock_logging.call_args[0][0]
        assert record['user'] == 'test_user'
        assert record['tiles_downloaded'] == 3