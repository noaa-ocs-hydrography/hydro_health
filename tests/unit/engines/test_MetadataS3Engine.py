import json
import io
import pathlib
import pytest
import requests
import geopandas as gpd
from unittest.mock import MagicMock, patch, mock_open
from shapely.geometry import Polygon

import sys
HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
print(HYDRO_HEALTH_MODULE)
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.MetadataS3Engine import MetadataS3Engine


METADATA_PATH = 'hydro_health.engines.MetadataS3Engine'


@pytest.fixture
def victim():
    """Fixture returning an instance of MetadataS3Engine with mocked side effects."""
    engine = MetadataS3Engine()
    engine.write_message = MagicMock()
    return engine


def test_upload_metadata_to_s3_success(victim):
    """Tests happy path where XML is fetched, parsed, and written back to S3."""
    param_inputs = ['Metadata', 'http://example.com', 's3://bucket/provider_dir', '/tmp/out']

    # Constructed using explicit tags:
    xml_str = (
        "<root>"
        "  <time-frame>"
        "    <description>Dataset A</description>"
        "    <start-date-time>2020-01-01</start-date-time>"
        "    <end-date-time>2021-01-01</end-date-time>"
        "  </time-frame>"
        "</root>"
    )
    xml_payload = xml_str.encode('utf-8')

    mock_response = MagicMock()
    mock_response.content = xml_payload

    mock_s3_file = MagicMock()

    with patch(f'{METADATA_PATH}.requests.get', return_value=mock_response) as mock_get, \
         patch(f'{METADATA_PATH}.s3fs.S3FileSystem') as mock_s3_cls:

        mock_s3_instance = mock_s3_cls.return_value
        mock_s3_instance.open.return_value.__enter__.return_value = mock_s3_file

        victim.upload_metadata_to_s3(param_inputs)

        mock_get.assert_called_once_with('http://example.com/inport-xml', timeout=10)
        mock_s3_instance.open.assert_called_once_with('s3://bucket/provider_dir/metadata.txt', 'w')

        # Check expected content written to the S3 file handle
        written_content = "".join([call.args[0] for call in mock_s3_file.write.call_args_list])
        assert "Description: Dataset A" in written_content
        assert "2020-01-01, 2021-01-01" in written_content
        victim.write_message.assert_called_with(' - stored metadata: s3://bucket/provider_dir', '/tmp/out')


def test_upload_metadata_to_s3_connection_error(victim):
    """Tests that ConnectionError is caught gracefully without throwing an exception."""

    param_inputs = ['Metadata', 'http://invalid-url.com', 's3://bucket/provider_dir', '/tmp/out']

    with patch(f'{METADATA_PATH}.requests.get', side_effect=requests.exceptions.ConnectionError), \
         patch(f'{METADATA_PATH}.s3fs.S3FileSystem'):

        victim.upload_metadata_to_s3(param_inputs)

        # Ensure error log message was recorded
        victim.write_message.assert_called_once()
        assert "Metadata error:" in victim.write_message.call_args[0][0]


def test_upload_metadata_to_s3_skipped_label(victim):
    """Tests skipping network call when label is not 'Metadata'."""

    param_inputs = ['ISO metadata', 'http://example.com', 's3://bucket/provider_dir', '/tmp/out']

    with patch(f'{METADATA_PATH}.requests.get') as mock_get, \
         patch(f'{METADATA_PATH}.s3fs.S3FileSystem'):

        victim.upload_metadata_to_s3(param_inputs)

        mock_get.assert_not_called()
        victim.write_message.assert_called_once_with(' - skipping invalid metdata: ISO metadata', '/tmp/out')


def test_read_json_files(victim):
    """Tests parsing S3 JSON files and delegating to ThreadPoolExecutor."""
    mock_json_content = json.dumps({
        "Metalink": "http://example.com/link",
        "ExternalProviderLink": [
            {"label": "Metadata", "link": "http://example.com/meta"}
        ]
    })

    with patch(f'{METADATA_PATH}.s3fs.S3FileSystem') as mock_s3_cls, \
         patch(f'{METADATA_PATH}.ThreadPoolExecutor') as mock_executor_cls:

        mock_s3 = mock_s3_cls.return_value
        mock_s3.glob.return_value = ["s3://bucket/folder/data.json"]
        
        # Use io.StringIO so json.load() can call .read() on the returned mock file
        mock_s3.open.return_value.__enter__.return_value = io.StringIO(mock_json_content)

        mock_executor = mock_executor_cls.return_value.__enter__.return_value

        # Pass as a forward-slash S3 URI string or call .as_posix()
        victim.read_json_files("s3://bucket/digital_coast", "/tmp/out")

        mock_s3.glob.assert_called_once_with("s3://bucket/digital_coast/**/*.json")
        
        # Verify map was called with expected metadata params
        mock_executor.map.assert_called_once()
        args = mock_executor.map.call_args[0]
        assert args[0] == victim.upload_metadata_to_s3
        assert args[1] == [['Metadata', 'http://example.com/link', 's3://bucket/folder', '/tmp/out']]


def test_get_ecoregion_geometry_strings(victim):
    """Tests spatial reprojection and geometry extraction for a specific EcoRegion."""

    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1', 'ER2'],
        'geometry': [
            Polygon([(0, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 11), (10, 11)])
        ]
    }, crs="EPSG:4326")

    results = victim.get_ecoregion_geometry_strings(mock_gdf, 'ER1')

    assert len(results) == 1
    assert results[0].geom_type == 'Polygon'


def test_run(victim):
    """Tests full execution orchestration for all EcoRegions in the GeoDataFrame."""

    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1', 'ER2'],
        'geometry': [
            Polygon([(0, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 11), (10, 11)])
        ]
    }, crs="EPSG:4326")

    with patch.object(victim, 'read_json_files') as mock_read_json, \
         patch(f'{METADATA_PATH}.get_config_item') as mock_cfg:

        # Config mock side effect for SHARED output bucket and DIGITALCOAST subfolder
        mock_cfg.side_effect = lambda domain, key: "my-bucket" if domain == "SHARED" else "subfolder"

        victim.run(mock_gdf, output_prefix="prefix_dir", outputs="/tmp/out")

        assert mock_read_json.call_count == 2
        mock_read_json.assert_any_call("s3://my-bucket/prefix_dir/ER1/subfolder/DigitalCoast", "/tmp/out")
        mock_read_json.assert_any_call("s3://my-bucket/prefix_dir/ER2/subfolder/DigitalCoast", "/tmp/out")


def test_write_message(tmp_path):
    """Tests writing logs locally using builtins/pathlib."""

    engine = MetadataS3Engine()

    with patch('builtins.open', mock_open()) as mock_file:
        engine.write_message("Test status", str(tmp_path))

        mock_file.assert_called_once_with(tmp_path / 'log_prints.txt', 'a')
        mock_file.return_value.write.assert_called_once_with("Test status\n")