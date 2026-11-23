import json
import pathlib
import sys
from unittest.mock import MagicMock, mock_open, patch

import geopandas as gpd
import pytest
import requests
from shapely.geometry import Polygon

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[3] / 'src'
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.engines.MetadataEngine import MetadataEngine
from hydro_health.helpers.tools import Param

METADATA_PATH = 'hydro_health.engines.MetadataEngine'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


@pytest.fixture
def victim():
    """Fixture returning a local MetadataEngine instance with mocked side effects."""
    engine = MetadataEngine({'env': 'local', 'output_directory': Param(OUTPUTS)})
    engine.write_message = MagicMock()
    engine.write_run_manifest = MagicMock()
    return engine


@pytest.fixture
def s3_victim():
    """Fixture returning an AWS/S3 MetadataEngine instance with mocked S3FileSystem."""
    with patch(f'{METADATA_PATH}.s3fs.S3FileSystem') as mock_s3_cls:
        engine = MetadataEngine({'env': 'aws', 'output_directory': Param(OUTPUTS)})
        engine.write_message = MagicMock()
        engine.write_run_manifest = MagicMock()
        engine.s3_files = mock_s3_cls.return_value
        yield engine


def test_download_metadata_success(victim, tmp_path):
    """Tests happy path where XML is fetched, parsed, and written to a local metadata.txt file."""
    provider_folder = tmp_path / "provider_dir"
    provider_folder.mkdir()
    param_inputs = ['ISO metadata', 'http://example.com', provider_folder, str(tmp_path / "out")]

    xml_payload = (
        b"<root><time-frame>"
        b"<description>Dataset A</description>"
        b"<start-date-time>2020-01-01</start-date-time>"
        b"<end-date-time>2021-01-01</end-date-time>"
        b"</time-frame></root>"
    )

    mock_response = MagicMock()
    mock_response.content = xml_payload

    with patch(f'{METADATA_PATH}.requests.get', return_value=mock_response) as mock_get:
        victim.download_metadata(param_inputs)

        mock_get.assert_called_once_with('http://example.com/inport-xml', timeout=10)

        out_file = provider_folder / 'metadata.txt'
        assert out_file.exists()
        written_content = out_file.read_text(encoding='utf-8')
        assert "Description: Dataset A" in written_content
        assert "2020-01-01, 2021-01-01" in written_content
        assert victim.write_message.call_count == 2


def test_download_metadata_s3_fallback_description(s3_victim):
    """Tests S3 write and fallback to provider folder name when <description> is absent."""
    provider_folder = pathlib.Path('my-bucket/ER1/sub/DigitalCoast/provider_dir')
    param_inputs = ['Metadata', 'http://example.com', provider_folder, 's3://my-bucket/out']

    xml_payload = (
        b'<root><time-frame>'
        b'<start-date-time>2022-01-01</start-date-time>'
        b'<end-date-time>2022-12-31</end-date-time>'
        b'</time-frame></root>'
    )
    mock_response = MagicMock(content=xml_payload)
    mock_s3_stream = mock_open()
    s3_victim.s3_files.open = mock_s3_stream

    with patch(f'{METADATA_PATH}.requests.get', return_value=mock_response):
        s3_victim.download_metadata(param_inputs)

        mock_s3_stream.assert_called_once_with(
            'my-bucket\\ER1\\sub\\DigitalCoast\\provider_dir/metadata.txt', 'w'
        )
        handle = mock_s3_stream()
        handle.write.assert_any_call('Description: provider_dir\n')
        handle.write.assert_any_call('2022-01-01, 2022-12-31\n')


def test_download_metadata_connection_error(victim, tmp_path):
    """Tests that requests.exceptions.RequestException is caught gracefully."""
    provider_folder = tmp_path / "provider_dir"
    provider_folder.mkdir()
    param_inputs = ['Metadata', 'http://invalid-url.com', provider_folder, str(tmp_path / "out")]

    with patch(f'{METADATA_PATH}.requests.get', side_effect=requests.exceptions.ConnectionError("boom")):
        victim.download_metadata(param_inputs)

        assert victim.write_message.call_count == 2
        error_msg = victim.write_message.call_args_list[1][0][0]
        assert "Metadata error [boom]: http://invalid-url.com/inport-xml" in error_msg


def test_download_metadata_skipped_label(victim, tmp_path):
    """Tests skipping network call when label is not in {'Metadata', 'ISO metadata'}."""
    provider_folder = tmp_path / "provider_dir"
    param_inputs = ['Other', 'http://example.com', provider_folder, str(tmp_path / "out")]

    with patch(f'{METADATA_PATH}.requests.get') as mock_get:
        victim.download_metadata(param_inputs)

        mock_get.assert_not_called()
        assert victim.write_message.call_count == 2
        skip_msg = victim.write_message.call_args_list[1][0][0]
        assert "skipping invalid metadata: Other" in skip_msg


def test_extract_metadata_link_schemas():
    """Tests extracting metadata links from both ExternalProviderLink and links schemas."""
    external_iso = {
        "Metalink": "http://example.com/iso_meta",
        "ExternalProviderLink": [
            {"label": "Metadata", "link": "http://example.com/iso/details"}
        ],
    }
    assert MetadataEngine._extract_metadata_link(external_iso) == (
        "ISO metadata",
        "http://example.com/iso_meta",
    )

    links_schema = {
        "links": [{"linkTypeName": "Metadata", "uri": "http://example.com/standard_meta"}]
    }
    assert MetadataEngine._extract_metadata_link(links_schema) == (
        "Metadata",
        "http://example.com/standard_meta",
    )


def test_read_json_files(victim, tmp_path):
    """Tests discovering JSON files locally and delegating to ThreadPoolExecutor."""
    digital_coast = tmp_path / "DigitalCoast"
    provider_dir = digital_coast / "provider_a"
    provider_dir.mkdir(parents=True)

    (provider_dir / "feature.json").write_text(json.dumps({
        "links": [{"linkTypeName": "Metadata", "uri": "http://example.com/meta_link"}]
    }))

    unused_dir = digital_coast / "unused_providers" / "provider_b"
    unused_dir.mkdir(parents=True)
    (unused_dir / "feature.json").write_text(json.dumps({
        "links": [{"linkTypeName": "Metadata", "uri": "http://example.com/ignore"}]
    }))

    with patch(f'{METADATA_PATH}.ThreadPoolExecutor') as mock_executor_cls:
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        victim.read_json_files(digital_coast, str(tmp_path / "out"))

        mock_executor.map.assert_called_once()
        func, params = mock_executor.map.call_args[0]
        assert func == victim.download_metadata
        assert params == [['Metadata', 'http://example.com/meta_link', provider_dir, str(tmp_path / "out")]]


def test_read_json_files_s3(s3_victim):
    """Tests discovering JSON files via s3fs.glob and filtering unused_providers."""
    s3_victim.s3_files.glob.return_value = [
        "my-bucket/ER1/sub/DigitalCoast/provider_a/feature.json",
        "my-bucket/ER1/sub/DigitalCoast/unused_providers/provider_b/feature.json",
    ]
    json_content = json.dumps({
        "Metalink": "http://example.com/s3_meta",
        "ExternalProviderLink": [{"altlabel": "Metadata", "link": "http://example.com/regular"}],
    })
    s3_victim.s3_files.open = mock_open(read_data=json_content)

    with patch(f'{METADATA_PATH}.ThreadPoolExecutor') as mock_executor_cls:
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        s3_victim.read_json_files("s3://my-bucket/ER1/sub/DigitalCoast", "s3://my-bucket/out")

        func, params = mock_executor.map.call_args[0]
        assert func == s3_victim.download_metadata
        assert params == [[
            'Metadata',
            'http://example.com/s3_meta',
            pathlib.Path("my-bucket/ER1/sub/DigitalCoast/provider_a"),
            "s3://my-bucket/out",
        ]]


def test_get_ecoregion_geometry_strings(victim):
    """Tests spatial reprojection and geometry extraction for a specific EcoRegion."""
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1', 'ER2'],
        'geometry': [
            Polygon([(0, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 11), (10, 11)]),
        ],
    }, crs="EPSG:4326")

    results = victim.get_ecoregion_geometry_strings(mock_gdf, 'ER1')

    assert len(results) == 1
    assert results[0].geom_type == 'Polygon'


def test_run_without_prefix(victim, tmp_path):
    """Tests main run logic when output_prefix is False/None."""
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1'],
        'geometry': [Polygon([(0, 0), (1, 1), (0, 1)])],
    }, crs="EPSG:4326")

    with patch.object(victim, 'read_json_files') as mock_read_json, \
         patch.object(victim, '_find_files', return_value=['file1.txt']), \
         patch(f'{METADATA_PATH}.get_config_item', return_value="subfolder"):

        victim.run(mock_gdf, output_prefix=False)

        expected_path = OUTPUTS / "ER1/subfolder/DigitalCoast"
        mock_read_json.assert_called_once_with(expected_path, OUTPUTS)
        victim.write_run_manifest.assert_called_once_with(
            pathlib.Path("ER1/subfolder/DigitalCoast"), {'metadata': 1}
        )


def test_run_s3_with_prefix(s3_victim):
    """Tests main run logic in S3 mode with output_prefix provided."""
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1'],
        'geometry': [Polygon([(0, 0), (1, 1), (0, 1)])],
    }, crs="EPSG:4326")

    def fake_config(section, key):
        return {'SUBFOLDER': 'subfolder', 'OUTPUT_BUCKET': 'my-bucket'}[key]

    with patch.object(s3_victim, 'read_json_files') as mock_read_json, \
         patch.object(s3_victim, '_find_files', return_value=['a', 'b']), \
         patch(f'{METADATA_PATH}.get_config_item', side_effect=fake_config):

        s3_victim.run(mock_gdf, output_prefix="prefix_dir")

        expected_s3_path = "s3://my-bucket/prefix_dir/ER1/subfolder/DigitalCoast"
        mock_read_json.assert_called_once_with(expected_s3_path, OUTPUTS)
        s3_victim.write_run_manifest.assert_called_once_with(
            pathlib.Path("prefix_dir/ER1/subfolder/DigitalCoast"), {'metadata': 2}
        )


def test_write_message(tmp_path):
    """Tests writing logs to a local file path using builtins."""
    engine = MetadataEngine({'env': 'local'})

    with patch('builtins.open', mock_open()) as mock_file:
        engine.write_message("Test status message", str(tmp_path))

        mock_file.assert_called_once_with(tmp_path / 'log_prints.txt', 'a')
        mock_file.return_value.write.assert_called_once_with("Test status message\n")