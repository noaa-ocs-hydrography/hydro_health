import json
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

from hydro_health.engines.MetadataEngine import MetadataEngine

METADATA_PATH = 'hydro_health.engines.MetadataEngine'


@pytest.fixture
def victim():
    """Fixture returning an instance of local MetadataEngine with mocked side effects."""
    engine = MetadataEngine()
    engine.write_message = MagicMock()
    return engine


def test_download_metadata_success(victim, tmp_path):
    """Tests happy path where XML is fetched, parsed, and written to a local metadata.txt file."""

    provider_folder = tmp_path / "provider_dir"
    provider_folder.mkdir()
    param_inputs = ['Metadata', 'http://example.com', provider_folder, str(tmp_path / "out")]

    # Construct XML using chr() so angle brackets are preserved without rendering issues
    xml_str = (
        chr(60) + "root" + chr(62)
        + chr(60) + "time-frame" + chr(62)
        + chr(60) + "description" + chr(62) + "Dataset A" + chr(60) + "/description" + chr(62)
        + chr(60) + "start-date-time" + chr(62) + "2020-01-01" + chr(60) + "/start-date-time" + chr(62)
        + chr(60) + "end-date-time" + chr(62) + "2021-01-01" + chr(60) + "/end-date-time" + chr(62)
        + chr(60) + "/time-frame" + chr(62)
        + chr(60) + "/root" + chr(62)
    )
    xml_payload = xml_str.encode('utf-8')

    mock_response = MagicMock()
    mock_response.content = xml_payload

    with patch(f'{METADATA_PATH}.requests.get', return_value=mock_response) as mock_get:
        victim.download_metadata(param_inputs)

        mock_get.assert_called_once_with('http://example.com/inport-xml')
        
        # Verify local file creation and content
        out_file = provider_folder / 'metadata.txt'
        assert out_file.exists()
        written_content = out_file.read_text()
        assert "Description: Dataset A" in written_content
        assert "2020-01-01, 2021-01-01" in written_content
        
        # Verify status messages logged
        assert victim.write_message.call_count == 2


def test_download_metadata_connection_error(victim, tmp_path):
    """Tests that requests.exceptions.ConnectionError is caught gracefully."""

    provider_folder = tmp_path / "provider_dir"
    provider_folder.mkdir()
    param_inputs = ['Metadata', 'http://invalid-url.com', provider_folder, str(tmp_path / "out")]

    with patch(f'{METADATA_PATH}.requests.get', side_effect=requests.exceptions.ConnectionError):
        victim.download_metadata(param_inputs)

        # Ensure second write_message call recorded the error
        assert victim.write_message.call_count == 2
        error_msg = victim.write_message.call_args_list[1][0][0]
        assert "Metadata error:" in error_msg


def test_download_metadata_skipped_label(victim, tmp_path):
    """Tests skipping network call when label is not 'Metadata'."""
    provider_folder = tmp_path / "provider_dir"
    param_inputs = ['ISO metadata', 'http://example.com', provider_folder, str(tmp_path / "out")]

    with patch(f'{METADATA_PATH}.requests.get') as mock_get:
        victim.download_metadata(param_inputs)

        mock_get.assert_not_called()
        assert victim.write_message.call_count == 2
        skip_msg = victim.write_message.call_args_list[1][0][0]
        assert "skipping invalid metdata: ISO metadata" in skip_msg


def test_read_json_files(victim, tmp_path):
    """Tests discovering feature.json files locally and delegating to ThreadPoolExecutor."""
    # Set up dummy folder hierarchy on local disk
    digital_coast = tmp_path / "DigitalCoast"
    provider_dir = digital_coast / "provider_a"
    provider_dir.mkdir(parents=True)

    feature_json_path = provider_dir / "feature.json"
    feature_json_path.write_text(json.dumps({
        "links": [
            {"linkTypeName": "Metadata", "uri": "http://example.com/meta_link"}
        ]
    }))

    # Also set up an 'unused_providers' directory to verify filtering
    unused_dir = digital_coast / "unused_providers" / "provider_b"
    unused_dir.mkdir(parents=True)
    (unused_dir / "feature.json").write_text(json.dumps({
        "links": [{"linkTypeName": "Metadata", "uri": "http://example.com/ignore"}]
    }))

    with patch(f'{METADATA_PATH}.ThreadPoolExecutor') as mock_executor_cls:
        mock_executor = mock_executor_cls.return_value.__enter__.return_value

        victim.read_json_files(digital_coast, str(tmp_path / "out"))

        # Verify map was called with only valid provider params
        mock_executor.map.assert_called_once()
        args = mock_executor.map.call_args[0]
        assert args[0] == victim.download_metadata
        assert len(args[1]) == 1
        assert args[1][0] == ['Metadata', 'http://example.com/meta_link', provider_dir, str(tmp_path / "out")]


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


def test_run_with_prefix(victim, tmp_path):
    """Tests main run logic with output_prefix provided."""
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1', 'ER2'],
        'geometry': [
            Polygon([(0, 0), (1, 1), (0, 1)]),
            Polygon([(10, 10), (11, 11), (10, 11)])
        ]
    }, crs="EPSG:4326")

    with patch.object(victim, 'read_json_files') as mock_read_json, \
         patch(f'{METADATA_PATH}.get_config_item', return_value="subfolder"):

        victim.run(mock_gdf, output_prefix="prefix_dir", outputs=str(tmp_path))

        assert mock_read_json.call_count == 2
        expected_path_er1 = tmp_path / "prefix_dir" / "ER1" / "subfolder" / "DigitalCoast"
        expected_path_er2 = tmp_path / "prefix_dir" / "ER2" / "subfolder" / "DigitalCoast"

        mock_read_json.assert_any_call(expected_path_er1, str(tmp_path))
        mock_read_json.assert_any_call(expected_path_er2, str(tmp_path))


def test_run_without_prefix(victim, tmp_path):
    """Tests main run logic when output_prefix is False/None."""
    mock_gdf = gpd.GeoDataFrame({
        'EcoRegion': ['ER1'],
        'geometry': [Polygon([(0, 0), (1, 1), (0, 1)])]
    }, crs="EPSG:4326")

    with patch.object(victim, 'read_json_files') as mock_read_json, \
         patch(f'{METADATA_PATH}.get_config_item', return_value="subfolder"):

        victim.run(mock_gdf, output_prefix=False, outputs=str(tmp_path))

        expected_path = tmp_path / "ER1" / "subfolder" / "DigitalCoast"
        mock_read_json.assert_called_once_with(expected_path, str(tmp_path))


def test_write_message(tmp_path):
    """Tests writing logs to a local file path using builtins."""
    engine = MetadataEngine()

    with patch('builtins.open', mock_open()) as mock_file:
        engine.write_message("Test status message", str(tmp_path))

        mock_file.assert_called_once_with(tmp_path / 'log_prints.txt', 'a')
        mock_file.return_value.write.assert_called_once_with("Test status message\n")