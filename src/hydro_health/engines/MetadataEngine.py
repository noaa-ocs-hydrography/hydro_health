import json
import geopandas as gpd
import os
import pathlib
import requests
import s3fs

from typing import Generator, TextIO
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


class MetadataEngine(Engine):
    """Class for parallel processing metadata for a region"""

    def __init__(self, param_lookup: dict[dict]) -> None:
        """Initialize MetadataEngine with environment lookup and configuration"""

        super().__init__()
        self.param_lookup = param_lookup
        self.is_s3 = self.param_lookup.get('env') == 'aws'
        self.s3_files = s3fs.S3FileSystem() if self.is_s3 else None

    @contextmanager
    def _open(self, file_path: str | pathlib.Path, mode: str = 'r') -> Generator[TextIO, None, None]:
        """Context manager to open either a local or S3 file stream"""

        if self.is_s3:
            with self.s3_files.open(str(file_path), mode) as stream:
                yield stream
        else:
            with open(
                file_path, mode, encoding='utf-8' if 'b' not in mode else None
            ) as stream:
                yield stream

    def _find_files(self, base_folder: str | pathlib.Path, pattern: str) -> list[str]:
        """Find files matching a glob pattern across local paths or S3 paths"""

        if self.is_s3:
            return list(self.s3_files.glob(f'{base_folder}/**/{pattern}'))
        return [str(p) for p in pathlib.Path(base_folder).rglob(pattern)]

    @staticmethod
    def _extract_metadata_link(feature: dict) -> tuple[str, str] | None:
        """Extract label and URL supporting both schema formats"""

        if 'ExternalProviderLink' in feature:
            metadata_values = {'Metadata', 'ISO metadata'}
            for external_data in feature.get('ExternalProviderLink', []):
                current_labels = {
                    external_data.get('altlabel'),
                    external_data.get('label'),
                }
                if current_labels & metadata_values:
                    is_iso = 'iso' in external_data.get('link', '').lower()
                    label = 'ISO metadata' if is_iso else 'Metadata'
                    return label, feature.get('Metalink')

        if 'links' in feature:
            for link in feature.get('links', []):
                if link.get('linkTypeName') == 'Metadata':
                    return 'Metadata', link.get('uri')

        return None

    def download_metadata(self, param_inputs: list) -> None:
        """Parallel process and download metadata dates"""

        label, download_link, provider_folder, outputs = param_inputs
        provider_stem = pathlib.PurePosixPath(str(provider_folder)).name
        self.write_message(f' - getting metadata: {provider_stem}', outputs)

        if label in {'Metadata', 'ISO metadata'}:
            full_list_url = download_link.rstrip('/') + '/inport-xml'
            try:
                metadata_response = requests.get(full_list_url, timeout=10)
                metadata_response.raise_for_status()

                metadata_xml = metadata_response.content
                xml_root = BeautifulSoup(metadata_xml, features='xml')
                time_frames = xml_root.find_all('time-frame')

                target_file = (
                    f'{provider_folder}/metadata.txt'
                    if self.is_s3
                    else pathlib.Path(provider_folder) / 'metadata.txt'
                )

                with self._open(target_file, 'w') as writer:
                    for time in time_frames:
                        desc_elem = time.find('description')
                        description = (desc_elem.get_text() if desc_elem else provider_stem)
                        start = time.find('start-date-time')
                        end = time.find('end-date-time')
                        start_str = start.get_text() if start else ''
                        end_str = end.get_text() if end else ''

                        writer.write(f'Description: {description}\n')
                        writer.write(f'{start_str}, {end_str}\n')

                self.write_message(f' - stored metadata: {provider_folder}', outputs)

            except requests.RequestException as err:
                self.write_message(
                    f'#####################\nMetadata error [{err}]: '
                    f'{full_list_url}',
                    outputs,
                )
        else:
            self.write_message(f' - skipping invalid metadata: {label}', outputs)

    def read_json_files(self, digital_coast_path: str | pathlib.Path, outputs: str) -> None:
        """Read JSON files to download metadata information"""

        print(' - Reading DigitalCoast JSON files')
        raw_json_files = self._find_files(digital_coast_path, '*.json')
        json_files = [f for f in raw_json_files if 'unused_providers' not in str(f)]

        metadata_params = []
        for json_path in json_files:
            provider_folder = str(pathlib.PurePosixPath(json_path).parent)
            try:
                with self._open(json_path, 'r') as json_file:
                    feature = json.load(json_file)
            except (json.JSONDecodeError, OSError):
                continue

            link_info = self._extract_metadata_link(feature)
            if link_info:
                label, uri = link_info
                metadata_params.append([label, uri, provider_folder, outputs])

        worker_count = max(1, (os.cpu_count() or 4) - 2)
        with ThreadPoolExecutor(worker_count) as meta_pool:
            meta_pool.map(self.download_metadata, metadata_params)

    def get_ecoregion_geometry_strings(self, tile_gdf: gpd.GeoDataFrame, ecoregion: str) -> list:
        """Build bbox string dictionary of tiles in web mercator projection"""

        geometry_coords = []
        ecoregion_groups = tile_gdf.groupby('EcoRegion')
        for er_id, ecoregion_group in ecoregion_groups:
            if er_id == ecoregion:
                ecoregion_group_web_mercator = ecoregion_group.to_crs(4269)
                ecoregion_group_web_mercator['geom_type'] = 'Polygon'
                tile_geometries = ecoregion_group_web_mercator[['geom_type', 'geometry']]
                tile_boundary = tile_geometries.dissolve(by='geom_type')
                tile_wkt = tile_boundary.iloc[0].geometry
                geometry_coords.append(tile_wkt)

        return geometry_coords

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str | bool, outputs: str,) -> None:
        """Main entry point for creating metadata.txt for tracking year-pairs"""

        print(f"Starting MetadataEngine ({'aws' if self.is_s3 else 'local'})")

        ecoregions = list(tile_gdf['EcoRegion'].unique())
        subfolder = get_config_item('DIGITALCOAST', 'SUBFOLDER')

        for ecoregion in ecoregions:
            print('- Starting:', ecoregion)
            ecoregion_subpath = f'{ecoregion}/{subfolder}/DigitalCoast'
            if output_prefix:
                ecoregion_subpath = f'{output_prefix}/{ecoregion_subpath}'

            if self.is_s3:
                target_folder = f's3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/{ecoregion_subpath}'
            else:
                base_dir = (pathlib.Path(output_prefix) if output_prefix else pathlib.Path(outputs))
                target_folder = base_dir / ecoregion_subpath

            self.read_json_files(target_folder, outputs)
            found_metadata = self._find_files(target_folder, 'metadata.txt')
            self.write_run_manifest(ecoregion_subpath, {'metadata': len(found_metadata)})