"""Class for obtaining all available files"""

import boto3
import json
import os
import re
import zipfile
import requests
import shutil
import sys
import geopandas as gpd
import pathlib

from bs4 import BeautifulSoup
from botocore.client import Config
from botocore import UNSIGNED
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import set_executable

from hydro_health.helpers.tools import get_config_item


set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class MetadataEngine:
    """Class for parallel processing metadata for a region"""

    def download_metadata(self, param_inputs: list[list]) -> None:
        """Parallel process and download metadata dates"""

        label, download_link, provider_folder, outputs = param_inputs
        # self.write_message(f' - getting metadata: {provider_folder.stem}', outputs)
        if label == 'Metadata':
            full_list_url = download_link + '/inport-xml'
            try:
                metadata_response = requests.get(full_list_url)
                metadata_xml = metadata_response.content
                xml_root = BeautifulSoup(metadata_xml, features="xml")
                time_frames = xml_root.find_all('time-frame')
                with open(provider_folder / 'metadata.txt', 'w') as writer:
                    for time in time_frames:
                        description = time.find('description').get_text() if time.find('description') else provider_folder.stem
                        start = time.find('start-date-time')
                        end = time.find('end-date-time')
                        writer.write(f'Description: {description}\n')
                        writer.write(f'{start.get_text()}, {end.get_text()}\n')
                # self.write_message(f' - stored metadata: {provider_folder}', outputs)
            except requests.exceptions.ConnectionError:
                # self.write_message(f'#####################\nMetadata error: {full_list_url}', outputs)
                pass
        else:
            # handle CUDEM data
            cudem_tiffs = provider_folder.rglob('*.tif')  # will CUDEM files not end in "YYYYv1.tif"?
            years = sorted([tif.stem[-6:-2] for tif in cudem_tiffs])
            start = years[0]
            end = years[-1]  # if only 1 year, -1 will still work
            with open(provider_folder / 'metadata.txt', 'w') as writer:
                writer.write(f'Description: CUDEM\n')
                writer.write(f'{start}, {end}\n')
            # self.write_message(f' - stored metadata: {provider_folder}', outputs)

    def read_json_files(self, digital_coast_folder: pathlib.Path, ecoregion: str, outputs: str) -> None:
        """Read JSON files to download metadata information"""

        print('- Reading DigitalCoast JSON files')
        feature_json_files = [feature_json for feature_json in digital_coast_folder.rglob('feature.json') if 'unused_providers' not in str(feature_json)]
        metadata_params = []
        for feature_json in feature_json_files:
            provider_folder = feature_json.parents[0]
            with open(feature_json, 'r') as json_file:
                feature = json.load(json_file)
            external_provider_links = feature['ExternalProviderLink']
            for external_data in external_provider_links:
                if external_data['label'] in ['Metadata', 'ISO metadata']:
                    if 'iso' in external_data['link']:
                        label = 'ISO metadata'
                    else:
                        label = 'Metadata'
                    break
            metadata_params.append([label, feature['Metalink'], provider_folder, outputs])

        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as meta_pool:
            meta_pool.map(self.download_metadata, metadata_params)

    def get_ecoregion_geometry_strings(self, tile_gdf: gpd.GeoDataFrame, ecoregion: str) -> str:
        """Build bbox string dictionary of tiles in web mercator projection"""

        geometry_coords = []
        ecoregion_groups = tile_gdf.groupby('EcoRegion')
        for er_id, ecoregion_group in ecoregion_groups:
            if er_id == ecoregion:
                ecoregion_group_web_mercator = ecoregion_group.to_crs(4269)  # POST request only allows this EPSG
                ecoregion_group_web_mercator['geom_type'] = 'Polygon'
                tile_geometries = ecoregion_group_web_mercator[['geom_type', 'geometry']]
                tile_boundary = tile_geometries.dissolve(by='geom_type')
                tile_wkt = tile_boundary.iloc[0].geometry
                geometry_coords.append(tile_wkt)

        return geometry_coords

    def run(self, ecoregions: list[str], outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        print('Downloading Digital Coast Datasets')
        for ecoregion in ecoregions:
            print('Starting:', ecoregion)
            digital_coast_folder = pathlib.Path(outputs) / ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            self.read_json_files(digital_coast_folder, ecoregion, outputs)

    def write_message(self, message: str, output_folder: str) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')
