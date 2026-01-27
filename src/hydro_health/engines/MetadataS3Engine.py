"""Class for obtaining all available files"""

import boto3
import json
import os
import re
import requests
import geopandas as gpd
import pathlib
import s3fs

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from hydro_health.helpers.tools import get_config_item


class MetadataS3Engine:
    """Class for parallel processing metadata for a region"""

    def upload_metadata_to_s3(self, param_inputs: list[list]) -> None:
        """Parallel process and download metadata dates"""

        label, download_link, provider_folder, outputs = param_inputs
        provider_stem = pathlib.Path(provider_folder.split('/')[-1]).stem
        # self.write_message(f' - getting metadata: {provider_stem}', outputs)
        s3_files = s3fs.S3FileSystem()
        if label == 'Metadata':
            full_list_url = download_link + '/inport-xml'
            try:
                metadata_response = requests.get(full_list_url, timeout=10)
                metadata_response.raise_for_status()
                metadata_xml = metadata_response.content
                xml_root = BeautifulSoup(metadata_xml, features="xml")
                time_frames = xml_root.find_all('time-frame')
                with s3_files.open(provider_folder + '/metadata.txt', 'w') as writer:
                    for time in time_frames:
                        description = time.find('description').get_text() if time.find('description') else provider_stem
                        start = time.find('start-date-time')
                        end = time.find('end-date-time')
                        writer.write(f'Description: {description}\n')
                        writer.write(f'{start.get_text()}, {end.get_text()}\n')
                self.write_message(f' - stored metadata: {provider_folder}', outputs)
            except requests.exceptions.ConnectionError:
                self.write_message(f'#####################\nMetadata error: {full_list_url}', outputs)
                pass
        else:
            # handle CUDEM data
            cudem_tiffs = s3_files.glob(f"{provider_folder}/**/*.json")  # will CUDEM files not end in "YYYYv1.tif"?
            years = sorted([pathlib.Path(tif.split('/')[-1]).stem[-6:-2] for tif in cudem_tiffs])
            start = years[0]
            end = years[-1]  # if only 1 year, -1 will still work
            with s3_files.open(provider_folder + '/metadata.txt', 'w') as writer:
                writer.write(f'Description: CUDEM\n')
                writer.write(f'{start}, {end}\n')
            self.write_message(f' - stored metadata: {provider_folder}', outputs)

    def read_json_files(self, digital_coast_path: pathlib.Path, outputs: str) -> None:
        """Read JSON files to download metadata information"""

        print('- Reading DigitalCoast JSON files')
        s3_files = s3fs.S3FileSystem()
        feature_json_files = s3_files.glob(f"{digital_coast_path}/**/*.json")
        metadata_params = []
        for feature_json in feature_json_files:
            provider_folder = '/'.join(feature_json.split('/')[:-1])
            with s3_files.open(feature_json, 'r') as json_file:
                feature = json.load(json_file)
            external_provider_links = feature['ExternalProviderLink']
            metadata_values = {'Metadata', 'ISO metadata'}
            for external_data in external_provider_links:
                current_labels = {external_data.get('altlabel'), external_data.get('label')}
                if current_labels & metadata_values:  # See if current_labels are in metadata_values set
                    label = 'ISO metadata' if 'iso' in external_data.get('link', '').lower() else 'Metadata'
                    metadata_params.append([label, feature['Metalink'], provider_folder, outputs])
                    break
            
        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as meta_pool:
            meta_pool.map(self.upload_metadata_to_s3, metadata_params)

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

    def run(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False) -> None:
        """Main entry point for creating metadata.txt for tracking year-pairs"""

        print('Downloading Metadata Datasets')
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            print('Starting:', ecoregion)
            digital_coast_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/testing/{ecoregion}/{get_config_item('DIGITALCOAST', 'SUBFOLDER')}/DigitalCoast"
            self.read_json_files(digital_coast_path, outputs)

    def write_message(self, message: str, output_folder: str) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')








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
        self.write_message(f' - getting metadata: {provider_folder.stem}', outputs)
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
                self.write_message(f' - stored metadata: {provider_folder}', outputs)
            except requests.exceptions.ConnectionError:
                self.write_message(f'#####################\nMetadata error: {full_list_url}', outputs)
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
            self.write_message(f' - stored metadata: {provider_folder}', outputs)

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
                json_label = external_data['label'] if external_data['label'] else external_data['altlabel']  # Some labels are empty
                if json_label in ['Metadata', 'ISO metadata']:
                    if 'iso' in external_data['link']:
                        label = 'ISO metadata'
                    else:
                        label = 'Metadata'
                    metadata_params.append([label, feature['Metalink'], provider_folder, outputs])
                    break
            
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

    def run(self, tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Main entry point for creating metadata.txt for tracking year-pairs"""

        print('Downloading Metadata Datasets')
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            print('Starting:', ecoregion)
            digital_coast_folder = pathlib.Path(outputs) / ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            self.read_json_files(digital_coast_folder, ecoregion, outputs)

    def write_message(self, message: str, output_folder: str) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')
