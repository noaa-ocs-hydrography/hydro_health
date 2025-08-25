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


class DigitalCoastEngine:
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self) -> None:
        self.approved_size = 200000000  # 2015 USACE polygon was 107,987,252 sq. meters
        self.tile_index_links = None  # Store for use with CUDEM

    def approved_dataset(self, feature_json: dict[dict]) -> bool:
        """Only allow certain provider types"""

        # CUDEM: NOAA NCEI
        provider_list_text = ['USACE', 'NCMP', 'NGS', 'NOAA NCEI']
        for text in provider_list_text:
            if text in feature_json['attributes']['provider_results_name']:
                return True
        return False

    def check_tile_index_areas(self, digital_coast_folder, outputs) -> None:
        """Exclude any small area surveys"""

        self.write_message(f'Checking area size of tileindex files', outputs)
        tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('tileindex*.shp') if 'unused_providers' not in str(folder)]
        move_providers = []
        for shp_path in tile_index_shapefiles:
            shp_df = gpd.read_file(shp_path).to_crs(9822)  # Albers Equal Area
            shp_df['area'] = shp_df['geometry'].area
            total_area = shp_df["area"].sum()
            if total_area < self.approved_size:
                self.write_message(f' - provider too small: {total_area} - {shp_path}', outputs)
                move_providers.append(shp_path.parents[2])
        for provider in move_providers:
            unused_provider_folder = digital_coast_folder / 'unused_providers'
            if not unused_provider_folder.exists():
                unused_provider_folder.mkdir()
            if pathlib.Path(unused_provider_folder / provider.stem).exists():
                shutil.rmtree(unused_provider_folder / provider.stem)
            shutil.move(provider, unused_provider_folder)

    def cleansed_url(self, url: str) -> str:
        """Remove found illegal characters from URLs"""

        illegal_chars = ['{', '}']
        for char in illegal_chars:
            url = url.replace(char, '')
        return url

    def delete_unused_folder(self, digital_coast_folder: pathlib.Path, outputs: str) -> None:
        """Delete any provider folders without a subfolder"""

        if digital_coast_folder.exists():
            self.write_message('Deleting empty provider folders', outputs)
            provider_folders = os.listdir(digital_coast_folder)
            provider_folders = [folder for folder in digital_coast_folder.glob('*') if folder.is_dir()]
            for provider in provider_folders:
                if 'unused_providers' != provider.stem:
                    provider_folder = digital_coast_folder / provider
                    if not provider_folder.suffix and 'dem' not in os.listdir(provider_folder):
                        self.write_message(f' - removing empty provider: {provider_folder}', outputs)
                        shutil.rmtree(provider_folder)

    def download_intersected_datasets(self, param_inputs: list[list]) -> None:
        """Parallel process spatial filter and download of datasets"""

        tile_gdf, shp_path, outputs = param_inputs
        shp_df = gpd.read_file(shp_path).to_crs(4326)
        shp_df.columns = shp_df.columns.str.lower()  # make url column all lowercase
        if 'tile' in shp_df.columns:
            # drop previous tile merged column
            shp_df.drop('tile', axis=1, inplace=True)
        df_joined = shp_df.sjoin(df=tile_gdf, how='left')
        df_joined.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
        if df_joined['url'].any():
            df_joined = df_joined.loc[df_joined['tile'].notnull()]
            shp_folder = shp_path.parents[0]
            urls = df_joined['url'].unique()
            for i, url in enumerate(urls):
                cleansed_url = self.cleansed_url(url)
                # Only download .tif files
                if not cleansed_url.endswith('.tif'):
                    continue
                dataset_name = cleansed_url.split('/')[-1]
                output_file = shp_folder / dataset_name
                if os.path.exists(output_file):
                    self.write_message(f' - ({i} of {len(urls)}) Skipping data: {output_file.stem}', outputs)
                    continue
                else:
                    self.write_message(f' - ({i} of {len(urls)}) Downloading data: {output_file.stem}', outputs)
                try:
                    intersected_response = requests.get(cleansed_url, timeout=15)
                except requests.exceptions.ConnectionError:
                    self.write_message(f'#####################\nTimeout error: {cleansed_url}', outputs)
                    continue

                if intersected_response.status_code == 200:
                    with open(output_file, 'wb') as file:
                        file.write(intersected_response.content)
                else:
                    return f'Failed to download: {cleansed_url}'
            return f'- {shp_path.stem}'
        else:
            return f'- No intersect: {shp_path.stem}'

    def download_metadata(self, param_inputs: list[list]) -> None:
        """Parallel process and download metadata dates"""

        download_link, provider_folder, outputs = param_inputs
        full_list_url = download_link + '/inport-xml'
        try:
            metadata_response = requests.get(full_list_url)
            metadata_xml = metadata_response.content
            xml_root = BeautifulSoup(metadata_xml, features="xml")
            time_frames = xml_root.find_all('time-frame')
            with open(provider_folder / 'metadata.txt', 'w') as writer:
                for time in time_frames:
                    description = time.find('description')
                    start = time.find('start-date-time')
                    end = time.find('end-date-time')
                    writer.write(f'Description: {description.get_text()}\n')
                    writer.write(f'{start.get_text()}, {end.get_text()}\n')
            self.write_message(f' - stored metadata: {provider_folder}', outputs)
            self.write_message(f' - metadata URL: {full_list_url}')
        except requests.exceptions.ConnectionError:
            self.write_message(f'#####################\nMetadata error: {full_list_url}', outputs)
            pass

    def download_support_files(self, digital_coast_folder: pathlib.Path, tile_gdf: gpd.GeoDataFrame, ecoregion: str, outputs: str) -> None:
        """Download tile_index shapefiles"""

        self.write_message('Download Support Files', outputs)
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)

        for geometry_coords in ecoregion_geom_strings:
            self.tile_index_links = self.get_available_datasets(geometry_coords, ecoregion, outputs)  # TODO return all object keys
            bulk_download_params = [[link_dict['link'], link_dict['provider_path'], outputs] for link_dict in self.tile_index_links if link_dict['label'] == 'Bulk Download']
            metadata_params = [[link_dict['link'], link_dict['provider_path'], outputs] for link_dict in self.tile_index_links if link_dict['label'] == 'Metadata']

            with ThreadPoolExecutor(int(os.cpu_count() - 2)) as bulk_pool:
                bulk_pool.map(self.download_tile_index, bulk_download_params)
            with ThreadPoolExecutor(int(os.cpu_count() - 2)) as meta_pool:
                meta_pool.map(self.download_metadata, metadata_params)
            self.unzip_all_files(digital_coast_folder)

    def download_tile_index(self, param_inputs: list[list]) -> None:
        """Parallel process and download tile index shapefiles"""

        download_link, provider_folder, outputs = param_inputs
        # TODO we already only choose USACE NCMP providers, so this might always be from this s3 bucket
        # only allow dem data type.  Others are: laz
        if get_config_item('DIGITALCOAST', 'BUCKET') in download_link and 'dem' in download_link:
            _, data_file = download_link.replace('/index.html', '').split('.com')
            lidar_bucket = self.get_bucket()
            for obj_summary in lidar_bucket.objects.filter(Prefix=f"{data_file[1:]}"):
                if 'tileindex' in obj_summary.key:
                    output_zip_file = provider_folder / obj_summary.key
                    file_parent_folder = output_zip_file.parents[0]
                    shp_path = output_zip_file.parents[0] / pathlib.Path(str(output_zip_file.stem) + '.shp')
                    provider_and_file = str(pathlib.Path(*shp_path.parts[-4:]))
                    if os.path.exists(shp_path):
                        self.write_message(f' - Skipping index: {provider_and_file}', outputs)
                        continue
                    else:
                        self.write_message(f' - Downloading index: {provider_and_file}', outputs)
                    file_parent_folder.mkdir(parents=True, exist_ok=True) 
                    with open(output_zip_file, 'wb') as tile_index:
                        lidar_bucket.download_fileobj(obj_summary.key, tile_index)

    def get_available_datasets(self, geometry_coords: str, ecoregion_id: str, outputs: str) -> None:
        """Query NOWCoast REST API for available datasets"""

        payload = {
            "aoi": f"SRID=4269;{geometry_coords}",
            "published": "true",
            "dataTypes": ["Lidar", "DEM"],
            "dialect": "arcgis",
        }
        response = requests.post(get_config_item('DIGITALCOAST', 'API'), data=payload)
        datasets_json = response.json()

        if response.status_code == 404:
            raise Exception(f"Digital Coast Error: {response.reason}")

        tile_index_links = []
        for feature in datasets_json['features']:
            if not self.approved_dataset(feature):
                continue
            folder_name = re.sub('\W+',' ', feature['attributes']['provider_results_name']).strip().replace(' ', '_') + '_' + str(feature['attributes']['Year'])  # remove illegal chars
            output_folder_path = pathlib.Path(outputs) / ecoregion_id / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast' / f"{folder_name}_{feature['attributes']['ID']}"
            output_folder_path.mkdir(parents=True, exist_ok=True)

            # Write out JSON
            output_json = pathlib.Path(output_folder_path) / 'feature.json'
            external_provider_links = json.loads(feature['attributes']['ExternalProviderLink'])['links']
            feature['attributes']['ExternalProviderLink'] = external_provider_links
            with open(output_json, 'a') as writer:
                writer.write(json.dumps(feature['attributes'], indent=4))

            for external_data in external_provider_links:
                if external_data['label'] == 'Bulk Download':
                    tile_index_links.append({'label': 'Bulk Download', 'link': external_data['link'], 'provider_path': output_folder_path})
                elif external_data['label'] == 'Metadata':
                    tile_index_links.append({'label': 'Metadata', 'link': external_data['link'], 'provider_path': output_folder_path})
                elif external_data['label'] == 'ISO metadata':
                    tile_index_links.append({'label': 'ISO metadata', 'provider_path': output_folder_path}) 
        return tile_index_links

    def get_bucket(self) -> boto3.resource:
        """Connect to anonymous OCS S3 Bucket"""

        creds = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "config": Config(signature_version=UNSIGNED),
        }
        s3 = boto3.resource('s3', **creds)
        nbs_bucket = s3.Bucket(get_config_item('DIGITALCOAST', 'BUCKET'))
        return nbs_bucket

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

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)

    def run(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        print('Downloading Digital Coast Datasets')
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                digital_coast_folder = pathlib.Path(outputs) / ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
                # tile_gdf.to_file(rF'{OUTPUTS}\tile_gdf.shp', driver='ESRI Shapefile')
                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]
                self.download_support_files(digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs)
                self.check_tile_index_areas(digital_coast_folder, outputs)
                self.process_intersected_datasets(digital_coast_folder, ecoregion_tile_gdf, outputs)
                self.store_cudem_metadata(ecoregion_tile_gdf, ecoregion, outputs)
                if digital_coast_folder.exists():
                    self.delete_unused_folder(digital_coast_folder, outputs)

    def process_intersected_datasets(self, digital_coast_folder: pathlib.Path, ecoregion_tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Download intersected Digital Coast files"""

        self.write_message('Downloading elevation datasets', outputs)
        tile_index_shapefiles = digital_coast_folder.rglob('*.shp')
        param_inputs = [[ecoregion_tile_gdf, shp_path, outputs] for shp_path in tile_index_shapefiles]
        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as intersected_pool:
            self.print_async_results(intersected_pool.map(self.download_intersected_datasets, param_inputs), outputs)

    def store_cudem_metadata(self, tile_gdf: gpd.GeoDataFrame, ecoregion: str, outputs: str) -> None:
        """Parallel process and download CUDEM metadata"""

        self.write_message('Storing CUDEM metadata', outputs)
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)

        for _ in ecoregion_geom_strings:
            iso_metadata_params = [[link_dict['provider_path'], outputs] for link_dict in self.tile_index_links if link_dict['label'] == 'ISO metadata']
            for provider_info in iso_metadata_params:
                provider_folder, outputs = provider_info
                # CUDEM seems to have same provider name with unique tiff names
                # only using first result for now unless provider name changes
                cudem_tiffs = provider_folder.rglob('*.tif')  # will CUDEM files not end in "YYYYv1.tif"?
                years = sorted([tif.stem[-6:-2] for tif in cudem_tiffs])
                start = years[0]
                end = years[-1]  # if only 1 year, -1 will still work
                with open(provider_folder / 'metadata.txt', 'w') as writer:
                    writer.write(f'Description: CUDEM\n')
                    writer.write(f'{start}, {end}\n')
                self.write_message(f' - stored metadata: {provider_folder}', outputs)
                break

    def unzip_all_files(self, output_folder: str) -> None:
        """Unzip all zip files in a folder"""

        for zipped_file in pathlib.Path(output_folder).rglob('*.zip'):
            with zipfile.ZipFile(zipped_file, 'r') as zipped:
                zipped.extractall(str(zipped_file.parents[0]))
            # Delete zip file after extract
            zipped_file.unlink()

    def write_message(self, message: str, output_folder: str) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')
