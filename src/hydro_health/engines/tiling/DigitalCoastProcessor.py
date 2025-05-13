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

from botocore.client import Config
from botocore import UNSIGNED
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import set_executable


set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class DigitalCoastProcessor:
    """Class for parallel processing all BlueTopo tiles for a region"""

    def approved_dataset(self, feature_json: dict[dict]) -> bool:
        """Only allow certain provider types"""

        provider_list_text = ['USACE', 'NCMP', 'NGS']
        for text in provider_list_text:
            if text in feature_json['attributes']['provider_results_name']:
                return True
        return False
    
    def cleansed_url(self, url: str) -> str:
        """Remove found illegal characters from URLs"""

        illegal_chars = ['{', '}']
        for char in illegal_chars:
            url = url.replace(char, '')
        return url
    
    def delete_unused_folder(self, digital_coast_folder: pathlib.Path) -> None:
        """Delete any provider folders without a subfolder"""

        if digital_coast_folder.exists():
            self.write_message('Deleting empty provider folders', str(digital_coast_folder.parents[1]))
            provider_folders = os.listdir(digital_coast_folder)
            for provider in provider_folders:
                provider_folder = digital_coast_folder / provider
                if not provider_folder.suffix and 'dem' not in os.listdir(provider_folder):
                    shutil.rmtree(provider_folder)

    def download_intersected_datasets(self, param_inputs: list[list]) -> None:
        """Parallel process spatial filter and download of datasets"""

        tile_gdf, shp_path = param_inputs
        shp_df = gpd.read_file(shp_path).to_crs(4326)
        shp_df.columns = shp_df.columns.str.lower()  # make url column all lowercase
        if 'tile' in shp_df.columns:
            # drop previous tile merged column
            shp_df.drop('tile', axis=1, inplace=True)
        df_joined = shp_df.sjoin(df=tile_gdf, how='left')
        df_joined.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
        if df_joined['url'].any():
            df_joined = df_joined.loc[df_joined['tile'].notnull()]
            # Update datset to be only intesersected features
            # df_joined.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
            shp_folder = shp_path.parents[0]
            # df_joined.to_file(fr'{OUTPUTS}\{shp_path.stem}', driver='ESRI Shapefile')
            urls = df_joined['url'].unique()
            for i, url in enumerate(urls):
                cleansed_url = self.cleansed_url(url)
                # Only download .tif files
                if not cleansed_url.endswith('.tif'):
                    continue
                dataset_name = cleansed_url.split('/')[-1]
                output_file = shp_folder / dataset_name
                if os.path.exists(output_file):
                    self.write_message(f' - ({i} of {len(urls)}) Skipping data: {output_file.stem}', shp_folder.parents[4])
                    continue
                else:
                    self.write_message(f' - ({i} of {len(urls)}) Downloading data: {output_file.stem}', shp_folder.parents[4])
                
                try:
                    intersected_response = requests.get(cleansed_url, timeout=15)
                except requests.exceptions.ConnectionError:
                    self.write_message(f'#####################\nTimeout error: {cleansed_url}', shp_folder.parents[4])
                    continue

                if intersected_response.status_code == 200:
                    with open(output_file, 'wb') as file:
                        file.write(intersected_response.content)
                else:
                    return f'Failed to download: {cleansed_url}'
            return f'- {shp_path.stem}'
        else:
            return f'- No intersect: {shp_path.stem}'

    def download_tile_index(self, param_inputs: list[list]) -> None:
        """Parallel process and download tile index shapefiles"""

        download_link, output_folder_path = param_inputs
        # TODO we already only choose USACE NCMP providers, so this might always be from this s3 bucket
        # only allow dem data type.  Others are: laz
        if 'noaa-nos-coastal-lidar-pds' in download_link and 'dem' in download_link:
            _, data_file = download_link.replace('/index.html', '').split('.com')
            lidar_bucket = self.get_bucket()
            for obj_summary in lidar_bucket.objects.filter(Prefix=f"{data_file[1:]}"):
                if 'tileindex' in obj_summary.key:
                    output_zip_file = output_folder_path / obj_summary.key
                    file_parent_folder = output_zip_file.parents[0]
                    shp_path = output_zip_file.parents[0] / pathlib.Path(str(output_zip_file.stem) + '.shp')
                    if os.path.exists(shp_path):
                        self.write_message(f' - Skipping index: {output_zip_file.name}', output_folder_path.parents[2])
                        continue
                    else:
                        self.write_message(f' - Downloading index: {output_zip_file.name}', output_folder_path.parents[2])
                    file_parent_folder.mkdir(parents=True, exist_ok=True) 
                    with open(output_zip_file, 'wb') as tile_index:
                        lidar_bucket.download_fileobj(obj_summary.key, tile_index)
            return f'- {output_folder_path.parents[0]}/{data_file}'

    def get_available_datasets(self, geometry_coords: str, ecoregion_id: str, outputs: str) -> None:
        """Query NOWCoast REST API for available datasets"""

        base_url = 'https://maps.coast.noaa.gov/arcgis/rest/services/DAV/ElevationFootprints/MapServer/0/query?returnGeometry=false&f=json&where=1%3D1&outfields=%2A&spatialRel=esriSpatialRelIntersects&geometry='
        elevation_footprints_url = base_url + geometry_coords
        datasets_json = requests.get(elevation_footprints_url).json()

        tile_index_links = []
        for feature in datasets_json['features']:
            if not self.approved_dataset(feature):
                continue
            folder_name = re.sub('\W+',' ', feature['attributes']['provider_results_name']).strip().replace(' ', '_') + '_' + str(feature['attributes']['Year'])  # remove illegal chars
            output_folder_path = pathlib.Path(outputs) / ecoregion_id / 'DigitalCoast' / folder_name
            output_folder_path.mkdir(parents=True, exist_ok=True)

            # Write out JSON
            output_json = pathlib.Path(output_folder_path) / 'feature.json'
            if os.path.exists(output_json):
                # Read the existing json
                with open(output_json) as reader:
                    provider_json = json.load(reader)
                    external_provider_links = provider_json['ExternalProviderLink']
            else:
                external_provider_links = json.loads(feature['attributes']['ExternalProviderLink'])['links']
                feature['attributes']['ExternalProviderLink'] = external_provider_links
                with open(output_json, 'a') as writer:
                    writer.write(json.dumps(feature['attributes'], indent=4))
                
            # Look for Bulk Download
            for external_data in external_provider_links:
                if external_data['label'] == 'Bulk Download':
                    download_link = external_data['link']
                    tile_index_links.append({'link': download_link, 'output_path': output_folder_path})
        return tile_index_links
    
    def get_bucket(self) -> boto3.resource:
        """Connect to anonymous OCS S3 Bucket"""

        bucket = "noaa-nos-coastal-lidar-pds"
        creds = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "config": Config(signature_version=UNSIGNED),
        }
        s3 = boto3.resource('s3', **creds)
        nbs_bucket = s3.Bucket(bucket)
        return nbs_bucket
    
    def get_ecoregion_geometry_strings(self, tile_gdf: gpd.GeoDataFrame, ecoregion: str) -> str:
        """Build bbox string dictionary of tiles in web mercator projection"""

        geometry_coords = []
        ecoregion_groups = tile_gdf.groupby('EcoRegion')
        for er_id, ecoregion_group in ecoregion_groups:
            if er_id == ecoregion:
                ecoregion_group_web_mercator = ecoregion_group.to_crs(3857)
                ecoregion_group_web_mercator['geom_type'] = 'Polygon'
                tile_geometries = ecoregion_group_web_mercator[['geom_type', 'geometry']]
                tile_boundary = tile_geometries.dissolve(by='geom_type')
                bbox = json.loads(tile_boundary.bounds.to_json())
                # lower-left to upper-right
                geometry_coords.append(f"{bbox['minx']['Polygon']},{bbox['miny']['Polygon']},{bbox['maxx']['Polygon']},{bbox['maxy']['Polygon']}")
            
        return geometry_coords
    
    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)
    
    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                digital_coast_folder = pathlib.Path(outputs) / ecoregion / 'DigitalCoast'
                # tile_gdf.to_file(rF'{OUTPUTS}\tile_gdf.shp', driver='ESRI Shapefile')
                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]
                self.process_tile_index(digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs)
                self.process_intersected_datasets(digital_coast_folder, ecoregion_tile_gdf)
                if digital_coast_folder.exists():
                    self.delete_unused_folder(digital_coast_folder)

    def process_intersected_datasets(self, digital_coast_folder: pathlib.Path, ecoregion_tile_gdf: gpd.GeoDataFrame) -> None:
        """Download intersected Digital Coast files"""

        self.write_message('Downloading elevation datasets', str(digital_coast_folder.parents[1]))
        tile_index_shapefiles = digital_coast_folder.rglob('*.shp')
        param_inputs = [[ecoregion_tile_gdf, shp_path] for shp_path in tile_index_shapefiles]
        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as intersected_pool:
            self.print_async_results(intersected_pool.map(self.download_intersected_datasets, param_inputs), str(digital_coast_folder.parents[1]))

    def process_tile_index(self, digital_coast_folder: pathlib.Path, tile_gdf: gpd.GeoDataFrame, ecoregion: str, outputs: str) -> None:
        """Download tile_index shapefiles"""

        self.write_message('Download Tile Index shapefiles', str(digital_coast_folder.parents[1]))
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)

        for geometry_coords in ecoregion_geom_strings:
            tile_index_links = self.get_available_datasets(geometry_coords, ecoregion, outputs)  # TODO return all object keys
            param_inputs = [[link_dict['link'], link_dict['output_path']] for link_dict in tile_index_links]
            with ThreadPoolExecutor(int(os.cpu_count() - 2)) as tile_index_pool:
                self.print_async_results(tile_index_pool.map(self.download_tile_index, param_inputs), str(digital_coast_folder.parents[1]))
            self.unzip_all_files(digital_coast_folder)

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