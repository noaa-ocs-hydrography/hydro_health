"""Class for obtaining all available files"""

import boto3
import json
import os
import re
import zipfile
import requests
import sys
import geopandas as gpd
import pathlib
import multiprocessing as mp
import numpy as np

from botocore.client import Config
from botocore import UNSIGNED


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class DigitalCoastProcessor:
    """Class for parallel processing all BlueTopo tiles for a region"""

    def download_tile_index(self, download_link, output_folder_path) -> None:
        if 'noaa-nos-coastal-lidar-pds' in download_link:
            _, data_file = download_link.replace('/index.html', '').split('.com')
            lidar_bucket = self.get_bucket()
            for obj_summary in lidar_bucket.objects.filter(Prefix=f"{data_file[1:]}"):
                if 'tileindex' in obj_summary.key:
                    output_zip_file = output_folder_path / obj_summary.key
                    file_parent_folder = output_zip_file.parents[0]
                    if os.path.exists(output_zip_file):
                        print(f'Skipping: {output_zip_file.name}', output_folder_path)
                        continue
                    else:
                        print(f'Downloading: {output_zip_file.name}', output_folder_path)
                    file_parent_folder.mkdir(parents=True, exist_ok=True) 
                    with open(output_zip_file, 'wb') as tile_index:
                        lidar_bucket.download_fileobj(obj_summary.key, tile_index)

    def get_available_datasets(self, geometry_coords: str, outputs) -> None:
        """Query NOWCoast REST API for available datasets"""

        base_url = 'https://maps.coast.noaa.gov/arcgis/rest/services/DAV/ElevationFootprints/MapServer/0/query?returnGeometry=false&f=json&where=1%3D1&outfields=%2A&spatialRel=esriSpatialRelIntersects&state=FL&geometry='
        elevation_footprints_url = base_url + geometry_coords
        datasets_json = requests.get(elevation_footprints_url).json()

        tile_index_links = []
        for feature in datasets_json['features']:
            feature_json = json.dumps(feature, indent=4) + '\n\n'
            folder_name = re.sub('\W+',' ', feature['attributes']['provider_results_name']).strip().replace(' ', '_') + '_' + str(feature['attributes']['Year'])  # remove illegal chars
            output_folder_path = pathlib.Path(outputs) / 'DigitalCoast' / folder_name
            output_folder_path.mkdir(parents=True, exist_ok=True)

            # Write out JSON
            output_json = pathlib.Path(output_folder_path) / 'feature.json'
            if os.path.exists(output_json):
                output_json.unlink()
            external_data_json = json.loads(feature['attributes']['ExternalProviderLink'])
            with open(output_json, 'a') as writer:
                writer.write(feature_json + '\n\n')
                writer.write(json.dumps(external_data_json['links'], indent=4) + '\n\n')
            
            # Look for Bulk Download
            for external_data in external_data_json['links']:
                if external_data['label'] == 'Bulk Download':
                    download_link = external_data['link']
                    tile_index_links.append({'link': download_link, 'output_path': output_folder_path})
        return tile_index_links
            

    def get_bucket(self):
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
    
    def get_geometry_string(self, tile_gdf: gpd.GeoDataFrame) -> str:
        """Build bbox string of tiles"""

        tile_gdf_web_mercator = tile_gdf.to_crs(3857)
        tile_gdf_web_mercator['geom_type'] = 'Polygon'
        tile_geometries = tile_gdf_web_mercator[['geom_type', 'geometry']]
        tile_boundary = tile_geometries.dissolve(by='geom_type')
        bbox = json.loads(tile_boundary.bounds.to_json())
        # lower-left to upper-right
        geometry_coords = f"{bbox['minx']['Polygon']},{bbox['miny']['Polygon']},{bbox['maxx']['Polygon']},{bbox['maxy']['Polygon']}"
        return geometry_coords

    def get_pool(self, processes=int(mp.cpu_count() / 2)):
        """Obtain a multiprocessing Pool"""

        return mp.Pool(processes=processes)
    
    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        geometry_coords = self.get_geometry_string(tile_gdf)
        tile_index_links = self.get_available_datasets(geometry_coords, outputs)  # TODO return all object keys
        
        with self.get_pool() as process_pool:
            results = [process_pool.apply_async(self.download_tile_index, [link_dict['link'], link_dict['output_path']]) for link_dict in tile_index_links]
            for result in results:
                result.get()
        self.unzip_all_files(pathlib.Path(outputs) / 'DigitalCoast')

    def unzip_all_files(self, output_folder) -> None:
        """Unzip all zip files in a folder"""

        for zipped_file in pathlib.Path(output_folder).rglob('*.zip'):
            with zipfile.ZipFile(zipped_file, 'r') as zipped:
                zipped.extractall(str(zipped_file.parents[0]))

    def write_message(self, message, output_folder):
        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')