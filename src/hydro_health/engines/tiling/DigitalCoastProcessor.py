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
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import set_executable


set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


class DigitalCoastProcessor:
    """Class for parallel processing all BlueTopo tiles for a region"""

    def approved_dataset(self, feature_json: dict[dict]) -> bool:
        """Only allow certain provider types"""

        provider_list_text = ['National Coastal Mapping Program', 'USACE NCMP']
        for text in provider_list_text:
            if text in feature_json['attributes']['provider_details_name']:
                return True
        return False
    
    def delete_unused_folder(self, digital_coast_folder: pathlib.Path) -> None:
        """Delete any provider folders without a subfolder"""

        print('Deleting empty provider folders')
        provider_folders = os.listdir(digital_coast_folder)
        for provider in provider_folders:
            provider_folder = digital_coast_folder / provider
            if 'dem' not in os.listdir(provider_folder):
                print(f'{provider_folder}')
                shutil.rmtree(provider_folder)

    def download_intersected_datasets(self, param_inputs: list[list]) -> None:
        """Parallel process spatial filter and download of datasets"""

        tile_gdf, shp_path = param_inputs
        shp_df = gpd.read_file(shp_path).to_crs(4326)
        shp_df.columns = shp_df.columns.str.lower()  # make url column all lowercase
        df_joined = shp_df.sjoin(df=tile_gdf, how='left', predicate='intersects')
        df_joined = df_joined.loc[df_joined['tile'].notnull()]
        shp_folder = shp_path.parents[0]
        if df_joined['url'].any():
            # df_joined.to_file(fr'{OUTPUTS}\{shp_path.stem}', driver='ESRI Shapefile')
            for url in df_joined['url'].unique():
                # Only download .tif files
                # TODO try to handle this earlier when laz or dem folder is created
                # otherwise we need to delete the provider folders with laz only
                if not url.endswith('.tif'):
                    continue
                dataset_name = url.split('/')[-1]
                output_file = shp_folder / dataset_name
                if os.path.exists(output_file):
                    continue
                intersected_response = requests.get(url)
                if intersected_response.status_code == 200:
                    with open(output_file, 'wb') as file:
                        file.write(intersected_response.content)
                else:
                    return f'Failed to download: {url}'
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
                    if os.path.exists(output_zip_file):
                        print(f'Skipping: {output_zip_file.name}', output_folder_path)
                        continue
                    else:
                        print(f'Downloading: {output_zip_file.name}', output_folder_path)
                    file_parent_folder.mkdir(parents=True, exist_ok=True) 
                    with open(output_zip_file, 'wb') as tile_index:
                        lidar_bucket.download_fileobj(obj_summary.key, tile_index)
            return f'- {output_folder_path.parents[0]}/{data_file}'

    def get_available_datasets(self, geometry_coords: str, outputs) -> None:
        """Query NOWCoast REST API for available datasets"""

        base_url = 'https://maps.coast.noaa.gov/arcgis/rest/services/DAV/ElevationFootprints/MapServer/0/query?returnGeometry=false&f=json&where=1%3D1&outfields=%2A&spatialRel=esriSpatialRelIntersects&state=FL&geometry='
        elevation_footprints_url = base_url + geometry_coords
        datasets_json = requests.get(elevation_footprints_url).json()

        tile_index_links = []
        for feature in datasets_json['features']:
            feature_json = json.dumps(feature, indent=4) + '\n\n'
            if not self.approved_dataset(feature):
                continue
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
        """Build bbox string of tiles in web mercator projection"""

        tile_gdf_web_mercator = tile_gdf.to_crs(3857)
        tile_gdf_web_mercator['geom_type'] = 'Polygon'
        tile_geometries = tile_gdf_web_mercator[['geom_type', 'geometry']]
        tile_boundary = tile_geometries.dissolve(by='geom_type')
        bbox = json.loads(tile_boundary.bounds.to_json())
        # lower-left to upper-right
        geometry_coords = f"{bbox['minx']['Polygon']},{bbox['miny']['Polygon']},{bbox['maxx']['Polygon']},{bbox['maxy']['Polygon']}"
        return geometry_coords
    
    def print_async_results(self, results) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                print(result)
    
    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        """Main entry point for downloading Digital Coast data"""

        digital_coast_folder = pathlib.Path(outputs) / 'DigitalCoast'

        # tile_gdf.to_file(rF'{OUTPUTS}\tile_gdf.shp', driver='ESRI Shapefile')
        self.process_tile_index(digital_coast_folder, tile_gdf, outputs)
        self.process_intersected_datasets(digital_coast_folder, tile_gdf)
        self.delete_unused_folder(digital_coast_folder)

    def process_intersected_datasets(self, digital_coast_folder, tile_gdf) -> None:
        """Download intersected Digital Coast files"""

        print('Downloading elevation datasets')
        tile_index_shapefiles = digital_coast_folder.rglob('*.shp')
        param_inputs = [[tile_gdf, shp_path] for shp_path in tile_index_shapefiles]
        with ProcessPoolExecutor() as intersected_pool:
            self.print_async_results(intersected_pool.map(self.download_intersected_datasets, param_inputs))

    def process_tile_index(self, digital_coast_folder, tile_gdf, outputs) -> None:
        """Download tile_index shapefiles"""

        print('Download Tile Index shapefiles')
        geometry_coords = self.get_geometry_string(tile_gdf)
        tile_index_links = self.get_available_datasets(geometry_coords, outputs)  # TODO return all object keys
        param_inputs = [[link_dict['link'], link_dict['output_path']] for link_dict in tile_index_links]
        with ProcessPoolExecutor() as tile_index_pool:
            self.print_async_results(tile_index_pool.map(self.download_tile_index, param_inputs))
        self.unzip_all_files(digital_coast_folder)
        # TODO delete *.zip

    def unzip_all_files(self, output_folder) -> None:
        """Unzip all zip files in a folder"""

        for zipped_file in pathlib.Path(output_folder).rglob('*.zip'):
            with zipfile.ZipFile(zipped_file, 'r') as zipped:
                zipped.extractall(str(zipped_file.parents[0]))
            # Delete zip file after extract
            zipped_file.unlink()

    def write_message(self, message, output_folder):
        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')