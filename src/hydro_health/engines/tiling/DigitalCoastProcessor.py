"""Class for obtaining all available files"""

import boto3
import json
import os
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

    def download_bulk_data(self, download_link) -> None:
        if 'noaa-nos-coastal-lidar-pds' in download_link:
            _, data_file = download_link.replace('/index.html', '').split('.com')
            lidar_bucket = self.get_bucket()
            print('file:', data_file)
            for obj_summary in lidar_bucket.objects.filter(Prefix=f"{data_file[1:]}"):
                if 'tileindex' in obj_summary.key:
                    print(obj_summary)
                    # TODO each bucket folder has a tileindex zip file containing a shapefile, or a geopackage
                    # 1. download shapefile or geopackage
                    # 2. spatial query against drawn polygon
                    # 3. loop through bucket again and download intersected tiles

    def get_available_datasets(self, geometry_coords: str, outputs) -> None:
        """Query NOWCoast REST API for available datasets"""

        base_url = 'https://maps.coast.noaa.gov/arcgis/rest/services/DAV/ElevationFootprints/MapServer/0/query?returnGeometry=false&f=json&where=1%3D1&outfields=%2A&spatialRel=esriSpatialRelIntersects&state=FL&geometry='
        elevation_footprints_url = base_url + geometry_coords
        print(elevation_footprints_url)
        datasets_json = requests.get(elevation_footprints_url).json()

        output_json = pathlib.Path(outputs) / 'json_log.txt'
        if os.path.exists(output_json):
            output_json.unlink()
        with open(output_json, 'a') as writer:
            for feature in datasets_json['features']:
                # TODO maybe store full JSON in a text in each data folder
                writer.write(json.dumps(feature, indent=4) + '\n\n')
                folder_name = ''.join(char for char in feature['attributes']['Name'] if char.isalnum())  # strips out illegal characters
                print('folder:', folder_name)
                external_data_json = json.loads(feature['attributes']['ExternalProviderLink'])
                # Look for Bulk Download
                for external_data in external_data_json['links']:
                    if external_data['label'] == 'Bulk Download':
                        download_link = external_data['link']
                        self.download_bulk_data(download_link)
                writer.write(json.dumps(external_data_json['links'], indent=4) + '\n\n')

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
        print(geometry_coords)
        return geometry_coords

    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        geometry_coords = self.get_geometry_string(tile_gdf)

        # 1. build URL and request JSON output
        self.get_available_datasets(geometry_coords, outputs)


        # with self.get_pool() as process_pool:
        #     results = [process_pool.apply_async(self.process_tile, [outputs, row]) for _, row in tile_gdf.iterrows()]
        #     for result in results:
        #         result.get()

        # # log all tiles using tile_gdf
        # tiles = list(tile_gdf['tile'])
        # record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        # hibase_logging.send_record(record, table='bluetopo_test')  # TODO update to prod hibase