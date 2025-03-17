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

from hydro_health.helpers import hibase_logging
from botocore.client import Config
from botocore import UNSIGNED


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class DigitalCoastProcessor:
    """Class for parallel processing all BlueTopo tiles for a region"""

    def get_available_datasets(self, geometry_coords: str) -> None:
        """Query NOWCoast REST API for available datasets"""

        base_url = 'https://maps.coast.noaa.gov/arcgis/rest/services/DAV/ElevationFootprints/MapServer/0/query?returnGeometry=false&f=json&where=1%3D1&outfields=%2A&spatialRel=esriSpatialRelIntersects&geometry='
        elevation_footprints_url = base_url + geometry_coords
        datasets_json = requests.get(elevation_footprints_url).json()
        for feature in datasets_json['features']:
            print(feature)
            break
            external_data_json = json.loads(feature['attributes']['ExternalProviderLink'])
            for link in external_data_json['links']:
                print(link['label'], link['link'])

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

    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        geometry_coords = self.get_geometry_string(tile_gdf)

        # 1. build URL and request JSON output
        self.get_available_datasets(geometry_coords)


        # with self.get_pool() as process_pool:
        #     results = [process_pool.apply_async(self.process_tile, [outputs, row]) for _, row in tile_gdf.iterrows()]
        #     for result in results:
        #         result.get()

        # # log all tiles using tile_gdf
        # tiles = list(tile_gdf['tile'])
        # record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        # hibase_logging.send_record(record, table='bluetopo_test')  # TODO update to prod hibase