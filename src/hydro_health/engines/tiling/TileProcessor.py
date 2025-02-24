"""Class for processing everything for a single tile"""

import os
import sys
import geopandas as gpd
import pathlib
import multiprocessing as mp
import boto3

from hydro_health.helpers import hibase_logging
from botocore.client import Config
from botocore import UNSIGNED
from rasterio.features import shapes

mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class TileProcessor:
    def download_nbs_tile(self, output_folder: str, tile_id: str):
        """Download all NBS files for a single tile"""

        nbs_bucket = self.get_bucket()
        output_pathlib = pathlib.Path(output_folder)
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            output_tile_path = output_pathlib / obj_summary.key
            tile_folder = output_tile_path.parents[0]
            if os.path.exists(output_tile_path):
                self.write_message(f'Skipping: {output_tile_path.name}', output_folder)
                continue
            else:
                self.write_message(f'Downloading: {output_tile_path.name}', output_folder)
            tile_folder.mkdir(parents=True, exist_ok=True)   
            nbs_bucket.download_file(obj_summary.key, output_tile_path)
        
        return tile_folder

    def get_bucket(self):
        """Connect to anonymous OCS S3 Bucket"""

        bucket = "noaa-ocs-nationalbathymetry-pds"
        creds = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "config": Config(signature_version=UNSIGNED),
        }
        s3 = boto3.resource('s3', **creds)
        nbs_bucket = s3.Bucket(bucket)
        return nbs_bucket

    def get_pool(self, processes=int(mp.cpu_count() / 2)):
        """Obtain a multiprocessing Pool"""

        return mp.Pool(processes=processes)
    
    def write_message(self, message, output_folder):
        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')
    
    def process_tile(self, output_folder: str, index: int, row: gpd.GeoSeries):
        """Handle processing of a single tile"""

        tile_id = row[0]
        tile_folder = self.download_nbs_tile(output_folder, tile_id)
        # add general logging

        # load tile
    
    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        with self.get_pool() as process_pool:
            results = [process_pool.apply_async(self.process_tile, [outputs, index, row]) for index, row in tile_gdf.iterrows()]
            for result in results:
                result.get()
        
        # log all tiles using tile_gdf
        tiles = list(tile_gdf['Tilename'])
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test')  # TODO update to prod hibase
