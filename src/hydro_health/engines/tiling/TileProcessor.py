"""Class for processing everything for a single tile"""

import boto3
import os
import sys
import geopandas as gpd
import pathlib
import multiprocessing as mp
import numpy as np

from hydro_health.helpers import hibase_logging
from botocore.client import Config
from botocore import UNSIGNED
from osgeo import gdal


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class TileProcessor:
    def download_nbs_tile(self, output_folder: str, tile_id: str):
        """Download all NBS files for a single tile"""

        nbs_bucket = self.get_bucket()
        output_pathlib = pathlib.Path(output_folder)
        tiff_file_path = False
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            output_tile_path = output_pathlib / obj_summary.key
            # Store the path to the tile, not the xml
            if output_tile_path.suffix == '.tiff':
                tiff_file_path = output_tile_path

            tile_folder = output_tile_path.parents[0]
            if os.path.exists(output_tile_path):
                self.write_message(f'Skipping: {output_tile_path.name}', output_folder)
                continue
            else:
                self.write_message(f'Downloading: {output_tile_path.name}', output_folder)
            tile_folder.mkdir(parents=True, exist_ok=True)   
            nbs_bucket.download_file(obj_summary.key, output_tile_path)

        return tiff_file_path

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

    def multiband_to_singleband(self, tiff_file_path: pathlib.Path) -> None:
        """Convert multiband BlueTopo raster into a singleband file"""
        
        multiband_tile_name = tiff_file_path.parents[0] / tiff_file_path.name
        output_name = str(tiff_file_path.name).replace('_mb', '')
        singleband_tile_name = tiff_file_path.parents[0] / output_name
        gdal.Translate(
            singleband_tile_name,
            multiband_tile_name,
            bandList=[1],
            creationOptions=["COMPRESS:DEFLATE", "TILED:NO"],
            callback=gdal.TermProgress_nocb
        )

    def rename_multiband(self, tiff_file_path) -> pathlib.Path:
        """Update file name for singleband conversion"""

        new_name = str(tiff_file_path).replace('.tiff', '_mb.tiff')
        mb_tiff_file = tiff_file_path.replace(pathlib.Path(new_name))
        return mb_tiff_file

    def process_tile(self, output_folder: str, index: int, row: gpd.GeoSeries):
        """Handle processing of a single tile"""

        tile_id = row[0]
        tiff_file_path = self.download_nbs_tile(output_folder, tile_id)
        if tiff_file_path:
            mb_tiff_file = self.rename_multiband(tiff_file_path)
            self.multiband_to_singleband(mb_tiff_file)
            mb_tiff_file.unlink()
            self.set_ground_to_nodata(tiff_file_path)

    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        with self.get_pool() as process_pool:
            results = [process_pool.apply_async(self.process_tile, [outputs, index, row]) for index, row in tile_gdf.iterrows()]
            for result in results:
                # try to add logging here
                # self.write_message(f'testing: {str(result)}', outputs)
                result.get()

        # log all tiles using tile_gdf
        tiles = list(tile_gdf['tile'])
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test')  # TODO update to prod hibase

    def write_message(self, message, output_folder):
        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value"""
        
        raster_ds = gdal.Open(tiff_file_path, gdal.GA_Update)
        no_data = -999999
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array < 0, raster_array, no_data)
        raster_ds.GetRasterBand(1).WriteArray(meters_array)
        raster_ds = None