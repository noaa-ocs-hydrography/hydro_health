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
from concurrent.futures import ProcessPoolExecutor


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class BlueTopoProcessor:
    """Class for parallel processing all BlueTopo tiles for a region"""

    def create_rugosity(self, tiff_file_path: pathlib.Path) -> None:
        """Generate a rugosity/roughness raster from the DEM"""

        rugosity_name = str(tiff_file_path.stem) + '_rugosity.tiff'
        rugosity_file_path = tiff_file_path.parents[0] / rugosity_name
        gdal.DEMProcessing(rugosity_file_path, tiff_file_path, 'Roughness')

    def create_slope(self, tiff_file_path: pathlib.Path) -> None:
        """Generate a slope raster from the DEM"""

        slope_name = str(tiff_file_path.stem) + '_slope.tiff'
        slope_file_path = tiff_file_path.parents[0] / slope_name
        gdal.DEMProcessing(slope_file_path, tiff_file_path, 'slope')

    def download_nbs_tile(self, output_folder: str, row: gpd.GeoSeries):
        """Download all NBS files for a single tile"""

        tile_id = row[0]
        ecoregion_id = row[1]
        nbs_bucket = self.get_bucket()
        output_pathlib = pathlib.Path(output_folder)
        tiff_file_path = False
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            output_tile_path = output_pathlib / ecoregion_id / obj_summary.key
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

    def multiband_to_singleband(self, tiff_file_path: pathlib.Path) -> None:
        """Convert multiband BlueTopo raster into a singleband file"""

        multiband_tile_name = tiff_file_path.parents[0] / tiff_file_path.name
        # temporarily rename file to output singleband with original name
        output_name = str(tiff_file_path.name).replace('_mb', '')
        singleband_tile_name = tiff_file_path.parents[0] / output_name

        gdal.Translate(
            singleband_tile_name,
            multiband_tile_name,
            bandList=[1],
            creationOptions=["COMPRESS:DEFLATE", "TILED:NO"],
            callback=gdal.TermProgress_nocb
        )
        multiband_tile_name.unlink()

    def rename_multiband(self, tiff_file_path) -> pathlib.Path:
        """Update file name for singleband conversion"""

        new_name = str(tiff_file_path).replace('.tiff', '_mb.tiff')
        mb_tiff_file = tiff_file_path.replace(pathlib.Path(new_name))
        return mb_tiff_file

    def print_async_results(self, results, output_folder) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(result, output_folder)

    def process_tile(self, param_inputs: list[list]) -> None:
        """Handle processing of a single tile"""

        output_folder, row = param_inputs
        tiff_file_path = self.download_nbs_tile(output_folder, row)
        if tiff_file_path:
            mb_tiff_file = self.rename_multiband(tiff_file_path)
            self.multiband_to_singleband(mb_tiff_file)
            self.set_ground_to_nodata(tiff_file_path)
            self.create_slope(tiff_file_path)
            self.create_rugosity(tiff_file_path)
        return f'- {row["EcoRegion"]}'

    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        param_inputs = [[outputs, row] for _, row in tile_gdf.iterrows() if isinstance(row[1], str)]  # rows out of ER will be nan
        with ProcessPoolExecutor(int(os.cpu_count()/2)) as intersected_pool:
            self.print_async_results(intersected_pool.map(self.process_tile, param_inputs), outputs)

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
        raster_ds.GetRasterBand(1).SetNoDataValue(no_data)  # took forever to find this gem
        raster_ds = None
