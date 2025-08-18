"""Class for processing everything for a single tile"""

import boto3
import os
import rasterio
import sys
import geopandas as gpd
import pandas as pd
import pathlib
import multiprocessing as mp
import numpy as np

from hydro_health.helpers import hibase_logging
from datetime import datetime
from botocore.client import Config
from botocore import UNSIGNED
from lxml import etree
from osgeo import gdal
from concurrent.futures import ThreadPoolExecutor

from hydro_health.helpers.tools import get_config_item


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


class BlueTopoEngine:
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

    def create_survey_end_date_tiff(self, tiff_file_path: pathlib.Path) -> None:
        """Create survey end date tiffs from contributor band values in the XML file."""        
        
        with rasterio.open(tiff_file_path) as src:
            contributor_band_values = src.read(3)
            transform = src.transform
            nodata = src.nodata 
            width, height = src.width, src.height  

        xml_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}.tiff.aux.xml'
        tree = etree.parse(xml_file_path)
        root = tree.getroot()

        contributor_band_xml = root.xpath("//PAMRasterBand[Description='Contributor']")
        rows = contributor_band_xml[0].xpath(".//GDALRasterAttributeTable/Row")

        table_data = []
        for row in rows:
            fields = row.xpath(".//F")
            field_values = [field.text for field in fields]

            data = {
                "value": float(field_values[0]),
                "survey_date_end": (
                    datetime.strptime(field_values[17], "%Y-%m-%d").date() 
                    if field_values[17] != "N/A" 
                    else None  
                )
            }
            table_data.append(data)
        attribute_table_df = pd.DataFrame(table_data)

        attribute_table_df['survey_year_end'] = attribute_table_df['survey_date_end'].apply(lambda x: x.year if pd.notna(x) else 0)
        attribute_table_df['survey_year_end'] = attribute_table_df['survey_year_end'].round(2)

        date_mapping = attribute_table_df[['value', 'survey_year_end']].drop_duplicates()
        reclass_matrix = date_mapping.to_numpy()
        reclass_dict = {row[0]: row[1] for row in reclass_matrix}

        reclassified_band = np.vectorize(lambda x: reclass_dict.get(x, nodata))(contributor_band_values)
        reclassified_band = np.where(reclassified_band == None, nodata, reclassified_band)

        survey_date_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}_survey_end_date.tiff'
        with rasterio.open(
            survey_date_file_path,
            "w",
            driver="GTiff",
            count=1,
            width=width,
            height=height,
            dtype=rasterio.float32,
            compress="lzw",
            crs=src.crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(reclassified_band, 1)

    def download_nbs_tile(self, output_folder: str, row: gpd.GeoSeries) -> pathlib.Path:
        """Download all NBS files for a single tile"""

        tile_id = row[0]
        ecoregion_id = row[1]
        nbs_bucket = self.get_bucket()
        output_pathlib = pathlib.Path(output_folder)
        output_tile_path = False
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            current_file = output_pathlib / ecoregion_id / get_config_item('BLUETOPO', 'SUBFOLDER') / obj_summary.key
            # Store the path to the tile, not the xml
            if current_file.suffix == '.tiff':
                if current_file.exists():
                    self.write_message(f'Skipping: {current_file.name}', output_folder)
                    return output_tile_path
                output_tile_path = current_file
            tile_folder = current_file.parents[0]
            self.write_message(f'Downloading: {current_file.name}', output_folder)
            tile_folder.mkdir(parents=True, exist_ok=True)   
            nbs_bucket.download_file(obj_summary.key, current_file)
        return output_tile_path

    def get_bucket(self) -> boto3.resource:
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

    def multiband_to_singleband(self, tiff_file_path: pathlib.Path, band: int) -> None:
        """Convert multiband BlueTopo raster into a singleband file"""

        band_name_lookup = {
            1: '',
            2: '_unc'
        }
        # temporarily rename file to output singleband with original name
        output_name = str(tiff_file_path.name).replace('_mb', band_name_lookup[band])
        singleband_tile_name = tiff_file_path.parents[0] / output_name

        gdal.Translate(
            singleband_tile_name,
            tiff_file_path,
            bandList=[band],
            creationOptions=["COMPRESS:DEFLATE", "TILED:NO"],
            callback=gdal.TermProgress_nocb
        )

    def rename_multiband(self, tiff_file_path: pathlib.Path) -> pathlib.Path:
        """Update file name for singleband conversion"""

        new_name = str(tiff_file_path).replace('.tiff', '_mb.tiff')
        mb_tiff_file = tiff_file_path.replace(pathlib.Path(new_name))
        return mb_tiff_file

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(result, output_folder)

    def process_tile(self, param_inputs: list[list]) -> None:
        """Handle processing of a single tile"""

        output_folder, row = param_inputs
        tiff_file_path = self.download_nbs_tile(output_folder, row)
        if tiff_file_path:
            self.create_survey_end_date_tiff(tiff_file_path)
            mb_tiff_file = self.rename_multiband(tiff_file_path)
            self.multiband_to_singleband(mb_tiff_file, band=1)
            self.multiband_to_singleband(mb_tiff_file, band=2)
            mb_tiff_file.unlink() # delete the original multiband file
            self.set_ground_to_nodata(tiff_file_path)
            self.create_slope(tiff_file_path)
            self.create_rugosity(tiff_file_path)

    def run(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False) -> None:
        print('Downloading BlueTopo Datasets')
        param_inputs = [[outputs, row] for _, row in tile_gdf.iterrows() if isinstance(row[1], str)]  # rows out of ER will be nan
        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as intersected_pool:
            self.print_async_results(intersected_pool.map(self.process_tile, param_inputs), outputs)

        # log all tiles using tile_gdf
        tiles = list(tile_gdf['tile'])
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test')  # TODO update to prod hibase

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value"""

        raster_ds = gdal.Open(tiff_file_path, gdal.GA_Update)
        no_data = -999999
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array < 0, raster_array, no_data)
        raster_ds.GetRasterBand(1).WriteArray(meters_array)
        raster_ds.GetRasterBand(1).SetNoDataValue(no_data)  # took forever to find this gem
        raster_ds = None

    def write_message(self, message: str, output_folder: str|pathlib.Path) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')
