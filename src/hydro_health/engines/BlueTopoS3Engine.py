"""Class for processing everything for a single tile"""

import tempfile
import boto3
import os
import pathlib
import os
import rasterio

import geopandas as gpd
import pandas as pd
import pathlib
import numpy as np
import os
import sys

from multiprocessing import set_executable
from hydro_health.helpers import hibase_logging
from datetime import datetime
from botocore.client import Config
from botocore import UNSIGNED
from lxml import etree
from osgeo import gdal

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


def _process_tile(param_inputs: list[list]) -> None:
    """
    Static function that handles processing of a single tile.
    Refactored for S3-to-S3 workflow using ephemeral storage.
    """

    # outputs folder available for logging
    s3_output_bucket, outputs, s3_prefix, tile_id, ecoregion_id = param_inputs

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        engine = BlueTopoS3Engine()

        # 1. DOWNLOAD (Updated to save to temp_path)
        tiff_file_path = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id)
        
        if tiff_file_path:
            # 2. PROCESS (Your existing logic runs locally on the temp drive)
            # No changes needed to these methods!
            engine.create_survey_end_date_tiff(tiff_file_path)
            mb_tiff_file = engine.rename_multiband(tiff_file_path)
            engine.multiband_to_singleband(mb_tiff_file, band=1)
            engine.multiband_to_singleband(mb_tiff_file, band=2)
            mb_tiff_file.unlink() 
            engine.set_ground_to_nodata(tiff_file_path)
            engine.create_slope(tiff_file_path)
            engine.create_rugosity(tiff_file_path)

            # 3. UPLOAD (Sync the temp folder results back to S3)
            # We assume anything left in the temp folder is a result we want.
            engine.upload_directory_to_s3(
                local_dir=temp_path, 
                bucket_name=s3_output_bucket, 
                s3_prefix=f"{s3_prefix}/{ecoregion_id}/{tile_id}"
            )

class BlueTopoS3Engine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self):
        super().__init__()

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

    def download_nbs_tile(self, temp_folder: pathlib.Path, tile_id: str, ecoregion_id: str) -> pathlib.Path:
        """
        Modified to accept a pathlib.Path object for output_folder 
        and download to that specific local path.
        """

        nbs_bucket = self.get_bucket()
        output_tile_path = None
        
        # TODO Determine the destination subfolder structure inside the temp dir
        # e.g., /tmp/xyz/US4NC/BlueTopo/...
        
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            # We flatten the path structure slightly for the temp dir processing
            # or keep it if your processing logic relies on specific parent folder structures.
            file_name = pathlib.Path(obj_summary.key).name
            local_file = temp_folder / file_name
            
            if local_file.suffix == '.tiff':
                output_tile_path = local_file
            
            # Download to the temp directory
            # No need to check if exists, because temp dir is always empty on start
            nbs_bucket.download_file(obj_summary.key, str(local_file))
            
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

    def run(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False) -> None:
        print('Downloading BlueTopo Datasets')

        output_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        self.setup_dask()
        param_inputs = [[outputs, output_bucket, row[0], row[1]] for _, row in tile_gdf.iterrows() if isinstance(row[1], str)]  # rows out of ER will be nan
        future_tiles = self.client.map(_process_tile, param_inputs)
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)
        self.close_dask()

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

    def upload_directory_to_s3(self, local_dir: pathlib.Path, bucket_name: str, s3_prefix: str) -> None:
        """
        Walks the temporary directory and uploads all generated files to S3.
        """
        s3_client = boto3.client('s3') # Use standard client for uploads
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Construct the S3 key
                # This puts the file at: s3://bucket/prefix/filename.tiff
                s3_key = f"{s3_prefix}/{file}"
                
                print(f"Uploading {file} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(local_path, bucket_name, s3_key)
