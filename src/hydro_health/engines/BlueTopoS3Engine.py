"""Class for processing everything for a single tile"""

import tempfile
import boto3
import os
import pathlib
import sys
import math
import rasterio

from rasterio.session import AWSSession
import geopandas as gpd
import pandas as pd
import numpy as np

from multiprocessing import set_executable
from hydro_health.helpers import hibase_logging
from datetime import datetime, date
from botocore.client import Config
from botocore import UNSIGNED
from botocore.exceptions import ClientError
from lxml import etree
from osgeo import gdal

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine, supersession, catzoc, decay

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


def _process_tile(param_inputs: list[list]) -> None:
    """
    Static function that handles processing of a single tile.
    Refactored for S3-to-S3 workflow using ephemeral storage.
    """

    # outputs folder available for logging
    param_lookup, s3_output_bucket, tile_id, ecoregion_id, output_prefix = param_inputs
    
    print(f"[{tile_id}] Initiating processing for ecoregion {ecoregion_id}...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        engine = BlueTopoS3Engine(param_lookup)

        # Pass s3_output_bucket and output_prefix so we can check if it already exists
        tiff_file_path = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id, s3_output_bucket, output_prefix)
        
        # Will be False if it was already downloaded and skipped
        if tiff_file_path:
            print(f"[{tile_id}] Download complete. Resampling and reprojecting...")
            # First, resample and reproject to 100m before running downstream tasks
            engine.resample_and_reproject(tiff_file_path)

            print(f"[{tile_id}] Creating survey end date TIFF...")
            engine.create_survey_end_date_tiff(tiff_file_path)

            print(f"[{tile_id}] Creating CATZOC all...")
            engine.create_catzoc_all(tiff_file_path)
            # engine.create_catzoc_latest(tiff_file_path)

            print(f"[{tile_id}] Extracting singlebands...")
            mb_tiff_file = engine.rename_multiband(tiff_file_path)
            engine.multiband_to_singleband(mb_tiff_file, band=1)
            engine.multiband_to_singleband(mb_tiff_file, band=2)
            mb_tiff_file.unlink() 

            print(f"[{tile_id}] Setting ground to nodata...")
            engine.set_ground_to_nodata(tiff_file_path)

            print(f"[{tile_id}] Creating slope...")
            engine.create_slope(tiff_file_path)
            # engine.create_rugosity(tiff_file_path)

            print(f"[{tile_id}] Finalizing COG format...")
            engine.finalize_cog(tiff_file_path)      

            print(f"[{tile_id}] Uploading final tiles to S3...")
            engine.upload_current_tiles_to_s3(tiff_file_path.parents[0], s3_output_bucket, ecoregion_id, output_prefix)
            print(f"[{tile_id}] Processing successfully completed.")
        else:
            print(f"[{tile_id}] Processing skipped (tile already exists).")


class BlueTopoS3Engine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self, param_lookup: dict[dict]):
        super().__init__()
        self.param_lookup = param_lookup
        # Set target resolution and CRS
        self.target_resolution = 100.0
        self.target_crs = "EPSG:32617"

    def resample_and_reproject(self, tiff_path: pathlib.Path) -> None:
        """Warp the downloaded raster to target resolution and CRS."""
        
        output_folder = self.param_lookup['output_directory'].valueAsText
        msg = f"Resampling {tiff_path.name} to {self.target_resolution}m and {self.target_crs}..."
        print(msg)
        self.write_message(msg, output_folder)
        
        temp_tiff = tiff_path.parent / f"warped_{tiff_path.name}"
        
        # We MUST use NearestNeighbor to preserve the categorical integer IDs in the Contributor band (Band 3)
        gdal.Warp(
            str(temp_tiff),
            str(tiff_path),
            xRes=self.target_resolution,
            yRes=self.target_resolution,
            dstSRS=self.target_crs,
            resampleAlg=gdal.GRA_NearestNeighbour,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
        )
        
        if temp_tiff.exists():
            tiff_path.unlink()
            temp_tiff.rename(tiff_path)

    def create_catzoc_all(self, tiff_file_path: pathlib.Path) -> None:
        """
        Generate a CATZOC score raster of unique values for each survey area
        """

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
        rat_node = root.find(".//GDALRasterAttributeTable")
        field_names = [f.find('Name').text for f in rat_node.findall('FieldDefn')]

        table_data = []
        for row in rows:  # Can sort rows and then use the last index range or no loop
            row_data = {field_names[i]: f_val.text for i, f_val in enumerate(row.findall('F'))}
            data = {
                "value": float(row_data.get('value')),
                'start_date': (
                    datetime.strptime(row_data.get('survey_date_start'), "%Y-%m-%d").date() 
                    if row_data.get('survey_date_start') != "N/A" 
                    else None
                ),
                "end_date": (
                    datetime.strptime(row_data.get('survey_date_end'), "%Y-%m-%d").date() 
                    if row_data.get('survey_date_end') != "N/A" 
                    else None
                ),
                'from_filename': row_data.get('source_survey_id'),
                'feat_detect': bool(int(row_data.get('significant_features', 0))),
                'feat_least_depth': bool(int(row_data.get('feature_least_depth', 0))),
                'complete_coverage': bool(int(row_data.get('bathy_coverage', 0))),
                'horiz_uncert_fixed': float(row_data.get('horizontal_uncert_fixed', 0)),
                'horiz_uncert_vari': float(row_data.get('horizontal_uncert_var', 0)),
                'vert_uncert_fixed': float(row_data.get('vertical_uncert_fixed', 0)),
                'vert_uncert_vari': float(row_data.get('vertical_uncert_var', 0)),
                'interpolated': ".interpolated" in row_data.get('source_survey_id', '').lower()
            }
            if data['start_date'] or data['end_date']:
                table_data.append(data)

        # Add CATZOC necessary columns
        for meta in table_data:
            ss_score = supersession(meta)
            meta['supersession_score'] = ss_score
            meta['catzoc'] = catzoc(meta)
            today = date.today()
            
            try:
                meta['catzoc_decay'] = decay(meta, today)
            except ValueError as e:
                if "Decay Score less than 1" in str(e):
                    meta['catzoc_decay'] = 1.0 
                else:
                    raise e

        attribute_table_df = pd.DataFrame(table_data)

        decay_mapping = attribute_table_df[['value', 'catzoc_decay']].drop_duplicates()
        reclass_matrix = decay_mapping.to_numpy()
        reclass_dict = {row[0]: row[1] for row in reclass_matrix}

        reclassified_band = np.vectorize(lambda x: reclass_dict.get(x, nodata))(contributor_band_values)
        reclassified_band = np.where(reclassified_band == None, nodata, reclassified_band)

        survey_date_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}_catzoc_decay_all.tiff'
        with rasterio.open(
            survey_date_file_path,
            "w",
            driver="GTiff",
            count=1,
            width=width,
            height=height,
            dtype=rasterio.float32,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            crs=src.crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(reclassified_band, 1)
            
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_catzoc_latest(self, tiff_file_path: pathlib.Path) -> None:
        """Generate a CATZOC score raster using the most recent survey date"""

        with rasterio.open(tiff_file_path) as src:
            contributor_band_values = src.read(3)
            transform = src.transform
            nodata = src.nodata 
            width, height = src.width, src.height 
            crs = src.crs

        xml_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}.tiff.aux.xml'
        tree = etree.parse(xml_file_path)
        root = tree.getroot()

        contributor_band_xml = root.xpath("//PAMRasterBand[Description='Contributor']")
        rows = contributor_band_xml[0].xpath(".//GDALRasterAttributeTable/Row")
        rat_node = root.find(".//GDALRasterAttributeTable")
        field_names = [f.find('Name').text for f in rat_node.findall('FieldDefn')]

        all_surveys = []
        for row in rows:
            row_dict = {field_names[i]: f_val.text for i, f_val in enumerate(row.findall('F'))}
            
            end_date_str = row_dict.get('survey_date_end')
            end_date = (
                datetime.strptime(end_date_str, "%Y-%m-%d").date() 
                if end_date_str and end_date_str != "N/A" 
                else date.min
            )

            meta = {
                "end_date": end_date,
                "feat_detect": bool(int(row_dict.get('significant_features', 0))),
                "feat_least_depth": bool(int(row_dict.get('feature_least_depth', 0))),
                "complete_coverage": bool(int(row_dict.get('bathy_coverage', 0))),
                "horiz_uncert_fixed": float(row_dict.get('horizontal_uncert_fixed', 0)),
                "horiz_uncert_vari": float(row_dict.get('horizontal_uncert_var', 0)),
                "vert_uncert_fixed": float(row_dict.get('vertical_uncert_fixed', 0)),
                "vert_uncert_vari": float(row_dict.get('vertical_uncert_var', 0)),
                'interpolated': ".interpolated" in row_dict.get('source_survey_id', '').lower()
            }
            all_surveys.append(meta)

        measured_surveys = [s for s in all_surveys if not s.get('interpolated')]  
        surveys_to_rank = measured_surveys if measured_surveys else all_surveys
        most_recent_survey = max(surveys_to_rank, key=lambda x: x['end_date'])

        today = date.today()
        most_recent_survey['supersession_score'] = supersession(most_recent_survey)
        most_recent_survey['catzoc'] = catzoc(most_recent_survey)
        
        try:
            most_recent_survey['catzoc_decay'] = decay(most_recent_survey, today)
        except ValueError as e:
            if "Decay Score less than 1" in str(e):
                most_recent_survey['catzoc_decay'] = 1.0
            else:
                raise e

        if nodata is not None and np.isnan(nodata):
            is_nodata = np.isnan(contributor_band_values)
        else:
            is_nodata = (contributor_band_values == nodata)

        reclassified_band = np.where(is_nodata, nodata, most_recent_survey['catzoc_decay']).astype(np.float32)

        survey_date_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}_catzoc_decay_latest.tiff'
        with rasterio.open(
            survey_date_file_path,
            "w",
            driver="GTiff",
            count=1,
            width=width,
            height=height,
            dtype=rasterio.float32,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            crs=src.crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(reclassified_band, 1)
            
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_rugosity(self, tiff_file_path: pathlib.Path) -> None:
        """Generate a rugosity/roughness raster from the DEM"""

        rugosity_name = str(tiff_file_path.stem) + '_rugosity.tiff'
        rugosity_file_path = tiff_file_path.parents[0] / rugosity_name
        gdal.DEMProcessing(str(rugosity_file_path), str(tiff_file_path), 'Roughness')

    def create_slope(self, tiff_file_path: pathlib.Path) -> None:
        """Generate a slope raster from the DEM"""

        slope_name = str(tiff_file_path.stem) + '_slope.tiff'
        slope_file_path = tiff_file_path.parents[0] / slope_name
        gdal.DEMProcessing(str(slope_file_path), str(tiff_file_path), 'slope')

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

    def download_nbs_tile(self, temp_folder: pathlib.Path, tile_id: str, ecoregion_id: str, s3_output_bucket: str, output_prefix: str|bool) -> pathlib.Path | bool:
        """
        Modified to check S3 bucket to skip already processed tiles.
        Also validates that existing tiles are properly resampled to 100m.
        """

        nbs_bucket = self.get_bucket()
        output_tile_path = False
        output_folder = self.param_lookup['output_directory'].valueAsText
        
        s3_client = boto3.client('s3')

        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            current_file = temp_folder / ecoregion_id / get_config_item('BLUETOPO', 'SUBFOLDER') / obj_summary.key
            
            if current_file.suffix == '.tiff':
                ecoregion_index = current_file.parts.index(ecoregion_id)
                s3_subpath = pathlib.Path(*current_file.parts[ecoregion_index:])
                
                s3_key = str(s3_subpath).replace("\\", "/") 
                if output_prefix:
                    s3_key = f'{output_prefix}/{s3_key}'
                    
                try:
                    # Retrieve the object metadata from S3
                    head_response = s3_client.head_object(Bucket=s3_output_bucket, Key=s3_key)
                    
                    is_valid_resolution = False
                    
                    # Method 1: Check the custom tags applied during upload (Fastest)
                    s3_meta = head_response.get('Metadata', {})
                    if s3_meta.get('resolution') == str(self.target_resolution) and s3_meta.get('crs') == self.target_crs:
                        is_valid_resolution = True
                    else:
                        # Method 2: Check deep metadata if tags were missing/old via rasterio
                        s3_uri = f"s3://{s3_output_bucket}/{s3_key}"
                        try:
                            with rasterio.Env(AWSSession(boto3.Session())):
                                with rasterio.open(s3_uri) as src:
                                    res_x, _ = src.res
                                    # Allow for slight floating point precision variances
                                    if abs(res_x - self.target_resolution) < 1.0:
                                        is_valid_resolution = True
                        except Exception:
                            pass # We will fall back to re-downloading below

                    if is_valid_resolution:
                        skip_msg = f"Skipping already downloaded {self.target_resolution}m tile: {tile_id} for ecoregion: {ecoregion_id}"
                        print(skip_msg)
                        self.write_message(skip_msg, output_folder)
                        return False 
                    else:
                        redownload_msg = f"Tile {tile_id} exists but is not {self.target_resolution}m. Re-downloading and resampling..."
                        print(redownload_msg)
                        self.write_message(redownload_msg, output_folder)

                except ClientError as e:
                    # 404 indicates it doesn't exist yet, proceed with download
                    if e.response['Error']['Code'] == "404":
                        pass
                    else:
                        raise 
                
                print(f"Downloading and processing tile: {tile_id} for ecoregion: {ecoregion_id}")
                output_tile_path = current_file
                
            tile_folder = current_file.parents[0]
            self.write_message(f'Downloading: {current_file.name}', output_folder)
            tile_folder.mkdir(parents=True, exist_ok=True)   

            nbs_bucket.download_file(obj_summary.key, str(current_file))

        return output_tile_path
    
    def finalize_cog(self, tiff_path: pathlib.Path) -> None:
        """The final pass to ensure perfect COG layout and overviews."""

        temp_cog = tiff_path.parent / f"temp_{tiff_path.name}"
        
        ds = gdal.Open(str(tiff_path), gdal.GA_Update)
        if ds is not None:
            ds.BuildOverviews("BILINEAR", [2, 4, 8, 16])
            ds = None

        gdal.Translate(
            str(temp_cog),
            str(tiff_path),
            creationOptions=[
                "COMPRESS=DEFLATE",
                "PREDICTOR=3",
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512",
                "COPY_SRC_OVERVIEWS=YES"
            ]
        )
        
        if temp_cog.exists():
            tiff_path.unlink()
            temp_cog.rename(tiff_path)

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
        output_name = str(tiff_file_path.name).replace('_mb', band_name_lookup[band])
        singleband_tile_name = tiff_file_path.parents[0] / output_name

        gdal.Translate(
            str(singleband_tile_name),
            str(tiff_file_path),
            bandList=[band],
            creationOptions=["COMPRESS=DEFLATE"]
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

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str|bool=False) -> None:
        print('Downloading BlueTopo Datasets')

        output_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        self.setup_dask(self.param_lookup['env'], threads_per_worker=1)
        param_inputs = [[self.param_lookup, output_bucket, row[0], row[1], output_prefix] for _, row in tile_gdf.iterrows() if isinstance(row[1], str)] 
        
        print(f"Submitting {len(param_inputs)} tiles to Dask workers...")
        future_tiles = self.client.map(_process_tile, param_inputs)
        
        print("Waiting for all Dask workers to complete...")
        tile_results = self.client.gather(future_tiles)
        print("All Dask workers finished successfully.")
        
        self.print_async_results(tile_results, self.param_lookup['output_directory'].valueAsText)
        self.close_dask()
        for ecoregion in self.param_lookup['eco_regions'].value:
            s3_path = f"{output_prefix}/{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo" if output_prefix else f"{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            self.write_run_manifest(s3_path)

        # log all tiles using tile_gdf
        tiles = list(tile_gdf['tile'])
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test') 

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value"""

        raster_ds = gdal.Open(str(tiff_file_path), gdal.GA_Update)
        no_data = -999999
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array < 0, raster_array, no_data)
        raster_ds.GetRasterBand(1).WriteArray(meters_array)
        raster_ds.GetRasterBand(1).SetNoDataValue(no_data)  
        raster_ds = None

    def upload_current_tiles_to_s3(self, tile_folder: pathlib.Path, bucket_name: str, ecoregion_id: str, output_prefix: str|bool) -> None:
        """Upload all tiff files to s3 for current tile with Resolution metadata"""

        s3_client = boto3.client('s3')
        
        # Tag objects with the target resolution / CRS so we can easily verify them later
        s3_metadata_args = {
            'Metadata': {
                'resolution': str(self.target_resolution),
                'crs': self.target_crs
            }
        }
        
        for tiff_file in tile_folder.glob('*'):
            ecoregion_index = tiff_file.parts.index(ecoregion_id)
            s3_subpath = pathlib.Path(*tiff_file.parts[ecoregion_index:])
            s3_path = s3_subpath if not output_prefix else f'{output_prefix}/{s3_subpath}'
            
            s3_path_formatted = str(s3_path).replace("\\", "/")
            
            self.write_message(f'Uploading {tiff_file} to s3://{bucket_name}/{s3_path_formatted}', self.param_lookup['output_directory'].valueAsText)
            s3_client.upload_file(str(tiff_file), bucket_name, f'{str(s3_path_formatted)}', ExtraArgs=s3_metadata_args)