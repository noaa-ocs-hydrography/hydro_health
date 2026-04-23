"""Class for processing everything for a single tile"""

import tempfile
import boto3
import os
import pathlib
import os
import rasterio
import os
import sys
import pathlib
import math

import geopandas as gpd
import pandas as pd
import numpy as np

from multiprocessing import set_executable
from hydro_health.helpers import hibase_logging
from datetime import datetime, date
from botocore.client import Config
from botocore import UNSIGNED
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
    param_lookup, s3_output_bucket, tile_id, ecoregion_id = param_inputs
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        engine = BlueTopoS3Engine(param_lookup)

        tiff_file_path = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id)
        if tiff_file_path:
            engine.create_survey_end_date_tiff(tiff_file_path)
            # engine.create_catzoc_all(tiff_file_path)
            # engine.create_catzoc_latest(tiff_file_path)
            mb_tiff_file = engine.rename_multiband(tiff_file_path)
            engine.multiband_to_singleband(mb_tiff_file, band=1)
            engine.multiband_to_singleband(mb_tiff_file, band=2)
            mb_tiff_file.unlink() 
            engine.set_ground_to_nodata(tiff_file_path)
            engine.create_slope(tiff_file_path)
            engine.create_rugosity(tiff_file_path)
            engine.finalize_cog(tiff_file_path)      

            engine.upload_current_tiles_to_s3(tiff_file_path.parents[0], s3_output_bucket, ecoregion_id)


class BlueTopoS3Engine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self, param_lookup: dict[dict]):
        super().__init__()
        self.param_lookup = param_lookup

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
            # self.write_message(f"dates: {meta['start_date']}, {meta['end_date']}", self.param_lookup['output_directory'].valueAsText)
            ss_score = supersession(meta)
            meta['supersession_score'] = ss_score
            meta['catzoc'] = catzoc(meta)
            today = date.today()
            meta['catzoc_decay'] = decay(meta, today)

        attribute_table_df = pd.DataFrame(table_data)

        decay_mapping = attribute_table_df[['value', 'catzoc_decay']].drop_duplicates()
        reclass_matrix = decay_mapping.to_numpy()
        reclass_dict = {row[0]: row[1] for row in reclass_matrix}

        reclassified_band = np.vectorize(lambda x: reclass_dict.get(x, nodata))(contributor_band_values)
        reclassified_band = np.where(reclassified_band == None, nodata, reclassified_band)

        survey_date_file_path = tiff_file_path.parents[1] / f'{tiff_file_path.stem}_catzoc_decay_all.tiff'
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

        measured_surveys = [s for s in all_surveys if not s.get('interpolated')]  # Skip interpolated layers to use actual surveys
        surveys_to_rank = measured_surveys if measured_surveys else all_surveys
        most_recent_survey = max(surveys_to_rank, key=lambda x: x['end_date'])

        today = date.today()
        most_recent_survey['supersession_score'] = supersession(most_recent_survey)
        most_recent_survey['catzoc'] = catzoc(most_recent_survey)
        most_recent_survey['catzoc_decay'] = decay(most_recent_survey, today)

        # output_folder = self.param_lookup['output_directory'].valueAsText
        # self.write_message(f"  Raw Score: {most_recent_survey['supersession_score']:.2f}", output_folder)
        # self.write_message(f"  Decayed Score: {catzoc_decay:.2f}", output_folder)
        # self.write_message(f"  CATZOC Category: {most_recent_survey['catzoc']}", output_folder)

        if nodata is not None and np.isnan(nodata):
            is_nodata = np.isnan(contributor_band_values)
        else:
            is_nodata = (contributor_band_values == nodata)

        reclassified_band = np.where(is_nodata, nodata, most_recent_survey['catzoc_decay']).astype(np.float32)

        survey_date_file_path = tiff_file_path.parents[1] / f'{tiff_file_path.stem}_catzoc_decay_latest.tiff'
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

    def download_nbs_tile(self, temp_folder: pathlib.Path, tile_id: str, ecoregion_id: str) -> pathlib.Path:
        """
        Modified to accept a pathlib.Path object for output_folder 
        and download to that specific local path.
        """

        nbs_bucket = self.get_bucket()
        output_tile_path = False

        output_folder = self.param_lookup['output_directory'].valueAsText
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            current_file = temp_folder / ecoregion_id / get_config_item('BLUETOPO', 'SUBFOLDER') / obj_summary.key
            # Store the path to the tile, not the xml
            if current_file.suffix == '.tiff':
                # TODO this would need to check S3 now and not the temp folder
                if current_file.exists():
                    self.write_message(f'Skipping: {current_file.name}', output_folder)
                    return output_tile_path
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
        # temporarily rename file to output singleband with original name
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

    def run(self, tile_gdf: gpd.GeoDataFrame) -> None:
        print('Downloading BlueTopo Datasets')

        output_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        self.setup_dask(self.param_lookup['env'])
        param_inputs = [[self.param_lookup, output_bucket, row[0], row[1]] for _, row in tile_gdf.iterrows() if isinstance(row[1], str)]  # rows out of ER will be nan
        future_tiles = self.client.map(_process_tile, param_inputs)
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, self.param_lookup['output_directory'].valueAsText)
        self.close_dask()

        # log all tiles using tile_gdf
        tiles = list(tile_gdf['tile'])
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test')  # TODO update to prod hibase

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value"""

        raster_ds = gdal.Open(str(tiff_file_path), gdal.GA_Update)
        no_data = -999999
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array < 0, raster_array, no_data)
        raster_ds.GetRasterBand(1).WriteArray(meters_array)
        raster_ds.GetRasterBand(1).SetNoDataValue(no_data)  # took forever to find this gem
        raster_ds = None

    def upload_current_tiles_to_s3(self, tile_folder: pathlib.Path, bucket_name: str, ecoregion_id: str) -> None:
        """Upload all tiff files to s3 for current tile"""

        s3_client = boto3.client('s3')
        for tiff_file in tile_folder.glob('*'):
            ecoregion_index = tiff_file.parts.index(ecoregion_id)
            s3_path = pathlib.Path(*tiff_file.parts[ecoregion_index:])
            self.write_message(f'Uploading {tiff_file} to s3://{bucket_name}/{s3_path}', self.param_lookup['output_directory'].valueAsText)
            s3_client.upload_file(str(tiff_file), bucket_name, f'{str(s3_path)}')
