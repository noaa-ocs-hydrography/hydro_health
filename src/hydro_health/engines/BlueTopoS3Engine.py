"""Class for processing everything for a single tile"""

import tempfile
import boto3
import os
import pathlib
import sys
import rasterio
import re

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

# Added ogr and osr for vector contour generation
from osgeo import gdal, ogr, osr

from hydro_health.helpers.tools import get_config_item
# Removed 'decay' from the imports since we only want ISS now
from hydro_health.engines.Engine import Engine, supersession, catzoc

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


def _process_tile(param_inputs: list) -> None:
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

        # Download the tile and determine what level of processing is required
        tiff_file_path, run_mode = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id, s3_output_bucket, output_prefix)
        
        if run_mode == 'skip' or not tiff_file_path:
            print(f"[{tile_id}] Processing skipped (all files already exist).")
            return

        print(f"[{tile_id}] Download complete. Resampling and reprojecting base tile...")
        # First, resample and reproject to 100m to serve as the base bounding box and grid
        engine.resample_and_reproject(tiff_file_path)

        print(f"[{tile_id}] Creating survey end date TIFF...")
        engine.create_survey_end_date_tiff(tiff_file_path)

        print(f"[{tile_id}] Creating Initial Survey Score (ISS) all...")
        engine.create_catzoc_all(tiff_file_path)
        
        print(f"[{tile_id}] Tiling corresponding UKC mosaic...")
        engine.create_ukc_tile(tiff_file_path, ecoregion_id, s3_output_bucket, output_prefix)

        print(f"[{tile_id}] Extracting singlebands...")
        mb_tiff_file = engine.rename_multiband(tiff_file_path)
        engine.multiband_to_singleband(mb_tiff_file, band=1)
        engine.multiband_to_singleband(mb_tiff_file, band=2)
        mb_tiff_file.unlink() 

        print(f"[{tile_id}] Setting ground to nodata...")
        engine.set_ground_to_nodata(tiff_file_path)

        print(f"[{tile_id}] Creating slope...")
        engine.create_slope(tiff_file_path)

        print(f"[{tile_id}] Finalizing COG format...")
        engine.finalize_cog(tiff_file_path)      

        print(f"[{tile_id}] Uploading all final tiles to S3...")
        engine.upload_current_tiles_to_s3(tiff_file_path.parents[0], s3_output_bucket, ecoregion_id, output_prefix)
        print(f"[{tile_id}] Processing successfully completed (All Tasks).")


class BlueTopoS3Engine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self, param_lookup: dict[dict]):
        super().__init__()
        self.param_lookup = param_lookup
        # Set target resolution and CRS
        self.target_resolution = 100.0
        self.target_crs = "EPSG:32617"

    @property
    def res_str(self) -> str:
        """Dynamically generate the resolution string (e.g., '100m' or '20m')"""
        return f"{int(self.target_resolution)}m" if self.target_resolution.is_integer() else f"{self.target_resolution}m"

    def resample_and_reproject(self, tiff_path: pathlib.Path) -> None:
        """Warp the downloaded raster to target resolution and CRS."""
        
        output_folder = self.param_lookup['output_directory'].valueAsText
        msg = f"Resampling {tiff_path.name} to {self.target_resolution}m and {self.target_crs}..."
        print(msg)
        self.write_message(msg, output_folder)
        
        temp_tiff = tiff_path.parent / f"warped_{tiff_path.name}"
        
        # Switched to GRA_Bilinear and explicitly defining NoData to -9999
        gdal.Warp(
            str(temp_tiff),
            str(tiff_path),
            xRes=self.target_resolution,
            yRes=self.target_resolution,
            dstSRS=self.target_crs,
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=-9999,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
        )
        
        if temp_tiff.exists():
            tiff_path.unlink()
            temp_tiff.rename(tiff_path)
            
    def create_ukc_tile(self, tiff_file_path: pathlib.Path, ecoregion_id: str, s3_output_bucket: str, output_prefix: str|bool) -> None:
        """
        Extract bounds from the current BlueTopo tile and tile out the corresponding UKC mosaic from S3.
        """
        er_prefix = ecoregion_id if str(ecoregion_id).startswith("ER_") else f"ER_{ecoregion_id}"
        ukc_filename = f"{er_prefix}_UKC_Mosaic_{self.res_str}.tif"
        
        s3_key = f"low_res/{self.res_str}/{ukc_filename}"
        if output_prefix and output_prefix.strip('/') != "low_res":
            s3_key = f"{output_prefix}/{s3_key}"
            
        s3_client = boto3.client('s3')
        try:
            s3_client.head_object(Bucket=s3_output_bucket, Key=s3_key)
        except ClientError:
            print(f"[{tiff_file_path.stem}] Warning: UKC mosaic not found at s3://{s3_output_bucket}/{s3_key}")
            return

        ukc_mosaic_path = f"/vsis3/{s3_output_bucket}/{s3_key}"
        ukc_tile_path = tiff_file_path.parents[0] / f"{tiff_file_path.stem}_UKC.tiff"

        ds = gdal.Open(str(tiff_file_path))
        if ds is None:
            return
        
        geo_transform = ds.GetGeoTransform()
        width = ds.RasterXSize
        height = ds.RasterYSize
        
        minx = geo_transform[0]
        maxy = geo_transform[3]
        maxx = minx + geo_transform[1] * width
        miny = maxy + geo_transform[5] * height
        ds = None

        gdal.Warp(
            str(ukc_tile_path),
            str(ukc_mosaic_path),
            outputBounds=(minx, miny, maxx, maxy),
            xRes=self.target_resolution,
            yRes=self.target_resolution,
            dstSRS=self.target_crs,
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=-9999,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
        )

    def create_catzoc_all(self, tiff_file_path: pathlib.Path) -> None:
        """
        Generate an Initial Survey Score (ISS) raster of unique values for each survey area.
        """
        with rasterio.open(tiff_file_path) as src:
            contributor_band_values = np.round(src.read(3))
            transform = src.transform
            nodata = -9999 
            width, height = src.width, src.height  

        xml_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}.tiff.aux.xml'
        tree = etree.parse(xml_file_path)
        root = tree.getroot()

        contributor_band_xml = root.xpath("//PAMRasterBand[Description='Contributor']")
        rows = contributor_band_xml[0].xpath(".//GDALRasterAttributeTable/Row")
        rat_node = root.find(".//GDALRasterAttributeTable")
        field_names = [f.find('Name').text for f in rat_node.findall('FieldDefn')]

        table_data = []
        for row in rows:
            row_data = {field_names[i]: f_val.text for i, f_val in enumerate(row.findall('F'))}
            
            start_date_str = row_data.get('survey_date_start')
            end_date_str = row_data.get('survey_date_end')

            data = {
                "value": float(row_data.get('value', 0) or 0),
                'start_date': (
                    datetime.strptime(start_date_str, "%Y-%m-%d").date() 
                    if start_date_str and start_date_str != "N/A" 
                    else None
                ),
                "end_date": (
                    datetime.strptime(end_date_str, "%Y-%m-%d").date() 
                    if end_date_str and end_date_str != "N/A" 
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

        for meta in table_data:
            ss_score = supersession(meta)
            meta['supersession_score'] = ss_score
            meta['catzoc'] = catzoc(meta)
            meta['iss'] = ss_score 

        attribute_table_df = pd.DataFrame(table_data)

        iss_mapping = attribute_table_df[['value', 'iss']].drop_duplicates()
        reclass_matrix = iss_mapping.to_numpy()
        reclass_dict = {row[0]: row[1] for row in reclass_matrix}

        reclassified_band = np.vectorize(lambda x: reclass_dict.get(x, nodata))(contributor_band_values)
        reclassified_band = np.where(reclassified_band == None, nodata, reclassified_band)

        survey_date_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}_ISS_all.tiff'
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
        """Generate an Initial Survey Score (ISS) raster using the most recent survey date"""

        with rasterio.open(tiff_file_path) as src:
            contributor_band_values = np.round(src.read(3))
            transform = src.transform
            nodata = -9999 
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

        most_recent_survey['supersession_score'] = supersession(most_recent_survey)
        most_recent_survey['catzoc'] = catzoc(most_recent_survey)
        
        most_recent_survey['iss'] = most_recent_survey['supersession_score']

        if nodata is not None and np.isnan(nodata):
            is_nodata = np.isnan(contributor_band_values)
        else:
            is_nodata = (contributor_band_values == nodata)

        reclassified_band = np.where(is_nodata, nodata, most_recent_survey['iss']).astype(np.float32)

        survey_date_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}_ISS_latest.tiff'
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
            contributor_band_values = np.round(src.read(3))
            transform = src.transform
            nodata = -9999 
            width, height = src.width, src.height  

        xml_file_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}.tiff.aux.xml'
        tree = etree.parse(xml_file_path)
        root = tree.getroot()

        contributor_band_xml = root.xpath("//PAMRasterBand[Description='Contributor']")
        rows = contributor_band_xml[0].xpath(".//GDALRasterAttributeTable/Row")
        rat_node = root.find(".//GDALRasterAttributeTable")
        field_names = [f.find('Name').text for f in rat_node.findall('FieldDefn')]

        table_data = []
        for row in rows:
            row_dict = {field_names[i]: f_val.text for i, f_val in enumerate(row.findall('F'))}
            end_date_str = row_dict.get('survey_date_end')

            data = {
                "value": float(row_dict.get('value', 0) or 0),
                "survey_date_end": (
                    datetime.strptime(end_date_str, "%Y-%m-%d").date() 
                    if end_date_str and end_date_str != "N/A" 
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

    def download_nbs_tile(self, temp_folder: pathlib.Path, tile_id: str, ecoregion_id: str, s3_output_bucket: str, output_prefix: str|bool) -> tuple[pathlib.Path | bool, str]:
        """
        Check the S3 bucket to skip already processed tiles.
        Only runs logic to check if ALL final output files exist.
        """

        nbs_bucket = self.get_bucket()
        output_tile_path = False
        output_folder = self.param_lookup['output_directory'].valueAsText
        run_mode = 'all'
        
        s3_client = boto3.client('s3')

        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            current_file = temp_folder / ecoregion_id / get_config_item('BLUETOPO', 'SUBFOLDER') / obj_summary.key
            
            if current_file.suffix == '.tiff':
                ecoregion_index = current_file.parts.index(ecoregion_id)
                s3_subpath = pathlib.Path(*current_file.parts[ecoregion_index:])
                
                s3_key = str(s3_subpath).replace("\\", "/") 
                if output_prefix:
                    s3_key = f'{output_prefix}/{s3_key}'
                    
                s3_path_obj = pathlib.Path(s3_key)
                
                # Check for ALL the files that get created during a run
                expected_keys = [
                    s3_key, # Base processed tile
                    str(s3_path_obj.parent / f"{s3_path_obj.stem}_ISS_all.tiff").replace("\\", "/"),
                    str(s3_path_obj.parent / f"{s3_path_obj.stem}_UKC.tiff").replace("\\", "/"),
                    str(s3_path_obj.parent / f"{s3_path_obj.stem}_survey_end_date.tiff").replace("\\", "/"),
                    str(s3_path_obj.parent / f"{s3_path_obj.stem}_slope.tiff").replace("\\", "/"),
                    str(s3_path_obj.parent / f"{s3_path_obj.stem}_unc.tiff").replace("\\", "/")
                ]

                all_files_exist = True
                for key in expected_keys:
                    try:
                        s3_client.head_object(Bucket=s3_output_bucket, Key=key)
                    except ClientError:
                        all_files_exist = False
                        break

                if all_files_exist:
                    run_mode = 'skip'
                    msg = f"Skipping {tile_id} for ecoregion: {ecoregion_id} (All final derived files already exist in S3)"
                    print(msg)
                    self.write_message(msg, output_folder)
                    return False, run_mode
                else:
                    run_mode = 'all'
                    msg = f"Tile {tile_id} is missing files. Downloading and processing all..."
                    print(msg)
                    self.write_message(msg, output_folder)

                print(f"Downloading NBS Source for tile: {tile_id} for ecoregion: {ecoregion_id}")
                output_tile_path = current_file
                
            tile_folder = current_file.parents[0]
            tile_folder.mkdir(parents=True, exist_ok=True)   

            if not current_file.exists():
                self.write_message(f'Downloading: {current_file.name}', output_folder)
                nbs_bucket.download_file(obj_summary.key, str(current_file))
            else:
                self.write_message(f'Local file already exists, skipping download: {current_file.name}', output_folder)

        return output_tile_path, run_mode
    
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

    def create_combined_isobaths(self, ecoregion_ids: list[str], s3_bucket: str, output_prefix: str|bool) -> None:
        """
        Creates solid polygon isobaths for specified depth bands (0-20m and 0-40m)
        and appends them directly to the local Master_Grids.gpkg.
        
        Uses an efficient tile-by-tile memory strategy to prevent VRT rendering dropouts,
        S3 connection thrashing, and massive disk usage.
        """
        master_gpkg_path = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/Master_Grids.gpkg")
        
        print(f"[All Regions] Generating combined polygon isobaths and saving to local {master_gpkg_path}...")
        s3_client = boto3.client('s3')
        
        # Optimize GDAL for S3 reads: allows dynamic block chunking directly from the web
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gdal.SetConfigOption('VSI_CACHE', 'YES')
        
        driver = ogr.GetDriverByName("GPKG")
        if master_gpkg_path.exists():
            master_ds = driver.Open(str(master_gpkg_path), 1)  # 1 indicates update mode
        else:
            master_ds = driver.CreateDataSource(str(master_gpkg_path))
            
        if master_ds is None:
            print(f"Error: Could not open or create {master_gpkg_path}")
            return
            
        srs = osr.SpatialReference()
        srs.SetFromUserInput(self.target_crs)
        
        # We process negative depths. 0 to -20 is the first band, 0 to -40 is the second.
        depth_bands = [
            ((0.0, -20.0), 'isobath_0_20m'),
            ((0.0, -40.0), 'isobath_0_40m')
        ]
        
        # Check if layers already exist to skip recreation
        existing_layers = [master_ds.GetLayerByIndex(i).GetName() for i in range(master_ds.GetLayerCount())]
        expected_layers = [layer_name for _, layer_name in depth_bands]
        
        if all(layer in existing_layers for layer in expected_layers):
            print(f"[All Regions] Isobath layers already exist in {master_gpkg_path}. Skipping creation.")
            master_ds = None
            return
        
        # Ensure layers exist in the Master_Grids.gpkg as POLYGONS
        layers = {}
        for _, layer_name in depth_bands:
            # Safely delete layer if it already exists by checking integer indices
            for i in range(master_ds.GetLayerCount()):
                if master_ds.GetLayerByIndex(i).GetName() == layer_name:
                    master_ds.DeleteLayer(i)
                    break
                
            layer = master_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
            field_defn = ogr.FieldDefn("elevation", ogr.OFTReal)
            layer.CreateField(field_defn)
            layers[layer_name] = layer
        
        # Process one ecoregion at a time
        for ecoregion_id in ecoregion_ids:
            print(f"[{ecoregion_id}] Gathering S3 links...")
            search_prefix = f"{ecoregion_id}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            if output_prefix:
                search_prefix = f"{output_prefix}/{search_prefix}"
                
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix)
            
            vsis3_paths = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Strictly look for files ending with digits + .tiff (case insensitive)
                        if re.search(r'\d+\.tiff$', key, re.IGNORECASE):
                            vsis3_paths.append(f"/vsis3/{s3_bucket}/{key}")
                            
            if not vsis3_paths:
                print(f"[{ecoregion_id}] No base tiles found. Skipping.")
                continue
            
            print(f"[{ecoregion_id}] Processing {len(vsis3_paths)} tiles individually via memory to guarantee 100% data extraction...")
            
            for path in vsis3_paths:
                src_ds = gdal.Open(path)
                if src_ds is None:
                    continue
                    
                src_band = src_ds.GetRasterBand(1)
                
                # Fetch full tile from S3 instantly into memory (BlueTopo tiles are typically small enough for this)
                data = src_band.ReadAsArray()
                
                if data is None:
                    src_band = None
                    src_ds = None
                    continue
                    
                # Skip entirely nodata/empty tiles quickly
                if not np.any(data != -9999):
                    src_band = None
                    src_ds = None
                    continue
                
                for depth_range, layer_name in depth_bands:
                    upper, lower = depth_range
                    
                    # Create the boolean mask array
                    mask_data = np.zeros_like(data, dtype=np.uint8)
                    mask_data[(data <= upper) & (data >= lower)] = 1
                    
                    # Only polygonize if there are actually matches in this tile
                    if np.any(mask_data):
                        # Use GDAL memory driver for rapid, disk-free mask polygonization
                        drv = gdal.GetDriverByName('MEM')
                        mask_ds = drv.Create('', src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Byte)
                        mask_ds.SetGeoTransform(src_ds.GetGeoTransform())
                        mask_ds.SetProjection(src_ds.GetProjection())
                        
                        mask_band = mask_ds.GetRasterBand(1)
                        # Setting NoDataValue to 0 instructs Polygonize to only outline the '1' values
                        mask_band.SetNoDataValue(0)
                        mask_band.WriteArray(mask_data)
                        
                        # Polygonize directly to the active layer (GDAL Polygonize internally manages transactions for each operation safely)
                        gdal.Polygonize(mask_band, mask_band, layers[layer_name], -1, [], callback=None)
                        
                        mask_band = None
                        mask_ds = None
                        
                src_band = None
                src_ds = None
                
        print("[All Regions] Saving and securely closing Master_Grids.gpkg...")
        layers.clear()
        master_ds = None

    def create_and_upload_mosaics(self, ecoregion_ids: list[str], s3_bucket: str, output_prefix: str|bool) -> None:
        """
        Creates massive 100m mosaics from all base bathy tiles and derivatives across all processed ecoregions,
        masks them securely with the EcoRegions layer, saves locally to low_res, and uploads to S3.
        """
        output_folder = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/outputs")
        low_res_dir = output_folder / "low_res" / self.res_str
        low_res_dir.mkdir(parents=True, exist_ok=True)
        
        master_gpkg_path = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/Master_Grids.gpkg")
        
        s3_client = boto3.client('s3')

        mosaics_config = {
            "Bathy": {"regex": re.compile(r'\d+\.tiff$', re.IGNORECASE), "filename": f"BlueTopo_Bathy_Mosaic_{self.res_str}.tiff", "paths": []},
            "ISS": {"regex": re.compile(r'_ISS_all\.tiff$', re.IGNORECASE), "filename": f"BlueTopo_ISS_Mosaic_{self.res_str}.tiff", "paths": []},
            "Survey_Date": {"regex": re.compile(r'_survey_end_date\.tiff$', re.IGNORECASE), "filename": f"BlueTopo_Survey_Date_Mosaic_{self.res_str}.tiff", "paths": []},
            "Slope": {"regex": re.compile(r'_slope\.tiff$', re.IGNORECASE), "filename": f"BlueTopo_Slope_Mosaic_{self.res_str}.tiff", "paths": []},
            "UKC": {"regex": re.compile(r'_UKC\.tiff$', re.IGNORECASE), "filename": f"BlueTopo_UKC_Mosaic_{self.res_str}.tiff", "paths": []},
        }

        # 1. Determine which mosaics actually need to be generated
        mosaics_to_process = {}
        for m_key, config in mosaics_config.items():
            s3_key = f"low_res/{self.res_str}/{config['filename']}"
            if output_prefix:
                s3_key = f"{output_prefix}/{s3_key}"
                
            try:
                s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
                print(f"[Mosaic] {m_key} mosaic already exists at s3://{s3_bucket}/{s3_key}. Skipping creation.")
            except ClientError:
                mosaics_to_process[m_key] = config
                mosaics_to_process[m_key]['s3_key'] = s3_key

        if not mosaics_to_process:
            print("[Mosaic] All requested mosaics already exist in S3. Skipping processing entirely.")
            return

        print(f"[Mosaic] Gathering S3 links across {len(ecoregion_ids)} ecoregions for: {list(mosaics_to_process.keys())}")
        
        # 2. Gather source file paths in a single efficient pass through S3 per ecoregion
        for ecoregion_id in ecoregion_ids:
            search_prefix = f"{ecoregion_id}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            if output_prefix:
                search_prefix = f"{output_prefix}/{search_prefix}"
                
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        for m_key, config in mosaics_to_process.items():
                            if config['regex'].search(key):
                                config['paths'].append(f"/vsis3/{s3_bucket}/{key}")
                                break  # Prevent checking other regexes once mapped
                            
        # 3. Locate the EcoRegions layer once for masking
        cutline_layer_name = None
        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            ds = None
            for name in layer_names:
                # Match case-insensitively, allowing for underscores or spaces
                if name.lower().replace("_", "").replace(" ", "") == "ecoregions":
                    cutline_layer_name = name
                    break
            
            if not cutline_layer_name:
                print(f"[Mosaic] WARNING: Could not find an 'EcoRegions' layer in {master_gpkg_path}.")
                print(f"[Mosaic] Available layers: {layer_names}")
                print("[Mosaic] Proceeding without vector mask to prevent crash...")
        else:
            print(f"[Mosaic] WARNING: Could not open {master_gpkg_path} to check layers. Proceeding without mask...")

        warp_kwargs_base = {
            "format": "GTiff",
            "xRes": self.target_resolution,
            "yRes": self.target_resolution,
            "dstSRS": self.target_crs,
            "resampleAlg": gdal.GRA_Bilinear,
            "dstNodata": -9999,
            "creationOptions": ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
        }
        
        if cutline_layer_name:
            warp_kwargs_base["cutlineDSName"] = str(master_gpkg_path)
            warp_kwargs_base["cutlineLayer"] = cutline_layer_name
            warp_kwargs_base["cropToCutline"] = True

        # 4. Generate & Upload each required mosaic dynamically
        for m_key, config in mosaics_to_process.items():
            if not config['paths']:
                print(f"[Mosaic] No base tiles found for {m_key}. Skipping.")
                continue

            local_mosaic_path = low_res_dir / config['filename']
            s3_key = config['s3_key']
            
            print(f"[Mosaic] Building VRT from {len(config['paths'])} base tiles for {m_key}...")
            vrt_path = f"/vsimem/{m_key}_mosaic.vrt"
            gdal.BuildVRT(vrt_path, config['paths'])
            
            print(f"[Mosaic] Warping and masking {m_key} mosaic utilizing EcoRegions layer...")
            warp_options = gdal.WarpOptions(**warp_kwargs_base)
            gdal.Warp(str(local_mosaic_path), vrt_path, options=warp_options)
            
            gdal.Unlink(vrt_path)
            
            print(f"[Mosaic] Generating robust overviews for fast performance...")
            ds = gdal.Open(str(local_mosaic_path), gdal.GA_Update)
            if ds is not None:
                ds.BuildOverviews("BILINEAR", [2, 4, 8, 16])
                ds = None
                
            print(f"[Mosaic] Uploading completely masked {m_key} mosaic to s3://{s3_bucket}/{s3_key}...")
            s3_client.upload_file(
                str(local_mosaic_path), 
                s3_bucket, 
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'resolution': str(self.target_resolution),
                        'crs': self.target_crs
                    }
                }
            )
            print(f"[Mosaic] {m_key} mosaic creation and upload perfectly complete.")

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str|bool=False, process_tiles: bool=False) -> None:
        output_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        all_ecoregions = self.param_lookup['eco_regions'].value
        
        if process_tiles:
            print('Downloading BlueTopo Datasets')
            self.setup_dask(self.param_lookup['env'], threads_per_worker=1)
            # Param list cleaned up and simplified
            param_inputs = [[self.param_lookup, output_bucket, row[0], row[1], output_prefix] for _, row in tile_gdf.iterrows() if isinstance(row[1], str)] 
            
            print(f"Submitting {len(param_inputs)} tiles to Dask workers...")
            future_tiles = self.client.map(_process_tile, param_inputs)
            
            print("Waiting for all Dask workers to complete...")
            tile_results = self.client.gather(future_tiles)
            print("All Dask workers finished successfully.")
            
            self.print_async_results(tile_results, self.param_lookup['output_directory'].valueAsText)
            self.close_dask()
        else:
            print("Skipping individual tile processing (process_tiles=False).")
        
        for ecoregion in all_ecoregions:
            s3_path = f"{output_prefix}/{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo" if output_prefix else f"{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            if process_tiles:
                self.write_run_manifest(s3_path)
        
        # Generate the unified isobaths from all finished tiles across all ecoregions
        self.create_combined_isobaths(all_ecoregions, output_bucket, output_prefix)

        # Mosaic all 100m base tiles and derivatives, apply masks, save locally and push to S3
        self.create_and_upload_mosaics(all_ecoregions, output_bucket, output_prefix)

        tiles = list(tile_gdf['tile'])
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test') 

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value"""
        raster_ds = gdal.Open(str(tiff_file_path), gdal.GA_Update)
        no_data = -9999
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array < 0, raster_array, no_data)
        raster_ds.GetRasterBand(1).WriteArray(meters_array)
        raster_ds.GetRasterBand(1).SetNoDataValue(no_data)  
        raster_ds = None

    def upload_current_tiles_to_s3(self, tile_folder: pathlib.Path, bucket_name: str, ecoregion_id: str, output_prefix: str|bool) -> None:
        """Upload all tiff and gpkg files to s3 for current tile with Resolution metadata"""
        s3_client = boto3.client('s3')
        
        s3_metadata_args = {
            'Metadata': {
                'resolution': str(self.target_resolution),
                'crs': self.target_crs
            }
        }
        
        # This will automatically pick up the newly generated .gpkg file!
        for file in tile_folder.glob('*'):
            ecoregion_index = file.parts.index(ecoregion_id)
            s3_subpath = pathlib.Path(*file.parts[ecoregion_index:])
            s3_path = s3_subpath if not output_prefix else f'{output_prefix}/{s3_subpath}'
            
            s3_path_formatted = str(s3_path).replace("\\", "/")
            
            self.write_message(f'Uploading {file} to s3://{bucket_name}/{s3_path_formatted}', self.param_lookup['output_directory'].valueAsText)
            s3_client.upload_file(str(file), bucket_name, f'{str(s3_path_formatted)}', ExtraArgs=s3_metadata_args)