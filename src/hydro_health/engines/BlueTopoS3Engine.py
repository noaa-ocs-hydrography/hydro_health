"""Class for processing everything for a single tile"""

import tempfile
import boto3
import os
import pathlib
import sys
import rasterio
import re
import s3fs
import psutil  # Added for memory tracking
import shutil  # Added for backup/restore of base tiles
import concurrent.futures # Added for aggressive parallel S3 downloading
import gc # Added for aggressive RAM purging

from rasterio.session import AWSSession
import geopandas as gpd
import pandas as pd
import numpy as np

from multiprocessing import set_executable
from hydro_health.helpers import hibase_logging
from datetime import datetime, date, timezone
from botocore.client import Config
from botocore import UNSIGNED
from botocore.exceptions import ClientError
from lxml import etree

# Added ogr and osr for vector contour generation
from osgeo import gdal, ogr, osr

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine, supersession, catzoc

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'



def _process_tile(param_inputs: list) -> str:
    """
    Static function that handles processing of a single tile.
    Refactored for S3-to-S3 workflow using ephemeral storage.
    """

    param_lookup, tile_id, ecoregion_id, output_prefix, target_res = param_inputs
    
    print(f"[{tile_id}] Initiating processing for ecoregion {ecoregion_id}...")

    engine = BlueTopoS3Engine(param_lookup)
        
    existing_files = engine.check_s3_existing_files(output_prefix, target_res, ecoregion_id, tile_id, overwrite=False)
        
    if all(existing_files.values()):
        msg = f'[{tile_id}] BlueTopo tile and all derivatives already exist. Skipping download.'
        print(msg)
        return msg
    else:
        # Determine exactly which files are missing to inform the user
        missing_files = [k for k, v in existing_files.items() if not v]
        print(f"[{tile_id}] Missing derivatives detected: {missing_files}. Downloading raw NBS source to generate them...")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            tiff_file_path = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id, output_prefix, target_res)
            if not tiff_file_path:
                msg = f"[{tile_id}] Processing skipped (Source tile could not be downloaded or was invalid)."
                print(msg)
                return msg
            
            # Create a pristine backup of the downloaded base tile. 
            # If variable generation corrupts a file, we can restore from here and remake them without re-downloading.
            backup_path = temp_path / f"{tiff_file_path.name}.backup"
            shutil.copy(tiff_file_path, backup_path)
            
            max_proc_retries = 3
            proc_attempt = 0
            all_valid = False
            tile_folder = tiff_file_path.parents[0]

            while proc_attempt < max_proc_retries and not all_valid:
                if proc_attempt > 0:
                    print(f"[{tile_id}] Retrying derivative generation (Attempt {proc_attempt+1}/{max_proc_retries})...")
                    # Clean up all potentially corrupt TIFFs generated in the previous attempt
                    if tile_folder.exists():
                        for f in tile_folder.glob('*.tiff'):
                            f.unlink()
                    # Restore the pristine downloaded base tile
                    tile_folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy(backup_path, tiff_file_path)

                try:
                    # Explicitly and safely warp the base tile to the target CRS and Resolution
                    print(f"[{tile_id}] Warping raw tile in memory to align generated derivatives...")
                    engine.warp_bluetopo_tile(tiff_file_path, target_res)
                    
                    if not existing_files['survey_end_date']:
                        engine.create_survey_end_date_tiff(tiff_file_path)
                        
                    if not existing_files['iss_110']:
                        engine.create_catzoc_all(tiff_file_path, increased_scale=True)

                    if not existing_files['hurricane']:
                        engine.create_hurricane_tile(tiff_file_path, target_res)
                    
                    # Only parse the multiband source if we are actively generating the base tile or uncertainty tile
                    if not existing_files['base'] or not existing_files['unc']:
                        mb_tiff_file = engine.rename_multiband(tiff_file_path)
                        if not existing_files['base']:
                            engine.multiband_to_singleband(mb_tiff_file, band=1)
                        if not existing_files['unc']:
                            engine.multiband_to_singleband(mb_tiff_file, band=2)
                        if mb_tiff_file.exists():
                            mb_tiff_file.unlink()
                    
                    if not existing_files['base']:
                        engine.set_ground_to_nodata(tiff_file_path)
                        engine.finalize_cog(tiff_file_path)     

                    # Crop intermediate tiffs dynamically if we are in 20m resolution mode
                    if target_res == 20:
                        print(f"[{tile_id}] Resolution is 20m. Cropping all local intermediate tiffs to ecoregion {ecoregion_id}...")
                        stem = tiff_file_path.stem
                        for local_file in tile_folder.glob(f"{stem}*.tiff"):
                            engine.crop_to_ecoregion(local_file, ecoregion_id, target_res)

                    # ---------------------------------------------------------
                    # VALIDATE ALL GENERATED DERIVATIVES BEFORE S3 UPLOAD
                    # ---------------------------------------------------------
                    print(f"[{tile_id}] Validating all generated derivative TIFFs before upload...")
                    all_valid = True
                    for local_file in tile_folder.glob('*.tiff'):
                        if not engine._is_valid_tiff(local_file):
                            print(f"[{tile_id}] ERROR: Generated derivative {local_file.name} failed deep validation (Corrupt/ZIPDecode error).")
                            all_valid = False
                            break # Break out of file check loop to trigger a full generation retry
                            
                except Exception as e:
                    print(f"[{tile_id}] Exception occurred during derivative generation: {e}")
                    all_valid = False
                
                proc_attempt += 1
            
            if not all_valid:
                msg = f"[{tile_id}] FAILED: One or more generated derivative TIFFs were persistently corrupt after {max_proc_retries} attempts. Aborting S3 upload to protect mosaic."
                print(msg)
                return msg

            print(f"[{tile_id}] Uploading strictly ONLY the newly generated missing files to S3...")
            # Pass existing files here to strictly prevent overwriting existing derivatives
            engine.upload_current_tiles_to_s3(tile_folder, temp_path, existing_files)
            
            msg = f"[{tile_id}] Processing successfully completed."
            print(msg)
            return msg


def _parse_survey_date(date_str: str) -> date | None:
    """Robustly parse survey dates from metadata strings handling various formats and extracting years."""
    if not date_str or str(date_str).strip().upper() in ["N/A", "UNKNOWN", "NULL", "NONE", "NAN", ""]:
        return None
        
    date_str = str(date_str).strip()
    
    # Try common exact formats
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m", "%Y%m%d", "%Y"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
            
    # Fallback: extract the first numeric sequence (integer or float) that looks like a year and round to nearest whole year
    match = re.search(r'\b((?:17|18|19|20)\d{2}(?:\.\d+)?)\b', date_str)
    if match:
        year = int(round(float(match.group(1))))
        return date(year, 1, 1)
        
    return None


class BlueTopoS3Engine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self, param_lookup: dict[dict]):
        super().__init__()
        self.param_lookup = param_lookup
        self.target_crs = "EPSG:6350"
        # Tiling is bypassed to exclusively remake mosaics
        self.skip_tiling = True

        # Apply global GDAL S3 Network and Overview Optimizations
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gdal.SetConfigOption('VSI_CACHE', 'YES')
        gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'tif,tiff')
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
        
        # FIXED: Drastically reduced GDAL cache to 1GB to prevent EC2 OS thrashing/freezing
        gdal.SetConfigOption('GDAL_CACHEMAX', '1024MB')
        
        # Add HTTP retries to prevent massive mosaics from silently stalling due to network blips
        gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '5')
        gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '3')

        # Attempt to automatically fix invalid vector geometries (TopologyExceptions) on the fly
        gdal.SetConfigOption('OGR_ENABLE_MAKE_VALID', 'YES')

    def crop_to_ecoregion(self, tiff_path: pathlib.Path, ecoregion_id: str, resolution: int) -> None:
        """Crop a single tile's TIFF to its ecoregion boundary if resolution is 20m."""

        if resolution != 20:
            return

        master_gpkg_path = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

        # Find cutline layer name (Enhanced_EcoRegions)
        cutline_layer_name = None
        er_field_name = None
        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            for name in layer_names:
                if name.lower().replace("_", "").replace(" ", "") == "enhancedecoregions":
                    cutline_layer_name = name
                    break
            if cutline_layer_name:
                layer = ds.GetLayerByName(cutline_layer_name)
                layer_defn = layer.GetLayerDefn()
                field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
                for expected in ['ecoregion_id', 'ecoregion', 'er', 'region', 'name', 'id']:
                    for field in field_names:
                        if field.lower() == expected:
                            er_field_name = field
                            break
                    if er_field_name:
                        break
            ds = None

        if not cutline_layer_name:
            print(f"[{tiff_path.name}] Enhanced_EcoRegions layer not found in Master_Grids.gpkg, skipping crop.")
            return

        temp_cropped = tiff_path.parent / f"crop_{tiff_path.name}"

        # Build query robustly matching various formats (e.g., 'ER_1', 'ER1', '1')
        match = re.search(r'\d+', str(ecoregion_id))
        clean_er_num = match.group(0) if match else str(ecoregion_id)
        er_prefix_val = f"ER_{clean_er_num}"
        er_val_no_underscore = f"ER{clean_er_num}"

        where_clauses = []
        if er_field_name:
            where_clauses.extend([
                f"{er_field_name} = '{er_prefix_val}'",
                f"{er_field_name} = '{er_val_no_underscore}'",
                f"{er_field_name} = '{ecoregion_id}'",
                f"{er_field_name} = '{clean_er_num}'"
            ])
            if clean_er_num.isdigit():
                where_clauses.append(f"{er_field_name} = {clean_er_num}")
        else:
            print(f"[{tiff_path.name}] Warning: Ecoregion ID field name not found. Attempting crop without filter...")

        cutline_where = " OR ".join(where_clauses) if where_clauses else None

        # Pass srcSRS explicitly so GDAL always aligns the source data and cutline CRS correctly
        warp_kwargs = {
            "format": "GTiff",
            "srcSRS": self.target_crs,
            "dstSRS": self.target_crs,
            "dstNodata": -9999,
            "cutlineDSName": str(master_gpkg_path),
            "cutlineLayer": cutline_layer_name,
            "cropToCutline": True,
            "creationOptions": ["COMPRESS=DEFLATE"]
        }
        if cutline_where:
            warp_kwargs["cutlineWhere"] = cutline_where

        warp_options = gdal.WarpOptions(**warp_kwargs)

        try:
            ds_crop = gdal.Warp(str(temp_cropped), str(tiff_path), options=warp_options)
            ds_crop = None # EXPLICIT CLEANUP
            
            if temp_cropped.exists():
                tiff_path.unlink()
                temp_cropped.rename(tiff_path)
        except Exception as e:
            print(f"[{tiff_path.name}] Failed to crop to ecoregion: {e}")
            if temp_cropped.exists():
                temp_cropped.unlink()

    def create_hurricane_tile(self, tiff_file_path: pathlib.Path, resolution: int) -> None:
        """
        Generate a hurricane count raster by evaluating the survey year and 
        accumulating the corresponding yearly hurricane counts for each cell.
        """

        print(f"[{tiff_file_path.name}] Creating cumulative hurricane count tile...")
        stem = tiff_file_path.stem
        hurricane_tile_path = tiff_file_path.parents[0] / f'{stem}_hurricane.tiff'

        # Local survey path always exists because we are enforcing unconditional local reprocessing
        local_survey_path = tiff_file_path.parents[0] / f'{stem}_survey_end_date.tiff'
        survey_ds_path = str(local_survey_path)

        try:
            with rasterio.open(survey_ds_path) as src:
                survey_years = src.read(1)
                transform = src.transform
                nodata = -9999
                width = src.width
                height = src.height
        except rasterio.errors.RasterioIOError:
            print(f"[{stem}] Error: Survey end date raster required for Hurricane calculation could not be loaded from {survey_ds_path}")
            return

        minx = transform.c
        maxy = transform.f
        maxx = minx + transform.a * width
        miny = maxy + transform.e * height

        # Create a rigorous valid mask specifically ignoring 0, nodata, or nan
        valid_mask = (survey_years != nodata) & (survey_years > 0) & ~np.isnan(survey_years)

        unique_years = np.unique(survey_years[valid_mask])
        valid_years = [y for y in unique_years]

        hurricane_path_prefix = get_config_item('HURRICANE', 'COUNT_RASTER_PATH')

        # Initialize everything to explicit nodata.
        result_array = np.full(survey_years.shape, nodata, dtype=np.float32)

        if not valid_years:
            with rasterio.open(
                hurricane_tile_path,
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
                crs=self.target_crs,
                transform=transform,
                nodata=nodata,
            ) as dst:
                dst.write(result_array, 1)
            return

        def get_year_count_array(target_year):
            # Clamp year to valid data range
            target_year = max(1851, min(2023, int(target_year)))

            # Construct both the .tiff and the .tif fallbacks
            s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            s3_uri_tiff = f"/vsis3/{s3_bucket}/{hurricane_path_prefix}/cumulative_count_{target_year}.tiff"
            s3_uri_tif = f"/vsis3/{s3_bucket}/{hurricane_path_prefix}/cumulative_count_{target_year}.tif"

            warp_options = gdal.WarpOptions(
                format="MEM",
                outputBounds=(minx, miny, maxx, maxy),
                width=width,   
                height=height,  
                dstSRS=self.target_crs,
                resampleAlg=gdal.GRA_NearestNeighbour, # Counts should not be interpolated
                dstNodata=-9999
            )

            mem_ds = None
            gdal.PushErrorHandler('CPLQuietErrorHandler')

            # 1. Attempt the .tiff version first (catching RuntimeError if UseExceptions is active)
            try:
                mem_ds = gdal.Warp('', s3_uri_tiff, options=warp_options)
            except Exception:
                mem_ds = None

            # 2. If the .tiff version failed, try the .tif version as a fallback
            if mem_ds is None:
                try:
                    mem_ds = gdal.Warp('', s3_uri_tif, options=warp_options)
                except Exception:
                    mem_ds = None

            gdal.PopErrorHandler()

            if mem_ds is None:
                print(f"[{stem}] Warning: Could not load hurricane raster for year {target_year} at either {s3_uri_tiff} or {s3_uri_tif}. Returning zeros.")
                return np.zeros((height, width), dtype=np.float32)

            arr = mem_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            arr[arr == -9999] = 0.0 
            
            mem_ds = None # EXPLICIT CLEANUP TO PREVENT LEAK
            return arr

        # Start hurricane count at 0.0 ONLY for cells with a valid survey year
        result_array[valid_mask] = 0.0

        min_survey_year = max(1851, int(min(valid_years)))
        max_hurricane_year = 2023

        print(f"[{stem}] Aggregating yearly hurricane counts from {min_survey_year} to {max_hurricane_year}...")
        for target_year in range(min_survey_year, max_hurricane_year + 1):
            year_arr = get_year_count_array(target_year)

            # Add this year's count to any valid pixel whose survey year is <= target_year
            add_mask = valid_mask & (survey_years <= target_year)
            result_array[add_mask] += year_arr[add_mask]

        # FINAL SAFETY NET: forcefully reset any invalid cells (like survey_year = 0) back to nodata
        result_array[~valid_mask] = nodata

        with rasterio.open(
            hurricane_tile_path,
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
            crs=self.target_crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(result_array, 1)
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_catzoc_all(self, tiff_file_path: pathlib.Path, increased_scale: bool=False) -> None:
        """Generate an Initial Survey Score (ISS) raster of unique values for each survey area."""

        print(f"[{tiff_file_path.name}] Creating Initial Survey Score (ISS) all...")
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
                'start_date': _parse_survey_date(start_date_str),
                "end_date": _parse_survey_date(end_date_str),
                'from_filename': row_data.get('source_survey_id'),
                'feat_detect': bool(int(row_data.get('significant_features', 0))),
                'feat_least_depth': bool(int(row_data.get('feature_least_depth', 0))),
                'complete_coverage': bool(int(row_data.get('bathy_coverage', 0))),
                'horiz_uncert_fixed': float(row_data.get('horizontal_uncert_fixed', 0)),
                'horiz_uncert_vari': float(row_data.get('horizontal_uncert_var', 0)),
                'vert_uncert_fixed': float(row_data.get('vertical_uncert_fixed', 0)),
                'vert_uncert_vari': float(row_data.get('vertical_uncert_var', 0)),
                'interpolated': ".interpolated" in row_data.get('source_survey_id', '').lower(),
                'increased_scale': increased_scale
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

        survey_date_file_path = tiff_file_path.parents[0] / f"{tiff_file_path.stem}_ISS_all{'_110' if increased_scale else ''}.tiff"
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
            crs=self.target_crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(reclassified_band, 1)

            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_catzoc_latest(self, tiff_file_path: pathlib.Path, increased_scale: bool=False) -> None:
        """Generate an Initial Survey Score (ISS) raster using the most recent survey date"""

        print(f"[{tiff_file_path.name}] Creating Initial Survey Score (ISS) latest...")
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

        all_surveys = []
        for row in rows:
            row_dict = {field_names[i]: f_val.text for i, f_val in enumerate(row.findall('F'))}

            end_date_str = row_dict.get('survey_date_end')
            end_date = _parse_survey_date(end_date_str) or date.min

            meta = {
                "end_date": end_date,
                "feat_detect": bool(int(row_dict.get('significant_features', 0))),
                "feat_least_depth": bool(int(row_dict.get('feature_least_depth', 0))),
                "complete_coverage": bool(int(row_dict.get('bathy_coverage', 0))),
                "horiz_uncert_fixed": float(row_dict.get('horizontal_uncert_fixed', 0)),
                "horiz_uncert_vari": float(row_dict.get('horizontal_uncert_var', 0)),
                "vert_uncert_fixed": float(row_dict.get('vertical_uncert_fixed', 0)),
                "vert_uncert_vari": float(row_dict.get('vertical_uncert_var', 0)),
                'interpolated': ".interpolated" in row_dict.get('source_survey_id', '').lower(),
                'increased_scale': increased_scale
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
            crs=self.target_crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(reclassified_band, 1)

            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_survey_end_date_tiff(self, tiff_file_path: pathlib.Path) -> None:
        """Create survey end date tiffs from contributor band values in the XML file."""        

        print(f"[{tiff_file_path.name}] Creating survey end date TIFF...")
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
                "survey_date_end": _parse_survey_date(end_date_str)
            }
            table_data.append(data)
        attribute_table_df = pd.DataFrame(table_data)

        # Force parsed years to be rounded and converted into solid clean integers (no fractional floats)
        attribute_table_df['survey_year_end'] = attribute_table_df['survey_date_end'].apply(
            lambda x: int(round(x.year)) if pd.notna(x) else 0
        )

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
            crs=self.target_crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(reclassified_band, 1)

    def _is_valid_tiff(self, file_path: pathlib.Path) -> bool:
        """
        Deeply inspect a TIFF file by calculating its checksum for every band.
        This forces the underlying engine to decompress and read every single 
        pixel block, guaranteeing no ZIPDecode errors exist.
        """
        if not file_path or not file_path.exists():
            return False
            
        try:
            with rasterio.open(file_path) as src:
                for i in src.indexes:
                    src.checksum(i) # Forces a full block-by-block deep read
            return True
        except Exception:
            # rasterio will throw an exception (RasterioIOError) if a block is corrupt
            return False

    def download_nbs_tile(self, temp_folder: pathlib.Path, tile_id: str, ecoregion_id: str, output_prefix: str|bool, target_res:  int) -> pathlib.Path|bool:
        """Unconditionally download the NBS source tile and stage it for full processing."""

        nbs_bucket = self.get_bucket()
        output_tile_path = False
        output_folder = self.param_lookup['output_directory'].valueAsText

        msg = f"Tile {tile_id} targeted for full processing. Downloading fresh NBS Source and metadata..."
        self.write_message(msg, output_folder)

        found_files = False

        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            found_files = True
            if output_prefix == 'low_res':
                beginning_prefix = temp_folder / output_prefix / f'{target_res}m'
            elif output_prefix:
                beginning_prefix = temp_folder / output_prefix
            else:
                beginning_prefix = temp_folder
            current_file = beginning_prefix / ecoregion_id / get_config_item('BLUETOPO', 'SUBFOLDER') / obj_summary.key
            # We need the main raster (.tiff/.tif) and the attribute table metadata (.aux.xml)
            if current_file.suffix in ('.tiff', '.tif', '.xml'):
                tile_folder = current_file.parents[0]
                tile_folder.mkdir(parents=True, exist_ok=True)   
                
                max_retries = 3
                attempt = 0
                is_valid = False
                
                while attempt < max_retries and not is_valid:
                    if attempt == 0:
                        self.write_message(f'Downloading: {current_file.name}', output_folder)
                    else:
                        self.write_message(f'Retrying Download: {current_file.name} (Attempt {attempt+1}/{max_retries})', output_folder)
                        
                    try:
                        if current_file.exists():
                            current_file.unlink() # Remove partial/corrupt file before retry
                            
                        nbs_bucket.download_file(obj_summary.key, str(current_file))
                        
                        # Deeply validate the TIFF to prevent ZIPDecode errors later
                        if current_file.suffix in ('.tiff', '.tif'):
                            if self._is_valid_tiff(current_file):
                                is_valid = True
                                output_tile_path = current_file
                            else:
                                print(f"[{tile_id}] Corrupt TIFF (ZIPDecode Error) detected after download. Retrying...")
                                attempt += 1
                        else:
                            is_valid = True # XML files assume valid if download succeeds
                            
                    except Exception as e:
                        print(f"[{tile_id}] Download error for {current_file.name}: {e}")
                        attempt += 1

                if not is_valid:
                    error_msg = f"[{tile_id}] FAILED to download a valid, uncorrupted copy of {current_file.name} after {max_retries} attempts. Skipping tile."
                    print(error_msg)
                    self.write_message(error_msg, output_folder)
                    if current_file.suffix in ('.tiff', '.tif'):
                        return False # Fail the whole tile processing if the base TIFF is fatally corrupt

        if not found_files:
            error_msg = f"[{tile_id}] CRITICAL: No source files found in NBS S3 bucket for prefix 'BlueTopo/{tile_id}'. Tile missing."
            print(error_msg)
            self.write_message(error_msg, output_folder)
            return False

        return output_tile_path

    def check_s3_existing_files(self, output_prefix: str|bool, target_res: int, ecoregion_id: str, tile_id: str, overwrite: bool=False) -> dict:
        """Check which derivatives already exist in S3 for a tile"""
        existing = {
            'base': False,
            'iss_110': False,
            'survey_end_date': False,
            'hurricane': False,
            'unc': False
        }

        if overwrite:
            return existing
            
        s3_files = s3fs.S3FileSystem()
        if output_prefix == 'low_res':
            beginning_prefix = f'{output_prefix}/{target_res}m/{ecoregion_id}'
        elif output_prefix:
            beginning_prefix = f'{output_prefix}/{ecoregion_id}'
        else:
            beginning_prefix = ecoregion_id
            
        bluetopo_tile_folder = f'{get_config_item("SHARED", "OUTPUT_BUCKET")}/{beginning_prefix}/{get_config_item("BLUETOPO", "SUBFOLDER")}/BlueTopo/{tile_id}'
        search_path = f"{bluetopo_tile_folder}/**/*.tiff"
        found_files = s3_files.glob(search_path)
        
        for file in found_files:
            filename = pathlib.Path(file).name
            # Exclude cropped temporary versions if they somehow got uploaded
            if filename.startswith('crop_') or filename.startswith('temp_'):
                continue
                
            if len(filename.split('_')) == 3:
                existing['base'] = True
            elif filename.endswith('_ISS_all_110.tiff'):
                existing['iss_110'] = True
            elif filename.endswith('_survey_end_date.tiff'):
                existing['survey_end_date'] = True
            elif filename.endswith('_hurricane.tiff'):
                existing['hurricane'] = True
            elif filename.endswith('_unc.tiff'):
                existing['unc'] = True
                
        return existing

    def finalize_cog(self, tiff_path: pathlib.Path) -> None:
        """The final pass to ensure perfect COG layout and overviews."""

        print(f"[{tiff_path.name}] Finalizing COG format...")
        temp_cog = tiff_path.parent / f"temp_{tiff_path.name}"

        ds = gdal.Open(str(tiff_path), gdal.GA_Update)
        if ds is not None:
            ds.BuildOverviews("BILINEAR", [2, 4, 8, 16])
            ds = None # EXPLICIT CLEANUP

        ds_trans = gdal.Translate(
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
        ds_trans = None # EXPLICIT CLEANUP

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

        ds_trans = gdal.Translate(
            str(singleband_tile_name),
            str(tiff_file_path),
            bandList=[band],
            creationOptions=["COMPRESS=DEFLATE"]
        )
        ds_trans = None # EXPLICIT CLEANUP

    def rename_multiband(self, tiff_file_path: pathlib.Path) -> pathlib.Path:
        """Update file name for singleband conversion"""

        new_name = str(tiff_file_path).replace('.tiff', '_mb.tiff')
        mb_tiff_file = tiff_file_path.replace(pathlib.Path(new_name))
        return mb_tiff_file

    def warp_bluetopo_tile(self, tiff_file_path: pathlib.Path, target_res: int) -> None:
        """
        Safely reprojects a 3-band BlueTopo tile to EPSG:6350.
        Uses Bilinear for Elevation (B1) & Uncertainty (B2), 
        and NearestNeighbour for the Contributor metadata (B3) to prevent category corruption.
        """
        ds = gdal.Open(str(tiff_file_path))
        if ds is None: return
        
        proj = ds.GetProjection()
        src_srs = osr.SpatialReference(wkt=proj)
        src_srs.AutoIdentifyEPSG()
        auth_code = src_srs.GetAuthorityCode(None)
        gt = ds.GetGeoTransform()
        x_res = abs(gt[1])
        ds = None # EXPLICIT CLEANUP
        
        # If it's already exactly EPSG:6350 AND the correct resolution, no warp needed
        if str(auth_code) == "6350" and abs(x_res - target_res) < 0.1:
            print(f"[{tiff_file_path.name}] Tile is already {self.target_crs} at {target_res}m. Skipping warp.")
            return

        print(f"[{tiff_file_path.name}] Reprojecting and resampling base tile to {self.target_crs} at {target_res}m...")
        
        # Keep original XML safe
        xml_path = tiff_file_path.parent / f"{tiff_file_path.name}.aux.xml"
        safe_xml = tiff_file_path.parent / "safe.xml"
        if xml_path.exists():
            import shutil
            shutil.copy(xml_path, safe_xml)
            
        temp_b12_src = tiff_file_path.parent / f"b12_src_{tiff_file_path.name}"
        temp_b3_src = tiff_file_path.parent / f"b3_src_{tiff_file_path.name}"
        temp_b12_warp = tiff_file_path.parent / f"b12_warp_{tiff_file_path.name}"
        temp_b3_warp = tiff_file_path.parent / f"b3_warp_{tiff_file_path.name}"
        final_warp = tiff_file_path.parent / f"warp_{tiff_file_path.name}"
        
        # Extract the bands using gdal.Translate
        ds1 = gdal.Translate(str(temp_b12_src), str(tiff_file_path), bandList=[1, 2], creationOptions=["COMPRESS=DEFLATE"])
        ds1 = None
        ds2 = gdal.Translate(str(temp_b3_src), str(tiff_file_path), bandList=[3], creationOptions=["COMPRESS=DEFLATE"])
        ds2 = None
        
        # Warp Bands 1 & 2 (Bilinear)
        ds3 = gdal.Warp(str(temp_b12_warp), str(temp_b12_src), options=gdal.WarpOptions(
            format="GTiff", dstSRS=self.target_crs, xRes=target_res, yRes=target_res,
            resampleAlg=gdal.GRA_Bilinear, targetAlignedPixels=True,
            creationOptions=["COMPRESS=DEFLATE"]
        ))
        ds3 = None
        
        # Warp Band 3 (Nearest Neighbor)
        ds4 = gdal.Warp(str(temp_b3_warp), str(temp_b3_src), options=gdal.WarpOptions(
            format="GTiff", dstSRS=self.target_crs, xRes=target_res, yRes=target_res,
            resampleAlg=gdal.GRA_NearestNeighbour, targetAlignedPixels=True,
            creationOptions=["COMPRESS=DEFLATE"]
        ))
        ds4 = None
        
        # Combine them using rasterio
        if temp_b12_warp.exists() and temp_b3_warp.exists():
            with rasterio.open(temp_b12_warp) as src12, rasterio.open(temp_b3_warp) as src3:
                profile = src12.profile
                profile.update(count=3)
                
                with rasterio.open(final_warp, 'w', **profile) as dst:
                    dst.write(src12.read(1), 1)
                    dst.write(src12.read(2), 2)
                    dst.write(src3.read(1), 3)
            
            # Clean up all intermediates and replace original
            tiff_file_path.unlink()
            final_warp.rename(tiff_file_path)
            
            if temp_b12_src.exists(): temp_b12_src.unlink()
            if temp_b3_src.exists(): temp_b3_src.unlink()
            if temp_b12_warp.exists(): temp_b12_warp.unlink()
            if temp_b3_warp.exists(): temp_b3_warp.unlink()
            
            # Restore the RAT XML to the final warped file
            if safe_xml.exists():
                safe_xml.rename(tiff_file_path.parent / f"{tiff_file_path.name}.aux.xml")

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""
        for result in results:
            if result:
                self.write_message(result, output_folder)

    def create_combined_isobaths(self, ecoregion_ids: list[str], s3_bucket: str, current_res: int, output_prefix: str|bool) -> None:
        """
        Creates solid polygon isobaths for specified depth bands (0-20m and 0-40m)
        and appends them directly to the local isobath_layers.gpkg.
        
        Uses an efficient tile-by-tile memory strategy to prevent VRT rendering dropouts,
        S3 connection thrashing, and massive disk usage.
        """

        if current_res == 20:
            print(f"[All Regions] Target resolution ({current_res}m) is 20m. Skipping isobath generation.")
            return

        # Changed to save to a separate newly created GPKG
        isobath_gpkg_path = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/isobath_layers.gpkg")

        print(f"[All Regions] Generating combined polygon isobaths and saving to local {isobath_gpkg_path}...")
        s3_client = boto3.client('s3')

        # Optimize GDAL for S3 reads: allows dynamic block chunking directly from the web
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gdal.SetConfigOption('VSI_CACHE', 'YES')

        driver = ogr.GetDriverByName("GPKG")
        if isobath_gpkg_path.exists():
            isobath_ds = driver.Open(str(isobath_gpkg_path), 1)  # 1 indicates update mode
        else:
            isobath_ds = driver.CreateDataSource(str(isobath_gpkg_path))

        if isobath_ds is None:
            print(f"Error: Could not open or create {isobath_gpkg_path}")
            return

        srs = osr.SpatialReference()
        srs.SetFromUserInput(self.target_crs)

        # We process negative depths. 0 to -20 is the first band, 0 to -40 is the second.
        depth_bands = [
            ((0.0, -20.0), 'isobath_0_20m'),
            ((0.0, -40.0), 'isobath_0_40m')
        ]

        # Check if layers already exist to skip recreation
        existing_layers = [isobath_ds.GetLayerByIndex(i).GetName() for i in range(isobath_ds.GetLayerCount())]
        expected_layers = [layer_name for _, layer_name in depth_bands]

        # Skip if they exist
        if all(layer in existing_layers for layer in expected_layers):
            print(f"[All Regions] Isobath layers already exist in {isobath_gpkg_path}. Skipping creation.")
            isobath_ds = None
            return

        # Ensure layers exist in the isobath_layers.gpkg as POLYGONS (unconditionally overwrite if we reach here)
        layers = {}
        for _, layer_name in depth_bands:
            # Safely delete layer if it already exists by checking integer indices
            for i in range(isobath_ds.GetLayerCount()):
                if isobath_ds.GetLayerByIndex(i).GetName() == layer_name:
                    isobath_ds.DeleteLayer(i)
                    break

            layer = isobath_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
            field_defn = ogr.FieldDefn("elevation", ogr.OFTReal)
            layer.CreateField(field_defn)
            layers[layer_name] = layer

        # Process one ecoregion at a time
        for ecoregion_id in ecoregion_ids:
            print(f"[{ecoregion_id}] Gathering S3 links...")
            search_prefix = f"low_res/{current_res}m/{ecoregion_id}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            if output_prefix and output_prefix.strip('/') != "low_res":
                search_prefix = f"{output_prefix}/{search_prefix}"

            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix)

            vsis3_paths = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Strictly look for files ending with digits + .tiff (case insensitive)
                        if re.search(r'_[0-9]{8}\.tiff?$', key, re.IGNORECASE):
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

        print(f"[All Regions] Saving and securely closing {isobath_gpkg_path}...")
        layers.clear()
        isobath_ds = None

    def _download_vsis3_paths_concurrently(self, vsis3_paths: list[str], temp_dir: pathlib.Path) -> list[str]:
        """
        [OPTION 1 FIX] Bulk download tiles from S3 to local disk before GDAL processing.
        Significantly reduces network thrashing inside the GDAL Warp operations.
        """
        local_paths = []
        download_tasks = []

        for vsi_path in vsis3_paths:
            parts = vsi_path.replace("/vsis3/", "").split("/", 1)
            if len(parts) == 2:
                bucket, key = parts
                # Flatten the path to prevent folder creation issues and guarantee uniqueness
                safe_filename = key.replace("/", "_")
                local_path = temp_dir / safe_filename
                download_tasks.append((bucket, key, local_path))
        
        # FIXED: Initialize one single globally shared Boto3 client context pool
        s3_pool_client = boto3.client('s3', config=Config(max_pool_connections=32))

        def _download_task(task_info):
            bucket, key, local_path = task_info
            if not local_path.exists():
                s3_pool_client.download_file(bucket, key, str(local_path))
            return str(local_path)

        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"      [S3 Pre-Download] FAST FETCH: Fetching {len(download_tasks)} files locally to eliminate S3 warp latency... (RAM: {mem_mb:.1f}MB)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(_download_task, task) for task in download_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    local_paths.append(future.result())
                except Exception as e:
                    print(f"      [S3 Pre-Download] WARNING: File download failed: {e}")

        print(f"      [S3 Pre-Download] Complete. {len(local_paths)} files perfectly staged locally.")
        
        # Clear out Boto3 client connections to free RAM
        del s3_pool_client
        gc.collect()
        
        return local_paths

    def _build_local_mosaic(self, target_path: pathlib.Path, paths: list[str], warp_kwargs: dict) -> None:
        """
        Process a massive list of inputs locally using a highly-optimized Two-Step Translate/Warp Pipeline.
        Step 1: Translate VRT -> Stitched Temp TIFF (Very Fast, no vector intersection math)
        Step 2: Warp Temp TIFF -> Final TIFF (Applies cutline precisely on a single file, rather than 700+)
        """
        gdal.UseExceptions()
        output_folder = self.param_lookup['output_directory'].valueAsText if 'output_directory' in self.param_lookup else None

        with tempfile.TemporaryDirectory(dir=target_path.parent) as temp_dir:
            temp_dir_path = pathlib.Path(temp_dir)
            
            # 1. PULL EVERYTHING TO LOCAL FAST STORAGE FIRST
            local_paths = self._download_vsis3_paths_concurrently(paths, temp_dir_path)

            if not local_paths:
                raise RuntimeError("No files were successfully downloaded to build the mosaic.")

            print(f"      [Mosaic Final Render] Building Virtual Raster (VRT) for {len(local_paths)} local files...")
            final_vrt = str(temp_dir_path / "final_merge.vrt")
            
            # Use the provided warp arguments directly (now handles clipping + mosaicking simultaneously)
            warp_kwargs_base = warp_kwargs.copy()
            
            # Extract final-stage only parameters for vector masking so they don't break the Translate phase
            cutline_ds = warp_kwargs_base.pop("cutlineDSName", None)
            cutline_layer = warp_kwargs_base.pop("cutlineLayer", None)
            cutline_where = warp_kwargs_base.pop("cutlineWhere", None)
            crop_to_cutline = warp_kwargs_base.pop("cropToCutline", False)

            vrt_options = gdal.BuildVRTOptions(
                resampleAlg=warp_kwargs_base.get("resampleAlg", gdal.GRA_NearestNeighbour),
                srcNodata=warp_kwargs_base.get("srcNodata", -9999),
                VRTNodata=warp_kwargs_base.get("dstNodata", -9999)
            )
            vrt_ds = gdal.BuildVRT(final_vrt, local_paths, options=vrt_options)
            vrt_ds = None # Must flush to disk, or VRT remains cached indefinitely in RAM

            # Custom progress callback function for GDAL
            def custom_progress_callback(complete, message, cb_data):
                percent = int(complete * 100)
                if percent > cb_data['last_printed']:
                    print(f"        -> {cb_data['step_name']} Progress: {percent}%")
                    sys.stdout.flush() # Force output to console immediately so it doesn't look hung
                    cb_data['last_printed'] = percent
                return 1

            solid_tiff = str(temp_dir_path / "solid_mosaic.tiff")

            print(f"      [Mosaic Final Render] Executing Step 1/2: Stitching 700+ tiles into a single master raster... (Fast Translate)")
            try:
                # Step 1: Rapidly stitch without worrying about vector math bounds
                cb_data_1 = {'last_printed': -1, 'step_name': 'Stitching'}
                trans_opts = gdal.TranslateOptions(
                    format="GTiff",
                    creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"],
                    noData=-9999,
                    callback=custom_progress_callback,
                    callback_data=cb_data_1
                )
                trans_ds = gdal.Translate(solid_tiff, final_vrt, options=trans_opts)
                trans_ds = None
                
                print(f"      [Mosaic Final Render] Executing Step 2/2: Applying Vector Mask... (Single File Warp)")
                # Step 2: Now that it's a single file, safely apply the vector mask without crashing GDAL threading
                if cutline_ds:
                    mask_kwargs = warp_kwargs_base.copy()
                    mask_kwargs["cutlineDSName"] = cutline_ds
                    if cutline_layer:
                        mask_kwargs["cutlineLayer"] = cutline_layer
                    if cutline_where:
                        mask_kwargs["cutlineWhere"] = cutline_where
                    mask_kwargs["cropToCutline"] = crop_to_cutline
                    
                    # Prevent I/O thread lockup, the single file warp is natively fast enough
                    mask_kwargs["multithread"] = False 
                    
                    cb_data_2 = {'last_printed': -1, 'step_name': 'Masking'}
                    warp_ds = gdal.Warp(
                        str(target_path), 
                        solid_tiff, 
                        options=gdal.WarpOptions(**mask_kwargs), 
                        callback=custom_progress_callback, 
                        callback_data=cb_data_2
                    )
                    warp_ds = None
                else:
                    # Fallback if no mask was provided
                    shutil.copy(solid_tiff, str(target_path))

            except Exception as e:
                warn_msg = f"        -> FATAL: Mosaic stitch/mask failed. Error: {e}"
                print(warn_msg)
                if output_folder: self.write_message(warn_msg, output_folder)
                raise e
                
        # Force Python garbage collection
        gc.collect()

    def create_and_upload_er_mosaics(self, ecoregion_ids: list[str], s3_bucket: str, current_res: int, output_prefix: str|bool, increased_scale: bool=False) -> None:
        """
        Creates isolated mosaics for each individual ecoregion and precisely masks them
        to their specific boundary polygon from the EcoRegions layer.
        """

        low_res_dir = OUTPUTS / "low_res" / f'{current_res}m'
        low_res_dir.mkdir(parents=True, exist_ok=True)

        master_gpkg_path = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

        s3_client = boto3.client('s3')

        # Flattened S3 Pagination: Page bucket ONCE instead of inside loop
        print(f"[ER Mosaic] Paginating S3 to gather all keys upfront to minimize API calls...")
        search_prefix_base = f"low_res/{current_res}m/"
        if output_prefix and output_prefix.strip('/') != "low_res":
            search_prefix_base = f"{output_prefix}/{search_prefix_base}"
        
        all_s3_keys = []
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix_base)
        
        for page_num, page in enumerate(pages, start=1):
            if 'Contents' in page:
                for obj in page['Contents']:
                    all_s3_keys.append(obj['Key'])
            
            # Print S3 pagination progress
            if page_num % 10 == 0:
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                print(f"      [S3 Paginator] Scanned {page_num} pages, found {len(all_s3_keys)} keys so far... (RAM: {mem_mb:.1f}MB)")

        print(f"[ER Mosaic] Finished scanning S3. Total keys retrieved: {len(all_s3_keys)}")
        all_s3_keys_set = set(all_s3_keys)

        # Locate the EcoRegions layer once for masking
        cutline_layer_name = None
        er_field_name = None

        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            for name in layer_names:
                if name.lower().replace("_", "").replace(" ", "") == "enhancedecoregions":
                    cutline_layer_name = name
                    break

            # Intelligently discover the column name containing the Ecoregion ID
            if cutline_layer_name:
                layer = ds.GetLayerByName(cutline_layer_name)
                layer_defn = layer.GetLayerDefn()
                field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

                # Prioritize field names to avoid grabbing a generic 'id' if 'ecoregion' exists
                for expected in ['ecoregion_id', 'ecoregion', 'er', 'region', 'name', 'id']:
                    for field in field_names:
                        if field.lower() == expected:
                            er_field_name = field
                            break
                    if er_field_name:
                        break
            ds = None
        else:
            print(f"[ER Mosaic] WARNING: Could not open {master_gpkg_path}. Proceeding without mask...")

        # Base warp arguments allowing GDAL to naturally unify projections perfectly
        # Set targetAlignedPixels to perfectly align the grids eliminating edge artifact grid lines
        # FIXED: Explicitly reduced warpMemoryLimit to 512MB to prevent OS Swap thrashing
        # FIXED: Replaced ALL_CPUS with 4, stopping the 16/32-core EC2 RAM explosions!
        warp_kwargs_base = {
            "format": "GTiff",
            "dstSRS": self.target_crs,
            "xRes": current_res,
            "yRes": current_res,
            "srcNodata": -9999,
            "dstNodata": -9999,
            "targetAlignedPixels": True, 
            "multithread": False, # Stop complex Vector/Thread lockups here
            "warpOptions": ["NUM_THREADS=4"], # Safe chunk-based threading (prevents vector deadlock)
            "warpMemoryLimit": 512 * 1024 * 1024, # 512MB Limit to brutally stop OS RAM Thrashing
            "creationOptions": ["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
        }

        for ecoregion_id in ecoregion_ids:
            # -------------------------------------------------------------------------------------
            # SKIP LOGIC: Only generate ER mosaics for ER 6 as requested
            # -------------------------------------------------------------------------------------
            match = re.search(r'\d+', str(ecoregion_id))
            clean_er_num = match.group(0) if match else str(ecoregion_id)

            if clean_er_num not in ["6"]:
                print(f"[ER Mosaic] Skipping mosaic generation for {ecoregion_id} as only ER 6 is requested.")
                continue
            # -------------------------------------------------------------------------------------

            er_prefix = ecoregion_id if str(ecoregion_id).startswith("ER_") else f"ER_{ecoregion_id}"

            # Apply a specific SQL WHERE clause to the vector file to grab ONLY this Ecoregion's polygon
            er_warp_kwargs = warp_kwargs_base.copy()
            temp_mask_path = str(low_res_dir / f"{ecoregion_id}_mask.geojson")

            if cutline_layer_name and er_field_name:
                # Robust extraction of the ecoregion number to prevent ER_1/ER1 split matching errors
                er_prefix_val = f"ER_{clean_er_num}"
                er_val_no_underscore = f"ER{clean_er_num}"

                # Build a robust SQL query that perfectly matches any possible GPKG ID format
                where_clauses = [
                    f"{er_field_name} = '{er_prefix_val}'",
                    f"{er_field_name} = '{er_val_no_underscore}'",
                    f"{er_field_name} = '{ecoregion_id}'",
                    f"{er_field_name} = '{clean_er_num}'"
                ]
                if clean_er_num.isdigit():
                    where_clauses.append(f"{er_field_name} = {clean_er_num}")
                    
                print(f"[{ecoregion_id}] Extracting vector mask to memory-safe GeoJSON to prevent SQLite thread locking...")
                ds_gpkg = ogr.Open(str(master_gpkg_path))
                if ds_gpkg:
                    layer = ds_gpkg.GetLayerByName(cutline_layer_name)
                    layer.SetAttributeFilter(" OR ".join(where_clauses))
                    
                    if os.path.exists(temp_mask_path):
                        os.remove(temp_mask_path)
                        
                    drv = ogr.GetDriverByName("GeoJSON")
                    out_ds = drv.CreateDataSource(temp_mask_path)
                    out_layer = out_ds.CreateLayer("mask", srs=layer.GetSpatialRef(), geom_type=layer.GetGeomType())
                    
                    for feat in layer:
                        out_layer.CreateFeature(feat)
                        
                    out_ds = None
                    ds_gpkg = None
                    
                    er_warp_kwargs["cutlineDSName"] = temp_mask_path
                    er_warp_kwargs["cropToCutline"] = True

            # Map configuration properties dictating regex mapping
            # REMOVED Slope from base config to calculate at the mosaic level
            base_mosaics_config = {
                "Bathy": re.compile(r'_[0-9]{8}\.tiff?$', re.IGNORECASE),
                "ISS": re.compile(rf"_ISS_all{'_110' if increased_scale else ''}\.tiff?$", re.IGNORECASE),
                "Survey_Date": re.compile(r'_survey_end_date\.tiff?$', re.IGNORECASE),
                "Hurricane": re.compile(r'_hurricane\.tiff?$', re.IGNORECASE),
            }

            mosaics_config = {}
            for name, regex in base_mosaics_config.items():
                # Omit BlueTopo prefix if we are generating Hurricane mosaics
                bt_prefix = "" if name.lower() == "hurricane" else "BlueTopo_"

                mosaics_config[name] = {
                    "regex": regex,
                    "filename": f"{er_prefix}_{bt_prefix}{name}_Mosaic_{current_res}m{'_110' if increased_scale and name == 'ISS' else ''}.tiff",
                    "paths": [],
                    "exclude_nearshore_bands": False
                }

            mosaics_to_process = mosaics_config
            for m_key, config in mosaics_to_process.items():
                s3_key = f"low_res/{current_res}m/{ecoregion_id}/{config['filename']}"
                if output_prefix and output_prefix.strip('/') != "low_res":
                    s3_key = f"{output_prefix}/{s3_key}"
                config['s3_key'] = s3_key

            print(f"[{ecoregion_id}] Mapping local S3 keys for ER specific mosaics: {list(mosaics_to_process.keys())}")
            
            # Use dynamic folder checking to ensure robust mapping whether it's '1' or 'ER_1'
            valid_folders = [f"/{ecoregion_id}/", f"/ER_{clean_er_num}/", f"/ER{clean_er_num}/"]

            for key in all_s3_keys:
                if not any(f in f"/{key}" for f in valid_folders):
                    continue
                if f"/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo/" not in f"/{key}":
                    continue
                    
                is_nearshore = bool(re.search(r'/BH[45]', key, re.IGNORECASE))
                for m_key, config in mosaics_to_process.items():
                    if config['regex'].search(key):
                        if config.get('exclude_nearshore_bands') and is_nearshore:
                            continue
                        config['paths'].append(f"/vsis3/{s3_bucket}/{key}")

            # Process standard mosaics before offshore mosaics
            ordered_mosaics = sorted(mosaics_to_process.items(), key=lambda item: 1 if "Offshore" in item[0] else 0)
            for m_key, config in ordered_mosaics:
                if not config['paths']:
                    print(f"[{ecoregion_id}] No base tiles found for {m_key}. Skipping.")
                    continue

                local_mosaic_path = low_res_dir / config['filename']
                s3_key = config['s3_key']

                # Categorical components require nearest neighbor resampling
                m_key_lower = m_key.lower()
                
                # Check if mosaics already exist to skip heavy processing
                if m_key_lower == "bathy" or m_key_lower == "bathy_offshore":
                    slope_s3_key = s3_key.replace("Bathy", "Slope")
                    if s3_key in all_s3_keys_set and slope_s3_key in all_s3_keys_set:
                        print(f"[{ecoregion_id}] {m_key} and its Slope mosaic already exist in S3. Skipping.")
                        continue
                else:
                    if s3_key in all_s3_keys_set:
                        print(f"[{ecoregion_id}] {m_key} mosaic already exists in S3. Skipping.")
                        continue

                if "bathy" in m_key_lower or "unc" in m_key_lower:
                    resample_alg = gdal.GRA_Bilinear
                else:
                    resample_alg = gdal.GRA_NearestNeighbour

                current_warp_kwargs = er_warp_kwargs.copy()
                current_warp_kwargs["resampleAlg"] = resample_alg

                print(f"[{ecoregion_id}] Warping and precisely masking {m_key} ER mosaic dynamically from {len(config['paths'])} source files...")
                
                # Single pass directly to output
                try:
                    self._build_local_mosaic(local_mosaic_path, config['paths'], current_warp_kwargs)
                except Exception as e:
                    error_msg = f"[{ecoregion_id}] ERROR: Mosaic generation failed for {m_key}. Exception details: {e}"
                    print(error_msg)
                    if 'output_directory' in self.param_lookup:
                        self.write_message(error_msg, self.param_lookup['output_directory'].valueAsText)
                    continue # Safely skip to the next mosaic type instead of crashing the whole pipeline

                print(f"[{ecoregion_id}] Generating robust overviews for {m_key}...")
                ds = gdal.Open(str(local_mosaic_path), gdal.GA_Update)
                if ds is not None:
                    ds.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                    ds = None

                print(f"[{ecoregion_id}] Uploading {m_key} ER mosaic to s3://{s3_bucket}/{s3_key}...")
                s3_client.upload_file(
                    str(local_mosaic_path), 
                    s3_bucket, 
                    s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'resolution': str(current_res),
                            'crs': self.target_crs
                        }
                    }
                )
                print(f"[{ecoregion_id}] {m_key} ER mosaic creation and upload perfectly complete.")

                # Generate Slope directly from Bathy to prevent tile-edge artifacts
                if m_key_lower == "bathy":
                    slope_key = m_key.replace("Bathy", "Slope")
                    slope_filename = config['filename'].replace("Bathy", "Slope")
                    slope_local_path = low_res_dir / slope_filename
                    slope_s3_key = config['s3_key'].replace("Bathy", "Slope")
                    
                    print(f"[{ecoregion_id}] Generating {slope_key} mosaic directly from {m_key} mosaic to prevent edge artifacts...")
                    slope_ds = gdal.DEMProcessing(
                        str(slope_local_path), 
                        str(local_mosaic_path), 
                        'slope', 
                        creationOptions=["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
                    )
                    slope_ds = None # FIXED: Close dataset immediately
                    
                    print(f"[{ecoregion_id}] Generating robust overviews for {slope_key}...")
                    ds_slope = gdal.Open(str(slope_local_path), gdal.GA_Update)
                    if ds_slope is not None:
                        ds_slope.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                        ds_slope = None
                        
                    print(f"[{ecoregion_id}] Uploading completely masked {slope_key} mosaic to s3://{s3_bucket}/{slope_s3_key}...")
                    s3_client.upload_file(
                        str(slope_local_path), 
                        s3_bucket, 
                        slope_s3_key,
                        ExtraArgs={
                            'Metadata': {
                                'resolution': str(current_res),
                                'crs': self.target_crs
                            }
                        }
                    )
                    print(f"[{ecoregion_id}] {slope_key} ER mosaic creation and upload perfectly complete.")
                    if slope_local_path.exists():
                        slope_local_path.unlink()

                if local_mosaic_path.exists():
                    local_mosaic_path.unlink() # Save disk space
            
            # Clean up the temporary extracted mask for this ER
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
                
            gc.collect()

    def create_and_upload_mosaics(self, ecoregion_ids: list[str], s3_bucket: str, current_res: int, output_prefix: str|bool, increased_scale: bool=False) -> None:
        """
        Creates massive mosaics from all base bathy tiles and derivatives across all processed ecoregions,
        masks them securely with the EcoRegions layer, saves locally to low_res, and uploads to S3.
        """

        if current_res < 50:
            print(f"[Mosaic] Target resolution ({current_res}m) is under 50m. Skipping massive mosaic generation to prevent extreme file sizes.")
            return

        # Restrict processing to only ER 6 as requested
        ecoregion_ids = [er for er in ecoregion_ids if (re.search(r'\d+', str(er)).group(0) if re.search(r'\d+', str(er)) else str(er)) in ["6"]]
        if not ecoregion_ids:
            print("[Mosaic] No requested Ecoregions (6) found. Skipping mosaic generation.")
            return

        low_res_dir = OUTPUTS / "low_res" / f'{current_res}m'
        low_res_dir.mkdir(parents=True, exist_ok=True)

        master_gpkg_path = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

        s3_client = boto3.client('s3')

        # Configuration holding the regex mapping (All variables uncommented)
        # REMOVED Slope from base config to calculate at the mosaic level
        base_mosaics_config = {
            "Bathy": re.compile(r'_[0-9]{8}\.tiff?$', re.IGNORECASE),
            "ISS": re.compile(rf"_ISS_all{'_110' if increased_scale else ''}\.tiff?$", re.IGNORECASE),
            "Survey_Date": re.compile(r'_survey_end_date\.tiff?$', re.IGNORECASE),
            "Hurricane": re.compile(r'_hurricane\.tiff?$', re.IGNORECASE),
        }

        mosaics_config = {}
        for name, regex in base_mosaics_config.items():
            # Drop BlueTopo prefix if we are processing Hurricane
            bt_prefix = "" if name.lower() == "hurricane" else "BlueTopo_"

            # Commented out the standard national 100m mosaics as requested
            if current_res != 100:
                mosaics_config[name] = {
                    "regex": regex,
                    "filename": f"{bt_prefix}{name}_Mosaic_{current_res}m{'_110' if increased_scale and name == 'ISS' else ''}.tiff",
                    "paths": [],
                    "exclude_nearshore_bands": False
                }
            # Create a secondary non-Band4/5 version specifically for 100m runs
            if current_res == 100:
                mosaics_config[f"{name}_Offshore"] = {
                    "regex": regex,
                    "filename": f"{bt_prefix}{name}_Mosaic_Offshore_{current_res}m{'_110' if increased_scale and name == 'ISS' else ''}.tiff",
                    "paths": [],
                    "exclude_nearshore_bands": True
                }

        # Unconditionally process all configured mosaics
        mosaics_to_process = mosaics_config
        for m_key, config in mosaics_to_process.items():
            s3_key = f"low_res/{current_res}m/{config['filename']}"
            if output_prefix and output_prefix.strip('/') != "low_res":
                s3_key = f"{output_prefix}/{s3_key}"
            config['s3_key'] = s3_key

        print(f"[Mosaic] Gathering S3 links across {len(ecoregion_ids)} ecoregions for: {list(mosaics_to_process.keys())}")

        # Flattened S3 Pagination: Page bucket ONCE instead of inside loop
        search_prefix_base = f"low_res/{current_res}m/"
        if output_prefix and output_prefix.strip('/') != "low_res":
            search_prefix_base = f"{output_prefix}/{search_prefix_base}"
        
        all_s3_keys = []
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix_base)
        print(f"[Mosaic] Paginating S3 to gather all keys upfront to minimize API calls...")
        
        for page_num, page in enumerate(pages, start=1):
            if 'Contents' in page:
                for obj in page['Contents']:
                    all_s3_keys.append(obj['Key'])
                    
            # Print S3 pagination progress
            if page_num % 10 == 0:
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                print(f"      [S3 Paginator] Scanned {page_num} pages, found {len(all_s3_keys)} keys so far... (RAM: {mem_mb:.1f}MB)")

        print(f"[Mosaic] Finished scanning S3. Total keys retrieved: {len(all_s3_keys)}")
        all_s3_keys_set = set(all_s3_keys)

        # 2. Gather source file paths in a single efficient pass through S3 per ecoregion
        for ecoregion_id in ecoregion_ids:
            
            # Use dynamic folder checking to ensure robust mapping whether it's '1' or 'ER_1'
            match = re.search(r'\d+', str(ecoregion_id))
            clean_er_num = match.group(0) if match else str(ecoregion_id)
            valid_folders = [f"/{ecoregion_id}/", f"/ER_{clean_er_num}/", f"/ER{clean_er_num}/"]

            for key in all_s3_keys:
                if not any(f in f"/{key}" for f in valid_folders):
                    continue
                if f"/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo/" not in f"/{key}":
                    continue

                is_nearshore = bool(re.search(r'/BH[45]', key, re.IGNORECASE))
                for m_key, config in mosaics_to_process.items():
                    if config['regex'].search(key):
                        if config.get('exclude_nearshore_bands') and is_nearshore:
                            continue
                        config['paths'].append(f"/vsis3/{s3_bucket}/{key}")

        # 3. Locate the EcoRegions layer once for masking
        cutline_layer_name = None
        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            for name in layer_names:
                # Match case-insensitively, allowing for underscores or spaces
                if name.lower().replace("_", "").replace(" ", "") == "enhancedecoregions":
                    cutline_layer_name = name
                    break

            if not cutline_layer_name:
                print(f"[Mosaic] WARNING: Could not find an 'Enhanced_EcoRegions' layer in {master_gpkg_path}.")
                print(f"[Mosaic] Available layers: {layer_names}")
                print("[Mosaic] Proceeding without vector mask to prevent crash...")

            ds = None
        else:
            print(f"[Mosaic] WARNING: Could not open {master_gpkg_path} to check layers. Proceeding without mask...")

        temp_mask_path = str(low_res_dir / "full_mask.geojson")
        if cutline_layer_name:
            print(f"[Mosaic] Extracting full vector mask to memory-safe GeoJSON to prevent SQLite thread locking...")
            ds_gpkg = ogr.Open(str(master_gpkg_path))
            if ds_gpkg:
                layer = ds_gpkg.GetLayerByName(cutline_layer_name)
                
                if os.path.exists(temp_mask_path):
                    os.remove(temp_mask_path)
                    
                drv = ogr.GetDriverByName("GeoJSON")
                out_ds = drv.CreateDataSource(temp_mask_path)
                out_layer = out_ds.CreateLayer("mask", srs=layer.GetSpatialRef(), geom_type=layer.GetGeomType())
                
                for feat in layer:
                    out_layer.CreateFeature(feat)
                    
                out_ds = None
                ds_gpkg = None

        # Base warp arguments allowing GDAL to naturally unify projections perfectly
        # Set targetAlignedPixels to perfectly align the grids eliminating edge artifact grid lines
        # FIXED: Explicitly reduced warpMemoryLimit to 512MB to prevent OS Swap thrashing
        warp_kwargs_base = {
            "format": "GTiff",
            "dstSRS": self.target_crs,
            "xRes": current_res,
            "yRes": current_res,
            "srcNodata": -9999,
            "dstNodata": -9999,
            "targetAlignedPixels": True,
            "multithread": False, # Stop complex Vector/Thread lockups here
            "warpOptions": ["NUM_THREADS=4"], # Safe chunk-based threading (prevents vector deadlock)
            "warpMemoryLimit": 512 * 1024 * 1024, # 512MB Limit to brutally stop OS RAM Thrashing
            "creationOptions": ["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
        }

        if cutline_layer_name:
            warp_kwargs_base["cutlineDSName"] = temp_mask_path
            warp_kwargs_base["cropToCutline"] = True
            # For the combined 100m all-tiles mosaic, we intentionally DO NOT pass a cutlineWhere clause.
            # This forces GDAL to utilize every single polygon inside the Enhanced_EcoRegions layer 
            # to mask the entire dataset simultaneously, avoiding SQL OR-clause limit truncations 
            # which cause some tiles to miss their mask.

        # 4. Generate & Upload each required mosaic dynamically
        # Process standard mosaics before offshore mosaics
        ordered_mosaics = sorted(mosaics_to_process.items(), key=lambda item: 1 if "Offshore" in item[0] else 0)
        for m_key, config in ordered_mosaics:
            if not config['paths']:
                print(f"[Mosaic] No base tiles found for {m_key}. Skipping.")
                continue

            local_mosaic_path = low_res_dir / config['filename']
            s3_key = config['s3_key']

            # Categorical components require nearest neighbor resampling
            m_key_lower = m_key.lower()
            
            # Check if mosaics already exist to skip heavy processing
            if m_key_lower == "bathy" or m_key_lower == "bathy_offshore":
                slope_s3_key = s3_key.replace("Bathy", "Slope")
                if s3_key in all_s3_keys_set and slope_s3_key in all_s3_keys_set:
                    print(f"[Mosaic] {m_key} and its Slope mosaic already exist in S3. Skipping.")
                    continue
            else:
                if s3_key in all_s3_keys_set:
                    print(f"[Mosaic] {m_key} mosaic already exists in S3. Skipping.")
                    continue

            if "bathy" in m_key_lower or "unc" in m_key_lower:
                resample_alg = gdal.GRA_Bilinear
            else:
                resample_alg = gdal.GRA_NearestNeighbour

            current_warp_kwargs = warp_kwargs_base.copy()
            current_warp_kwargs["resampleAlg"] = resample_alg

            print(f"[Mosaic] Warping and masking {m_key} mosaic directly from {len(config['paths'])} source files...")

            # Single pass directly to output
            try:
                self._build_local_mosaic(local_mosaic_path, config['paths'], current_warp_kwargs)
            except Exception as e:
                error_msg = f"[Mosaic] ERROR: Massive mosaic generation failed for {m_key}. Exception details: {e}"
                print(error_msg)
                if 'output_directory' in self.param_lookup:
                    self.write_message(error_msg, self.param_lookup['output_directory'].valueAsText)
                continue # Safely skip to the next mosaic type instead of crashing the whole pipeline

            print(f"[Mosaic] Generating robust overviews for fast performance...")
            ds = gdal.Open(str(local_mosaic_path), gdal.GA_Update)
            if ds is not None:
                ds.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                ds = None

            print(f"[Mosaic] Uploading completely masked {m_key} mosaic to s3://{s3_bucket}/{s3_key}...")
            s3_client.upload_file(
                str(local_mosaic_path), 
                s3_bucket, 
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'resolution': str(current_res),
                        'crs': self.target_crs
                    }
                }
            )
            print(f"[Mosaic] {m_key} mosaic creation and upload perfectly complete.")

            # Generate Slope directly from Bathy to prevent tile-edge artifacts
            if m_key_lower == "bathy" or m_key_lower == "bathy_offshore":
                slope_key = m_key.replace("Bathy", "Slope")
                slope_filename = config['filename'].replace("Bathy", "Slope")
                slope_local_path = low_res_dir / slope_filename
                slope_s3_key = config['s3_key'].replace("Bathy", "Slope")
                
                print(f"[Mosaic] Generating {slope_key} mosaic directly from {m_key} mosaic to prevent edge artifacts...")
                slope_ds = gdal.DEMProcessing(
                    str(slope_local_path), 
                    str(local_mosaic_path), 
                    'slope', 
                    creationOptions=["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
                )
                slope_ds = None # FIXED: Close dataset immediately
                
                print(f"[Mosaic] Generating robust overviews for {slope_key}...")
                ds_slope = gdal.Open(str(slope_local_path), gdal.GA_Update)
                if ds_slope is not None:
                    ds_slope.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                    ds_slope = None
                    
                print(f"[Mosaic] Uploading completely masked {slope_key} mosaic to s3://{s3_bucket}/{slope_s3_key}...")
                s3_client.upload_file(
                    str(slope_local_path), 
                    s3_bucket, 
                    slope_s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'resolution': str(current_res),
                            'crs': self.target_crs
                        }
                    }
                )
                print(f"[Mosaic] {slope_key} mosaic creation and upload perfectly complete.")
                if slope_local_path.exists():
                    slope_local_path.unlink()

            if local_mosaic_path.exists():
                local_mosaic_path.unlink() # Save disk space
        
        # Cleanup full mask
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)
            
        gc.collect()

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str|bool, resolution: list[int] = None) -> None:
        # Force resolution to 100m only to remake offshore mosaics
        resolution = [100]
        
        output_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        # Standardize the ecoregions mapped in param_lookup globally to "ER_#"
        all_ecoregions_raw = self.param_lookup['eco_regions'].value
        all_ecoregions = []
        for er in all_ecoregions_raw:
            match = re.search(r'\d+', str(er))
            all_ecoregions.append(f"ER_{match.group(0)}" if match else str(er))

        # Sort ecoregions in numerical order (ER1 to ER6)
        def _get_er_num(er_str):
            match = re.search(r'\d+', str(er_str))
            return int(match.group(0)) if match else 9999
            
        all_ecoregions = sorted(list(set(all_ecoregions)), key=_get_er_num)
        
        # Restrict to only ER 6
        all_ecoregions = [er for er in all_ecoregions if str(_get_er_num(er)) in ["6"]]
        
        print(f"[BlueTopo Engine] Processing ecoregions in sequential order (Restricted to ER 6): {all_ecoregions}")
        
        if not all_ecoregions:
            print("[BlueTopo Engine] No requested Ecoregions (ER 6) found in input. Exiting run.")
            return

        self.setup_dask(self.param_lookup['env'], threads_per_worker=1)
        for current_res in resolution:
            print(f"\n{'='*50}\n[BlueTopo Engine] STARTING PROCESSING FOR {current_res}m\n{'='*50}")

            if not self.skip_tiling:
                print(f'Checking and processing BlueTopo Datasets for {current_res}...')
                param_inputs = []

                # Determine the correct columns in tile_gdf dynamically
                tile_col = None
                er_col = None

                for col in tile_gdf.columns:
                    col_lower = str(col).lower()
                    if col_lower in ['tile', 'tile_id', 'name', 'id', 'bluetopo']:
                        tile_col = col
                    if col_lower in ['ecoregion', 'ecoregion_id', 'er', 'region', 'eco_region']:
                        er_col = col

                # Fallback by inspecting data if names don't match
                if not tile_col or not er_col:
                    if not tile_gdf.empty:
                        for col in tile_gdf.columns:
                            val = str(tile_gdf.iloc[0][col]).upper()
                            if not tile_col and re.search(r'(B[A-Z0-9]{7})', val):
                                tile_col = col
                            elif not er_col and ('ER_' in val or str(val).isdigit()):
                                er_col = col

                # Absolute fallback to original behavior if everything fails
                if not tile_col: tile_col = tile_gdf.columns[0]
                if not er_col: er_col = tile_gdf.columns[1] if len(tile_gdf.columns) > 1 else tile_gdf.columns[0]
                
                print(f"[BlueTopo Engine] Dynamically mapped Tile column: '{tile_col}' and EcoRegion column: '{er_col}'")

                for _, row in tile_gdf.iterrows():
                    # Handle raw numerical data safely using the dynamically found column names
                    if pd.isna(row[er_col]):
                        continue
                    
                    # Normalize whatever the user inputted (1, "1.0", "ER_1") securely into "ER_1"
                    raw_er = str(row[er_col]).strip()
                    er_match = re.search(r'\d+', raw_er)
                    normalized_er = f"ER_{er_match.group(0)}" if er_match else raw_er
                    
                    if normalized_er not in all_ecoregions:
                        continue
                    
                    raw_id = str(row[tile_col]).strip()
                    # Remove 'BlueTopo' to prevent the regex from accidentally matching the word 'BlueTopo' instead of the ID
                    clean_id = raw_id.upper().replace('BLUETOPO_', '').replace('BLUETOPO', '')
                    
                    # Force extract the standard 8-character ID (e.g., BH4K857L)
                    match = re.search(r'(B[A-Z0-9]{7})', clean_id)
                    if not match:
                        print(f"Skipping invalid tile format: {raw_id}")
                        continue
                    
                    tile_id = match.group(1)

                    # Look for BH4 or BH5 anywhere in the tile_id for safe targeting
                    is_nearshore_band = bool(re.search(r'BH[45]', tile_id, re.IGNORECASE))

                    # For 20m resolution, restrict processing to Band 4 and Band 5 tiles exclusively
                    if current_res == 20 and not is_nearshore_band:
                        continue

                    # For 100m resolution, exclude Band 4 and Band 5 tiles entirely
                    if current_res == 100 and is_nearshore_band:
                        continue

                    param_inputs.append([self.param_lookup, tile_id, normalized_er, output_prefix, current_res])

                # Sort parallel task configurations to submit them sequentially (ER1 -> ER2 -> ... -> ER6)
                # x[2] is the properly formatted ecoregion string now
                param_inputs = sorted(param_inputs, key=lambda x: _get_er_num(x[2]))

                print(f"Submitting {len(param_inputs)} tiles to Dask workers...")
                future_tiles = self.client.map(_process_tile, param_inputs)

                print("Waiting for all Dask workers to complete...")
                tile_results = self.client.gather(future_tiles)
                print("All Dask workers finished successfully.")

                self.print_async_results(tile_results, self.param_lookup['output_directory'].valueAsText)
                

                for ecoregion in all_ecoregions:
                    if output_prefix == 'low_res':
                        beginning_prefix = f'{output_prefix}/{current_res}m'
                    elif output_prefix:
                        beginning_prefix = output_prefix
                    else:
                        beginning_prefix = ''
                    s3_path = f"{beginning_prefix}/{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
                    self.write_run_manifest(s3_path, {'tiles': len(param_inputs)})
            else:   
                print(f"[BlueTopo Engine] skip_tiling is set to True. Bypassing individual tile generation entirely for {current_res}.")

            # TODO need to remove all of the mosaic logic and self.skip_tiling
            # Generate the unified isobaths from all finished tiles across all ecoregions (ER1 to ER6 order)
            # self.create_combined_isobaths(all_ecoregions, output_bucket, current_res, output_prefix)

            if current_res == 100:
                # When processing 100m, produce ONLY the unified big offshore mosaics
                self.create_and_upload_mosaics(all_ecoregions, output_bucket, current_res, output_prefix, increased_scale=True)
            elif current_res == 20:
                # Generate all individual cleanly masked mosaics per Ecoregion (ER1 to ER6 order)
                self.create_and_upload_er_mosaics(all_ecoregions, output_bucket, current_res, output_prefix, increased_scale=True)
                # NO FULL SIZE MOSAICS CREATED FOR 20M
        self.close_dask()

        # Keep logging safe by safely declaring tile_col even if loop is commented
        tile_col = None
        if not tile_gdf.empty:
            for col in tile_gdf.columns:
                col_lower = str(col).lower()
                if col_lower in ['tile', 'tile_id', 'name', 'id', 'bluetopo']:
                    tile_col = col
                    break

        tiles = list(tile_gdf[tile_col]) if tile_col and tile_col in tile_gdf.columns else []
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test') 

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value"""

        print(f"[{tiff_file_path.name}] Setting ground to nodata...")
        raster_ds = gdal.Open(str(tiff_file_path), gdal.GA_Update)
        no_data = -9999
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array < 0, raster_array, no_data)
        raster_ds.GetRasterBand(1).WriteArray(meters_array)
        raster_ds.GetRasterBand(1).SetNoDataValue(no_data)  
        raster_ds = None

    def upload_current_tiles_to_s3(self, tile_folder: pathlib.Path, temp_folder: pathlib.Path, existing_files: dict = None) -> None:
        """Upload all tiff files to s3 for current tile, strictly skipping those that already existed in S3."""

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        for tiff_file in tile_folder.glob('*.tiff'):
            filename = tiff_file.name
            
            # Prevent overwriting existing derivatives if they were already marked as present in S3
            if existing_files:
                if len(filename.split('_')) == 3 and existing_files.get('base'):
                    continue
                if filename.endswith('_ISS_all_110.tiff') and existing_files.get('iss_110'):
                    continue
                if filename.endswith('_survey_end_date.tiff') and existing_files.get('survey_end_date'):
                    continue
                if filename.endswith('_hurricane.tiff') and existing_files.get('hurricane'):
                    continue
                if filename.endswith('_unc.tiff') and existing_files.get('unc'):
                    continue

            s3_path = tiff_file.relative_to(temp_folder)  # Strip off the temp_folder parts
            self.write_message(f'Uploading {filename} to s3://{bucket_name}/{s3_path}', self.param_lookup['output_directory'].valueAsText)
            s3_client.upload_file(str(tiff_file), bucket_name, f'{str(s3_path)}')