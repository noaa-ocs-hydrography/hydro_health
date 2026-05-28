"""Class for processing everything for a single tile"""

import tempfile
import boto3
import os
import pathlib
import sys
import rasterio
import re
import s3fs

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
    param_lookup, tile_id, ecoregion_id, output_prefix, target_res = param_inputs
    
    print(f"[{tile_id}] Initiating processing for ecoregion {ecoregion_id}...")

    engine = BlueTopoS3Engine(param_lookup)
    engine.target_resolution = target_res

    # TODO need to check if BlueTopo tile and other tiffs already exist
    # Unconditionally regenerate all layers
    missing_files = ['base', 'ISS', 'hurricane', 'survey_end_date', 'slope', 'unc']
    if engine.file_in_s3(ecoregion_id, tile_id):
        print(f'BlueTopo tile {tile_id} already exists.  Skipping download.')
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = pathlib.Path(temp_dir)

            tiff_file_path = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id, output_prefix)
            if not tiff_file_path:
                print(f"[{tile_id}] Processing skipped (Source tile could not be downloaded).")
                return
            
            engine.resample_and_reproject(tiff_file_path)
            engine.create_survey_end_date_tiff(tiff_file_path)
            engine.create_catzoc_all(tiff_file_path)
            engine.create_hurricane_tile(tiff_file_path, ecoregion_id, output_prefix)
            mb_tiff_file = engine.rename_multiband(tiff_file_path)
            engine.multiband_to_singleband(mb_tiff_file, band=1)
            engine.multiband_to_singleband(mb_tiff_file, band=2)
            mb_tiff_file.unlink()
            engine.set_ground_to_nodata(tiff_file_path)
            engine.create_slope(tiff_file_path)
            engine.finalize_cog(tiff_file_path)      

            # Crop intermediate tiffs dynamically if we are in 20m resolution mode
            if engine.target_resolution == 20.0:
                print(f"[{tile_id}] Resolution is 20m. Cropping all local intermediate tiffs to ecoregion {ecoregion_id}...")
                tile_folder = tiff_file_path.parent
                stem = tiff_file_path.stem
                for local_file in tile_folder.glob(f"{stem}*.tiff"):
                    engine.crop_to_ecoregion(local_file, ecoregion_id)

            print(f"[{tile_id}] Uploading newly generated tiles to S3...")
            engine.upload_missing_tiles_to_s3(tiff_file_path.parents[0], tiff_file_path.stem, ecoregion_id, output_prefix, missing_files)
            print(f"[{tile_id}] Processing successfully completed.")


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
        # Set target resolution and CRS
        self.target_resolution = 100.0
        self.target_crs = "EPSG:6350"
        # Tiling is strictly executed
        self.skip_tiling = False

    @property
    def res_str(self) -> str:
        """Dynamically generate the resolution string (e.g., '100m' or '20m')"""
        return f"{int(self.target_resolution)}m" if self.target_resolution.is_integer() else f"{self.target_resolution}m"

    def resample_and_reproject(self, tiff_path: pathlib.Path) -> None:
        """Warp the downloaded raster to target resolution and CRS, treating categorical bands appropriately."""

        print(f"  - Resampling and reprojecting base tile...")
        output_folder = self.param_lookup['output_directory'].valueAsText
        msg = f"Resampling {tiff_path.name} to {self.target_resolution}m and explicitly locking into {self.target_crs}..."
        print(msg)
        self.write_message(msg, output_folder)
        
        # Open source dataset to dynamically determine available band count
        src_ds = gdal.Open(str(tiff_path))
        if src_ds is None:
            raise FileNotFoundError(f"Could not open downloaded source TIFF: {tiff_path}")
        band_count = src_ds.RasterCount
        src_ds = None
        
        warped_bands = []
        temp_files_to_clean = []
        
        # Resample each band independently to match continuous vs categorical types
        for band_idx in range(1, band_count + 1):
            raw_band_path = tiff_path.parent / f"band_{band_idx}_raw.tiff"
            warped_band_path = tiff_path.parent / f"band_{band_idx}_warped.tiff"
            
            # Extract single band from source
            ds_ext = gdal.Translate(str(raw_band_path), str(tiff_path), bandList=[band_idx])
            ds_ext = None  # Flush and release lock
            temp_files_to_clean.append(raw_band_path)
            
            # Choose correct resampling algorithm
            # Bands 1 (Bathy) and 2 (Uncertainty) get Bilinear interpolation
            # Bands 3+ (Contributor IDs/Metadata indices) must get Nearest Neighbor to avoid float corruption
            if band_idx in (1, 2):
                resample_alg = gdal.GRA_Bilinear
            else:
                resample_alg = gdal.GRA_NearestNeighbour
            
            # Warp single band to output parameters
            ds_warp = gdal.Warp(
                str(warped_band_path),
                str(raw_band_path),
                xRes=self.target_resolution,
                yRes=self.target_resolution,
                targetAlignedPixels=True,
                dstSRS=self.target_crs,
                resampleAlg=resample_alg,
                dstNodata=-9999,
                creationOptions=["COMPRESS=DEFLATE"]
            )
            ds_warp = None  # Flush and release lock
            warped_bands.append(str(warped_band_path))
            temp_files_to_clean.append(warped_band_path)
            
        # Re-merge the correctly resampled single bands back into a multi-band dataset
        temp_tiff = tiff_path.parent / f"warped_{tiff_path.name}"
        vrt_path = tiff_path.parent / f"warped_multiband.vrt"
        
        vrt_ds = gdal.BuildVRT(str(vrt_path), warped_bands, separate=True)
        vrt_ds = None  # Flush and release lock
        
        # Explicitly apply the self.target_crs here to strictly guarantee the finalized TIFF doesn't inherit a corrupted VRT CRS
        ds_trans = gdal.Translate(
            str(temp_tiff),
            str(vrt_path),
            outputSRS=self.target_crs,
            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES"]
        )
        ds_trans = None  # Flush and release lock
        
        # Clean up temporary intermediate files securely
        if vrt_path.exists():
            vrt_path.unlink()
        for tmp_file in temp_files_to_clean:
            if tmp_file.exists():
                tmp_file.unlink()
                
        if temp_tiff.exists():
            tiff_path.unlink()
            temp_tiff.rename(tiff_path)

    def crop_to_ecoregion(self, tiff_path: pathlib.Path, ecoregion_id: str) -> None:
        """Crop a single tile's TIFF to its ecoregion boundary if resolution is 20m."""
        if self.target_resolution != 20.0:
            return
            
        inputs_folder = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs")
        master_gpkg_path = inputs_folder / "Master_Grids.gpkg"
        
        if not master_gpkg_path.exists():
            print(f"[{tiff_path.name}] Master_Grids.gpkg not found, skipping tile cropping.")
            return
            
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
            gdal.Warp(str(temp_cropped), str(tiff_path), options=warp_options)
            if temp_cropped.exists():
                tiff_path.unlink()
                temp_cropped.rename(tiff_path)
        except Exception as e:
            print(f"[{tiff_path.name}] Failed to crop to ecoregion: {e}")
            if temp_cropped.exists():
                temp_cropped.unlink()

    def create_hurricane_tile(self, tiff_file_path: pathlib.Path, ecoregion_id: str, output_prefix: str|bool) -> None:
        """
        Generate a hurricane count raster by evaluating the survey year and 
        accumulating the corresponding yearly hurricane counts for each cell.
        """

        print("  - Creating cumulative hurricane count tile...")
        stem = tiff_file_path.stem
        hurricane_tile_path = tiff_file_path.parents[0] / f'{stem}_hurricane.tiff'
        
        # Local survey path always exists because we are enforcing unconditional local reprocessing
        local_survey_path = tiff_file_path.parents[0] / f'{stem}_survey_end_date.tiff'
        survey_ds_path = str(local_survey_path)

        try:
            with rasterio.open(survey_ds_path) as src:
                survey_years = src.read(1)
                transform = src.transform
                # Explicitly cast nodata to float, defaulting to -9999.0
                nodata = float(src.nodata) if src.nodata is not None else -9999.0
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
            with rasterio.open(hurricane_tile_path, "w", driver="GTiff", count=1, width=width, height=height, dtype=rasterio.float32, compress="lzw", tiled=True, blockxsize=512, blockysize=512, crs=self.target_crs, transform=transform, nodata=nodata) as dst:
                dst.write(result_array, 1)
            return

        def get_year_count_array(target_year):
            # Clamp year to valid data range
            target_year = max(1851, min(2023, int(target_year)))
            
            # Construct both the .tiff and the .tif fallbacks
            s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            s3_uri_tiff = f"/vsis3/{s3_bucket}/{hurricane_path_prefix}/cumulative_count_{target_year}.tiff"
            s3_uri_tif = f"/vsis3/{s3_bucket}/{hurricane_path_prefix}/cumulative_count_{target_year}.tif"
            
            # Use MEM driver to warp just the exact bounding box needed directly from S3
            warp_options = gdal.WarpOptions(
                format="MEM",
                outputBounds=(minx, miny, maxx, maxy),
                xRes=self.target_resolution,
                yRes=self.target_resolution,
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
            return arr

        # Start hurricane count at 0.0 ONLY for cells with a valid survey year
        result_array[valid_mask] = 0.0

        min_survey_year = max(1851, int(min(valid_years)))
        max_hurricane_year = 2023

        print(f"[{stem}] Aggregating yearly hurricane counts from {min_survey_year} to {max_hurricane_year}...")
        for target_year in range(min_survey_year, max_hurricane_year + 1):
            year_arr = get_year_count_array(target_year)
            
            # Add this year's count to any valid pixel whose survey year is <= target_year
            # (i.e. the hurricane occurred during or after the survey year)
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

    def create_catzoc_all(self, tiff_file_path: pathlib.Path) -> None:
        """
        Generate an Initial Survey Score (ISS) raster of unique values for each survey area.
        """

        print("  - Creating Initial Survey Score (ISS) all...")
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
            crs=self.target_crs,
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
            crs=self.target_crs,
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

        print("  - Creating slope...")
        slope_name = str(tiff_file_path.stem) + '_slope.tiff'
        slope_file_path = tiff_file_path.parents[0] / slope_name
        gdal.DEMProcessing(str(slope_file_path), str(tiff_file_path), 'slope')

    def create_survey_end_date_tiff(self, tiff_file_path: pathlib.Path) -> None:
        """Create survey end date tiffs from contributor band values in the XML file."""        

        print("  - Creating survey end date TIFF...")
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

    def download_nbs_tile(self, temp_folder: pathlib.Path, tile_id: str, ecoregion_id: str, output_prefix: str|bool) -> tuple[pathlib.Path | bool, list]:
        """Unconditionally download the NBS source tile and stage it for full processing."""

        nbs_bucket = self.get_bucket()
        output_tile_path = False
        output_folder = self.param_lookup['output_directory'].valueAsText

        msg = f"Tile {tile_id} targeted for full processing. Downloading fresh NBS Source and metadata..."
        print(msg)
        self.write_message(msg, output_folder)

        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            current_file = temp_folder / ecoregion_id / get_config_item('BLUETOPO', 'SUBFOLDER') / obj_summary.key
            
            # We need the main raster (.tiff/.tif) and the attribute table metadata (.aux.xml)
            if current_file.suffix in ('.tiff', '.tif', '.xml'):
                tile_folder = current_file.parents[0]
                tile_folder.mkdir(parents=True, exist_ok=True)   
                self.write_message(f'Downloading: {current_file.name}', output_folder)
                nbs_bucket.download_file(obj_summary.key, str(current_file))
                
                if current_file.suffix in ('.tiff', '.tif'):
                    output_tile_path = current_file

        return output_tile_path
    
    def file_in_s3(ecoregion_id: str, tile_id: str) -> bool:
        """Check if BlueTopo tile already exists in S3"""

        s3_files = s3fs.S3FileSystem()
        bluetopo_tile_folder = f'{get_config_item("SHARED", "OUTPUT_BUCKET")}/ER_3/{get_config_item("BLUETOPO", "SUBFOLDER")}/BlueTopo/{tile_id}'
        search_path = f"{bluetopo_tile_folder}/**/*.tiff"
        found_files = s3_files.glob(search_path)
        for file in found_files:
            filename = pathlib.Path(file).name
            if len(filename.split('_')) == 3:
                # Found tile example: BlueTopo_BC25V26P_20250916.tiff
                return True
        return False
    
    def finalize_cog(self, tiff_path: pathlib.Path) -> None:
        """The final pass to ensure perfect COG layout and overviews."""

        print("  - Finalizing COG format...")
        temp_cog = tiff_path.parent / f"temp_{tiff_path.name}"
        
        ds = gdal.Open(str(tiff_path), gdal.GA_Update)
        if ds is not None:
            ds.BuildOverviews("BILINEAR", [2, 4, 8, 16])
            ds = None

        gdal.Translate(
            str(temp_cog),
            str(tiff_path),
            outputSRS=self.target_crs,
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
            outputSRS=self.target_crs,
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
        and appends them directly to the local isobath_layers.gpkg.
        
        Uses an efficient tile-by-tile memory strategy to prevent VRT rendering dropouts,
        S3 connection thrashing, and massive disk usage.
        """
        if self.target_resolution == 20.0:
            print(f"[All Regions] Target resolution ({self.res_str}) is 20m. Skipping isobath generation.")
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
            search_prefix = f"low_res/{self.res_str}/{ecoregion_id}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
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
                
        print(f"[All Regions] Saving and securely closing {isobath_gpkg_path}...")
        layers.clear()
        isobath_ds = None

    def _build_mosaic_chunked(self, target_path: pathlib.Path, paths: list[str], warp_kwargs: dict) -> None:
        """
        Process a massive list of S3 inputs in memory-safe chunks. This handles heterogeneous 
        projections dynamically via Warp without hitting "Too many open files" OS limits.
        """
        chunk_size = 250
        path_chunks = [paths[i:i + chunk_size] for i in range(0, len(paths), chunk_size)]
        
        for i, chunk in enumerate(path_chunks):
            if i == 0:
                # First chunk establishes the raster extent (computed from cutline if supplied)
                gdal.Warp(str(target_path), chunk, options=gdal.WarpOptions(**warp_kwargs))
            else:
                # Subsequent chunks append onto the newly established target dataset.
                append_kwargs = warp_kwargs.copy()
                if 'creationOptions' in append_kwargs:
                    del append_kwargs['creationOptions']
                if 'cropToCutline' in append_kwargs:
                    del append_kwargs['cropToCutline']
                    
                dest_ds = gdal.Open(str(target_path), gdal.GA_Update)
                if dest_ds is None:
                    raise RuntimeError(f"Failed to open {target_path} for appending chunk {i+1}.")
                
                gdal.Warp(dest_ds, chunk, options=gdal.WarpOptions(**append_kwargs))
                dest_ds = None

    def create_and_upload_er_mosaics(self, ecoregion_ids: list[str], s3_bucket: str, output_prefix: str|bool, only_offshore: bool = False) -> None:
        """
        Creates isolated mosaics for each individual ecoregion and precisely masks them
        to their specific boundary polygon from the EcoRegions layer.
        """
        output_folder = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/outputs")
        low_res_dir = output_folder / "low_res" / self.res_str
        low_res_dir.mkdir(parents=True, exist_ok=True)
        
        inputs_folder = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs")
        master_gpkg_path = inputs_folder / "Master_Grids.gpkg"
        
        s3_client = boto3.client('s3')

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
        # Set targetAlignedPixels to perfectly align the 100m grids eliminating edge artifact grid lines
        warp_kwargs_base = {
            "format": "GTiff",
            "dstSRS": self.target_crs,
            "xRes": self.target_resolution,
            "yRes": self.target_resolution,
            "dstNodata": -9999,
            "targetAlignedPixels": True, 
            "multithread": True,
            "creationOptions": ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
        }
        
        if cutline_layer_name:
            warp_kwargs_base["cutlineDSName"] = str(master_gpkg_path)
            warp_kwargs_base["cutlineLayer"] = cutline_layer_name
            warp_kwargs_base["cropToCutline"] = True

        for ecoregion_id in ecoregion_ids:
            er_prefix = ecoregion_id if str(ecoregion_id).startswith("ER_") else f"ER_{ecoregion_id}"
            
            # Map configuration properties dictating regex mapping
            base_mosaics_config = {
                "Bathy": re.compile(r'\d+\.tiff$', re.IGNORECASE),
                "ISS": re.compile(r'_ISS_all\.tiff$', re.IGNORECASE),
                "Survey_Date": re.compile(r'_survey_end_date\.tiff$', re.IGNORECASE),
                "Hurricane": re.compile(r'_hurricane\.tiff$', re.IGNORECASE),
                "Slope": re.compile(r'_slope\.tiff$', re.IGNORECASE),
            }

            mosaics_config = {}
            for name, regex in base_mosaics_config.items():
                # Omit BlueTopo prefix if we are generating Hurricane mosaics
                bt_prefix = "" if name.lower() == "hurricane" else "BlueTopo_"

                if not only_offshore:
                    mosaics_config[name] = {
                        "regex": regex,
                        "filename": f"{er_prefix}_{bt_prefix}{name}_Mosaic_{self.res_str}.tiff",
                        "paths": [],
                        "exclude_bh4": False
                    }
                # Create a secondary non-Band4 version specifically for 100m runs
                if self.target_resolution == 100.0:
                    mosaics_config[f"{name}_Offshore"] = {
                        "regex": regex,
                        "filename": f"{er_prefix}_{bt_prefix}{name}_Mosaic_Offshore_{self.res_str}.tiff",
                        "paths": [],
                        "exclude_bh4": True
                    }

            # Unconditionally process all configured mosaics
            mosaics_to_process = mosaics_config
            for m_key, config in mosaics_to_process.items():
                s3_key = f"low_res/{self.res_str}/{ecoregion_id}/{config['filename']}"
                if output_prefix and output_prefix.strip('/') != "low_res":
                    s3_key = f"{output_prefix}/{s3_key}"
                config['s3_key'] = s3_key

            print(f"[{ecoregion_id}] Gathering S3 links for ER specific mosaics: {list(mosaics_to_process.keys())}")
            search_prefix = f"low_res/{self.res_str}/{ecoregion_id}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            if output_prefix and output_prefix.strip('/') != "low_res":
                search_prefix = f"{output_prefix}/{search_prefix}"
                
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        is_bh4 = bool(re.search(r'/BH4[A-Za-z]', key, re.IGNORECASE))
                        for m_key, config in mosaics_to_process.items():
                            if config['regex'].search(key):
                                if config['exclude_bh4'] and is_bh4:
                                    continue
                                config['paths'].append(f"/vsis3/{s3_bucket}/{key}")
                                
            # Apply a specific SQL WHERE clause to the vector file to grab ONLY this Ecoregion's polygon
            er_warp_kwargs = warp_kwargs_base.copy()
            if cutline_layer_name and er_field_name:
                # Robust extraction of the ecoregion number to prevent ER_1/ER1 split matching errors
                match = re.search(r'\d+', str(ecoregion_id))
                clean_er_num = match.group(0) if match else str(ecoregion_id)
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
                    
                er_warp_kwargs["cutlineWhere"] = " OR ".join(where_clauses)

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
                if "bathy" in m_key_lower or "unc" in m_key_lower or "slope" in m_key_lower:
                    resample_alg = gdal.GRA_Bilinear
                else:
                    resample_alg = gdal.GRA_NearestNeighbour
                
                current_warp_kwargs = er_warp_kwargs.copy()
                current_warp_kwargs["resampleAlg"] = resample_alg

                print(f"[{ecoregion_id}] Warping and precisely masking {m_key} ER mosaic dynamically from {len(config['paths'])} source files...")
                
                gdal.UseExceptions()
                try:
                    warp_options = gdal.WarpOptions(**current_warp_kwargs)
                    gdal.Warp(str(local_mosaic_path), config['paths'], options=warp_options)
                except RuntimeError as e:
                    print(f"[{ecoregion_id}] WARNING: Masking failed with specific WHERE clause. Falling back to general cutline...")
                    print(f"[{ecoregion_id}] GDAL Error: {e}")
                    
                    if local_mosaic_path.exists():
                        local_mosaic_path.unlink()
                        
                    fallback_kwargs = current_warp_kwargs.copy()
                    if "cutlineWhere" in fallback_kwargs:
                        del fallback_kwargs["cutlineWhere"]
                    
                    try:
                        fallback_options = gdal.WarpOptions(**fallback_kwargs)
                        gdal.Warp(str(local_mosaic_path), config['paths'], options=fallback_options)
                    except RuntimeError as e2:
                        print(f"[{ecoregion_id}] CRITICAL WARNING: General mask also failed! Generating unclipped mosaic...")
                        print(f"[{ecoregion_id}] GDAL Error: {e2}")
                        
                        if local_mosaic_path.exists():
                            local_mosaic_path.unlink()
                            
                        unclipped_kwargs = {k: v for k, v in fallback_kwargs.items() if k not in ["cutlineDSName", "cutlineLayer", "cropToCutline", "cutlineWhere"]}
                        unclipped_options = gdal.WarpOptions(**unclipped_kwargs)
                        gdal.Warp(str(local_mosaic_path), config['paths'], options=unclipped_options)

                print(f"[{ecoregion_id}] Generating robust overviews...")
                ds = gdal.Open(str(local_mosaic_path), gdal.GA_Update)
                if ds is not None:
                    ds.BuildOverviews("BILINEAR", [2, 4, 8, 16])
                    ds = None
                    
                print(f"[{ecoregion_id}] Uploading {m_key} ER mosaic to s3://{s3_bucket}/{s3_key}...")
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
                
                if local_mosaic_path.exists():
                    local_mosaic_path.unlink() # Save disk space
                
                print(f"[{ecoregion_id}] {m_key} ER mosaic fully completed.")

    def create_and_upload_mosaics(self, ecoregion_ids: list[str], s3_bucket: str, output_prefix: str|bool) -> None:
        """
        Creates massive mosaics from all base bathy tiles and derivatives across all processed ecoregions,
        masks them securely with the EcoRegions layer, saves locally to low_res, and uploads to S3.
        """
        if self.target_resolution < 50.0:
            print(f"[Mosaic] Target resolution ({self.res_str}) is under 50m. Skipping massive mosaic generation to prevent extreme file sizes.")
            return

        output_folder = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/outputs")
        low_res_dir = output_folder / "low_res" / self.res_str
        low_res_dir.mkdir(parents=True, exist_ok=True)
        
        inputs_folder = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs")
        master_gpkg_path = inputs_folder / "Master_Grids.gpkg"
        
        s3_client = boto3.client('s3')

        # Configuration holding the regex mapping
        base_mosaics_config = {
            "Bathy": re.compile(r'\d+\.tiff$', re.IGNORECASE),
            "ISS": re.compile(r'_ISS_all\.tiff$', re.IGNORECASE),
            "Survey_Date": re.compile(r'_survey_end_date\.tiff$', re.IGNORECASE),
            "Hurricane": re.compile(r'_hurricane\.tiff$', re.IGNORECASE),
            "Slope": re.compile(r'_slope\.tiff$', re.IGNORECASE),
        }

        mosaics_config = {}
        for name, regex in base_mosaics_config.items():
            # Drop BlueTopo prefix if we are processing Hurricane
            bt_prefix = "" if name.lower() == "hurricane" else "BlueTopo_"

            mosaics_config[name] = {
                "regex": regex,
                "filename": f"{bt_prefix}{name}_Mosaic_{self.res_str}.tiff",
                "paths": [],
                "exclude_bh4": False
            }
            # Create a secondary non-Band4 version specifically for 100m runs
            if self.target_resolution == 100.0:
                mosaics_config[f"{name}_Offshore"] = {
                    "regex": regex,
                    "filename": f"{bt_prefix}{name}_Mosaic_Offshore_{self.res_str}.tiff",
                    "paths": [],
                    "exclude_bh4": True
                }

        # Unconditionally process all configured mosaics
        mosaics_to_process = mosaics_config
        for m_key, config in mosaics_to_process.items():
            s3_key = f"low_res/{self.res_str}/{config['filename']}"
            if output_prefix and output_prefix.strip('/') != "low_res":
                s3_key = f"{output_prefix}/{s3_key}"
            config['s3_key'] = s3_key

        print(f"[Mosaic] Gathering S3 links across {len(ecoregion_ids)} ecoregions for: {list(mosaics_to_process.keys())}")
        
        # 2. Gather source file paths in a single efficient pass through S3 per ecoregion
        for ecoregion_id in ecoregion_ids:
            search_prefix = f"low_res/{self.res_str}/{ecoregion_id}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            if output_prefix and output_prefix.strip('/') != "low_res":
                search_prefix = f"{output_prefix}/{search_prefix}"
                
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix)
            
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        is_bh4 = bool(re.search(r'/BH4[A-Za-z]', key, re.IGNORECASE))
                        for m_key, config in mosaics_to_process.items():
                            if config['regex'].search(key):
                                if config['exclude_bh4'] and is_bh4:
                                    continue
                                config['paths'].append(f"/vsis3/{s3_bucket}/{key}")
                            
        # 3. Locate the EcoRegions layer once for masking
        cutline_layer_name = None
        er_field_name = None
        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            for name in layer_names:
                # Match case-insensitively, allowing for underscores or spaces
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
            
            if not cutline_layer_name:
                print(f"[Mosaic] WARNING: Could not find an 'Enhanced_EcoRegions' layer in {master_gpkg_path}.")
                print(f"[Mosaic] Available layers: {layer_names}")
                print("[Mosaic] Proceeding without vector mask to prevent crash...")
            
            ds = None
        else:
            print(f"[Mosaic] WARNING: Could not open {master_gpkg_path} to check layers. Proceeding without mask...")

        # Base warp arguments allowing GDAL to naturally unify projections perfectly
        # Set targetAlignedPixels to perfectly align the 100m grids eliminating edge artifact grid lines
        warp_kwargs_base = {
            "format": "GTiff",
            "dstSRS": self.target_crs,
            "xRes": self.target_resolution,
            "yRes": self.target_resolution,
            "dstNodata": -9999,
            "targetAlignedPixels": True,
            "multithread": True,
            "creationOptions": ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
        }
        
        if cutline_layer_name:
            warp_kwargs_base["cutlineDSName"] = str(master_gpkg_path)
            warp_kwargs_base["cutlineLayer"] = cutline_layer_name
            warp_kwargs_base["cropToCutline"] = True
            
            if er_field_name:
                where_clauses = []
                for er_id in ecoregion_ids:
                    match = re.search(r'\d+', str(er_id))
                    clean_er_num = match.group(0) if match else str(er_id)
                    where_clauses.extend([
                        f"{er_field_name} = 'ER_{clean_er_num}'",
                        f"{er_field_name} = 'ER{clean_er_num}'",
                        f"{er_field_name} = '{er_id}'",
                        f"{er_field_name} = '{clean_er_num}'"
                    ])
                    if clean_er_num.isdigit():
                        where_clauses.append(f"{er_field_name} = {clean_er_num}")
                warp_kwargs_base["cutlineWhere"] = " OR ".join(where_clauses)

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
            if "bathy" in m_key_lower or "unc" in m_key_lower or "slope" in m_key_lower:
                resample_alg = gdal.GRA_Bilinear
            else:
                resample_alg = gdal.GRA_NearestNeighbour
            
            current_warp_kwargs = warp_kwargs_base.copy()
            current_warp_kwargs["resampleAlg"] = resample_alg
            
            print(f"[Mosaic] Warping and masking {m_key} mosaic directly from {len(config['paths'])} source files...")
            
            gdal.UseExceptions()
            try:
                warp_options = gdal.WarpOptions(**current_warp_kwargs)
                gdal.Warp(str(local_mosaic_path), config['paths'], options=warp_options)
            except RuntimeError as e:
                print(f"[Mosaic] WARNING: Masking failed with specific WHERE clause. Falling back to general cutline...")
                print(f"[Mosaic] GDAL Error: {e}")
                
                if local_mosaic_path.exists():
                    local_mosaic_path.unlink()
                    
                # Fall back to using the cutline layer without the WHERE filter
                fallback_kwargs = current_warp_kwargs.copy()
                if "cutlineWhere" in fallback_kwargs:
                    del fallback_kwargs["cutlineWhere"]
                
                try:
                    fallback_options = gdal.WarpOptions(**fallback_kwargs)
                    gdal.Warp(str(local_mosaic_path), config['paths'], options=fallback_options)
                except RuntimeError as e2:
                    print(f"[Mosaic] CRITICAL WARNING: General mask also failed! Generating unclipped mosaic...")
                    print(f"[Mosaic] GDAL Error: {e2}")
                    
                    if local_mosaic_path.exists():
                        local_mosaic_path.unlink()
                        
                    unclipped_kwargs = {k: v for k, v in fallback_kwargs.items() if k not in ["cutlineDSName", "cutlineLayer", "cropToCutline", "cutlineWhere"]}
                    unclipped_options = gdal.WarpOptions(**unclipped_kwargs)
                    gdal.Warp(str(local_mosaic_path), config['paths'], options=unclipped_options)

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

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str|bool=False) -> None:
        output_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        all_ecoregions = self.param_lookup['eco_regions'].value
        
        # Sort ecoregions in numerical order (ER1 to ER6)
        def _get_er_num(er_str):
            match = re.search(r'\d+', str(er_str))
            return int(match.group(0)) if match else 9999
        all_ecoregions = sorted(all_ecoregions, key=_get_er_num)
        print(f"[BlueTopo Engine] Processing ecoregions in sequential order: {all_ecoregions}")
        
        for current_res in [100.0, 20.0]:
            self.target_resolution = current_res
            print(f"\n{'='*50}\n[BlueTopo Engine] STARTING PROCESSING FOR {self.res_str}\n{'='*50}")

            if not self.skip_tiling:
                print(f'Checking and processing BlueTopo Datasets for {self.res_str}...')
                self.setup_dask(self.param_lookup['env'], threads_per_worker=1)
                
                param_inputs = []
                for _, row in tile_gdf.iterrows():
                    if not isinstance(row[1], str):
                        continue
                    tile_id = str(row[0])
                    
                    is_band4 = bool(re.match(r'^BH4[A-Za-z]', tile_id))
                    
                    # For 20m resolution, restrict processing to Band 4 tiles exclusively (BH4 followed by a letter)
                    if self.target_resolution == 20.0 and not is_band4:
                        continue
                        
                    param_inputs.append([self.param_lookup, tile_id, row[1], output_prefix, self.target_resolution])
                
                # Sort parallel task configurations to submit them sequentially (ER1 -> ER2 -> ... -> ER6)
                param_inputs = sorted(param_inputs, key=lambda x: _get_er_num(x[3]))
                
                print(f"Submitting {len(param_inputs)} tiles to Dask workers...")
                future_tiles = self.client.map(_process_tile, param_inputs)
                
                print("Waiting for all Dask workers to complete...")
                tile_results = self.client.gather(future_tiles)
                print("All Dask workers finished successfully.")
                
                self.print_async_results(tile_results, self.param_lookup['output_directory'].valueAsText)
                self.close_dask()
                
                for ecoregion in all_ecoregions:
                    s3_path = f"low_res/{self.res_str}/{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
                    if output_prefix and output_prefix.strip('/') != "low_res":
                        s3_path = f"{output_prefix}/{s3_path}"
                    self.write_run_manifest(s3_path)
            else:
                print(f"[BlueTopo Engine] skip_tiling is set to True. Bypassing individual tile generation entirely for {self.res_str}.")
            
            # Generate the unified isobaths from all finished tiles across all ecoregions (ER1 to ER6 order)
            self.create_combined_isobaths(all_ecoregions, output_bucket, output_prefix)

            if self.target_resolution == 100.0:
                # When processing 100m, produce unified big mosaics first, and then ONLY produce offshore ER individual mosaics
                self.create_and_upload_mosaics(all_ecoregions, output_bucket, output_prefix)
                self.create_and_upload_er_mosaics(all_ecoregions, output_bucket, output_prefix, only_offshore=True)
            elif self.target_resolution == 20.0:
                # Generate all individual cleanly masked mosaics per Ecoregion (ER1 to ER6 order)
                self.create_and_upload_er_mosaics(all_ecoregions, output_bucket, output_prefix, only_offshore=False)
                # NO FULL SIZE MOSAICS CREATED FOR 20M

        tiles = list(tile_gdf['tile']) if 'tile' in tile_gdf.columns else [str(row[0]) for _, row in tile_gdf.iterrows()]
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test') 

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value"""

        print("  - Setting ground to nodata...")
        raster_ds = gdal.Open(str(tiff_file_path), gdal.GA_Update)
        no_data = -9999
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array < 0, raster_array, no_data)
        raster_ds.GetRasterBand(1).WriteArray(meters_array)
        raster_ds.GetRasterBand(1).SetNoDataValue(no_data)  
        raster_ds = None

    def upload_missing_tiles_to_s3(self, tile_folder: pathlib.Path, stem: str, ecoregion_id: str, output_prefix: str|bool, missing_files: list) -> None:
        """Upload all newly generated files to S3"""
        s3_client = boto3.client('s3')
        
        s3_metadata_args = {
            'Metadata': {
                'resolution': str(self.target_resolution),
                'crs': self.target_crs
            }
        }
        
        # Map our logical component names to the exact file structures we generate
        file_map = {
            'base': f"{stem}.tiff",
            'ISS': f"{stem}_ISS_all.tiff",
            'hurricane': f"{stem}_hurricane.tiff",
            'survey_end_date': f"{stem}_survey_end_date.tiff",
            'slope': f"{stem}_slope.tiff",
            'unc': f"{stem}_unc.tiff"
        }
        
        for mf in missing_files:
            file_name = file_map.get(mf)
            if not file_name:
                continue
                
            local_file = tile_folder / file_name
            if local_file.exists():
                # Reconstruct the exact S3 prefix we need based on the dynamic resolution string
                ecoregion_index = local_file.parts.index(ecoregion_id)
                s3_subpath = pathlib.Path(*local_file.parts[ecoregion_index:])
                
                formatted_subpath = str(s3_subpath).replace('\\', '/')
                s3_path_formatted = f"low_res/{self.res_str}/{formatted_subpath}"
                if output_prefix and output_prefix.strip('/') != "low_res":
                    s3_path_formatted = f"{output_prefix}/{s3_path_formatted}"
                
                bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
                self.write_message(f'Uploading {local_file.name} to s3://{bucket_name}/{s3_path_formatted}', self.param_lookup['output_directory'].valueAsText)
                s3_client.upload_file(str(local_file), bucket_name, s3_path_formatted, ExtraArgs=s3_metadata_args)