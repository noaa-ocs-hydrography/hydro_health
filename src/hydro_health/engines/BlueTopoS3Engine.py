"""Class for processing everything for a single tile"""

import tempfile
import boto3
import os
import pathlib
import sys
import rasterio
import re
import s3fs
import shutil

import geopandas as gpd
import pandas as pd
import numpy as np

from multiprocessing import set_executable
from hydro_health.helpers import hibase_logging
from datetime import datetime, date
from botocore.client import Config
from botocore import UNSIGNED
from lxml import etree
from osgeo import gdal, osr

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine, supersession, catzoc

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


def _process_tile(param_inputs: list) -> str:
    """
    Static function that handles processing of a single tile.
    Refactored for S3-to-S3 workflow using ephemeral storage on NVMe disk.
    """

    param_lookup, tile_id, ecoregion_id, output_prefix, target_res = param_inputs
    
    print(f"[{tile_id}] Initiating processing for ecoregion {ecoregion_id}...")

    engine = BlueTopoS3Engine(param_lookup)
    
    print(f"[{tile_id}] Downloading raw NBS source to generate derivatives...")

    # Direct temporary workspace to the engine's NVMe scratch directory
    with tempfile.TemporaryDirectory(dir=engine.scratch_dir) as temp_dir:
        temp_path = pathlib.Path(temp_dir)

        tiff_file_path = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id, output_prefix, target_res)
        if not tiff_file_path:
            msg = f"[{tile_id}] Processing skipped (Source tile could not be downloaded or was invalid)."
            print(msg)
            return msg
        
        tile_folder = tiff_file_path.parents[0]

        # Explicitly and safely warp the base tile to the target CRS and Resolution
        print(f"[{tile_id}] Warping raw tile in memory to align generated derivatives...")
        engine.warp_bluetopo_tile(tiff_file_path, target_res)
        
        engine.create_survey_end_date_tiff(tiff_file_path)
        engine.create_catzoc_all(tiff_file_path, increased_scale=True)
        engine.create_catzoc_latest(tiff_file_path, increased_scale=True)
        engine.create_rugosity(tiff_file_path)
        engine.create_slope(tiff_file_path)
        
        mb_tiff_file = engine.rename_multiband(tiff_file_path)
        engine.multiband_to_singleband(mb_tiff_file, band=1)
        engine.multiband_to_singleband(mb_tiff_file, band=2)
        if mb_tiff_file.exists():
            mb_tiff_file.unlink()
        
        engine.set_ground_to_nodata(tiff_file_path)
        engine.finalize_cog(tiff_file_path)    

        print(f"[{tile_id}] Uploading newly generated files to S3...")
        engine.upload_current_tiles_to_s3(tile_folder, temp_path)
        
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
        self.skip_tiling = False
        
        # Local NVMe scratch directory isolated to BlueTopoS3Engine for current user
        self.scratch_dir = pathlib.Path.home() / "scratch_tmp" / "BlueTopoS3Engine"
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def create_catzoc_all(self, tiff_file_path: pathlib.Path, increased_scale: bool=False) -> None:
        """Generate an Initial Survey Score (ISS) raster of unique values for each survey area."""

        print(f"[{tiff_file_path.name}] Creating Initial Survey Score (ISS) all...")
        nodata = -9999 
        with rasterio.open(tiff_file_path) as src:
            band3_raw = src.read(3)
            # Safely replace NaNs with nodata before converting to int32
            contributor_band_values = np.nan_to_num(np.round(band3_raw), nan=nodata).astype(np.int32)
            transform = src.transform
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
        reclass_dict = {int(row[0]): float(row[1]) for row in iss_mapping.to_numpy()}

        # Fast Array Indexing Reclassification (Replaces np.vectorize to conserve RAM)
        max_val = int(contributor_band_values.max()) if contributor_band_values.size > 0 else 0
        lookup_array = np.full(max_val + 1, nodata, dtype=np.float32)

        for val, iss in reclass_dict.items():
            if 0 <= val <= max_val:
                lookup_array[val] = iss

        valid_mask = (contributor_band_values >= 0) & (contributor_band_values <= max_val)
        reclassified_band = np.full_like(contributor_band_values, nodata, dtype=np.float32)
        reclassified_band[valid_mask] = lookup_array[contributor_band_values[valid_mask]]

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
        nodata = -9999 
        with rasterio.open(tiff_file_path) as src:
            band3_raw = src.read(3)
            # Safely replace NaNs with nodata before converting to int32
            contributor_band_values = np.nan_to_num(np.round(band3_raw), nan=nodata).astype(np.int32)
            transform = src.transform
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

        reclassified_band = np.where(contributor_band_values == nodata, nodata, most_recent_survey['iss']).astype(np.float32)

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

        print(f"[{tiff_file_path.name}] Creating survey end date TIFF...")
        nodata = -9999
        with rasterio.open(tiff_file_path) as src:
            band3_raw = src.read(3)
            # Safely replace NaNs with nodata before converting to int32
            contributor_band_values = np.nan_to_num(np.round(band3_raw), nan=nodata).astype(np.int32)
            transform = src.transform
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
        reclass_dict = {int(row[0]): int(row[1]) for row in date_mapping.to_numpy()}

        # Fast Array Indexing Reclassification (Replaces np.vectorize to conserve RAM)
        max_val = int(contributor_band_values.max()) if contributor_band_values.size > 0 else 0
        lookup_array = np.full(max_val + 1, nodata, dtype=np.float32)

        for val, yr in reclass_dict.items():
            if 0 <= val <= max_val:
                lookup_array[val] = yr

        valid_mask = (contributor_band_values >= 0) & (contributor_band_values <= max_val)
        reclassified_band = np.full_like(contributor_band_values, nodata, dtype=np.float32)
        reclassified_band[valid_mask] = lookup_array[contributor_band_values[valid_mask]]

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
                
                self.write_message(f'Downloading: {current_file.name}', output_folder)
                
                if current_file.exists():
                    current_file.unlink() # Remove partial/corrupt file before retry
                    
                nbs_bucket.download_file(obj_summary.key, str(current_file))
                
                if current_file.suffix in ('.tiff', '.tif'):
                    output_tile_path = current_file

        if not found_files:
            error_msg = f"[{tile_id}] CRITICAL: No source files found in NBS S3 bucket for prefix 'BlueTopo/{tile_id}'. Tile missing."
            print(error_msg)
            self.write_message(error_msg, output_folder)
            return False

        return output_tile_path

    def finalize_cog(self, tiff_path: pathlib.Path) -> None:
        """The final pass to ensure perfect COG layout and overviews."""

        print(f"[{tiff_path.name}] Finalizing COG format...")
        temp_cog = tiff_path.parent / f"temp_{tiff_path.name}"

        gdal.Translate(
            str(temp_cog),
            str(tiff_path),
            format="COG",
            outputSRS=self.target_crs,
            creationOptions=[
                "COMPRESS=DEFLATE",
                "PREDICTOR=3",
                "BLOCKSIZE=512",
                "OVERVIEW_RESAMPLING=BILINEAR"
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
        mb_tiff_file = pathlib.Path(new_name)
        return mb_tiff_file

    def warp_bluetopo_tile(self, tiff_file_path: pathlib.Path, target_res: int) -> None:
        """
        Safely reprojects a 3-band BlueTopo tile to EPSG:6350.
        Uses Bilinear for Elevation (B1) & Uncertainty (B2), 
        and NearestNeighbour for the Contributor metadata (B3) to prevent category corruption.
        Intermediate files are placed in an isolated temporary directory to avoid polluting
        the tile folder.
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
            shutil.copy(xml_path, safe_xml)
            
        # Place all intermediate files inside an isolated temp directory on NVMe storage
        with tempfile.TemporaryDirectory(dir=self.scratch_dir) as warp_tmpdir:
            tmp_path = pathlib.Path(warp_tmpdir)
            temp_b12_src = tmp_path / f"b12_src_{tiff_file_path.name}"
            temp_b3_src = tmp_path / f"b3_src_{tiff_file_path.name}"
            temp_b12_warp = tmp_path / f"b12_warp_{tiff_file_path.name}"
            temp_b3_warp = tmp_path / f"b3_warp_{tiff_file_path.name}"
            final_warp = tmp_path / f"warp_{tiff_file_path.name}"
            
            # Common creation options to prevent 4GB BigTIFF limits and improve performance
            creation_opts = [
                "COMPRESS=DEFLATE",
                "BIGTIFF=IF_NEEDED",
                "TILED=YES",
                "BLOCKXSIZE=512",
                "BLOCKYSIZE=512"
            ]

            # Extract bands 1 & 2 vs band 3
            ds1 = gdal.Translate(str(temp_b12_src), str(tiff_file_path), bandList=[1, 2], creationOptions=creation_opts)
            ds1 = None
            ds2 = gdal.Translate(str(temp_b3_src), str(tiff_file_path), bandList=[3], creationOptions=creation_opts)
            ds2 = None
            
            # Warp Bands 1 & 2 (Bilinear)
            ds3 = gdal.Warp(str(temp_b12_warp), str(temp_b12_src), options=gdal.WarpOptions(
                format="GTiff", dstSRS=self.target_crs, xRes=target_res, yRes=target_res,
                resampleAlg=gdal.GRA_Bilinear, targetAlignedPixels=True,
                creationOptions=creation_opts
            ))
            ds3 = None
            
            # Warp Band 3 (Nearest Neighbor)
            ds4 = gdal.Warp(str(temp_b3_warp), str(temp_b3_src), options=gdal.WarpOptions(
                format="GTiff", dstSRS=self.target_crs, xRes=target_res, yRes=target_res,
                resampleAlg=gdal.GRA_NearestNeighbour, targetAlignedPixels=True,
                creationOptions=creation_opts
            ))
            ds4 = None
            
            # Zero-RAM Streamed Merge using GDAL VRT
            if temp_b12_warp.exists() and temp_b3_warp.exists():
                vrt_path = tmp_path / f"stacked_{tiff_file_path.name}.vrt"
                
                gdal.BuildVRT(str(vrt_path), [str(temp_b12_warp), str(temp_b3_warp)], options=gdal.BuildVRTOptions(separate=True))
                
                gdal.Translate(
                    str(final_warp), 
                    str(vrt_path), 
                    bandList=[1, 2, 3], 
                    creationOptions=creation_opts
                )
                
                # Replace original tile with final warped file
                tiff_file_path.unlink()
                shutil.move(str(final_warp), str(tiff_file_path))
                
                # Restore the RAT XML to the final warped file
                if safe_xml.exists():
                    safe_xml.rename(tiff_file_path.parent / f"{tiff_file_path.name}.aux.xml")

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""
        for result in results:
            if result:
                self.write_message(result, output_folder)

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str|bool, resolution: list[int] = None) -> None:
        output_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        # Standardize the ecoregions mapped in param_lookup globally to "ER_#"
        all_ecoregions_raw = self.param_lookup['eco_regions'].value
        all_ecoregions = []
        for er in all_ecoregions_raw:
            match = re.search(r'\d+', str(er))
            if match:
                all_ecoregions.append(f"ER_{match.group(0)}")
                
        # Sort ecoregions in numerical order (ER_1 to ER_6)
        def _get_er_num(er_str):
            match = re.search(r'\d+', str(er_str))
            return int(match.group(0)) if match else 9999
            
        all_ecoregions = sorted(set(all_ecoregions), key=_get_er_num)
        
        print(f"[BlueTopo Engine] Processing ecoregions in sequential order: {all_ecoregions}")
        
        if not all_ecoregions:
            print("[BlueTopo Engine] No requested Ecoregions found in input. Exiting run.")
            return

        # Explicitly setting single worker / thread to protect 16GB RAM limit
        self.setup_dask(self.param_lookup['env'], n_workers=1, threads_per_worker=1)
        
        # We explicitly rely on 'EcoRegion' as requested
        er_col = 'EcoRegion' 
        
        # Fallback list to quickly grab the tile ID column without iterating the dataframe
        tile_col = next((c for c in tile_gdf.columns if str(c).lower() in ['tile', 'tile_id', 'name', 'id', 'bluetopo']), tile_gdf.columns[0])

        for current_res in resolution:
            print(f"\n{'='*50}\n[BlueTopo Engine] STARTING PROCESSING FOR {current_res}m\n{'='*50}")

            if not self.skip_tiling:
                print(f'Checking and processing BlueTopo Datasets for {current_res}...')
                param_inputs = []

                print(f"[BlueTopo Engine] Mapped Tile column: '{tile_col}' and EcoRegion column: '{er_col}'")

                for _, row in tile_gdf.iterrows():
                    # Check for NaNs safely
                    if pd.isna(row.get(er_col)):
                        continue
                    
                    # We can safely assume row[er_col] is an int (1-6) per system design
                    ecoregion_id = row[er_col]
                    
                    if ecoregion_id not in all_ecoregions:
                        continue
                    
                    raw_id = str(row[tile_col]).strip()
                    clean_id = raw_id.upper().replace('BLUETOPO_', '').replace('BLUETOPO', '')
                    
                    match = re.search(r'(B[A-Z0-9]{7})', clean_id)
                    if not match:
                        print(f"Skipping invalid tile format: {raw_id}")
                        continue
                    
                    tile_id = match.group(1)
                    param_inputs.append([self.param_lookup, tile_id, ecoregion_id, output_prefix, current_res])

                # Sort by ecoregion number to group them together logically for Dask
                param_inputs = sorted(param_inputs, key=lambda x: _get_er_num(x[2]))

                if param_inputs:
                    print(f"Submitting {len(param_inputs)} tiles to Dask workers...")
                    future_tiles = self.client.map(_process_tile, param_inputs)

                    print("Waiting for all Dask workers to complete...")
                    tile_results = self.client.gather(future_tiles)
                    print("All Dask workers finished successfully.")

                    self.print_async_results(tile_results, self.param_lookup['output_directory'].valueAsText)
                else:
                    print(f"[BlueTopo Engine] No tiles to process for {current_res}m.")

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

        self.close_dask()

        tiles = list(tile_gdf[tile_col]) if tile_col and tile_col in tile_gdf.columns else []
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test') 

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation to no data value using windowed block processing."""

        print(f"[{tiff_file_path.name}] Setting ground to nodata...")
        no_data = -9999
        
        with rasterio.open(tiff_file_path, "r+") as src:
            for _, window in src.block_windows(1):
                band1 = src.read(1, window=window)
                band1 = np.where(band1 < 0, band1, no_data)
                src.write(band1, 1, window=window)

    def upload_current_tiles_to_s3(self, tile_folder: pathlib.Path, temp_folder: pathlib.Path) -> None:
        """Upload all tiff files to s3 for current tile."""

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        for tiff_file in tile_folder.glob('*.tiff'):
            filename = tiff_file.name
            
            s3_path = tiff_file.relative_to(temp_folder)  # Strip off the temp_folder parts
            self.write_message(f'Uploading {filename} to s3://{bucket_name}/{s3_path}', self.param_lookup['output_directory'].valueAsText)
            s3_client.upload_file(str(tiff_file), bucket_name, str(s3_path))