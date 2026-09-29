"""
Unified BlueTopoEngine for Local (Windows) and AWS S3/EC2 processing environments.
Handles NBS tile downloading, resampling, reclassification, slope derivation, and COG formatting.
"""

import os
import sys
import re
import shutil
import pathlib
import tempfile
from datetime import datetime, date
from multiprocessing import set_executable
from contextlib import nullcontext

import boto3
from botocore.client import Config
from botocore import UNSIGNED
from lxml import etree
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from osgeo import gdal

from hydro_health.helpers import hibase_logging
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine, supersession, catzoc


def _process_tile(param_inputs: list) -> str:
    """
    Static top-level entry point for Dask parallel workers.
    Dispatches processing through the unified BlueTopoEngine.
    """

    param_lookup, tile_id, ecoregion_id, output_prefix, target_res = param_inputs
    engine = BlueTopoEngine(param_lookup)

    # Use TemporaryDirectory for AWS/S3 mode, otherwise nullcontext for local mode
    ctx = tempfile.TemporaryDirectory(dir=engine.scratch_dir) if param_lookup['env'] == 'aws' else nullcontext()

    with ctx as temp_dir:
        temp_path = pathlib.Path(temp_dir) if temp_dir else None
        
        tiff_file_path = engine.download_nbs_tile(
            tile_id, ecoregion_id, output_prefix, target_res, temp_folder=temp_path
        )
        if not tiff_file_path:
            msg = f"[{tile_id}] Skipped (Source tile download or resolution failed)."
            print(msg)
            return msg

        engine.resample_and_reproject(tiff_file_path, target_res)
        engine.create_survey_end_date_tiff(tiff_file_path)
        engine.create_catzoc_all(tiff_file_path, increased_scale=True)
        engine.create_catzoc_latest(tiff_file_path, increased_scale=True)
        engine.create_slope(tiff_file_path)

        mb_tiff_file = engine.rename_multiband(tiff_file_path)
        engine.multiband_to_singleband(mb_tiff_file, band=1)
        engine.multiband_to_singleband(mb_tiff_file, band=2)
        if mb_tiff_file.exists():
            mb_tiff_file.unlink()

        engine.set_ground_to_nodata(tiff_file_path)
        engine.finalize_cog(tiff_file_path)

        if param_lookup['env'] == 'aws':
            print(f"[{tile_id}] Uploading output rasters to S3...")
            tile_folder = tiff_file_path.parents[0]
            engine.upload_current_tiles_to_s3(tile_folder, temp_path)

    msg = f"[{tile_id}] Processing successfully completed."
    print(msg)
    return msg


class BlueTopoEngine(Engine):
    """Unified engine for local Windows and AWS EC2/S3 BlueTopo workflow processing."""

    def __init__(self, param_lookup: dict):
        super().__init__()
        self.param_lookup = param_lookup
        self.skip_tiling = False

        # Cloud scratch space initialization
        self.scratch_dir = pathlib.Path.home() / "scratch_tmp" / "BlueTopoEngine"
        if self.param_lookup['env'] == 'aws':
            self.scratch_dir.mkdir(parents=True, exist_ok=True)

    def get_bucket(self) -> boto3.resource:
        """Connect to public anonymous NOAA OCS National Bathymetry S3 Bucket."""

        bucket_name = "noaa-ocs-nationalbathymetry-pds"
        creds = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "config": Config(signature_version=UNSIGNED),
        }
        s3 = boto3.resource('s3', **creds)
        return s3.Bucket(bucket_name)

    def download_nbs_tile(
        self, 
        tile_id: str, 
        ecoregion_id: str, 
        output_prefix: str | bool, 
        target_res: int, 
        temp_folder: pathlib.Path | None = None
    ) -> pathlib.Path | None:
        """
        Download required source NBS GeoTIFF and PAM XML metadata from S3.
        Routes destination paths based on local disk or cloud temp staging.
        """

        nbs_bucket = self.get_bucket()
        output_tile_path = None
        output_folder = self.param_lookup['output_directory'].valueAsText

        base_dest_dir = temp_folder if self.param_lookup['env'] == 'aws' and temp_folder else pathlib.Path(output_folder)

        if output_prefix == 'low_res':
            beginning_prefix = base_dest_dir / output_prefix / f'{target_res}m'
        elif output_prefix:
            beginning_prefix = base_dest_dir / output_prefix
        else:
            beginning_prefix = base_dest_dir

        found_files = False
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            found_files = True
            current_file = beginning_prefix / ecoregion_id / get_config_item('BLUETOPO', 'SUBFOLDER') / obj_summary.key

            if current_file.suffix in ('.tiff', '.tif', '.xml'):
                tile_folder = current_file.parents[0]
                tile_folder.mkdir(parents=True, exist_ok=True)

                self.write_message(f'Downloading: {current_file.name}', output_folder)

                if current_file.exists():
                    current_file.unlink()  # Clean partial downloads

                nbs_bucket.download_file(obj_summary.key, str(current_file))

                if current_file.suffix in ('.tiff', '.tif'):
                    output_tile_path = current_file

        if not found_files:
            error_msg = f"[{tile_id}] CRITICAL: No source files found in NBS bucket under 'BlueTopo/{tile_id}'."
            print(error_msg)
            self.write_message(error_msg, output_folder)
            return None

        return output_tile_path

    def resample_and_reproject(self, tiff_file_path: pathlib.Path, target_res: int) -> None:
        """
        Resamples a 3-band BlueTopo tile to target_res while preserving its local UTM projection.
        Uses Bilinear for Elevation (B1) & Uncertainty (B2), and NearestNeighbour for the Contributor metadata (B3).
        Intermediate files are placed in an isolated temporary directory.
        """
        
        ds = gdal.Open(str(tiff_file_path))
        if ds is None: return
        
        gt = ds.GetGeoTransform()
        x_res = abs(gt[1])
        ds = None # EXPLICIT CLEANUP (No locked GDAL pointers!)
        
        # If the tile is already at the correct resolution, skip resampling
        if abs(x_res - target_res) < 0.1:
            print(f"[{tiff_file_path.name}] Tile is already at target resolution {target_res}m. Skipping resampling.")
            return

        print(f"[{tiff_file_path.name}] Resampling base tile to {target_res}m...")
        
        # Keep original XML safe using tile-unique filename to prevent Dask collisions
        xml_path = tiff_file_path.parent / f"{tiff_file_path.name}.aux.xml"
        safe_xml = tiff_file_path.parent / f"safe_{tiff_file_path.stem}.xml"
        if xml_path.exists():
            shutil.copy(xml_path, safe_xml)
            
        # Place all intermediate files inside an isolated temp directory on NVMe/scratch storage
        scratch_base = self.scratch_dir if self.param_lookup['env'] == 'aws' else tiff_file_path.parent
        with tempfile.TemporaryDirectory(dir=scratch_base) as warp_tmpdir:
            tmp_path = pathlib.Path(warp_tmpdir)
            temp_b12_src = tmp_path / f"b12_src_{tiff_file_path.name}"
            temp_b3_src = tmp_path / f"b3_src_{tiff_file_path.name}"
            temp_b12_warp = tmp_path / f"b12_warp_{tiff_file_path.name}"
            temp_b3_warp = tmp_path / f"b3_warp_{tiff_file_path.name}"
            final_warp = tmp_path / f"warp_{tiff_file_path.name}"
            
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
            
            # Resample Bands 1 & 2 (Bilinear)
            ds3 = gdal.Warp(str(temp_b12_warp), str(temp_b12_src), options=gdal.WarpOptions(
                format="GTiff", xRes=target_res, yRes=target_res,
                resampleAlg=gdal.GRA_Bilinear, targetAlignedPixels=True,
                creationOptions=creation_opts
            ))
            ds3 = None
            
            # Resample Band 3 (Nearest Neighbor)
            ds4 = gdal.Warp(str(temp_b3_warp), str(temp_b3_src), options=gdal.WarpOptions(
                format="GTiff", xRes=target_res, yRes=target_res,
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
                
                # Replace original tile with final warped file safely
                if tiff_file_path.exists():
                    tiff_file_path.unlink()
                shutil.move(str(final_warp), str(tiff_file_path))
                
                # Restore the RAT XML to the final warped file (Cross-platform overwrite fix)
                if safe_xml.exists():
                    target_xml = tiff_file_path.parent / f"{tiff_file_path.name}.aux.xml"
                    if target_xml.exists():
                        target_xml.unlink()
                    shutil.move(str(safe_xml), str(target_xml))

    def create_catzoc_all(self, tiff_file_path: pathlib.Path, increased_scale: bool = False) -> None:
        """Generate an Initial Survey Score (ISS) raster for all combined survey areas."""

        print(f"[{tiff_file_path.name}] Creating Initial Survey Score (ISS) all...")
        nodata = -9999
        with rasterio.open(tiff_file_path) as src:
            band3_raw = src.read(3)
            contributor_band_values = np.nan_to_num(np.round(band3_raw), nan=nodata).astype(np.int32)
            transform, width, height, native_crs = src.transform, src.width, src.height, src.crs

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
            data = {
                "value": float(row_data.get('value', 0) or 0),
                'start_date': self.parse_survey_date(row_data.get('survey_date_start')),
                "end_date": self.parse_survey_date(row_data.get('survey_date_end')),
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

        max_val = int(contributor_band_values.max()) if contributor_band_values.size > 0 else 0
        lookup_array = np.full(max_val + 1, nodata, dtype=np.float32)
        for val, iss in reclass_dict.items():
            if 0 <= val <= max_val:
                lookup_array[val] = iss

        valid_mask = (contributor_band_values >= 0) & (contributor_band_values <= max_val)
        reclassified_band = np.full_like(contributor_band_values, nodata, dtype=np.float32)
        reclassified_band[valid_mask] = lookup_array[contributor_band_values[valid_mask]]

        output_path = tiff_file_path.parents[0] / f"{tiff_file_path.stem}_ISS_all{'_110' if increased_scale else ''}.tiff"
        with rasterio.open(
            output_path, "w", driver="GTiff", count=1, width=width, height=height,
            dtype=rasterio.float32, compress="lzw", tiled=True, blockxsize=512, blockysize=512,
            crs=native_crs, transform=transform, nodata=nodata
        ) as dst:
            dst.write(reclassified_band, 1)
            dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_catzoc_latest(self, tiff_file_path: pathlib.Path, increased_scale: bool = False) -> None:
        """Generate an Initial Survey Score (ISS) raster using the most recent survey date."""

        print(f"[{tiff_file_path.name}] Creating Initial Survey Score (ISS) latest...")
        nodata = -9999
        with rasterio.open(tiff_file_path) as src:
            band3_raw = src.read(3)
            contributor_band_values = np.nan_to_num(np.round(band3_raw), nan=nodata).astype(np.int32)
            transform, width, height, native_crs = src.transform, src.width, src.height, src.crs

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
            meta = {
                "end_date": self.parse_survey_date(row_dict.get('survey_date_end')) or date.min,
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

        output_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}_ISS_latest.tiff'
        with rasterio.open(
            output_path, "w", driver="GTiff", count=1, width=width, height=height,
            dtype=rasterio.float32, compress="lzw", tiled=True, blockxsize=512, blockysize=512,
            crs=native_crs, transform=transform, nodata=nodata
        ) as dst:
            dst.write(reclassified_band, 1)
            dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_survey_end_date_tiff(self, tiff_file_path: pathlib.Path) -> None:
        """Create survey end date tiffs from contributor band values in the XML file."""

        print(f"[{tiff_file_path.name}] Creating survey end date TIFF...")
        nodata = -9999
        with rasterio.open(tiff_file_path) as src:
            band3_raw = src.read(3)
            contributor_band_values = np.nan_to_num(np.round(band3_raw), nan=nodata).astype(np.int32)
            transform, width, height, native_crs = src.transform, src.width, src.height, src.crs

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
            table_data.append({
                "value": float(row_dict.get('value', 0) or 0),
                "survey_date_end": self.parse_survey_date(row_dict.get('survey_date_end'))
            })

        attribute_table_df = pd.DataFrame(table_data)
        attribute_table_df['survey_year_end'] = attribute_table_df['survey_date_end'].apply(
            lambda x: int(round(x.year)) if pd.notna(x) else 0
        )

        date_mapping = attribute_table_df[['value', 'survey_year_end']].drop_duplicates()
        reclass_dict = {int(row[0]): int(row[1]) for row in date_mapping.to_numpy()}

        max_val = int(contributor_band_values.max()) if contributor_band_values.size > 0 else 0
        lookup_array = np.full(max_val + 1, nodata, dtype=np.float32)
        for val, yr in reclass_dict.items():
            if 0 <= val <= max_val:
                lookup_array[val] = yr

        valid_mask = (contributor_band_values >= 0) & (contributor_band_values <= max_val)
        reclassified_band = np.full_like(contributor_band_values, nodata, dtype=np.float32)
        reclassified_band[valid_mask] = lookup_array[contributor_band_values[valid_mask]]

        output_path = tiff_file_path.parents[0] / f'{tiff_file_path.stem}_survey_end_date.tiff'
        with rasterio.open(
            output_path, "w", driver="GTiff", count=1, width=width, height=height,
            dtype=rasterio.float32, compress="lzw", crs=native_crs, transform=transform, nodata=nodata
        ) as dst:
            dst.write(reclassified_band, 1)

    def create_slope(self, tiff_file_path: pathlib.Path) -> None:
        """Generate a slope raster from the DEM."""

        slope_name = str(tiff_file_path.stem) + '_slope.tiff'
        slope_file_path = tiff_file_path.parents[0] / slope_name
        gdal.DEMProcessing(str(slope_file_path), str(tiff_file_path), 'slope')

    def set_ground_to_nodata(self, tiff_file_path: pathlib.Path) -> None:
        """Set positive elevation values to nodata (-9999) using memory-safe windowed processing."""

        no_data = -9999
        with rasterio.open(tiff_file_path, "r+") as src:
            for _, window in src.block_windows(1):
                band1 = src.read(1, window=window)
                band1 = np.where(band1 < 0, band1, no_data)
                src.write(band1, 1, window=window)

    def finalize_cog(self, tiff_path: pathlib.Path) -> None:
        """Final pass guaranteeing Cloud-Optimized GeoTIFF (COG) layout and interior overviews."""
        temp_cog = tiff_path.parent / f"temp_{tiff_path.name}"

        gdal.Translate(
            str(temp_cog),
            str(tiff_path),
            format="COG",
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

    def rename_multiband(self, tiff_file_path: pathlib.Path) -> pathlib.Path:
        """Update file name for singleband conversion"""

        new_name = str(tiff_file_path).replace('.tiff', '_mb.tiff')
        mb_tiff_file = tiff_file_path.replace(pathlib.Path(new_name))
        mb_tiff_file = pathlib.Path(new_name)
        return mb_tiff_file

    def multiband_to_singleband(self, tiff_file_path: pathlib.Path, band: int) -> None:
        """Split a multiband BlueTopo raster into individual singleband rasters."""

        band_name_lookup = {1: '', 2: '_unc'}
        output_name = str(tiff_file_path.name).replace('_mb', band_name_lookup[band])
        singleband_tile_name = tiff_file_path.parents[0] / output_name

        gdal.Translate(
            str(singleband_tile_name),
            str(tiff_file_path),
            bandList=[band],
            creationOptions=["COMPRESS=DEFLATE"]
        )

    def upload_current_tiles_to_s3(self, tile_folder: pathlib.Path, temp_folder: pathlib.Path) -> None:
        """Upload target generated GeoTIFF files to destination S3 output bucket."""

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')

        for tiff_file in tile_folder.glob('*.tiff'):
            s3_path = tiff_file.relative_to(temp_folder)
            self.write_message(
                f'Uploading {tiff_file.name} to s3://{bucket_name}/{s3_path}',
                self.param_lookup['output_directory'].valueAsText
            )
            s3_client.upload_file(str(tiff_file), bucket_name, str(s3_path))

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Log async job outputs."""

        for result in results:
            if result:
                self.write_message(result, output_folder)

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str | bool, resolution: list[int]) -> None:
        """Main execution workflow running parallel tile processing via Dask."""

        print('Starting BlueTopoEngine')

        all_ecoregions = self.param_lookup['eco_regions'].value

        if self.param_lookup['env'] == 'aws':
            self.setup_dask(self.param_lookup['env'], n_workers=1, threads_per_worker=1)
        else:
            self.setup_dask(self.param_lookup['env'])

        tile_col = next((c for c in tile_gdf.columns if str(c).lower() in ['tile', 'tile_id', 'name', 'id', 'bluetopo']), tile_gdf.columns[0])
        er_col = 'EcoRegion' if 'EcoRegion' in tile_gdf.columns else tile_gdf.columns[1]

        for current_res in resolution:
            print(f"- Processing resolution {current_res}m")

            if not self.skip_tiling:
                param_inputs = []
                for _, row in tile_gdf.iterrows():
                    if pd.isna(row.get(er_col)):
                        continue

                    ecoregion_id = row[er_col]
                    if all_ecoregions and ecoregion_id not in all_ecoregions:
                        continue

                    raw_id = str(row[tile_col]).strip()
                    clean_id = raw_id.upper().replace('BLUETOPO_', '').replace('BLUETOPO', '')
                    match = re.search(r'(B[A-Z0-9]{7})', clean_id)
                    if not match:
                        continue

                    tile_id = match.group(1)
                    param_inputs.append([self.param_lookup, tile_id, ecoregion_id, output_prefix, current_res])

                if param_inputs:
                    future_tiles = self.client.map(_process_tile, param_inputs)
                    tile_results = self.client.gather(future_tiles)
                    self.print_async_results(tile_results, self.param_lookup['output_directory'].valueAsText)

                for ecoregion in all_ecoregions:
                    if output_prefix == 'low_res':
                        beginning_prefix = f'{output_prefix}/{current_res}m'
                    elif output_prefix:
                        beginning_prefix = output_prefix
                    else:
                        beginning_prefix = ''
                    
                    manifest_path = f"{beginning_prefix}/{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
                    self.write_run_manifest(manifest_path, {'tiles': len(param_inputs)})

        self.close_dask()

        tiles = list(tile_gdf[tile_col]) if tile_col in tile_gdf.columns else []
        record = {'data_source': 'hydro_health', 'user': os.getlogin(), 'tiles_downloaded': len(tiles), 'tile_list': tiles}
        hibase_logging.send_record(record, table='bluetopo_test')