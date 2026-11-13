import os
import pathlib
import posixpath
import sys
import time
import uuid

import boto3
from osgeo import gdal

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.engines.DigitalCoastEngine import _reproject_raster_gdal, _trim_memory
from hydro_health.engines.GridDigitalCoastEngine import GridDigitalCoastEngine
from hydro_health.engines.MetadataS3Engine import MetadataS3Engine
from hydro_health.engines.RasterMaskS3Engine import RasterMaskS3Engine
from hydro_health.engines.RasterVRTEngine import RasterVRTEngine
from hydro_health.helpers.tools import Param, get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class ManualDownloadEngine:
    """Class for processing manually downloaded DEM files"""

    def __init__(self, param_lookup) -> None:
        self.param_lookup = param_lookup

    def reproject_manual_downloads(self, output_prefix: str | bool = False, target_crs: str = "EPSG:32617") -> None:
        """
        Archives raw manual download GeoTIFFs into an /original/ S3 subfolder
        and overwrites the main provider keys with EPSG:32617 reprojected rasters.
        """
        s3_client = boto3.client('s3', region_name='us-east-2')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
        dc_subfolder = get_config_item('DIGITALCOAST', 'SUBFOLDER')

        scratch_dir = pathlib.Path.home() / "temp_scratch"
        scratch_dir.mkdir(parents=True, exist_ok=True)

        gdal.SetCacheMax(256 * 1024 * 1024)
        gdal.SetConfigOption("GDAL_TMPDIR", str(scratch_dir))
        gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")

        eco_regions = (
            self.param_lookup['eco_regions'].valueAsText
            if 'eco_regions' in self.param_lookup
            else ['ER_3']
        )

        for ecoregion in eco_regions:
            base_prefix = f"{ecoregion}/{dc_subfolder}/Digital_Coast_Manual_Downloads"
            if output_prefix:
                base_prefix = f"{str(output_prefix).strip('/')}/{base_prefix}"

            print(f"Scanning S3 prefix for manual downloads: s3://{bucket_name}/{base_prefix}")

            paginator = s3_client.get_paginator('list_objects_v2')
            existing_keys = set()
            for page in paginator.paginate(Bucket=bucket_name, Prefix=base_prefix):
                for obj in page.get('Contents', []):
                    existing_keys.add(obj['Key'])

            # Filter for top-level provider GeoTIFFs (exclude anything already inside /original/)
            tif_keys = sorted([
                key for key in existing_keys
                if key.lower().endswith(('.tif', '.tiff'))
                and '/original/' not in key
            ])

            print(f"Found {len(tif_keys)} manual GeoTIFF(s) under {base_prefix}")

            for i, s3_key in enumerate(tif_keys, start=1):
                parent_prefix = posixpath.dirname(s3_key)
                filename = posixpath.basename(s3_key)
                original_key = posixpath.join(parent_prefix, "original", filename)

                if original_key in existing_keys:
                    print(f" - ({i}/{len(tif_keys)}) Skipping (already archived & reprojected): {filename}")
                    continue

                unique_id = uuid.uuid4().hex[:8]
                tmp_raw_path = scratch_dir / f"manual_raw_{unique_id}.tif"
                tmp_rep_path = scratch_dir / f"manual_rep_{unique_id}.tif"

                try:
                    print(f" - ({i}/{len(tif_keys)}) Processing: {s3_key}")

                    s3_client.download_file(bucket_name, s3_key, str(tmp_raw_path))

                    # Reproject to target CRS locally via GDAL streaming
                    _reproject_raster_gdal(tmp_raw_path, tmp_rep_path, target_crs=target_crs)

                    # Server-side copy original S3 object to .../original/<filename>
                    s3_client.copy(
                        {'Bucket': bucket_name, 'Key': s3_key},
                        bucket_name,
                        original_key
                    )

                    s3_client.upload_file(str(tmp_rep_path), bucket_name, s3_key)
                    existing_keys.add(original_key)

                except Exception as e:
                    print(f"Error reprojecting {s3_key}: {e}")
                    continue
                finally:
                    if tmp_raw_path.exists():
                        tmp_raw_path.unlink()
                    if tmp_rep_path.exists():
                        tmp_rep_path.unlink()
                    _trim_memory()

    def process_grid_tiles(self, output_prefix) -> None:
        """Tile manually downloaded VRT files to BlueTopo"""

        engine = GridDigitalCoastEngine(self.param_lookup)
        engine.run(output_prefix=output_prefix, manual_downloads=False)

        engine = GridDigitalCoastEngine(self.param_lookup)
        engine.run(output_prefix=output_prefix, manual_downloads=True)

    def process_masks(self, output_prefix) -> None:
        """Create masks with manual download additions"""

        engine = RasterMaskS3Engine(self.param_lookup, pilot_mode=False)
        engine.run(str(OUTPUTS), output_prefix, manual_downloads=True)

    def process_metadata(self) -> None:
        """Build metadata text file for manually downloaded DEMs"""

        engine = MetadataS3Engine()
        digital_coast_path = (
            f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/ER_3/"
            f"{get_config_item('DIGITALCOAST', 'SUBFOLDER')}/Digital_Coast_Manual_Downloads"
        )
        engine.read_json_files(digital_coast_path, OUTPUTS)

    def process_vrt(self, output_prefix) -> None:
        """Create VRT files for each manually downloaded provider"""

        engine = RasterVRTEngine(self.param_lookup)
        engine.setup_dask('aws')
        engine.run(OUTPUTS, 'NCMP', 'ER_3', 'DigitalCoast', manual_downloads=True)
        engine.close_dask()

    def rebuild_training_mask(self) -> None:
        engine = RasterMaskS3Engine(self.param_lookup)
        result = engine.remerge_training_mask("ER_3", OUTPUTS)
        print(result)

    def run(self) -> None:
        if os.path.exists(OUTPUTS / 'log_prints.txt'):
            now = time.time()
            os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')

        # TODO need to add output_prefix to calls
        output_prefix = False

        # Reproject manual downloads first so VRTs, masks, and tiles use EPSG:32617
        self.reproject_manual_downloads(output_prefix)

        # # TODO VRT process runs both normal and manual downloads if using manual_downloads param
        # self.process_vrt(output_prefix)
        # self.process_masks(output_prefix)  # mask process runs DigitalCoast and Digital_Coast_Manual_Downloads
        # # self.rebuild_training_mask()
        # # TODO this grid tiling process calls the regular and manual downloads processes
        # # Need to run both folders separately or fix code
        # # self.process_grid_tiles(output_prefix)
        print('Done')


if __name__ == '__main__':
    param_lookup = {
        'output_directory': Param(str(OUTPUTS)),
        'eco_regions': Param(['ER_3']),
        'env': 'aws'
    }
    engine = ManualDownloadEngine(param_lookup)
    engine.run()