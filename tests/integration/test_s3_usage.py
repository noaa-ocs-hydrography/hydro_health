import rasterio
import sys
import pathlib
import s3fs
import random

import numpy as np

HYDRO_HEALTH_MODULE = pathlib.Path(__file__).parents[2] / 'src'
print(HYDRO_HEALTH_MODULE)
sys.path.append(str(HYDRO_HEALTH_MODULE))

from hydro_health.helpers.tools import get_config_item


"""
This is an integration test suite for design concerns we need to make sure never change.
This test file needs to be run on the EC2 in order to access S3.
Steps:
1. SFTP the file to the EC2
2. SSH onto the EC2
3. Change directory to Repos/hydro_health/tests/integration
4. run the test: pytest test_s3_usage.py -s
"""


def test_provider_folder_has_vrt():
    """Check if each provider has a VRT file"""

    s3_files = s3fs.S3FileSystem()
    for i in range(1, 7):
        ecoregion = f'ER_{i}'
        digital_coast_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET', 'aws')}/{ecoregion}/{get_config_item('DIGITALCOAST', 'SUBFOLDER', 'aws')}/DigitalCoast"
        prefixes = list(s3_files.glob(f'{digital_coast_path}/*'))

        vrt_files = []
        provider_folders = []
        for prefix in prefixes:
            if '.vrt' in prefix:
                vrt_files.append(prefix)
            else:
                provider_folders.append(prefix)
        for provider in provider_folders:
            provider_path = pathlib.Path(provider)
            assert any(provider_path.name in vrt for vrt in vrt_files)


def test_bluetopo_tiling_count():
    """Test BlueTopo folders in tiled folder <= number in main BlueTopo folder"""

    s3_files = s3fs.S3FileSystem()
    for i in range(1, 7):
        ecoregion = f'ER_{i}'
        pre_processed_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET', 'aws')}/{ecoregion}/model_variables/Prediction/pre_processed"
        bluetopo_folders = list(s3_files.glob(f'{pre_processed_path}/BlueTopo/*'))
        tiled_folders = list(s3_files.glob(f'{pre_processed_path}/tiled/*'))
        for tiled_folder in tiled_folders:
            tiled_folder_path = pathlib.Path(tiled_folder)
            assert any(tiled_folder_path.name in bt for bt in bluetopo_folders)


def test_provider_vrt_crs_matches_geotiff():
    """Check if VRT has same CRS as geotiffs"""

    s3_files = s3fs.S3FileSystem()
    for i in range(1, 7):
        ecoregion = f'ER_{i}'
        digital_coast_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET', 'aws')}/{ecoregion}/{get_config_item('DIGITALCOAST', 'SUBFOLDER', 'aws')}/DigitalCoast"
        prefixes = list(s3_files.glob(f'{digital_coast_path}/*'))
        vrt_files = []
        provider_folders = []
        for prefix in prefixes:
            if '.vrt' in prefix:
                vrt_files.append(prefix)
            else:
                provider_folders.append(prefix)
        for provider in provider_folders:
            provider_path = pathlib.Path(provider)
            vrt_found = None
            for vrt in vrt_files:
                if provider_path.name in vrt:
                    vrt_found = vrt
                    break
            if vrt_found:
                geotiff = random.choice(list(s3_files.glob(f'{provider}/dem/**/*.tif')))
                with rasterio.open(f's3://{vrt_found}') as vrt, rasterio.open(f's3://{geotiff}') as src:
                    if vrt.crs != src.crs:
                        # TODO Sometimes geotiff has a compound CRS
                        print(vrt_found, vrt.crs)
                        print(geotiff, src.crs)
                    assert vrt.crs == src.crs
                    assert vrt.driver == 'VRT'


def test_mask_metadata_and_values():
    s3_files = s3fs.S3FileSystem()
    for i in range(1, 7):
        ecoregion = f'ER_{i}'
        training_mask = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET', 'aws')}/masks/{ecoregion}/training_mask_{ecoregion}.tif"
        if s3_files.exists(training_mask):
            with rasterio.open(training_mask) as src:
                assert len(src.overviews(1)) > 0
                data = src.read(1)
                unique_values = np.unique(data)
                for val in [0, 1, 2]:
                    assert val in unique_values, f"Value {val} missing from {training_mask}"
                assert set(unique_values).issubset(set([0, 1, 2]))
        prediction_mask = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET', 'aws')}/masks/{ecoregion}/prediction_mask_{ecoregion}.tif"
        if s3_files.exists(prediction_mask):
            with rasterio.open(prediction_mask) as src:
                assert len(src.overviews(1)) > 0
                data = src.read(1)
                unique_values = np.unique(data)
                for val in [0, 1]:
                    assert val in unique_values, f"Value {val} missing from {prediction_mask}"
                assert set(unique_values).issubset(set([0, 1]))
