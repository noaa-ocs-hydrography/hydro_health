"""Class for obtaining all available files"""

import boto3
import os
import zipfile
import requests
import shutil
import geopandas as gpd
import pathlib
import os
import sys
import tempfile
import logging
import platform
import s3fs
import rasterio

from rasterio.io import MemoryFile
from multiprocessing import set_executable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict
from botocore.client import Config
from botocore import UNSIGNED
from io import BytesIO

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


if platform.system() == 'Windows':
    set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))

os.environ['SHAPE_RESTORE_SHX'] = 'YES'  # Reruns are throwing fiona DriverError for .SHX, line 60

# RedHat UTF encodings
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

logging.getLogger("pyproj").setLevel(logging.ERROR)


def _download_tile_index(param_inputs: list[list]) -> None:
    """Parallel process and download tile index shapefiles"""

    download_link, provider_folder, param_lookup, outputs = param_inputs

    engine = DigitalCoastS3Engine(param_lookup)

    # TODO Removed laz download until HH 2.0
    if get_config_item('DIGITALCOAST', 'BUCKET') in download_link and ('dem' in download_link):  # or 'laz' in download_link):
        _, data_file = download_link.replace('/index.html', '').split('.com')
        lidar_bucket = engine.get_bucket()
        for obj_summary in lidar_bucket.objects.filter(Prefix=f"{data_file[1:]}"):
            if 'tileindex' in obj_summary.key and obj_summary.key.endswith('.zip'):
                output_zip_file = provider_folder / obj_summary.key
                file_parent_folder = output_zip_file.parents[0]
                shp_path = output_zip_file.parents[0] / pathlib.Path(str(output_zip_file.stem) + '.shp')
                provider_and_file = str(pathlib.Path(*shp_path.parts[-4:]))
                if os.path.exists(shp_path):
                    engine.write_message(f' - Skipping index: {provider_and_file}', outputs)
                    continue
                else:
                    engine.write_message(f' - Downloading index: {provider_and_file}', outputs)
                file_parent_folder.mkdir(parents=True, exist_ok=True) 
                with open(output_zip_file, 'wb') as tile_index:
                    lidar_bucket.download_fileobj(obj_summary.key, tile_index)


def _download_intersected_datasets(param_inputs: list[list]) -> None:
    """Parallel process spatial filter and download of datasets"""

    tile_gdf, shp_path, outputs, param_lookup = param_inputs

    engine = DigitalCoastS3Engine(param_lookup)

    shp_df = gpd.read_file(shp_path, engine="pyogrio")
    shp_df['geometry'] = shp_df.geometry.make_valid()
    shp_df = shp_df.to_crs(4326)

    shp_df.columns = shp_df.columns.str.lower()  # make url column all lowercase
    if 'tile' in shp_df.columns:
        # drop previous tile merged column
        shp_df.drop('tile', axis=1, inplace=True)
    df_joined = shp_df.sjoin(df=tile_gdf, how='left')

    if 'index_right' in df_joined.columns:
        df_joined = df_joined.rename(columns={'index_right': 'idx_right'})
    shp_df = None
    df_joined.to_file(shp_path, driver='ESRI Shapefile')
    if df_joined['url'].any():
        df_joined = df_joined.loc[df_joined['tile'].notnull()]
        shp_folder = shp_path.parents[0]
        urls = df_joined['url'].unique()
        engine.write_message(f'{shp_path.name} URLs: {len(urls)}', outputs)
        for i, url in enumerate(urls):
            cleansed_url = engine.cleansed_url(url)
            # Only download .tif files
            if not cleansed_url.endswith('.tif'):
                continue
            dataset_name = cleansed_url.split('/')[-1]
            output_file = shp_folder / dataset_name
            try:
                retry_strategy = Retry(
                    total=3,  # retries
                    backoff_factor=1,  # delay in seconds
                    status_forcelist=[404],  # Status codes to retry on
                    allowed_methods=["GET"]
                )
                adapter = HTTPAdapter(max_retries=retry_strategy)
                request_session = requests.Session()
                request_session.mount("https://", adapter)
                request_session.mount("http://", adapter)

                intersected_response = request_session.get(cleansed_url, timeout=5)
            except requests.exceptions.ConnectionError:
                engine.write_message(f'Timeout error: {cleansed_url}', outputs)
                continue
                
            if intersected_response.status_code == 200:
                # Open the downloaded bytes in memory and update for COG
                with MemoryFile(intersected_response.content) as memfile:
                    with memfile.open() as src:
                        cog_buffer = BytesIO()
                        dst_profile = src.profile.copy()
                        dst_profile.update({
                            "driver": "GTiff",
                            "tiled": True,
                            "blockxsize": 512,
                            "blockysize": 512,
                            "compress": "deflate",
                            "predictor": 3,
                            "interleave": "pixel"
                        })

                        with MemoryFile(cog_buffer) as out_memfile:
                            with out_memfile.open(**dst_profile) as dst:
                                dst.write(src.read())
                                factors = [2, 4, 8, 16]
                                dst.build_overviews(factors, rasterio.enums.Resampling.average)
                                dst.update_tags(ns='rio_overview', resampling='average')
                            
                            cog_buffer.seek(0)
                            
                            ecoregion_folder = output_file.parents[8]
                            s3_prefix = str(output_file.relative_to(ecoregion_folder)).replace("\\", "/")
                            engine.upload_bytes_to_s3(cog_buffer, s3_prefix)
            else:
                return f'Failed to download: {cleansed_url}'
        return f'- {shp_path.stem}'
    else:
        return f'- No intersect: {shp_path.stem}'
        

class DigitalCoastS3Engine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self, param_lookup: dict[dict]):
        super().__init__()
        self.param_lookup = param_lookup

    def breakup_cudem(self, ecoregion: str) -> None:
        """
        THIS IS NOT USED
        Create unique year folders for CUDEM data
        """

        s3_files = s3fs.S3FileSystem()
        digital_coast_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/testing/{ecoregion}/{get_config_item('DIGITALCOAST', 'SUBFOLDER')}/DigitalCoast"
        cudem_folders = s3_files.glob(f"{digital_coast_path}/NOAA_NCEI_0*")
        for folder in cudem_folders:
            year_tifs = defaultdict(list)
            tif_files = s3_files.glob(f'{folder}/**/*.tif')

            for tif_file in tif_files:
                filename = tif_file.split('/')[-1]
                year = filename[-10:-6]
                year_tifs[year].append(tif_file)
            for year in year_tifs:
                print(f'Organizing {year} CUDEM files in S3')
                first_tif = year_tifs[year][0]
                project_folder_path = "/".join(first_tif.split('/')[:-1])
                project_folder_name = project_folder_path.split('/')[-1]
                folder_suffix = folder.split('_')[-1]
                year_folder = f'{digital_coast_path}/NOAA_NCEI_{year}_{folder_suffix}/dem/{project_folder_name}'
                
                # Copy year tifs to new folder
                for tif_file in year_tifs[year]:
                    filename = tif_file.split('/')[-1]
                    output_tif_path = f'{year_folder}/{filename}'
                    s3_files.cp(tif_file, output_tif_path)
                    s3_files.rm(tif_file)

                # Copy tileindex (SHP files) with geopandas
                tile_index_files = s3_files.glob(f'{project_folder_path}/*index*.shp')
                for tile_index in tile_index_files:
                    tile_index_name = tile_index.split('/')[-1]                    
                    base_name = tile_index.rsplit('.', 1)[0]  # Get path without '.shp'
                    all_components = s3_files.glob(f'{base_name}.*') 

                    for component in all_components:
                        ext = component.split('.')[-1]
                        target_path = f'{year_folder}/{tile_index_name.replace("shp", ext)}'
                        s3_files.cp(component, target_path)
                    
                # Copy feature.json
                path_parts = project_folder_path.split('/')
                feature_json_path = "/".join(path_parts[:-2]) + '/feature.json'
                if s3_files.exists(feature_json_path):
                    target_json = "/".join(year_folder.split('/')[:-2]) + '/feature.json'
                    s3_files.cp(feature_json_path, target_json)

            s3_files.rm(folder, recursive=True)

    def check_tile_index_areas(self, tile_index_shapefiles: list[pathlib.Path], outputs: str) -> None:
        """Exclude any small area surveys"""

        self.write_message(f'Checking area size of tileindex files', outputs)
        providers = []
        for shp_path in tile_index_shapefiles:
            shp_df = gpd.read_file(shp_path).to_crs(9822)  # Albers Equal Area
            shp_df['area'] = shp_df['geometry'].area
            total_area = shp_df["area"].sum()
            if total_area < self.approved_size:
                self.write_message(f' - provider too small: {total_area} - {shp_path}', outputs)
            else:
                providers.append(shp_path)
        return providers

    def delete_unused_folder(self, digital_coast_folder: pathlib.Path, outputs: str) -> None:
        """Delete any provider folders without a subfolder"""

        if digital_coast_folder.exists():
            self.write_message('Deleting empty provider folders', outputs)
            provider_folders = os.listdir(digital_coast_folder)
            provider_folders = [folder for folder in digital_coast_folder.glob('*') if folder.is_dir()]
            for provider in provider_folders:
                if 'unused_providers' != provider.stem:
                    provider_folder = digital_coast_folder / provider
                    data_types = os.listdir(provider_folder)
                    if not provider_folder.suffix and 'dem' not in data_types and 'laz' not in data_types:
                        self.write_message(f' - removing empty provider: {provider_folder}', outputs)
                        shutil.rmtree(provider_folder)

    def download_support_files(self, tile_gdf: gpd.GeoDataFrame, ecoregion: str, digital_coast_folder: pathlib.Path, temp_digital_coast_outputs: pathlib.Path, outputs: pathlib.Path) -> None:
        """Download tile_index shapefiles"""

        print('Download Support Files')
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)

        for geometry_coords in ecoregion_geom_strings:
            tile_index_links = self.get_available_datasets(geometry_coords, ecoregion, str(temp_digital_coast_outputs))  # TODO return all object keys
            bulk_download_params = [[link_dict['link'], link_dict['provider_path'], self.param_lookup, outputs] for link_dict in tile_index_links if link_dict['label'] == 'Bulk Download']
            future_tiles = self.client.map(_download_tile_index, bulk_download_params)
            _ = self.client.gather(future_tiles)

            self.unzip_all_files(digital_coast_folder)

    def get_bucket(self) -> boto3.resource:
        """Connect to anonymous OCS S3 Bucket"""

        creds = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "config": Config(signature_version=UNSIGNED),
        }
        s3 = boto3.resource('s3', **creds)
        nbs_bucket = s3.Bucket(get_config_item('DIGITALCOAST', 'BUCKET'))
        return nbs_bucket

    def print_async_results(self, results: list[str], output_folder: str = None) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                # print('Result:', result)
                self.write_message(f'Result: {result}', output_folder)

    def run(self, tile_gdf: gpd.GeoDataFrame) -> None:
        """Main entry point for downloading Digital Coast data"""

        print('Downloading Digital Coast Datasets')
        outputs = self.param_lookup['output_directory'].valueAsText
        provider_log = pathlib.Path(outputs) / "processed_providers.log"
        provider_log.touch(exist_ok=True) # Ensure file exists
        self.setup_dask(self.param_lookup['env'])
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_digital_coast_outputs = pathlib.Path(temp_dir)                    
                    digital_coast_folder = temp_digital_coast_outputs / ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
                    self.download_support_files(ecoregion_tile_gdf, ecoregion, digital_coast_folder, temp_digital_coast_outputs, outputs)

                    tile_index_shapefiles = [shp for shp in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(shp)]
                    approved_index_shapefiles = self.check_tile_index_areas(tile_index_shapefiles, outputs)

                    chunk_size = 5
                    for i in range(0, len(approved_index_shapefiles), chunk_size):
                        current_files = approved_index_shapefiles[i:i+chunk_size]
                        provider_folders = [shp_path.parents[0].name for shp_path in current_files]

                        for provider in provider_folders:
                            print(f'Processing provider: {provider}')
                        for provider in provider_folders:
                            completed_providers = set(provider_log.read_text().splitlines())
                            if provider in completed_providers:
                                print(f'Skipping already processed provider: {provider}')
                                continue
                            
                            param_inputs = [[ecoregion_tile_gdf, shp_path, outputs, self.param_lookup] for shp_path in current_files]
                            future_intersected_datasets = self.client.map(_download_intersected_datasets, param_inputs)
                            self.client.gather(future_intersected_datasets)
                            with open(provider_log, "a") as f:
                                f.write(f"{provider}\n")

                        # Upload tile_index files
                        for provider_shp in current_files:
                            provider_path_parts = provider_shp.parts
                            digital_coast_index = provider_path_parts.index('DigitalCoast')
                            provider_folder = pathlib.Path(*provider_path_parts[:digital_coast_index + 2])
                            self.upload_files_to_s3(provider_folder)

                # self.breakup_cudem(ecoregion)
        self.close_dask()

    def process_intersected_datasets(self, digital_coast_folder: pathlib.Path, ecoregion_tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Download intersected Digital Coast files"""

        self.write_message('Downloading elevation datasets', outputs)
        tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(folder)]
        param_inputs = [[ecoregion_tile_gdf, shp_path, outputs, self.param_lookup] for shp_path in tile_index_shapefiles]
        future_tiles = self.client.map(_download_intersected_datasets, param_inputs)
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)

    def unzip_all_files(self, output_folder: pathlib.Path) -> None:
        """Unzip all zip files in a folder"""

        for zipped_file in output_folder.rglob('*.zip'):
            with zipfile.ZipFile(zipped_file, 'r') as zipped:
                zipped.extractall(str(zipped_file.parents[0]))
            # Delete zip file after extract
            zipped_file.unlink()

    def upload_bytes_to_s3(self, file_object: BytesIO, s3_prefix: str) -> None:
        """Stream object to S3"""

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
        s3_client.upload_fileobj(file_object, bucket_name, f'testing/{s3_prefix}')

    def upload_files_to_s3(self, provider_folder: pathlib.Path) -> None:
        """Upload all tiff files to s3 for current tile"""
        
        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
        for found_file in provider_folder.rglob('*'):
            if found_file.is_file():
                s3_prefix = "/".join(found_file.parts[3:])
                print(f'Uploading {found_file} to {s3_prefix}')
                s3_client.upload_file(str(found_file), bucket_name, f'testing/{s3_prefix}')
