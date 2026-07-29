"""Class for obtaining all available files"""

import boto3
import os
import zipfile
import requests
import shutil
import geopandas as gpd
import pathlib
import rasterio
import sys
import tempfile

import numpy as np

from multiprocessing import set_executable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict
from botocore.client import Config
from botocore import UNSIGNED
from boto3.s3.transfer import TransferConfig

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
os.environ['SHAPE_RESTORE_SHX'] = 'YES'  # Reruns throwing fiona DriverError for .SHX

S3_SYNC_CONFIG = TransferConfig(use_threads=False)


def _to_win_long_path(path: pathlib.Path | str) -> str:
    """Formats path for native Python file I/O (open, shutil, os.path) on Windows."""

    path_str = str(pathlib.Path(path).resolve())
    if os.name == 'nt' and not path_str.startswith(r'\\?\\'):
        if path_str.startswith(r'\\'):  # UNC network path
            return r'\\?\UNC' + path_str[1:]
        return r'\\?\\' + path_str
    return path_str


def _get_short_temp_dir() -> str:
    """Returns a short base directory for GDAL temporary processing."""

    c_temp = pathlib.Path(r"C:\Temp")
    if c_temp.exists() or os.name == 'nt':
        try:
            c_temp.mkdir(parents=True, exist_ok=True)
            return str(c_temp)
        except Exception:
            pass
    return tempfile.gettempdir()


def _download_tile_index(param_inputs: list[list]) -> None:
    """Parallel process and download tile index shapefiles"""

    download_link, provider_folder, param_lookup, outputs = param_inputs

    engine = DigitalCoastEngine(param_lookup)

    if get_config_item('DIGITALCOAST', 'BUCKET') in download_link and ('dem' in download_link):
        _, data_file = download_link.replace('/index.html', '').split('.com')
        lidar_bucket = engine.get_bucket()
        
        for obj_summary in lidar_bucket.objects.filter(Prefix=f"{data_file[1:]}"):
            if 'tileindex' in obj_summary.key and obj_summary.key.endswith('.zip'):
                output_zip_file = provider_folder / obj_summary.key
                file_parent_folder = output_zip_file.parents[0]
                shp_path = file_parent_folder / pathlib.Path(str(output_zip_file.stem) + '.shp')
                provider_and_file = str(pathlib.Path(*shp_path.parts[-4:]))
                
                # Check existence using extended path in Python
                abs_shp = _to_win_long_path(shp_path)

                if os.path.exists(abs_shp):
                    engine.write_message(f' - Skipping index: {provider_and_file}', outputs)
                    continue
                else:
                    engine.write_message(f' - Downloading index: {provider_and_file}', outputs)

                # Ensure target destination parents exist
                pathlib.Path(_to_win_long_path(file_parent_folder)).mkdir(parents=True, exist_ok=True)

                abs_zip_path = _to_win_long_path(output_zip_file)

                # Single-threaded transfer prevents Dask C-extension/GIL crashes
                lidar_bucket.download_file(
                    Key=obj_summary.key, 
                    Filename=abs_zip_path,
                    Config=S3_SYNC_CONFIG
                )


def _download_intersected_datasets(param_inputs: list[list]) -> None:
    """Parallel process spatial filter and download of datasets using a short temp working dir for GDAL"""

    tile_gdf, shp_path, param_lookup, outputs = param_inputs

    engine = DigitalCoastEngine(param_lookup)
    shp_path_obj = pathlib.Path(shp_path)

    # Use a short temporary directory for GDAL/Fiona operations
    with tempfile.TemporaryDirectory(dir=_get_short_temp_dir()) as tmp_dir:
        tmp_folder = pathlib.Path(tmp_dir)
        temp_shp = tmp_folder / shp_path_obj.name

        # Copy original shapefile sidecar files (.shp, .shx, .dbf, .prj) to temp dir using Python
        target_dir_long = _to_win_long_path(shp_path_obj.parent)
        stem = shp_path_obj.stem
        
        if os.path.exists(target_dir_long):
            for filename in os.listdir(target_dir_long):
                if filename.startswith(stem + "."):
                    src_file = _to_win_long_path(shp_path_obj.parent / filename)
                    shutil.copy(src_file, str(tmp_folder / filename))

        shp_df = gpd.read_file(str(temp_shp)).to_crs(4326)
        shp_df.columns = shp_df.columns.str.lower()
        if 'tile' in shp_df.columns:
            shp_df.drop('tile', axis=1, inplace=True)
            
        df_joined = shp_df.sjoin(df=tile_gdf, how='left')

        try:
            df_joined.to_file(str(temp_shp), driver='ESRI Shapefile', engine='pyogrio', encoding='utf-8')
        except Exception:
            df_joined.to_file(str(temp_shp), driver='ESRI Shapefile', engine='fiona', mode='w', encoding='utf-8')

        # 3. Move updated shapefile back to final long output path using standard Python
        for sidecar in tmp_folder.glob(f"{stem}.*"):
            dst_file = _to_win_long_path(shp_path_obj.parent / sidecar.name)
            shutil.copy(str(sidecar), dst_file)

        if df_joined['url'].any():
            df_joined = df_joined.loc[df_joined['tile'].notnull()]
            shp_folder = shp_path_obj.parents[0]
            urls = df_joined['url'].unique()
            for i, url in enumerate(urls):
                cleansed_url = engine.cleansed_url(url)
                if not cleansed_url.endswith('.tif'):
                    continue
                dataset_name = cleansed_url.split('/')[-1]
                output_file = shp_folder / dataset_name
                
                win_out_file = _to_win_long_path(output_file)

                if os.path.exists(win_out_file):
                    engine.write_message(f' - ({i} of {len(urls)}) Skipping data: {output_file.stem}', outputs)
                    continue
                else:
                    engine.write_message(f' - ({i} of {len(urls)}) Downloading data: {output_file.stem}', outputs)

                try:
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[404],
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
                    with open(win_out_file, 'wb') as file:
                        file.write(intersected_response.content)
                else:
                    return f'Failed to download: {cleansed_url}'
            return f'- {shp_path_obj.stem}'
        else:
            return f'- No intersect: {shp_path_obj.stem}'


class DigitalCoastEngine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self, param_lookup: dict[dict]):
        super().__init__()
        self.param_lookup = param_lookup

    def check_tile_index_areas(self, digital_coast_folder, outputs) -> None:
        """Exclude any small area surveys using a temp location for GeoPandas area calculations"""

        self.write_message('Checking area size of tileindex files', outputs)
        tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(folder)]
        
        for shp_path in tile_index_shapefiles:
            with tempfile.TemporaryDirectory(dir=_get_short_temp_dir()) as tmp_dir:
                tmp_folder = pathlib.Path(tmp_dir)
                temp_shp = tmp_folder / shp_path.name
                
                # Copy sidecar files to temp dir
                target_dir_long = _to_win_long_path(shp_path.parent)
                stem = shp_path.stem
                if os.path.exists(target_dir_long):
                    for filename in os.listdir(target_dir_long):
                        if filename.startswith(stem + "."):
                            src_file = _to_win_long_path(shp_path.parent / filename)
                            shutil.copy(src_file, str(tmp_folder / filename))

                shp_df = gpd.read_file(str(temp_shp)).to_crs(9822)  # Albers Equal Area
                shp_df['area'] = shp_df['geometry'].area
                total_area = shp_df["area"].sum()
                
                if total_area < self.approved_size:
                    self.write_message(f' - provider too small: {total_area} - {shp_path}', outputs)
                    provider_dir = _to_win_long_path(shp_path.parents[2])
                    if os.path.exists(provider_dir):
                        shutil.rmtree(provider_dir)

    def delete_unused_folder(self, digital_coast_folder: pathlib.Path, outputs: str) -> None:
        """Delete any provider folders without a subfolder"""

        dc_folder_long = _to_win_long_path(digital_coast_folder)
        if os.path.exists(dc_folder_long):
            self.write_message('Deleting empty provider folders', outputs)
            provider_folders = [f for f in pathlib.Path(dc_folder_long).glob('*') if f.is_dir()]
            for provider in provider_folders:
                if 'unused_providers' != provider.stem:
                    provider_folder = digital_coast_folder / provider.name
                    provider_folder_long = _to_win_long_path(provider_folder)
                    data_types = os.listdir(provider_folder_long)
                    if not provider_folder.suffix and 'dem' not in data_types and 'laz' not in data_types:
                        self.write_message(f' - removing empty provider: {provider_folder}', outputs)
                        shutil.rmtree(provider_folder_long)

    def download_support_files(self, digital_coast_folder: pathlib.Path, tile_gdf: gpd.GeoDataFrame, ecoregion: str, outputs: str) -> None:
        """Download tile_index shapefiles"""

        self.write_message('Download Support Files', outputs)
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)
        for geometry_coords in ecoregion_geom_strings:
            tile_index_links = self.get_available_datasets(geometry_coords, digital_coast_folder)
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

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str) -> None:
        """Main entry point for downloading Digital Coast data"""

        print('Downloading Digital Coast Datasets')
        outputs = self.param_lookup['output_directory'].valueAsText
        self.setup_dask(self.param_lookup['env'])
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                if output_prefix:
                    digital_coast_folder = pathlib.Path(outputs) / output_prefix / ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
                else:
                    digital_coast_folder = pathlib.Path(outputs) / ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'

                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]
                self.download_support_files(digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs)
                self.check_tile_index_areas(digital_coast_folder, outputs)
                self.process_intersected_datasets(digital_coast_folder, ecoregion_tile_gdf, outputs)
                if os.path.exists(_to_win_long_path(digital_coast_folder)):
                    self.delete_unused_folder(digital_coast_folder, outputs)
        self.close_dask()

    def process_intersected_datasets(self, digital_coast_folder: pathlib.Path, ecoregion_tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Download intersected Digital Coast files"""

        self.write_message('Downloading elevation datasets', outputs)
        tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(folder)]
        param_inputs = [[ecoregion_tile_gdf, shp_path, self.param_lookup, outputs] for shp_path in tile_index_shapefiles]
        future_tiles = self.client.map(_download_intersected_datasets, param_inputs)
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)

    def unzip_all_files(self, digital_coast_folder: str) -> None:
        """Unzip all zip files in a folder"""

        for zipped_file in pathlib.Path(digital_coast_folder).rglob('*.zip'):
            zip_path_str = _to_win_long_path(zipped_file)
            extract_dir_str = _to_win_long_path(zipped_file.parents[0])

            with zipfile.ZipFile(zip_path_str, 'r') as zipped:
                zipped.extractall(extract_dir_str)

            try:
                zipped_file.unlink()
            except OSError:
                os.remove(zip_path_str)