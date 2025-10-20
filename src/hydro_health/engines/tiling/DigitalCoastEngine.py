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

from multiprocessing import set_executable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from collections import defaultdict
from botocore.client import Config
from botocore import UNSIGNED

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
os.environ['SHAPE_RESTORE_SHX'] = 'YES'  # Reruns are throwing fiona DriverError for .SHX, line 60


def _download_tile_index(param_inputs: list[list]) -> None:
    """Parallel process and download tile index shapefiles"""

    download_link, provider_folder, outputs = param_inputs

    engine = DigitalCoastEngine()

    if get_config_item('DIGITALCOAST', 'BUCKET') in download_link and ('dem' in download_link or 'laz' in download_link):
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

    tile_gdf, shp_path, outputs = param_inputs

    engine = DigitalCoastEngine()

    shp_df = gpd.read_file(shp_path).to_crs(4326)
    shp_df.columns = shp_df.columns.str.lower()  # make url column all lowercase
    if 'tile' in shp_df.columns:
        # drop previous tile merged column
        shp_df.drop('tile', axis=1, inplace=True)
    df_joined = shp_df.sjoin(df=tile_gdf, how='left')
    df_joined.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')
    if df_joined['url'].any():
        df_joined = df_joined.loc[df_joined['tile'].notnull()]
        shp_folder = shp_path.parents[0]
        urls = df_joined['url'].unique()
        for i, url in enumerate(urls):
            cleansed_url = engine.cleansed_url(url)
            # Only download .tif files
            if not cleansed_url.endswith('.tif'):
                continue
            dataset_name = cleansed_url.split('/')[-1]
            output_file = shp_folder / dataset_name
            if os.path.exists(output_file):
                engine.write_message(f' - ({i} of {len(urls)}) Skipping data: {output_file.stem}', outputs)
                continue
            else:
                engine.write_message(f' - ({i} of {len(urls)}) Downloading data: {output_file.stem}', outputs)
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
                with open(output_file, 'wb') as file:
                    file.write(intersected_response.content)
            else:
                return f'Failed to download: {cleansed_url}'
        return f'- {shp_path.stem}'
    else:
        return f'- No intersect: {shp_path.stem}'
        

class DigitalCoastEngine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self):
        super().__init__()

    def breakup_cudem(self, digital_coast_folder) -> None:
        """Create unique year folders for CUDEM data"""

        cudem_folders = digital_coast_folder.glob('NOAA_NCEI_0*')
        for folder in cudem_folders:
            year_tifs = defaultdict(list)
            tif_files = folder.rglob('*.tif')
            for tif_file in tif_files:
                year = str(tif_file.stem)[-6:-2]
                year_tifs[year].append(tif_file)
            for year in year_tifs:
                print(f'Creating {year} folder for CUDEM')
                project_folder = year_tifs[year][0].parents[0]
                # Create new year folder
                year_folder = digital_coast_folder / f'NOAA_NCEI_{year}_{str(folder.stem)[-4:]}' / 'dem' / project_folder.stem
                year_folder.mkdir(parents=True, exist_ok=True)
                # Copy year tifs to new folder
                for tif_file in year_tifs[year]:
                    output_tif_file = year_folder / tif_file.name
                    if output_tif_file.exists():
                        output_tif_file.unlink()
                    shutil.move(tif_file, year_folder)
                # Copy tileindex
                tile_index_files = project_folder.glob('*index*.shp')
                for tile_index in tile_index_files:
                    # use gpd to copy all shp files at once
                    tile_index_df = gpd.read_file(tile_index)
                    tile_index_df.to_file(year_folder / tile_index.name)
                # Copy feature.json
                feature_json = project_folder.parents[1] / 'feature.json'
                shutil.copy(feature_json, year_folder.parents[1])
            # Delete old CUDEM folder
            shutil.rmtree(folder)

    def check_tile_index_areas(self, digital_coast_folder, outputs) -> None:
        """Exclude any small area surveys"""

        self.write_message(f'Checking area size of tileindex files', outputs)
        tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(folder)]
        move_providers = []
        for shp_path in tile_index_shapefiles:
            shp_df = gpd.read_file(shp_path).to_crs(9822)  # Albers Equal Area
            shp_df['area'] = shp_df['geometry'].area
            total_area = shp_df["area"].sum()
            if total_area < self.approved_size:
                self.write_message(f' - provider too small: {total_area} - {shp_path}', outputs)
                move_providers.append(shp_path.parents[2])
        for provider in move_providers:
            unused_provider_folder = digital_coast_folder / 'unused_providers'
            if not unused_provider_folder.exists():
                unused_provider_folder.mkdir()
            if pathlib.Path(unused_provider_folder / provider.stem).exists():
                shutil.rmtree(unused_provider_folder / provider.stem)
            shutil.move(provider, unused_provider_folder)

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

    def download_support_files(self, digital_coast_folder: pathlib.Path, tile_gdf: gpd.GeoDataFrame, ecoregion: str, outputs: str) -> None:
        """Download tile_index shapefiles"""

        self.write_message('Download Support Files', outputs)
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)

        for geometry_coords in ecoregion_geom_strings:
            tile_index_links = self.get_available_datasets(geometry_coords, ecoregion, outputs)  # TODO return all object keys
            bulk_download_params = [[link_dict['link'], link_dict['provider_path'], outputs] for link_dict in tile_index_links if link_dict['label'] == 'Bulk Download']
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

    def run(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        print('Downloading Digital Coast Datasets')
        self.setup_dask()
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                digital_coast_folder = pathlib.Path(outputs) / ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
                # tile_gdf.to_file(rF'{OUTPUTS}\tile_gdf.shp', driver='ESRI Shapefile')
                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]
                self.download_support_files(digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs)
                self.check_tile_index_areas(digital_coast_folder, outputs)
                self.process_intersected_datasets(digital_coast_folder, ecoregion_tile_gdf, outputs)
                if digital_coast_folder.exists():
                    self.delete_unused_folder(digital_coast_folder, outputs)
                self.breakup_cudem(digital_coast_folder)
        self.close_dask()

    def process_intersected_datasets(self, digital_coast_folder: pathlib.Path, ecoregion_tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Download intersected Digital Coast files"""

        self.write_message('Downloading elevation datasets', outputs)
        tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(folder)]
        param_inputs = [[ecoregion_tile_gdf, shp_path, outputs] for shp_path in tile_index_shapefiles]
        future_tiles = self.client.map(_download_intersected_datasets, param_inputs)
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)

    def unzip_all_files(self, output_folder: str) -> None:
        """Unzip all zip files in a folder"""

        for zipped_file in pathlib.Path(output_folder).rglob('*.zip'):
            with zipfile.ZipFile(zipped_file, 'r') as zipped:
                zipped.extractall(str(zipped_file.parents[0]))
            # Delete zip file after extract
            zipped_file.unlink()

