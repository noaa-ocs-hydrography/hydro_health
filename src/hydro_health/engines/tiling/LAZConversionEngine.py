import os
import pathlib
import pickle
import requests
import subprocess
import sys
import shutil
import geopandas as gpd

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import set_executable
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
HELPERS = pathlib.Path(__file__).parents[2] / 'helpers'


class LAZConversionEngine(Engine):
    """Class for parallel processing all LAZ to GeoTiff conversion for a region"""

    def __init__(self):
        super().__init__()
        self.laz_folders = None
        self.delete_files = []

    def check_tile_index_areas(self) -> None:
        """Exclude any small area surveys"""

        print(f'- Checking area size of tileindex files')
        remove_laz = []
        for i, laz_folder in enumerate(self.laz_folders):
            shp_path = list(laz_folder.rglob('*.shp'))
            if shp_path:
                shp_path = shp_path[0]
                shp_df = gpd.read_file(shp_path).to_crs(9822)  # Albers Equal Area
                shp_df['area'] = shp_df['geometry'].area
                total_area = shp_df["area"].sum()
                if total_area < self.approved_size:
                    print(f' - provider too small: {total_area} - {shp_path}')
                    # dynamically obtain digital coast and provider folder
                    # sometimes less subfolders
                    digital_coast_folder = [path for path in shp_path.parents if path.stem == 'DigitalCoast'][0]
                    provider_index = [i-1 for i, path in enumerate(shp_path.parents) if path.stem == 'DigitalCoast'][0]
                    provider = shp_path.parents[provider_index]
                    unused_provider_folder = digital_coast_folder / 'unused_providers'
                    if not unused_provider_folder.exists():
                        unused_provider_folder.mkdir()
                    if pathlib.Path(unused_provider_folder / provider.stem).exists():
                        shutil.rmtree(unused_provider_folder / provider.stem)
                    shutil.move(provider, unused_provider_folder)
                    remove_laz.append(i)
            else:
                print(f' - missing tileindex: {laz_folder}')
        for index in remove_laz:
            print(f' - deleting {self.laz_folders[index]}')
            del self.laz_folders[index]

    def convert_laz_file(self, input_file) -> None:
        """Convert an individual LAZ file with threading"""

        output_file = input_file.parents[0] / f'{input_file.stem}.tif'
        pipeline_json = {
            "pipeline": [
                {"type": "readers.las", "filename": str(input_file)},
                {
                    "type": "writers.gdal",
                    "gdaldriver": "GTiff",
                    "resolution": 8.0,
                    "dimension": "Z",
                    "output_type": "max",
                    "filename": str(output_file),
                },
            ]
        }

        output_pickle_file = input_file.parents[0] / f'{input_file.stem}.pkl'
        with open(output_pickle_file, 'wb') as picklish:
            pickle.dump(pipeline_json, picklish)
        try:
            subprocess.call(
                [
                    "conda",
                    "run",
                    "-p",
                    pathlib.Path.home() / r"AppData\Local\ESRI\conda\envs\pdal-workshop",
                    "python",
                    HELPERS / 'pdal_separate_script.py',
                    str(output_pickle_file),
                ]
            )
            output_pickle_file.unlink()
            return f'- {output_file.name}'
        except Exception as e:
            print(f'Failure: {e}')
            return False
    
    def download_intersected_datasets(self, param_inputs) -> None:
        """Parallel process spatial filter and download of datasets"""

        tile_gdf, shp_path, outputs = param_inputs
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

            param_inputs = [(url, shp_folder, outputs) for url in urls]
            with ThreadPoolExecutor(int(os.cpu_count() - 2)) as bulk_pool:
                print('- Starting LAZ conversion')
                bulk_pool.map(self.download_single_laz, param_inputs)

    def download_single_laz(self, param_inputs) -> None:
        """Threading method for downloading and converting and LAZ file"""

        url, shp_folder, outputs = param_inputs

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

        cleansed_url = self.cleansed_url(url)
        dataset_name = cleansed_url.split('/')[-1]
        output_laz = shp_folder / dataset_name
        converted_laz = output_laz.parents[0] / f'{output_laz.stem}.tif'

        if converted_laz.exists():
            print(f' - found converted: {converted_laz.name}')
            if output_laz.exists():
                output_laz.unlink()
        else:
            if output_laz.exists():
                converted = self.convert_laz_file(output_laz)
                if converted:
                    print(f' - converted: {output_laz}')
                    output_laz.unlink()
            else:
                try:
                    intersected_response = request_session.get(cleansed_url)
                    if intersected_response.status_code == 200:
                        with open(output_laz, 'wb') as file:
                            file.write(intersected_response.content)
                        converted = self.convert_laz_file(output_laz)
                        if converted:
                            print(f' - converted: {output_laz}')
                            output_laz.unlink()
                    else:
                        self.write_message(f'LAZ Download failed, {intersected_response.status_code}: {cleansed_url}', outputs)
                except requests.exceptions.ConnectionError:
                    self.write_message(f'Timeout error: {cleansed_url}', outputs)

    def get_laz_providers(self, tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Obtain all provider URL info by ecoregion"""

        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            print(f'Processing LAZ files for {ecoregion}')
            digital_coast_folder = pathlib.Path(outputs) / ecoregion /  get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            self.laz_folders = [folder for folder in digital_coast_folder.glob('**/laz') if 'unused_providers' not in str(folder)]

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)

    def process_laz_files(self, tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Sequentially download and convert LAZ files"""

        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]
            for laz_folder in self.laz_folders:
                 self.download_intersected_datasets([ecoregion_tile_gdf, list(laz_folder.rglob('*.shp'))[0], outputs])

    def run(self, tiles: gpd.GeoDataFrame, outputs: str) -> None:
        """Main entry point for converting LAZ data"""

        self.get_laz_providers(tiles, outputs)
        self.check_tile_index_areas()
        self.process_laz_files(tiles, outputs)
