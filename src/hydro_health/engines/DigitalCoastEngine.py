"""Class for obtaining all available files"""

import ctypes
import gc
import logging
import os
import pathlib
import shutil
import tempfile
import uuid
import zipfile
from collections import defaultdict
from multiprocessing import set_executable

import boto3
import geopandas as gpd
import requests
from botocore import UNSIGNED
from botocore.client import Config
from osgeo import gdal
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

os.environ['SHAPE_RESTORE_SHX'] = 'YES'

# RedHat UTF encodings
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

logging.getLogger("pyproj").setLevel(logging.ERROR)


def _trim_memory() -> None:
    """Forces Linux glibc to release unmanaged C/GDAL memory arenas back to the OS."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _reproject_raster_gdal(
    src_path: str | pathlib.Path,
    dst_path: str | pathlib.Path,
    target_crs: str = "EPSG:32617",
) -> None:
    """
    Native C++ streaming reprojection and COG conversion using gdal.Warp.
    Streams block-by-block with a strict 128MB warp cache limit.
    """
    src_path_str = str(src_path)
    dst_path_str = str(dst_path)

    # Determine predictor: 3 for floating point DEMs, 2 for integers
    src_ds = gdal.Open(src_path_str, gdal.GA_ReadOnly)
    if not src_ds:
        raise RuntimeError(f"Could not open source raster: {src_path_str}")

    band = src_ds.GetRasterBand(1)
    band_type = gdal.GetDataTypeName(band.DataType)
    predictor = "3" if "Float" in band_type else "2"
    src_ds = None  # Close handle

    warp_options = gdal.WarpOptions(
        dstSRS=target_crs,
        resampleAlg="bilinear",
        format="GTiff",
        warpMemoryLimit=128 * 1024 * 1024,  # Exactly 128 MB in bytes
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=DEFLATE",
            f"PREDICTOR={predictor}",
            "BIGTIFF=IF_SAFER"
        ]
    )

    # Stream warp directly to destination file
    dst_ds = gdal.Warp(dst_path_str, src_path_str, options=warp_options)
    if dst_ds is None:
        raise RuntimeError(f"gdal.Warp failed for {src_path_str}")

    # Build internal overviews
    min_dim = min(dst_ds.RasterXSize, dst_ds.RasterYSize)
    potential_ov = [2, 4, 8, 16]
    valid_ov = [o for o in potential_ov if (min_dim // o) >= 4]
    if valid_ov:
        dst_ds.BuildOverviews("AVERAGE", valid_ov)

    # Close and flush destination dataset
    dst_ds = None


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


def _download_intersected_datasets_local(param_inputs: list) -> str:
    """
    Downloads raw GeoTIFFs to an 'original' subfolder and reprojects
    them into the parent folder as EPSG:32617.
    """

    tile_gdf, shp_path, param_lookup, outputs = param_inputs
    shp_path = pathlib.Path(shp_path)
    engine = DigitalCoastEngine(param_lookup)

    shp_df = gpd.read_file(shp_path).to_crs(4326)
    shp_df.columns = shp_df.columns.str.lower()
    
    if 'tile' in shp_df.columns:
        shp_df.drop('tile', axis=1, inplace=True)
        
    df_joined = shp_df.sjoin(df=tile_gdf, how='left')
    df_joined.to_file(shp_path, driver='ESRI Shapefile', encoding='utf-8')

    if df_joined['url'].any():
        df_joined = df_joined.loc[df_joined['tile'].notnull()]
        shp_folder = shp_path.parent
        
        original_folder = shp_folder / "original"
        original_folder.mkdir(parents=True, exist_ok=True)
        
        urls = df_joined['url'].unique()
        
        for i, url in enumerate(urls):
            cleansed_url = engine.cleansed_url(url)
            if not cleansed_url.endswith('.tif'):
                continue
                
            dataset_name = cleansed_url.split('/')[-1]
            raw_output_file = original_folder / dataset_name
            reprojected_output_file = shp_folder / dataset_name

            raw_target = str(raw_output_file)
            rep_target = str(reprojected_output_file)

            if os.path.exists(rep_target):
                engine.write_message(f' - ({i} of {len(urls)}) Skipping (already reprojected): {reprojected_output_file.stem}', outputs)
                continue

            # 1. Download to original/
            if not os.path.exists(raw_target):
                engine.write_message(f' - ({i} of {len(urls)}) Downloading original: {raw_output_file.stem}', outputs)
                try:
                    retry_strategy = Retry(
                        total=3,
                        backoff_factor=1,
                        status_forcelist=[404, 500, 502, 503, 504],
                        allowed_methods=["GET"]
                    )
                    adapter = HTTPAdapter(max_retries=retry_strategy)
                    request_session = requests.Session()
                    request_session.mount("https://", adapter)
                    request_session.mount("http://", adapter)

                    intersected_response = request_session.get(cleansed_url, timeout=10)
                except requests.exceptions.RequestException:
                    engine.write_message(f'Connection/Timeout error: {cleansed_url}', outputs)
                    continue

                if intersected_response.status_code == 200:
                    with open(raw_target, 'wb') as file:
                        file.write(intersected_response.content)
                else:
                    return f'Failed to download: {cleansed_url}'

            # 2. Native streaming reprojection
            try:
                engine.write_message(f' - ({i} of {len(urls)}) Reprojecting to EPSG:32617: {reprojected_output_file.stem}', outputs)
                _reproject_raster_gdal(raw_target, rep_target, target_crs="EPSG:32617")
            except Exception as e:
                engine.write_message(f'Reprojection error on {dataset_name}: {e}', outputs)
                continue
            finally:
                _trim_memory()
                
        return f'- {shp_path.stem}'
    else:
        return f'- No intersect: {shp_path.stem}'


def _download_intersected_datasets_s3(param_inputs: list) -> str:
    """
    Downloads DigitalCoast files, preserves raw original in S3 under /original/,
    reprojects to EPSG:32617 using GDAL C-streaming, and uploads to S3.
    """

    ecoregion, tile_gdf, shp_path, outputs, param_lookup, temp_root = param_inputs
    shp_path_obj = pathlib.Path(shp_path)
    engine = DigitalCoastEngine(param_lookup)
    
    scratch_dir = pathlib.Path.home() / "temp_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Correct cache limit: 256 MB in bytes (not 256 bytes)
    gdal.SetCacheMax(256 * 1024 * 1024)
    gdal.SetConfigOption("GDAL_TMPDIR", str(scratch_dir))
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")

    shp_df = gpd.read_file(shp_path, engine="pyogrio")
    shp_df['geometry'] = shp_df.geometry.make_valid()
    shp_df = shp_df.to_crs(4326)
    shp_df.columns = shp_df.columns.str.lower()

    cols_to_clear = ['tile', 'idx_right', 'index_right', 'index_left']
    laundered_cols = [c for c in shp_df.columns if c.startswith('ecoregio_')]
    shp_df = shp_df.drop(columns=cols_to_clear + laundered_cols, errors='ignore')

    df_joined = shp_df.sjoin(df=tile_gdf, how='left')

    if 'index_right' in df_joined.columns:
        df_joined = df_joined.rename(columns={'index_right': 'idx_right'})

    df_joined = df_joined.loc[:, ~df_joined.columns.duplicated()].copy()
    df_joined.to_file(shp_path, driver='ESRI Shapefile')
    
    urls = []
    if not df_joined.empty and 'url' in df_joined.columns:
        urls = df_joined.loc[df_joined['tile'].notnull(), 'url'].unique().tolist()
    
    del shp_df
    del df_joined
    _trim_memory()

    if urls:
        retry_strategy = Retry(
            total=3, 
            backoff_factor=1, 
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        request_session = requests.Session()
        request_session.mount("https://", adapter)

        for url in urls:
            cleansed_url = engine.cleansed_url(url)
            if not cleansed_url.endswith('.tif'): 
                continue

            unique_id = uuid.uuid4().hex[:8]
            tmp_raw_path = scratch_dir / f"raw_{unique_id}.tif"
            tmp_rep_path = scratch_dir / f"rep_{unique_id}.tif"

            try:
                dataset_name = cleansed_url.split('/')[-1]
                local_parent_dir = shp_path_obj.parent
                relative_s3_path = local_parent_dir.relative_to(temp_root)

                # 1. Download directly to scratch file
                with request_session.get(cleansed_url, timeout=20, stream=True) as r:
                    r.raise_for_status()
                    with open(tmp_raw_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024 * 1024): 
                            f.write(chunk)

                # 2. Upload raw file to S3 under /original/
                s3_original_prefix = f"{relative_s3_path.as_posix()}/original/{dataset_name}"
                engine.upload_file_to_s3(str(tmp_raw_path), s3_original_prefix)

                # 3. Stream warp to EPSG:32617 and build overviews
                _reproject_raster_gdal(tmp_raw_path, tmp_rep_path, target_crs="EPSG:32617")

                # 4. Upload reprojected raster to S3
                s3_prefix = f"{relative_s3_path.as_posix()}/{dataset_name}"
                engine.upload_file_to_s3(str(tmp_rep_path), s3_prefix)

            except Exception as e:
                engine.write_message(f"Error processing {cleansed_url}: {e}", outputs)
                continue
            finally:
                # Always remove temp scratch files and trim memory
                if tmp_raw_path.exists():
                    tmp_raw_path.unlink()
                if tmp_rep_path.exists():
                    tmp_rep_path.unlink()
                _trim_memory()

        request_session.close()

    return f'- {shp_path_obj.stem}'


def _download_intersected_datasets(param_inputs: list) -> str:
    """Parallel process spatial filter and download of datasets (local disk or S3 upload)"""
    
    if len(param_inputs) == 6:
        return _download_intersected_datasets_s3(param_inputs)
    return _download_intersected_datasets_local(param_inputs)


class DigitalCoastEngine(Engine):
    """Class for parallel processing all BlueTopo tiles for a region"""

    def __init__(self, param_lookup: dict[dict]):
        super().__init__()
        self.param_lookup = param_lookup

    def breakup_cudem(self, digital_coast_folder: pathlib.Path) -> None:
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
                year_folder = digital_coast_folder / f'NOAA_NCEI_{year}_{str(folder.stem)[-4:]}' / 'dem' / project_folder.stem
                year_folder.mkdir(parents=True, exist_ok=True)
                for tif_file in year_tifs[year]:
                    output_tif_file = year_folder / tif_file.name
                    if output_tif_file.exists():
                        output_tif_file.unlink()
                    shutil.move(tif_file, year_folder)
                tile_index_files = project_folder.glob('*index*.shp')
                for tile_index in tile_index_files:
                    tile_index_df = gpd.read_file(tile_index)
                    tile_index_df.to_file(year_folder / tile_index.name)
                feature_json = project_folder.parents[1] / 'feature.json'
                shutil.copy(feature_json, year_folder.parents[1])
            shutil.rmtree(folder)

    def check_tile_index_areas(
        self, 
        digital_coast_folder: pathlib.Path | list[pathlib.Path], 
        outputs: str
    ) -> list[pathlib.Path] | None:
        """Exclude any small area surveys"""

        self.write_message('Checking area size of tileindex files', outputs)
        if isinstance(digital_coast_folder, (list, tuple)):
            tile_index_shapefiles = digital_coast_folder
            is_list_mode = True
        else:
            tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(folder)]
            is_list_mode = False

        providers = []
        for shp_path in tile_index_shapefiles:
            shp_df = gpd.read_file(shp_path).to_crs(5072)  # latest (NAD83 (NSRS2007) / Conus Albers)
            shp_df['area'] = shp_df['geometry'].area
            total_area = shp_df["area"].sum()
            if total_area < self.approved_size:
                self.write_message(f' - provider too small: {total_area}\n{shp_path}', outputs)
                if not is_list_mode:
                    shutil.rmtree(shp_path.parents[2])
            else:
                providers.append(shp_path)

        if is_list_mode:
            return providers
        return None

    def delete_unused_folder(self, digital_coast_folder: pathlib.Path, outputs: str) -> None:
        """Delete any provider folders without a subfolder"""

        if digital_coast_folder.exists():
            self.write_message('Deleting empty provider folders', outputs)
            provider_folders = [folder for folder in digital_coast_folder.glob('*') if folder.is_dir()]
            for provider in provider_folders:
                if 'unused_providers' != provider.stem:
                    provider_folder = digital_coast_folder / provider
                    data_types = os.listdir(provider_folder)
                    if not provider_folder.suffix and 'dem' not in data_types and 'laz' not in data_types:
                        self.write_message(f' - removing empty provider: {provider_folder}', outputs)
                        shutil.rmtree(provider_folder)

    def download_support_files(
        self, 
        digital_coast_folder: pathlib.Path | gpd.GeoDataFrame, 
        tile_gdf: gpd.GeoDataFrame | str, 
        ecoregion: str | pathlib.Path, 
        outputs: pathlib.Path | str
    ) -> None:
        """Download tile_index shapefiles"""

        if isinstance(digital_coast_folder, gpd.GeoDataFrame):
            tile_gdf, ecoregion, digital_coast_folder = digital_coast_folder, tile_gdf, ecoregion

        outputs_str = str(outputs)
        self.write_message('Download Support Files', outputs_str)
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)
        for geometry_coords in ecoregion_geom_strings:
            tile_index_links = self.get_available_datasets(geometry_coords, digital_coast_folder)
            bulk_download_params = [
                [link_dict['link'], link_dict['provider_path'], self.param_lookup, outputs_str] 
                for link_dict in tile_index_links if link_dict['label'] == 'Bulk Download'
            ]
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
                self.write_message(f'Result: {result}', output_folder)

    def process_intersected_datasets(self, digital_coast_folder: pathlib.Path, ecoregion_tile_gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Download intersected Digital Coast files"""

        self.write_message('Downloading elevation datasets', outputs)
        tile_index_shapefiles = [folder for folder in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(folder)]
        param_inputs = [[ecoregion_tile_gdf, shp_path, self.param_lookup, outputs] for shp_path in tile_index_shapefiles]
        future_tiles = self.client.map(_download_intersected_datasets, param_inputs)
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)

    def unzip_all_files(self, digital_coast_folder: pathlib.Path | str) -> None:
        """Unzip all zip files in a folder"""

        for zipped_file in pathlib.Path(digital_coast_folder).rglob('*.zip'):
            with zipfile.ZipFile(zipped_file, 'r') as zipped:
                zipped.extractall(str(zipped_file.parents[0]))
            zipped_file.unlink()

    def upload_file_to_s3(self, temp_file_path: str, s3_prefix: str) -> None:
        """Simple S3 file upload with only region setting"""

        s3_client = boto3.client('s3', region_name='us-east-2')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
        s3_client.upload_file(str(temp_file_path), bucket_name, s3_prefix.lstrip('/'))

    def upload_files_to_s3(self, provider_folder: pathlib.Path, temp_root: pathlib.Path) -> None:
        """Upload all files to s3 for current tile maintaining structural integrity"""

        s3_client = boto3.client('s3', region_name='us-east-2')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        for found_file in provider_folder.rglob('*'):
            if found_file.is_file():
                relative_path = found_file.relative_to(temp_root)
                s3_prefix = relative_path.as_posix()
                print(f'Uploading {found_file} to {s3_prefix}')
                s3_client.upload_file(str(found_file), bucket_name, s3_prefix)

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str | bool = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        print('Starting DigitalCoastEngine')
        if self.param_lookup['env'] == 'aws':
            self._run_s3(tile_gdf, str(output_prefix) if output_prefix else "")
        else:
            self._run_local(tile_gdf, output_prefix)

    def _run_local(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str | bool) -> None:
        """Local pipeline execution"""

        outputs = pathlib.Path(self.param_lookup['output_directory'].valueAsText)
        outputs.mkdir(parents=True, exist_ok=True)
        outputs_str = str(outputs)

        self.setup_dask(self.param_lookup['env'])
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                digital_coast_subfolder = pathlib.Path(ecoregion) / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
                if output_prefix:
                    digital_coast_subfolder = pathlib.Path(output_prefix) / digital_coast_subfolder
                digital_coast_folder = outputs / digital_coast_subfolder
                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]
                self.download_support_files(digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs_str)
                self.check_tile_index_areas(digital_coast_folder, outputs_str)
                self.process_intersected_datasets(digital_coast_folder, ecoregion_tile_gdf, outputs_str)
                if digital_coast_folder.exists():
                    self.delete_unused_folder(digital_coast_folder, outputs_str)
            found_providers = [folder for folder in digital_coast_folder.glob('*') if folder.is_dir()]
            self.write_run_manifest(digital_coast_subfolder, {'providers': len(found_providers)})
        self.close_dask()

    def _run_s3(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str) -> None:
        """S3 cloud upload pipeline execution"""

        outputs = pathlib.Path(self.param_lookup['output_directory'].valueAsText)
        outputs.mkdir(parents=True, exist_ok=True)
        provider_log = outputs / "processed_providers.log"
        provider_log.touch(exist_ok=True)
        outputs_str = str(outputs)

        self.setup_dask(self.param_lookup['env'], threads_per_worker=1)
        ecoregions = list(tile_gdf['EcoRegion'].unique())
        
        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf['EcoRegion'] == ecoregion]

                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_digital_coast_outputs = pathlib.Path(temp_dir)

                    digital_coast_subfolder = pathlib.Path(ecoregion) / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
                    if output_prefix:
                        digital_coast_subfolder = pathlib.Path(output_prefix) / digital_coast_subfolder
                    
                    digital_coast_folder = temp_digital_coast_outputs / digital_coast_subfolder
                    
                    self.download_support_files(digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs_str)
                    tile_index_shapefiles = [shp for shp in digital_coast_folder.rglob('*index*.shp') if 'unused_providers' not in str(shp)]
                    approved_index_shapefiles = self.check_tile_index_areas(tile_index_shapefiles, outputs_str) or []

                    chunk_size = 5
                    for i in range(0, len(approved_index_shapefiles), chunk_size):
                        current_files = approved_index_shapefiles[i:i + chunk_size]
                        
                        completed_providers = set(provider_log.read_text().splitlines()) if provider_log.exists() else set()
                        files_to_process = [
                            shp for shp in current_files 
                            if shp.parents[0].name not in completed_providers
                        ]

                        if not files_to_process:
                            continue

                        for shp in files_to_process:
                            print(f'Processing provider: {shp.parents[0].name}')

                        param_inputs = [
                            [ecoregion, ecoregion_tile_gdf, shp_path, outputs_str, self.param_lookup, temp_digital_coast_outputs] 
                            for shp_path in files_to_process
                        ]
                        future_intersected_datasets = self.client.map(_download_intersected_datasets, param_inputs)
                        self.client.gather(future_intersected_datasets)

                        # Record completion
                        with open(provider_log, "a") as f:
                            for shp in files_to_process:
                                f.write(f"{shp.parents[0].name}\n")

                        # Upload tile_index shapefiles
                        uploaded_providers = set()
                        for provider_shp in files_to_process:
                            provider_path_parts = provider_shp.parts
                            digital_coast_index = provider_path_parts.index('DigitalCoast')
                            provider_folder = pathlib.Path(*provider_path_parts[:digital_coast_index + 2])
                            if provider_folder not in uploaded_providers:
                                self.upload_files_to_s3(provider_folder, temp_digital_coast_outputs)
                                uploaded_providers.add(provider_folder)

                    found_providers = [folder for folder in digital_coast_folder.glob('*') if folder.is_dir()]
                    self.write_run_manifest(digital_coast_subfolder, {'providers': len(found_providers)})
        self.close_dask()