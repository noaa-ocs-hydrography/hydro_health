from __future__ import annotations

import re
import cProfile
import pstats
import pathlib
import fsspec
import geopandas as gpd
import fnmatch
import datetime
import xarray as xr
import numpy as np
import s3fs
import tempfile
import time

from collections import deque
from collections.abc import Iterator

from ftplib import FTP
from upath import UPath
from osgeo import gdal, osr

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class HydroHealthCredentialsError(Exception):
    pass

class HydroHealthConfig:
    def __init__(self) -> None:
        super().__init__()
        self.config_path = INPUTS / 'lookups' / 'glob_tsm.config'
        self.username = None
        self.password = None
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Read credentials from config file"""

        # you need the inputs/lookups/glob_tsm.config file which is not stored in the repo for security reasons
        if not self.config_path.exists():
            raise HydroHealthCredentialsError(f"Config file not found: {self.config_path}")

        creds = {}
        with open(self.config_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if '=' in line:
                    key, value = line.split("=", 1)
                    creds[key.strip()] = value.strip()

        try:
            self.username = creds["username"]
            self.password = creds["password"]
        except KeyError as e:
            raise HydroHealthCredentialsError(f"Missing value in config: {e}")

class CreateTSMLayerEngine(Engine):
    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']
        self.output_prefix = output_prefix

        creds = HydroHealthConfig()
        self.username = creds.username
        self.password = creds.password
        self.server = 'ftp.hermes.acri.fr'
        self.downloaded_files = []
        self.ftp_file_paths = {}
        self.ftp_directory_cache = {}
        self.output_folder = OUTPUTS
        
        # Setup paths
        tsm_data_path = get_config_item("TSM", "DATA_PATH")
        
        if get_environment() in ["local", "remote"]:
            base_raster = OUTPUTS / 'ER_3' / get_config_item("TSM", "SUBFOLDER")
            self.raster_path = base_raster / 'mean_rasters'
            self.year_pair_path = base_raster / 'TSM_year_pair_rasters'
            self.nc_files_path = self.output_folder / str(tsm_data_path).lstrip('/')
        else:
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            subfolder = get_config_item('TSM', 'SUBFOLDER')
            self.raster_path = f"{bucket}/ER_3/{subfolder}/mean_rasters"
            self.year_pair_path = f"{bucket}/ER_3/{subfolder}/TSM_year_pair_rasters"
            self.nc_files_path = f"{bucket}/{tsm_data_path}"

    def _save_raster_data(self, data: np.array, path: str|pathlib.Path, transform: rasterio.transform.Affine, crs: str, nodata_val: float) -> None:
        """Standardized Rasterio write method"""

        with rasterio.open(
            path, 
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform,
            nodata=nodata_val,
            compress='deflate',
            tiled=True,
            zlevel=9
        ) as dst:
            dst.write(data, 1)

    def extract_start_date(self, filename: str) -> str:
        """Parse file name for start date"""

        match = re.search(r'(\d{8})-\d{8}', filename)
        return match.group(1) if match else '99999999'

    def write_download_log(self) -> None:
        """Helper function to log downloaded NC files"""

        self.write_message("Writing download log...", OUTPUTS)
        sorted_files = sorted(self.downloaded_files, key=self.extract_start_date)
        log_path = self.download_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            for name in sorted_files:
                f.write(name + '\n')
        self.write_message(f"\nDownload log written to: {log_path}", OUTPUTS)

    def download_recursive(self, ftp: FTP, current_dir: str) -> None:
        """Recursive method to download all NetCDF files"""

        try:
            ftp.cwd(current_dir)
            normalized = current_dir.lower().strip('/')
            
            skip_patterns = (
                '*meris/*', '*viirsn*', '*seawifs*', '*modis*',
                '*viirsj1*', '*meris4rp*', '*olcib*',
                '*/month/*', '*/day/*', '*/track/*',
                '*/olcia/*/2019/*', '*/olcia/*/2020/*', '*/olcia/*/2021/*',
                '*/olcia/*/2022/*', '*/olcia/*/2023/*', '*/olcia/*/2024/*', '*/olcia/*/2025/*'
            )

            if any(fnmatch.fnmatch(normalized, pattern.strip('/')) for pattern in skip_patterns):
                return

            items = list(ftp.mlsd())
            for name, facts in items:
                item_path = f"{current_dir}/{name}"
                if facts.get("type") == "dir":
                    self.download_recursive(ftp, item_path)
                elif name.endswith('.nc') and 'TSM_8D' in name and '_4' in name and 'L3m' in name:
                    # Date filtering for OLCIA
                    if 'AV-OLA' in name:
                        match = re.search(r'L3m_(\d{8})-', name)
                        if match:
                            file_date = datetime.datetime.strptime(match.group(1), "%Y%m%d")
                            if file_date > datetime.datetime(2018, 5, 1):
                                continue

                    if get_environment() == 'aws':
                        s3_files = s3fs.S3FileSystem()
                        s3_prefix = f'{self.nc_files_path}/{name}'
                        
                        # Never redownload an existing NetCDF file.
                        if not s3_files.exists(s3_prefix):
                            print(f"Downloading {name} to S3...")
                            with s3_files.open(s3_prefix, 'wb') as writer:
                                ftp.retrbinary(f'RETR {name}', writer.write)
                            self.downloaded_files.append(name)
                    else:
                        local_path = self.nc_files_path / name
                        self.nc_files_path.mkdir(parents=True, exist_ok=True)
                        
                        # Never redownload an existing NetCDF file.
                        if not local_path.exists():
                            print(f"Downloading {name} locally...")
                            with open(local_path, 'wb') as f:
                                ftp.retrbinary(f'RETR {name}', f.write)
                            self.downloaded_files.append(name)
        except Exception as e:
            self.write_message(f"Error accessing {current_dir}: {e}", OUTPUTS)

    def _download_verified_nc(self, ftp: FTP, remote_path: str, destination: str) -> None:
        """Stage on disk and validate all TSM chunks before replacing the destination."""

        with tempfile.TemporaryDirectory() as temporary_dir:
            local_path = pathlib.Path(temporary_dir) / pathlib.PurePosixPath(destination).name
            ftp.voidcmd('TYPE I')
            expected_size = ftp.size(remote_path)
            size_label = f'{expected_size / 1024**2:.1f} MiB' if expected_size is not None else 'size unknown'
            received = 0
            last_report = time.monotonic()
            with local_path.open('wb') as writer:
                def write_block(block: bytes) -> None:
                    nonlocal received, last_report
                    writer.write(block)
                    received += len(block)
                    now = time.monotonic()
                    if now - last_report >= 2:
                        progress = f'{received / 1024**2:.1f} MiB'
                        if expected_size:
                            progress += f' / {expected_size / 1024**2:.1f} MiB ({100 * received / expected_size:.0f}%)'
                        last_report = now
                ftp.retrbinary(f'RETR {remote_path}', write_block, blocksize=256 * 1024)
            self.write_message(f'  Download complete: {received / 1024**2:.1f} MiB', OUTPUTS)
            if expected_size is not None and local_path.stat().st_size != expected_size:
                raise IOError(f'Incomplete FTP download: {remote_path}')
            self.write_message('  Validating downloaded NetCDF...', OUTPUTS)
            with xr.open_dataset(local_path, engine='h5netcdf', cache=False) as ds:
                variable = ds['TSM_mean'].squeeze()
                if variable.shape != (4320, 8640):
                    raise ValueError(f'Unexpected NetCDF shape: {variable.shape}')
                # Read small row blocks so validation does not load a global array.
                for row in range(0, variable.shape[0], 128):
                    variable.isel({variable.dims[0]: slice(row, row + 128)}).values
            self.write_message(f'  Validation passed. Uploading replacement to {destination}...', OUTPUTS)
            self.filesystem.put_file(str(local_path), str(destination))
            self.filesystem.invalidate_cache(str(destination))
            self.write_message('  Replacement complete.', OUTPUTS)

    def _find_ftp_file(self, ftp: FTP, name: str) -> str:
        """Search compatible sensor/year branches, caching paths and listings."""

        if name in self.ftp_file_paths:
            self.write_message(f'  Using cached FTP path: {self.ftp_file_paths[name]}', OUTPUTS)
            return self.ftp_file_paths[name]
        date_match = re.search(r'L3m_(\d{4})', name)
        year = date_match.group(1) if date_match else None
        sensor_match = re.search(r'AV-([A-Z0-9]+)_', name)
        sensor = sensor_match.group(1) if sensor_match else None
        aliases = {
            'MER': ('mer', 'meris', 'meris4rp'),
            'OLA': ('ola', 'olcia'), 'OLB': ('olb', 'olcib'),
        }
        sensor_names = aliases.get(sensor, ())
        known_sensors = ('meris', 'olcia', 'olcib', 'modis', 'viirs', 'seawifs')
        directories = deque(['/GLOB'])
        started = time.monotonic()
        self.write_message(f'  Finding FTP file (sensor={sensor}, year={year})...', OUTPUTS)
        while directories:
            if time.monotonic() - started > 180:
                raise TimeoutError(f'FTP file search exceeded 180 seconds: {name}')
            directory = directories.popleft()
            self.write_message(f'    Searching {directory}', OUTPUTS)
            if directory not in self.ftp_directory_cache:
                self.ftp_directory_cache[directory] = list(ftp.mlsd(directory))
            children = []
            for item_name, facts in self.ftp_directory_cache[directory]:
                remote_path = f'{directory}/{item_name}'
                if facts.get('type') != 'dir':
                    if item_name.endswith('.nc'):
                        self.ftp_file_paths[item_name] = remote_path
                    if item_name == name:
                        self.write_message(f'  Found FTP file: {remote_path}', OUTPUTS)
                        return remote_path
                    continue
                lower = item_name.lower()
                if year and re.fullmatch(r'(?:19|20)\d{2}', lower) and lower != year:
                    continue
                if lower in ('day', 'daily', 'month', 'monthly', 'track'):
                    continue
                if sensor_names and any(label in lower for label in known_sensors):
                    if not any(label in lower for label in sensor_names):
                        continue
                priority = 0 if lower == year or any(label in lower for label in sensor_names) else 1
                children.append((priority, remote_path))
            # Prefer matching branches immediately over generic sibling directories.
            for _, child in reversed(sorted(children)):
                directories.appendleft(child)
        raise FileNotFoundError(f'NetCDF not found in compatible FTP branches: {name}')

    def _redownload_corrupted_nc(self, destination: str) -> None:
        """Find and replace only the damaged file, with visible progress."""

        name = pathlib.PurePosixPath(destination).name
        self.write_message(f'  Repairing corrupted NetCDF: {name}', OUTPUTS)
        self.write_message(f'  Connecting to FTP server {self.server}...', OUTPUTS)
        with FTP(self.server, timeout=30) as ftp:
            ftp.login(user=self.username, passwd=self.password)
            remote_path = self._find_ftp_file(ftp, name)
            self._download_verified_nc(ftp, remote_path, destination)
            self.downloaded_files.append(name)

    def _read_tsm_array(self, path: str | pathlib.Path) -> np.ndarray:
        """Retry once after repairing an HDF5 corruption/read failure."""

        for attempt in range(2):
            try:
                with self.filesystem.open(str(path), 'rb') as source:
                    with xr.open_dataset(source, engine='h5netcdf', cache=False) as ds:
                        arr = ds['TSM_mean'].values.squeeze().astype(np.float32)
                if arr.shape != (4320, 8640):
                    raise ValueError(f'Unexpected NetCDF shape {arr.shape}; expected (4320, 8640)')
                arr[arr == 0] = np.nan
                return arr
            except (OSError, ValueError) as error:
                message = str(error).lower()
                corruption = any(token in message for token in (
                    'truncated file', 'file signature not found', 'bad object header',
                    'inflate() failed', 'filter returned failure', 'checksum',
                    'unable to synchronously read', 'address overflow',
                ))
                if attempt or not corruption:
                    raise
                self._redownload_corrupted_nc(str(path))

    def download_tsm_data(self) -> None:
        """FTP startup method to download all TSM data"""

        self.write_message('Connecting to FTP server...', OUTPUTS)
        ftp = FTP(self.server)
        ftp.login(user=self.username, passwd=self.password)
        try:
            self.download_recursive(ftp, '/GLOB')
        finally:
            ftp.quit()
        self.write_download_log()

    def create_mean_year_rasters(self) -> None:
        """Create mean year raster files from all NetCDF files"""

        grid_shape = (4320, 8640)
        transform = (-180.0, 360.0 / grid_shape[1], 0.0,
                     90.0, 0.0, -180.0 / grid_shape[0])
        crs = "EPSG:4326" # the nc files use this crs
        nodata_val = np.nan

        for year in range(1998, 2025):
            filename = f"TSM_mean_{year}.tif"
            if not self.overwrite:
                if get_environment() in ['local', 'remote']:
                    output_exists = (self.raster_path / filename).exists()
                else:
                    output_exists = s3_files.exists(f"s3://{self.raster_path}/{filename}")
                if output_exists:
                    print(f"Skipping existing annual raster: {filename}")
                    continue

            print(f'Processing year: {year}')
            if get_environment() in ['local', 'remote']:
                if not self.nc_files_path.exists():
                    print(f'  No NetCDF files found for {year}')
                    continue
                nc_files = [f for f in os.listdir(self.nc_files_path) if f"L3m_{year}" in f and f.endswith('.nc')]
            else:
                s3_tsm_files = s3_files.ls(self.nc_files_path)
                nc_files = [p for p in s3_tsm_files if f"L3m_{year}" in p and p.endswith('.nc')]

            if not nc_files:
                self.write_message(f'  No NetCDF files found for {year}', OUTPUTS)
                continue

            running_sum, valid_count = None, None
            annual_mean = None
            for fname in nc_files:
                try:
                    arr = self._read_tsm_array(fname)
                    if running_sum is None:
                        running_sum = np.zeros_like(arr)
                        valid_count = np.zeros_like(arr, dtype=np.uint16)
                    valid = np.isfinite(arr)
                    np.add(running_sum, arr, out=running_sum, where=valid)
                    np.add(valid_count, 1, out=valid_count, where=valid)
                    del arr, valid
                except MemoryError:
                    raise
                except Exception as e:
                    self.write_message(f"  Failed: {fname}: {e}", OUTPUTS)

            if running_sum is None:
                continue

            annual_mean = np.where(valid_count > 0, running_sum / valid_count, nodata_val)

            if get_environment() in ['local', 'remote']:
                self.raster_path.mkdir(parents=True, exist_ok=True)
                self._save_raster_data(annual_mean, self.raster_path / filename, transform, crs, nodata_val)
            else:
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                    tmp_path = tmp.name
                try:
                    self._save_raster_data(annual_mean, tmp_path, transform, crs, nodata_val)
                    s3_dest = f"s3://{self.raster_path}/{filename}"
                    s3_files.put(tmp_path, s3_dest)
                finally:
                    if os.path.exists(tmp_path): os.remove(tmp_path)

    def year_pair_rasters(self, start_year: int, end_year: int) -> None:
        """Build and store year-pair mean and cumulative rasters"""

        s3_files = s3fs.S3FileSystem()
        out_name_mean = f'tsm_mean_{start_year}_{end_year}.tif'
        if not self.overwrite:
            if get_environment() in ['local', 'remote']:
                output_exists = (self.year_pair_path / out_name_mean).exists()
            else:
                output_exists = s3_files.exists(f"s3://{self.year_pair_path}/{out_name_mean}")
            if output_exists:
                print(f"Skipping existing year-pair raster: {out_name_mean}")
                return

        if get_environment() in ['local', 'remote']:
            raster_files = [os.path.join(self.raster_path, f) for f in os.listdir(self.raster_path)
                            if f.endswith(".tif") and any(str(y) in f for y in range(start_year, end_year + 1))]
        else:
            s3_list = s3_files.ls(self.raster_path)
            raster_files = [f"s3://{f}" for f in s3_list if f.endswith(".tif") 
                            and any(str(y) in f for y in range(start_year, end_year + 1))]

        if not raster_files: return

        with rasterio.open(raster_files[0]) as src:
            meta = src.meta.copy()
            sum_array = np.zeros(src.shape, dtype=np.float32)
            count_array = np.zeros(src.shape, dtype=np.int32)

        for f in raster_files:
            with rasterio.open(f) as src:
                data = src.read(1)
                mask = ~np.isnan(data)
                sum_array[mask] += data[mask]
                count_array[mask] += 1

        # Calculate average and cumulative arrays
        avg_array = np.where(count_array > 0, sum_array / count_array, np.nan)
        # cumulative_array = np.where(count_array > 0, sum_array, np.nan)

        # Output filenames place year pairs at the end
        # out_name_cumulative = f'tsm_cumulative_{start_year}_{end_year}.tif'

        if get_environment() in ['local', 'remote']:
            self.year_pair_path.mkdir(parents=True, exist_ok=True)
            # Save Mean
            self._save_raster_data(avg_array, os.path.join(self.year_pair_path, out_name_mean), meta['transform'], meta['crs'], meta['nodata'])
            
            # Save Cumulative
            # self._save_raster_data(cumulative_array, os.path.join(self.year_pair_path, out_name_cumulative), meta['transform'], meta['crs'], meta['nodata'])
        else:
            # Save Mean to S3
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_mean:
                self._save_raster_data(avg_array, tmp_mean.name, meta['transform'], meta['crs'], meta['nodata'])
                s3_dest_mean = f"s3://{self.year_pair_path}/{out_name_mean}"
                s3_files.put(tmp_mean.name, s3_dest_mean)
                print(f"Mean raster saved: {s3_dest_mean}")    
            os.remove(tmp_mean.name)

            # Save Cumulative to S3
            # with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_cumul:
            #     self._save_raster_data(cumulative_array, tmp_cumul.name, meta['transform'], meta['crs'], meta['nodata'])
            #     s3_dest_cumul = f"s3://{self.year_pair_path}/{out_name_cumulative}"
            #     s3_files.put(tmp_cumul.name, s3_dest_cumul)
            #     print(f"Cumulative raster saved: {s3_dest_cumul}")    
            # os.remove(tmp_cumul.name)

    def run(self) -> None:
        """Main entry point for the engine; orchestrates downloading, processing, and raster creation."""
    
        for eco_region in self.param_lookup['eco_regions'].value:
            self._resolve_paths(eco_region)
            self.download_tsm_data()
            self._prepare_region_mask()
            self.create_mean_year_rasters()
            for start_year, end_year in self.year_ranges:
                self.year_pair_rasters(start_year, end_year)



if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    engine = CreateTSMLayerEngine()
    engine.run()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  