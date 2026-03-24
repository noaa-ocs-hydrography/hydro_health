import os
import re
import pathlib
import fnmatch
import datetime
import xarray as xr
import numpy as np
import rasterio
import s3fs
import tempfile

from ftplib import FTP
from rasterio.transform import from_bounds

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_environment


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class HydroHealthCredentialsError(Exception):
    pass


class HydroHealthConfig:
    def __init__(self):
        super().__init__()
        self.config_path = INPUTS / 'lookups' / 'hydro_health.config'
        self.username = None
        self.password = None
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Read credentials from config file"""

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
    def __init__(self):
        super().__init__()

        creds = HydroHealthConfig()
        self.username = creds.username
        self.password = creds.password
        self.server = 'ftp.hermes.acri.fr'
        self.downloaded_files = []
        self.output_folder = OUTPUTS
        
        # Setup paths
        if get_environment() in ["local", "remote"]:
            base_raster = OUTPUTS / 'ER_3' / get_config_item("TSM", "SUBFOLDER")
            self.raster_path = base_raster / 'mean_rasters'
            self.year_pair_path = base_raster / 'TSM_year_pair_rasters'
        else:
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            subfolder = get_config_item('TSM', 'SUBFOLDER')
            self.raster_path = f"{bucket}/ER_3/{subfolder}/mean_rasters"
            self.year_pair_path = f"{bucket}/ER_3/{subfolder}/TSM_year_pair_rasters"

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

        print("Writing download log...")
        sorted_files = sorted(self.downloaded_files, key=self.extract_start_date)
        log_path = self.output_folder / 'downloaded_files.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            for name in sorted_files:
                f.write(name + '\n')
        print(f"\nDownload log written to: {log_path}")

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
                        s3_prefix = f'{get_config_item("SHARED", "OUTPUT_BUCKET")}/ER_3/{get_config_item("TSM", "SUBFOLDER")}/nc_files/{name}'
                        if not s3_files.exists(s3_prefix):
                            print(f"Downloading {name} to S3...")
                            with s3_files.open(s3_prefix, 'wb') as writer:
                                ftp.retrbinary(f'RETR {name}', writer.write)
                            self.downloaded_files.append(name)
                    else:
                        local_path = self.output_folder / f'ER_3/{get_config_item("TSM", "SUBFOLDER")}/nc_files/{name}'
                        self.output_folder.mkdir(parents=True, exist_ok=True)
                        if not local_path.exists():
                            print(f"Downloading {name} locally...")
                            with open(local_path, 'wb') as f:
                                ftp.retrbinary(f'RETR {name}', f.write)
                            self.downloaded_files.append(name)
        except Exception as e:
            print(f"Error accessing {current_dir}: {e}")

    def download_tsm_data(self) -> None:
        """FTP startup method to download all TSM data"""

        print('Connecting to FTP server...')
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
        transform = from_bounds(-180, -90, 180, 90, grid_shape[1], grid_shape[0])
        crs = "EPSG:4326"
        nodata_val = np.nan
        s3_files = s3fs.S3FileSystem()

        for year in range(1998, 2025):
            print(f'Processing year: {year}')
            if get_environment() in ['local', 'remote']:
                nc_files_folder = f'{self.output_folder}/ER_3/{get_config_item("TSM", "SUBFOLDER")}/nc_files'
                nc_files = [f for f in os.listdir(nc_files_folder) if f"L3m_{year}" in f and f.endswith('.nc')]
            else:
                s3_tsm_files = s3_files.ls(f'{get_config_item("SHARED", "OUTPUT_BUCKET")}/ER_3/{get_config_item("TSM", "SUBFOLDER")}/nc_files')
                nc_files = [p for p in s3_tsm_files if f"L3m_{year}" in p and p.endswith('.nc')]

            if not nc_files:
                print(f'  No NetCDF files found for {year}')
                continue

            running_sum, valid_count = None, None
            for fname in nc_files:
                path = self.output_folder / fname if get_environment() in ['local', 'remote'] else f's3://{fname}'
                try:
                    with xr.open_dataset(path, engine='h5netcdf') as ds:
                        arr = ds['TSM_mean'].values.squeeze().astype(np.float32)
                        arr[arr == 0] = np.nan
                        if running_sum is None:
                            running_sum = np.zeros_like(arr)
                            valid_count = np.zeros_like(arr, dtype=np.uint16)
                        valid = ~np.isnan(arr)
                        running_sum[valid] += arr[valid]
                        valid_count[valid] += 1
                except Exception as e:
                    print(f"  Failed: {fname}: {e}")

            if running_sum is None:
                continue

            annual_mean = np.where(valid_count > 0, running_sum / valid_count, nodata_val)
            filename = f"TSM_mean_{year}.tif"

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
        """Build and store year-pair mean rasters"""

        s3_files = s3fs.S3FileSystem()
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

        avg_array = np.where(count_array > 0, sum_array / count_array, np.nan)
        out_name = f'{start_year}_{end_year}_tsm_mean.tif'

        if get_environment() in ['local', 'remote']:
            self._save_raster_data(avg_array, os.path.join(self.year_pair_path, out_name), meta['transform'], meta['crs'], meta['nodata'])
        else:
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                self._save_raster_data(avg_array, tmp.name, meta['transform'], meta['crs'], meta['nodata'])
                s3_dest = f"s3://{self.year_pair_path}/{out_name}"
                s3_files.put(tmp.name, s3_dest)
                os.remove(tmp.name)

    def run(self) -> None:
        # self.download_tsm_data()
        self.create_mean_year_rasters()
        for start_year, end_year in self.year_ranges:
            self.year_pair_rasters(start_year, end_year)