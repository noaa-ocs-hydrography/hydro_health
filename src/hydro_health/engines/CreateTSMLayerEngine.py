import os
import re
import pathlib
import fnmatch
import datetime
from ftplib import FTP
import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


class CreateTSMLayerEngine(Engine):
    """Class to hold the logic for processing the TSM layer"""

    def __init__(self):
        super().__init__()
        self.year_ranges = [
            (2004, 2006),
            (2006, 2010),
            (2010, 2015),
            (2015, 2022)]
        self.username = 'ftp_gc_AMcCutchan'
        self.password = 'AMcCutchan_4915'
        self.server = 'ftp.hermes.acri.fr'
        self.downloaded_files = []

    def download_tsm_data(self):
        print('Connecting to FTP server...')
        ftp = FTP(self.server)
        ftp.login(user=self.username, passwd=self.password)

        root_dir = '/GLOB'
        print(f"Starting recursive download from: {root_dir}")

        try:
            self.download_recursive(ftp, root_dir)
        except Exception as e:
            print(f"Error during download: {e}")

        ftp.quit()
        self.write_download_log()

    def write_download_log(self):
        def extract_start_date(filename):
            match = re.search(r'(\d{8})-\d{8}', filename)
            return match.group(1) if match else '99999999'

        sorted_files = sorted(self.downloaded_files, key=extract_start_date)
        log_path = pathlib.Path(get_config_item('TSM', 'DATA_PATH')) / 'downloaded_files.txt'
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_path, 'w') as f:
            for name in sorted_files:
                f.write(name + '\n')

        print(f"\nDownload log written to: {log_path}")
    
    def download_recursive(self, ftp, current_dir):
        try:
            ftp.cwd(current_dir)
            normalized = current_dir.lower().strip('/')

            skip_patterns = (
                '*meris/*', '*viirsn*', '*seawifs*', '*modis*',
                '*viirsj1*', '*meris4rp*', '*olcib*',
                '*/1998/*', '*/1999/*', '*/2000/*', '*/2001/*', '*/2002/*', '*/2003/*',
                '*/month/*', '*/day/*', '*/track/*',
                '*/olcia/*/2019/*', '*/olcia/*/2020/*', '*/olcia/*/2021/*',
                '*/olcia/*/2022/*', '*/olcia/*/2023/*', '*/olcia/*/2024/*', '*/olcia/*/2025/*'
            )

            if any(fnmatch.fnmatch(normalized, pattern.strip('/')) for pattern in skip_patterns):
                return

            items = list(ftp.mlsd())
            print(f"Found {len(items)} items in {current_dir}")

            for name, facts in items:
                item_path = f"{current_dir}/{name}"
                if facts.get("type") == "dir":
                    self.download_recursive(ftp, item_path)
                elif name.endswith('.nc') and 'TSM_8D' in name and '_4' in name and 'L3m' in name:
                    # Skip OLCIA files after 20180501 bc merged files for A/B satellites exist after this data
                    if 'AV-OLA' in name:
                        match = re.search(r'L3m_(\d{8})-', name)
                        if match:
                            file_date = datetime.datetime.strptime(match.group(1), "%Y%m%d")
                            if file_date > datetime.datetime(2018, 5, 1):
                                print(f"Skipping late OLCIA file: {name}")
                                continue  

                    local_dir = pathlib.Path('TSM_download/nc_files')
                    local_path = local_dir / name
                    local_dir.mkdir(parents=True, exist_ok=True)

                    if not local_path.exists():
                        print(f"Downloading {name} from {item_path}")
                        with open(local_path, 'wb') as f:
                            ftp.retrbinary(f'RETR {name}', f.write)
                        self.downloaded_files.append(name)

        except Exception as e:
            print(f"Error accessing {current_dir}: {e}")

    def create_rasters(self):
        """Create annual mean TSM rasters and write full-resolution GeoTIFFs"""
        
        folder_path = pathlib.Path(get_config_item('TSM', 'DATA_PATH'))
        output_folder = pathlib.Path(get_config_item('TSM', 'MEAN_RASTER_PATH')) 

        grid_shape = (4320, 8640)  # (height, width)
        transform = from_bounds(-180, -90, 180, 90, grid_shape[1], grid_shape[0])
        crs = "EPSG:4326"
        nodata_val = np.nan

        for year in range(2004, 2025):
            print(f'Processing year: {year}')
            nc_files = [f for f in os.listdir(folder_path) if f"L3m_{year}" in f and f.endswith('.nc')]
            if not nc_files:
                print(f"  No NetCDF files found for {year}")
                continue

            running_sum = None
            valid_count = None

            for fname in nc_files:
                path = folder_path / fname
                try:
                    ds = xr.open_dataset(path)
                    arr = ds['TSM_mean'].values.squeeze().astype(np.float32)
                    arr[arr == 0] = np.nan 
                    ds.close()

                    if running_sum is None:
                        running_sum = np.zeros_like(arr, dtype=np.float32)
                        valid_count = np.zeros_like(arr, dtype=np.uint16)

                    valid = ~np.isnan(arr)
                    running_sum[valid] += arr[valid]
                    valid_count[valid] += 1

                except Exception as e:
                    print(f"  Failed to process {fname}: {e}")

            if running_sum is None:
                print(f"  No valid rasters for {year}")
                continue

            with np.errstate(invalid='ignore'):
                annual_mean = running_sum / valid_count
            annual_mean[valid_count == 0] = nodata_val

            out_path = output_folder / f"TSM_mean_{year}.tif"
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=annual_mean.shape[0],
                width=annual_mean.shape[1],
                count=1,
                dtype='float32',
                crs=crs,
                transform=transform,
                nodata=nodata_val,
                compress='deflate',
                tiled=True,
                zlevel=9,
            ) as dst:
                dst.write(annual_mean, 1)

            print(f"Saved raster to {out_path}")

    def year_pair_rasters(self, start_year, end_year):
        """ Create a raster average for years in the range of start_year and end_year

        :param _type_ start_year: first year in the range
        :param _type_ end_year: last year in the range
        """        

        input_folder = pathlib.Path(get_config_item('TSM', 'MEAN_RASTER_PATH')) 
        output_folder = pathlib.Path(get_config_item('TSM', 'YEAR_PAIR_RASTER_PATH'))
        output_name=f"tsm_mean.tif"

        raster_files = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.endswith(".tif") and any(str(year) in f for year in range(start_year, end_year + 1))
        ]

        with rasterio.open(raster_files[0]) as src:
            meta = src.meta.copy()
            meta.update(dtype="float32", nodata=np.nan, compress="lzw")
            raster_shape = src.shape

        sum_array = np.zeros(raster_shape, dtype=np.float32)
        count_array = np.zeros(raster_shape, dtype=np.int32)

        for raster_file in raster_files:
            with rasterio.open(raster_file) as src:
                data = src.read(1)

                valid_mask = ~np.isnan(data)
                sum_array[valid_mask] += data[valid_mask]
                count_array[valid_mask] += 1

        with np.errstate(divide='ignore', invalid='ignore'):
            average_array = np.where(count_array > 0, sum_array / count_array, np.nan) 

        output_path = os.path.join(output_folder, f'{start_year}_{end_year}_{output_name}')
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(average_array, 1)

        print(f"Averaged raster saved to {output_path}.")

    def start(self):
        """Entrypoint for processing the TSM layer"""
  
        self.download_tsm_data()
        self.create_rasters()

        for start_year, end_year in self.year_ranges:    
            self.year_pair_rasters(start_year=start_year, end_year=end_year)   

