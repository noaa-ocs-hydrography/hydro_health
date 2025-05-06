import os
import pathlib
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
        # self.pattern = 'L3m_'  # You can refine this later with fnmatch if needed    

    def download_tsm_data(self):
        print('Downloading data...')
        ftp = FTP(self.server)
        ftp.login(user=self.username, passwd=self.password)

        ftp.cwd('globcolour/GLOB')

        sensors = ftp.nlst()  # List all sensors: merged, seawifs, meris, modis, etc.

        for sensor in sensors:
            if sensor == 'merged':
                continue  # Skip 'merged' sensor

            base_dir = f'/globcolour/GLOB/{sensor}/8-day'
            print(f"Starting download for sensor: {sensor}")

            try:
                self.download_recursive(ftp, base_dir)
            except Exception as e:
                print(f"Skipping sensor {sensor} due to error: {e}")

        ftp.quit()

    def download_recursive(self, ftp, current_dir):
        try:
            ftp.cwd(current_dir)
            items = ftp.nlst()  # List everything in current folder

            for item in items:
                path = f'{current_dir}/{item}'
                print(path)
                try:
                    ftp.cwd(path)  # Try to move into it
                    # If success, it was a folder: recurse inside
                    self.download_recursive(ftp, path)
                    ftp.cwd('..')  # Move back up after recursion
                except Exception:
                    # If cannot move into it, it's a file
                    if item.endswith('.nc') and 'TSM_8D' in item:
                        local_path = pathlib.Path(get_config_item('TSM', 'DATA_PATH')) / 'nc_files' / item
                        print(local_path)
                        if not local_path.exists():
                            print(f'Downloading {item} from {path}')
                            with open(local_path, 'wb') as f:
                                ftp.retrbinary(f'RETR {item}', f.write)
        except Exception as e:
            print(f"Error accessing {current_dir}: {e}")

    def create_rasters(self):
        """Create annual mean TSM rasters and write full-resolution GeoTIFFs"""
        
        folder_path = pathlib.Path(get_config_item('TSM', 'DATA_PATH'))
        output_folder = pathlib.Path(get_config_item('TSM', 'DATA_PATH')) / 'tsm_rasters' / 'mean_rasters'

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
                    arr[arr == 0] = np.nan  # treat 0s as invalid, # TODO double check
                    ds.close()

                    if running_sum is None: # TODO double check the mean and count
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

            with np.errstate(invalid='ignore'):  # suppress warnings from divide-by-zero
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

        input_folder = pathlib.Path(get_config_item('TSM', 'DATA_PATH')) / 'tsm_rasters' / 'mean_rasters'
        output_folder = pathlib.Path(get_config_item('TSM', 'DATA_PATH')) / 'tsm_rasters' / 'year_pair_rasters'
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
  
        # self.download_tsm_data()
        self.create_rasters()

        for start_year, end_year in self.year_ranges:    
            self.year_pair_rasters(start_year=start_year, end_year=end_year)   

