import os
import pathlib
import geopandas as gpd
from ftplib import FTP
import xarray as xr
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from rasterio.warp import reproject
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item
import rioxarray
import xarray as xr
import dask.array as da
from dask.distributed import Client
import gc


# INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
# OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'
# DATA_PATH = pathlib.Path(r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\HHM_Run\ER_3\original_data_files\tsm_data")
DATA_PATH = pathlib.Path(r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\tsm_data')


class CreateTSMLayerEngine(Engine):
    """Class to hold the logic for processing the TSM layer"""

    def __init__(self,
                 param_lookup:dict=None):
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

        self.input_directory = pathlib.Path(DATA_PATH / 'tsm_rasters' / 'mean_rasters')
        self.mask_path = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\masks\prediction.mask.UTM17_8m.tif" # temp path
        # self.mask_path = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\HHM_Run\ER_3\prediction_masks\prediction.mask.UTM17_8m.tif"
        self.output_directory = DATA_PATH / 'tsm_rasters' / 'clipped_mean_rasters'

        if param_lookup:
            self.param_lookup = param_lookup
            if self.param_lookup['input_directory'].valueAsText:
                global INPUTS
                # INPUTS = pathlib.Path(self.param_lookup['input_directory'].valueAsText)
            if self.param_lookup['output_directory'].valueAsText:
                global OUTPUTS
                # OUTPUTS = pathlib.Path(self.param_lookup['output_directory'].valueAsText)

    def download_tsm_data(self):
        print('Downloading data...')
        ftp = FTP(self.server)
        ftp.login(user=self.username, passwd=self.password)

        # Move into /globcolour/GLOB
        ftp.cwd('globcolour')
        ftp.cwd('GLOB')

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
                        local_path = DATA_PATH / 'nc_files' / item
                        print(local_path)
                        if not local_path.exists():
                            print(f'Downloading {item} from {path}')
                            with open(local_path, 'wb') as f:
                                ftp.retrbinary(f'RETR {item}', f.write)
        except Exception as e:
            print(f"Error accessing {current_dir}: {e}")

    # def create_rasters(self):
        """ Create rasters from the downloaded TSM data"""      
        
        # folder_path = pathlib.Path(DATA_PATH / 'nc_files')
        folder_path = pathlib.Path(DATA_PATH) # temp

        output_folder = DATA_PATH / 'tsm_rasters' / 'mean_rasters'
        print(output_folder)
        shapefile = gpd.read_file(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\coastal_boundary_dataset\50m_isobath_polygon\50m_isobath_polygon.shp')

        for year in range(2004, 2025):
            print(f'Creating {year} raster...')
            files = [f for f in os.listdir(folder_path) if f"L3m_{year}" in f]
            grid_shape = (4320, 8640) 
            file_count = len(files) 

            datasets = np.full((file_count, *grid_shape), np.nan) 

            for i, file in enumerate(files):
                print(file)
                file_path = os.path.join(folder_path, file)
                ds = xr.open_dataset(file_path)
                
                tsm_data = ds['TSM_mean'].values  # Shape: (time, lat, lon)
                lon = ds['lon'].values
                lat = ds['lat'].values  
                
                grid_shape = tsm_data.shape[-2:]  # Last two dimensions (lat, lon)
                transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), grid_shape[1], grid_shape[0])

                shapefile = shapefile.to_crs("EPSG:4326") 

                shapes = [(geom, 1) for geom in shapefile.geometry]
                rasterized_mask = rasterize(shapes, out_shape=grid_shape, transform=transform, fill=0, dtype="uint8")
                tsm_data[rasterized_mask == 0] = np.nan
                
                datasets[i, :, :] = tsm_data  
            
            annual_mean = np.nanmean(datasets, axis=0)
            annual_mean[annual_mean == 0] = np.nan
            
            # Save un-clipped mean raster to disk first
            mean_raster_path = os.path.join(output_folder, f"mean_{year}.tif")
            height, width = annual_mean.shape

            with rasterio.open(
                mean_raster_path,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=transform,
                nodata=np.nan,
                compress="lzw",
            ) as dst:
                dst.write(annual_mean, 1)

            # Then clip it using the mask
            output_raster_path = os.path.join(self.output_directory, f"clipped_{year}.tif")
            print(output_raster_path)
            self.clip_raster_to_match(mean_raster_path, self.mask_path, output_raster_path)

    from rasterio.windows import from_bounds
    from rasterio.io import MemoryFile

    def create_rasters(self):
        """Create annual mean TSM rasters clipped to the extent of a raster mask."""

        folder_path = pathlib.Path(DATA_PATH)
        output_folder = folder_path / 'tsm_rasters' / 'mean_rasters'
        output_folder.mkdir(parents=True, exist_ok=True)

        grid_shape = (4320, 8640)
        transform = from_bounds(-180, -90, 180, 90, grid_shape[1], grid_shape[0])
        crs = "EPSG:4326"

        # Get clipping bounds from the mask
        with rasterio.open(self.mask_path) as mask_src:
            mask_bounds = mask_src.bounds

        for year in range(2004, 2025):
            print(f'Processing year: {year}')
            nc_files = [f for f in os.listdir(folder_path) if f"L3m_{year}" in f and f.endswith('.nc')]
            if not nc_files:
                print(f"  No NetCDF files found for {year}")
                continue

            tsm_stack = []

            for fname in nc_files:
                path = folder_path / fname
                try:
                    ds = xr.open_dataset(path)
                    arr = ds['TSM_mean'].values.squeeze().astype(np.float32)
                    arr[arr == 0] = np.nan  # replace 0 with nan
                    tsm_stack.append(arr)
                    ds.close()
                except Exception as e:
                    print(f"  Failed to process {fname}: {e}")

            if not tsm_stack:
                print(f"  No valid rasters for {year}")
                continue

            # Compute the mean
            print("  Computing mean...")
            annual_mean = np.nanmean(np.stack(tsm_stack, axis=0), axis=0)
            del tsm_stack
            gc.collect()

            # Clip to bounds
            print("  Clipping to mask...")
            window = window_from_bounds(*mask_bounds, transform=transform)
            window = window.round_offsets().round_lengths()
            row_off, col_off = int(window.row_off), int(window.col_off)
            height, width = int(window.height), int(window.width)
            clipped_array = annual_mean[row_off:row_off+height, col_off:col_off+width]
            clipped_transform = window_transform(window, transform)

            # Save
            print("  Saving clipped raster...")
            out_path = output_folder / f"TSM_mean_{year}.tif"
            with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=1,
                dtype='float32',
                crs=crs,
                transform=clipped_transform,
                nodata=np.nan,
                compress='lzw',
            ) as dst:
                dst.write(clipped_array, 1)

            print(f"  Done: {out_path}")

    def clip_raster_to_match(self, raster_path, clip_raster_path, output_raster_path):
        
        size = 512
        # Open source raster (with Dask chunks)
        src_data = rioxarray.open_rasterio(raster_path, chunks={'x': size, 'y': size})
        
        # Open mask raster (also chunked)
        clip_mask = rioxarray.open_rasterio(clip_raster_path, chunks={'x': size, 'y': size})
        
        # Reproject to match the mask
        reprojected = src_data.rio.reproject_match(clip_mask)

        masked = reprojected.where(clip_mask != 0)

        # Trigger Dask chunk-parallel computation before writing
        masked = masked.compute()

        masked.rio.to_raster(output_raster_path, compress='LZW')

    def year_pair_rasters(self, start_year, end_year):
        """_summary_

        :param _type_ start_year: _description_
        :param _type_ end_year: _description_
        """        


        input_folder = DATA_PATH / 'tsm_rasters' / 'clipped_mean_rasters'
        output_folder = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\HHM_Run\ER_3\model_variables\Prediction\pre_processed\tsm_year_pairs" #temp
        # output_folder = DATA_PATH / 'tsm_rasters' / 'year_pair_averages' #fix
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

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f'{start_year}_{end_year}_{output_name}')
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(average_array, 1)

        print(f"Averaged raster saved to {output_path}.")

    def start(self):
        """Entrypoint for processing the TSM layer"""

        # client = Client()  # Or Client(n_workers=4, threads_per_worker=2)
        # print(client)   
        # self.download_tsm_data()
        self.create_rasters()

        for start_year, end_year in self.year_ranges:    
            self.year_pair_rasters(start_year=start_year, end_year=end_year)   

