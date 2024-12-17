import os
import pathlib
from ftplib import FTP
import xarray as xr
import matplotlib.pyplot as plt
import glob
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'

def download_tsm_data(directory):
    server = 'ftp.hermes.acri.fr'
    username = 'ftp_hermes'
    password = 'hermes%'

    download_folder = INPUTS / 'hurricane_data' / 'tsm_data'

    ftp = FTP(server)
    ftp.login(user=username, passwd=password)
    ftp.cwd(directory)

    files = ftp.nlst()

    for file_name in files:
        local_file_path = os.path.join(download_folder, file_name)
        print(f'Downloading {file_name} to {local_file_path}...')
        with open(local_file_path, 'wb') as file:
            ftp.retrbinary(f'RETR {file_name}', file.write)

    ftp.quit()    

def read_tsm_data():
    file_path = INPUTS / 'hurricane_data' / 'tsm_data' / 'L3m_20160422-20160429__GLOB_4_AV-OLA_TSM_8D_00.nc'
    ds = xr.open_dataset(file_path)

    print(ds)
    print(f'Variables:', list(ds.data_vars))


    # Check the 'TSM_mean' variable
    # print(ds['TSM_mean'])
    # print("TSM_mean Min:", ds['TSM_mean'].min().values)
    # print("TSM_mean Max:", ds['TSM_mean'].max().values)
    # print("Number of NaN values:", ds['TSM_mean'].isnull().sum().values)

    # lat_min, lat_max = -90, 90  # Expand to global for debugging
    # lon_min, lon_max = -180, 180
    # lat_min, lat_max = 24.5, 31.0  # Expand to global for debugging
    # lon_min, lon_max = -87.6, -79.8
    # subset = ds['TSM_mean'].sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    # print("Subset shape:", subset.shape)
    # print("Subset min value:", subset.min().values)
    # print("Subset max value:", subset.max().values)
    # print("Longitude range:", ds['lon'].min().values, "to", ds['lon'].max().values)

    # # TSM_mean = ds['TSM_mean']

    # plt.figure(figsize=(10, 5))
    # plt.imshow(subset.values, origin='lower', extent=[lon_min, lon_max, lat_min, lat_max], cmap='viridis')
    # plt.colorbar(label='TSM_mean (g/m3)')
    # plt.title("Subset Data")
    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # TSM_mean.plot(cmap='viridis', robust=True)
    # plt.title('TSM mean')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.show()

    ds.close

def create_rasters_and_plots():
    folder_path = INPUTS / 'hurricane_data' / 'tsm_data'
    output_folder = INPUTS / 'hurricane_data' / 'tsm_data' / 'tsm_rasters'
    shapefile = gpd.read_file(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\coastal_boundary_dataset\50m_isobath_polygon\50m_isobath_polygon.shp')

    florida_bounds = (24, 31, -87, -79)

    for year in range(2016, 2025):
        files = [f for f in os.listdir(folder_path) if f"L3m_{year}" in f]
        grid_shape = (4320, 8640) 
        file_count = len(files) 

        datasets = np.full((file_count, *grid_shape), np.nan)  # Preallocate with NaNs for missing values

        for i, file in enumerate(files):
            print(file)
            file_path = os.path.join(folder_path, file)
            ds = xr.open_dataset(file_path)
            
            tsm_data = ds['TSM_mean'].values  # Shape: (time, lat, lon)
            lon = ds['lon'].values
            lat = ds['lat'].values  
            
            grid_shape = tsm_data.shape[-2:]  # Last two dimensions (lat, lon)
            transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), grid_shape[1], grid_shape[0])

            # Reproject shapefile to match TSM data CRS
            shapefile = shapefile.to_crs("EPSG:4326")  # Assuming TSM data uses WGS 84

            # Rasterize the shapefile to match the TSM data grid
            shapes = [(geom, 1) for geom in shapefile.geometry]
            rasterized_mask = rasterize(shapes, out_shape=grid_shape, transform=transform, fill=0, dtype="uint8")
            
            # Mask out values where the rasterized shapefile is 0
            tsm_data[rasterized_mask == 0] = np.nan
            
            # Store the data in the preallocated array (index by file)
            datasets[i, :, :] = tsm_data  # Assuming you want the first time step (index 0)
        
        # annual_variance = np.nanvar(np.stack(datasets), axis=0)  
        annual_stddev = np.nansum(datasets, axis=0)
        annual_stddev[annual_stddev == 0] = np.nan
        # annual_stddev = np.nansum(np.stack(datasets), axis=0)
        
        raster_path = os.path.join(output_folder, f"sum_{year}.tif")
        height, width = annual_stddev.shape
        
        out_meta = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",  
            "transform": transform,
            "nodata": np.nan,
        }
        with rasterio.open(raster_path, "w", **out_meta) as dest:
            dest.write(annual_stddev, 1)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))  # Adjust grid for years
    axs = axs.ravel()
    for i, year in enumerate(range(2016, 2025)):
        raster_path = os.path.join(output_folder, f"stddev_{year}.tif")
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            # Clip to Florida bounds for plotting
            lats = np.linspace(bounds.top, bounds.bottom, data.shape[0])
            lons = np.linspace(bounds.left, bounds.right, data.shape[1])
            lat_mask = (lats >= florida_bounds[0]) & (lats <= florida_bounds[1])
            lon_mask = (lons >= florida_bounds[2]) & (lons <= florida_bounds[3])
            clipped_data = data[np.ix_(lat_mask, lon_mask)]
            
            # Extent for Florida bounds
            extent = [florida_bounds[2], florida_bounds[3], florida_bounds[0], florida_bounds[1]]
            
            axs[i].imshow(clipped_data, cmap="viridis", extent=extent, origin="upper")
            axs[i].set_title(f"Year {year}")
            axs[i].set_xlabel("Longitude")
            axs[i].set_ylabel("Latitude")

    plt.tight_layout()
    plt.show()        

def load_data(data_folder):
    file_paths = sorted(glob.glob(os.path.join(data_folder, '*.nc')))
    frames = []
    dates = []
    for file_path in file_paths:
        ds = xr.open_dataset(file_path)
        tsm_data = ds['TSM_mean']
        frames.append(tsm_data)
        date_range = ds.attrs['product_name'].split('_')[1].split('-')
        dates.append(f"{date_range[0]} to {date_range[1]}")
        ds.close()
    return frames, dates

def preprocess_frame(frame, florida_bbox):
    lat_min, lat_max, lon_min, lon_max = florida_bbox
    subset = frame.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    flipped_subset = subset[::-1]  # Flip latitudes for proper orientation
    return flipped_subset

def plot_frame(frame, date, florida_bbox, norm):
    florida_lat_min, florida_lat_max, florida_lon_min, florida_lon_max = florida_bbox

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(
        frame,
        origin='lower',
        extent=(florida_lon_min, florida_lon_max, florida_lat_min, florida_lat_max),
        cmap="viridis",
        norm=norm  # Use the consistent normalization
    )
    ax.set_title(f"TSM Data - {date}")
    fig.colorbar(im, ax=ax, label="TSM (g/m³)")
    return fig

def create_video(frames, dates, florida_bbox, output_video):
    global_min = 0
    global_max = 50
    norm = Normalize(vmin=global_min, vmax=global_max)

    images = []
    for frame, date in zip(frames, dates):
        subset_frame = preprocess_frame(frame, florida_bbox)
        fig = plot_frame(subset_frame, date, florida_bbox, norm)
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(Image.fromarray(image))
        plt.close(fig)
    
    images[0].save(output_video, save_all=True, append_images=images[1:], duration=200, loop=0)

def main():
    data_folder = INPUTS / 'hurricane_data' / 'tsm_data'
    output_video = 'tsm_video.gif'
    florida_bbox = (24, 31, -87, -79)  # Latitude/Longitude for Florida
    frames, dates = load_data(data_folder)
    create_video(frames, dates, florida_bbox, output_video)

def plot_tsm_over_time(nc_dir, target_lat, target_lon):
    nc_files = list(nc_dir.glob("*GLOB_4_AV-OLA_TSM_8D_00.nc"))
    tsm_values = []
    dates = []

    for nc_file in nc_files:
        ds = xr.open_dataset(nc_file)

        lat = ds['lat'].values
        lon = ds['lon'].values

        lat_idx = (np.abs(lat - target_lat)).argmin()
        lon_idx = (np.abs(lon - target_lon)).argmin()

        product_name = ds.attrs.get('product_name', '')
        date_str = product_name.split('__')[0].split('-')[-1]  
        date_for_file = datetime.strptime(date_str, "%Y%m%d")
        dates.append(date_for_file)


        tsm_value_for_file = ds['TSM_mean'].isel(lat=lat_idx, lon=lon_idx).values
        tsm_values.append(tsm_value_for_file)
        
        ds.close()

    time_dates = pd.to_datetime(dates, format='%Y%m%d')
    time_dates_year = time_dates.year
    time_dates_days = time_dates.dayofyear
    year_data = pd.DataFrame({'Date': time_dates, 'TSM': tsm_values, 'Year': time_dates_year, 'Day': time_dates_days})

    plt.figure(figsize=(10, 6))
    for year in year_data['Year'].unique():
        year_subset = year_data[year_data['Year'] == year]
        plt.plot(year_subset['Day'], year_subset['TSM'], label=str(year), marker='o', linestyle='-', markersize=4)

    plt.xlabel('Day of Year')
    plt.ylabel('TSM Mean (g/m3)')
    plt.title(f'TSM Mean Over Time for W Florida Coast Location at ({target_lat}, {target_lon})')
    plt.legend(title="Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

nc_dir = Path(INPUTS / 'hurricane_data' / 'tsm_data')  # Specify your directory here
target_lat = 28.052267727
target_lon = -82.9725695

# plot_tsm_over_time(nc_dir, target_lat, target_lon) 

# read_tsm_data()
create_rasters_and_plots()
# main()   
    
# download_tsm_data(directory='350121757')   
