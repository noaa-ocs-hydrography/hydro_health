import os
import pathlib
from ftplib import FTP
import xarray as xr
import matplotlib.pyplot as plt
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.warp import reproject

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
    print(files)
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

def create_rasters():
    folder_path = INPUTS / 'hurricane_data' / 'tsm_data'
    output_folder = INPUTS / 'hurricane_data' / 'tsm_data' / 'tsm_rasters / mean_rasters'
    shapefile = gpd.read_file(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\coastal_boundary_dataset\50m_isobath_polygon\50m_isobath_polygon.shp')

    for year in range(2004, 2025):
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
        
        raster_path = os.path.join(output_folder, f"mean_{year}.tif")
        height, width = annual_mean.shape
        
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
            dest.write(annual_mean, 1)

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
        norm=norm
    )
    ax.set_title(f"TSM Data - {date}")
    fig.colorbar(im, ax=ax, label="TSM (g/m³)")
    return fig

# def create_video(frames, dates, florida_bbox, output_video):
#     global_min = 0
#     global_max = 50
#     norm = Normalize(vmin=global_min, vmax=global_max)

#     images = []
#     for frame, date in zip(frames, dates):
#         subset_frame = preprocess_frame(frame, florida_bbox)
#         fig = plot_frame(subset_frame, date, florida_bbox, norm)
#         fig.canvas.draw()
#         image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
#         image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#         images.append(Image.fromarray(image))
#         plt.close(fig)
    
#     images[0].save(output_video, save_all=True, append_images=images[1:], duration=200, loop=0)

# def main():
#     data_folder = INPUTS / 'hurricane_data' / 'tsm_data'
#     output_video = 'tsm_video.gif'
#     florida_bbox = (24, 31, -87, -79)  # Latitude/Longitude for Florida
#     frames, dates = load_data(data_folder)
#     create_video(frames, dates, florida_bbox, output_video)

# def plot_tsm_over_time(nc_dir, target_lat, target_lon):
#     nc_files = list(nc_dir.glob("*GLOB_4_AV-OLA_TSM_8D_00.nc"))
#     tsm_values = []
#     dates = []

#     for nc_file in nc_files:
#         ds = xr.open_dataset(nc_file)

#         lat = ds['lat'].values
#         lon = ds['lon'].values

#         lat_idx = (np.abs(lat - target_lat)).argmin()
#         lon_idx = (np.abs(lon - target_lon)).argmin()

#         product_name = ds.attrs.get('product_name', '')
#         date_str = product_name.split('__')[0].split('-')[-1]  
#         date_for_file = datetime.strptime(date_str, "%Y%m%d")
#         dates.append(date_for_file)


#         tsm_value_for_file = ds['TSM_mean'].isel(lat=lat_idx, lon=lon_idx).values
#         tsm_values.append(tsm_value_for_file)
        
#         ds.close()

#     time_dates = pd.to_datetime(dates, format='%Y%m%d')
#     time_dates_year = time_dates.year
#     time_dates_days = time_dates.dayofyear
#     year_data = pd.DataFrame({'Date': time_dates, 'TSM': tsm_values, 'Year': time_dates_year, 'Day': time_dates_days})

#     plt.figure(figsize=(10, 6))
#     for year in year_data['Year'].unique():
#         year_subset = year_data[year_data['Year'] == year]
#         plt.plot(year_subset['Day'], year_subset['TSM'], label=str(year), marker='o', linestyle='-', markersize=4)

#     plt.xlabel('Day of Year')
#     plt.ylabel('TSM Mean (g/m3)')
#     plt.title(f'TSM Mean Over Time for W Florida Coast Location at ({target_lat}, {target_lon})')
#     plt.legend(title="Year")
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.show()

# nc_dir = Path(INPUTS / 'hurricane_data' / 'tsm_data')  # Specify your directory here
# target_lat = 28.052267727
# target_lon = -82.9725695

def clip_raster_to_match(input_raster_path, clip_raster_path, output_raster_path):
    with rasterio.open(clip_raster_path) as clip_raster:
        clip_transform = clip_raster.transform
        clip_crs = clip_raster.crs
        clip_width = clip_raster.width
        clip_height = clip_raster.height
        clip_data = clip_raster.read(1) 

    with rasterio.open(input_raster_path) as src:
        resampled_data = np.zeros((src.count, clip_height, clip_width), dtype=src.meta['dtype'])

        for i in range(src.count):
            reproject(
                source=rasterio.band(src, i+1),
                destination=resampled_data[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=clip_transform,
                dst_crs=clip_crs,
                resampling=Resample.bilinear  
            )
        
        resampled_data = resampled_data.astype(np.float32)
        resampled_data[:, clip_data == 0] = np.nan

        output_meta = src.meta.copy()
        output_meta.update({
            "driver": "GTiff",
            "height": clip_height,
            "width": clip_width,
            "transform": clip_transform,
            "crs": clip_crs,
            "dtype": "float32",  
            "nodata": np.nan
        })

        with rasterio.open(output_raster_path, "w", **output_meta) as dst:
            dst.write(resampled_data)

def process_yearly_rasters(input_dir, clip_raster_path, output_dir, start_year, end_year):

    for year in range(start_year, end_year + 1):
        input_raster_path = os.path.join(input_dir, f"mean_{year}.tif")
        output_raster_path = os.path.join(output_dir, f"clipped_{year}.tif")

        clip_raster_to_match(input_raster_path, clip_raster_path, output_raster_path)

def average_rasters(start_year, end_year):
    input_folder=r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\tsm_data\tsm_rasters\clipped_mean_rasters'
    output_folder=r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\year_averages'
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

input_directory = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\tsm_data\tsm_rasters\mean_rasters'
clip_raster = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\prediction.mask.tif"
output_directory = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\tsm_data\tsm_rasters\clipped_mean_rasters'

create_rasters()
process_yearly_rasters(input_directory, clip_raster, output_directory, start_year=2004, end_year=2025)  

year_ranges = [
    (2004, 2006),
    (2006, 2010),
    (2010, 2015),
    (2015, 2022),
]
for start_year, end_year in year_ranges:    
    average_rasters(start_year=start_year, end_year=end_year)            
