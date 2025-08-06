import os
import re
import math
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
import contextily as cx
from matplotlib import colors

def plot_rasters_by_year(input_folder, output_folder):
    year_datasets = {}

    filename_pattern = re.compile(r'(.+)_(\d{4})_resampled\.tif', re.IGNORECASE)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            match = filename_pattern.search(filename)
            if match:
                dataset_name = match.group(1).split('_')[-1]
                year = match.group(2)
                
                title_name = f"{dataset_name}_{year}"
                
                if year not in year_datasets:
                    year_datasets[year] = []
                year_datasets[year].append({'path': os.path.join(input_folder, filename), 'title': title_name})

    if not year_datasets:
        print("No raster files found matching the naming convention.")
        return

    target_crs = 'EPSG:3857'
    global_min = np.inf
    global_max = -np.inf
    global_left = np.inf
    global_bottom = np.inf
    global_right = -np.inf
    global_top = -np.inf
    reference_res = None

    # Step 1: Scan rasters for global min/max and global bounds in target CRS
    for year in year_datasets:
        for dataset in year_datasets[year]:
            with rasterio.open(dataset['path']) as src:
                data = src.read(1, masked=True)
                if data.min() < global_min:
                    global_min = data.min()
                if data.max() > global_max:
                    global_max = data.max()

                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)
                
                left = transform.c
                top = transform.f
                right = left + width * transform.a
                bottom = top + height * transform.e

                global_left = min(global_left, left)
                global_bottom = min(global_bottom, bottom)
                global_right = max(global_right, right)
                global_top = max(global_top, top)

                if reference_res is None:
                    reference_res = (abs(transform.a), abs(transform.e))  # (xres, yres)

    # Step 2: Define a common transform and shape
    xres, yres = reference_res
    dst_width = int((global_right - global_left) / xres)
    dst_height = int((global_top - global_bottom) / yres)
    common_transform = from_bounds(global_left, global_bottom, global_right, global_top, dst_width, dst_height)
    common_extent = (global_left, global_right, global_bottom, global_top)

    # Step 3: Prepare subplot grid
    sorted_years = sorted(year_datasets.keys())
    num_years = len(sorted_years)
    cols = math.ceil(math.sqrt(num_years))
    rows = math.ceil(num_years / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
    if num_years == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    im = None
    
    # Create a colormap from 'viridis' and set the color for NaN values to white
    cmap = plt.cm.get_cmap('viridis').copy()
    cmap.set_bad(color='white')
    
    for i, year in enumerate(sorted_years):
        ax = axes[i]
        datasets = year_datasets[year]
        
        title_lines = [dataset['title'] for dataset in datasets]
        ax.set_title('\n'.join(title_lines), loc='left', fontsize=12, fontweight='bold')

        for j, dataset in enumerate(datasets):
            with rasterio.open(dataset['path']) as src:
                destination = np.zeros((dst_height, dst_width), dtype=src.dtypes[0])
                reproject(
                    source=rasterio.band(src, 1),
                    destination=destination,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=common_transform,
                    dst_crs=target_crs,
                    dst_nodata=src.nodata,
                    resampling=Resampling.bilinear
                )
                
                # After reprojection, create a masked array based on the nodata value
                if src.nodata is not None:
                    masked_destination = np.ma.masked_equal(destination, src.nodata)
                else:
                    masked_destination = destination
                
                im = ax.imshow(masked_destination, extent=common_extent, cmap=cmap, vmin=global_min, vmax=global_max)

        ax.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax.set_aspect('equal')

    # Step 4: Turn off unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Step 5: Add colorbar
    if global_min != np.inf and global_max != -np.inf and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.set_label('Depth (meters)', fontsize=20, rotation=270, labelpad=20)

    plt.suptitle("Raster Datasets by Year", fontsize=24)

    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'raster_plots_all_years.png'
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

def plot_consecutive_year_differences(input_folder, output_folder):
    year_datasets = {}
    filename_pattern = re.compile(r'(.+)_(\d{4})_resampled\.tif', re.IGNORECASE)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            match = filename_pattern.search(filename)
            if match:
                year = match.group(2)
                if year not in year_datasets:
                    year_datasets[year] = []
                year_datasets[year].append(os.path.join(input_folder, filename))

    if not year_datasets:
        print("No raster files found matching the naming convention.")
        return

    sorted_years = sorted(year_datasets.keys())
    if len(sorted_years) < 2:
        print("Not enough years to calculate differences.")
        return

    target_crs = 'EPSG:3857'
    
    # First pass to determine common extent and resolution
    global_left = np.inf
    global_bottom = np.inf
    global_right = -np.inf
    global_top = -np.inf
    reference_res = None
    
    for year in sorted_years:
        for path in year_datasets[year]:
            with rasterio.open(path) as src:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)
                
                global_left = min(global_left, transform.c)
                global_bottom = min(global_bottom, transform.f + height * transform.e)
                global_right = max(global_right, transform.c + width * transform.a)
                global_top = max(global_top, transform.f)
                
                if reference_res is None:
                    reference_res = (abs(transform.a), abs(transform.e))

    xres, yres = reference_res
    dst_width = int((global_right - global_left) / xres)
    dst_height = int((global_top - global_bottom) / yres)
    common_transform = from_bounds(global_left, global_bottom, global_right, global_top, dst_width, dst_height)
    common_extent = (global_left, global_right, global_bottom, global_top)
    
    # Store reprojected data for all years
    reprojected_data = {}
    for year in sorted_years:
        # Assuming only one raster per year for difference calculation
        path = year_datasets[year][0]
        with rasterio.open(path) as src:
            destination = np.zeros((dst_height, dst_width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=common_transform,
                dst_crs=target_crs,
                dst_nodata=src.nodata,
                resampling=Resampling.bilinear
            )
            reprojected_data[year] = np.ma.masked_equal(destination, src.nodata)

    # Calculate differences and determine global min/max for differences
    diff_data = []
    global_diff_min = np.inf
    global_diff_max = -np.inf
    for i in range(len(sorted_years) - 1):
        year1 = sorted_years[i]
        year2 = sorted_years[i+1]
        
        data1 = reprojected_data[year1]
        data2 = reprojected_data[year2]
        
        # Create a combined mask
        combined_mask = np.ma.mask_or(data1.mask, data2.mask)
        
        # Calculate difference only where both are valid
        diff = np.ma.array(data2 - data1, mask=combined_mask)
        diff_data.append({'diff': diff, 'title': f'Difference: {year1} to {year2}'})
        
        if diff.min() < global_diff_min:
            global_diff_min = diff.min()
        if diff.max() > global_diff_max:
            global_diff_max = diff.max()

    # Create custom colormap and norm for difference plots
    cmap = plt.cm.get_cmap('RdYlBu').copy()
    # Define custom color for masked values (black)
    cmap.set_bad(color='black')
    
    # Plotting differences
    num_diffs = len(diff_data)
    cols = math.ceil(math.sqrt(num_diffs))
    rows = math.ceil(num_diffs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
    if num_diffs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
        
    im_diff = None
    for i, diff_dict in enumerate(diff_data):
        ax = axes[i]
        diff_array = diff_dict['diff']
        title = diff_dict['title']
        
        im_diff = ax.imshow(diff_array, extent=common_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        
        ax.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        ax.set_aspect('equal')
        
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_diff_min != np.inf and global_diff_max != -np.inf and im_diff is not None:
        cbar = fig.colorbar(im_diff, ax=axes.ravel().tolist())
        cbar.set_label('Difference (m)', fontsize=20, rotation=270, labelpad=20)
    
    plt.suptitle("Consecutive Year Differences", fontsize=24)
    
    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'consecutive_year_differences.png'
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Difference plot saved to: {output_path}")

# Set input/output folders
raster_folder = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast_resampled"
plot_output_folder = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast_plots"

plot_rasters_by_year(raster_folder, plot_output_folder)
plot_consecutive_year_differences(raster_folder, plot_output_folder)