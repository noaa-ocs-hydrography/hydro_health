import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
from rasterio.features import shapes
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import math
from shapely.geometry import shape

def plot_rasters_by_year(input_folder, output_folder, mask_path):
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

    # Open the mask file once and keep the object for the whole function
    mask_src = rasterio.open(mask_path)
    
    # --- Part 1: Determine Global Min/Max and Common Extent for both plots ---
    global_min = np.inf
    global_max = -np.inf
    global_left = np.inf
    global_bottom = np.inf
    global_right = -np.inf
    global_top = -np.inf
    reference_res = None
    
    all_raster_paths = [dataset['path'] for year in year_datasets for dataset in year_datasets[year]]
    all_paths_for_extent = all_raster_paths + [mask_path]

    for path in all_paths_for_extent:
        with rasterio.open(path) as src:
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
                reference_res = (abs(transform.a), abs(transform.e))

    xres, yres = reference_res
    dst_width = int((global_right - global_left) / xres)
    dst_height = int((global_top - global_bottom) / yres)
    common_transform = from_bounds(global_left, global_bottom, global_right, global_top, dst_width, dst_height)
    common_extent = (global_left, global_right, global_bottom, global_top)
    
    # Reproject the mask once to the common extent
    mask_destination_global = np.zeros((dst_height, dst_width), dtype=mask_src.dtypes[0])
    reproject(
        source=rasterio.band(mask_src, 1),
        destination=mask_destination_global,
        src_transform=mask_src.transform,
        src_crs=mask_src.crs,
        dst_transform=common_transform,
        dst_crs=target_crs,
        dst_nodata=mask_src.nodata,
        resampling=Resampling.nearest
    )
    mask_boolean_global = mask_destination_global != mask_src.nodata
    mask_geometries_global = [shape(geom) for geom, val in shapes(mask_boolean_global.astype(np.uint8), transform=common_transform) if val > 0]

    # Recalculate global min/max with the combined mask
    for path in all_raster_paths:
        with rasterio.open(path) as src:
            reprojected_data = np.zeros((dst_height, dst_width), dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=reprojected_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=common_transform,
                dst_crs=target_crs,
                dst_nodata=src.nodata,
                resampling=Resampling.bilinear
            )
            combined_mask = np.logical_or(reprojected_data == src.nodata, ~mask_boolean_global)
            masked_data = np.ma.array(reprojected_data, mask=combined_mask)

            if masked_data.min() < global_min:
                global_min = masked_data.min()
            if masked_data.max() > global_max:
                global_max = masked_data.max()

    cmap = plt.colormaps['ocean'].copy()
    cmap.set_bad(color='white')

    # --- Part 2: Plotting with Global Extent ---
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
    
    for i, year in enumerate(sorted_years):
        ax = axes[i]
        dataset = year_datasets[year][0] # Assuming one dataset per year for now
        
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
            
            combined_mask = np.logical_or(destination == src.nodata, ~mask_boolean_global)
            masked_destination = np.ma.array(destination, mask=combined_mask)
            im = ax.imshow(masked_destination, extent=common_extent, cmap=cmap, vmin=global_min, vmax=global_max)

        for geom in mask_geometries_global:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.5, zorder=10)

        ax.set_title(dataset['title'], loc='left', fontsize=12, fontweight='bold')
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)
        ax.set_aspect('equal')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_min != np.inf and global_max != -np.inf and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.set_label('Depth (meters)', fontsize=20, rotation=270, labelpad=20)

    plt.suptitle("Raster Datasets by Year (Global Extent)", fontsize=24)

    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'raster_plots_global_extent.png'
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Global extent plot saved to: {output_path}")

    # ---------------------------------------------------------------------------------------------
    
# --- Part 3: Plotting with Individual Extents (Single Figure) ---
    
    # Create the output directory
    output_folder_individual = os.path.join(output_folder, 'individual_extents')
    os.makedirs(output_folder_individual, exist_ok=True)

    # Get sorted years and calculate grid size
    sorted_years = sorted(year_datasets.keys())
    num_years = len(sorted_years)
    cols = math.ceil(math.sqrt(num_years))
    rows = math.ceil(num_years / cols)

    # Create a single figure with a grid of subplots, disabling shared axes
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True, sharex=False, sharey=False)
    
    if num_years == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    im = None
    
    # Use the existing cmap from earlier in the script
    cmap = plt.colormaps['ocean'].copy()
    cmap.set_bad(color='white')

    # Re-open mask_src since the previous with block closed it.
    with rasterio.open(mask_path) as mask_src:
        for i, year in enumerate(sorted_years):
            ax = axes[i]
            dataset = year_datasets[year][0]
            
            with rasterio.open(dataset['path']) as src:
                # Calculate local transform and extent for this specific subplot
                local_transform, local_width, local_height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds)
                local_extent = (local_transform.c, local_transform.c + local_width * local_transform.a, 
                                local_transform.f + local_height * local_transform.e, local_transform.f)
                
                destination = np.zeros((local_height, local_width), dtype=src.dtypes[0])
                reproject(source=rasterio.band(src, 1), destination=destination, src_transform=src.transform, src_crs=src.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=src.nodata, resampling=Resampling.bilinear)
                
                mask_destination_local = np.zeros((local_height, local_width), dtype=mask_src.dtypes[0])
                reproject(source=rasterio.band(mask_src, 1), destination=mask_destination_local, src_transform=mask_src.transform, src_crs=mask_src.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=mask_src.nodata, resampling=Resampling.nearest)
                
                mask_boolean_local = mask_destination_local != mask_src.nodata
                combined_mask = np.logical_or(destination == src.nodata, ~mask_boolean_local)
                masked_destination = np.ma.array(destination, mask=combined_mask)
                
                im = ax.imshow(masked_destination, extent=local_extent, cmap=cmap, vmin=global_min, vmax=global_max)
                
                mask_geometries_local = [shape(geom) for geom, val in shapes(mask_boolean_local.astype(np.uint8), transform=local_transform) if val > 0]
                for geom in mask_geometries_local:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color='red', linewidth=1.5, zorder=10)

                ax.set_title(dataset['title'], loc='left', fontsize=12, fontweight='bold')
                ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
                ax.set_aspect('equal')
    
    # --- Finalize the multi-subplot plot ---
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_min != np.inf and global_max != -np.inf and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.set_label('Depth (meters)', fontsize=20, rotation=270, labelpad=20)

    plt.suptitle("Raster Datasets by Year (Individual Extent)", fontsize=24)
    
    # Save the single figure
    individual_path = os.path.join(output_folder_individual, 'individual_extent_plots.png')
    plt.savefig(individual_path, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Individual extent plot saved to: {individual_path}")
    
def plot_consecutive_year_differences(input_folder, output_folder, mask_path):
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
    
    # First pass to determine common extent and resolution, including the mask file
    global_left = np.inf
    global_bottom = np.inf
    global_right = -np.inf
    global_top = -np.inf
    reference_res = None
    
    all_paths = [path for year in sorted_years for path in year_datasets[year]] + [mask_path]

    for path in all_paths:
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
    
    # Reproject the separate mask file and generate geometries once
    with rasterio.open(mask_path) as mask_src:
        mask_destination = np.zeros((dst_height, dst_width), dtype=mask_src.dtypes[0])
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=mask_destination,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=common_transform,
            dst_crs=target_crs,
            dst_nodata=mask_src.nodata,
            resampling=Resampling.nearest
        )
    
    mask_boolean = mask_destination != mask_src.nodata
    mask_geometries = [shape(geom) for geom, val in shapes(mask_boolean.astype(np.uint8), transform=common_transform) if val > 0]
    
    # Store reprojected data for all years
    reprojected_data = {}
    for year in sorted_years:
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
            # Combine the internal raster nodata mask with the external mask
            combined_raster_mask = np.logical_or(destination == src.nodata, ~mask_boolean)
            reprojected_data[year] = np.ma.array(destination, mask=combined_raster_mask)

    # Calculate differences and determine global min/max for differences
    diff_data = []
    global_diff_min = np.inf
    global_diff_max = -np.inf
    for i in range(len(sorted_years) - 1):
        year1 = sorted_years[i]
        year2 = sorted_years[i+1]
        
        data1 = reprojected_data[year1]
        data2 = reprojected_data[year2]
        
        # The mask is already handled when we created reprojected_data
        combined_diff_mask = np.ma.mask_or(data1.mask, data2.mask)
        
        diff = np.ma.array(data2 - data1, mask=combined_diff_mask)
        diff_data.append({'diff': diff, 'title': f'Difference: {year1} to {year2}'})
        
        if diff.min() < global_diff_min:
            global_diff_min = diff.min()
        if diff.max() > global_diff_max:
            global_diff_max = diff.max()

    # Create custom colormap and norm for difference plots
    cmap = plt.colormaps['ocean'].copy()
    cmap.set_bad(color='white')
    
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
        
        # Plot the mask outline on the current subplot
        for geom in mask_geometries:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.5, zorder=10)
        
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
mask_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\masks\training_mask_ER_3.tif"

plot_rasters_by_year(raster_folder, plot_output_folder, mask_path)
plot_consecutive_year_differences(raster_folder, plot_output_folder, mask_path)