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
from itertools import combinations

# This environment variable is important for handling large raster files
os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'

def plot_rasters_by_year(input_folder, output_folder, mask_path):
    """
    Plots individual raster datasets by year in two ways:
    1. A single figure with subplots all sharing a common, global extent.
    2. A single figure with subplots each using an individual extent.

    Args:
        input_folder (str): Path to the folder containing the raster files.
        output_folder (str): Path to the folder to save the plots.
        mask_path (str): Path to the GeoTIFF file used as a mask.
    """
    year_datasets = {}
    filename_pattern = re.compile(r'(.+)_(\d{4})_resampled\.tif', re.IGNORECASE)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            match = filename_pattern.search(filename)
            if match:
                dataset_name = match.group(1).split('_')[-1]
                year = match.group(2)
                
                if year not in year_datasets:
                    year_datasets[year] = []
                year_datasets[year].append({'path': os.path.join(input_folder, filename), 'dataset_name': dataset_name})

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
    
    # Create a flat list of all datasets for determining global extent and min/max
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
        
        # Build the title with all dataset names for the year
        dataset_titles = [f"{year} {d['dataset_name']}" for d in year_datasets[year]]
        
        # Get the path to the raster file for the current year
        year_path = year_datasets[year][0]['path']
        
        with rasterio.open(year_path) as src:
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
            
            # Calculate and add total area to the title
            valid_pixels = np.sum(~combined_mask)
            pixel_area_m2 = abs(common_transform.a * common_transform.e)
            total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
            title_text = "\n".join(dataset_titles) + f"\nTotal Area: {total_area_km2:.2f} km$^2$"

        for geom in mask_geometries_global:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='red', linewidth=0.5, zorder=10)

        ax.set_title(title_text, loc='left', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        
        # Add x and y axis labels in kilometers
        ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
        ax.set_xlabel('X distance (km)', fontsize=10)
        ax.set_ylabel('Y distance (km)', fontsize=10)
        
        # Convert tick labels to kilometers
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xtick_labels = [f'{val/1000:.1f}' for val in xticks]
        ytick_labels = [f'{val/1000:.1f}' for val in yticks]
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_yticklabels(ytick_labels, fontsize=8)


    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_min != np.inf and global_max != -np.inf and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.set_label('Depth (meters)', fontsize=20, rotation=270, labelpad=20)

    plt.suptitle("Raster Datasets by Year (Global Extent)", fontsize=24)

    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'year_plots_global_extent.png'
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Global extent plot saved to: {output_path}")

    # --- Part 3: Plotting with Individual Extents (Single Figure) ---

    # Get sorted years and calculate grid size
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
            
            # Build the title with all dataset names for the year
            dataset_titles = [f"{year} {d['dataset_name']}" for d in year_datasets[year]]
            
            # Plot each dataset for the current year on the same subplot
            for dataset in year_datasets[year]:
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
                    
                    # Calculate and add total area to the title
                    valid_pixels = np.sum(~combined_mask)
                    pixel_area_m2 = abs(local_transform.a * local_transform.e)
                    total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
                    title_text = "\n".join(dataset_titles) + f"\nTotal Area: {total_area_km2:.2f} km$^2$"
                    
                    mask_geometries_local = [shape(geom) for geom, val in shapes(mask_boolean_local.astype(np.uint8), transform=local_transform) if val > 0]
                    for geom in mask_geometries_local:
                        x, y = geom.exterior.xy
                        ax.plot(x, y, color='red', linewidth=1.5, zorder=10)

            ax.set_title(title_text, loc='left', fontsize=12, fontweight='bold')
            ax.set_aspect('equal')

            # Add x and y axis labels in kilometers
            ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
            ax.set_xlabel('X distance (km)', fontsize=10)
            ax.set_ylabel('Y distance (km)', fontsize=10)

            # Convert tick labels to kilometers
            xticks = ax.get_xticks()
            yticks = ax.get_yticks()
            xtick_labels = [f'{val/1000:.1f}' for val in xticks]
            ytick_labels = [f'{val/1000:.1f}' for val in yticks]
            ax.set_xticklabels(xtick_labels, fontsize=8)
            ax.set_yticklabels(ytick_labels, fontsize=8)
    
    # --- Finalize the multi-subplot plot ---
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_min != np.inf and global_max != -np.inf and im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.set_label('Depth (meters)', fontsize=20, rotation=270, labelpad=20)

    plt.suptitle("Raster Datasets by Year (Individual Extent)", fontsize=24)
    
    individual_path = os.path.join(output_folder, 'year_plots_individual_extents.png')
    plt.savefig(individual_path, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Individual extent plot saved to: {individual_path}")
    
def plot_consecutive_year_differences(input_folder, output_folder, mask_path):
    """
    Calculates and plots the difference between consecutive years of raster data,
    including the percentage and area of data overlap in the plot titles.
    
    Args:
        input_folder (str): Path to the folder containing the raster files.
        output_folder (str): Path to the folder to save the plots.
        mask_path (str): Path to the GeoTIFF file used as a mask.
    """
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
    
    # Store reprojected data and pixel counts for all years
    reprojected_data = {}
    valid_pixel_counts = {}
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
            # Count the number of valid pixels for this year
            valid_pixel_counts[year] = np.sum(~combined_raster_mask)

    # Calculate differences, overlap percentages, and determine global min/max
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
        
        # Calculate overlap percentage and area
        overlapping_pixels = np.sum(~combined_diff_mask)
        total_pixels_year1 = valid_pixel_counts[year1]
        total_pixels_year2 = valid_pixel_counts[year2]
        
        smaller_year_pixels = min(total_pixels_year1, total_pixels_year2)
        if smaller_year_pixels > 0:
            overlap_percentage = (overlapping_pixels / smaller_year_pixels) * 100
            pixel_area_m2 = abs(common_transform.a * common_transform.e)
            overlap_area_km2 = (overlapping_pixels * pixel_area_m2) / 1e6
            overlap_text = f"Overlap: {overlap_percentage:.2f}% ({overlap_area_km2:.2f} km$^2$)"
        else:
            overlap_text = "No overlap"

        diff = np.ma.array(data2 - data1, mask=combined_diff_mask)
        title = f'Difference: {year1} to {year2}\n{overlap_text}'
        diff_data.append({'diff': diff, 'title': title})
        
        if diff.min() < global_diff_min:
            global_diff_min = diff.min()
        if diff.max() > global_diff_max:
            global_diff_max = diff.max()

    # Create custom colormap for difference plots
    cmap = plt.colormaps['ocean'].copy()
    cmap.set_bad(color='white')
    
    # --- Part 1: Plotting differences with Global Extent ---
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
            ax.plot(x, y, color='red', linewidth=0.5, zorder=10)
        
        ax.set_aspect('equal')
        
        # Add x and y axis labels in kilometers
        ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
        ax.set_xlabel('X distance (km)', fontsize=10)
        ax.set_ylabel('Y distance (km)', fontsize=10)
        
        # Convert tick labels to kilometers
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xtick_labels = [f'{val/1000:.1f}' for val in xticks]
        ytick_labels = [f'{val/1000:.1f}' for val in yticks]
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_yticklabels(ytick_labels, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_diff_min != np.inf and global_diff_max != -np.inf and im_diff is not None:
        cbar = fig.colorbar(im_diff, ax=axes.ravel().tolist())
        cbar.set_label('Difference (m)', fontsize=20, rotation=270, labelpad=20)
    
    plt.suptitle("Consecutive Year Differences (Global Extent)", fontsize=24)
    
    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'consecutive_year_differences_global_extent.png'
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Global extent difference plot saved to: {output_path}")

    # --- Part 2: Plotting differences with Individual Extents ---

    num_diffs = len(diff_data)
    cols = math.ceil(math.sqrt(num_diffs))
    rows = math.ceil(num_diffs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True, sharex=False, sharey=False)

    if num_diffs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    im_diff = None
    
    with rasterio.open(mask_path) as mask_src:
        for i, diff_dict in enumerate(diff_data):
            ax = axes[i]
            year1 = sorted_years[i]
            year2 = sorted_years[i+1]
            title = diff_dict['title']
            
            path1 = year_datasets[year1][0]
            path2 = year_datasets[year2][0]

            with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
                # Determine local extent for the current difference pair
                local_left = min(src1.bounds.left, src2.bounds.left)
                local_bottom = min(src1.bounds.bottom, src2.bounds.bottom)
                local_right = max(src1.bounds.right, src2.bounds.right)
                local_top = max(src1.bounds.top, src2.bounds.top)

                local_transform, local_width, local_height = calculate_default_transform(
                    src1.crs, target_crs, src1.width, src1.height, local_left, local_bottom, local_right, local_top)
                
                local_extent = (local_transform.c, local_transform.c + local_width * local_transform.a, 
                                local_transform.f + local_height * local_transform.e, local_transform.f)

                # Reproject both rasters to this local extent
                destination1 = np.zeros((local_height, local_width), dtype=src1.dtypes[0])
                reproject(source=rasterio.band(src1, 1), destination=destination1, src_transform=src1.transform, src_crs=src1.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=src1.nodata, resampling=Resampling.bilinear)
                
                destination2 = np.zeros((local_height, local_width), dtype=src2.dtypes[0])
                reproject(source=rasterio.band(src2, 1), destination=destination2, src_transform=src2.transform, src_crs=src2.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=src2.nodata, resampling=Resampling.bilinear)

                # Reproject the mask to this local extent
                mask_destination_local = np.zeros((local_height, local_width), dtype=mask_src.dtypes[0])
                reproject(source=rasterio.band(mask_src, 1), destination=mask_destination_local, src_transform=mask_src.transform, src_crs=mask_src.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=mask_src.nodata, resampling=Resampling.nearest)
                
                mask_boolean_local = mask_destination_local != mask_src.nodata

                # Calculate the difference and apply the combined mask
                combined_mask_local = np.logical_or(destination1 == src1.nodata, destination2 == src2.nodata)
                combined_mask_local = np.logical_or(combined_mask_local, ~mask_boolean_local)

                diff_local = np.ma.array(destination2 - destination1, mask=combined_mask_local)
                
                im_diff = ax.imshow(diff_local, extent=local_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
                
                mask_geometries_local = [shape(geom) for geom, val in shapes(mask_boolean_local.astype(np.uint8), transform=local_transform) if val > 0]
                for geom in mask_geometries_local:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color='red', linewidth=1.5, zorder=10)

                ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                
                # Add x and y axis labels in kilometers
                ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
                ax.set_xlabel('X distance (km)', fontsize=10)
                ax.set_ylabel('Y distance (km)', fontsize=10)

                # Convert tick labels to kilometers
                xticks = ax.get_xticks()
                yticks = ax.get_yticks()
                xtick_labels = [f'{val/1000:.1f}' for val in xticks]
                ytick_labels = [f'{val/1000:.1f}' for val in yticks]
                ax.set_xticklabels(xtick_labels, fontsize=8)
                ax.set_yticklabels(ytick_labels, fontsize=8)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_diff_min != np.inf and global_diff_max != -np.inf and im_diff is not None:
        cbar = fig.colorbar(im_diff, ax=axes.ravel().tolist())
        cbar.set_label('Difference (m)', fontsize=20, rotation=270, labelpad=20)
    
    plt.suptitle("Consecutive Year Differences (Individual Extent)", fontsize=24)
    
    output_filename_individual = 'consecutive_year_differences_individual_extents.png'
    output_path_individual = os.path.join(output_folder, output_filename_individual)
    plt.savefig(output_path_individual, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Individual extent difference plot saved to: {output_path_individual}")

def plot_all_year_differences(input_folder, output_folder, mask_path):
    """
    Calculates and plots the difference between all possible year pairs of raster data,
    only plotting pairs with >= 5% data overlap.

    Args:
        input_folder (str): Path to the folder containing the raster files.
        output_folder (str): Path to the folder to save the plots.
        mask_path (str): Path to the GeoTIFF file used as a mask.
    """
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
    
    # --- Part 1: Determine Global Extent and Reproject all data once ---
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
    
    # Store reprojected data and pixel counts for all years
    reprojected_data = {}
    valid_pixel_counts = {}
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
            # Count the number of valid pixels for this year
            valid_pixel_counts[year] = np.sum(~combined_raster_mask)

    # --- Part 2: Calculate differences for all year pairs and filter by overlap ---
    diff_data_filtered = []
    global_diff_min = np.inf
    global_diff_max = -np.inf
    
    # Use itertools.combinations to get all unique pairs of years
    for year1, year2 in combinations(sorted_years, 2):
        data1 = reprojected_data[year1]
        data2 = reprojected_data[year2]
        
        combined_diff_mask = np.ma.mask_or(data1.mask, data2.mask)
        overlapping_pixels = np.sum(~combined_diff_mask)
        
        total_pixels_year1 = valid_pixel_counts[year1]
        total_pixels_year2 = valid_pixel_counts[year2]
        
        smaller_year_pixels = min(total_pixels_year1, total_pixels_year2)
        
        if smaller_year_pixels > 0:
            overlap_percentage = (overlapping_pixels / smaller_year_pixels) * 100
        else:
            overlap_percentage = 0
            
        # Only proceed if overlap is 5% or more
        if overlap_percentage >= 5:
            diff = np.ma.array(data2 - data1, mask=combined_diff_mask)
            pixel_area_m2 = abs(common_transform.a * common_transform.e)
            overlap_area_km2 = (overlapping_pixels * pixel_area_m2) / 1e6
            title = f'Difference: {year1} to {year2}\nOverlap: {overlap_percentage:.2f}% ({overlap_area_km2:.2f} km$^2$)'
            diff_data_filtered.append({'diff': diff, 'title': title, 'year1': year1, 'year2': year2})
            
            if diff.count() > 0: # Ensure there are valid pixels to get min/max from
                if diff.min() < global_diff_min:
                    global_diff_min = diff.min()
                if diff.max() > global_diff_max:
                    global_diff_max = diff.max()
    
    if not diff_data_filtered:
        print("No year pairs found with an overlap of 5% or more.")
        return

    # Create a custom colormap for the difference plots
    cmap = plt.colormaps['ocean'].copy() # Changed from 'coolwarm' to 'ocean'
    cmap.set_bad(color='white')
    
    # --- Part 3: Plotting all year differences with Global Extent ---
    num_diffs = len(diff_data_filtered)
    cols = math.ceil(math.sqrt(num_diffs))
    rows = math.ceil(num_diffs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
    if num_diffs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
        
    im_diff = None
    for i, diff_dict in enumerate(diff_data_filtered):
        ax = axes[i]
        diff_array = diff_dict['diff']
        title = diff_dict['title']
        
        im_diff = ax.imshow(diff_array, extent=common_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
        ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
        
        for geom in mask_geometries:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='red', linewidth=0.5, zorder=10)
        
        ax.set_aspect('equal')
        
        # Add x and y axis labels in kilometers
        ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
        ax.set_xlabel('X distance (km)', fontsize=10)
        ax.set_ylabel('Y distance (km)', fontsize=10)
        
        # Convert tick labels to kilometers
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        xtick_labels = [f'{val/1000:.1f}' for val in xticks]
        ytick_labels = [f'{val/1000:.1f}' for val in yticks]
        ax.set_xticklabels(xtick_labels, fontsize=8)
        ax.set_yticklabels(ytick_labels, fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_diff_min != np.inf and global_diff_max != -np.inf and im_diff is not None:
        cbar = fig.colorbar(im_diff, ax=axes.ravel().tolist())
        cbar.set_label('Difference (m)', fontsize=20, rotation=270, labelpad=20)
    
    plt.suptitle("All Year Differences with >= 5% Overlap (Global Extent)", fontsize=24)
    
    os.makedirs(output_folder, exist_ok=True)
    output_filename = 'all_year_differences_global_extent.png'
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path, dpi=1200, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Global extent plot saved to: {output_path}")

    # --- Part 4: Plotting all year differences with Individual Extents ---
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True, sharex=False, sharey=False)
    if num_diffs == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    im_diff = None
    
    with rasterio.open(mask_path) as mask_src:
        for i, diff_dict in enumerate(diff_data_filtered):
            ax = axes[i]
            year1 = diff_dict['year1']
            year2 = diff_dict['year2']
            title = diff_dict['title']
            
            path1 = year_datasets[year1][0]
            path2 = year_datasets[year2][0]

            with rasterio.open(path1) as src1, rasterio.open(path2) as src2:
                local_left = min(src1.bounds.left, src2.bounds.left)
                local_bottom = min(src1.bounds.bottom, src2.bounds.bottom)
                local_right = max(src1.bounds.right, src2.bounds.right)
                local_top = max(src1.bounds.top, src2.bounds.top)
                
                local_transform, local_width, local_height = calculate_default_transform(
                    src1.crs, target_crs, src1.width, src1.height, local_left, local_bottom, local_right, local_top)
                
                local_extent = (local_transform.c, local_transform.c + local_width * local_transform.a, 
                                local_transform.f + local_height * local_transform.e, local_transform.f)

                destination1 = np.zeros((local_height, local_width), dtype=src1.dtypes[0])
                reproject(source=rasterio.band(src1, 1), destination=destination1, src_transform=src1.transform, src_crs=src1.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=src1.nodata, resampling=Resampling.bilinear)
                
                destination2 = np.zeros((local_height, local_width), dtype=src2.dtypes[0])
                reproject(source=rasterio.band(src2, 1), destination=destination2, src_transform=src2.transform, src_crs=src2.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=src2.nodata, resampling=Resampling.bilinear)

                mask_destination_local = np.zeros((local_height, local_width), dtype=mask_src.dtypes[0])
                reproject(source=rasterio.band(mask_src, 1), destination=mask_destination_local, src_transform=mask_src.transform, src_crs=mask_src.crs, dst_transform=local_transform, dst_crs=target_crs, dst_nodata=mask_src.nodata, resampling=Resampling.nearest)
                
                mask_boolean_local = mask_destination_local != mask_src.nodata

                combined_mask_local = np.logical_or(destination1 == src1.nodata, destination2 == src2.nodata)
                combined_mask_local = np.logical_or(combined_mask_local, ~mask_boolean_local)

                diff_local = np.ma.array(destination2 - destination1, mask=combined_mask_local)
                
                im_diff = ax.imshow(diff_local, extent=local_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
                
                mask_geometries_local = [shape(geom) for geom, val in shapes(mask_boolean_local.astype(np.uint8), transform=local_transform) if val > 0]
                for geom in mask_geometries_local:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color='red', linewidth=1.5, zorder=10)

                ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                
                ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
                ax.set_xlabel('X distance (km)', fontsize=10)
                ax.set_ylabel('Y distance (km)', fontsize=10)

                xticks = ax.get_xticks()
                yticks = ax.get_yticks()
                xtick_labels = [f'{val/1000:.1f}' for val in xticks]
                ytick_labels = [f'{val/1000:.1f}' for val in yticks]
                ax.set_xticklabels(xtick_labels, fontsize=8)
                ax.set_yticklabels(ytick_labels, fontsize=8)
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    if global_diff_min != np.inf and global_diff_max != -np.inf and im_diff is not None:
        cbar = fig.colorbar(im_diff, ax=axes.ravel().tolist())
        cbar.set_label('Difference (m)', fontsize=20, rotation=270, labelpad=20)
    
    plt.suptitle("All Year Differences with >= 5% Overlap (Individual Extent)", fontsize=24)
    
    output_filename_individual = 'all_year_differences_individual_extents.png'
    output_path_individual = os.path.join(output_folder, output_filename_individual)
    plt.savefig(output_path_individual, dpi=600, bbox_inches='tight', format='png')
    plt.close(fig)
    print(f"Individual extent difference plot saved to: {output_path_individual}")

# Set input/output folders
raster_folder = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast_resampled"
plot_output_folder = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast_plots"
mask_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\masks\training_mask_ER_3.tif"

plot_rasters_by_year(raster_folder, plot_output_folder, mask_path)
plot_consecutive_year_differences(raster_folder, plot_output_folder, mask_path)
# Call the new function to plot all year differences
plot_all_year_differences(raster_folder, plot_output_folder, mask_path)
