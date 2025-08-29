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
import geopandas as gpd
from matplotlib.lines import Line2D

# This environment variable is important for handling large raster files
os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'

def getYearDatasets(input_folder):
    """Parses filenames to group raster datasets by year."""
    year_datasets = {}
    filename_pattern = re.compile(r'(.+)_(\d{4})_\d+_resampled.tif', re.IGNORECASE)

    for filename in os.listdir(input_folder):
        if filename.endswith('.tif'):
            match = filename_pattern.search(filename)
            if match:
                dataset_info, year = match.groups()
                
                if "USACE" in dataset_info and "NCMP" in dataset_info:
                    dataset_name = "USACE"
                else:
                    dataset_name = dataset_info.split('_')[-1]
                
                if year not in year_datasets:
                    year_datasets[year] = []
                year_datasets[year].append({
                    'path': os.path.join(input_folder, filename),
                    'dataset_name': dataset_name
                })
    return year_datasets

def calculateExtent(paths, target_crs):
    """Calculates a common extent for a list of raster files."""
    local_left, local_bottom, local_right, local_top = np.inf, np.inf, -np.inf, -np.inf
    reference_res = None

    for path in paths:
        with rasterio.open(path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds)
            
            left, top = transform.c, transform.f
            right = left + width * transform.a
            bottom = top + height * transform.e

            local_left = min(local_left, left)
            local_bottom = min(local_bottom, bottom)
            local_right = max(local_right, right)
            local_top = max(local_top, top)

            if reference_res is None:
                # Use the resolution of the first raster as the reference
                reference_res = (abs(transform.a), abs(transform.e))

    if reference_res is None:
        return None, None, None

    xres, yres = reference_res
    dst_width = int((local_right - local_left) / xres)
    dst_height = int((local_top - local_bottom) / yres)
    common_transform = from_bounds(local_left, local_bottom, local_right, local_top, dst_width, dst_height)
    common_extent = (local_left, local_right, local_bottom, local_top)
    
    return common_transform, common_extent, (dst_height, dst_width)

def reprojectToGrid(path, transform, shape, target_crs, resampling_method=Resampling.bilinear):
    """Reprojects a raster to a specified grid."""
    with rasterio.open(path) as src:
        destination = np.zeros(shape, dtype=src.dtypes[0])
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=target_crs,
            dst_nodata=src.nodata,
            resampling=resampling_method
        )
        return destination, src.nodata

def setupSubplot(ax, shp_gdf, extent):
    """Configures the appearance of a subplot."""
    shp_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, zorder=11)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect('equal')
    ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
    ax.set_xlabel('X distance (km)', fontsize=10)
    ax.set_ylabel('Y distance (km)', fontsize=10)
    
    xticks = ax.get_xticks()
    yticks = ax.get_yticks()
    ax.set_xticklabels([f'{val/1000:.1f}' for val in xticks], fontsize=8)
    ax.set_yticklabels([f'{val/1000:.1f}' for val in yticks], fontsize=8)

def finalizeFigure(fig, axes, im, cbar_label, suptitle, output_path, dpi):
    """Adds final touches to a figure and saves it."""
    if im is not None and im.get_array() is not None and im.get_array().count() > 0:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.set_label(cbar_label, fontsize=20, rotation=270, labelpad=20)

    legend_lines = [
        Line2D([0], [0], color='black', lw=0.5, label='Mask Outline'),
        Line2D([0], [0], color='gray', lw=0.5, label='50m Isobath')
    ]
    fig.legend(handles=legend_lines, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=2, fontsize=14)
    
    plt.suptitle(suptitle, fontsize=24)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, format='png', bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to: {output_path}")

def plot_rasters_by_year(input_folder, output_folder, mask_path, shp_path):
    """Plots individual raster datasets by year using a global extent."""
    year_datasets = getYearDatasets(input_folder)
    if not year_datasets:
        print("No raster files found.")
        return

    target_crs = 'EPSG:4326'
    shp_gdf = gpd.read_file(shp_path).to_crs(target_crs)
    
    all_raster_paths = [ds['path'] for year in year_datasets for ds in year_datasets[year]]
    common_transform, common_extent, common_shape = calculateExtent(all_raster_paths + [mask_path], target_crs)
    
    mask_reproj, mask_nodata = reprojectToGrid(mask_path, common_transform, common_shape, target_crs, Resampling.nearest)
    mask_boolean_global = mask_reproj != mask_nodata
    mask_geometries_global = [shape(geom) for geom, val in shapes(mask_boolean_global.astype(np.uint8), transform=common_transform) if val > 0]

    global_min, global_max = np.inf, -np.inf
    for path in all_raster_paths:
        reprojected_data, nodata = reprojectToGrid(path, common_transform, common_shape, target_crs)
        combined_mask = np.logical_or(reprojected_data == nodata, ~mask_boolean_global)
        masked_data = np.ma.array(reprojected_data, mask=combined_mask)
        if masked_data.count() > 0:
            global_min = min(global_min, masked_data.min())
            global_max = max(global_max, masked_data.max())

    cmap = plt.colormaps['ocean'].copy()
    cmap.set_bad(color='white')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'pink']
    sorted_years = sorted(year_datasets.keys())
    num_years = len(sorted_years)
    cols = math.ceil(math.sqrt(num_years))
    rows = math.ceil(num_years / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    im = None
    
    for i, year in enumerate(sorted_years):
        ax = axes[i]
        final_area_mask = np.zeros(common_shape, dtype=bool)
        legend_handles = [] # Create a list to hold legend handles

        for j, dataset in enumerate(year_datasets[year]):
            destination, nodata = reprojectToGrid(dataset['path'], common_transform, common_shape, target_crs)
            
            current_area_mask = destination != nodata
            final_area_mask = np.logical_or(final_area_mask, current_area_mask)
            
            data_geometries = [shape(geom) for geom, val in shapes(current_area_mask.astype(np.uint8), transform=common_transform) if val > 0]
            for geom in data_geometries:
                x, y = geom.exterior.xy
                ax.plot(x, y, color=colors[j % len(colors)], linewidth=0.5, zorder=12)
            
            # Add a custom line to our list of legend handles
            legend_handles.append(Line2D([0], [0], color=colors[j % len(colors)], lw=2, label=dataset['dataset_name']))
            
            display_mask = np.logical_or(destination == nodata, ~mask_boolean_global)
            masked_destination = np.ma.array(destination, mask=display_mask)
            im = ax.imshow(masked_destination, extent=common_extent, cmap=cmap, vmin=global_min, vmax=global_max)

        for geom in mask_geometries_global:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.5, zorder=10)
        
        setupSubplot(ax, shp_gdf, common_extent)
        
        # --- MODIFIED SECTION START ---
        # Calculate area
        valid_pixels = np.sum(final_area_mask)
        pixel_area_m2 = abs(common_transform.a * common_transform.e)
        total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
        
        # Set a single title with the year and total area
        ax.set_title(f"{year}\nTotal Area: {total_area_km2:.2f} km$^2$", fontsize=10)
        
        # Add the legend underneath the subplot
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), 
                  fancybox=True, shadow=False, ncol=2, fontsize=8)
        # --- MODIFIED SECTION END ---

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    finalizeFigure(fig, axes, im, 'Depth (meters)', "Raster Datasets by Year (Global Extent)", os.path.join(output_folder, 'year_plots_global_extent.png'), 1200)

def plot_rasters_by_year_individual(input_folder, output_folder, mask_path, shp_path):
    """Plots individual raster datasets by year using individual extents."""
    year_datasets = getYearDatasets(input_folder)
    if not year_datasets:
        print("No raster files found.")
        return

    target_crs = 'EPSG:4326'
    shp_gdf = gpd.read_file(shp_path).to_crs(target_crs)
    
    all_raster_paths = [ds['path'] for year in year_datasets for ds in year_datasets[year]]
    global_transform, _, global_shape = calculateExtent(all_raster_paths + [mask_path], target_crs)
    mask_reproj_global, mask_nodata_global = reprojectToGrid(mask_path, global_transform, global_shape, target_crs, Resampling.nearest)
    mask_boolean_global_scope = mask_reproj_global != mask_nodata_global
    
    global_min, global_max = np.inf, -np.inf
    for path in all_raster_paths:
        reprojected_data, nodata = reprojectToGrid(path, global_transform, global_shape, target_crs)
        combined_mask = np.logical_or(reprojected_data == nodata, ~mask_boolean_global_scope)
        masked_data = np.ma.array(reprojected_data, mask=combined_mask)
        if masked_data.count() > 0:
            global_min = min(global_min, masked_data.min())
            global_max = max(global_max, masked_data.max())

    cmap = plt.colormaps['ocean'].copy()
    cmap.set_bad(color='white')
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'cyan', 'magenta', 'yellow', 'pink']
    sorted_years = sorted(year_datasets.keys())
    num_years = len(sorted_years)
    cols = math.ceil(math.sqrt(num_years))
    rows = math.ceil(num_years / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True, sharex=False, sharey=False)
    axes = np.atleast_1d(axes).flatten()
    im = None

    for i, year in enumerate(sorted_years):
        ax = axes[i]
        year_paths = [ds['path'] for ds in year_datasets[year]]
        local_transform, local_extent, local_shape = calculateExtent(year_paths + [mask_path], target_crs)
        
        mask_reproj_local, mask_nodata_local = reprojectToGrid(mask_path, local_transform, local_shape, target_crs, Resampling.nearest)
        mask_boolean_local = mask_reproj_local != mask_nodata_local
        mask_geometries_local = [shape(geom) for geom, val in shapes(mask_boolean_local.astype(np.uint8), transform=local_transform) if val > 0]
        
        final_area_mask = np.zeros(local_shape, dtype=bool)
        legend_handles = [] # Create a list to hold legend handles

        for j, dataset in enumerate(year_datasets[year]):
            destination, nodata = reprojectToGrid(dataset['path'], local_transform, local_shape, target_crs)
            
            current_area_mask = destination != nodata
            final_area_mask = np.logical_or(final_area_mask, current_area_mask)
            
            data_geometries = [shape(geom) for geom, val in shapes(current_area_mask.astype(np.uint8), transform=local_transform) if val > 0]
            for geom in data_geometries:
                x, y = geom.exterior.xy
                ax.plot(x, y, color=colors[j % len(colors)], linewidth=0.5, zorder=12)

            # Add a custom line to our list of legend handles
            legend_handles.append(Line2D([0], [0], color=colors[j % len(colors)], lw=2, label=dataset['dataset_name']))
            
            display_mask = np.logical_or(destination == nodata, ~mask_boolean_local)
            masked_destination = np.ma.array(destination, mask=display_mask)
            im = ax.imshow(masked_destination, extent=local_extent, cmap=cmap, vmin=global_min, vmax=global_max)

        for geom in mask_geometries_local:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.5, zorder=10)
        
        setupSubplot(ax, shp_gdf, local_extent)
        
        # --- MODIFIED SECTION START ---
        # Calculate area
        valid_pixels = np.sum(final_area_mask)
        pixel_area_m2 = abs(local_transform.a * local_transform.e)
        total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
        
        # Set a single title with the year and total area
        ax.set_title(f"{year}\nTotal Area: {total_area_km2:.2f} km$^2$", fontsize=10)
        
        # Add the legend underneath the subplot
        ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), 
                  fancybox=True, shadow=False, ncol=2, fontsize=8)
        # --- MODIFIED SECTION END ---

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    finalizeFigure(fig, axes, im, 'Depth (meters)', "Raster Datasets by Year (Individual Extent)", os.path.join(output_folder, 'year_plots_individual_extent.png'), 600)

def plot_difference(input_folder, output_folder, mask_path, shp_path, mode, use_individual_extent=False):
    """Calculates and plots raster differences for consecutive or all year pairs."""
    year_datasets_map = getYearDatasets(input_folder)
    if not year_datasets_map or len(year_datasets_map) < 2:
        print("Not enough years of data to calculate differences.")
        return
    
    year_datasets = {year: data[0]['path'] for year, data in year_datasets_map.items()}

    target_crs = 'EPSG:4326'
    shp_gdf = gpd.read_file(shp_path).to_crs(target_crs)
    sorted_years = sorted(year_datasets.keys())
    
    # Global min/max calculation for consistent color scale across all difference plots
    all_paths = list(year_datasets.values()) + [mask_path]
    global_transform, global_extent, global_shape = calculateExtent(all_paths, target_crs)
    mask_reproj_global, mask_nodata_global = reprojectToGrid(mask_path, global_transform, global_shape, target_crs, Resampling.nearest)
    mask_boolean_global = mask_reproj_global != mask_nodata_global
    
    reprojected_global = {}
    for year in sorted_years:
        data, nodata = reprojectToGrid(year_datasets[year], global_transform, global_shape, target_crs)
        reprojected_global[year] = np.ma.array(data, mask=np.logical_or(data == nodata, ~mask_boolean_global))

    global_diff_min, global_diff_max = np.inf, -np.inf
    year_pairs_for_minmax = list(combinations(sorted_years, 2))
    for year1, year2 in year_pairs_for_minmax:
        diff = reprojected_global[year2] - reprojected_global[year1]
        if diff.count() > 0:
            global_diff_min = min(global_diff_min, diff.min())
            global_diff_max = max(global_diff_max, diff.max())

    year_pairs = list(combinations(sorted_years, 2)) if mode == 'all' else list(zip(sorted_years, sorted_years[1:]))
    
    diff_data = []
    for year1, year2 in year_pairs:
        path1, path2 = year_datasets[year1], year_datasets[year2]
        
        # Use globally reprojected data for overlap calculation to be consistent
        data1_global, data2_global = reprojected_global[year1], reprojected_global[year2]
        combined_mask_global = np.ma.mask_or(data1_global.mask, data2_global.mask)
        overlapping_pixels = np.sum(~combined_mask_global)
        smaller_year_pixels = min(np.sum(~data1_global.mask), np.sum(~data2_global.mask))
        overlap_percentage = (overlapping_pixels / smaller_year_pixels * 100) if smaller_year_pixels > 0 else 0
        
        if mode == 'all' and overlap_percentage < 5:
            continue
        
        pixel_area_m2 = abs(global_transform.a * global_transform.e)
        overlap_area_km2 = (overlapping_pixels * pixel_area_m2) / 1e6
        overlap_text = f"Overlap: {overlap_percentage:.2f}% ({overlap_area_km2:.2f} km$^2$)"
        title = f'Difference: {year1} to {year2}\n{overlap_text}'
        
        diff_data.append({'title': title, 'path1': path1, 'path2': path2, 'year1': year1, 'year2': year2})
        
    if not diff_data:
        print(f"No year pairs found for '{mode}' mode with sufficient overlap.")
        return

    num_diffs = len(diff_data)
    cols = math.ceil(math.sqrt(num_diffs))
    rows = math.ceil(num_diffs / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True, sharex=not use_individual_extent, sharey=not use_individual_extent)
    axes = np.atleast_1d(axes).flatten()
    im_diff = None
    cmap = plt.colormaps['ocean'].copy()
    cmap.set_bad(color='white')

    for i, data_dict in enumerate(diff_data):
        ax = axes[i]
        
        if use_individual_extent:
            local_transform, local_extent, local_shape = calculateExtent([data_dict['path1'], data_dict['path2'], mask_path], target_crs)
            
            data1, nodata1 = reprojectToGrid(data_dict['path1'], local_transform, local_shape, target_crs)
            data2, nodata2 = reprojectToGrid(data_dict['path2'], local_transform, local_shape, target_crs)
            mask_reproj, mask_nodata = reprojectToGrid(mask_path, local_transform, local_shape, target_crs, Resampling.nearest)
            
            mask_boolean = mask_reproj != mask_nodata
            combined_mask = np.logical_or.reduce((data1 == nodata1, data2 == nodata2, ~mask_boolean))
            diff = np.ma.array(data2 - data1, mask=combined_mask)
            
            im_diff = ax.imshow(diff, extent=local_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
            mask_geometries = [shape(geom) for geom, val in shapes(mask_boolean.astype(np.uint8), transform=local_transform) if val > 0]
            setupSubplot(ax, shp_gdf, local_extent)
        else:
            diff = reprojected_global[data_dict['year2']] - reprojected_global[data_dict['year1']]
            im_diff = ax.imshow(diff, extent=global_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
            mask_boolean = mask_reproj_global != mask_nodata_global
            mask_geometries = [shape(geom) for geom, val in shapes(mask_boolean.astype(np.uint8), transform=global_transform) if val > 0]
            setupSubplot(ax, shp_gdf, global_extent)

        ax.set_title(data_dict['title'], loc='left', fontsize=12, fontweight='bold')
        for geom in mask_geometries:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='black', linewidth=0.5, zorder=10)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    extent_type = "Individual" if use_individual_extent else "Global"
    suptitle = f"{mode.capitalize().replace('_', ' ')} Year Differences ({extent_type} Extent)"
    if mode == 'all':
        suptitle = f"All Year Differences >= 5% Overlap ({extent_type} Extent)"
    
    output_filename = f'{mode}_year_differences_{extent_type.lower()}_extent.png'
    dpi = 600 if use_individual_extent else 1200
    finalizeFigure(fig, axes, im_diff, 'Difference (m)', suptitle, os.path.join(output_folder, output_filename), dpi)

def plot_consecutive_year_differences(input_folder, output_folder, mask_path, shp_path):
    plot_difference(input_folder, output_folder, mask_path, shp_path, 'consecutive', use_individual_extent=False)
    plot_difference(input_folder, output_folder, mask_path, shp_path, 'consecutive', use_individual_extent=True)

def plot_all_year_differences(input_folder, output_folder, mask_path, shp_path):
    plot_difference(input_folder, output_folder, mask_path, shp_path, 'all', use_individual_extent=False)
    plot_difference(input_folder, output_folder, mask_path, shp_path, 'all', use_individual_extent=True)

# Set input/output folders
raster_folder = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast_resampled"
plot_output_folder = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast_plots"
mask_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\masks\training_mask_ER_3.tif"
shp_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\coastal_boundary_dataset\50m_isobath_polygon\50m_isobath_polygon.shp"

# Call the functions
plot_rasters_by_year(raster_folder, plot_output_folder, mask_path, shp_path)
plot_rasters_by_year_individual(raster_folder, plot_output_folder, mask_path, shp_path)
plot_consecutive_year_differences(raster_folder, plot_output_folder, mask_path, shp_path)
plot_all_year_differences(raster_folder, plot_output_folder, mask_path, shp_path)