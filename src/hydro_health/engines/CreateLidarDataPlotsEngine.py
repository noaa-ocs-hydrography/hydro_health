import os
import re
import math
import numpy as np
import geopandas as gpd

from shapely.geometry import shape
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from itertools import combinations

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject

import dask
from dask.distributed import Client, print

import glob

from osgeo import gdal

from hydro_health.helpers.tools import get_config_item

os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'

class CreateLidarDataPlotsEngine():
    """Class to hold the logic for processing the Lidar Data Plots"""

    def __init__(self):
        super().__init__()

    def calculateExtent(self, paths, target_crs) -> tuple:
        """
        Calculates a common extent for a list of raster files.
        param list paths: List of paths to raster files.
        param str target_crs: Target coordinate reference system (e.g., 'EPSG:3857').
        return: Affine transform, extent (left, right, bottom, top), and shape (height, width) of the common grid.
        """

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

    def extract_date_from_metadata(self, metadata_path) -> list:
        """
        Extracts the first date in YYYY-MM format from a metadata file.
        param str metadata_path: Path to the metadata file.
        return list: Date(s) found in the metadata file.
        """

        try:
            with open(metadata_path, 'r') as f:
                content = f.read()
                date_matches_ym = re.findall(r'(?:19|20)\d{2}-\d{2}', content)
                if date_matches_ym:
                    return sorted(list(set(date_matches_ym)))

                date_matches_y = re.findall(r'(?:19|20)\d{2}', content)
                if date_matches_y:
                    return sorted(list(set(date_matches_y)))

                return 'Missing'
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: Error reading metadata file {metadata_path}: {e}")
        return None  

    def finalizeFigure(self, fig, axes, im, cbar_label, suptitle, output_path, dpi) -> None:
        """
        Adds final touches to a figure and saves it.
        param object fig: Matplotlib figure object.
        param object axes: Matplotlib axes object.
        param object im: Matplotlib image object.
        param str cbar_label: Label for the colorbar.
        param str suptitle: Super title for the figure.
        param str output_path: Path to save the figure.
        param int dpi: DPI for saving the figure.
        return: None
        """

        if im is not None and im.get_array() is not None and im.get_array().count() > 0:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist())
            cbar.set_label(cbar_label, fontsize=14, rotation=270, labelpad=20)

        # Added a legend for the new 'Overlap Area' outline
        legend_lines = [
            Line2D([0], [0], color='black', lw=0.5, label='Mask Outline'),
            Line2D([0], [0], color='gray', lw=0.5, label='50m Isobath'),
            Line2D([0], [0], color='red', lw=1.5, label='Overlap Area') # New legend entry
        ]
        fig.legend(handles=legend_lines, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=1, fontsize=8)
        
        plt.suptitle(suptitle, fontsize=14)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=dpi, format='png', bbox_inches='tight')
        plt.close(fig)
        print(f"Plot saved to: {output_path}")

    def getYearDatasets(self, input_folder) -> dict:
        """
        Parses filenames to group raster datasets by year.
        If year is not in filename, it searches metadata for a date to use for grouping.
        param str input_folder: Folder containing the input raster files.
        return dict: Dictionary with years as keys and lists of dataset info as values.
        """

        year_datasets = {}
        filename_pattern = re.compile(r'(.+?)(?:_(\d{4}))?_(\d+)_resampled\.tif', re.IGNORECASE)
        
        base_folder = os.path.dirname(input_folder)
        base_folder = os.path.join(base_folder, 'DigitalCoast')

        for filename in os.listdir(input_folder):
            if filename.endswith('.tif'):
                match = filename_pattern.search(filename)
                if match:
                    dataset_info, year_from_filename, unique_id = match.groups()
                    
                    if "USACE" in dataset_info and "NCMP" in dataset_info:
                        dataset_name = "USACE"
                    else:
                        dataset_name = dataset_info.split('_')[-1]
                    
                    acquisition_date = None
                    try:
                        for folder_name in os.listdir(base_folder):
                            potential_path = os.path.join(base_folder, folder_name)
                            if os.path.isdir(potential_path) and f'_{unique_id}' in folder_name:
                                metadata_path = os.path.join(potential_path, 'metadata.txt')
                                acquisition_date = self.extract_date_from_metadata(metadata_path)
                                if acquisition_date:
                                    break
                    except FileNotFoundError:
                        # This warning is helpful if the base 'DigitalCoast' folder is missing
                        print(f"Warning: Base folder '{base_folder}' not found. Cannot search for metadata.")
                    
                    # Determine the year for grouping data and the string for the plot title
                    grouping_year = year_from_filename
                    title_str = year_from_filename

                    if not grouping_year:
                        grouping_year = 'No Year Found'

                    # Set a specific title for plots where year was not in the filename
                    if not title_str:
                        title_str = 'Date Not in Filename'

                    if grouping_year not in year_datasets:
                        year_datasets[grouping_year] = []
                    
                    year_datasets[grouping_year].append({
                        'path': os.path.join(input_folder, filename),
                        'dataset_name': dataset_name,
                        'date': acquisition_date,  # For the legend (e.g., '2019-05')
                        'title': title_str         # For the subplot title (e.g., '2019' or 'Date Not in Filename')
                    })
        return year_datasets

    def plot_rasters_by_year(self, input_folder, output_folder, mask_path, shp_path) -> None:
        """
        Plots individual raster datasets by year using a global extent.
        param str input_folder: Folder containing the input raster files.
        param str output_folder: Folder to save the output plots.
        param str mask_path: Path to the mask raster file.
        param str shp_path: Path to the shapefile for coastline plotting.
        return: None
        """

        year_datasets = self.getYearDatasets(input_folder)
        if not year_datasets:
            print("No raster files found.")
            return

        target_crs = 'EPSG:3857'
        shp_gdf = gpd.read_file(shp_path).to_crs(target_crs)
        
        all_raster_paths = [ds['path'] for year in year_datasets for ds in year_datasets[year]]
        common_transform, common_extent, common_shape = self.calculateExtent(all_raster_paths + [mask_path], target_crs)
        
        mask_reproj, mask_nodata = self.reprojectToGrid(mask_path, common_transform, common_shape, target_crs, Resampling.nearest)
        mask_boolean_global = mask_reproj != mask_nodata
        mask_geometries_global = [shape(geom) for geom, val in shapes(mask_boolean_global.astype(np.uint8), transform=common_transform) if val > 0]

        global_min, global_max = np.inf, -np.inf
        for path in all_raster_paths:
            reprojected_data, nodata = self.reprojectToGrid(path, common_transform, common_shape, target_crs)
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
            legend_handles = []

            for j, dataset in enumerate(year_datasets[year]):
                destination, nodata = self.reprojectToGrid(dataset['path'], common_transform, common_shape, target_crs)

                # Update the current_area_mask to exclude values >= 0
                current_area_mask = np.logical_and(destination != nodata, destination < 0)
                final_area_mask = np.logical_or(final_area_mask, current_area_mask)
                
                data_geometries = [shape(geom) for geom, val in shapes(current_area_mask.astype(np.uint8), transform=common_transform) if val > 0]
                for geom in data_geometries:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color=colors[j % len(colors)], linewidth=0.5, zorder=12)
                
                # The legend label includes the acquisition date if available
                date_str = f" ({dataset['date']})" if dataset.get('date') else ""
                label_text = f"{dataset['dataset_name']}{date_str}"
                legend_handles.append(Line2D([0], [0], color=colors[j % len(colors)], lw=2, label=label_text))
                
                # The display_mask is now simpler since the >= 0 values are already handled
                display_mask = np.logical_or(destination == nodata, ~mask_boolean_global)
                masked_destination = np.ma.array(destination, mask=display_mask)
                im = ax.imshow(masked_destination, extent=common_extent, cmap=cmap, vmin=global_min, vmax=global_max)

            for geom in mask_geometries_global:
                x, y = geom.exterior.xy
                ax.plot(x, y, color='black', linewidth=0.5, zorder=10)
            
            self.setupSubplot(ax, shp_gdf, common_extent)
            
            valid_pixels = np.sum(final_area_mask)
            pixel_area_m2 = abs(common_transform.a * common_transform.e)
            total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
            
            subplot_title = year_datasets[year][0]['title']
            ax.set_title(f"{subplot_title}\nTotal Area: {total_area_km2:.2f} km$^2$", fontsize=8)
            
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                    fancybox=True, shadow=False, ncol=1, fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        self.finalizeFigure(fig, axes, im, 'Depth (meters)', "Raster Datasets by Year (Global Extent)", os.path.join(output_folder, 'year_plots_global_extent.png'), 1200)

    def plot_rasters_by_year_individual(self, input_folder, output_folder, mask_path, shp_path) -> None:
        """
        Plots individual raster datasets by year using individual extents.
        param str input_folder: Folder containing the input raster files.
        param str output_folder: Folder to save the output plots.
        param str mask_path: Path to the mask raster file.
        param str shp_path: Path to the shapefile for coastline plotting.
        return: None
        """

        year_datasets = self.getYearDatasets(input_folder)
        if not year_datasets:
            print("No raster files found.")
            return

        target_crs = 'EPSG:3857'
        shp_gdf = gpd.read_file(shp_path).to_crs(target_crs)
        
        all_raster_paths = [ds['path'] for year in year_datasets for ds in year_datasets[year]]
        global_transform, _, global_shape = self.calculateExtent(all_raster_paths + [mask_path], target_crs)
        mask_reproj_global, mask_nodata_global = self.reprojectToGrid(mask_path, global_transform, global_shape, target_crs, Resampling.nearest)
        mask_boolean_global_scope = mask_reproj_global != mask_nodata_global
        
        global_min, global_max = np.inf, -np.inf
        for path in all_raster_paths:
            reprojected_data, nodata = self.reprojectToGrid(path, global_transform, global_shape, target_crs)
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
            local_transform, local_extent, local_shape = self.calculateExtent(year_paths + [mask_path], target_crs)
            
            mask_reproj_local, mask_nodata_local = self.reprojectToGrid(mask_path, local_transform, local_shape, target_crs, Resampling.nearest)
            mask_boolean_local = mask_reproj_local != mask_nodata_local
            mask_geometries_local = [shape(geom) for geom, val in shapes(mask_boolean_local.astype(np.uint8), transform=local_transform) if val > 0]
            
            final_area_mask = np.zeros(local_shape, dtype=bool)
            legend_handles = []

            for j, dataset in enumerate(year_datasets[year]):
                destination, nodata = self.reprojectToGrid(dataset['path'], local_transform, local_shape, target_crs)
                
                current_area_mask = np.logical_and(destination != nodata, destination < 0)
                final_area_mask = np.logical_or(final_area_mask, current_area_mask)
                
                data_geometries = [shape(geom) for geom, val in shapes(current_area_mask.astype(np.uint8), transform=local_transform) if val > 0]
                for geom in data_geometries:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color=colors[j % len(colors)], linewidth=0.5, zorder=12)

                date_str = f" ({dataset['date']})" if dataset.get('date') else ""
                label_text = f"{dataset['dataset_name']}{date_str}"
                legend_handles.append(Line2D([0], [0], color=colors[j % len(colors)], lw=2, label=label_text))
                
                display_mask = np.logical_or(destination == nodata, ~mask_boolean_local)
                masked_destination = np.ma.array(destination, mask=display_mask)
                im = ax.imshow(masked_destination, extent=local_extent, cmap=cmap, vmin=global_min, vmax=global_max)

            for geom in mask_geometries_local:
                x, y = geom.exterior.xy
                ax.plot(x, y, color='black', linewidth=0.5, zorder=10)
            
            self.setupSubplot(ax, shp_gdf, local_extent)
            
            valid_pixels = np.sum(final_area_mask)
            pixel_area_m2 = abs(local_transform.a * local_transform.e)
            total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
            
            # Use the 'title' field from the dataset dictionary for the subplot title
            subplot_title = year_datasets[year][0]['title']
            ax.set_title(f"{subplot_title}\nTotal Area: {total_area_km2:.2f} km$^2$", fontsize=8)
            
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), 
                    fancybox=True, shadow=False, ncol=1, fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        self.finalizeFigure(fig, axes, im, 'Depth (meters)', "Raster Datasets by Year (Individual Extent)", os.path.join(output_folder, 'year_plots_individual_extent.png'), 600)
    
    def process_single_vrt(self, vrt_file, output_dir, x_res, y_res, resampling_method, creation_options, warpOptions, multithread) -> None:
        """
        Processes a single VRT file. This function will be executed by a Dask worker.
        """
        
        try:
            base_name = os.path.basename(vrt_file)
            output_filename = os.path.join(output_dir, base_name.replace(".vrt", "_resampled.tif"))

            # Check if the output file already exists. If so, skip processing.
            if os.path.exists(output_filename):
                skip_message = f"Output file {output_filename} already exists. Skipping."
                print(skip_message)

            print(f"Processing {base_name}...")

            gdal.Warp(
                output_filename,
                vrt_file,
                xRes=x_res,
                yRes=y_res,
                resampleAlg=resampling_method,
                creationOptions=creation_options,
                warpOptions=warpOptions,   
                multithread=multithread     
            )

            print(f"Successfully resampled and saved to {output_filename}")
        except Exception as e:
            print(f"Error processing {vrt_file}: {e}")

    def reprojectToGrid(self, path, transform, shape, target_crs, resampling_method=Resampling.bilinear) -> tuple:
        """
        Reprojects a raster to a specified grid.
        param str path: Path to the input raster file.
        param object transform: Affine transform for the target grid.
        param tuple shape: (height, width) of the target grid.
        param str target_crs: Target coordinate reference system (e.g., 'EPSG:3857').
        param object resampling_method: Resampling method from rasterio.enums.Resampling.
        return: Reprojected raster data as a NumPy array and the nodata value.
        """

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

    def resample_vrt_files(self, x_res=0.0359, y_res=0.0359, resampling_method='nearest') -> None:
        """
        Resample all .vrt files in the input directory in parallel using Dask
        and save them to the output directory.

        param x_res: Horizontal resolution for resampling
        param y_res: Vertical resolution for resampling
        param resampling_method: Resampling method to use (e.g., 'bilinear', 'cubic')
        """

        input_dir = get_config_item("LIDAR_PLOTS", "INPUT_VRTS")
        output_dir = get_config_item("LIDAR_PLOTS", "RESAMPLED_VRTS")

        os.makedirs(output_dir, exist_ok=True)

        creation_options = [
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=YES"
        ]

        vrt_files = glob.glob(os.path.join(input_dir, "*.vrt"))

        if not vrt_files:
            print(f"No .vrt files found in {input_dir}")
            return

        print(f"Found {len(vrt_files)} VRT files to process.")
        print(f"Unique items in vrt_files: {len(set(vrt_files))}")

        tasks = []

        for vrt_file in vrt_files:
            task = dask.delayed(self.process_single_vrt)(
                vrt_file,
                output_dir,
                x_res,
                y_res,
                resampling_method,
                creation_options,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                multithread=True
            )
            tasks.append(task)

        # dask.compute(*tasks)
        print("All vrts have been resampled.")

    def plot_difference(self, input_folder, output_folder, mask_path, shp_path, mode, use_individual_extent=False) -> None:
        """
        Calculates and plots raster differences for consecutive or all year pairs.
        param str input_folder: Folder containing the input raster files.
        param str output_folder: Folder to save the output plots.
        param str mask_path: Path to the mask raster file.
        param str shp_path: Path to the shapefile for coastline plotting.
        param str mode: 'consecutive' for consecutive year differences, 'all' for all year pairs with >=5% overlap.
        param bool use_individual_extent: Whether to use individual extents for each subplot.
        return: None
        """

        year_datasets_map = self.getYearDatasets(input_folder)
        if not year_datasets_map or len(year_datasets_map) < 2:
            print("Not enough years of data to calculate differences.")
            return
        
        year_datasets = {year: data[0]['path'] for year, data in year_datasets_map.items()}

        target_crs = 'EPSG:3857'
        shp_gdf = gpd.read_file(shp_path).to_crs(target_crs)
        sorted_years = sorted(year_datasets.keys())
        
        all_paths = list(year_datasets.values()) + [mask_path]
        global_transform, global_extent, global_shape = self.calculateExtent(all_paths, target_crs)
        mask_reproj_global, mask_nodata_global = self.reprojectToGrid(mask_path, global_transform, global_shape, target_crs, Resampling.nearest)
        mask_boolean_global = mask_reproj_global != mask_nodata_global
        
        reprojected_global = {}
        for year in sorted_years:
            data, nodata = self.reprojectToGrid(year_datasets[year], global_transform, global_shape, target_crs)
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
                local_transform, local_extent, local_shape = self.calculateExtent([data_dict['path1'], data_dict['path2'], mask_path], target_crs)
                
                data1, nodata1 = self.reprojectToGrid(data_dict['path1'], local_transform, local_shape, target_crs)
                data2, nodata2 = self.reprojectToGrid(data_dict['path2'], local_transform, local_shape, target_crs)
                mask_reproj, mask_nodata = self.reprojectToGrid(mask_path, local_transform, local_shape, target_crs, Resampling.nearest)
                
                mask_boolean = mask_reproj != mask_nodata
                combined_mask = np.logical_or.reduce((data1 == nodata1, data2 == nodata2, ~mask_boolean))
                diff = np.ma.array(data2 - data1, mask=combined_mask)
                
                im_diff = ax.imshow(diff, extent=local_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
                mask_geometries = [shape(geom) for geom, val in shapes(mask_boolean.astype(np.uint8), transform=local_transform) if val > 0]
                self.setupSubplot(ax, shp_gdf, local_extent)
            else:
                diff = reprojected_global[data_dict['year2']] - reprojected_global[data_dict['year1']]
                im_diff = ax.imshow(diff, extent=global_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
                mask_boolean = mask_reproj_global != mask_nodata_global
                mask_geometries = [shape(geom) for geom, val in shapes(mask_boolean.astype(np.uint8), transform=global_transform) if val > 0]
                self.setupSubplot(ax, shp_gdf, global_extent)

            ax.set_title(data_dict['title'], loc='left', fontsize=8, fontweight='bold')
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
        self.finalizeFigure(fig, axes, im_diff, 'Difference (m)', suptitle, os.path.join(output_folder, output_filename), dpi)

    def run(self) -> None:
        """Entrypoint for processing the Lidar Data Plots"""

        # client = Client(n_workers=7, threads_per_worker=2, memory_limit="32GB")
        # print(f"Dask Dashboard: {client.dashboard_link}")
        
        # self.resample_vrt_files()

        # client.close()

        raster_folder = get_config_item("LIDAR_PLOTS", "RESAMPLED_VRTS")
        plot_output_folder = get_config_item("LIDAR_PLOTS", "PLOT_OUTPUTS")
        mask_path = get_config_item("MASK", "MASK_TRAINING_PATH")
        shp_path = get_config_item("MASK", "COAST_BOUNDARY_PATH")

        self.plot_rasters_by_year(raster_folder, plot_output_folder, mask_path, shp_path)
        # self.plot_rasters_by_year_individual(raster_folder, plot_output_folder, mask_path, shp_path)

        # self.plot_difference(raster_folder, plot_output_folder, mask_path, shp_path, 'consecutive', use_individual_extent=False)
        # self.plot_difference(raster_folder, plot_output_folder, mask_path, shp_path, 'consecutive', use_individual_extent=True)

        self.plot_difference(raster_folder, plot_output_folder, mask_path, shp_path, 'all', use_individual_extent=False)
        # self.plot_difference(raster_folder, plot_output_folder, mask_path, shp_path, 'all', use_individual_extent=True)


        print("Lidar Data Plots processing complete.")

    def setupSubplot(self, ax, shp_gdf, extent) -> None:
        """
        Configures the appearance of a subplot.
        param object ax: Matplotlib axes object.
        param object shp_gdf: GeoDataFrame containing the shapefile data.
        param tuple extent: (left, right, bottom, top) for setting axis limits.
        return: None
        """

        shp_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, zorder=11)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
        ax.set_xlabel('X distance (km)', fontsize=8)
        ax.set_ylabel('Y distance (km)', fontsize=8)
        
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        ax.set_xticklabels([f'{val/1000:.1f}' for val in xticks], fontsize=8)
        ax.set_yticklabels([f'{val/1000:.1f}' for val in yticks], fontsize=8)
