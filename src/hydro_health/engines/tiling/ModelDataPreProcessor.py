"""Class for parallel preprocessing all model data"""

import os
import geopandas as gpd
import pathlib
import numpy as np
import rasterio
import shutil

from scipy.ndimage import generic_filter


OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


class ModelDataPreProcessor:
    """Class for parallel preprocessing all model data"""

    def fill_with_fallback(input_file, output_file, max_iters=10, fallback_repeats=5, w=3):
        """
        Attempt in-memory iterative focal fill. If that fails, fall back to disk-based repeated fill.
        
        Parameters:
        - input_file: path to input raster
        - output_file: path to write final filled raster
        - max_iters: max iterations for in-memory method
        - fallback_repeats: fallback pass count for disk-based method
        - w: window size for focal filtering
        """
        layer_name = os.path.basename(input_file)
        print(f"Attempting iterative fill for {layer_name}")
        
        kernel = np.ones((w, w))

        try:
            # Load input raster
            with rasterio.open(input_file) as src:
                profile = src.profile
                data = src.read(1).astype(float)
                nodata = src.nodata
                data[data == nodata] = np.nan

            # Inline iterative_focal_fill logic
            for _ in range(max_iters):
                if not np.isnan(data).any():
                    break
                filled = generic_filter(
                    data,
                    function=lambda values: np.nanmean(values),
                    footprint=kernel,
                    mode='constant',
                    cval=np.nan
                )
                data = np.where(np.isnan(data), filled, data)

            # Save result
            with rasterio.open(output_file, 'w', **profile) as dst:
                dst.write(np.nan_to_num(data, nan=nodata), 1)

            print(f"Iterative fill succeeded: {layer_name}")
            return

        except Exception as e:
            print(f"Iterative fill failed for {layer_name} - {e}")
            print(f"🛠️  Fallback disk-based fill starting for {layer_name}")

        # Fallback to disk-based focal repeat
        temp_file = input_file

        for i in range(1, fallback_repeats + 1):
            print(f" {layer_name} - Disk-Based Focal Fill Iteration {i} of {fallback_repeats}")
            out_path = os.path.join(
                os.path.dirname(output_file),
                f"{os.path.splitext(os.path.basename(input_file))[0]}_f{i}.tif"
            )

            if os.path.exists(out_path):
                os.remove(out_path)

            try:
                with rasterio.open(temp_file) as src:
                    profile = src.profile
                    data = src.read(1).astype(float)
                    nodata = src.nodata
                    data[data == nodata] = np.nan

                filled = generic_filter(
                    data,
                    function=lambda values: np.nanmean(values),
                    footprint=kernel,
                    mode='constant',
                    cval=np.nan
                )
                output_data = np.where(np.isnan(data), filled, data)

                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(np.nan_to_num(output_data, nan=nodata), 1)

                temp_file = out_path

            except Exception as e2:
                print(f" Focal failed at iteration {i} - {e2}")

        if os.path.exists(temp_file):
            shutil.move(temp_file, output_file)
            print(f" Final filled raster saved as: {os.path.basename(output_file)}")
        else:
            print(f" Final result file was not created for {layer_name}")

    def iterative_focal_fill(r, max_iters=10, w=3):
        """
        Iteratively fills NaN values in a 2D NumPy array using a focal mean filter.
        
        Parameters:
        - r: 2D NumPy array with NaNs representing missing data
        - max_iters: maximum number of focal iterations to run
        - w: window size for the focal kernel (must be odd)
        
        Returns:
        - A NumPy array with missing values filled
        """
        footprint = np.ones((w, w))
        
        for _ in range(max_iters):
            if not np.isnan(r).any():
                break
            filled = generic_filter(
                r,
                lambda values: np.nanmean(values),
                footprint=footprint,
                mode='constant',
                cval=np.nan
            )
            r = np.where(np.isnan(r), filled, r)
        
        return r
    
    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        """Main entry point for downloading Digital Coast data"""

        digital_coast_folder = pathlib.Path(outputs) / 'DigitalCoast'

        # tile_gdf.to_file(rF'{OUTPUTS}\tile_gdf.shp', driver='ESRI Shapefile')
        self.process_tile_index(digital_coast_folder, tile_gdf, outputs)
        self.process_intersected_datasets(digital_coast_folder, tile_gdf)
        if digital_coast_folder.exists():
            self.delete_unused_folder(digital_coast_folder)

    def repeat_disk_focal_fill(input_file, output_final, output_dir, n_repeats=5, w=3, layer_name="unknown"):
        """
        Repeatedly fills NaN values in a raster by applying focal mean filtering
        and writing intermediate results to disk. Uses disk I/O for each pass.
        
        Parameters:
        - input_file: path to input raster
        - output_final: path to final output raster
        - output_dir: directory to store intermediate outputs
        - n_repeats: number of focal passes
        - w: window size for focal mean (must be odd)
        - layer_name: optional name for logging
        """
        temp_file = input_file
        kernel = np.ones((w, w))

        for i in range(1, n_repeats + 1):
            print(f" {layer_name} - Disk-Based Focal Fill Iteration {i} of {n_repeats}")
            
            out_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(input_file))[0]}_f{i}.tif")

            if os.path.exists(out_path):
                os.remove(out_path)

            try:
                with rasterio.open(temp_file) as src:
                    profile = src.profile
                    data = src.read(1).astype(float)
                    data[data == src.nodata] = np.nan

                filled = generic_filter(
                    data,
                    function=lambda values: np.nanmean(values),
                    footprint=kernel,
                    mode='constant',
                    cval=np.nan
                )

                output_data = np.where(np.isnan(data), filled, data)

                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(np.nan_to_num(output_data, nan=profile['nodata']), 1)

                temp_file = out_path

            except Exception as e:
                print(f" Focal failed at iteration {i} - {e}")

        if os.path.exists(temp_file):
            shutil.move(temp_file, output_final)
            print(f" Final filled raster saved as: {os.path.basename(output_final)}")
        else:
            print(f" Final result file was not created for {layer_name}")

    def run_gap_fill(bathy_files, output_dir, max_iters=10, fallback_repeats=10, w=3):
        """
        Sequentially fills gaps in raster files using iterative focal fill with fallback strategy.
        
        Parameters:
        - bathy_files: list of input raster file paths
        - output_dir: directory to save filled raster files
        - max_iters: max iterations for the iterative fill method
        - fallback_repeats: how many fallback passes to try
        - w: kernel/window size
        """
        print("Starting gap fill module...")

        for file_path in bathy_files:
            output_file = os.path.join(
                output_dir, os.path.splitext(os.path.basename(file_path))[0] + "_filled.tif"
            )
            fill_with_fallback(
                input_file=file_path,
                output_file=output_file,
                max_iters=max_iters,
                fallback_repeats=fallback_repeats,
                w=w
            )

        print("Gap fill process complete.")
