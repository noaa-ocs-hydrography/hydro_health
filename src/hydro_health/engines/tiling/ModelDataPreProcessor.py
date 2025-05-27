"""Class for parallel preprocessing all model data"""

import os
import geopandas as gpd
import pathlib
import numpy as np
import rasterio
import shutil
import rioxarray
import xarray as xr
from scipy.ndimage import uniform_filter
from dask import array as da

from scipy.ndimage import generic_filter
import dask
dask.config.set(scheduler='threads', num_workers=2)


OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'
from hydro_health.helpers.tools import get_config_item


class ModelDataPreProcessor:
    """Class for parallel preprocessing all model data"""

    def focal_fill_block(self, block, w=3):
        """Efficient nan-aware focal mean using uniform_filter."""
        block = block.astype(np.float32)
        nan_mask = np.isnan(block)

        # sum of values in the window
        data_sum = uniform_filter(np.nan_to_num(block, nan=0.0), size=w, mode="constant", cval=0.0)
        
        # count of non-nan cells in the window
        valid_count = uniform_filter((~nan_mask).astype(np.float32), size=w, mode="constant", cval=0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            filled = data_sum / valid_count

        return np.where(nan_mask, filled, block)

    def fill_with_fallback(self, input_file, output_file, max_iters=10, fallback_repeats=5, w=3, chunk_size=1024):
        """
        Attempt chunked iterative focal fill using Dask and rioxarray.
        """
        print(f"Attempting chunked fill for {os.path.basename(input_file)}")

        # Open input raster lazily with Dask
        da_chunk = {"x": chunk_size, "y": chunk_size}
        ds = rioxarray.open_rasterio(input_file, chunks=da_chunk)
        nodata = ds.rio.nodata
        da = ds.squeeze().astype("float32")
        da = da.where(da != nodata)

        # Iterative focal filling using Dask
        for i in range(max_iters):
            print(f"  Iteration {i+1}")
            da_prev = da
            # Apply focal filter using map_blocks
            da = xr.apply_ufunc(
                self.focal_fill_block,
                da,
                kwargs={"w": w},
                input_core_dims=[["y", "x"]],
                output_core_dims=[["y", "x"]],
                dask="parallelized",
                dask_gufunc_kwargs={"allow_rechunk": True},  # ← add this line
                output_dtypes=[da.dtype],
            )

            # Replace nans only
            da = xr.where(np.isnan(da_prev), da, da_prev)

            # Early stop if no nans (optional, but cost of full eval is high)
            # nan_check = da.isnull().any().compute()
            # if not nan_check:
            #     break

        # Replace remaining nans with nodata
        da = da.fillna(nodata)
        da = da.expand_dims(dim="band")

        # Save result to disk
        da.rio.write_nodata(nodata, inplace=True)
        da.rio.to_raster(output_file)
        print(f"Filled raster written to: {output_file}")

    def iterative_focal_fill(self, r, max_iters=10, w=3):
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
    
    # def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
    #     """Main entry point for downloading Digital Coast data"""

    #     digital_coast_folder = pathlib.Path(outputs) / 'DigitalCoast'

    #     # tile_gdf.to_file(rF'{OUTPUTS}\tile_gdf.shp', driver='ESRI Shapefile')
    #     self.process_tile_index(digital_coast_folder, tile_gdf, outputs)
    #     self.process_intersected_datasets(digital_coast_folder, tile_gdf)
    #     if digital_coast_folder.exists():
    #         self.delete_unused_folder(digital_coast_folder)

    def repeat_disk_focal_fill(self, input_file, output_final, output_dir, n_repeats=5, w=3, layer_name="unknown"):
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

    def run_gap_fill(self, output_dir, max_iters=10, fallback_repeats=10, w=3):
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

        # bathy_files = get_config_item('DIGITALCOAST', 'TILED_DATA')

        # for file_path in bathy_files:
        file_path = r'C:\Users\aubrey.mccutchan\Documents\bathy_2004.tif'
        output_file = os.path.join(
            output_dir, os.path.splitext(os.path.basename(file_path))[0] + "_filled_python.tif"
        )
        self.fill_with_fallback(
            input_file=file_path,
            output_file=output_file,
            max_iters=max_iters,
            fallback_repeats=fallback_repeats,
            w=w
        )

        print("Gap fill process complete.")
