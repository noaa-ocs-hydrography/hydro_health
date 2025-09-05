import os
import pathlib
import geopandas as gpd
import pandas as pd
import time
import shutil
from scipy.ndimage import generic_filter, uniform_filter
import rioxarray
import xarray as xr

from osgeo import ogr, osr, gdal
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item
import os
import glob
import time
import re
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import generic_filter
from dask.distributed import Client, LocalCluster
import dask


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'
gpkg_path = INPUTS / 'sediment_data' / 'sediment_data.gpkg'


class CreateSeabedTerrainLayerEngine(Engine):
    """Class to hold the logic for processing the Seabed Terrain layer"""

    def __init__(self, 
                 param_lookup:dict=None):
        super().__init__()
        self.sediment_types = ['Gravel', 'Sand', 'Mud', 'Clay']

    def focal_fill_block(self, block: np.ndarray, w=3) -> np.ndarray:
        """
        Performs a single, efficient, nan-aware focal mean on a NumPy array block.

        This is a low-level helper function that operates on a small chunk of data.
        It calculates a focal mean while correctly handling NaN (missing) values.
        
        Parameters:
        - block: A 2D NumPy array with NaNs to be filled.
        - w: The window size for the focal mean. Must be an odd number.
        
        Returns:
        - A NumPy array with NaNs filled by the focal mean.
        """
        block = block.astype(np.float32)
        nan_mask = np.isnan(block)

        # sum of values in the window
        # np.nan_to_num replaces NaNs with 0, which allows uniform_filter to work.
        # The nan_mask is used later to only fill original NaNs.
        data_sum = uniform_filter(np.nan_to_num(block, nan=0.0), size=w, mode="constant", cval=0.0)
        
        # count of non-nan cells in the window
        # This creates a "valid" count for each window to get a true mean.
        valid_count = uniform_filter((~nan_mask).astype(np.float32), size=w, mode="constant", cval=0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            # The filled values are the sum divided by the count.
            filled = data_sum / valid_count

        # The core logic: only replace the original NaNs with the filled values.
        return np.where(nan_mask, filled, block)

    def fill_with_fallback(self, input_file, output_file, max_iters=10, chunk_size=1024) -> None:
        """
        Performs chunked iterative focal fill on a raster file using Dask and rioxarray.

        This is the main workhorse function. It opens a large raster lazily (using Dask)
        to avoid loading it all into memory. It then iteratively applies the
        focal_fill_block function to fill small gaps.
        
        Parameters:
        - input_file: Path to the input raster file.
        - output_file: Path where the filled raster will be saved.
        - max_iters: The maximum number of fill iterations to run.
        - chunk_size: The size of the chunks to process at a time.
        """
        print(f"Attempting chunked fill for {os.path.basename(input_file)}")

        # Open input raster lazily with Dask. This is the key to handling large files.
        da_chunk = {"x": chunk_size, "y": chunk_size}
        ds = rioxarray.open_rasterio(input_file, chunks=da_chunk)
        nodata = ds.rio.nodata
        da = ds.squeeze().astype("float32")
        da = da.where(da != nodata)

        # Iterative focal filling using Dask's map_blocks for parallel processing.
        for i in range(max_iters):
            print(f"  Iteration {i+1}")
            da_prev = da
            
            # Apply focal_fill_block across all chunks in parallel.
            da = xr.apply_ufunc(
                self.focal_fill_block,
                da,
                kwargs={"w": 3},
                input_core_dims=[["y", "x"]],
                output_core_dims=[["y", "x"]],
                dask="parallelized",
                dask_gufunc_kwargs={"allow_rechunk": True},
                output_dtypes=[da.dtype],
            )

            # Important: only replace NaN values from the previous iteration.
            da = xr.where(np.isnan(da_prev), da, da_prev)

        # Replace any remaining NaNs with the original NoData value and save.
        da = da.fillna(nodata)
        da = da.expand_dims(dim="band")
        da.rio.write_nodata(nodata, inplace=True)
        da.rio.to_raster(output_file)
        print(f"Filled raster written to: {output_file}")

    def run_gap_fill(self, input_file, output_dir, max_iters=10, w=3) -> None:
        """
        The main entry point for the gap-filling process.

        This function sets up the file paths and orchestrates the call to the
        chunked, Dask-based fill process.
        
        Parameters:
        - input_file: Path to the single input raster file.
        - output_dir: Directory where the output file will be saved.
        - max_iters: Maximum number of focal fill iterations.
        - w: The focal window size.
        """
        print("Starting gap fill module...")

        output_file = os.path.join(
            output_dir, os.path.splitext(os.path.basename(input_file))[0] + "_filled_python.tif"
        )
        self.fill_with_fallback(
            input_file=input_file,
            output_file=output_file,
            max_iters=max_iters,
            w=w
        )

        print("Gap fill process complete.")

    # ==============================================================================
    #
    #   Strategic & Automated Benthic Terrain Modeler (BTM) - Spatially Consistent
    #                         (Parallel Python Version)
    #
    # ==============================================================================
    #
    # Purpose:
    # This script implements a two-phase workflow to ensure spatially consistent
    # terrain classification across large, tiled datasets. It is designed to:
    #
    #   1. PHASE 1 (Pre-computation): Scan all input files, group them by year, and
    #      take a large, representative sample of pixels from each year's data.
    #      From this sample, it creates a single, regionally-consistent
    #      classification dictionary for each year.
    #
    #   2. PHASE 2 (Parallel Processing): Process all individual raster tiles in
    #      parallel. Each worker loads the appropriate pre-computed dictionary for
    #      its tile's year, ensuring all tiles from the same year are classified
    #      using the exact same rules and thresholds.
    #
    # This prevents artificial seams at tile boundaries and ensures a consistent
    # classification across the entire study area for each time-slice.
    #
    # ==============================================================================

    # ==============================================================================
    #   CORE BTM HELPER FUNCTIONS
    # ==============================================================================

    def calculate_bpi(self, bathy_array, cell_size, inner_radius, outer_radius):
        """Calculates the Bathymetric Position Index for a numpy array."""
        inner_cells = int(round(inner_radius / cell_size))
        outer_cells = int(round(outer_radius / cell_size))
        
        y, x = np.ogrid[-outer_cells:outer_cells + 1, -outer_cells:outer_cells + 1]
        mask = x**2 + y**2 <= outer_cells**2
        mask[x**2 + y**2 <= inner_cells**2] = False
        
        def annulus_mean(buffer):
            # Create a view of the buffer that corresponds to the mask
            masked_buffer = buffer.reshape(mask.shape)
            return np.nanmean(masked_buffer[mask])

        footprint = np.ones((2 * outer_cells + 1, 2 * outer_cells + 1))
        mean_annulus = generic_filter(
            bathy_array,
            function=annulus_mean,
            size=footprint.shape,
            mode='mirror'
        )
        
        return bathy_array - mean_annulus

    def standardize_raster_array(self, input_array):
        """Standardizes a numpy array (mean=0, sd=1)."""
        mean = np.nanmean(input_array)
        std = np.nanstd(input_array)
        if std == 0:
            return np.zeros_like(input_array)
        return (input_array - mean) / std

    def calculate_slope_and_tri(self, bathy_array, cell_size):
        """Calculates slope (in degrees) and Terrain Ruggedness Index (TRI)."""
        gy, gx = np.gradient(bathy_array, cell_size)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(slope_rad)
        
        def tri_func(buffer):
            center = buffer[len(buffer)//2]
            return np.nanmean(np.abs(buffer - center))
            
        footprint = np.ones((3, 3))
        tri = generic_filter(bathy_array, function=tri_func, footprint=footprint, mode='mirror')
        
        return slope_deg, tri

    def create_classification_dictionary(self, bpi_broad_std_sample, bpi_fine_std_sample, slope_sample):
        """Creates a data-driven classification dictionary from representative sample arrays."""
        broad_breaks = np.nanquantile(bpi_broad_std_sample, [0.15, 0.85])
        fine_breaks = np.nanquantile(bpi_fine_std_sample, [0.15, 0.85])
        slope_break = np.nanquantile(slope_sample, 0.85)

        nan = np.nan
        dictionary_data = {
            'Class_ID': range(1, 9),
            'Zone_Name': ["Broad Flat/Plain", "Broad Depression", "Broad Crest",
                        "Fine Crest on Broad Flat", "Fine Depression on Broad Flat",
                        "Crest on Broad Crest", "Depression on Broad Crest", "Steep Slope"],
            'BroadBPI_Lower': [broad_breaks[0], nan, broad_breaks[1], broad_breaks[0], broad_breaks[0], broad_breaks[1], broad_breaks[1], nan],
            'BroadBPI_Upper': [broad_breaks[1], broad_breaks[0], nan, broad_breaks[1], broad_breaks[1], nan, nan, nan],
            'FineBPI_Lower': [fine_breaks[0], nan, nan, fine_breaks[1], nan, fine_breaks[1], nan, nan],
            'FineBPI_Upper': [fine_breaks[1], nan, nan, nan, fine_breaks[0], nan, fine_breaks[0], nan],
            'Slope_Lower': [nan, nan, nan, nan, nan, nan, nan, slope_break],
            'Slope_Upper': [slope_break, slope_break, slope_break, slope_break, slope_break, slope_break, slope_break, nan]
        }
        df = pd.DataFrame(dictionary_data)
        df.fillna({'BroadBPI_Lower': -9999, 'BroadBPI_Upper': 9999,
                'FineBPI_Lower': -9999, 'FineBPI_Upper': 9999,
                'Slope_Lower': -9999, 'Slope_Upper': 9999}, inplace=True)
        return df

    # ==============================================================================
    #   PHASE 1: PRE-COMPUTATION OF CONSISTENT DICTIONARIES
    # ==============================================================================

    def create_regionally_consistent_dictionaries(self, all_files, best_radii, output_dir, max_sample_files=10, pixels_per_file=20000):
        """
        Scans files, groups by year, samples pixels, and creates a single consistent
        classification dictionary for each year.
        """
        print("\n--- PHASE 1: Creating Regionally Consistent Dictionaries ---")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Group files by year using regex to find 4-digit years
        year_groups = {}
        for f in all_files:
            match = re.search(r'(\d{4})', os.path.basename(f))
            if match:
                year = match.group(1)
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(f)
        
        # Handle the generic 'bt.bathy.tif' case
        generic_bathy = [f for f in all_files if 'bt.bathy' in os.path.basename(f)]
        if generic_bathy:
            year_groups['bt_bathy'] = generic_bathy

        # 2. Process each year group
        for year, files in year_groups.items():
            print(f"\n  - Processing group: {year} ({len(files)} files found)")
            
            # Take a subset of files to sample from for efficiency
            files_to_sample = files if len(files) <= max_sample_files else np.random.choice(files, max_sample_files, replace=False)
            
            # 3. Collect samples from the subset of files
            all_samples = {'slope': [], 'bpi_fine_std': [], 'bpi_broad_std': []}
            
            for f in files_to_sample:
                try:
                    with rasterio.open(f) as src:
                        bathy_array = src.read(1)
                        bathy_array[bathy_array == src.nodata] = np.nan
                        cell_size = src.res[0]
                        
                        # Take a random sample of valid pixel indices
                        valid_pixels = np.argwhere(~np.isnan(bathy_array))
                        if len(valid_pixels) > pixels_per_file:
                            sample_indices = valid_pixels[np.random.choice(len(valid_pixels), pixels_per_file, replace=False)]
                        else:
                            sample_indices = valid_pixels
                        
                        # Create a small array around each sample point to handle focal operations
                        # This is an approximation but avoids processing the whole raster
                        slope_sample, _ = self.calculate_slope_and_tri(bathy_array, cell_size)
                        
                        bpi_fine_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                        bpi_broad_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                        
                        # Get the sampled values
                        rows, cols = sample_indices[:, 0], sample_indices[:, 1]
                        all_samples['slope'].append(slope_sample[rows, cols])
                        all_samples['bpi_fine_std'].append(self.standardize_raster_array(bpi_fine_sample)[rows, cols])
                        all_samples['bpi_broad_std'].append(self.standardize_raster_array(bpi_broad_sample)[rows, cols])

                except Exception as e:
                    print(f"    - Warning: Could not sample from {os.path.basename(f)}. Reason: {e}")
                    continue
            
            # 4. Create and save the dictionary for this year
            if all_samples['slope']:
                slope_agg = np.concatenate(all_samples['slope'])
                fine_agg = np.concatenate(all_samples['bpi_fine_std'])
                broad_agg = np.concatenate(all_samples['bpi_broad_std'])
                
                year_dictionary = self.create_classification_dictionary(broad_agg, fine_agg, slope_agg)
                dict_path = os.path.join(output_dir, f"dictionary_{year}.csv")
                year_dictionary.to_csv(dict_path, index=False)
                print(f"  - Saved consistent dictionary for year {year} to: {dict_path}")
            else:
                print(f"  - No valid samples collected for year {year}. Skipping dictionary creation.")
                
        print("\n--- PHASE 1 Complete ---")

    # ==============================================================================
    #   PHASE 2: PARALLEL PROCESSING OF INDIVIDUAL RASTERS
    # ==============================================================================

    def generate_terrain_products_python(self, bathy_path, best_radii, dictionary_dir):
        """
        Main function to process one bathymetry raster using a pre-computed dictionary.
        """
        try:
            base_name = os.path.splitext(os.path.basename(bathy_path))[0]
            output_dir = os.path.dirname(bathy_path)
            
            # Determine the year and load the correct dictionary
            year = 'bt_bathy' # default for generic name
            match = re.search(r'(\d{4})', base_name)
            if match:
                year = match.group(1)
                
            dict_path = os.path.join(dictionary_dir, f"dictionary_{year}.csv")
            if not os.path.exists(dict_path):
                raise FileNotFoundError(f"Dictionary not found for year {year} at {dict_path}")
            
            unique_dictionary = pd.read_csv(dict_path)
            
            print(f"\n--- Starting processing for: {base_name} (using '{year}' dictionary) ---")

            with rasterio.open(bathy_path) as src:
                bathy_array = src.read(1)
                bathy_array[bathy_array == src.nodata] = np.nan
                profile = src.profile
                cell_size = src.res[0]

            print("  - Generating derivatives...")
            slope, rugosity = self.calculate_slope_and_tri(bathy_array, cell_size)
            bpi_fine = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
            bpi_broad = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
            bpi_fine_std = self.standardize_raster_array(bpi_fine)
            bpi_broad_std = self.standardize_raster_array(bpi_broad)
            
            print("  - Classifying terrain...")
            classified_array = np.zeros_like(bathy_array, dtype=np.uint8)
            
            for index, rule in unique_dictionary.iterrows():
                matches = (
                    (bpi_broad_std >= rule['BroadBPI_Lower']) & (bpi_broad_std <= rule['BroadBPI_Upper']) &
                    (bpi_fine_std >= rule['FineBPI_Lower']) & (bpi_fine_std <= rule['FineBPI_Upper']) &
                    (slope >= rule['Slope_Lower']) & (slope <= rule['Slope_Upper'])
                )
                classified_array[matches & (classified_array == 0)] = rule['Class_ID']
                
            print("  - Saving output files...")
            outputs = {
                "_slope.tif": slope, "_rugosity_tri.tif": rugosity,
                "_bpi_fine_std.tif": bpi_fine_std, "_bpi_broad_std.tif": bpi_broad_std,
                "_terrain_classification.tif": classified_array
            }
            
            for suffix, data_array in outputs.items():
                out_path = os.path.join(output_dir, base_name + suffix)
                profile.update(dtype=data_array.dtype.name, nodata=0, count=1)
                with rasterio.open(out_path, 'w', **profile) as dst:
                    dst.write(data_array.astype(profile['dtype']), 1)

            print(f"  - Successfully processed and saved all products for {base_name}")
            return f"Success: {bathy_path}"

        except Exception as e:
            error_msg = f"FAILED to process {bathy_path}: {e}"
            print(error_msg)
            return error_msg

    # ==============================================================================
    #   MAIN ORCHESTRATION SCRIPT
    # ==============================================================================
    def start(self):
        """Main function to find all bathy files and process them in parallel."""
        # --- 1. DEFINE USER INPUTS ---
        root_dirs = [
            r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\pre_processed\BlueTopo",
            r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\pre_processed\tiled"
        ]

        main_output_dir = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\BTM_outputs"
        dictionary_output_dir = os.path.join(main_output_dir, "dictionaries")

        self.run_gap_fill(input_file_path, output_directory, max_iters=5)

        client = Client(n_workers=7, threads_per_worker=2, memory_limit="32GB")
        print(f"Dask Dashboard: {client.dashboard_link}")

        best_radii = {'fine': (8, 32), 'broad': (80, 240)}

        # --- 2. FIND ALL BATHY FILES TO PROCESS ---
        bathy_files_to_process = []
        print("Scanning for bathymetry files...")
        for root in root_dirs:
            if os.path.isdir(root):
                tile_folders = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
                for tile in tile_folders:
                    tif_files = glob.glob(os.path.join(root, tile, "*.tif"))
                    bathy_files_to_process.extend(tif_files)
        
        bathy_files_to_process = [
            f for f in bathy_files_to_process 
            if not any(suffix in f for suffix in ['_slope', '_rugosity', '_bpi', '_classification'])
        ]
        
        if not bathy_files_to_process:
            print("No bathymetry files found to process. Exiting.")
            return

        print(f"Found {len(bathy_files_to_process)} bathymetry files to process.")

        # --- 3. RUN PHASE 1: PRE-COMPUTE DICTIONARIES ---
        self.create_regionally_consistent_dictionaries(bathy_files_to_process, best_radii, dictionary_output_dir)

        # --- 4. RUN PHASE 2: PARALLEL CLASSIFICATION ---
        cluster = LocalCluster()
        client = Client(cluster)
        print(f"\nDask dashboard available at: {client.dashboard_link}")
        
        tasks = []
        for bathy_file in bathy_files_to_process:
            task = dask.delayed(self.generate_terrain_products_python)(bathy_file, best_radii, dictionary_output_dir)
            tasks.append(task)
            
        print("\nStarting parallel processing of all files...")
        results = dask.compute(*tasks)
        
        print("\n--- Processing Complete ---")
        
        success_count = sum(1 for r in results if isinstance(r, str) and r.startswith("Success"))
        fail_count = len(results) - success_count
        print(f"Successfully processed: {success_count} files")
        print(f"Failed to process: {fail_count} files")
        
        if fail_count > 0:
            print("\n--- Failures ---")
            for r in results:
                if isinstance(r, str) and r.startswith("FAILED"):
                    print(r)
                    
        client.close()
        cluster.close()
