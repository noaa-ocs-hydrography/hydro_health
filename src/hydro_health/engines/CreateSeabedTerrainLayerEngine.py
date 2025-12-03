import os
import re
import shutil

import dask
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
from dask.distributed import Client, print
from dask import delayed, compute
from scipy.ndimage import binary_erosion, generic_filter, uniform_filter
import itertools
from pathlib import Path
from collections import defaultdict
import re
from pathlib import Path
from typing import List, Dict, Optional
import dask
import rioxarray


from hydro_health.helpers.tools import get_config_item
# from hydro_health.engines.Engine import Engine


class CreateSeabedTerrainLayerEngine():
    """Class to hold the logic for processing the Seabed Terrain layer"""

    def __init__(self):
        self.input_dir = Path(get_config_item('TERRAIN', 'FILLED_DIR'))
        self.output_dir = Path(get_config_item('TERRAIN', 'COMBINED_LIDAR_DIR'))
        self.target_vars = ["terrain", "slope"]
        super().__init__()
        self.year_ranges = [
            (1998, 2004),
            (2004, 2006),
            (2006, 2007),
            (2007, 2010),
            (2010, 2015),
            (2014, 2022),
            (2016, 2017),
            (2017, 2018),
            (2018, 2019),
            (2020, 2022),
            (2022, 2024)
        ]


    def focal_fill_block(self, block: np.ndarray, w=3) -> np.ndarray:
        """
        Performs a single, efficient, nan-aware focal mean on a NumPy array block.

        This is a low-level helper function that operates on a small chunk of data.
        It calculates a focal mean while correctly handling NaN (missing) values.

        param np.ndarray block: 2D NumPy array representing a chunk of raster data.
        param int w: Size of the moving window (must be odd).
        return: 2D NumPy array with NaNs filled in the block.
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

    def fill_with_fallback(self, input_file, output_file, max_iters=5, chunk_size=1024) -> None:
        """
        Performs chunked iterative focal fill on a raster file using Dask and rioxarray.

        This is the main workhorse function. It opens a large raster lazily (using Dask)
        to avoid loading it all into memory. It then iteratively applies the
        focal_fill_block function to fill small gaps.
        
        param str input_file: Path to the input raster file.
        param str output_file: Path where the filled raster will be saved.
        param int max_iters: Maximum number of fill iterations to perform.
        param int chunk_size: Size of the chunks to process at a time.
        return: None
        """

        print(f"Attempting chunked fill for {os.path.basename(input_file)}")

        # Open input raster lazily with Dask. This is the key to handling large files.
        da_chunk = {"x": chunk_size, "y": chunk_size}
        ds = rioxarray.open_rasterio(input_file, chunks=da_chunk)
        nodata = ds.rio.nodata
        da = ds.squeeze().astype("float32")
        da = da.where(da != nodata)
        
        # --- NEW OPTIMIZATION ---
        # A more advanced check for missing data to distinguish between
        # exterior (boundary) NaNs and interior gaps.
        print("Checking for interior gaps...")
        
        # We need to compute the NaN mask to use binary_erosion from SciPy.
        nan_mask = da.isnull().compute()

        interior_nan_count = binary_erosion(nan_mask, structure=np.ones((3,3))).sum()
        print(interior_nan_count)
        
        # Erode the mask. Any remaining 'True' values indicate an interior NaN.
        # This is a much more precise check.
        if not binary_erosion(nan_mask, structure=np.ones((3,3))).any():
            print(f"No interior gaps found in {os.path.basename(input_file)}. Skipping fill process.")
            shutil.copyfile(input_file, output_file)
            print(f"File copied to: {output_file}")
            return

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

        da.rio.write_crs(ds.rio.crs, inplace=True)
        da.rio.write_transform(ds.rio.transform(), inplace=True)
        
        da.rio.write_nodata(nodata, inplace=True)
        da.rio.to_raster(output_file, compress='LZW')
        print(f"Filled raster written to: {output_file}")

    def run_gap_fill(self, input_file, output_dir, max_iters) -> None:
        """
        The main entry point for the gap-filling process.
        This function sets up the file paths and orchestrates the call to the
        chunked, Dask-based fill process.
        
        param str input_file: Path to the input raster file.
        param str output_dir: Directory where the filled raster will be saved.
        param int max_iters: Maximum number of fill iterations to perform.
        return: None
        """

        print("Starting gap fill module...")

        output_file = os.path.join(
            output_dir, os.path.splitext(os.path.basename(input_file))[0] + "_filled.tif"
        )

        if os.path.exists(output_file):
            print(f"File already exists, skipping gap fill: {os.path.basename(output_file)}")
            return
        
        self.fill_with_fallback(
            input_file=input_file,
            output_file=output_file,
            max_iters=max_iters
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

    def calculate_bpi(self, bathy_array, cell_size, inner_radius, outer_radius) -> np.ndarray:
        """Calculates the Bathymetric Position Index for a numpy array.
         param np.ndarray bathy_array: 2D numpy array of bathymetry values.
         param float cell_size: Size of each cell in the raster (assumed square).
         param float inner_radius: Inner radius for the annulus (in same units as cell_size).
         param float outer_radius: Outer radius for the annulus (in same units as cell_size).
         return: 2D numpy array of BPI values.

        """
        
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

    def standardize_raster_array(self, input_array) -> np.ndarray:
        """Standardizes a numpy array (mean=0, sd=1).
        
        param np.ndarray input_array: 2D numpy array to standardize.
        return: 2D numpy array of standardized values.
        """

        mean = np.nanmean(input_array)
        std = np.nanstd(input_array)
        if std == 0:
            return np.zeros_like(input_array)
        return (input_array - mean) / std

    def calculate_slope_and_tri(self, bathy_array, cell_size) -> tuple[np.ndarray, np.ndarray]:
        """Calculates slope (in degrees) and Terrain Ruggedness Index (TRI).
        param np.ndarray bathy_array: 2D numpy array of bathymetry values.
        param float cell_size: Size of each cell in the raster (assumed square).
        return: Tuple of 2D numpy arrays (slope in degrees, TRI).
        """

        gy, gx = np.gradient(bathy_array, cell_size)
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
        slope_deg = np.degrees(slope_rad)
        
        def tri_func(buffer):
            center = buffer[len(buffer)//2]
            return np.nanmean(np.abs(buffer - center))
            
        footprint = np.ones((3, 3))
        tri = generic_filter(bathy_array, function=tri_func, footprint=footprint, mode='mirror')
        
        return slope_deg, tri

    def create_classification_dictionary(self, bpi_broad_std_sample, bpi_fine_std_sample, slope_sample) -> pd.DataFrame:
        """Creates a data-driven classification dictionary from representative sample arrays.

        param np.ndarray bpi_broad_std_sample: 1D array of standardized broad BPI samples.
        param np.ndarray bpi_fine_std_sample: 1D array of standardized fine BPI samples.
        param np.ndarray slope_sample: 1D array of slope samples.
        return: Pandas DataFrame representing the classification dictionary.
        """

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

    def create_regionally_consistent_dictionaries(self, all_files, best_radii, output_dir, max_sample_files=10, pixels_per_file=20000) -> None:
        """
        Scans files, groups by year, samples pixels, and creates a single consistent
        classification dictionary for each year.

        param list all_files: List of file paths to process.
        param dict best_radii: Dictionary with 'fine' and 'broad' radius tuples.
        param str output_dir: Directory where dictionaries will be saved.
        param int max_sample_files: Max number of files to sample per year.
        param int pixels_per_file: Number of random pixels to sample from each file.
        return: None
        """

        print("\n--- PHASE 1: Creating Regionally Consistent Dictionaries ---")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        valid_years = {year for year_pair in self.year_ranges for year in year_pair}
        print(valid_years)    

        # 1. Group files by year using regex to find 4-digit years
        year_groups = {}
        for f in all_files:
            match = re.search(r'((?:19|20)\d{2})', os.path.basename(f))

            if match and int(match.group(1)) in valid_years:
                year = match.group(1)
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(f)
        
        # Handle the generic 'BLueTopo.tif' case
        generic_bathy = [f for f in all_files if 'BlueTopo' in os.path.basename(f)]
        if generic_bathy:
            year_groups['BlueTopo'] = generic_bathy
        # 2. Process each year group
        for year, files in year_groups.items():
            print(f"\n  - Processing group: {year} ({len(files)} files found)")

            # 1. Get sizes for all files (stored as tuples to avoid hitting disk twice)
            # Format: [(filepath, size), (filepath, size), ...]
            file_data = [(f, os.path.getsize(f)) for f in files]

            # 2. Calculate the 30th percentile threshold
            # We extract just the sizes to do the math
            all_sizes = [x[1] for x in file_data]
            size_threshold = np.percentile(all_sizes, 30)

            # 3. Create a pool of only the smallest 30% of files
            # We filter the original list based on the calculated threshold
            small_files_pool = [x[0] for x in file_data if x[1] <= size_threshold]

            # 4. Apply your sampling logic to this specific pool
            files_to_sample = (
                small_files_pool 
                if len(small_files_pool) <= max_sample_files 
                else list(np.random.choice(small_files_pool, max_sample_files, replace=False))
            )
            # 3. Collect samples from the subset of files
            all_samples = {'slope': [], 'bpi_fine_std': [], 'bpi_broad_std': []}
            
            for f in files_to_sample:
                print(f"    Sampling from: {os.path.basename(f)}")    
                try:
                    with rasterio.open(f) as src:
                        bathy_array = src.read(1)
                        bathy_array[bathy_array == src.nodata] = np.nan
                        cell_size = src.res[0]
                        
                        # Take a random sample of valid pixel indices
                        print(f"      - Extracting up to {pixels_per_file} random valid pixels...")
                        valid_pixels = np.argwhere(~np.isnan(bathy_array))
                        if len(valid_pixels) > pixels_per_file:
                            sample_indices = valid_pixels[np.random.choice(len(valid_pixels), pixels_per_file, replace=False)]
                        else:
                            sample_indices = valid_pixels
                        
                        # Create a small array around each sample point to handle focal operations
                        # This is an approximation but avoids processing the whole raster
                        print("      - Calculating derivatives for sampled pixels...")
                        slope_sample, _ = self.calculate_slope_and_tri(bathy_array, cell_size)
                        print("      - Calculating BPI for sampled pixels...")
                        
                        bpi_fine_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                        bpi_broad_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                        
                        # Get the sampled values
                        print("      - Collecting sampled values...")
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

    def generate_neighborhood_statistics(self, file_path:Path) -> None:
        """
        Calculates focal mean and standard deviation for a given raster file
        and saves them as new .tif files.
        """
        print(f"Calculating neighborhood statistics for: {file_path.name}")
        size = 3  # Fixed neighborhood size

        base_name = file_path.stem  # e.g., "bathy_2004"
        out_dir = file_path.parent
        
        out_sd = out_dir / f"{base_name}_sd{size}.tif"
        out_mean = out_dir / f"{base_name}_mean{size}.tif"

        if os.path.exists(out_sd) or 'mean' in file_path.name:
            print(f"Output files already exist, skipping: {out_sd.name}")
            return 
        
        if os.path.exists(out_mean) or 'sd3' in file_path.name:
            print(f"Output files already exist, skipping: {out_mean.name}")
            return
        
        rds = rioxarray.open_rasterio(file_path, chunks=True).isel(band=0)
        
        # Create the rolling window (kernel)
        # 'center=True' mimics the R 'focal' behavior
        window = rds.rolling(x=size, y=size, center=True)

        # Calculate focal mean and standard deviation
        r_mean = window.mean()
        r_sd = window.std()
        
        r_mean.rio.to_raster(out_mean, driver="GTiff", compress="LZW")
        r_sd.rio.to_raster(out_sd, driver="GTiff", compress="LZW")

    def load_and_average(self, paths: List[Path]) -> xr.DataArray:
        """Loads a list of rasters and returns the mean array.
           Uses skipna=True to ensure valid data is averaged even if some layers have NoData.
        """
        # masked=True converts the file's NoData value (e.g., -9999) to np.nan
        das = [rioxarray.open_rasterio(p, chunks=None, masked=True).isel(band=0) for p in paths]
        
        if len(das) == 1:
            return das[0]
        
        combined = xr.concat(das, dim="merge_dim")
        
        # CRITICAL FIX: skipna=True ensures we calculate a 'nanmean'.
        # If Pixel A is 10.0 in file1 and NaN in file2, the result is 10.0 (not NaN).
        averaged = combined.mean(dim="merge_dim", keep_attrs=True, skipna=True)
        
        # Explicitly write CRS as 'mean' operation can sometimes drop it
        if das[0].rio.crs:
            averaged.rio.write_crs(das[0].rio.crs, inplace=True)
            
        return averaged

    @dask.delayed
    def process_delta_task(self, paths_t0: List[Path], paths_t1: List[Path], out_path: Path) -> str:
        """Calculates delta between two sets of file paths.
           Executes as a single synchronous unit of work on the worker.
        """
        if out_path.exists():
            return f"Skipped (Exists): {out_path.name}"

        try:
            # 1. Load data into memory (NumPy arrays)
            da_t0 = self.load_and_average(paths_t0)
            da_t1 = self.load_and_average(paths_t1)

            # 2. Alignment Check
            # Using reproject_match on in-memory arrays is simpler and safer than on dask arrays
            # However, if you are CERTAIN grids are identical, you can skip this to save time.
            # We keep it here to fix your 'bounds' error from earlier.
            if da_t0.rio.bounds() != da_t1.rio.bounds():
                 da_t1 = da_t1.rio.reproject_match(da_t0)

            # 3. Calculate Delta
            delta = da_t1 - da_t0

            # 4. Restore Metadata
            delta.rio.write_crs(da_t0.rio.crs, inplace=True)
            delta.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

            # 5. Write to disk
            # We use 'tiled=True' here for the output format, but the calculation happened in RAM
            delta.rio.to_raster(
                out_path,
                driver="GTiff",
                compress="LZW",
                tiled=True,
                windowed=True
            )
            return f"Created: {out_path.name}"

        except Exception as e:
            return f"Error on {out_path.name}: {str(e)}"

    def _get_tile_id(self, filename: str) -> Optional[str]:
            """Extracts the 8-character alphanumeric Tile ID.
            Handles IDs surrounded by underscores (e.g., _BH4Q958G_).
            Ignores 8-digit dates (e.g., 20240917).
            """
            # Look for exactly 8 uppercase letters or digits
            # We use findall to get EVERY match in the string
            candidates = re.findall(r"[A-Z0-9]{8}", filename)
            
            for cand in candidates:
                # Return the first candidate that is NOT purely numbers
                # (This filters out dates like '20240917' but keeps 'BH4Q958G')
                if not cand.isdigit():
                    return cand
                    
            return None

    def _get_year(self, filename: str) -> Optional[int]:
        """Extracts the 4-digit year.
           param str filename: Input filename.
           return: Integer year or None.
        """
        pattern = r"(?P<year>199\d|20[0-2]\d)"
        match = re.search(pattern, filename)
        return int(match.group("year")) if match else None

    def _get_variable_type(self, filename: str) -> Optional[str]:
        """Determines if file matches target variables.
           param str filename: Input filename.
           return: Variable name ('terrain' or 'slope') or None.
        """
        # Assumes self.target_vars is defined in your class __init__
        fname_lower = filename.lower()
        for var in self.target_vars:
            if var in fname_lower:
                return var
        return None

    def group_files(self) -> Dict:
        """Groups files by Tile and Variable, then by Year.
           return: Nested dictionary structure: dict[tile_id][variable][year] = [List of Paths]
        """
        groups = {}

        print(len(list(self.input_dir.rglob("*.tif"))))


        # Assumes self.input_dir is defined in your class __init__
        for f_path in list(self.input_dir.rglob("*.tif")):
            f_name = f_path.name
            
            print(f_name)
            tile_id = self._get_tile_id(f_name)
            print(tile_id)
            year = self._get_year(f_name)
            print(year)
            var = self._get_variable_type(f_name)
            print(var)  

            if tile_id and year and var:
                if tile_id not in groups:
                    groups[tile_id] = {}
                if var not in groups[tile_id]:
                    groups[tile_id][var] = {}
                if year not in groups[tile_id][var]:
                    groups[tile_id][var][year] = []
                
                groups[tile_id][var][year].append(f_path)
        
        return groups

    def process_delta_rasters(self):
        """Executes the delta processing pipeline.
           return: None
        """
        # Assumes self.year_ranges and self.output_dir are defined in your class __init__
        groups = self.group_files()
        delayed_tasks = []

        print(f"Scanning complete. Found groups for {len(groups)} tiles.")

        for tile_id, var_dict in groups.items():
            for var, year_map in var_dict.items():
                
                for y0, y1 in self.year_ranges:
                    if y0 in year_map and y1 in year_map:
                        
                        paths_t0 = year_map[y0]
                        paths_t1 = year_map[y1]
                        
                        if len(paths_t0) > 1:
                            print(f"  [Info] Averaging {len(paths_t0)} datasets for {tile_id} - Year {y0} ({var})")
                        
                        if len(paths_t1) > 1:
                            print(f"  [Info] Averaging {len(paths_t1)} datasets for {tile_id} - Year {y1} ({var})")

                        out_name = f"delta_{tile_id}_{y0}_{y1}_{var}.tif"
                        out_path = self.output_dir / out_name
                        
                        task = self.process_delta_task(paths_t0, paths_t1, out_path)
                        delayed_tasks.append(task)

        if not delayed_tasks:
            print("No matching year pairs found.")
            return

        print(f"Queued {len(delayed_tasks)} delta calculations. Computing...")
        
        results = dask.compute(*delayed_tasks)
        
        for res in results:
            print(res)

    def group_files_simple(self) -> Dict:
        """Groups files strictly by Tile ID and Year.
           Used for the batch averaging process.
           return: dict[tile_id][year] = [List of Paths]
        """
        groups = {}
        
        # Ensure input_dir is a Path object
        input_path = Path(self.input_dir)

        for f_path in input_path.glob("*.tif"):
            f_name = f_path.name
            
            tile_id = self._get_tile_id(f_name)
            year = self._get_year(f_name)

            if tile_id and year:
                if tile_id not in groups:
                    groups[tile_id] = {}
                if year not in groups[tile_id]:
                    groups[tile_id][year] = []
                
                groups[tile_id][year].append(f_path)
        
        return groups

    @dask.delayed
    def process_combination_task(self, paths: List[Path], out_path: Path) -> str:
        """Loads one or more rasters, averages them, and saves to new name.
           Handles both single files (copy/format) and multiple files (average).
        """
        if out_path.exists():
            return f"Skipped (Exists): {out_path.name}"

        try:
            # Reuses your existing robust load logic (in-memory, no chunks)
            da_avg = self.load_and_average(paths)
            
            # Ensure spatial dimensions are set correctly before writing
            da_avg.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True, compress='LZW')

            da_avg.rio.to_raster(
                out_path,
                driver="GTiff",
                compress="LZW",
                tiled=True,
                windowed=True
            )
            return f"Created: {out_path.name}"
            
        except Exception as e:
            return f"Error on {out_path.name}: {str(e)}"

    def run_bathy_combination(self):
        """Orchestrates the finding, averaging, and renaming of bathy files."""
        groups = self.group_files_simple()
        delayed_tasks = []

        print(f"Scanning complete. Found groups for {len(groups)} tiles.")

        for tile_id, year_map in groups.items():
            for year, paths in year_map.items():
                
                # Naming Convention: combined_bathy_{tileid}_{year}.tif
                out_name = f"combined_bathy_{tile_id}_{year}.tif"
                out_path = self.output_dir / out_name
                
                # Optional: Logging for averaging vs single file
                if len(paths) > 1:
                    print(f"  [Info] Averaging {len(paths)} files for {out_name}")
                
                task = self.process_combination_task(paths, out_path)
                delayed_tasks.append(task)

        if not delayed_tasks:
            print("No matching files found to process.")
            return

        print(f"Queued {len(delayed_tasks)} combination tasks. Computing...")
        
        results = dask.compute(*delayed_tasks)
        
        for res in results:
            print(res)

    def generate_terrain_products_python(self, bathy_path, best_radii, dictionary_dir) -> str:
        """
        Main function to process one bathymetry raster using a pre-computed dictionary.

        param str bathy_path: Path to the input bathymetry raster file.
        param dict best_radii: Dictionary with 'fine' and 'broad' radius tuples.
        param str dictionary_dir: Directory where pre-computed dictionaries are stored.
        return: Status message indicating success or failure.
        """
        
        base_name = os.path.splitext(os.path.basename(bathy_path))[0]
        output_dir = os.path.join(get_config_item('TERRAIN', 'OUTPUTS'), 'BTM_outputs')

        classification_file = os.path.join(output_dir, base_name + "_terrain_classification.tif")
        if os.path.exists(classification_file):
            print(f"Output files already exist, skipping: {base_name}")
            return 
        
        # Determine the year and load the correct dictionary
        year = 'bt_bathy' # default for generic name
        match = re.search(r'((?:19|20)\d{2})', base_name)
        if match:
            year = match.group(1)
            print(year)
            
        dict_path = os.path.join(dictionary_dir, f"dictionary_{year}.csv")
        if not os.path.exists(dict_path):
            print(f"Dictionary not found for year {year} at {dict_path} for {base_name}")
            return None
        
        unique_dictionary = pd.read_csv(dict_path)
        
        print(f"\n--- Processing {base_name} (using '{year}' dictionary) ---")

        with rasterio.open(bathy_path) as src:
            bathy_array = src.read(1)
            bathy_array[bathy_array == src.nodata] = np.nan
            profile = src.profile
            cell_size = src.res[0]

        # print("  - Generating derivatives...")
        if "slope" not in base_name.lower():
            slope, rugosity = self.calculate_slope_and_tri(bathy_array, cell_size)
        bpi_fine = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
        bpi_broad = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
        bpi_fine_std = self.standardize_raster_array(bpi_fine)
        bpi_broad_std = self.standardize_raster_array(bpi_broad)
        
        # print("  - Classifying terrain...")
        classified_array = np.zeros_like(bathy_array, dtype='float32')
        
        for index, rule in unique_dictionary.iterrows():
            matches = (
                (bpi_broad_std >= rule['BroadBPI_Lower']) & (bpi_broad_std <= rule['BroadBPI_Upper']) &
                (bpi_fine_std >= rule['FineBPI_Lower']) & (bpi_fine_std <= rule['FineBPI_Upper']) &
                (slope >= rule['Slope_Lower']) & (slope <= rule['Slope_Upper'])
            )
            classified_array[matches & (classified_array == 0)] = rule['Class_ID']
            
        outputs = {
            "_slope.tif": slope, "_rugosity_tri.tif": rugosity,
            "_bpi_fine_std.tif": bpi_fine_std, "_bpi_broad_std.tif": bpi_broad_std,
            "_terrain_classification.tif": classified_array
        }
        
        for suffix, data_array in outputs.items():
            out_path = os.path.join(output_dir, base_name + suffix)
                    
            profile.update(
                dtype=data_array.dtype.name, 
                nodata=np.nan, 
                count=1,
                compress='LZW' 
            )
            
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(data_array.astype(profile['dtype']), 1)

        return f"Success: {bathy_path}"

    # ==============================================================================
    #   MAIN ORCHESTRATION SCRIPT
    # ==============================================================================
    def process(self) -> None:
        """Main function to find all bathy files and process them in parallel."""

        client = Client(n_workers=4, threads_per_worker=2, memory_limit="32GB")
        print(f"Dask Dashboard: {client.dashboard_link}")

        main_output_dir = get_config_item('TERRAIN', 'OUTPUTS')
        dictionary_output_dir = os.path.join(main_output_dir, "dictionaries")

        input_dir = get_config_item('TERRAIN', 'INPUT_DIR')
        filled_dir = get_config_item('TERRAIN', 'FILLED_DIR')

        keywords_to_exclude = ['tsm', 'hurr', 'sed', 'bluetopo']
        lidar_data_paths = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(('.tif', '.tiff')) and not any(keyword in f.lower() for keyword in keywords_to_exclude)
        ]        

        tasks = []
        # for file in lidar_data_paths:
        #     task = dask.delayed(self.run_gap_fill(file, filled_dir, max_iters=3))
        #     tasks.append(task)
            
        # dask.compute(*tasks)

        # --- 2. FIND ALL BATHY FILES TO PROCESS ---
        bathy_files_to_process = [os.path.join(filled_dir, f) for f in os.listdir(filled_dir)]

        # Adding Blutopo files to the lists
        bathy_files_to_process.extend(
            [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
            if f.endswith(('.tif', '.tiff')) and 'BlueTopo' in f]
        )

        vars_to_exclude = ["unc","slope", "rugosity", "bpi_fine", "bpi_broad", "terrain_classification", "survey_end_date"]

        bathy_files_to_process = [
            f for f in bathy_files_to_process 
            if not any(v in os.path.basename(f) for v in vars_to_exclude)
        ]

        print(f"Found {len(bathy_files_to_process)} bathymetry files to process.")

        # TODO best radii might need changed if different eco region
        best_radii = {'fine': (8, 32), 'broad': (80, 240)}
        # --- 3. RUN PHASE 1: PRE-COMPUTE DICTIONARIES ---
        # self.create_regionally_consistent_dictionaries(bathy_files_to_process, best_radii, dictionary_output_dir)

        # --- 4. RUN PHASE 2: PARALLEL CLASSIFICATION ---
        print("\n--- PHASE 2: Parallel Processing of terrain products")
        tasks = []
        for bathy_file in bathy_files_to_process:
            task = dask.delayed(self.generate_terrain_products_python)(bathy_file, best_radii, dictionary_output_dir)
            tasks.append(task)
            
        # dask.compute(*tasks)

        WORKING_DIR = get_config_item('TERRAIN', 'OUTPUTS')
        STATE_VARS = ["filled", "slope", "rugosity", "bpi_fine", "bpi_broad", "terrain_classification"]


        print(f"Starting raster processing in: {WORKING_DIR}")
        print(f"Finding files for: {', '.join(STATE_VARS)}")
        
        state_files_map = defaultdict(list)
        all_state_files_to_process = []

        WORKING_DIR = Path(WORKING_DIR)

        valid_years = {str(year) for year_pair in self.year_ranges for year in year_pair}

        year_pattern_str = "|".join(valid_years)
        var_pattern_str = "|".join(STATE_VARS)

        FILE_PATTERN = re.compile(rf".*?_({year_pattern_str})\d*?_.*_({var_pattern_str}).*?$")

        for f in WORKING_DIR.rglob("*.tif"):
            match = FILE_PATTERN.search(f.name)
            if match:
                year = int(match.group(1))
                var_name = match.group(2)

                file_info = {'path': f, 'year': year}
                
                if var_name not in state_files_map:
                    state_files_map[var_name] = []
                    
                state_files_map[var_name].append(file_info)
                all_state_files_to_process.append(f)
                
        print(f"Found {len(set(all_state_files_to_process))} rasters for neighborhood stats.")
        print(f"Found {len(state_files_map)} variable groups for delta calculation.")
        
        # --- Job 1: Process Neighborhood Stats in Parallel ---
        delayed_tasks = []
        delayed_tasks = [
            dask.delayed(self.generate_neighborhood_statistics)(file_path)
            for file_path in set(all_state_files_to_process)
        ]

        # dask.compute(*delayed_tasks)

        # --- Job 2: Create Delta Rasters in Parallel ---
        # We need to pass the dict for each variable group
        delta_groups = [{'var': var_name, 'files': files} for var_name, files in state_files_map.items() if len(files) > 1]
        
        print(f"\nRunning delta raster creation for {len(delta_groups)} variable groups")

        # self.process_delta_rasters()
        self.run_bathy_combination()
        
        # print("\n--- Delta Results ---")
        # for res_list in delta_results:
        #     for res in res_list:
        #         print(res)
        
        print("\n--- Processing Complete ---")
                    
        client.close()
