import os
import re
import shutil
import tempfile
import itertools
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional

import s3fs
from upath import UPath
import numpy as np
import pandas as pd
import rasterio
import rioxarray
import xarray as xr
import dask
from dask.distributed import Client, print
from dask import delayed, compute
from scipy.ndimage import binary_erosion, generic_filter, uniform_filter

# Check for WhiteboxTools dependency
try:
    from whitebox import WhiteboxTools
except ImportError:
    print("\nCRITICAL ERROR: The 'whitebox' library is not installed.")
    print("Please install it by running: conda install whitebox\n")
    raise

from hydro_health.helpers.tools import get_config_item, get_environment
# from hydro_health.engines.Engine import Engine


class CreateSeabedTerrainLayerEngine():
    """Class to hold the logic for processing the Seabed Terrain layer"""

    def __init__(self):
        self.is_aws = (get_environment() == 'aws')
        self.fs = s3fs.S3FileSystem(anon=False)
        
        # Initialize WhiteboxTools
        self.wbt = WhiteboxTools()
        self.wbt.verbose = False
        self.wbt.set_compress_rasters(True)
        
        # super().__init__() # Uncomment if inheriting from Engine
        
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

        # Variables that do not change regardless of time step
        self.static_vars = [
            "grain_size_layer.tif",
            "prim_sed_layer.tif",
            "survey_end_date.tif",
            "bt.UC.tif",
            "bt.bathy.tif",
            "bt.bpi_broad.tif",
            "bt.bpi_fine.tif",
            "bt.curv_plan.tif",
            "bt.curv_profile.tif",
            "bt.curv_total.tif",
            "bt.flowacc.tif",
            "bt.flowdir.tif",
            "bt.gradmag.tif",
            "bt.rugosity.tif",
            "bt.shearproxy.tif",
            "bt.slope.tif",
            "bt.slope_deg.tif",
            "bt.tci.tif",
            "bt.terrain_classification.tif"
        ]

        # Variables formatted as {stem}_{start_year}.tif
        self.dynamic_single_year_stems = [
            "bathy_{}_filled", 
            "bpi_broad",
            "bpi_fine",
            "curv_plan",
            "curv_profile",
            "curv_total",
            "flowacc",
            "flowdir",
            "gradmag",
            "rugosity",
            "shearproxy",
            "slope",
            "slope_deg",
            "tci",
            "terrain_classification"
        ]

        # Variables formatted as {stem}_{start_year}_{end_year}.tif
        self.dynamic_range_stems = [
            "hurr_count",
            "hurr_strength",
            "tsm"
        ]

    # ==============================================================================
    #  PRIVATE HELPER FUNCTIONS
    # ==============================================================================

    def _exists(self, path) -> bool:
        """Checks if a file exists, natively compatible with both local and S3 paths via UPath."""
        return UPath(path).exists()

    def _get_tile_id(self, filename: str) -> Optional[str]:
        """Extracts the 8-character alphanumeric Tile ID."""
        candidates = re.findall(r"[A-Z0-9]{8}", filename)
        for cand in candidates:
            if not cand.isdigit():
                return cand
        return None

    def _get_variable_type(self, filename: str) -> Optional[str]:
        """Determines if file matches target variables."""
        fname_lower = filename.lower()
        # Fallback to empty list if target_vars isn't defined
        for var in getattr(self, 'target_vars', []):
            if var in fname_lower:
                return var
        return None

    def _get_year(self, filename: str) -> Optional[int]:
        """Extracts the 4-digit year."""
        pattern = r"(?P<year>199\d|20[0-2]\d)"
        match = re.search(pattern, filename)
        return int(match.group("year")) if match else None

    def _getsize(self, path) -> int:
        """Gets size of file safely using UPath"""
        return UPath(path).stat().st_size

    def _join_paths(self, *args) -> str:
        """Safely joins paths for any protocol returning string layout"""
        if not args:
            return ""
        return str(UPath(args[0]).joinpath(*args[1:]))

    def _safe_ls(self, path) -> List[str]:
        """Safely list directory contents, returning empty list if missing."""
        try:
            p = UPath(path)
            if not p.exists():
                return []
            return [str(child) for child in p.iterdir()]
        except FileNotFoundError:
            return []

    # ==============================================================================
    #  PUBLIC FUNCTIONS
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

    def create_file_paths(self):
        filled_dir = get_config_item('TERRAIN', 'FILLED_DIR')
        combined_dir = get_config_item('TERRAIN', 'COMBINED_LIDAR_DIR')

        if get_environment() == 'remote':
            self.input_dir = UPath(filled_dir)
            self.output_dir = UPath(combined_dir)

        elif get_environment() == 'aws':
            bucket = get_config_item('S3', 'BUCKET_NAME').strip('/')
            base_path = UPath(f"s3://{bucket}")
            self.input_dir = base_path / filled_dir.strip('/')
            self.output_dir = base_path / combined_dir.strip('/')

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
        out_path = UPath(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        valid_years = {year for year_pair in self.year_ranges for year in year_pair}
        print(valid_years)    

        # 1. Group files by year
        year_groups = {}
        for f in all_files:
            match = re.search(r'((?:19|20)\d{2})', os.path.basename(str(f)))

            if match and int(match.group(1)) in valid_years:
                year = match.group(1)
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(f)
        
        # Handle the generic 'BlueTopo.tif' case
        generic_bathy = [f for f in all_files if 'BlueTopo' in os.path.basename(str(f))]
        if generic_bathy:
            year_groups['BlueTopo'] = generic_bathy

        # 2. Process each year group
        for year, files in year_groups.items():
            print(f"\n  - Processing group: {year} ({len(files)} files found)")

            file_data = [(f, self._getsize(f)) for f in files]
            all_sizes = [x[1] for x in file_data]
            size_threshold = np.percentile(all_sizes, 30)
            small_files_pool = [x[0] for x in file_data if x[1] <= size_threshold]

            files_to_sample = (
                small_files_pool 
                if len(small_files_pool) <= max_sample_files 
                else list(np.random.choice(small_files_pool, max_sample_files, replace=False))
            )
            
            all_samples = {'slope': [], 'bpi_fine_std': [], 'bpi_broad_std': []}
            
            for f in files_to_sample:
                print(f"    Sampling from: {os.path.basename(str(f))}")    
                try:
                    with rasterio.open(f) as src:
                        bathy_array = src.read(1)
                        bathy_array[bathy_array == src.nodata] = np.nan
                        cell_size = src.res[0]
                        
                        print(f"      - Extracting up to {pixels_per_file} random valid pixels...")
                        valid_pixels = np.argwhere(~np.isnan(bathy_array))
                        if len(valid_pixels) > pixels_per_file:
                            sample_indices = valid_pixels[np.random.choice(len(valid_pixels), pixels_per_file, replace=False)]
                        else:
                            sample_indices = valid_pixels
                        
                        print("      - Calculating derivatives for sampled pixels...")
                        slope_sample, _ = self.calculate_slope_and_tri(bathy_array, cell_size)
                        print("      - Calculating BPI for sampled pixels...")
                        
                        bpi_fine_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                        bpi_broad_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                        
                        print("      - Collecting sampled values...")
                        rows, cols = sample_indices[:, 0], sample_indices[:, 1]
                        all_samples['slope'].append(slope_sample[rows, cols])
                        all_samples['bpi_fine_std'].append(self.standardize_raster_array(bpi_fine_sample)[rows, cols])
                        all_samples['bpi_broad_std'].append(self.standardize_raster_array(bpi_broad_sample)[rows, cols])

                except Exception as e:
                    print(f"    - Warning: Could not sample from {os.path.basename(str(f))}. Reason: {e}")
                    continue
            
            # Create and save the dictionary for this year
            if all_samples['slope']:
                slope_agg = np.concatenate(all_samples['slope'])
                fine_agg = np.concatenate(all_samples['bpi_fine_std'])
                broad_agg = np.concatenate(all_samples['bpi_broad_std'])
                
                year_dictionary = self.create_classification_dictionary(broad_agg, fine_agg, slope_agg)
                dict_path = UPath(self._join_paths(output_dir, f"dictionary_{year}.csv"))
                
                with dict_path.open('w') as fh:
                    year_dictionary.to_csv(fh, index=False)
                
                print(f"  - Saved consistent dictionary for year {year} to: {dict_path}")
            else:
                print(f"  - No valid samples collected for year {year}. Skipping dictionary creation.")
                
        print("\n--- PHASE 1 Complete ---")

    def fill_with_fallback(self, input_file, output_file, max_iters=5, chunk_size=1024) -> None:
        """
        Performs chunked iterative focal fill on a raster file using Dask and rioxarray.
        param str input_file: Path to the input raster file.
        param str output_file: Path where the filled raster will be saved.
        param int max_iters: Maximum number of fill iterations to perform.
        param int chunk_size: Size of the chunks to process at a time.
        return: None
        """
        print(f"Attempting chunked fill for {os.path.basename(str(input_file))}")

        da_chunk = {"x": chunk_size, "y": chunk_size}
        ds = rioxarray.open_rasterio(input_file, chunks=da_chunk)
        nodata = ds.rio.nodata
        da = ds.squeeze().astype("float32")
        da = da.where(da != nodata)
        
        print("Checking for interior gaps...")
        
        nan_mask = da.isnull().compute()
        interior_nan_count = binary_erosion(nan_mask, structure=np.ones((3,3))).sum()
        print(interior_nan_count)
        
        if not binary_erosion(nan_mask, structure=np.ones((3,3))).any():
            print(f"No interior gaps found in {os.path.basename(str(input_file))}. Skipping fill process.")
            src = UPath(input_file)
            dst = UPath(output_file)
            if src.protocol == "s3":
                src.fs.copy(str(src), str(dst))
            else:
                shutil.copyfile(src, dst)
            print(f"File copied to: {output_file}")
            return

        for i in range(max_iters):
            print(f"  Iteration {i+1}")
            da_prev = da
            
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

            da = xr.where(np.isnan(da_prev), da, da_prev)

        da = da.fillna(nodata)
        da = da.expand_dims(dim="band")

        da.rio.write_crs(ds.rio.crs, inplace=True)
        da.rio.write_transform(ds.rio.transform(), inplace=True)
        da.rio.write_nodata(nodata, inplace=True)
        da.rio.to_raster(output_file, compress='LZW')
        print(f"Filled raster written to: {output_file}")

    def focal_fill_block(self, block: np.ndarray, w=3) -> np.ndarray:
        """
        Performs a single, efficient, nan-aware focal mean on a NumPy array block.
        param np.ndarray block: 2D NumPy array representing a chunk of raster data.
        param int w: Size of the moving window (must be odd).
        return: 2D NumPy array with NaNs filled in the block.
        """
        block = block.astype(np.float32)
        nan_mask = np.isnan(block)

        # sum of values in the window
        data_sum = uniform_filter(np.nan_to_num(block, nan=0.0), size=w, mode="constant", cval=0.0)
        
        # count of non-nan cells in the window
        valid_count = uniform_filter((~nan_mask).astype(np.float32), size=w, mode="constant", cval=0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            filled = data_sum / valid_count

        return np.where(nan_mask, filled, block)

    def generate_neighborhood_statistics(self, file_path: Path) -> None:
        """
        Calculates focal mean and standard deviation for a given raster file
        and saves them as new .tif files.
        """
        print(f"Calculating neighborhood statistics for: {file_path.name}")
        size = 3  # Fixed neighborhood size

        base_name = file_path.stem 
        out_dir = file_path.parent
        
        out_sd = out_dir / f"{base_name}_sd{size}.tif"
        out_mean = out_dir / f"{base_name}_mean{size}.tif"

        if self._exists(out_sd) or 'mean' in file_path.name:
            print(f"Output files already exist, skipping: {out_sd.name}")
            return 
        
        if self._exists(out_mean) or 'sd3' in file_path.name:
            print(f"Output files already exist, skipping: {out_mean.name}")
            return
        
        rds = rioxarray.open_rasterio(file_path, chunks=True).isel(band=0)
        
        window = rds.rolling(x=size, y=size, center=True)
        r_mean = window.mean()
        r_sd = window.std()
        
        r_mean.rio.to_raster(out_mean, driver="GTiff", compress="LZW")
        r_sd.rio.to_raster(out_sd, driver="GTiff", compress="LZW")

    def generate_terrain_products_python(self, bathy_path, best_radii, dictionary_dir) -> str:
        """
        Main function to process one bathymetry raster.
        Integrates WhiteboxTools for morphometrics and NumPy for BPI/Classification.

        param str bathy_path: Path to the input bathymetry raster file.
        param dict best_radii: Dictionary with 'fine' and 'broad' radius tuples.
        param str dictionary_dir: Directory where pre-computed dictionaries are stored.
        return: Status message indicating success or failure.
        """
        base_name = os.path.splitext(os.path.basename(str(bathy_path)))[0]
        output_dir = self._join_paths(get_config_item('TERRAIN', 'OUTPUTS'), 'BTM_outputs')
        
        out_dir_path = UPath(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        # Define Output Paths
        out_slope_deg = self._join_paths(output_dir, base_name + "_slope_deg.tif")
        out_gradmag   = self._join_paths(output_dir, base_name + "_gradmag.tif") 
        out_flowdir   = self._join_paths(output_dir, base_name + "_flowdir.tif") 
        out_prof      = self._join_paths(output_dir, base_name + "_curv_profile.tif")
        out_plan      = self._join_paths(output_dir, base_name + "_curv_plan.tif")
        out_total     = self._join_paths(output_dir, base_name + "_curv_total.tif")
        out_tci       = self._join_paths(output_dir, base_name + "_tci.tif")
        out_flowacc   = self._join_paths(output_dir, base_name + "_flowacc.tif")
        out_shear     = self._join_paths(output_dir, base_name + "_shearproxy.tif")
        
        # --- 1. WhiteboxTools Processing (Direct to Disk) ---
        # Calculate Slope (Deg/Rad), Aspect, Curvatures, TCI, FlowAcc
        
        # Slope (Degrees)
        if not self._exists(out_slope_deg): 
            try: self.wbt.slope(str(bathy_path), out_slope_deg, units="degrees")
            except Exception as e: print(f"WBT Slope (Deg) Error on {base_name}: {e}")

        # Gradient Magnitude (Radians)
        if not self._exists(out_gradmag):
            try: self.wbt.slope(str(bathy_path), out_gradmag, units="radians")
            except Exception as e: print(f"WBT Slope (Rad) Error on {base_name}: {e}")

        # Aspect / Flow Direction
        if not self._exists(out_flowdir):
            try: self.wbt.aspect(str(bathy_path), out_flowdir)
            except Exception as e: print(f"WBT Aspect Error on {base_name}: {e}")

        # Curvatures
        if not self._exists(out_prof):
            try: self.wbt.profile_curvature(str(bathy_path), out_prof)
            except Exception as e: print(f"WBT Profile Curvature Error on {base_name}: {e}")
            
        if not self._exists(out_plan):
            try: self.wbt.plan_curvature(str(bathy_path), out_plan)
            except Exception as e: print(f"WBT Plan Curvature Error on {base_name}: {e}")
            
        if not self._exists(out_total):
            try: self.wbt.total_curvature(str(bathy_path), out_total)
            except Exception as e: print(f"WBT Total Curvature Error on {base_name}: {e}")

        # TCI (Convergence Index) - With fallback
        if not self._exists(out_tci):
            try: 
                self.wbt.convergence_index(str(bathy_path), out_tci)
            except AttributeError:
                # Attempt direct tool run if method missing in wrapper
                try:
                    self.wbt.run_tool("ConvergenceIndex", [f"--dem={str(bathy_path)}", f"--output={out_tci}"])
                except Exception as e:
                    print(f"WBT TCI RunTool Error on {base_name}: {e}")
            except Exception as e: 
                print(f"WBT TCI Error on {base_name}: {e}")

        # Flow Accumulation
        if not self._exists(out_flowacc):
            try: self.wbt.d8_flow_accumulation(str(bathy_path), out_flowacc, out_type="cells")
            except Exception as e: print(f"WBT FlowAcc Error on {base_name}: {e}")

        # --- 2. Shear Proxy (Hybrid: Read WBT outputs -> Calc in Memory -> Write) ---
        if not self._exists(out_shear):
            try:
                # Ensure inputs exist before trying to read them
                if self._exists(out_slope_deg) and self._exists(out_plan):
                    with rasterio.open(out_slope_deg) as s, rasterio.open(out_plan) as p:
                        slope_arr = s.read(1)
                        plan_arr = p.read(1)
                        meta = s.meta.copy()
                    
                    shear = slope_arr * np.abs(plan_arr)
                    
                    meta.update(compress='LZW')
                    with rasterio.open(out_shear, "w", **meta) as dst:
                        dst.write(shear.astype("float32"), 1)
                else:
                    missing = []
                    if not self._exists(out_slope_deg): missing.append("Slope")
                    if not self._exists(out_plan): missing.append("Plan Curvature")
                    print(f"Skipping Shear Proxy for {base_name}: Inputs missing ({', '.join(missing)})")
            except Exception as e:
                print(f"Shear Proxy Calc Error {base_name}: {e}")

        # --- 3. BTM Classification (NumPy In-Memory) ---
        # Check if we need to run numpy math
        numpy_outputs = {
            "_rugosity_tri.tif": None,
            "_bpi_fine_std.tif": None, 
            "_bpi_broad_std.tif": None,
            "_terrain_classification.tif": None
        }
        
        # Check which files are missing
        missing_numpy = False
        for suffix in numpy_outputs.keys():
             if not self._exists(self._join_paths(output_dir, base_name + suffix)):
                  missing_numpy = True
                  break

        if missing_numpy:
            # Get dictionary year
            year = 'bt_bathy'
            match = re.search(r'((?:19|20)\d{2})', base_name)
            if match:
                year = match.group(1)
                
            dict_path = UPath(self._join_paths(dictionary_dir, f"dictionary_{year}.csv"))
            if not dict_path.exists():
                return f"Dictionary missing for {year}"
            
            with dict_path.open('r') as fh:
                unique_dictionary = pd.read_csv(fh)

            # Load Bathy for BPI/Rugosity
            with rasterio.open(bathy_path) as src:
                bathy_array = src.read(1)
                bathy_array[bathy_array == src.nodata] = np.nan
                profile = src.profile
                cell_size = src.res[0]

            # In-memory derivatives for classification
            slope, rugosity = self.calculate_slope_and_tri(bathy_array, cell_size)
            
            bpi_fine = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
            bpi_broad = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
            bpi_fine_std = self.standardize_raster_array(bpi_fine)
            bpi_broad_std = self.standardize_raster_array(bpi_broad)
            
            classified_array = np.zeros_like(bathy_array, dtype='float32')
            
            for index, rule in unique_dictionary.iterrows():
                matches = (
                    (bpi_broad_std >= rule['BroadBPI_Lower']) & (bpi_broad_std <= rule['BroadBPI_Upper']) &
                    (bpi_fine_std >= rule['FineBPI_Lower']) & (bpi_fine_std <= rule['FineBPI_Upper']) &
                    (slope >= rule['Slope_Lower']) & (slope <= rule['Slope_Upper'])
                )
                classified_array[matches & (classified_array == 0)] = rule['Class_ID']
                
            # Write BTM-specific outputs
            outputs = {
                "_rugosity_tri.tif": rugosity,
                "_bpi_fine_std.tif": bpi_fine_std, 
                "_bpi_broad_std.tif": bpi_broad_std,
                "_terrain_classification.tif": classified_array
            }
            
            for suffix, data_array in outputs.items():
                out_path = self._join_paths(output_dir, base_name + suffix)
                if not self._exists(out_path):
                    profile.update(dtype=data_array.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    with rasterio.open(out_path, 'w', **profile) as dst:
                        dst.write(data_array.astype(profile['dtype']), 1)
        else:
             print(f"Skipping Numpy BTM calc for {base_name}, all files exist.")

        return f"Success: {base_name}"

    def group_files(self) -> Dict:
        """Groups files by Tile and Variable, then by Year."""
        groups = {}
        
        try:
            file_paths = list(UPath(self.input_dir).rglob("*.tif"))
        except FileNotFoundError:
            file_paths = []

        print(len(file_paths))

        for f_path in file_paths:
            f_name = f_path.name
            tile_id = self._get_tile_id(f_name)
            year = self._get_year(f_name)
            var = self._get_variable_type(f_name)

            if tile_id and year and var:
                if tile_id not in groups:
                    groups[tile_id] = {}
                if var not in groups[tile_id]:
                    groups[tile_id][var] = {}
                if year not in groups[tile_id][var]:
                    groups[tile_id][var][year] = []
                
                groups[tile_id][var][year].append(f_path)
        return groups

    def group_files_simple(self) -> Dict:
        """Groups files strictly by Tile ID and Year."""
        groups = {}
        
        try:
            file_paths = list(UPath(self.input_dir).glob("*.tif"))
        except FileNotFoundError:
            file_paths = []

        for f_path in file_paths:
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

    def load_and_average(self, paths: List[Path]) -> xr.DataArray:
        """Loads a list of rasters, ALIGNS them to a common grid, and returns the mean."""
        das = [rioxarray.open_rasterio(p, chunks=None, masked=True).isel(band=0) for p in paths]
        
        if not das:
            raise ValueError("No file paths provided to load_and_average.")

        if len(das) == 1:
            return das[0]
        
        master = das[0]
        aligned_das = [master]

        for da in das[1:]:
            if da.rio.bounds() != master.rio.bounds() or da.shape != master.shape:
                da = da.rio.reproject_match(master)
            aligned_das.append(da)

        combined = xr.concat(aligned_das, dim="merge_dim")
        averaged = combined.mean(dim="merge_dim", keep_attrs=True, skipna=True)
        
        if master.rio.crs:
            averaged.rio.write_crs(master.rio.crs, inplace=True)
            
        return averaged

    def process(self) -> None:
        """Main function to find all bathy files and process them in parallel."""

        self.create_file_paths()

        main_output_dir = get_config_item('TERRAIN', 'OUTPUTS')
        dictionary_output_dir = self._join_paths(main_output_dir, "dictionaries")

        input_dir = self.input_dir
        filled_dir = self.output_dir

        keywords_to_exclude = ['tsm', 'hurr', 'sed', 'bluetopo']

        input_files = self._safe_ls(input_dir)

        lidar_data_paths = [
            f for f in input_files
            if f.lower().endswith(('.tif', '.tiff')) and not any(keyword in os.path.basename(f).lower() for keyword in keywords_to_exclude)
        ]        

        # 1. GAP FILL
        tasks = []
        for file in lidar_data_paths:
            task = dask.delayed(self.run_gap_fill(file, filled_dir, max_iters=3))
            tasks.append(task)
        dask.compute(*tasks)

        self.run_bathy_combination()

        # 2. FIND ALL BATHY FILES TO PROCESS
        # Re-list directory AFTER combination step to ensure we capture all created tifs
        filled_files = self._safe_ls(filled_dir)

        bathy_files_to_process = [f for f in filled_files]
        bathy_files_to_process.extend(
            [f for f in input_files 
            if f.endswith(('.tif', '.tiff')) and 'BlueTopo' in os.path.basename(f)]
        )

        vars_to_exclude = ["unc","slope", "rugosity", "bpi_fine", "bpi_broad", "terrain_classification", "survey_end_date"]
        bathy_files_to_process = [
            f for f in bathy_files_to_process 
            if not any(v in os.path.basename(f) for v in vars_to_exclude)
        ]

        print(f"Found {len(bathy_files_to_process)} bathymetry files to process.")

        best_radii = {'fine': (8, 32), 'broad': (80, 240)}
        
        # 3. RUN PHASE 1: PRE-COMPUTE DICTIONARIES
        self.create_regionally_consistent_dictionaries(bathy_files_to_process, best_radii, dictionary_output_dir)

        # 4. RUN PHASE 2: PARALLEL CLASSIFICATION
        print("\n--- PHASE 2: Parallel Processing of terrain products")
        tasks = []
        for bathy_file in bathy_files_to_process:
            task = dask.delayed(self.generate_terrain_products_python)(bathy_file, best_radii, dictionary_output_dir)
            tasks.append(task)
            
        dask.compute(*tasks)

        WORKING_DIR = get_config_item('TERRAIN', 'OUTPUTS')
        STATE_VARS = ["filled", "slope", "rugosity", "bpi_fine", "bpi_broad", "terrain_classification"]

        print(f"Starting raster processing in: {WORKING_DIR}")
        print(f"Finding files for: {', '.join(STATE_VARS)}")
        
        state_files_map = defaultdict(list)
        all_state_files_to_process = []

        valid_years = {str(year) for year_pair in self.year_ranges for year in year_pair}

        year_pattern_str = "|".join(valid_years)
        var_pattern_str = "|".join(STATE_VARS)

        FILE_PATTERN = re.compile(rf".*?_({year_pattern_str})\d*?_.*_({var_pattern_str}).*?$")

        working_dir_path = UPath(WORKING_DIR)
        if self.is_aws:
            bucket = get_config_item('S3', 'BUCKET_NAME').strip('/')
            out_dir = str(WORKING_DIR).replace("s3://", "").strip('/')
            if not out_dir.startswith(bucket):
                working_dir_path = UPath(f"s3://{bucket}/{out_dir}")
            else:
                working_dir_path = UPath(f"s3://{out_dir}")

        try:
            all_files = list(working_dir_path.rglob("*.tif"))
        except FileNotFoundError:
            all_files = []

        for f in all_files:
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
        
        print("\n--- Processing Complete ---")

    @dask.delayed
    def process_combination_task(self, paths: List[Path], out_path: Path) -> str:
        """Loads one or more rasters, averages them, and saves to new name."""
        if self._exists(out_path):
            return f"Skipped (Exists): {out_path.name}"

        try:
            da_avg = self.load_and_average(paths)
            da_avg.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

            da_avg = da_avg.fillna(-9999.0)
            da_avg.rio.write_nodata(-9999.0, inplace=True)

            da_avg.rio.to_raster(
                out_path,
                driver="GTiff",
                compress="LZW",
                tiled=True,
                windowed=True,
                nodata=-9999.0
            )
            return f"Created: {out_path.name}"
            
        except Exception as e:
            return f"Error on {out_path.name}: {str(e)}"

    @dask.delayed
    def process_delta_task(self, paths_t0: List[Path], paths_t1: List[Path], out_path: Path) -> str:
        """Calculates delta between two sets of file paths."""
        if self._exists(out_path):
            return f"Skipped (Exists): {out_path.name}"

        try:
            da_t0 = self.load_and_average(paths_t0)
            da_t1 = self.load_and_average(paths_t1)

            if da_t0.rio.bounds() != da_t1.rio.bounds():
                 da_t1 = da_t1.rio.reproject_match(da_t0)

            delta = da_t1 - da_t0
            delta.rio.write_crs(da_t0.rio.crs, inplace=True)
            delta.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

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

    def run_bathy_combination(self):
        """Orchestrates the finding, averaging, and renaming of bathy files."""
        groups = self.group_files_simple()
        delayed_tasks = []

        print(f"Scanning complete. Found groups for {len(groups)} tiles.")

        for tile_id, year_map in groups.items():
            for year, paths in year_map.items():
                
                out_name = f"combined{len(paths)}_bathy_{tile_id}_{year}.tif"
                # Use _join_paths for robust creation, then convert to Path/UPath
                out_path = UPath(self._join_paths(self.output_dir, out_name))
                
                task = self.process_combination_task(paths, out_path)
                delayed_tasks.append(task)

        if not delayed_tasks:
            print("No matching files found to process.")
            return

        print(f"Queued {len(delayed_tasks)} combination tasks. Computing...")
        results = dask.compute(*delayed_tasks)
        for res in results:
            print(res)

    def run_gap_fill(self, input_file, output_dir, max_iters) -> None:
        """
        The main entry point for the gap-filling process.
        param str input_file: Path to the input raster file.
        param str output_dir: Directory where the filled raster will be saved.
        param int max_iters: Maximum number of fill iterations to perform.
        return: None
        """
        print("Starting gap fill module...")

        output_file = self._join_paths(
            output_dir, os.path.splitext(os.path.basename(str(input_file)))[0] + "_filled.tif"
        )

        if self._exists(output_file):
            print(f"File already exists, skipping gap fill: {os.path.basename(str(output_file))}")
            return
        
        self.fill_with_fallback(
            input_file=input_file,
            output_file=output_file,
            max_iters=max_iters
        )

        print("Gap fill process complete.")

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