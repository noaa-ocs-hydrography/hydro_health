import os
# [MEMORY FIX 1]: Force glibc to release memory back to the OS immediately
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"

import re
import shutil
import tempfile
import itertools
import warnings
import gc
import ctypes
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

# [MEMORY FIX 4]: Prevent xarray from secretly holding onto file memory
xr.set_options(file_cache_maxsize=1)

import dask
import dask.array as da
from dask.distributed import Client, print as dask_print
from dask import delayed, compute
from scipy.ndimage import binary_erosion, uniform_filter, convolve

# Check for WhiteboxTools dependency
try:
    from whitebox import WhiteboxTools
except ImportError:
    print("\nCRITICAL ERROR: The 'whitebox' library is not installed.")
    print("Please install it by running: conda install whitebox\n")
    raise

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_environment


# ==============================================================================
#  WORKER SINGLETONS AND DASK WRAPPERS
# ==============================================================================
# By moving these functions to the top level, Dask no longer serializes
# the entire Class instance for each task. This eliminates the massive delay.

_worker_engine = None

def _get_worker_engine(overwrite, pilot_mode):
    """Initializes a singleton engine per Dask worker process to avoid serialization overhead."""
    global _worker_engine
    if _worker_engine is None:
        _worker_engine = CreateSeabedTerrainLayerEngine(overwrite=overwrite, pilot_mode=pilot_mode)
        _worker_engine.create_file_paths()
    # Always ensure overwrite flag is synced
    _worker_engine.overwrite = overwrite
    return _worker_engine

def _run_gap_fill_task(input_file, output_dir, max_iters, overwrite, pilot_mode):
    engine = _get_worker_engine(overwrite, pilot_mode)
    return engine.run_gap_fill(input_file, output_dir, max_iters)

def _run_combination_task(paths, out_path_str, overwrite, pilot_mode):
    engine = _get_worker_engine(overwrite, pilot_mode)
    return engine.process_combination_task(paths, out_path_str)

def _run_terrain_task(bathy_file, best_radii, dictionary_output_dir, main_output_dir, overwrite, pilot_mode):
    engine = _get_worker_engine(overwrite, pilot_mode)
    return engine.generate_terrain_products_python(bathy_file, best_radii, dictionary_output_dir, main_output_dir)


class ModelDataPreProcessor(Engine):
    """Class for parallel preprocessing all model data"""

    def __init__(self, overwrite: bool = False, pilot_mode: bool = False):
        super().__init__()
        self.target_crs = "EPSG:32617"
        self.target_res = 8
        self.pilot_mode = pilot_mode
        self.overwrite = overwrite


class CreateSeabedTerrainLayerEngine(ModelDataPreProcessor):
    """Class to hold the logic for processing the Seabed Terrain layer"""

    def __init__(self, overwrite: bool = False, pilot_mode: bool = False):
        # Inherit settings (including overwrite) from the preprocessor class
        super().__init__(overwrite=overwrite, pilot_mode=pilot_mode)
        self.is_aws = (get_environment() == 'aws')
        
        # Ensure our custom temp directory exists
        self.local_tmp_dir = Path.home() / "hydro_health_local_tmp"
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize s3fs auth purely for side-effect (don't store it on self to prevent Dask graph bloat)
        _ = s3fs.S3FileSystem(anon=False)
        
        # Initialize WhiteboxTools
        self.wbt = WhiteboxTools()
        self.wbt.verbose = False
        self.wbt.set_compress_rasters(True)

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
    #  STATE MANAGEMENT FOR DASK GRAPH EFFICIENCY
    # ==============================================================================

    def __getstate__(self):
        """
        Called automatically by Dask/cloudpickle before distributing tasks to workers.
        Removes large, cached, or unpicklable state to prevent Dask graph bloat.
        """
        state = self.__dict__.copy()
        
        # Drop WhiteboxTools instance and re-init on workers to be absolutely safe
        if 'wbt' in state:
            del state['wbt']
            
        return state

    def __setstate__(self, state):
        """
        Called automatically by Dask/cloudpickle on the worker nodes.
        Re-initializes the dropped objects with fresh caches so the worker can process the task.
        """
        self.__dict__.update(state)
        
        # Guarantee the local temp directory exists physically on this specific worker node
        if hasattr(self, 'local_tmp_dir'):
            self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Re-initialize s3fs auth side-effect
        _ = s3fs.S3FileSystem(anon=False)
        
        # Re-initialize WhiteboxTools locally on the worker
        self.wbt = WhiteboxTools()
        self.wbt.verbose = False
        self.wbt.set_compress_rasters(True)

    # ==============================================================================
    #  PRIVATE HELPER FUNCTIONS
    # ==============================================================================

    def _save_raster_da(self, da: xr.DataArray, out_path: str, **kwargs):
        """Safely saves an xarray DataArray to S3 by writing locally first."""
        out_u = UPath(out_path)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=str(self.local_tmp_dir)) as tmp_file:
            local_tmp_path = tmp_file.name

        try:
            # [MEMORY FIX 6]: Run the actual rasterization single-threaded to prevent internal hidden threads
            with dask.config.set(scheduler='single-threaded'):
                da.rio.to_raster(local_tmp_path, **kwargs)
            
            if out_u.protocol == "s3":
                out_u.fs.put(local_tmp_path, str(out_u))
            else:
                shutil.copyfile(local_tmp_path, str(out_path))
            
            # Standard print goes to worker log safely
            print(f"Successfully wrote raster DataArray file to: {out_path}")
        finally:
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)

    def _save_numpy_to_raster(self, data_array: np.ndarray, out_path: str, profile: dict):
        """Safely saves a numpy array to S3 by writing locally first."""
        out_u = UPath(out_path)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=str(self.local_tmp_dir)) as tmp_file:
            local_tmp_path = tmp_file.name
            
        try:
            with rasterio.open(local_tmp_path, 'w', **profile) as dst:
                dst.write(data_array, 1)
                
            if out_u.protocol == "s3":
                out_u.fs.put(local_tmp_path, str(out_u))
            else:
                shutil.copyfile(local_tmp_path, str(out_path))
                
            # Standard print goes to worker log safely
            print(f"Successfully wrote NumPy array file to: {out_path}")
        finally:
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)

    def _exists(self, path) -> bool:
        """Checks if a file exists, natively compatible with both local and S3 paths via UPath."""
        return UPath(path).exists()

    def _get_tile_id(self, filename: str) -> Optional[str]:
        """Extracts the 8-character alphanumeric Tile ID."""
        candidates = re.findall(r"B[A-Z0-9]{7}", filename)
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
        """Calculates the Bathymetric Position Index chunked purely in Dask to prevent massive mem allocation."""
        inner_cells = int(round(inner_radius / cell_size))
        outer_cells = int(round(outer_radius / cell_size))
        
        y, x = np.ogrid[-outer_cells:outer_cells + 1, -outer_cells:outer_cells + 1]
        mask = x**2 + y**2 <= outer_cells**2
        mask[x**2 + y**2 <= inner_cells**2] = False
        
        kernel = mask.astype(np.float32)
        
        chunk_size = 1024
        d_bathy = da.from_array(bathy_array, chunks=(chunk_size, chunk_size))
        
        # Lazy map blocks replaces massive in-memory valid_mask array
        d_valid = da.map_blocks(lambda b: (~np.isnan(b)).astype(np.float32), d_bathy, dtype=np.float32)
        
        # Lazy zero fill replaces massive in-memory zero array
        d_bathy_zeroed = da.where(da.isnan(d_bathy), 0.0, d_bathy)
        
        def _conv(block):
            return convolve(block, kernel, mode='mirror')
            
        # [MEMORY FIX 2]: Use scheduler='single-threaded' inside the worker task 
        # to prevent thread-arena memory bloat from "nested Dask".
        sum_array = d_bathy_zeroed.map_overlap(_conv, depth=outer_cells, boundary='reflect')
        count_array = d_valid.map_overlap(_conv, depth=outer_cells, boundary='reflect')
        
        mean_annulus = da.where(count_array > 0, sum_array / count_array, np.nan)
        bpi_lazy = d_bathy - mean_annulus
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return bpi_lazy.compute(scheduler='single-threaded')

    def calculate_slope(self, bathy_array, cell_size) -> np.ndarray:
        """Calculates slope (in degrees) using inplace operations to minimize memory."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            # Manual float32 gradient using slicing prevents upcasting to float64
            gy = np.empty_like(bathy_array, dtype=np.float32)
            gx = np.empty_like(bathy_array, dtype=np.float32)
            
            # Y-gradient
            gy[1:-1, :] = (bathy_array[2:, :] - bathy_array[:-2, :]) / (2 * cell_size)
            gy[0, :] = (bathy_array[1, :] - bathy_array[0, :]) / cell_size
            gy[-1, :] = (bathy_array[-1, :] - bathy_array[-2, :]) / cell_size
            
            # X-gradient
            gx[:, 1:-1] = (bathy_array[:, 2:] - bathy_array[:, :-2]) / (2 * cell_size)
            gx[:, 0] = (bathy_array[:, 1] - bathy_array[:, 0]) / cell_size
            gx[:, -1] = (bathy_array[:, -1] - bathy_array[:, -2]) / cell_size
            
            np.square(gx, out=gx)
            np.square(gy, out=gy)
            gx += gy  # gx now holds gx^2 + gy^2
            
            del gy    # Free memory immediately
            
            np.sqrt(gx, out=gx)
            np.arctan(gx, out=gx)
            slope_deg = np.degrees(gx, out=gx)
            
        return slope_deg

    def calculate_tri(self, bathy_array) -> np.ndarray:
        """Calculates Terrain Ruggedness Index (TRI) using vectorized slices natively."""
        tri_sum = np.zeros_like(bathy_array, dtype=np.float32)
        valid_count = np.zeros_like(bathy_array, dtype=np.float32)
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                    
                y1 = max(0, dy)
                y2 = bathy_array.shape[0] + min(0, dy)
                x1 = max(0, dx)
                x2 = bathy_array.shape[1] + min(0, dx)
                
                sy1 = max(0, -dy)
                sy2 = bathy_array.shape[0] + min(0, -dy)
                sx1 = max(0, -dx)
                sx2 = bathy_array.shape[1] + min(0, -dx)
                
                neighbor = bathy_array[y1:y2, x1:x2]
                center = bathy_array[sy1:sy2, sx1:sx2]
                
                diff = neighbor - center
                np.abs(diff, out=diff)
                
                invalid_mask = np.isnan(neighbor)
                diff[invalid_mask] = 0.0
                
                tri_sum[sy1:sy2, sx1:sx2] += diff
                valid_count[sy1:sy2, sx1:sx2] += (~invalid_mask).astype(np.float32)
                
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            tri = np.where(valid_count > 0, tri_sum / valid_count, np.nan)
            
        return tri

    def create_classification_dictionary(self, bpi_broad_std_sample, bpi_fine_std_sample, slope_sample) -> pd.DataFrame:
        """Creates a data-driven classification dictionary from representative sample arrays."""
        valid_broad = bpi_broad_std_sample[~np.isnan(bpi_broad_std_sample)]
        valid_fine = bpi_fine_std_sample[~np.isnan(bpi_fine_std_sample)]
        valid_slope = slope_sample[~np.isnan(slope_sample)]

        broad_breaks = np.nanquantile(valid_broad, [0.15, 0.85]) if len(valid_broad) > 0 else [np.nan, np.nan]
        fine_breaks = np.nanquantile(valid_fine, [0.15, 0.85]) if len(valid_fine) > 0 else [np.nan, np.nan]
        slope_break = np.nanquantile(valid_slope, 0.85) if len(valid_slope) > 0 else np.nan

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
        uncombined_dir = get_config_item('MODEL', 'UNCOMBINED_LIDAR_DIR')
        processed_dir = get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR')

        if get_environment() == 'remote':
            # Strictly enforcing strings directly off UPath evaluation keeps Dask payload clean
            self.filled_dir = str(UPath(filled_dir))
            self.combined = str(UPath(combined_dir))
            self.uncombined_dir = str(UPath(uncombined_dir))
            self.processed_dir = str(UPath(processed_dir))

        elif get_environment() == 'aws':
            bucket = get_config_item('S3', 'BUCKET_NAME').strip('/')
            base_path = UPath(f"s3://{bucket}")
            self.filled_dir = str(base_path / filled_dir.strip('/'))
            self.combined = str(base_path / combined_dir.strip('/'))
            self.uncombined_dir = str(base_path / uncombined_dir.strip('/'))
            self.processed_dir = str(base_path / processed_dir.strip('/'))

    def create_regionally_consistent_dictionaries(self, all_files, best_radii, output_dir, max_sample_files=10, pixels_per_file=20000) -> None:
        """Scans files, groups by year, samples pixels, and creates dictionaries."""
        print("\n--- PHASE 1: Creating Regionally Consistent Dictionaries ---")
        print(f"Dictionaries will be written to: {output_dir}")
        out_path = UPath(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        valid_years = {year for year_pair in self.year_ranges for year in year_pair}
        print(f"DEBUG: Valid years mapped from engine.year_ranges: {sorted(list(valid_years))}")   

        # 1. Group files by year
        year_groups = {}
        all_found_years = set()
        
        for f in all_files:
            match = re.search(r'((?:19|20)\d{2})', os.path.basename(str(f)))
            if match:
                extracted_year = int(match.group(1))
                all_found_years.add(extracted_year)
                
                if extracted_year in valid_years:
                    year = match.group(1)
                    if year not in year_groups:
                        year_groups[year] = []
                    year_groups[year].append(f)
                    
        print(f"DEBUG: All actual years extracted from filenames: {sorted(list(all_found_years))}")
        
        # Handle the generic 'BlueTopo.tif' case (using case-insensitive matching)
        generic_bathy = [f for f in all_files if 'bluetopo' in os.path.basename(str(f)).lower()]
        if generic_bathy:
            year_groups['BlueTopo'] = generic_bathy
            
        print(f"DEBUG: Final grouped year-keys intended for processing: {list(year_groups.keys())}")

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
                    with rasterio.open(str(f)) as src:
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
                        slope_sample = self.calculate_slope(bathy_array, cell_size)
                        print("      - Calculating BPI for sampled pixels...")
                        
                        bpi_fine_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                        bpi_broad_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                        
                        print("      - Collecting sampled values...")
                        rows, cols = sample_indices[:, 0], sample_indices[:, 1]
                        all_samples['slope'].append(slope_sample[rows, cols])
                        all_samples['bpi_fine_std'].append(self.standardize_raster_array(bpi_fine_sample)[rows, cols])
                        all_samples['bpi_broad_std'].append(self.standardize_raster_array(bpi_broad_sample)[rows, cols])

                        # Explicitly release large arrays 
                        del bathy_array, slope_sample, bpi_fine_sample, bpi_broad_sample
                        gc.collect()

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

    def fill_with_fallback(self, input_file, output_file, max_iters=5, chunk_size=512) -> None:
        """Performs chunked iterative focal fill on a raster file using Dask and rioxarray."""
        print(f"Attempting chunked fill for {os.path.basename(str(input_file))}")

        da_chunk = {"x": chunk_size, "y": chunk_size}
        ds = rioxarray.open_rasterio(str(input_file), chunks=da_chunk)
        nodata = ds.rio.nodata
        da = ds.squeeze().astype("float32")
        da = da.where(da != nodata)
        
        print("Checking for interior gaps...")
        
        # [MEMORY FIX 6]: single threaded compute on boolean mask
        nan_mask = da.isnull().compute(scheduler='single-threaded')
        
        if not binary_erosion(nan_mask, structure=np.ones((3,3))).any():
            print(f"No interior gaps found in {os.path.basename(str(input_file))}. Skipping fill process.")
            src = UPath(input_file)
            dst = UPath(output_file)
            if src.protocol == "s3":
                src.fs.copy(str(src), str(dst))
            else:
                shutil.copyfile(src, dst)
            
            dask_print(f"✅ Copied/skipped gap fill for: {os.path.basename(str(output_file))}")
            return

        for i in range(max_iters):
            da_prev = da
            
            # Use dask.config context around ufunc to prevent threaded bloat
            with dask.config.set(scheduler='single-threaded'):
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
        
        self._save_raster_da(da, output_file, compress='LZW')
        dask_print(f"✅ Filled raster process completed for: {os.path.basename(str(output_file))}")

    def focal_fill_block(self, block: np.ndarray, w=3) -> np.ndarray:
        """Performs a single, efficient, nan-aware focal mean on a NumPy array block."""
        block = block.astype(np.float32)
        nan_mask = np.isnan(block)

        data_sum = uniform_filter(np.nan_to_num(block, nan=0.0), size=w, mode="constant", cval=0.0)
        valid_count = uniform_filter((~nan_mask).astype(np.float32), size=w, mode="constant", cval=0.0)

        with np.errstate(invalid='ignore', divide='ignore'):
            filled = data_sum / valid_count

        return np.where(nan_mask, filled, block)

    def generate_neighborhood_statistics(self, file_path: Path) -> None:
        """Calculates focal mean and standard deviation for a given raster file."""
        print(f"Calculating neighborhood statistics for: {file_path.name}")
        size = 3

        base_name = file_path.stem 
        out_dir = file_path.parent
        
        out_sd = out_dir / f"{base_name}_sd{size}.tif"
        out_mean = out_dir / f"{base_name}_mean{size}.tif"

        if 'mean' in file_path.name or (not self.overwrite and self._exists(out_sd)):
            print(f"Output files already exist, skipping: {out_sd.name}")
            return 
        
        if 'sd3' in file_path.name or (not self.overwrite and self._exists(out_mean)):
            print(f"Output files already exist, skipping: {out_mean.name}")
            return
        
        rds = rioxarray.open_rasterio(str(file_path), chunks=True).isel(band=0)
        
        window = rds.rolling(x=size, y=size, center=True)
        r_mean = window.mean()
        r_sd = window.std()
        
        self._save_raster_da(r_mean, str(out_mean), driver="GTiff", compress="LZW")
        self._save_raster_da(r_sd, str(out_sd), driver="GTiff", compress="LZW")

    def generate_terrain_products_python(self, bathy_path, best_radii, dictionary_dir, main_output_dir) -> str:
        """Main function to process one bathymetry raster."""
        base_name = os.path.splitext(os.path.basename(str(bathy_path)))[0]
        output_dir = self._join_paths(main_output_dir, 'BTM_outputs')
        
        out_dir_path = UPath(output_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        out_slope_deg = self._join_paths(output_dir, base_name + "_slope_deg.tif")
        out_gradmag   = self._join_paths(output_dir, base_name + "_gradmag.tif") 
        out_flowdir   = self._join_paths(output_dir, base_name + "_flowdir.tif") 
        out_prof      = self._join_paths(output_dir, base_name + "_curv_profile.tif")
        out_plan      = self._join_paths(output_dir, base_name + "_curv_plan.tif")
        out_total     = self._join_paths(output_dir, base_name + "_curv_total.tif")
        out_tci       = self._join_paths(output_dir, base_name + "_tci.tif")
        out_flowacc   = self._join_paths(output_dir, base_name + "_flowacc.tif")
        out_shear     = self._join_paths(output_dir, base_name + "_shearproxy.tif")
        
        with tempfile.TemporaryDirectory(dir=str(self.local_tmp_dir)) as tmpdir:
            local_bathy = os.path.join(tmpdir, "bathy.tif")
            local_slope = os.path.join(tmpdir, "slope_deg.tif")
            local_gradmag = os.path.join(tmpdir, "gradmag.tif")
            local_flowdir = os.path.join(tmpdir, "flowdir.tif")
            local_prof = os.path.join(tmpdir, "prof.tif")
            local_plan = os.path.join(tmpdir, "plan.tif")
            local_total = os.path.join(tmpdir, "total.tif")
            local_tci = os.path.join(tmpdir, "tci.tif")
            local_flowacc = os.path.join(tmpdir, "flowacc.tif")

            outputs_wbt = [
                (out_slope_deg, lambda i, o: self.wbt.slope(i, o, units="degrees"), local_slope),
                (out_gradmag, lambda i, o: self.wbt.slope(i, o, units="radians"), local_gradmag),
                (out_flowdir, self.wbt.aspect, local_flowdir),
                (out_prof, self.wbt.profile_curvature, local_prof),
                (out_plan, self.wbt.plan_curvature, local_plan),
                (out_total, self.wbt.total_curvature, local_total),
                (out_flowacc, lambda i, o: self.wbt.d8_flow_accumulation(i, o, out_type="cells"), local_flowacc)
            ]

            missing_wbt = [item for item in outputs_wbt if self.overwrite or not self._exists(item[0])]
            missing_tci = self.overwrite or not self._exists(out_tci)
            missing_shear = self.overwrite or not self._exists(out_shear)

            numpy_outputs = {
                "_rugosity_tri.tif": None,
                "_bpi_fine_std.tif": None, 
                "_bpi_broad_std.tif": None,
                "_terrain_classification.tif": None
            }
            missing_numpy = any(self.overwrite or not self._exists(self._join_paths(output_dir, base_name + suffix)) for suffix in numpy_outputs.keys())

            needs_processing = missing_wbt or missing_tci or missing_shear or missing_numpy
            if not needs_processing:
                print(f"Skipped (All exist): {base_name}")
                return f"Skipped (All exist): {base_name}"

            print(f"Processing terrain classification for: {base_name}")

            with UPath(bathy_path).open('rb') as f_in:
                with open(local_bathy, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            if missing_wbt:
                print(f"[{base_name}] Generating {len(missing_wbt)} required WBT layer(s)...")
            for out_s3, wbt_func, local_out in missing_wbt:
                try:
                    wbt_func(local_bathy, local_out)
                    if os.path.exists(local_out):
                        with open(local_out, 'rb') as f_in, UPath(out_s3).open('wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        print(f"Successfully wrote WBT layer file to: {out_s3}")
                except Exception as e:
                    dask_print(f"❌ WBT Error on {base_name} for {out_s3}: {e}")

            if missing_tci:
                print(f"[{base_name}] Generating TCI layer...")
                try: 
                    self.wbt.convergence_index(local_bathy, local_tci)
                except AttributeError:
                    try:
                        self.wbt.run_tool("ConvergenceIndex", [f"--dem={local_bathy}", f"--output={local_tci}"])
                    except Exception as e:
                        dask_print(f"❌ WBT TCI RunTool Error on {base_name}: {e}")
                except Exception as e: 
                    dask_print(f"❌ WBT TCI Error on {base_name}: {e}")

                if os.path.exists(local_tci):
                    with open(local_tci, 'rb') as f_in, UPath(out_tci).open('wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    print(f"Successfully wrote TCI layer file to: {out_tci}")

            if missing_shear:
                print(f"[{base_name}] Generating Shear Proxy layer...")
                try:
                    slope_src = local_slope if os.path.exists(local_slope) else None
                    plan_src = local_plan if os.path.exists(local_plan) else None

                    if not slope_src and self._exists(out_slope_deg):
                        slope_src = local_slope
                        with UPath(out_slope_deg).open('rb') as f_in, open(slope_src, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                            
                    if not plan_src and self._exists(out_plan):
                        plan_src = local_plan
                        with UPath(out_plan).open('rb') as f_in, open(plan_src, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    if slope_src and plan_src:
                        with rasterio.open(slope_src) as s, rasterio.open(plan_src) as p:
                            slope_arr = s.read(1)
                            plan_arr = p.read(1)
                            meta = s.meta.copy()
                        
                        shear = slope_arr * np.abs(plan_arr)
                        meta.update(compress='LZW')
                        self._save_numpy_to_raster(shear.astype("float32"), out_shear, meta)
                    else:
                        dask_print(f"⚠️ Skipping Shear Proxy for {base_name}: Inputs missing (could not resolve local copies)")
                except Exception as e:
                    dask_print(f"❌ Shear Proxy Calc Error {base_name}: {e}")

            if missing_numpy:
                print(f"[{base_name}] Generating NumPy BTM classification layers...")
                
                # Fix parsing to ensure BlueTopo files look for the correct dictionary
                if 'bluetopo' in base_name.lower():
                    year = 'BlueTopo'
                else:
                    year = 'bt_bathy'
                    match = re.search(r'((?:19|20)\d{2})', base_name)
                    if match:
                        year = match.group(1)
                    
                dict_path = UPath(self._join_paths(dictionary_dir, f"dictionary_{year}.csv"))
                if not dict_path.exists():
                    # Fallback to case-insensitive scan of the directory
                    dict_files = self._safe_ls(dictionary_dir)
                    found = False
                    expected_suffix = f"dictionary_{year.lower()}.csv"
                    for df in dict_files:
                        if df.lower().endswith(expected_suffix):
                            dict_path = UPath(df)
                            found = True
                            break
                    if not found:
                        dask_print(f"❌ Dictionary missing for {year} on {base_name}")
                        return f"Dictionary missing for {year}"
                
                with dict_path.open('r') as fh:
                    unique_dictionary = pd.read_csv(fh)

                # ==========================================================
                # MEMORY CONSERVATIVE PROCESSING VIA MEMMAP AND CHUNKING
                # ==========================================================
                with rasterio.open(local_bathy) as src:
                    nodata_val = src.nodata
                    profile = src.profile
                    cell_size = src.res[0]
                    shape_2d = (src.height, src.width)
                    
                    # Create memory mapped array to prevent loading 1.6GB+ into RAM
                    bathy_array = np.memmap(os.path.join(tmpdir, "bathy.dat"), dtype='float32', mode='w+', shape=shape_2d)
                    src.read(1, out=bathy_array)
                    if nodata_val is not None:
                        bathy_array[bathy_array == nodata_val] = np.nan

                # Process 1: Rugosity
                out_path_rug = self._join_paths(output_dir, base_name + "_rugosity_tri.tif")
                if self.overwrite or not self._exists(out_path_rug):
                    rugosity = self.calculate_tri(bathy_array)
                    profile.update(dtype=rugosity.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    self._save_numpy_to_raster(rugosity, out_path_rug, profile)
                    del rugosity  # FREE MEMORY EARLY
                    gc.collect()

                # Process 2: Slope (Memmapped to avoid RAM accumulation)
                slope_raw = self.calculate_slope(bathy_array, cell_size)
                slope = np.memmap(os.path.join(tmpdir, "slope.dat"), dtype='float32', mode='w+', shape=shape_2d)
                slope[:] = slope_raw[:]
                del slope_raw
                gc.collect()

                # Process 3: Fine BPI
                out_path_fine = self._join_paths(output_dir, base_name + "_bpi_fine_std.tif")
                if self.overwrite or not self._exists(out_path_fine):
                    bpi_fine = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                    bpi_fine_std_raw = self.standardize_raster_array(bpi_fine)
                    del bpi_fine
                    
                    bpi_fine_std = np.memmap(os.path.join(tmpdir, "fine.dat"), dtype='float32', mode='w+', shape=shape_2d)
                    bpi_fine_std[:] = bpi_fine_std_raw[:]
                    del bpi_fine_std_raw
                    
                    profile.update(dtype=bpi_fine_std.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    self._save_numpy_to_raster(bpi_fine_std, out_path_fine, profile)
                    gc.collect()
                else:
                    # If skipped, load a read-only memmap for classification
                    with rasterio.open(out_path_fine) as src_f:
                        bpi_fine_std = np.memmap(os.path.join(tmpdir, "fine.dat"), dtype='float32', mode='w+', shape=shape_2d)
                        src_f.read(1, out=bpi_fine_std)

                # Process 4: Broad BPI
                out_path_broad = self._join_paths(output_dir, base_name + "_bpi_broad_std.tif")
                if self.overwrite or not self._exists(out_path_broad):
                    bpi_broad = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                    bpi_broad_std_raw = self.standardize_raster_array(bpi_broad)
                    del bpi_broad
                    
                    bpi_broad_std = np.memmap(os.path.join(tmpdir, "broad.dat"), dtype='float32', mode='w+', shape=shape_2d)
                    bpi_broad_std[:] = bpi_broad_std_raw[:]
                    del bpi_broad_std_raw
                    
                    profile.update(dtype=bpi_broad_std.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    self._save_numpy_to_raster(bpi_broad_std, out_path_broad, profile)
                else:
                    # If skipped, load a read-only memmap for classification
                    with rasterio.open(out_path_broad) as src_b:
                        bpi_broad_std = np.memmap(os.path.join(tmpdir, "broad.dat"), dtype='float32', mode='w+', shape=shape_2d)
                        src_b.read(1, out=bpi_broad_std)

                # WE ARE COMPLETELY DONE WITH BATHY ARRAY - HUGE MEMORY WIN
                del bathy_array
                gc.collect()
                
                # Process 5: Terrain Classification
                out_path_class = self._join_paths(output_dir, base_name + "_terrain_classification.tif")
                if self.overwrite or not self._exists(out_path_class):
                    # Memmap classified array and iterate in chunks to prevent boolean math RAM spikes
                    classified_array = np.memmap(os.path.join(tmpdir, "class.dat"), dtype='float32', mode='w+', shape=shape_2d)
                    classified_array[:] = 0.0
                    
                    chunk_s = 2048
                    for i in range(0, shape_2d[0], chunk_s):
                        for j in range(0, shape_2d[1], chunk_s):
                            s_chunk = slope[i:i+chunk_s, j:j+chunk_s]
                            b_chunk = bpi_broad_std[i:i+chunk_s, j:j+chunk_s]
                            f_chunk = bpi_fine_std[i:i+chunk_s, j:j+chunk_s]
                            c_chunk = classified_array[i:i+chunk_s, j:j+chunk_s]
                            
                            for index, rule in unique_dictionary.iterrows():
                                matches = (
                                    (b_chunk >= rule['BroadBPI_Lower']) & (b_chunk <= rule['BroadBPI_Upper']) &
                                    (f_chunk >= rule['FineBPI_Lower']) & (f_chunk <= rule['FineBPI_Upper']) &
                                    (s_chunk >= rule['Slope_Lower']) & (s_chunk <= rule['Slope_Upper'])
                                )
                                c_chunk[matches & (c_chunk == 0)] = rule['Class_ID']
                        
                    profile.update(dtype=classified_array.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    self._save_numpy_to_raster(classified_array, out_path_class, profile)
                    del classified_array

                # Final cleanup
                del slope, bpi_fine_std, bpi_broad_std
                gc.collect()

        # Target 1 status print over the network per successful file
        dask_print(f"✅ Successfully finished terrain processing: {base_name}")
        
        # [MEMORY FIX 2]: Force OS to reclaim the memory pool at the end of heavy worker tasks
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
            
        return f"Success: {base_name}"

    def group_files(self) -> Dict:
        """Groups files by Tile and Variable, then by Year."""
        groups = {}
        try:
            file_paths = list(UPath(self.filled_dir).rglob("*.tif"))
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
        filled_dir_path = UPath(self.filled_dir)

        try:
            if hasattr(filled_dir_path, 'fs'):
                filled_dir_path.fs.invalidate_cache(filled_dir_path.path) 
        except Exception:
            pass 

        try:
            file_paths = [
                p for p in filled_dir_path.iterdir() 
                if p.suffix.lower() in ('.tif', '.tiff')
            ]
        except (NotADirectoryError, FileNotFoundError):
            print(f"Directory {self.filled_dir} does not exist yet or is empty. Proceeding with 0 existing files.")
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
                groups[tile_id][year].append(str(f_path))
        return groups

    # [MEMORY FIX 3]: Close datasets cleanly to release open file handles in xarray
    def load_and_average(self, paths: List[str]) -> xr.DataArray:
        """Loads a list of rasters, ALIGNS them to a common grid, and returns the mean."""
        das = []
        for p in paths:
            # Masked=True can be memory intensive, be cautious
            da = rioxarray.open_rasterio(str(p), chunks={"x": 512, "y": 512}, masked=True).isel(band=0)
            das.append(da)
        
        if not das:
            raise ValueError("No file paths provided to load_and_average.")

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
            
        # Force computation to memory so we can close the file handles
        with dask.config.set(scheduler='single-threaded'):
            averaged.load() 
            
        # Close all opened file handles to free unmanaged memory
        for da in das:
            da.close()
            
        return averaged

    def process(self) -> None:
        """Main function to find all bathy files and process them in parallel."""
        self.create_file_paths()

        raw_output_dir = get_config_item('TERRAIN', 'OUTPUTS')
        if self.is_aws:
            bucket = get_config_item('S3', 'BUCKET_NAME').strip('/')
            clean_out_dir = str(raw_output_dir).replace("s3://", "").strip('/')
            if not clean_out_dir.startswith(bucket):
                main_output_dir = f"s3://{bucket}/{clean_out_dir}"
            else:
                main_output_dir = f"s3://{clean_out_dir}"
        else:
            main_output_dir = raw_output_dir

        dictionary_output_dir = self._join_paths(main_output_dir, "dictionaries")
        print(f"Dictionaries path set to: {dictionary_output_dir}")

        keywords_to_exclude = ['tsm', 'hurr', 'sed', 'bluetopo']
        input_files = self._safe_ls(self.uncombined_dir) 

        valid_input_files = [
            f for f in input_files
            if str(f).lower().endswith(('.tif', '.tiff')) and not any(keyword in UPath(f).name.lower() for keyword in keywords_to_exclude)
        ]

        lidar_data_paths = []
        for f in valid_input_files:
            expected_filename = os.path.splitext(UPath(f).name)[0] + "_filled.tif"
            expected_output_path = UPath(self.filled_dir) / expected_filename
            if self.overwrite or not expected_output_path.exists():
                lidar_data_paths.append(f)

        print(f"Found {len(lidar_data_paths)} new lidar files to gap fill (skipped {len(valid_input_files) - len(lidar_data_paths)} existing).")

        if lidar_data_paths:
            print("Starting gap fill process in parallel...")
            tasks = []
            for file in lidar_data_paths:
                # OPTIMIZATION: Dask delayed now targets the top-level wrapper function
                task = dask.delayed(_run_gap_fill_task)(str(file), str(self.filled_dir), 3, self.overwrite, self.pilot_mode)
                tasks.append(task)
            
            dask.compute(*tasks)
        else:
            print("All files have already been gap-filled. Skipping computation.")

        print("Gap fill process complete. Starting bathymetry combination...")
        self.run_bathy_combination()

        filled_files = self._safe_ls(self.combined)
        bathy_files_to_process = [f for f in filled_files]

        vars_to_exclude = ["unc", "slope", "rugosity", "bpi_fine", "bpi_broad", "terrain_classification", "survey_end_date"]

        found_bluetopo_files = [
            f for f in UPath(self.processed_dir).glob("*.tif*")
            if f.suffix.lower() in {'.tif', '.tiff'} 
            and 'bluetopo' in f.name.lower()
            and not any(v in f.name for v in vars_to_exclude)
        ]
        
        bathy_files_to_process.extend(found_bluetopo_files)
        print(f"{len(filled_files)} lidar and {len(found_bluetopo_files)} Bluetopo files found for dictionaries.")

        best_radii = {'fine': (8, 32), 'broad': (80, 240)}
        # self.create_regionally_consistent_dictionaries(bathy_files_to_process, best_radii, dictionary_output_dir)

        print("\n--- PHASE 2: Pre-flight Check for Terrain Products ---")
        tasks = []
        
        # Accumulators for summary output
        total_missing_wbt = 0
        total_missing_tci = 0
        total_missing_shear = 0
        total_missing_numpy = 0
        total_skipped = 0

        # Optimization: Fetch existing files once to avoid repeated network calls
        output_dir = self._join_paths(main_output_dir, 'BTM_outputs')
        existing_files = {UPath(f).name for f in self._safe_ls(output_dir)}

        # Sequentially check all files BEFORE queueing Dask tasks
        for bathy_file in bathy_files_to_process:
            base_name = os.path.splitext(os.path.basename(str(bathy_file)))[0]

            wbt_suffixes = [
                "_slope_deg.tif", "_gradmag.tif", "_flowdir.tif", 
                "_curv_profile.tif", "_curv_plan.tif", "_curv_total.tif", "_flowacc.tif"
            ]
            
            # Sub-second O(1) set lookup instead of self._exists()
            missing_wbt = [s for s in wbt_suffixes if self.overwrite or (base_name + s) not in existing_files]
            missing_tci = self.overwrite or (base_name + "_tci.tif") not in existing_files
            missing_shear = self.overwrite or (base_name + "_shearproxy.tif") not in existing_files

            numpy_outputs = ["_rugosity_tri.tif", "_bpi_fine_std.tif", "_bpi_broad_std.tif", "_terrain_classification.tif"]
            missing_numpy = any(self.overwrite or (base_name + suffix) not in existing_files for suffix in numpy_outputs)

            total_missing_wbt += len(missing_wbt)
            total_missing_tci += 1 if missing_tci else 0
            total_missing_shear += 1 if missing_shear else 0
            total_missing_numpy += 1 if missing_numpy else 0

            needs_processing = len(missing_wbt) > 0 or missing_tci or missing_shear or missing_numpy

            if needs_processing:
                # OPTIMIZATION: Dask delayed now targets the top-level wrapper function 
                task = dask.delayed(_run_terrain_task)(
                    str(bathy_file), 
                    best_radii, 
                    str(dictionary_output_dir), 
                    str(main_output_dir), 
                    self.overwrite, 
                    self.pilot_mode
                )
                tasks.append(task)
            else:
                total_skipped += 1
                
        print(f"Total bathymetry files checked: {len(bathy_files_to_process)}")
        print(f"Total missing WBT layers:   {total_missing_wbt}")
        print(f"Total missing TCI layers:   {total_missing_tci}")
        print(f"Total missing Shear layers: {total_missing_shear}")
        print(f"Total missing NumPy layers: {total_missing_numpy}")
        print(f"Files skipped (All exist):  {total_skipped}")
            
        print(f"\n--- Queuing {len(tasks)} parallel tasks for terrain processing ---")
        if tasks:
            results = dask.compute(*tasks)
            # The return statements from the delayed functions will print here in the main thread
            for res in results:
                print(res)

        WORKING_DIR = main_output_dir
        STATE_VARS = ["filled", "slope", "rugosity", "bpi_fine", "bpi_broad", "terrain_classification"]

        print(f"\nStarting raster processing in: {WORKING_DIR}")
        print(f"Finding files for: {', '.join(STATE_VARS)}")
        
        state_files_map = defaultdict(list)
        all_state_files_to_process = []

        valid_years = {str(year) for year_pair in self.year_ranges for year in year_pair}
        year_pattern_str = "|".join(valid_years)
        var_pattern_str = "|".join(STATE_VARS)

        FILE_PATTERN = re.compile(rf".*?_({year_pattern_str})\d*?_.*_({var_pattern_str}).*?$")
        working_dir_path = UPath(WORKING_DIR)

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
        print(f"Found {len(state_files_map)} variable groups for calculation.")
        
        print("\n--- Processing Complete ---")

    def process_combination_task(self, paths: List[str], out_path_str: str) -> str:
        """Loads one or more rasters, averages them, and saves to new name."""
        out_path = UPath(out_path_str)
        if not self.overwrite and self._exists(out_path):
            return f"Skipped (Exists): {out_path.name}"

        da_avg = None # initialize early for finally block cleanup safety
        try:
            da_avg = self.load_and_average(paths)
            da_avg.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

            da_avg = da_avg.fillna(-9999.0)
            da_avg.rio.write_nodata(-9999.0, inplace=True)

            self._save_raster_da(
                da_avg, 
                str(out_path),
                driver="GTiff",
                compress="LZW",
                tiled=True,
                windowed=True,
                nodata=-9999.0
            )
            dask_print(f"✅ Combined: {out_path.name}")
            return f"Created: {out_path.name}"
            
        except Exception as e:
            dask_print(f"❌ Error combining {out_path.name}: {str(e)}")
            return f"Error on {out_path.name}: {str(e)}"
        
        finally:
            if da_avg is not None:
                da_avg.close()

    def process_delta_task(self, paths_t0: List[str], paths_t1: List[str], out_path_str: str) -> str:
        """Calculates delta between two sets of file paths."""
        out_path = UPath(out_path_str)
        if not self.overwrite and self._exists(out_path):
            return f"Skipped (Exists): {out_path.name}"

        try:
            da_t0 = self.load_and_average(paths_t0)
            da_t1 = self.load_and_average(paths_t1)

            if da_t0.rio.bounds() != da_t1.rio.bounds():
                 da_t1 = da_t1.rio.reproject_match(da_t0)

            delta = da_t1 - da_t0
            delta.rio.write_crs(da_t0.rio.crs, inplace=True)
            delta.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

            self._save_raster_da(
                delta,
                str(out_path),
                driver="GTiff",
                compress="LZW",
                tiled=True,
                windowed=True
            )
            dask_print(f"✅ Created Delta: {out_path.name}")
            return f"Created: {out_path.name}"

        except Exception as e:
            dask_print(f"❌ Error creating delta {out_path.name}: {str(e)}")
            return f"Error on {out_path.name}: {str(e)}"

    def run_bathy_combination(self):
        """Orchestrates the finding, averaging, and renaming of bathy files."""
        groups = self.group_files_simple()
        delayed_tasks = []

        print(f"Scanning complete. Found groups for {len(groups)} tiles.")

        for tile_id, year_map in groups.items():
            for year, paths in year_map.items():
                
                out_name = f"combined{len(paths)}_bathy_{tile_id}_{year}.tif"
                out_path_str = self._join_paths(self.combined, out_name)
                
                # OPTIMIZATION: Dask delayed now targets the top-level wrapper function
                task = dask.delayed(_run_combination_task)(paths, out_path_str, self.overwrite, self.pilot_mode)
                delayed_tasks.append(task)

        if not delayed_tasks:
            print("No matching files found to process.")
            return

        print(f"Queued {len(delayed_tasks)} combination tasks. Computing...")
        results = dask.compute(*delayed_tasks)
        for res in results:
            print(res)

    def run_gap_fill(self, input_file, output_dir, max_iters) -> None:
        """The main entry point for the gap-filling process."""
        output_file = self._join_paths(
            output_dir, os.path.splitext(os.path.basename(str(input_file)))[0] + "_filled.tif"
        )

        if not self.overwrite and self._exists(output_file):
            print(f"File already exists, skipping gap fill: {os.path.basename(str(output_file))}")
            return
        
        self.fill_with_fallback(
            input_file=input_file,
            output_file=output_file,
            max_iters=max_iters
        )

    def standardize_raster_array(self, input_array) -> np.ndarray:
        """Standardizes a numpy array (mean=0, sd=1)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean = np.nanmean(input_array)
            std = np.nanstd(input_array)
            
        if np.isnan(std) or std == 0:
            return np.zeros_like(input_array)
        return (input_array - mean) / std