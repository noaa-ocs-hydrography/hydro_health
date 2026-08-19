"""Class for parallel processing terrain products (Slope, Rugosity, BPI, Classification)"""

import os
import re
import gc
import shutil
import ctypes
import logging
import tempfile
import warnings
import traceback
import pathlib
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional, Union

import numpy as np
import pandas as pd
import rasterio
from scipy.signal import fftconvolve

import dask
import dask.array as da
from dask.distributed import Client, as_completed
from upath import UPath
import s3fs

try:
    from whitebox import WhiteboxTools
except ImportError:
    raise ImportError(
        "CRITICAL ERROR: The 'whitebox' library is not installed. "
        "Please install it by running: conda install whitebox"
    )

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)

class TerrainProductsEngine(Engine):
    """Class for parallel generation of seabed terrain layers, highly optimized for memory management."""

    def __init__(self, param_lookup: dict, output_prefix: Union[str, bool] = False) -> None:
        """Initialize paths, configurations, and environment for terrain products"""
        super().__init__()
        self.param_lookup = param_lookup
        
        def _get_val(key, default=None):
            val = self.param_lookup.get(key, default)
            if hasattr(val, 'valueAsText') and val.valueAsText is not None:
                return val.valueAsText
            if hasattr(val, 'value') and val.value is not None:
                return val.value
            return val
            
        self.local_tmp_dir = pathlib.Path(_get_val('local_tmp_dir', str(Path.home() / "hydro_health_local_tmp")))
        
        # Pre-run cleanup: Wipe out any existing tmp folder to prevent disk space issues
        if self.local_tmp_dir.exists():
            try:
                shutil.rmtree(self.local_tmp_dir)
                logger.info(f"Cleaned up existing temp directory at startup: {self.local_tmp_dir}")
            except Exception as e:
                logger.warning(f"Could not perform startup cleanup of temp directory {self.local_tmp_dir}: {e}")
                
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        env = _get_val('env', 'local')
        self.is_aws = env in ['remote', 'aws']
        self.overwrite = _get_val('overwrite', False)
        
        self.gdal_env_vars = _get_val('gdal_env_vars', {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'AWS_NO_SIGN_REQUEST': 'YES'
        } if self.is_aws else {})
        
        logger.info(f"Environment detected: {'AWS/Remote' if self.is_aws else 'Local'}")

        self.repo_root = pathlib.Path(__file__).resolve().parents[4]
        
        in_dir = _get_val('input_directory')
        out_dir = _get_val('output_directory')
        
        base_in_dir = pathlib.Path(in_dir) if in_dir else self.repo_root / 'inputs'
        base_out_dir = pathlib.Path(out_dir) if out_dir else self.repo_root / 'outputs'
        
        if hasattr(output_prefix, 'valueAsText') and output_prefix.valueAsText is not None:
            output_prefix_str = output_prefix.valueAsText
        elif hasattr(output_prefix, 'value') and output_prefix.value is not None:
            output_prefix_str = output_prefix.value
        else:
            output_prefix_str = output_prefix

        self.inputs_dir = base_in_dir
        self.outputs_dir = base_out_dir / output_prefix_str if output_prefix_str and isinstance(output_prefix_str, str) else base_out_dir
        
        eco_val = _get_val('eco_regions')
        self.ecoregion = ''
        
        if isinstance(eco_val, list) and eco_val:
            self.ecoregion = eco_val[0]
        elif isinstance(eco_val, str) and eco_val:
            import ast
            try:
                parsed = ast.literal_eval(eco_val)
                if isinstance(parsed, list) and parsed:
                    self.ecoregion = parsed[0]
                else:
                    self.ecoregion = eco_val.strip("[]'\" ")
            except Exception:
                self.ecoregion = eco_val.strip("[]'\" ")

        if not self.ecoregion:
            er_dirs = [d.name for d in self.outputs_dir.glob('ER_*') if d.is_dir()]
            if er_dirs:
                self.ecoregion = er_dirs[0]

        if self.ecoregion:
            self.outputs_dir = self.outputs_dir / self.ecoregion

        # Ensure base output directory exists for logging
        if isinstance(self.outputs_dir, pathlib.Path):
            self.outputs_dir.mkdir(parents=True, exist_ok=True)

        def _resolve_path(config_value: str, is_output: bool = False) -> UPath:
            if self.is_aws:
                bucket = get_config_item('S3', 'BUCKET_NAME')
                return UPath(f"s3://{bucket}/{config_value}")
            else:
                p = pathlib.Path(config_value)
                if p.is_absolute():
                    return UPath(p)
                if p.parts:
                    if p.parts[0] == 'inputs':
                        return UPath(self.inputs_dir / pathlib.Path(*p.parts[1:]))
                    elif p.parts[0] == 'outputs':
                        return UPath(self.outputs_dir / pathlib.Path(*p.parts[1:]))
                base_dir = self.outputs_dir if is_output else self.inputs_dir
                return UPath(base_dir / p)

        self.combined_bathy_dir = _resolve_path(get_config_item('TERRAIN', 'COMBINED_LIDAR_DIR'), is_output=True)
        self.terrain_outputs_dir = _resolve_path(get_config_item('TERRAIN', 'OUTPUTS'), is_output=True)
        self.dictionary_dir = _resolve_path(get_config_item('TERRAIN', 'DICTIONARIES_DIR', default="dictionaries"), is_output=True)
        
        self._init_wbt()

    def _init_wbt(self):
        """Initializes WhiteboxTools securely."""
        self.wbt = WhiteboxTools()
        self.wbt.verbose = False
        self.wbt.set_compress_rasters(True)
        self.wbt.max_procs = 2 

    def __getstate__(self):
        """Exclude unpicklable attributes when serializing to Dask workers."""
        state = self.__dict__.copy()
        state.pop('wbt', None)
        state.pop('client', None)
        state.pop('cluster', None)
        state.pop('param_lookup', None)  # CRITICAL: Removes unpicklable ArcPy objects
        return state

    def __setstate__(self, state):
        """Reconstruct state on the worker nodes."""
        self.__dict__.update(state)
        if hasattr(self, 'local_tmp_dir'):
            self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        self._init_wbt()

    def _exists(self, path) -> bool:
        return UPath(path).exists()

    def _getsize(self, path) -> int:
        return UPath(path).stat().st_size

    def _join_paths(self, *args) -> str:
        if not args: return ""
        return str(UPath(args[0]).joinpath(*args[1:]))

    def _safe_ls(self, path) -> List[str]:
        try:
            p = UPath(path)
            if not p.exists(): return []
            return [str(child) for child in p.iterdir()]
        except FileNotFoundError:
            return []

    def _save_numpy_to_raster(self, data_array: np.ndarray, out_path: str, profile: dict, log_prefix: str = ""):
        """Safely saves a numpy array to S3 or local by writing to tmp first."""
        out_u = UPath(out_path)
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=str(self.local_tmp_dir)) as tmp_file:
            local_tmp_path = tmp_file.name
            
        try:
            with rasterio.open(local_tmp_path, 'w', **profile) as dst:
                dst.write(data_array, 1)
                
            if out_u.protocol == "s3":
                out_u.fs.put_file(local_tmp_path, str(out_u))
            else:
                shutil.copyfile(local_tmp_path, str(out_path))
                
            logger.info(f"{log_prefix}Successfully wrote NumPy layer to: {out_path}")
        finally:
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)

    def calculate_bpi(self, bathy_array: np.ndarray, cell_size: float, inner_radius: float, outer_radius: float) -> np.ndarray:
        """Calculates BPI using chunked Dask logic to prevent massive mem allocations."""
        if cell_size <= 0:
            raise ValueError(f"Invalid cell_size ({cell_size}). Cannot calculate BPI.")
            
        inner_cells = int(round(inner_radius / cell_size))
        outer_cells = int(round(outer_radius / cell_size))
        
        if outer_cells > 2000:
            raise ValueError(f"outer_cells ({outer_cells}) is too large (cell_size: {cell_size}). Skipping.")
        
        y, x = np.ogrid[-outer_cells:outer_cells + 1, -outer_cells:outer_cells + 1]
        mask = x**2 + y**2 <= outer_cells**2
        mask[x**2 + y**2 <= inner_cells**2] = False
        
        kernel = mask.astype(np.float32)
        chunk_size = 1024
        d_bathy = da.from_array(bathy_array, chunks=(chunk_size, chunk_size))
        
        d_valid = da.map_blocks(lambda b: (~np.isnan(b)).astype(np.float32), d_bathy, dtype=np.float32)
        d_bathy_zeroed = da.where(da.isnan(d_bathy), 0.0, d_bathy)
        
        def _conv(block):
            return fftconvolve(block, kernel, mode='same').astype(np.float32)
            
        sum_array = d_bathy_zeroed.map_overlap(_conv, depth=outer_cells, boundary='reflect')
        count_array = d_valid.map_overlap(_conv, depth=outer_cells, boundary='reflect')
        
        mean_annulus = da.where(count_array > 0, sum_array / count_array, np.nan)
        bpi_lazy = d_bathy - mean_annulus
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return bpi_lazy.compute(scheduler='single-threaded')

    def calculate_slope(self, bathy_array: np.ndarray, cell_size: float) -> np.ndarray:
        """Calculates slope (degrees) using inplace operations to minimize memory."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gy = np.empty_like(bathy_array, dtype=np.float32)
            gx = np.empty_like(bathy_array, dtype=np.float32)
            
            gy[1:-1, :] = (bathy_array[2:, :] - bathy_array[:-2, :]) / (2 * cell_size)
            gy[0, :] = (bathy_array[1, :] - bathy_array[0, :]) / cell_size
            gy[-1, :] = (bathy_array[-1, :] - bathy_array[-2, :]) / cell_size
            
            gx[:, 1:-1] = (bathy_array[:, 2:] - bathy_array[:, :-2]) / (2 * cell_size)
            gx[:, 0] = (bathy_array[:, 1] - bathy_array[:, 0]) / cell_size
            gx[:, -1] = (bathy_array[:, -1] - bathy_array[:, -2]) / cell_size
            
            np.square(gx, out=gx)
            np.square(gy, out=gy)
            gx += gy 
            del gy 
            
            np.sqrt(gx, out=gx)
            np.arctan(gx, out=gx)
            slope_deg = np.degrees(gx, out=gx)
            
        return slope_deg

    def calculate_tri(self, bathy_array: np.ndarray) -> np.ndarray:
        """Calculates Terrain Ruggedness Index (TRI) via vectorized slices."""
        tri_sum = np.zeros_like(bathy_array, dtype=np.float32)
        valid_count = np.zeros_like(bathy_array, dtype=np.float32)
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                y1, y2 = max(0, dy), bathy_array.shape[0] + min(0, dy)
                x1, x2 = max(0, dx), bathy_array.shape[1] + min(0, dx)
                sy1, sy2 = max(0, -dy), bathy_array.shape[0] + min(0, -dy)
                sx1, sx2 = max(0, -dx), bathy_array.shape[1] + min(0, -dx)
                
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
            return np.where(valid_count > 0, tri_sum / valid_count, np.nan)

    def create_classification_dictionary(self, bpi_broad_sample: np.ndarray, bpi_fine_sample: np.ndarray, slope_sample: np.ndarray) -> pd.DataFrame:
        """Creates a data-driven classification dictionary from sample arrays."""
        valid_broad = bpi_broad_sample[~np.isnan(bpi_broad_sample)]
        valid_fine = bpi_fine_sample[~np.isnan(bpi_fine_sample)]
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

    def create_regionally_consistent_dictionaries(self, all_files: List[str], best_radii: Dict[str, Tuple[int, int]], valid_years: Optional[Set[int]] = None) -> None:
        """Groups files by year, samples valid pixels, and generates regional class limits."""
        logger.info(f"\n--- PHASE 1: Building Regional Classification Limits in {self.dictionary_dir} ---")
        out_path = UPath(self.dictionary_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        year_groups = {}
        for f in all_files:
            match = re.search(r'((?:19|20)\d{2})', os.path.basename(str(f)))
            if match:
                extracted_year = int(match.group(1))
                if valid_years is None or extracted_year in valid_years:
                    year = match.group(1)
                    year_groups.setdefault(year, []).append(f)
                    
        generic_bathy = [f for f in all_files if 'bluetopo' in os.path.basename(str(f)).lower()]
        if generic_bathy:
            year_groups['BlueTopo'] = generic_bathy
            
        for year, files in year_groups.items():
            dict_path = UPath(self._join_paths(str(self.dictionary_dir), f"dictionary_{year}.csv"))
            if not self.overwrite and dict_path.exists():
                logger.info(f"  - Skipping: {year} (Dictionary exists)")
                continue

            logger.info(f"  - Processing: {year} ({len(files)} files)")
            file_data = [(f, self._getsize(f)) for f in files]
            all_sizes = [x[1] for x in file_data]
            size_threshold = np.percentile(all_sizes, 30)
            small_files_pool = [x[0] for x in file_data if x[1] <= size_threshold]

            files_to_sample = small_files_pool if len(small_files_pool) <= 10 else list(np.random.choice(small_files_pool, 10, replace=False))
            all_samples = {'slope': [], 'bpi_fine': [], 'bpi_broad': []}
            
            for f in files_to_sample:
                try:
                    with rasterio.open(str(f)) as src:
                        crs = src.crs
                        epsg_code = crs.to_epsg() if crs else None
                        if epsg_code != 32617: continue

                        bathy_array = src.read(1)
                        if src.nodata is not None and not (isinstance(src.nodata, (float, np.floating)) and np.isnan(src.nodata)):
                            bathy_array[bathy_array == src.nodata] = np.nan
                            
                        cell_size = src.res[0]
                        valid_pixels = np.argwhere(~np.isnan(bathy_array))
                        if len(valid_pixels) == 0: continue
                        
                        sample_indices = valid_pixels[np.random.choice(len(valid_pixels), min(len(valid_pixels), 20000), replace=False)]
                        
                        slope_sample = self.calculate_slope(bathy_array, cell_size)
                        bpi_fine_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                        bpi_broad_sample = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                        
                        rows, cols = sample_indices[:, 0], sample_indices[:, 1]
                        all_samples['slope'].append(slope_sample[rows, cols])
                        all_samples['bpi_fine'].append(bpi_fine_sample[rows, cols])
                        all_samples['bpi_broad'].append(bpi_broad_sample[rows, cols])
                        del bathy_array, slope_sample, bpi_fine_sample, bpi_broad_sample
                        gc.collect()
                except Exception:
                    continue
                finally:
                    # [MEMORY FIX] Aggressively collect garbage after every file is opened and sampled
                    gc.collect()
                    try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                    except Exception: pass
            
            if all_samples['slope']:
                slope_agg = np.concatenate(all_samples['slope'])
                fine_agg = np.concatenate(all_samples['bpi_fine'])
                broad_agg = np.concatenate(all_samples['bpi_broad'])
                year_dictionary = self.create_classification_dictionary(broad_agg, fine_agg, slope_agg)
                
                with dict_path.open('w') as fh:
                    year_dictionary.to_csv(fh, index=False)
                logger.info(f"  - Saved dictionary for {year}.")
                
            # [MEMORY FIX] Force memory release after processing all samples for a given year
            gc.collect()
            try: ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception: pass

    def process_terrain_raster(self, bathy_path: str, best_radii: Dict[str, Tuple[int, int]], current_index: int = None, total_count: int = None) -> tuple:
        """Processes one bathymetry raster returning a Success/Failure boolean tuple."""
        base_name = os.path.splitext(os.path.basename(str(bathy_path)))[0]
        progress_str = f"[{current_index}/{total_count}] " if current_index and total_count else ""
        
        out_dir_path = UPath(self.terrain_outputs_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        def resolve_out_path(suffix):
            return self._join_paths(str(self.terrain_outputs_dir), base_name + suffix)

        out_slope_deg = resolve_out_path("_slope_deg.tif")
        out_gradmag   = resolve_out_path("_gradmag.tif") 
        out_flowdir   = resolve_out_path("_flowdir.tif") 
        out_prof      = resolve_out_path("_curv_profile.tif")
        out_plan      = resolve_out_path("_curv_plan.tif")
        out_total     = resolve_out_path("_curv_total.tif")
        out_tci       = resolve_out_path("_tci.tif")
        out_flowacc   = resolve_out_path("_flowacc.tif")
        out_shear     = resolve_out_path("_shearproxy.tif")

        out_rug = resolve_out_path("_rugosity_tri.tif")
        out_slope = resolve_out_path("_slope.tif")
        out_fine = resolve_out_path("_bpi_fine.tif")
        out_broad = resolve_out_path("_bpi_broad.tif")
        out_class = resolve_out_path("_terrain_classification.tif")
        
        try:
            with tempfile.TemporaryDirectory(dir=str(self.local_tmp_dir)) as tmpdir:
                local_bathy = os.path.join(tmpdir, "bathy.tif")
                local_slope = os.path.join(tmpdir, "slope_deg.tif")
                local_flowdir = os.path.join(tmpdir, "flowdir.tif")
                local_prof = os.path.join(tmpdir, "prof.tif")
                local_plan = os.path.join(tmpdir, "plan.tif")
                local_total = os.path.join(tmpdir, "total.tif")
                local_tci = os.path.join(tmpdir, "tci.tif")
                local_flowacc = os.path.join(tmpdir, "flowacc.tif")
                local_gradmag = os.path.join(tmpdir, "gradmag.tif")

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

                missing_numpy_dict = {
                    "_rugosity_tri.tif": self.overwrite or not self._exists(out_rug),
                    "_slope.tif": self.overwrite or not self._exists(out_slope),
                    "_bpi_fine.tif": self.overwrite or not self._exists(out_fine),
                    "_bpi_broad.tif": self.overwrite or not self._exists(out_broad),
                    "_terrain_classification.tif": self.overwrite or not self._exists(out_class)
                }
                missing_numpy = any(missing_numpy_dict.values())

                if not (len(missing_wbt) > 0 or missing_tci or missing_shear or missing_numpy):
                     return (True, f"Skipped: {base_name} (All exist)")

                logger.info(f"-> [STARTING] {progress_str}Generating products for: {base_name}")

                with UPath(bathy_path).open('rb') as f_in:
                    with open(local_bathy, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
                # [MEMORY FIX] Clear memory buffers after large file copy
                gc.collect()

                with rasterio.open(local_bathy) as src:
                    crs = src.crs
                    epsg_code = crs.to_epsg() if crs else None
                    if epsg_code != 32617:
                        return (False, f"Skipped (Invalid CRS): {base_name}")

                for out_s3, wbt_func, local_out in missing_wbt:
                    try:
                        wbt_func(local_bathy, local_out)
                        if os.path.exists(local_out):
                            with open(local_out, 'rb') as f_in, UPath(out_s3).open('wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                            if local_out not in [local_slope, local_plan]:
                                try: os.remove(local_out)
                                except OSError: pass
                                
                        # [MEMORY FIX] Collect garbage after each WBT product is finished and file is closed
                        gc.collect()
                    except Exception as e:
                        logger.error(f"WBT Error on {base_name}: {e}")

                if missing_tci:
                    try: 
                        self.wbt.convergence_index(local_bathy, local_tci)
                        if os.path.exists(local_tci):
                            with open(local_tci, 'rb') as f_in, UPath(out_tci).open('wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                            try: os.remove(local_tci)
                            except OSError: pass
                            
                        # [MEMORY FIX] Clear WBT TCI file handles
                        gc.collect()
                    except Exception as e: 
                        logger.error(f"WBT TCI Error on {base_name}: {e}")

                if missing_shear:
                    try:
                        slope_src = local_slope if os.path.exists(local_slope) else None
                        plan_src = local_plan if os.path.exists(local_plan) else None

                        if slope_src and plan_src:
                            with rasterio.open(slope_src) as s, rasterio.open(plan_src) as p:
                                meta = s.meta.copy()
                                s_nodata = s.nodata if s.nodata is not None else -9999.0
                                p_nodata = p.nodata if p.nodata is not None else -9999.0
                                
                                meta.update(compress='LZW', nodata=s_nodata, dtype='float32')
                                
                                out_u = UPath(out_shear)
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=str(self.local_tmp_dir)) as tmp_file:
                                    local_shear_path = tmp_file.name
                                    
                                with rasterio.open(local_shear_path, 'w', **meta) as dst:
                                    for ji, window in s.block_windows(1):
                                        slope_chunk = s.read(1, window=window).astype(np.float32)
                                        plan_chunk = p.read(1, window=window).astype(np.float32)
                                        
                                        valid_mask = ~np.isnan(slope_chunk) & ~np.isnan(plan_chunk)
                                        
                                        shear_chunk = np.full_like(slope_chunk, s_nodata, dtype=np.float32)
                                        shear_chunk[valid_mask] = slope_chunk[valid_mask] * np.abs(plan_chunk[valid_mask])
                                        
                                        dst.write(shear_chunk, 1, window=window)
                                        
                                        # [MEMORY FIX] Clear arrays inside the block window loop
                                        del slope_chunk, plan_chunk, valid_mask, shear_chunk
                                        gc.collect()
                                        
                                if out_u.protocol == "s3":
                                    out_u.fs.put_file(local_shear_path, str(out_u))
                                else:
                                    shutil.copyfile(local_shear_path, str(out_shear))
                                    
                                os.remove(local_shear_path)
                    except Exception as e:
                        logger.error(f"Shear Proxy Error {base_name}: {e}")
                    finally:
                        for tmp_f in [local_slope, local_plan]:
                            if tmp_f and os.path.exists(tmp_f):
                                try: os.remove(tmp_f)
                                except OSError: pass

                if missing_numpy:
                    year = 'BlueTopo' if 'bluetopo' in base_name.lower() else 'bt_bathy'
                    match = re.search(r'((?:19|20)\d{2})', base_name)
                    if match: year = match.group(1)
                        
                    dict_path = UPath(self._join_paths(str(self.dictionary_dir), f"dictionary_{year}.csv"))
                    if missing_numpy_dict["_terrain_classification.tif"] and not dict_path.exists():
                        return (False, f"Dictionary missing for {year}")
                    elif missing_numpy_dict["_terrain_classification.tif"]:
                        with dict_path.open('r') as fh:
                            unique_dictionary = pd.read_csv(fh)

                    with rasterio.open(local_bathy) as src:
                        profile = src.profile
                        cell_size = src.res[0]
                        shape_2d = (src.height, src.width)
                        
                        bathy_array = np.memmap(os.path.join(tmpdir, "bathy.dat"), dtype='float32', mode='w+', shape=shape_2d)
                        for ji, window in src.block_windows(1):
                            chunk = src.read(1, window=window)
                            if src.nodata is not None and not (isinstance(src.nodata, (float, np.floating)) and np.isnan(src.nodata)):
                                chunk[chunk == src.nodata] = np.nan
                            bathy_array[window.toslices()] = chunk
                            
                            # [MEMORY FIX] Clear chunk memory inside block window loop
                            del chunk
                            gc.collect()

                    if missing_numpy_dict["_rugosity_tri.tif"]:
                        rugosity = self.calculate_tri(bathy_array)
                        profile.update(dtype=rugosity.dtype.name, nodata=np.nan, count=1, compress='LZW')
                        self._save_numpy_to_raster(rugosity, out_rug, profile, log_prefix=progress_str)
                        del rugosity; gc.collect()

                    if missing_numpy_dict["_slope.tif"]:
                        slope_raw = self.calculate_slope(bathy_array, cell_size)
                        profile.update(dtype=slope_raw.dtype.name, nodata=np.nan, count=1, compress='LZW')
                        self._save_numpy_to_raster(slope_raw, out_slope, profile, log_prefix=progress_str)
                        if missing_numpy_dict["_terrain_classification.tif"]:
                            slope = np.memmap(os.path.join(tmpdir, "s.dat"), dtype='float32', mode='w+', shape=shape_2d)
                            slope[:] = slope_raw[:]
                        del slope_raw
                    elif missing_numpy_dict["_terrain_classification.tif"]:
                        with rasterio.open(out_slope) as src_s:
                            slope = np.memmap(os.path.join(tmpdir, "s.dat"), dtype='float32', mode='w+', shape=shape_2d)
                            for ji, window in src_s.block_windows(1):
                                slope[window.toslices()] = src_s.read(1, window=window)

                    if missing_numpy_dict["_bpi_fine.tif"]:
                        bpi_fine = self.calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                        profile.update(dtype=bpi_fine.dtype.name, nodata=np.nan, count=1, compress='LZW')
                        self._save_numpy_to_raster(bpi_fine, out_fine, profile, log_prefix=progress_str)
                        if missing_numpy_dict["_terrain_classification.tif"]:
                            bpi_fine_mem = np.memmap(os.path.join(tmpdir, "f.dat"), dtype='float32', mode='w+', shape=shape_2d)
                            bpi_fine_mem[:] = bpi_fine[:]
                        del bpi_fine
                    elif missing_numpy_dict["_terrain_classification.tif"]:
                        with rasterio.open(out_fine) as src_f:
                            bpi_fine_mem = np.memmap(os.path.join(tmpdir, "f.dat"), dtype='float32', mode='w+', shape=shape_2d)
                            for ji, window in src_f.block_windows(1):
                                bpi_fine_mem[window.toslices()] = src_f.read(1, window=window)

                    if missing_numpy_dict["_bpi_broad.tif"]:
                        bpi_broad = self.calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                        profile.update(dtype=bpi_broad.dtype.name, nodata=np.nan, count=1, compress='LZW')
                        self._save_numpy_to_raster(bpi_broad, out_broad, profile, log_prefix=progress_str)
                        if missing_numpy_dict["_terrain_classification.tif"]:
                            bpi_broad_mem = np.memmap(os.path.join(tmpdir, "b.dat"), dtype='float32', mode='w+', shape=shape_2d)
                            bpi_broad_mem[:] = bpi_broad[:]
                        del bpi_broad
                    elif missing_numpy_dict["_terrain_classification.tif"]:
                        with rasterio.open(out_broad) as src_b:
                            bpi_broad_mem = np.memmap(os.path.join(tmpdir, "b.dat"), dtype='float32', mode='w+', shape=shape_2d)
                            for ji, window in src_b.block_windows(1):
                                bpi_broad_mem[window.toslices()] = src_b.read(1, window=window)

                    del bathy_array; gc.collect()
                    try: os.remove(os.path.join(tmpdir, "bathy.dat"))
                    except OSError: pass
                    
                    if missing_numpy_dict["_terrain_classification.tif"]:
                        classified_array = np.memmap(os.path.join(tmpdir, "c.dat"), dtype='float32', mode='w+', shape=shape_2d)
                        classified_array[:] = np.nan 
                        
                        chunk_s = 2048
                        for i in range(0, shape_2d[0], chunk_s):
                            for j in range(0, shape_2d[1], chunk_s):
                                s_c = slope[i:i+chunk_s, j:j+chunk_s]
                                b_c = bpi_broad_mem[i:i+chunk_s, j:j+chunk_s]
                                f_c = bpi_fine_mem[i:i+chunk_s, j:j+chunk_s]
                                c_c = classified_array[i:i+chunk_s, j:j+chunk_s]
                                
                                valid_mask = ~np.isnan(s_c) & ~np.isnan(b_c) & ~np.isnan(f_c)
                                
                                for _, rule in unique_dictionary.iterrows():
                                    matches = ((b_c >= rule['BroadBPI_Lower']) & (b_c <= rule['BroadBPI_Upper']) &
                                               (f_c >= rule['FineBPI_Lower']) & (f_c <= rule['FineBPI_Upper']) &
                                               (s_c >= rule['Slope_Lower']) & (s_c <= rule['Slope_Upper']))
                                    c_c[valid_mask & matches & np.isnan(c_c)] = rule['Class_ID']
                        
                        profile.update(dtype=classified_array.dtype.name, nodata=np.nan, count=1, compress='LZW')
                        self._save_numpy_to_raster(classified_array, out_class, profile, log_prefix=progress_str)
                        del classified_array

                    if 'slope' in locals():
                        del slope, bpi_fine_mem, bpi_broad_mem
                    gc.collect()
                    for f in ["s.dat", "f.dat", "b.dat", "c.dat"]:
                        try: os.remove(os.path.join(tmpdir, f))
                        except OSError: pass

                logger.info(f" - [✓ SUCCESS] {progress_str}Completed terrain processing: {base_name}")
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception: pass
                    
                return (True, f"Success: {base_name}")
            
        except Exception as e:
            err_msg = traceback.format_exc()
            logger.error(f"❌ [{base_name}] Error during terrain product generation:\n{err_msg}")
            return (False, f"Failed: {base_name} - {str(e)}")

    def _log_system_metrics(self) -> str:
        """Helper to collect and format EC2 system metrics (RAM, Disk Space, Temp Size)."""
        try:
            total, used, free = shutil.disk_usage(self.local_tmp_dir)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            tmp_size_bytes = 0
            if self.local_tmp_dir.exists():
                tmp_size_bytes = sum(f.stat().st_size for f in self.local_tmp_dir.rglob('*') if f.is_file())
            tmp_mb = tmp_size_bytes / (1024**2)
            
            ram_info = "Unknown"
            try:
                import psutil
                vm = psutil.virtual_memory()
                ram_info = f"Free: {vm.available / (1024**3):.1f}GB / {vm.total / (1024**3):.1f}GB (Used: {vm.percent}%)"
            except ImportError:
                if os.path.exists('/proc/meminfo'):
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    import re
                    mem_avail = re.search(r'MemAvailable:\s+(\d+)\s+kB', meminfo)
                    mem_total = re.search(r'MemTotal:\s+(\d+)\s+kB', meminfo)
                    if mem_avail and mem_total:
                        avail_gb = int(mem_avail.group(1)) / (1024**2)
                        tot_gb = int(mem_total.group(1)) / (1024**2)
                        pct = 100 - (avail_gb / tot_gb * 100)
                        ram_info = f"Free: {avail_gb:.1f}GB / {tot_gb:.1f}GB (Used: {pct:.1f}%)"
            
            return f"   [SysMetrics] RAM | {ram_info} || Disk Free | {free_gb:.1f}GB / {total_gb:.1f}GB || Tmp Dir Size | {tmp_mb:.1f}MB"
        except Exception as e:
            return f"   [SysMetrics] Error collecting system metrics: {e}"

    def _cleanup_resources(self, client: Client) -> None:
        """Safely tears down Dask and forces deletion of the massive temp directory."""
        logger.info("Cleaning up resources and shutting down Dask...")
        try:
            if client:
                client.close()
            self.close_dask()
        except Exception as e:
            logger.error(f"Could not cleanly close client/cluster: {e}")

        logger.info(f"Cleaning up master temp directory: {self.local_tmp_dir}")
        try:
            if self.local_tmp_dir.exists():
                shutil.rmtree(self.local_tmp_dir)
                logger.info(f"Successfully removed temp directory: {self.local_tmp_dir}")
        except Exception as e:
            logger.error(f"Could not delete temp directory {self.local_tmp_dir}: {e}")

    def run(self, max_concurrent: int = 4) -> None:
        """Main entry point for generating terrain products with aggressive memory management."""
        
        env_val = self.param_lookup.get('env', 'local')
        env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        
        self.setup_dask(env)
        client = getattr(self, 'client', None)
        if not client:
            client = Client()

        try:
            logger.info(f"Scanning combined lidar directory: {self.combined_bathy_dir}")
            potential_inputs = []
            for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
                potential_inputs.extend(list(UPath(self.combined_bathy_dir).rglob(ext)))

            valid_files = [f for f in potential_inputs if 'iss' not in UPath(f).name.lower()]
            
            logger.info(f"Found {len(valid_files)} combined bathymetry files.")
            if not valid_files:
                logger.info("No bathymetry files found to process.")
                return

            best_radii = {'fine': (8, 32), 'broad': (80, 240)}
            self.create_regionally_consistent_dictionaries(valid_files, best_radii)

            logger.info(f"\n--- PHASE 2: Parallel Terrain Product Generation ---")
            
            total_tasks = len(valid_files)
            task_iterator = iter(enumerate(valid_files))
            task_stream = as_completed()
            
            def submit_terrain_task(item):
                i, file_path = item
                return client.submit(
                    self.process_terrain_raster,
                    str(file_path),
                    best_radii,
                    current_index=i + 1,
                    total_count=total_tasks
                )

            # Initial queue fill
            for _ in range(min(max_concurrent, total_tasks)):
                try:
                    task_stream.add(submit_terrain_task(next(task_iterator)))
                except StopIteration:
                    break

            # Dynamic Dask execution loop with strict garbage collection
            for future in task_stream:
                try:
                    success, result_msg = future.result() 
                    if success:
                        logger.info(f" - [SUCCESS] {result_msg}")
                    else:
                        logger.error(f" - [ERROR/SKIP] {result_msg}")
                        
                    metrics_msg = self._log_system_metrics()
                    logger.info(metrics_msg)
                    
                except Exception as e:
                    logger.error(f" - [FATAL ERROR] Worker task crashed: {e}")
                    
                try:
                    task_stream.add(submit_terrain_task(next(task_iterator)))
                except StopIteration:
                    pass
                
                # [MEMORY FIX] Aggressively force memory release back to the OS after every single file finishes
                gc.collect()
                try: ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception: pass

            logger.info("\n[SUCCESS] Terrain raster processing complete.")
            
        finally:
            self._cleanup_resources(client)