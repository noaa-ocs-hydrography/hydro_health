"""Class for parallel processing terrain products (Slope, Rugosity, BPI, Classification)"""

import os
import re
import gc
import shutil
import platform
import subprocess
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
from dask.distributed import Client
from upath import UPath
from whitebox import WhiteboxTools

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

# Global Base Paths
INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'

def _save_numpy_to_raster(data_array: np.ndarray, out_path: str, profile: dict, local_tmp_dir: str, log_prefix: str = ""):
    """Safely saves a numpy array to S3 or local by writing to tmp first."""
    out_u = UPath(out_path)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=local_tmp_dir) as tmp_file:
        local_tmp_path = tmp_file.name
        
    try:
        with rasterio.open(local_tmp_path, 'w', **profile) as dst:
            dst.write(data_array, 1)
            
        if out_u.protocol == "s3":
            out_u.fs.put_file(local_tmp_path, str(out_u))
        else:
            shutil.copyfile(local_tmp_path, str(out_path))
            
        Engine.write_message_dask(f"{log_prefix}Successfully wrote NumPy layer to: {out_path}", OUTPUTS)
    finally:
        if os.path.exists(local_tmp_path):
            try:
                os.remove(local_tmp_path)
            except OSError:
                pass

def _calculate_bpi(bathy_array: np.ndarray, cell_size: float, inner_radius: float, outer_radius: float) -> np.ndarray:
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

def _calculate_slope(bathy_array: np.ndarray, cell_size: float) -> np.ndarray:
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

def _calculate_tri(bathy_array: np.ndarray) -> np.ndarray:
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

def _create_classification_dictionary(bpi_broad_sample: np.ndarray, bpi_fine_sample: np.ndarray, slope_sample: np.ndarray) -> pd.DataFrame:
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

def _create_regional_dictionary_worker(year: str, files: List[str], dictionary_dir: str, best_radii: Dict[str, Tuple[int, int]], local_tmp_dir: str) -> Tuple[bool, str]:
    """Module-level worker to sample valid pixels and generate regional class limits."""
    dict_path = UPath(dictionary_dir) / f"dictionary_{year}.csv"
    
    # Implicit skip pattern
    if dict_path.exists():
        return True, f"Skipping dictionary creation for {year} (Already exists)"
        
    Engine.write_message_dask(f"Processing regional dictionary for: {year} ({len(files)} files)", OUTPUTS)
    
    try:
        def _getsize(path):
            try: return UPath(path).stat().st_size
            except Exception: return 0

        file_data = [(f, _getsize(f)) for f in files]
        all_sizes = [x[1] for x in file_data if x[1] > 0]
        size_threshold = np.percentile(all_sizes, 30) if all_sizes else 0
        small_files_pool = [x[0] for x in file_data if x[1] <= size_threshold]

        files_to_sample = small_files_pool if len(small_files_pool) <= 10 else list(np.random.choice(small_files_pool, 10, replace=False))
        all_samples = {'slope': [], 'bpi_fine': [], 'bpi_broad': []}
        
        for f in files_to_sample:
            try:
                with rasterio.open(str(f)) as src:
                    bathy_array = src.read(1).astype(np.float32)
                    
                    if src.nodata is not None and not np.isnan(src.nodata):
                        bathy_array[bathy_array == src.nodata] = np.nan
                        
                    bathy_array[bathy_array < -9998.0] = np.nan
                    bathy_array[bathy_array >= 0.0] = np.nan
                        
                    cell_size = src.res[0]
                    valid_pixels = np.argwhere(~np.isnan(bathy_array))
                    if len(valid_pixels) == 0: continue
                    
                    sample_indices = valid_pixels[np.random.choice(len(valid_pixels), min(len(valid_pixels), 20000), replace=False)]
                    
                    slope_sample = _calculate_slope(bathy_array, cell_size)
                    bpi_fine_sample = _calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                    bpi_broad_sample = _calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                    
                    rows, cols = sample_indices[:, 0], sample_indices[:, 1]
                    all_samples['slope'].append(slope_sample[rows, cols])
                    all_samples['bpi_fine'].append(bpi_fine_sample[rows, cols])
                    all_samples['bpi_broad'].append(bpi_broad_sample[rows, cols])
                    del bathy_array, slope_sample, bpi_fine_sample, bpi_broad_sample
                    gc.collect()
            except Exception:
                continue
            finally:
                gc.collect()
        
        if all_samples['slope']:
            slope_agg = np.concatenate(all_samples['slope'])
            fine_agg = np.concatenate(all_samples['bpi_fine'])
            broad_agg = np.concatenate(all_samples['bpi_broad'])
            year_dictionary = _create_classification_dictionary(broad_agg, fine_agg, slope_agg)
            
            with dict_path.open('w') as fh:
                year_dictionary.to_csv(fh, index=False)
            Engine.write_message_dask(f"Saved dictionary for {year}.", OUTPUTS)
            
        return True, f"Successfully created dictionary for {year}."
    except Exception as e:
        err = traceback.format_exc()
        return False, f"Failed dictionary creation for {year}: {e}\n{err}"
    finally:
        gc.collect()

def _process_terrain_raster_worker(bathy_path: str, best_radii: Dict[str, Tuple[int, int]], terrain_outputs_dir: str, prediction_output_dir: str, dictionary_dir: str, local_tmp_dir: str, current_index: int, total_count: int) -> Tuple[bool, str]:
    """Module-level worker to process one bathymetry raster, completely detached from the class."""
    import sys
    if sys.stdout is None:
        class DummyFile:
            def write(self, x): pass
            def flush(self): pass
        sys.stdout = DummyFile()
    if sys.stderr is None:
        class DummyFile:
            def write(self, x): pass
            def flush(self): pass
        sys.stderr = DummyFile()

    if platform.system() == "Windows" and not hasattr(subprocess, "_wbt_patched"):
        _orig_popen = subprocess.Popen
        def _no_window_popen(*args, **kwargs):
            kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000)
            return _orig_popen(*args, **kwargs)
        subprocess.Popen = _no_window_popen
        
        try:
            import whitebox.whitebox_tools
            whitebox.whitebox_tools.Popen = _no_window_popen
        except Exception:
            pass
        subprocess._wbt_patched = True

    try:
        base_name = os.path.splitext(os.path.basename(str(bathy_path)))[0]
        progress_str = f"[{current_index}/{total_count}] " if current_index and total_count else ""
        
        is_bluetopo = 'bluetopo' in base_name.lower()
        
        out_dir_path = UPath(terrain_outputs_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        
        def resolve_out_path(suffix):
            return str(out_dir_path / (base_name + suffix))

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
            from whitebox import WhiteboxTools
        except ImportError:
            return (False, f"Failed: Whitebox library is not installed in the worker environment.")

        import whitebox
        if not os.path.exists(os.path.join(os.path.dirname(whitebox.__file__), "WBT", "whitebox_tools.exe")):
            whitebox.download_wbt()

        wbt = WhiteboxTools()
        wbt.verbose = False
        wbt.set_default_callback(lambda x: None)
        wbt.set_compress_rasters(True)
        wbt.max_procs = 2 

        tmpdir = tempfile.mkdtemp(dir=str(local_tmp_dir))
        
        try:
            wbt.set_working_dir(tmpdir)
            
            local_bathy_raw = os.path.join(tmpdir, "bathy_raw.tif")
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
                (out_gradmag, lambda i, o: wbt.slope(i, o, units="radians"), local_gradmag),
                (out_flowdir, wbt.aspect, local_flowdir),
                (out_prof, wbt.profile_curvature, local_prof),
                (out_plan, wbt.plan_curvature, local_plan),
                (out_total, wbt.total_curvature, local_total),
                (out_flowacc, lambda i, o: wbt.d8_flow_accumulation(i, o, out_type="cells"), local_flowacc)
            ]
            
            if not is_bluetopo:
                outputs_wbt.insert(0, (out_slope_deg, lambda i, o: wbt.slope(i, o, units="degrees"), local_slope))

            # Implicit Skipping Logic
            missing_wbt = [item for item in outputs_wbt if not UPath(item[0]).exists()]
            missing_tci = not UPath(out_tci).exists()
            missing_shear = not UPath(out_shear).exists()

            missing_numpy_dict = {
                "_rugosity_tri.tif": False if is_bluetopo else (not UPath(out_rug).exists()),
                "_slope.tif": False if is_bluetopo else (not UPath(out_slope).exists()),
                "_bpi_fine.tif": not UPath(out_fine).exists(),
                "_bpi_broad.tif": not UPath(out_broad).exists(),
                "_terrain_classification.tif": not UPath(out_class).exists()
            }
            missing_numpy = any(missing_numpy_dict.values())

            if not (len(missing_wbt) > 0 or missing_tci or missing_shear or missing_numpy):
                 return (True, f"Skipped: {base_name} (All exist)")

            Engine.write_message_dask(f"-> [STARTING] {progress_str}Generating products for: {base_name}", OUTPUTS)

            with UPath(bathy_path).open('rb') as f_in, open(local_bathy_raw, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
            with rasterio.open(local_bathy_raw) as src:
                profile = src.profile
                s_nodata = src.nodata
                profile.update(nodata=-9999.0, dtype='float32')
                
                with rasterio.open(local_bathy, 'w', **profile) as dst:
                    for ji, window in src.block_windows(1):
                        chunk = src.read(1, window=window).astype(np.float32)
                        if s_nodata is not None and not np.isnan(s_nodata):
                            chunk[chunk == s_nodata] = -9999.0
                        
                        chunk[chunk < -9998.0] = -9999.0
                        chunk[chunk >= 0.0] = -9999.0
                        chunk[np.isnan(chunk)] = -9999.0
                        
                        dst.write(chunk, 1, window=window)

            try: os.remove(local_bathy_raw)
            except OSError: pass
            gc.collect()
                    
            if is_bluetopo:
                ext = UPath(bathy_path).suffix
                match = re.search(r'(BlueTopo_[A-Za-z0-9_]+_\d{8})', base_name, re.IGNORECASE)
                if match:
                    core_name = match.group(1)
                    bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{core_name}_slope{ext}")
                else:
                    bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{base_name}_slope{ext}")
                    
                if UPath(bluetopo_slope_path).exists():
                    with UPath(bluetopo_slope_path).open('rb') as f_in, open(local_slope, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                else:
                    return (False, f"Required slope file missing for BlueTopo: {bluetopo_slope_path}")

            for out_s3, wbt_func, local_out in missing_wbt:
                try:
                    ret_code = wbt_func(local_bathy, local_out)
                    if ret_code != 0:
                        Engine.write_message_dask(f"[ERROR] WBT Failure on {base_name}: Task returned exit code {ret_code} for {local_out}", OUTPUTS)
                    elif not os.path.exists(local_out):
                        Engine.write_message_dask(f"[ERROR] WBT Silent Failure on {base_name}: Tool succeeded but output {local_out} is missing.", OUTPUTS)
                    else:
                        with open(local_out, 'rb') as f_in, UPath(out_s3).open('wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        if local_out not in [local_slope, local_plan]:
                            try: os.remove(local_out)
                            except OSError: pass
                    gc.collect()
                except Exception as e:
                    Engine.write_message_dask(f"WBT Error on {base_name}: {e}", OUTPUTS)

            if missing_tci:
                try: 
                    ret_code = wbt.convergence_index(local_bathy, local_tci)
                    if ret_code == 0 and os.path.exists(local_tci):
                        with open(local_tci, 'rb') as f_in, UPath(out_tci).open('wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        try: os.remove(local_tci)
                        except OSError: pass
                    gc.collect()
                except Exception as e: 
                    Engine.write_message_dask(f"WBT TCI Error on {base_name}: {e}", OUTPUTS)

            if missing_shear:
                try:
                    if not is_bluetopo and not os.path.exists(local_slope) and UPath(out_slope_deg).exists():
                        with UPath(out_slope_deg).open('rb') as f_in, open(local_slope, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    if not os.path.exists(local_plan) and UPath(out_plan).exists():
                        with UPath(out_plan).open('rb') as f_in, open(local_plan, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    slope_src = local_slope if os.path.exists(local_slope) else None
                    plan_src = local_plan if os.path.exists(local_plan) else None

                    if slope_src and plan_src:
                        with rasterio.open(slope_src) as s, rasterio.open(plan_src) as p:
                            meta = s.meta.copy()
                            s_nodata = s.nodata if s.nodata is not None else -9999.0
                            p_nodata = p.nodata if p.nodata is not None else -9999.0
                            
                            meta.update(compress='LZW', nodata=s_nodata, dtype='float32')
                            
                            out_u = UPath(out_shear)
                            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=str(local_tmp_dir)) as tmp_file:
                                local_shear_path = tmp_file.name
                                
                            with rasterio.open(local_shear_path, 'w', **meta) as dst:
                                for ji, window in s.block_windows(1):
                                    slope_chunk = s.read(1, window=window).astype(np.float32)
                                    plan_chunk = p.read(1, window=window).astype(np.float32)
                                    
                                    valid_mask = ~np.isnan(slope_chunk) & ~np.isnan(plan_chunk) & (slope_chunk != s_nodata) & (plan_chunk != p_nodata)
                                    
                                    shear_chunk = np.full_like(slope_chunk, s_nodata, dtype=np.float32)
                                    shear_chunk[valid_mask] = slope_chunk[valid_mask] * np.abs(plan_chunk[valid_mask])
                                    
                                    dst.write(shear_chunk, 1, window=window)
                                    
                                    del slope_chunk, plan_chunk, valid_mask, shear_chunk
                                    gc.collect()
                                    
                            if out_u.protocol == "s3":
                                out_u.fs.put_file(local_shear_path, str(out_u))
                            else:
                                shutil.copyfile(local_shear_path, str(out_shear))
                                
                            os.remove(local_shear_path)
                except Exception as e:
                    Engine.write_message_dask(f"Shear Proxy Error {base_name}: {e}", OUTPUTS)

            if missing_numpy:
                if is_bluetopo:
                    year = 'BlueTopo'
                else:
                    year = 'bt_bathy'
                    match = re.search(r'((?:19|20)\d{2})', base_name)
                    if match: year = match.group(1)
                    
                dict_path = UPath(dictionary_dir) / f"dictionary_{year}.csv"
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
                        chunk = src.read(1, window=window).astype(np.float32)
                        
                        if src.nodata is not None and not np.isnan(src.nodata):
                            chunk[chunk == src.nodata] = np.nan
                            
                        bathy_array[window.toslices()] = chunk
                        del chunk
                        gc.collect()

                if missing_numpy_dict["_rugosity_tri.tif"]:
                    rugosity = _calculate_tri(bathy_array)
                    profile.update(dtype=rugosity.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    _save_numpy_to_raster(rugosity, out_rug, profile, local_tmp_dir, log_prefix=progress_str)
                    del rugosity; gc.collect()

                if missing_numpy_dict["_slope.tif"]:
                    slope_raw = _calculate_slope(bathy_array, cell_size)
                    profile.update(dtype=slope_raw.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    _save_numpy_to_raster(slope_raw, out_slope, profile, local_tmp_dir, log_prefix=progress_str)
                    if missing_numpy_dict["_terrain_classification.tif"]:
                        slope = np.memmap(os.path.join(tmpdir, "s.dat"), dtype='float32', mode='w+', shape=shape_2d)
                        slope[:] = slope_raw[:]
                    del slope_raw
                elif missing_numpy_dict["_terrain_classification.tif"]:
                    with rasterio.open(local_slope if is_bluetopo else out_slope) as src_s:
                        slope = np.memmap(os.path.join(tmpdir, "s.dat"), dtype='float32', mode='w+', shape=shape_2d)
                        for ji, window in src_s.block_windows(1):
                            slope_chunk = src_s.read(1, window=window).astype(np.float32)
                            if src_s.nodata is not None and not np.isnan(src_s.nodata):
                                slope_chunk[slope_chunk == src_s.nodata] = np.nan
                            slope[window.toslices()] = slope_chunk

                if missing_numpy_dict["_bpi_fine.tif"]:
                    bpi_fine = _calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                    profile.update(dtype=bpi_fine.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    _save_numpy_to_raster(bpi_fine, out_fine, profile, local_tmp_dir, log_prefix=progress_str)
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
                    bpi_broad = _calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                    profile.update(dtype=bpi_broad.dtype.name, nodata=np.nan, count=1, compress='LZW')
                    _save_numpy_to_raster(bpi_broad, out_broad, profile, local_tmp_dir, log_prefix=progress_str)
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
                    _save_numpy_to_raster(classified_array, out_class, profile, local_tmp_dir, log_prefix=progress_str)
                    del classified_array

                if 'slope' in locals():
                    del slope, bpi_fine_mem, bpi_broad_mem
                gc.collect()

            Engine.write_message_dask(f" - [SUCCESS] {progress_str}Completed terrain processing: {base_name}", OUTPUTS)
            return (True, f"Success: {base_name}")
        
        finally:
            def _close_mmap(arr):
                if arr is not None:
                    try:
                        if hasattr(arr, '_mmap'): arr._mmap.close()
                        if hasattr(arr, 'base') and hasattr(arr.base, 'close'): arr.base.close()
                    except Exception: pass
            
            locs = locals()
            for key in ['bathy_array', 'slope', 'bpi_fine_mem', 'bpi_broad_mem', 'classified_array']:
                _close_mmap(locs.get(key))

            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)
        
    except Exception as e:
        err_msg = traceback.format_exc()
        base_name_err = os.path.splitext(os.path.basename(str(bathy_path)))[0]
        Engine.write_message_dask(f"[FATAL ERROR] [{base_name_err}] CRASH during terrain product generation:\n{err_msg}", OUTPUTS)
        return (False, f"Fatal Crash: {base_name_err} - {str(e)}")
    finally:
        gc.collect()

class TerrainProductsEngine(Engine):
    """Class for parallel generation of seabed terrain layers, highly optimized for memory management."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize configurations and environment for terrain products."""
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        
        # Setup local temp dir mapping to ensure EC2 limits aren't exceeded
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "terrain_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']
        self.inputs_dir = INPUTS

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f"TerrainProductsEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        combined_bathy = get_config_item('TERRAIN', 'COMBINED_LIDAR_DIR')
        self.combined_bathy_dir = UPath(f"{s3_dir_base}/{combined_bathy}") if self.is_aws else UPath(self.outputs_dir / combined_bathy)

        terrain_outputs = get_config_item('TERRAIN', 'OUTPUTS')
        self.terrain_outputs_dir = UPath(f"{s3_dir_base}/{terrain_outputs}") if self.is_aws else UPath(self.outputs_dir / terrain_outputs)

        dictionaries_dir = get_config_item('TERRAIN', 'DICTIONARIES_DIR')
        self.dictionary_dir = UPath(f"{s3_dir_base}/{dictionaries_dir}") if self.is_aws else UPath(self.outputs_dir / dictionaries_dir)

        pred_dir = get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR')
        self.prediction_output_dir = UPath(f"{s3_dir_base}/{pred_dir}") if self.is_aws else UPath(self.outputs_dir / pred_dir)

        if not self.is_aws:
            self.combined_bathy_dir.mkdir(parents=True, exist_ok=True)
            self.terrain_outputs_dir.mkdir(parents=True, exist_ok=True)
            self.dictionary_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_output_dir.mkdir(parents=True, exist_ok=True)

    def _get_files_to_process(self) -> List[str]:
        """Scans the directory and filters out invalid or 'iss' specific files."""
        potential_inputs = set()
        
        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            for f in UPath(self.combined_bathy_dir).rglob(ext):
                potential_inputs.add(str(f))

        if hasattr(self, 'prediction_output_dir') and self.prediction_output_dir:
            for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
                for f in UPath(self.prediction_output_dir).glob(ext):
                    potential_inputs.add(str(f))

        valid_files = []
        for f_str in list(potential_inputs):
            fname = UPath(f_str).name.lower()
            if 'iss' in fname:
                continue
                
            if 'bluetopo' in fname:
                if not re.match(r'^bluetopo_[a-z0-9_]+_\d{8}\.tiff?$', fname):
                    continue

            valid_files.append(f_str)
            
        self.write_message(f"Final filtered list: Found {len(valid_files)} bathymetry files to process.", OUTPUTS)
        return valid_files

    def _execute_regional_limits(self, valid_files: List[str], best_radii: Dict[str, Tuple[int, int]]) -> None:
        """Step 1: Execute regional dictionary limits."""
        self.write_message(f"--- PHASE 1: Building Regional Classification Limits in {self.dictionary_dir} ---", OUTPUTS)
        
        year_groups = {}
        for f in valid_files:
            fname = os.path.basename(str(f)).lower()
            if 'bluetopo' in fname:
                year_groups.setdefault('BlueTopo', []).append(f)
            else:
                match = re.search(r'((?:19|20)\d{2})', fname)
                if match:
                    year = match.group(1)
                    year_groups.setdefault(year, []).append(f)

        years = list(year_groups.keys())
        files_lists = list(year_groups.values())
        dict_dirs = [str(self.dictionary_dir)] * len(years)
        radiis = [best_radii] * len(years)
        tmp_dirs = [str(self.local_tmp_dir)] * len(years)

        if years:
            dict_futures = self.client.map(
                _create_regional_dictionary_worker, 
                years, files_lists, dict_dirs, radiis, tmp_dirs
            )
            dict_results = self.client.gather(dict_futures)
            for success, msg in dict_results:
                self.write_message(msg, OUTPUTS)
        
        self.write_message(self.log_system_metrics(), OUTPUTS)

    def _execute_terrain_generation(self, valid_files: List[str], best_radii: Dict[str, Tuple[int, int]]) -> None:
        """Step 2: Generate terrain products iteratively."""
        self.write_message(f"--- PHASE 2: Parallel Terrain Product Generation ---", OUTPUTS)
        
        paths = valid_files
        radiis_list = [best_radii] * len(paths)
        terr_outs = [str(self.terrain_outputs_dir)] * len(paths)
        pred_outs = [str(self.prediction_output_dir)] * len(paths)
        dict_dirs_list = [str(self.dictionary_dir)] * len(paths)
        loc_tmps = [str(self.local_tmp_dir)] * len(paths)
        indices = list(range(1, len(paths) + 1))
        totals = [len(paths)] * len(paths)
        
        terrain_futures = self.client.map(
            _process_terrain_raster_worker, 
            paths, radiis_list, terr_outs, pred_outs, dict_dirs_list, loc_tmps, indices, totals
        )
        
        terrain_results = self.client.gather(terrain_futures)
        
        for success, msg in terrain_results:
            if success:
                self.write_message(f"[SUCCESS] {msg}", OUTPUTS)
            else:
                self.write_message(f"[ERROR] {msg}", OUTPUTS)
                
        self.write_message(self.log_system_metrics(), OUTPUTS)
        self.write_message("[SUCCESS] Terrain raster processing complete.", OUTPUTS)

    def run(self, max_concurrent: int = 4) -> None:
        """Main entry point for evaluating directories and processing rasters in parallel."""
        env = self.param_lookup.get('env', 'local')
        
        try:
            self.setup_dask(env, n_workers=max_concurrent, threads_per_worker=1, memory_limit="6GB")

            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)
                
                valid_files = self._get_files_to_process()
                if not valid_files:
                    self.write_message(f"No bathymetry files found to process for region {eco_region}.", OUTPUTS)
                    continue
                
                best_radii = {'fine': (8, 32), 'broad': (80, 240)}
                
                # Step 1: Execute Regional Limits
                self._execute_regional_limits(valid_files, best_radii)
                
                # Step 2: Parallel Terrain Product Generation
                self._execute_terrain_generation(valid_files, best_radii)
                
        finally:
            self.cleanup_resources(OUTPUTS)