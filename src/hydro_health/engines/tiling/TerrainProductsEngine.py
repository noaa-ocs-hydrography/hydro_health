"""Class for parallel processing terrain products (Slope, Rugosity, BPI, Classification, TCI)"""

import os

# Bound native-library caches and thread pools before importing GDAL, NumPy,
# SciPy, or Whitebox. This keeps a single Dask tile task from silently creating
# additional native worker pools or an oversized GDAL block cache.
os.environ.setdefault("GDAL_CACHEMAX", "512")
os.environ.setdefault("VSI_CACHE", "FALSE")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "16384")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import gc
import shutil
import tempfile
import warnings
import traceback
import pathlib
import sys
import subprocess
import logging
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.signal import fftconvolve

import dask
import dask.array as da
from upath import UPath

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

# Global Base Paths
INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'

DEFAULT_DASK_CHUNK_SIZE = 1024
MAX_BPI_OUTER_CELLS = 512

# Tags invalidate legacy outputs whose filename stayed the same while the
# calculation changed. They prevent old aspect/TRI rasters from being mistaken
# for the corrected D8-pointer/surface-rugosity products.
PRODUCT_VERSIONS = {
    "_flowdir.tif": "d8_pointer_v1",
    "_rugosity.tif": "surface_area_ratio_v1",
    "_tci.tif": "neighbor_variance_v2",
    "_slope.tif": "central_difference_nodata_edges_v2",
    "_gradmag.tif": "slope_radians_zero_valid_v2",
    "_bpi_fine.tif": "annular_bpi_nodata_edges_v2",
    "_bpi_broad.tif": "annular_bpi_nodata_edges_v2",
    "_terrain_classification.tif": "bpi_nodata_edges_v2",
}


def _silence_aws_credential_discovery_logs() -> None:
    """Hide repetitive IAM credential discovery while preserving real warnings."""
    logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
    logging.getLogger("aiobotocore.credentials").setLevel(logging.WARNING)


_silence_aws_credential_discovery_logs()


def _configure_whitebox_headless_windows(whitebox_tools_class) -> None:
    """Prevent Whitebox executable console windows from flashing on Windows."""
    if os.name != "nt":
        return

    whitebox_module = sys.modules.get(whitebox_tools_class.__module__)
    if whitebox_module is None:
        return
    if getattr(whitebox_module, "_hydro_health_hidden_popen", False):
        return

    original_popen = getattr(whitebox_module, "Popen", None)
    if original_popen is None:
        return

    def hidden_popen(*args, **kwargs):
        kwargs["creationflags"] = (
            kwargs.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
        )

        startup_info = kwargs.get("startupinfo")
        if startup_info is None:
            startup_info = subprocess.STARTUPINFO()
            kwargs["startupinfo"] = startup_info
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup_info.wShowWindow = 0  # Windows SW_HIDE

        return original_popen(*args, **kwargs)

    whitebox_module.Popen = hidden_popen
    whitebox_module._hydro_health_hidden_popen = True


def _is_s3_path(path: str | UPath) -> bool:
    """Return True for S3 UPaths across upath/fsspec versions."""
    protocol = UPath(path).protocol
    if isinstance(protocol, (tuple, list)):
        return "s3" in protocol
    return protocol == "s3"


def _validate_raster(
    path: str | UPath,
    expected_shape: Tuple[int, int] | None = None,
    expected_crs=None,
    expected_transform=None,
    full_read: bool = False,
) -> Tuple[bool, str]:
    """Validate that a raster exists, opens, has data, and matches its expected grid."""
    raster_path = UPath(path)
    try:
        if not raster_path.exists():
            return False, "missing"
        if raster_path.stat().st_size <= 0:
            return False, "empty file"

        with rasterio.open(str(raster_path)) as src:
            if src.count < 1 or src.width <= 0 or src.height <= 0:
                return False, "invalid dimensions or band count"
            if expected_shape is not None and src.shape != expected_shape:
                return False, f"shape {src.shape} != {expected_shape}"
            if expected_crs is not None and src.crs != expected_crs:
                return False, f"CRS {src.crs} != {expected_crs}"
            if expected_transform is not None and not src.transform.almost_equals(expected_transform):
                return False, "transform mismatch"

            # Force GDAL to read raster bytes rather than accepting metadata alone.
            if full_read:
                for _, window in src.block_windows(1):
                    src.read(1, window=window)
            else:
                src.read(1, window=Window(0, 0, 1, 1))
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _validate_product_raster(
    path: str | UPath,
    suffix: str,
    expected_shape: Tuple[int, int] | None = None,
    expected_crs=None,
    expected_transform=None,
) -> Tuple[bool, str]:
    """Validate a raster and, when defined, its algorithm-version tag."""
    valid, reason = _validate_raster(
        path,
        expected_shape=expected_shape,
        expected_crs=expected_crs,
        expected_transform=expected_transform,
    )
    if not valid:
        return valid, reason

    expected_version = PRODUCT_VERSIONS.get(suffix)
    if expected_version is None:
        return True, ""

    try:
        with rasterio.open(str(UPath(path))) as src:
            actual_version = src.tags().get("hydro_health_product_version")
        if actual_version != expected_version:
            return False, (
                f"legacy or mismatched product version {actual_version!r}; "
                f"expected {expected_version!r}"
            )
        return True, ""
    except Exception as exc:
        return False, str(exc)


def _publish_local_raster(local_path: str, out_path: str) -> None:
    """Validate locally, then publish through a temporary destination name."""
    # Read every encoded block locally so a damaged LZW tile is caught before
    # it can be published to local storage or S3.
    valid, reason = _validate_raster(local_path, full_read=True)
    if not valid:
        raise RuntimeError(f"Refusing to publish invalid raster {local_path}: {reason}")

    out_u = UPath(out_path)
    token = uuid.uuid4().hex
    if _is_s3_path(out_u):
        out_u.parent.mkdir(parents=True, exist_ok=True)
        # A completed S3 PUT becomes visible atomically. Uploading directly
        # avoids a second server-side copy and prevents GDAL from seeing a key
        # while an immediately-following move/overwrite is still settling.
        out_u.fs.put_file(local_path, str(out_u))
        try:
            out_u.fs.invalidate_cache(str(out_u))
        except Exception:
            pass
    else:
        destination = pathlib.Path(str(out_u))
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Do not append the token to the already-long raster filename. Windows
        # commonly enforces a 260-character path limit, and terrain filenames
        # can be close to it before an atomic-write suffix is added.
        partial_path = destination.parent / f".hh-{token[:8]}.tmp"
        try:
            shutil.copyfile(local_path, partial_path)
            os.replace(partial_path, destination)
        finally:
            try:
                if partial_path.exists():
                    partial_path.unlink()
            except OSError:
                pass


def _materialize_raster(source_path: str | UPath, local_path: str) -> str:
    """Copy a local or remote raster to a fresh local file for reliable reads."""
    with UPath(source_path).open("rb") as source, open(local_path, "wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    valid, reason = _validate_raster(local_path, full_read=True)
    if not valid:
        raise RuntimeError(f"Invalid materialized raster {UPath(source_path).name}: {reason}")
    return local_path


def _metric_cell_size(src: rasterio.io.DatasetReader) -> float:
    """Return square pixel size in metres, rejecting incompatible grids."""
    if src.crs is None or not src.crs.is_projected:
        raise ValueError(f"Raster must use a projected CRS in metres; got {src.crs}.")

    try:
        unit_name, unit_factor = src.crs.linear_units_factor
    except Exception as exc:
        raise ValueError(f"Could not determine CRS linear units for {src.crs}.") from exc

    if not np.isclose(float(unit_factor), 1.0):
        raise ValueError(
            f"Raster linear units must be metres; got {unit_name} "
            f"(metre conversion factor {unit_factor})."
        )

    x_size, y_size = abs(float(src.res[0])), abs(float(src.res[1]))
    if x_size <= 0 or y_size <= 0:
        raise ValueError(f"Invalid raster resolution: {src.res}.")
    if not np.isclose(x_size, y_size, rtol=1e-4, atol=1e-9):
        raise ValueError(f"BPI and slope require square pixels; got {src.res}.")
    return x_size

def _save_memmap_to_raster(
    mmap_array: np.ndarray,
    out_path: str,
    profile: dict,
    local_tmp_dir: str,
    log_prefix: str = "",
    product_suffix: str | None = None,
):
    """Stream a memory-mapped array block-by-block into a validated raster."""
    out_u = UPath(out_path)
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=local_tmp_dir) as tmp_file:
        local_tmp_path = tmp_file.name
        
    try:
        with rasterio.open(local_tmp_path, 'w', **profile) as dst:
            for ji, window in dst.block_windows(1):
                dst.write(mmap_array[window.toslices()], 1, window=window)
            if product_suffix in PRODUCT_VERSIONS:
                dst.update_tags(
                    hydro_health_product_version=PRODUCT_VERSIONS[product_suffix]
                )
                
        _publish_local_raster(local_tmp_path, str(out_u))
            
        Engine.write_message_dask(f"{log_prefix}Successfully wrote layer to: {out_path}", OUTPUTS)
    finally:
        if os.path.exists(local_tmp_path):
            try: os.remove(local_tmp_path)
            except OSError: pass

def _save_dask_to_raster(
    d_array: da.Array,
    out_path: str,
    profile: dict,
    local_tmp_dir: str,
    log_prefix: str = "",
    product_suffix: str | None = None,
):
    """Compute a chunked Dask array into a disk memmap, then publish it."""
    with tempfile.NamedTemporaryFile(suffix='.dat', delete=False, dir=local_tmp_dir) as tmp_mmap:
        mmap_path = tmp_mmap.name
        
    try:
        # The final array remains disk-backed; individual calculations such as
        # FFT convolution still need bounded in-memory working arrays.
        mmap_arr = np.memmap(mmap_path, dtype='float32', mode='w+', shape=d_array.shape)
        
        # This is a nested, tile-local graph. Keep it inside the worker instead
        # of submitting its array/kernel graph to the distributed scheduler.
        with dask.config.set(scheduler='single-threaded'):
            da.store(d_array, mmap_arr, compute=True)
            
        _save_memmap_to_raster(
            mmap_arr,
            out_path,
            profile,
            local_tmp_dir,
            log_prefix,
            product_suffix=product_suffix,
        )
    finally:
        if 'mmap_arr' in locals():
            try: 
                mmap_arr._mmap.close()
            except Exception: pass
        if os.path.exists(mmap_path):
            try: os.remove(mmap_path)
            except OSError: pass

def _get_worker_metrics(tmp_dir: str) -> str:
    """Safely fetches worker system metrics (RAM, Disk, Tmp Size)."""
    try:
        import psutil
        import shutil
        import os
        
        vm = psutil.virtual_memory()
        ram_free = vm.available / (1024**3)
        ram_total = vm.total / (1024**3)
        ram_used_pct = vm.percent
        
        du = shutil.disk_usage(tmp_dir)
        disk_free = du.free / (1024**3)
        disk_total = du.total / (1024**3)
        
        tmp_size = 0
        if os.path.exists(tmp_dir):
            for path, dirs, files in os.walk(tmp_dir):
                for f in files:
                    fp = os.path.join(path, f)
                    if not os.path.islink(fp):
                        tmp_size += os.path.getsize(fp)
        tmp_mb = tmp_size / (1024**2)
        
        return f"[SysMetrics] RAM | Free: {ram_free:.1f}GB / {ram_total:.1f}GB (Used: {ram_used_pct}%) || Disk Free | {disk_free:.1f}GB / {disk_total:.1f}GB || Tmp Dir Size | {tmp_mb:.1f}MB"
    except Exception:
        return ""

def _calculate_bpi_dask(d_bathy: da.Array, cell_size: float, inner_radius: float, outer_radius: float) -> da.Array:
    """Return a lazy, NoData-aware annular BPI array."""
    if cell_size <= 0:
        raise ValueError(f"Invalid cell_size ({cell_size}). Cannot calculate BPI.")
        
    inner_cells = int(round(inner_radius / cell_size))
    outer_cells = int(round(outer_radius / cell_size))
    
    if inner_cells < 0 or outer_cells <= inner_cells:
        raise ValueError(
            f"Invalid BPI radii after conversion to cells: inner={inner_cells}, "
            f"outer={outer_cells}."
        )
    if outer_cells > MAX_BPI_OUTER_CELLS:
        raise ValueError(
            f"BPI outer radius becomes {outer_cells} cells at {cell_size} m resolution. "
            f"The safe limit is {MAX_BPI_OUTER_CELLS}; resample the raster or reduce the radius."
        )
    
    y, x = np.ogrid[-outer_cells:outer_cells + 1, -outer_cells:outer_cells + 1]
    mask = x**2 + y**2 <= outer_cells**2
    mask[x**2 + y**2 <= inner_cells**2] = False
    
    kernel = mask.astype(np.float32)
    
    d_valid = da.map_blocks(lambda b: (~np.isnan(b)).astype(np.float32), d_bathy, dtype=np.float32)
    d_bathy_zeroed = da.where(da.isnan(d_bathy), 0.0, d_bathy).astype(np.float32)
    
    def _conv(block):
        return fftconvolve(block, kernel, mode='same').astype(np.float32)
        
    # Outside the raster is NoData. Reflecting the seabed at tile edges biases BPI.
    sum_array = d_bathy_zeroed.map_overlap(
        _conv, depth=outer_cells, boundary=0.0, dtype=np.float32
    )
    count_array = d_valid.map_overlap(
        _conv, depth=outer_cells, boundary=0.0, dtype=np.float32
    )

    def _safe_mean(sum_block, count_block):
        result = np.full(sum_block.shape, np.nan, dtype=np.float32)
        np.divide(
            sum_block,
            count_block,
            out=result,
            where=count_block > 0.5,
        )
        return result

    # da.where evaluates both branches, so sum/count still emitted divide-by-zero
    # warnings over NoData regions. np.divide(where=...) avoids doing the invalid
    # operation in the first place.
    mean_annulus = da.map_blocks(
        _safe_mean,
        sum_array,
        count_array,
        dtype=np.float32,
    )
    return d_bathy - mean_annulus.astype(np.float32)

def _calculate_bpi(bathy_array: np.ndarray, cell_size: float, inner_radius: float, outer_radius: float) -> np.ndarray:
    """Wrapper for dictionary creation worker which operates on small sampled numpy arrays."""
    d_bathy = da.from_array(
        bathy_array,
        chunks=(DEFAULT_DASK_CHUNK_SIZE, DEFAULT_DASK_CHUNK_SIZE),
    )
    bpi_lazy = _calculate_bpi_dask(d_bathy, cell_size, inner_radius, outer_radius)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return bpi_lazy.compute(scheduler='single-threaded')

def _calculate_slope(bathy_array: np.ndarray, cell_size: float) -> np.ndarray:
    """Core slope calculation (degrees). Optimized for small arrays or Dask chunks."""
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

def _calculate_slope_dask(d_bathy: da.Array, cell_size: float) -> da.Array:
    """Dask wrapper to stream slope calculation natively across blocks without RAM accumulation."""
    return d_bathy.map_overlap(
        lambda block: _calculate_slope(block, cell_size),
        depth=1,
        boundary=np.nan,
        dtype=np.float32,
    )

def _calculate_tci(bathy_array: np.ndarray) -> np.ndarray:
    """Core Terrain Complexity Index (TCI)."""
    sum_diff = np.zeros_like(bathy_array, dtype=np.float32)
    sum_sq_diff = np.zeros_like(bathy_array, dtype=np.float32)
    valid_count = np.zeros_like(bathy_array, dtype=np.float32)
    
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            y1, y2 = max(0, dy), bathy_array.shape[0] + min(0, dy)
            x1, x2 = max(0, dx), bathy_array.shape[1] + min(0, dx)
            sy1, sy2 = max(0, -dy), bathy_array.shape[0] + min(0, -dy)
            sx1, sx2 = max(0, -dx), bathy_array.shape[1] + min(0, -dx)
            
            neighbor = bathy_array[y1:y2, x1:x2]
            center = bathy_array[sy1:sy2, sx1:sx2]
            
            invalid_mask = np.isnan(neighbor) | np.isnan(center)
            diff = np.where(invalid_mask, 0.0, neighbor - center)
            
            sum_diff[sy1:sy2, sx1:sx2] += diff
            sum_sq_diff[sy1:sy2, sx1:sx2] += diff ** 2
            valid_count[sy1:sy2, sx1:sx2] += (~invalid_mask).astype(np.float32)
            
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_diff = sum_diff / valid_count
        variance = (sum_sq_diff / valid_count) - (mean_diff ** 2)
        return np.where(valid_count > 1, np.sqrt(np.maximum(variance, 0.0)), np.nan)

def _calculate_tci_dask(d_bathy: da.Array) -> da.Array:
    """Dask wrapper to stream TCI natively across blocks."""
    return d_bathy.map_overlap(_calculate_tci, depth=1, boundary=np.nan, dtype=np.float32)

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

def _create_regional_dictionary_worker(year: str, files: List[str], dictionary_dir: str, best_radii: Dict[str, Tuple[int, int]], local_tmp_dir: str, prediction_output_dir: str = "") -> Tuple[bool, str]:
    """Module-level worker to sample valid pixels and generate regional class limits."""
    dict_path = UPath(dictionary_dir) / f"dictionary_{year}.csv"
    
    if dict_path.exists():
        return True, f"Skipping dictionary creation for {year} (Already exists)"
        
    Engine.write_message_dask(f"Processing regional dictionary for: {year} ({len(files)} files)", OUTPUTS)
    metrics = _get_worker_metrics(str(local_tmp_dir))
    if metrics: Engine.write_message_dask(metrics, OUTPUTS)
    
    try:
        def _getsize(path):
            try: return UPath(path).stat().st_size
            except Exception: return 0

        file_data = [(f, _getsize(f)) for f in files]
        all_sizes = [x[1] for x in file_data if x[1] > 0]
        size_threshold = np.percentile(all_sizes, 30) if all_sizes else 0
        small_files_pool = [x[0] for x in file_data if 0 < x[1] <= size_threshold]

        if not small_files_pool:
            return False, f"Failed dictionary creation for {year}: no readable input files."

        files_to_sample = small_files_pool if len(small_files_pool) <= 10 else list(np.random.choice(small_files_pool, 10, replace=False))
        all_samples = {'slope': [], 'bpi_fine': [], 'bpi_broad': []}
        sample_errors = []
        
        for f in files_to_sample:
            try:
                with rasterio.open(str(f)) as src:
                    try:
                        cell_size = _metric_cell_size(src)
                    except ValueError as exc:
                        Engine.write_message_dask(
                            f"[ERROR] Skipping file {UPath(f).name} for dictionary sampling: {exc}",
                            OUTPUTS,
                        )
                        continue

                    bathy_array = src.read(1).astype(np.float32)
                    
                    if src.nodata is not None and not np.isnan(src.nodata):
                        bathy_array[bathy_array == src.nodata] = np.nan
                        
                    bathy_array[bathy_array < -9998.0] = np.nan
                    bathy_array[bathy_array >= 0.0] = np.nan
                        
                    slope_sample = None
                    if year == 'BlueTopo' and prediction_output_dir:
                        base_name = UPath(f).stem
                        ext = UPath(f).suffix
                        match = re.search(r'(BlueTopo_[A-Za-z0-9_]+_\d{8})', base_name, re.IGNORECASE)
                        core_name = match.group(1) if match else base_name
                        
                        bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{core_name}_slope{ext}")
                        if not UPath(bluetopo_slope_path).exists():
                            bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{base_name}_slope{ext}")
                            
                        if UPath(bluetopo_slope_path).exists():
                            with rasterio.open(bluetopo_slope_path) as src_ext:
                                ext_slope = src_ext.read(1).astype(np.float32)
                                e_nodata = src_ext.nodata
                                if ext_slope.shape == bathy_array.shape:
                                    if e_nodata is not None and not np.isnan(e_nodata):
                                        ext_slope[ext_slope == e_nodata] = np.nan
                                    
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore", category=RuntimeWarning)
                                        artifact_mask = ext_slope > 75.0
                                        
                                    bathy_array[artifact_mask] = np.nan
                                    slope_sample = ext_slope
                                    slope_sample[artifact_mask] = np.nan
                                    
                    if slope_sample is None:
                        slope_sample = _calculate_slope(bathy_array, cell_size)

                    valid_pixels = np.argwhere(~np.isnan(bathy_array))
                    if len(valid_pixels) == 0: continue
                    
                    sample_indices = valid_pixels[np.random.choice(len(valid_pixels), min(len(valid_pixels), 20000), replace=False)]
                    
                    bpi_fine_sample = _calculate_bpi(bathy_array, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                    bpi_broad_sample = _calculate_bpi(bathy_array, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                    
                    rows, cols = sample_indices[:, 0], sample_indices[:, 1]
                    all_samples['slope'].append(slope_sample[rows, cols])
                    all_samples['bpi_fine'].append(bpi_fine_sample[rows, cols])
                    all_samples['bpi_broad'].append(bpi_broad_sample[rows, cols])
                    del bathy_array, slope_sample, bpi_fine_sample, bpi_broad_sample
                    gc.collect()
            except Exception as exc:
                sample_errors.append(f"{UPath(f).name}: {exc}")
                Engine.write_message_dask(
                    f"[WARNING] Dictionary sampling failed for {UPath(f).name}: {exc}",
                    OUTPUTS,
                )
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

        details = "; ".join(sample_errors[:5]) if sample_errors else "no valid bathymetry pixels"
        return False, f"Failed dictionary creation for {year}: {details}"
    except Exception as e:
        err = traceback.format_exc()
        return False, f"Failed dictionary creation for {year}: {e}\n{err}"
    finally:
        gc.collect()

def _check_unprocessed_worker(bathy_path: str, terrain_outputs_dir: str, prediction_output_dir: str) -> bool:
    """Return True when any required output is missing or unreadable."""
    try:
        base_name = os.path.splitext(os.path.basename(str(bathy_path)))[0]
        is_bluetopo = 'bluetopo' in base_name.lower()
        
        out_dir_path = UPath(terrain_outputs_dir)
        def resolve_out_path(suffix): return str(out_dir_path / (base_name + suffix))
        
        wbt_suffixes = [
            "_slope_deg.tif", "_flowdir.tif", "_curv_profile.tif", 
            "_curv_plan.tif", "_curv_total.tif", "_flowacc.tif"
        ]
        for suffix in wbt_suffixes:
            if not _validate_product_raster(resolve_out_path(suffix), suffix)[0]:
                return True
            
        if not _validate_raster(resolve_out_path("_shearproxy.tif"))[0]:
            return True
        
        numpy_suffixes = [
            "_rugosity.tif", "_bpi_fine.tif", "_bpi_broad.tif", 
            "_terrain_classification.tif", "_gradmag.tif", "_tci.tif"
        ]
        for suffix in numpy_suffixes:
            if not _validate_product_raster(resolve_out_path(suffix), suffix)[0]:
                return True
            
        if not is_bluetopo and not _validate_product_raster(
            resolve_out_path("_slope.tif"), "_slope.tif"
        )[0]:
            return True
        
        return False
    except Exception:
        # Default to processing if the check fails for any reason
        return True

def _process_terrain_raster_worker(bathy_path: str, current_index: int, total_count: int, best_radii: Dict[str, Tuple[int, int]], terrain_outputs_dir: str, prediction_output_dir: str, dictionary_dir: str, local_tmp_dir: str) -> Tuple[bool, str]:
    """Module-level worker to process one bathymetry raster, completely detached from the class."""

    # Dask worker processes configure logging independently of the driver.
    _silence_aws_credential_discovery_logs()

    # WhiteboxTools assumes stdout/stderr are valid streams and calls flush()
    # internally. Dask worker processes on Windows can have either set to None.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

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
        out_flowacc   = resolve_out_path("_flowacc.tif")
        out_shear     = resolve_out_path("_shearproxy.tif")
        out_tci       = resolve_out_path("_tci.tif")
        out_rug       = resolve_out_path("_rugosity.tif")
        out_slope     = resolve_out_path("_slope.tif")
        out_fine      = resolve_out_path("_bpi_fine.tif")
        out_broad     = resolve_out_path("_bpi_broad.tif")
        out_class     = resolve_out_path("_terrain_classification.tif")
        
        try:
            from whitebox import WhiteboxTools
        except ImportError:
            return (False, f"Failed: Whitebox library is not installed in the worker environment.")

        _configure_whitebox_headless_windows(WhiteboxTools)
        wbt = WhiteboxTools()
        if os.name == "nt":
            # Whitebox's wrapper supplies STARTUPINFO when this flag is true;
            # the headless Popen wrapper above changes it from minimized to hidden.
            wbt.start_minimized = True
        wbt.verbose = False
        wbt.set_default_callback(lambda x: None)
        if wbt.set_compress_rasters(True) != 0:
            return (False, "Failed to configure Whitebox raster compression.")
        if wbt.set_max_procs(1) != 0:
            return (False, "Failed to configure Whitebox maximum processors.")

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
            local_flowacc = os.path.join(tmpdir, "flowacc.tif")
            local_rug = os.path.join(tmpdir, "rugosity.tif")

            outputs_wbt = [
                (out_slope_deg, lambda i, o: wbt.slope(i, o, units="degrees"), local_slope, "_slope_deg.tif"),
                (out_flowdir, lambda i, o: wbt.d8_pointer(i, o, esri_pntr=False), local_flowdir, "_flowdir.tif"),
                (out_prof, wbt.profile_curvature, local_prof, "_curv_profile.tif"),
                (out_plan, wbt.plan_curvature, local_plan, "_curv_plan.tif"),
                (out_total, wbt.total_curvature, local_total, "_curv_total.tif"),
                (out_flowacc, lambda i, o: wbt.d8_flow_accumulation(i, o, out_type="cells"), local_flowacc, "_flowacc.tif"),
                (out_rug, wbt.surface_area_ratio, local_rug, "_rugosity.tif"),
            ]

            missing_wbt = [
                item for item in outputs_wbt
                if not _validate_product_raster(item[0], item[3])[0]
            ]
            missing_shear = not _validate_raster(out_shear)[0]

            missing_numpy_dict = {
                "_slope.tif": False if is_bluetopo else (not _validate_product_raster(out_slope, "_slope.tif")[0]),
                "_bpi_fine.tif": not _validate_product_raster(out_fine, "_bpi_fine.tif")[0],
                "_bpi_broad.tif": not _validate_product_raster(out_broad, "_bpi_broad.tif")[0],
                "_terrain_classification.tif": not _validate_product_raster(out_class, "_terrain_classification.tif")[0],
                "_gradmag.tif": not _validate_product_raster(out_gradmag, "_gradmag.tif")[0],
                "_tci.tif": not _validate_product_raster(out_tci, "_tci.tif")[0],
            }

            # Classification must never consume a BPI file that previously
            # failed during encoded-tile reads. If classification is missing,
            # rebuild both BPI dependencies in this same worker invocation.
            if missing_numpy_dict["_terrain_classification.tif"]:
                missing_numpy_dict["_bpi_fine.tif"] = True
                missing_numpy_dict["_bpi_broad.tif"] = True

            missing_numpy = any(missing_numpy_dict.values())

            if not (len(missing_wbt) > 0 or missing_shear or missing_numpy):
                 return (True, f"Skipped: {base_name} (All exist)")

            Engine.write_message_dask(f"-> [STARTING] {progress_str}Generating products for: {base_name}", OUTPUTS)
            metrics = _get_worker_metrics(str(local_tmp_dir))
            if metrics: Engine.write_message_dask(metrics, OUTPUTS)
            product_errors = []

            with UPath(bathy_path).open('rb') as f_in, open(local_bathy_raw, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
                
            with rasterio.open(local_bathy_raw) as src:
                try:
                    cell_size = _metric_cell_size(src)
                except ValueError as exc:
                    err_msg = f"ERROR: Invalid grid for {base_name}: {exc}"
                    Engine.write_message_dask(err_msg, OUTPUTS)
                    return (False, err_msg)

                profile = src.profile.copy()
                s_nodata = src.nodata
                source_shape = src.shape
                source_crs = src.crs
                source_transform = src.transform
                
                profile.update(nodata=-9999.0, dtype='float32', tiled=True, blockxsize=256, blockysize=256)
                
                with rasterio.open(local_bathy, 'w', **profile) as dst:
                    for ji, window in src.block_windows(1):
                        chunk = src.read(1, window=window).astype(np.float32)
                        if s_nodata is not None and not np.isnan(s_nodata):
                            chunk[np.isclose(chunk, s_nodata)] = -9999.0
                        
                        chunk[chunk < -9998.0] = -9999.0
                        chunk[chunk >= -0.01] = -9999.0
                        chunk[np.isnan(chunk)] = -9999.0
                        
                        dst.write(chunk, 1, window=window)

            try: os.remove(local_bathy_raw)
            except OSError: pass
            gc.collect()

            for out_s3, wbt_func, local_out, product_suffix in missing_wbt:
                try:
                    ret_code = wbt_func(local_bathy, local_out)
                    if ret_code != 0:
                        message = f"WBT returned exit code {ret_code} for {os.path.basename(local_out)}"
                        product_errors.append(message)
                        Engine.write_message_dask(f"[ERROR] {base_name}: {message}", OUTPUTS)
                    elif not os.path.exists(local_out):
                        message = f"WBT reported success but {os.path.basename(local_out)} is missing"
                        product_errors.append(message)
                        Engine.write_message_dask(f"[ERROR] {base_name}: {message}", OUTPUTS)
                    else:
                        if product_suffix in PRODUCT_VERSIONS:
                            with rasterio.open(local_out, "r+") as product_dst:
                                product_dst.update_tags(
                                    hydro_health_product_version=PRODUCT_VERSIONS[product_suffix]
                                )
                        _publish_local_raster(local_out, out_s3)
                        if local_out not in [local_slope, local_plan]:
                            try: os.remove(local_out)
                            except OSError: pass
                    gc.collect()
                except Exception as e:
                    message = f"WBT error for {os.path.basename(local_out)}: {e}"
                    product_errors.append(message)
                    Engine.write_message_dask(f"[ERROR] {base_name}: {message}", OUTPUTS)

            if missing_shear:
                try:
                    if not os.path.exists(local_slope) and UPath(out_slope_deg).exists():
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

                            meta.update(compress='LZW', tiled=True, blockxsize=256, blockysize=256, nodata=s_nodata, dtype='float32')

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
                                    
                            _publish_local_raster(local_shear_path, str(out_u))
                            os.remove(local_shear_path)
                    else:
                        message = "Cannot generate shear proxy because slope or plan curvature is missing"
                        product_errors.append(message)
                        Engine.write_message_dask(f"[ERROR] {base_name}: {message}", OUTPUTS)
                except Exception as e:
                    message = f"Shear proxy error: {e}"
                    product_errors.append(message)
                    Engine.write_message_dask(f"[ERROR] {base_name}: {message}", OUTPUTS)

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
                    profile = src.profile.copy()
                    cell_size = _metric_cell_size(src)
                    shape_2d = (src.height, src.width)
                    
                    bathy_array = np.memmap(os.path.join(tmpdir, "bathy.dat"), dtype='float32', mode='w+', shape=shape_2d)
                    for ji, window in src.block_windows(1):
                        chunk = src.read(1, window=window).astype(np.float32)
                        if src.nodata is not None and not np.isnan(src.nodata):
                            chunk[chunk == src.nodata] = np.nan
                        bathy_array[window.toslices()] = chunk
                        del chunk
                        
                # -------------------------------------------------------------
                # Block-wise Masking and Gradmag Generation for BlueTopo
                # -------------------------------------------------------------
                if is_bluetopo:
                    ext = UPath(bathy_path).suffix
                    match = re.search(r'(BlueTopo_[A-Za-z0-9_]+_\d{8})', base_name, re.IGNORECASE)
                    core_name = match.group(1) if match else base_name
                    
                    bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{core_name}_slope{ext}")
                    if not UPath(bluetopo_slope_path).exists():
                        bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{base_name}_slope{ext}")

                    if not UPath(bluetopo_slope_path).exists():
                        raise FileNotFoundError(
                            f"Missing external BlueTopo slope required for masking and gradmag: "
                            f"{bluetopo_slope_path}"
                        )

                    with rasterio.open(bluetopo_slope_path) as src_ext:
                        aligned, reason = _validate_raster(
                            bluetopo_slope_path,
                            expected_shape=shape_2d,
                            expected_crs=source_crs,
                            expected_transform=source_transform,
                        )
                        if not aligned:
                            raise ValueError(
                                f"External BlueTopo slope is not aligned with {base_name}: {reason}"
                            )

                        if missing_numpy_dict["_gradmag.tif"]:
                            gradmag_mmap = np.memmap(
                                os.path.join(tmpdir, "gradmag.dat"),
                                dtype='float32',
                                mode='w+',
                                shape=shape_2d,
                            )

                        e_nodata = src_ext.nodata
                        for ji, window in src_ext.block_windows(1):
                            ext_chunk = src_ext.read(1, window=window).astype(np.float32)
                            if e_nodata is not None and not np.isnan(e_nodata):
                                ext_chunk[ext_chunk == e_nodata] = np.nan

                            artifact_mask = (
                                ~np.isfinite(ext_chunk)
                                | (ext_chunk < 0.0)
                                | (ext_chunk > 75.0)
                            )

                            bathy_chunk = bathy_array[window.toslices()]
                            bathy_chunk[artifact_mask] = np.nan
                            bathy_array[window.toslices()] = bathy_chunk

                            if missing_numpy_dict["_gradmag.tif"]:
                                ext_chunk[artifact_mask] = np.nan
                                # A zero-degree slope is valid flat terrain, not NoData.
                                gradmag_mmap[window.toslices()] = np.radians(ext_chunk)

                        if missing_numpy_dict["_gradmag.tif"]:
                            profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                            _save_memmap_to_raster(gradmag_mmap, out_gradmag, profile, local_tmp_dir, log_prefix=progress_str, product_suffix="_gradmag.tif")
                            del gradmag_mmap
                gc.collect()

                # Stream Memmap as Dask for remaining products without allocating the full array
                d_bathy = da.from_array(
                    bathy_array,
                    chunks=(DEFAULT_DASK_CHUNK_SIZE, DEFAULT_DASK_CHUNK_SIZE),
                )

                if missing_numpy_dict["_tci.tif"]: 
                    tci_lazy = _calculate_tci_dask(d_bathy)
                    profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                    _save_dask_to_raster(tci_lazy, out_tci, profile, local_tmp_dir, log_prefix=progress_str, product_suffix="_tci.tif")
                    del tci_lazy; gc.collect()

                if missing_numpy_dict["_slope.tif"]:
                    slope_lazy = _calculate_slope_dask(d_bathy, cell_size)
                    profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                    _save_dask_to_raster(slope_lazy, out_slope, profile, local_tmp_dir, log_prefix=progress_str, product_suffix="_slope.tif")
                    del slope_lazy; gc.collect()

                # Gradmag and classification both consume the custom slope.
                # Fully decode a local copy before reuse because opening an S3
                # GeoTIFF and reading its first block cannot detect corruption
                # in a later compressed tile. Regenerate only this dependency
                # when an existing slope fails the full local read.
                local_validated_slope = None
                slope_dependency_needed = (
                    not is_bluetopo
                    and (
                        missing_numpy_dict["_gradmag.tif"]
                        or missing_numpy_dict["_terrain_classification.tif"]
                    )
                )
                if slope_dependency_needed:
                    local_validated_slope = os.path.join(tmpdir, "validated_slope.tif")
                    try:
                        _materialize_raster(out_slope, local_validated_slope)
                    except RuntimeError as exc:
                        Engine.write_message_dask(
                            f"[WARNING] {base_name}: Existing slope is unreadable; "
                            f"regenerating it before dependent products. {exc}",
                            OUTPUTS,
                        )
                        try:
                            if os.path.exists(local_validated_slope):
                                os.remove(local_validated_slope)
                        except OSError:
                            pass

                        slope_lazy = _calculate_slope_dask(d_bathy, cell_size)
                        profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                        _save_dask_to_raster(
                            slope_lazy,
                            out_slope,
                            profile,
                            local_tmp_dir,
                            log_prefix=progress_str,
                            product_suffix="_slope.tif",
                        )
                        del slope_lazy
                        gc.collect()
                        _materialize_raster(out_slope, local_validated_slope)
                
                if missing_numpy_dict["_gradmag.tif"] and not is_bluetopo:
                    # Leverage block iteration against the newly generated or existing slope TIF to save recalculating slope and save RAM
                    gradmag_mmap = np.memmap(os.path.join(tmpdir, "g.dat"), dtype='float32', mode='w+', shape=shape_2d)
                    with rasterio.open(local_validated_slope) as src_s:
                        for ji, window in src_s.block_windows(1):
                            s_chunk = src_s.read(1, window=window).astype(np.float32)
                            if src_s.nodata is not None and not np.isnan(src_s.nodata):
                                s_chunk[s_chunk == src_s.nodata] = np.nan
                            gradmag_mmap[window.toslices()] = np.radians(s_chunk)
                            
                    profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                    _save_memmap_to_raster(gradmag_mmap, out_gradmag, profile, local_tmp_dir, log_prefix=progress_str, product_suffix="_gradmag.tif")
                    del gradmag_mmap; gc.collect()

                if missing_numpy_dict["_bpi_fine.tif"]:
                    bpi_fine_lazy = _calculate_bpi_dask(d_bathy, cell_size, best_radii['fine'][0], best_radii['fine'][1])
                    profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                    _save_dask_to_raster(bpi_fine_lazy, out_fine, profile, local_tmp_dir, log_prefix=progress_str, product_suffix="_bpi_fine.tif")
                    del bpi_fine_lazy; gc.collect()

                if missing_numpy_dict["_bpi_broad.tif"]:
                    bpi_broad_lazy = _calculate_bpi_dask(d_bathy, cell_size, best_radii['broad'][0], best_radii['broad'][1])
                    profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                    _save_dask_to_raster(bpi_broad_lazy, out_broad, profile, local_tmp_dir, log_prefix=progress_str, product_suffix="_bpi_broad.tif")
                    del bpi_broad_lazy; gc.collect()

                # -------------------------------------------------------------
                # Block-wise Classification Processing
                # -------------------------------------------------------------
                if missing_numpy_dict["_terrain_classification.tif"]:
                    slope_src_path = local_validated_slope or out_slope
                    
                    if is_bluetopo:
                        ext = UPath(bathy_path).suffix
                        match = re.search(r'(BlueTopo_[A-Za-z0-9_]+_\d{8})', base_name, re.IGNORECASE)
                        core_name = match.group(1) if match else base_name
                        
                        bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{core_name}_slope{ext}")
                        if not UPath(bluetopo_slope_path).exists():
                            bluetopo_slope_path = str(UPath(prediction_output_dir) / f"{base_name}_slope{ext}")
                            
                        if UPath(bluetopo_slope_path).exists():
                            slope_src_path = bluetopo_slope_path
                        else:
                            raise FileNotFoundError(f"Missing external BlueTopo slope for classification: {bluetopo_slope_path}")

                    classified_array = np.memmap(os.path.join(tmpdir, "c.dat"), dtype='float32', mode='w+', shape=shape_2d)
                    classified_array[:] = np.nan 

                    # Use fresh local copies instead of reopening newly
                    # overwritten S3 keys through GDAL's /vsis3 range cache.
                    if not is_bluetopo and local_validated_slope:
                        local_class_slope = local_validated_slope
                    else:
                        local_class_slope = _materialize_raster(
                            slope_src_path,
                            os.path.join(tmpdir, "class_slope.tif"),
                        )

                    def materialize_or_rebuild_bpi(
                        output_path: str,
                        local_name: str,
                        radii_key: str,
                        product_suffix: str,
                    ) -> str:
                        local_path = os.path.join(tmpdir, local_name)
                        try:
                            return _materialize_raster(output_path, local_path)
                        except RuntimeError as exc:
                            Engine.write_message_dask(
                                f"[WARNING] {base_name}: Existing {product_suffix} is "
                                f"unreadable; regenerating only that BPI. {exc}",
                                OUTPUTS,
                            )
                            try:
                                if os.path.exists(local_path):
                                    os.remove(local_path)
                            except OSError:
                                pass

                            bpi_lazy = _calculate_bpi_dask(
                                d_bathy,
                                cell_size,
                                best_radii[radii_key][0],
                                best_radii[radii_key][1],
                            )
                            profile.update(
                                dtype='float32',
                                nodata=np.nan,
                                count=1,
                                compress='LZW',
                                tiled=True,
                                blockxsize=256,
                                blockysize=256,
                            )
                            _save_dask_to_raster(
                                bpi_lazy,
                                output_path,
                                profile,
                                local_tmp_dir,
                                log_prefix=progress_str,
                                product_suffix=product_suffix,
                            )
                            del bpi_lazy
                            gc.collect()
                            return _materialize_raster(output_path, local_path)

                    local_class_broad = materialize_or_rebuild_bpi(
                        out_broad,
                        "class_broad.tif",
                        "broad",
                        "_bpi_broad.tif",
                    )
                    local_class_fine = materialize_or_rebuild_bpi(
                        out_fine,
                        "class_fine.tif",
                        "fine",
                        "_bpi_fine.tif",
                    )

                    with rasterio.open(local_class_slope) as src_s, rasterio.open(local_class_broad) as src_b, rasterio.open(local_class_fine) as src_f:
                        for label, dataset in (
                            ("classification slope", src_s),
                            ("broad BPI", src_b),
                            ("fine BPI", src_f),
                        ):
                            if dataset.shape != shape_2d:
                                raise ValueError(
                                    f"{label} shape {dataset.shape} does not match {shape_2d}."
                                )
                            if dataset.crs != source_crs:
                                raise ValueError(
                                    f"{label} CRS {dataset.crs} does not match {source_crs}."
                                )
                            if not dataset.transform.almost_equals(source_transform):
                                raise ValueError(f"{label} transform does not match bathymetry.")

                        for ji, window in src_s.block_windows(1):
                            s_c = src_s.read(1, window=window).astype(np.float32)
                            b_c = src_b.read(1, window=window).astype(np.float32)
                            f_c = src_f.read(1, window=window).astype(np.float32)
                            
                            if src_s.nodata is not None and not np.isnan(src_s.nodata):
                                s_c[s_c == src_s.nodata] = np.nan
                            if src_b.nodata is not None and not np.isnan(src_b.nodata):
                                b_c[b_c == src_b.nodata] = np.nan
                            if src_f.nodata is not None and not np.isnan(src_f.nodata):
                                f_c[f_c == src_f.nodata] = np.nan
                                
                            c_c = np.full_like(s_c, np.nan)
                            valid_mask = ~np.isnan(s_c) & ~np.isnan(b_c) & ~np.isnan(f_c)
                            
                            for _, rule in unique_dictionary.iterrows():
                                matches = ((b_c >= rule['BroadBPI_Lower']) & (b_c <= rule['BroadBPI_Upper']) &
                                           (f_c >= rule['FineBPI_Lower']) & (f_c <= rule['FineBPI_Upper']) &
                                           (s_c >= rule['Slope_Lower']) & (s_c <= rule['Slope_Upper']))
                                c_c[valid_mask & matches & np.isnan(c_c)] = rule['Class_ID']
                                
                            classified_array[window.toslices()] = c_c

                    profile.update(dtype='float32', nodata=np.nan, count=1, compress='LZW', tiled=True, blockxsize=256, blockysize=256)
                    _save_memmap_to_raster(classified_array, out_class, profile, local_tmp_dir, log_prefix=progress_str, product_suffix="_terrain_classification.tif")
                    del classified_array
                del d_bathy
            gc.collect()

            required_outputs = [
                (out_slope_deg, "_slope_deg.tif"),
                (out_flowdir, "_flowdir.tif"),
                (out_prof, "_curv_profile.tif"),
                (out_plan, "_curv_plan.tif"),
                (out_total, "_curv_total.tif"),
                (out_flowacc, "_flowacc.tif"),
                (out_shear, "_shearproxy.tif"),
                (out_tci, "_tci.tif"),
                (out_rug, "_rugosity.tif"),
                (out_fine, "_bpi_fine.tif"),
                (out_broad, "_bpi_broad.tif"),
                (out_class, "_terrain_classification.tif"),
                (out_gradmag, "_gradmag.tif"),
            ]
            if not is_bluetopo:
                required_outputs.append((out_slope, "_slope.tif"))

            for output_path, product_suffix in required_outputs:
                valid, reason = _validate_product_raster(
                    output_path,
                    product_suffix,
                    expected_shape=source_shape,
                    expected_crs=source_crs,
                    expected_transform=source_transform,
                )
                if not valid:
                    product_errors.append(
                        f"Invalid required output {UPath(output_path).name}: {reason}"
                    )

            if product_errors:
                unique_errors = list(dict.fromkeys(product_errors))
                return (
                    False,
                    f"Failed: {base_name} - " + "; ".join(unique_errors),
                )

            Engine.write_message_dask(f" - [SUCCESS] {progress_str}Completed terrain processing: {base_name}", OUTPUTS)
            metrics = _get_worker_metrics(str(local_tmp_dir))
            if metrics: Engine.write_message_dask(metrics, OUTPUTS)
            return (True, f"Success: {base_name}")
        
        finally:
            def _close_mmap(arr):
                if arr is not None:
                    try:
                        if hasattr(arr, '_mmap'): arr._mmap.close()
                        if hasattr(arr, 'base') and hasattr(arr.base, 'close'): arr.base.close()
                    except Exception: pass
            
            locs = locals()
            for key in ['bathy_array', 'classified_array', 'gradmag_mmap']:
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
            else:
                # STRICT FILTER: Ensure files include "combined" if they are not BlueTopo.
                # This prevents processing weather/hurricane variables mixed into the same directories.
                if 'combined' not in fname:
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
        
        if years:
            # Scatter massive data lists to workers before mapping to prevent >100MB Dask graph transmission bottlenecks
            files_futures = self.client.scatter(files_lists)
            
            dict_futures = self.client.map(
                _create_regional_dictionary_worker, 
                years, 
                files_futures, 
                dictionary_dir=str(self.dictionary_dir), 
                best_radii=best_radii, 
                local_tmp_dir=str(self.local_tmp_dir), 
                prediction_output_dir=str(self.prediction_output_dir)
            )
            dict_results = self.client.gather(dict_futures)
            for success, msg in dict_results:
                self.write_message(msg, OUTPUTS)
        
        self.write_message(self.log_system_metrics(), OUTPUTS)

    def _execute_terrain_generation(self, valid_files: List[str], best_radii: Dict[str, Tuple[int, int]]) -> None:
        """Step 2: Generate terrain products iteratively."""
        self.write_message(f"--- PHASE 2: Parallel Terrain Product Generation ---", OUTPUTS)
        
        # --- Pre-scan to filter out already processed tiles ---
        self.write_message("Scanning tiles to calculate remaining work...", OUTPUTS)
        
        # One lightweight task per tile provides correct worker-level load balancing.
        check_futures = self.client.map(
            _check_unprocessed_worker,
            valid_files,
            terrain_outputs_dir=str(self.terrain_outputs_dir),
            prediction_output_dir=str(self.prediction_output_dir)
        )
        check_results = self.client.gather(check_futures)
        
        paths = [f for f, needs_work in zip(valid_files, check_results) if needs_work]
        skipped_count = len(valid_files) - len(paths)
        
        if skipped_count > 0:
            self.write_message(f"Skipped {skipped_count} tiles (products already exist).", OUTPUTS)
            
        if not paths:
            self.write_message("[SUCCESS] All terrain products are up to date. No remaining tiles.", OUTPUTS)
            return
            
        self.write_message(f"Sorting {len(paths)} remaining tiles by size (smallest to largest)...", OUTPUTS)
        
        def _get_size(path_str):
            try:
                return UPath(path_str).stat().st_size
            except Exception:
                return float('inf') # Move unreadable sizes to end of queue
                
        paths.sort(key=_get_size)
        
        self.write_message(f"Processing {len(paths)} remaining tiles in optimized order...", OUTPUTS)
        
        indices = list(range(1, len(paths) + 1))

        # Each future owns one tile. Large BPI arrays remain inside that worker
        # and are never embedded in the distributed scheduler graph.
        terrain_futures = self.client.map(
            _process_terrain_raster_worker,
            paths,
            indices,
            total_count=len(paths),
            best_radii=best_radii,
            terrain_outputs_dir=str(self.terrain_outputs_dir),
            prediction_output_dir=str(self.prediction_output_dir),
            dictionary_dir=str(self.dictionary_dir),
            local_tmp_dir=str(self.local_tmp_dir)
        )
        
        terrain_results = self.client.gather(terrain_futures)
        
        failed_results = []

        for success, msg in terrain_results:
            if success:
                self.write_message(f"[SUCCESS] {msg}", OUTPUTS)
            else:
                failed_results.append(msg)
                self.write_message(f"[ERROR] {msg}", OUTPUTS)

        self.write_message(self.log_system_metrics(), OUTPUTS)

        if failed_results:
            self.write_message(
                f"[ERROR] Terrain raster processing completed with "
                f"{len(failed_results)} failed tile(s) out of {len(terrain_results)}.",
                OUTPUTS
            )
        else:
            self.write_message(
                f"[SUCCESS] Terrain raster processing complete. "
                f"{len(terrain_results)} tile(s) processed successfully.",
                OUTPUTS
            )

    def _generate_pdf_report(self, valid_files: List[str]) -> None:
        """Step 3: Generate visual plots showing corresponding terrain maps."""
        self.write_message("--- PHASE 3: Generating PDF Reports ---", OUTPUTS)
        
        lidar_files = [f for f in valid_files if 'bluetopo' not in os.path.basename(f).lower()]
        bt_files = [f for f in valid_files if 'bluetopo' in os.path.basename(f).lower()]
        
        matched_lidar = None
        matched_bt = None
        
        # Try to find a matching tile
        for bt in bt_files:
            bt_name = os.path.basename(bt)
            match = re.search(r'BlueTopo_([A-Za-z0-9_]+)_\d{8}', bt_name, re.IGNORECASE)
            if match:
                tile_id = match.group(1)
                
                # Find ALL matching LiDAR files for this tile
                matching_lidars = []
                for lf in lidar_files:
                    if tile_id in os.path.basename(lf):
                        matching_lidars.append(lf)
                
                if matching_lidars:
                    matched_bt = bt
                    # Prioritize the FIRST match to select a different LiDAR year for the report
                    matching_lidars.sort()
                    matched_lidar = matching_lidars[0]
                    break
                
        if not matched_lidar and lidar_files: matched_lidar = lidar_files[0] # Try first if no match
        if not matched_bt and bt_files: matched_bt = bt_files[0]
        
        files_to_plot = []
        if matched_lidar: files_to_plot.append(("LiDAR", matched_lidar))
        if matched_bt: files_to_plot.append(("BlueTopo", matched_bt))
        
        if not files_to_plot:
            self.write_message("No files found to plot.", OUTPUTS)
            return

        pdf_path = self.outputs_dir / f"Terrain_Products_Report_{self.param_lookup.get('eco_regions').value[0]}.pdf"
        
        try:
            with PdfPages(str(pdf_path)) as pdf:
                # Sort files to ensure LiDAR plots first. This allows us to capture its 'correct' colorbar bounds.
                files_to_plot.sort(key=lambda x: 0 if x[0] == "LiDAR" else 1)
                
                # Store colorbar limits (vmin, vmax) from LiDAR to apply to BlueTopo
                synced_colorbar_bounds = {}
                
                for src_type, file_path in files_to_plot:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    
                    # Dynamically build products to plot to enforce exactly 2 slope layers per source
                    products_to_plot = [
                        ("_slope.tif" if src_type == "BlueTopo" else "_slope_deg.tif", "Slope [Degrees]"),
                        ("_gradmag.tif", "Slope/Gradient [Radians]"),
                        ("_rugosity.tif", "Surface Rugosity Ratio [Unitless]"),
                        ("_tci.tif", "Terrain Complexity (TCI) [Meters]"),
                        ("_bpi_fine.tif", "BPI Fine [Meters]"),
                        ("_bpi_broad.tif", "BPI Broad [Meters]"),
                        ("_terrain_classification.tif", "Terrain Classification [Class]"),
                        ("_curv_profile.tif", "Profile Curvature [1/Meters]"),
                        ("_curv_plan.tif", "Plan Curvature [1/Meters]"),
                        ("_curv_total.tif", "Total Curvature [1/Meters]"),
                        ("_flowdir.tif", "D8 Flow Pointer [Whitebox Encoding]"),
                        ("_flowacc.tif", "Flow Accumulation [Cells]"),
                        ("_shearproxy.tif", "Shear Stress Proxy [Unitless]")
                    ]
                    
                    # Expanding grid to 4x4 to fit 13 layers
                    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
                    fig.suptitle(f"Terrain Output Profile - {src_type} (13 Products)\nSource: {base_name}", fontsize=18, fontweight='bold', y=0.98)
                    axes = axes.flatten()
                    
                    # Turn off axes for unused subplots
                    for i in range(len(products_to_plot), len(axes)):
                        axes[i].axis('off')
                    
                    for idx, (suffix, title) in enumerate(products_to_plot):
                        ax = axes[idx]
                        layer_path = UPath(self.terrain_outputs_dir) / (base_name + suffix)
                        
                        # Correctly route BlueTopo slope to PREDICTION_OUTPUT_DIR
                        if src_type == "BlueTopo":
                            ext = UPath(file_path).suffix
                            match = re.search(r'(BlueTopo_[A-Za-z0-9_]+_\d{8})', base_name, re.IGNORECASE)
                            core_name = match.group(1) if match else base_name
                            
                            if suffix == "_slope.tif":
                                layer_path = UPath(self.prediction_output_dir) / f"{core_name}_slope{ext}"
                                if not layer_path.exists():
                                    layer_path = UPath(self.prediction_output_dir) / f"{base_name}_slope{ext}"

                        if layer_path.exists():
                            local_tmp = None
                            try:
                                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False, dir=str(self.local_tmp_dir)) as tmp_file:
                                    local_tmp = tmp_file.name
                                with layer_path.open('rb') as f_in, open(local_tmp, 'wb') as f_out:
                                    shutil.copyfileobj(f_in, f_out)
                                
                                with rasterio.open(local_tmp) as src:
                                    decimation = max(1, src.width // 1000)
                                    arr = src.read(1, out_shape=(1, int(src.height // decimation), int(src.width // decimation)))
                                    
                                    nodata = src.nodata
                                    if nodata is not None:
                                        arr = np.where(arr == nodata, np.nan, arr)
                                    
                                    # Robust fallback to catch un-flagged NoData (-9999), Inf, or extreme anomalies in BlueTopo
                                    arr = np.where((arr < -9998) | (arr > 1e30) | np.isinf(arr), np.nan, arr)
                                    
                                    valid_mask = ~np.isnan(arr)
                                    if np.any(valid_mask):
                                        if title.startswith('Terrain Classification'):
                                            # Discrete integer scaling for 8 classification zones
                                            cmap = plt.get_cmap('tab10', 8)
                                            norm = mcolors.BoundaryNorm(np.arange(0.5, 9.5, 1), cmap.N)
                                            im = ax.imshow(arr, cmap=cmap, norm=norm, interpolation='nearest')
                                            
                                            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=np.arange(1, 9))
                                        else:
                                            # Retrieve synced limits if we are on BlueTopo and already calculated LiDAR limits
                                            if src_type == "BlueTopo" and suffix in synced_colorbar_bounds:
                                                vmin, vmax = synced_colorbar_bounds[suffix]
                                            else:
                                                # Calculate robust min/max to filter out 1st/99th percentile outliers
                                                vmin, vmax = np.percentile(arr[valid_mask], [1, 99])
                                                # Fallback if the data is entirely uniform
                                                if vmin == vmax or np.isnan(vmin):
                                                    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
                                                
                                                # Save LiDAR limits for the next dataset
                                                if src_type == "LiDAR":
                                                    synced_colorbar_bounds[suffix] = (vmin, vmax)
                                                
                                            im = ax.imshow(arr, cmap='viridis', vmin=vmin, vmax=vmax)
                                            
                                            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                                    else:
                                        ax.text(0.5, 0.5, 'All NoData', ha='center', va='center')
                                        
                                ax.set_title(title)
                                ax.axis('off')
                            except Exception as e:
                                ax.text(0.5, 0.5, f"Error reading data:\n{e}", ha='center', va='center', wrap=True)
                                ax.set_title(title)
                                ax.axis('off')
                            finally:
                                if local_tmp and os.path.exists(local_tmp):
                                    try:
                                        os.remove(local_tmp)
                                    except OSError:
                                        pass
                        else:
                            ax.text(0.5, 0.5, 'Layer Missing or Not Generated', 
                                    horizontalalignment='center', verticalalignment='center', 
                                    transform=ax.transAxes, color='red', fontsize=10)
                            ax.set_title(title)
                            ax.axis('off')
                            
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    pdf.savefig(fig)
                    plt.close(fig)
                    
            self.write_message(f"Successfully generated PDF report at: {pdf_path}", OUTPUTS)
        except Exception as e:
            self.write_message(f"Failed to generate PDF report: {e}", OUTPUTS)

    def run(self) -> None:
        """Main entry point for evaluating directories and processing rasters in parallel."""
        env = self.param_lookup.get('env', 'local')
        
        try:
            self.setup_dask(env, n_workers=2, threads_per_worker=1, memory_limit="13GB") 

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
                
                # Step 3: Output PDF Visual Plot
                self._generate_pdf_report(valid_files)
                
        finally:
            self.cleanup_resources(OUTPUTS)  
