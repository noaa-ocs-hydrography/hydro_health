"""Class engine for iteratively filling gaps in bathymetric/lidar raster data using Dask and Xarray."""

import os
import shutil
import tempfile
import pathlib
import gc
from pathlib import Path
import re

import numpy as np
import xarray as xr
import rioxarray
import dask
from scipy.ndimage import uniform_filter, binary_fill_holes
from upath import UPath
import s3fs

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

# Global Base Paths
INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'

def _dask_worker_logger(msg: str, dest: str = "") -> None:
    """
    Module-level logger for Dask workers.
    Passed as 'log_func' to avoid pickling the Engine class instance.
    Dask automatically captures standard print statements into cluster logs.
    """
    print(msg)


def focal_fill_block(block: np.ndarray, w=3) -> np.ndarray:
    """
    Performs a single, efficient, nan-aware focal mean on a NumPy array block.
    Designed to be mapped across Dask chunks safely inside workers.
    """
    block = block.astype(np.float32)
    nan_mask = np.isnan(block)

    # Sum valid data within the window
    data_sum = uniform_filter(np.nan_to_num(block, nan=0.0), size=w, mode="constant", cval=0.0)
    
    # Count number of valid pixels within the window
    valid_count = uniform_filter((~nan_mask).astype(np.float32), size=w, mode="constant", cval=0.0)

    with np.errstate(invalid='ignore', divide='ignore'):
        filled = data_sum / valid_count

    return np.where(nan_mask, filled, block)


def _chunked_fill_holes(block: np.ndarray) -> np.ndarray:
    """
    Performs binary hole filling per-chunk to avoid OOM errors on massive rasters.
    """
    return binary_fill_holes(block.astype(bool))

def _process_gap_fill(params: list) -> tuple:
    """Worker function: Performs iterative focal fill on a single raster using Dask and rioxarray."""
    input_path, output_path, max_iters, chunk_size, local_tmp_dir, is_aws, log_func = params
    raster_name = Path(input_path).name
    expected_path = UPath(output_path)
    
    # Implicit skip pattern - verify if output already exists before doing any work
    if expected_path.exists():
        log_func(f"[SKIP] Gap fill already processed for {raster_name}. Output exists.", OUTPUTS)
        return True, str(output_path)
        
    log_func(f"Worker attempting gap fill for: {raster_name}", OUTPUTS)

    with tempfile.TemporaryDirectory(dir=local_tmp_dir) as task_tmp_dir:
        try:
            da_chunk = {"x": chunk_size, "y": chunk_size}
            
            with rioxarray.open_rasterio(input_path, chunks=da_chunk) as ds:
                nodata = ds.rio.nodata
                da = ds.isel(band=0).astype("float32")
                
                # Mask 1: Land/Surface Mask
                land_mask = (da > 0)
                if nodata is not None:
                    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
                        land_mask = land_mask & da.notnull()
                        da = da.where(da.notnull() & (da != 0.0))
                    else:
                        land_mask = land_mask & (da != nodata)
                        da = da.where((da != nodata) & (da != 0.0))
                else:
                    da = da.where(da != 0.0)
                    
                # Skip if empty
                if not da.notnull().any().compute().item():
                    detailed_msg = f"Skipped {raster_name} - no valid data remaining."
                    log_func(f"[SKIP] {detailed_msg}", OUTPUTS)
                    return False, detailed_msg

                # Map hole filling across Dask chunks
                valid_mask_da = da.notnull()
                with dask.config.set(scheduler='single-threaded'):
                    allowed_footprint_da = valid_mask_da.data.map_overlap(
                        _chunked_fill_holes,
                        depth={0: 10, 1: 10},
                        boundary="reflect",
                        dtype=bool
                    )
                
                allowed_da = xr.DataArray(allowed_footprint_da, coords=da.coords, dims=da.dims)
                nan_mask_da = ~valid_mask_da
                interior_gaps_exist = (nan_mask_da & allowed_da).any().compute(scheduler='single-threaded').item()
                
                if not interior_gaps_exist:
                    log_func(f"[INFO] No interior gaps in {raster_name}. Applying simple masks.", OUTPUTS)
                    da = da.where(~land_mask)
                else:
                    for _ in range(max_iters):
                        da_prev = da
                        window_size = 5
                        margin = window_size // 2 
                        
                        with dask.config.set(scheduler='single-threaded'):
                            filled_data = da.data.map_overlap(
                                focal_fill_block,
                                depth={0: margin, 1: margin},
                                boundary="reflect",
                                dtype=da.dtype,
                                w=window_size
                            )
                            da_filled = xr.DataArray(filled_data, coords=da.coords, dims=da.dims)
                            
                        da = xr.where(np.isnan(da_prev), da_filled, da_prev)

                    da = da.where(allowed_da)
                    da = da.where(~land_mask)

                # Ensure < 0 logic remains intact
                da = da.where(da < 0, np.nan)
                da = da.expand_dims(dim="band")
                da.rio.write_crs(ds.rio.crs, inplace=True)
                da.rio.write_transform(ds.rio.transform(), inplace=True)
                da.rio.write_nodata(np.nan, inplace=True)
                
                tmp_dst_path = str(output_path)
                if is_aws or expected_path.protocol == "s3":
                    tmp_dst_path = str(Path(task_tmp_dir) / "filled_tmp.tif")

                with dask.config.set(scheduler='single-threaded'):
                    da.rio.to_raster(tmp_dst_path, lock=False, compress='LZW')

                # Push to target destination
                if is_aws or expected_path.protocol == "s3":
                    fs = s3fs.S3FileSystem()
                    fs.put(tmp_dst_path, str(output_path))
                else:
                    if tmp_dst_path != str(output_path):
                        shutil.copyfile(tmp_dst_path, str(output_path))

                # Explicitly delete temp file immediately to free EC2 disk space
                if tmp_dst_path != str(output_path) and Path(tmp_dst_path).exists():
                    try:
                        os.remove(tmp_dst_path)
                    except Exception as e:
                        log_func(f"Failed to explicitly delete temp file {tmp_dst_path}: {e}", OUTPUTS)

                return True, f"Gap fill complete for: {raster_name}"

        except Exception as e:
            return False, f"Failed gap fill for {raster_name}: {e}"
        finally:
            gc.collect()

def _process_combine_group(params: list) -> tuple:
    """Worker function: Combines and averages multiple overlapping Lidar datasets."""
    group_files, output_path, chunk_size, local_tmp_dir, is_aws, log_func = params
    expected_path = UPath(output_path)
    
    # Implicit skip pattern
    if expected_path.exists():
        log_func(f"[SKIP] Combined Lidar already processed. Output exists: {output_path}", OUTPUTS)
        return True, f"Already exists: {output_path}"
        
    file_count = len(group_files)
    log_func(f"Starting combine task for {file_count} files -> {output_path}", OUTPUTS)
    
    try:
        if file_count == 1:
            single_file = group_files[0]
            if is_aws or expected_path.protocol == "s3":
                fs = s3fs.S3FileSystem()
                fs.copy(single_file, str(output_path))
            else:
                shutil.copyfile(single_file, str(output_path))
            return True, f"Copied single dataset to {output_path}"
            
        with tempfile.TemporaryDirectory(dir=local_tmp_dir) as task_tmp_dir:
            das = []
            for p in group_files:
                da = rioxarray.open_rasterio(p, chunks={"x": chunk_size, "y": chunk_size})
                if da.rio.nodata is not None:
                    da = da.where(da != da.rio.nodata)
                da = da.where(da != 0.0)
                da = da.assign_coords(x=np.round(da.x, decimals=4), y=np.round(da.y, decimals=4))
                das.append(da)
                
            all_xs = np.concatenate([da.x.values for da in das])
            all_ys = np.concatenate([da.y.values for da in das])
            min_x, max_x = all_xs.min(), all_xs.max()
            min_y, max_y = all_ys.min(), all_ys.max()
            
            res_x = abs(das[0].x.values[1] - das[0].x.values[0])
            res_y = abs(das[0].y.values[1] - das[0].y.values[0])
            req_width = int((max_x - min_x) / res_x)
            req_height = int((max_y - min_y) / res_y)
            total_pixels = req_width * req_height
            
            if total_pixels > 2_500_000_000:
                return False, f"Grid is too large: {req_width}x{req_height} ({total_pixels} pixels)."
                
            aligned_das = xr.align(*das, join="outer")
            stacked = xr.concat(aligned_das, dim="dataset")
            combined = stacked.mean(dim="dataset", skipna=True)
            combined = combined.where(combined < 0, np.nan)
            
            base_da = das[0]
            combined.rio.write_crs(base_da.rio.crs, inplace=True)
            combined.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
            combined.rio.write_nodata(np.nan, inplace=True)
            
            if "band" not in combined.dims:
                combined = combined.expand_dims(dim="band")
                
            tmp_dst_path = str(output_path)
            if is_aws or expected_path.protocol == "s3":
                tmp_dst_path = str(Path(task_tmp_dir) / "combined_tmp.tif")
                
            with dask.config.set(scheduler='single-threaded'):
                combined.rio.to_raster(tmp_dst_path, lock=False, compress='LZW')
                
            if is_aws or expected_path.protocol == "s3":
                fs = s3fs.S3FileSystem()
                fs.put(tmp_dst_path, str(output_path))
            else:
                if tmp_dst_path != str(output_path):
                    shutil.copyfile(tmp_dst_path, str(output_path))
                    
            if tmp_dst_path != str(output_path) and Path(tmp_dst_path).exists():
                try:
                    os.remove(tmp_dst_path)
                except Exception:
                    pass
                    
            return True, f"Combined {file_count} datasets and saved to {output_path}"
            
    except Exception as e:
        return False, f"Failed combining datasets: {e}"
    finally:
        gc.collect()

class LidarGapFillEngine(Engine):
    """
    Engine dedicated solely to iteratively filling gaps (NoData holes) 
    in bathymetric/lidar raster data using distributed Dask architecture.
    """

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize configurations and environment for gap filling."""
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        
        # Setup local temp dir mapping to ensure EC2 limits aren't exceeded
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "gap_fill_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']
        
        self.inputs_dir = INPUTS

    def log_system_metrics(self) -> str:
        """Helper to collect and format EC2 system metrics (RAM, Disk Space, Temp Size)."""
        try:
            total, used, free = shutil.disk_usage(self.local_tmp_dir)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            tmp_size_bytes = sum(f.stat().st_size for f in self.local_tmp_dir.rglob('*') if f.is_file()) if self.local_tmp_dir.exists() else 0
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
                    mem_avail = re.search(r'MemAvailable:\s+(\d+)\s+kB', meminfo)
                    mem_total = re.search(r'MemTotal:\s+(\d+)\s+kB', meminfo)
                    if mem_avail and mem_total:
                        avail_gb = int(mem_avail.group(1)) / (1024**2)
                        tot_gb = int(mem_total.group(1)) / (1024**2)
                        ram_info = f"Free: {avail_gb:.1f}GB / {tot_gb:.1f}GB (Used: {100 - (avail_gb / tot_gb * 100):.1f}%)"
            
            return f"   [SysMetrics] RAM | {ram_info} || Disk Free | {free_gb:.1f}GB / {total_gb:.1f}GB || Tmp Dir Size | {tmp_mb:.1f}MB"
        except Exception as e:
            return f"   [SysMetrics] Error collecting system metrics: {e}"

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f"LidarGapFillEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        # Setup standard data directories
        tiled_lidar_dir = get_config_item('MODEL', 'TILED_LIDAR_PROC')
        self.tiled_lidar_dir = UPath(f"{s3_dir_base}/{tiled_lidar_dir}") if self.is_aws else UPath(self.outputs_dir / tiled_lidar_dir)

        filled_out_dir = get_config_item('TERRAIN', 'FILLED_DIR')
        self.filled_out_dir = UPath(f"{s3_dir_base}/{filled_out_dir}") if self.is_aws else UPath(self.outputs_dir / filled_out_dir)

        combined_lidar_dir = get_config_item('TERRAIN', 'COMBINED_LIDAR_DIR')
        self.combined_lidar_dir = UPath(f"{s3_dir_base}/{combined_lidar_dir}") if self.is_aws else UPath(self.outputs_dir / combined_lidar_dir)

        if not self.is_aws:
            self.tiled_lidar_dir.mkdir(parents=True, exist_ok=True)
            self.filled_out_dir.mkdir(parents=True, exist_ok=True)
            self.combined_lidar_dir.mkdir(parents=True, exist_ok=True)

    def _execute_gap_fill(self, max_iters: int, chunk_size: int) -> None:
        """Glob inputs and distribute gap fill tasks via Dask."""
        self.write_message(f"Globbing inputs from directory: {self.tiled_lidar_dir}", OUTPUTS)
        
        files_to_process = []
        for ext in ["*.tif", "*.tiff"]:
            files_to_process.extend(list(self.tiled_lidar_dir.rglob(ext)))

        if not files_to_process:
            self.write_message("No rasters found to gap fill.", OUTPUTS)
            return

        self.write_message(f"Queuing {len(files_to_process)} gap fill tasks...", OUTPUTS)
        
        params_list = []
        for f in files_to_process:
            out_name = f.stem + "_filled.tif"
            output_path = self.filled_out_dir / out_name
            params_list.append([
                str(f),
                str(output_path),
                max_iters,
                chunk_size,
                str(self.local_tmp_dir),
                self.is_aws,
                _dask_worker_logger
            ])

        # Execute map cleanly outside the object logic
        futures = self.client.map(_process_gap_fill, params_list)
        results = self.client.gather(futures)

        for success, msg in results:
            if success:
                self.write_message(f"[SUCCESS] {msg}", OUTPUTS)
            else:
                self.write_message(f"[ERROR/SKIP] {msg}", OUTPUTS)
                
        self.write_message(self.log_system_metrics(), OUTPUTS)

    def _execute_combine_lidar(self, chunk_size: int) -> None:
        """Glob filled inputs, group them, and distribute merge tasks via Dask."""
        self.write_message("Moving to combine datasets step...", OUTPUTS)
        
        input_paths = []
        for ext in ["*.tif", "*.tiff"]:
            input_paths.extend(list(self.filled_out_dir.rglob(ext)))
            
        if not input_paths:
            self.write_message("[SKIP] No datasets found to combine.", OUTPUTS)
            return

        # Group files by Year AND Tile Name
        files_by_group = {}
        for p in input_paths:
            year_match = re.search(r'(19\d{2}|20\d{2})', p.name)
            year = year_match.group(1) if year_match else "unknown_year"
            
            tile_match = re.search(r'(?<![a-zA-Z0-9])(B[A-Z0-9]{4,15})(?![a-zA-Z0-9])', p.name.upper())
            tile = tile_match.group(1) if tile_match else "unknown_tile"
            
            group_key = (year, tile)
            if group_key not in files_by_group:
                files_by_group[group_key] = []
            files_by_group[group_key].append(str(p))

        self.write_message(f"Found {len(input_paths)} datasets across {len(files_by_group)} unique Year+Tile combinations.", OUTPUTS)
        
        params_list = []
        for (year, tile), group_files in files_by_group.items():
            file_count = len(group_files)
            if file_count == 1:
                year_out_filename = f"combined1_{Path(group_files[0]).name}"
            else:
                base_name = "combined_lidar.tif".replace("combined_", "", 1)
                year_out_filename = f"combined{file_count}_{year}_{tile}_{base_name}"
                
            output_path = self.combined_lidar_dir / year_out_filename
            params_list.append([
                group_files,
                str(output_path),
                chunk_size,
                str(self.local_tmp_dir),
                self.is_aws,
                _dask_worker_logger
            ])

        # Execute map cleanly outside the object logic
        futures = self.client.map(_process_combine_group, params_list)
        results = self.client.gather(futures)

        for success, msg in results:
            if success:
                self.write_message(f"[SUCCESS] {msg}", OUTPUTS)
            else:
                self.write_message(f"[ERROR/SKIP] {msg}", OUTPUTS)
                
        self.write_message(self.log_system_metrics(), OUTPUTS)

    def run(self, max_iters: int = 5, chunk_size: int = 1024) -> None:
        """Main entry point for evaluating directories and processing rasters in parallel."""
        env = self.param_lookup.get('env', 'local')
        
        try:
            self.setup_dask(env, n_workers=4, threads_per_worker=1, memory_limit="6GB")
            
            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)
                
                # Step 1: Execute Iterative Gap Fill
                self._execute_gap_fill(max_iters, chunk_size)
                
                # Step 2: Combine and Average Matching LiDAR Groups
                self._execute_combine_lidar(chunk_size)
                
        finally:
            self.cleanup_resources(OUTPUTS)