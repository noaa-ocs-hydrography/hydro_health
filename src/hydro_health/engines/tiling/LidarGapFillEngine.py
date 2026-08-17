import os
import shutil
import tempfile
import logging
import pathlib
from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
import rioxarray
import dask
from scipy.ndimage import uniform_filter, binary_fill_holes

from dask.distributed import Client, as_completed
from upath import UPath
import s3fs

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)

class LidarGapFillEngine(Engine):
    """
    Engine dedicated solely to iteratively filling gaps (NoData holes) 
    in bathymetric/lidar raster data using Dask and Xarray.
    Refactored to support distributed parallel processing.
    """

    def __init__(self, param_lookup: dict, output_prefix: Union[str, bool] = False) -> None:
        """Initialize paths, configurations, and environment for gap filling"""
        super().__init__()
        self.param_lookup = param_lookup
        
        # Helper to extract raw values from potential ArcGIS/custom Param objects
        def _get_val(key, default=None):
            val = self.param_lookup.get(key, default)
            if hasattr(val, 'valueAsText') and val.valueAsText is not None:
                return val.valueAsText
            if hasattr(val, 'value') and val.value is not None:
                return val.value
            return val
            
        self.local_tmp_dir = pathlib.Path(_get_val('local_tmp_dir', str(Path.home() / "hydro_health_local_tmp")))
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
        
        # Use param_lookup paths if provided, else default to repo root
        base_in_dir = pathlib.Path(in_dir) if in_dir else self.repo_root / 'inputs'
        base_out_dir = pathlib.Path(out_dir) if out_dir else self.repo_root / 'outputs'
        
        # Ensure output_prefix is correctly evaluated
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
        if eco_val:
            self.ecoregion = eco_val[0] if isinstance(eco_val, list) else str(eco_val).strip("[]'\" ")

        if self.ecoregion:
            self.outputs_dir = self.outputs_dir / self.ecoregion

        def _resolve_path(config_value: str, is_output: bool = False) -> UPath:
            """Helper to safely build paths for AWS or Local execution"""
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

        # Map to appropriate filled output directory
        self.filled_out_dir = _resolve_path(get_config_item('TERRAIN', 'FILLED_DIR'), is_output=True)

    def __getstate__(self):
        """
        Exclude unpicklable attributes (like Dask Client/Cluster and raw Params)
        when serializing this instance to send to Dask worker nodes.
        """
        state = self.__dict__.copy()
        state.pop('client', None)
        state.pop('cluster', None)
        state.pop('param_lookup', None)
        return state

    @staticmethod
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

    def process_gap_fill(self, input_path: str, output_path: str, max_iters: int = 5, chunk_size: int = 512, current_index: int = None, total_count: int = None) -> None:
        """Worker function: Performs iterative focal fill on a single raster using Dask and rioxarray."""
        raster_name = pathlib.Path(input_path).name
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        
        logger.info(f"-> [STARTING]{progress_str} Worker attempting gap fill for: {raster_name}")

        with tempfile.TemporaryDirectory(dir=self.local_tmp_dir) as task_tmp_dir:
            try:
                da_chunk = {"x": chunk_size, "y": chunk_size}
                ds = rioxarray.open_rasterio(str(input_path), chunks=da_chunk)
                nodata = ds.rio.nodata
                
                da = ds.isel(band=0).astype("float32")
                
                # --- MASK 1: LAND/SURFACE MASK ---
                land_mask = (da > 0)
                if nodata is not None:
                    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
                        land_mask = land_mask & da.notnull()
                    else:
                        land_mask = land_mask & (da != nodata)
                    
                # Convert official nodata AND exact 0.0 values to NaN so they are recognized as gaps.
                if nodata is not None:
                    if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
                        da = da.where(da.notnull() & (da != 0.0))
                    else:
                        da = da.where((da != nodata) & (da != 0.0))
                else:
                    da = da.where(da != 0.0)
                    
                # Check if there is any valid data left to process
                if not da.notnull().any().compute().item():
                    logger.info(f"- [SKIP]{progress_str} {raster_name} contains no valid data.")
                    return

                valid_mask_mem = da.notnull().compute(scheduler='single-threaded').values
                allowed_footprint = binary_fill_holes(valid_mask_mem)
                allowed_da = xr.DataArray(allowed_footprint, coords=da.coords, dims=da.dims)
                
                nan_mask = ~valid_mask_mem
                interior_gaps_exist = (nan_mask & allowed_footprint).any()
                
                needs_fill = interior_gaps_exist
                
                if not needs_fill:
                    logger.info(f"- [INFO]{progress_str} No interior gaps in {raster_name}. Skipping compute, applying simple masks.")
                    da = da.where(~land_mask)
                    da = da.where(da < 0)
                else:
                    # Iteratively fill gaps if they exist
                    for _ in range(max_iters):
                        da_prev = da
                        with dask.config.set(scheduler='single-threaded'):
                            da = xr.apply_ufunc(
                                self.focal_fill_block,
                                da,
                                kwargs={"w": 5},
                                input_core_dims=[["y", "x"]],
                                output_core_dims=[["y", "x"]],
                                dask="parallelized",
                                dask_gufunc_kwargs={"allow_rechunk": True},
                                output_dtypes=[da.dtype],
                            )
                        da = xr.where(np.isnan(da_prev), da, da_prev)

                    # Restrict filled data strictly to interior footprint
                    da = da.where(allowed_da)
                    # Re-apply land mask
                    da = da.where(~land_mask)
                    da = da.where(da < 0)

                if nodata is not None:
                    da = da.fillna(nodata)
                da = da.expand_dims(dim="band")

                da.rio.write_crs(ds.rio.crs, inplace=True)
                da.rio.write_transform(ds.rio.transform(), inplace=True)
                if nodata is not None:
                    da.rio.write_nodata(nodata, inplace=True)
                
                # Setup local temp export path
                tmp_dst_path = str(output_path)
                if self.is_aws or UPath(output_path).protocol == "s3":
                    tmp_dst_path = str(Path(task_tmp_dir) / "filled_tmp.tif")

                # Stream to local disk chunk-by-chunk using single-threaded dask scheduler
                with dask.config.set(scheduler='single-threaded'):
                    da.rio.to_raster(tmp_dst_path, lock=False, compress='LZW')

                # Push to target destination
                if self.is_aws or UPath(output_path).protocol == "s3":
                    fs = s3fs.S3FileSystem()
                    fs.put(tmp_dst_path, str(output_path))
                else:
                    if tmp_dst_path != str(output_path):
                        shutil.copyfile(tmp_dst_path, str(output_path))

                logger.info(f" - [✓ SUCCESS]{progress_str} Gap fill complete for: {raster_name}")

            except Exception as e:
                logger.exception(f"Unexpected failure during gap fill for {raster_name}: {e}")

    def combine_lidar_datasets(self, input_paths: list, output_path: str, chunk_size: int = 512) -> None:
        """
        Combines and averages multiple overlapping Lidar datasets into a single raster.
        Aligns to a common outer bounding box, calculating the mean of overlapping pixels.
        """
        logger.info(f"-> [STARTING] Combining and averaging {len(input_paths)} datasets into {pathlib.Path(output_path).name}")
        
        with tempfile.TemporaryDirectory(dir=self.local_tmp_dir) as task_tmp_dir:
            try:
                das = []
                for p in input_paths:
                    da = rioxarray.open_rasterio(str(p), chunks={"x": chunk_size, "y": chunk_size})
                    
                    # Convert nodata to NaN for proper nanmean calculations
                    if da.rio.nodata is not None:
                        da = da.where(da != da.rio.nodata)
                    # Also treat exact 0.0 as nodata if required by context
                    da = da.where(da != 0.0)
                    das.append(da)
                    
                if not das:
                    logger.warning("- [SKIP] No datasets provided to combine.")
                    return
                
                # Align datasets to a common outer grid based on their coordinates
                logger.info("Aligning datasets to global grid...")
                aligned_das = xr.align(*das, join="outer")
                
                # Stack and compute the mean across overlapping areas
                logger.info("Calculating mean across overlapping regions...")
                stacked = xr.concat(aligned_das, dim="dataset")
                combined = stacked.mean(dim="dataset", skipna=True)
                
                # Restore spatial attributes from the first input
                base_da = das[0]
                combined.rio.write_crs(base_da.rio.crs, inplace=True)
                combined.rio.write_transform(base_da.rio.transform(), inplace=True)
                
                # Use standard nodata value (e.g. from the base layer)
                nodata_val = base_da.rio.nodata if base_da.rio.nodata is not None else -9999.0
                combined = combined.fillna(nodata_val)
                combined.rio.write_nodata(nodata_val, inplace=True)
                
                # Ensure we retain the band dimension for raster export
                if "band" not in combined.dims:
                    combined = combined.expand_dims(dim="band")
                
                tmp_dst_path = str(output_path)
                if self.is_aws or UPath(output_path).protocol == "s3":
                    tmp_dst_path = str(Path(task_tmp_dir) / "combined_tmp.tif")

                logger.info("Writing combined raster to disk...")
                # Stream chunk-by-chunk to avoid OOM
                with dask.config.set(scheduler='single-threaded'):
                    combined.rio.to_raster(tmp_dst_path, lock=False, compress='LZW')

                # Push to target destination
                if self.is_aws or UPath(output_path).protocol == "s3":
                    fs = s3fs.S3FileSystem()
                    fs.put(tmp_dst_path, str(output_path))
                else:
                    if tmp_dst_path != str(output_path):
                        shutil.copyfile(tmp_dst_path, str(output_path))

                logger.info(f" - [✓ SUCCESS] Combined Lidar datasets saved to: {pathlib.Path(output_path).name}")

            except Exception as e:
                logger.exception(f"Unexpected failure during lidar combination: {e}")

    def run(self, max_concurrent: int = 5, max_iters: int = 5, chunk_size: int = 512) -> None:
        """Main entry point for evaluating directories and processing rasters in parallel."""
        env_val = self.param_lookup.get('env', 'local')
        env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        
        # Initialize Dask via Base Engine
        self.setup_dask(env, n_workers=3, threads_per_worker=1, memory_limit="9.5GB")
        
        client = getattr(self, 'client', None)
        if not client:
            client = Client()

        # Glob input files (e.g., from combined lidar outputs)
        potential_inputs = []
        for ext in ["*.tif", "*.tiff"]:
            potential_inputs.extend(list(self.inputs_dir.rglob(ext)))

        # Identify existing outputs to optionally skip
        existing_outputs = set()
        for ext in ["*.tif", "*.tiff"]:
            existing_outputs.update({f.name for f in self.filled_out_dir.rglob(ext)})

        files_to_process = []
        removed_existing = 0

        for f in potential_inputs:
            out_name = os.path.splitext(f.name)[0] + "_filled.tif"
            if not self.overwrite and out_name in existing_outputs:
                removed_existing += 1
                continue
            files_to_process.append(f)

        skip_msg = f" (Skipping {removed_existing} existing)" if not self.overwrite else " (Overwrite enabled)"
        
        logger.info(f"Outputting filled rasters to: {self.filled_out_dir}")
        logger.info(f"Queuing {len(files_to_process)} gap fill files{skip_msg}...")

        # DYNAMIC DASK TASK STREAM
        total_files = len(files_to_process)
        iterator = iter(enumerate(files_to_process))
        seq = as_completed()

        def submit_task(item):
            i, file_path = item
            out_name = os.path.splitext(file_path.name)[0] + "_filled.tif"
            output_path = self.filled_out_dir / out_name
            return client.submit(
                self.process_gap_fill,
                str(file_path),
                str(output_path),
                max_iters,
                chunk_size,
                current_index=i + 1,
                total_count=total_files
            )

        # Initial queue fill
        for _ in range(min(max_concurrent, total_files)):
            try:
                seq.add(submit_task(next(iterator)))
            except StopIteration:
                break

        # Process stream
        for future in seq:
            future.result() 
            try:
                seq.add(submit_task(next(iterator)))
            except StopIteration:
                pass

        if total_files > 0:
            logger.info("[SUCCESS] Gap Fill processing complete.")
        else:
            logger.info("No new rasters to gap fill.")

        # Cleanly shut down Dask
        try:
            client.close()
            self.close_dask()
        except Exception as e:
            logger.error(f"Could not cleanly close client/cluster: {e}")