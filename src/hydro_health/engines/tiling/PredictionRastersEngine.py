"""Class for reading and warping TIFF files using parallel processing"""

import os
import gc
import shutil
import psutil
import pathlib
import tempfile
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import s3fs
from osgeo import gdal
from scipy.ndimage import uniform_filter
from rasterio.warp import transform_bounds
from shapely.geometry import box
from dask.distributed import Client, as_completed
from upath import UPath
from hydro_health.helpers.tools import get_config_item

from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)

class PredictionRastersEngine(Engine):
    """Class for processing prediction rasters in parallel using Dask"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize the engine with necessary configuration paths and settings"""
        super().__init__()
        
        # Constrain GDAL's background Block Cache to 256MB to prevent memory bloat
        gdal.SetCacheMax(256 * 1024 * 1024)
        
        # Unwrap values from arcpy Param objects so standard dictionary .get() works cleanly everywhere
        self.param_lookup = {}
        for k, v in (param_lookup or {}).items():
            if hasattr(v, 'valueAsText') and v.valueAsText is not None:
                self.param_lookup[k] = v.valueAsText
            elif hasattr(v, 'value') and v.value is not None:
                self.param_lookup[k] = v.value
            else:
                self.param_lookup[k] = v
                
        self.target_crs = 'EPSG:32617'
        self.target_res = 8.0
        
        self.local_tmp_dir = pathlib.Path(self.param_lookup.get('local_tmp_dir', '/tmp'))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        env = self.param_lookup.get('env', 'local')
        self.is_aws = env in ['remote', 'aws']
        self.overwrite = self.param_lookup.get('overwrite', False)
        self.gdal_env_vars = self.param_lookup.get('gdal_env_vars', {})
        
        print(f"Environment detected: {'AWS/Remote' if self.is_aws else 'Local'}")

        self.repo_root = pathlib.Path(__file__).resolve().parents[4]
        
        in_dir = self.param_lookup.get('input_directory')
        out_dir = self.param_lookup.get('output_directory')
        
        # Use param_lookup paths if provided (and not empty strings), else default to repo root
        base_in_dir = pathlib.Path(in_dir) if in_dir else self.repo_root / 'inputs'
        base_out_dir = pathlib.Path(out_dir) if out_dir else self.repo_root / 'outputs'
        
        # Mimic RasterMaskEngine logic: append output_prefix to output folder if it exists
        self.inputs_dir = base_in_dir
        self.outputs_dir = base_out_dir / output_prefix if output_prefix and isinstance(output_prefix, str) else base_out_dir
        
        # Dynamically determine the ecoregion (e.g., 'ER_3') to append to output paths
        eco_val = self.param_lookup.get('eco_regions')
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
                # 1. If the config provides an absolute path, just use it
                if p.is_absolute():
                    return UPath(p)
                # 2. If the config value already starts with 'inputs' or 'outputs', 
                #    strip the first folder and attach it directly to the correctly built 
                #    inputs_dir or outputs_dir so that ecoregion prefixes are respected!
                if p.parts:
                    if p.parts[0] == 'inputs':
                        return UPath(self.inputs_dir / pathlib.Path(*p.parts[1:]))
                    elif p.parts[0] == 'outputs':
                        return UPath(self.outputs_dir / pathlib.Path(*p.parts[1:]))
                # 3. Otherwise, map flat filenames/paths to the correct base directory
                base_dir = self.outputs_dir if is_output else self.inputs_dir
                return UPath(base_dir / p)

        self.mask_prediction_pq = _resolve_path(get_config_item('MASK', 'PREDICTION_MASK_PQ'), is_output=True)
        self.preprocessed_dir = _resolve_path(get_config_item('MODEL', 'PREPROCESSED_DIR'), is_output=True)
        
        self.prediction_out_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'), is_output=True)
        self.uncombined_lidar_dir = _resolve_path(get_config_item('MODEL', 'TILED_LIDAR_PROC'), is_output=True)

        filled_dir_path = get_config_item('TERRAIN', 'FILLED_DIR')
        if filled_dir_path:
            self.filled_folder_name = UPath(filled_dir_path).name.lower()
        else:
            print("Could not load TERRAIN/FILLED_DIR from config. Falling back to default 'filled_tifs'.")
            self.filled_folder_name = "filled_tifs"
            
        self.preprocessed_subdirs = {
            'bluetopo': _resolve_path(get_config_item('PREPROCESSED', 'BLUETOPO'), is_output=True),
            'hurricane': _resolve_path(get_config_item('PREPROCESSED', 'HURRICANE'), is_output=True),
            'lidar': _resolve_path(get_config_item('MODEL', 'TILED_LIDAR_DIR'), is_output=True),
            'sediment': _resolve_path(get_config_item('PREPROCESSED', 'SEDIMENT'), is_output=True),
            'tsm': _resolve_path(get_config_item('PREPROCESSED', 'TSM'), is_output=True)
        }

    def __getstate__(self):
        """
        Exclude unpicklable attributes (like Dask Client/Cluster and raw Params)
        when serializing this instance to send to Dask worker nodes.
        """
        state = self.__dict__.copy()
        
        # Drop the active connection handlers which crash the serializer
        state.pop('client', None)
        state.pop('cluster', None)
        
        # Drop raw parameter lookups in case they contain unpicklable COM objects
        state.pop('param_lookup', None)
        
        return state
        
    def _log_system_metrics(self) -> str:
        """Helper to collect and format EC2 system metrics (RAM, Disk Space, Temp Size)."""
        try:
            # 1. Overall Hard Drive (EBS) Space for the partition holding the temp directory
            total, used, free = shutil.disk_usage(self.local_tmp_dir)
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            # 2. Temp Storage Folder Size
            tmp_size_bytes = 0
            if self.local_tmp_dir.exists():
                tmp_size_bytes = sum(f.stat().st_size for f in self.local_tmp_dir.rglob('*') if f.is_file())
            tmp_mb = tmp_size_bytes / (1024**2)
            
            # 3. RAM (Using psutil if available, fallback to /proc/meminfo for EC2/Linux)
            ram_info = "Unknown"
            try:
                import psutil
                vm = psutil.virtual_memory()
                ram_info = f"Free: {vm.available / (1024**3):.1f}GB / {vm.total / (1024**3):.1f}GB (Used: {vm.percent}%)"
            except ImportError:
                # Fallback to reading native Linux memory info if psutil is not installed
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

    def prepare_spatial_masks(self) -> tuple:
        """Process and validate geometries for prediction masks"""

        mask_pred_gdf = gpd.read_parquet(str(self.mask_prediction_pq))

        if hasattr(self, 'target_crs') and self.target_crs:
            if mask_pred_gdf.crs is None: 
                mask_pred_gdf = mask_pred_gdf.set_crs(self.target_crs)
            elif mask_pred_gdf.crs != self.target_crs: 
                mask_pred_gdf = mask_pred_gdf.to_crs(self.target_crs)

        mask_pred_gdf['geometry'] = mask_pred_gdf.geometry.make_valid().buffer(0)
        
        # Clean up any empty geometries resulting from buffer(0)
        mask_pred_gdf = mask_pred_gdf[~mask_pred_gdf.is_empty & mask_pred_gdf.geometry.notnull()]
        
        if mask_pred_gdf.empty:
            raise ValueError("Prediction Mask GeoDataFrame is empty after validation. Check your input geometries.")

        mask_pred_bounds = mask_pred_gdf.total_bounds
        
        pred_cutline_path = str(self.local_tmp_dir / "pred_cutline.gpkg")
        
        # Using GPKG natively builds a spatial R-Tree index
        mask_pred_gdf.to_file(pred_cutline_path, driver='GPKG')

        return mask_pred_bounds, pred_cutline_path

    def get_valid_source_files(self) -> list:
        """Scan and filter input directories for valid TIFF files"""

        if not self.is_aws:
            self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_out_dir.mkdir(parents=True, exist_ok=True)

        existing_pred_outputs = {f.name for f in self.prediction_out_dir.rglob("*") if f.suffix.lower() in ['.tif', '.tiff']}
        existing_uncombined_outputs = {f.name for f in self.uncombined_lidar_dir.rglob("*") if f.suffix.lower() in ['.tif', '.tiff']}
        
        all_existing_pred_outputs = existing_pred_outputs.union(existing_uncombined_outputs)

        potential_files = []
        
        print(f"\n{'='*80}")
        print("SCANNING DIRECTORIES FOR SOURCE TIFF FILES")
        print(f"{'='*80}")

        for data_type, directory in self.preprocessed_subdirs.items():
            print(f"-> [{data_type.upper()}] Scanning: {directory}")
            found_files = [f for f in directory.rglob("*") if f.suffix.lower() in ['.tif', '.tiff']]
            
            if not found_files:
                print(f"   WARNING: No .tif or .tiff files found here.")
            else:
                print(f"   Found {len(found_files)} potential TIFF files.")
                potential_files.extend(found_files)

        excluded_folders = {self.filled_folder_name, 'filled_tifs', 'filled_lidar'}
        print(f"\nApplying Exclusion Rules (Folders to skip: {excluded_folders})")
        
        valid_source_files = []
        excluded_count = 0
        excluded_by_folder_counts = {folder: 0 for folder in excluded_folders}

        for f in potential_files:
            if "sand_mud_mask" in f.name:
                excluded_count += 1
                continue
                
            # Check if any parent folder is in the exclusion list
            skip_file = False
            for folder in excluded_folders:
                if folder in f.parts:
                    excluded_by_folder_counts[folder] += 1
                    excluded_count += 1
                    skip_file = True
                    break
                    
            if skip_file:
                continue
                
            valid_source_files.append(f)

        print(f"Excluded {excluded_count} total files.")
        for folder, count in excluded_by_folder_counts.items():
            if count > 0:
                print(f"   - {count} files dropped because they were inside a '{folder}' folder.")
        
        print(f"Remaining valid source files: {len(valid_source_files)}")

        prediction_files = []
        already_processed_count = 0
        
        for f in valid_source_files:
            if not self.overwrite and f.name in all_existing_pred_outputs:
                already_processed_count += 1
                continue
            prediction_files.append(f)

        print(f"Skipped {already_processed_count} files that already exist in outputs (overwrite=False).")
        print(f"Final queue size to process: {len(prediction_files)}")
        print(f"{'='*80}\n")

        return prediction_files

    def process_prediction_raster(self, raster_path: str, mask_bounds: tuple, output_path: str, cutline_path: str) -> None:
        """Reprojects, resamples, and crops a raster for prediction."""
        
        raster_name = pathlib.Path(raster_path).name.lower()
        open_path = str(raster_path)
        
        if self.is_aws and open_path.startswith('s3://'):
            open_path = open_path.replace('s3://', '/vsis3/')
            
        print(f"-> [STARTING] Worker executing prediction on: {raster_name}")

        try:
            with rasterio.open(open_path) as src:
                src_nodata = src.nodata
                raster_crs = src.crs
                raster_bounds = src.bounds
        except Exception as e:
            print(f"Could not open {raster_name} with rasterio. File might be corrupted: {e}")
            return

        if raster_crs is not None:
            try:
                target_crs_obj = rasterio.crs.CRS.from_string(self.target_crs)
                if raster_crs != target_crs_obj:
                    left, bottom, right, top = transform_bounds(raster_crs, target_crs_obj, *raster_bounds)
                else:
                    left, bottom, right, top = raster_bounds
            except Exception as e:
                print(f"Failed to transform bounds for {raster_name}: {e}. Bypassing intersection check for safety.")
                left, bottom, right, top = None, None, None, None
        else:
            left, bottom, right, top = raster_bounds

        bounds_geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top)) if left is not None else None

        if bounds_geom is not None:
            try:
                mask_box = box(*mask_bounds)
                if not mask_box.intersects(bounds_geom):
                    print(f"- [SKIP] Bounding box does not intersect prediction raster {raster_name}.")
                    return
            except Exception as e:
                print(f"Bounding box check failed for {raster_name}: {e}")
                return

        print(f" [PROCESSING] Starting warp on prediction file {raster_name}...")
        should_crop = any(k in raster_name for k in ["tsm", "sed", "hurr"])
        is_tsm = "tsm" in raster_name or "strength" in raster_name

        with tempfile.TemporaryDirectory(dir=self.local_tmp_dir) as task_tmp_dir:
            try:
                self._warp_to_cutline(
                    raster_path, 
                    output_path, 
                    cutline_path, 
                    task_tmp_dir=task_tmp_dir,
                    dst_crs=self.target_crs, 
                    x_res=self.target_res, 
                    y_res=self.target_res,
                    crop_to_cutline=should_crop,
                    src_nodata=src_nodata,
                    apply_tsm_smoothing=is_tsm,
                    resample_alg='bilinear' 
                )
            except Exception as e:
                print(f"Unexpected failure during _warp_to_cutline for {raster_name}: {e}")
            finally:
                # Force memory release back to the OS after every single file
                gc.collect()
                print(self._log_system_metrics())

    def _warp_to_cutline(self, src_path: str, dst_path: str, cutline_path: str, task_tmp_dir: str = None, **kwargs) -> None:
        """Helper to handle GDAL Warp boilerplate."""
        
        src_str = str(src_path)
        dst_str = str(dst_path)

        if self.is_aws and src_str.startswith('s3://'):
            src_str = src_str.replace('s3://', '/vsis3/')

        if self.is_aws:
            gdal_dst_str = str(pathlib.Path(task_tmp_dir) / "warp_tmp.tif")
        else:
            gdal_dst_str = dst_str

        resample_alg = kwargs.pop('resample_alg', None) 

        warp_opts = {
            'cutlineDSName': cutline_path,
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE'],
            'creationOptions': [
                'TILED=YES', 
                'BLOCKXSIZE=512',
                'BLOCKYSIZE=512',
                'COMPRESS=LZW',
                'BIGTIFF=YES',
                'NUM_THREADS=1'
            ],
            'multithread': False,
            'warpMemoryLimit': 256, # Lowered from 1024 MB to conserve RAM per worker
            'outputType': gdal.GDT_Float32 
        }
        
        if resample_alg:
            warp_opts['resampleAlg'] = resample_alg 
            
        if 'dst_crs' in kwargs: warp_opts['dstSRS'] = kwargs.pop('dst_crs')
        if 'x_res' in kwargs: warp_opts['xRes'] = kwargs.pop('x_res')
        if 'y_res' in kwargs: warp_opts['yRes'] = kwargs.pop('y_res')
        if 'crop_to_cutline' in kwargs: warp_opts['cropToCutline'] = kwargs.pop('crop_to_cutline')
        if 'src_nodata' in kwargs: warp_opts['srcNodata'] = kwargs.pop('src_nodata')
        if 'dst_nodata' in kwargs: warp_opts['dstNodata'] = kwargs.pop('dst_nodata')
        
        apply_tsm_smoothing = kwargs.pop('apply_tsm_smoothing', False)
        
        try:
            ds = gdal.Warp(gdal_dst_str, src_str, **warp_opts)

            if ds is None:
                raise RuntimeError(f"gdal.Warp returned None for {os.path.basename(src_str)}")

            ds = None

            if apply_tsm_smoothing:
                self._apply_tsm_smoothing(gdal_dst_str, src_str, warp_opts)
            
            if self.is_aws:
                fs = s3fs.S3FileSystem()
                fs.put(gdal_dst_str, dst_str)
                print(f" - [✓ SUCCESS] Wrote to S3 successfully: {os.path.basename(dst_str)}")
            else:
                print(f" - [✓ SUCCESS] Wrote locally successfully: {os.path.basename(dst_str)}")

        except Exception as e:
            print(f" - [✗ ERROR] GDAL Warp/Upload failed for {os.path.basename(src_str)}! Exception: {e}")
            raise e
        finally:
            # Explicitly delete the temp file immediately to free EC2 disk space
            if gdal_dst_str != dst_str and pathlib.Path(gdal_dst_str).exists():
                try:
                    os.remove(gdal_dst_str)
                except Exception as e:
                    logger.warning(f"Failed to explicitly delete temp file {gdal_dst_str}: {e}")

            if hasattr(gdal, 'VSICurlClearCache'):
                gdal.VSICurlClearCache()

    def _apply_tsm_smoothing(self, gdal_dst_str: str, src_str: str, warp_opts: dict) -> None:
        """Applies uniform filter smoothing to a raster dataset."""
        tmp_ds = gdal.Open(gdal_dst_str)
        pixel_size = tmp_ds.GetGeoTransform()[1]
        tmp_ds = None 
        
        mem = psutil.virtual_memory()
        print(f" [SMOOTHING INIT] {os.path.basename(src_str)} | Sys RAM: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")

        # Cap the radius pixels to prevent massive padding allocations on high-res files
        radius_pixels = min(1000, int(2000 / abs(pixel_size)))
        size = radius_pixels * 2 + 1
        
        dst_path_obj = pathlib.Path(gdal_dst_str)
        smoothed_tmp = str(dst_path_obj.with_name(f"{dst_path_obj.stem}_smoothed{dst_path_obj.suffix}"))

        with rasterio.open(gdal_dst_str) as src:
            kwargs = src.meta.copy()
            kwargs.update({
                'dtype': 'float32', 
                'tiled': True,
                'blockxsize': 512,
                'blockysize': 512,
                'compress': 'lzw', 
                'bigtiff': 'yes' 
            })
            nodata = src.nodata if src.nodata is not None else warp_opts.get('dstNodata', warp_opts.get('srcNodata', -9999.0))
            
            # Lowered chunk size to 512 to preserve RAM
            block_size = 512
            total_chunks_x = (src.width + block_size - 1) // block_size
            total_chunks_y = (src.height + block_size - 1) // block_size
            total_chunks = total_chunks_x * total_chunks_y
            current_chunk = 0
            
            with rasterio.Env(CHECK_DISK_FREE_SPACE="FALSE"): 
                with rasterio.open(smoothed_tmp, 'w', **kwargs) as dst:
                    for y in range(0, src.height, block_size):
                        for x in range(0, src.width, block_size):
                            
                            current_chunk += 1
                            if current_chunk % max(1, total_chunks // 10) == 0 or current_chunk == total_chunks:
                                mem = psutil.virtual_memory()
                                print(f"   -> [PROGRESS] {os.path.basename(src_str)} Smoothing: Chunk {current_chunk}/{total_chunks} | Sys RAM: {mem.percent}%")

                            core_width = min(block_size, src.width - x)
                            core_height = min(block_size, src.height - y)
                            window = rasterio.windows.Window(x, y, core_width, core_height)
                            
                            read_xoff = max(0, x - radius_pixels)
                            read_yoff = max(0, y - radius_pixels)
                            read_right = min(src.width, x + core_width + radius_pixels)
                            read_bottom = min(src.height, y + core_height + radius_pixels)
                            
                            read_window = rasterio.windows.Window(
                                read_xoff, 
                                read_yoff, 
                                read_right - read_xoff, 
                                read_bottom - read_yoff
                            )
                            
                            array = src.read(1, window=read_window).astype(np.float32)
                            
                            if pd.isna(nodata):
                                valid_mask = (~np.isnan(array)).astype(np.float32)
                                array[np.isnan(array)] = 0
                            else:
                                valid_mask = ((array != nodata) & ~np.isnan(array)).astype(np.float32)
                                array[array == nodata] = 0
                                array[np.isnan(array)] = 0

                            smoothed = uniform_filter(array, size=size, mode='constant', cval=0.0)
                            weights = uniform_filter(valid_mask, size=size, mode='constant', cval=0.0)
                            
                            with np.errstate(divide='ignore', invalid='ignore'):
                                final_array = np.where(weights > 0, smoothed / weights, nodata)
                            
                            row_offset = int(y - read_yoff)
                            col_offset = int(x - read_xoff)
                            core_array = final_array[row_offset:row_offset + core_height, col_offset:col_offset + core_width]
                            
                            original_core = array[row_offset:row_offset + core_height, col_offset:col_offset + core_width]
                            
                            if pd.isna(nodata):
                                out_of_bounds_mask = np.isnan(original_core)
                            else:
                                out_of_bounds_mask = (original_core == nodata)
                            
                            core_array[out_of_bounds_mask] = nodata
                            
                            dst.write(core_array.astype(kwargs['dtype']), 1, window=window)

        if os.path.exists(gdal_dst_str):
            os.remove(gdal_dst_str)
        shutil.move(smoothed_tmp, gdal_dst_str)
        
    def _cleanup_resources(self):
        """Cleanup logic to run after processing stream is complete."""
        # Wipe the master local temp directory to ensure EC2 disk is completely cleared
        if self.local_tmp_dir.exists():
            try:
                shutil.rmtree(self.local_tmp_dir)
                print(f"Successfully wiped master local temp directory: {self.local_tmp_dir}")
            except Exception as e:
                logger.warning(f"Failed to wipe master local temp directory: {e}")

    def _initialize_cluster(self) -> Client:
        """Initializes the Dask cluster based on environment configuration."""
        print('Initializing Cluster')
        env = self.param_lookup.get('env', 'local')
        # Changed to 3 workers, 1 thread each, with a hard 9GB cap
        self.setup_dask(env, n_workers=3, threads_per_worker=1, memory_limit="9GB")
        return getattr(self, 'client', None)

    def _execute_task_stream(self, client: Client, prediction_files: list, mask_pred_bounds: tuple, pred_cutline_path: str) -> None:
        """Manages the asynchronous Dask task stream and queue for raster processing."""
        print('Starting Dask Task Stream')
        max_concurrent = 200 
        total_pred = len(prediction_files)
        prediction_iterator = iter(enumerate(prediction_files))
        seq = as_completed()
        
        def submit_pred_task(item):
            i, file_path = item
            base_out = self.uncombined_lidar_dir if "mosaic" in file_path.name.lower() else self.prediction_out_dir
            output_path = base_out / file_path.name
            return client.submit(
                self.process_prediction_raster,
                str(file_path), 
                mask_pred_bounds, 
                str(output_path),
                pred_cutline_path
            )

        # Initial queue fill
        for _ in range(min(max_concurrent, total_pred)):
            try:
                seq.add(submit_pred_task(next(prediction_iterator)))
            except StopIteration:
                break

        # Process stream
        for future in seq:
            try:
                future.result() 
            except Exception as e:
                print(f" [!] Task failed in stream processing: {e}")
            
            try:
                seq.add(submit_pred_task(next(prediction_iterator)))
            except StopIteration:
                pass

    def _shutdown_cluster(self, client: Client) -> None:
        """Safely shuts down the Dask client and cluster."""
        print('Processing Complete. Shutting down Dask cluster to free memory.')
        if client:
            try:
                client.close()
            except Exception as e:
                print(f"Could not cleanly close client: {e}")
                
        try:
            self.close_dask()
        except Exception as e:
            print(f"Could not cleanly close cluster before next execution step: {e}")

    def run(self) -> None:
        """Main entry point for processing prediction rasters in parallel"""
        
        # 1. Initialize Distributed Computing
        client = self._initialize_cluster()

        # 2. Prepare Data & Masks
        print('Processing Spatial Masks...')
        mask_pred_bounds, pred_cutline_path = self.prepare_spatial_masks()
        prediction_files = self.get_valid_source_files()

        # 3. Execute Pipeline
        if prediction_files and client:
            self._execute_task_stream(client, prediction_files, mask_pred_bounds, pred_cutline_path)
        else:
            print("No valid source files to process or Dask client failed to initialize.")

        # 4. Teardown & Cleanup
        self._shutdown_cluster(client)
        self._cleanup_resources()