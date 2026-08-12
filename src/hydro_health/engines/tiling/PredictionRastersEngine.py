"""Class for reading and warping TIFF files using parallel processing"""

import os
import shutil
import psutil
import pathlib
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import s3fs
from osgeo import gdal
from scipy.ndimage import uniform_filter
from rasterio.warp import transform_bounds
from shapely.geometry import box
from dask.distributed import LocalCluster, Client, as_completed
from upath import UPath
from hydro_health.helpers.tools import get_config_item

from hydro_health.engines.Engine import Engine

class PredictionRastersEngine(Engine):
    """Class for processing prediction rasters in parallel using Dask"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize the engine with necessary configuration paths and settings"""
        super().__init__()
        self.param_lookup = param_lookup
        
        # Helper to safely extract values whether they are plain types or Param objects
        def _get_val(key, default=None):
            val = self.param_lookup.get(key, default)
            if hasattr(val, 'valueAsText') and val.valueAsText is not None:
                return val.valueAsText
            if hasattr(val, 'value') and val.value is not None:
                return val.value
            return val
            
        self.local_tmp_dir = pathlib.Path(_get_val('local_tmp_dir', '/tmp'))
        # FIX: Explicitly create the temp directory so Fiona, Dask, and Tempfile don't crash
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_crs = _get_val('target_crs')
        self.target_res = _get_val('target_res', 10.0)
        
        env = _get_val('env', 'local')
        self.is_aws = env in ['remote', 'aws']
        self.overwrite = _get_val('overwrite', False)
        self.gdal_env_vars = _get_val('gdal_env_vars', {})
        
        print(f"Environment detected: {'AWS/Remote' if self.is_aws else 'Local'}")

        # ---------------------------------------------------------
        # NEW FIX: Dynamically determine Repo Root and base folders
        # __file__ = src/hydro_health/engines/tiling/PredictionRastersEngine.py
        # parents[4] points to the hydro_health/ repository root
        # ---------------------------------------------------------
        self.repo_root = pathlib.Path(__file__).resolve().parents[4]
        
        in_dir = _get_val('input_directory')
        out_dir = _get_val('output_directory')
        
        # Use param_lookup paths if provided (and not empty strings), else default to repo root
        base_in_dir = pathlib.Path(in_dir) if in_dir else self.repo_root / 'inputs'
        base_out_dir = pathlib.Path(out_dir) if out_dir else self.repo_root / 'outputs'
        
        # Mimic RasterMaskEngine logic: append output_prefix to output folder if it exists
        self.inputs_dir = base_in_dir
        self.outputs_dir = base_out_dir / output_prefix if output_prefix and isinstance(output_prefix, str) else base_out_dir
        
        # Dynamically determine the ecoregion (e.g., 'ER_3') to append to output paths
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

        # If it wasn't explicitly provided in param_lookup, scan the directory
        if not self.ecoregion:
            er_dirs = [d.name for d in self.outputs_dir.glob('ER_*') if d.is_dir()]
            if er_dirs:
                self.ecoregion = er_dirs[0]

        # Finally, append the ecoregion into the path!
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

        # Apply the path resolver to all attributes
        self.mask_prediction_pq = _resolve_path(get_config_item('MASK', 'PREDICTION_MASK_PQ'), is_output=True)
        self.preprocessed_dir = _resolve_path(get_config_item('MODEL', 'PREPROCESSED_DIR'), is_output=True)
        
        self.prediction_out_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'), is_output=True)
        self.uncombined_lidar_dir = _resolve_path(get_config_item('MODEL', 'TILED_LIDAR_PROC'), is_output=True)

        try:
            filled_dir_path = get_config_item('TERRAIN', 'FILLED_DIR')
            self.filled_folder_name = UPath(filled_dir_path).name.lower()
        except Exception:
            print("Could not load TERRAIN/FILLED_DIR from config. Falling back to default 'filled_tifs'.")
            self.filled_folder_name = "filled_tifs"

        self.preprocessed_subdirs = {
            'bluetopo': _resolve_path(get_config_item('PREPROCESSED', 'BLUETOPO'), is_output=True),
            'hurricane': _resolve_path(get_config_item('PREPROCESSED', 'HURRICANE'), is_output=True),
            'lidar': _resolve_path(get_config_item('MODEL', 'TILED_LIDAR_DIR'), is_output=True),
            'sediment': _resolve_path(get_config_item('PREPROCESSED', 'SEDIMENT'), is_output=True),
            'tsm': _resolve_path(get_config_item('PREPROCESSED', 'TSM'), is_output=True)
        }

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
            # Added missing directory creation for prediction outputs using robust path methods
            self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_out_dir.mkdir(parents=True, exist_ok=True)

        existing_pred_outputs = set()
        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            existing_pred_outputs.update({f.name for f in self.prediction_out_dir.rglob(ext)})
        
        existing_uncombined_outputs = set()
        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            existing_uncombined_outputs.update({f.name for f in self.uncombined_lidar_dir.rglob(ext)})
        
        all_existing_pred_outputs = existing_pred_outputs.union(existing_uncombined_outputs)

        potential_files = []

        for data_type, directory in self.preprocessed_subdirs.items():
            found_files = []
            for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
                found_files.extend(list(directory.rglob(ext)))
            
            if not found_files:
                raise RuntimeError(f"CRITICAL ERROR: Missing data for '{data_type}'. No .tif or .tiff files were found in {directory} or any of its subfolders.")
                
            potential_files.extend(found_files)

        excluded_folders = {self.filled_folder_name, 'filled_tifs', 'filled_lidar'}
        valid_source_files = []

        for f in potential_files:
            if "sand_mud_mask" in f.name:
                continue
                
            if any(folder in f.parts for folder in excluded_folders):
                continue
                
            valid_source_files.append(f)

        prediction_files = []
        
        for f in valid_source_files:
            if not self.overwrite and f.name in all_existing_pred_outputs:
                continue
            prediction_files.append(f)

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
                # Use robust bounding box construction strictly handling minimums and maximums
                if raster_crs != target_crs_obj:
                    left, bottom, right, top = transform_bounds(raster_crs, target_crs_obj, *raster_bounds)
                    bounds_geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top))
                else:
                    bounds_geom = box(min(raster_bounds[0], raster_bounds[2]), min(raster_bounds[1], raster_bounds[3]), max(raster_bounds[0], raster_bounds[2]), max(raster_bounds[1], raster_bounds[3]))
            except Exception as e:
                print(f"Failed to transform bounds for {raster_name}: {e}. Bypassing intersection check for safety.")
                bounds_geom = None
        else:
            bounds_geom = box(min(raster_bounds[0], raster_bounds[2]), min(raster_bounds[1], raster_bounds[3]), max(raster_bounds[0], raster_bounds[2]), max(raster_bounds[1], raster_bounds[3]))

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
            'warpMemoryLimit': 1024, # 512 MB
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
            if hasattr(gdal, 'VSICurlClearCache'):
                gdal.VSICurlClearCache()

    def _apply_tsm_smoothing(self, gdal_dst_str: str, src_str: str, warp_opts: dict) -> None:
        """Applies uniform filter smoothing to a raster dataset."""
        tmp_ds = gdal.Open(gdal_dst_str)
        pixel_size = tmp_ds.GetGeoTransform()[1]
        tmp_ds = None 
        
        mem = psutil.virtual_memory()
        print(f" [SMOOTHING INIT] {os.path.basename(src_str)} | Sys RAM: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)")

        radius_pixels = int(2000 / abs(pixel_size))
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
            
            block_size = 1024
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

    def run(self) -> None:
        """Main entry point for processing prediction rasters in parallel"""

        print('Initializing Cluster and Processing Masks')
        
        # Use param_lookup and the base Engine class to initialize Dask
        env_val = self.param_lookup.get('env', 'local')
        env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        
        self.setup_dask(env)
        
        # Pull the client configured by setup_dask, or fallback to connecting to the default cluster
        client = getattr(self, 'client', None)
        if not client:
            client = Client()

        mask_pred_bounds, pred_cutline_path = self.prepare_spatial_masks()
        prediction_files = self.get_valid_source_files()

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

        print('Processing Complete. Shutting down Dask cluster to free memory.')
        try:
            client.close()
            self.close_dask()
        except Exception as e:
            print(f"Could not cleanly close client/cluster before next execution step: {e}")