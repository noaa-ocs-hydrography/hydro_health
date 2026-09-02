"""Class for reading and warping TIFF files using parallel processing"""

import os
import gc
import shutil
import psutil
import pathlib
import tempfile
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import s3fs
from pathlib import Path
from osgeo import gdal
from scipy.ndimage import uniform_filter
from rasterio.warp import transform_bounds
from shapely.geometry import box
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

INPUTS = pathlib.Path(__file__).resolve().parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).resolve().parents[4] / 'outputs'


def _apply_tsm_smoothing(gdal_dst_str: str, src_str: str, warp_opts: dict) -> None:
    """Applies uniform filter smoothing to a raster dataset."""
    tmp_ds = gdal.Open(gdal_dst_str)
    pixel_size = tmp_ds.GetGeoTransform()[1]
    tmp_ds = None 
    
    mem = psutil.virtual_memory()
    Engine.write_message_dask(f"[SMOOTHING INIT] {os.path.basename(src_str)} | Sys RAM: {mem.percent}% ({mem.used / 1024**3:.1f}GB / {mem.total / 1024**3:.1f}GB)", OUTPUTS)

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
                            Engine.write_message_dask(f"[PROGRESS] {os.path.basename(src_str)} Smoothing: Chunk {current_chunk}/{total_chunks} | Sys RAM: {mem.percent}%", OUTPUTS)

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


def _warp_to_cutline(src_path: str, dst_path: str, cutline_path: str, task_tmp_dir: str, is_aws: bool, target_crs: str, target_res: float, src_nodata, is_tsm: bool) -> None:
    """Helper to handle GDAL Warp boilerplate."""
    
    src_str = str(src_path)
    dst_str = str(dst_path)

    if is_aws and src_str.startswith('s3://'):
        src_str = src_str.replace('s3://', '/vsis3/')

    if is_aws:
        gdal_dst_str = str(pathlib.Path(task_tmp_dir) / "warp_tmp.tif")
    else:
        gdal_dst_str = dst_str

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
        'warpMemoryLimit': 1024,
        'outputType': gdal.GDT_Float32,
        'resampleAlg': 'bilinear',
        'dstSRS': target_crs,
        'xRes': target_res,
        'yRes': target_res,
        'cropToCutline': any(k in os.path.basename(src_str).lower() for k in ["tsm", "sed", "hurr"]),
    }
    
    if src_nodata is not None:
        warp_opts['srcNodata'] = src_nodata

    try:
        ds = gdal.Warp(gdal_dst_str, src_str, **warp_opts)

        if ds is None:
            raise RuntimeError(f"gdal.Warp returned None for {os.path.basename(src_str)}")
        
        ds = None

        if is_tsm:
            _apply_tsm_smoothing(gdal_dst_str, src_str, warp_opts)
        
        if is_aws:
            fs = s3fs.S3FileSystem()
            fs.put(gdal_dst_str, dst_str)
            Engine.write_message_dask(f"[SUCCESS] Wrote to S3 successfully: {os.path.basename(dst_str)}", OUTPUTS)
            if os.path.exists(gdal_dst_str):
                os.remove(gdal_dst_str)
        else:
            Engine.write_message_dask(f"[SUCCESS] Wrote locally successfully: {os.path.basename(dst_str)}", OUTPUTS)

    except Exception as e:
        Engine.write_message_dask(f"[ERROR] GDAL Warp/Upload failed for {os.path.basename(src_str)}! Exception: {e}", OUTPUTS)
        raise e
    finally:
        if hasattr(gdal, 'VSICurlClearCache'):
            gdal.VSICurlClearCache()


def _process_prediction_raster(params: list) -> None:
    """Core worker task for processing prediction rasters. Designed for dask pickling."""
    
    raster_path, mask_bounds, output_path, cutline_path, is_aws, target_crs, target_res, local_tmp_dir = params
    raster_name = pathlib.Path(raster_path).name.lower()
    
    # Implicit Skipping 
    if UPath(output_path).exists():
        Engine.write_message_dask(f"[SKIP] Output already exists for {raster_name}. Skipping computation.", OUTPUTS)
        return

    open_path = str(raster_path)
    if is_aws and open_path.startswith('s3://'):
        open_path = open_path.replace('s3://', '/vsis3/')
        
    Engine.write_message_dask(f"[STARTING] Worker executing prediction on: {raster_name}", OUTPUTS)

    try:
        with rasterio.open(open_path) as src:
            src_nodata = src.nodata
            raster_crs = src.crs
            raster_bounds = src.bounds
    except Exception as e:
        Engine.write_message_dask(f"[ERROR] Could not open {raster_name} with rasterio. File might be corrupted: {e}", OUTPUTS)
        return

    if raster_crs is not None:
        try:
            target_crs_obj = rasterio.crs.CRS.from_string(target_crs)
            if raster_crs != target_crs_obj:
                left, bottom, right, top = transform_bounds(raster_crs, target_crs_obj, *raster_bounds)
            else:
                left, bottom, right, top = raster_bounds
        except Exception as e:
            Engine.write_message_dask(f"[WARNING] Failed to transform bounds for {raster_name}: {e}. Bypassing intersection check for safety.", OUTPUTS)
            left, bottom, right, top = None, None, None, None
    else:
        left, bottom, right, top = raster_bounds

    bounds_geom = box(min(left, right), min(bottom, top), max(left, right), max(bottom, top)) if left is not None else None

    if bounds_geom is not None:
        try:
            mask_box = box(*mask_bounds)
            if not mask_box.intersects(bounds_geom):
                Engine.write_message_dask(f"[SKIP] Bounding box does not intersect prediction raster {raster_name}.", OUTPUTS)
                return
        except Exception as e:
            Engine.write_message_dask(f"[WARNING] Bounding box check failed for {raster_name}: {e}", OUTPUTS)
            return

    Engine.write_message_dask(f"[PROCESSING] Starting warp on prediction file {raster_name}...", OUTPUTS)
    is_tsm = "tsm" in raster_name or "strength" in raster_name

    try:
        with tempfile.TemporaryDirectory(dir=local_tmp_dir) as task_tmp_dir:
            _warp_to_cutline(
                src_path=raster_path, 
                dst_path=output_path, 
                cutline_path=cutline_path, 
                task_tmp_dir=task_tmp_dir,
                is_aws=is_aws,
                target_crs=target_crs,
                target_res=target_res,
                src_nodata=src_nodata,
                is_tsm=is_tsm
            )
    except Exception as e:
        Engine.write_message_dask(f"[ERROR] Unexpected failure during _warp_to_cutline for {raster_name}: {e}", OUTPUTS)
    finally:
        gc.collect()


class PredictionRastersEngine(Engine):
    """Class for processing prediction rasters in parallel using Dask"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize the engine with necessary flat configurations"""
        super().__init__()
        
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
        self.output_prefix = output_prefix
        
        self.env = self.param_lookup.get('env', 'local')
        self.is_aws = self.env in ['remote', 'aws']
        self.gdal_env_vars = self.param_lookup.get('gdal_env_vars', {})
        
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "prediction_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        self.inputs_dir = INPUTS

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f"PredictionRastersEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        mask_pq = get_config_item('MASK', 'PREDICTION_MASK_PQ')
        self.mask_prediction_pq = UPath(f"{s3_dir_base}/{mask_pq}") if self.is_aws else UPath(self.outputs_dir / mask_pq)
        
        preprocessed_dir = get_config_item('MODEL', 'PREPROCESSED_DIR')
        self.preprocessed_dir = UPath(f"{s3_dir_base}/{preprocessed_dir}") if self.is_aws else UPath(self.outputs_dir / preprocessed_dir)
        
        pred_out_dir = get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR')
        self.prediction_out_dir = UPath(f"{s3_dir_base}/{pred_out_dir}") if self.is_aws else UPath(self.outputs_dir / pred_out_dir)
        
        tiled_lidar = get_config_item('MODEL', 'TILED_LIDAR_PROC')
        self.uncombined_lidar_dir = UPath(f"{s3_dir_base}/{tiled_lidar}") if self.is_aws else UPath(self.outputs_dir / tiled_lidar)

        filled_dir_path = get_config_item('TERRAIN', 'FILLED_DIR')
        self.filled_folder_name = UPath(filled_dir_path).name.lower() if filled_dir_path else "filled_tifs"
            
        self.preprocessed_subdirs = {
            'bluetopo': UPath(f"{s3_dir_base}/{get_config_item('PREPROCESSED', 'BLUETOPO')}") if self.is_aws else UPath(self.outputs_dir / get_config_item('PREPROCESSED', 'BLUETOPO')),
            'hurricane': UPath(f"{s3_dir_base}/{get_config_item('PREPROCESSED', 'HURRICANE')}") if self.is_aws else UPath(self.outputs_dir / get_config_item('PREPROCESSED', 'HURRICANE')),
            'lidar': UPath(f"{s3_dir_base}/{get_config_item('MODEL', 'TILED_LIDAR_DIR')}") if self.is_aws else UPath(self.outputs_dir / get_config_item('MODEL', 'TILED_LIDAR_DIR')),
            'sediment': UPath(f"{s3_dir_base}/{get_config_item('PREPROCESSED', 'SEDIMENT')}") if self.is_aws else UPath(self.outputs_dir / get_config_item('PREPROCESSED', 'SEDIMENT')),
            'tsm': UPath(f"{s3_dir_base}/{get_config_item('PREPROCESSED', 'TSM')}") if self.is_aws else UPath(self.outputs_dir / get_config_item('PREPROCESSED', 'TSM'))
        }

        if not self.is_aws:
            self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_out_dir.mkdir(parents=True, exist_ok=True)


    def prepare_spatial_masks(self, region: str) -> tuple:
        """Process and validate geometries for prediction masks"""

        self.write_message(f"Preparing spatial masks for region {region}...", OUTPUTS)
        mask_pred_gdf = gpd.read_parquet(str(self.mask_prediction_pq))

        if hasattr(self, 'target_crs') and self.target_crs:
            if mask_pred_gdf.crs is None: 
                mask_pred_gdf = mask_pred_gdf.set_crs(self.target_crs)
            elif mask_pred_gdf.crs != self.target_crs: 
                mask_pred_gdf = mask_pred_gdf.to_crs(self.target_crs)

        mask_pred_gdf['geometry'] = mask_pred_gdf.geometry.make_valid().buffer(0)
        mask_pred_gdf = mask_pred_gdf[~mask_pred_gdf.is_empty & mask_pred_gdf.geometry.notnull()]
        
        if mask_pred_gdf.empty:
            raise ValueError("Prediction Mask GeoDataFrame is empty after validation. Check your input geometries.")

        mask_pred_bounds = mask_pred_gdf.total_bounds
        pred_cutline_path = str(self.local_tmp_dir / f"pred_cutline_{region}.gpkg")
        
        mask_pred_gdf.to_file(pred_cutline_path, driver='GPKG')

        return mask_pred_bounds, pred_cutline_path


    def get_valid_source_files(self) -> list:
        """Scan and filter input directories for valid TIFF files"""

        potential_files = []
        self.write_message("Scanning directories for source TIFF files", OUTPUTS)

        for data_type, directory in self.preprocessed_subdirs.items():
            self.write_message(f"-> [{data_type.upper()}] Scanning: {directory}", OUTPUTS)
            if directory.exists() or self.is_aws: 
                found_files = [f for f in directory.rglob("*") if f.suffix.lower() in ['.tif', '.tiff']]
                if not found_files:
                    self.write_message("   [WARNING] No .tif or .tiff files found here.", OUTPUTS)
                else:
                    self.write_message(f"   Found {len(found_files)} potential TIFF files.", OUTPUTS)
                    potential_files.extend(found_files)

        excluded_folders = {self.filled_folder_name, 'filled_tifs', 'filled_lidar'}
        self.write_message(f"Applying Exclusion Rules (Folders to skip: {excluded_folders})", OUTPUTS)
        
        valid_source_files = []
        excluded_count = 0
        excluded_by_folder_counts = {folder: 0 for folder in excluded_folders}

        for f in potential_files:
            if "sand_mud_mask" in f.name:
                excluded_count += 1
                continue
                
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

        self.write_message(f"Excluded {excluded_count} total files.", OUTPUTS)
        self.write_message(f"Remaining valid source files: {len(valid_source_files)}", OUTPUTS)

        return valid_source_files


    def run(self) -> None:
        """Main entry point for processing prediction rasters in parallel"""
        
        self.write_message("Initializing Cluster and Processing Masks", OUTPUTS)
        
        try:
            self.setup_dask(self.env, n_workers=4, threads_per_worker=1, memory_limit="6GB")
            
            regions = self.param_lookup.get('eco_regions', [])
            if not isinstance(regions, list):
                regions = [regions]

            for eco_region in regions:
                region_str = str(eco_region).strip("[]'\" ")
                self.write_message(f"--- Starting Pipeline for Ecoregion: {region_str} ---", OUTPUTS)
                self.write_message(self.log_system_metrics(), OUTPUTS)

                self._resolve_paths(region_str)
                mask_pred_bounds, pred_cutline_path = self.prepare_spatial_masks(region_str)
                prediction_files = self.get_valid_source_files()

                self.write_message(f"Building task stream for {len(prediction_files)} files...", OUTPUTS)
                
                params_list = []
                for file_path in prediction_files:
                    base_out = self.uncombined_lidar_dir if "mosaic" in file_path.name.lower() else self.prediction_out_dir
                    output_path = base_out / file_path.name
                    
                    params_list.append([
                        str(file_path), 
                        mask_pred_bounds, 
                        str(output_path),
                        pred_cutline_path,
                        self.is_aws,
                        self.target_crs,
                        self.target_res,
                        str(self.local_tmp_dir)
                    ])

                if params_list:
                    self.write_message(f"Submitting {len(params_list)} prediction tasks to Dask client map...", OUTPUTS)
                    futures = self.client.map(_process_prediction_raster, params_list)
                    self.client.gather(futures)

                self.write_message(f"--- Finished Ecoregion: {region_str} ---", OUTPUTS)
                self.write_message(self.log_system_metrics(), OUTPUTS)

        finally:
            self.write_message("Processing Complete. Shutting down Dask cluster to free memory.", OUTPUTS)
            self.cleanup_resources(OUTPUTS)