"""Class for reading in tiff files and warping as needed for training"""

import logging
import tempfile
import pathlib
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import box

import dask
from dask.distributed import Client, LocalCluster, as_completed
from upath import UPath
import s3fs

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)


class TrainingRastersEngine(Engine):
    """Class for parallel processing training rasters and applying mathematical masks"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize paths, configurations, and environment for training rasters"""
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
            
        self.local_tmp_dir = pathlib.Path(_get_val('local_tmp_dir', str(Path.home() / "hydro_health_local_tmp")))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        self.target_crs = _get_val('target_crs', "EPSG:4326")
        self.target_res = float(_get_val('target_res', 1.0))
        
        env = _get_val('env', 'local')
        self.is_aws = env in ['remote', 'aws']
        self.overwrite = _get_val('overwrite', False)
        
        self.gdal_env_vars = _get_val('gdal_env_vars', {
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'AWS_NO_SIGN_REQUEST': 'YES'
        } if self.is_aws else {})
        
        logger.info(f"Environment detected: {'AWS/Remote' if self.is_aws else 'Local'}")

        # ---------------------------------------------------------
        # Dynamically determine Repo Root and base folders
        # __file__ = src/hydro_health/engines/tiling/TrainingRastersEngine.py
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

        self.mask_training_pq = _resolve_path(get_config_item('MASK', 'TRAINING_MASK_PQ'), is_output=True)
        self.train_mask_path = _resolve_path(get_config_item('MASK', 'MASK_TRAINING_PATH'), is_output=True)
        self.prediction_out_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'), is_output=True)
        self.training_out_dir = _resolve_path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'), is_output=True)
        self.training_tiles_dir = _resolve_path(get_config_item('MODEL', 'TRAINING_TILES_DIR'), is_output=True)
        
        try:
            filled_dir_path = get_config_item('TERRAIN', 'FILLED_DIR')
            self.filled_folder_name = UPath(filled_dir_path).name.lower()
        except Exception:
            logger.warning("Could not load TERRAIN/FILLED_DIR from config. Falling back to default 'filled_tifs'.")
            self.filled_folder_name = "filled_tifs"

        self.subgrid_paths = {
            'training': _resolve_path(get_config_item('MODEL', 'TRAINING_SUB_GRIDS'), is_output=True)
        }

    def _create_binary_mask_raster(self, cutline_path: str, bounds: tuple, output_path: str) -> tuple:
        """Creates a global binary raster mask from a vector layer before running Dask processes"""
        
        logger.info(f"Burning vector mask into binary raster: {output_path}")
        gdf = gpd.read_file(cutline_path)
        
        minx, miny, maxx, maxy = bounds
        res = self.target_res
        
        # Snap bounds to target resolution grid
        minx = np.floor(minx / res) * res
        maxy = np.ceil(maxy / res) * res
        maxx = np.ceil(maxx / res) * res
        miny = np.floor(miny / res) * res
        
        width = int((maxx - minx) / res)
        height = int((maxy - miny) / res)
        
        # Generate the affine transform
        transform = from_origin(minx, maxy, res, res)
        
        # Rasterize shapes (burn value 1)
        shapes_gen = ((geom, 1) for geom in gdf.geometry)
        mask_arr = rasterize(
            shapes=shapes_gen,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )
        
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'uint8',
            'crs': self.target_crs,
            'transform': transform,
            'compress': 'lzw',
            'tiled': True,
            'nodata': 0
        }
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask_arr, 1)
            
        return (output_path,)

    def process_training_raster(self, raster_path: str, mask_bounds: tuple, output_path: str, global_mask_path: str, current_index: int = None, total_count: int = None) -> None:
        """Process a training raster by extracting array blocks and masking them mathematically"""
        
        raster_name = pathlib.Path(raster_path).name.lower()
        open_path = str(raster_path)
        
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        
        if self.is_aws and open_path.startswith('s3://'):
            open_path = open_path.replace('s3://', '/vsis3/')

        logger.info(f"-> [STARTING]{progress_str} Worker executing training array mask on: {raster_name}")

        with tempfile.TemporaryDirectory(dir=self.local_tmp_dir) as task_tmp_dir:
            try:
                with rasterio.open(open_path) as src_pred:
                    src_nodata = src_pred.nodata if src_pred.nodata is not None else np.nan
                    
                    # Check bounding box intersections quickly using safe bounds
                    rb = src_pred.bounds
                    raster_bounds_geom = box(min(rb[0], rb[2]), min(rb[1], rb[3]), max(rb[0], rb[2]), max(rb[1], rb[3]))
                    mask_box = box(*mask_bounds)
                    
                    if not mask_box.intersects(raster_bounds_geom):
                        logger.info(f"- [SKIP]{progress_str} Bounding box does not intersect raster {raster_name}. Skipping.")
                        return
                    
                    meta = src_pred.meta.copy()
                    meta.update({
                        'nodata': np.nan if np.isnan(src_nodata) else src_nodata,
                        'compress': 'lzw',
                        'tiled': True
                    })

                    # Setup temporary local path inside quarantined task directory
                    tmp_dst_path = str(output_path)
                    if self.is_aws:
                        tmp_dst_path = str(Path(task_tmp_dir) / "train_mask_tmp.tif")

                    with rasterio.open(global_mask_path) as src_mask:
                        # Virtual re-alignment ensures the mask array is perfectly registered 
                        # to the incoming prediction raster (even if it was cropped/offset slightly)
                        with WarpedVRT(src_mask, crs=src_pred.crs, transform=src_pred.transform, 
                                       height=src_pred.height, width=src_pred.width, 
                                       resampling=Resampling.nearest) as vrt_mask:
                            
                            with rasterio.Env(CHECK_DISK_FREE_SPACE="FALSE"):
                                with rasterio.open(tmp_dst_path, 'w', **meta) as dest:
                                    
                                    # Evaluate the arrays safely in memory chunks to prevent Dask limits from being exceeded
                                    for ji, window in src_pred.block_windows(1):
                                        pred_arr = src_pred.read(1, window=window)
                                        mask_arr = vrt_mask.read(1, window=window)

                                        # Cast integers to floats if the nodata value is NaN
                                        if np.isnan(meta['nodata']) and pred_arr.dtype not in (np.float32, np.float64):
                                            pred_arr = pred_arr.astype(np.float32)

                                        # Apply mask logic via numpy 
                                        masked_data = np.where(mask_arr == 1, pred_arr, meta['nodata'])
                                        dest.write(masked_data, 1, window=window)
                    
                    # If on AWS, push complete file from fast local disk to S3 bucket
                    if self.is_aws:
                        import s3fs
                        fs = s3fs.S3FileSystem()
                        fs.put(tmp_dst_path, str(output_path))

                logger.info(f" - [✓ SUCCESS]{progress_str} Processed training raster via array masking: {raster_name}")
                
            except Exception as e:
                logger.exception(f"Unexpected failure during array masking for {raster_name}.")

    def run(self, mask_train_bounds: tuple, train_cutline_path: str, max_concurrent: int = 10) -> None:
        """Main entry point for generating binary masks and processing training rasters in parallel"""
        
        # Use param_lookup and the base Engine class to initialize Dask
        env_val = self.param_lookup.get('env', 'local')
        env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        
        self.setup_dask(env)
        
        # Pull the client configured by setup_dask, or fallback to connecting to the default cluster
        client = getattr(self, 'client', None)
        if not client:
            client = Client()
        
        global_mask_path = str(self.local_tmp_dir / "global_train_mask.tif")
        self._create_binary_mask_raster(train_cutline_path, mask_train_bounds, global_mask_path)

        # Case-insensitive recursive glob for input files
        potential_train_inputs = []
        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            potential_train_inputs.extend(list(self.prediction_out_dir.rglob(ext)))

        training_files = []
        removed_existing_train = 0
        
        # Case-insensitive recursive glob for existing outputs
        existing_train_outputs = set()
        for ext in ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]:
            existing_train_outputs.update({f.name for f in self.training_out_dir.rglob(ext)})
        
        for f in potential_train_inputs:
            name_lower = f.name.lower()
            parts_lower = [p.lower() for p in f.parts]

            # Ensure filled lidar directory is excluded for training 
            if self.filled_folder_name in parts_lower or 'filled_lidar' in parts_lower or 'filled_tifs' in parts_lower:
                continue
            
            # Ensure we use combined lidar rather than filled lidar for mosaics
            if 'mosaic' in name_lower and 'combined' not in name_lower and 'combined_lidar' not in parts_lower:
                continue
            
            # Exclude BlueTopo files strictly from being processed into training datasets
            # EXCEPTION: Keep it if it is the survey_end_date layer
            if ('bluetopo' in name_lower or name_lower.startswith('bt.')) and 'survey_end_date' not in name_lower:
                continue
                
            if not self.overwrite and f.name in existing_train_outputs:
                removed_existing_train += 1
                continue
                
            training_files.append(f)

        skip_train_msg = f" (Skipping {removed_existing_train} existing)" if not self.overwrite else " (Overwrite enabled)"
        
        logger.info(f"Outputting training rasters to: {self.training_out_dir}")
        logger.info(f"Queuing {len(training_files)} training files{skip_train_msg}...")

        # -------------------------------------------------------------
        # DYNAMIC DASK TASK STREAM (TRAINING)
        # Prevents idle workers by instantly replacing completed tasks
        # -------------------------------------------------------------
        total_train = len(training_files)
        training_iterator = iter(enumerate(training_files))
        seq_train = as_completed()
        
        def submit_train_task(item):
            i, file_path = item
            output_path = self.training_out_dir / file_path.name
            return client.submit(
                self.process_training_raster,
                str(file_path), 
                mask_train_bounds, 
                str(output_path),
                global_mask_path,
                current_index=i + 1,
                total_count=total_train
            )

        # Initial queue fill
        for _ in range(min(max_concurrent, total_train)):
            try:
                seq_train.add(submit_train_task(next(training_iterator)))
            except StopIteration:
                break

        # Process stream
        for future in seq_train:
            future.result() 
            try:
                seq_train.add(submit_train_task(next(training_iterator)))
            except StopIteration:
                pass

        if total_train > 0:
            logger.info("[SUCCESS] Training raster processing complete.")
        else:
            logger.info("No new training rasters to process.")
            
        # Cleanly shut down
        try:
            client.close()
            self.close_dask()
        except Exception as e:
            logger.error(f"Could not cleanly close client/cluster: {e}")