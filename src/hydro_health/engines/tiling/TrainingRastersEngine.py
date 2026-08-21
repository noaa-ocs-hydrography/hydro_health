import os
import gc
import tempfile
import pathlib
import shutil
import s3fs
import numpy as np
import rasterio

from pathlib import Path
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from shapely.geometry import box
from dask.distributed import Client
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


def _process_training_raster(params: list) -> None:
    """Process a training raster by extracting array blocks and masking them mathematically"""

    raster_path, mask_bounds, output_path, global_mask_path, current_index, total_count, param_lookup, output_prefix = params

    engine = TrainingRastersEngine(param_lookup, output_prefix)

    raster_name = pathlib.Path(raster_path).name.lower()
    open_path = str(raster_path)

    progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""

    if engine.is_aws and open_path.startswith('s3://'):
        open_path = open_path.replace('s3://', '/vsis3/')

    engine.write_message(f"-> [STARTING]{progress_str} Worker executing training array mask on: {raster_name}", OUTPUTS)

    tmp_dst_path = str(output_path)

    try:
        with tempfile.TemporaryDirectory(dir=engine.local_tmp_dir) as task_tmp_dir:
            try:
                with rasterio.open(open_path) as src_pred:
                    src_nodata = src_pred.nodata if src_pred.nodata is not None else np.nan

                    rb = src_pred.bounds
                    raster_bounds_geom = box(min(rb[0], rb[2]), min(rb[1], rb[3]), max(rb[0], rb[2]), max(rb[1], rb[3]))
                    mask_box = box(*mask_bounds)

                    if not mask_box.intersects(raster_bounds_geom):
                        engine.write_message(f"- [SKIP]{progress_str} Bounding box does not intersect raster {raster_name}. Skipping.", OUTPUTS)
                        return

                    meta = src_pred.meta.copy()
                    meta.update({
                        'nodata': np.nan if np.isnan(src_nodata) else src_nodata,
                        'compress': 'lzw',
                        'tiled': True
                    })

                    if engine.is_aws:
                        tmp_dst_path = str(Path(task_tmp_dir) / "train_mask_tmp.tif")

                    open_mask_path = str(global_mask_path)
                    if engine.is_aws and open_mask_path.startswith('s3://'):
                        open_mask_path = open_mask_path.replace('s3://', '/vsis3/')

                    with rasterio.open(open_mask_path) as src_mask:
                        mask_nodata = src_mask.nodata if src_mask.nodata is not None else 255
                        with WarpedVRT(
                            src_mask,
                            crs=src_pred.crs,
                            transform=src_pred.transform,
                            height=src_pred.height,
                            width=src_pred.width,
                            resampling=Resampling.nearest,
                            src_nodata=mask_nodata,
                            nodata=mask_nodata,
                        ) as vrt_mask:
                            with rasterio.Env(CHECK_DISK_FREE_SPACE="FALSE"):
                                with rasterio.open(tmp_dst_path, 'w', **meta) as dest:

                                    for ji, window in src_pred.block_windows(1):
                                        pred_arr = src_pred.read(1, window=window)
                                        mask_arr = vrt_mask.read(1, window=window)

                                        if np.isnan(meta['nodata']) and pred_arr.dtype not in (np.float32, np.float64):
                                            pred_arr = pred_arr.astype(np.float32)

                                        masked_data = np.where(mask_arr == 2, pred_arr, meta['nodata'])
                                        dest.write(masked_data, 1, window=window)

                    if engine.is_aws:
                        fs = s3fs.S3FileSystem()
                        fs.put(tmp_dst_path, str(output_path))

                engine.write_message(f" - [✓ SUCCESS]{progress_str} Processed training raster via array masking: {raster_name}", OUTPUTS)
                engine.write_message(engine.log_system_metrics(), OUTPUTS)

            except Exception as e:
                engine.write_message(f"Unexpected failure during array masking for {raster_name}: {e}", OUTPUTS)

            finally:
                if tmp_dst_path != str(output_path) and Path(tmp_dst_path).exists():
                    try:
                        os.remove(tmp_dst_path)
                    except Exception as e:
                        engine.write_message(f"Failed to explicitly delete temp file {tmp_dst_path}: {e}", OUTPUTS)
    finally:
        gc.collect()


class TrainingRastersEngine(Engine):
    """Class for parallel processing training rasters and applying mathematical masks"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize paths, configurations, and environment for training rasters"""

        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
                
        self.is_aws = param_lookup['env'] in ['remote', 'aws']  

        self.inputs_dir = INPUTS
        self.outputs_dir = OUTPUTS / output_prefix if output_prefix else OUTPUTS

        bucket = get_config_item('S3', 'BUCKET_NAME')
        mask_training_path = get_config_item('MASK', 'MASK_TRAINING_PATH')
        self.train_mask_path = UPath(f"s3://{bucket}/{mask_training_path}") if self.is_aws else UPath(self.outputs_dir / mask_training_path)
        prediction_output_dir = get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR')
        self.prediction_out_dir = UPath(f"s3://{bucket}/{prediction_output_dir}") if self.is_aws else UPath(self.outputs_dir / prediction_output_dir)
        training_out_dir = get_config_item('MODEL', 'TRAINING_OUTPUT_DIR')
        self.training_out_dir = UPath(f"s3://{bucket}/{training_out_dir}") if self.is_aws else UPath(self.outputs_dir / training_out_dir)
        self.training_out_dir.mkdir(parents=True, exist_ok=True)
        
        self.filled_folder_name = UPath(get_config_item('TERRAIN', 'FILLED_DIR')).name.lower()
        self.filled_folder_name = "filled_tifs"

        training_subgrid_path = get_config_item('MODEL', 'TRAINING_SUB_GRIDS')
        self.subgrid_paths = {
            'training': UPath(f"s3://{bucket}/{training_subgrid_path}") if self.is_aws else UPath(self.outputs_dir / training_subgrid_path)
        }

    def log_system_metrics(self) -> str:
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

    def prepare_mask_bounds(self) -> tuple:
        """Extract spatial bounds directly from training mask TIFF."""

        global_mask_path = str(self.train_mask_path)
        open_mask_path = global_mask_path
        
        if self.is_aws and open_mask_path.startswith('s3://'):
            open_mask_path = open_mask_path.replace('s3://', '/vsis3/')

        self.write_message(f"Extracting spatial bounds directly from training mask TIFF: {open_mask_path}", OUTPUTS)
        with rasterio.open(open_mask_path) as src:
            mask_train_bounds = src.bounds
            
        return global_mask_path, mask_train_bounds

    def collect_training_files(self) -> list:
        """Scan directories and collect target raster paths ready for processing."""

        potential_train_inputs = []
        for ext in ["*.tif", "*.tiff"]:
            potential_train_inputs.extend(list(self.prediction_out_dir.rglob(ext)))

        training_files = []
        removed_existing_train = 0
        
        existing_train_outputs = set()
        for ext in ["*.tif", "*.tiff"]:
            existing_train_outputs.update({f.name for f in self.training_out_dir.rglob(ext)})
        
        for f in potential_train_inputs:
            name_lower = f.name.lower()
            parts_lower = [p.lower() for p in f.parts]

            if self.filled_folder_name in parts_lower or 'filled_lidar' in parts_lower or 'filled_tifs' in parts_lower:
                continue
            
            if 'mosaic' in name_lower and 'combined' not in name_lower and 'combined_lidar' not in parts_lower:
                continue
                
            if f.name in existing_train_outputs:
                removed_existing_train += 1
                continue
                
            training_files.append(f)

        skip_train_msg = f" (Skipping {removed_existing_train} existing)" if removed_existing_train > 0 else ""
        self.write_message(f"Queuing {len(training_files)} training files{skip_train_msg}...", OUTPUTS)
        
        return training_files

    def execute_tasks(self, training_files: list, mask_train_bounds: tuple, global_mask_path: str):
        """Schedule and execute training tasks directly via dask map."""

        total_train = len(training_files)
        
        params = [
            [
                str(file_path),
                mask_train_bounds,
                str(self.training_out_dir / file_path.name),
                global_mask_path,
                i + 1,
                total_train,
                self.param_lookup,
                self.output_prefix
            ]
            for i, file_path in enumerate(training_files)
        ]

        futures = self.client.map(_process_training_raster, params)
        self.client.gather(futures)

        self.write_message("[SUCCESS] Training raster processing complete.", OUTPUTS)
        self.write_message(self.log_system_metrics(), OUTPUTS)

    def cleanup_resources(self):
        """Wipe temp disks and safely teardown parallel execution pools."""

        self.close_dask()

        if hasattr(self, 'local_tmp_dir') and self.local_tmp_dir.exists():
            try:
                shutil.rmtree(self.local_tmp_dir)
                self.write_message("Successfully wiped master local temp directory.", OUTPUTS)
            except Exception as e:
                self.write_message(f"Failed to wipe master local temp directory: {e}", OUTPUTS)

    def run(self) -> None:
        """Main entry point for evaluating training masks and processing rasters in parallel"""

        try:
            self.setup_dask(self.param_lookup['env'])
            global_mask_path, mask_train_bounds = self.prepare_mask_bounds()
            training_files = self.collect_training_files()

            if training_files:
                self.write_message(f"Outputting training rasters to: {self.training_out_dir}", OUTPUTS)
                self.execute_tasks(training_files, mask_train_bounds, global_mask_path)
            else:
                self.write_message("No new training rasters to process.", OUTPUTS)

        finally:
            self.cleanup_resources()
