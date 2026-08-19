"""Class engine for subtiling raster data into geoparquet files"""

import re
import os
import gc
import shutil
import logging
from pathlib import Path
import pathlib
import tempfile

import dask
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from upath import UPath
from dask.distributed import as_completed, Client

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

logger = logging.getLogger(__name__)

class SubgridTilingEngine(Engine):
    """Class for subtiling raster data into geoparquet files"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize the SubgridTilingEngine"""
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
            
        env = _get_val('env', 'local')
        self.is_aws = env in ['remote', 'aws']
        self.overwrite = _get_val('overwrite', False)

        # Extended static patterns to ensure ungridded data like 'grain', 'survey', and 'sed' are captured 
        self.static_patterns = ['sed', 'tsm', 'hurr', 'grain', 'survey']
        
        logger.info(f"Environment detected: {'AWS/Remote' if self.is_aws else 'Local'}")

        # ---------------------------------------------------------
        # Dynamically determine Repo Root and base folders
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

        self.prediction_out_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'), is_output=True)
        self.training_out_dir = _resolve_path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'), is_output=True)
        self.training_tiles_dir = _resolve_path(get_config_item('MODEL', 'TRAINING_TILES_DIR'), is_output=True)
        self.prediction_tiles_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_TILES_DIR'), is_output=True)
        
        # Establish master local temp dir for processing memory mitigation
        self.local_tmp_dir = Path(tempfile.gettempdir()) / "subgrid_tiling_tmp"
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure the local output directories exist before workers try to write to them
        if not self.is_aws:
            pathlib.Path(self.prediction_out_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.training_out_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.training_tiles_dir).mkdir(parents=True, exist_ok=True)
            pathlib.Path(self.prediction_tiles_dir).mkdir(parents=True, exist_ok=True)

        # Dynamically retrieve the filled terrain directory from config to ensure accurate exclusion
        try:
            filled_dir_path = get_config_item('TERRAIN', 'FILLED_DIR')
            self.filled_folder_name = UPath(filled_dir_path).name.lower()
        except Exception:
            logger.warning("Could not load TERRAIN/FILLED_DIR from config. Falling back to default 'filled_tifs'.")
            self.filled_folder_name = "filled_tifs"

        self.subgrid_paths = {
            'training': _resolve_path(get_config_item('MODEL', 'TRAINING_SUB_GRIDS'), is_output=True),
            'prediction': _resolve_path(get_config_item('MODEL', 'PREDICTION_SUB_GRIDS'), is_output=True)
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

    def _cleanup_resources(self):
        """Wipe the master local temp directory to ensure EC2 disk is completely cleared"""
        if hasattr(self, 'local_tmp_dir') and self.local_tmp_dir.exists():
            try:
                shutil.rmtree(self.local_tmp_dir)
                logger.info(f"Cleaned up local temporary directory: {self.local_tmp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp dir {self.local_tmp_dir}: {e}")

    def _process_pipeline(self, raster_dir: UPath, output_dir: UPath, data_type: str) -> None:
        """Orchestrates the tile processing for a specific data type."""
        logger.info(f"--- Starting {data_type.upper()} pipeline ---")
        logger.info(self._log_system_metrics())
        
        sub_grids = self._load_subgrids(data_type)
        if sub_grids is None or sub_grids.empty:
            return
            
        all_raster_files = self._get_filtered_raster_files(raster_dir, data_type)
        gridded_files, ungridded_files = self._partition_raster_files(all_raster_files, sub_grids)
        all_tasks, total_to_write = self._build_tile_tasks(sub_grids, output_dir, data_type)
        
        results_list = self._execute_tile_tasks(
            all_tasks=all_tasks,
            gridded_files=gridded_files,
            ungridded_files=ungridded_files,
            data_type=data_type,
            total_to_write=total_to_write
        )
        
        self._save_statistics(results_list, output_dir, data_type)
        logger.info(self._log_system_metrics())
        logger.info(f"Finished processing {data_type} rasters by tile.")

    def _load_subgrids(self, data_type: str) -> gpd.GeoDataFrame:
        """Loads the subgrids definition for the given data type."""
        sub_grid_path = self.subgrid_paths.get(data_type)
        if not sub_grid_path:
            logger.error(f"No subgrid path defined for {data_type}")
            return None
        
        logger.info(f"Loading subgrids from: {sub_grid_path}")
        try:
             sub_grids = gpd.read_file(str(sub_grid_path))
             logger.info(f"Successfully loaded {sub_grids.shape[0]} subgrids.")
             return sub_grids
        except Exception:
             logger.exception(f"Reading subgrids from {sub_grid_path} failed.")
             return None

    def _get_filtered_raster_files(self, raster_dir: UPath, data_type: str) -> list:
        """Scans and filters raster files based on type rules."""
        logger.info("Scanning directory for raster files...")
        raster_dir_upath = UPath(raster_dir)
        all_raster_files = []
        
        for f in raster_dir_upath.rglob("*"):
            if f.suffix.lower() in {'.tif', '.tiff'}:
                name_lower = f.name.lower()
                parts_lower = [p.lower() for p in f.parts]
                
                # Double-check exclusion of filled lidar directory
                if self.filled_folder_name in parts_lower or "filled_lidar" in parts_lower or "filled_tifs" in parts_lower:
                    continue
                
                # Exclude specific files globally
                if "unc" in name_lower:
                    continue
                    
                if "tsm_cumulative" in name_lower or \
                   "hurr_count_mean" in name_lower or \
                   "hurr_count_cumulative" in name_lower or \
                   "hurr_strength_cumulative" in name_lower or \
                   re.search(r"hurr_count_\d{4}_\d{4}", name_lower) or \
                   re.search(r"hurr_strength_\d{4}_\d{4}", name_lower):
                    continue
                    
                if data_type == 'training':
                    # Exclude BlueTopo files strictly from being processed into training datasets (except survey_end_date)
                    if ("bluetopo" in name_lower or name_lower.startswith("bt.")) and "survey_end_date" not in name_lower:
                        continue
                elif data_type == 'prediction':
                    # Prediction parquet uses the bluetopo files and not the filled lidar bathy
                    if "bathy" in name_lower and "bluetopo" not in name_lower and not name_lower.startswith("bt."):
                        continue
                        
                all_raster_files.append(str(f))
                
        logger.info(f"Found {len(all_raster_files)} valid raster files after filtering.")
        return all_raster_files

    def _partition_raster_files(self, all_raster_files: list, sub_grids: gpd.GeoDataFrame) -> tuple:
        """Pre-partition files to prevent S3 API connection exhaustion"""
        valid_tids = [str(tid) for tid in sub_grids['original_tile'].unique() if pd.notna(tid) and str(tid).strip()]
        valid_tid_patterns = [re.compile(rf"(?:^|_){tid}(?:_|\.)") for tid in valid_tids]
        
        gridded_files = []
        ungridded_files = []
        for f in all_raster_files:
            fname = Path(f).name
            if any(p.search(fname) for p in valid_tid_patterns):
                gridded_files.append(f)
            else:
                ungridded_files.append(f)
                
        logger.info(f" -> Pre-partitioned into {len(gridded_files)} gridded files (tiled) and {len(ungridded_files)} ungridded files (global).")
        return gridded_files, ungridded_files

    def _build_tile_tasks(self, sub_grids: gpd.GeoDataFrame, output_dir: UPath, data_type: str) -> tuple:
        """Determines which tiles need to be built and prepares task dictionaries."""
        all_tasks = []
        write_counter = 0
        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id']
            output_folder = output_dir / tile_name
            expected_final_path = output_folder / f"{tile_name}_{data_type}_formatted.parquet"
            
            should_write = self.overwrite or not expected_final_path.exists()
            if should_write:
                write_counter += 1
                all_tasks.append({
                    'sub_grid': sub_grid,
                    'output_folder': output_folder,
                    'tile_name': tile_name,
                    'should_write': True,
                    'write_index': write_counter
                })
            else:
                all_tasks.append({
                    'sub_grid': sub_grid,
                    'output_folder': output_folder,
                    'tile_name': tile_name,
                    'should_write': False,
                    'expected_output_path': expected_final_path
                })
                
        logger.info(f"--- Subtiling Parquet Summary ({data_type}) ---")
        logger.info(f" -> Total subgrid tiles: {len(sub_grids)}")
        logger.info(f" -> Tiles needing Wide Parquet generation: {write_counter}")
        logger.info(f" -> Existing tiles (skipped): {len(sub_grids) - write_counter}")
        return all_tasks, write_counter

    def _execute_tile_tasks(self, all_tasks: list, gridded_files: list, ungridded_files: list, data_type: str, total_to_write: int) -> list:
        """Executes the Dask computation graph for all queued tasks."""
        client = getattr(self, 'client', None)
        if not client:
            client = Client()
            
        max_concurrent = 25 
        logger.info(f"Building dynamic Dask task stream (max {max_concurrent} concurrent)...")
        
        sub_grid_iterator = iter(all_tasks)
        total_grids = len(all_tasks)
        seq = as_completed()
        results_list = []
        
        def submit_next_tile(task_item):
            tile_name = task_item['tile_name']
            if not task_item['should_write']:
                expected_output_path = task_item['expected_output_path']
                logger.info(f" [SKIP] Tile already processed: {tile_name}. Queuing stats generation only.")
                stats_task = dask.delayed(SubgridTilingEngine._generate_stats_from_existing)(str(expected_output_path), tile_name)
                return client.compute(stats_task)
            else:
                sub_grid = task_item['sub_grid']
                output_folder = task_item['output_folder']
                write_idx = task_item['write_index']
                
                gridded_task = dask.delayed(SubgridTilingEngine.subtile_process_gridded)(
                    sub_grid, gridded_files, self.is_aws
                )
                combined_task = dask.delayed(SubgridTilingEngine.subtile_process_ungridded)(
                    sub_grid, ungridded_files, gridded_task, self.static_patterns, self.is_aws
                )
                save_task = dask.delayed(SubgridTilingEngine.save_combined_data)(
                    combined_task, 
                    output_folder, 
                    data_type, 
                    tile_id=tile_name,
                    is_aws=self.is_aws,
                    local_tmp_dir=str(self.local_tmp_dir),
                    current_index=write_idx,
                    total_count=total_to_write
                )
                return client.compute(save_task)

        # Initial queue fill
        for _ in range(min(max_concurrent, total_grids)):
            try:
                seq.add(submit_next_tile(next(sub_grid_iterator)))
            except StopIteration:
                break
                
        # Process stream
        for future in seq:
            stats_df, cols_list, t_id = future.result()
            
            print(f"\n{'='*75}\n✅ [TILE COMPLETED: {t_id}]\nPARQUET FILE CONTAINS THE FOLLOWING {len(cols_list)} COLUMNS:\n{', '.join(cols_list)}\n{'='*75}\n")
            
            results_list.append(stats_df)
            
            # Force memory release back to the OS after every single file
            gc.collect()
            
            try:
                seq.add(submit_next_tile(next(sub_grid_iterator)))
            except StopIteration:
                pass

        logger.info("Dask computation across dynamic task stream finished successfully.")
        return results_list

    def _save_statistics(self, results_list: list, output_dir: UPath, data_type: str) -> None:
        """Combines the result stats and saves to a CSV."""
        if not results_list:
            return
            
        logger.info(f"Concatenating {len(results_list)} tile result dataframes for stats...")
        final_results_df = pd.concat(results_list, ignore_index=True)
        
        output_csv_path = output_dir.parent / f"year_pair_nan_counts_{data_type}.csv"
        final_results_df.to_csv(str(output_csv_path), index=False, na_rep='NA')
        logger.info(f"[SUCCESS] Statistics successfully saved to: {output_csv_path}")

    def run(self) -> None:
        """Main entry point for executing the clipping pipelines"""
        
        # Use param_lookup and the base Engine class to initialize Dask
        env_val = self.param_lookup.get('env', 'local')
        env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        
        self.setup_dask(env)

        try:
            # Process Prediction Data
            self._process_pipeline(
                raster_dir=self.prediction_out_dir, 
                output_dir=self.prediction_tiles_dir, 
                data_type="prediction"
            )

            # Process Training Data
            self._process_pipeline(
                raster_dir=self.training_out_dir, 
                output_dir=self.training_tiles_dir, 
                data_type="training"
            )
        finally:
            self._cleanup_resources()
            try:
                self.close_dask()
            except Exception as e:
                logger.error(f"Could not cleanly close client/cluster: {e}")

    @staticmethod
    def _generate_stats_from_existing(filepath: str, tile_id: str) -> tuple:
        """Reads an existing parquet file to generate nan stats without reprocessing. Returns tuple for main thread unpacking."""
        try:
            df = pd.read_parquet(filepath)
            cols = df.columns.tolist()
            
            # Memory cleanup in the worker
            stats = SubgridTilingEngine.create_nan_stats_csv(df, tile_id)
            del df
            gc.collect()
            
            return stats, cols, tile_id
        except Exception:
            logger.exception(f"Failed to read existing tile {filepath} for stats.")
            return pd.DataFrame(), [], tile_id

    @staticmethod
    def subtile_process_gridded(sub_grid, raster_files, is_aws: bool) -> pd.DataFrame:
        """Process gridded rasters for a single tile dynamically and avoid sequential merging."""
        original_tile = str(sub_grid['original_tile'])
        pattern = re.compile(rf"(?:^|_){original_tile}(?:_|\.)")
        
        filtered_files = [f for f in raster_files if pattern.search(Path(f).name)]
        
        if not filtered_files:
            logger.warning(f"⚠️ MISSING GRIDDED DATA: No tile-specific raster files found for tile '{original_tile}'.")
            return pd.DataFrame()

        tile_extent = sub_grid.geometry.bounds
        data_arrays = {}
        common_window = None
        common_transform = None
        
        # Read all aligned band arrays in a single open/read pass
        for file in filtered_files:
            open_path = str(file)
            if is_aws and open_path.startswith("s3://"):
                open_path = open_path.replace("s3://", "/vsis3/")
                
            try:
                with rasterio.open(open_path) as src:
                    if common_window is None:
                        common_window = src.window(*tile_extent)
                        common_transform = src.window_transform(common_window)
                    
                    data = src.read(1, window=common_window, boundless=True, fill_value=src.nodata)
                    col_name = Path(file).stem
                    col_name = SubgridTilingEngine._standardize_col_name(col_name, original_tile)
                    data_arrays[col_name] = (data, src.nodata)
            except Exception as e:
                logger.warning(f"Error reading gridded file {file}: {e}")
                
        if not data_arrays:
            return pd.DataFrame()
            
        master_mask = None
        for col_name, (data, nodata) in data_arrays.items():
            if nodata is not None and not np.isnan(nodata):
                mask = data != nodata
            else:
                mask = ~np.isnan(data)
            if master_mask is None:
                master_mask = mask.copy()
            else:
                master_mask |= mask
                
        if master_mask is None or not master_mask.any():
            logger.warning(f"⚠️ MISSING VALID DATA: Gridded files exist for '{original_tile}', but all pixels are NoData/NaN.")
            return pd.DataFrame()
            
        rows, cols = np.where(master_mask)
        xs, ys = rasterio.transform.xy(common_transform, rows, cols, offset='center')
        
        df_dict = {
            'X': np.round(xs, 3),
            'Y': np.round(ys, 3)
        }
        
        for col_name, (data, nodata) in data_arrays.items():
            vals = data[master_mask].astype(np.float32)
            if nodata is not None and not np.isnan(nodata):
                vals[vals == nodata] = np.nan
            df_dict[col_name] = vals
            
        combined_data = pd.DataFrame(df_dict)
        combined_data = combined_data.drop_duplicates(subset=['X', 'Y'])
        
        # Free memory in worker
        del data_arrays
        del master_mask
        gc.collect()
        
        return combined_data

    @staticmethod
    def subtile_process_ungridded(sub_grid, raster_files, gridded_df, static_patterns: list, is_aws: bool) -> pd.DataFrame:
        """Process ungridded rasters by translating spatial locations directly to pixel indices instead of merging."""
        if gridded_df is None or gridded_df.empty:
            logger.warning(f"⚠️ MISSING SPATIAL GRID: Cannot process ungridded data for tile '{sub_grid.get('original_tile', 'Unknown')}' because gridded dataframe is empty.")
            return pd.DataFrame()

        combined_df = gridded_df.copy()
        xs = combined_df['X'].values
        ys = combined_df['Y'].values
        tile_extent = sub_grid.geometry.bounds
        original_tile = str(sub_grid.get('original_tile', ''))

        for pattern in static_patterns:
            current_files = [f for f in raster_files if pattern in Path(f).name]
            if not current_files:
                logger.warning(f"⚠️ MISSING UNGRIDDED DATA: No global files found matching pattern '{pattern}' for tile '{original_tile}'.")

            for file in current_files:
                col_name = Path(file).stem
                col_name = SubgridTilingEngine._standardize_col_name(col_name, original_tile)
                
                open_path = str(file)
                if is_aws and open_path.startswith("s3://"):
                    open_path = open_path.replace("s3://", "/vsis3/")
                    
                try:
                    with rasterio.open(open_path) as src:
                        window = src.window(*tile_extent)
                        if window.width <= 0 or window.height <= 0:
                            if col_name not in combined_df:
                                combined_df[col_name] = np.full(len(xs), np.nan, dtype=np.float32)
                            continue

                        win_data = src.read(1, window=window)
                        win_transform = src.window_transform(window)
                        
                        win_rows, win_cols = rasterio.transform.rowcol(win_transform, xs, ys)
                        win_rows = np.array(win_rows)
                        win_cols = np.array(win_cols)
                        
                        win_valid = (win_rows >= 0) & (win_rows < win_data.shape[0]) & \
                                    (win_cols >= 0) & (win_cols < win_data.shape[1])
                        
                        vals = np.full(len(xs), np.nan, dtype=np.float32)
                        
                        if win_valid.any():
                            extracted_vals = win_data[win_rows[win_valid], win_cols[win_valid]].astype(np.float32)
                            if src.nodata is not None:
                                nodata_val = src.nodata
                                if not np.isnan(nodata_val):
                                    extracted_vals[extracted_vals == nodata_val] = np.nan
                            vals[win_valid] = extracted_vals
                            
                        if col_name in combined_df:
                            existing = combined_df[col_name].values
                            existing_nan = np.isnan(existing)
                            existing[existing_nan] = vals[existing_nan]
                            combined_df[col_name] = existing
                        else:
                            combined_df[col_name] = vals

                except Exception as e:
                    logger.warning(f"Failed to sample ungridded raster {file}: {e}")
                    if col_name not in combined_df:
                        combined_df[col_name] = np.full(len(xs), np.nan, dtype=np.float32)
        
        # Original df cleanup
        del gridded_df
        gc.collect()

        return combined_df

    @staticmethod
    def save_combined_data(combined_df, output_folder, data_type, tile_id, is_aws: bool, local_tmp_dir: str, current_index=None, total_count=None) -> tuple:
        """Combine dataframes, explicitly save to temporary disk, move to output, and aggressively drop memory."""
        if combined_df is None or combined_df.empty:
            logger.warning(f"⚠️ CRITICAL MISSING DATA: No data assembled for tile '{tile_id}'. Creating empty parquet file to register completion.")
            combined_df = pd.DataFrame(columns=['FID', 'tile_id', 'X', 'Y'])

        # Dynamically calculate delta_bathy for the wide/tall format before saving
        if data_type in ["training", "prediction"]:
            bathy_years = set()
            for c in combined_df.columns:
                if data_type == "training":
                    if c.startswith('bathy_'):
                        m = re.search(r'_(\d{4})(?:_filled)?$', c, re.IGNORECASE)
                        if m: bathy_years.add(int(m.group(1)))
                else:  # prediction
                    if c.startswith('bt.'):
                        m = re.search(r'\.(\d{4})$', c)
                        if m: bathy_years.add(int(m.group(1)))
                        
            sorted_years = sorted(list(bathy_years))
            dynamic_year_ranges = [(sorted_years[i], sorted_years[i+1]) for i in range(len(sorted_years)-1)]
            
            valid_pairs = []
            for y0, y1 in dynamic_year_ranges:
                y0_str, y1_str = str(y0), str(y1)
                
                if data_type == "training":
                    pattern_0 = re.compile(rf"^bathy_{y0_str}_filled$", re.IGNORECASE)
                    pattern_1 = re.compile(rf"^bathy_{y1_str}_filled$", re.IGNORECASE)
                else:
                    pattern_0 = re.compile(rf"^bt\.(?:bluetopo_)?{y0_str}$", re.IGNORECASE)
                    pattern_1 = re.compile(rf"^bt\.(?:bluetopo_)?{y1_str}$", re.IGNORECASE)
                
                c_0 = [c for c in combined_df.columns if pattern_0.match(c)]
                c_1 = [c for c in combined_df.columns if pattern_1.match(c)]
                
                if data_type == "training":
                    f_0 = [c for c in c_0 if "filled" in c.lower()]
                    f_1 = [c for c in c_1 if "filled" in c.lower()]
                    b_y0 = f_0[0] if f_0 else (c_0[0] if c_0 else None)
                    b_y1 = f_1[0] if f_1 else (c_1[0] if c_1 else None)
                else:
                    b_y0 = c_0[0] if c_0 else None
                    b_y1 = c_1[0] if c_1 else None
                
                if b_y0 and b_y1:
                    delta_name = f"delta_bathy_{y0_str}_{y1_str}"
                    combined_df[delta_name] = combined_df[b_y1] - combined_df[b_y0]
                    valid_pairs.append(f"{y0_str}_{y1_str}")
                else:
                    missing_parts = []
                    if not b_y0: missing_parts.append(f"bathy for {y0_str}")
                    if not b_y1: missing_parts.append(f"bathy for {y1_str}")
                    logger.warning(f"⚠️ MISSING BATHY DATA: Cannot calculate delta_bathy for {y0_str}_{y1_str} on tile '{tile_id}'. Missing: {', '.join(missing_parts)}")
            
            cols_to_drop = []
            for c in combined_df.columns:
                m = re.search(r"(\d{4}_\d{4})$", c)
                if m and not c.startswith("delta_bathy_"):
                    if m.group(1) not in valid_pairs:
                        cols_to_drop.append(c)
            if cols_to_drop:
                combined_df.drop(columns=cols_to_drop, inplace=True)

        if 'tile_id' not in combined_df.columns:
            combined_df['tile_id'] = tile_id
        if 'FID' not in combined_df.columns:
            combined_df.insert(0, 'FID', np.arange(len(combined_df)))

        # Handle writing to a local temp file first to protect memory / S3 writes
        output_folder_path = UPath(output_folder)
        if not is_aws: 
            output_folder_path.mkdir(parents=True, exist_ok=True)
            
        final_save_path = str(output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet")
        tmp_dst_path = str(Path(local_tmp_dir) / f"{tile_id}_{data_type}_clipped_data.parquet")

        combined_df.to_parquet(tmp_dst_path, engine="pyarrow", index=False)
        
        # Transfer the temp file over to the final path
        if is_aws and final_save_path.startswith("s3://"):
            import s3fs
            fs = s3fs.S3FileSystem()
            fs.put(tmp_dst_path, final_save_path)
        else:
            shutil.copy(tmp_dst_path, final_save_path)
        
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        logger.info(f"{progress_str} [SUCCESS] Saved combined tile data to: {final_save_path}")
        
        cols_list = combined_df.columns.tolist()
        cols_str = ", ".join(cols_list)
        logger.info(f"{progress_str}    -> CREATED PARQUET COLUMNS: {cols_str}")
        
        # Pre-calc statistics before dataframe destruction
        stats_df = SubgridTilingEngine.create_nan_stats_csv(combined_df, tile_id)
        
        # Drop references and explicitly invoke the GC to release RAM
        del combined_df
        gc.collect()

        # Explicitly delete the temp file immediately to free EC2 disk space
        if tmp_dst_path != final_save_path and Path(tmp_dst_path).exists():
            try:
                os.remove(tmp_dst_path)
            except Exception as e:
                logger.warning(f"Failed to explicitly delete temp file {tmp_dst_path}: {e}")

        return stats_df, cols_list, tile_id

    @staticmethod
    def create_nan_stats_csv(df: pd.DataFrame, tile_id: str) -> pd.DataFrame:
        """Calculates NaN stats for a tile."""
        if df.empty:
            return pd.DataFrame()
        new_row = {'tile_id': tile_id}
        
        change_cols = [c for c in df.columns if c.startswith('delta_bathy_')]
        for col in change_cols:
            year_pair = col.replace('delta_bathy_', '')
            new_row[f"{year_pair}_nan_percent"] = round(df[col].isna().mean() * 100, 2)
            
        return pd.DataFrame([new_row])

    @staticmethod
    def _standardize_col_name(col_name: str, original_tile: str = "") -> str:
        """Cleans raster filenames into consistent column names, standardizing years and prefixes."""
        clean_name = col_name
        
        if "survey_end_date" in clean_name.lower():
            return "survey_end_date"
        
        is_bluetopo = "bluetopo" in clean_name.lower() or clean_name.startswith("bt.")
        if is_bluetopo:
            clean_name = re.sub(r"(?i)^bluetopo_?", "", clean_name)
            if clean_name.startswith("bt."):
                clean_name = clean_name[3:]
                
        clean_name = re.sub(r"(?i)^combined\d*_", "", clean_name)
        
        if original_tile and original_tile in clean_name:
            clean_name = clean_name.replace(f"_{original_tile}", "").replace(f"{original_tile}_", "").replace(original_tile, "")
            
        clean_name = re.sub(r"(?<!\d)((?:19|20)\d{2})\d{4}(?!\d)", r"\1", clean_name)
        clean_name = clean_name.strip("_")
        
        m_pair = re.search(r"(\d{4}_\d{4})", clean_name)
        if m_pair:
            year_pair = m_pair.group(1)
            base = clean_name.replace(year_pair, "").strip("_")
            base = re.sub(r"__+", "_", base)
            if base:
                final_name = f"{base}_{year_pair}"
            else:
                final_name = year_pair
        else:
            m_single = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", clean_name)
            if m_single:
                year = m_single.group(1)
                base = clean_name.replace(year, "").strip("_")
                base = re.sub(r"__+", "_", base)
                
                base_lower = base.lower()
                if base_lower == "bathy" or base_lower == "bathy_filled":
                    final_name = f"bathy_{year}_filled"
                elif base_lower.startswith("bathy_"):
                    base = base[6:].strip("_")
                    final_name = f"{base}_{year}"
                elif base:
                    final_name = f"{base}_{year}"
                else:
                    final_name = year
            else:
                final_name = clean_name
                
        if is_bluetopo:
            return f"bt.{final_name}"
            
        return final_name
