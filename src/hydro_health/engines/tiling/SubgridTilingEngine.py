"""Class engine for subtiling raster data into geoparquet files"""

import re
import logging
from pathlib import Path
import pathlib

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
        # __file__ = src/hydro_health/engines/tiling/SubgridTilingEngine.py
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

        self.prediction_out_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'), is_output=True)
        self.training_out_dir = _resolve_path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'), is_output=True)
        self.training_tiles_dir = _resolve_path(get_config_item('MODEL', 'TRAINING_TILES_DIR'), is_output=True)
        self.prediction_tiles_dir = _resolve_path(get_config_item('MODEL', 'PREDICTION_TILES_DIR'), is_output=True)
        
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

    def clip_rasters_by_tile(self, raster_dir: UPath, output_dir: UPath, data_type: str) -> None:
        """Clip raster files by tile and save data in memory-managed batches"""
        
        logger.info(f"Clipping {data_type} rasters by tile...")
        
        sub_grid_path = self.subgrid_paths.get(data_type)
        if not sub_grid_path:
            logger.error(f"No subgrid path defined for {data_type}")
            return
        
        logger.info(f"Loading subgrids from: {sub_grid_path}")
        try:
             sub_grids = gpd.read_file(str(sub_grid_path))
             logger.info("Successfully loaded subgrids.")
        except Exception:
             logger.exception(f"Reading subgrids from {sub_grid_path} failed.")
             return
        
        logger.info(f"Number of tiles to process: {sub_grids.shape[0]}")
        logger.info(f"Raster directory: {raster_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        logger.info("Scanning directory for raster files... (This will only happen once)")
        raster_dir_upath = UPath(raster_dir)
        
        all_raster_files = []
        for f in raster_dir_upath.rglob("*"):
            if f.suffix.lower() in {'.tif', '.tiff'}:
                name_lower = f.name.lower()
                parts_lower = [p.lower() for p in f.parts]
                
                # Double-check exclusion of filled lidar directory in clipping step
                if self.filled_folder_name in parts_lower or "filled_lidar" in parts_lower or "filled_tifs" in parts_lower:
                    continue
                
                # Exclude 'unc', specific hurricane, and tsm_cumulative files globally for BOTH modes
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
                    # Exclude BlueTopo files strictly from being processed into training datasets
                    # EXCEPTION: Keep it if it is the survey_end_date layer
                    if ("bluetopo" in name_lower or name_lower.startswith("bt.")) and "survey_end_date" not in name_lower:
                        continue
                elif data_type == 'prediction':
                    # Prediction parquet uses the bluetopo files and not the filled lidar bathy
                    if "bathy" in name_lower and "bluetopo" not in name_lower and not name_lower.startswith("bt."):
                        continue
                        
                all_raster_files.append(str(f))
        
        logger.info(f"Filtered out TIFF files containing 'unc' and specific hurricane/tsm derivatives from the {data_type} parquet files.")
        if data_type == 'training':
            logger.info(" -> Specifically excluded BlueTopo files for training (except survey_end_date).")
        elif data_type == 'prediction':
            logger.info(" -> Specifically excluded standard Lidar (bathy) files for prediction (using BlueTopo instead).")

        logger.info(f"Found {len(all_raster_files)} raster files.")
        
        # Pre-partition files to prevent S3 API connection exhaustion
        # Separates files into strictly tiled (gridded) vs global mosaics (ungridded)
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

        # Parquet counting & pre-calculations
        all_tasks = []
        write_counter = 0
        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id']
            output_folder = output_dir / tile_name
            # Look for the final formatted file to determine if we need to process this tile
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
        
        total_to_write = write_counter
        logger.info(f"--- Subtiling Wide Parquet Summary ({data_type}) ---")
        logger.info(f" -> Total subgrid tiles: {len(sub_grids)}")
        logger.info(f" -> Tiles needing Wide Parquet generation: {total_to_write}")
        logger.info(f" -> Existing tiles (skipped): {len(sub_grids) - total_to_write}")
        logger.info("--------------------------------------------------")

        # Dynamic Dask task stream
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
                
                # We specifically pass the pre-partitioned list and explicit variables to avoid serializing self
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
            results_list.append(future.result())
            try:
                seq.add(submit_next_tile(next(sub_grid_iterator)))
            except StopIteration:
                pass

        logger.info("Dask computation across dynamic task stream finished successfully.")

        logger.info(f"Combining {data_type} tile results and calculating statistics...")
        logger.info(f"Concatenating {len(results_list)} tile result dataframes...")
        final_results_df = pd.concat(results_list, ignore_index=True)
        logger.info(f"Final combined dataframe shape: {final_results_df.shape}")
        
        output_csv_path = output_dir.parent / f"year_pair_nan_counts_{data_type}.csv"
        
        logger.info("Generating statistics and saving to CSV format...")
        final_results_df.to_csv(str(output_csv_path), index=False, na_rep='NA')
        logger.info(f"[SUCCESS] Statistics successfully saved to: {output_csv_path}")
        logger.info(f"Finished clipping {data_type} rasters by tile.")

    def run(self) -> None:
        """Main entry point for executing the clipping pipelines"""
        
        # Use param_lookup and the base Engine class to initialize Dask
        env_val = self.param_lookup.get('env', 'local')
        env = env_val.valueAsText if hasattr(env_val, 'valueAsText') and env_val.valueAsText else (env_val.value if hasattr(env_val, 'value') and env_val.value else env_val)
        
        self.setup_dask(env)

        try:
            self.clip_rasters_by_tile(
                raster_dir=self.prediction_out_dir, 
                output_dir=self.prediction_tiles_dir, 
                data_type="prediction"
            )

            self.clip_rasters_by_tile(
                raster_dir=self.training_out_dir, 
                output_dir=self.training_tiles_dir, 
                data_type="training"
            )
        finally:
            try:
                self.close_dask()
            except Exception as e:
                logger.error(f"Could not cleanly close client/cluster: {e}")

    @staticmethod
    def _generate_stats_from_existing(filepath: str, tile_id: str) -> pd.DataFrame:
        """Reads an existing parquet file to generate nan stats without reprocessing."""
        try:
            df = pd.read_parquet(filepath)
            return SubgridTilingEngine.create_nan_stats_csv(df, tile_id)
        except Exception:
            logger.exception(f"Failed to read existing tile {filepath} for stats.")
            return pd.DataFrame()

    @staticmethod
    def subtile_process_gridded(sub_grid, raster_files, is_aws: bool) -> pd.DataFrame:
        """Process gridded rasters for a single tile dynamically and avoid sequential merging."""
        original_tile = str(sub_grid['original_tile'])
        pattern = re.compile(rf"(?:^|_){original_tile}(?:_|\.)")
        
        filtered_files = [
            f for f in raster_files
            if pattern.search(Path(f).name)
        ]
        
        if not filtered_files:
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
                    
                    # Add boundless=True to pad dimensions when the window crosses the raster edges,
                    # ensuring that all array dimensions match exactly for the master_mask |= mask bitwise operator.
                    data = src.read(1, window=common_window, boundless=True, fill_value=src.nodata)
                    
                    col_name = Path(file).stem
                    col_name = SubgridTilingEngine._standardize_col_name(col_name, original_tile)
                    data_arrays[col_name] = (data, src.nodata)
            except Exception as e:
                logger.warning(f"Error reading gridded file {file}: {e}")
                
        if not data_arrays:
            return pd.DataFrame()
            
        # Create a unified master mask across all bands
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
            return pd.DataFrame()
            
        # Compute spatial coordinates only ONCE for the whole tile
        rows, cols = np.where(master_mask)
        xs, ys = rasterio.transform.xy(common_transform, rows, cols, offset='center')
        
        # Instantiate the DataFrame in one go to bypass intermediate allocation
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
        return combined_data

    @staticmethod
    def subtile_process_ungridded(sub_grid, raster_files, gridded_df, static_patterns: list, is_aws: bool) -> pd.DataFrame:
        """Process ungridded rasters by translating spatial locations directly to pixel indices instead of merging."""
        if gridded_df is None or gridded_df.empty:
            return pd.DataFrame()

        # Copy dataframe structure to insert matching ungridded bands directly
        combined_df = gridded_df.copy()
        
        xs = combined_df['X'].values
        ys = combined_df['Y'].values
        tile_extent = sub_grid.geometry.bounds
        original_tile = str(sub_grid.get('original_tile', ''))

        for pattern in static_patterns:
            current_files = [f for f in raster_files if pattern in Path(f).name]

            for file in current_files:
                col_name = Path(file).stem
                col_name = SubgridTilingEngine._standardize_col_name(col_name, original_tile)
                
                open_path = str(file)
                if is_aws and open_path.startswith("s3://"):
                    open_path = open_path.replace("s3://", "/vsis3/")
                    
                try:
                    with rasterio.open(open_path) as src:
                        window = src.window(*tile_extent)
                        
                        # Guard against non-intersecting / empty coordinate windows
                        if window.width <= 0 or window.height <= 0:
                            if col_name not in combined_df:
                                combined_df[col_name] = np.full(len(xs), np.nan, dtype=np.float32)
                            continue

                        win_data = src.read(1, window=window)
                        win_transform = src.window_transform(window)
                        
                        # Translate spatial coordinates directly into row-column positions on this band
                        win_rows, win_cols = rasterio.transform.rowcol(win_transform, xs, ys)
                        win_rows = np.array(win_rows)
                        win_cols = np.array(win_cols)
                        
                        # Check inside-image boundary conditions
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
                            
                        # Safely insert without destroying data from other tiles mapped to the same col_name
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

        return combined_df

    @staticmethod
    def save_combined_data(combined_df, output_folder, data_type, tile_id, is_aws: bool, current_index=None, total_count=None) -> pd.DataFrame:
        """Combine dataframes and save to parquet."""
        if combined_df is None or combined_df.empty:
            return pd.DataFrame()

        # Dynamically calculate delta_bathy for the wide/tall format before saving
        if data_type in ["training", "prediction"]:
            bathy_years = set()
            # Find all base bathy columns natively
            for c in combined_df.columns:
                if data_type == "training":
                    if c.startswith('bathy_'):
                        m = re.search(r'_(\d{4})(?:_filled)?$', c, re.IGNORECASE)
                        if m:
                            bathy_years.add(int(m.group(1)))
                else:  # prediction
                    if c.startswith('bt.'):
                        m = re.search(r'\.(\d{4})$', c)
                        if m:
                            bathy_years.add(int(m.group(1)))
                        
            sorted_years = sorted(list(bathy_years))
            # Create dynamic sequential pairs: e.g., (2004, 2006), (2006, 2010)
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
            
            # Drop year-pair variables that do not have a matching delta_bathy
            cols_to_drop = []
            for c in combined_df.columns:
                m = re.search(r"(\d{4}_\d{4})$", c)
                if m and not c.startswith("delta_bathy_"):
                    if m.group(1) not in valid_pairs:
                        cols_to_drop.append(c)
            if cols_to_drop:
                combined_df.drop(columns=cols_to_drop, inplace=True)

        # Ensure tile_id and FID are included
        if 'tile_id' not in combined_df.columns:
            combined_df['tile_id'] = tile_id
        if 'FID' not in combined_df.columns:
            combined_df.insert(0, 'FID', np.arange(len(combined_df)))

        output_folder_path = UPath(output_folder)
        
        if not is_aws: 
            output_folder_path.mkdir(parents=True, exist_ok=True)
            
        output_path = output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet"
        save_path = str(output_path)

        combined_df.to_parquet(save_path, engine="pyarrow", index=False)
        
        progress_str = f" [{current_index}/{total_count}]" if current_index and total_count else ""
        
        # Print columns directly to terminal the moment Dask finishes saving
        logger.info(f"{progress_str} [SUCCESS] Saved combined tile data to: {save_path}")
        
        cols_str = ", ".join(combined_df.columns.tolist())
        logger.info(f"{progress_str}    -> CREATED PARQUET COLUMNS: {cols_str}")
        
        return SubgridTilingEngine.create_nan_stats_csv(combined_df, tile_id)

    @staticmethod
    def create_nan_stats_csv(df: pd.DataFrame, tile_id: str) -> pd.DataFrame:
        """Calculates NaN stats for a tile."""
        if df.empty:
            return pd.DataFrame()
        new_row = {'tile_id': tile_id}
        
        # Now searches directly for the dynamically built delta columns!
        change_cols = [c for c in df.columns if c.startswith('delta_bathy_')]
        for col in change_cols:
            year_pair = col.replace('delta_bathy_', '')
            new_row[f"{year_pair}_nan_percent"] = round(df[col].isna().mean() * 100, 2)
            
        return pd.DataFrame([new_row])

    @staticmethod
    def _standardize_col_name(col_name: str, original_tile: str = "") -> str:
        """Cleans raster filenames into consistent column names, standardizing years and prefixes."""
        clean_name = col_name
        
        # Explicit override for survey_end_date to ensure it gets exactly this column name
        if "survey_end_date" in clean_name.lower():
            return "survey_end_date"
        
        # 1. Clean bluetopo prefix and tags
        is_bluetopo = "bluetopo" in clean_name.lower() or clean_name.startswith("bt.")
        if is_bluetopo:
            clean_name = re.sub(r"(?i)^bluetopo_?", "", clean_name)
            if clean_name.startswith("bt."):
                clean_name = clean_name[3:]
                
        # 2. Strip 'combined_' legacy prefixes
        clean_name = re.sub(r"(?i)^combined\d*_", "", clean_name)
        
        # 3. Strip original tile
        if original_tile and original_tile in clean_name:
            clean_name = clean_name.replace(f"_{original_tile}", "").replace(f"{original_tile}_", "").replace(original_tile, "")
            
        # 4. Truncate 8-digit YYYYMMDD dates to 4-digit YYYY
        clean_name = re.sub(r"(?<!\d)((?:19|20)\d{2})\d{4}(?!\d)", r"\1", clean_name)
        clean_name = clean_name.strip("_")
        
        # 5. Extract and shift year / year-pairs dynamically
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
                # Enforce '_filled' for root bathy models
                if base_lower == "bathy" or base_lower == "bathy_filled":
                    final_name = f"bathy_{year}_filled"
                # Strip root bathy prefixes from dependent derivatives to keep them succinct 
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