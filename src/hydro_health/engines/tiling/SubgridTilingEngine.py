"""Class engine for subtiling raster data into geoparquet files"""

import re
import os
import gc
import shutil
import tempfile
import pathlib
from pathlib import Path

import s3fs
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from upath import UPath

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs' 


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
        final_name = f"{base}_{year_pair}" if base else year_pair
    else:
        m_single = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", clean_name)
        if m_single:
            year = m_single.group(1)
            base = clean_name.replace(year, "").strip("_")
            base = re.sub(r"__+", "_", base)

            base_lower = base.lower()
            if base_lower in ["bathy", "bathy_filled"]:
                final_name = f"bathy_{year}_filled"
            elif base_lower.startswith("bathy_"):
                base = base[6:].strip("_")
                final_name = f"{base}_{year}"
            else:
                final_name = f"{base}_{year}" if base else year
        else:
            final_name = clean_name

    if is_bluetopo:
        return f"bt.{final_name}"

    return final_name


def _create_nan_stats_csv(df: pd.DataFrame, tile_id: str) -> pd.DataFrame:
    """Calculates NaN stats for a tile."""
    
    if df.empty:
        return pd.DataFrame()
    
    new_row = {'tile_id': tile_id}
    change_cols = [c for c in df.columns if c.startswith('delta_bathy_')]
    for col in change_cols:
        year_pair = col.replace('delta_bathy_', '')
        new_row[f"{year_pair}_nan_percent"] = round(df[col].isna().mean() * 100, 2)

    return pd.DataFrame([new_row])


def _subtile_process_gridded(sub_grid: pd.Series, raster_files: list, is_aws: bool) -> pd.DataFrame:
    """Process gridded rasters for a single tile dynamically and avoid sequential merging."""
    
    original_tile = str(sub_grid['original_tile'])
    pattern = re.compile(rf"(?:^|_){original_tile}(?:_|\.)")

    filtered_files = [f for f in raster_files if pattern.search(Path(f).name)]

    if not filtered_files:
        return pd.DataFrame()

    tile_extent = sub_grid.geometry.bounds
    data_arrays = {}
    common_window = None
    common_transform = None

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
                col_name = _standardize_col_name(col_name, original_tile)
                data_arrays[col_name] = (data, src.nodata)
        except Exception as e:
            # Warnings remain active so true pipeline failures are identifiable
            Engine.write_message_dask(f"WARNING: Error reading gridded file {file}: {e}", OUTPUTS)

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

    combined_data = pd.DataFrame(df_dict).drop_duplicates(subset=['X', 'Y'])
    
    del data_arrays, master_mask
    return combined_data


def _subtile_process_ungridded(sub_grid: pd.Series, raster_files: list, gridded_df: pd.DataFrame, static_patterns: list, is_aws: bool) -> pd.DataFrame:
    """Process ungridded rasters by translating spatial locations directly to pixel indices instead of merging."""
    
    if gridded_df is None or gridded_df.empty:
        return pd.DataFrame()

    combined_df = gridded_df.copy()
    xs = combined_df['X'].values
    ys = combined_df['Y'].values
    tile_extent = sub_grid.geometry.bounds
    original_tile = str(sub_grid.get('original_tile', ''))

    for pattern in static_patterns:
        current_files = [f for f in raster_files if pattern in Path(f).name]
        for file in current_files:
            col_name = _standardize_col_name(Path(file).stem, original_tile)
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
                        if src.nodata is not None and not np.isnan(src.nodata):
                            extracted_vals[extracted_vals == src.nodata] = np.nan
                        vals[win_valid] = extracted_vals

                    if col_name in combined_df:
                        existing = combined_df[col_name].values
                        existing_nan = np.isnan(existing)
                        existing[existing_nan] = vals[existing_nan]
                        combined_df[col_name] = existing
                    else:
                        combined_df[col_name] = vals

            except Exception as e:
                Engine.write_message_dask(f"WARNING: Failed to sample ungridded raster {file}: {e}", OUTPUTS)
                if col_name not in combined_df:
                    combined_df[col_name] = np.full(len(xs), np.nan, dtype=np.float32)

    return combined_df


def _save_combined_data(combined_df: pd.DataFrame, output_folder: str, data_type: str, tile_id: str, is_aws: bool, local_tmp_dir: str, current_index: int, total_count: int, verbose: bool) -> tuple:
    """Combine dataframes, explicitly save to temporary disk, move to output, and aggressively drop memory."""
    
    if combined_df is None or combined_df.empty:
        if verbose:
            Engine.write_message_dask(f" [SKIP] Tile '{tile_id}': No valid data assembled. Skipping file creation.", OUTPUTS)
        return pd.DataFrame()

    if data_type in ["training", "prediction"]:
        bathy_years = set()
        for c in combined_df.columns:
            if data_type == "training":
                m = re.search(r'_(\d{4})(?:_filled)?$', c, re.IGNORECASE) if c.startswith('bathy_') else None
                if m: bathy_years.add(int(m.group(1)))
            else:
                m = re.search(r'\.(\d{4})$', c) if c.startswith('bt.') else None
                if m: bathy_years.add(int(m.group(1)))

        sorted_years = sorted(list(bathy_years))
        dynamic_year_ranges = [(sorted_years[i], sorted_years[i+1]) for i in range(len(sorted_years)-1)]
        valid_pairs = []

        for y0, y1 in dynamic_year_ranges:
            y0_str, y1_str = str(y0), str(y1)

            if data_type == "training":
                c_0 = [c for c in combined_df.columns if re.match(rf"^bathy_{y0_str}_filled$", c, re.IGNORECASE)]
                c_1 = [c for c in combined_df.columns if re.match(rf"^bathy_{y1_str}_filled$", c, re.IGNORECASE)]
                b_y0 = [c for c in c_0 if "filled" in c.lower()][0] if c_0 else None
                b_y1 = [c for c in c_1 if "filled" in c.lower()][0] if c_1 else None
            else:
                c_0 = [c for c in combined_df.columns if re.match(rf"^bt\.(?:bluetopo_)?{y0_str}$", c, re.IGNORECASE)]
                c_1 = [c for c in combined_df.columns if re.match(rf"^bt\.(?:bluetopo_)?{y1_str}$", c, re.IGNORECASE)]
                b_y0 = c_0[0] if c_0 else None
                b_y1 = c_1[0] if c_1 else None

            if b_y0 and b_y1:
                combined_df[f"delta_bathy_{y0_str}_{y1_str}"] = combined_df[b_y1] - combined_df[b_y0]
                valid_pairs.append(f"{y0_str}_{y1_str}")
            else:
                Engine.write_message_dask(f"WARNING: MISSING BATHY DATA: Cannot calculate delta_bathy for {y0_str}_{y1_str} on tile '{tile_id}'.", OUTPUTS)

        cols_to_drop = [c for c in combined_df.columns if re.search(r"(\d{4}_\d{4})$", c) and not c.startswith("delta_bathy_") and re.search(r"(\d{4}_\d{4})$", c).group(1) not in valid_pairs]
        if cols_to_drop:
            combined_df.drop(columns=cols_to_drop, inplace=True)

    if 'tile_id' not in combined_df.columns:
        combined_df['tile_id'] = tile_id
    if 'FID' not in combined_df.columns:
        combined_df.insert(0, 'FID', np.arange(len(combined_df)))

    output_folder_path = UPath(output_folder)
    if not is_aws: 
        output_folder_path.mkdir(parents=True, exist_ok=True)

    final_save_path = str(output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet")
    tmp_dst_path = str(Path(local_tmp_dir) / f"{tile_id}_{data_type}_clipped_data.parquet")

    combined_df.to_parquet(tmp_dst_path, engine="pyarrow", index=False)

    if is_aws and final_save_path.startswith("s3://"):
        s3fs.S3FileSystem().put(tmp_dst_path, final_save_path)
    else:
        shutil.copy(tmp_dst_path, final_save_path)

    if verbose:
        Engine.write_message_dask(f" [{current_index}/{total_count}] [SUCCESS] Saved combined tile data to: {final_save_path}", OUTPUTS)

    stats_df = _create_nan_stats_csv(combined_df, tile_id)

    if tmp_dst_path != final_save_path and Path(tmp_dst_path).exists():
        os.remove(tmp_dst_path)

    return stats_df


def _process_tile(params: list) -> pd.DataFrame:
    """Core worker task for processing a single tile. Designed for dask pickling."""
    
    # Unpack the new verbose flag passed via params
    sub_grid, gridded_files, ungridded_files, static_patterns, is_aws, output_folder, data_type, tile_name, local_tmp_dir, current_index, total_count, verbose = params

    expected_path = UPath(output_folder) / f"{tile_name}_{data_type}_clipped_data.parquet"

    if verbose:
        Engine.write_message_dask(f"Processing tile {tile_name} ({current_index}/{total_count})...", OUTPUTS)

    # Prediction must have bluetopo data. Training MUST have combined lidar data.
    has_bluetopo = any("bluetopo" in Path(f).name.lower() or Path(f).name.lower().startswith("bt.") for f in gridded_files + ungridded_files)
    has_combined_lidar = any("combined" in Path(f).name.lower() for f in gridded_files + ungridded_files)

    if data_type == "prediction" and not has_bluetopo:
        if verbose:
            Engine.write_message_dask(f" [SKIP] Tile {tile_name}: Missing required BlueTopo data for prediction.", OUTPUTS)
        return pd.DataFrame()

    if data_type == "training" and not has_combined_lidar:
        if verbose:
            Engine.write_message_dask(f" [SKIP] Tile {tile_name}: Missing required combined LiDAR data for training.", OUTPUTS)
        return pd.DataFrame()

    try:
        # Check if the output file already exists. If so, skip raster processing and just read it to get the NaN statistics.
        if expected_path.exists():
            if verbose:
                Engine.write_message_dask(f" [SKIP] Tile already processed: {tile_name}. Compiling statistics from existing data.", OUTPUTS)
            df = pd.read_parquet(expected_path)
            stats = _create_nan_stats_csv(df, tile_name)
            del df
            return stats

        # Run fresh raster extraction
        gridded_df = _subtile_process_gridded(sub_grid, gridded_files, is_aws)
        combined_df = _subtile_process_ungridded(sub_grid, ungridded_files, gridded_df, static_patterns, is_aws)
        
        # Save and calculate final table statistics
        stats = _save_combined_data(combined_df, output_folder, data_type, tile_name, is_aws, local_tmp_dir, current_index, total_count, verbose)
        del combined_df
        return stats
    
    except Exception as e:
        Engine.write_message_dask(f"ERROR: Error processing tile {tile_name}: {e}", OUTPUTS)
        return pd.DataFrame()
    finally:
        gc.collect()


class SubgridTilingEngine(Engine):
    """Class for parallel subtiling of raster data into geoparquet files"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize paths, configurations, and environment variables"""
        
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix

        # Setup local temp dir mapping to ensure EC2 limits aren't exceeded
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "subgrid_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)

        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']
        self.static_patterns = ['sed', 'tsm', 'hurr', 'grain', 'survey']

        self.inputs_dir = INPUTS

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""
        
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f"SubgridTilingEngine resolved outputs_dir for region {region}: {self.outputs_dir}", OUTPUTS)

        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        # Model output directories 
        prediction_output_dir = get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR')
        self.prediction_out_dir = UPath(f"{s3_dir_base}/{prediction_output_dir}") if self.is_aws else UPath(self.outputs_dir / prediction_output_dir)

        training_out_dir = get_config_item('MODEL', 'TRAINING_OUTPUT_DIR')
        self.training_out_dir = UPath(f"{s3_dir_base}/{training_out_dir}") if self.is_aws else UPath(self.outputs_dir / training_out_dir)

        # Tile directories
        training_tiles_dir = get_config_item('MODEL', 'TRAINING_TILES_DIR')
        self.training_tiles_dir = UPath(f"{s3_dir_base}/{training_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / training_tiles_dir)

        prediction_tiles_dir = get_config_item('MODEL', 'PREDICTION_TILES_DIR')
        self.prediction_tiles_dir = UPath(f"{s3_dir_base}/{prediction_tiles_dir}") if self.is_aws else UPath(self.outputs_dir / prediction_tiles_dir)

        if not self.is_aws:
            self.training_tiles_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_tiles_dir.mkdir(parents=True, exist_ok=True)

        # Subgrid definitions 
        training_subgrid_path = get_config_item('MODEL', 'TRAINING_SUB_GRIDS')
        prediction_subgrid_path = get_config_item('MODEL', 'PREDICTION_SUB_GRIDS')
        self.subgrid_paths = {
            'training': UPath(f"{s3_dir_base}/{training_subgrid_path}") if self.is_aws else UPath(self.outputs_dir / training_subgrid_path),
            'prediction': UPath(f"{s3_dir_base}/{prediction_subgrid_path}") if self.is_aws else UPath(self.outputs_dir / prediction_subgrid_path)
        }

        # Terrain defaults
        filled_dir = get_config_item('TERRAIN', 'FILLED_DIR')
        self.filled_out_dir = UPath(f"{s3_dir_base}/{filled_dir}") if self.is_aws else UPath(self.outputs_dir / filled_dir)
        self.filled_folder_name = self.filled_out_dir.name.lower()

        # Config item for combined LiDAR dir for training files 
        combined_lidar_dir = get_config_item('TERRAIN', 'COMBINED_LIDAR_DIR')
        self.combined_lidar_dir = UPath(f"{s3_dir_base}/{combined_lidar_dir}") if self.is_aws else UPath(self.outputs_dir / combined_lidar_dir)

    def _load_subgrids(self, data_type: str) -> gpd.GeoDataFrame:
        """Loads the subgrids definition for the given data type."""
        
        sub_grid_path = self.subgrid_paths.get(data_type)
        if not sub_grid_path:
            return None

        self.write_message(f"Loading subgrids from: {sub_grid_path}", OUTPUTS)
        try:
            return gpd.read_file(str(sub_grid_path))
        except Exception as e:
            self.write_message(f"EXCEPTION: Reading subgrids from {sub_grid_path} failed. {e}", OUTPUTS)
            return None


    def _get_filtered_raster_files(self, raster_dirs: list, data_type: str) -> list:
        """Scans and filters raster files based on type rules."""
        
        self.write_message("Scanning directories for raster files...", OUTPUTS)
        all_raster_files = []

        for raster_dir in raster_dirs:
            if not raster_dir.exists():
                self.write_message(f"Directory not found, skipping: {raster_dir}", OUTPUTS)
                continue
                
            for f in raster_dir.rglob("*"):
                if f.suffix.lower() in {'.tif', '.tiff'}:
                    name_lower = f.name.lower()
                    parts_lower = [p.lower() for p in f.parts]

                    # Skip filled tifs ONLY IF they aren't our required combined files for training
                    if (self.filled_folder_name in parts_lower or "filled_lidar" in parts_lower or "filled_tifs" in parts_lower) and "combined" not in name_lower:
                        continue
                    if "unc" in name_lower:
                        continue
                    
                    exclude_patterns = ["tsm_cumulative", "hurr_count_mean", "hurr_count_cumulative", "hurr_strength_cumulative"]
                    if any(x in name_lower for x in exclude_patterns) or \
                       re.search(r"hurr_count_\d{4}_\d{4}", name_lower) or \
                       re.search(r"hurr_strength_\d{4}_\d{4}", name_lower):
                        continue

                    if data_type == 'training' and ("bluetopo" in name_lower or name_lower.startswith("bt.")) and "survey_end_date" not in name_lower:
                        continue
                    elif data_type == 'prediction' and "bathy" in name_lower and "bluetopo" not in name_lower and not name_lower.startswith("bt."):
                        continue

                    all_raster_files.append(str(f))

        return all_raster_files


    def _partition_raster_files(self, all_raster_files: list, sub_grids: gpd.GeoDataFrame) -> tuple:
        """Pre-partition files into gridded vs ungridded lists."""
        
        valid_tids = [str(tid) for tid in sub_grids['original_tile'].unique() if pd.notna(tid) and str(tid).strip()]
        valid_tid_patterns = [re.compile(rf"(?:^|_){tid}(?:_|\.)") for tid in valid_tids]

        gridded_files = [f for f in all_raster_files if any(p.search(Path(f).name) for p in valid_tid_patterns)]
        ungridded_files = [f for f in all_raster_files if f not in gridded_files]

        return gridded_files, ungridded_files


    def _process_pipeline(self, raster_dirs: list, output_dir: UPath, data_type: str, verbose_workers: bool = False) -> None:
        """Orchestrates the tile processing for a specific data type via Dask mapping."""
        
        self.write_message(f"--- Starting {data_type.upper()} pipeline ---", OUTPUTS)
        self.write_message(self.log_system_metrics(), OUTPUTS)

        sub_grids = self._load_subgrids(data_type)
        if sub_grids is None or sub_grids.empty:
            return

        all_raster_files = self._get_filtered_raster_files(raster_dirs, data_type)
        gridded_files, ungridded_files = self._partition_raster_files(all_raster_files, sub_grids)

        # Build task payloads
        params_list = []
        total_tiles = len(sub_grids)
        for i, (_, sub_grid) in enumerate(sub_grids.iterrows()):
            tile_name = sub_grid['tile_id']
            output_folder = output_dir / tile_name
            params_list.append([
                sub_grid,
                gridded_files,
                ungridded_files,
                self.static_patterns,
                self.is_aws,
                str(output_folder),
                data_type,
                tile_name,
                str(self.local_tmp_dir),
                i + 1,
                total_tiles,
                verbose_workers # Passed dynamically to control per-tile spam
            ])

        self.write_message(f"Submitting {total_tiles} tile tasks to Dask client map... (Worker logs muted)", OUTPUTS)
        futures = self.client.map(_process_tile, params_list)
        results = self.client.gather(futures)

        # Concatenate returned dataframes to construct final stats summary
        valid_results = [res for res in results if res is not None and not res.empty]
        if valid_results:
            final_results_df = pd.concat(valid_results, ignore_index=True)
            output_csv_path = output_dir.parent / f"year_pair_nan_counts_{data_type}.csv"
            final_results_df.to_csv(str(output_csv_path), index=False, na_rep='NA')
            self.write_message(f"[SUCCESS] Statistics successfully saved to: {output_csv_path}", OUTPUTS)

        self.write_message(self.log_system_metrics(), OUTPUTS)


    def run(self) -> None:
        """Main entry point for evaluating training masks and processing rasters in parallel"""
        
        env = self.param_lookup.get('env', 'local')
        
        try:
            self.setup_dask(env, n_workers=4, threads_per_worker=1, memory_limit="6GB")

            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)

                # Process Prediction Data
                self._process_pipeline(
                    raster_dirs=[self.prediction_out_dir], 
                    output_dir=self.prediction_tiles_dir, 
                    data_type="prediction",
                    verbose_workers=False  # Controls task-level logging verbosity
                )

                # Process Training Data
                self._process_pipeline(
                    raster_dirs=[self.training_out_dir, self.combined_lidar_dir], 
                    output_dir=self.training_tiles_dir, 
                    data_type="training",
                    verbose_workers=False  # Controls task-level logging verbosity
                )
        finally:
            self.cleanup_resources(OUTPUTS)