"""Class for data acquisition and preprocessing of model data"""

import os
import re
import pathlib
import yaml
import warnings
from typing import List, Tuple, Literal
from pathlib import Path
from rasterio.features import shapes
from shapely.geometry import shape
import s3fs

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import dask
from dask.distributed import Client
from osgeo import gdal
from shapely.geometry import Point, box

from hydro_health.helpers.tools import get_config_item, get_environment
from hydro_health.engines.CreateSeabedTerrainLayerEngine import CreateSeabedTerrainLayerEngine

os.environ["GDAL_CACHEMAX"] = "64"


class ModelDataPreProcessor:
    """Class for parallel preprocessing all model data"""

    def __init__(self):
        self.target_crs = "EPSG:32617"
        self.target_res = 8

        self.fs = s3fs.S3FileSystem(anon=False)

        self.static_patterns = ['sed_size_raster', 'sed_type_raster', 'tsm_mean', 'hurr']
        
        self.year_ranges = [
            (1998, 2004), (2004, 2006), (2006, 2007), (2007, 2010),
            (2010, 2015), (2014, 2022), (2016, 2017), (2017, 2018),
            (2018, 2019), (2020, 2022), (2022, 2024)
        ]

        self.re_year_suffix = re.compile(r"_\d{4}$")
        self.re_year_extract = re.compile(r"_(\d{4})")
        self.re_bt_prefix = re.compile(r"^bt\.")
        self.re_flowdir = re.compile(r"flowdir")

        self.excluded_keys = self._load_exclusion_config()
        self.is_aws = (get_environment() == 'aws')

    def create_file_paths(self):
        if get_environment() == 'remote':
            self.mask_prediction_pq = pathlib.Path(get_config_item('MASK', 'PREDICTION_MASK_PQ'))
            self.mask_training_pq = pathlib.Path(get_config_item('MASK', 'TRAINING_MASK_PQ'))
            self.pred_mask_path = pathlib.Path(get_config_item('MASK', 'MASK_PRED_PATH'))
            self.train_mask_path = pathlib.Path(get_config_item('MASK', 'MASK_TRAINING_PATH'))
            self.preprocessed_dir = pathlib.Path(get_config_item('MODEL', 'PREPROCESSED_DIR'))
            self.prediction_out_dir = pathlib.Path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'))
            self.training_out_dir = pathlib.Path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'))
            self.training_tiles_dir = pathlib.Path(get_config_item('MODEL', 'TRAINING_TILES_DIR'))
            self.prediction_tiles_dir = pathlib.Path(get_config_item('MODEL', 'PREDICTION_TILES_DIR'))
            self.uncombined_lidar_dir = pathlib.Path(get_config_item('MODEL', 'UNCOMBINED_LIDAR_DIR'))
            self.subgrid_paths = {
                'training': pathlib.Path(get_config_item('MODEL', 'TRAINING_SUB_GRIDS')),
                'prediction': pathlib.Path(get_config_item('MODEL', 'PREDICTION_SUB_GRIDS'))
            }

        elif get_environment() == 'aws':
            bucket = get_config_item('S3', 'BUCKET_NAME')
            self.mask_prediction_pq = f"s3://{bucket}/{get_config_item('MASK', 'PREDICTION_MASK_PQ')}"
            self.mask_training_pq = f"s3://{bucket}/{get_config_item('MASK', 'TRAINING_MASK_PQ')}"
            self.pred_mask_path = f"s3://{bucket}/{get_config_item('MASK', 'MASK_PRED_PATH')}"
            self.train_mask_path = f"s3://{bucket}/{get_config_item('MASK', 'MASK_TRAINING_PATH')}"
            self.preprocessed_dir = f"s3://{bucket}/{get_config_item('MODEL', 'PREPROCESSED_DIR')}"
            self.prediction_out_dir = f"s3://{bucket}/{get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR')}"
            self.training_out_dir = f"s3://{bucket}/{get_config_item('MODEL', 'TRAINING_OUTPUT_DIR')}"
            self.training_tiles_dir = f"s3://{bucket}/{get_config_item('MODEL', 'TRAINING_TILES_DIR')}"
            self.prediction_tiles_dir = f"s3://{bucket}/{get_config_item('MODEL', 'PREDICTION_TILES_DIR')}"
            self.uncombined_lidar_dir = f"s3://{bucket}/{get_config_item('MODEL', 'UNCOMBINED_LIDAR_DIR')}"
            self.subgrid_paths = {
                'training': f"s3://{bucket}/{get_config_item('MODEL', 'TRAINING_SUB_GRIDS')}",
                'prediction': f"s3://{bucket}/{get_config_item('MODEL', 'PREDICTION_SUB_GRIDS')}"
            }

    def _load_exclusion_config(self) -> set:
        """Loads dataset exclusion keys from YAML config."""
        try:
            inputs_root = pathlib.Path(__file__).parents[4] / 'inputs'
            config_path = inputs_root / 'lookups' / 'ER_3_lidar_data_config.yaml'
            
            if not config_path.exists():
                print(f"Warning: Config path {config_path} not found.")
                return set()

            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)

            excluded = {
                key for key, data in config_data.get('EcoRegion-3', {}).items()
                if data.get('use') is False
            }
            
            if excluded:
                print(f"Loaded {len(excluded)} exclusion keys from config.")
            
            return excluded
        except Exception as e:
            print(f"Error loading exclusion config: {e}")
            return set()

    def process(self) -> None:
        """Main function to process model data."""     
        # TODO use the base engine function for this, change the memory limit for AWS
        client = Client(
            n_workers=8, 
            threads_per_worker=2, 
            memory_limit="32GB"
        )
        print(f"Dask Dashboard: {client.dashboard_link}")

        self.create_file_paths()

        # self.raster_to_spatial_df(self.pred_mask_path, process_type='prediction')
        # self.raster_to_spatial_df(self.train_mask_path, process_type='training')

        input("Press Enter to continue...")

        self.create_subgrids(mask_gdf=self.mask_prediction_pq, output_dir=self.subgrid_paths['prediction'], process_type='prediction')
        self.create_subgrids(mask_gdf=self.mask_training_pq, output_dir=self.subgrid_paths['training'], process_type='training')

        input("Press Enter to continue...")

        try:
            mask_pred_gdf = gpd.read_parquet(self.mask_prediction_pq)
            mask_future_pred = client.scatter(mask_pred_gdf.union_all, broadcast=True)

            mask_train_gdf = gpd.read_parquet(self.mask_training_pq)
            mask_future_train = client.scatter(mask_train_gdf.union_all, broadcast=True)

            self.parallel_processing_rasters(self.preprocessed_dir, mask_future_pred, mask_future_train)

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
            
            self.batch_long_format_transformation(base_dir=self.training_tiles_dir, mode="training")

        finally:
            client.close()

    def parallel_processing_rasters(self, input_directory, mask_future_pred, mask_future_train) -> None:
        """Process prediction and training rasters in parallel using Dask."""
        
        if not self.is_aws:
            input_directory = pathlib.Path(input_directory)
            self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)
        # AWS: s3fs handles "mkdir" implicitly by object existence

        # --- 1. Identify Existing Outputs ---
        # Get existing prediction outputs
        potential_files = (
            [pathlib.Path(f) for f in self.fs.glob(f"{self.prediction_out_dir}/*")]
            if self.is_aws 
            else list(self.prediction_out_dir.glob("*"))
        )
        existing_pred_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }

        # Get existing uncombined outputs
        potential_files = (
            [pathlib.Path(f) for f in self.fs.glob(f"{self.uncombined_lidar_dir}/*")]
            if self.is_aws
            else list(self.uncombined_lidar_dir.glob("*"))
        )
        existing_uncombined_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        
        all_existing_outputs = existing_pred_outputs.union(existing_uncombined_outputs)
        
        # --- 2. Gather Prediction Inputs ---
        excluded_folders = {'lidar_filled_tifs', 'uncombined_lidar_tifs'}

        potential_files = (
            [pathlib.Path(f) for f in self.fs.glob(f"{input_directory}/**/*") if f.lower().endswith(('.tif', '.tiff'))]
            if self.is_aws
            else [f for f in input_directory.rglob("*") if f.suffix.lower() in {'.tif', '.tiff'}]
        )
    
        prediction_files = [
            f for f in potential_files
            if f.name not in all_existing_outputs
            and not any(key in f.name for key in self.excluded_keys)
            and not any(folder in f.parts for folder in excluded_folders)
        ]

        print(f"Processing {len(prediction_files)} prediction files (Skipping {len(all_existing_outputs)} existing)...")

        # --- 3. Queue Prediction Tasks ---
        prediction_tasks = []
        for file_path in prediction_files:
            # Handle Path Joining (AWS String vs Local Path)
            if self.is_aws:
                # AWS S3 Path Construction
                base_out = self.uncombined_lidar_dir if "mosaic" in file_path.name.lower() else self.prediction_out_dir
                output_path = f"{base_out.rstrip('/')}/{file_path.name}"
                # Add s3:// for input reading
                input_str = f"s3://{file_path.as_posix()}"
            else:
                # Local Path Construction
                base_out = self.uncombined_lidar_dir if "mosaic" in file_path.name.lower() else self.prediction_out_dir
                output_path = base_out / file_path.name
                input_str = str(file_path)

            task = dask.delayed(self.process_prediction_raster)(
                input_str, 
                mask_future_pred, 
                str(output_path)
            )
            prediction_tasks.append(task)

        if prediction_tasks:
            dask.compute(*prediction_tasks)
            print("Prediction raster processing complete.")
        else:
            print("No new prediction rasters to process.")

        print("Running Seabed Terrain Layer Engine...")
        engine = CreateSeabedTerrainLayerEngine()
        engine.process()

        if not self.is_aws:
            self.training_out_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. Gather Training Inputs (from Prediction Outputs) ---
        # Check what is already processed in training dir
        potential_files = (
            [pathlib.Path(f) for f in self.fs.glob(f"{self.training_out_dir}/*")]
            if self.is_aws
            else list(self.training_out_dir.glob("*"))
        )

        existing_train_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }

        # Candidate files are in prediction_out_dir
        potential_files = (
            [pathlib.Path(f) for f in self.fs.glob(f"{self.prediction_out_dir}/*")]
            if self.is_aws
            else list(self.prediction_out_dir.glob("*"))
        )

        training_candidates = [
            f for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        ]

        training_files = [
            f for f in training_candidates
            if f.name not in existing_train_outputs
            and not ('mosaic' in f.name and 'filled' not in f.name)
        ]

        print(f"Processing {len(training_files)} training files (Skipping {len(existing_train_outputs)} existing)...")

        # --- 5. Queue Training Tasks ---
        training_tasks = []
        for file_path in training_files:
            if self.is_aws:
                output_path = f"{self.training_out_dir.rstrip('/')}/{file_path.name}"
                input_str = f"s3://{file_path.as_posix()}"
            else:
                output_path = self.training_out_dir / file_path.name
                input_str = str(file_path)
            
            task = dask.delayed(self.process_training_raster)(
                input_str, 
                mask_future_train, 
                str(output_path)
            )
            training_tasks.append(task)

        if training_tasks:
            dask.compute(*training_tasks)
            print("Training raster processing complete.")
        else:
            print("No new training rasters to process.")

    def process_prediction_raster(self, raster_path, mask_union, output_path) -> None:
        """Reprojects, resamples, and crops a raster for prediction."""
        raster_name = pathlib.Path(raster_path).name.lower()
        
        # Ensure S3 prefix if needed
        if self.is_aws and not str(raster_path).startswith('s3://'):
            raster_path = f"s3://{str(raster_path).lstrip('/')}"

        try:
            with rasterio.open(raster_path) as src:
                src_nodata = src.nodata
        except Exception as e:
            print(f"Error: Could not open {raster_name}. {e}")
            return

        print(f'Processing prediction file {raster_name}...')
        should_crop = any(k in raster_name for k in ["tsm", "sed", "hurr"])
        
        self._warp_to_cutline(
            str(raster_path), 
            str(output_path), 
            mask_geometry=mask_union, 
            dst_crs=self.target_crs, 
            x_res=self.target_res, 
            y_res=self.target_res,
            crop_to_cutline=should_crop,
            src_nodata=src_nodata
        )

    def process_training_raster(self, raster_path, mask_union, output_path) -> None:
        """Process a training raster by clipping it with a mask."""
        raster_path_obj = pathlib.Path(raster_path)
        raster_name = raster_path_obj.name.lower()

        # Ensure S3 prefix if needed
        if self.is_aws and not str(raster_path).startswith('s3://'):
            raster_path = f"s3://{str(raster_path).lstrip('/')}"

        try:
            with rasterio.open(raster_path) as src:
                src_nodata = src.nodata
                raster_bounds = box(*src.bounds)

            if not mask_union.intersects(raster_bounds):
                print(f"Mask does not intersect raster {raster_name}. Skipping.")
                return

            print(f'Processing training file {raster_name}...')
            
            self._warp_to_cutline(
                str(raster_path),
                str(output_path),
                mask_geometry=mask_union,
                src_nodata=src_nodata,
                dst_nodata=np.nan
            )
        except Exception as e:
            print(f"Error processing {raster_name}: {e}")

    def _warp_to_cutline(self, src_path, dst_path, mask_geometry, **kwargs):
        """Helper to handle GDAL Warp boilerplate and in-memory cutlines."""
        
        # --- 1. PREPARE PATHS FOR GDAL ---
        src_str = str(src_path)
        dst_str = str(dst_path)

        # If AWS, strip s3:// if present, then prepend /vsis3/
        if self.is_aws:
            if src_str.startswith('s3://'): src_str = src_str.replace('s3://', '')
            if dst_str.startswith('s3://'): dst_str = dst_str.replace('s3://', '')
            
            if not src_str.startswith('/vsis3/'): src_str = f"/vsis3/{src_str}"
            if not dst_str.startswith('/vsis3/'): dst_str = f"/vsis3/{dst_str}"

        # --- 2. SETUP CUTLINE ---
        raster_name = os.path.basename(src_str)
        in_memory_cutline = f'/vsimem/cutline_{os.getpid()}_{hash(raster_name)}.geojson'
        
        try:
            gdf = gpd.GeoDataFrame(geometry=[mask_geometry], crs=self.target_crs)
            gdf.to_file(in_memory_cutline, driver='GeoJSON')
        except Exception as e:
            print(f"Error creating cutline for {raster_name}: {e}")
            return

        # --- 3. CONFIGURE WARP ---
        warp_opts = {
            'cutlineDSName': in_memory_cutline,
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE'],
            'creationOptions': ['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES'],
            'multithread': True,
            'warpMemoryLimit': 2048,
            'resampleAlg': 'bilinear'
        }
        
        if 'dst_crs' in kwargs: warp_opts['dstSRS'] = kwargs.pop('dst_crs')
        if 'x_res' in kwargs: warp_opts['xRes'] = kwargs.pop('x_res')
        if 'y_res' in kwargs: warp_opts['yRes'] = kwargs.pop('y_res')
        if 'crop_to_cutline' in kwargs: warp_opts['cropToCutline'] = kwargs.pop('crop_to_cutline')
        if 'src_nodata' in kwargs: warp_opts['srcNodata'] = kwargs.pop('src_nodata')
        if 'dst_nodata' in kwargs: warp_opts['dstNodata'] = kwargs.pop('dst_nodata')
        
        # --- 4. EXECUTE WARP ---
        try:
            gdal.Warp(dst_str, src_str, **warp_opts)
        except Exception as e:
            print(f"GDAL Warp failed for {src_str}: {e}")
        finally:
            gdal.Unlink(in_memory_cutline)

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type) -> None:
        """Clip raster files by tile and save data."""
        print(f" - Clipping {data_type} rasters by tile...")
        
        sub_grid_path = self.subgrid_paths.get(data_type)
        if not sub_grid_path:
            print(f"Error: No subgrid path defined for {data_type}")
            return

        # Ensure reading works on AWS
        sub_grid_read_path = sub_grid_path
        if self.is_aws and not str(sub_grid_path).startswith('s3://'):
             sub_grid_read_path = f"s3://{str(sub_grid_path).lstrip('/')}"
        
        try:
             sub_grids = gpd.read_file(sub_grid_read_path)
        except Exception as e:
             print(f"Error reading subgrids from {sub_grid_read_path}: {e}")
             return
        
        print(f"Number of tiles: {sub_grids.shape[0]}")
        
        tasks = []
        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id']

            if "BH4SD56H" not in tile_name:
                continue
            
            # Handle Path Joining
            if self.is_aws:
                output_folder = f"{str(output_dir).rstrip('/')}/{tile_name}"
            else:
                output_folder = pathlib.Path(output_dir) / tile_name

            gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, raster_dir)
            ungridded_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, raster_dir)
            
            tasks.append(
                self.save_combined_data(gridded_task, ungridded_task, str(output_folder), data_type, tile_id=tile_name)
            )

        results_list = dask.compute(*tasks)

        print("Combining results...")
        final_results_df = pd.concat(results_list, ignore_index=True)
        
        if self.is_aws:
             output_csv_path = f"{str(output_dir).rstrip('/')}/../year_pair_nan_counts_{data_type}.csv"
        else:
             output_csv_path = pathlib.Path(output_dir).parent / f"year_pair_nan_counts_{data_type}.csv"
        
        final_results_df.to_csv(output_csv_path, index=False, na_rep='NA')
        print(f"Stats saved to: {output_csv_path}")

    def subtile_process_gridded(self, sub_grid, raster_dir) -> pd.DataFrame:
        """Process gridded rasters for a single tile."""

        original_tile = sub_grid['original_tile']
                
        # Handle AWS vs Local file listing
        if self.is_aws:
            # Use raw string for glob on AWS
            raster_dir_str = str(raster_dir).rstrip('/')
            potential_files = [pathlib.Path(f) for f in self.fs.glob(f"{raster_dir_str}/**/*")]
        else:
            raster_dir_path = pathlib.Path(raster_dir)
            potential_files = list(raster_dir_path.rglob("*"))

        raster_files = [
            f for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'} 
            and original_tile in f.name
        ]

        tile_extent = sub_grid.geometry.bounds
        dfs = []
        for file in raster_files:
            # Add prefix if AWS
            file_str = f"s3://{file.as_posix()}" if self.is_aws else str(file)
            
            df = self._extract_raster_to_df(file_str, tile_extent)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()

        combined_data = pd.concat(dfs, ignore_index=True)

        combined_data['X'] = combined_data['X'].round(3)
        combined_data['Y'] = combined_data['Y'].round(3)

        combined_data = combined_data.drop_duplicates(subset=['X', 'Y', 'Raster'])

        combined_data = combined_data.pivot(index=['X', 'Y'], columns='Raster', values='Value').reset_index()
        
        combined_data['geometry'] = [Point(x, y) for x, y in zip(combined_data['X'], combined_data['Y'])]
        combined_data = gpd.GeoDataFrame(combined_data, geometry='geometry', crs=self.target_crs)
        
        return combined_data

    def subtile_process_ungridded(self, sub_grid, raster_dir) -> pd.DataFrame:
        """Process ungridded rasters for a single tile."""

        if not self.is_aws:
            raster_dir = pathlib.Path(raster_dir)

        dfs = []
        tile_extent = sub_grid.geometry.bounds

        for pattern in self.static_patterns:
            if self.is_aws:
                raster_dir_str = str(raster_dir).rstrip('/')
                # glob returns strings, add s3:// prefix
                current_files = [f"s3://{f}" for f in self.fs.glob(f"{raster_dir_str}/**/*{pattern}*.tif")]
            else:
                current_files = list(raster_dir.rglob(f"*{pattern}*.tif"))

            for file in current_files:
                df = self._extract_raster_to_df(str(file), tile_extent)
                if not df.empty:
                    dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined_data = pd.concat(dfs, ignore_index=True)

        combined_data['X'] = combined_data['X'].round(3)
        combined_data['Y'] = combined_data['Y'].round(3)
        combined_data = combined_data.drop_duplicates(subset=['X', 'Y', 'Raster'])

        combined_data = combined_data.pivot(index=['X', 'Y'], columns='Raster', values='Value').reset_index()
        return combined_data

    def _extract_raster_to_df(self, raster_path, tile_extent) -> pd.DataFrame:
        """Helper to read a window of a raster and convert to DataFrame."""

        try:
            with rasterio.open(raster_path) as src:
                window = src.window(*tile_extent)
                data = src.read(1, window=window)
                transform = src.window_transform(window)
                mask = data != src.nodata
                
                if not mask.any():
                    return pd.DataFrame()

                rows, cols = np.where(mask)
                values = data[mask]
                xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
                
                return pd.DataFrame({
                    'X': xs, 'Y': ys, 'Value': values, 'Raster': pathlib.Path(raster_path).stem
                })
        except Exception as e:
            print(f"Error reading {raster_path}: {e}")
            return pd.DataFrame()

    @dask.delayed
    def save_combined_data(self, gridded_df, ungridded_df, output_folder, data_type, tile_id) -> pd.DataFrame:
        """Combine dataframes and save to parquet."""
        if gridded_df is None or gridded_df.empty:
            return pd.DataFrame()

        # Handle Directory Creation
        if not self.is_aws:
            output_folder_path = pathlib.Path(output_folder)
            output_folder_path.mkdir(parents=True, exist_ok=True)
            output_path = output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet"
            save_path = str(output_path)
        else:
            # AWS S3 (no mkdir needed)
            output_folder = str(output_folder).rstrip('/')
            save_path = f"s3://{output_folder}/{tile_id}_{data_type}_clipped_data.parquet"
            # Strip extra s3:// if double added
            if save_path.startswith("s3://s3://"): save_path = save_path.replace("s3://s3://", "s3://")

        if ungridded_df is not None and not ungridded_df.empty:
            combined = pd.merge(gridded_df, ungridded_df, on=['X', 'Y'], how='left')
        else:
            combined = gridded_df

        combined.to_parquet(save_path, engine="pyarrow", index=False)
        return self.create_nan_stats_csv(combined, tile_id)

    def create_nan_stats_csv(self, df, tile_id) -> pd.DataFrame:
        """Calculates NaN stats for a tile."""

        if df.empty:
            return pd.DataFrame()
        new_row = {'tile_id': tile_id}
        change_cols = [c for c in df.columns if c.startswith('b.change.')]
        for col in change_cols:
            year_pair = col.replace('b.change.', '')
            new_row[f"{year_pair}_nan_percent"] = round(df[col].isna().mean() * 100, 2)
        return pd.DataFrame([new_row])

    def batch_long_format_transformation(self, base_dir, mode: Literal["training", "prediction"]):
        """Orchestrator for transforming wide tiles to long year-pair format."""

        print("Starting Long Format Transformation...")
        print(f"Starting Dask Transformation Batch: {mode}")

        file_suffix = f"_{mode}_clipped_data.parquet"

        # Fix: Convert base_dir to Path only on Local
        if self.is_aws:
            base_dir_str = str(base_dir).rstrip('/')
            files_to_process = [pathlib.Path(f) for f in self.fs.glob(f"{base_dir_str}/**/*{file_suffix}")]
        else:
            files_to_process = list(pathlib.Path(base_dir).rglob(f"*{file_suffix}"))

        if not files_to_process:
            print("No files found for transformation.")
            return

        print(f"Queueing {len(files_to_process)} tiles...")

        tasks = []
        for fp in files_to_process:
            # Format path for passing to worker
            if self.is_aws:
                file_str = f"s3://{fp.as_posix()}"
            else:
                file_str = str(fp)
            
            tasks.append(dask.delayed(self._transform_tile_task)(file_str, mode))

        results = dask.compute(*tasks)

        success = sum(1 for r in results if r.startswith("Success"))
        failed = len(results) - success
        
        print(f"\n Transformation Complete. Success: {success}, Failed: {failed}")
        if failed > 0:
            print("Errors:\n" + "\n".join([r for r in results if not r.startswith("Success")]))

    def _transform_tile_task(self, f_path: str, mode: Literal["training", "prediction"]) -> str:
        """Dask Worker: Reads file -> Calls specific processor -> Returns status."""
        try:
            tile_name = os.path.basename(f_path).split("_")[0]
            output_dir = os.path.dirname(f_path)

            try:
                gdf = gpd.read_parquet(f_path)
            except Exception:
                gdf = gpd.GeoDataFrame(pd.read_parquet(f_path))

            if mode == "training":
                saved = self._process_and_save_training_tile(gdf, output_dir, tile_name)
            else:
                saved = self._process_and_save_prediction_tile(gdf, output_dir, tile_name)
            
            del gdf
            return f"Success: {tile_name} ({len(saved)} pairs)"

        except Exception as e:
            return f"Failed: {os.path.basename(f_path)} - {str(e)}"

    def _process_and_save_training_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str) -> List[str]:
        """Processes a training tile and writes outputs immediately to disk."""
        self._transform_flowdir_cols_inplace(gdf)
        col_meta = self._get_column_metadata(gdf.columns.tolist())
        
        saved_files = []

        for y0, y1 in self.year_ranges:
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"

            cols_t_meta = col_meta[col_meta["year"] == y0]
            cols_t1_meta = col_meta[col_meta["year"] == y1]

            common_vars = set(cols_t_meta["var_base"]).intersection(cols_t1_meta["var_base"])
            
            if "bathy_filled" not in common_vars:
                continue

            t_map = dict(zip(cols_t_meta["var_base"], cols_t_meta["colname"]))
            t1_map = dict(zip(cols_t1_meta["var_base"], cols_t1_meta["colname"]))

            sorted_common = sorted(common_vars)
            cols_t_exist = [t_map[v] for v in sorted_common]
            cols_t1_exist = [t1_map[v] for v in sorted_common]

            forcing_pattern = f"{y0_str}_{y1_str}$"
            forcing_cols = [c for c in gdf.columns if re.search(forcing_pattern, c)]
            
            delta_bathy_col = f"delta_bathy_{pair_name}"
            forcing_cols = [c for c in forcing_cols if c != delta_bathy_col]

            static_vars = ["grain_size_layer", "prim_sed_layer", "survey_end_date"]
            static_cols = [c for c in static_vars if c in gdf.columns]
            
            id_cols = [c for c in ["X", "Y", "FID", "tile_id", "geometry"] if c in gdf.columns]

            cols_to_grab = id_cols + cols_t_exist + cols_t1_exist + forcing_cols + static_cols
            if delta_bathy_col in gdf.columns:
                cols_to_grab.append(delta_bathy_col)

            if any(c not in gdf.columns for c in cols_to_grab):
                continue

            pair_gdf = gdf[cols_to_grab].drop_duplicates()

            new_names_t = [f"{v}_t" for v in sorted_common]
            new_names_t1 = [f"{v}_t1" for v in sorted_common]
            rename_map = dict(zip(cols_t_exist + cols_t1_exist, new_names_t + new_names_t1))
            
            pair_gdf.rename(columns=rename_map, inplace=True)
            
            if delta_bathy_col in pair_gdf.columns:
                pair_gdf.rename(columns={delta_bathy_col: "delta_bathy"}, inplace=True)

            pair_gdf["year_t"] = y0
            pair_gdf["year_t1"] = y1

            if "delta_bathy" not in pair_gdf.columns and "bathy_t" in pair_gdf.columns and "bathy_t1" in pair_gdf.columns:
                pair_gdf["delta_bathy"] = pair_gdf["bathy_t1"] - pair_gdf["bathy_t"]

            filled_cols = [c for c in pair_gdf.columns if "_filled" in c]
            if filled_cols:
                pair_gdf.rename(columns={c: c.replace("_filled", "") for c in filled_cols}, inplace=True)

            target_col = "bathy_t1"
            predictor_cols_t = [c for c in pair_gdf.columns if c.endswith("_t")]
            predictor_cols_delta = [c for c in pair_gdf.columns if c.startswith("delta_")]
            
            final_cols = id_cols + ["year_t", "year_t1", target_col] + predictor_cols_t + predictor_cols_delta + forcing_cols + static_cols
            final_cols = [c for c in final_cols if c in pair_gdf.columns]
            
            out_name = f"{tile_name}_{pair_name}_long.parquet"
            # Join paths carefully
            if output_dir.startswith("s3://"):
                out_path = f"{output_dir.rstrip('/')}/{out_name}"
            else:
                out_path = os.path.join(output_dir, out_name)
            
            pair_gdf[final_cols].to_parquet(out_path, index=None)
            saved_files.append(out_name)
            del pair_gdf

        return saved_files

    def _process_and_save_prediction_tile(self, gdf: gpd.GeoDataFrame, output_dir: str, tile_name: str) -> List[str]:
        """Processes a prediction tile and writes outputs immediately."""
        self._transform_flowdir_cols_inplace(gdf)
        
        bt_cols = [c for c in gdf.columns if self.re_bt_prefix.match(c)]
        new_t_names = [self.re_bt_prefix.sub("", c) + "_t" for c in bt_cols]
        
        gdf.rename(columns=dict(zip(bt_cols, new_t_names)), inplace=True)
        
        saved_files = []

        for y0, y1 in self.year_ranges:
            y0_str, y1_str = str(y0), str(y1)
            pair_name = f"{y0_str}_{y1_str}"
            
            forcing_pattern = f"{y0_str}_{y1_str}$"
            forcing_cols = [c for c in gdf.columns if re.search(forcing_pattern, c)]

            static_vars = ["grain_size_layer", "prim_sed_layer", "survey_end_date"]
            static_cols = [c for c in static_vars if c in gdf.columns]
            id_cols = [c for c in ["X", "Y", "FID", "tile_id", "geometry"] if c in gdf.columns]

            final_cols = id_cols + new_t_names + forcing_cols + static_cols
            final_cols = [c for c in final_cols if c in gdf.columns]

            pair_gdf = gdf[final_cols].drop_duplicates()
            
            out_name = f"{tile_name}_{pair_name}_prediction_long.parquet"
            # Join paths carefully
            if output_dir.startswith("s3://"):
                out_path = f"{output_dir.rstrip('/')}/{out_name}"
            else:
                out_path = os.path.join(output_dir, out_name)
            
            pair_gdf.to_parquet(out_path, index=None)
            saved_files.append(out_name)
            del pair_gdf

        return saved_files

    def _transform_flowdir_cols_inplace(self, df: pd.DataFrame) -> None:
        """Modifies DataFrame in-place to replace flow direction angles."""
        flow_cols = [c for c in df.columns if self.re_flowdir.search(c)]
        if not flow_cols:
            return

        radians = np.deg2rad(df[flow_cols])
        for col in flow_cols:
            df[f"{col}_sin"] = np.sin(radians[col])
            df[f"{col}_cos"] = np.cos(radians[col])

        df.drop(columns=flow_cols, inplace=True)

    def _get_column_metadata(self, columns: List[str]) -> pd.DataFrame:
        """Efficiently parses column names to extract variables and years."""
        potential_cols = [
            c for c in columns 
            if "_" in c and c[-4:].isdigit() and not (c[-9] == "_" and c[-4:].isdigit() and c[-9:-5].isdigit())
        ]
        
        if not potential_cols:
            return pd.DataFrame(columns=["colname", "year", "var_base"])

        meta = pd.DataFrame({"colname": potential_cols})
        meta["year"] = meta["colname"].str.slice(-4).astype(int)
        meta["var_base"] = meta["colname"].str.replace(self.re_year_suffix, "", regex=True)
        return meta
    
    def raster_to_spatial_df(self, raster_path, process_type)-> gpd.GeoDataFrame:
        """ Convert a raster file to a GeoDataFrame by extracting shapes and their geometries."""   

        print(f'Creating {process_type} mask data frame..')

        open_path = str(raster_path)

        with rasterio.open(open_path) as src:
            mask = src.read(1, out_dtype='uint8')

            if process_type == 'prediction':
                valid_mask = mask == 1
            elif process_type == 'training':
                valid_mask = mask == 2

            shapes_gen = shapes(mask, valid_mask, transform=src.transform)  

            gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, _ in shapes_gen]}, crs=src.crs)
            gdf = gdf.to_crs("EPSG:32617")   

            masks_dir_conf = get_config_item('MASK', 'MASKS_DIR')
            
            if self.is_aws:
                bucket = get_config_item('S3', 'BUCKET_NAME')
                mask_path = f"s3://{bucket}/{masks_dir_conf}/{process_type}_mask.parquet"
            else:
                masks_dir = Path(masks_dir_conf)
                mask_path = masks_dir / f"{process_type}_mask.parquet"

            print(f"Saving {process_type} mask GeoDataFrame to: {mask_path}")    

            gdf.to_parquet(str(mask_path))

            return gdf
        
    def create_subgrids(self, mask_gdf, output_dir, process_type)-> None:
        """ Create subgrids layer by intersecting grid tiles with the mask geometries"""        
        
        # Ensure we read the parquet from S3 properly
        mask_gdf_path = str(mask_gdf)
        print(f"----- Reading mask GeoDataFrame from: {mask_gdf_path}")

        grid_gpkg = get_config_item('MODEL', 'SUBGRIDS')
        
        print(f"Preparing {process_type} sub-grids...")

        mask_gdf_df = gpd.read_parquet(mask_gdf_path, filesystem=self.fs if self.is_aws else None)
        combined_geometry = mask_gdf_df.union_all()
        mask_gdf_df = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf_df.crs)

        sub_grids = gpd.read_file(grid_gpkg, layer='prediction_subgrid').to_crs(mask_gdf_df.crs)

        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf_df, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
        
        # Output handling
        if self.is_aws:
             output_path = f"{str(output_dir).rstrip('/')}/{process_type}_intersecting_subgrids.gpkg"
        else:
             output_path = os.path.join(output_dir, f"{process_type}_intersecting_subgrids.gpkg")

        intersecting_sub_grids.to_file(output_path, driver="GPKG") 

        return
    