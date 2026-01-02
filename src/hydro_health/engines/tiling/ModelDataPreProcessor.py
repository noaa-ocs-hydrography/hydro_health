"""Class for data acquisition and preprocessing of model data"""

import os
import re
import pathlib
import yaml
import warnings
from typing import List, Tuple, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import dask
from dask.distributed import Client
from osgeo import gdal
from shapely.geometry import Point, box

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.CreateSeabedTerrainLayerEngine import CreateSeabedTerrainLayerEngine

os.environ["GDAL_CACHEMAX"] = "64"


class ModelDataPreProcessor:
    """Class for parallel preprocessing all model data"""

    def __init__(self):
        self.target_crs = "EPSG:32617"
        self.target_res = 8

        self.mask_prediction_pq = pathlib.Path(get_config_item('MASK', 'PREDICTION_MASK_PQ'))
        self.mask_training_pq = pathlib.Path(get_config_item('MASK', 'TRAINING_MASK_PQ'))
        self.preprocessed_dir = pathlib.Path(get_config_item('MODEL', 'PREPROCESSED_DIR'))
        
        self.prediction_out_dir = pathlib.Path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'))
        self.training_out_dir = pathlib.Path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'))
        
        self.training_tiles_dir = pathlib.Path(get_config_item('MODEL', 'TRAINING_TILES_DIR'))
        self.prediction_tiles_dir = pathlib.Path(get_config_item('MODEL', 'PREDICTION_TILES_DIR'))
        
        # Directory for uncombined mosaic files
        self.uncombined_lidar_dir = pathlib.Path(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\pre_processed\uncombined_lidar_tifs")

        self.subgrid_paths = {
            'training': pathlib.Path(get_config_item('MODEL', 'TRAINING_SUB_GRIDS')),
            'prediction': pathlib.Path(get_config_item('MODEL', 'PREDICTION_SUB_GRIDS'))
        }

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
        client = Client(
            n_workers=4, 
            threads_per_worker=2, 
            memory_limit="32GB"
        )
        print(f"Dask Dashboard: {client.dashboard_link}")

        try:
            mask_pred_gdf = gpd.read_parquet(self.mask_prediction_pq)
            mask_future_pred = client.scatter(mask_pred_gdf.unary_union, broadcast=True)

            mask_train_gdf = gpd.read_parquet(self.mask_training_pq)
            mask_future_train = client.scatter(mask_train_gdf.unary_union, broadcast=True)

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

            print("\nStarting Long Format Transformation...")
            
            # self.batch_long_format_transformation(base_dir=self.training_tiles_dir, mode="training")

        finally:
            client.close()

    def parallel_processing_rasters(self, input_directory, mask_future_pred, mask_future_train) -> None:
        """Process prediction and training rasters in parallel using Dask."""
        
        input_directory = pathlib.Path(input_directory)
        
        # Ensure the uncombined directory exists
        self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)
                
        existing_pred_outputs = {
            f.name for f in self.prediction_out_dir.glob("*") 
            if f.suffix.lower() in {'.tif', '.tiff'}
        }

        # Also check uncombined directory for existing files to avoid re-processing
        existing_uncombined_outputs = {
            f.name for f in self.uncombined_lidar_dir.glob("*") 
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        
        all_existing_outputs = existing_pred_outputs.union(existing_uncombined_outputs)
        
        # Explicitly exclude these folders from the search
        excluded_folders = {'lidar_filled_tifs', 'uncombined_lidar_tifs'}

        prediction_files = [
            f for f in input_directory.rglob("*")
            if f.suffix.lower() in {'.tif', '.tiff'}
            and f.name not in all_existing_outputs
            and not any(key in f.name for key in self.excluded_keys)
            and not any(folder in f.parts for folder in excluded_folders)
        ]

        print(f"Processing {len(prediction_files)} prediction files (Skipping {len(all_existing_outputs)} existing)...")

        prediction_tasks = []
        for file_path in prediction_files:
            # Route files with 'mosaic' in the name to the uncombined directory
            if "mosaic" in file_path.name.lower():
                output_path = self.uncombined_lidar_dir / file_path.name
            else:
                output_path = self.prediction_out_dir / file_path.name
            
            task = dask.delayed(self.process_prediction_raster)(
                str(file_path), 
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

        self.training_out_dir.mkdir(parents=True, exist_ok=True)
        existing_train_outputs = {
            f.name for f in self.training_out_dir.glob("*") 
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        
        training_candidates = [
            f for f in self.prediction_out_dir.rglob("*")
            if f.suffix.lower() in {'.tif', '.tiff'}
        ]
        
        training_files = [
            f for f in training_candidates
            if f.name not in existing_train_outputs
            and not ('mosaic' in f.name and 'filled' not in f.name)
        ]

        print(f"Processing {len(training_files)} training files (Skipping {len(existing_train_outputs)} existing)...")

        training_tasks = []
        for file_path in training_files:
            output_path = self.training_out_dir / file_path.name
            
            task = dask.delayed(self.process_training_raster)(
                str(file_path), 
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
        raster_path = pathlib.Path(raster_path)
        raster_name = raster_path.name.lower()

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
        raster_name = os.path.basename(src_path)
        in_memory_cutline = f'/vsimem/cutline_{os.getpid()}_{hash(raster_name)}.geojson'
        
        try:
            gdf = gpd.GeoDataFrame(geometry=[mask_geometry], crs=self.target_crs)
            gdf.to_file(in_memory_cutline, driver='GeoJSON')
        except Exception as e:
            print(f"Error creating cutline for {raster_name}: {e}")
            return

        warp_opts = {
            'cutlineDSName': in_memory_cutline,
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE'],
            'creationOptions': ['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES'],
            'multithread': True,
            'warpMemoryLimit': 2048,
        }
        
        if 'dst_crs' in kwargs: warp_opts['dstSRS'] = kwargs.pop('dst_crs')
        if 'x_res' in kwargs: warp_opts['xRes'] = kwargs.pop('x_res')
        if 'y_res' in kwargs: warp_opts['yRes'] = kwargs.pop('y_res')
        if 'crop_to_cutline' in kwargs: warp_opts['cropToCutline'] = kwargs.pop('crop_to_cutline')
        if 'src_nodata' in kwargs: warp_opts['srcNodata'] = kwargs.pop('src_nodata')
        if 'dst_nodata' in kwargs: warp_opts['dstNodata'] = kwargs.pop('dst_nodata')
        
        if 'resampleAlg' not in kwargs: warp_opts['resampleAlg'] = 'bilinear'

        try:
            gdal.Warp(dst_path, src_path, **warp_opts)
        except Exception as e:
            print(f"Error during gdal.Warp for {raster_name}: {e}")
            if os.path.exists(dst_path):
                try: os.remove(dst_path)
                except OSError: pass
        finally:
            gdal.Unlink(in_memory_cutline)

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type) -> None:
        """Clip raster files by tile and save data."""
        print(f" - Clipping {data_type} rasters by tile...")
        
        sub_grid_path = self.subgrid_paths.get(data_type)
        if not sub_grid_path:
            print(f"Error: No subgrid path defined for {data_type}")
            return

        sub_grids = gpd.read_file(sub_grid_path)
        
        print(f"Number of tiles: {sub_grids.shape[0]}")
        
        tasks = []
        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id']

            if "BH4SD56H" not in tile_name:
                continue
            output_folder = pathlib.Path(output_dir) / tile_name

            gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, raster_dir)
            ungridded_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, raster_dir)
            
            tasks.append(
                self.save_combined_data(gridded_task, ungridded_task, output_folder, data_type, tile_id=tile_name)
            )

        results_list = dask.compute(*tasks)

        print("Combining results...")
        final_results_df = pd.concat(results_list, ignore_index=True)
        output_csv_path = pathlib.Path(output_dir).parent / f"year_pair_nan_counts_{data_type}.csv"
        final_results_df.to_csv(output_csv_path, index=False, na_rep='NA')
        print(f"Stats saved to: {output_csv_path}")

    def subtile_process_gridded(self, sub_grid, raster_dir) -> pd.DataFrame:
        """Process gridded rasters for a single tile."""

        original_tile = sub_grid['original_tile']
        
        raster_dir = pathlib.Path(raster_dir)
        raster_files = [
            f for f in raster_dir.rglob("*") 
            if f.suffix in {'.tif', '.tiff'} and original_tile in f.name
        ]

        tile_extent = sub_grid.geometry.bounds
        dfs = []
        for file in raster_files:
            df = self._extract_raster_to_df(file, tile_extent)
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

        raster_dir = pathlib.Path(raster_dir)
        dfs = []
        tile_extent = sub_grid.geometry.bounds

        for pattern in self.static_patterns:
            for file in raster_dir.rglob(f"*{pattern}*.tif"):
                 df = self._extract_raster_to_df(file, tile_extent)
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

        output_folder = pathlib.Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        
        output_path = output_folder / f"{tile_id}_{data_type}_clipped_data.parquet"

        if ungridded_df is not None and not ungridded_df.empty:
            combined = pd.merge(gridded_df, ungridded_df, on=['X', 'Y'], how='left')
        else:
            combined = gridded_df

        combined.to_parquet(output_path, engine="pyarrow", index=False)
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

    def batch_long_format_transformation(self, base_dir: pathlib.Path, mode: Literal["training", "prediction"]):
        """Orchestrator for transforming wide tiles to long year-pair format."""
        print(f"\nStarting Dask Transformation Batch: {mode}")

        base_dir = pathlib.Path(base_dir)
        file_suffix = f"_{mode}_clipped_data.parquet"
        files_to_process = list(base_dir.rglob(f"*{file_suffix}"))

        if not files_to_process:
            print("No files found for transformation.")
            return

        print(f"Queueing {len(files_to_process)} tiles...")

        tasks = [
            dask.delayed(self._transform_tile_task)(str(fp), mode) 
            for fp in files_to_process
        ]

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