"""Class for data acquisition and preprocessing of model data"""

import os
import re
import pathlib
import yaml
import warnings
import tempfile
from typing import List, Tuple, Literal
from pathlib import Path
from rasterio.features import shapes, geometry_mask # <--- Added geometry_mask for clipping smoothed edges
from rasterio.warp import transform_bounds # <--- Added to transform raster bounds to target CRS
from shapely.geometry import shape
import s3fs
from scipy.ndimage import convolve, uniform_filter # <--- Added for TSM smoothing

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import dask
from dask.distributed import Client
from osgeo import gdal
from shapely.geometry import Point, box
from upath import UPath 

from hydro_health.helpers.tools import get_config_item, get_environment
from hydro_health.engines.CreateSeabedTerrainLayerEngine import CreateSeabedTerrainLayerEngine

# GDAL Configuration and S3 Network Optimizations
os.environ["GDAL_CACHEMAX"] = "64"
os.environ["GDAL_HTTP_MAX_RETRY"] = "5"            # Retry failed HTTP requests up to 5 times
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"          # Wait 3 seconds between retries
os.environ["AWS_MAX_CONNECTIONS"] = "32"           # Increase S3 connection pool for Dask workers
os.environ["VSI_CACHE"] = "TRUE"                   # Enable VSI read cache to reduce redundant network calls
os.environ["VSI_CACHE_SIZE"] = "50000000"          # 50 MB cache limit


class ModelDataPreProcessor:
    """Class for parallel preprocessing all model data"""

    def __init__(self, overwrite: bool = False, pilot_mode: bool=False):
        self.target_crs = "EPSG:32617"
        self.target_res = 8
        self.pilot_mode = pilot_mode
        self.overwrite = overwrite

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
        """Creates unified UPath objects that work both locally and on S3."""
        # Determine the base prefix depending on the environment
        prefix = f"s3://{get_config_item('S3', 'BUCKET_NAME', pilot_mode=self.pilot_mode)}/" if self.is_aws else ""
        
        self.mask_prediction_pq = UPath(f"{prefix}{get_config_item('MASK', 'PREDICTION_MASK_PQ', pilot_mode=self.pilot_mode)}")
        self.mask_training_pq = UPath(f"{prefix}{get_config_item('MASK', 'TRAINING_MASK_PQ', pilot_mode=self.pilot_mode)}")
        self.grid_gpkg = UPath(f"{prefix}{get_config_item('MODEL', 'SUBGRIDS', pilot_mode=self.pilot_mode)}")
        self.pred_mask_path = UPath(f"{prefix}{get_config_item('MASK', 'MASK_PRED_PATH', pilot_mode=self.pilot_mode)}")
        self.train_mask_path = UPath(f"{prefix}{get_config_item('MASK', 'MASK_TRAINING_PATH', pilot_mode=self.pilot_mode)}")
        self.preprocessed_dir = UPath(f"{prefix}{get_config_item('MODEL', 'PREPROCESSED_DIR', pilot_mode=self.pilot_mode)}")
        self.prediction_out_dir = UPath(f"{prefix}{get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR', pilot_mode=self.pilot_mode)}")
        self.training_out_dir = UPath(f"{prefix}{get_config_item('MODEL', 'TRAINING_OUTPUT_DIR', pilot_mode=self.pilot_mode)}")
        self.training_tiles_dir = UPath(f"{prefix}{get_config_item('MODEL', 'TRAINING_TILES_DIR', pilot_mode=self.pilot_mode)}")
        self.prediction_tiles_dir = UPath(f"{prefix}{get_config_item('MODEL', 'PREDICTION_TILES_DIR', pilot_mode=self.pilot_mode)}")
        self.uncombined_lidar_dir = UPath(f"{prefix}{get_config_item('MODEL', 'UNCOMBINED_LIDAR_DIR', pilot_mode=self.pilot_mode)}")
        self.subgrid_paths = {
            'training': UPath(f"{prefix}{get_config_item('MODEL', 'TRAINING_SUB_GRIDS', pilot_mode=self.pilot_mode)}"),
            'prediction': UPath(f"{prefix}{get_config_item('MODEL', 'PREDICTION_SUB_GRIDS', pilot_mode=self.pilot_mode)}")
        }

        self.preprocessed_subdirs = {
            # 'bluetopo': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'BLUETOPO', pilot_mode=self.pilot_mode)}"),
            'hurricane': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'HURRICANE', pilot_mode=self.pilot_mode)}"),
            # 'lidar': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'LIDAR', pilot_mode=self.pilot_mode)}"),
            # 'sediment': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'SEDIMENT', pilot_mode=self.pilot_mode)}"),
            'tsm': UPath(f"{prefix}{get_config_item('PREPROCESSED', 'TSM', pilot_mode=self.pilot_mode)}")
        }

    def _load_exclusion_config(self) -> set:
        """Loads dataset exclusion keys from YAML config."""
        try:
            inputs_root = pathlib.Path(__file__).parents[4] / 'inputs'
            config_path = inputs_root / 'lookups' / 'ER_3_lidar_data_config.yaml'
            
            if not config_path.exists():
                print(f"[WARNING] Exclusion config path not found: {config_path}")
                return set()

            with open(config_path, 'r') as file:
                config_data = yaml.safe_load(file)

            excluded = {
                key for key, data in config_data.get('EcoRegion-3', {}).items()
                if data.get('use') is False
            }
            
            if excluded:
                print(f"[INFO] Loaded {len(excluded)} exclusion keys from config.")
            
            return excluded
        except Exception as e:
            print(f"[ERROR] Loading exclusion config failed: {e}")
            return set()

    def process(self) -> None:
        """Main function to process model data."""    
        # TODO use the base engine function for this, change the memory limit for AWS
        client = Client(
            n_workers=8, 
            threads_per_worker=2, 
            memory_limit="32GB"
        )
        print(f"\n[INFO] Starting preprocessing workflow...")
        print(f"[INFO] Dask Dashboard: {client.dashboard_link}")

        self.create_file_paths()

        self.raster_to_spatial_df(self.pred_mask_path, process_type='prediction')
        self.raster_to_spatial_df(self.train_mask_path, process_type='training')


        self.create_subgrids(mask_gdf=self.mask_prediction_pq, output_path=self.subgrid_paths['prediction'], process_type='prediction')
        self.create_subgrids(mask_gdf=self.mask_training_pq, output_path=self.subgrid_paths['training'], process_type='training')

        try:
            mask_pred_gdf = gpd.read_parquet(str(self.mask_prediction_pq))
            mask_train_gdf = gpd.read_parquet(str(self.mask_training_pq))

            mask_pred_union = mask_pred_gdf.union_all()
            mask_train_union = mask_train_gdf.union_all()

            # Scatter the single geometry objects to ensure they stay as single unified objects
            mask_future_pred = client.scatter([mask_pred_union], broadcast=True)[0]
            mask_future_train = client.scatter([mask_train_union], broadcast=True)[0]

            # Inside your parallel workers, use the single geometry directly to mask the rasters
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
        input_directory = UPath(input_directory)
        
        if not self.is_aws:
            self.uncombined_lidar_dir.mkdir(parents=True, exist_ok=True)

        # --- 1. Identify Existing Outputs ---
        potential_files = list(self.prediction_out_dir.rglob("*"))
        existing_pred_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        print(f"\n[INFO] Checking existing outputs to prevent redundant processing...")
        print(f"  -> Found {len(existing_pred_outputs)} files in prediction output directory.")

        # Get existing uncombined outputs - changed to .rglob("*")
        potential_files = list(self.uncombined_lidar_dir.rglob("*"))
        existing_uncombined_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }
        print(f"  -> Found {len(existing_uncombined_outputs)} files in uncombined lidar directory.")

        all_existing_outputs = existing_pred_outputs.union(existing_uncombined_outputs)

        # Looks for preprocessed lidar, bluetopo, sediment, TSM, and hurricane rasters in the preprocessed directory and all subdirectories
        potential_files = []
        print(f"\n[INFO] Scanning for preprocessed input rasters in: {self.preprocessed_dir}")

        for data_type, directory in self.preprocessed_subdirs.items():
            # 1. Search for TIFF files in this specific directory
            found_files = [
                f for f in directory.rglob("*") 
                if f.suffix.lower() in {'.tif', '.tiff'}
            ]
            print(f"  -> {data_type.capitalize()} directory: Found {len(found_files)} files.")
            
            # 2. Check if this specific directory came up empty
            if not found_files:
                raise RuntimeError(
                    f"CRITICAL ERROR: Missing data for '{data_type}'. "
                    f"No .tif files were found in {directory}."
                )
                
            # 3. If files exist, add them to our main list
            potential_files.extend(found_files)

        print(f"\n[INFO] Found {len(potential_files)} total potential prediction files in input directories.")

        

        excluded_folders = {'filled_tifs', 'combined_lidar_tifs'} # Exclude intermediate lidar tifs
        prediction_files = [] 

        # Initialize counters and our tracking lists
        removed_existing = 0
        removed_folders = 0
        files_removed_by_keys = []
        kept_mosaic_files = [] # New list to track our kept "mosaic" files

        for f in potential_files:
            # Condition 1: Is it already in existing outputs?
            if not self.overwrite and f.name in all_existing_outputs:
                removed_existing += 1
                continue
                
            # Condition 2: Does it contain an excluded key?
            if any(key in f.name for key in self.excluded_keys):
                files_removed_by_keys.append(f.name) 
                continue
                
            # Condition 3: Is it in an excluded folder?
            if any(folder in f.parts for folder in excluded_folders):
                removed_folders += 1
                continue
                
            # If it passes all checks, keep it in our main list
            prediction_files.append(f)
            
            # Check if the successfully kept file has "mosaic" in its name
            if "mosaic" in f.name.lower():
                kept_mosaic_files.append(f.name)

        # Print the numerical summary
        print(f"\n[INFO] --- File Filtering Summary ---")
        print(f"  -> Total potential files: {len(potential_files)}")
        print(f"  -> Removed (already existing): {removed_existing}")
        print(f"  -> Removed (excluded keys): {len(files_removed_by_keys)}") 
        print(f"  -> Removed (excluded folders): {removed_folders}")
        print(f"  -> Files kept for processing: {len(prediction_files)}")
        print(f"------------------------------------")

        skip_msg = f" (Skipping {removed_existing} existing)" if not self.overwrite else " (Overwrite enabled)"

        print(f"\n[INFO] Found {len(kept_mosaic_files)} lidar mosaic files to process.")
        print(f"[INFO] Outputting uncombined lidar to: {self.uncombined_lidar_dir}")
        print(f"[INFO] Outputting prediction rasters to: {self.prediction_out_dir}")
        print(f"[INFO] Queuing {len(prediction_files)} prediction files{skip_msg}...")
        
        # --- 3. Queue Prediction Tasks ---
        prediction_tasks = []
        for file_path in prediction_files:
            base_out = self.uncombined_lidar_dir if "mosaic" in file_path.name.lower() else self.prediction_out_dir
            output_path = base_out / file_path.name
            
            task = dask.delayed(self.process_prediction_raster)(
                str(file_path), 
                mask_future_pred, 
                str(output_path)
            )
            prediction_tasks.append(task)

        if prediction_tasks:
            dask.compute(*prediction_tasks)
            print("[SUCCESS] Prediction raster processing complete.")
        else:
            print("[INFO] No new prediction rasters to process.")

        input("\nPress Enter to continue...")
        print("\n[INFO] Running Seabed Terrain Layer Engine...")
        engine = CreateSeabedTerrainLayerEngine()
        engine.process()

        if not self.is_aws:
            self.training_out_dir.mkdir(parents=True, exist_ok=True)

        # --- 4. Gather Training Inputs (from Prediction Outputs) ---
        potential_files = list(self.training_out_dir.rglob("*"))
        existing_train_outputs = {
            f.name for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        }

        potential_files = list(self.prediction_out_dir.rglob("*"))
        training_candidates = [
            f for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'}
        ]

        training_files = [
            f for f in training_candidates
            if (self.overwrite or f.name not in existing_train_outputs)
            and not ('mosaic' in f.name and 'filled' not in f.name)
        ]

        skip_train_msg = f" (Skipping {len(existing_train_outputs)} existing)" if not self.overwrite else " (Overwrite enabled)"
        
        print(f"\n[INFO] Outputting training rasters to: {self.training_out_dir}")
        print(f"[INFO] Queuing {len(training_files)} training files{skip_train_msg}...")

        # --- 5. Queue Training Tasks ---
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
            print("[SUCCESS] Training raster processing complete.")
        else:
            print("[INFO] No new training rasters to process.")

    def process_prediction_raster(self, raster_path, mask_union, output_path) -> None:
        """Reprojects, resamples, and crops a raster for prediction."""
        # Using pathlib to extract name safely from the string
        raster_name = pathlib.Path(raster_path).name.lower()

        try:
            with rasterio.open(raster_path) as src:
                src_nodata = src.nodata
                raster_crs = src.crs
                raster_bounds = src.bounds
        except Exception as e:
            print(f"[ERROR] Could not open {raster_name}. {e}")
            return

        # CRS Reprojection Fix: Transform raster bounds to the target CRS before checking for intersection
        if raster_crs is not None:
            try:
                target_crs_obj = rasterio.crs.CRS.from_string(self.target_crs)
                if raster_crs != target_crs_obj:
                    left, bottom, right, top = transform_bounds(raster_crs, target_crs_obj, *raster_bounds)
                    bounds_geom = box(left, bottom, right, top)
                else:
                    bounds_geom = box(*raster_bounds)
            except Exception as e:
                print(f"[WARNING] Failed to transform bounds for {raster_name}: {e}. Falling back to native bounds.")
                bounds_geom = box(*raster_bounds)
        else:
            bounds_geom = box(*raster_bounds)

        if not mask_union.intersects(bounds_geom):
            print(f"  [SKIP] Mask does not intersect prediction raster {raster_name}.")
            return

        print(f"  [PROCESS] Processing prediction file {raster_name}...")
        should_crop = any(k in raster_name for k in ["tsm", "sed", "hurr"])
        is_tsm = "tsm" in raster_name or "strength" in raster_name # Identify if this is a TSM or hurr strength rasters that need smoothed

        self._warp_to_cutline(
            raster_path, 
            output_path, 
            mask_geometry=mask_union, 
            dst_crs=self.target_crs, 
            x_res=self.target_res, 
            y_res=self.target_res,
            crop_to_cutline=should_crop,
            src_nodata=src_nodata,
            apply_tsm_smoothing=is_tsm # Pass the flag to trigger smoothing
        )

    def process_training_raster(self, raster_path, mask_union, output_path) -> None:
        """Process a training raster by clipping it with a mask."""
        raster_name = pathlib.Path(raster_path).name.lower()

        try:
            with rasterio.open(raster_path) as src:
                src_nodata = src.nodata
                raster_crs = src.crs
                raster_bounds = src.bounds

            # CRS Reprojection Fix: Transform raster bounds to the target CRS before checking for intersection
            if raster_crs is not None:
                try:
                    target_crs_obj = rasterio.crs.CRS.from_string(self.target_crs)
                    if raster_crs != target_crs_obj:
                        left, bottom, right, top = transform_bounds(raster_crs, target_crs_obj, *raster_bounds)
                        bounds_geom = box(left, bottom, right, top)
                    else:
                        bounds_geom = box(*raster_bounds)
                except Exception as e:
                    print(f"[WARNING] Failed to transform bounds for {raster_name}: {e}. Falling back to native bounds.")
                    bounds_geom = box(*raster_bounds)
            else:
                bounds_geom = box(*raster_bounds)

            if not mask_union.intersects(bounds_geom):
                print(f"  [SKIP] Mask does not intersect training raster {raster_name}.")
                return

            print(f"  [PROCESS] Processing training file {raster_name}...")
            
            self._warp_to_cutline(
                raster_path,
                output_path,
                mask_geometry=mask_union,
                src_nodata=src_nodata,
                dst_nodata=np.nan
            )
        except Exception as e:
            print(f"[ERROR] Error processing {raster_name}: {e}")

    def _warp_to_cutline(self, src_path, dst_path, mask_geometry, **kwargs):
        """Helper to handle GDAL Warp boilerplate and in-memory cutlines."""
        
        # --- 1. PREPARE PATHS FOR GDAL ---
        src_str = str(src_path)
        dst_str = str(dst_path)

        # GDAL uses /vsis3/ instead of s3:// for READING
        if self.is_aws and src_str.startswith('s3://'):
            src_str = src_str.replace('s3://', '/vsis3/')

        # --- 2. SETUP TEMP OUTPUT PATH FOR AWS ---
        if self.is_aws:
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_out:
                gdal_dst_str = tmp_out.name
        else:
            gdal_dst_str = dst_str

        # --- 3. SETUP CUTLINE ---
        # Use a physical temporary file to avoid Fiona/Pyogrio compatibility issues with /vsimem/
        with tempfile.NamedTemporaryFile(suffix='.geojson', delete=False) as tmp:
            cutline_path = tmp.name
        
        try:
            gdf = gpd.GeoDataFrame(geometry=[mask_geometry], crs=self.target_crs)
            gdf.to_file(cutline_path, driver='GeoJSON')
        except Exception as e:
            print(f"[ERROR] Creating cutline for {os.path.basename(src_str)} failed: {e}")
            if os.path.exists(cutline_path):
                os.remove(cutline_path)
            if self.is_aws and os.path.exists(gdal_dst_str):
                os.remove(gdal_dst_str)
            return

        # --- 4. CONFIGURE WARP ---
        warp_opts = {
            'cutlineDSName': cutline_path,
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE'],
            'creationOptions': [
                'TILED=YES', 
                'BLOCKXSIZE=512',       # NEW: Sets tile width to 512 pixels
                'BLOCKYSIZE=512',       # NEW: Sets tile height to 512 pixels
                'COMPRESS=LZW',         # (You can also try 'COMPRESS=ZSTD' here)
                'BIGTIFF=YES',
                'NUM_THREADS=ALL_CPUS'  # NEW: Speeds up the compression process
            ],
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
        
        apply_tsm_smoothing = kwargs.pop('apply_tsm_smoothing', False)
        
        # --- 5. EXECUTE WARP ---
        try:
            # Output directly to our safe local file
            ds = gdal.Warp(gdal_dst_str, src_str, **warp_opts)

            # --- 6. OPTIONAL TSM SMOOTHING PASS ---
            if apply_tsm_smoothing and ds is not None:
                print(f"  [SMOOTHING] Applying focal mean smoothing to {src_str} raster...")

                # IMPORTANT: Close the dataset returned by gdal.Warp to flush writes
                # and reopen it in Update mode to safely write into the LZW-compressed format.
                ds = None
                ds = gdal.Open(gdal_dst_str, gdal.GA_Update)
                
                if ds is not None:
                    band = ds.GetRasterBand(1)
                    array = band.ReadAsArray()
                    
                    if array is not None:
                        array = array.astype(np.float32)
                        nodata = band.GetNoDataValue()
                        
                        # Fallback if no specific nodata returned
                        if nodata is None:
                            nodata = warp_opts.get('dstNodata', warp_opts.get('srcNodata', -9999.0))

                        # Convert your 2000 Map Units into a pixel radius
                        pixel_size = ds.GetGeoTransform()[1]
                        radius_pixels = int(2000 / pixel_size)
                        size = radius_pixels * 2 + 1

                        # Safe NoData mask creation that handles np.nan
                        if pd.isna(nodata):
                            valid_mask = (~np.isnan(array)).astype(np.float32)
                            array[np.isnan(array)] = 0  # Temporarily zero out NoData
                        else:
                            valid_mask = (array != nodata).astype(np.float32)
                            array[array == nodata] = 0  # Temporarily zero out NoData

                        # Convolve data and mask, then divide to get the true mean ignoring NoData.
                        # We use uniform_filter (O(1) sliding square window) instead of dense convolve (O(N*M))
                        # to fix MemoryError / KilledWorker timeout issues on extremely large rasters.
                        smoothed = uniform_filter(array, size=size, mode='constant', cval=0.0)
                        weights = uniform_filter(valid_mask, size=size, mode='constant', cval=0.0)

                        with np.errstate(divide='ignore', invalid='ignore'):
                            final_array = np.where(weights > 0, smoothed / weights, nodata)

                        # Re-mask to the input geometry mask. The smoothing filter extends data edges 
                        # outwards; this strictly clips the values back to the original prediction mask bounds.
                        gt = ds.GetGeoTransform()
                        transform = rasterio.transform.Affine.from_gdal(*gt)
                        
                        out_of_bounds_mask = geometry_mask(
                            [mask_geometry],
                            out_shape=final_array.shape,
                            transform=transform,
                            invert=False  # Returns True for pixels OUTSIDE the geometry
                        )
                        
                        final_array[out_of_bounds_mask] = nodata

                        # Write array directly back to the warp dataset output
                        band.WriteArray(final_array)
                        band.FlushCache()

            # Explicitly force garbage collection to flush and close the file
            ds = None 
            
            if self.is_aws:
                # Bypass /vsis3/ entirely and use our robust s3fs client to upload the file
                self.fs.put(gdal_dst_str, dst_str)
                print(f"  [SUCCESS] Uploaded raster to S3: {dst_str}")
            else:
                print(f"  [SUCCESS] Saved raster locally to: {dst_str}")

        except Exception as e:
            print(f"[ERROR] GDAL Warp/Upload failed for {src_str}: {e}")
        finally:
            try:
                # Clean up all explicitly created temporary files
                if os.path.exists(cutline_path):
                    os.remove(cutline_path)
                if self.is_aws and os.path.exists(gdal_dst_str):
                    os.remove(gdal_dst_str)
            except Exception as e:
                # Failing to unlink shouldn't kill the distributed worker
                print(f"[WARNING] Failed to cleanup temp files: {e}")

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type) -> None:
        """Clip raster files by tile and save data."""
        
        sub_grid_path = self.subgrid_paths.get(data_type)
        if not sub_grid_path:
            print(f"[ERROR] No subgrid path defined for {data_type}")
            return
        
        try:
             sub_grids = gpd.read_file(str(sub_grid_path))
        except Exception as e:
             print(f"[ERROR] Reading subgrids from {sub_grid_path} failed: {e}")
             return
        
        print(f"\n[INFO] Clipping {data_type} rasters into {sub_grids.shape[0]} sub-grid tiles...")
        print(f"[INFO] Outputting clipped {data_type} tiles to: {output_dir}")
        
        tasks = []
        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id']

            # if "BH4SD56H" not in tile_name:
            #     continue
            
            output_folder = output_dir / tile_name

            gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, raster_dir)
            ungridded_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, raster_dir)
            
            tasks.append(
                self.save_combined_data(gridded_task, ungridded_task, output_folder, data_type, tile_id=tile_name)
            )

        results_list = dask.compute(*tasks)

        print(f"\n[INFO] Combining {data_type} tile results and calculating statistics...")
        final_results_df = pd.concat(results_list, ignore_index=True)
        
        output_csv_path = output_dir.parent / f"year_pair_nan_counts_{data_type}.csv"
        
        final_results_df.to_csv(str(output_csv_path), index=False, na_rep='NA')
        print(f"[SUCCESS] Statistics saved to: {output_csv_path}")

    def subtile_process_gridded(self, sub_grid, raster_dir) -> pd.DataFrame:
        """Process gridded rasters for a single tile."""
        original_tile = sub_grid['original_tile']
                
        # UPath natively rglobs through AWS without needing s3fs.glob manual string parsing
        raster_dir_upath = UPath(raster_dir)
        potential_files = list(raster_dir_upath.rglob("*"))

        raster_files = [
            f for f in potential_files
            if f.suffix.lower() in {'.tif', '.tiff'} 
            and original_tile in f.name
        ]

        tile_extent = sub_grid.geometry.bounds
        dfs = []
        for file in raster_files:
            # str(file) gives full S3 URL or local path automatically
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
        
        combined_data = gpd.GeoDataFrame(
            combined_data, 
            geometry=gpd.points_from_xy(combined_data['X'], combined_data['Y']), 
            crs=self.target_crs
        )
        
        return combined_data

    def subtile_process_ungridded(self, sub_grid, raster_dir) -> pd.DataFrame:
        """Process ungridded rasters for a single tile."""
        raster_dir_upath = UPath(raster_dir)
        dfs = []
        tile_extent = sub_grid.geometry.bounds

        for pattern in self.static_patterns:
            current_files = list(raster_dir_upath.rglob(f"*{pattern}*.tif"))

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
            print(f"[ERROR] Reading raster window from {raster_path} failed: {e}")
            return pd.DataFrame()

    @dask.delayed
    def save_combined_data(self, gridded_df, ungridded_df, output_folder, data_type, tile_id) -> pd.DataFrame:
        """Combine dataframes and save to parquet."""
        if gridded_df is None or gridded_df.empty:
            return pd.DataFrame()

        output_folder_path = UPath(output_folder)
        
        if not self.is_aws: 
            output_folder_path.mkdir(parents=True, exist_ok=True)
            
        output_path = output_folder_path / f"{tile_id}_{data_type}_clipped_data.parquet"
        save_path = str(output_path)

        if ungridded_df is not None and not ungridded_df.empty:
            combined = pd.merge(gridded_df, ungridded_df, on=['X', 'Y'], how='left')
        else:
            combined = gridded_df

        combined.to_parquet(save_path, engine="pyarrow", index=False)
        print(f"  [SUCCESS] Saved combined tile data to: {save_path}")
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
        print(f"\n[INFO] Starting Long Format Transformation (Batch: {mode})...")

        file_suffix = f"_{mode}_clipped_data.parquet"

        base_dir_upath = UPath(base_dir)
        files_to_process = list(base_dir_upath.rglob(f"*{file_suffix}"))

        if not files_to_process:
            print(f"[WARNING] No files found for {mode} transformation in {base_dir}")
            return

        print(f"[INFO] Outputting transformed {mode} long-format tiles to: {base_dir}")
        print(f"[INFO] Queueing {len(files_to_process)} tiles...")

        tasks = []
        for fp in files_to_process:
            tasks.append(dask.delayed(self._transform_tile_task)(str(fp), mode))

        results = dask.compute(*tasks)

        success = sum(1 for r in results if r.startswith("Success"))
        failed = len(results) - success
        
        print(f"\n[SUCCESS] Transformation Complete. Success: {success}, Failed: {failed}")
        if failed > 0:
            print("[ERROR] Transformation Errors:\n" + "\n".join([r for r in results if not r.startswith("Success")]))

    def _transform_tile_task(self, f_path: str, mode: Literal["training", "prediction"]) -> str:
        """Dask Worker: Reads file -> Calls specific processor -> Returns status."""
        try:
            tile_name = os.path.basename(f_path).split("_")[0]
            output_dir = os.path.dirname(f_path)

            try:
                gdf = gpd.read_parquet(f_path)
            except Exception:
                df = pd.read_parquet(f_path)
                geometry_col = 'geometry' if 'geometry' in df.columns else None
                gdf = gpd.GeoDataFrame(df, geometry=geometry_col)

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
            
            # Replaced manual path joining with UPath!
            out_path = str(UPath(output_dir) / out_name)
            
            pair_gdf[final_cols].to_parquet(out_path, index=None)
            print(f"  [SUCCESS] Saved training long-format tile to: {out_path}")
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
            
            # Replaced manual path joining with UPath!
            out_path = str(UPath(output_dir) / out_name)
            
            pair_gdf.to_parquet(out_path, index=None)
            print(f"  [SUCCESS] Saved prediction long-format tile to: {out_path}")
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

        print(f"\n[INFO] Creating {process_type} mask GeoDataFrame from: {raster_path}")

        open_path = str(raster_path)

        with rasterio.open(open_path) as src:
            mask = src.read(1, out_dtype='uint8')

            if process_type == 'prediction':
                valid_mask = mask == 1
            elif process_type == 'training':
                if self.pilot_mode:
                    valid_mask = mask == 1
                else:
                    # TODO we need to double check what er3 mask uses
                    valid_mask = mask == 2

            shapes_gen = shapes(mask, valid_mask, transform=src.transform)  

            gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, _ in shapes_gen]}, crs=src.crs)
            gdf = gdf.to_crs(self.target_crs)   

            pilot_mode = 'pilot'
            masks_dir_conf = get_config_item('MASK', 'MASKS_DIR', pilot_mode=self.pilot_mode)
            prefix = f"s3://{get_config_item('S3', 'BUCKET_NAME', pilot_mode=self.pilot_mode)}/" if self.is_aws else ""
            mask_path = UPath(f"{prefix}{masks_dir_conf}/{process_type}_mask_pilot.parquet")
            
            if not self.is_aws:
                mask_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] Saving {process_type} mask GeoDataFrame to: {mask_path}")    

            gdf.to_parquet(str(mask_path))

            return gdf
        
    def create_subgrids(self, mask_gdf, output_path, process_type) -> None:
        """ Create subgrids layer by intersecting grid tiles with the mask geometries"""        
        
        mask_gdf_path = str(mask_gdf)
        print(f"\n[INFO] Preparing {process_type} sub-grids...")
        print(f"  -> Reading mask GeoDataFrame from: {mask_gdf_path}")

        mask_gdf_df = gpd.read_parquet(mask_gdf_path, filesystem=self.fs if self.is_aws else None)
        combined_geometry = mask_gdf_df.union_all()
        mask_gdf_df = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf_df.crs)

        # FIX A: Use /vsis3/ for reading GPKG via GeoPandas/Fiona on AWS
        grid_gpkg_str = str(self.grid_gpkg)
        if self.is_aws and grid_gpkg_str.startswith("s3://"):
            grid_gpkg_str = grid_gpkg_str.replace("s3://", "/vsis3/")

        sub_grids = gpd.read_file(grid_gpkg_str, layer='prediction_subgrid').to_crs(mask_gdf_df.crs)

        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf_df, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
        
        if self.is_aws:
            with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
                local_tmp_path = tmp.name
                
            print(f"  -> Writing GPKG locally to {local_tmp_path} before uploading...")
            intersecting_sub_grids.to_file(local_tmp_path, driver="GPKG")
            
            print(f"  -> Uploading subgrids to S3: {output_path}")
            self.fs.put(local_tmp_path, str(output_path))
            os.remove(local_tmp_path)
        else:
            # FIX B: Ensure local directory exists before saving
            output_upath = UPath(output_path)
            output_upath.parent.mkdir(parents=True, exist_ok=True)
            intersecting_sub_grids.to_file(str(output_upath), driver="GPKG") 

        print(f"[SUCCESS] Successfully saved {process_type} subgrids to: {output_path}")
        return 
    