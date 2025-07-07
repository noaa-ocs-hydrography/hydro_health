"""Class for processing and tiling model prediction and training data."""

import os
import pathlib
from datetime import datetime
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point


import numpy as np
import pandas as pd
import xarray as xr

import dask
from dask import compute, delayed
from dask.distributed import Client

import geopandas as gpd
import rioxarray
from shapely.geometry import box, shape

import rasterio
from rasterio.enums import Resampling
from rasterio import windows
from rasterio.features import rasterize, shapes
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds, transform as window_transform
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

from hydro_health.helpers.tools import get_config_item

INPUTS = pathlib.Path(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\HHM_Run\ER_3')

class ModelDataProcessor:
    """Class for parallel processing all model data"""

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type):
        """ Clip raster files by tile and save the data in a specified directory.
        :param str raster_dir: directory containing the raster files to be processed
        :param str output_dir: directory to save the output files
        :param str data_type: Specifies the type of data being processed (e.g., "prediction" or "training").
        """        
        # sub_grid_gpkg = rf"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\{data_type}_tiles\intersecting_sub_grids.gpkg"

        sub_grid_gpkg = pathlib.Path(get_config_item('MODEL', 'PREDICTION_SUB_GRIDS'))
        sub_grids = gpd.read_file(sub_grid_gpkg, layer=f'{data_type}_intersecting_subgrids')
        

        tasks = []

        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id'] 
            output_path = os.path.join(output_dir, f"{tile_name}_{data_type}_clipped_data.parquet")

            # Delay each part separately
            gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, raster_dir, output_dir, data_type)
            ungridded_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, raster_dir, output_dir, data_type)

            # Delay the combining + saving
            combined_task = self.save_combined_data(gridded_task, ungridded_task, output_path, data_type)

            tasks.append(combined_task)

        dask.compute(*tasks) 

    @dask.delayed
    def save_combined_data(self, gridded_df, ungridded_df, output_path):
        # logger.info(f"Started processing tile: ")
        combined = pd.concat([gridded_df, ungridded_df], ignore_index=True)
        combined.to_parquet(output_path, engine="pyarrow", index=False)   

    def parallel_processing_rasters(self, input_directory, mask_pred, mask_train):
        """Process prediction and training rasters in parallel using Dask.
        :param gdf mask_pred: Prediction mask GeoDataFrame
        :param gdf mask_train: Training mask GeoDataFrame
        :return: None
        """   

        input_directory = Path(input_directory)
        prediction_files = [
            f for f in input_directory.iterdir()
            if f.is_file() and f.suffix == '.tif' and 'sed' in f.stem
        ]
        # prediction_files = list(Path(input_directory).rglob("*.tif"))
        prediction_out = pathlib.Path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'))

        prediction_tasks = []
        for file in prediction_files:
            print(f'Processing {file}...')
            input_path = os.path.join(input_directory, file)
            output_path = os.path.join(prediction_out, file.name)
            print(output_path)

            prediction_tasks.append(dask.delayed(self.process_raster)(input_path, mask_pred, output_path))

        dask.compute(*prediction_tasks)  
        
        training_files = [f for f in os.listdir(prediction_out) if f.endswith('.tif')]
        training_out = pathlib.Path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'))

        training_tasks  = []
        for file in training_files:
            print(f'Processing {file}...')
            input_path = os.path.join(prediction_out, file)
            output_path = os.path.join(training_out, file)
            training_tasks .append(dask.delayed(self.process_raster)(input_path, mask_train, output_path))

        dask.compute(*training_tasks)    

    def create_subgrids(self, mask_gdf, output_dir, process_type):
        """ Create subgrids layer by intersecting grid tiles with the mask geometries
        :param gdf mask_gdf: GeoDataFrame containing the mask geometries
        :param str output_dir: directory to save the output files
        :return: None
        """        
        
        grid_gpkg = get_config_item('MODEL', 'SUBGRIDS')

        print(f"Preparing {process_type} sub-grids...")
        
        sub_grids = gpd.read_file(grid_gpkg, layer='prediction_subgrid').to_crs(mask_gdf.crs)
        
        # TODO will need to change which valid value it looks for in the mask
        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
        intersecting_sub_grids.to_file(os.path.join(output_dir, f"{process_type}_intersecting_subgrids.gpkg"), driver="GPKG") 
   
    def process(self):
        """ Main function to process model data.
        """        

        subgrids_output_dir = pathlib.Path(get_config_item('MODEL', 'MODEL_SUBGRIDS'))


        # prediction_mask_df = self.raster_to_spatial_df(pathlib.Path(get_config_item('MODEL', 'PREDICTION_MASK')), process_type='prediction')
        mask_prediction_pq = pathlib.Path(get_config_item('MODEL', 'PREDICTION_MASK_PQ'))
        prediction_mask_df = gpd.read_parquet(mask_prediction_pq)
        # training_mask_df = prediction_mask_df
        # training_mask_df = self.raster_to_spatial_df(pathlib.Path(get_config_item('MODEL', 'TRAINING_MASK')), process_type='training')
        mask_training_pq = pathlib.Path(get_config_item('MODEL', 'TRAINING_MASK_PQ'))
        training_mask_df = gpd.read_parquet(mask_training_pq)

        # self.create_subgrids(mask_gdf=prediction_mask_df, output_dir=subgrids_output_dir, process_type = 'prediction')
        # self.create_subgrids(mask_gdf=training_mask_df, output_dir=subgrids_output_dir, process_type = 'training') # 254 seconds

        # cluster = LocalCluster(n_workers=8, threads_per_worker=1)
        # client = Client(cluster)
        client = Client(n_workers=4, threads_per_worker=2, memory_limit="16GB")
        # client = Client(memory_limit="16GB")
        print(f"Dask Dashboard: {client.dashboard_link}")

        preprocessed_dir = pathlib.Path(get_config_item('MODEL', 'PREPROCESSED_DIR'))
        processed_dir = pathlib.Path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'))
        output_dir_pred = pathlib.Path(get_config_item('MODEL', 'PREDICTION_TILES_DIR'))

        self.parallel_processing_rasters(preprocessed_dir, prediction_mask_df, training_mask_df)
        # input_dir_pred = Path(input_directory) / 'model_variables' / 'Prediction' / 'processed'
        # input_dir_train = Path(input_directory) / 'model_variables' / 'Training' / 'processed'
        # output_dir_pred = Path(input_directory) / 'model_variables' / 'Prediction' / 'prediction_tiles'
        # output_dir_train = Path(input_directory) / 'model_variables' / 'Training' / 'training_tiles'

        print(" - Clipping prediction rasters by tile...")
        # self.clip_rasters_by_tile(raster_dir=processed_dir, output_dir=output_dir_pred, data_type="prediction")
        print(" - Clipping training rasters by tile...")
        # self.clip_rasters_by_tile(raster_dir=input_dir_train, output_dir=output_dir_train, data_type="training")
    
        client.close()

    def process_raster(self, raster_path, mask_gdf, output_path, target_crs="EPSG:32617", target_res=8):
            with rasterio.open(raster_path) as src:
                mask_gdf = mask_gdf.to_crs(src.crs)

                # --- 1. Crop raster to bounding box of mask ---
                bounds = mask_gdf.total_bounds  # [minx, miny, maxx, maxy]
                window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
                data = src.read(1, window=window)
                transform = src.window_transform(window)

                # --- 2. Create a rasterized mask array where inside = 1, outside = 0 ---
                mask_array = rasterize(
                    [(geom, 1) for geom in mask_gdf.geometry],
                    out_shape=data.shape,
                    transform=transform,
                    fill=0,
                    dtype='uint8',
                    nodata=0
                )

                # --- 3. Set values outside geometry to NaN ---
                data = np.where(mask_array == 1, data, np.nan)

                # --- 4. Apply bathymetry filter if needed ---
                if 'bathy_' in os.path.basename(raster_path):
                    data = np.where(data <= 0, data, np.nan)

                # --- 5. Reproject ---
                dst_transform, dst_width, dst_height = calculate_default_transform(
                    src.crs, target_crs, data.shape[1], data.shape[0],
                    *bounds, resolution=target_res
                )

                kwargs = src.meta.copy()
                kwargs.update({
                    'crs': target_crs,
                    'transform': dst_transform,
                    'width': dst_width,
                    'height': dst_height,
                    'compress': 'LZW',
                    'tiled': True,
                    'blockxsize': 1024,
                    'blockysize': 1024,
                    'dtype': 'float32',
                    'count': 1,
                    'nodata': -9999
                })

                with rasterio.open(output_path, 'w', **kwargs) as dst:
                    reproject(
                        source=data,
                        destination=rasterio.band(dst, 1),
                        src_transform=transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=-9999,
                        dst_nodata=np.nan
                    )

    def raster_to_spatial_df(self, raster_path, process_type):
        """ Convert a raster file to a GeoDataFrame by extracting shapes and their geometries.

        :param str raster_path: path to the raster file
        :return gdf: GeoDataFrame containing the mask shapes and geometries
        """        

        print(f'Creating {process_type} mask data frame..')

        with rasterio.open(raster_path) as src:
            mask = src.read(1, out_dtype='uint8')

            if process_type == 'prediction':
                valid_mask = mask == 1

            elif process_type == 'training':
                valid_mask = mask == 2

            shapes_gen = shapes(mask, valid_mask, transform=src.transform)

            gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, _ in shapes_gen]}, crs=src.crs)
            gdf = gdf.to_crs("EPSG:32617") 

            masks_dir = Path(get_config_item("MODEL", "MASKS_DIR"))  # this is a folder path
            mask_path = masks_dir / f"{process_type}_mask.parquet"

            gdf.to_parquet(mask_path)

        return gdf
     
    def subtile_process_gridded(self, sub_grid, raster_dir, output_dir):
        """ Process a single tile by clipping gridded Bluetopo and Digital raster data and saving the data in a specified directory.

        :param _type_ sub_grid: sub-grid information containing tile_id and geometry
        :param _type_ raster_dir: input directory containing the raster files to be processed
        :param _type_ output_dir: output directory to save the processed data
        :param _type_ data_type: predicts or training data
        """        
        
        sub_tile_name = sub_grid['tile_id']
        print(f"Processing tile: {sub_tile_name}")
        original_tile = sub_grid['original_tile']  
        print(f"Original tile: {original_tile}")

        raster_files = [
            os.path.join(raster_dir, f)
            for f in os.listdir(raster_dir)
            if f.endswith('.tif') and original_tile in f
        ]

        if not raster_files:
            print(f"No matching raster files found for tile {sub_tile_name}, skipping...")

            return

        tile_extent = sub_grid.geometry.bounds
        
        tile_dir = os.path.join(output_dir, sub_tile_name)
        os.makedirs(tile_dir, exist_ok=True)
            
        # print(f"Processing {data_type} tile: {sub_tile_name}")
        
        clipped_data = []
        for file in raster_files:

            with rasterio.open(file) as src:
                window = src.window(tile_extent[0], tile_extent[1], tile_extent[2], tile_extent[3])
                cropped_r = src.read(1, window=window)  
                transform = src.window_transform(window)   

                mask = cropped_r != src.nodata
                raster_data = np.column_stack(np.where(mask)) 
                raster_values = cropped_r[mask]  

                row_vals = raster_data[:, 0]
                col_vals = raster_data[:, 1]

                xs, ys = rasterio.transform.xy(transform, row_vals, col_vals, offset='center')
                fids = np.ravel_multi_index((row_vals, col_vals), cropped_r.shape)

                raster_df = pd.DataFrame({
                    'X': xs,
                    'Y': ys,
                    'FID': fids,
                    'Value': raster_values,
                    'Raster': os.path.splitext(os.path.basename(file))[0]
                })

                clipped_data.append(raster_df)

        combined_data = pd.concat(clipped_data, axis=0, join='outer', ignore_index=True)
        combined_data_pivot = combined_data.pivot(index=['X', 'Y', 'FID'], columns='Raster', values='Value').reset_index()
        combined_data = combined_data_pivot

        combined_data['geometry'] = [Point(x, y) for x, y in zip(combined_data['X'], combined_data['Y'])]
        combined_data = gpd.GeoDataFrame(combined_data, geometry='geometry', crs='EPSG:32617') 

        # TODO I need a way to deal with the different years that might exist in a raster file and also get corresponding nonstatic ungridded data

        nan_percentage = combined_data['bathy_2006'].isna().mean() * 100
        print(f"Percentage of NaNs in bathy_2006: {nan_percentage:.2f}%")
        
        combined_data['b.change.2004_2006'] = combined_data['bathy_2006'] - combined_data['bathy_2004']
        combined_data['b.change.2006_2010'] = combined_data['bathy_2010'] - combined_data['bathy_2006']
        combined_data['b.change.2010_2015'] = combined_data['bathy_2015'] - combined_data['bathy_2010']
        combined_data['b.change.2015_2022'] = combined_data['bathy_2022'] - combined_data['bathy_2015']

        return combined_data

    def subtile_process_ungridded(self, sub_grid, raster_dir, output_dir):
        """ Process a single tile by clipping non Bluetopo raster data and saving the data in a specified directory.
       
        :param _type_ sub_grid: sub-grid information containing tile_id and geometry
        :param _type_ raster_dir: input directory containing the raster files to be processed
        :param _type_ output_dir: output directory to save the processed data
        :param _type_ data_type: predicts or training data
        """      
        
        sub_tile_name = sub_grid['tile_id']
        print(f"Processing tile: {sub_tile_name}")

        static_data = ['sed_size_raster', 'sed_type_raster']
        clipped_data = []

        for data in static_data:
            raster_files = [
                os.path.join(raster_dir, f)
                for f in os.listdir(raster_dir)
                if f.endswith('.tif') and data in f
            ]

            if not raster_files:
                print(f"No matching raster files found for tile {sub_tile_name}, skipping...")

                return

            tile_extent = sub_grid.geometry.bounds
            
            tile_dir = os.path.join(output_dir, sub_tile_name)
            os.makedirs(tile_dir, exist_ok=True)            

            for file in raster_files:

                with rasterio.open(file) as src:
                    window = src.window(tile_extent[0], tile_extent[1], tile_extent[2], tile_extent[3])
                    cropped_r = src.read(1, window=window)
                    transform = src.window_transform(window)   

                    mask = cropped_r != src.nodata
                    raster_data = np.column_stack(np.where(mask)) 
                    raster_values = cropped_r[mask]  

                    row_vals = raster_data[:, 0]
                    col_vals = raster_data[:, 1]

                    # Convert pixel row/col to spatial coordinates
                    xs, ys = rasterio.transform.xy(transform, row_vals, col_vals, offset='center')

                    fids = np.ravel_multi_index((row_vals, col_vals), cropped_r.shape)

                    raster_df = pd.DataFrame({
                        'X': xs,
                        'Y': ys,
                        'FID': fids,
                        'Value': raster_values,
                        'Raster': os.path.splitext(os.path.basename(file))[0]
                    })

                    clipped_data.append(raster_df)

        combined_data = pd.concat(clipped_data, axis=0, join='outer', ignore_index=True)
        combined_data_pivot = combined_data.pivot(index=['X', 'Y', 'FID'], columns='Raster', values='Value').reset_index()
        combined_data = combined_data_pivot
        combined_data['geometry'] = [Point(x, y) for x, y in zip(combined_data['X'], combined_data['Y'])]

        raster_gdf = gpd.GeoDataFrame(combined_data, geometry='geometry', crs='EPSG:32617') 

        return raster_gdf

