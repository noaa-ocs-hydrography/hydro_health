"""Class for processing and tiling model prediction and training data."""

import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import dask
import rioxarray

from pathlib import Path
from lxml import etree
from datetime import datetime
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape
from rasterio.features import shapes
from dask.distributed import Client


class ModelDataProcessor:
    """Class for parallel processing all model data"""

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type):
        """ Clip raster files by tile and save the data in a specified directory.
        :param str raster_dir: directory containing the raster files to be processed
        :param str output_dir: directory to save the output files
        :param str data_type: Specifies the type of data being processed (e.g., "prediction" or "training").
        """        
        # sub_grid_gpkg = rf"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\{data_type}_tiles\intersecting_sub_grids.gpkg"
        sub_grid_gpkg = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\intersecting_sub_grids.gpkg"
        sub_grids = gpd.read_file(sub_grid_gpkg, layer='intersecting_sub_grids')
        
        tasks = []
        for _, sub_grid in sub_grids.iterrows():
            # tasks.append(dask.delayed(self.tile_process)(sub_grid, raster_dir, output_dir, data_type))
            self.tile_process(sub_grid, raster_dir, output_dir, data_type)

        # dask.compute(*tasks) 

    def parallel_processing_rasters(self, input_directory, mask_pred, mask_train):
        """Process prediction and training rasters in parallel using Dask.
        :param gdf mask_pred: Prediction mask GeoDataFrame
        :param gdf mask_train: Training mask GeoDataFrame
        :return: None
        """   

        # prediction_dir = Path(input_directory) / 'model_variables' / 'Prediction' / 'preprocessed' 
        prediction_dir  = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\sample_files" # temp for testing
        # prediction_out =  Path(input_directory) / 'model_variables' / 'Prediction' / 'processed'

        prediction_out = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\outputs' # temp for testing
        prediction_files = list(Path(prediction_dir).rglob("*.tiff"))
        # prediction_files = list(Path(prediction_dir).rglob("*slope*.tiff")) # temp for testing
        # prediction_files = list(Path(prediction_dir).rglob("*slope*BH4TC564*.tiff"))

        prediction_tasks = []
        for file in prediction_files:
            print(f'Processing {file}...')
            input_path = os.path.join(prediction_dir, file)
            output_path = os.path.join(prediction_out, file.name)

            prediction_tasks.append(dask.delayed(self.process_raster)(input_path, mask_pred, output_path))

        dask.compute(*prediction_tasks)  
        
        training_files = [f for f in os.listdir(prediction_out) if f.endswith('.tif')]
        training_out = Path(input_directory) / 'model_variables' / 'Training' / 'processed'

        training_tasks  = []
        for file in training_files:
            print(f'Processing {file}...')
            input_path = os.path.join(prediction_out, file)
            output_path = os.path.join(training_out, file)
            training_tasks .append(dask.delayed(self.process_raster)(input_path, mask_train, output_path))

        dask.compute(*training_tasks)    

    def create_subgrids(self, mask_gdf, output_dir):
        """ Create subgrids layer by intersecting grid tiles with the mask geometries
        :param gdf mask_gdf: GeoDataFrame containing the mask geometries
        :param str output_dir: directory to save the output files
        :return: None
        """        
        
        grid_gpkg = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\HHM_Run\ER_3\processing_grids\Prediction.subgrid.WGS84_8m.gpkg' 

        print("Preparing grid tiles and sub-grids...")
        
        sub_grids = gpd.read_file(grid_gpkg, layer='prediction_subgrid').to_crs(mask_gdf.crs)
        
        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
        intersecting_sub_grids.to_file(os.path.join(output_dir, "intersecting_sub_grids.gpkg"), driver="GPKG") 
   
    def process(self, input_directory):
        """ Main function to process model data.
        """        

        output_dir_pred = Path(input_directory) / 'Outputs' / 'prediction_data_grid_tiles' 
        output_dir_train = Path(input_directory) / 'Outputs' / 'training_data_grid_tiles' 

        mask_prediction_path = Path(input_directory) / 'prediction_masks' / 'prediction.mask.UTM17_8m.tif'
        # mask_training_path = Path(input_directory) / 'training_masks' / 'ER_3_mask.tif'
        mask_prediction_path = r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\masks\prediction.mask.UTM17_8m.tif'
        mask_training_path = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\masks\prediction.mask.UTM17_8m.tif"

        print('Creating prediction mask data frame..')
        # self.raster_to_spatial_df(mask_prediction_path)
        # prediction_mask_df = self.raster_to_spatial_df(mask_prediction_path)
        # prediction_mask_df = gpd.read_parquet(r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\prediction_mask.parquet')

        print('Creating training mask data frame..')
        # training_mask_df = self.raster_to_spatial_df(mask_training_path)
        # training_mask_df = prediction_mask_df

        # self.create_subgrids(mask_gdf=prediction_mask_df, output_dir=output_dir_pred) # TODO may not need to do this at the ER scale # 210 seconds
        # self.create_subgrids(mask_gdf=training_mask_df, output_dir=output_dir_train) # 254 seconds

        # cluster = LocalCluster(n_workers=8, threads_per_worker=1)
        # client = Client(cluster)
        client = Client(n_workers=4, threads_per_worker=2, memory_limit="16GB")
        # client = Client(memory_limit="16GB")
        print(f"Dask Dashboard: {client.dashboard_link}")

        # self.parallel_processing_rasters(input_directory, prediction_mask_df, training_mask_df)
        # input_dir_pred = Path(input_directory) / 'model_variables' / 'Prediction' / 'processed'
        # input_dir_train = Path(input_directory) / 'model_variables' / 'Training' / 'processed'
        input_dir_pred = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\outputs' # temp
        output_dir_pred = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\outputs' # temp
        
        # TODO need to have these use corresponding parent tile
        self.clip_rasters_by_tile(raster_dir=input_dir_pred, output_dir=output_dir_pred, data_type="prediction")
        # self.clip_rasters_by_tile(raster_dir=input_dir_train, output_dir=output_dir_train, data_type="training")
    
        client.close()   

    def process_raster(self, raster_path, mask_gdf, output_path, target_crs="EPSG:32617", target_res=8):
        """Process a raster file by applying a mask and reprojecting it to a target CRS and resolution,
        then save as a compressed, tiled GeoTIFF.
        
        :param str raster_path: Path to the raster file to be processed
        :param gdf mask_gdf: GeoDataFrame containing the mask geometries
        :param str output_path: Path to the output raster file
        :param str target_crs: Target coordinate reference system (default is EPSG:32617)
        :param int target_res: Target resolution in meters (default is 8m)
        """

        # Read raster with rioxarray, using Dask for chunking
        with rioxarray.open_rasterio(raster_path, chunks={"x": 1024, "y": 1024}) as ds:            
            mask_gdf = mask_gdf.to_crs(ds.rio.crs)
            clipped = ds.rio.clip(mask_gdf.geometry.values, mask_gdf.crs, drop=True, invert=False)

        if 'bathy_' in os.path.basename(raster_path):
            clipped = clipped.where(clipped <= 0) # sets values > 0 to NaN

        clipped = clipped.rio.reproject(
            target_crs,
            resolution=target_res,
            resampling=Resampling.bilinear,
            nodata=np.nan)


        clipped.rio.to_raster(
            output_path,
            compress="LZW",
            tiled=True,
            blockxsize=1024,
            blockysize=1024,
            dtype='float32'
        )    

        zarr_path = os.path.splitext(output_path)[0] + ".zarr"
        clipped.to_zarr(os.path.join(zarr_path, '.zarr'), mode="w", consolidated=True)

    
    # def process_raster(self, raster_path, mask_gdf, output_path, target_crs="EPSG:32617", target_res=8):
    #     """Process a raster file by applying a mask and reprojecting it to a target CRS and resolution.
    #     All rasters will have the same X, Y, and FID values.
    #     This function uses Dask for chunking and parallel processing.

    #     :param str raster_path: path to the raster file to be processed
    #     :param gdf mask_gdf: GeoDataFrame containing the mask geometries
    #     :param str output_path: path to the output raster file
    #     :param str target_crs: target coordinate reference system, defaults to "EPSG:32617"
    #     :param int target_res: raster resolution, defaults to 8 m
    #     """        

    #     with rasterio.open(raster_path) as src:
    #         mask_gdf = mask_gdf.to_crs(src.crs)
    #         mask_shapes = [geom for geom in mask_gdf.geometry]
    #         data, mask_transform = mask(src, mask_shapes, crop=True, nodata=np.nan) # Apply mask

    #         # Set bathymetry values > 0 to NaNs
    #         if 'bathy_' in os.path.basename(raster_path):
    #             data[data > 0] = np.nan

    #         transform, width, height = calculate_default_transform(
    #             src.crs, target_crs, data.shape[-1], data.shape[-2], *mask_gdf.total_bounds, resolution=target_res
    #         )

    #         kwargs = src.meta.copy()
    #         kwargs.update({
    #             'crs': target_crs,
    #             'transform': transform,
    #             'width': width,
    #             'height': height,
    #             'dtype': 'float32',
    #             'compress': 'lzw'
    #         })

    #         with rasterio.open(output_path, 'w', **kwargs) as dst:
    #             reproject(
    #                 source=data,
    #                 destination=rasterio.band(dst, 1),
    #                 src_transform=mask_transform,
    #                 src_crs=src.crs,
    #                 dst_transform=transform, 
    #                 dst_crs=target_crs,
    #                 resampling=Resampling.nearest,
    #                 dst_nodata=np.nan
    #             )

    def raster_to_spatial_df(self, raster_path):
        """ Convert a raster file to a GeoDataFrame by extracting shapes and their geometries.

        :param str raster_path: path to the raster file
        :return gdf: GeoDataFrame containing the mask shapes and geometries
        """        
        with rasterio.open(raster_path) as src:
            mask = src.read(1, out_dtype='uint8')
            valid_mask = mask == 1
            shapes_gen = shapes(mask, valid_mask, transform=src.transform)

            gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, _ in shapes_gen]}, crs=src.crs)
            gdf.to_parquet(r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\prediction_mask.parquet')

        # return gdf
     
    def tile_process(self, sub_grid, raster_dir, output_dir, data_type):
        """ Process a single tile by clipping raster files and saving the data in a specified directory.

        :param _type_ sub_grid: _description_
        :param _type_ raster_dir: _description_
        :param _type_ output_dir: _description_
        :param _type_ data_type: _description_
        """        
        
        sub_tile_name = sub_grid['tile_id']
        original_tile = sub_grid['original_tile']  

        raster_files = [
            os.path.join(raster_dir, f)
            for f in os.listdir(raster_dir)
            if f.endswith('.tiff') and original_tile in f
        ]

        if not raster_files:
            print(f"No matching raster files found for tile {sub_tile_name}, skipping...")

            return

        tile_extent = sub_grid.geometry.bounds

        print(raster_files)
        
        tile_dir = os.path.join(output_dir, sub_tile_name)
        os.makedirs(tile_dir, exist_ok=True)
            
        print(f"Processing {data_type} tile: {sub_tile_name}")
        
        clipped_data = []
        for file in raster_files:

            with rasterio.open(file) as src:
                window = src.window(tile_extent[0], tile_extent[1], tile_extent[2], tile_extent[3])
                cropped_r = src.read(1, window=window)  

                mask = cropped_r != src.nodata
                raster_data = np.column_stack(np.where(mask)) 
                raster_values = cropped_r[mask]  

                x_vals = raster_data[:, 1]  
                y_vals = raster_data[:, 0]  

                fids = np.ravel_multi_index((y_vals, x_vals), cropped_r.shape)

                raster_df = pd.DataFrame({
                    'X': x_vals,
                    'Y': y_vals,
                    'FID': fids,
                    'Value': raster_values,
                    'Raster': os.path.splitext(os.path.basename(file))[0]
                })

                clipped_data.append(raster_df)

        combined_data = pd.concat(clipped_data, axis=0, join='outer', ignore_index=True)
        combined_data_pivot = combined_data.pivot(index=['X', 'Y', 'FID'], columns='Raster', values='Value').reset_index()
        combined_data = combined_data_pivot # remove if the pivot is not needed
        print(combined_data.head(5))
        nan_percentage = combined_data['bathy_2006'].isna().mean() * 100
        print(f"Percentage of NaNs in bathy_2006: {nan_percentage:.2f}%")
        
        combined_data['b.change.2004_2006'] = combined_data['bathy_2006'] - combined_data['bathy_2004']
        combined_data['b.change.2006_2010'] = combined_data['bathy_2010'] - combined_data['bathy_2006']
        combined_data['b.change.2010_2015'] = combined_data['bathy_2015'] - combined_data['bathy_2010']
        combined_data['b.change.2015_2022'] = combined_data['bathy_2022'] - combined_data['bathy_2015']
        
        clipped_data_path = os.path.join(tile_dir, f"{sub_tile_name}_{data_type}_clipped_data.parquet")
        combined_data.to_parquet(clipped_data_path)
