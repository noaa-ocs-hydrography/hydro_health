"""Class for processing and tiling model prediction and training data."""

import os
import glob
import pandas as pd
import geopandas as gpd
import rasterio
import numpy as np
import dask
import rioxarray
import xarray as xr

from pathlib import Path
from lxml import etree
from datetime import datetime
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape
from rasterio.features import shapes
from dask.distributed import Client, LocalCluster


class ModelDataProcessor:
    """Class for parallel processing all model data"""

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type):
        """ Clip raster files by tile and save the data in a specified directory.
        :param str raster_dir: directory containing the raster files to be processed
        :param str output_dir: directory to save the output files
        :param str data_type: Specifies the type of data being processed (e.g., "prediction" or "training").
        """        
        sub_grid_gpkg = rf"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\{data_type}_tiles\intersecting_sub_grids.gpkg"
        sub_grids = gpd.read_file(sub_grid_gpkg, layer='intersecting_sub_grids')
        
        tasks = []
        for _, sub_grid in sub_grids.iterrows():
            tasks.append(dask.delayed(self.tile_process)(sub_grid, raster_dir, output_dir, data_type))

        dask.compute(*tasks) 

    # TODO move to other processing step with slope and rugosity
    def create_survey_end_date_tiffs(self):
        """Create survey end date tiffs from contributor band values in the XML file.
        """        

        input_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\UTM17'
        output_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\survey_date_end_python'
        kml_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\RATs'
        os.makedirs(output_dir, exist_ok=True)

        tiff_files = glob.glob(os.path.join(input_dir, "*.tiff"))
        for raster_path in tiff_files:
            base_name = os.path.splitext(os.path.basename(raster_path))[0]
            kml_file_path = os.path.join(kml_dir, f'{base_name}.tiff.aux.xml')

            with rasterio.open(raster_path) as src:
                contributor_band_values = src.read(3)
                transform = src.transform
                nodata = src.nodata 
                width, height = src.width, src.height  

            tree = etree.parse(kml_file_path)
            root = tree.getroot()

            contributor_band_xml = root.xpath("//PAMRasterBand[Description='Contributor']")
            rows = contributor_band_xml[0].xpath(".//GDALRasterAttributeTable/Row")

            table_data = []
            for row in rows:
                fields = row.xpath(".//F")
                field_values = [field.text for field in fields]

                data = {
                    "value": float(field_values[0]),
                    "survey_date_end": (
                        datetime.strptime(field_values[17], "%Y-%m-%d").date() 
                        if field_values[17] != "N/A" 
                        else None  
                    )
                }
                table_data.append(data)
            attribute_table_df = pd.DataFrame(table_data)

            attribute_table_df['survey_year_end'] = attribute_table_df['survey_date_end'].apply(lambda x: x.year if pd.notna(x) else 0)
            attribute_table_df['survey_year_end'] = attribute_table_df['survey_year_end'].round(2)

            date_mapping = attribute_table_df[['value', 'survey_year_end']].drop_duplicates()
            reclass_matrix = date_mapping.to_numpy()
            reclass_dict = {row[0]: row[1] for row in reclass_matrix}

            reclassified_band = np.vectorize(lambda x: reclass_dict.get(x, nodata))(contributor_band_values)
            reclassified_band = np.where(reclassified_band == None, nodata, reclassified_band)

            output_file = os.path.join(output_dir, f'{base_name}.tiff')

            with rasterio.open(
            output_file, 
            'w', 
            driver='GTiff',
            count=1, 
            width=width,
            height=height,
            dtype=rasterio.float32,
            compress='lzw', 
            crs=src.crs, 
            transform=transform,
            nodata=nodata) as dst: dst.write(reclassified_band, 1)       
    
    def parallel_processing_rasters(self, input_directory, mask_pred, mask_train):
        """Process prediction and training rasters in parallel using Dask.
        :param gdf mask_pred: Prediction mask GeoDataFrame
        :param gdf mask_train: Training mask GeoDataFrame
        :return: None
        """   

        prediction_dir = Path(input_directory) / 'model_variables' / 'Prediction' / 'pre_processed' # TODO this might be preproceesed depending on the dataset
        prediction_out =  Path(input_directory) / 'model_variables' / 'Prediction' / 'processed'

        prediction_files = list(Path(prediction_dir).rglob("*.tif"))

        prediction_tasks = []
        for file in prediction_files:
            print(f'Processing {file}...')
            input_path = os.path.join(prediction_dir, file)
            output_path = os.path.join(prediction_out, file)

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
        
        grid_gpkg = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Tessellation\Master_Grids.gpkg" #TODO update for new grid

        print("Preparing grid tiles and sub-grids...")
        
        sub_grids = gpd.read_file(grid_gpkg, layer="Model_sub_Grid_Tiles").to_crs(mask_gdf.crs)
        
        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
        intersecting_sub_grids.to_file(os.path.join(output_dir, "intersecting_sub_grids.gpkg"), driver="GPKG") 
   
    def process(self, input_directory):
        """ Main function to process model data.
        """        

        output_dir_pred = Path(input_directory) / 'model_variables' / 'Prediction' / 'prediction_tiles'
        output_dir_train = Path(input_directory) / 'model_variables' / 'Prediction' / 'training_tiles'

        mask_prediction = Path(input_directory) / 'prediction_masks' / 'prediction.mask.UTM17_8m.tif'
        mask_training = Path(input_directory) / 'prediction_masks' / 'training.mask.UTM17_8m.tif'

        prediction_mask_df = self.raster_to_spatial_df(mask_prediction)
        training_mask_df = self.raster_to_spatial_df(mask_training)

        self.create_subgrids(mask_gdf=prediction_mask_df, output_dir=output_dir_pred)
        self.create_subgrids(mask_gdf=training_mask_df, output_dir=output_dir_train) 

        cluster = LocalCluster(n_workers=8, threads_per_worker=1)
        client = Client(cluster)
        # client = Client(n_workers=1, threads_per_worker=1, memory_limit="16GB")
        # print(f"Dask Dashboard: {client.dashboard_link}")

        self.parallel_processing_rasters(input_directory, prediction_mask_df, training_mask_df)
        input_dir_pred = Path(input_directory) / 'model_variables' / 'Prediction' / 'processed'
        input_dir_train = Path(input_directory) / 'model_variables' / 'Training' / 'processed'
        
        self.clip_rasters_by_tile(raster_dir=input_dir_pred, output_dir=output_dir_pred, data_type="prediction")
        self.clip_rasters_by_tile(raster_dir=input_dir_train, output_dir=output_dir_train, data_type="training")
    
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
        ds = rioxarray.open_rasterio(raster_path, chunks={"x": 1024, "y": 1024})

        mask_gdf = mask_gdf.to_crs(ds.rio.crs)

        clipped = ds.rio.clip(mask_gdf.geometry.values, mask_gdf.crs, drop=True, invert=False)

        if 'bathy_' in os.path.basename(raster_path):
            clipped = clipped.where(clipped <= 0)

        clipped = clipped.rio.reproject(
            target_crs,
            resolution=target_res,
            resampling=Resampling.nearest
            source=data,
            nodata=np.nan)

        clipped.rio.to_raster(
            output_path,
            compress="LZW",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            dtype='float32'
        )    

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
            mask = src.read(1)
            valid_mask = mask == 1
            shapes_gen = shapes(mask, valid_mask, transform=src.transform)

            gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, _ in shapes_gen]}, crs=src.crs)

        return gdf
     
    def tile_process(self, sub_grid, raster_dir, output_dir, data_type):
        """ Process a single tile by clipping raster files and saving the data in a specified directory.

        :param _type_ sub_grid: _description_
        :param _type_ raster_dir: _description_
        :param _type_ output_dir: _description_
        :param _type_ data_type: _description_
        """        
        raster_files = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]

        tile_name = sub_grid['Tile_ID']
        tile_extent = sub_grid.geometry.bounds
        
        tile_dir = os.path.join(output_dir, tile_name)
        os.makedirs(tile_dir, exist_ok=True)
            
        print(f"Processing {data_type} tile: {tile_name}")
        
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
        nan_percentage = combined_data['bathy_2006'].isna().mean() * 100
        print(f"Percentage of NaNs in bathy_2006: {nan_percentage:.2f}%")
        
        combined_data['b.change.2004_2006'] = combined_data['bathy_2006'] - combined_data['bathy_2004']
        combined_data['b.change.2006_2010'] = combined_data['bathy_2010'] - combined_data['bathy_2006']
        combined_data['b.change.2010_2015'] = combined_data['bathy_2015'] - combined_data['bathy_2010']
        combined_data['b.change.2015_2022'] = combined_data['bathy_2022'] - combined_data['bathy_2015']
        
        clipped_data_path = os.path.join(tile_dir, f"{tile_name}_{data_type}_clipped_data.parquet")
        combined_data.to_parquet(clipped_data_path)
