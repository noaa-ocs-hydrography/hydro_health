"""Class for parallel preprocessing all model data"""

import os
import pathlib
import re
from pathlib import Path

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dask.distributed import Client, print
from osgeo import gdal
from rasterio.features import shapes
from shapely.geometry import Point, box, shape

os.environ["GDAL_CACHEMAX"] = "64"

from hydro_health.helpers.tools import get_config_item

OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


class ModelDataPreProcessor:
    """Class for parallel preprocessing all model data"""
    def __init__(self):
        self.year_ranges = [
            (2004, 2006),
            (2006, 2010),
            (2010, 2015),
            (2015, 2022)]
    
    def create_nan_stats_csv(self, df, tile_id)-> pd.DataFrame:
        """Analyzes a DataFrame for a single tile and returns its NaN stats.

        :param df: DataFrame containing the data for a single tile
        :param tile_id: Identifier for the tile being processed
        :return: DataFrame with NaN statistics for the tile
        """

        if df.empty:
            return pd.DataFrame() 
        
        new_row = {'tile_id': tile_id}
        
        # Find all 'b.change' columns and calculate NaN percentages
        change_columns = [col for col in df.columns if col.startswith('b.change.')]
        for col_name in change_columns:
            year_pair = col_name.replace('b.change.', '')
            nan_percent = df[col_name].isna().mean() * 100
            
            new_row[f"{year_pair}_nan_percent"] = round(nan_percent, 2)

        return pd.DataFrame([new_row])

    def clip_rasters_by_tile(self, raster_dir, output_dir, data_type)-> None:
        """ Clip raster files by tile and save the data in a specified directory.

        :param str raster_dir: directory containing the raster files to be processed
        :param str output_dir: directory to save the output files
        :param str data_type: Specifies the type of data being processed either "prediction" or "training"
        """        

        sub_grid_gpkg = pathlib.Path(get_config_item('MODEL', f'{data_type.upper()}_SUB_GRIDS'))
        sub_grids = gpd.read_file(sub_grid_gpkg)
        
        tasks = []

        num_tiles = sub_grids.shape[0]
        print(f"Number of tiles: {num_tiles}")

        for _, sub_grid in sub_grids.iterrows():
            tile_name = sub_grid['tile_id'] 
            output_folder = os.path.join(output_dir, tile_name)

            # Delay each part separately
            gridded_task = dask.delayed(self.subtile_process_gridded)(sub_grid, raster_dir)
            ungridded_task = dask.delayed(self.subtile_process_ungridded)(sub_grid, raster_dir)

            # Delay the combining + saving
            combined_task = self.save_combined_data(gridded_task, ungridded_task, output_folder, data_type, tile_id=tile_name)

            tasks.append(combined_task)

        results_list = dask.compute(*tasks)

        # 3. Combine the list of result DataFrames into one final DataFrame.
        print("Combining results...")
        final_results_df = pd.concat(results_list, ignore_index=True)

        output_csv_path = f"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\year_pair_nan_counts_{data_type}.csv"
        final_results_df.to_csv(output_csv_path, index=False, na_rep='NA')

    def create_subgrids(self, mask_gdf, output_dir, process_type)-> None:
        """ Create subgrids layer by intersecting grid tiles with the mask geometries

        :param gdf mask_gdf: GeoDataFrame containing the mask geometries
        :param str output_dir: directory to save the output files
        :param str process_type: Specifies the type of data being processed either "prediction" or "training"
        :return: None
        """        
        
        grid_gpkg = get_config_item('MODEL', 'SUBGRIDS')

        print(f"Preparing {process_type} sub-grids...")

        mask_gdf = gpd.read_parquet(mask_gdf)
        combined_geometry = mask_gdf.unary_union
        mask_gdf = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf.crs)
        
        sub_grids = gpd.read_file(grid_gpkg, layer='prediction_subgrid').to_crs(mask_gdf.crs)
        
        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
        intersecting_sub_grids.to_file(os.path.join(output_dir, f"{process_type}_intersecting_subgrids.gpkg"), driver="GPKG") 

    def parallel_processing_rasters(self, input_directory, mask_pred, mask_train)-> None:
        """Process prediction and training rasters in parallel using Dask.

        :param str input_directory: Directory containing the raster files to be processed
        :param gdf mask_pred: Prediction mask GeoDataFrame
        :param gdf mask_train: Training mask GeoDataFrame
        :return: None
        """   

        input_directory = Path(input_directory)
        prediction_out = Path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'))
        existing_outputs = {f.name for f in prediction_out.glob("*") if f.suffix.lower() in ['.tif', '.tiff']}

        prediction_files = [
            f for f in input_directory.rglob("*")
            if (
                f.suffix.lower() in ['.tif', '.tiff'] and
                f.name not in existing_outputs
            )
        ]

        print(f"Total prediction files found (excluding already processed): {len(prediction_files)}")

        mask_gdf = gpd.read_parquet(mask_pred)
        combined_geometry = mask_gdf.unary_union
        mask_gdf = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf.crs)

        prediction_tasks = []
        for file in prediction_files:
            input_path = os.path.join(input_directory, file)
            output_path = os.path.join(prediction_out, file.name)
            prediction_tasks.append(dask.delayed(self.process_prediction_raster)(input_path, mask_gdf, output_path))

        dask.compute(*prediction_tasks)  
        
        training_out = pathlib.Path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'))
        existing_outputs = {f.name for f in training_out.glob("*") if f.suffix.lower() in ['.tif', '.tiff']}

        training_files = [
            f for f in prediction_out.rglob("*")
            if f.suffix.lower() in ['.tif', '.tiff'] and f.name not in existing_outputs
        ]

        print(f"Total training files found (excluding already processed): {len(training_files)}")

        mask_gdf = gpd.read_parquet(mask_train)
        combined_geometry = mask_gdf.unary_union
        mask_gdf = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf.crs)

        training_tasks  = []
        for file in training_files:
            input_path = os.path.join(prediction_out, file)
            output_path = os.path.join(training_out, file.name)
            training_tasks.append(dask.delayed(self.process_training_raster)(input_path, mask_gdf, output_path))

        dask.compute(*training_tasks)    

    def process(self)-> None:
        """ Main function to process model data.
        """        

        subgrids_output_dir = pathlib.Path(get_config_item('MODEL', 'MODEL_SUBGRIDS'))

        # prediction_mask_df = self.raster_to_spatial_df(pathlib.Path(get_config_item('MODEL', 'PREDICTION_MASK')), process_type='prediction')
        # training_mask_df = self.raster_to_spatial_df(pathlib.Path(get_config_item('MODEL', 'TRAINING_MASK')), process_type='training')

        mask_prediction_pq = pathlib.Path(get_config_item('MODEL', 'PREDICTION_MASK_PQ'))
        mask_training_pq = pathlib.Path(get_config_item('MODEL', 'TRAINING_MASK_PQ'))

        # self.create_subgrids(mask_gdf=mask_prediction_pq, output_dir=subgrids_output_dir, process_type = 'prediction')
        # self.create_subgrids(mask_gdf=mask_training_pq, output_dir=subgrids_output_dir, process_type = 'training') 

        client = Client(n_workers=7, threads_per_worker=2, memory_limit="32GB")
        print(f"Dask Dashboard: {client.dashboard_link}")

        preprocessed_dir = pathlib.Path(get_config_item('MODEL', 'PREPROCESSED_DIR'))
        processed_dir = pathlib.Path(get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR'))

        # self.parallel_processing_rasters(preprocessed_dir, mask_prediction_pq, mask_training_pq)

        input_dir_train = pathlib.Path(get_config_item('MODEL', 'TRAINING_OUTPUT_DIR'))
        output_dir_pred = pathlib.Path(get_config_item('MODEL', 'PREDICTION_TILES_DIR'))
        output_dir_train = pathlib.Path(get_config_item('MODEL', 'TRAINING_TILES_DIR'))

        print(" - Clipping prediction rasters by tile...")
        self.clip_rasters_by_tile(raster_dir=processed_dir, output_dir=output_dir_pred, data_type="prediction")
        print(" - Clipping training rasters by tile...")
        # self.clip_rasters_by_tile(raster_dir=input_dir_train, output_dir=output_dir_train, data_type="training")
    
        client.close()

    def process_prediction_raster(self, raster_path, mask_gdf, output_path, target_crs="EPSG:32617", target_res=8)-> None:
        """ Reprojects, resamples, and crops a raster to a target CRS, resolution, and extent

        :param str raster_path: path to the raster file
        :param gdf mask_gdf: GeoDataFrame containing the mask geometries
        :param str output_path: path to save the processed raster
        :param str target_crs: target coordinate reference system (default is "EPSG:32617")
        :param int target_res: target resolution in meters (default is 8)
        :return: None
        """  

        with rasterio.open(raster_path) as src:
            src_nodata = src.nodata

        keywords = ["tsm", "sed", "hurricane"] 
        raster_name = os.path.basename(str(raster_path)).lower()
        print(f'Processing prediction file {raster_name}...')

        should_crop = any(keyword in raster_name.lower() for keyword in keywords)

        in_memory_cutline = f'/vsimem/cutline_{raster_name}.geojson'   

        mask_gdf.to_file(in_memory_cutline, driver='GeoJSON')

        gdal.Warp(
            output_path,
            str(raster_path),
            dstSRS=target_crs,
            xRes=target_res,
            yRes=target_res,
            cutlineDSName=in_memory_cutline,
            cropToCutline=should_crop, 
            resampleAlg='bilinear',
            srcNodata=src_nodata,
            dstNodata=np.nan,
            creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES'],
            multithread=True,
            warpMemoryLimit=2048,
            warpOptions=['CUTLINE_ALL_TOUCHED=TRUE']
        )

        src = None

        if os.path.exists(in_memory_cutline):
            gdal.Unlink(in_memory_cutline)

    def process_training_raster(self, raster_path, mask_gdf, output_path)-> None:
        """ Process a training raster by clipping it with a mask GeoDataFrame and saving the output.

        :param str raster_path: path to the raster file
        :param gdf mask_gdf: GeoDataFrame containing the mask geometries
        :param str output_path: path to save the processed raster
        :return: None
        """

        with rasterio.open(raster_path) as src:
            src_nodata = src.nodata
            raster_bounds = box(*src.bounds)

        raster_name = os.path.basename(str(raster_path)).lower()

        if not mask_gdf.unary_union.intersects(raster_bounds):
            print(f"Mask does not intersect raster {raster_name}. Skipping output.")
            return
        
        print(f'Processing training file {raster_name}...')

        in_memory_cutline = f'/vsimem/cutline_{raster_name}.geojson'   

        mask_gdf.to_file(in_memory_cutline, driver='GeoJSON')

        try:
            result = gdal.Warp(
                output_path,
                str(raster_path),
                cutlineDSName=in_memory_cutline,
                srcNodata=src_nodata,
                dstNodata=np.nan, 
                warpOptions=['CUTLINE_ALL_TOUCHED=TRUE'],
                creationOptions=['TILED=YES', 'COMPRESS=LZW', 'BIGTIFF=YES']
            )

            if result is None:
                raise RuntimeError("gdal.Warp failed and returned None")

        except Exception as e: # Delete output file if it was partially written
            print(f"Error during gdal.Warp: {e}")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                    print(f"Removed partial output: {output_path}")
                except Exception as rm_error:
                    print(f"Failed to remove partial output: {rm_error}")
            return  # Exit task early
        finally:
            if gdal.VSIStatL(in_memory_cutline) == 0:
                gdal.Unlink(in_memory_cutline)  

    def raster_to_spatial_df(self, raster_path, process_type)-> gpd.GeoDataFrame:
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

    @dask.delayed
    def save_combined_data(self, gridded_df, ungridded_df, output_folder, data_type, tile_id)-> pd.DataFrame:
        """ Combine gridded and ungridded dataframes and save to a parquet file.

        :param gridded_df: delayed gridded data DataFrame
        :param ungridded_df: delayed ungridded data DataFrame
        :param output_path: path to save the combined DataFrame
        :param data_type: type of data being processed (e.g., "prediction" or "training")
        :param tile_id: identifier for the tile being processed
        :return: None
        """

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, f"{tile_id}_{data_type}_clipped_data.parquet")

        combined = pd.merge(gridded_df, ungridded_df, on=['X', 'Y'], how='outer')
        combined.to_parquet(output_path, engine="pyarrow", index=False)  
        nan_row = self.create_nan_stats_csv(combined, tile_id) 

        return nan_row

    def subtile_process_gridded(self, sub_grid, raster_dir)-> pd.DataFrame:
        """ Process a single tile by clipping gridded raster data and saving the data in a specified directory.

        :param _type_ sub_grid: sub-grid information containing tile_id and geometry
        :param _type_ raster_dir: input directory containing the raster files to be processed
        :return: GeoDataFrame containing the clipped raster data
        """        
        
        sub_tile_name = sub_grid['tile_id']
        print(f"Processing tile: {sub_tile_name}")
        original_tile = sub_grid['original_tile']  

        raster_files = [
            os.path.join(raster_dir, f)
            for f in os.listdir(raster_dir)
            if (f.endswith('.tif') or f.endswith('.tiff')) and original_tile in f
        ]

        if not raster_files:
            print(f"No matching gridded raster files found for tile {sub_tile_name}, skipping...")
            return

        tile_extent = sub_grid.geometry.bounds
                    
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

                raster_df = pd.DataFrame({
                    'X': xs,
                    'Y': ys,
                    'Value': raster_values,
                    'Raster': os.path.splitext(os.path.basename(file))[0]
                })

                clipped_data.append(raster_df)

        combined_data = pd.concat(clipped_data, axis=0, join='outer', ignore_index=True)
        combined_data_pivot = combined_data.pivot(index=['X', 'Y'], columns='Raster', values='Value').reset_index()
        combined_data = combined_data_pivot

        combined_data['geometry'] = [Point(x, y) for x, y in zip(combined_data['X'], combined_data['Y'])]
        combined_data = gpd.GeoDataFrame(combined_data, geometry='geometry', crs='EPSG:32617') 

        exclude_keywords = ['rugosity', 'slope', 'survey_end_date', 'unc']
        keep_unchanged = ['X', 'Y', 'geometry']

        new_columns = {}

        for col in combined_data.columns:
            if col in keep_unchanged:
                new_columns[col] = col  # leave unchanged
            else:
                year_match = re.search(r'(19|20)\d{2}', col)
                if year_match:
                    year = year_match.group()
                    matched_keyword = None
                    for keyword in exclude_keywords:
                        if keyword in col:
                            matched_keyword = keyword
                            break
                    if matched_keyword:
                        new_columns[col] = f"{matched_keyword}_{year}"
                    elif 'bluetopo' in col.lower():
                        new_columns[col] = f"bt_bathy_{year}"
                    else:
                        new_columns[col] = f"bathy_{year}"

        combined_data = combined_data.rename(columns=new_columns)

        combined_data = combined_data.loc[:, list(new_columns.values())]

        bathy_cols = [
            col for col in combined_data.columns
            if col.startswith("bathy_") and re.fullmatch(r'.*_(\d{4})', col)
        ]

        if len(bathy_cols) > 1:
            bathy_cols_sorted = sorted(bathy_cols, key=lambda x: int(re.search(r'\d{4}', x).group()))

            for i in range(len(bathy_cols_sorted) - 1):
                col1 = bathy_cols_sorted[i]
                col2 = bathy_cols_sorted[i + 1]

                year1 = re.search(r'\d{4}', col1).group()
                year2 = re.search(r'\d{4}', col2).group()

                new_col_name = f'b.change.{year1}_{year2}'
                combined_data[new_col_name] = combined_data[col2] - combined_data[col1]

        else:
            print(f"Only {len(bathy_cols)} year of bathy data found for tile {sub_tile_name}, skipping bathy change calculations.")
            return combined_data 

        return combined_data

    def subtile_process_ungridded(self, sub_grid, raster_dir)-> gpd.GeoDataFrame:
        """ Process a single tile by clipping ungridded raster data and saving the data in a specified directory.

        :param gpkg sub_grid: sub-grid information containing tile_id and geometry
        :param str raster_dir: input directory containing the raster files to be processed
        :return: GeoDataFrame containing the clipped raster data
        """
        
        sub_tile_name = sub_grid['tile_id']
        static_data = ['sed_size_raster', 'sed_type_raster', 'tsm_mean', 'hurricane']
        clipped_data = []

        for data in static_data:
            raster_files = [
                os.path.join(raster_dir, f)
                for f in os.listdir(raster_dir)
                if f.endswith('.tif') and data in f
            ]

            # if not raster_files:
                # print(f"No static {data} raster found for tile {sub_tile_name}, skipping...")

            tile_extent = sub_grid.geometry.bounds        

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

                    raster_df = pd.DataFrame({
                        'X': xs,
                        'Y': ys,
                        'Value': raster_values,
                        'Raster': os.path.splitext(os.path.basename(file))[0]
                    })

                    clipped_data.append(raster_df)

        combined_data = pd.concat(clipped_data, axis=0, join='outer', ignore_index=True)
        combined_data_pivot = combined_data.pivot(index=['X', 'Y'], columns='Raster', values='Value').reset_index()
        combined_data = combined_data_pivot
        combined_data['geometry'] = [Point(x, y) for x, y in zip(combined_data['X'], combined_data['Y'])]

        raster_gdf = gpd.GeoDataFrame(combined_data, geometry='geometry', crs='EPSG:32617') 

        return raster_gdf
    