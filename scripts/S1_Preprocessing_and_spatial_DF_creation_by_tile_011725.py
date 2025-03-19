import os
import glob
import gc
from lxml import etree
from datetime import datetime
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from shapely.geometry import Polygon, shape
import dask.array as da
from rasterio.features import shapes
import pyreadr # only for reading the .rds files, I need to save them as something different
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
# import zarr
import dask
from dask import delayed, compute
from dask.distributed import Client, progress
from dask.diagnostics import ProgressBar
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.inspection import partial_dependence
import joblib # for exporting the model file
import mlflow
import mlflow.sklearn
import cProfile
import pstats


# 1. Location of support rasters for processing- N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw
# training_out = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Model_variables\Training\processed_python'
# os.makedirs(training_out, exist_ok=True)  

# 1. create survey date tiffs, works correctly but check the crs
def create_survey_end_date_tiffs():
    input_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\UTM17'
    output_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\survey_date_end_python'
    kml_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\RATs'
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = glob.glob(os.path.join(input_dir, "*.tiff"))
    for raster_path in tiff_files:
        # 2. get file name
        base_name = os.path.splitext(os.path.basename(raster_path))[0]
        # 3. get the XML metadata file for the current tile
        kml_file_path = os.path.join(kml_dir, f'{base_name}.tiff.aux.xml')

        # 4. locate contributor band 3
        with rasterio.open(raster_path) as src:
            contributor_band_values = src.read(3)
            transform = src.transform
            nodata = src.nodata 
            width, height = src.width, src.height  

        # 5. read the XML file
        tree = etree.parse(kml_file_path)
        root = tree.getroot()

        # 6. use the contributor band to get rows from XML file
        contributor_band_xml = root.xpath("//PAMRasterBand[Description='Contributor']")
        rows = contributor_band_xml[0].xpath(".//GDALRasterAttributeTable/Row")

        table_data = []
        for row in rows:
            fields = row.xpath(".//F")
            field_values = [field.text for field in fields]

        # 7. store "value" and "survey_data_end" from the XML rows in a dataframe
            data = {
                "value": float(field_values[0]),  # Convert the first field to a float
                "survey_date_end": (
                    datetime.strptime(field_values[17], "%Y-%m-%d").date() 
                    if field_values[17] != "N/A" 
                    else None  # Use None or another placeholder for 'N/A'
                )
            }
            table_data.append(data)
        attribute_table_df = pd.DataFrame(table_data)

        # 8. create "survey_year_end" column from "survey_date_end"
        attribute_table_df['survey_year_end'] = attribute_table_df['survey_date_end'].apply(lambda x: x.year if pd.notna(x) else 0)
        # 9. round the date to 2 digits
        attribute_table_df['survey_year_end'] = attribute_table_df['survey_year_end'].round(2)

        # 10. build dataframe to map raster values by the year column
        date_mapping = attribute_table_df[['value', 'survey_year_end']].drop_duplicates()
        reclass_matrix = date_mapping.to_numpy()
        # Create a reclassification lookup dictionary from reclass_matrix
        reclass_dict = {row[0]: row[1] for row in reclass_matrix}

        # 11. reclassify contributor band value to the year it was surveyed
        reclassified_band = np.vectorize(lambda x: reclass_dict.get(x, nodata))(contributor_band_values)
        reclassified_band = np.where(reclassified_band == None, nodata, reclassified_band)

        output_file = os.path.join(output_dir, f'{base_name}.tiff')

        # 12. write out new file with processed survey information
        with rasterio.open(
        output_file, 
        'w', 
        driver='GTiff',
        count=1, 
        width=width,
        height=height,
        dtype=rasterio.float32,
        compress='lzw', #lzw
        crs=src.crs, 
        transform=transform,
        nodata=nodata) as dst: dst.write(reclassified_band, 1)       

## 2. standardize all rasters to have same X, Y, FID 
def process_raster(raster_path, mask_gdf, output_path, target_crs="EPSG:32617", target_res=8):
    with rasterio.open(raster_path) as src:
        mask_gdf = mask_gdf.to_crs(src.crs)

        # Get mask geometries
        mask_shapes = [geom for geom in mask_gdf.geometry]
        # Apply mask
        data, mask_transform = mask(src, mask_shapes, crop=True, nodata=np.nan)

        # Set bathymetry NaN condition
        if 'bathy_' in os.path.basename(raster_path):
            data[data > 0] = np.nan

        # Compute new transform for the masked raster (not the full original raster!)
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, data.shape[-1], data.shape[-2], *mask_gdf.total_bounds, resolution=target_res
        )

        kwargs = src.meta.copy()
        kwargs.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'dtype': 'float32',
            'compress': 'lzw'
        })

        # Write the output raster
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            reproject(
                source=data,  # Use masked data
                destination=rasterio.band(dst, 1),
                src_transform=mask_transform,  # Use the transform of the clipped raster
                src_crs=src.crs,
                dst_transform=transform,  # Use new projection transform
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
                dst_nodata=np.nan
            )

## 2. Standardize all rasters (PREDICTION Extent first as its larger)----
#makes all same extent, for processing into spatial points dataframe and removes all land based elevation values > 0 as well
#### NOTE:------ It is critical that BOTH the prediction and training datasets must have the same values of X, Y and FID within each, ---###
# although they are different extents, the smaller training data must be a direct subset of the prediction data
# for variables they have in common, even if the datasets vary between the two final datasets, we will divide the pertiant 
#columns afterward.

def standardize_rasters(mask_pred, mask_train):
    prediction_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Model_variables\Prediction\raw_testing'
    prediction_out = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\prediction_processed" 
    os.makedirs(prediction_out, exist_ok=True)

    prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.tif')]
    # prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.tif') and 'bathy_2004' in f]

    # scattered_mask = Client.scatter(mask)
    tasks = []
    # for file in prediction_files:
    #     print(f'Processing {file}...')
    #     input_path = os.path.join(prediction_dir, file)
    #     output_path = os.path.join(prediction_out, file)

    #     # Use the scattered mask instead of passing the large object directly?
    #     tasks.append(dask.delayed(process_raster)(input_path, mask_pred, output_path))

    # dask.compute(*tasks)  
    
    # # ## 2 Standardize all rasters (TRAINING Extent - sub sample of prediction extent)----
    # # #- THIS IS A DIRECT SUBSET OF THE PREDICTION AREA - clipped using the training mask. 
    training_files = [f for f in os.listdir(prediction_out) if f.endswith('.tif')]
    training_out = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\training_processed" 

    for file in training_files:
        print(f'Processing {file}...')
        input_path = os.path.join(prediction_out, file)  # use processed prediction rasters?
        output_path = os.path.join(training_out, file)
        tasks.append(dask.delayed(process_raster)(input_path, mask_train, output_path))

    dask.compute(*tasks)    

# Create dataframes for Training and Prediction masks
def raster_to_spatial_df(raster_path):
    with rasterio.open(raster_path) as src:
        mask = src.read(1)

        # Extract geometries where mask == 1
        shapes_gen = shapes(mask, mask=mask == 1, transform=src.transform)

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, value in shapes_gen]}, crs=src.crs)
        # gdf = gdf.to_crs(epsg=32617)  # Skips if already in utm zone 17

    return gdf

# 2. prepare_subgrids - saves intersecting sub-grids to the gpkg, filters w/ masks
# TODO this is duplicating sub tiles, added a line to temp fix it but should review
def prepare_subgrids(grid_gpkg, mask_gdf, output_dir):
    print("Preparing grid tiles and sub-grids...")
    
    # # Read and transform the grid tiles to match the CRS of the mask
    # grid_tiles = gpd.read_file(grid_gpkg)
    # grid_tiles = grid_tiles.to_crs(mask_gdf.crs)
    
    # Use the subgrids from master grids geopackage and matches the CRS of the mask
    sub_grids = gpd.read_file(grid_gpkg, layer="Model_sub_Grid_Tiles").to_crs(mask_gdf.crs)
    
    # Filter sub-grids that intersect with the mask points
    intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf, how="inner", predicate='intersects')
    
    # Save intersecting sub-grids to a GeoPackage
    intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
    intersecting_sub_grids.to_file(os.path.join(output_dir, "intersecting_sub_grids.gpkg"), driver="GPKG")
    
    # # Save the training sub-grid extents, # TODO I'm not sure if grid_tile_extents is used later
    # intersecting_sub_grids[['tile', 'geometry']].to_file(os.path.join(output_dir, "grid_tile_extents.gpkg"), driver="GPKG")    
    
    return intersecting_sub_grids

# 5. process_rasters - processes raster files by clipping, extracting values, 
# combining them, calculating depth changes, and saving the data per tile
def clip_rasters_by_tile(sub_grid_gpkg, raster_dir, output_dir, data_type):
    # check sub_grid_gpkg path
    sub_grids = gpd.read_file(sub_grid_gpkg, layer='intersecting_sub_grids')
    
    raster_files = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]
    
    for _, sub_grid in sub_grids.iterrows():
        tile_name = sub_grid['Tile_ID']  # Ensure sub-grid has a `tile` column
        tile_extent = sub_grid.geometry.bounds  # Get spatial extent of the tile
        
        # Create sub-folder for the tile if it doesn't exist
        tile_dir = os.path.join(output_dir, tile_name)
        os.makedirs(tile_dir, exist_ok=True)
        
        # Path to save the clipped raster data
        
        print(f"Processing {data_type} tile: {tile_name}")
        
        # Clip rasters to the tile extent and process
        clipped_data = []
        for file in raster_files:
            with rasterio.open(file) as src:
                # Crop to tile extent
                window = src.window(tile_extent[0], tile_extent[1], tile_extent[2], tile_extent[3])
                cropped_r = src.read(1, window=window)  # Read the data within the window

                # Extract raster values along with X and Y coordinates (exclude nodata values)
                raster_data = np.column_stack(np.where(cropped_r != src.nodata))  # Get coordinates (X, Y)
                raster_values = cropped_r[cropped_r != src.nodata]  # Get raster values

                # Create a DataFrame for this specific raster
                raster_df = pd.DataFrame(raster_data, columns=['X', 'Y'])
                raster_df['Value'] = raster_values
                # raster_df['FID'] = [src.index(x, y)[0] for x, y in zip(raster_df['X'], raster_df['Y'])]  # Add FID
                raster_df['Raster'] = os.path.splitext(os.path.basename(file))[0]  # Add raster file name as a column

                # Append the raster DataFrame to the list
                clipped_data.append(raster_df)

        # Combine all raster data into a single DataFrame
        combined_data = pd.concat(clipped_data, axis=0, join='outer', ignore_index=True)

        # Pivot the combined data so each raster type (e.g., 'bathy_2004', 'bathy_2006') has its own column
        combined_data_pivot = combined_data.pivot(index=['X', 'Y'], columns='Raster', values='Value').reset_index()

        # print(combined_data_pivot)
        combined_data = combined_data_pivot # TODO remove if the pivot is not needed
        # print(combined_data['bathy_2006'])
        nan_percentage = combined_data['bathy_2006'].isna().mean() * 100
        print(f"Percentage of NaNs in bathy_2006: {nan_percentage:.2f}%")
        
        # Calculate depth change columns
        combined_data['b.change.2004_2006'] = combined_data['bathy_2006'] - combined_data['bathy_2004']
        combined_data['b.change.2006_2010'] = combined_data['bathy_2010'] - combined_data['bathy_2006']
        combined_data['b.change.2010_2015'] = combined_data['bathy_2015'] - combined_data['bathy_2010']
        combined_data['b.change.2015_2022'] = combined_data['bathy_2022'] - combined_data['bathy_2015']
        
        clipped_data_path = os.path.join(tile_dir, f"{tile_name}_{data_type}_clipped_data.parquet")
        combined_data.to_parquet(clipped_data_path)

        # clipped_data_path = os.path.join(tile_dir, f"{tile_name}_{data_type}_clipped_data.csv")
        # combined_data.to_csv(clipped_data_path, index=False)

def dask_workflow():
    client = Client(n_workers=3, threads_per_worker=2, memory_limit="16GB")
    print(f"Dask Dashboard: {client.dashboard_link}")

    with ProgressBar():  # Enables progress bar
        standardize_rasters(prediction_mask_df, training_mask_df)

    client.close()   

def tile_model_training(tiles_df, output_dir_train, year_pairs):
    print("Starting model training for all tiles...")

    # mlflow.sklearn.autolog() # I had bugs with the autolog, so I'm doing it manually
    # Set MLflow log location, will auto create the database
    mlflow.set_tracking_uri("sqlite:///C:/Users/aubrey.mccutchan/Documents/HydroHealth/mlflow.db")
        
    # 1 Static predictors (used across all year pairs)
    static_predictors = ["prim_sed_layer", "grain_size_layer", "survey_end_date"]
    results_summary = []

    for _, tile in tiles_df.iterrows():
        tile_id = tile['Tile_ID']
        tile_dir = os.path.join(output_dir_train, f'{tile_id}')
        # os.makedirs(tile_dir, exist_ok=True)

        experiment_name = f"sediment_change_model_{tile_id}"

        # Create experiment
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(name=experiment_name, artifact_location=tile_dir)

        mlflow.set_experiment(experiment_name)

        print(f"Processing tile: {tile_id}")

        # 2 Load corresponding training data
        training_data_path = os.path.join(tile_dir, f"{tile_id}_training_clipped_data.parquet")
        if not os.path.exists(training_data_path):
            print(f" - Training data missing for tile: {tile_id}")
            continue

        # 1. Load the training data from a Parquet file
        training_data = pd.read_parquet(training_data_path)

        # 2. Parse year pair
        for pair in year_pairs:
            start_year, end_year = map(int, pair.split("_"))

            # 3. Select dynamic predictors matching the year pair
            dynamic_predictors = [
                f"bathy_{start_year}", f"bathy_{end_year}",
                f"slope_{start_year}", f"slope_{end_year}",
                f"Rugosity_{start_year}_nbh9", f"Rugosity_{end_year}_nbh9",
                f"hurr_count_{pair}", f"hurr_strength_{pair}",
                f"tsm_{pair}"
            ]

            # 4. Combine dynamic and static predictors
            predictors = list(set(dynamic_predictors + static_predictors) & set(training_data.columns))
            response_var = f"b.change.{pair}"

            # 5. Filter and select data for the year pair
            subgrid_data = training_data.dropna(subset=[f"bathy_{start_year}", f"bathy_{end_year}"])[predictors + [response_var]]

            if subgrid_data.empty:
                print(f"  No valid data for year pair: {pair}")
                continue

            # 7 Train Random Forest
            X = subgrid_data[predictors]
            y = subgrid_data[response_var]
            
            with mlflow.start_run():
                xgb_model = xgb.XGBRegressor(
                    n_estimators=500,  # Number of boosting rounds (equivalent to number of trees)
                    max_depth=None,  # XGBoost does not have max_features like RF, but you can control depth
                    subsample=1.0,  # Bootstrap sampling equivalent (default is 1.0, which means no row sampling)
                    colsample_bytree=0.5,  # Similar to max_features="sqrt" in RF
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )

                xgb_model.fit(X, y)

                rf_model= xgb_model

                # 8 Save model-MLflow autolog would do this but decided not to use it
                # mlflow.log_param("tile_id", tile_id)
                # mlflow.log_param("year_pair", pair)

                model_path = os.path.join(tile_dir, f"model_{pair}.pkl")
                joblib.dump(rf_model, model_path, compress=3) # compression otherwise large 5 GB files

                # mlflow.sklearn.log_model(rf_model, artifact_path=tile_dir)
                print(f" - {pair} model saved for {tile_id} in {tile_dir}")

                predictions = rf_model.predict(X)
                r2 = r2_score(y, predictions)
                mse = root_mean_squared_error(y, predictions)

                mlflow.log_metric("R2", r2)
                mlflow.log_metric("ResidualError", mse)
                
                importance_df = pd.DataFrame({
                    "Variable": predictors,
                    "Importance": rf_model.feature_importances_
                }).sort_values(by="Importance", ascending=False)
                
                # 10 Compute Performance Metrics 
                results_summary.append({
                    "Tile": tile_id,
                    "YearPair": pair,
                    "R2": r2,
                    "ResidualError": mse,
                    "Importance": list(importance_df["Importance"]),
                    "Variable": list(importance_df["Variable"])
                })

                # 9 Generate Partial Dependence Plots and Initialize PDP data
                # generate_pdp(rf_model, X, tile_dir, pair)
    
    results_df = pd.DataFrame(results_summary)
    results_csv_path = os.path.join(output_dir_train, "model_performance_summary.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("Processing complete and summary saved with mlflow.")

def generate_pdp(model, X, tile_dir, pair):
    pdp_data = {}
    
    for feature in X.columns:
        pdp_values = partial_dependence(model, X, [feature])
        pdp_data[feature] = pd.DataFrame({
            "Predictor": feature,
            "Value": pdp_values["grid_values"][0],
            "Prediction": pdp_values["average"][0]
        })
    
    pdp_df = pd.concat(pdp_data.values(), ignore_index=True)
    pdp_path = os.path.join(tile_dir, f"pdp_{pair}.pkl")
    joblib.dump(pdp_df, pdp_path)
    print(f"  PDP saved for year pair: {pair}")
    
    plt.figure(figsize=(12, 8))
    for feature, df in pdp_data.items():
        sns.lineplot(x=df["Value"], y=df["Prediction"], label=feature)
    
    plt.xlabel("Feature Value")
    plt.ylabel("Predicted Bathy Change (m)")
    plt.legend()
    plt.title(f"Partial Dependence Plot for {pair}")
    plt.savefig(os.path.join(tile_dir, f"pdp_{pair}.jpeg"), dpi=300)
    plt.close()
    print(f"  PDP plot saved for year pair: {pair}")

def clean_tile_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'Tile' in os.path.basename(dirpath):  # Check if folder name contains 'Tile'
            for filename in filenames:
                if 'Tile' not in filename:  # Delete files that don't have 'Tile' in their name
                    file_path = os.path.join(dirpath, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

## code for processing the model input rasters. Input variables are stored in a single zarr file for each tile
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    # create_survey_end_date_tiffs() # works

    mask_prediction = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\prediction.mask.UTM17_8m.tif"
    mask_training = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\training.mask.UTM17_8m.tif"

    prediction_mask_df = raster_to_spatial_df(mask_prediction) # part 3b, create dataframe from Prediction extent mask
    training_mask_df = raster_to_spatial_df(mask_training) # part 3a, create dataframe from Training extent mask

    dask_workflow()  # Executes the entire Dask workflow to standarize rasters

    # input_dir_pred = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" 
    # output_dir_pred = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles_testing"\
    # input_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed"
    # output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles_testing"

    input_dir_pred = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\prediction_processed" 
    output_dir_pred = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\prediction_tiles" 
    input_dir_train = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\training_processed"   
    output_dir_train = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\training_tiles"
    model_outputs_path = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\model_outputs"
   
    os.makedirs(output_dir_pred, exist_ok=True)   
    os.makedirs(input_dir_train, exist_ok=True)    
    os.makedirs(output_dir_train, exist_ok=True)
    os.makedirs(model_outputs_path, exist_ok=True)

    grid_gpkg = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Tessellation\Master_Grids.gpkg"

    # prepare_subgrids(grid_gpkg=grid_gpkg, mask_gdf=prediction_mask_df, output_dir=output_dir_pred)
    # prepare_subgrids(grid_gpkg=grid_gpkg, mask_gdf=training_mask_df, output_dir=output_dir_train) 

    prediction_sub_grids = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\prediction_tiles\intersecting_sub_grids.gpkg"
    training_sub_grids = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\training_tiles\intersecting_sub_grids.gpkg"

    clip_rasters_by_tile(sub_grid_gpkg=prediction_sub_grids, raster_dir=input_dir_pred, output_dir=output_dir_pred, data_type="prediction")
    clip_rasters_by_tile(sub_grid_gpkg=training_sub_grids, raster_dir=input_dir_pred, output_dir=output_dir_train, data_type="training")

    tiles_df = gpd.read_file(training_sub_grids)
    year_pairs = ["2004_2006", "2006_2010", "2010_2015", "2015_2022"]

    # tile_model_training(tiles_df, output_dir_train, year_pairs)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(20) 

######### part 2 model training #########
# if __name__ == "__main__":
    # clean_tile_folders(r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\Training.data.grid_tiles')

    # client = Client(n_workers=1, threads_per_worker=2, memory_limit="16GB")
    # client.run(gc.collect)
    # print(client)

    # print("Dask Dashboard is running at http://localhost:8787")

    # with cProfile.Profile() as pr:
    #     tiles_df = gpd.read_file(r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\Training.data.grid_tiles\intersecting_sub_grids.gpkg")

    #     delayed_task = delayed(tile_model_training)(tiles_df, output_dir_train, year_pairs)
    #     print("Dask Task Progress:")
    #     result = compute(delayed_task)  # Compute once
    #     progress(result)  # Track progress

    # stats = pstats.Stats(pr)
    # stats.strip_dirs().sort_stats("cumulative").print_stats(10)  
    
