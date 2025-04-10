import os
import glob
from lxml import etree
from datetime import datetime
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
from shapely.geometry import shape
from rasterio.features import shapes
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar
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
                    else None  
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

        with rasterio.open(output_path, 'w', **kwargs) as dst:
            reproject(
                source=data,  # Use masked data
                destination=rasterio.band(dst, 1),
                src_transform=mask_transform,  # Use the transform of the clipped raster
                src_crs=src.crs,
                dst_transform=transform,  # Use new projection transform
                dst_crs=target_crs,
                resampling=Resampling.nearest,
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

    tasks = []
    for file in prediction_files:
        print(f'Processing {file}...')
        input_path = os.path.join(prediction_dir, file)
        output_path = os.path.join(prediction_out, file)

        tasks.append(dask.delayed(process_raster)(input_path, mask_pred, output_path))

    dask.compute(*tasks)  
    
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
        valid_mask = mask == 1

        shapes_gen = shapes(mask, valid_mask, transform=src.transform)

        gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, value in shapes_gen]}, crs=src.crs)
        # gdf = gdf.to_crs(epsg=32617)  # Skips if already in utm zone 17

    return gdf

# 2. prepare_subgrids - saves intersecting sub-grids to the gpkg, filters w/ masks
def prepare_subgrids(mask_gdf, output_dir):
    grid_gpkg = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Tessellation\Master_Grids.gpkg"

    print("Preparing grid tiles and sub-grids...")
    
    sub_grids = gpd.read_file(grid_gpkg, layer="Model_sub_Grid_Tiles").to_crs(mask_gdf.crs)
    
    # Filter sub-grids that intersect with the mask points
    intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf, how="inner", predicate='intersects')
    
    intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")
    intersecting_sub_grids.to_file(os.path.join(output_dir, "intersecting_sub_grids.gpkg"), driver="GPKG") 
    
    return intersecting_sub_grids

def tile_process(sub_grid, raster_dir, output_dir, data_type):
    raster_files = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]

    tile_name = sub_grid['Tile_ID']  # Ensure sub-grid has a `Tile_ID` column
    tile_extent = sub_grid.geometry.bounds  # Get spatial extent of the tile
    
    tile_dir = os.path.join(output_dir, tile_name)
    os.makedirs(tile_dir, exist_ok=True)
        
    print(f"Processing {data_type} tile: {tile_name}")
    
    clipped_data = []
    for file in raster_files:

        with rasterio.open(file) as src:
            # Crop to tile extent
            window = src.window(tile_extent[0], tile_extent[1], tile_extent[2], tile_extent[3])
            cropped_r = src.read(1, window=window)  

            mask = cropped_r != src.nodata
            raster_data = np.column_stack(np.where(mask))  # Get coordinates (Y, X)
            raster_values = cropped_r[mask]  # Get raster values

            x_vals = raster_data[:, 1]  # Column indices (X values)
            y_vals = raster_data[:, 0]  # Row indices (Y values)

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
    
    # Calculate depth change columns
    combined_data['b.change.2004_2006'] = combined_data['bathy_2006'] - combined_data['bathy_2004']
    combined_data['b.change.2006_2010'] = combined_data['bathy_2010'] - combined_data['bathy_2006']
    combined_data['b.change.2010_2015'] = combined_data['bathy_2015'] - combined_data['bathy_2010']
    combined_data['b.change.2015_2022'] = combined_data['bathy_2022'] - combined_data['bathy_2015']
    
    clipped_data_path = os.path.join(tile_dir, f"{tile_name}_{data_type}_clipped_data.parquet")
    combined_data.to_parquet(clipped_data_path)

# 5. process_rasters - processes raster files by clipping, extracting values, 
# combining them, calculating depth changes, and saving the data per tile
def clip_rasters_by_tile(sub_grid_gpkg, raster_dir, output_dir, data_type):
    sub_grids = gpd.read_file(sub_grid_gpkg, layer='intersecting_sub_grids')
    
    tasks = []
    for _, sub_grid in sub_grids.iterrows():
        tasks.append(dask.delayed(tile_process)(sub_grid, raster_dir, output_dir, data_type))

    dask.compute(*tasks) 

def dask_workflow():
    client = Client(n_workers=1, threads_per_worker=1, memory_limit="16GB")
    print(f"Dask Dashboard: {client.dashboard_link}")

    with ProgressBar():  
        standardize_rasters(prediction_mask_df, training_mask_df)

        clip_rasters_by_tile(sub_grid_gpkg=prediction_sub_grids, raster_dir=input_dir_pred, output_dir=output_dir_pred, data_type="prediction")
        clip_rasters_by_tile(sub_grid_gpkg=training_sub_grids, raster_dir=input_dir_pred, output_dir=output_dir_train, data_type="training")
    
    client.close()   

# TODO more work is needed on the training and prediction code
def tile_model_training(tiles_df, output_dir_train, year_pairs):
    print("Starting model training for all tiles...")

    mlflow.set_tracking_uri("sqlite:///C:/Users/aubrey.mccutchan/Documents/HydroHealth/mlflow.db")
    
    # Static predictors (used across all year pairs)
    static_predictors = ["prim_sed_layer", "grain_size_layer", "survey_end_date"]

    for _, tile in tiles_df.iterrows():
        tile_id = tile['Tile_ID']
        tile_dir = os.path.join(output_dir_train, f'{tile_id}')

        experiment_name = f"sediment_change_model_{tile_id}"

        # Create experiment if it doesn't exist
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(name=experiment_name, artifact_location=tile_dir)

        mlflow.set_experiment(experiment_name)
        print(f"Processing tile: {tile_id}")

        training_data_path = os.path.join(tile_dir, f"{tile_id}_training_clipped_data.parquet")
        if not os.path.exists(training_data_path):
            print(f" - Training data missing for tile: {tile_id}")
            count += 1
            continue

        training_data = pd.read_parquet(training_data_path)

        for pair in year_pairs:
            start_year, end_year = map(int, pair.split("_"))

            # Define dynamic predictors for the year pair
            dynamic_predictors = [
                f"bathy_{start_year}", f"bathy_{end_year}",
                f"slope_{start_year}", f"slope_{end_year}",
                f"Rugosity_{start_year}_nbh9", f"Rugosity_{end_year}_nbh9",
                f"hurr_count_{pair}", f"hurr_strength_{pair}",
                f"tsm_{pair}"
            ]

            # Combine dynamic and static predictors
            predictors = list(set(dynamic_predictors + static_predictors) & set(training_data.columns))
            response_var = f"b.change.{pair}"

            # Filter and select data for the current year pair
            subgrid_data = training_data.dropna(subset=[f"bathy_{start_year}", f"bathy_{end_year}"])[predictors + [response_var]]

            if subgrid_data.empty:
                print(f"  No valid data for year pair: {pair}")
                continue

            # Train the model
            X = subgrid_data[predictors]
            y = subgrid_data[response_var]
            
            with mlflow.start_run():
                # Initialize XGBoost regressor
                # Create DMatrix for XGBoost (this is the optimized data structure used by XGBoost)
                dtrain = xgb.DMatrix(X, label=y)

                # Set the hyperparameters
                params = {
                    'max_depth': 6,                # Controls the depth of the trees
                    'eta': 0.01,                   # Learning rate (shrinkage factor)
                    'subsample': 0.7,              # Fraction of data to use per boosting round (bootstrapping)
                    'colsample_bytree': 0.8,       # Fraction of features to use per tree (feature bagging)
                    # 'objective': 'reg:squarederror', # Regression objective
                    # 'eval_metric': 'rmse',          # Evaluation metric (Root Mean Squared Error)
                    # 'nthread': 1                    # Number of threads for parallel processing (optional)
                }

                # Specify the number of boosting rounds (iterations)
                nrounds = 500

                # Train the model using xgb.train (equivalent to xgb.train in R)
                xgb_model = xgb.train(params, dtrain, num_boost_round=nrounds)

                mlflow.log_param("tile_id", tile_id)
                mlflow.log_param("year_pair", pair)
                mlflow.log_param("n_estimators", 500)
                mlflow.log_param("max_depth", None)
                mlflow.log_param("subsample", 1.0)
                mlflow.log_param("colsample_bytree", 0.5)

                model_path = os.path.join(tile_dir, f"model_{pair}.ubj")
                xgb_model.save_model(model_path)
                print(f" - Model saved locally at: {model_path}")

                # Log the model as an artifact in MLflow
                # mlflow.log_artifact(model_path, artifact_path="model")

                # Log feature importances as an artifact (optional)
                # importance_df = pd.DataFrame({
                #     "Variable": predictors,
                #     "Importance": xgb_model.feature_importances_
                # }).sort_values(by="Importance", ascending=False)

                # # Save feature importance as an artifact in MLflow
                # importance_path = os.path.join(tile_dir, f"importance_{pair}.json")
                # importance_df.to_json(importance_path, orient="records")  # Save as JSON instead of CSV
                # mlflow.log_artifact(importance_path)

                # Optionally: you can also log the performance summary as an artifact.
                # summary_metrics = {
                #     "Tile": tile_id,
                #     "YearPair": pair,
                #     "Description": "Model trained but no predictions performed"
                # }

                # Log performance summary as an artifact (if you want a JSON file for reference)
                # summary_metrics_path = os.path.join(tile_dir, f"summary_{pair}.json")
                # with open(summary_metrics_path, 'w') as f:
                #     json.dump(summary_metrics, f)
                # mlflow.log_artifact(summary_metrics_path)


                # 9 Generate Partial Dependence Plots and Initialize PDP data
                # generate_pdp(xgb_model, X, tile_dir, pair)

    print("Model training complete and logged to MLflow.")

def tile_model_predictions(tiles_df, output_dir_train, year_pairs):
    print("Starting model prediction for all tiles...")

    static_predictors = ["prim_sed_layer", "grain_size_layer", "survey_end_date"]
    results_summary = []

    for _, tile in tiles_df.iterrows():
        tile_id = tile['Tile_ID']
        tile_dir = os.path.join(output_dir_train, f'{tile_id}')

        print(f"Processing tile: {tile_id}")

        # 1. Load the pre-trained models and make predictions for each year pair
        for pair in year_pairs:
            start_year, end_year = map(int, pair.split("_"))

            # 2. Select dynamic predictors matching the year pair
            dynamic_predictors = [
                f"bathy_{start_year}", f"bathy_{end_year}",
                f"slope_{start_year}", f"slope_{end_year}",
                f"Rugosity_{start_year}_nbh9", f"Rugosity_{end_year}_nbh9",
                f"hurr_count_{pair}", f"hurr_strength_{pair}",
                f"tsm_{pair}"
            ]

            # 3. Combine dynamic and static predictors
            predictors = list(set(dynamic_predictors + static_predictors) & set(tile.columns))
            response_var = f"b.change.{pair}"

            # 4. Filter and select data for the year pair
            prediction_data = tile.dropna(subset=[f"bathy_{start_year}", f"bathy_{end_year}"])[predictors]

            if prediction_data.empty:
                print(f"  No valid data for year pair: {pair}")
                continue

            # 5. Load the pre-trained model for this pair
            model_path = os.path.join(tile_dir, f"model_{pair}.bin")
            if not os.path.exists(model_path):
                print(f"  Model missing for year pair: {pair} in {tile_id}")
                continue

            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(model_path)

            predictions = xgb_model.predict(prediction_data[predictors])

            results_summary.append({
                "Tile": tile_id,
                "YearPair": pair,
                "Predictions": predictions.tolist()
            })
    
    results_df = pd.DataFrame(results_summary)
    results_csv_path = os.path.join(output_dir_train, "model_predictions.csv")
    results_df.to_csv(results_csv_path, index=False)
    print("Prediction complete and results saved.")

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

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    create_survey_end_date_tiffs()

    input_dir_pred = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\prediction_processed" 
    output_dir_pred = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\prediction_tiles" 
    input_dir_train = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\training_processed"   
    output_dir_train = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\training_tiles"
    model_outputs_path = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\model_outputs"

    mask_prediction = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\prediction.mask.UTM17_8m.tif"
    mask_training = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\training.mask.UTM17_8m.tif"

    prediction_mask_df = raster_to_spatial_df(mask_prediction) # part 3b, create dataframe from Prediction extent mask
    training_mask_df = raster_to_spatial_df(mask_training) # part 3a, create dataframe from Training extent mask

    prepare_subgrids(mask_gdf=prediction_mask_df, output_dir=output_dir_pred)
    prepare_subgrids(mask_gdf=training_mask_df, output_dir=output_dir_train) 

    prediction_sub_grids = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\prediction_tiles\intersecting_sub_grids.gpkg"
    training_sub_grids = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\model_data\training_tiles\intersecting_sub_grids.gpkg"

    dask_workflow()  

    tiles_training_df = gpd.read_file(training_sub_grids)
    tiles_prediction_df = gpd.read_file(prediction_sub_grids)

    year_pairs = ["2004_2006", "2006_2010", "2010_2015", "2015_2022"]

    tile_model_training(tiles_training_df, output_dir_train, year_pairs)
    tile_model_predictions(tiles_prediction_df, output_dir_train, year_pairs)

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  
    