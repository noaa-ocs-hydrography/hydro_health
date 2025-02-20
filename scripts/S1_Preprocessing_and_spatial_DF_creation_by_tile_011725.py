import os
import glob
from lxml import etree
from datetime import datetime
import pandas as pd
import rasterio
import numpy as np
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import Polygon, shape
from rasterio.features import shapes
import pyreadr # only for reading the .rds files, I need to save them as something different
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from dask import delayed, compute
from dask.distributed import Client, progress
import cProfile
import pstats
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.inspection import partial_dependence
import gc


## 1. is preprocess all lidar data
input_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\UTM17'
output_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\survey_date_end_python'
kml_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\RATs'

# 1. Location of support rasters for processing- N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw
prediction_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Model_variables\Prediction\raw_testing'
prediction_out = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Model_variables\Prediction\processed_python'
training_out = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Model_variables\Training\processed_python'

mask_prediction = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\prediction.mask.tif'
mask_training = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\training.mask.tif'

grid_gpkg = 'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg' # from Blue topo
output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles_testing"
output_dir_pred = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles_testing"
input_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed" # raster data 
input_dir_pred = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # raster data 
training_sub_grids = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles_testing/intersecting_sub_grids.gpkg"
prediction_sub_grids = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles_testing/intersecting_sub_grids.gpkg"


target_crs = 'EPSG:4326'  #WGS84, # TODO this needs to be changed to take crs from the grid, will be a UTM zone
# that will changed based on location
target_res = 8  # 8m resolution

# Create output "processed" folders if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(prediction_out, exist_ok=True)
os.makedirs(training_out, exist_ok=True)  
os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_pred, exist_ok=True)      

# 1. get all tiff files in a specific folder, should work correctly but check the crs
def create_survey_end_date_tiffs():
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
        compress='lzw',
        crs=src.crs, 
        transform=transform,
        nodata=nodata) as dst: dst.write(reclassified_band, 1)       

## 2. standardize all rasters to have same X, Y, FID # TODO need to double check I am not missing any steps
# for training vs predicted
# 1. Location of support rasters for processing- N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw

def process_raster(raster_path, output_path, target_crs="EPSG:4326", target_res=8):
    # Step 1: Use GDAL to reproject & resample in one go
    temp_path = raster_path.replace(".tif", "_temp.tif")
    gdal.Warp(
            temp_path, raster_path,
            dstSRS=target_crs,
            xRes=target_res, yRes=target_res,
            resampleAlg=gdal.GRA_Bilinear,
            format="GTiff",
            options=gdal.WarpOptions(options=["COMPRESS=LZW"])
        )

    # Step 2: Apply bathymetry mask if needed
    with rasterio.open(temp_path) as src:
        data = src.read(1)
        transform = src.transform

        if '_bathy' in os.path.basename(raster_path):
            data[data > 0] = np.nan  # Set positive values to NaN

        # Step 3: Write final output
        out_meta = src.meta.copy()
        out_meta.update({"dtype": "float32"})

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.write(data, 1)

    # Cleanup temporary file
    os.remove(temp_path)

## 2. Standardize all rasters (PREDICTION Extent first as its larger)----
#makes all same extent, for processing into spatial points dataframe and removes all land based elevation values > 0 as well
#### NOTE:------ It is critical that BOTH the prediction and training datasets must have the same values of X, Y and FID within each, ---###
# although they are different extents, the smaller training data must be a direct subset of the prediction data
# for variables they have in common, even if the datasets vary between the two final datasets, we will divide the pertiant 
#columns afterward.
def standardize_rasters():
    max_size = 50 * 1024 * 1024  # 100 MB in bytes
    prediction_files = [
    f for f in os.listdir(prediction_dir)
    if f.endswith('.tif') and os.path.getsize(os.path.join(prediction_dir, f)) < max_size]

    for file in prediction_files:
        print(f'Processing {file}...')
        input_path = os.path.join(prediction_dir, file)
        output_path = os.path.join(prediction_out, file)
        # 12. write out the updated files to the processed folder 
        process_raster(input_path, mask_prediction, output_path)

    ## 2 Standardize all rasters (TRAINING Extent - sub sample of prediction extent)----
    #- THIS IS A DIRECT SUBSET OF THE PREDICTION AREA - clipped using the training mask. 
    training_files = [f for f in os.listdir(prediction_out) if f.endswith('.tif') and not f.startswith('blue_topo')]

    for file in training_files:
        print(f'Processing {file}...')
        input_path = os.path.join(prediction_out, file)  # use processed prediction rasters
        output_path = os.path.join(training_out, file)
        # 12. write out the updated files to the processed folder 
        process_raster(input_path, mask_training, output_path, is_training=True)

# ## 3. Create dataframes for Training and Prediction masks
def raster_to_spatial_df(raster_path):
    with rasterio.open(raster_path) as src:
        mask = src.read(1)  # Read first band

        # Extract geometries where mask == 1
        shapes_gen = shapes(mask, mask=mask == 1, transform=src.transform)

        # Convert to GeoDataFrame
        gdf = gpd.GeoDataFrame({'geometry': [shape(geom) for geom, value in shapes_gen]}, crs=src.crs)

        gdf = gdf.to_crs(epsg=4326)  # Skips if already in wgs84

    return gdf

# ## 4. Helper functions, need to check the updated code from steph
# 1. split_tile - subdivide a tile into smaller grids, this function is used in prepare_subgrids
def split_tile(tile, crs):
    # Get the bounding box of the input tile (xmin, ymin, xmax, ymax)
    xmin, ymin, xmax, ymax = tile.geometry.bounds

    # Calculate the midpoints for x and y
    mid_x = (xmin + xmax) / 2
    mid_y = (ymin + ymax) / 2
    
    # Define the sub-grid polygons in clockwise order
    sub_grids = [
        Polygon([(xmin, ymin), (mid_x, ymin), (mid_x, mid_y), (xmin, mid_y)]),  # Bottom-left
        Polygon([(mid_x, ymin), (xmax, ymin), (xmax, mid_y), (mid_x, mid_y)]),  # Bottom-right
        Polygon([(mid_x, mid_y), (xmax, mid_y), (xmax, ymax), (mid_x, ymax)]),  # Top-right
        Polygon([(xmin, mid_y), (mid_x, mid_y), (mid_x, ymax), (xmin, ymax)])   # Top-left
    ]

    # Create sub-grid IDs based on the original tile's ID (tile + 1, 2, 3, 4)
    sub_grid_ids = [f"Tile_{tile['tile']}_{i+1}" for i in range(4)]

    # Create a DataFrame with the sub-grid details
    sub_grid_data = {
        'tile': sub_grid_ids,
        'xmin': [s.bounds[0] for s in sub_grids],
        'ymin': [s.bounds[1] for s in sub_grids],
        'xmax': [s.bounds[2] for s in sub_grids],
        'ymax': [s.bounds[3] for s in sub_grids],
        'geometry': sub_grids
    }
    
    # Convert the dictionary into a GeoDataFrame
    sub_grid_df = gpd.GeoDataFrame(sub_grid_data, crs=crs)
    
    return sub_grid_df
# 2. prepare_subgrids - splits grid tiles into sub-grids, filters w/ mask, 
# saves intersecting sub-grids to the gpkg
# TODO this is duplicating sub tiles, added a line to temp fix it but should review when time
def prepare_subgrids(grid_gpkg, mask_gdf, output_dir):
    print("Preparing grid tiles and sub-grids...")
    
    # Read and transform the grid tiles to match the CRS of the mask
    grid_tiles = gpd.read_file(grid_gpkg)
    grid_tiles = grid_tiles.to_crs(mask_gdf.crs)
    
    # Split grid tiles into sub-grids
    sub_grids_list = [split_tile(tile, crs=mask_gdf.crs) for _, tile in grid_tiles.iterrows()]
    sub_grids = gpd.GeoDataFrame(pd.concat(sub_grids_list, ignore_index=True), crs=grid_tiles.crs)
    
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
# Need for the both the training and the prediction data
def clip_rasters_by_tile(sub_grid_gpkg, raster_dir, output_dir, data_type):
    # check sub_grid_gpkg path
    sub_grids = gpd.read_file(sub_grid_gpkg, layer='intersecting_sub_grids')
    
    raster_files = [os.path.join(raster_dir, f) for f in os.listdir(raster_dir) if f.endswith('.tif')]
    
    for _, sub_grid in sub_grids.iterrows():
        tile_name = sub_grid['tile']  # Ensure sub-grid has a `tile` column
        tile_extent = sub_grid.geometry.bounds  # Get spatial extent of the tile
        
        # Create sub-folder for the tile if it doesn't exist
        tile_dir = os.path.join(output_dir, tile_name)
        os.makedirs(tile_dir, exist_ok=True)
        
        # Path to save the clipped raster data
        clipped_data_path = os.path.join(tile_dir, f"{tile_name}_{data_type}_clipped_data.csv")
        
        print(f"Processing {data_type} tile: {tile_name}")
        
        # Clip rasters to the tile extent and process
        clipped_data = []
        for file in raster_files:
            with rasterio.open(file) as src:
                # Crop to tile extent
                window = src.window(tile_extent[0], tile_extent[1], tile_extent[2], tile_extent[3])
                cropped_r = src.read(1, window=window)  # Read the data within the window, # TODO might be able to use the
                # mask method for simplicity but need to check the subgrid geopackage, getting an error
                
                # Extract raster values along with X and Y coordinates
                raster_data = np.column_stack(np.where(cropped_r != src.nodata))  # Get coordinates
                raster_values = cropped_r[cropped_r != src.nodata]  # Get raster values
                raster_df = pd.DataFrame(raster_data, columns=['X', 'Y'])
                raster_df['Value'] = raster_values
                raster_df['FID'] = [src.index(x, y)[0] for x, y in zip(raster_df['X'], raster_df['Y'])]  # Add FID
                raster_df['Raster'] = os.path.splitext(os.path.basename(file))[0]  # Add raster file name as column
                
                clipped_data.append(raster_df)
        
        # Combine all rasters into a single data frame
        combined_data = pd.concat(clipped_data, axis=0, join='outer', ignore_index=True)
        
        # Calculate depth change columns
        combined_data['b.change.2004_2006'] = combined_data['bathy_2006'] - combined_data['bathy_2004']
        combined_data['b.change.2006_2010'] = combined_data['bathy_2010'] - combined_data['bathy_2006']
        combined_data['b.change.2010_2015'] = combined_data['bathy_2015'] - combined_data['bathy_2010']
        combined_data['b.change.2015_2022'] = combined_data['bathy_2022'] - combined_data['bathy_2015']
        
        # Save the data as CSV? she saves as an rds file
        # TODO need to think how we want to access the data for each tile
        combined_data.to_csv(clipped_data_path, index=False)


######## part 2 Model training, I'm puting everything in this script bc overlapping files
# Initialize Dask Client for parallel processing

year_pairs = ["2004_2006", "2006_2010", "2010_2015", "2015_2022"]
# 1. Model training over all subgrids

def process_tiles(tiles_df, output_dir_train, year_pairs):
    print("Starting processing of all tiles...")

    # Start MLflow Run
    # mlflow.sklearn.autolog()
    # Set MLflow log location, will auto create the database
    mlflow.set_tracking_uri("sqlite:///C:/Users/aubrey.mccutchan/Documents/HydroHealth/mlflow.db")
        

    # 1 Static predictors (used across all year pairs)
    static_predictors = ["prim_sed_layer", "grain_size_layer", "survey_end_date"]
    results_summary = []

    for _, tile in tiles_df.iterrows():
        tile_id = tile["tile"]
        tile_dir = os.path.join(output_dir_train, f'{tile_id}')
        # os.makedirs(tile_dir, exist_ok=True)

        experiment_name = "sediment_change_model"

        # Create experiment only if it doesn't exist
        if not mlflow.get_experiment_by_name(experiment_name):
            mlflow.create_experiment(name=experiment_name, artifact_location=tile_dir)

        mlflow.set_experiment(experiment_name)

        print(f"Processing tile: {tile_id}")

        # 2 Load corresponding training data
        # training_data_path = os.path.join(tile_dir, f"{tile_id}_training_clipped_data.pkl")
        training_data_path = os.path.join(tile_dir, f"{tile_id}_training_clipped_data.rds")
        if not os.path.exists(training_data_path):
            print(f"  Training data missing for tile: {tile_id}")
            continue

        # training_data = pd.read_pickle(training_data_path)
        result = pyreadr.read_r(training_data_path)  # Returns a dictionary, only for .rds file type
        training_data = next(iter(result.values()))  # Extracts the dataframe, only for .rds file type

        # 3. Parse year pair
        for pair in year_pairs:
            start_year, end_year = map(int, pair.split("_"))

            # 4 Select dynamic predictors matching the year pair
            dynamic_predictors = [
                f"bathy_{start_year}", f"bathy_{end_year}",
                f"slope_{start_year}", f"slope_{end_year}",
                f"hurr_count_{pair}", f"hurr_strength_{pair}",
                f"tsm_{pair}"
            ]

            # 5 Combine dynamic and static predictors
            predictors = list(set(dynamic_predictors + static_predictors) & set(training_data.columns))
            response_var = f"b.change.{pair}"

            # 6 Filter and select data for the year pair
            subgrid_data = training_data.dropna(subset=[f"bathy_{start_year}", f"bathy_{end_year}"])[predictors + [response_var]].dropna()

            if subgrid_data.empty:
                print(f"  No valid data for year pair: {pair}")
                continue

            # 7 Train Random Forest
            X = subgrid_data[predictors]
            y = subgrid_data[response_var]
            
            with mlflow.start_run():
                rf_model = RandomForestRegressor(
                n_estimators=500, # number of trees in the forest
                max_features="sqrt",
                bootstrap=True,
                random_state=42)
                # n_jobs=-1
                
                rf_model.fit(X, y)
                # rf_model.estimators_ = None 

                # 8 Save model-MLflow should auto do it
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


# ## 6. Run model processing
# if __name__ == '__main__':
    # create_survey_end_date_tiffs() # part 1

    # standardize_rasters() # part 2, #TODO fix mask, and is there a faster process to use or cloud solution?

    # training_mask_df = raster_to_spatial_df(mask_training) # part 3a, create dataframe from Training extent mask
    # prediction_mask_df = raster_to_spatial_df(mask_prediction) # part 3b, create dataframe from Prediction extent mask

    # prepare_subgrids(grid_gpkg=grid_gpkg, mask_gdf=training_mask_df, output_dir=output_dir_train) # part 4a
    # prepare_subgrids(grid_gpkg=grid_gpkg, mask_gdf=prediction_mask_df, output_dir=output_dir_pred) # part 4b

    # clip_rasters_by_tile(sub_grid_gpkg=training_sub_grids, raster_dir=input_dir_train, output_dir=output_dir_train, data_type="training") # part 5a
    # clip_rasters_by_tile(sub_grid_gpkg=prediction_sub_grids, raster_dir=input_dir_pred, output_dir=output_dir_pred, data_type="prediction") # part 5a

# model_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/Tile_BH4S2577_4/model_2015_2022.pkl"
# mod1 = joblib.load(model_path)

def clean_tile_folders(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if 'Tile' in os.path.basename(dirpath):  # Check if folder name contains 'Tile'
            for filename in filenames:
                if 'Tile' not in filename:  # Delete files that don't have 'Tile' in their name
                    file_path = os.path.join(dirpath, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")

tiles_df = gpd.read_file(r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\Training.data.grid_tiles\intersecting_sub_grids.gpkg")
output_dir_train = r"C:\Users\aubrey.mccutchan\Documents\HydroHealth\Training.data.grid_tiles"

if __name__ == "__main__":
    clean_tile_folders(r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\Training.data.grid_tiles')

    client = Client(n_workers=1, threads_per_worker=2, memory_limit="16GB")
    client.run(gc.collect)
    print(client)

    print("Dask Dashboard is running at http://localhost:8787")
    with cProfile.Profile() as pr:
        delayed_task = delayed(process_tiles)(tiles_df, output_dir_train, year_pairs)  
        result = compute(delayed_task)  # Dask actually executes the function
    
    stats = pstats.Stats(pr)
    stats.strip_dirs().sort_stats("cumulative").print_stats(10)
    print(mlflow.get_artifact_uri())


    print("Dask Task Progress:")
    progress(result)
    
