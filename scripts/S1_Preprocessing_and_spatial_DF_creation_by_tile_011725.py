import os
import glob
from lxml import etree
from datetime import datetime
import pandas as pd
import rasterio
import numpy as np
import geopandas as gpd
from osgeo import gdal
import multiprocessing
from shapely.geometry import Polygon



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


target_crs = 'EPSG:4326'  #WGS84
target_res = 8  # 8m resolution

# Create output "processed" folders if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(prediction_out, exist_ok=True)
os.makedirs(training_out, exist_ok=True)    

# 1. get all tiff files in a specific folder
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

# create_survey_end_date_tiffs()        

## 2. standardize all rasters to have same X, Y, FID # TODO need to double check I am not missing any steps
# for training vs predicted
# 1. Location of support rasters for processing- N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw
# 2. Create output "processed" folders if they don't exist

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
    # Open the raster
    with rasterio.open(raster_path) as src:
        # Extract coordinates and values from the raster
        rows, cols = src.shape
        x_min, y_min, x_max, y_max = src.bounds
        x_res = (x_max - x_min) / cols
        y_res = (y_max - y_min) / rows
        
        # 1. Extract mask coordinates and match X and Y from mask with 1 and 2 coords f - Review lin 363
        # Generate the coordinates (rows, cols) for each pixel
        xs, ys = rasterio.transform.xy(src.transform, range(rows), range(cols))
        coords = [(x, y) for x in xs for y in ys]
        
        # Extract the pixel values
        values = list(src.read(1).flatten())
        
        # 2. Create FID column based on unique XY columns
        df = pd.DataFrame({'X': [coord[0] for coord in coords],
                           'Y': [coord[1] for coord in coords],
                           'Value': values})
        
        # 3. filter DF where mask == 1
        df = df[df['Value'] == 1]
        
        # 4. write out the filtered dataframe
        geometry = [Point(x, y) for x, y in zip(df['X'], df['Y'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
        
        return gdf

# # Save as .pkl, Python's version of .Rds
# training_mask_df.to_pickle('N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/training_mask_df.pkl')
# prediction_mask_df.to_pickle('N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/prediction_mask_df.pkl')

# # Function to create a spatial dataframe from a raster
# training_mask_df_loaded = pd.read_pickle('N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/training_mask_df.pkl')
# prediction_mask_df_loaded = pd.read_pickle('N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/prediction_mask_df.pkl')


# ## 4. Helper functions, need to check the updated code from steph
# 1. split_tile - subdivide a tile into smaller grids
def split_tile(tile):
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

    # Create sub-grid IDs based on the original tile's ID (tile_id + 1, 2, 3, 4)
    sub_grid_ids = [f"Tile_{tile['tile_id']}_{i+1}" for i in range(4)]

    # Create a DataFrame with the sub-grid details
    sub_grid_data = {
        'tile_id': sub_grid_ids,
        'xmin': [s.bounds[0] for s in sub_grids],
        'ymin': [s.bounds[1] for s in sub_grids],
        'xmax': [s.bounds[2] for s in sub_grids],
        'ymax': [s.bounds[3] for s in sub_grids],
        'geometry': sub_grids
    }
    
    # Convert the dictionary into a GeoDataFrame
    sub_grid_df = gpd.GeoDataFrame(sub_grid_data, crs=tile.crs)
    
    return sub_grid_df
# 2. prepare_subgrids - splits grid tiles into sub-grids, filters w/ mask, 
# saves intersecting sub-grids to the gpkg
def prepare_subgrids(grid_gpkg, mask_df, mask, output_dir):
    print("Preparing grid tiles and sub-grids...")
    
    # Read and transform the grid tiles to match the CRS of the mask
    grid_tiles = gpd.read_file(grid_gpkg)
    grid_tiles = grid_tiles.to_crs(mask.crs)
    
    # Convert mask_df to a GeoDataFrame
    training_gdf = gpd.GeoDataFrame(mask_df, geometry=gpd.points_from_xy(mask_df['X'], mask_df['Y']), crs=mask.crs)
    
    # Split grid tiles into sub-grids
    sub_grids_list = [split_tile(tile) for _, tile in grid_tiles.iterrows()]
    sub_grids = gpd.GeoDataFrame(pd.concat(sub_grids_list, ignore_index=True), crs=grid_tiles.crs)
    
    # Filter sub-grids that intersect with the mask points
    intersecting_sub_grids = gpd.sjoin(sub_grids, training_gdf, how="inner", op='intersects')
    
    # Save intersecting sub-grids to a GeoPackage
    intersecting_sub_grids.to_file(os.path.join(output_dir, "intersecting_sub_grids.gpkg"), driver="GPKG")
    
    # Save the training sub-grid extents
    intersecting_sub_grids[['tile_id', 'geometry']].to_file(os.path.join(output_dir, "grid_tile_extents.gpkg"), driver="GPKG")    
    
    return intersecting_sub_grids

# 3. process_rasters - processes raster files by clipping, extracting values, 
# combining them, calculating depth changes, and saving the data per tile


# ## 5. Run model processing
# 1. run prediction sub grid 
# 2. run training sub grid
# 3. call process_rasters() for training 
# 4. call process_rasters() for prediction

if __name__ == '__main__':
    # create_survey_end_date_tiffs() 
    standardize_rasters() # part 2
    # make a dataframe from Training extent mask
    # training_mask_df = raster_to_spatial_df(mask_training)
    # make a dataframe from Prediction extent mask
    # prediction_mask_df = raster_to_spatial_df(mask_prediction)