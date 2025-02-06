import os
import glob
from lxml import etree
from datetime import datetime
import pandas as pd
import rasterio
import numpy as np
import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import geometry_mask
from shapely.geometry import box, mapping
import dask.array as da  # Dask for parallel computation
from dask.diagnostics import visualize
from dask.diagnostics import ProgressBar


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

def process_raster(raster_path, mask_path, output_path, target_crs = 'EPSG:4326', target_res=8):
    # 3.Load mask data
    with rasterio.open(mask_path) as mask_src:
        mask_crs = mask_src.crs
        mask_gdf = gpd.GeoDataFrame(geometry=[box(*mask_src.bounds)], crs=mask_crs)
        mask_gdf = mask_gdf.to_crs(target_crs)
        mask_geom = [mapping(mask_gdf.geometry.values[0])]

    with rasterio.open(raster_path) as src:
        src_crs = src.crs
        transform = src.transform

        # 8. Loop through tiff files and make sure they have the same CRS - WGS84
        if src_crs != target_crs:
            print('Reprojecting...')
            transform, width, height = calculate_default_transform(
                src_crs, target_crs, src.width, src.height, *src.bounds
            )
            data = da.from_array(np.empty((height, width), dtype=np.float32), chunks=(512, 512))
            with ProgressBar():
                result = da.compute(data)
            reproject(
                source=src.read(1),  
                destination=data.compute(),
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )
        else:
            data = da.from_array(src.read(1), chunks=(512, 512))

        # 9. verify all rasters have the same extent
        # data, transform = mask(src, mask_geom, crop=True, filled=True)

        # 10. Resample all rasters to 8m resolution
        print('Resampling...')
        new_width = int((src.bounds.right - src.bounds.left) / target_res)
        new_height = int((src.bounds.top - src.bounds.bottom) / target_res)
        resampled_data = da.from_array(np.empty((new_height, new_width), dtype=np.float32), chunks=(512, 512))

        reproject(
            source=data.compute(),
            destination=resampled_data.compute(),
            src_transform=transform,
            src_crs=target_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear
        )

        # 11. Set ground values > 0 to NA for the "_bathy" raster files
        if '_bathy' in os.path.basename(raster_path):
            print('Applying bathy mask...')
            resampled_data = da.where(resampled_data > 0, np.nan, resampled_data)

        out_meta = src.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': resampled_data.shape[0],
            'width': resampled_data.shape[1],
            'transform': transform,
            'crs': target_crs,
            'dtype': 'float32',
            'compress': 'lzw'
        })

        print('Writing output...')
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(resampled_data.compute(), 1)

## 2. Standardize all rasters (PREDICTION Extent first as its larger)----
#makes all same extent, for processing into spatial points dataframe and removes all land based elevation values > 0 as well

#### NOTE:------ It is critical that BOTH the prediction and training datasets must have the same values of X, Y and FID within each, ---###
# although they are different extents, the smaller training data must be a direct subset of the prediction data
# for variables they have in common, even if the datasets vary between the two final datasets, we will divide the pertiant 
#columns afterward.

def standardize_rasters():
    max_size = 100 * 1024 * 1024  # 100 MB in bytes
    prediction_files = [f for f in os.listdir(prediction_dir) 
                    if f.endswith('.tif')]

    for file in prediction_files:
        input_path = os.path.join(prediction_dir, file)
        output_path = os.path.join(prediction_out, file)
        # 12. write out the updated files to the processed folder 
        process_raster(input_path, mask_prediction, output_path)

    ## 2 Standardize all rasters (TRAINING Extent - sub sample of prediction extent)----
    #- THIS IS A DIRECT SUBSET OF THE PREDICTION AREA - clipped using the training mask. 
    training_files = [f for f in os.listdir(prediction_out) if f.endswith('.tif') and not f.startswith('blue_topo')]

    for file in training_files:
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
# 2. prepare_subgrids - TODO need to review
# 3. process_rasters - TODO need to review


# ## 5. Run model processing
# 1. run prediction sub grid 
# 2. run training sub grid
# 3. call process_rasters() for training 
# 4. call process_rasters() for prediction

if __name__ == '__main__':
    
    # create_survey_end_date_tiffs() 
    print('starting')
    standardize_rasters() # part 2
    # make a dataframe from Training extent mask
    # training_mask_df = raster_to_spatial_df(mask_training)
    # make a dataframe from Prediction extent mask
    # prediction_mask_df = raster_to_spatial_df(mask_prediction)