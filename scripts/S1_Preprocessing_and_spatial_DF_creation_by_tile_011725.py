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


## 1. is preprocess all lidar data
input_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\UTM17'
output_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\survey_date_end'
kml_dir = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\Now_Coast_NBS_Data\Modeling\RATs'

# 1. get all tiff files in a specific folder
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

    output_file = os.path.join(output_dir, f'{base_name}_test.tiff')

    # 12. write out new file with processed survey information
    with rasterio.open(
    output_file, 
    'w', 
    driver='GTiff',
    count=1, 
    width=width,
    height=height,
    dtype=reclassified_band.dtype, 
    crs=src.crs, 
    transform=transform,
    nodata=nodata) as dst: dst.write(reclassified_band, 1)

## 2. standardize all rasters to have same X, Y, FID # TODO need to double check I am not missing any steps
# for training vs predicted
# 1. Location of support rasters for processing- N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw
prediction_dir = 'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw'
prediction_out = 'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed'
training_out = 'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed'

mask_prediction = 'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.tif'
mask_training = 'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.tif'

target_crs = 'EPSG:4326'  #WGS84
target_res = 8  # 8m resolution

# 2. Create output "processed" folders if they don't exist
os.makedirs(prediction_out, exist_ok=True)
os.makedirs(training_out, exist_ok=True)        

def process_raster(raster_path, mask_path, output_path, is_training=False):
    # 3.Load mask data
    with rasterio.open(mask_path) as mask_src:
        mask_meta = mask_src.meta.copy()
        mask_meta.update({'driver': 'gtiff', 'crs': target_crs, 'transform': mask_src.transform})

        # 7. Loop through tiff files to make a dataframe with make vertices/points? and current tiff value at points
        mask_data = mask_src.read(1)
        mask_geom = [gpd.GeoDataFrame(geometry=mask_src.bounds.__geo_interface__).geometry.values[0]]

    with rasterio.open(raster_path) as src:
        src_crs = src.crs

        # 8. Loop through tiff files and make sure they have the same CRS - WGS84
        if src_crs != target_crs:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )
            kwargs = src.meta.copy()
            kwargs.update({'crs': target_crs, 'transform': transform, 'width': width, 'height': height})
            
            with rasterio.open(output_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

        # 9. verify all rasters have the same extent
        with rasterio.open(output_path) as temp_src:
            out_image, out_transform = mask(temp_src, mask_geom, crop=True, filled=True)
            out_meta = temp_src.meta.copy()
            out_meta.update({'height': out_image.shape[1], 'width': out_image.shape[2], 'transform': out_transform})

        # 10. Resample all rasters to 8m resolution
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            reproject(
                source=rasterio.band(temp_src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=temp_src.transform,
                src_crs=target_crs,
                dst_transform=out_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear
            )

        # 11. Set ground values > 0 to NA for the "_bathy" raster files
        if '_bathy' in os.path.basename(raster_path):
            with rasterio.open(output_path, 'r+') as bathy_ras:
                data = bathy_ras.read(1)
                data[data > 0] = np.nan  # set positive values to NaN
                bathy_ras.write(data, 1)

    print(f'Processed: {os.path.basename(raster_path)} → {output_path}')

## 2. Standardize all rasters (PREDICTION Extent first as its larger)----
#makes all same extent, for processing into spatial points dataframe and removes all land based elevation values > 0 as well

#### NOTE:------ It is critical that BOTH the prediction and training datasets must have the same values of X, Y and FID within each, ---###
# although they are different extents, the smaller training data must be a direct subset of the prediction data
# for variables they have in common, even if the datasets vary between the two final datasets, we will divide the pertiant 
#columns afterward.
prediction_files = [f for f in os.listdir(prediction_dir) if f.endswith('.tif')]

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
# make a dataframe from Training extent mask
# 1. Extract mask coordinates and match X and Y from mask with 1 and 2 coords f - Review lin 363
# 2. Create FID column based on unique XY columns
# 3. filter DF where mask == 1
# 4. write out the filtered dataframe
# make a dataframe from Prediction extent mask
# 1. Extract mask coordinates and match X and Y from mask with 1 and 2 coords f - Review lin 363
# 2. Create FID column based on unique XY columns
# 3. filter DF where mask == 1
# 4. write out the filtered dataframe


# ## 4. Helper functions
# 1. split_tile - subdivide a tile into smaller grids
# 2. prepare_subgrids - TODO need to review
# 3. process_rasters - TODO need to review


# ## 5. Run model processing
# 1. run prediction sub grid 
# 2. run training sub grid
# 3. call process_rasters() for training 
# 4. call process_rasters() for prediction