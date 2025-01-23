# ## 1. is preprocess all lidar data
# Directory for tiff files is: N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/UTM17

# 1. get all tiff files in a specific folder
# 2. get file name
# 3. get the XML metadata file for the current tile
# 4. locate contributor band
# 5. read the XML file
# 6. use the contributor band to get rows from XML file
# 7. store "value" and "survey_data_end" from the XML rows in a dataframe
# 8. create "survey_year_end" column from "survey_date_end"
# 9. round the date to 2 digits
# 10. build dataframe to map raster values by the year column
# 11. reclassify contributor band value to the year it was surveyed
# 12. write out new file with processed survey information


# ## 2. standardize all rasters to have same X, Y, FID
# Directory is N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model

# 1. Load the prediction mask
# 2. get the CRS of the mask
# 3. convert mask to dataframe 
# 4. get X and Y values from mask dataframe
# 5. Load list of support rasters for processing- N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw
# 6. Create output "processed" folder
# 7. Loop through tiff files and make sure they have the same CRS - 3747? NAD83
# 8. Loop through tiff files and adjust extents to make prediction mask (maybe same loop as above?)
# 9. Resample all rasters to 5m resolution
# 10. Loop through tiff files to make a dataframe with make vertices/points? and current tiff value at points
# 11. Set ground values > 0 to NA for the "_bathy" raster files
# 12. write out the updated files to the processed folder
# 13. verify all rasters have the same extent
