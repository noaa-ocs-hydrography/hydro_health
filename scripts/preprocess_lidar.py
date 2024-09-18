import arcpy
import pathlib


base_dir = pathlib.Path(r'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/Florida_all_survey_years_raw')


# get list of all rasters with rglob
for raster in base_dir.rglob('*.tif'):

# convert feet to meters by * 0.3048006096
    print(raster)

# reproject to NAD83 UTM Zone 17N if raster in survey feet, 26917?

# resample to 5x5m resolution

# convert positive values to NaN

# check if units and res changed

# compress file using LZ77

