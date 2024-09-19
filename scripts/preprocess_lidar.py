
import pathlib
import numpy as np
import shutil
import os

from osgeo import gdal, osr


BASE_DIR = pathlib.Path(r'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/Florida_all_survey_years_raw')

OUTPUT = pathlib.Path(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\sediment\dev')


# get list of all rasters with rglob
for i, raster in enumerate(BASE_DIR.rglob('*.tif')):

    # 1. Copy out file
    no_data = -999999
    print(i, raster)
    copy_raster = str(OUTPUT / raster.name)
    shutil.copyfile(str(raster), copy_raster)

    # 2. Load dataset
    raster_ds = gdal.Open(copy_raster, gdal.GA_Update)
    projection = osr.SpatialReference(wkt=raster_ds.GetProjection())
    projection.AutoIdentifyEPSG()
    epsg = projection.GetAttrValue('AUTHORITY', 1)
    units = projection.GetAttrValue('UNIT')
    
    # conversion
    if epsg != '3747':
        # 2. reproject
        output_nad83_raster = str(OUTPUT / f'{raster.stem}_nad83.tif')
        raster_warp = gdal.Warp(output_nad83_raster, raster_ds, dstSRS='EPSG:3747')

        # reset raster_ds
        raster_ds = None
        raster_ds = raster_warp
    if units != 'metre':
        # 3. change units if project didn't do that already?!
        raster_array = raster_ds.ReadAsArray()
        meters_array = np.where(raster_array == no_data, raster_array, raster_array * 0.3048006096)

        raster_ds.GetRasterBand(1).WriteArray(meters_array)
    
    _, cell_x, _, _, _, cell_y = raster_ds.GetGeoTransform()
    if cell_x != 5.0 and cell_y != 5.0:
        # 4. Resample to 5x5
        output_nad83_raster = str(OUTPUT / f'{raster.stem}_nad83.tif')
        raster_ds = gdal.Warp(output_nad83_raster, raster_ds, 
                       format="GTiff",
                       xRes=5.0, yRes=5.0,
                       creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"])
        _, cell_x, _, _, _, cell_y = raster_ds.GetGeoTransform()
        print(cell_x, -cell_y)
  
    # 5. convert positive values to NaN
    raster_array = raster_ds.ReadAsArray()
    meters_array = np.where(raster_array < 0, raster_array, no_data)
    raster_ds.GetRasterBand(1).WriteArray(meters_array)

    # check if units and res changed
    raster_ds = None

    if os.path.exists(str(OUTPUT / f'{raster.stem}_nad83.tif')):
        os.remove(str(OUTPUT / raster.name))
    if i == 3:
        break

# 6. Build a VRT of all final raster files