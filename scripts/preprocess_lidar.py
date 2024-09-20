import pathlib
import numpy as np
import shutil
import os

from osgeo import gdal, osr


BASE_DIR = pathlib.Path(r'N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/Florida_all_survey_years_raw')
OUTPUT = pathlib.Path(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\sediment\dev')


def create_raster_vrt():
    geotiffs = list(OUTPUT.rglob('*_5_5.tif'))
    vrt_filename = str(OUTPUT / 'JABLTCX_mosaic.vrt')
    gdal.BuildVRT(vrt_filename, geotiffs, callback=gdal.TermProgress_nocb)
    final_image = gdal.Open(vrt_filename, gdal.GA_ReadOnly)
    gdal.Translate(str(OUTPUT / 'JABLTCX_final.tif'), final_image, format='GTiff',
               creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
               callback=gdal.TermProgress_nocb)
    final_image = None


def convert_band_ft_to_m(dataset):
    no_data = -999999
    raster_array = dataset.ReadAsArray()
    meters_array = np.where(raster_array == no_data, raster_array, raster_array * 0.3048006096)
    dataset.GetRasterBand(1).WriteArray(meters_array)


def get_epsq(dataset):
    projection = osr.SpatialReference(wkt=dataset.GetProjection())
    projection.AutoIdentifyEPSG()
    return projection.GetAttrValue('AUTHORITY', 1)


def get_units(dataset):
    projection = osr.SpatialReference(wkt=dataset.GetProjection())
    return projection.GetAttrValue('UNIT')


def get_raster_cell_size(dataset):
    _, cell_x, _, _, _, cell_y = dataset.GetGeoTransform()
    return cell_x, cell_y * -1  # inverse Y value


def reproject_raster(dataset, raster_path):
    output_nad83_raster = str(OUTPUT / f"{raster_path.stem}_nad83.tif")
    gdal.Warp(
        output_nad83_raster, 
        dataset,
        format="GTiff",
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"],
        dstSRS="EPSG:3747")

    # reset raster_ds
    dataset = None
    raster_ds = gdal.Open(output_nad83_raster, gdal.GA_Update)
    return raster_ds


def resize_raster_grid(dataset, raster_path):
    output_nad83_raster = str(OUTPUT / f'{raster_path.stem}_nad83_5_5.tif')
    gdal.Warp(
        output_nad83_raster,
        dataset, # TODO are brackets needed? they got rid of error messages, but not Z aware now?
        format="GTiff",
        xRes=5.0,
        yRes=5.0,
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"],
    )

    # reset raster_ds
    dataset = None
    raster_ds = gdal.Open(output_nad83_raster, gdal.GA_Update)
    return raster_ds


def set_ground_to_nodata(dataset):
    no_data = -999999
    raster_array = dataset.ReadAsArray()
    meters_array = np.where(raster_array < 0, raster_array, no_data)
    dataset.GetRasterBand(1).WriteArray(meters_array)


def process():
    # get list of all rasters with rglob
    for i, raster_path in enumerate(BASE_DIR.rglob('*.tif')):
        # Copy out file
        print(i, raster_path)
        copy_raster = str(OUTPUT / raster_path.name)
        shutil.copyfile(str(raster_path), copy_raster)

        # Load copied dataset
        raster_ds = gdal.Open(copy_raster, gdal.GA_Update)
        units = get_units(raster_ds)
        epsg = get_epsq(raster_ds)

        # This was failing when running it last
        set_ground_to_nodata(raster_ds)
        
        if epsg != '3747':  # should we use Albers or some other projection?
            raster_ds = reproject_raster(raster_ds, raster_path)

        if units != 'metre':
            convert_band_ft_to_m(raster_ds)
        print('Output units:', get_units(raster_ds))
        
        cell_x, cell_y = get_raster_cell_size(raster_ds)
        if cell_x != 5.0 and cell_y != 5.0:
            raster_ds = resize_raster_grid(raster_ds, raster_path)
            cell_x, cell_y = get_raster_cell_size(raster_ds)
            print('Output cell size:', cell_x, -cell_y)

        raster_ds = None

        # Remove intermediate files
        os.remove(str(OUTPUT / raster_path.name))
        if os.path.exists(str(OUTPUT / f'{raster_path.stem}_nad83.tif')):
            os.remove(str(OUTPUT / f'{raster_path.stem}_nad83.tif'))


        if i == 4:
            break

    # 6. Build a VRT of all final raster files
    create_raster_vrt()


if __name__ == "__main__":
    process()
