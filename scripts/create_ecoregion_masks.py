import pathlib
import datetime
import json
from osgeo import gdal, ogr, osr
gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


def build_prediction_masks():
    ecoregions = ['ER_3']
    
    gpkg = INPUTS / 'Master_Grids.gpkg'
    gpkg_ds = ogr.Open(gpkg)
    ecoregions_50m = gpkg_ds.GetLayerByName('EcoRegions_50m')
    xmin, xmax, ymin, ymax = ecoregions_50m.GetExtent()
    pixel_size = .00025
    nodata = 0.0
    x_res = int((xmax - xmin) / pixel_size)
    y_res = int((ymax - ymin) / pixel_size)

    ecoregions_50m.ResetReading()
    in_memory = ogr.GetDriverByName('Memory')
    for feature in ecoregions_50m:
        feature_json = json.loads(feature.ExportToJson())
        ecoregion_id = feature_json['properties']['EcoRegion']
        if ecoregion_id in ecoregions:
            in_memory_ds = in_memory.CreateDataSource(str(OUTPUTS / f'output_layer_{ecoregion_id}.shp'))
            in_memory_layer = in_memory_ds.CreateLayer(f'poly_{ecoregion_id}', srs=ecoregions_50m.GetSpatialRef(), geom_type=ogr.wkbPolygon)
            mem_poly = ogr.Feature(in_memory_layer.GetLayerDefn())
            mem_poly.SetGeometry(feature.GetGeometryRef())
            in_memory_layer.CreateFeature(mem_poly)

            mask_path = OUTPUTS / f'ecoregions_50m_mask_{ecoregion_id}.tif'
            with gdal.GetDriverByName("GTiff").Create(
                str(mask_path),
                x_res,
                y_res,
                1,
                gdal.GDT_Float32,
                options=["COMPRESS=LZW"],
            ) as target_ds:
                target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                target_ds.SetProjection(srs.ExportToWkt())
                band = target_ds.GetRasterBand(1)
                band.SetNoDataValue(nodata)

                # Rasterize
                gdal.RasterizeLayer(target_ds, [1], in_memory_layer, burn_values=[1])
            

            in_memory_layer = None
            break


def process():
    start = datetime.datetime.now()
    print('Starting')
    build_prediction_masks()
    print(f'Done: {datetime.datetime.now() - start}')

if __name__ == '__main__':
    process()
