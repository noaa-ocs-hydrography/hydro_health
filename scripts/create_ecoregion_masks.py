import pathlib
import datetime
import json
from osgeo import gdal, ogr, osr
gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


def get_transformation():
    # transform geometry
    utm17_gdal = osr.SpatialReference()
    utm17_gdal.ImportFromEPSG(3747)
    wgs84_gdal = osr.SpatialReference()
    wgs84_gdal.ImportFromEPSG(4326)
    wgs84_to_utm17_transform = osr.CoordinateTransformation(wgs84_gdal, utm17_gdal)
    return wgs84_to_utm17_transform


def build_prediction_masks():
   

    # Project EcoRegions_50m to UTM
    gpkg = INPUTS / 'Master_Grids.gpkg'
    gpkg_ds = ogr.Open(gpkg)
    ecoregions_50m = gpkg_ds.GetLayerByName('EcoRegions_50m')
    # in_memory_driver = ogr.GetDriverByName('Memory')
    in_memory_driver = ogr.GetDriverByName('ESRI Shapefile')
    
    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(3747)
    output_ds = in_memory_driver.CreateDataSource(str(OUTPUTS / 'output_ecoregions.shp'))
    output_layer = output_ds.CreateLayer(f'ecoregions', geom_type=ogr.wkbPolygon, srs=output_srs)

    input_layer_definition = ecoregions_50m.GetLayerDefn()
    for i in range(input_layer_definition.GetFieldCount()):
        fieldDefn = input_layer_definition.GetFieldDefn(i)
        output_layer.CreateField(fieldDefn)

    output_layer_definition = output_layer.GetLayerDefn()
    ecoregions_50m.ResetReading()
    for feature in ecoregions_50m:
        geom = feature.GetGeometryRef()
        geom.Transform(get_transformation())
        output_feature = ogr.Feature(output_layer_definition)
        output_feature.SetGeometry(geom)

        for i in range(input_layer_definition.GetFieldCount()):
            output_feature.SetField(input_layer_definition.GetFieldDefn(i).GetNameRef(), feature.GetField(i))

        output_layer.CreateFeature(output_feature)
        output_feature = None
    
    # TODO see if I can build the raster right after the vector transformation
    # Rebuild the extent from the single polygon to cut down on size

    # Build output UTM raster
    ecoregions = ['ER_3']
    # xmin, xmax, ymin, ymax = output_layer.GetExtent()
    pixel_size = 8
    nodata = 0.0
    # x_res = int((xmax - xmin) / pixel_size)
    # y_res = int((ymax - ymin) / pixel_size)

    output_layer.ResetReading()
    in_memory = ogr.GetDriverByName('Memory')
    for feature in output_layer:
        feature_json = json.loads(feature.ExportToJson())
        ecoregion_id = feature_json['properties']['EcoRegion']
        
        if ecoregion_id in ecoregions:
            print(ecoregion_id)
            # Create in memory layer and single polygon
            in_memory_ds = in_memory.CreateDataSource(f'output_layer_{ecoregion_id}')
            in_memory_layer = in_memory_ds.CreateLayer(f'poly_{ecoregion_id}', srs=output_layer.GetSpatialRef(), geom_type=ogr.wkbPolygon)
            temp_feature = ogr.Feature(in_memory_layer.GetLayerDefn())

            geometry = feature.GetGeometryRef().ExportToWkt()
            polygon = ogr.CreateGeometryFromWkt(geometry)
            temp_feature.SetGeometry(polygon)
            in_memory_layer.CreateFeature(temp_feature)

            xmin, xmax, ymin, ymax = in_memory_layer.GetExtent()
            x_res = int((xmax - xmin) / pixel_size)
            y_res = int((ymax - ymin) / pixel_size)

            mask_path = OUTPUTS / f'ecoregions_50m_mask_{ecoregion_id}.tif'
            with gdal.GetDriverByName("GTiff").Create(
                str(mask_path),
                x_res,
                y_res,
                1,
                gdal.GDT_Byte,
                options=["COMPRESS=LZW", f"NUM_THREADS=ALL_CPUS"],
            ) as target_ds:
                target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(3747)
                target_ds.SetProjection(srs.ExportToWkt())
                band = target_ds.GetRasterBand(1)
                band.SetNoDataValue(nodata)

                # Rasterize
                gdal.RasterizeLayer(target_ds, [1], in_memory_layer, burn_values=[1])
            in_memory_layer = None
            in_memory_ds = None
            print('finished rasterize')
            break



def process():
    start = datetime.datetime.now()
    print('Starting')
    build_prediction_masks()
    print(f'Done: {datetime.datetime.now() - start}')

if __name__ == '__main__':
    process()
