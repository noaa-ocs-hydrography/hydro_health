import pathlib
import datetime
import json
import pandas as pd
import geopandas as gpd
import shutil

from osgeo import gdal, ogr, osr
osr.DontUseExceptions()

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
    in_memory_driver = ogr.GetDriverByName('Memory')
    # in_memory_driver = ogr.GetDriverByName('ESRI Shapefile')
    
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


    # Build output UTM raster
    pixel_size = 8
    nodata = 0
    output_layer.ResetReading()
    in_memory = ogr.GetDriverByName('Memory')
    for feature in output_layer:
        feature_json = json.loads(feature.ExportToJson())
        ecoregion_id = feature_json['properties']['EcoRegion']
        print('Building prediction mask:', ecoregion_id)
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

        ecoregion_folder = OUTPUTS / ecoregion_id
        ecoregion_folder.mkdir(parents=True, exist_ok=True)
        mask_path = ecoregion_folder / f'prediction_mask_{ecoregion_id}.tif'
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
            gdal.RasterizeLayer(target_ds, [1], in_memory_layer, burn_values=[1])

        in_memory_layer = None
        in_memory_ds = None
        print('finished rasterize')


def build_training_masks():
    print('Building training masks')
    ecoregion_folders = [folder for folder in OUTPUTS.glob('ER_*') if folder.is_dir()]
    for ecoregion in ecoregion_folders:
        prediction_file = ecoregion / f'prediction_mask_{ecoregion.stem}.tif'
        training_data_outline = ecoregion / 'training_data_outlines.shp'
        if prediction_file.exists() and training_data_outline.exists():
            training_file = ecoregion / f'training_mask_{ecoregion.stem}.tif'
            shutil.copy(prediction_file, training_file)
            training_ds = gdal.Open(training_file, gdal.GA_Update)
            shp_driver = ogr.GetDriverByName("ESRI Shapefile")
            training_data_outline_ds = shp_driver.Open(training_data_outline)
            for layer in training_data_outline_ds:
                # TODO try to load the layer without looping in case more than 1 layer
                # TODO verify if tile_index is used as full dataset or subset and we need to save that dataset
                # TODO try to silence errors? or fix them?
                gdal.PushErrorHandler('CPLQuietErrorHandler')
                gdal.RasterizeLayer(training_ds, [1], layer, burn_values=[2])
                break
            training_ds = None
            training_data_outline_ds = None


def get_approved_area_files():
    print('Getting approved files')
    ecoregion_folders = [folder for folder in OUTPUTS.glob('ER_*') if folder.is_dir()]
    for ecoregion in ecoregion_folders:
        digital_coast = ecoregion / 'DigitalCoast'
        # digital_coast = pathlib.Path(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast')
        if digital_coast.is_dir():
            json_files = digital_coast.rglob('feature.json')
            approved_files = {}
            for file in json_files:
                ecoregion = file.parents[2].stem
                if ecoregion not in approved_files:
                    approved_files[ecoregion] = []
                if json.load(open(file))['Shape_Area'] > 5000000:
                    parent_project = file.parents[0]
                    tile_index_shps = list(parent_project.rglob('tileindex*.shp'))
                    if tile_index_shps:
                        approved_files[ecoregion].append(tile_index_shps[0])
    return approved_files


def dissolve_tile_index_shapefiles(approved_files):
    """Create dissolved single polygons to use as burn rasters"""

    print('Dissolving polygons')
    ecoregion_folders = [folder.stem for folder in OUTPUTS.glob('ER_*') if folder.is_dir()]
    for ecoregion in ecoregion_folders:
        if ecoregion in approved_files:
            for tile_index_path in approved_files[ecoregion]:
                print(f'Dissolving {tile_index_path}')
                tile_index = gpd.read_file(tile_index_path).dissolve()
                dissolved_tile_index = tile_index_path.parents[0] / pathlib.Path(tile_index_path.stem + '_dis.shp')
                tile_index.to_file(dissolved_tile_index)


def merge_dissolved_polygons():
    ecoregion_folders = [folder for folder in OUTPUTS.glob('ER_*') if folder.is_dir()]
    for ecoregion in ecoregion_folders:
        digital_coast = ecoregion / 'DigitalCoast'
        if digital_coast.is_dir():
            dissolved_shapefiles = digital_coast.rglob('*_dis.shp')
            print('Merging dissolved datasets')
            merged_training_ds = pd.concat([gpd.read_file(shp) for shp in dissolved_shapefiles]).dissolve()
            training_datasets = ecoregion / 'training_data_outlines.shp'
            print('Saving merged training outlines')
            merged_training_ds.to_file(training_datasets)


def process():
    start = datetime.datetime.now()
    print('Starting')
    build_prediction_masks()
    approved_files = get_approved_area_files()
    dissolve_tile_index_shapefiles(approved_files)
    merge_dissolved_polygons()
    build_training_masks()
    print(f'Done: {datetime.datetime.now() - start}')

if __name__ == '__main__':
    process()
