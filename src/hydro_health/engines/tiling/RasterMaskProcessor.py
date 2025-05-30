import os
import sys
import json
import pathlib
import shutil
import multiprocessing as mp
import pandas as pd
import geopandas as gpd

from osgeo import ogr, osr, gdal
from concurrent.futures import ThreadPoolExecutor


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))
INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


class RasterMaskProcessor:
    def create_training_mask(self, ecoregion):
        """Create training mask for current ecoregion"""

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

    def create_prediction_mask(self, ecoregion: pathlib.Path) -> None:
        """Create prediction mask for current ecoregion"""

        # TODO try to share in_memory layer creation with multiprocessing
        # Project EcoRegions_50m to UTM
        gpkg = INPUTS / 'Master_Grids.gpkg'
        gpkg_ds = ogr.Open(gpkg)
        ecoregions_50m = gpkg_ds.GetLayerByName('EcoRegions_50m')
        in_memory_driver = ogr.GetDriverByName('Memory')
        # in_memory_driver = ogr.GetDriverByName('ESRI Shapefile')
        
        output_srs = osr.SpatialReference()
        output_srs.ImportFromEPSG(3747)
        output_ds = in_memory_driver.CreateDataSource('output_ecoregion')
        output_layer = output_ds.CreateLayer(f'ecoregions', geom_type=ogr.wkbPolygon, srs=output_srs)

        input_layer_definition = ecoregions_50m.GetLayerDefn()
        for i in range(input_layer_definition.GetFieldCount()):
            fieldDefn = input_layer_definition.GetFieldDefn(i)
            output_layer.CreateField(fieldDefn)

        output_layer_definition = output_layer.GetLayerDefn()
        ecoregions_50m.ResetReading()
        for feature in ecoregions_50m:
            # TODO how to get field
            ecoregion_id = json.loads(feature.ExportToJson())['properties']['EcoRegion']
            if ecoregion_id == ecoregion.stem:
                geom = feature.GetGeometryRef()
                geom.Transform(self.get_transformation())
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

            mask_path = ecoregion / f'prediction_mask_{ecoregion_id}.tif'
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
        gpkg_ds = None
        output_ds = None

    def delete_intermediate_files(self, outputs) -> None:
        """Delete any intermediate shapefiles"""

        merged_shapefiles = pathlib.Path(outputs).rglob('training_data*')
        for file in merged_shapefiles:
            file.unlink()

    def dissolve_tile_index_shapefiles(self, approved_files, ecoregions) -> None:
        """Create dissolved single polygons to use as burn rasters"""

        for ecoregion in ecoregions:
            if ecoregion.stem in approved_files:
                for tile_index_path in approved_files[ecoregion.stem]:
                    tile_index = gpd.read_file(tile_index_path).dissolve()
                    dissolved_tile_index = tile_index_path.parents[0] / pathlib.Path(tile_index_path.stem + '_dis.shp')
                    tile_index.to_file(dissolved_tile_index)

    def get_approved_area_files(self, ecoregions: list[pathlib.Path]):
        """Get list of intersected features from ecoregion tile index shapefile"""

        approved_files = {}
        for ecoregion in ecoregions:
            digital_coast = ecoregion / 'DigitalCoast'
            if digital_coast.is_dir():
                json_files = digital_coast.rglob('feature.json')
                for file in json_files:
                    if ecoregion.stem not in approved_files:
                        approved_files[ecoregion.stem] = []
                    # Disregard small area tile index
                    parent_project = file.parents[0]
                    if json.load(open(file))['Shape_Area'] > 5000000:
                        tile_index_shps = list(parent_project.rglob('tileindex*.shp'))
                        if tile_index_shps:
                            approved_files[ecoregion.stem].append(tile_index_shps[0])
                    else:
                        # delete digital coast folder if small area
                        print(f' - Deleting small area: {parent_project.stem}')
                        if parent_project.exists():
                            # TODO will all ecoregions have 'mosaic_NCMP_6326' ? 
                            vrt_file = parent_project.parents[0] / f'mosaic_NCMP_6326_{parent_project.stem}.vrt'
                            if vrt_file.exists():
                                vrt_file.unlink()
                            shutil.rmtree(parent_project)
        return approved_files

    def get_transformation(self) -> osr.CoordinateTransformation:
        """Transformation object for WGS84 to UTM17"""

        utm17_gdal = osr.SpatialReference()
        utm17_gdal.ImportFromEPSG(3747)
        wgs84_gdal = osr.SpatialReference()
        wgs84_gdal.ImportFromEPSG(4326)
        wgs84_to_utm17_transform = osr.CoordinateTransformation(wgs84_gdal, utm17_gdal)
        return wgs84_to_utm17_transform

    def merge_dissolved_polygons(self, ecoregions):
        """Creating full training outline shapefile for ecoregion"""

        for ecoregion in ecoregions:
            digital_coast = ecoregion / 'DigitalCoast'
            if digital_coast.is_dir():
                dissolved_shapefiles = digital_coast.rglob('*_dis.shp')
                merged_training_ds = pd.concat([gpd.read_file(shp) for shp in dissolved_shapefiles]).dissolve()
                training_datasets = ecoregion / 'training_data_outlines.shp'
                merged_training_ds.to_file(training_datasets)

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)

    def process(self, outputs: str) -> None:
        ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
        print('Creating prediction masks')
        self.process_prediction_masks(ecoregions, outputs)
        approved_files = self.get_approved_area_files(ecoregions)
        print(f' - Found files for ecoregions: {approved_files.keys()}')
        print('Dissolving tile index shapefiles')
        self.dissolve_tile_index_shapefiles(approved_files, ecoregions)
        print('Merging all dissolved tile index shapfiles')
        self.merge_dissolved_polygons(ecoregions)
        print('Creating training masks')
        self.process_training_masks(ecoregions, outputs)
        self.delete_intermediate_files(outputs)

    def process_prediction_masks(self, ecoregions: list[pathlib.Path], outputs: str) -> None:
        """Multiprocessing entrypoint for creating prediction masks"""

        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as pool:
            self.print_async_results(pool.map(self.create_prediction_mask, ecoregions), outputs)

    def process_training_masks(self, ecoregions: list[pathlib.Path], outputs: str) -> None:
        """Multiprocessing entrypoint for creating training masks"""

        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as pool:
            self.print_async_results(pool.map(self.create_training_mask, ecoregions), outputs)
