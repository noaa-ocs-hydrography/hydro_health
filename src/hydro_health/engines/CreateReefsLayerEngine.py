import os
import pathlib
import subprocess
# import geopandas as gpd

from osgeo import ogr, osr, gdal
from Engine import Engine
from hydro_health.helpers.tools import get_config_item


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateReefsLayerException(Exception):
    """Custom exception for tool"""

    pass


class CreateReefsLayerEngine(Engine):
    """Class to hold the logic for processing the Reefs layer"""

    # def __init__(self, param_lookup: dict=None) -> None:
    #     # TODO copy over functions that process Reefs
    #     pass

    # def create_gpd_multiple_buffer(self, reef_polygons: str) -> None:
    #     """Create multiple buffers around reef polygons using Geopandas"""

    #     buffer_distance = get_config_item('REEF', 'BUFFER_DISTANCE')
    #     tanker_buffer_distance = get_config_item(
    #         'REEF', 'TANKER_BUFFER_DISTANCE')

    #     reef_polygons_buffer = str(OUTPUTS / 'reef_polygons_buffer.shp')
    #     reef_polygons_tanker_buffer = str(
    #         OUTPUTS / 'reef_polygons_tanker_buffer.shp')

    #     # def multiring_buffer(df, distances):
    #     #     buffers = []
    #     #     for distance in distances:
    #     #         reef_buffer = reef_gdf.geometry.buffer(distance)
    #     #         buffers.append(reef_buffer.difference(reef_gdf['geometry']))
    #     #     return buffers

    #     buffers = {reef_polygons_buffer: buffer_distance,
    #                reef_polygons_tanker_buffer: tanker_buffer_distance}
    #     for buffer_file, distances in buffers.items():
    #         reef_gdf = gpd.read_file(reef_polygons)
    #         reef_polygons_crs = reef_gdf.crs
    #         print(f'Attempting: {buffer_file}')
    #         if os.path.exists(reef_polygons_buffer) and os.path.exists(reef_polygons_tanker_buffer):
    #             print('Buffer files already exist')
    #             return [reef_polygons_buffer, reef_polygons_tanker_buffer]

    #         # reef_gdf['buffers'] = reef_gdf.apply(lambda row: multiring_buffer(row, distances))
    #         for i, row in reef_gdf.iterrows():
    #             print(i)
    #             buffers = []
    #             for distance in distances:
    #                 reef_buffer = row.geometry.buffer(distance)
    #                 buffers.append(reef_buffer.difference(row['geometry']))
    #             row['buffers'] = buffers
    #         reef_gdf = reef_gdf.explode('buffers')
    #         reef_gdf = reef_gdf.set_geometry('buffers')
    #         reef_gdf = reef_gdf.drop(columns=['geometry']).rename_geometry(
    #             'geometry').set_crs(reef_polygons_crs)
    #         reef_gdf = gpd.GeoDataFrame(
    #             geometry=[reef_gdf.unary_union]).explode(
    #                 index_parts=False).reset_index(
    #                 drop=True)
    #         reef_gdf.to_file(buffer_file)
    #     return buffers.keys()

    # def create_gdal_multiple_buffer(self, reef_polygons: str) -> None:
    #     """Create multiple buffers around reef polygons using GDAL"""

    #     buffer_distance = get_config_item('REEF', 'REVERSE_BUFFER_DISTANCE')
    #     tanker_buffer_distance = get_config_item('REEF', 'REVERSE_TANKER_BUFFER_DISTANCE')

    #     driver = ogr.GetDriverByName('ESRI Shapefile')
    #     reef_data = ogr.Open(reef_polygons)
    #     reef_layer = reef_data.GetLayer()

    #     reef_polygons_buffer = str(OUTPUTS / 'reef_polygons_buffer.shp')
    #     reef_polygons_tanker_buffer = str(OUTPUTS / 'reef_polygons_tanker_buffer.shp')

    #     buffers = {reef_polygons_buffer: buffer_distance,
    #                reef_polygons_tanker_buffer: tanker_buffer_distance}
    #     for buffer_file, distances in buffers.items():
    #         print(f'Attempting: {buffer_file}')
    #         if os.path.exists(reef_polygons_buffer) and os.path.exists(reef_polygons_tanker_buffer):
    #             print('Buffer files already exist')
    #             return [reef_polygons_buffer, reef_polygons_tanker_buffer]
    #         buffer_data = driver.CreateDataSource(buffer_file)
    #         buffer_layer = buffer_data.CreateLayer(
    #             'buffer_file', geom_type=ogr.wkbPolygon)
    #         distance_field = ogr.FieldDefn('distance', ogr.OFTReal)
    #         buffer_layer.CreateField(distance_field)
    #         buffer_lyr_definition = buffer_layer.GetLayerDefn()
    #         for distance in distances:
    #             print(distance)
    #             # -0.15
    #             # -0.25
    #             # 0.15
    #             # 0.25
    #             previous_outter_buffer = None
    #             previous_inner_buffer = None
    #             for feature in reef_layer:
    #                 ingeom = feature.GetGeometryRef()
    #                 geomBuffer = ingeom.Buffer(distance, 30)
    #                 geom_difference = geomBuffer.Difference(ingeom)
    #                 first_inner = True
    #                 first_outter = True
    #                 if distance < 0: # OG is big, buffer is small, 
    #                     # first time - difference, and buffer
    #                     # second time - need difference of previous buffer and new buffer
    #                     if first_inner: # first
    #                         first_inner = False
    #                         previous_inner_buffer = geomBuffer
    #                         # difference
    #                         buffered_feature = ogr.Feature(buffer_lyr_definition)
    #                         buffered_feature.SetField('distance', distance)
    #                         buffered_feature.SetGeometry(geom_difference)
    #                         buffer_layer.CreateFeature(buffered_feature)
    #                         # buffer
    #                         buffered_feature = ogr.Feature(buffer_lyr_definition)
    #                         buffered_feature.SetField('distance', distance)
    #                         buffered_feature.SetGeometry(geomBuffer)
    #                         buffer_layer.CreateFeature(buffered_feature)
    #                     else: # second
    #                         # previous difference
    #                         previous_geom_difference = previous_inner_buffer.Difference(geomBuffer)
    #                         buffered_feature = ogr.Feature(buffer_lyr_definition)
    #                         buffered_feature.SetField('distance', distance)
    #                         buffered_feature.SetGeometry(previous_geom_difference)
    #                         buffer_layer.CreateFeature(buffered_feature)
    #                 else: # buffer is big, OG is small
    #                     # difference
    #                     # multiple bigger, needs difference of previous buffer and new buffer
    #                     if first_outter: # first
    #                         first_outter = False
    #                         previous_outter_buffer = geomBuffer
    #                         # difference
    #                         buffered_feature = ogr.Feature(buffer_lyr_definition)
    #                         buffered_feature.SetField('distance', distance)
    #                         buffered_feature.SetGeometry(geom_difference)
    #                         buffer_layer.CreateFeature(buffered_feature)
    #                     else: # second
    #                         # previous difference
    #                         previous_geom_difference = previous_outter_buffer.Difference(geomBuffer)
    #                         buffered_feature = ogr.Feature(buffer_lyr_definition)
    #                         buffered_feature.SetField('distance', distance)
    #                         buffered_feature.SetGeometry(previous_geom_difference)
    #                         buffer_layer.CreateFeature(buffered_feature)
    #                 original_feature = None
    #                 outFeature = None
    #             # TODO try to clip las geometry from each distance geomBuffer
    #             # might need to write out separate shapefiles and use the final clipped one
    #         buffer_data = None
    #     reef_data = None

    #     return buffers.keys()

    def create_gdal_multiple_buffer(self, reef_polygons: str) -> None:
        """Create multiple buffers around reef polygons using GDAL"""

        buffer_distance = list(reversed(get_config_item('REEF', 'BUFFER_DISTANCE')))
        tanker_buffer_distance = list(reversed(get_config_item('REEF', 'TANKER_BUFFER_DISTANCE')))
        print(buffer_distance, tanker_buffer_distance)
        driver = ogr.GetDriverByName('ESRI Shapefile')
        reef_data = ogr.Open(reef_polygons)
        reef_layer = reef_data.GetLayer()

        reef_polygons_buffer = OUTPUTS / 'reef_polygons_buffer.shp'
        reef_polygons_tanker_buffer = OUTPUTS / 'reef_polygons_tanker_buffer.shp'

        buffers = {reef_polygons_buffer: buffer_distance,
                   reef_polygons_tanker_buffer: tanker_buffer_distance}
        for buffer_path, distances in buffers.items():
            buffer_file = str(buffer_path)
            if os.path.exists(reef_polygons_buffer) and os.path.exists(reef_polygons_tanker_buffer):
                print('Buffer files already exist')
                return [reef_polygons_buffer, reef_polygons_tanker_buffer]
            
            print(f'Attempting: {buffer_file}')
            buffer_data = driver.CreateDataSource(buffer_file)
            buffer_layer = buffer_data.CreateLayer(
                "reef_polygons", geom_type=ogr.wkbPolygon)
            distance_field = ogr.FieldDefn('distance', ogr.OFTReal)
            buffer_layer.CreateField(distance_field)
            buffer_lyr_definition = buffer_layer.GetLayerDefn()
            for i, distance in enumerate(distances):
                for feature in reef_layer:
                    ingeom = feature.GetGeometryRef()
                    geomBuffer = ingeom.Buffer(distance, 10)
                    buffered_feature = ogr.Feature(buffer_lyr_definition)
                    if i == len(distances) - 1: # original and difference
                        original_feature = ogr.Feature(buffer_lyr_definition)
                        original_feature.SetField('distance', 0)
                        original_feature.SetGeometry(ingeom)
                        buffer_layer.CreateFeature(original_feature)
                        original_feature = None

                        buffered_feature.SetField('distance', distance)
                        geom_difference = geomBuffer.Difference(ingeom)
                        buffered_feature.SetGeometry(geom_difference)
                        buffer_layer.CreateFeature(buffered_feature)
                    else: # previous distance difference only
                        buffered_feature.SetField('distance', distance)
                        previous_distance_buffer = ingeom.Buffer(distances[i+1], 10)  # plus to run reversed buffers
                        geom_difference = geomBuffer.Difference(previous_distance_buffer)
                        buffered_feature.SetGeometry(geom_difference)
                        buffer_layer.CreateFeature(buffered_feature)
                    buffered_feature = None
            buffer_data = None
            self.make_esri_projection(buffer_path.stem, 5070)
        reef_data = None

        return buffers.keys()

    def create_reef_rasters(self, reef_buffer_paths: str) -> None:
        """Create GeoTiff raster files from each shapefile"""

        raster_files = []
        for shp_path in reef_buffer_paths:
            raster_name = pathlib.Path(shp_path).stem
            reef_raster = str(OUTPUTS / f'{raster_name}.tif')
            if os.path.exists(reef_raster):
                print(f'Raster file already exists: {reef_raster}')
                raster_files.append(reef_raster)
                continue
            input_shp = ogr.Open(shp_path)
            shp_layer = input_shp.GetLayer()
            pixel_size = 0.01
            xmin, xmax, ymin, ymax = shp_layer.GetExtent()
            print(f'Rasterizing: {reef_raster}')
            no_data = -999
            raster = gdal.Rasterize(reef_raster, shp_path,
                                    noData=no_data,
                                    format='GTiff',
                                    outputType=gdal.GDT_Float32,
                                    outputBounds=[xmin, ymin, xmax, ymax],
                                    attribute='distance',
                                    xRes=pixel_size,
                                    yRes=pixel_size)
            raster = None
        return raster_files

    def clip_reef_shapefile(self, projected_reef_shp):
        driver = ogr.GetDriverByName('ESRI Shapefile')

        # valid_projected_reef = str(OUTPUTS / 'valid_projected_reef.shp')
        # subprocess.run([
        #     'ogr2ogr',
        #     f'{valid_projected_reef}',
        #     f'{projected_reef_shp}',
        #     '-makevalid'
        # ])

        projected_reef_data = driver.Open(projected_reef_shp, 0)
        projected_reef_layer = projected_reef_data.GetLayer()

        reef_extent_data = driver.Open(
            str(INPUTS / 'north_america_clip_wgs84.shp'), 0)  # using Albers for clip does not work with GDAL?!
        reef_extent_layer = reef_extent_data.GetLayer()

        clipped_reef_shp = str(OUTPUTS / 'reef_polygons_clip.shp')
        output_reef_clip_data = driver.CreateDataSource(clipped_reef_shp)
        output_reef_clip_layer = output_reef_clip_data.CreateLayer(
            "reef_polygons", geom_type=ogr.wkbMultiPolygon)

        ogr.Layer.Clip(projected_reef_layer, reef_extent_layer,
                       output_reef_clip_layer)

        self.make_esri_projection('reef_polygons_clip')

        projected_reef_data = None
        reef_extent_data = None
        output_reef_clip_data = None

        return clipped_reef_shp
    
    def dissolve_overlapping_buffers(self, buffer_paths):
        """Dissolve overlapping buffers"""

        dissolved_buffers = [self.dissolve_overlapping_polygons(buffer_shp, with_distance=True) for buffer_shp in buffer_paths]

        return dissolved_buffers
    
    def dissolve_overlapping_polygons(self, input_shapefile, with_distance=False):
        """Dissolve simplified polygons"""

        shp_path = pathlib.Path(input_shapefile)
        dissolved_reef_polygons = str(OUTPUTS / f'dis_{shp_path.stem}.shp')
        if os.path.exists(dissolved_reef_polygons):
            print('Dissolved Reef shapefile already exists')
            return dissolved_reef_polygons
        print('Dissolving simplified reef polygons')
        if with_distance:
            sql = f"SELECT ST_Union(geometry), distance FROM {shp_path.stem} GROUP BY distance"
        else:
            sql = f"SELECT ST_Union(geometry) FROM {shp_path.stem}"
        subprocess.run([
            'ogr2ogr',
            '-explodecollections',
            '-f',
            "ESRI Shapefile",
            f'{dissolved_reef_polygons}', 
            f'{input_shapefile}', 
            '-dialect', 
            'sqlite', 
            '-sql', 
            sql
        ])
        self.make_esri_projection(f'dis_{shp_path.stem}', 5070)

        return dissolved_reef_polygons
    
    def make_esri_projection(self, file_name, epsg=4326):
        """Create an Esri .prj file for a shapefile"""

        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(epsg)
        output_projection.MorphToESRI()
        file = open(OUTPUTS / f'{file_name}.prj', 'w')
        file.write(output_projection.ExportToWkt())
        file.close()
    
    def project_reef_shapefile(self, shp_path: str) -> str:
        """Reproject a shapefile using GDAL"""

        driver = ogr.GetDriverByName('ESRI Shapefile')
        input_projection = osr.SpatialReference()
        input_projection.ImportFromEPSG(4326)
        input_projection.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) # 4326 requires lat/lon, but all coords are read as lon/lat.  This sets order.
        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(5070)
        coord_transformation = osr.CoordinateTransformation(input_projection, output_projection)
        reef_data = driver.Open(shp_path)
        reef_layer = reef_data.GetLayer()
        projected_reef_shp = str(OUTPUTS / 'projected_reef_polygons.shp')
        if os.path.exists(projected_reef_shp):
            print('Projected Reef shapefile already exists')
            return projected_reef_shp
            # driver.DeleteDataSource(projected_reef_shp)

        output_reef_data = driver.CreateDataSource(projected_reef_shp)
        output_reef_layer = output_reef_data.CreateLayer(
            "reef_polygons", 
            geom_type=ogr.wkbMultiPolygon)

        # Add fields
        reef_lyr_definition = reef_layer.GetLayerDefn()
        for i in range(0, reef_lyr_definition.GetFieldCount()):
            fieldDefn = reef_lyr_definition.GetFieldDefn(i)
            output_reef_layer.CreateField(fieldDefn)

        # Transform geometry
        output_reef_lyr_definition = output_reef_layer.GetLayerDefn()

        for feature in reef_layer:
            geom = feature.GetGeometryRef()
            geom.Transform(coord_transformation)
            outFeature = ogr.Feature(output_reef_lyr_definition)
            outFeature.SetGeometry(geom)
            for i in range(0, output_reef_lyr_definition.GetFieldCount()):
                outFeature.SetField(
                    output_reef_lyr_definition.GetFieldDefn(i).GetNameRef(),
                    feature.GetField(i)
                )
            output_reef_layer.CreateFeature(outFeature)
            outFeature = None

        # Write .prj files
        self.make_esri_projection('projected_reef_polygons', 5070)

        # Close files
        reef_data = None
        output_reef_data = None

        return projected_reef_shp

    def simplify_reef_shapefile(self, cilpped_reef_shp):
        """Simplify the 1km reef polygons with a small buffer"""

        simplified_reef_polygons = str(OUTPUTS / 'simp_reef_poly.shp')
        if os.path.exists(simplified_reef_polygons):
            print('Simplified Reef shapefile already exists')
            return simplified_reef_polygons
        print('Simplifying 1km reef polygons')
        driver = ogr.GetDriverByName('ESRI Shapefile')
        reef_data = ogr.Open(cilpped_reef_shp)
        reef_layer = reef_data.GetLayer()
        buffer_data = driver.CreateDataSource(simplified_reef_polygons)
        buffer_layer = buffer_data.CreateLayer(
            "reef_polygons", geom_type=ogr.wkbMultiPolygon)
        buffer_lyr_definition = buffer_layer.GetLayerDefn()

        for feature in reef_layer:
            ingeom = feature.GetGeometryRef()
            geomBuffer = ingeom.Buffer(100, 30)
            simplified_feature = ogr.Feature(buffer_lyr_definition)
            simplified_feature.SetGeometry(geomBuffer)
            buffer_layer.CreateFeature(simplified_feature)
        buffer_data = None
        self.make_esri_projection('simp_reef_poly', 5070)

        return simplified_reef_polygons

    def start(self):
        """Entrypoint for processing Reefs layer""" 

        # TODO load other shapefile and buffer backwards
        reef_polygons = str(OUTPUTS / get_config_item('REEF', 'POLYGONS_1KM'))
        clipped_reef_shp = self.clip_reef_shapefile(reef_polygons)
        projected_reef_shp = self.project_reef_shapefile(clipped_reef_shp)
        simplified_reef_shp = self.simplify_reef_shapefile(projected_reef_shp)
        dissolved_reef_shp = self.dissolve_overlapping_polygons(simplified_reef_shp)
        reef_buffer_paths = self.create_gdal_multiple_buffer(dissolved_reef_shp)
        dissolved_buffer_paths = self.dissolve_overlapping_buffers(reef_buffer_paths)
        self.create_reef_rasters(dissolved_buffer_paths)

    def validate_reef_shapefile(self, clipped_reef_shp):
        valid_projected_reef = str(OUTPUTS / 'valid_clipped_reef.shp')
        subprocess.run([
            'ogr2ogr',
            f'{valid_projected_reef}',
            f'{clipped_reef_shp}',
            '-makevalid'
        ])
        self.make_esri_projection('valid_clipped_reef')
        return valid_projected_reef
