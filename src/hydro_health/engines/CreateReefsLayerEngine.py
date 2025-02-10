import os
import pathlib
import subprocess

from osgeo import ogr, osr, gdal
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateReefsLayerException(Exception):
    """Custom exception for tool"""

    pass


class CreateReefsLayerEngine(Engine):
    """Class to hold the logic for processing the Reefs layer"""

    def __init__(self, param_lookup:dict=None):
        super().__init__()
        if param_lookup:
            self.param_lookup = param_lookup
            if self.param_lookup['input_directory'].valueAsText:
                global INPUTS
                INPUTS = pathlib.Path(self.param_lookup['input_directory'].valueAsText)
            if self.param_lookup['output_directory'].valueAsText:
                global OUTPUTS
                OUTPUTS = pathlib.Path(self.param_lookup['output_directory'].valueAsText)

    def create_gdal_multiple_buffer(self, reef_polygons: str) -> None:
        """Create multiple buffers around reef polygons using GDAL"""

        buffer_distance = list(reversed(get_config_item('REEF', 'BUFFER_DISTANCE')))
        tanker_buffer_distance = list(reversed(get_config_item('REEF', 'TANKER_BUFFER_DISTANCE')))

        driver = ogr.GetDriverByName('ESRI Shapefile')
        reef_data = ogr.Open(reef_polygons)
        reef_layer = reef_data.GetLayer()

        buffer_types = {}
        buffer_distances = {'regular': buffer_distance, 'tanker': tanker_buffer_distance}
        for name, distances in buffer_distances.items():
            buffer_paths = []
            for distance in distances:
                buffer_path = OUTPUTS / f'reef_{name}_buffer_{distance}.shp'
                buffer_file = str(buffer_path)
                buffer_paths.append(buffer_file)
                # if os.path.exists(buffer_file):
                #     self.message(f'Buffer file already exists: {buffer_file}')
                #     continue
                self.message(f'Attempting: {buffer_file}')
                buffer_data = driver.CreateDataSource(buffer_file)
                buffer_layer = buffer_data.CreateLayer("buffer_polygons", geom_type=ogr.wkbPolygon)
                distance_field = ogr.FieldDefn('distance', ogr.OFTReal)
                buffer_layer.CreateField(distance_field)
                buffer_lyr_definition = buffer_layer.GetLayerDefn()
                # Buffer each feature to the current distance and output shapefile
                for feature in reef_layer:
                    ingeom = feature.GetGeometryRef()
                    geomBuffer = ingeom.Buffer(distance, 10)
                    buffered_feature = ogr.Feature(buffer_lyr_definition)
                    buffered_feature.SetField('distance', distance)
                    buffered_feature.SetGeometry(geomBuffer)
                    buffer_layer.CreateFeature(buffered_feature)
                    buffered_feature = None
                buffer_data = None
                self.make_esri_projection(buffer_path.stem, 5070)
            buffer_types[name] = buffer_paths
        reef_data = None
        return buffer_types
    

    def rasterize_buffer_polygons(self, buffer_types: dict=None) -> None:
        """Merge buffer polygons into single shapefiles for rasterization"""
        
        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(5070)
        # gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
        raster_files = {}
        for buffer_type, buffer_files in buffer_types.items():
            raster_files[buffer_type] = []
            for buffer_file in buffer_files:
                buffer_size = pathlib.Path(buffer_file).stem.split('_')[-1]
                reef_raster = str(pathlib.Path(OUTPUTS / f'reef_{buffer_type}_{buffer_size}_raster.tif'))
                # if os.path.exists(reef_raster):
                #     self.message(f'Raster file already exists: {reef_raster}')
                #     raster_files[buffer_type].append(reef_raster)    
                #     continue
                raster_files[buffer_type].append(reef_raster)    
                input_shp = ogr.Open(buffer_file)
                shp_layer = input_shp.GetLayer()
                pixel_size = 200
                xmin, xmax, ymin, ymax = shp_layer.GetExtent()
                self.message(f'Rasterizing: {buffer_file}')
                no_data = -999
                raster = gdal.Rasterize(reef_raster, buffer_file,
                                        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"],
                                        noData=no_data,
                                        format='GTiff',
                                        outputType=gdal.GDT_Int32,
                                        outputBounds=[xmin, ymin, xmax, ymax],
                                        attribute='distance',
                                        xRes=pixel_size,
                                        yRes=pixel_size)
                raster = None
        
        for buffer_type, buffer_files in raster_files.items():
            merged_final_raster = str(OUTPUTS / f'merged_{buffer_type}_raster.tif')
            # if os.path.exists(merged_final_raster):
            #     self.message(f'Merged raster already exists: {merged_final_raster}')
            #     continue
            self.message(f'Merging {buffer_type} rasters')
            raster = gdal.Warp(merged_final_raster, buffer_files, format="GTiff",
                        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"])
            raster = None # Close file and flush to disk

    def create_reef_rasters(self, reef_buffer_paths: str) -> None:
        """Create GeoTiff raster files from each shapefile"""

        raster_files = []
        for shp_path in reef_buffer_paths:
            raster_name = pathlib.Path(shp_path).stem
            reef_raster = str(OUTPUTS / f'{raster_name}.tif')
            # if os.path.exists(reef_raster):
            #     self.message(f'Raster file already exists: {reef_raster}')
            #     raster_files.append(reef_raster)
            #     continue

            input_shp = ogr.Open(shp_path)
            shp_layer = input_shp.GetLayer()
            pixel_size = 0.01
            xmin, xmax, ymin, ymax = shp_layer.GetExtent()
            self.message(f'Rasterizing: {reef_raster}')
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
        """Clip global reef polygons to a North America window"""

        clipped_reef_shp = str(OUTPUTS / 'reef_polygons_clip.shp')
        # if os.path.exists(clipped_reef_shp):
        #     self.message('Clipped Reef shapefile already exists')
        #     return clipped_reef_shp
        self.message('Clipping reef polygons to North America')
        driver = ogr.GetDriverByName('ESRI Shapefile')
        projected_reef_data = driver.Open(projected_reef_shp, 0)
        projected_reef_layer = projected_reef_data.GetLayer()

        wgs84_bbox = str(INPUTS / get_config_item('SHARED', 'BBOX_SHP'))
        reef_extent_data = driver.Open(wgs84_bbox, 0)  # using Albers for clip does not work with GDAL?!
        reef_extent_layer = reef_extent_data.GetLayer()

        
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
    
    def dissolve_overlapping_buffers(self, buffer_paths: dict) -> dict:
        """Dissolve overlapping buffers"""

        dissolved_buffers = {}
        for buffer_type, buffer_files in buffer_paths.items():
            dissolved_buffers[buffer_type] = []
            for buffer_file in buffer_files:
                dissolved_buffers[buffer_type].append(self.dissolve_overlapping_polygons(buffer_file))
        return dissolved_buffers
    
    def dissolve_overlapping_polygons(self, input_shapefile, with_distance=True):
        """Dissolve buffer polygons"""

        shp_path = pathlib.Path(input_shapefile)
        dissolved_reef_polygons = str(OUTPUTS / f'dis_{shp_path.stem}.shp')
        # if os.path.exists(dissolved_reef_polygons):
        #     self.message('Dissolved Buffer shapefile already exists')
        #     return dissolved_reef_polygons
        self.message(f'Dissolving buffer: {shp_path.stem}')
        if with_distance:
            sql = f"SELECT ST_Union(geometry), distance FROM {shp_path.stem} GROUP BY distance"
        else:
            sql = f"SELECT ST_Union(geometry) FROM {shp_path.stem}"
        subprocess.run([
            str(INPUTS / 'ogr2ogr.exe'),
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
        # if os.path.exists(projected_reef_shp):
        #     self.message('Projected Reef shapefile already exists')
        #     return projected_reef_shp
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
        # if os.path.exists(simplified_reef_polygons):
        #     self.message('Simplified Reef shapefile already exists')
        #     return simplified_reef_polygons
        self.message('Simplifying 1km reef polygons')
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

        # TODO try in-memory datasets: https://stackoverflow.com/questions/44293091/use-gdal-open-to-open-an-in-memory-gdal-dataset
        reef_polygons = str(INPUTS / get_config_item('REEF', 'POLYGONS_1KM'))
        clipped_reef_shp = self.clip_reef_shapefile(reef_polygons)
        projected_reef_shp = self.project_reef_shapefile(clipped_reef_shp)
        simplified_reef_shp = self.simplify_reef_shapefile(projected_reef_shp)
        dissolved_reef_shp = self.dissolve_overlapping_polygons(simplified_reef_shp, with_distance=False)
        reef_buffer_paths = self.create_gdal_multiple_buffer(dissolved_reef_shp)
        dissolved_buffer_paths = self.dissolve_overlapping_buffers(reef_buffer_paths)
        self.rasterize_buffer_polygons(dissolved_buffer_paths)
        # self.create_reef_rasters(dissolved_buffer_paths)

    def validate_reef_shapefile(self, clipped_reef_shp):
        valid_projected_reef = str(OUTPUTS / 'valid_clipped_reef.shp')
        subprocess.run([
            str(INPUTS / 'ogr2ogr.exe'),
            f'{valid_projected_reef}',
            f'{clipped_reef_shp}',
            '-makevalid'
        ])
        self.make_esri_projection('valid_clipped_reef')
        return valid_projected_reef
