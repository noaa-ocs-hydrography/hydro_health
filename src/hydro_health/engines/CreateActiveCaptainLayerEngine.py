import pathlib
import json

from osgeo import ogr, osr
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateActiveCaptainLayerException(Exception):
    """Custom exception for tool"""

    pass


class CreateActiveCaptainLayerEngine(Engine):
    """Class to hold the logic for processing the Reefs layer"""


    def __init__(self, param_lookup:dict=None):
        if param_lookup:
            self.param_lookup = param_lookup
            if self.param_lookup['input_directory'].valueAsText:
                global INPUTS
                INPUTS = pathlib.Path(self.param_lookup['input_directory'].valueAsText)
            if self.param_lookup['output_directoty'].valueAsText:
                global OUTPUTS
                OUTPUTS = pathlib.Path(self.param_lookup['output_directoty'].valueAsText)

    def create_ac_shapefile(self, ac_points_json:str) -> str:
        """Create Active Captain Point shapefile"""

        print('Creating Active Captain point shapefile')
        with open(ac_points_json) as file:
            ac_points = json.load(file)
        driver = ogr.GetDriverByName('ESRI Shapefile')
        ac_points_path = OUTPUTS / 'active_captain_points.shp'
        ac_points_file = str(ac_points_path)
        
        ac_points_data = driver.CreateDataSource(ac_points_file)
        ac_points_layer = ac_points_data.CreateLayer("ac_points", geom_type=ogr.wkbPoint)

        # Create fields
        icon_field = ogr.FieldDefn('iconUrl', ogr.OFTString)
        point_count_field = ogr.FieldDefn('poiCount', ogr.OFTInteger)
        point_id = ogr.FieldDefn('pointId', ogr.OFTInteger64)
        fields = [icon_field, point_count_field, point_id]
        for field in fields:
            ac_points_layer.CreateField(field)
        ac_points_lyr_definition = ac_points_layer.GetLayerDefn()
        for point in ac_points['pointsOfInterest']:
            if not self.within_extent(driver, point['mapLocation']['longitude'], point['mapLocation']['latitude']):
                continue
            feature = ogr.Feature(ac_points_lyr_definition)
            feature.SetField('iconUrl', point['iconUrl'])
            feature.SetField('poiCount', point['poiCount'])
            feature.SetField('pointId', point['id'])
            geom = ogr.Geometry(ogr.wkbPoint)
            geom.AddPoint(point['mapLocation']['longitude'], point['mapLocation']['latitude'])
            feature.SetGeometry(geom)
            ac_points_layer.CreateFeature(feature)
            feature = None
        ac_points_data = None
        self.make_esri_projection(ac_points_path.stem)

        return ac_points_file

    def start(self):
        """Entrypoint for processing Reefs layer""" 

        ac_points_json = str(INPUTS / get_config_item('ACTIVECAPTAIN', 'JSON'))
        ac_points_shp = self.create_ac_shapefile(ac_points_json)