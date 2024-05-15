import csv
import pathlib
import requests
# import geopandas as gpd

from osgeo import ogr
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateGroundingsLayerException(Exception):
    """Custom exception for tool"""

    pass


class CreateGroundingsLayerEngine(Engine):
    """Class to hold the logic for processing the Reefs layer"""


    def __init__(self, param_lookup:dict=None):
        super().__init__()
        if param_lookup:
            self.param_lookup = param_lookup
            if self.param_lookup['input_directory'].valueAsText:
                global INPUTS
                INPUTS = pathlib.Path(self.param_lookup['input_directory'].valueAsText)
            if self.param_lookup['output_directoty'].valueAsText:
                global OUTPUTS
                OUTPUTS = pathlib.Path(self.param_lookup['output_directoty'].valueAsText)

    def get_grounding_incidents(self) -> str:
        """Convert NOAA Office of Response and Restoration CSV file to shapefile"""

        self.message('Creating ORR Groundings point shapefile')
        incidents_csv = str(INPUTS / get_config_item('GROUNDINGS', 'CSV'))
        with open(incidents_csv, 'r', encoding='iso8859_7') as orr_data:
            data_reader = csv.DictReader(orr_data)
            fields = data_reader.fieldnames
            driver = ogr.GetDriverByName('ESRI Shapefile')
            output_path = OUTPUTS / 'orr_groundings.shp'
            output_shp = str(output_path)
            points_data = driver.CreateDataSource(output_shp)
            points_layer = points_data.CreateLayer("points", geom_type=ogr.wkbPoint)

            # Create fields
            field_map = {} # gdal shortens field names
            for field in fields:
                shp_field = field[:10]  # TODO shapefile field limit of 10, switch to GPKG?
                ogr_field = ogr.FieldDefn(shp_field, ogr.OFTString)
                points_layer.CreateField(ogr_field)
                field_map[field] = shp_field 
                
            points_lyr_definition = points_layer.GetLayerDefn()
            for row in data_reader:
                if 'Grounding' not in row['tags']:
                    continue

                point = (float(row['lon']), float(row['lat']))
                if not self.within_extent(driver, *point):
                    continue
                feature = ogr.Feature(points_lyr_definition)
                try:
                    for key, value in row.items():
                        feature.SetField(field_map[key], value)
                except:
                    self.log_error()
                geom = ogr.Geometry(ogr.wkbPoint)
                geom.AddPoint(*point)
                feature.SetGeometry(geom)
                points_layer.CreateFeature(feature)
                feature = None
            points_data = None
            self.make_esri_projection(output_path.stem)

        return output_shp
    
    # def get_uscg_groundings(self) -> gpd.GeoDataFrame:
    #     """Convert US Coast Guard Maritime Incidents service to shapefile"""

    #     print('Creating USCG Groundings point shapefile')
    #     url = get_config_item('GROUNDINGS', 'SERVICE')
    #     grounding_json = requests.get(url).json()
    #     groundings_df = gpd.GeoDataFrame.from_features(grounding_json['features'])
    #     output_shp = str(OUTPUTS / 'uscg_groundings.shp')
    #     groundings_df.to_file(output_shp, driver='ESRI Shapefile')

    #     return output_shp

    def start(self):
        """Entrypoint for processing Groundings layer""" 

        self.get_grounding_incidents()
        # self.get_uscg_groundings()
        self.check_logging()
