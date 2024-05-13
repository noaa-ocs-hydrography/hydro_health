import pathlib
import requests
import pandas as pd
import geopandas as gpd

from osgeo import osr
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateGroundingsLayerException(Exception):
    """Custom exception for tool"""

    pass


class CreateGroundingsLayerEngine(Engine):
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

    def get_grounding_incidents(self) -> gpd.GeoDataFrame:
        """Convert NOAA Office of Response and Restoration CSV file to shapefile"""

        print('Creating ORR Groundings point shapefile')
        incidents_csv = str(INPUTS / get_config_item('GROUNDINGS', 'CSV'))
        filter = ['Grounding']
        groundings_gdf = gpd.read_file(incidents_csv,
                                       encoding='iso8859_7', # Needed iso encoding for bytes field
                                       X_POSSIBLE_NAMES="lon",
                                       Y_POSSIBLE_NAMES="lat").query('tags in @filter')
        output_shp = str(OUTPUTS / 'groundings.shp')
        # Shapefile only allows 10 char attributes
        # TODO to_file('dataframe.gpkg', driver='GPKG', layer='name')  
        groundings_gdf.to_file(output_shp, driver='ESRI Shapefile')

        return output_shp
    
    def get_uscg_groundings(self) -> gpd.GeoDataFrame:
        """Convert US Coast Guard Maritime Incidents service to shapefile"""

        print('Creating USCG Groundings point shapefile')
        url = get_config_item('GROUNDINGS', 'SERVICE')
        grounding_json = requests.get(url).json()
        groundings_df = gpd.GeoDataFrame.from_features(grounding_json['features'])
        output_shp = str(OUTPUTS / 'uscg_groundings.shp')
        groundings_df.to_file(output_shp, driver='ESRI Shapefile')

        return output_shp

    def start(self):
        """Entrypoint for processing Groundings layer""" 

        groundings_df = self.get_grounding_incidents()
        self.get_uscg_groundings()
