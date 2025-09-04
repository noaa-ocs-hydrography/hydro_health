import pathlib
import sys
import json
import logging
import re
import requests
import geopandas as gpd

from osgeo import osr, gdal
from hydro_health.helpers.tools import get_config_item


gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class Engine:
    """Base class for all Engines"""

    def __init__(self):
        self.approved_size = 200000000  # 2015 USACE polygon was 107,987,252 sq. meters

    # def __init__(self):
    #     # set up logging
    #     # TODO do we need daily logs?
    #     self.log_path = str(OUTPUTS / 'hh_log.txt')
    #     self.logger = logging.getLogger(__name__)
    #     logging.basicConfig(filename=self.log_path,
    #                 format='%(levelname)s:%(asctime)s %(message)s',
    #                 level=logging.INFO)
    #     self.logged = False

    def approved_dataset(self, feature_json: dict[dict]) -> bool:
        """Only allow certain provider types"""

        # CUDEM: NOAA NCEI
        provider_list_text = ['USACE', 'NCMP', 'NGS', 'NOAA NCEI', 'USGS']
        for text in provider_list_text:
            if text in feature_json['attributes']['provider_results_name']:
                return True
        return False
    
    def check_logging(self) -> None:
        if self.logged:
            self.message(f'Check log: {self.log_path}')

    def cleansed_url(self, url: str) -> str:
        """Remove found illegal characters from URLs"""

        illegal_chars = ['{', '}']
        for char in illegal_chars:
            url = url.replace(char, '')
        return url
    
    def get_available_datasets(self, geometry_coords: str, ecoregion_id: str, outputs: str) -> None:
        """Query NOWCoast REST API for available datasets"""

        payload = {
            "aoi": f"SRID=4269;{geometry_coords}",
            "published": "true",
            "dataTypes": ["Lidar", "DEM"],
            "dialect": "arcgis",
        }
        response = requests.post(get_config_item('DIGITALCOAST', 'API'), data=payload)
        datasets_json = response.json()

        if response.status_code == 404:
            raise Exception(f"Digital Coast Error: {response.reason}")

        tile_index_links = []
        for feature in datasets_json['features']:
            # print(feature['attributes']['DataType'], feature['attributes']['Year'], feature['attributes']['provider_results_name'])
            if not self.approved_dataset(feature):
                continue
            folder_name = re.sub('\W+',' ', feature['attributes']['provider_results_name']).strip().replace(' ', '_') + '_' + str(feature['attributes']['Year'])  # remove illegal chars
            output_folder_path = pathlib.Path(outputs) / ecoregion_id / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast' / f"{folder_name}_{feature['attributes']['ID']}"
            output_folder_path.mkdir(parents=True, exist_ok=True)

            # Write out JSON
            output_json = pathlib.Path(output_folder_path) / 'feature.json'
            external_provider_links = json.loads(feature['attributes']['ExternalProviderLink'])['links']
            feature['attributes']['ExternalProviderLink'] = external_provider_links
            with open(output_json, 'w') as writer:
                writer.write(json.dumps(feature['attributes'], indent=4))

            for external_data in external_provider_links:
                if external_data['label'] == 'Bulk Download':
                    tile_index_links.append({'label': 'Bulk Download', 'data_type': feature['attributes']['DataType'], 'link': external_data['link'], 'provider_path': output_folder_path}) 
        return tile_index_links

    def get_ecoregion_geometry_strings(self, tile_gdf: gpd.GeoDataFrame, ecoregion: str) -> str:
        """Build bbox string dictionary of tiles in web mercator projection"""

        geometry_coords = []
        ecoregion_groups = tile_gdf.groupby('EcoRegion')
        for er_id, ecoregion_group in ecoregion_groups:
            if er_id == ecoregion:
                ecoregion_group_web_mercator = ecoregion_group.to_crs(4269)  # POST request only allows this EPSG
                ecoregion_group_web_mercator['geom_type'] = 'Polygon'
                tile_geometries = ecoregion_group_web_mercator[['geom_type', 'geometry']]
                tile_boundary = tile_geometries.dissolve(by='geom_type')
                tile_wkt = tile_boundary.iloc[0].geometry
                geometry_coords.append(tile_wkt)

        return geometry_coords
    
    def log_error(self) -> None:
        self.logger.error(gdal.GetLastErrorMsg())
        if not self.logged:
            self.logged = True
    
    def message(self, content:str) -> None:
        """Wrap Arcpy for printing"""

        if 'arcpy' in sys.modules:
            module = __import__('arcpy')
            getattr(module, 'AddMessage')(content)
        else:
            print(content)

    def make_esri_projection(self, file_name, epsg=4326):
        """Create an Esri .prj file for a shapefile"""

        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(epsg)
        output_projection.MorphToESRI()
        file = open(OUTPUTS / f'{file_name}.prj', 'w')
        file.write(output_projection.ExportToWkt())
        file.close()

    def write_message(self, message: str, output_folder: str) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')