import yaml
import pathlib

from osgeo import osr
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class Engine:
    def get_config_item(item:str) -> str:
        """Load config and return speciific key"""

        with open(str(INPUTS / 'config.yaml'), 'r') as lookup:
            config = yaml.safe_load(lookup)
        return config[item]
    
    def within_extent(self, shapefile_driver, longitude: float, latitude: float) -> bool:
        """Check if lat/lon is within the WGS84 extent"""

        wgs84_bbox = str(INPUTS / get_config_item('SHARED', 'BBOX_SHP'))
        bbox_data = shapefile_driver.Open(wgs84_bbox)
        bbox_extent = bbox_data.GetLayer().GetExtent()
        if bbox_extent[0] < longitude < bbox_extent[1] and bbox_extent[2] < latitude < bbox_extent[3]:
            return True
        else:
            return False

    def make_esri_projection(self, file_name, epsg=4326):
        """Create an Esri .prj file for a shapefile"""

        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(epsg)
        output_projection.MorphToESRI()
        file = open(OUTPUTS / f'{file_name}.prj', 'w')
        file.write(output_projection.ExportToWkt())
        file.close()