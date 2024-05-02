import yaml
import pathlib

from osgeo import osr


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class Engine:
    def get_config_item(item:str) -> str:
        """Load config and return speciific key"""

        with open(str(INPUTS / 'config.yaml'), 'r') as lookup:
            config = yaml.safe_load(lookup)
        return config[item]

    def make_esri_projection(self, file_name, epsg=4326):
        """Create an Esri .prj file for a shapefile"""

        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(epsg)
        output_projection.MorphToESRI()
        file = open(OUTPUTS / f'{file_name}.prj', 'w')
        file.write(output_projection.ExportToWkt())
        file.close()