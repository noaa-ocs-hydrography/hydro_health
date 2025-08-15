import pathlib
import sys
import logging

from osgeo import osr, gdal


gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class Engine:
    def __init__(self):
        # set up logging
        # TODO do we need daily logs?
        self.log_path = str(OUTPUTS / 'hh_log.txt')
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename=self.log_path,
                    format='%(levelname)s:%(asctime)s %(message)s',
                    level=logging.INFO)
        self.logged = False

    def check_logging(self) -> None:
        if self.logged:
            self.message(f'Check log: {self.log_path}')

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