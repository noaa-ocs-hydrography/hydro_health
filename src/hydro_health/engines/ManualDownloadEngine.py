import pathlib
import sys

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.helpers.tools import Param
from hydro_health.engines.MetadataS3Engine import MetadataS3Engine
from hydro_health.engines.RasterVRTS3Engine import RasterVRTS3Engine
from hydro_health.engines.tiling.RasterMaskS3Engine import RasterMaskS3Engine
from hydro_health.engines.tiling.GridDigitalCoastEngine import GridDigitalCoastEngine
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class ManualDownloadEngine:
    """Class for processing manually downloaded DEM files"""

    def __init__(self, param_lookup) -> None:
        self.param_lookup = param_lookup

    def process_masks(self) -> None:
        """Create masks with manual download additions"""

        engine = RasterMaskS3Engine(self.param_lookup)
        engine.run(str(OUTPUTS), manual_downloads=True)

    def process_metadata(self) -> None:
        """Build metdata text file for manually downloaded DEMs"""

        engine = MetadataS3Engine()
        digital_coast_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/ER_3/{get_config_item('DIGITALCOAST', 'SUBFOLDER')}/DigitalCoast_manual_downloads"
        engine.read_json_files(digital_coast_path, OUTPUTS)
        
    def process_vrt(self) -> None:
        """Create VRT files for each manually downloaded provider"""
        
        engine = RasterVRTS3Engine(self.param_lookup)
        engine.setup_dask('aws')
        engine.create_raster_vrts('NCMP', 'ER_3', 'DigitalCoast', data_folder='DigitalCoast_manual_downloads', skip_existing=False)
        engine.close_dask()

    def run(self) -> None:
        # self.process_vrt()
        self.process_masks()
        # Rerun Grid Tiling Engine
        print('Done')

if __name__ == '__main__':
    param_lookup = {
        'output_directory': Param(str(OUTPUTS)),
        'env': 'aws'
    }
    engine = ManualDownloadEngine(param_lookup)
    engine.run()