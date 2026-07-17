import pathlib
import sys
import os
import time

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

    def process_grid_tiles(self) -> None:
        """Tile manually downloaded VRT files to BlueTopo"""

        engine = GridDigitalCoastEngine(self.param_lookup)
        engine.run(manual_download=True)

    def process_masks(self) -> None:
        """Create masks with manual download additions"""

        engine = RasterMaskS3Engine(self.param_lookup)
        engine.run(str(OUTPUTS), manual_downloads=True)

    def process_metadata(self) -> None:
        """Build metdata text file for manually downloaded DEMs"""

        engine = MetadataS3Engine()
        digital_coast_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/ER_3/{get_config_item('DIGITALCOAST', 'SUBFOLDER')}/Digital_Coast_Manual_Downloads"
        engine.read_json_files(digital_coast_path, OUTPUTS)
        
    def process_vrt(self) -> None:
        """Create VRT files for each manually downloaded provider"""
        
        engine = RasterVRTS3Engine(self.param_lookup)
        engine.setup_dask('aws')
        engine.run(OUTPUTS, 'NCMP', 'ER_3', 'DigitalCoast', data_folder='Digital_Coast_Manual_Downloads', skip_existing=False)
        engine.close_dask()

    def rebuild_training_mask(self) -> None:
        engine = RasterMaskS3Engine(self.param_lookup)
        result = engine.remerge_training_mask("ER_3", OUTPUTS)
        print(result)

    def run(self) -> None:
        if os.path.exists(OUTPUTS / 'log_prints.txt'):
            now = time.time()
            os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
        
        # TODO this process creates missing Manual Download VRTs
        # Need to run normal VRT process before
        self.process_vrt()
        self.process_masks()  # mask process runs DigitalCoast and Digital_Coast_Manual_Downoads
        # self.rebuild_training_mask()
        # TODO this grid tiling process only does the manual downloads
        # Need to run both folders separately or fix code
        self.process_grid_tiles()
        print('Done')

if __name__ == '__main__':
    param_lookup = {
        'output_directory': Param(str(OUTPUTS)),
        'env': 'aws'
    }
    engine = ManualDownloadEngine(param_lookup)
    engine.run()