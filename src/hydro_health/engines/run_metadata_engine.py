import pathlib
import time
import os
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import Param, get_environment, get_config_item
from hydro_health.helpers.runners import run_metadata_engine


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


if __name__ == '__main__':
    env = get_environment()
    if env == 'local':
        param_lookup = {
            'input_directory': Param(''),
            'output_directory': Param(str(OUTPUTS)),
            'eco_regions': Param(''),
            'drawn_polygon': Param(str(INPUTS / 'drawn_polygons.geojson'))
        }
    else:
        param_lookup = {
            'input_directory': Param(''),
            'output_directory': Param(get_config_item('SHARED', 'OUTPUT_FOLDER')),
            'eco_regions': Param(''),
            'drawn_polygon': Param('')
        }

    start = time.time()
    print('starting')
    output_folder = pathlib.Path(param_lookup['output_directory'].valueAsText)
    ecoregions = [file_path.stem for file_path in output_folder.rglob('ER_*') if file_path.is_dir()]
    print(f"Running Hydro Health for ecoregions: {ecoregions}")
    run_metadata_engine(param_lookup['output_directory'].valueAsText)
    end = time.time()
    print(f'Total Runtime: {end - start}') # Florida-West - 640.7945353984833 seconds or 10.67990892330806 minutes, 7.23GB, 727 folders, 1454 files