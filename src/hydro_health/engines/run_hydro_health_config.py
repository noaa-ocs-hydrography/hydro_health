import pathlib
import time
import os
import yaml
from osgeo import gdal, osr, ogr
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))

from hydro_health.helpers import tools


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


def run_hydro_health(config_name) -> None:
    start = time.time()
    config_path = INPUTS / 'lookups' / 'run_configs' / config_name
    with open(config_path, 'r') as lookup:
        config = yaml.safe_load(lookup)
        for step in config['steps']:
            if step['tool'] == 'ModelDataPreProcessor' and step['run']:
                 tools.run_model_data_processor()
            elif step['tool'] == 'BlueTopoProcessor' and step['run']:
                 tools.process_bluetopo_tiles()
            elif step['tool'] == 'DigitalCoastProcessor' and step['run']:
                 tools.process_digital_coast_files()
            elif step['tool'] == 'RasterMaskProcessor' and step['run']:
                 tools.process_create_masks()
            elif step['tool'] == 'GridDigitalCoastProcessor' and step['run']:
                 tools.grid_vrt_files()
                 
    end = time.time()
    print(f'Total Runtime: {end - start}')
    print('done')


if __name__ == '__main__':
    config_name = 'hydro_health_07082025.yaml'
    run_hydro_health(config_name)
