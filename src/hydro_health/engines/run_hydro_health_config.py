import pathlib
import time
import sys
import yaml

HH_MODEL = pathlib.Path(__file__).parents[2]
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
            if step['tool'] == 'process_model_data' and step['run']:
                 tools.process_model_data()
            elif step['tool'] == 'process_bluetopo_tiles' and step['run']:
                 tools.process_bluetopo_tiles()
            elif step['tool'] == 'process_digital_coast_files' and step['run']:
                 tools.process_digital_coast_files()
            elif step['tool'] == 'process_raster_vrts' and step['run']:
                 tools.process_raster_vrts()
            elif step['tool'] == 'process_create_masks' and step['run']:
                 tools.process_create_masks()
            elif step['tool'] == 'grid_digital_coast_files' and step['run']:
                 tools.grid_digital_coast_files()
                 
    end = time.time()
    print(f'Total Runtime: {end - start}')
    print('done')


if __name__ == '__main__':
    config_name = 'hydro_health_07082025.yaml'
    run_hydro_health(config_name)
