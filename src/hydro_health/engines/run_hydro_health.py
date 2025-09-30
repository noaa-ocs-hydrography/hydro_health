import os
import pathlib
import time
import sys
import yaml

from datetime import datetime

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.helpers import tools, runners


INPUTS = pathlib.Path(__file__).parents[3] / "inputs"
OUTPUTS = pathlib.Path(__file__).parents[3] / "outputs"


def get_env_param_lookup(env: str) -> dict[str]:
    """Isolate logic for setting param_lookup"""

    if env == 'local':
        param_lookup = {
            'input_directory': tools.Param(''),
            'output_directory': tools.Param(str(OUTPUTS)),
            'eco_regions': tools.Param(''),
            'drawn_polygon': tools.Param(str(INPUTS / 'drawn_polygons.geojson')),
            'env': env
        }
    else:
        param_lookup = {
            'input_directory': tools.Param(''),
            'output_directory': tools.Param(tools.get_config_item('SHARED', 'OUTPUT_FOLDER')),
            'eco_regions': tools.Param(''),
            'env': env
        }
    return param_lookup


def run_hydro_health(config_name: str) -> None:
    start = time.time()
    env = tools.get_environment()
    print('Environment:', env)
    param_lookup = get_env_param_lookup(env)

    output_directory = pathlib.Path(param_lookup['output_directory'].valueAsText)
    if os.path.exists(output_directory / 'log_prints.txt'):
        now = time.time()
        os.rename(output_directory / 'log_prints.txt', output_directory / f'log_prints_{now}.txt')
    print('Output folder:', output_directory)

    config_path = INPUTS / "run_configs" / config_name
    with open(config_path, "r") as lookup:
        config = yaml.safe_load(lookup)
        print(f'Script has been run {len(config["runtimes"])} time(s)')
        # load ecoregions from config for remote run
        param_lookup['eco_regions'].value = ';'.join(config['ecoregions']) if env == 'remote' else ''
        print(f"Running Hydro Health for ecoregions: {param_lookup['eco_regions'].valueAsText}")
        tiles = tools.get_ecoregion_tiles(param_lookup)
        for step in config["steps"]:
            if step["tool"] == "run_bluetopo_tile_engine" and step["run"]:
                runners.run_bluetopo_tile_engine(tiles, param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "run_digital_coast_engine" and step["run"]:
                runners.run_digital_coast_engine(tiles, param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "run_laz_conversion_engine" and step["run"]:
                runners.run_laz_conversion_engine(tiles, param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "run_metadata_engine" and step["run"]:
                runners.run_metadata_engine(param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "run_vrt_creation" and step["run"]:
                runners.run_raster_vrt_engine(param_lookup, skip_existing=False)
            elif step["tool"] == "run_raster_mask_engine" and step["run"]:
                runners.run_raster_mask_engine(param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "grid_digital_coast_files" and step["run"]:
                tools.grid_digital_coast_files(param_lookup['output_directory'].valueAsText, 'DigitalCoast')
            elif step["tool"] == "run_sediment_layer_engine" and step["run"]:
                runners.run_sediment_layer_engine()
            elif step["tool"] == "run_tsm_layer_engine" and step["run"]:
                runners.run_tsm_layer_engine()
            elif step["tool"] == "run_hurricane_layer_engine" and step["run"]:
                runners.run_hurricane_layer_engine  
    update_config_runtime(config_path, config)
    end = time.time()
    print(f"Total Runtime: {(end - start) / 60}")
    for ecoregion in pathlib.Path(param_lookup['output_directory'].valueAsText).glob('ER_*'):
        bluetopo_path = pathlib.Path(ecoregion / tools.get_config_item('BLUETOPO', 'SUBFOLDER') / 'BlueTopo')
        if bluetopo_path.is_dir():
            print(f'{ecoregion.stem} BlueTopo tiles:', len(next(os.walk(str(bluetopo_path)))[1]))
        else:
            print('No tiles downloaded')
    print("done")


def update_config_runtime(config_path: pathlib.Path, config: dict[list]) -> None:
    """Update run config with run time"""

    config_path = INPUTS / "run_configs" / config_name
    with open(config_path, "w") as config_file:
        current_day = datetime.now()
        timestamp = current_day.strftime("%m%d%Y")
        config['runtimes'].append(str(timestamp))
        print(f'Updating config runtimes for date: {timestamp}')
        yaml.safe_dump(config, config_file, sort_keys=False)


if __name__ == "__main__":
    config_name = "hydro_health_session_08272025.yaml"
    run_hydro_health(config_name)
