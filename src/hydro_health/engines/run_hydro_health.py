import os
import pathlib
import time
import sys
import yaml

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.helpers import tools, runners


INPUTS = pathlib.Path(__file__).parents[3] / "inputs"
OUTPUTS = pathlib.Path(__file__).parents[3] / "outputs"


def run_hydro_health(config_name) -> None:
    start = time.time()
    env = tools.get_environment()
    if env == 'local':
        param_lookup = {
            'input_directory': tools.Param(''),
            'output_directory': tools.Param(str(OUTPUTS)),
            'eco_regions': tools.Param(''),
            'drawn_polygon': tools.Param(str(INPUTS / 'drawn_polygons.geojson'))
        }
    else:
        param_lookup = {
            'input_directory': tools.Param(''),
            'output_directory': tools.Param(str(OUTPUTS)),
            'eco_regions': tools.Param('ER_3'),
            'drawn_polygon': tools.Param('')
        }
    tiles = tools.get_ecoregion_tiles(param_lookup)
    print('Environment:', env)
    print('Output folder:', param_lookup['output_directory'].valueAsText)

    config_path = INPUTS / "run_configs" / config_name
    with open(config_path, "r") as lookup:
        config = yaml.safe_load(lookup)
        for step in config["steps"]:
            if step["tool"] == "run_tsm_layer_engine" and step["run"]:
                runners.run_tsm_layer_engine()
            elif step["tool"] == "run_sediment_layer_engine" and step["run"]:
                runners.run_sediment_layer_engine()
            elif step["tool"] == "run_bluetopo_tile_engine" and step["run"]:
                runners.run_bluetopo_tile_engine(tiles, param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "run_digital_coast_engine" and step["run"]:
                runners.run_digital_coast_engine(tiles, param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "run_vrt_creation" and step["run"]:
                tools.run_vrt_creation(param_lookup)
            elif step["tool"] == "run_raster_mask_engine" and step["run"]:
                runners.run_raster_mask_engine(param_lookup['output_directory'].valueAsText)
            elif step["tool"] == "grid_digital_coast_files" and step["run"]:
                tools.grid_digital_coast_files(param_lookup['output_directory'].valueAsText, 'DigitalCoast')

    end = time.time()
    print(f"Total Runtime: {end - start}")
    for ecoregion in pathlib.Path(param_lookup['output_directory'].valueAsText).glob('ER_*'):
        if pathlib.Path(ecoregion / 'BlueTopo').is_dir():
            print(f'{ecoregion.stem} BlueTopo tiles:', len(next(os.walk(os.path.join(param_lookup['output_directory'].valueAsText, ecoregion, 'BlueTopo')))[1]))
        else:
            print('No tiles downloaded')
    print("done")


if __name__ == "__main__":
    config_name = "hydro_health_session_07082025.yaml"
    run_hydro_health(config_name)
