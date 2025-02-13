import pathlib
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import process_tiles, get_ecoregion_tiles, Param


OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


if __name__ == '__main__':
    output = OUTPUTS / 'junk'
    output.mkdir(parents=True, exist_ok=True)
    param_lookup = {
        'input_directory': Param(''),
        'output_directory': Param(output),
        'eco_regions': Param('ER_3-Florida-West;')
    }
    tiles = get_ecoregion_tiles(param_lookup)
    process_tiles(tiles, param_lookup['output_directory'].valueAsText)
    print('done')