import pathlib
import time
import os
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import (
    Param,
    grid_vrt_files
)


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


if __name__ == '__main__':
    if os.path.exists(OUTPUTS / 'log_prints.txt'):
        now = time.time()
        os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
    param_lookup = {
        'input_directory': Param(''),
        'output_directory': Param(str(OUTPUTS)),
        'eco_regions': Param(''),
        'drawn_polygon': Param(str(OUTPUTS / 'drawn_polygons.geojson'))
        # 'drawn_polygon': Param('')
    }
    
    start = time.time()
    grid_vrt_files(param_lookup['output_directory'].valueAsText, 'DigitalCoast')

    end = time.time()
    print(f'Total Runtime: {end - start}') # Florida-West - 640.7945353984833 seconds or 10.67990892330806 minutes, 7.23GB, 727 folders, 1454 files
    # takes 102 seconds to verify if already downloaded
    
    print('done')
