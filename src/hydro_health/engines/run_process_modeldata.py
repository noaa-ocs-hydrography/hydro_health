import pathlib
import time
import os
import cProfile
import pstats
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import process_model_data, Param


INPUTS = pathlib.Path(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\HHM_Run\ER_3')
# OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    # if os.path.exists(OUTPUTS / 'log_prints.txt'):
    #     now = time.time()
    #     os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
    param_lookup = {
        'input_directory': Param(str(INPUTS))
    }
    
    process_model_data(param_lookup['input_directory'].valueAsText)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  