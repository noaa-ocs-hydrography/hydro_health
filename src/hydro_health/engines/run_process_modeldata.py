import pathlib
import time
import os
import cProfile
import pstats
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))


# OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'
from hydro_health.engines.tiling.ModelDataProcessor import ModelDataProcessor


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    # if os.path.exists(OUTPUTS / 'log_prints.txt'):
    #     now = time.time()
    #     os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
    
    processor = ModelDataProcessor()
    processor.process()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  