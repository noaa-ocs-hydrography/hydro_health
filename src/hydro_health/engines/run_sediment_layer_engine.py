import pathlib
import cProfile
import pstats

HYDRO_HEALTH = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HYDRO_HEALTH))

from hydro_health.engines.CreateSedimentLayerEngine import CreateSedimentLayerEngine

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    engine = CreateSedimentLayerEngine()
    engine.run()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10) 
