import cProfile
import pstats
import pathlib
import sys


HYDRO_HEALTH = pathlib.Path(__file__).parents[3]
sys.path.append(str(HYDRO_HEALTH))


from hydro_health.engines.tiling.ModelDataPreProcessor import ModelDataPreProcessor
from hydro_health.engines.CreateTSMLayerEngine import CreateTSMLayerEngine
from hydro_health.engines.CreateSedimentLayerEngine import CreateSedimentLayerEngine


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    processor = ModelDataPreProcessor()

    processor.process()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)