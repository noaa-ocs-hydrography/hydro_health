import cProfile
import pstats
import pathlib
import sys


HYDRO_HEALTH = pathlib.Path(__file__).parents[2]
sys.path.append(str(HYDRO_HEALTH))


from hydro_health.engines.CreateTSMLayerEngine import CreateTSMLayerEngine

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    engine = CreateTSMLayerEngine()
    engine.run()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  