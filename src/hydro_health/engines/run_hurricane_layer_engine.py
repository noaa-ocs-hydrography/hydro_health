import cProfile
import pstats
import pathlib
import sys


HYDRO_HEALTH = pathlib.Path(__file__).parents[2]
sys.path.append(str(HYDRO_HEALTH))


from hydro_health.engines.CreateHurricaneLayerEngine import CreateHurricaneLayerEngine

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    engine = CreateHurricaneLayerEngine()
    engine.run()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  