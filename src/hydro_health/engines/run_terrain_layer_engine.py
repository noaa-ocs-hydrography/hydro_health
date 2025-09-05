import pathlib
import cProfile
import pstats
import sys

HYDRO_HEALTH = pathlib.Path(__file__).parents[2]

sys.path.append(str(HYDRO_HEALTH))

from hydro_health.engines.CreateSeabedTerrainLayerEngine import CreateSeabedTerrainLayerEngine

if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()

    engine = CreateSeabedTerrainLayerEngine()
    # engine.run_gap_fill(r"C:\Users\aubrey.mccutchan\Documents")
    engine.process()

    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.strip_dirs().sort_stats('cumulative').print_stats(10)  
