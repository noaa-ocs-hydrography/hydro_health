import pathlib
import sys

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.engines.RasterMaskChecker import RasterMaskChecker


mapper = RasterMaskChecker()
mapper.create_interactive_map()