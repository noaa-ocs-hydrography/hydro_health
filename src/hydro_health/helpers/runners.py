import pathlib
import geopandas as gpd

from hydro_health.engines.BlueTopoEngine import BlueTopoEngine
from hydro_health.engines.tiling.DigitalCoastEngine import DigitalCoastEngine
from hydro_health.engines.tiling.RasterMaskEngine import RasterMaskEngine
from hydro_health.engines.tiling.SurgeTideForecastEngine import SurgeTideForecastEngine
from hydro_health.engines.CreateTSMLayerEngine import CreateTSMLayerEngine
from hydro_health.engines.CreateSedimentLayerEngine import CreateSedimentLayerEngine
from hydro_health.engines.CreateHurricaneLayerEngine import CreateHurricaneLayerEngine
from hydro_health.helpers.tools import Param, get_ecoregion_tiles, get_environment


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


def run_bluetopo_tile_engine(tiles: gpd.GeoDataFrame, outputs:str) -> None:
    """Entry point for parallel processing of BlueTopo tiles"""

    engine = BlueTopoEngine()
    engine.run(tiles, outputs)


def run_raster_mask_engine(outputs:str) -> None:
    """Create prediction and training masks for found ecoregions"""

    engine = RasterMaskEngine()
    engine.run(outputs)


def run_digital_coast_engine(tiles: gpd.GeoDataFrame, outputs: str) -> None:
    """Entry point for parallel proccessing of Digital Coast data"""
    
    engine = DigitalCoastEngine()
    engine.run(tiles, outputs)


def run_tsm_layer_engine() -> None:
    """Entry point for parallel processing of TSM model data"""

    engine = CreateTSMLayerEngine()
    engine.run()


def run_sediment_layer_engine() -> None:
    """Entry point for parallel processing of sediment model data"""

    engine = CreateSedimentLayerEngine()
    engine.run()

def run_hurricane_layer_engine() -> None:
    """Entry point for parallel processing of sediment model data"""

    engine = CreateHurricaneLayerEngine()
    engine.run()    


def run_stofs_engine(tiles: gpd.GeoDataFrame, outputs: str) -> None:
    """Entry point for parallel processing of STOFS data"""

    engine = SurgeTideForecastEngine()
    engine.run(tiles, outputs)