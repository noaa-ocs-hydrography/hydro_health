import pathlib
import geopandas as gpd

from hydro_health.engines.BlueTopoEngine import BlueTopoEngine
from hydro_health.engines.BlueTopoS3Engine import BlueTopoS3Engine
from hydro_health.engines.tiling.DigitalCoastEngine import DigitalCoastEngine
from hydro_health.engines.tiling.DigitalCoastS3Engine import DigitalCoastS3Engine
from hydro_health.engines.MetadataEngine import MetadataEngine
from hydro_health.engines.MetadataS3Engine import MetadataS3Engine
from hydro_health.engines.tiling.LAZConversionEngine import LAZConversionEngine
from hydro_health.engines.tiling.RasterMaskEngine import RasterMaskEngine
from hydro_health.engines.tiling.SurgeTideForecastEngine import SurgeTideForecastEngine
from hydro_health.engines.CreateTSMLayerEngine import CreateTSMLayerEngine
from hydro_health.engines.CreateSedimentLayerEngine import CreateSedimentLayerEngine
from hydro_health.engines.CreateHurricaneLayerEngine import CreateHurricaneLayerEngine
from hydro_health.engines.RasterVRTEngine import RasterVRTEngine
from hydro_health.engines.RasterVRTS3Engine import RasterVRTS3Engine

from hydro_health.helpers.tools import get_ecoregion_folders


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


def run_bluetopo_tile_engine(tiles: gpd.GeoDataFrame, param_lookup: dict[dict]) -> None:
    """Entry point for parallel processing of BlueTopo tiles"""

    if param_lookup['env'] in ['local', 'remote']:
        run_bluetopo_tile_engine_local(tiles, param_lookup)
    else:
        run_bluetopo_tile_engine_s3(tiles, param_lookup)


def run_bluetopo_tile_engine_local(tiles: gpd.GeoDataFrame, param_lookup: dict[dict]) -> None:
    """Entry point for parallel processing of BlueTopo tiles"""

    engine = BlueTopoEngine(param_lookup)
    engine.run(tiles)


def run_bluetopo_tile_engine_s3(tiles: gpd.GeoDataFrame,  param_lookup: dict[dict]) -> None:
    """Entry point for parallel processing of BlueTopo tiles on AWS VM"""

    engine = BlueTopoS3Engine(param_lookup)
    engine.run(tiles)


def run_raster_mask_engine(outputs:str) -> None:
    """Create prediction and training masks for found ecoregions"""

    engine = RasterMaskEngine()
    engine.run(outputs)


# def run_raster_mask_local(outputs:str) -> None:
#     """Create prediction and training masks for found ecoregions"""

#     engine = RasterMaskEngine()
#     engine.run(outputs)


# def run_raster_mask_s3(outputs:str) -> None:
#     """Create prediction and training masks for found ecoregions"""

#     engine = RasterMaskS3Engine()
#     engine.run(outputs)


def run_digital_coast_engine(tiles: gpd.GeoDataFrame, param_lookup: dict[dict]) -> None:
    """Entry point for parallel processing of Digital Coast data"""

    if param_lookup['env'] in ['local', 'remote']:
        run_digital_coast_engine_local(tiles, param_lookup)
    else:
        run_digital_coast_engine_s3(tiles, param_lookup)


def run_digital_coast_engine_local(tiles: gpd.GeoDataFrame, param_lookup: dict[dict]) -> None:
    """Entry point for parallel proccessing of Digital Coast data"""
    
    engine = DigitalCoastEngine(param_lookup)
    engine.run(tiles)


def run_digital_coast_engine_s3(tiles: gpd.GeoDataFrame, param_lookup: dict[dict]) -> None:
    """Entry point for parallel proccessing of Digital Coast data on AWS VM"""
    
    engine = DigitalCoastS3Engine(param_lookup)
    engine.run(tiles)


def run_laz_conversion_engine(tiles: gpd.GeoDataFrame, outputs: str) -> None:
    """Entry point for converting all LAZ files to TIF"""

    engine = LAZConversionEngine()
    engine.run(tiles, outputs)


def run_metadata_engine(tiles: gpd.GeoDataFrame, param_lookup: dict[dict]) -> None:
    """Entry point for parallel processing of provider metadata"""

    if param_lookup['env'] in ['local', 'remote']:
        engine = MetadataEngine()
    else:
        engine = MetadataS3Engine()
    outputs = param_lookup['output_directory'].valueAsText
    engine.run(tiles, outputs)


def run_raster_vrt_engine(param_lookup: dict[str], skip_existing=False) -> None:
    """Entry point for building VRT files for BlueTopo and Digital Coast data"""
    
    if param_lookup['env'] in ['local', 'remote']:
        engine = RasterVRTEngine(param_lookup)
    else:
        engine = RasterVRTS3Engine(param_lookup)
    for ecoregion in get_ecoregion_folders(param_lookup):
        # for dataset in ['elevation', 'slope', 'rugosity', 'uncertainty', 'catzoc_score_all', 'catzoc_score_latest', 'catzoc_decay_all', 'catzoc_decay_latest']:
        for dataset in ['elevation', 'slope', 'rugosity', 'uncertainty']:
            print(f'Building {ecoregion} - {dataset} VRT file')
            engine.run(param_lookup['output_directory'].valueAsText, dataset, ecoregion, 'BlueTopo', skip_existing=skip_existing)
        print(f'Building {ecoregion} - DigitalCoast VRT files')
        engine.run(param_lookup['output_directory'].valueAsText, 'NCMP', ecoregion, 'DigitalCoast', skip_existing=skip_existing)


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