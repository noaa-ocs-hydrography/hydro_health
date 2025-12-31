import pathlib
import time
import os
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import get_ecoregion_tiles, Param
from hydro_health.helpers.runners import run_bluetopo_tile_engine, run_raster_vrt_engine


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


if __name__ == '__main__':
    if os.path.exists(OUTPUTS / 'log_prints.txt'):
        now = time.time()
        os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
    param_lookup = {
        'input_directory': Param(''),
        'output_directory': Param(str(OUTPUTS)),
        'eco_regions': Param('ER_3-Florida-West;'),
        'drawn_polygon': Param(str(INPUTS / 'drawn_polygons.geojson')),
        # 'drawn_polygon': Param(''),
        'env': 'local'
    }
    
    tiles = get_ecoregion_tiles(param_lookup)
    print(f'Selected tiles: {tiles.shape[0]}')
    start = time.time()
    run_bluetopo_tile_engine(tiles, param_lookup)
    run_raster_vrt_engine(param_lookup, skip_existing=False)
    
    end = time.time()
    print(f'Total Runtime: {end - start}') # Florida-West - 640.7945353984833 seconds or 10.67990892330806 minutes, 7.23GB, 727 folders, 1454 files
    found_bluetopo_files = []
    for tiff_file in pathlib.Path(param_lookup['output_directory'].valueAsText).rglob('*.tiff'):
        if 'BlueTopo' in str(tiff_file):
            found_bluetopo_files.append(tiff_file)
    if found_bluetopo_files:
        print('BlueTopo files:', len(found_bluetopo_files))
    else:
        print('No tiles downloaded')
    print('done')