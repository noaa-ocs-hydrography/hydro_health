import pathlib
import time
import os
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import process_bluetopo_tiles, get_ecoregion_tiles, Param, create_raster_vrt


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

def create_vrt_output(output_folder):
    ecoregion_folders = pathlib.Path(param_lookup['output_directory'].valueAsText).glob('ER*')
    for dataset in ['elevation', 'slope', 'rugosity']:
        for folder in ecoregion_folders:
            create_raster_vrt(output_folder, dataset, folder.stem)


if __name__ == '__main__':
    if os.path.exists(OUTPUTS / 'log_prints.txt'):
        now = time.time()
        os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
    param_lookup = {
        'input_directory': Param(''),
        'output_directory': Param(str(OUTPUTS)),
        'eco_regions': Param('ER_3-Florida-West;'),
        'drawn_polygon': Param(str(OUTPUTS / 'drawn_polygons.geojson'))
        # 'drawn_polygon': Param('')
    }
    
    tiles = get_ecoregion_tiles(param_lookup)
    start = time.time()
    process_bluetopo_tiles(tiles, param_lookup['output_directory'].valueAsText)
    create_vrt_output(param_lookup['output_directory'].valueAsText)
    
    end = time.time()
    print(f'Total Runtime: {end - start}') # Florida-West - 640.7945353984833 seconds or 10.67990892330806 minutes, 7.23GB, 727 folders, 1454 files
    # takes 102 seconds to verify if already downloaded
    ecoregion_folders = pathlib.Path(param_lookup['output_directory'].valueAsText).glob('ER*')
    if ecoregion_folders:
        for folder in ecoregion_folders:
            print(f'Downloaded {folder.stem} tiles:', len(next(os.walk(folder))[1]))
    else:
        print('No tiles downloaded')
    print('done')