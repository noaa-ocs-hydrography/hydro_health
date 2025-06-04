import pathlib
import time
import os
from osgeo import gdal, osr, ogr
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import (
    process_bluetopo_tiles,
    process_digital_coast_files,
    get_ecoregion_tiles,
    get_ecoregion_folders,
    Param,
    create_raster_vrts,
    process_create_masks,
    grid_digitalcoast_files
)


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


if __name__ == '__main__':
    if os.path.exists(OUTPUTS / 'log_prints.txt'):
        now = time.time()
        os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
    param_lookup = {
        'input_directory': Param(''),
        'output_directory': Param(str(OUTPUTS)),
        'eco_regions': Param(''),
        'drawn_polygon': Param(str(OUTPUTS / 'drawn_polygons.geojson'))
        # 'drawn_polygon': Param('')
    }
    
    start = time.time()
    tiles = get_ecoregion_tiles(param_lookup)
    print(f'Selected tiles: {tiles.shape[0]}')
    process_bluetopo_tiles(tiles, param_lookup['output_directory'].valueAsText)
    process_digital_coast_files(tiles, param_lookup['output_directory'].valueAsText)
    
    for ecoregion in get_ecoregion_folders(param_lookup):
        for dataset in ['elevation', 'slope', 'rugosity', 'uncertainty']:
            print(f'Building {ecoregion} - {dataset} VRT file')
            create_raster_vrts(param_lookup['output_directory'].valueAsText, dataset, ecoregion, 'BlueTopo')
        create_raster_vrts(param_lookup['output_directory'].valueAsText, 'NCMP', ecoregion, 'DigitalCoast')
    
    process_create_masks(param_lookup['output_directory'].valueAsText)
    grid_digitalcoast_files(param_lookup['output_directory'].valueAsText)

    end = time.time()
    print(f'Total Runtime: {end - start}') # Florida-West - 640.7945353984833 seconds or 10.67990892330806 minutes, 7.23GB, 727 folders, 1454 files
    # takes 102 seconds to verify if already downloaded
    
    for ecoregion in pathlib.Path(param_lookup['output_directory'].valueAsText).glob('ER_*'):
        if pathlib.Path(ecoregion / 'BlueTopo').is_dir():
            print(f'{ecoregion.stem} BlueTopo tiles:', len(next(os.walk(os.path.join(param_lookup['output_directory'].valueAsText, ecoregion, 'BlueTopo')))[1]))
        else:
            print('No tiles downloaded')
    print('done')
