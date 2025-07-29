import pathlib
import time
import os
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import process_digital_coast_files, get_ecoregion_tiles, Param, create_raster_vrt


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs' 


if __name__ == '__main__':
    # if os.path.exists(OUTPUTS / 'log_prints.txt'):
    #     now = time.time()
    #     os.rename(OUTPUTS / 'log_prints.txt', OUTPUTS / f'log_prints_{now}.txt')
    param_lookup = {
        'input_directory': Param(''),
        'output_directory': Param(str(OUTPUTS)),
        'eco_regions': Param('ER_3-Florida-West;'),
        'drawn_polygon': Param(str(OUTPUTS / 'drawn_polygons.geojson'))
        # 'drawn_polygon': Param('')
    }

    log_file_path = pathlib.Path(param_lookup['output_directory'].valueAsText) / 'log_prints.txt'
    # Delete log file on startup
    if log_file_path.exists():
        log_file_path.unlink()
    open(log_file_path, 'a').close()

    
    tiles = get_ecoregion_tiles(param_lookup)
    print(f'Selected tiles: {tiles.shape[0]}')
    start = time.time()
    process_digital_coast_files(tiles, param_lookup['output_directory'].valueAsText)
    create_raster_vrt(param_lookup['output_directory'].valueAsText, 'NCMP', 'DigitalCoast')

    digital_coast_data = pathlib.Path(param_lookup['output_directory'].valueAsText) / 'DigitalCoast'
    if digital_coast_data.exists():
        digital_coast_folders = digital_coast_data.rglob('*')
        data_providers = []
        for path in digital_coast_folders:
            if path.is_dir():
                data_providers.append(path)
        print(f'Downloaded digital coast providers: {len(data_providers)}')
    else:
        print('No data downloaded')
        
    end = time.time()
    print(f'Total Runtime: {end - start}') # Florida-West - 640.7945353984833 seconds or 10.67990892330806 minutes, 7.23GB, 727 folders, 1454 files
    print('done')