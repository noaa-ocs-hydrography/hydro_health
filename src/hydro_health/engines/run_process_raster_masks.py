import pathlib
import time
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))
from hydro_health.helpers.tools import process_create_masks


OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


if __name__ == "__main__":
    start = time.time()
    process_create_masks(str(OUTPUTS))
    print('Finished prediction masks')
    print('Run time:', time.time() - start)
    print('Done')