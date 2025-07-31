import pathlib
import time
import os
HH_MODEL = pathlib.Path(__file__).parents[2]

import sys
sys.path.append(str(HH_MODEL))

from hydro_health.helpers.runners import run_validate_extents_engine


def run() -> None: 
    run_validate_extents_engine()


if __name__ == "__main__":
    run()