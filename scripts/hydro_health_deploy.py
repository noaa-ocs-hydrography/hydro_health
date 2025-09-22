"""Script for migrating the Hydro Health library to the N: drive"""

import os
import sys
import shutil
import pathlib


CODE_FOLDER = pathlib.Path(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Code\hydro_health")
INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
HYDRO_HEALTH = pathlib.Path(__file__).parents[1] / 'src' / 'hydro_health'
SRC_FOLDER = HYDRO_HEALTH.parents[1]


def check_vpn(path=None):
    spacer = '#######################################################################'
    if not os.path.exists(CODE_FOLDER):
        print(f'{spacer}\n\nError - Unable to deploy to N: drive.  Check VPN connection.\n\n{spacer}')
        sys.exit(1)


def clear_folder(folder):
    """Recursively clear contents of a folder"""
    
    for path in folder.iterdir():
        if 'Aubrey' in path.stem:
            continue
        if path.is_file():
            path.unlink()
        else:
            clear_folder(path)


def deploy_hydro_health_to_working():
    # Minimum files needed
    # inputs
    #   /lookups/*
    #   /Master_Grids.gpkg
    # src/hydro_health
    #   /engines/[^run_]*
    #   *Tool.py
    #   *Toolbox.pys
    #   __init__.py
    # README.md

    clear_folder(CODE_FOLDER)
    CODE_FOLDER.mkdir(parents=True, exist_ok=True)


    # inputs folder
    shutil.copytree(INPUTS / 'lookups', CODE_FOLDER / 'inputs' / 'lookups', dirs_exist_ok=True)
    shutil.copytree(INPUTS / 'run_configs', CODE_FOLDER / 'inputs' / 'run_configs', dirs_exist_ok=True)
    shutil.copy2(INPUTS / 'Master_Grids.gpkg', CODE_FOLDER / 'inputs' / 'Master_Grids.gpkg')

    # src/hydro_health 
    shutil.copytree(HYDRO_HEALTH / 'engines', 
                    CODE_FOLDER / 'src' / 'hydro_health' / 'engines', 
                    dirs_exist_ok=True,)
                    # ignore=shutil.ignore_patterns('run*.py'))
    for tool in HYDRO_HEALTH.glob('*Tool*.py'):
        shutil.copy2(tool, CODE_FOLDER / 'src' / 'hydro_health' / tool.name)

    for toolbox in HYDRO_HEALTH.glob('*Tool*.pyt'):
        shutil.copy2(toolbox, CODE_FOLDER / 'src' / 'hydro_health' / toolbox.name)

    shutil.copytree(HYDRO_HEALTH / 'helpers', CODE_FOLDER / 'src' / 'hydro_health' / 'helpers', dirs_exist_ok=True)
        
    shutil.copy2(HYDRO_HEALTH / '__init__.py', CODE_FOLDER / 'src' / 'hydro_health' / '__init__.py')


    # pydro specific files
    shutil.copy2(SRC_FOLDER / 'README.md', CODE_FOLDER.parents[0] / 'README.md')


if __name__ == "__main__":
    check_vpn()
    deploy_hydro_health_to_working()
    print('Done')
    