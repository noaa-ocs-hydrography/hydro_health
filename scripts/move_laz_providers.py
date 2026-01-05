import pathlib
import shutil

from osgeo import gdal


DIGITALCOAST = pathlib.Path(r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast')
OUTPUTS =  pathlib.Path(r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast\unused_providers\LAZ')

laz_folders = [folder for folder in DIGITALCOAST.glob('**/laz') if 'unused_providers' not in str(folder)]
vrt_files = [file for file in DIGITALCOAST.glob('*.vrt')]


"""
Helper script to move all LAZ providers and their VRT file to the unused_providers/LAZ folder for future use with HH 2.0
"""

for folder in laz_folders:
    print(folder)
    provider_folder = folder.parents[0]
    # Moving provider to LAZ folder
    print(' - ', provider_folder)
    shutil.move(provider_folder, OUTPUTS)

    # Moving provider VRT to LAZ folder
    for vrt in vrt_files:
        if folder.parents[0].stem in str(vrt):
            print(' - ', vrt)
            shutil.move(vrt, OUTPUTS)
            break