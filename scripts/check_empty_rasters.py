import pathlib

from osgeo import gdal


DIGITALCOAST = pathlib.Path(r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast')


tif_files = [tif_file for tif_file in DIGITALCOAST.rglob('*.tif') if 'unused_providers' not in str(tif_file)]


"""Helper script to check if any current TIF file is empty or has an error"""

empty_files = []
tif_file_length = len(tif_files)
for i, file in enumerate(tif_files):
    print(f'{i+1} of {tif_file_length}')
    try:
        tif_ds = gdal.Open(str(file))
        if tif_ds is None:
            print(f'empty file: {file}')
            empty_files.append(file)
            continue
        if tif_ds.RasterXSize == 0 or tif_ds.RasterYSize == 0:
            print(f'empty file XY: {file}')
            empty_files.append(file)

        tif_ds = None
    except Exception as e:
        print(f'File error: {file}')
        empty_files.append(file)

