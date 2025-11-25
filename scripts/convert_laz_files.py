import json
import sys
import numpy as np
import pathlib
import subprocess
import pickle
import rasterio   # missing rasterio pdal environment


INPUTS = pathlib.Path(__file__).parents[1] / "inputs"
OUTPUTS = pathlib.Path(__file__).parents[1] / "outputs"


def convert_laz_file():
    print('starting')
    print(OUTPUTS)
    input_file = pathlib.Path(r'C:\Users\Stephen.Patterson\Data\Repos\hydro_health\scripts\test_laz_files\20191218_585000e_3352500n.copc.laz')
          
    output_file = input_file.parents[0] / f'{input_file.stem}.tif'

    # Define the PDAL pipeline as a Python dictionary
    # This is the same as the JSON file from Step 2
    pipeline_json = {
        "pipeline": [
            # {"type": "readers.las", "filename": str(input_file)},
            # {
            #     "type": "writers.gdal",
            #     "gdaldriver": "GTiff",
            #     "resolution": 8.0,
            #     "dimension": "Z",
            #     "output_type": "max",
            #     "filename": str(output_file),
            # },
            {
                "type": "readers.las",
                "filename": str(input_file)
            },
            {
                "type": "writers.gdal",
                "filename": str(output_file),
                "resolution": 8,  # <--- The 8 meter smash!
                "output_type": "mean", # We take the average height. SMASH!
                "radius": 11 # Reach out to find points if the pixel is empty!
            }
        ]
    }
    output_pickle_file = input_file.parents[0] / f'{input_file.stem}.pkl'
    print('pickling:', output_pickle_file)
    with open(output_pickle_file, 'wb') as picklish:
        pickle.dump(pipeline_json, picklish)
    print('Calling pdal script')
    try:
        subprocess.call(
            [
                "conda",
                "run",
                "-p",
                r"C:\Users\Stephen.Patterson\AppData\Local\ESRI\conda\envs\pdal-workshop",
                "python",
                r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\scripts\pdal_separate_script.py",
                str(output_pickle_file),
            ]
        )
        output_pickle_file.unlink()
    except Exception as e:
        print(f'Failure: {e}')


if __name__ == "__main__":
    convert_laz_file()
