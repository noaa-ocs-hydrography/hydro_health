import json
import sys
import numpy as np
import pathlib
import subprocess
import pickle


INPUTS = pathlib.Path(__file__).parents[1] / "inputs"
# OUTPUTS = pathlib.Path(__file__).parents[1] / "outputs"
OUTPUTS = pathlib.Path(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\Digital_Cost_Manual_Downloads\laz_issues\laz\geoid18")


def convert_laz_files():
    print('starting')
    print(OUTPUTS)
    for input_file in OUTPUTS.rglob('*.laz'):
        output_file = input_file.parents[0] / f'{input_file.stem}.tif'

        # Define the PDAL pipeline as a Python dictionary
        # This is the same as the JSON file from Step 2
        pipeline_json = {
            "pipeline": [
                {"type": "readers.las", "filename": str(input_file)},
                {
                    "type": "writers.gdal",
                    "gdaldriver": "GTiff",
                    "resolution": 8.0,
                    "dimension": "Z",
                    "output_type": "max",
                    "filename": str(output_file),
                },
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
    convert_laz_files()
