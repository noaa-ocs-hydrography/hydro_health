import pdal
import json
import pickle
import sys
import os
import sys

from multiprocessing import set_executable

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


pickle_file_path = sys.argv[1]
with open(pickle_file_path, 'rb') as picklish:
    pipeline_json = pickle.load(picklish)
pipeline = pdal.Pipeline(json.dumps(pipeline_json))

# Execute the pipeline
try:
    count = pipeline.execute()
    print(f"Successfully processed {count} points.")
except RuntimeError as e:
    print(f"PDAL pipeline failed: {e}")