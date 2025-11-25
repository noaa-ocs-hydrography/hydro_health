import pdal
import json
import pickle
import sys
import os
import sys

from multiprocessing import set_executable

set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


pickle_file_path = sys.argv[1]
return_result = bool(sys.argv[2]) if len(sys.argv) > 2 else False
with open(pickle_file_path, 'rb') as picklish:
    pipeline_json = pickle.load(picklish)
pipeline = pdal.Pipeline(json.dumps(pipeline_json))

# Execute the pipeline
try:
    count = pipeline.execute()
    print(f"Successfully processed {count} points.")
    if return_result:
        print('- Writing SRS result to pickle file')
        metadata = pipeline.metadata
        reader_meta = metadata['metadata']['readers.las']
        srs_json = reader_meta.get('srs', {}).get('json', {})
        with open(pickle_file_path, 'wb') as picklish:
            pickle.dump(srs_json, picklish)
except RuntimeError as e:
    print(f"PDAL pipeline failed: {e}")