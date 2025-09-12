import pdal
import json
import pickle
import sys


pickle_file_path = sys.argv[1]
with open(pickle_file_path, 'rb') as picklish:
    pipeline_json = pickle.load(picklish)
print('loaded pickle:', pipeline_json)
print(type(pipeline_json))
pipeline = pdal.Pipeline(json.dumps(pipeline_json))

# Execute the pipeline
try:
    count = pipeline.execute()
    print(f"Successfully processed {count} points.")
except RuntimeError as e:
    print(f"PDAL pipeline failed: {e}")