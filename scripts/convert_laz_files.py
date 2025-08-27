import pdal
import json
import numpy as np

# Define the input and output filenames
input_file = 'input.laz'
output_file = 'output.tif'

# Define the PDAL pipeline as a Python dictionary
# This is the same as the JSON file from Step 2
pipeline_json = {
  "pipeline": [
    input_file,
    {
      "type": "writers.gdal",
      "gdaldriver": "GTiff",
      "resolution": 8.0,
      "output_type": "all",
      "filename": output_file
    }
  ]
}

# Create a PDAL pipeline object from the JSON dictionary
pipeline = pdal.Pipeline(json.dumps(pipeline_json))

# Execute the pipeline
try:
    count = pipeline.execute()
    print(f"Successfully processed {count} points.")
    print(f"Output GeoTIFF saved as {output_file}")
except RuntimeError as e:
    print(f"PDAL pipeline failed: {e}")