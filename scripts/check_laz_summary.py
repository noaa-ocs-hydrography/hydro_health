import pdal
import json
import pathlib

filename = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\scripts\test_laz_files\20191218_585000e_3352500n.copc.laz"



pipeline = pdal.Pipeline(json.dumps([
    {
        "type": "readers.las",
        "filename": filename,
        "count": 0
    }
]))

pipeline.execute()
metadata = pipeline.metadata
reader_meta = metadata['metadata']['readers.las']

# We look for the name of the cage (SRS)
srs_json = reader_meta.get('srs', {})
print("--- THE SRS REPORT ---")
print(json.dumps(srs_json, indent=2))

# Check the units!
if 'units' in srs_json.get('horizontal', {}):
    print(f"\nUNIT DETECTED: {srs_json['horizontal']['units']}")
elif 'wkt' in reader_meta and 'DEGREE' in reader_meta['wkt']:
    print("\nI SEE 'DEGREE' IN THE TEXT! DANGER!")
else:
    print("\nI see nothing! It might be raw!")