import os
import json
import rasterio
import numpy as np
from shapely.geometry import Polygon

def create_valid_data_footprint(root_directory: str):
    """Iterates through subfolders to create a GeoJSON footprint of valid data cells from TIF files.
    
    Skips subfolders where 'feature.json' already exists.

    param str root_directory: The path to the root directory containing subfolders with TIF files.
    """
    for subdir, _, files in os.walk(root_directory):
        
        # --- ADDED CHECK ---
        # Define the potential output path
        output_path = os.path.join(subdir, 'feature.json')
        
        # Check if the JSON file already exists
        if os.path.exists(output_path):
            print(f"Skipping {subdir}: feature.json already exists.")
            continue  # Move to the next subdirectory
        # --- END ADDED CHECK ---

        tif_files = [f for f in files if f.lower().endswith('.tif')]
        if not tif_files:
            continue

        features = []
        for tif_file in tif_files:
            tif_path = os.path.join(subdir, tif_file)
            try:
                with rasterio.open(tif_path) as src:
                    transform = src.transform
                    nodata_val = src.nodata
                    array = src.read(1)

                    if nodata_val is not None:
                        if np.isnan(nodata_val):
                            valid_cells = np.argwhere(~np.isnan(array))
                        else:
                            valid_cells = np.argwhere(array != nodata_val)
                    else:
                        # If no nodata value defined, assume all cells are valid
                        valid_cells = np.argwhere(np.ones_like(array, dtype=bool))

                    for row, col in valid_cells:
                        # Get the four corners of the pixel
                        top_left = transform * (col, row)
                        top_right = transform * (col + 1, row)
                        bottom_right = transform * (col + 1, row + 1)
                        bottom_left = transform * (col, row + 1)

                        poly = Polygon([top_left, top_right, bottom_right, bottom_left])
                        
                        # Manually create the GeoJSON Feature dictionary
                        feature = {
                            "type": "Feature",
                            "geometry": poly.__geo_interface__,
                            "properties": {}
                        }
                        features.append(feature)
            except rasterio.errors.RasterioIOError as e:
                print(f"Error reading {tif_path}: {e}. Skipping this file.")
                continue

        if features:
            # Manually create the GeoJSON FeatureCollection dictionary
            feature_collection = {
                "type": "FeatureCollection",
                "features": features
            }
            # 'output_path' was already defined above
            with open(output_path, 'w') as f:
                json.dump(feature_collection, f)
            print(f"Generated feature.json for {subdir}")


if __name__ == '__main__':
    main_data_folder = r"\\nos.noaa\ocs\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\Digital_Cost_Manual_Downloads"
    create_valid_data_footprint(main_data_folder)