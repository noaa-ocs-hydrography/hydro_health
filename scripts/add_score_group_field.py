import geopandas as gpd
import numpy as np
import pathlib


OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


# --- Configuration ---
gpkg_path = OUTPUTS / 'ER3_Grids_WGS84.gpkg'  # Path to your Geopackage
layer_name = "median_present_survey_score"      # Name of the layer within the GPKG
output_path = "your_data_updated.gpkg"

def categorize_score(score):
    """Logic to map median_score to score_group integers"""
    if score >= 90:
        return 1
    elif score >= 70:
        return 2
    elif score >= 50:
        return 3
    elif score >= 30:
        return 4
    else:
        return 5

# 1. Load the layer
gdf = gpd.read_file(gpkg_path, layer=layer_name)

# 2. Create and populate the 'score_group' field
# We use .apply() to iterate the logic across the 'median_score' column
gdf['score_group'] = gdf['median_score'].apply(categorize_score)

# 3. Ensure the data type is a short integer
# (In Python/Pandas, 'int32' or 'int16' maps to short integer in GIS)
gdf['score_group'] = gdf['score_group'].astype('int16')

# 4. Save the result back to a GeoPackage
# Note: Using 'pyogrio' or 'fiona' engine preserves GPKG standards well
gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")

print(f"Successfully processed {len(gdf)} rows and saved to {output_path}")