import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# Load GeoParquet
# combined = gpd.read_parquet(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Training\training_tiles\BH4QK58C_4\BH4QK58C_4_training_clipped_data.parquet")
combined = gpd.read_parquet(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Training\training_tiles\BH4SD56H_1\BH4SD56H_1_training_clipped_data.parquet")

# Verify CRS
crs = combined.crs
if crs is None:
    raise ValueError("CRS is missing from the GeoParquet.")

# Get raster bounds
bounds = combined.total_bounds  # minx, miny, maxx, maxy
resolution = 100  # meters per pixel

width = int(np.ceil((bounds[2] - bounds[0]) / resolution))
height = int(np.ceil((bounds[3] - bounds[1]) / resolution))

# Origin is upper-left corner
transform = from_origin(bounds[0], bounds[3], resolution, resolution)
columns_to_exclude = ['X', 'Y', 'geometry']
band_columns = combined.columns.drop(columns_to_exclude)
print(combined)
# print(combined['b.change.2006_2010'])  # Display first few rows of band columns for verification
# nan_count = combined['b.change.2006_2010'].isnull().sum()

# # Get the total number of values in the column
# total_count = combined['b.change.2006_2010'].size

# # Calculate the percentage
# percent_nans = (nan_count / total_count) * 100

# # Print the result formatted to two decimal places
# print(f"Percentage of NaNs in the column: {percent_nans:.2f}%")

# sub_data = combined['b.change.2006_2010'].dropna().reset_index(drop=True)
# print(sub_data)

band_arrays = []

for col in band_columns:
    shapes = zip(combined.geometry, combined[col])
    # Add this 'with' statement to temporarily allow the memory operation
    with rasterio.Env(GDAL_MEM_ENABLE_OPEN='YES'):
        band = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=np.nan,
            dtype='float32'
        )
    band_arrays.append(band)

# Stack bands into (bands, height, width)
bands_stack = np.stack(band_arrays)

# Output path for raster
raster_output_path = r"C:\Users\aubrey.mccutchan\Documents\prediction_output_multiband.tif"  # replace with your desired path

# Save as multiband GeoTIFF
with rasterio.open(
    raster_output_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=len(band_columns),
    nodata=np.nan,
    dtype='float32',
    crs=crs,
    transform=transform
) as dst:
    for i, band in enumerate(bands_stack, start=1):
        dst.write(band, i)
        dst.set_band_description(i, band_columns[i - 1])
