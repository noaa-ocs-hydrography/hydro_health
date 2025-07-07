import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

# Load GeoParquet
combined = gpd.read_parquet(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\prediction_tiles\BC25X26P_36_prediction_clipped_data.parquet")

print(combined.head(5))  # Display first few rows for verification
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

# Columns to rasterize
band_columns = ['sed_size_raster_100m', 'sed_type_raster_100m']
band_arrays = []

for col in band_columns:
    shapes = zip(combined.geometry, combined[col])
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
raster_output_path = r"C:\Users\aubrey.mccutchan\Desktop\output_multiband.tif"  # replace with your desired path

# Save as multiband GeoTIFF
with rasterio.open(
    raster_output_path,
    'w',
    driver='GTiff',
    height=height,
    width=width,
    count=len(band_columns),
    dtype='float32',
    crs=crs,
    transform=transform
) as dst:
    for i, band in enumerate(bands_stack, start=1):
        dst.write(band, i)
        dst.set_band_description(i, band_columns[i - 1])
