# import rasterio
import matplotlib.pyplot as plt
import numpy as np
import os
# import contextily as ctx
import cProfile
import io
import pstats

import os
import glob
from osgeo import gdal, osr

os.environ["GDAL_CACHEMAX"] = "64"

# Define input and output directories
input_dir = r"\\nos.noaa\ocs\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast"
output_dir = r"\\nos.noaa\ocs\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast_resampled"


# Define resampling parameters
target_crs = "EPSG:32617"
x_res = 0.0359/40
y_res = 0.0359/40
resampling_method = "bilinear"

# Define GDAL creation options for a chunked/tiled output
# This forces GDAL to process the data in manageable chunks.
creation_options = [
    "COMPRESS=DEFLATE",  # Use lossless DEFLATE compression
    "TILED=YES",         # Create a tiled TIFF
    "BIGTIFF=YES"        # Enable BigTIFF format for files > 4GB
]

# Find all .vrt files in the input directory
vrt_files = glob.glob(os.path.join(input_dir, "*.vrt"))
print(vrt_files)

if not vrt_files:
    print(f"No .vrt files found in {input_dir}")
else:
    for vrt_file in vrt_files:
        # Get the base name of the file to create the output filename
        base_name = os.path.basename(vrt_file)
        output_filename = os.path.join(output_dir, base_name.replace(".vrt", "_resampled.tif"))

        print(f"Processing {base_name}...")

        ds = gdal.Open(vrt_file)
        srs = osr.SpatialReference(wkt=ds.GetProjection())
        print(srs.GetAttrValue("AUTHORITY", 1))

        # Use gdal.Warp to resample the VRT file with chunked processing enabled
        try:
            gdal.Warp(
                output_filename,
                vrt_file,
                xRes=x_res,
                yRes=y_res,
                resampleAlg=resampling_method,
                creationOptions=creation_options
            )
            print(f"Successfully resampled and saved to {output_filename}")
        except Exception as e:
            print(f"Error processing {vrt_file}: {e}")

print("All .vrt files have been processed.")

# vrt_file_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\DigitalCoast\mosaic_NCMP_6326_USACE_USACE_NCMP_2024.vrt"

# def plot_raster_with_background():
#     try:
#         with rasterio.open(vrt_file_path) as src:
#             downsampled_data = src.read(
#                 out_shape=(src.count, 500, 500),
#                 resampling=rasterio.enums.Resampling.bilinear
#             )

#             out_transform = src.transform * src.transform.scale(
#                 (src.width / downsampled_data.shape[2]),
#                 (src.height / downsampled_data.shape[1])
#             )
            
#             fig, ax = plt.subplots(figsize=(10, 8))

#             raster_plot = ax.imshow(
#                 downsampled_data[0],
#                 cmap='viridis',
#                 extent=rasterio.plot.plotting_extent(src, out_transform),
#             )

#             ctx.add_basemap(ax, crs=src.crs)

#             ax.set_title(f"Downsampled Plot of VRT: {os.path.basename(vrt_file_path)}")
#             ax.set_xlabel('X-axis (projected coordinates)')
#             ax.set_ylabel('Y-axis (projected coordinates)')
#             plt.colorbar(raster_plot, ax=ax, label='Pixel Value')
#             plt.show()

#     except rasterio.errors.RasterioIOError as e:
#         print(f"Error opening the VRT file: {e}")
#         print("Please check that the VRT file path is correct and the source files it points to exist.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     pr = cProfile.Profile()
#     pr.enable()
#     plot_raster_with_background()
#     pr.disable()
#     s = io.StringIO()
#     sortby = 'cumulative'
#     ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
#     ps.print_stats()
#     print("-" * 50)
#     print("Profiling Report:")
#     print(s.getvalue())
