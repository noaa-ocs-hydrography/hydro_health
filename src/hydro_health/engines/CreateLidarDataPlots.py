import os
import pathlib
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import rasterio
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.transform import from_origin
import fiona

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

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


class CreateLidarDataPlots(Engine):
    """Class to hold the logic for processing the Lidar Data Plots"""

    def __init__(self):
        super().__init__()
        self.sediment_types = ['Gravel', 'Sand', 'Mud', 'Clay'] 
        self.sediment_data = None

    def resample_vrt_files(self, input_dir, output_dir, x_res=0.0359/40, y_res=0.0359/40, resampling_method='bilinear')-> None:
        """
        Resample all .vrt files in the input directory and save them to the output directory."""
        target_crs = "EPSG:32617"

        creation_options = [
            "COMPRESS=DEFLATE",  # Use lossless DEFLATE compression
            "TILED=YES",         # Create a tiled TIFF
            "BIGTIFF=YES"        # Enable BigTIFF format for files > 4GB
        ]

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

    def run(self):
        """Entrypoint for processing the Lidar Data Plots"""
        self.download_sediment_data()

