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
import matplotlib.pyplot as plt
import glob
from osgeo import gdal, osr
import dask
from dask.distributed import Client, LocalCluster

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

# os.environ["GDAL_CACHEMAX"] = "60048"

# Corrected the function signature to accept all arguments passed by Dask
def process_single_vrt(vrt_file, output_dir, x_res, y_res, resampling_method, creation_options, warpOptions, multithread):
    """
    Processes a single VRT file. This function will be executed by a Dask worker.
    """
    try:
        base_name = os.path.basename(vrt_file)
        output_filename = os.path.join(output_dir, base_name.replace(".vrt", "_resampled.tif"))

        # --- KEY CHANGE ---
        # Check if the output file already exists. If so, skip processing.
        if os.path.exists(output_filename):
            skip_message = f"Output file {output_filename} already exists. Skipping."
            print(skip_message)
            return skip_message

        print(f"Processing {base_name}...")

        gdal.Warp(
            output_filename,
            vrt_file,
            xRes=x_res,
            yRes=y_res,
            resampleAlg=resampling_method,
            creationOptions=creation_options,
            warpOptions=warpOptions,   
            multithread=multithread     
        )

        success_message = f"Successfully resampled and saved to {output_filename}"
        print(success_message)
        return success_message
    except Exception as e:
        error_message = f"Error processing {vrt_file}: {e}"
        print(error_message)
        return error_message

class CreateLidarDataPlotsEngine():
    """Class to hold the logic for processing the Lidar Data Plots"""

    def __init__(self):
        super().__init__()
        # self.sediment_data = None

    def resample_vrt_files(self, x_res=0.0359, y_res=0.0359, resampling_method='nearest')-> None:
        """
        Resample all .vrt files in the input directory in parallel using Dask
        and save them to the output directory.

        param x_res: Horizontal resolution for resampling
        param y_res: Vertical resolution for resampling
        param resampling_method: Resampling method to use (e.g., 'bilinear', 'cubic')
        """

        input_dir = get_config_item("LIDAR_PLOTS", "INPUT_VRTS")
        output_dir = get_config_item("LIDAR_PLOTS", "RESAMPLED_VRTS")

        os.makedirs(output_dir, exist_ok=True)

        creation_options = [
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=YES"
        ]

        vrt_files = glob.glob(os.path.join(input_dir, "*.vrt"))

        if not vrt_files:
            print(f"No .vrt files found in {input_dir}")
            return

        print(f"Found {len(vrt_files)} VRT files to process.")

        tasks = []

        for vrt_file in vrt_files:
            task = dask.delayed(process_single_vrt)(
                vrt_file,
                output_dir,
                x_res,
                y_res,
                resampling_method,
                creation_options,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                multithread=True
            )
            tasks.append(task)

        print("Starting parallel processing with Dask...")
        results = dask.compute(*tasks)
        print("All Dask tasks have been completed.")

    def run(self):
        """Entrypoint for processing the Lidar Data Plots"""
        with LocalCluster() as cluster, Client(cluster) as client:
            print(f"Dask dashboard link: {client.dashboard_link}")
            self.resample_vrt_files()