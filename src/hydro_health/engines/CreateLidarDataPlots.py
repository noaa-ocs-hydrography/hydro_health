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

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


class CreateLidarDataPlots(Engine):
    """Class to hold the logic for processing the Lidar Data Plots"""

    def __init__(self):
        super().__init__()
        self.sediment_types = ['Gravel', 'Sand', 'Mud', 'Clay'] 
        self.sediment_data = None

    def run(self):
        """Entrypoint for processing the Lidar Data Plots"""
        self.download_sediment_data()
        self.read_sediment_data()
        self.add_sed_size_column()
        self.determine_sed_types()
        self.add_sediment_mapping_column()
        self.create_point_layer()
        self.transform_points_to_polygons()
        self.convert_polys_to_raster('sed_type')
        self.convert_polys_to_raster('sed_size')
