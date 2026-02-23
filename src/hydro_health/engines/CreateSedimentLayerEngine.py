import sys
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
from hydro_health.helpers.tools import get_config_item, get_environment


OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateSedimentLayerEngine(Engine):
    """Class to hold the logic for processing the Sediment layer"""

    def __init__(self):
        super().__init__()
        self.sediment_types = ['Gravel', 'Sand', 'Mud', 'Clay'] 
        self.sediment_data = None
        self.data_path = (
            OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "DATA_PATH"))
            if get_environment() == "local"
            else pathlib.Path(get_config_item("SEDIMENT", "DATA_PATH"))
        )
        self.gpkg_path = (
            OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "GPKG_PATH"))
            if get_environment() == "local"
            else pathlib.Path(get_config_item("SEDIMENT", "GPKG_PATH"))
        )
        self.mask_path = (
            OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "MASK_PATH"))
            if get_environment() == "local"
            else pathlib.Path(get_config_item("SEDIMENT", "MASK_PATH"))
        )
        self.raster_path = (
            OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "RASTER_PATH"))
            if get_environment() == "local"
            else pathlib.Path(get_config_item("SEDIMENT", "RASTER_PATH"))
        )

    def add_sed_size_column(self):
        """
        Adds column for the sediment size in mm
        Renames Grainsze column to Size_phi to clarify size is in phi units
        """      

        self.sediment_data['sed_size'] = 2 ** -(self.sediment_data['Grainsze'])
        self.sediment_data = self.sediment_data.rename(columns={'Grainsze': 'Size_phi'})

    def add_sediment_mapping_column(self):   
        """Adds column for the integer value of each sediment time for rasterization"""    

        sediment_mapping = {
            'Gravel': 1,
            'Sand': 2,
            'Mud': 3,
            'Clay': 4
        }
        self.sediment_data['sed_int'] = self.sediment_data['sed_type'].map(sediment_mapping)  

    def convert_polys_to_raster(self, field_name, resolution=100):
        """Rasterize polygons using a field, clip with mask, and set custom resolution (e.g., 100m)."""

        print(f"Creating raster for {field_name} at {resolution} m resolution...")

        # TODO is MASK_PATH pointing to the output from the RasterMaskEngine?
        # if so, this engine should run after RasterMaskEngine
        with rasterio.open(str(self.mask_path)) as mask_src:
            bounds = mask_src.bounds
            crs = mask_src.crs

        xres, yres = resolution, resolution
        width = int((bounds.right - bounds.left) / xres)
        height = int((bounds.top - bounds.bottom) / yres)
        transform = from_origin(bounds.left, bounds.top, xres, yres)

        with fiona.open(self.gpkg_path, layer='sediment_polygons') as src:
            assert src.crs == crs.to_dict(), "CRS mismatch between mask and vector layer"

            shapes = (
                (feature["geometry"], feature["properties"][field_name])
                for feature in src if feature["properties"][field_name] is not None
            )

            nodata_val = -9999.0
            rasterized = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=nodata_val,
                dtype='float32',
                all_touched=False
            )

        with rasterio.open(self.mask_path) as mask_src:
            mask_reproj = np.empty((height, width), dtype='float32')
            rasterio.warp.reproject(
                source=rasterio.band(mask_src, 1),
                destination=mask_reproj,
                src_transform=mask_src.transform,
                src_crs=mask_src.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest
            )

        rasterized[mask_reproj == 0] = nodata_val

        file_path = self.raster_path / f"{field_name}_raster_{resolution}m.tif"
        with rasterio.open(
            file_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=1,
            dtype='float32',
            crs=crs,
            transform=transform,
            nodata=nodata_val,
            compress='lzw'
        ) as dst:
            dst.write(rasterized, 1)

        print(f"Raster saved to {file_path}")

    def correct_sed_type(self, row):
        """
        Corrects primary sediment type if sediment percerntages do not match grain size.
        :param dataframe row: Each row of the point sediment dataframe
        :return str: Primary sediment type
        """        

        # These size classification ranges are based on the Udden-Wentworth grain size chart
        if 0 < row['sed_size'] < 0.0039:
            return 'Clay'
        elif 0.0039 <= row['sed_size'] < 0.0625:
            return 'Mud'
        elif 0.0625 <= row['sed_size'] < 2:
            return 'Sand'
        elif 2 <= row['sed_size']: 
            return 'Gravel'
        else:
            return row['sed_type']    

    def create_point_layer(self):
        """Creates a point shapefile from the sediment GeoDataFrame"""    
        
        gdf = gpd.GeoDataFrame(self.sediment_data, 
                               geometry=gpd.points_from_xy(self.sediment_data['Longitude'], self.sediment_data['Latitude']))  
        gdf.set_crs(crs="EPSG:4326", inplace=True)
        gdf_reprojected = gdf.to_crs("EPSG:32617")
        
        gdf_reprojected.to_file(self.gpkg_path, layer='sediment_points', driver = 'GPKG', overwrite=True)
        self.message('Created sediment point layer.') 

    def determine_sed_types(self):
        """Determines the primary and secondarysediment types
        at each point based on type percentage"""  

        self.message('Calculating primary and secondary sediment types')
        prim_values = []
        sec_values = []

        for _, row in self.sediment_data.iterrows():
            sediments = row[self.sediment_types]
            sorted_sediments = sediments.sort_values(ascending=False).index  
            primary_sed = sorted_sediments[0]
            secondary_sed = sorted_sediments[1]
            prim_values.append(primary_sed)
            sec_values.append(secondary_sed)
        self.sediment_data['sed_type'] = prim_values  
        self.sediment_data['sec_sed'] = sec_values    

        self.sediment_data['sed_type'] = self.sediment_data.apply(self.correct_sed_type, axis=1)      

    def download_sediment_data(self):
        """Downloads the USGS sediment dataset"""     

        csv_path = self.data_path / 'US9_ONE.csv'

        # Only download if file doesn't exist
        if not csv_path.exists():
            print("Downloading sediment data...")
            try:
                response = requests.get(get_config_item('SEDIMENT', 'DATA_URL'))
                if response.status_code == 200:
                    with open(csv_path, "wb") as f:
                        f.write(response.content)
                        print("Sediment data downloaded successfully to:", csv_path)
                else:
                    print("Failed to download CSV file. Status code:", response.status_code)
            except (requests.exceptions.ConnectionError, FileNotFoundError) as e:
                print(f'Unable to download sediment CSV file. Error: {e}')
                sys.exit(1)
        
    def read_sediment_data(self):   
        """Reads and stores the data from the USGS sediment dataset CSV"""      

        csv_columns = ['Latitude', 'Longitude', 'Gravel', 'Sand', 'Mud', 'Clay', 'Grainsze']
        sediment_data_path = self.data_path / 'US9_ONE.csv'
        self.sediment_data = pd.read_csv(sediment_data_path, usecols=csv_columns)

        self.message('Filtering out rows with missing data') 
        self.message(f' - Rows before: {self.sediment_data.shape[0]}')
        self.sediment_data = self.sediment_data[~((self.sediment_data[self.sediment_types] == -99) | (self.sediment_data[self.sediment_types] == 0)).all(axis=1)]
        self.sediment_data = self.sediment_data[(self.sediment_data['Grainsze'] != -99)].reset_index(drop=True)
        self.message(f' - Rows after: {self.sediment_data.shape[0]}')

    def transform_points_to_polygons(self):
        """Polygonize the sediment points"""       

        print("Transforming sediment points to polygons...")
        gdf = gpd.read_file(self.gpkg_path, layer='sediment_points')
        coordinates_df = gdf.geometry.apply(lambda geom: geom.centroid.coords[0]).apply(pd.Series)
        coordinates_df.columns = ['Longitude', 'Latitude']
        sed_type_values = gdf['sed_int'].tolist()
        grain_size_values = gdf['sed_size'].tolist()
        
        vor = Voronoi(coordinates_df[['Longitude', 'Latitude']].values)
        polygons = []

        for point_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            polygon = Polygon([vor.vertices[i] for i in region])

            polygons.append({
                'geometry': polygon,
                'sed_type': sed_type_values[point_idx],
                'sed_size': grain_size_values[point_idx]
            })

        gdf_voronoi = gpd.GeoDataFrame(polygons, crs='EPSG:32617')
        gdf_voronoi.to_file(self.gpkg_path, layer='sediment_polygons', driver = 'GPKG', overwrite=True)   

    def run(self):
        """Entrypoint for processing the Sediment layer"""
        self.download_sediment_data()
        self.read_sediment_data()
        self.add_sed_size_column()
        self.determine_sed_types()
        self.add_sediment_mapping_column()
        self.create_point_layer()
        self.transform_points_to_polygons()
        self.convert_polys_to_raster('sed_type')
        self.convert_polys_to_raster('sed_size')
