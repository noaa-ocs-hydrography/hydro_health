import fiona
import sys
import pathlib
import requests
import rasterio
import fsspec
import shutil
import tempfile
import os
import boto3
import pandas as pd
import geopandas as gpd
import numpy as np

from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from rasterio.features import rasterize
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from urllib.parse import urlparse
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_environment

OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class CreateSedimentLayerEngine(Engine):
    """Class to hold the logic for processing the Sediment layer"""

    def __init__(self):
        super().__init__()
        self.sediment_types = ['Gravel', 'Sand', 'Mud', 'Clay'] 
        self.sediment_data = None
        self.sediment_prefix = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/ER_3/{get_config_item('SEDIMENT', 'SUBFOLDER')}"
        if get_environment() in ["local", "remote"]:
            self.data_path = OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "DATA_PATH"))
            self.gpkg_path = OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "GPKG_PATH"))
            self.mask_path = OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "MASK_PATH"))
            self.raster_path = OUTPUTS / pathlib.Path(get_config_item("SEDIMENT", "RASTER_PATH"))
        else:
            self.data_path = self.sediment_prefix
            self.gpkg_path = f"{self.sediment_prefix}/sediment_data.gpkg"
            self.mask_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/ER_3/{get_config_item('SEDIMENT', 'MASK_PATH')}"
            self.raster_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/ER_3/{get_config_item('SEDIMENT', 'RASTER_PATH')}"

    def _modify_gpkg(self, gdf: gpd.GeoDataFrame, layer_name: str, mode: str='w') -> None:
        """Handles the S3 round-trip for GeoPackages"""

        if get_environment() in ['local', 'remote']:
            gdf.to_file(self.gpkg_path, layer=layer_name, driver='GPKG', engine='pyogrio', mode=mode)
            return

        p = urlparse(self.gpkg_path)
        bucket, key = p.netloc, p.path.lstrip('/')
        s3 = boto3.client('s3')

        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Download existing gpkg if appending
            if mode == 'a':
                try:
                    s3.download_file(bucket, key, tmp_path)
                except Exception:
                    mode = 'w'

            gdf.to_file(tmp_path, layer=layer_name, driver='GPKG', engine='pyogrio', mode=mode)
            s3.upload_file(tmp_path, bucket, key)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def add_sed_size_column(self) -> None:
        """
        Adds column for the sediment size in mm
        Renames Grainsze column to Size_phi to clarify size is in phi units
        """      

        self.sediment_data['sed_size'] = 2 ** -(self.sediment_data['Grainsze'])
        self.sediment_data = self.sediment_data.rename(columns={'Grainsze': 'Size_phi'})

    def add_sediment_mapping_column(self) -> None:   
        """Adds column for the integer value of each sediment time for rasterization"""    

        sediment_mapping = {
            'Gravel': 1,
            'Sand': 2,
            'Mud': 3,
            'Clay': 4
        }
        self.sediment_data['sed_int'] = self.sediment_data['sed_type'].map(sediment_mapping)  

    def convert_polys_to_raster(self, field_name, resolution=100) -> None:
        """Rasterize polygons using a field, clip with mask, and set custom resolution."""

        print(f"Creating raster for {field_name} at {resolution} m resolution...")

        # Use the requested Master_Grids GeoPackage for bounds and valid area
        eco_path = "/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/Master_Grids.gpkg"
        print(f"Reading Enhanced_EcoRegions from {eco_path} for valid area mask...")
        eco_gdf = gpd.read_file(eco_path, layer='Enhanced_EcoRegions')
        
        # Ensure it matches the EPSG:6350 used for sediment polygons
        target_crs = "EPSG:6350"
        if eco_gdf.crs != target_crs:
            eco_gdf = eco_gdf.to_crs(target_crs)
            
        bounds = eco_gdf.total_bounds
        crs = eco_gdf.crs

        xres, yres = resolution, resolution
        width = int((bounds[2] - bounds[0]) / xres)
        height = int((bounds[3] - bounds[1]) / yres)
        transform = from_origin(bounds[0], bounds[3], xres, yres)

        # 1. Rasterize sediment polygons
        with fiona.open(self.gpkg_path, layer='sediment_polygons') as src:
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

        # 2. Rasterize Enhanced_EcoRegions to serve as the valid area mask
        eco_shapes = [(geom, 1) for geom in eco_gdf.geometry]
        valid_area_mask = rasterize(
            shapes=eco_shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=False
        )

        # Apply Enhanced_EcoRegions mask to nullify non-valid areas
        rasterized[valid_area_mask == 0] = nodata_val

        filename = f"{field_name}_raster_{resolution}m.tif"

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with rasterio.open(
                tmp_path,
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

            if get_environment() in ['local', 'remote']:
                final_local_path = self.raster_path / filename
                final_local_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tmp_path, str(final_local_path))
                print(f"Raster saved locally to {final_local_path}")
                
                # Upload the sand_mud_mask to S3 anyway
                if field_name == 'sand_mud_mask':
                    s3 = boto3.client('s3')
                    bucket = "ocs-dev-csdl-hydrohealth"
                    key = f"low_res/{filename}"
                    print(f"Uploading {field_name} to s3://{bucket}/{key}...")
                    s3.upload_file(str(final_local_path), bucket, key)
                    print(f"Raster uploaded successfully to S3.")

            else:
                # Handle standard S3 vs custom bucket/key for sand_mud_mask
                if field_name == 'sand_mud_mask':
                    bucket = "ocs-dev-csdl-hydrohealth"
                    key = f"low_res/{filename}"
                else:
                    p = urlparse(self.raster_path)
                    bucket = p.netloc
                    key = f"{p.path.lstrip('/')}/{filename}".replace("//", "/")
                
                print(f"Uploading to s3://{bucket}/{key}...")
                s3 = boto3.client('s3')
                s3.upload_file(tmp_path, bucket, key)
                print(f"Raster uploaded successfully.")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def correct_sed_type(self, row: gpd.GeoSeries) -> str:
        """Corrects primary sediment type if sediment percerntages do not match grain size"""        

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

    def create_point_layer(self) -> None:
        """Creates a point layer from the sediment GeoDataFrame"""    
        gdf = gpd.GeoDataFrame(
            self.sediment_data, 
            geometry=gpd.points_from_xy(self.sediment_data['Longitude'], self.sediment_data['Latitude'])
        )  
        gdf.set_crs(crs="EPSG:4326", inplace=True)
        gdf_reprojected = gdf.to_crs("EPSG:6350")

        self._modify_gpkg(gdf_reprojected, 'sediment_points', mode='w')
        print('Created sediment point layer.')

    def determine_sed_types(self) -> None:
        """Determines the primary and secondarysediment types at each point based on type percentage"""  

        print('Calculating primary and secondary sediment types')
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

    def download_sediment_data(self) -> None:
        """Downloads the USGS sediment dataset"""     

        csv_path = f"{self.data_path}/US9_ONE.csv"

        # fsspec works with local and s3 paths
        fs = fsspec.open(csv_path).fs 
        if not fs.exists(csv_path):
            print(f"Downloading sediment data to {csv_path}...")
            try:
                url = get_config_item('SEDIMENT', 'DATA_URL')
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()
                    
                    with fsspec.open(csv_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print("Sediment data downloaded successfully.")
            except Exception as e:
                print(f'Error downloading sediment CSV: {e}')
                sys.exit(1)
        else:
            print("File already exists. Skipping download.")

    def read_sediment_data(self) -> None:   
        """Reads and stores the data from the USGS sediment dataset CSV"""      

        csv_columns = ['Latitude', 'Longitude', 'Gravel', 'Sand', 'Mud', 'Clay', 'Grainsze']
        sediment_data_path = f"{self.data_path}/US9_ONE.csv"
        self.sediment_data = pd.read_csv(sediment_data_path, usecols=csv_columns)

        print('Filtering out rows with missing data') 
        print(f' - Rows before: {self.sediment_data.shape[0]}')
        self.sediment_data = self.sediment_data[~((self.sediment_data[self.sediment_types] == -99) | (self.sediment_data[self.sediment_types] == 0)).all(axis=1)]
        self.sediment_data = self.sediment_data[(self.sediment_data['Grainsze'] != -99)].reset_index(drop=True)
        print(f' - Rows after: {self.sediment_data.shape[0]}')

    def transform_points_to_polygons(self) -> None:
        """Polygonize the sediment points and append as a new layer"""    
        print("Transforming sediment points to polygons...")
        
        # Pyogrio + fsspec handles the S3 read better than the write
        gdf = gpd.read_file(self.gpkg_path, layer='sediment_points', engine='pyogrio')
        
        coordinates_df = gdf.geometry.apply(lambda geom: geom.centroid.coords[0]).apply(pd.Series)
        coordinates_df.columns = ['Longitude', 'Latitude']
        vor = Voronoi(coordinates_df[['Longitude', 'Latitude']].values)
        polygons = []
        for point_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0: continue
            
            sed_int_val = gdf['sed_int'].iloc[point_idx]
            
            polygons.append({
                'geometry': Polygon([vor.vertices[i] for i in region]),
                'sed_type': sed_int_val,
                'sed_size': gdf['sed_size'].iloc[point_idx],
                'sand_mud_mask': 1 if sed_int_val in [2, 3] else None
            })

        gdf_voronoi = gpd.GeoDataFrame(polygons, crs='EPSG:6350')

        self._modify_gpkg(gdf_voronoi, 'sediment_polygons', mode='a')
        print("Appended sediment polygons layer.")

    def run(self) -> None:
        """Entrypoint for processing the Sediment layer"""

        self.download_sediment_data()
        self.read_sediment_data()
        self.add_sed_size_column()
        self.determine_sed_types()
        self.add_sediment_mapping_column()
        self.create_point_layer()
        self.transform_points_to_polygons()
        
        # Rasterize layers
        self.convert_polys_to_raster('sed_type')
        self.convert_polys_to_raster('sed_size')
        self.convert_polys_to_raster('sand_mud_mask')