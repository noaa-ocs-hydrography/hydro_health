import os
import pathlib
import subprocess
import geopandas as gpd
import pandas as pd
import requests
import time
from osgeo import ogr, osr, gdal

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateSedimentLayerEngine(Engine):
    """Class to hold the logic for processing the Sediment layer"""

    def __init__(self, 
                 param_lookup:dict=None):
        # super().__init__()
        self.sediment_types = ['Gravel', 'Sand', 'Mud', 'Clay'] 
        self.sediment_data = None
        if param_lookup:
            self.param_lookup = param_lookup
            if self.param_lookup['input_directory'].valueAsText:
                global INPUTS
                INPUTS = pathlib.Path(self.param_lookup['input_directory'].valueAsText)
            if self.param_lookup['output_directory'].valueAsText:
                global OUTPUTS
                OUTPUTS = pathlib.Path(self.param_lookup['output_directoty'].valueAsText)

    def add_sed_size_column(self):
        """
        Adds column for the sediment size in mm.
        Renames Grainsze column to Size_phi to clarify size is in phi units.
        """      

        self.sediment_data['Size_mm'] = 2 ** -(self.sediment_data['Grainsze'])
        self.sediment_data = self.sediment_data.rename(columns={'Grainsze': 'Size_phi'})

    def correct_sed_type(self, row):
        """
        Corrects primary sediment type if sediment percerntages do not match grain size.
        :param dataframe row: Each row of the point sediment dataframe
        :return str: Primary sediment type
        """        

        # These size classification ranges are based on the Udden-Wentworth grain size chart
        if 0 < row['Size_mm'] < 0.0039:
            return 'Clay'
        elif 0.0039 <= row['Size_mm'] < 0.0625:
            return 'Mud'
        elif 0.0625 <= row['Size_mm'] < 2:
            return 'Sand'
        elif 2 <= row['Size_mm']: 
            return 'Gravel'
        else:
            return row['prim_sed']    

    def create_point_shapefile(self):
        """Creates a point shapefile from the sediment GeoDataFrame"""    

        print('Creating sediment point shapefile') 
        shapefile_path = str(OUTPUTS / 'sediment_shapefile.shp') 

        gdf = gpd.GeoDataFrame(self.sediment_data, 
                               geometry=gpd.points_from_xy(self.sediment_data['Longitude'], self.sediment_data['Latitude']))  
        gdf.set_crs(crs="EPSG:4326", inplace=True)
        gdf_reprojected = gdf.to_crs("EPSG:26917")
        
        gdf_reprojected.to_file(shapefile_path, driver = 'ESRI Shapefile')

        # GDAL code for creating the shp
        # driver = ogr.GetDriverByName('ESRI Shapefile')
        # shapefile = driver.CreateDataSource(shapefile_path)
        # srs = osr.SpatialReference()
        # srs.ImportFromEPSG(26917)
        # layer = shapefile.CreateLayer('sediment_points', srs, geom_type = ogr.wkbPoint)

        # field_types = {'Latitude': ogr.OFTReal,
        #                'Longitude': ogr.OFTReal,
        #                'Gravel': ogr.OFTReal,
        #                'Sand': ogr.OFTReal,
        #                'Mud': ogr.OFTReal,
        #                'Clay': ogr.OFTReal,
        #                'Grainsze': ogr.OFTReal,
        #                'Size (mm)': ogr.OFTReal,
        #                'prim_sed': ogr.OFTString, 
        #                'sec_sed': ogr.OFTString
        #                }
        
        # print(' - Creating fields')
        # for column, field_type in field_types.items():
        #     field = ogr.FieldDefn(column, field_type)
        #     layer.CreateField(field)

        # print(' - Writing points and field values')
        # for index, row in self.sediment_data.iterrows():
        #     print(index)
        #     point = ogr.Geometry(ogr.wkbPoint)
        #     # print(row['Latitude'], row['Longitude'])
        #     point.AddPoint(row['Longitude'], row['Latitude'])
        #     feature = ogr.Feature(layer.GetLayerDefn())
        #     feature.SetGeometry(point)

        #     column = field_types.keys()
        #     for column in self.sediment_data:
        #             feature.SetField(column, row[column])

        #     layer.CreateFeature(feature)  
        #     feature = None
        #     point = None

        # shapefile = None  

    def determine_sed_types(self):
        """Determines the primary and secondarysediment types
        at each point based on type percentage"""  

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
        self.sediment_data['prim_sed'] = prim_values  
        self.sediment_data['sec_sed'] = sec_values    

        self.sediment_data['prim_sed'] = self.sediment_data.apply(self.correct_sed_type, axis=1)      

    def read_sediment_data(self):   
        """ 
        Reads and stores the data from the USGS sediment dataset CSV.
        :param list csv_columns: Columns required from the CSV file
        """      

        csv_columns = ['Latitude', 'Longitude', 'Gravel', 'Sand', 'Mud', 'Clay', 'Grainsze']

        sediment_data_path = str(INPUTS / 'US9_ONE.csv')
        self.sediment_data = pd.read_csv(sediment_data_path, usecols=csv_columns)

        print('Filtering out rows with missing data') 
        print(f' - Rows before: {self.sediment_data.shape[0]}')
        self.sediment_data = self.sediment_data[~((self.sediment_data[self.sediment_types] == -99) | (self.sediment_data[self.sediment_types] == 0)).all(axis=1)]
        self.sediment_data = self.sediment_data[(self.sediment_data['Grainsze'] != -99)].reset_index(drop=True)
        print(f' - Rows after: {self.sediment_data.shape[0]}')

    # def create_thiessen_polygons(self):
    # def apply_coastal_buffer:
        # Use the 2018 one, put in the inputs folder
    # def apply_nbs_grid(self):
    # def assign_grid_id(self): 
    # def export_geotiff(self):
    #     # export it as a 32-bit floating point tiff
          # resample to 5 m x 5 m use bilinear interpolation for continuous data resampling

    def start(self):
        """Entrypoint for processing the Sediment layer"""
        start = time.time()
        # self.download_data()
        self.read_sediment_data()
        self.add_sed_size_column()
        self.determine_sed_types()
        self.create_point_shapefile()
        print(f'Run time: {(time.time() - start) / 60}')
