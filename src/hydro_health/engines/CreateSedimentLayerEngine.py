import pathlib
import geopandas as gpd
import pandas as pd
import time
from osgeo import ogr, osr, gdal
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'
gpkg_path = INPUTS / 'sediment_data' / 'sediment_data.gpkg'


class CreateSedimentLayerEngine(Engine):
    """Class to hold the logic for processing the Sediment layer"""

    def __init__(self, 
                 param_lookup:dict=None):
        super().__init__()
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
        Adds column for the sediment size in mm
        Renames Grainsze column to Size_phi to clarify size is in phi units
        """      

        self.sediment_data['Size_mm'] = 2 ** -(self.sediment_data['Grainsze'])
        self.sediment_data = self.sediment_data.rename(columns={'Grainsze': 'Size_phi'})

    def add_sediment_mapping_column(self):   
        """Adds column for the integer value of each sediment time for rasterization"""    

        sediment_mapping = {
            'Gravel': 1,
            'Sand': 2,
            'Mud': 3,
            'Clay': 4
        }
        self.sediment_data['sed_int'] = self.sediment_data['prim_sed'].map(sediment_mapping)

    def clip_polygons_with_mask(self):
        """Clips the polygon layer to match the tile mask extent"""  
              
        data_gdf = gpd.read_file(gpkg_path, layer='sediment_polgons')
        mask_gdf = gpd.read_file(tile_path)

        mask_gdf = mask_gdf[mask_gdf["Value"] == 1]

        clipped_gdf = gpd.clip(data_gdf, mask_gdf)
        clipped_gdf.to_file(gpkg_path, layer='sediment_polys_clipped', driver="GPKG", overwrite=True)   

    def clip_sediment_points(self):
        # TODO I need to run more testing to see if this first clip is needed, 
        # I think I needed it originally bc this same process was so slow in ArcGIS
        # but is actually fast in python
        """
        Clip the sediment point layer before converting to polygons to reduce polygon layer size.
        """        
        mask = gpd.read_file(tile_path)
        sed_point_shapefile = gpd.read_file(gpkg_path, layer='sediment_points')

        points = sed_point_shapefile.to_crs(mask.crs)

        clipped = gpd.sjoin(points, mask, how='inner', predicate='within')
        clipped.to_file(gpkg_path, layer='sediment_points_clipped', driver="GPKG", overwrite=True) 

    def convert_polys_to_raster(self, field_name):
        """ Rasterize the polygons with selected column as pixel value"""        

        gpkg = ogr.Open(gpkg_path, 1)
        layer = gpkg.GetLayerByName('sediment_polys_clipped')

        tiff = gdal.Open(tile_path)
        geotransform = tiff.GetGeoTransform()
        proj = tiff.GetProjection()
        cols = tiff.RasterXSize
        rows = tiff.RasterYSize

        driver = gdal.GetDriverByName('GTIFF')
        file_path = INPUTS / "sediment_data" / f"{tile_number}_{field_name}_raster.tif"
        out_raster = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
        out_raster.SetGeoTransform(geotransform)
        out_raster.SetProjection(proj)

        gdal.RasterizeLayer(out_raster, [1], layer, options=['ALL_TOUCHED=FALSE', 'ATTRIBUTE={field_name}'])

        band = out_raster.GetRasterBand(1)
        band.SetNoDataValue(0) 
        out_raster = None
        gpkg = None
        tiff = None      

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

    def create_point_layer(self):
        """Creates a point shapefile from the sediment GeoDataFrame"""    
        
        gdf = gpd.GeoDataFrame(self.sediment_data, 
                               geometry=gpd.points_from_xy(self.sediment_data['Longitude'], self.sediment_data['Latitude']))  
        gdf.set_crs(crs="EPSG:4326", inplace=True)
        gdf_reprojected = gdf.to_crs("EPSG:26917")
        
        gdf_reprojected.to_file(gpkg_path, layer='sediment_points', driver = 'GPKG', overwrite=True)
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
        self.sediment_data['prim_sed'] = prim_values  
        self.sediment_data['sec_sed'] = sec_values    

        self.sediment_data['prim_sed'] = self.sediment_data.apply(self.correct_sed_type, axis=1)      

    def read_sediment_data(self):   
        """Reads and stores the data from the USGS sediment dataset CSV"""      

        csv_columns = ['Latitude', 'Longitude', 'Gravel', 'Sand', 'Mud', 'Clay', 'Grainsze']
        sediment_data_path = str('N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Grid_Development\Data_Sourced\usSEABED_EEZ'
                                  / 'US9_ONE.csv')
        self.sediment_data = pd.read_csv(sediment_data_path, usecols=csv_columns)

        self.message('Filtering out rows with missing data') 
        self.message(f' - Rows before: {self.sediment_data.shape[0]}')
        self.sediment_data = self.sediment_data[~((self.sediment_data[self.sediment_types] == -99) | (self.sediment_data[self.sediment_types] == 0)).all(axis=1)]
        self.sediment_data = self.sediment_data[(self.sediment_data['Grainsze'] != -99)].reset_index(drop=True)
        self.message(f' - Rows after: {self.sediment_data.shape[0]}')

    def transform_points_to_polygons(self):
        """Polygonize the sediment points"""       

        gdf = gpd.read_file(gpkg_path, layer='sediment_points_clipped')
        coordinates_df = gdf.geometry.apply(lambda geom: geom.centroid.coords[0]).apply(pd.Series)
        coordinates_df.columns = ['Longitude', 'Latitude']
        prim_sed_values = gdf['prim_sed'].tolist()
        grain_size_values = gdf['Size_mm'].tolist()
        
        vor = Voronoi(coordinates_df[['Longitude', 'Latitude']].values)
        polygons = []

        for point_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                continue
            polygon = Polygon([vor.vertices[i] for i in region])

            polygons.append({
                'geometry': polygon,
                'prim_sed': prim_sed_values[point_idx],
                'Size_mm': grain_size_values[point_idx]
            })

        gdf_voronoi = gpd.GeoDataFrame(polygons, crs='EPSG:26917')
        gdf_voronoi.to_file(gpkg_path, layer='sediment_polgons', driver = 'GPKG', overwrite=True)   

    def start(self):
        """Entrypoint for processing the Sediment layer"""
        start = time.time()
        self.read_sediment_data()
        self.add_sed_size_column()
        self.add_sediment_mapping_column()
        self.determine_sed_types()
        self.create_point_layer()
        self.clip_sediment_points()
        self.transform_points_to_polygons()
        self.clip_polygons_with_mask()
        self.convert_polys_to_raster('sed_int')
        self.convert_polys_to_raster('Size_mm')
        self.message(f'Run time: {(time.time() - start) / 60}')
