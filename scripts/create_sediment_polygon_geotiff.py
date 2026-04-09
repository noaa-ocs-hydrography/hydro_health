import os
import pathlib
import pandas as pd
import geopandas as gpd
from osgeo import ogr, osr, gdal
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

projection_nad17 = osr.SpatialReference()
projection_nad17.ImportFromEPSG(26917)

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


mask_path = str(INPUTS / 'Spatial_Aggregate_Mosaic3.tif')
mask_nad17_path =  str(OUTPUTS / 'mask_nad17.tif')
mask_shapefile_path = str(OUTPUTS / 'mask_nad17.shp')
sediment_shapefile_path = str(OUTPUTS / 'sediment_shapefile.shp')
clipped_sed_points_path = str(OUTPUTS / 'clipped_sed_points.shp')
poly_shapefile_path = str(OUTPUTS / 'sediment_layer_poly.shp')
clipped_poly_path = str(OUTPUTS / 'sediment_poly_clipped.shp')
tif_prim_sed_path = str(OUTPUTS / 'prim_sed_layer_w_florida.tif')
tif_grn_size_path = str(OUTPUTS / 'grain_size_layer_w_florida.tif')


def reproject_the_mask():
    mask_raster = gdal.Open(mask_path)
    output_format = "GTiff"
    output_res = 5
    gdal.Warp(mask_nad17_path, mask_raster, xRes=output_res, yRes=output_res, dstSRS = projection_nad17, format=output_format)

def create_mask_shapefile():
    field_name = 'Value'
    ds = gdal.Open(mask_nad17_path)
    band = ds.GetRasterBand(1)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(mask_shapefile_path):
        driver.DeleteDataSource(mask_shapefile_path)
    shp_ds = driver.CreateDataSource(mask_shapefile_path)
    layer = shp_ds.CreateLayer('polygons', srs=projection_nad17)

    field_defn = ogr.FieldDefn(field_name, ogr.OFTReal)
    layer.CreateField(field_defn)

    gdal.Polygonize(band, None, layer, 0, callback=None)

    shp_ds = None
    ds = None

def clip_sediment_points():
    mask = gpd.read_file(mask_shapefile_path)
    sed_point_shapefile = gpd.read_file(sediment_shapefile_path)

    points = sed_point_shapefile.to_crs(mask.crs)

    clipped = gpd.sjoin(points, mask, how='inner', predicate='within')
    clipped.to_file(clipped_sed_points_path)

def transform_points_to_polygons():
    gdf = gpd.read_file(clipped_sed_points_path)
    coordinates_df = gdf.geometry.apply(lambda geom: geom.centroid.coords[0]).apply(pd.Series)
    coordinates_df.columns = ["Longitude", "Latitude"]
    prim_sed_values = gdf['prim_sed'].tolist()
    grain_size_values = gdf['Size_mm'].tolist()
    
    vor = Voronoi(coordinates_df[["Longitude","Latitude"]].values)
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
    gdf_voronoi.to_file(poly_shapefile_path)
        
def clip_polygons_with_mask():
    data_shp = gpd.read_file(poly_shapefile_path)
    mask_shp = gpd.read_file(mask_shapefile_path)

    mask_shp = mask_shp[mask_shp["Value"] == 1]

    clipped_shp = gpd.clip(data_shp, mask_shp)
    clipped_shp.to_file(clipped_poly_path)

def convert_prim_sed_to_tif():
    sediment_mapping = {
        'Gravel': 1,
        'Sand': 2,
        'Mud': 3,
        'Clay': 4
    }

    shp = ogr.Open(clipped_poly_path, 1)
    layer = shp.GetLayer()

    field_name = 'sed_int'

    if not layer.FindFieldIndex(field_name, True) >= 0:
        new_field = ogr.FieldDefn(field_name, ogr.OFTInteger)
        layer.CreateField(new_field)

        for feature in layer:
            prim_sed = feature.GetField('prim_sed')
            sed_int_value = sediment_mapping.get(prim_sed, 0)    
            feature.SetField(field_name, sed_int_value)
            layer.SetFeature(feature)

    tiff = gdal.Open(mask_nad17_path)
    geotransform = tiff.GetGeoTransform()
    proj = tiff.GetProjection()
    cols = tiff.RasterXSize
    rows = tiff.RasterYSize

    driver = gdal.GetDriverByName('GTIFF')
    out_raster = driver.Create(tif_prim_sed_path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(proj)

    gdal.RasterizeLayer(out_raster, [1], layer, options=['ALL_TOUCHED=FALSE', 'ATTRIBUTE=sed_int'])

    band = out_raster.GetRasterBand(1)
    band.SetNoDataValue(0) 
    out_raster = None
    shp = None
    tiff = None

def convert_grain_size_to_tif():
    shp = ogr.Open(clipped_poly_path, 1)
    layer = shp.GetLayer()

    tiff = gdal.Open(mask_nad17_path)
    geotransform = tiff.GetGeoTransform()
    proj = tiff.GetProjection()
    cols = tiff.RasterXSize
    rows = tiff.RasterYSize

    driver = gdal.GetDriverByName('GTIFF')
    out_raster = driver.Create(tif_grn_size_path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(proj)

    gdal.RasterizeLayer(out_raster, [1], layer, options=['ALL_TOUCHED=FALSE', 'ATTRIBUTE=Size_mm'])

    band = out_raster.GetRasterBand(1)
    band.SetNoDataValue(0) 
    out_raster = None
    shp = None
    tiff = None    

reproject_the_mask()
create_mask_shapefile()
clip_sediment_points()
transform_points_to_polygons()
clip_polygons_with_mask()
convert_prim_sed_to_tif()
convert_grain_size_to_tif()
