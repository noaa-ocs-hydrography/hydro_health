import rasterio
import pathlib
import geopandas as gpd
import boto3

from osgeo import gdal, osr
from botocore.client import Config
from botocore import UNSIGNED
from rasterio.features import shapes


OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'
INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'



def create_raster_vrt():
    blue_topo_folder = OUTPUTS / 'BlueTopo'
    geotiffs = list(blue_topo_folder.rglob('*.tiff'))
    vrt_filename = str(OUTPUTS / 'mask_tiles_mosaic.vrt')
    gdal.BuildVRT(vrt_filename, geotiffs, callback=gdal.TermProgress_nocb)
    final_image = gdal.Open(vrt_filename, gdal.GA_ReadOnly)

    # This makes a separate single raster
    # gdal.Translate(str(OUTPUTS / 'mask_tiles.tif'), final_image, format='GTiff',
    #            creationOptions=['COMPRESS:DEFLATE', 'TILED:YES'],
    #            callback=gdal.TermProgress_nocb)
    # final_image = None


def download_nbs_tiles(tiles):
    bucket = "noaa-ocs-nationalbathymetry-pds"
    creds = {
        "aws_access_key_id": "",
        "aws_secret_access_key": "",
        "config": Config(signature_version=UNSIGNED),
    }
    s3 = boto3.resource('s3', **creds)
    nbs_bucket = s3.Bucket(bucket)
    for tile_id in tiles:
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            print(f'downloading: {obj_summary.key}')
            output_tile = OUTPUTS / obj_summary.key
            output_folder = output_tile.parents[0]
            output_folder.mkdir(parents=True, exist_ok=True)
            nbs_bucket.download_file(obj_summary.key, output_tile)


def get_intersected_tiles(mask_polygons_gdf):
    mask_polygons_wgs84 = project_gdf(mask_polygons_gdf)
    nbs_tile_gpkg = INPUTS / 'BlueTopo_Tile_Scheme_20250117_104024.gpkg'
    nbs_tile_gdf = gpd.read_file(nbs_tile_gpkg, layer='BlueTopo_Tile_Scheme_20250117_104024')  # WGS_84
    intersected_gdf = gpd.sjoin(mask_polygons_wgs84, nbs_tile_gdf, how = 'left')
    tiles = intersected_gdf['tile'].unique()
    print(f'Intersected tiles: {tiles}')
    output_intersection = OUTPUTS / 'mask_tiles.shp'
    intersected_gdf.to_file(output_intersection, driver='ESRI Shapefile')

    return tiles


def get_mask_shapes(raster):
    with rasterio.Env():
        with rasterio.open(raster) as source:
            image = source.read(1)
            results = []
            for i, (s, v) in enumerate(shapes(image, mask=None, transform=source.transform)):
                if v == 1:
                    results.append({"properties": {"raster_val": v}, "geometry": s})
            # results = [
            #     {"properties": {"raster_val": v}, "geometry": s}
            #     for i, (s, v) in enumerate(
            #         shapes(image, mask=None, transform=source.transform)
            #     )
            # ]

            return results
        

def project_gdf(gdf, wkid=4326):
    return gdf.to_crs(wkid)


if __name__ == "__main__":
    mask = INPUTS / 'prediction.mask.tif'  # NAD_1983_UTM_Zone_17N CRS
    mask_polygons = get_mask_shapes(mask)
    mask_polygons_gdf  = gpd.GeoDataFrame.from_features(mask_polygons, crs=26917)
    tiles = get_intersected_tiles(mask_polygons_gdf)
    download_nbs_tiles(tiles)
    create_raster_vrt()