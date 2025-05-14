from osgeo import gdal, osr, ogr
import pathlib
import yaml

gdal.UseExceptions()

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


def get_config_item(parent: str, child: str=False) -> tuple[str, int]:
    """Load config and return speciific key"""

    with open(str(INPUTS / 'lookups' / 'config.yaml'), 'r') as lookup:
        config = yaml.safe_load(lookup)
        parent_item = config[parent]
        if child:
            return parent_item[child]
        else:
            return parent_item
        

# geotiff = pathlib.Path(r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\NOAA_NGS_2019\dem\NGS_NW_Florida_Topobathy_DEM_2020_9708\2020_455000e_3350000n_dem.tif")
outputs = OUTPUTS / 'ER_3' / 'DigitalCoast'
geotiffs = list(outputs.rglob('*.tif'))
for geotiff in geotiffs:
    print(f'Projecting to wgs84: {geotiff}')
    geotiff_ds = gdal.Open(geotiff)
    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    raster_wgs84 = geotiff.parents[0] / f'{geotiff.stem}_wgs84.tif'
    gdal.Warp(
        raster_wgs84,
        geotiff_ds,
        dstSRS=wgs84_srs
    )
    print('done')
    vrt = raster_wgs84.parents[0] / f'{raster_wgs84.stem}.vrt'
    gdal.BuildVRT(vrt, [raster_wgs84], callback=gdal.TermProgress_nocb)


    geotiff_ds = None
    wgs84_srs = None


    gpkg_ds = ogr.Open(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS'))
    blue_topo_layer = gpkg_ds.GetLayerByName(get_config_item('SHARED', 'TILES'))
    vrt_ds = gdal.Open(vrt)
    # finish tiling
    gt = vrt_ds.GetGeoTransform()
    raster_extent = (gt[0], gt[3], gt[0] + gt[1] * vrt_ds.RasterXSize, gt[3] + gt[5] * vrt_ds.RasterYSize)
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(raster_extent[0], raster_extent[1])
    ring.AddPoint(raster_extent[2], raster_extent[1])
    ring.AddPoint(raster_extent[2], raster_extent[3])
    ring.AddPoint(raster_extent[0], raster_extent[3])
    ring.AddPoint(raster_extent[0], raster_extent[1])
    raster_geom = ogr.Geometry(ogr.wkbPolygon)
    raster_geom.AddGeometry(ring)

    bluetopo_grids = ['BH4PN58F', 'BH4PP58F']
    print(bluetopo_grids)
    for feature in blue_topo_layer:
        folder_name = feature.GetField('tile')
        polygon = feature.GetGeometryRef()
        if folder_name in bluetopo_grids:
            print('Found folder:', folder_name)
            output_path = geotiff.parents[3] / f'tiled_{folder_name}.tif'
            if polygon.Intersects(raster_geom):
                print('intersects')
                gdal.Warp(
                    output_path,
                    vrt,
                    format='GTiff',
                    cutlineDSName=polygon,
                    cropToCutline=True,
                    dstNodata=vrt_ds.GetRasterBand(1).GetNoDataValue(),
                    cutlineSRS=vrt_ds.GetProjection(),
                    creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"]
                )
            else:
                print('nope')
    print('done')
    vrt_ds = None
    polygon = None
    raster_geom = None
    ring = None
    gt = None
    gpkg_ds = None

    break