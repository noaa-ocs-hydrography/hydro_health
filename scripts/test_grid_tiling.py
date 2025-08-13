from osgeo import gdal, osr, ogr

import os
import yaml
import pathlib
from socket import gethostname


INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'


def get_environment() -> str:
    """Determine current environment running code"""

    hostname = gethostname()
    if 'L' in hostname:
        return  'local'
    elif 'VS' in hostname:
        return 'remote'
    else:
        return 'remote'


def get_config_item(parent: str, child: str=False, env_string: str=False) -> str:
    """
    Load config and return speciific key
    :param str parent: Primary key in config
    :param str child: Secondary key in config
    :param str env_string: Optional explicit value of "local" or "remote"
    :returns str: Value from local or remote YAML config
    """

    # TODO add sample data and folders to input folder

    env = env_string if env_string else None
    if env is None:
        env = get_environment()
    
    with open(str(INPUTS / 'lookups' / f'{env}_path_config.yaml'), 'r') as lookup:
        config = yaml.safe_load(lookup)
        parent_item = config[parent]
        if child:
            return parent_item[child]
        else:
            return parent_item
        

def test_grid_tiling():
    

    blue_topo_folder = pathlib.Path(r'C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3') / 'BlueTopo'
    bluetopo_grids = [folder.stem for folder in blue_topo_folder.iterdir() if folder.is_dir()]
    print(bluetopo_grids)

    gpkg_ds = ogr.Open(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))
    blue_topo_layer = gpkg_ds.GetLayerByName(get_config_item('SHARED', 'TILES'))
    for feature in blue_topo_layer:
        # Clip VRT by current polygon
        polygon = feature.GetGeometryRef()
        folder_name = feature.GetField('tile')
        if folder_name in bluetopo_grids:
            vrt_files = [
                'mosaic_NCMP_6326_NOAA_NGS_2019', 
                'mosaic_NCMP_6326_USACE_NCMP_2010',
                'mosaic_NCMP_6326_USACE_NCMP_2020',
                'mosaic_NCMP_6326_USACE_NCMP_2022']
            for vrt_name in vrt_files:
                output_folder = pathlib.Path(r'c:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\\ER_3\DigitalCoast\tiled\BH4PQ58F')
                output_folder.mkdir(parents=True, exist_ok=True)

                output_clipped_vrt = output_folder / pathlib.Path(fr"{vrt_name}_BH4PQ58F.tiff")
                vrt = pathlib.Path(fr"c:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\{vrt_name}.vrt")
                vrt_ds = gdal.Open(str(vrt))
                vrt_data_folder = vrt.parents[0] / '_'.join(vrt.stem.split('_')[3:])
                vrt_tile_index = list(vrt_data_folder.rglob('*_dis.shp'))[0]
                shp_driver = ogr.GetDriverByName('ESRI Shapefile')
                vrt_tile_index_shp = shp_driver.Open(vrt_tile_index, 0)
                # get geometry of single feature
                dissolve_layer = vrt_tile_index_shp.GetLayer(0)
                raster_geom = None
                for dis_feature in dissolve_layer:
                    raster_geom = dis_feature.GetGeometryRef()
                    break

                if polygon.Intersects(raster_geom):
                    print('start', vrt_name)
                    gdal.Warp(
                        str(output_clipped_vrt),
                        str(vrt),
                        format='GTiff',
                        cutlineDSName=polygon,
                        cropToCutline=True,
                        dstNodata=vrt_ds.GetRasterBand(1).GetNoDataValue(),
                        cutlineSRS=vrt_ds.GetProjection(),
                        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"]
                    )
                    print('done')


def test_single_tiff():
    # load tiff
    # create polygon variable
    # set up gdal.Warp

    # input_tiff = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\USACE_NCMP_2010\dem\USACE_AL_FL_DEM_2010_9453\2010_NCMP_FL_03_FL_BareEarth_1mGrid.tif"
    # input_tiff = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\USACE_NCMP_2010\dem\USACE_AL_FL_DEM_2010_9453\2010_NCMP_FL_04_FL_BareEarth_1mGrid.tif"
    # input_tiff = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\USACE_NCMP_2010\dem\USACE_AL_FL_DEM_2010_9453\2010_NCMP_FL_21_FL_BareEarth_1mGrid.tif"
    # input_tiff = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\tiled\merged_test.tif"
    input_tiff = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\mosaic_NCMP_6326_USACE_NCMP_2010.vrt"
    # polygon = 'MULTIPOLYGON (((-87.45 30.3000000000001,-87.45 30.2250000000001,-87.3749999999999 30.2250000000001,-87.3749999999999 30.3000000000001,-87.45 30.3000000000001)))'
    polygon = 'POLYGON ((-87.45 30.3000000000001, -87.45 30.2250000000001, -87.3749999999999 30.2250000000001, -87.3749999999999 30.3000000000001, -87.45 30.3000000000001))'
    output_clipped_vrt = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\DigitalCoast\tiled\test.tiff"
    vrt_ds = gdal.Open(str(input_tiff))
    print(polygon, '\n', str(output_clipped_vrt), '\n', str(input_tiff))
    gdal.Warp(
        output_clipped_vrt,
        input_tiff,
        format='GTiff',
        cutlineDSName=polygon,
        cropToCutline=True,
        dstNodata=-9999,
        cutlineSRS=vrt_ds.GetProjection(),
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"]
    )
    vrt_ds = None


    # with rasterio.open(vrt) as raster_src:
    #     polygon_shapes = [from_wkt(polygon.wkt) for polygon in wkt.loads(current_tile_geom.ExportToWkt()).geoms]
    #     print(polygon_shapes)
    #     out_image, transform = rasterio.mask.mask(raster_src, polygon_shapes, crop=True)
    #     out_meta = raster_src.meta.copy()
    #     out_meta.update({
    #         "driver": "GTiff",
    #         "height": out_image.shape[1],
    #         "width": out_image.shape[2],
    #         "transform": transform
    #     })
    #     with rasterio.open(output_clipped_vrt, 'w', **out_meta) as raster_dst:
    #         print('output:', output_clipped_vrt)
    #         raster_dst.write(out_image)
    # polygon = [polygon.wkt for polygon in wkt.loads(current_tile_geom.ExportToWkt()).geoms][0]






if __name__ == '__main__':
    # test_grid_tiling()
    test_single_tiff()
