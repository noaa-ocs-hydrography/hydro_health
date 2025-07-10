"""Class for processing everything for a single tile"""

import os
import pathlib
import sys
import tempfile
import geopandas as gpd
import multiprocessing as mp
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from hydro_health.helpers.tools import get_config_item
from osgeo import gdal, ogr


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


class GridRastersProcessor:
    """Class for parallel processing the tiling of raster datasets"""

    def grid_vrt(vrt: pathlib.Path, ecoregion: str, outputs: str, data_type: str, blue_topo_layer: ogr.Layer, bluetopo_grids: list[str]) -> None:
        """Clip raster VRTs to BlueTopo grid"""

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
        dissolve_layer = None
        blue_topo_layer.ResetReading()
        for feature in blue_topo_layer:
            # Clip VRT by current polygon
            polygon = feature.GetGeometryRef()
            folder_name = feature.GetField('tile')
            output_path = ecoregion / data_type / 'tiled' / folder_name
            output_clipped_vrt = output_path / f'{vrt.stem}_{folder_name}.tiff'
            if output_clipped_vrt.exists():
                if output_clipped_vrt.stat().st_size == 0:
                    print(f're-warp empty raster: {output_clipped_vrt.name}')
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
            elif folder_name in bluetopo_grids:
                try:
                    if polygon.Intersects(raster_geom):
                        output_path.mkdir(parents=True, exist_ok=True)
                        print(f'Creating {output_clipped_vrt.name}')
                        # Try to force clear temp directory to conserve space
                        with tempfile.TemporaryDirectory() as temp:
                            gdal.SetConfigOption('CPL_TMPDIR', temp)
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
                except Exception as e:
                    print('failed:', e)
            polygon = None
        raster_geom = None
        vrt_ds = None
        
        return f'vrt: {vrt}'
        
    def grid_vrt_files(self, outputs: str, data_type: str) -> None:
        """Clip VRT files to BlueTopo grid"""

        gpkg_ds = ogr.Open(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))
        blue_topo_layer = gpkg_ds.GetLayerByName(get_config_item('SHARED', 'TILES'))
        ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
        for ecoregion in ecoregions:
            blue_topo_folder = ecoregion / 'BlueTopo'
            bluetopo_grids = [folder.stem for folder in blue_topo_folder.iterdir() if folder.is_dir()]
            data_folder = ecoregion / data_type
            vrt_files = data_folder.glob('*.vrt')
            # TODO call grid_vrt here for multiprocessing

            param_inputs = [[vrt, ecoregion, outputs, data_type, blue_topo_layer, bluetopo_grids] for vrt in vrt_files]  # rows out of ER will be nan
            with ThreadPoolExecutor(int(os.cpu_count() - 2)) as intersected_pool:
                self.print_async_results(intersected_pool.map(self.process_tile, param_inputs), outputs)

            
        gpkg_ds = None
        blue_topo_layer = None

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(result, output_folder)

    def process(self) -> None:
        self.grid_vrt_files()

    def write_message(self, message: str, output_folder: str|pathlib.Path) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')
