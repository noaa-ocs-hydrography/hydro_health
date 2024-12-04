"""Class for processing everything for a single tile"""

import os
import sys
import geopandas as gpd
import pathlib
import multiprocessing as mp

mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


# OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


def run_hydro_health(output_folder, index: int, row: gpd.GeoSeries):
    # Mocking full process
    # Normal process would take all inputs needed and run GDAL requirements
    filename = str(index) + '.txt'
    with open(os.path.join(output_folder, filename), 'w') as writer:
        writer.write(str(list(row)))


class TileProcessor:
    def get_pool(self, processes=4):
        return mp.Pool(processes=processes)
    
    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False):
        with self.get_pool() as process_pool:
            results = [process_pool.apply_async(run_hydro_health, [outputs, index, row]) for index, row in tile_gdf.iterrows()]
            for result in results:
                result.get()
                # Also tried calling self.write_output() here and that worked fine

