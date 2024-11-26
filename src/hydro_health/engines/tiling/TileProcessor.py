"""Class for processing everything for a single tile"""

import os
import sys
import geopandas as gpd
import pathlib
import multiprocessing as mp

mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))


OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


def run_hydro_health(index: int, row: gpd.GeoSeries):
    # Sample process
    return (index, f'Finished {index}: {row}')


class TileProcessor:
    def get_pool(self, processes=4):
        return mp.Pool(processes=processes)
    

    def process(self, tile_gdf: gpd.GeoDataFrame):
        with self.get_pool() as process_pool:
            results = [process_pool.apply_async(run_hydro_health, [index, row]) for index, row in tile_gdf.iterrows()]
            for result in results:
                indx, row_result = result.get()
                self.write_result(indx, row_result)

    def write_result(self, index, result: str) -> None:
        filename = str(index) + '.txt'
        with open(os.path.join(str(OUTPUTS), 'junk', filename), 'w') as writer:
            writer.write(result)
