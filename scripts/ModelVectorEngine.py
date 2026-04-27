import os
import sys
import pathlib
import time
import numpy as np
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from osgeo import gdal
from shapely.geometry import box

HH_MODEL = pathlib.Path(__file__).parents[1] / 'src'
sys.path.append(str(HH_MODEL))

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import Param, get_config_item, get_environment

os.environ['PROJ_LIB_CACHE'] = 'OFF'
os.environ['PROJ_NETWORK'] = 'OFF'
gdal.UseExceptions()


INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'


def _set_gdal_s3_options():
    """Optimized GDAL settings for S3 VSI stability."""

    gdal.SetConfigOption('GDAL_CACHEMAX', '512')
    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '5')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('GDAL_PROJ_THREAD_SAFE', 'YES')


def _zonal_worker(gdf_chunk: gpd.GeoDataFrame, raster_path: str) -> list:
    """Isolated worker for parallel processing."""

    _set_gdal_s3_options()
    try:
        with rasterio.open(raster_path) as src:
            stats = zonal_stats(
                gdf_chunk, 
                raster_path, 
                stats=['median'],
                nodata=src.nodata,
                affine=src.transform,
                all_touched=True
            )
        return [s['median'] for s in stats]
    except Exception as e:
        return [None] * len(gdf_chunk)


class ModelVectorEngine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        
        self.param_lookup = param_lookup
        self.gpkg_path = INPUTS / 'ER3_Grids_WGS84.gpkg'
        self.source_layer = 'Model_Subgrid'
        self.output_layer = 'median_present_survey_score'
        self.s3_raster = f"/vsis3/{get_config_item('SHARED', 'OUTPUT_BUCKET')}/ER_3/outputs/present_survey_score_int.tif"

    def _get_intersecting_polygons(self, outputs: str) -> gpd.GeoDataFrame:
        """Loads and filters polygons to those within the raster footprint."""

        with rasterio.open(self.s3_raster) as src:
            raster_crs = src.crs
            raster_bounds = src.bounds
            
            self.write_message(f"Reading and reprojecting {self.source_layer}...", outputs)
            gdf = gpd.read_file(self.gpkg_path, layer=self.source_layer).to_crs(raster_crs)
            
            # Spatial filter using raster bounding box
            search_envelope = box(*raster_bounds)
            intersecting = gdf[gdf.intersects(search_envelope)].copy()
            
            self.write_message(f"Retained {len(intersecting)} polygons within raster extent.", outputs)
            return intersecting

    def _compute_parallel_stats(self, gdf: gpd.GeoDataFrame, outputs: str, workers: int = 6) -> list:
        """Orchestrates the parallel worker swarm."""

        chunk_size = 1000
        chunks = [gdf.iloc[i:i+chunk_size].copy() for i in range(0, len(gdf), chunk_size)]
        
        worker_func = partial(_zonal_worker, raster_path=self.s3_raster)
        all_results = []
        
        self.write_message(f"Processing {len(chunks)} chunks across {workers} workers...", outputs)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for result in executor.map(worker_func, chunks):
                all_results.extend(result)
        
        return all_results

    def _write_to_geopackage(self, gdf: gpd.GeoDataFrame, outputs: str) -> None:
        """Standardizes CRS and writes the new layer to the input GPKG."""

        self.write_message(f"Writing results to layer: {self.output_layer}", outputs)
        
        # Revert to WGS84 for consistency with the master grid file
        gdf_out = gdf.to_crs("EPSG:4326")
        
        gdf_out.to_file(
            self.gpkg_path, 
            layer=self.output_layer, 
            driver="GPKG",
            engine="pyogrio"
        )

    def run(self, outputs: str) -> None:
        """Main entry point: Orchestrates the vectorization pipeline."""

        try:
            _set_gdal_s3_options()

            target_gdf = self._get_intersecting_polygons(outputs)
            if target_gdf.empty:
                self.write_message("Empty intersection set. Skipping.", outputs)
                return

            target_gdf['median_score'] = self._compute_parallel_stats(target_gdf, outputs)
            self._write_to_geopackage(target_gdf, outputs)
            self.write_message("Engine process complete.", outputs)

        except Exception as e:
            self.write_message(f"CRITICAL ERROR in GulfVectorizationEngine: {str(e)}", outputs)
            raise e
        

if __name__ == "__main__":
    start = time.time()
    env = get_environment()
    print('Environment:', env)
    param_lookup = {
        'input_directory': Param(''),
        'output_directory': Param(str(OUTPUTS)),
        'eco_regions': Param(''),
        'env': 'aws'  # force aws for now
    }

    # TODO EC2 instances have different hostnames.  02 returns "remote" for env
    engine = ModelVectorEngine(param_lookup)
    engine.run(OUTPUTS)
    end = time.time()
    print(f"Total Runtime: {(end - start) / 60} minutes")