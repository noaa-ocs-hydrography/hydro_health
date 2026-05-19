import sys
import os
import pathlib
import tempfile
import boto3
import shutil
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.transform import from_bounds

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_environment


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


def _rasterize_tile_worker(intersected_tiles: gpd.GeoDataFrame, bounds: list[float], crs: str, resolution: int, output_path: pathlib.Path) -> bool:
    """
    Isolated worker to build a local, binary raster mask bounded strictly by selected tiles.
    Areas covered by the tile polygons are burned as 1, background is 0.
    """

    minx, miny, maxx, maxy = bounds
    
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    
    if width <= 0 or height <= 0:
        return False

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    mask = np.zeros((height, width), dtype=rasterio.uint8)
    
    shapes_to_burn = [(geom, 1) for geom in intersected_tiles.geometry if geom is not None]
    if not shapes_to_burn:
        return False
        
    features.rasterize(
        shapes=shapes_to_burn,
        out_shape=(height, width),
        transform=transform,
        fill=0,             
        out=mask            
    )
    
    meta = {
        'driver': 'GTiff',
        'dtype': rasterio.uint8,
        'nodata': 0,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask, 1)
        
    return True


class ShoreTileEngine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup
        
        # Consistent Configuration Foundations
        self.target_crs = "EPSG:32617"  # UTM Zone 17N
        self.nearshore_res = 20         # 20 meters
        self.offshore_res = 100         # 100 meters

    def load_and_classify_tiles(self, geopackage_path: pathlib.Path, outputs: str) -> gpd.GeoDataFrame:
        """Loads BlueTopo tiles and pre-flags nearshore/offshore boundaries."""

        self.write_message(" - Selecting nearshore and offshore tiles", outputs)
        tiles = gpd.read_file(geopackage_path, layer=get_config_item('SHARED', 'TILES'))
        
        nearshore_mask = tiles['tile'].str.startswith('BH', na=False)
        offshore_mask = (~nearshore_mask) & (~tiles['tile'].str.match(r'^BF2H[0-9]', na=False))
        
        tiles['_is_nearshore'] = nearshore_mask
        tiles['_is_offshore'] = offshore_mask
        return tiles

    def load_and_align_ecoregions(self, geopackage_path: pathlib.Path, reference_crs: str, outputs: str) -> gpd.GeoDataFrame:
        """Loads ecoregions and ensures spatial reference matches the tiles."""

        self.write_message(" - Loading ecoregions layer", outputs)
        ecoregions = gpd.read_file(geopackage_path, layer="EcoRegions")
        
        if ecoregions.crs != reference_crs:
            self.write_message("Aligning coordinate reference systems for spatial matching...", outputs)
            ecoregions = ecoregions.to_crs(reference_crs)
        return ecoregions

    def process_ecoregion_tiles(self, ecoregion_id: str, ecoregion_row: gpd.GeoSeries, bluetopo_tiles: gpd.GeoDataFrame, outputs: str) -> str:
        """Extracts, projects, and passes targeted groupings down to rasterizers before uploading to S3."""

        eco_geom = gpd.GeoDataFrame([ecoregion_row], crs=bluetopo_tiles.crs)
        
        mask_sub = get_config_item('MASK', 'SUBFOLDER')
        s3_client = boto3.client('s3', region_name='us-east-2')

        nearshore_pool = bluetopo_tiles[bluetopo_tiles['_is_nearshore']]
        offshore_pool = bluetopo_tiles[bluetopo_tiles['_is_offshore']]

        intersecting_nearshore = gpd.sjoin(nearshore_pool, eco_geom, predicate='intersects')
        intersecting_offshore = gpd.sjoin(offshore_pool, eco_geom, predicate='intersects')
        
        with tempfile.TemporaryDirectory() as scratch_dir:
            scratch_path = pathlib.Path(scratch_dir)

            # Nearshore
            if not intersecting_nearshore.empty:
                self.write_message(f" - [{ecoregion_id}] starting nearshore", outputs)
                clean_nearshore = intersecting_nearshore.drop(columns=['index_right'], errors='ignore')
                clean_nearshore_utm = clean_nearshore.to_crs(self.target_crs)
                
                local_tif_path = scratch_path / 'nearshore_mask.tif'
                _rasterize_tile_worker(
                    intersected_tiles=clean_nearshore_utm,
                    bounds=clean_nearshore_utm.total_bounds, 
                    crs=self.target_crs,
                    resolution=self.nearshore_res,
                    output_path=local_tif_path
                )
                
                s3_key_tif = f"{ecoregion_id}/{mask_sub}/nearshore_mask.tif"
                if self.param_lookup['env'] == 'aws':
                    s3_client.upload_file(str(local_tif_path), get_config_item('SHARED', 'OUTPUT_BUCKET'), s3_key_tif)
                else:
                    os.makedirs(str(OUTPUTS / f"{ecoregion_id}/{mask_sub}"), exist_ok=True)
                    shutil.copy(str(local_tif_path), str(OUTPUTS / s3_key_tif))
            else:
                self.write_message(f"  No nearshore tiles for EcoRegion {ecoregion_id}", outputs)
                
            # Offshore
            if not intersecting_offshore.empty:
                self.write_message(f" - [{ecoregion_id}] starting offshore", outputs)
                clean_offshore = intersecting_offshore.drop(columns=['index_right'], errors='ignore')
                clean_offshore_utm = clean_offshore.to_crs(self.target_crs)
                
                local_tif_path = scratch_path / 'offshore_mask.tif'
                _rasterize_tile_worker(
                    intersected_tiles=clean_offshore_utm,
                    bounds=clean_offshore_utm.total_bounds, 
                    crs=self.target_crs,
                    resolution=self.offshore_res,
                    output_path=local_tif_path
                )
                
                s3_key_tif = f"{ecoregion_id}/{mask_sub}/offshore_mask.tif"
                if self.param_lookup['env'] == 'aws':
                    s3_client.upload_file(str(local_tif_path), get_config_item('SHARED', 'OUTPUT_BUCKET'), s3_key_tif)
                else:
                    os.makedirs(str(OUTPUTS / f"{ecoregion_id}/{mask_sub}"), exist_ok=True)
                    shutil.copy(str(local_tif_path), str(OUTPUTS / s3_key_tif))
            else:
                self.write_message(f"  No offshore tiles for EcoRegion {ecoregion_id}", outputs)
            
        return f"{ecoregion_id}: Shoreline tile compilation and S3 push completed."

    def run(self, outputs: str) -> None:
        """Procedural entry loop orchestration."""

        gpkg_path = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
        
        bluetopo_tiles = self.load_and_classify_tiles(gpkg_path, outputs)
        ecoregions = self.load_and_align_ecoregions(gpkg_path, bluetopo_tiles.crs, outputs)

        for index, row in ecoregions.iterrows():
            ecoregion_id = str(row['EcoRegion'])
            self.write_message(f" - Processing EcoRegion: {ecoregion_id}", outputs)
            result_log = self.process_ecoregion_tiles(ecoregion_id, row, bluetopo_tiles, outputs)
            self.write_message(result_log, outputs)
            
        self.write_message("Nearshore and Offshore masks finished", outputs)


if __name__ == "__main__":
    engine = ShoreTileS3Engine({'env': get_environment()})
    engine.run(OUTPUTS)