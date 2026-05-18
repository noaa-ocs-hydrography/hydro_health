import pathlib
import sys
import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_bounds

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.helpers.tools import Param, get_config_item

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'

# --- Configuration Constants ---
TARGET_CRS = "EPSG:32617"  # UTM Zone 17N (units are meters)
NODATA_VALUE = 0

# --- Resolution Nuances (Meters) ---
NEARSHORE_RES = 20
OFFSHORE_RES = 100


def create_raster_mask(intersected_tiles, bounds, crs, resolution, output_path):
    """
    Creates a binary raster mask in UTM 17N bounded strictly by the selected tiles.
    Accepts a dynamic resolution argument to handle nearshore vs offshore requirements.
    Areas covered by the tile polygons will be 1, background will be 0.
    """
    minx, miny, maxx, maxy = bounds
    
    # Calculate pixel dimensions dynamically based on the passed resolution
    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    
    if width <= 0 or height <= 0:
        print(f"Skipping {output_path.name}: Width/height calculated as 0.")
        return False

    # Create the affine transform matrix matching the UTM bounds and specified resolution
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    
    # Initialize the canvas entirely filled with 0s
    mask = np.zeros((height, width), dtype=rasterio.uint8)
    
    # Convert GeoSeries to (geometry, value) tuples for reliable burning
    shapes_to_burn = [(geom, 1) for geom in intersected_tiles.geometry if geom is not None]
    
    if not shapes_to_burn:
        print(f"Warning: No valid geometries found to burn for {output_path.name}")
    else:
        # Burn the tile geometries into the canvas
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
        'nodata': NODATA_VALUE,
        'width': width,
        'height': height,
        'count': 1,
        'crs': crs,
        'transform': transform,
        'compress': 'lzw'
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Safely clear disk file before rewriting
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass

    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask, 1)
        
    return True


def run():
    master_grid_geopackage = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
    
    print("Loading BlueTopo tiles...")
    bluetopo_tiles = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'))
    
    nearshore_mask = bluetopo_tiles['tile'].str.startswith('BH', na=False)
    offshore_mask = (~nearshore_mask) & (~bluetopo_tiles['tile'].str.match(r'^BF2H[0-9]', na=False))
    
    nearshore_tiles = bluetopo_tiles[nearshore_mask]
    offshore_tiles = bluetopo_tiles[offshore_mask]
    
    print("Loading ecoregions...")
    ecoregions = gpd.read_file(master_grid_geopackage, layer="EcoRegions")
    
    # Align initial vector inputs to the same native reference system for spatial join
    if ecoregions.crs != bluetopo_tiles.crs:
        print("Aligning coordinate reference systems for spatial matching...")
        ecoregions = ecoregions.to_crs(bluetopo_tiles.crs)

    for index, row in ecoregions.iterrows():
        ecoregion_id = str(row['EcoRegion'])
        print(f"Processing EcoRegion: {ecoregion_id}...")
        
        eco_geom = gpd.GeoDataFrame([row], crs=ecoregions.crs)
        
        intersecting_nearshore = gpd.sjoin(nearshore_tiles, eco_geom, predicate='intersects')
        intersecting_offshore = gpd.sjoin(offshore_tiles, eco_geom, predicate='intersects')
        
        ecoregion_out_dir = OUTPUTS / ecoregion_id
        ecoregion_out_dir.mkdir(parents=True, exist_ok=True) 
        
        # --- Handle Nearshore (20m Resolution) ---
        if not intersecting_nearshore.empty:
            nearshore_shp_path = ecoregion_out_dir / 'nearshore_tiles.shp'
            clean_nearshore = intersecting_nearshore.drop(columns=['index_right'], errors='ignore')
            
            clean_nearshore_utm = clean_nearshore.to_crs(TARGET_CRS)
            # clean_nearshore_utm.to_file(nearshore_shp_path, driver='ESRI Shapefile')
            
            nearshore_tile_bounds = clean_nearshore_utm.total_bounds
            nearshore_out_path = ecoregion_out_dir / 'nearshore_mask.tif'
            
            # --- UPDATED: Passing NEARSHORE_RES (20m) ---
            create_raster_mask(
                intersected_tiles=clean_nearshore_utm,
                bounds=nearshore_tile_bounds, 
                crs=TARGET_CRS,
                resolution=NEARSHORE_RES,
                output_path=nearshore_out_path
            )
        else:
            print(f"  No nearshore tiles found intersecting EcoRegion {ecoregion_id}")
            
        # --- Handle Offshore (100m Resolution) ---
        if not intersecting_offshore.empty:
            offshore_shp_path = ecoregion_out_dir / 'offshore_tiles.shp'
            clean_offshore = intersecting_offshore.drop(columns=['index_right'], errors='ignore')
            
            clean_offshore_utm = clean_offshore.to_crs(TARGET_CRS)
            # clean_offshore_utm.to_file(offshore_shp_path, driver='ESRI Shapefile')
            
            offshore_tile_bounds = clean_offshore_utm.total_bounds
            offshore_out_path = ecoregion_out_dir / 'offshore_mask.tif'
            
            # --- UPDATED: Passing OFFSHORE_RES (100m) ---
            create_raster_mask(
                intersected_tiles=clean_offshore_utm,
                bounds=offshore_tile_bounds, 
                crs=TARGET_CRS,
                resolution=OFFSHORE_RES,
                output_path=offshore_out_path
            )
        else:
            print(f"  No offshore tiles found intersecting EcoRegion {ecoregion_id}")
            
    print("Processing complete!")
    return


if __name__ == "__main__":
    run()