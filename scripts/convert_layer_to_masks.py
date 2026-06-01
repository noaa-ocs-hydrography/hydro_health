import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
from pathlib import Path
import concurrent.futures
import multiprocessing

# --- Configuration ---
GPKG_PATH = r"C:\Users\aubrey.mccutchan\Documents\Repo\hydro_health\inputs\Master_Grids.gpkg"
LAYER_NAME = "Enhanced_EcoRegions"
OUTPUT_DIR = r"C:\Users\aubrey.mccutchan\Documents"
TARGET_CRS = "EPSG:6350"
RESOLUTIONS = [8, 20, 100]  # in meters
CHUNK_SIZE = 4096  # Processes the raster in 4096x4096 blocks to prevent RAM overload

def process_mask_task(task_args):
    """Worker function to process a single mask in parallel."""
    er_val, res, minx, miny, maxx, maxy, shapes, output_dir, target_crs, chunk_size = task_args
    
    width = max(1, int(np.ceil((maxx - minx) / res)))
    height = max(1, int(np.ceil((maxy - miny) / res)))
    transform = from_origin(minx, maxy, res, res)
    
    # Format target_crs string (e.g. "EPSG:6350" -> "EPSG6350")
    crs_str = target_crs.replace(':', '')
    
    # Updated naming format: ER_1_binary_mask_EPSG6350_20m.tiff
    filename = f"ER_{int(er_val)}_binary_mask_{crs_str}_{res}m.tiff"
    out_path = os.path.join(output_dir, filename)
    
    profile = {
        'driver': 'GTiff',
        'height': height,
        'width': width,
        'count': 1,
        'dtype': 'uint8',
        'crs': target_crs,
        'transform': transform,
        'nodata': 0,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512
    }
    
    with rasterio.open(out_path, 'w', **profile) as dst:
        for row_start in range(0, height, chunk_size):
            for col_start in range(0, width, chunk_size):
                
                # Define the current window/chunk bounds
                w_width = min(chunk_size, width - col_start)
                w_height = min(chunk_size, height - row_start)
                window = Window(col_start, row_start, w_width, w_height)
                
                # Get the affine transform specifically for this window
                win_transform = rasterio.windows.transform(window, transform)
                
                # Rasterize only the chunk
                mask_chunk = rasterize(
                    shapes=shapes,
                    out_shape=(w_height, w_width),
                    transform=win_transform,
                    fill=0,
                    dtype='uint8'
                )
                
                # Write the chunk to disk
                dst.write(mask_chunk, 1, window=window)
                
    return f"  -> Saved: {out_path} ({width}x{height} pixels)"

def create_binary_masks():
    """Reads a Geopackage layer and exports binary masks per EcoRegion and resolution in chunks."""
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading layer '{LAYER_NAME}' from {GPKG_PATH}...")
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)
    
    if gdf.crs != TARGET_CRS:
        print(f"Reprojecting from {gdf.crs} to {TARGET_CRS}...")
        gdf = gdf.to_crs(TARGET_CRS)
    
    unique_ecoregions = gdf['EcoRegion'].unique()
    unique_ecoregions = [er for er in unique_ecoregions if not np.isnan(er)]
    
    print(f"Found {len(unique_ecoregions)} unique EcoRegions: {unique_ecoregions}")
    
    # Build a list of tasks for parallel execution
    tasks = []
    for res in RESOLUTIONS:
        for er_val in unique_ecoregions:
            # Isolate the geometry for the current EcoRegion
            er_gdf = gdf[gdf['EcoRegion'] == er_val]
            minx, miny, maxx, maxy = er_gdf.total_bounds
            shapes = [(geom, 1) for geom in er_gdf.geometry]
            
            # Append arguments as a tuple to pass to our worker function
            tasks.append((er_val, res, minx, miny, maxx, maxy, shapes, OUTPUT_DIR, TARGET_CRS, CHUNK_SIZE))
            
    # Start parallel processing using the process pool
    # By default, subtracting 1 core ensures your computer stays responsive while running
    num_cores = max(1, multiprocessing.cpu_count() - 1) 
    print(f"\nStarting parallel processing for {len(tasks)} tasks using {num_cores} cores...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Map returns results as they finish, processing all tasks concurrently
        for result in executor.map(process_mask_task, tasks):
            print(result)

if __name__ == "__main__":
    create_binary_masks()
    print("\nAll binary masks generated successfully!")