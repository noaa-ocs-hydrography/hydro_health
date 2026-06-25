import os
import warnings
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, geometry_mask
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.errors import NotGeoreferencedWarning
import concurrent.futures

def generate_masks():
    # --- Configuration and Paths ---
    gpkg_path = r"C:\Users\aubrey.mccutchan\Documents\Repo\hydro_health\inputs\Master_Grids.gpkg"
    
    # Layer names (often GeoPackages strip the 'main.' prefix depending on the software that created it. 
    # We will define them here to easily fall back if needed).
    tiles_layer_name = "main.BlueTopo_Tile_Scheme_20260505_132642"
    tiles_layer_fallback = "BlueTopo_Tile_Scheme_20260505_132642"
    
    eco_layer_name = "main.Enhanced_EcoRegions"
    eco_layer_fallback = "Enhanced_EcoRegions"

    out_dir = r"C:\Users\aubrey.mccutchan\Documents\Repo\hydro_health\inputs"
    target_crs = "EPSG:6350"
    nodata_val = 255 # Value for pixels outside the ecoregion mask

    print(f"Loading data from: {gpkg_path}...")

    # --- 1. Load Data ---
    # Try loading BlueTopo Tiles
    try:
        tiles_gdf = gpd.read_file(gpkg_path, layer=tiles_layer_name)
    except Exception as e:
        print(f"Failed to load '{tiles_layer_name}'. Trying fallback name...")
        tiles_gdf = gpd.read_file(gpkg_path, layer=tiles_layer_fallback)

    # Try loading Enhanced Ecoregions
    try:
        eco_gdf = gpd.read_file(gpkg_path, layer=eco_layer_name)
    except Exception as e:
        print(f"Failed to load '{eco_layer_name}'. Trying fallback name...")
        eco_gdf = gpd.read_file(gpkg_path, layer=eco_layer_fallback)

    # --- 2. Reproject to EPSG:6350 ---
    print(f"Reprojecting layers to {target_crs}...")
    if tiles_gdf.crs != target_crs:
        tiles_gdf = tiles_gdf.to_crs(target_crs)
    
    if eco_gdf.crs != target_crs:
        eco_gdf = eco_gdf.to_crs(target_crs)

    # --- 3. Assign Burn Values ---
    print("Assigning binary values (0 for BH4/BH5, 1 for others)...")
    # Identify rows where 'tile' contains BH4 or BH5
    condition = tiles_gdf['tile'].str.contains('BH4|BH5', case=False, na=False)
    
    # Create a new column 'burn_val'. condition=True gets 0, False gets 1
    tiles_gdf['burn_val'] = np.where(condition, 0, 1)

    # --- 4. Determine Raster Extent based on Ecoregions ---
    minx, miny, maxx, maxy = eco_gdf.total_bounds
    
    # Build spatial indexes for faster chunking queries
    print("Building spatial indexes for parallel processing...")
    tiles_sidx = tiles_gdf.sindex
    eco_sidx = eco_gdf.sindex
    
    # Helper function to generate rasters
    def create_raster(resolution, out_filename):
        print(f"\nProcessing {resolution}m resolution with chunking and parallelization...")
        
        # Calculate matrix dimensions based on bounding box and resolution
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))
        
        # Create affine transform 
        transform = from_bounds(minx, miny, maxx, maxy, width, height)
        
        print(f"Raster dimensions: Width={width}, Height={height}")

        out_path = os.path.join(out_dir, out_filename)
        
        # Profile with Tiling and Compression
        profile = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': rasterio.uint8,
            'crs': target_crs,
            'transform': transform,
            'nodata': nodata_val,
            'tiled': True,
            'blockxsize': 1024,
            'blockysize': 1024,
            'compress': 'deflate',  # DEFLATE is excellent for binary/categorical masks
            'zlevel': 6             # Compression level
        }
        
        chunk_size = 4096 # Process 4096x4096 pixels at a time
        
        windows = []
        for col_off in range(0, width, chunk_size):
            for row_off in range(0, height, chunk_size):
                w = min(chunk_size, width - col_off)
                h = min(chunk_size, height - row_off)
                windows.append(Window(col_off, row_off, w, h))
                
        def process_window(window):
            # GDAL requires a thread-local environment when running in parallel
            with rasterio.Env():
                # Suppress the harmless NotGeoreferencedWarning during in-memory rasterization
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                    
                    win_transform = rasterio.windows.transform(window, transform)
                    win_bounds = rasterio.windows.bounds(window, transform)
                    
                    # Intersect bounds with spatial index
                    eco_idx = list(eco_sidx.intersection(win_bounds))
                    
                    # If chunk is entirely outside ecoregions, return nodata immediately
                    if not eco_idx:
                        return window, np.full((window.height, window.width), nodata_val, dtype=np.uint8)
                        
                    local_eco = eco_gdf.iloc[eco_idx]
                    
                    tiles_idx = list(tiles_sidx.intersection(win_bounds))
                    
                    if not tiles_idx:
                        # If inside ecoregion but no tiles present, default to NoData
                        rasterized_tiles = np.full((window.height, window.width), nodata_val, dtype=np.uint8)
                    else:
                        local_tiles = tiles_gdf.iloc[tiles_idx]
                        shapes = ((geom, val) for geom, val in zip(local_tiles.geometry, local_tiles.burn_val))
                        rasterized_tiles = rasterize(
                            shapes=shapes,
                            out_shape=(window.height, window.width),
                            transform=win_transform,
                            fill=nodata_val,
                            dtype=np.uint8
                        )
                    
                    # Eco mask (invert=True means inside polygon is True)
                    eco_mask = geometry_mask(
                        geometries=local_eco.geometry,
                        out_shape=(window.height, window.width),
                        transform=win_transform,
                        invert=True 
                    )
                    
                    # Apply mask
                    chunk_array = np.where(eco_mask, rasterized_tiles, nodata_val)
                    return window, chunk_array

        print(f"Divided into {len(windows)} chunks. Starting parallel processing...")
        
        with rasterio.open(out_path, 'w', **profile) as dst:
            # ThreadPoolExecutor is used because rasterize/GDAL releases the GIL,
            # and it avoids the high overhead of pickling large GeoDataFrames.
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                futures = {executor.submit(process_window, w): w for w in windows}
                
                for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                    window, data = future.result()
                    dst.write(data, 1, window=window)
                    
                    # Progress indicator
                    if i % max(1, len(windows) // 10) == 0 or i == len(windows):
                        print(f"  Progress: {i}/{len(windows)} chunks processed ({(i/len(windows))*100:.1f}%)")
                        
        print(f"Finished {resolution}m raster and saved to {out_path}.")

    # --- 5. Execute for both 20m and 100m ---
    create_raster(20, "mask_20m_epsg6350.tif")
    create_raster(100, "mask_100m_epsg6350.tif")
    
    print("\nAll processing complete!")

if __name__ == "__main__":
    generate_masks()