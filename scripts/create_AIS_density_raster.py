import os
import fiona
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shutil
import concurrent.futures
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.enums import MergeAlg, Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, reproject
from shapely.geometry import box

# ==============================================================================
# PARALLEL WORKER FUNCTIONS (Must be at top-level to be pickled by multiprocessing)
# ==============================================================================

def _worker_density(layer, input_gdb, final_crs_wkt, mask_bounds, mask_crs_wkt, temp_dir, out_meta, height, width, transform, block_size):
    """Worker process for reading, filtering, and rasterizing a single month's density data."""
    try:
        print(f"  [Worker Started] Processing Density for Month: {layer}")
        with fiona.open(input_gdb, layer=layer) as src:
            layer_crs = src.crs
        
        # Prepare native bounding box
        mask_box_gdf = gpd.GeoDataFrame({'geometry': [box(*mask_bounds)]}, crs=mask_crs_wkt)
        native_bounds = tuple(mask_box_gdf.to_crs(layer_crs).total_bounds) if layer_crs else mask_bounds
        
        # Read the geodatabase limited to the bounding box
        gdf = gpd.read_file(input_gdb, layer=layer, bbox=native_bounds)
        
        if gdf.empty:
            print(f"  > [Worker {layer}] No tracks found within bounding box. Skipping.")
            return None
            
        if gdf.crs != final_crs_wkt:
            gdf = gdf.to_crs(final_crs_wkt)

        if 'Width' in gdf.columns:
            gdf['Width'] = pd.to_numeric(gdf['Width'], errors='coerce')
            gdf['Width'] = gdf['Width'].apply(lambda x: 1.0 if pd.isna(x) or x <= 0 else x)
            gdf['Width_m'] = gdf['Width']
        else:
            gdf['Width_m'] = 1.0

        sindex = gdf.sindex
        monthly_tif_path = os.path.join(temp_dir, f"{layer}.tiff")
        
        # Stream rasterization to disk
        with rasterio.open(monthly_tif_path, "w", **out_meta) as dest:
            for row in range(0, height, block_size):
                for col in range(0, width, block_size):
                    win_height = min(block_size, height - row)
                    win_width = min(block_size, width - col)
                    window = Window(col, row, win_width, win_height)
                    
                    win_transform = rasterio.windows.transform(window, transform)
                    win_bounds = rasterio.windows.bounds(window, transform)
                    
                    possible_matches_index = list(sindex.intersection(win_bounds))
                    
                    if not possible_matches_index:
                        dest.write(np.zeros((win_height, win_width), dtype=np.float32), 1, window=window)
                        continue
                        
                    possible_matches = gdf.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(box(*win_bounds))]
                    
                    if precise_matches.empty:
                        dest.write(np.zeros((win_height, win_width), dtype=np.float32), 1, window=window)
                        continue
                        
                    shapes = ((geom, val) for geom, val in zip(precise_matches.geometry, precise_matches['Width_m']))
                    burned = rasterize(
                        shapes=shapes,
                        out_shape=(win_height, win_width),
                        transform=win_transform,
                        fill=0,
                        all_touched=True,
                        merge_alg=MergeAlg.add,
                        dtype=np.float32
                    )
                    
                    dest.write(burned, 1, window=window)
                    
        print(f"  [Worker Finished] Month {layer} completed successfully.")
        return monthly_tif_path
    
    except Exception as e:
        print(f"  > [Worker {layer}] FATAL ERROR: {e}")
        return None

def _worker_draft(layer, input_gdb, orig_bounds, orig_crs_wkt, final_crs_wkt, temp_dir, out_meta, height, width, transform, block_size):
    """Worker process for reading, filtering, and rasterizing a single month's max draft data."""
    try:
        print(f"  [Worker Started] Processing Max Draft for Month: {layer}")
        with fiona.open(input_gdb, layer=layer) as src:
            layer_crs = src.crs
        
        # Safely convert bounds for reading
        if orig_crs_wkt and layer_crs:
            orig_box_gdf = gpd.GeoDataFrame({'geometry': [box(*orig_bounds)]}, crs=orig_crs_wkt)
            read_bounds = tuple(orig_box_gdf.to_crs(layer_crs).total_bounds)
        else:
            read_bounds = orig_bounds

        gdf = gpd.read_file(input_gdb, layer=layer, bbox=read_bounds)
        
        if gdf.empty:
            print(f"  > [Worker {layer}] No tracks found within bounding box. Skipping.")
            return None

        if gdf.crs != final_crs_wkt:
            gdf = gdf.to_crs(final_crs_wkt)

        # Handle Draft Column
        if 'Draft' in gdf.columns:
            if gdf['Draft'].dtype == object or pd.api.types.is_string_dtype(gdf['Draft']):
                gdf['Draft'] = gdf['Draft'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
            
            gdf['Draft'] = pd.to_numeric(gdf['Draft'], errors='coerce').fillna(0)
            gdf = gdf[gdf['Draft'] > 0] # Filter out missing/zero drafts
        else:
            print(f"  > [Worker {layer}] No 'Draft' column found. Skipping.")
            return None

        if gdf.empty:
            print(f"  > [Worker {layer}] No valid draft data (>0) found after filtering. Skipping.")
            return None

        # Sort ascending so the rasterize engine burns larger values LAST, overwriting smaller ones
        gdf = gdf.sort_values(by='Draft', ascending=True)

        sindex = gdf.sindex
        monthly_tif_path = os.path.join(temp_dir, f"{layer}_max_draft.tiff")
        
        with rasterio.open(monthly_tif_path, "w", **out_meta) as dest:
            for row in range(0, height, block_size):
                for col in range(0, width, block_size):
                    win_height = min(block_size, height - row)
                    win_width = min(block_size, width - col)
                    window = Window(col, row, win_width, win_height)
                    
                    win_transform = rasterio.windows.transform(window, transform)
                    win_bounds = rasterio.windows.bounds(window, transform)
                    
                    possible_matches_index = list(sindex.intersection(win_bounds))
                    
                    if not possible_matches_index:
                        dest.write(np.zeros((win_height, win_width), dtype=np.float32), 1, window=window)
                        continue
                        
                    possible_matches = gdf.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(box(*win_bounds))]
                    
                    if precise_matches.empty:
                        dest.write(np.zeros((win_height, win_width), dtype=np.float32), 1, window=window)
                        continue
                        
                    shapes = ((geom, val) for geom, val in zip(precise_matches.geometry, precise_matches['Draft']))
                    burned = rasterize(
                        shapes=shapes,
                        out_shape=(win_height, win_width),
                        transform=win_transform,
                        fill=0,
                        all_touched=True,
                        merge_alg=MergeAlg.replace, 
                        dtype=np.float32
                    )
                    
                    dest.write(burned, 1, window=window)
                    
        print(f"  [Worker Finished] Month {layer} completed successfully.")
        return monthly_tif_path

    except Exception as e:
        print(f"  > [Worker {layer}] FATAL ERROR: {e}")
        return None

# ==============================================================================
# MAIN GEOPROCESSING LOGIC
# ==============================================================================

def get_wkt(crs):
    """Safely converts a CRS object to its WKT or string representation."""
    if crs is None: return None
    return crs.to_wkt() if hasattr(crs, 'to_wkt') else str(crs)

def process_density_raster(input_gdb, mask_tif, output_raster, isobath_shp, temp_dir, max_cutoff=1000, target_crs=None, max_workers=8):
    """
    Processes the AIS geodatabase into a density raster matching the bounds/resolution 
    of a specific mask_tif, clips to the provided isobath shapefile, and outputs 
    only values <= the specified max_cutoff using concurrent multiprocessing.
    """
    os.makedirs(temp_dir, exist_ok=True)

    # 1. READ MASK FOR ENVIRONMENT SETTINGS
    print(f"\n=======================================================")
    print(f"Processing target: {os.path.basename(output_raster)}")
    with rasterio.open(mask_tif) as src:
        transform = src.transform
        width = src.width
        height = src.height
        mask_crs = src.crs
        mask_bounds = tuple(src.bounds) 
        
    final_crs = target_crs or mask_crs
    final_crs_wkt = get_wkt(final_crs)
    mask_crs_wkt = get_wkt(mask_crs)

    print(f"Target Resolution: {transform[0]}m x {-transform[4]}m")
    print(f"Target CRS: {final_crs}")
    
    # 2. READ ISOBATH POLYGON FOR CLIPPING
    print(f"Loading and preparing 50m isobath clipping mask...")
    isobath_gdf = gpd.read_file(isobath_shp)
    if isobath_gdf.crs != final_crs:
        isobath_gdf = isobath_gdf.to_crs(final_crs)
    isobath_sindex = isobath_gdf.sindex

    # 3. METADATA & CHUNKING SETTINGS
    final_nodata_value = -9999.0
    block_size = 4096 
    
    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "transform": transform,
        "crs": final_crs_wkt, # Passing wkt string for serialization safety
        "count": 1,
        "dtype": 'float32',
        "nodata": final_nodata_value,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 1024,
        "blockysize": 1024
    }

    # 4. PROCESS MONTHLY RASTERS (PARALLEL MULTIPROCESSING)
    all_layers = fiona.listlayers(input_gdb)
    ais_layers = [layer for layer in all_layers if layer.startswith("AIS_2024_")]
    valid_monthly_tifs = []

    if not ais_layers:
        print(f"No layers found in {input_gdb} matching 'AIS_2024_*'")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    print(f"Dispatching tasks across {max_workers} CPU cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for layer in ais_layers:
            # Submit worker arguments
            futures.append(executor.submit(
                _worker_density, 
                layer, input_gdb, final_crs_wkt, mask_bounds, mask_crs_wkt, 
                temp_dir, out_meta, height, width, transform, block_size
            ))
            
        for future in concurrent.futures.as_completed(futures):
            result_path = future.result()
            if result_path is not None:
                valid_monthly_tifs.append(result_path)

    # 5. AGGREGATE FINAL RASTER & APPLY CUTOFF
    if not valid_monthly_tifs:
        print("\nFAILURE: No valid features found across any month to create the final raster.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    print("\n=== Aggregating Annual Density, Clipping to Isobath, & Applying Cutoff ===")
    src_datasets = [rasterio.open(tif) for tif in valid_monthly_tifs]

    with rasterio.open(output_raster, "w", **out_meta) as dest:
        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                win_height = min(block_size, height - row)
                win_width = min(block_size, width - col)
                window = Window(col, row, win_width, win_height)
                win_transform = rasterio.windows.transform(window, transform)
                win_bounds = rasterio.windows.bounds(window, transform)
                
                chunk_sum = np.zeros((win_height, win_width), dtype=np.float32)
                
                for src in src_datasets:
                    chunk_sum += src.read(1, window=window)
                
                chunk_sum[chunk_sum == 0] = final_nodata_value

                # Apply 50m Isobath Clip
                possible_isobath_idx = list(isobath_sindex.intersection(win_bounds))
                if not possible_isobath_idx:
                    chunk_sum[:] = final_nodata_value
                else:
                    intersecting_isobath = isobath_gdf.iloc[possible_isobath_idx]
                    poly_mask = rasterize(
                        shapes=intersecting_isobath.geometry,
                        out_shape=(win_height, win_width),
                        transform=win_transform,
                        fill=0,             
                        default_value=1,    
                        dtype=np.uint8
                    )
                    chunk_sum[poly_mask == 0] = final_nodata_value
                
                # Apply cutoff
                chunk_sum[(chunk_sum != final_nodata_value) & (chunk_sum > max_cutoff)] = final_nodata_value
                dest.write(chunk_sum, 1, window=window)

    for src in src_datasets:
        src.close()

    print(f"Cleaning up temporary monthly rasters...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"SUCCESS! Annual density raster safely generated:\n{output_raster}")


def process_max_draft_raster(input_gdb, output_raster, bounds_tif, temp_dir, resolution=100.0, target_crs=None, max_workers=8):
    """
    Creates a raster at the specified resolution evaluating the greatest ship 'Draft' 
    for each cell across all months using concurrent multiprocessing.
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # 1. SETUP GRID BASED ON BOUNDS & DYNAMIC RESOLUTION
    print(f"\n=======================================================")
    print(f"Processing Max Draft target: {os.path.basename(output_raster)}")
    
    final_nodata_value = -9999.0
    block_size = 4096 

    try:
        with rasterio.open(bounds_tif) as src:
            orig_crs = src.crs
            orig_bounds = tuple(src.bounds)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to open bounds TIF: {e}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    target_crs = target_crs or orig_crs
    orig_crs_wkt = get_wkt(orig_crs)
    final_crs_wkt = get_wkt(target_crs)

    try:
        if orig_crs and orig_crs != target_crs:
            minx, miny, maxx, maxy = transform_bounds(orig_crs, target_crs, *orig_bounds)
        else:
            minx, miny, maxx, maxy = orig_bounds
    except Exception as e:
        print(f"  > Warning: Safe bounds transformation failed ({e}).")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    minx = np.floor(minx / resolution) * resolution
    miny = np.floor(miny / resolution) * resolution
    maxx = np.ceil(maxx / resolution) * resolution
    maxy = np.ceil(maxy / resolution) * resolution

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "transform": transform,
        "crs": final_crs_wkt, # Passing WKT to ensure workers can read it
        "count": 1,
        "dtype": 'float32',
        "nodata": final_nodata_value,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 1024,
        "blockysize": 1024
    }

    # 2. PROCESS MONTHLY RASTERS (PARALLEL MULTIPROCESSING)
    all_layers = fiona.listlayers(input_gdb)
    ais_layers = [layer for layer in all_layers if layer.startswith("AIS_2024_")]
    valid_monthly_tifs = []

    if not ais_layers:
        print(f"No layers found in {input_gdb} matching 'AIS_2024_*'")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    print(f"Dispatching Max Draft tasks across {max_workers} CPU cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for layer in ais_layers:
            # Submit worker arguments
            futures.append(executor.submit(
                _worker_draft, 
                layer, input_gdb, orig_bounds, orig_crs_wkt, final_crs_wkt, 
                temp_dir, out_meta, height, width, transform, block_size
            ))
            
        for future in concurrent.futures.as_completed(futures):
            result_path = future.result()
            if result_path is not None:
                valid_monthly_tifs.append(result_path)

    # 3. AGGREGATE FINAL MAX DRAFT RASTER & APPLY MASK
    if not valid_monthly_tifs:
        print("\nFAILURE: No valid features found to create the final max draft raster.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    print("\n=== Aggregating Annual Maximum Draft and Applying Source Mask ===")
    src_datasets = [rasterio.open(tif) for tif in valid_monthly_tifs]

    with rasterio.open(bounds_tif) as mask_src:
        mask_nodata = mask_src.nodata

        with rasterio.open(output_raster, "w", **out_meta) as dest:
            for row in range(0, height, block_size):
                for col in range(0, width, block_size):
                    win_height = min(block_size, height - row)
                    win_width = min(block_size, width - col)
                    window = Window(col, row, win_width, win_height)
                    win_transform = rasterio.windows.transform(window, transform)
                    
                    chunk_max = np.zeros((win_height, win_width), dtype=np.float32)
                    
                    for src in src_datasets:
                        chunk_max = np.maximum(chunk_max, src.read(1, window=window))
                    
                    chunk_max[chunk_max == 0] = final_nodata_value

                    # --- MASKING LOGIC ---
                    mask_chunk = np.zeros((win_height, win_width), dtype=np.float32)
                    reproject(
                        source=rasterio.band(mask_src, 1),
                        destination=mask_chunk,
                        src_transform=mask_src.transform,
                        src_crs=mask_src.crs,
                        dst_transform=win_transform,
                        dst_crs=target_crs,
                        resampling=Resampling.nearest
                    )

                    if mask_nodata is not None:
                        is_valid_mask = (mask_chunk == 1) | ((mask_chunk != mask_nodata) & (mask_chunk > 0))
                    else:
                        is_valid_mask = (mask_chunk == 1) | (mask_chunk > 0)

                    chunk_max[~is_valid_mask] = final_nodata_value
                    
                    dest.write(chunk_max, 1, window=window)

    for src in src_datasets:
        src.close()

    print(f"Cleaning up temporary monthly rasters...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"SUCCESS! Annual Maximum Draft raster safely generated:\n{output_raster}")


def main():
    # ---------------------------------------------------------
    # MASTER CONFIGURATION
    # ---------------------------------------------------------
    input_gdb = r"C:\Users\aubrey.mccutchan\Downloads\AIS_2024.gdb\AIS_2024.gdb"
    output_dir = r"C:\Users\aubrey.mccutchan\Documents"
    isobath_shp = r"C:\Users\aubrey.mccutchan\Documents\50m_isobath_polygon\50m_isobath_polygon.shp"
    
    # Updated network mask directory 
    mask_dir = r"C:\Users\aubrey.mccutchan\Documents"
    
    # ---------------------------------------------------------
    # PROCESSING MODES
    # ---------------------------------------------------------
    resolutions_to_run = [20]  # Adjust list to run specific modes (e.g., [20, 100])
    
    max_density_cutoff = 50000
    explicit_crs = "EPSG:6350"
    parallel_cores = 8
    
    er_regions = ["ER_1", "ER_2", "ER_3", "ER_4", "ER_5", "ER_6"]

    # 1. Iterate through the selected resolution modes
    for res in resolutions_to_run:
        print(f"\n\n{'#'*80}")
        print(f"### INITIALIZING PROCESSING MODE: {res}m RESOLUTION ###")
        print(f"{'#'*80}")
        
        # 2. ER Run Loop
        for er_name in er_regions:
            print(f"\n\n{'='*70}")
            print(f"Executing pipeline for: {er_name} at {res}m")
            print(f"{'='*70}")
            
            # Dynamically target the mask based on the current mode
            mask_filename = f"{er_name}_binary_mask_EPSG6350_{res}m.tiff"
            mask_tif = os.path.join(mask_dir, mask_filename)
            
            # Check if mask exists before firing processing logic
            if not os.path.exists(mask_tif):
                print(f"WARNING: Mask file not found -> {mask_tif}")
                print(f"Skipping {er_name} for the {res}m resolution mode.")
                continue
            
            # --- Density Raster Setup ---
            output_raster_density = os.path.join(output_dir, f"AIS_2024_Annual_Density_{er_name}_Max{max_density_cutoff}_{res}m.tiff")
            temp_dir_density = os.path.join(output_dir, f"temp_monthly_rasters_density_{er_name.lower()}_{res}m")
            
            # Uncomment to run density generator
            # process_density_raster(
            #     input_gdb, 
            #     mask_tif, 
            #     output_raster_density, 
            #     isobath_shp, 
            #     temp_dir_density, 
            #     max_cutoff=max_density_cutoff,
            #     target_crs=explicit_crs,
            #     max_workers=parallel_cores
            # )

            # --- Max Draft Raster Setup ---
            output_raster_draft = os.path.join(output_dir, f"{er_name}_AIS_2024_Annual_Max_Draft_{res}m.tiff")
            temp_dir_draft = os.path.join(output_dir, f"temp_monthly_rasters_draft_{er_name.lower()}_{res}m")
            
            process_max_draft_raster(
                input_gdb, 
                output_raster_draft, 
                mask_tif, 
                temp_dir_draft, 
                resolution=float(res),
                target_crs=explicit_crs,
                max_workers=parallel_cores
            )

if __name__ == "__main__":
    # Required for Windows multiprocessing safety
    main()