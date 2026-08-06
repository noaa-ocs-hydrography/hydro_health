import os
import glob
import numpy as np
import rioxarray
import geopandas as gpd

# Define input and output directories
input_folder = r"C:\Users\aubrey.mccutchan\Documents\tifs_to_mask"
output_folder = r"C:\Users\aubrey.mccutchan\Documents\tifs_to_mask_output"

# Define the Geopackage path and layer for masking
gpkg_path = r"C:\Users\aubrey.mccutchan\Documents\Repo\hydro_health\inputs\Master_Grids.gpkg"
layer_name = "Enhanced_EcoRegions"

# Create the new output folder
os.makedirs(output_folder, exist_ok=True)

# Find all files ending in .tif or .tiff
search_pattern_tif = os.path.join(input_folder, "*.tif")
search_pattern_tiff = os.path.join(input_folder, "*.tiff")

# Combine both lists and remove duplicates
files_to_process = list(set(glob.glob(search_pattern_tif) + glob.glob(search_pattern_tiff)))

# Target parameters
target_crs = "EPSG:6350"
target_resolution = 100

print("Loading masking polygons...")
# Read the GPKG layer and ensure it matches the target CRS
gdf_mask = gpd.read_file(gpkg_path, layer=layer_name)
gdf_mask = gdf_mask.to_crs(target_crs)

print(f"Found {len(files_to_process)} files to process.")

for file_path in files_to_process:
    # Get the exact original file name
    original_filename = os.path.basename(file_path)
    
    # Ensure the extension is strictly .tiff without changing the base name at all
    if original_filename.lower().endswith('.tif'):
        new_filename = original_filename[:-4] + ".tiff"
    else:
        new_filename = original_filename
        
    out_path = os.path.join(output_folder, new_filename)
    
    # Check if the output file already exists to avoid duplicate work
    if os.path.exists(out_path):
        print(f"  - Skipping: {new_filename} (already exists)")
        continue
    
    print(f"Processing: {original_filename} -> {new_filename}...")
    
    try:
        # 1. Open the raster with chunking enabled
        ds = rioxarray.open_rasterio(file_path, chunks={'x': 1024, 'y': 1024})
        
        # --- AGGRESSIVE NODATA SANITIZATION ---
        current_nodata = ds.rio.nodata
        is_float = np.issubdtype(ds.dtype, np.floating)
        
        # If the raster is decimal (float), use np.nan. If integer, use -9999.
        if is_float:
            clean_nodata = np.nan
            # 1a. If a nodata value exists, replace those pixels with NaN
            if current_nodata is not None:
                ds = ds.where(ds != current_nodata, clean_nodata)
            # 1b. Catch untagged GDAL float minimums (e.g. -3.4e38) and force them to NaN
            ds = ds.where(ds > -1e37, clean_nodata)
        else:
            clean_nodata = current_nodata if current_nodata is not None else -9999
            if current_nodata is not None:
                ds = ds.where(ds != current_nodata, clean_nodata)
            # Catch untagged extreme negative integers
            ds = ds.where(ds > -2147483600, clean_nodata)
            
        # Tag the dataset explicitly with the clean nodata value
        ds = ds.rio.write_nodata(clean_nodata, encoded=True)
        
        # 2. Reproject to EPSG:6350 and resample to exactly 100m resolution
        # Pass the clean_nodata directly so reprojection fill uses it
        ds_reprojected = ds.rio.reproject(
            target_crs, 
            resolution=target_resolution,
            nodata=clean_nodata
        )
        
        # 3. Mask the raster using the Geopackage polygons
        ds_masked = ds_reprojected.rio.clip(
            gdf_mask.geometry, 
            gdf_mask.crs, 
            drop=True
        )
        
        # 4. Write out the new file
        ds_masked.rio.to_raster(
            out_path, 
            tiled=True, 
            windowed=True,
            compress='LZW',
            profile_kwargs={'nodata': clean_nodata} # Force the header to recognize this
        )
        
        print(f"  ✓ Successfully saved: {new_filename}")
        
    except Exception as e:
        print(f"  X Error processing {file_path}: {e}")

print("Processing complete!")