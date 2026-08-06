import os
import glob
import rasterio
from rasterio.features import shapes, geometry_window, rasterize
from rasterio.windows import Window
from rasterio.errors import WindowError
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.crs import CRS
from rasterio.warp import transform_geom

def process_region_tifs(resolution):
    # File paths based on user specifications - dynamic based on resolution mode
    tifs_dir = r"C:\Users\aubrey.mccutchan\Documents\tifs_to_mask"
    
    if resolution == 100:
        output_dir = r"C:\Users\aubrey.mccutchan\Documents\Offshore_tiles_100m"
        mask_name_base = "offshore_mask"
    elif resolution == 20:
        output_dir = r"C:\Users\aubrey.mccutchan\Documents\Nearshore_tiles_20m"
        mask_name_base = "nearshore_mask"
    else:
        print(f"Unsupported resolution: {resolution}")
        return
    
    # Target CRS and resolution specifications
    target_crs = CRS.from_epsg(6350)
    target_res = (resolution, resolution)
    
    # Get all .tif and .tiff files from the input directory
    tifs_to_mask = []
    for ext in ('*.tif', '*.tiff', '*.TIF', '*.TIFF'):
        tifs_to_mask.extend(glob.glob(os.path.join(tifs_dir, ext)))
        
    tifs_to_mask = list(set(tifs_to_mask))
    
    if not tifs_to_mask:
        print(f"No TIFF files found in {tifs_dir}")
        return

    print(f"\n========================================")
    print(f" STARTING {resolution}m RESOLUTION MODE")
    print(f"========================================")

    for er_num in range(1, 7):
        er_folder = os.path.join(output_dir, f"ER_{er_num}")
        
        mask_path = os.path.join(er_folder, f"{mask_name_base}.tif")
        if not os.path.exists(mask_path):
            mask_path = os.path.join(er_folder, f"{mask_name_base}.tiff")
            
        if not os.path.exists(mask_path):
            print(f"Mask not found for ER_{er_num} in {er_folder}. Skipping.")
            continue
            
        print(f"\nProcessing ER_{er_num} ({resolution}m)...")
        
        # 1. Read the mask and extract valid geometries
        mask_shapes = []
        with rasterio.open(mask_path) as m_src:
            mask_data = m_src.read(1)
            
            if m_src.nodata is not None:
                valid_pixels = mask_data != m_src.nodata
            else:
                valid_pixels = mask_data > 0
                
            for geom, val in shapes(mask_data, mask=valid_pixels, transform=m_src.transform):
                if m_src.crs != target_crs:
                    geom = transform_geom(m_src.crs, target_crs, geom)
                mask_shapes.append(geom)
                
            if not mask_shapes:
                print(f"  No valid masking area found in {mask_path}. Skipping ER_{er_num}.")
                continue

        # 2. Process every input tif against this ER mask
        for tif_path in tifs_to_mask:
            filename = os.path.basename(tif_path)
            name, _ = os.path.splitext(filename)
            
            # Name formatting logic
            name_lower = name.lower()
            if "ukc" in name_lower or "hurricane" in name_lower:
                if name_lower.startswith("bluetopo"):
                    name = name[8:].lstrip("_- ")
            
            res_str = f"{resolution}m"
            if res_str not in name.lower():
                name = f"{name}_{res_str}"
            
            out_filename = f"{name}.tiff"
            out_path = os.path.join(er_folder, out_filename)
            
            # Skip if the file already exists
            if os.path.exists(out_path):
                print(f"  -> Skipping {out_filename} (Already exists)")
                continue
            
            print(f"  -> Cropping and masking to {out_filename} (Chunked)...")
            
            with rasterio.open(tif_path) as src:
                # 3. Use WarpedVRT with a memory limit to stabilize large file processing
                vrt_options = {
                    'crs': target_crs,
                    'resampling': Resampling.bilinear,
                    'resolution': target_res,
                    'warp_mem_limit': 2048 # Give GDAL up to 2GB RAM for internal warping
                }
                
                with WarpedVRT(src, **vrt_options) as vrt:
                    try:
                        # 4. Calculate the overlapping bounding box to avoid loading out-of-bounds data
                        full_vrt_window = Window(0, 0, vrt.width, vrt.height)
                        try:
                            geom_window = geometry_window(vrt, mask_shapes)
                            window = geom_window.intersection(full_vrt_window)
                        except WindowError:
                            print(f"     Skipping {out_filename}: Does not overlap with ER_{er_num} mask.")
                            continue
                            
                        # Snap to whole pixels
                        window = window.round_lengths().round_offsets()
                        out_height, out_width = int(window.height), int(window.width)
                        
                        if out_height <= 0 or out_width <= 0:
                            print(f"     Skipping {out_filename}: No valid overlap area.")
                            continue
                            
                        out_transform = vrt.window_transform(window)
                        
                        # 5. Prepare Output Metadata optimized for chunking
                        out_meta = vrt.meta.copy()
                        nodata_val = out_meta.get('nodata', 0) # Fallback to 0 if None
                        
                        out_meta.update({
                            "driver": "GTiff",
                            "height": out_height,
                            "width": out_width,
                            "transform": out_transform,
                            "crs": target_crs,
                            "compress": "lzw",
                            "tiled": True,        # Enable chunking/tiling
                            "blockxsize": 1024,   # Read/Write 1024x1024 pixel blocks
                            "blockysize": 1024,
                            "nodata": nodata_val
                        })
                        
                        # 6. Read, Mask, and Write in Blocks (Low Memory Footprint)
                        with rasterio.open(out_path, "w", **out_meta) as dest:
                            for ji, sub_window in dest.block_windows(1):
                                # Map the output block's coordinates back to the VRT's coordinate space
                                vrt_window = Window(
                                    col_off=window.col_off + sub_window.col_off,
                                    row_off=window.row_off + sub_window.row_off,
                                    width=sub_window.width,
                                    height=sub_window.height
                                )
                                
                                # Pull the reprojected data block lazily from the VRT without boundless=True
                                chunk_data = vrt.read(window=vrt_window)
                                chunk_transform = vrt.window_transform(vrt_window)
                                
                                # Create a boolean mask of the geometries for just this 1024x1024 block
                                chunk_mask = rasterize(
                                    mask_shapes,
                                    out_shape=(sub_window.height, sub_window.width),
                                    transform=chunk_transform,
                                    fill=0,           # 0 = outside geometry
                                    default_value=1,  # 1 = inside geometry
                                    dtype='uint8'
                                )
                                
                                # Apply the mask (replace pixels outside geometry with nodata)
                                if chunk_data.ndim == 3:
                                    for i in range(chunk_data.shape[0]):
                                        chunk_data[i][chunk_mask == 0] = nodata_val
                                else:
                                    chunk_data[chunk_mask == 0] = nodata_val
                                
                                # Save the processed block
                                dest.write(chunk_data, window=sub_window)
                                
                    except Exception as e:
                        print(f"     Error processing {out_filename}: {e}")

def main():
    resolutions = [100, 20]
    for res in resolutions:
        process_region_tifs(res)
        
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()