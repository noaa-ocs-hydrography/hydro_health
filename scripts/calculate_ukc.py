import os
import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.merge import merge
import boto3
from botocore.exceptions import ClientError
import gc

# ==============================================================================
# 1. FILE PATHS & S3 CONFIGURATION (Global)
# ==============================================================================
working_dir = os.getcwd() 

# Base folder on EC2 where draft tifs are located
ais_folder = "/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/AIS data"

# S3 Configuration
S3_BUCKET = "ocs-dev-csdl-hydrohealth" 

# Initialize boto3 S3 client
s3_client = boto3.client('s3')

# List of regions to process
er_regions = ["ER_1", "ER_2", "ER_3", "ER_4", "ER_5", "ER_6"]

# --- CONFIGURATIONS TO RUN ---
# Added 'bathy_tag' to explicitly define the S3 bathy file name variation
processing_configs = [
    # {"res": "20m", "suffix": "_nearshore", "bathy_tag": "", "upload_regional": True, "master_bathy": False},
    {"res": "100m", "suffix": "_offshore", "bathy_tag": "_Offshore", "upload_regional": False, "master_bathy": True}
    # {"res": "100m", "suffix": "", "bathy_tag": "", "upload_regional": False, "master_bathy": True} # NEW: 100m standard/nearshore bathy
]

for config in processing_configs:
    RESOLUTION = config["res"]
    OUT_SUFFIX = config["suffix"]
    BATHY_TAG = config["bathy_tag"]
    UPLOAD_REGIONAL = config.get("upload_regional", True)
    MASTER_BATHY = config.get("master_bathy", False)
    
    print(f"\n{'#'*80}")
    print(f"STARTING BATCH RUN FOR: {RESOLUTION} {OUT_SUFFIX.upper().strip('_')}")
    print(f"{'#'*80}")

    for ER_REGION in er_regions:
        print(f"\n{'='*80}")
        print(f"STARTING PROCESSING FOR REGION: {ER_REGION} AT {RESOLUTION} RESOLUTION")
        print(f"{'='*80}")

        # --- 1. Define Output Keys ---
        mosaic_tiff_filename = f"{ER_REGION}_UKC_Mosaic_{RESOLUTION}{OUT_SUFFIX}.tiff"
        s3_output_key = f"low_res/{mosaic_tiff_filename}"

        # --- 2. Define Input Paths & Verify ---
        draft_filename = f"{ER_REGION}_AIS_2024_Annual_Max_Draft_{RESOLUTION}.tiff"
        draft_raster_path = os.path.join(ais_folder, draft_filename)
        
        # Path to the pre-built bathy mosaic
        if MASTER_BATHY:
            # UPDATED: Injected {BATHY_TAG} here so it points to BlueTopo_Bathy_Mosaic_Offshore_100m.tiff
            s3_bathy_key = f"low_res/{RESOLUTION}/BlueTopo_Bathy_Mosaic{BATHY_TAG}_{RESOLUTION}.tiff"
        else:
            s3_bathy_key = f"low_res/{RESOLUTION}/{ER_REGION}/{ER_REGION}_BlueTopo_Bathy_Mosaic{BATHY_TAG}_{RESOLUTION}.tiff"
            
        bathy_s3_uri = f"s3://{S3_BUCKET}/{s3_bathy_key}"

        if not os.path.exists(draft_raster_path):
            print(f"Warning: Local Draft raster not found: {draft_raster_path}")
            continue

        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_bathy_key)
            print(f"Verified pre-built Bathy Mosaic exists on S3: {s3_bathy_key}")
        except ClientError:
            print(f"Warning: Remote Bathy Mosaic not found on S3: {s3_bathy_key}")
            continue

        # ==============================================================================
        # 3. BLOCK-BY-BLOCK UKC CALCULATION (Ultra-Low RAM)
        # ==============================================================================
        print(f"\nAligning {bathy_s3_uri}")
        print(f"Onto {draft_raster_path}...")
        
        out_nodata = -9999.0

        with rasterio.open(draft_raster_path) as draft_src:
            with rasterio.open(bathy_s3_uri) as bathy_src:
                
                # Use WarpedVRT to virtually align the S3 Bathy mosaic precisely to the Draft raster's grid
                vrt_options = {
                    'resampling': Resampling.average, 
                    'crs': draft_src.crs,
                    'transform': draft_src.transform,
                    'height': draft_src.height,
                    'width': draft_src.width,
                }
                
                with WarpedVRT(bathy_src, **vrt_options) as vrt_bathy:
                    
                    # Setup the output TIF profile
                    out_profile = draft_src.profile.copy()
                    out_profile.update({
                        "driver": "GTiff",
                        "dtype": rasterio.float32,
                        "nodata": out_nodata,
                        "compress": "lzw",
                        "tiled": True,       
                        "blockxsize": 1024,  
                        "blockysize": 1024,  
                        "BIGTIFF": "YES"     
                    })
                    
                    # Statistics trackers
                    total_cells = draft_src.height * draft_src.width
                    draft_nodata_count = 0
                    draft_negative_count = 0
                    total_valid = 0
                    total_negative = 0
                    total_zero = 0
                    total_0_to_5 = 0
                    total_5_to_10 = 0
                    total_10_to_20 = 0
                    total_over_20 = 0
                    
                    print("Calculating UKC and streaming blocks directly to S3 Memory Buffer...")
                    
                    # Construct the final TIF in memory and stream it directly to S3
                    with MemoryFile() as memfile:
                        with memfile.open(**out_profile) as dest:
                            
                            # Process both rasters simultaneously chunk-by-chunk
                            for _, window in draft_src.block_windows(1):
                                
                                # Read tiny chunks of both rasters into RAM
                                draft_chunk = draft_src.read(1, window=window)
                                bathy_chunk = vrt_bathy.read(1, window=window)
                                
                                # --- Draft Stats ---
                                draft_nodata_mask = np.isnan(draft_chunk) if np.isnan(draft_src.nodata) else (draft_chunk == draft_src.nodata)
                                draft_nodata_count += np.count_nonzero(draft_nodata_mask)
                                draft_negative_count += np.count_nonzero((draft_chunk < 0) & ~draft_nodata_mask)
                                
                                # --- Masks ---
                                valid_draft_mask = ~draft_nodata_mask
                                
                                if bathy_src.nodata is not None:
                                    valid_bathy_mask = ~np.isnan(bathy_chunk) if np.isnan(bathy_src.nodata) else (bathy_chunk != bathy_src.nodata)
                                else:
                                    valid_bathy_mask = np.ones_like(bathy_chunk, dtype=bool)
                                    
                                valid_combined = valid_draft_mask & valid_bathy_mask
                                
                                # --- Calculate UKC ---
                                ukc_chunk = np.full(draft_chunk.shape, out_nodata, dtype=np.float32)
                                
                                if np.any(valid_combined):
                                    # Convert Bathy to absolute depth, subtract Draft
                                    ukc_chunk[valid_combined] = np.abs(bathy_chunk[valid_combined]) - draft_chunk[valid_combined]
                                    
                                    # --- Stats Collection ---
                                    valid_vals = ukc_chunk[valid_combined]
                                    total_valid += valid_vals.size
                                    total_negative += np.count_nonzero(valid_vals < 0)
                                    total_zero += np.count_nonzero(valid_vals == 0)
                                    total_0_to_5 += np.count_nonzero((valid_vals > 0) & (valid_vals <= 5))
                                    total_5_to_10 += np.count_nonzero((valid_vals > 5) & (valid_vals <= 10))
                                    total_10_to_20 += np.count_nonzero((valid_vals > 10) & (valid_vals <= 20))
                                    total_over_20 += np.count_nonzero(valid_vals > 20)
                                    
                                # Write the chunk to the output TIFF buffer
                                dest.write(ukc_chunk, window=window, indexes=1)
                                
                        # Upload or Save the completed mosaic
                        memfile.seek(0)
                        if UPLOAD_REGIONAL:
                            print(f"\nUploading regional mosaic directly to S3...")
                            s3_client.upload_fileobj(memfile, S3_BUCKET, s3_output_key)
                            print(f"Successfully overwritten s3://{S3_BUCKET}/{s3_output_key}")
                        else:
                            local_path = os.path.join(working_dir, mosaic_tiff_filename)
                            print(f"\nSaving regional mosaic locally (temporary) to {local_path}...")
                            with open(local_path, "wb") as f:
                                f.write(memfile.read())

        # ==============================================================================
        # 4. FINAL REGIONAL SUMMARY
        # ==============================================================================
        total_no_overlap = total_cells - total_valid
        
        if total_valid > 0:
            percent_negative = (total_negative / total_valid) * 100
            percent_zero = (total_zero / total_valid) * 100
            percent_0_to_5 = (total_0_to_5 / total_valid) * 100
            percent_5_to_10 = (total_5_to_10 / total_valid) * 100
            percent_10_to_20 = (total_10_to_20 / total_valid) * 100
            percent_over_20 = (total_over_20 / total_valid) * 100
        else:
            percent_negative = percent_zero = percent_0_to_5 = percent_5_to_10 = percent_10_to_20 = percent_over_20 = 0.0

        if total_cells > 0:
            percent_valid = (total_valid / total_cells) * 100
            percent_no_overlap = (total_no_overlap / total_cells) * 100
        else:
            percent_valid = 0.0
            percent_no_overlap = 0.0
            
        print(f"\n=======================================================")
        print(f"Final Summary for {ER_REGION} ({RESOLUTION}):")
        print(f"  -> Total cells in Draft Raster region: {total_cells}")
        print(f"  -> Original Draft cells with NoData: {draft_nodata_count}")
        print(f"  -> Original Draft cells with negative values: {draft_negative_count}")
        print(f"  -> Total valid cells with calculated UKC: {total_valid} ({percent_valid:.2f}% of total region)")
        print(f"  -> Total cells with no overlap: {total_no_overlap} ({percent_no_overlap:.2f}% of total region)")
        print(f"  -> Total cells with negative UKC: {total_negative} ({percent_negative:.2f}% of valid cells)")
        print(f"  -> Total cells with 0m UKC: {total_zero} ({percent_zero:.2f}% of valid cells)")
        print(f"  -> Total cells with > 0m to 5m UKC: {total_0_to_5} ({percent_0_to_5:.2f}% of valid cells)")
        print(f"  -> Total cells with > 5m to 10m UKC: {total_5_to_10} ({percent_5_to_10:.2f}% of valid cells)")
        print(f"  -> Total cells with > 10m to 20m UKC: {total_10_to_20} ({percent_10_to_20:.2f}% of valid cells)")
        print(f"  -> Total cells with > 20m UKC: {total_over_20} ({percent_over_20:.2f}% of valid cells)")
        print(f"=======================================================\n")

        gc.collect()

    print(f"\nUKC regional processing fully complete for {RESOLUTION}{OUT_SUFFIX}.")

    # --- Skip Global Merge for 20m ---
    if RESOLUTION == "20m":
        print(f"\nSkipping global master mosaic generation for {RESOLUTION} resolution as requested.")
        continue

    # ==============================================================================
    # 5. CREATE AND UPLOAD MASTER MOSAIC OF ALL ER REGIONS (For this Resolution)
    # ==============================================================================
    print(f"\n{'='*80}")
    print(f"STARTING GLOBAL MERGE FOR ALL ER REGIONS AT {RESOLUTION}{OUT_SUFFIX}")
    print(f"{'='*80}")
    print("WARNING: 'rasterio.merge' requires massive RAM for full-coast merging.")
    print("If the script is killed during this step, the regional tiles were already successfully uploaded to S3.")

    # Gather the expected URIs for all completed regional mosaics for the current resolution/suffix
    if UPLOAD_REGIONAL:
        regional_uris = [f"s3://{S3_BUCKET}/low_res/{er}_UKC_Mosaic_{RESOLUTION}{OUT_SUFFIX}.tiff" for er in er_regions]
    else:
        regional_uris = [os.path.join(working_dir, f"{er}_UKC_Mosaic_{RESOLUTION}{OUT_SUFFIX}.tiff") for er in er_regions]
    
    sources = []

    for uri in regional_uris:
        try:
            src = rasterio.open(uri)
            sources.append(src)
            print(f"-> Successfully queued {uri} for merging.")
        except Exception as e:
            print(f"-> Warning: Could not open {uri}. Error: {e}")

    if len(sources) > 0:
        print("\nMerging all regional datasets in RAM... (This may take a moment)")
        
        # Merge returns the composite array and its updated coordinate transform
        mosaic_data, mosaic_transform = merge(sources, nodata=-9999.0)

        # Setup the metadata for the master BigTIFF
        out_meta = sources[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic_data.shape[1],
            "width": mosaic_data.shape[2],
            "transform": mosaic_transform,
            "nodata": -9999.0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 1024,
            "blockysize": 1024,
            "BIGTIFF": "YES"
        })

        master_s3_key = f"low_res/UKC_Mosaic_{RESOLUTION}{OUT_SUFFIX}.tiff"
        
        print(f"Constructing master mosaic buffer in memory and streaming directly to S3...")
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as dest:
                dest.write(mosaic_data)
                
            memfile.seek(0)
            s3_client.upload_fileobj(memfile, S3_BUCKET, master_s3_key)

        print(f"-> Global mosaic successfully overwritten directly to S3: {master_s3_key}")

        # Clean up open dataset connections
        for src in sources:
            src.close()
            
        # Clean up local files if they were temporary
        if not UPLOAD_REGIONAL:
            print(f"\nCleaning up temporary local files...")
            for uri in regional_uris:
                if os.path.exists(uri):
                    os.remove(uri)
                    print(f"-> Removed local temporary file {uri}")
            
    else:
        print(f"\nError: No regional TIFFs were successfully opened for {RESOLUTION}{OUT_SUFFIX}. Master mosaic aborted.")

print("\nALL CONFIGURATIONS PROCESSED SUCCESSFULLY!")