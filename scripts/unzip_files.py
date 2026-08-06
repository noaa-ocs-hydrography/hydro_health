import os
import glob
import zipfile
import rasterio
from rasterio.warp import transform_bounds
import time
import shutil

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\aubrey.mccutchan\Documents\Tampa Imagery"
MASK_PATH = r"C:\Users\aubrey.mccutchan\Documents\prediction_mask_pilot.tif"
OUTPUT_DIR_NAME = "Processed_TIFs" # Name of the new folder to move kept TIFs into
OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIR_NAME)
CHECK_INTERVAL = 10  # Seconds to wait between checking the folder for new zips

def check_intersection(bounds1, bounds2):
    """Checks if two bounding boxes intersect."""
    l1, b1, r1, t1 = bounds1
    l2, b2, r2, t2 = bounds2
    return not (r1 < l2 or l1 > r2 or t1 < b2 or b1 > t2)

def is_file_ready(filepath):
    """Checks if a file is fully downloaded and not locked by the browser."""
    if filepath.endswith(('.crdownload', '.part', '.tmp')):
        return False
    try:
        os.rename(filepath, filepath)
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            pass 
        return True
    except (PermissionError, zipfile.BadZipFile, OSError):
        return False

def process_tif(tif_path, mask_crs, mask_bounds):
    """Checks a TIF against the mask, moves it if it intersects, or deletes it."""
    try:
        # Open just to read metadata, then close immediately to avoid file lock issues
        with rasterio.open(tif_path) as tif_src:
            tif_crs = tif_src.crs
            tif_bounds = tif_src.bounds
            
        # Reproject if necessary
        if tif_crs != mask_crs:
            tif_bounds = transform_bounds(tif_crs, mask_crs, *tif_bounds)
        
        xml_path = os.path.splitext(tif_path)[0] + ".xml"

        # Check intersection
        if check_intersection(mask_bounds, tif_bounds):
            print(f"  [+] KEEPING & MOVING (Intersects): {os.path.basename(tif_path)}")
            # Move the TIF to the output directory
            shutil.move(tif_path, os.path.join(OUTPUT_DIR, os.path.basename(tif_path)))
            
            # Move companion XML if it exists
            if os.path.exists(xml_path):
                shutil.move(xml_path, os.path.join(OUTPUT_DIR, os.path.basename(xml_path)))
            return True
        else:
            print(f"  [-] DELETING (No Intersection): {os.path.basename(tif_path)}")
            os.remove(tif_path)
            
            # Clean up companion XML if it exists
            if os.path.exists(xml_path):
                os.remove(xml_path)
            return False
            
    except Exception as e:
        print(f"  [!] Error reading {os.path.basename(tif_path)}: {e}")
        # If it errors out, we'll try to move it anyway to be safe so it's not lost
        try:
            shutil.move(tif_path, os.path.join(OUTPUT_DIR, os.path.basename(tif_path)))
            print(f"  [~] Moved unreadable file to output directory to be safe.")
        except Exception as move_e:
            pass
        return True 

def cleanup_empty_folder(folder_path):
    """Removes a folder if it is completely empty."""
    try:
        if os.path.exists(folder_path) and not os.listdir(folder_path):
            os.rmdir(folder_path)
            print(f"  [~] Removed empty folder: {os.path.basename(folder_path)}")
    except OSError:
        pass # Folder wasn't empty or couldn't be deleted

def main():
    print(f"Reading mask file: {MASK_PATH}")
    try:
        with rasterio.open(MASK_PATH) as mask_src:
            mask_crs = mask_src.crs
            mask_bounds = mask_src.bounds
    except Exception as e:
        print(f"Could not open mask file. Please check the path. Error: {e}")
        return

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("-" * 50)
    print("Performing initial sweep of existing folders...")
    
    # Find all TIFs currently in the directory (and subdirectories)
    existing_tifs = glob.glob(os.path.join(BASE_DIR, "**", "*.tif"), recursive=True)
    
    for tif_path in existing_tifs:
        # Skip processing if the file is already in the output directory
        if os.path.abspath(OUTPUT_DIR) in os.path.abspath(tif_path):
            continue

        process_tif(tif_path, mask_crs, mask_bounds)
        # Try to clean up the folder right after processing its file
        cleanup_empty_folder(os.path.dirname(tif_path))

    print("-" * 50)
    print(f"Initial sweep complete. Valid files moved to '{OUTPUT_DIR_NAME}'.")
    print("Monitoring for new downloads... (Press Ctrl+C to stop)")

    try:
        while True:
            zip_files = glob.glob(os.path.join(BASE_DIR, "*.zip"))
            
            for zip_path in zip_files:
                if not is_file_ready(zip_path):
                    continue 

                extract_dir = os.path.splitext(zip_path)[0]
                os.makedirs(extract_dir, exist_ok=True)

                print(f"\n[✓] Download complete. Extracting: {os.path.basename(zip_path)}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)

                os.remove(zip_path)
                print(f"Deleted original zip: {os.path.basename(zip_path)}")

                # Check the newly extracted files
                extracted_tifs = glob.glob(os.path.join(extract_dir, "**", "*.tif"), recursive=True)
                
                for tif_path in extracted_tifs:
                    process_tif(tif_path, mask_crs, mask_bounds)
                    # Try cleaning up immediate parent directory of the processed file
                    cleanup_empty_folder(os.path.dirname(tif_path))
                
                # Clean up the main extracted folder if it is now empty (since files were moved/deleted)
                cleanup_empty_folder(extract_dir)
            
            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopping folder monitor. Goodbye!")

if __name__ == "__main__":
    main()