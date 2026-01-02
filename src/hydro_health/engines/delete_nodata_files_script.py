import os
import rasterio
import dask
from pathlib import Path
from dask import delayed

# Define the target folder from your request
DEFAULT_TARGET_FOLDER = r"\\nos.noaa\ocs\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\pre_processed\tiled"

def check_and_delete_if_empty(raster_path):
    """Checks a raster file for any valid data and deletes it if empty.
     
     param str or Path raster_path: Path to the raster file to check.
     return: A string indicating the status (KEPT, DELETED, or ERROR).
    """
    has_valid_data = False
    raster_name = os.path.basename(str(raster_path)).lower()

    try:
        with rasterio.open(raster_path) as src:
            # Check if the raster has any bands
            if src.count == 0:
                print(f'Skipping {raster_name}: No bands found.')
                has_valid_data = False
            else:
                # Iterate over blocks to avoid loading all data into memory
                for ji, window in src.block_windows(1):
                    
                    # Read the block, masking the nodata values
                    data_block = src.read(1, window=window, masked=True)
                    
                    # .count() returns the number of *valid* (unmasked) pixels
                    if data_block.count() > 0:
                        has_valid_data = True
                        break # Found valid data, no need to check other blocks

    except Exception as e:
        print(f'Error processing {raster_name}: {e}')
        return f"ERROR: {str(raster_path)}"

    if not has_valid_data:
        print(f'Deleting {raster_name}: No valid data found.')
        try:
            os.remove(raster_path)
            return f"DELETED: {str(raster_path)}"
        except Exception as e:
            print(f'Error deleting {raster_name}: {e}')
            return f"ERROR_DELETING: {str(raster_path)}"
    else:
        # Optional: Print message for files that are kept
        # print(f'Keeping {raster_name}: Valid data found.')
        return f"KEPT: {str(raster_path)}"

def main(target_folder):
    """
    Finds all TIF files in a folder and its subfolders, and uses Dask
    to parallel-check and delete them if they contain no valid data.
    """
    print(f"Scanning for raster files in: {target_folder}")
    
    target_path = Path(target_folder)
    
    # Use rglob to recursively find all .tif and .tiff files
    tif_files = list(target_path.rglob("*.tif"))
    tif_files.extend(list(target_path.rglob("*.tiff")))

    if not tif_files:
        print("No .tif or .tiff files found in the target directory.")
        return

    print(f"Found {len(tif_files)} files. Building Dask task list...")

    # Create a list of delayed tasks
    tasks = []
    for f_path in tif_files:
        task = delayed(check_and_delete_if_empty)(f_path)
        tasks.append(task)

    # Execute all tasks in parallel
    print("Starting Dask computation...")
    results = dask.compute(*tasks) # Unpack the list into arguments
    
    print("\n--- Computation Finished ---")
    
    # Optionally, print a summary of results
    deleted_count = sum(1 for r in results if r.startswith("DELETED"))
    kept_count = sum(1 for r in results if r.startswith("KEPT"))
    error_count = sum(1 for r in results if "ERROR" in r)

    print(f"Summary:")
    print(f"  Files Kept:   {kept_count}")
    print(f"  Files Deleted: {deleted_count}")
    print(f"  Errors:       {error_count}")
    print("----------------------------")


if __name__ == "__main__":
    # Ensure the Dask scheduler uses threads for I/O-bound tasks
    dask.config.set(scheduler='threads') 
    
    main(target_folder=DEFAULT_TARGET_FOLDER)