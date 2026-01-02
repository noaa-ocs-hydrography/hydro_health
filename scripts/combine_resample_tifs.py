import os
from osgeo import gdal, osr
import dask
from dask import delayed

def merge_and_resample_tifs(input_folder, output_tif, target_resolution=500.0):
    """Merges and resamples GeoTIFFs from a folder into a single raster.
      param str input_folder: Path to the folder containing input GeoTIFF files.
      param str output_tif: Path for the output merged and resampled GeoTIFF.
      param float target_resolution: The target resolution in meters.
      return: str Confirmation message.
    """
    # This function remains the same as before, but we'll return a string for clarity.
    folder_name = os.path.basename(input_folder)
    print(f"Starting processing for: {folder_name}")
    
    tif_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.tif', '.tiff'))]
    
    if not tif_files:
        msg = f"No TIFF files found in {folder_name}. Skipping."
        print(msg)
        return msg

    vrt_path = os.path.join(input_folder, "source.vrt")
    vrt_options = gdal.BuildVRTOptions(allowProjectionDifference=True)
    gdal.BuildVRT(vrt_path, tif_files, options=vrt_options)

    source_ds = gdal.Open(vrt_path)
    if source_ds is None:
        msg = f"Failed to open VRT for {folder_name}. Skipping."
        print(msg)
        return msg

    srs = osr.SpatialReference()
    srs.ImportFromWkt(source_ds.GetProjection())

    if srs.IsProjected():
        res = target_resolution
    else:
        res = target_resolution / 111320.0
        
    gdal.Warp(output_tif, source_ds, xRes=res, yRes=res, resampleAlg=gdal.GRA_Average)
    
    source_ds = None
    os.remove(vrt_path)
    
    msg = f"Finished. Output for {folder_name} saved to: {output_tif}"
    print(msg)
    return msg

if __name__ == '__main__':
    # --- User-defined variables ---
    parent_directory = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\Digital_Cost_Manual_Downloads'
    output_base_dir = r"C:\Users\aubrey.mccutchan\Documents\Processed_Mosaics"
    exclude_folder = 'laz_issues'
    resolution_meters = 500.0
    # --------------------------------

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    # --- Dask Implementation ---
    # 1. Create an empty list to hold the delayed tasks
    tasks = []

    print("Building task list for Dask...")
    for folder_name in os.listdir(parent_directory):
        folder_to_process = os.path.join(parent_directory, folder_name)

        if os.path.isdir(folder_to_process) and folder_name != exclude_folder:
            # --- Automated Naming ---
            name_parts = folder_name.split('_')
            data_name = '_'.join(name_parts[:2])
            output_raster = os.path.join(output_base_dir, f"mosaic_{data_name}_resampled.tif")
            
            # 2. Wrap the function call with `dask.delayed` and add it to the list
            # This doesn't run the function yet, it just adds it to the to-do list
            task = delayed(merge_and_resample_tifs)(folder_to_process, output_raster, resolution_meters)
            tasks.append(task)

    # 3. If tasks were created, run them all in parallel
    if tasks:
        print(f"\n{len(tasks)} tasks submitted. Starting parallel processing...")
        dask.compute(*tasks)
        print("\n✅ All folders have been processed!")
        print("\a")
    else:
        print("No folders found to process.")
    