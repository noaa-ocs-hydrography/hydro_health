# from osgeo import gdal
# import glob
# import os

# # 1. Define your paths
# input_folder = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\Pilot_Model_Outputs'
# output_vrt = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\Pilot_Model_Outputs\catzoc_decay_score_latest.vrt'

# # 2. Get a list of all .tif files in the folder
# search_path = os.path.join(input_folder, "BlueTopo*latest.tiff")
# file_list = glob.glob(search_path)

# # 3. Build the VRT
# # gdal.BuildVRT(destName, srcDSOrSrcDSTab, **kwargs)
# vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest', addAlpha=True)
# ds = gdal.BuildVRT(output_vrt, file_list, options=vrt_options)

# # 4. Close the dataset to write it to disk
# ds = None

# print(f"Successfully created {output_vrt} from {len(file_list)} files.")


from osgeo import gdal
import glob
import os

input_folder = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\Pilot_Model_Outputs'
final_vrt = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\Pilot_Model_Outputs\catzoc_supersession_score_all.vrt'
intermediate_vrt = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\Pilot_Model_Outputs\mosaic_26917.vrt'

# 1. Get the list of files
file_list = glob.glob(os.path.join(input_folder, "BlueTopo*score_all.tiff"))

# 2. STEP 1: Mosaic them into a single VRT (stays in original projection)
# This solves the "first source dataset" warning
gdal.BuildVRT(intermediate_vrt, file_list)

# 3. STEP 2: Warp that VRT into the new projection
# This creates a "Warped VRT"
warp_options = gdal.WarpOptions(
    format="VRT",
    srcSRS="EPSG:26917",
    dstSRS="EPSG:32617",
    resampleAlg="bilinear"
)

ds = gdal.Warp(final_vrt, intermediate_vrt, options=warp_options)
ds = None  # Flush to disk

print(f"Done! {final_vrt} now contains all tiles in EPSG:32617.")