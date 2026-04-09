# SAR VECTOR PROCESSED to RASTER
'''This script computes the final classified Search and Rescue (SAR) consequence raster from processed vector data compiled from SAR_RawToVector.py and processed
AIS data. The final raster is classified based on the SAR Coefficient Value defined as:

SAR Coefficient: log2(distance/300) + log10(Vessel Length)

where 300 represents the maximum searchable distances of Jayhawk and Dolphin assets.

Distance is classified following the table below:
    Distance                Classification
                         log base 2 (dist/300))
     < 75 naut mi                  -3
    75 - 150 naut mi               -2
   150 - 300 naut mi               -1
   300 - 600 naut mi                0
     > 600 naut mi                  1

Vessel Length is estimated differently for passenger and non-passenger vessels
                    Passenger
    Vessel Length (m)              Classification
                            log base 10 (Vessel Length)
          < 10                          0
        10 - 50                         1
        50 - 200                        2
          > 200                         3

                 Non- Passenger
    Vessel Presence                Classification
                            log base 10 (Vessel Length)
     Vessel Present                      0
     Vessel Absent                      -4

The maximum of the classified distance and length rasters are then added to compute a final SAR classified raster based on SAR Coefficient
SAR Coefficient: log2(distance/300) + log10(Vessel Length)

SAR Coefficient         Final Classification
    > 1                         5
   0 - 1                        4
  -1 - 0                        3
  -2 - -1                       2
  -3 - -2                       1
    < -3                        0

'''

import arcpy
import os
from arcpy import env
from arcpy.sa import *
arcpy.CheckOutExtension("Spatial")

# Define Variables
# Inputs
grid_extents = r"H:\NHSP_2_0\Christy\SAR\SAR.gdb\Grid_Model_Bounds"  # User-Defined (global)
mask = r"H:\GRID\Grid.gdb\USA_Shoreline_NAEAC_Explode_Gen100m_Raster"  # User -Defined (global)
output_gdb = r"H:\NHSP_2_0\Christy\SAR\SAR.gdb"  # User-Defined (global, eventually)
out_fldr = r'H:\NHSP_2_0\Christy\SAR'
sar_data = os.path.join(output_gdb, "SAR_VECTOR_PROCESSED")  # User - Defined
pass_length = r'H:\NHSP_2_0\Christy\SAR\sar_passl'  # User - Defined
nonpass_length = r'H:\NHSP_2_0\Christy\SAR\sar_nonpassl'  # User - Defined
prj_1 = 102008  # User - Defined
cell_size = 500  # USER DEFINED

# Outputs
sar_dist = os.path.join(out_fldr, "SAR_Dist")  # Delete
sar_dist_clip = os.path.join(out_fldr, "SAR_R")
sar_dist_reclass = os.path.join(out_fldr, "SAR_dis_RC")
sar_disL_pass = os.path.join(out_fldr, "sar_disPass")  # Delete
sar_disL_nonpass = os.path.join(out_fldr, "sar_disnPass")  # Delete
sar_pass_rct = os.path.join(out_fldr, "sar_pass_rct")
sar_nonpass_rct = os.path.join(out_fldr, "sar_npass_rct")
sar_pass_rc = os.path.join(out_fldr, "sar_pass_fin")
sar_nonpass_rc = os.path.join(out_fldr, "sar_npass_fin")

# Don't do this
# sar_final_t = "SAR_Final_t" #Delete
# sar_final_t2 = os.path.join(out_fldr,"SAR_Final_t2") # Delete
#sar_final = os.path.join(out_fldr, "SAR_Final")

# Set environmental settings
arcpy.env.extent = grid_extents

# Calculate Distance of SAR facility from each grid cell
t_start = time.time()
out_dist = EucDistance(sar_data, "", 500)
print(("Finished Euc Distance at %.1f secs" % (time.time() - t_start)))
out_dist.save(sar_dist)
print(("Finished Save Raster at %.1f secs" % (time.time() - t_start)))

# Clip Distance Raster to Shoreline-To-EEZ Raster
out_extract = ExtractByMask(sar_dist, mask)
out_extract.save(sar_dist_clip)

# Reclassify Raster based on distance from SAR facility
# SAR distance is classified as log base 2 (distance/300), assuming nautical mile units

# Find maximum value in raster
sar_dist_max_t = arcpy.GetRasterProperties_management(sar_dist_clip, "MAXIMUM")
sar_dist_max = (float(sar_dist_max_t.getOutput(0))) + 1111000  # Ensure maximum is not mis-classified.

# Define Reclassification
remap = RemapRange([[0, 138900, -3], [138900, 277800, -2], [277800, 555600, -1], [555600, 1111000, 0], [1111000, sar_dist_max, 1]])
sar_dist_reclass_out = Reclassify(sar_dist_clip, "Value", remap)
sar_dist_reclass_out.save(sar_dist_reclass)

# Must Add Dist Raster to Passenger Classified and to Non-Passenger classified, and take maximum, then reclassify
sar_dist_ras = arcpy.Raster(sar_dist_reclass)
pass_length_ras = arcpy.Raster(pass_length)
nonpass_length_ras = arcpy.Raster(nonpass_length)

sar_dist_pass_length = sar_dist_ras + pass_length_ras
sar_dist_pass_length.save(sar_disL_pass)
sar_dist_nonpass_length = sar_dist_ras + nonpass_length_ras
sar_dist_nonpass_length.save(sar_disL_nonpass)

# Reclassify Passenger and non-Passenger SAR Rasters, then classify again to get rid of zeros
r_pass = arcpy.sa.Reclassify(sar_disL_pass, "VALUE", "-10 -3.000001 NODATA;-3 -2.000010 1;-2 -1.000010 2;-1 0 3;0.000010 1 4;1 5 5;NODATA NODATA", "NODATA")
r_pass.save(sar_pass_rct)
r_pass_f = arcpy.sa.Reclassify(sar_pass_rct, "VALUE", "0 NODATA;1 1 1;2 2 2;3 3 3;4 4 4;5 5 5", "NODATA")
r_pass_f.save(sar_pass_rc)


r_npass = arcpy.sa.Reclassify(sar_disL_nonpass, "VALUE", "-10 -3.000001 NODATA;-3 -2.000010 1;-2 -1.000010 2;-1 0 3;0.000010 1 4;1 5 5;NODATA NODATA", "NODATA")
r_npass.save(sar_nonpass_rct)
r_npass_f = arcpy.sa.Reclassify(sar_nonpass_rct, "VALUE", "0 NODATA;1 1 1;2 2 2;3 3 3;3 4 4;5 5 5", "NODATA")
r_npass_f.save(sar_nonpass_rc)

# Don't Do this
#arcpy.MosaicToNewRaster_management([sar_disL_pass, sar_disL_nonpass], out_fldr, sar_final_t, prj_1, "32_BIT_SIGNED", str(cell_size), "1", "MAXIMUM", "FIRST")

#r = arcpy.sa.Reclassify(os.path.join(out_fldr,sar_final_t), "VALUE", "-10 -3.000001 NODATA;-3 -2.000010 1;-2 -1.000010 2;-1 0 3;0.000010 1 4;1 5 5;NODATA NODATA", "NODATA")
# r.save(sar_final_t2)
#r2 = arcpy.sa.Reclassify(sar_final_t2, "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", "DATA")
# r2.save(sar_final)

# Delete Management
del_fns = [sar_dist]
for fn in del_fns:  # enumerate retrieves index number
    print((str(fn)))
    arcpy.Delete_mangagement(str(fn))
