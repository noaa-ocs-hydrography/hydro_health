# SAR RAW to RASTER FINAL
'''This script computes the final classified Search and Rescue (SAR) consequence raster from raw SAR locations stored in a csv and processed
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
     Vessel Present-1                   -1
     Vessel Absent                      -5

The maximum of the classified distance and length rasters are then added to compute a final SAR classified raster based on SAR Coefficient
SAR Coefficient: log2(distance/300) + log10(Vessel Length)

SAR Coefficient         Final Classification
    > 1                         5
   0 - 1                        4
  -1 - 0                        3
  -2 - -1                       2
  -3 - -2                       1
    < -3                        0

When implemented,
Non Passenger Vessel
Non-Passenger Vessel Present
Present Value     Dist to SAR    Dist Val    Present + Dist Val     SAR Class (RC)
     0             < 75 nm        -3                -3                    1
     0            75 - 150 nm     -2                -2                    2
     0            150 - 300 nm    -1                -1                    3
     0            300 - 600 nm     0                0                     4
     0            > 600 nm         1                1                     5

Non-Passenger Vessel NOT Present
Present Value     Dist to SAR    Dist Val    Present + Dist Val     SAR Class (RC)
     -5             < 75 nm        -3                -8                    0
     -5            75 - 150 nm     -2                -7                    0
     -5            150 - 300 nm    -1                -6                    0
     -5            300 - 600 nm     0                -5                    0
     -5            > 600 nm         1                -4                    0

Passenger Vessel
Passenger Vessel Present
Present Value                 Dist to SAR        Dist Val    Present + Dist Val     SAR Class (RC)
see vessel length                < 75 nm           -3             Variable            Variable
see vessel length              75 - 150 nm         -2             Variable            Variable
see vessel length              150 - 300 nm        -1             Variable            Variable
see vessel length              300 - 600 nm         0             Variable            Variable
see vessel length                > 600 nm           1             Variable            Variable

'''
# Import modules
import os
import arcpy
import time
import arcgisscripting
from arcpy import env, sa
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values
from HSTB.ArcExt.NHSP import Functions
sar = Parameters("SAR")
aux = Parameters("AUX")
ais = Parameters("AIS")

# User-Defined Input Variables
sar_raw = sar.raw_filename("", False) + ".csv"
x_coords = "X"  # User-Defined
y_coords = "Y"  # User-Defined
sar_raw_input_prj = 4326  # User-Defined
ais_p_len = ais.raster_classified_filename("sar_p")  # p_len
ais_np_len = ais.raster_classified_filename("sar_np")

# Input Parameters
sar_raw_lyr = "SAR_RAW_lyr"

# Static Parameters
arcpy.env.extent = aux["grid_extents_ras"]
eez_0 = aux["eez_0_ras"]

# Output Parameters
sar_vp_fn = sar.vector_processed_filename()
sar_dist_t = sar.working_rastername("dis_t")  # Delete
sar_dist = sar.working_rastername("dis")  # Delete
sar_dist_rc = sar.raster_classified_filename("dis")
pass_sar_r = sar.raster_filename("pass")
npass_sar_r = sar.raster_filename("npass")
pass_sar_rc_t = sar.working_rastername("pass_rct")
pass_sar_rc = sar.raster_classified_filename("pass")
npass_sar_rc = sar.raster_classified_filename("npass")
pass_sar_rf = sar.raster_final_filename("pass")
npass_sar_rf = sar.raster_final_filename("npass")
eez_neg5_ras = sar.working_rastername("eez_neg5")

# VECTOR
sar_v = Functions.query_csv(sar_raw, x_coords, y_coords, sar, None, prj=4326)

# VECTOR PROCESSED
sar_vp = Functions.check_spatial_reference(sar_v, sar_vp_fn, sar.projection_name, sar.projection_number)

# Calculate Distance of SAR facility from each grid cell
out_dist = arcpy.sa.EucDistance(sar_vp, "", 500)
out_dist.save(sar_dist_t)

# Clip Distance Raster to EEZ Raster
# Clip Distance Raster to Shoreline-To-EEZ Raster
out_extract = arcpy.sa.ExtractByMask(sar_dist_t, eez_0)
out_extract.save(sar_dist)

# Reclassify Raster based on distance from SAR facility
# SAR distance is classified as log base 2 (distance/300), assuming nautical mile units

# Find maximum value in raster
sar_dist_max_t = arcpy.GetRasterProperties_management(sar_dist, "MAXIMUM")
sar_dist_max = (float(sar_dist_max_t.getOutput(0))) + 1111000  # Ensure maximum is not mis-classified.

arcpy.gp.Reclassify_sa(sar_dist, "VALUE", "0 138900 -3;138900 277800 -2;277800 555600 -1;555600 1111000 0;1111000 %s 1" % (sar_dist_max), sar_dist_rc, "NODATA")

# Must Add Dist Raster to Passenger Classified and to Non-Passenger classified, and take maximum, then reclassify
sar_dist_ras = arcpy.Raster(sar_dist_rc)
pass_length_ras = arcpy.Raster(ais_p_len)
npass_length_ras = arcpy.Raster(ais_np_len)

pass_sar_sum = sar_dist_ras + pass_length_ras
pass_sar_sum.save(pass_sar_r)
npass_sar_sum = sar_dist_ras + npass_length_ras
npass_sar_sum.save(npass_sar_r)

# Pad Passenger Raster with -4 values, where no vessel is present
eez_1_raster = arcpy.Raster(eez_0)
eez_neg4 = eez_1_raster + -5
eez_neg4.save(eez_neg5_ras)

arcpy.MosaicToNewRaster_management([pass_sar_r, eez_neg5_ras], os.path.dirname(pass_sar_rc_t), os.path.basename(pass_sar_rc_t), sar.projection_number, "32_BIT_SIGNED", sar.cell_size, "1", "MAXIMUM", "FIRST")

# Reclassify Raster
arcpy.gp.Reclassify_sa(pass_sar_rc_t, "VALUE", "-8 -4 0;-3 -3 1; -2 -2 2; -1 -1 3; 0 0 4; 1 10 5", pass_sar_rc, "NODATA")
arcpy.gp.Reclassify_sa(npass_sar_r, "VALUE", "-8 -4 0;-3 -3 1; -2 -2 2; -1 -1 3; 0 0 4; 1 10 5", npass_sar_rc, "NODATA")

# Copy as Raster Final
arcpy.CopyRaster_management(pass_sar_rc, pass_sar_rf)
arcpy.CopyRaster_management(npass_sar_rc, npass_sar_rf)
