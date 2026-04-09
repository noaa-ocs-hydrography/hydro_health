# Present Survey Score
# The present survey score is defined as the following:
# Initial Survey Score * Depreciation Coefficient
#      where,
#          Depreciation Coefficient = e^(Decay Coefficient*Survey Age)
# where
# Decay Coefficient = (Storms + Currents + Human Debris)*(0.022/4)


import arcpy
import os
from arcpy import sa
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values
iss = Parameters("ISS")
curr = Parameters("Curr")
hd = Parameters("HumanDebris")
storms = Parameters("Storms")
qual = Parameters("SurveyQuality")
pss = Parameters("PSS")
aux = Parameters("AUX")

# Input Parameters
iss_var = iss.raster_final_filename("2")
curr_var = curr.raster_final_filename()
storm_var = storms.raster_final_filename("svy")
hd_var = hd.raster_final_filename()
age_var = qual.aux_filename("age", False)

# Static Parameters
decay_factor = (0.022 / 4)  # Factor that makes survey decay from A to B in 10 years)
sandy_lt40m = aux["sandy_lt40m_ras"]
sandy_lt20m = aux["sandy_lt20m_ras"]
eez_1 = aux["eez_1_ras"]

# Output Variables
change_numterms = pss.working_filename("numterm", False)
decay_coeff_t_fn = pss.working_filename("decay_t", False)  # Delete
decay_coeff_t2_fn = pss.working_filename("decay_t2", False)  # Delete
decay_coeff_t3_fn = pss.working_filename("decay_t3", False)  # Delete
decay_coeff_fn = pss.raster_final_filename("decay")
sandy_lt20m_ras_bound_fn = pss.working_filename("s_lt20m_b", False)  # Delete
sandy_lt40m_ras_bound_fn = pss.working_filename("s_lt40m_b", False)  # Delete
depreciation_coeff_t_fn = pss.working_filename("dep_t", False)  # Delete
depreciation_coeff_fn = pss.raster_final_filename("dep")

pss_rf_fn = pss.raster_final_filename()

# Convert Files to Raster
# Input Parameters
iss_ras = arcpy.sa.Raster(iss_var)
curr_ras = arcpy.sa.Raster(curr_var)
storm_ras = arcpy.sa.Raster(storm_var)
hd_ras = arcpy.sa.Raster(hd_var)
age_ras = arcpy.sa.Raster(age_var)

eez_1_bound_ras = arcpy.sa.Raster(eez_1)
sandy_lt20m_ras = arcpy.sa.Raster(sandy_lt20m)
sandy_lt40m_ras = arcpy.sa.Raster(sandy_lt40m)


# Calculate Changeability Normalization Raster
# Set bounds of sandy < 20 m and sandy < 40 m files = 1
# See note below, should set sandy_lt20m_ras bounds to extents of currents raster to remove necessity to reclassify decay raster
sandy_lt20m_ras_bound = (1 + (sandy_lt20m_ras * 0))
sandy_lt20m_ras_bound.save(sandy_lt20m_ras_bound_fn)

sandy_lt40m_ras_bound = (1 + (sandy_lt40m_ras * 0))
sandy_lt40m_ras_bound.save(sandy_lt40m_ras_bound_fn)

sandy_lt20m_bound_ras = arcpy.sa.Raster(sandy_lt20m_ras_bound_fn)
sandy_lt40m_bound_ras = arcpy.sa.Raster(sandy_lt40m_ras_bound_fn)

# Sum extents of bounds
bounds_sum_list = [eez_1_bound_ras, sandy_lt20m_bound_ras, sandy_lt40m_bound_ras]
bounds_sum = arcpy.sa.CellStatistics(bounds_sum_list, "SUM", "DATA")
bounds_sum.save(change_numterms)

change_numterms_ras = arcpy.sa.Raster(change_numterms)

# Calculate Decay Coefficient
#############################

# Sum Change Variables
change_var_list = [curr_var, storm_var, hd_var]

change_sum = arcpy.sa.CellStatistics(change_var_list, "SUM", "DATA")
change_sum.save(decay_coeff_t_fn)
decay_coeff_ras = arcpy.sa.Raster(decay_coeff_t_fn)

# Subtract number of change terms from Sum Decay Coefficients and Multiply by 0.055
# (0.022/4 --> Factor that makes survey decay from A to B in 10 years)
decay_sum_minusnumterms = decay_coeff_ras - change_numterms_ras
decay_sum_minusnumterms.save(decay_coeff_t2_fn)

# Reclassify difference to remove any negative values. Note negative values result from currents raster.

# Extents of currents raster does not extend all the way to the inshore limit of the EEZ, therefore, there are some areas where currents should exist, but are NAN.
# As such, in these areas, we will treat them as if there were only two input change coefficient variables and set them to 0
# BETTER OPTION - Set Currents bounds to extents of currents raster
# Example
# Assume Storm = 1; HD = 1
# Within Sandy LT 20 m
# Curr = 5    # Num Terms = 3 # Diff = 4
# Outside Sandy LT 20 m
# Curr = NAN    # Num Terms = 2 # Diff = 0
# Within Sandy LT 20 m, but no current data
# Curr = NAN  # Num Terms = 3 # Diff = -1 # Set = 0, such that the same as outside sandy lt 20 m

arcpy.gp.Reclassify_sa(decay_coeff_t2_fn, "VALUE", "-1 0;0 0;1 1;2 2;3 3;4 4;5 5;6 6;7 7;8 8;9 9;10 10;11 11;12 12;13 13;14 14;15 15", decay_coeff_t3_fn, "DATA")
decay_sum_minusnumterms_ras = arcpy.sa.Raster(decay_coeff_t3_fn)

decay_coeff = decay_sum_minusnumterms_ras * decay_factor
decay_coeff.save(decay_coeff_fn)
decay_coeff_ras = arcpy.sa.Raster(decay_coeff_fn)

# Calculate Depreciation Coefficient
####################################
depreciation_term = arcpy.sa.Exp(-1 * decay_coeff_ras * age_ras)
depreciation_term.save(depreciation_coeff_t_fn)

# Calculate Present Survey Score
# Must Pad Depreciation Coefficient with Dep_Coeff = 1 to replace NALL values, then, take Minimum
arcpy.MosaicToNewRaster_management([eez_1, depreciation_coeff_t_fn], os.path.split(depreciation_coeff_fn)[0], os.path.split(depreciation_coeff_fn)[1], pss.projection_number, "32_BIT_FLOAT", str(pss.cell_size), "1", "MINIMUM", "FIRST")

depreciation_term_ras = arcpy.sa.Raster(depreciation_coeff_fn)

pss_term = iss_ras * depreciation_term_ras
pss_term.save(pss_rf_fn)