# Future Survey Score
# Must Run Present Survey Score Script First
# Must update to remove present survey score reference

import arcpy
import os
from arcpy import sa
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values
curr = Parameters("Curr")
hd = Parameters("HumanDebris")
storms = Parameters("Storms")
qual = Parameters("SurveyQuality")
pss = Parameters("PSS")
fss = Parameters("FSS")
aux = Parameters("AUX")

# Input Parameters
forecast_yr = 10  # USER-DEFINED # Number of years to forecast model
pss_var = pss.raster_final_filename("2")
curr_var = curr.raster_final_filename()
storm_var = storms.raster_final_filename("100")
hd_var = hd.raster_final_filename()
age_var = qual.aux_filename("age", False)

# Static Parameters
decay_factor = (0.022 / 4)  # Factor that makes survey decay from A to B in 10 years)
sandy_lt40m = aux["sandy_lt40m_ras"]
sandy_lt20m = aux["sandy_lt20m_ras"]
eez_1 = aux["eez_1_ras"]

# Convert Files to Raster
# Input Parameters
pss_ras = arcpy.sa.Raster(pss_var)
curr_ras = arcpy.sa.Raster(curr_var)
storm_100_ras = arcpy.sa.Raster(storm_var)
hd_ras = arcpy.sa.Raster(hd_var)
age_ras = arcpy.sa.Raster(age_var)
eez_ras = arcpy.sa.Raster(eez_1)

# Output Parameters
curr_ras_bound_fn = fss.working_rastername("curr_bnd")
storms_ras_bound_fn = fss.working_rastername("s_bnd")

curr_ras_bound_c_fn = curr_ras_bound_fn + "c"
storms_ras_bound_c_fn = storms_ras_bound_fn + "c"
hd_ras_bound_c_fn = eez_1 + "c"

future_numterms = fss.working_rastername("numterm")

decay_coeff_t_fn = fss.working_rastername("dec_t%s" % forecast_yr)  # Delete
decay_coeff_t2_fn = fss.working_rastername("dec_t2%s" % forecast_yr)  # Delete
decay_coeff_fn = fss.raster_final_filename("dec%s" % forecast_yr, False)

depreciation_coeff_t_fn = fss.working_rastername("dept%s" % forecast_yr)  # Delete
depreciation_coeff_fn = fss.raster_final_filename("dep%s" % forecast_yr)

forecast_years_fn = fss.working_rastername("%syr_ras" % forecast_yr)
fss_rf_fn = fss.raster_final_filename("%s" % forecast_yr)

# Convert to Arcpy Raster
eez_1_bound_ras = arcpy.sa.Raster(eez_1)
sandy_lt20m_ras = arcpy.sa.Raster(sandy_lt20m)
sandy_lt40m_ras = arcpy.sa.Raster(sandy_lt40m)

# Calculate Changeability Normalization Raster
##############################################

# Currents Bounds
curr_ras_bound = (1 + (curr_ras * 0))
curr_ras_bound.save(curr_ras_bound_fn)
# Storms Bounds
sandy_lt40m_ras_bound = (1 + (sandy_lt40m_ras * 0))
sandy_lt40m_ras_bound.save(storms_ras_bound_fn)
# Human Debris Bounds = EEZ

curr_bound_ras = arcpy.sa.Raster(curr_ras_bound_fn)
storms_bound_ras = arcpy.sa.Raster(storms_ras_bound_fn)
hd_bound_ras = arcpy.sa.Raster(eez_1)

# Sum extents of bounds
# bounds_sum_list = [curr_bound_ras, storms_bound_ras, hd_bound_ras]
bounds_sum_list = [curr_ras_bound_fn, storms_ras_bound_fn, eez_1]

bounds_sum_list_clipped = []
for bound in bounds_sum_list:
    m = arcpy.sa.ExtractByMask(bound, eez_1)
    bound_clipped_fn = bound + "c"
    m.save(bound_clipped_fn)
    bounds_sum_list_clipped.append(bound_clipped_fn)
    print(("Extracted %s" % (os.path.split(bound)[1])))

curr_ras_bound_c_ras = arcpy.sa.Raster(curr_ras_bound_c_fn)
storms_ras_bound_c_ras = arcpy.sa.Raster(storms_ras_bound_c_fn)
hd_ras_bound_c_ras = arcpy.sa.Raster(hd_ras_bound_c_fn)

bound_clipped_fn = [curr_ras_bound_c_ras, storms_ras_bound_c_ras, hd_ras_bound_c_ras]
bounds_sum = arcpy.sa.CellStatistics(bound_clipped_fn, "SUM", "DATA")
bounds_sum.save(future_numterms)

future_numterms_ras = arcpy.sa.Raster(future_numterms)

# Calculate Decay Coefficient
#############################

# Sum Forecast Variables
forecast_var_list = [curr_ras, storm_100_ras, hd_ras]
forecast_sum = arcpy.sa.CellStatistics(forecast_var_list, "SUM", "DATA")
forecast_sum.save(decay_coeff_t_fn)
future_sumterms_ras = arcpy.sa.Raster(decay_coeff_t_fn)

# Subtract number of change terms from Sum Decay Coefficients and Multiply by 0.055
# (0.022/4 --> Factor that makes survey decay from A to B in 10 years)
decay_sum_minusnumterms = future_sumterms_ras - future_numterms_ras
decay_sum_minusnumterms.save(decay_coeff_t2_fn)
future_sumterms_minusnmterms = arcpy.sa.Raster(decay_coeff_t2_fn)

decay_coeff = future_sumterms_minusnmterms * decay_factor
decay_coeff.save(decay_coeff_fn)

# Calculate Depreciation Coefficient
####################################

# START HERE FOR GENERATING UPDATED FSS Layers #
decay_coeff_ras = arcpy.sa.Raster(decay_coeff_fn)

forecast_ras_Xyrs = (forecast_yr + (eez_ras * 0))
forecast_ras_Xyrs.save(forecast_years_fn)
forecast_ras_Xyrs_ras = arcpy.sa.Raster(forecast_years_fn)

# Must Pad Depreciation Coefficient with Dep_Coeff = 1 to replace NALL values, then, take Minimum
depreciation_term = arcpy.sa.Exp(-1 * decay_coeff_ras * forecast_ras_Xyrs_ras)
depreciation_term.save(depreciation_coeff_t_fn)

# Calculate Present Survey Score
arcpy.MosaicToNewRaster_management([eez_1, depreciation_coeff_t_fn], os.path.split(depreciation_coeff_fn)[0], os.path.split(depreciation_coeff_fn)[1], fss.projection_number, "32_BIT_FLOAT", str(fss.cell_size), "1", "MINIMUM", "FIRST")
depreciation_term_ras = arcpy.sa.Raster(depreciation_coeff_fn)

fss_term = pss_ras * depreciation_term_ras
fss_term.save(fss_rf_fn)
