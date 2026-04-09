# Hydrographic Health

import arcpy
import os
from arcpy import sa
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values
pss = Parameters("PSS")
dss = Parameters("DSS")
hgap = Parameters("HGAP")
fss = Parameters("FSS")

# Input Parameters
forecast_years_list = [5, 10]  # User-Defined
dss_var = dss.raster_final_filename()
pss_var = pss.raster_final_filename("2")
fss_var = fss.raster_final_filename()

# Output Parameters
hgap_p_fn = hgap.raster_final_filename()

# Convert Inputs to Rasters
dss_ras = arcpy.sa.Raster(dss_var)
pss_ras = arcpy.sa.Raster(pss_var)

# Calculate PRESENT Hydro Gap
hydro_gap_p = dss_ras - pss_ras
hydro_gap_p.save(hgap.raster_final_filename())

# Calculate Future Hydro Gap for all Future Survey Score Files
for num in forecast_years_list:
    fss_fn = fss.raster_final_filename("%s" % (num))
    fss_ras = arcpy.sa.Raster(fss_fn)
    hydro_gap_f = dss_ras - fss_ras
    hydro_gap_f.save(hgap.raster_final_filename("%s" % (num)))
    print(("Exported %s" % (hgap.raster_final_filename("%s" % (num)))))


'''
fss = r'H:\Hydro_Health\2016\Raster_Final\PSS\present_ss'
risk_norm = r'H:\NHSP_2_0\Christy\RISK\risk'
out_fldr = r'H:\NHSP_2_0\Christy\HYDROGAP_HEALTH'
forecast_yrs = 10

# Output Parameters
hydro_gap_present = os.path.join(out_fldr, "H_Gap")
hydro_gap_future = os.path.join(out_fldr, "H_Gap_fut" + str(forecast_yrs))
hydro_health_present = os.path.join(out_fldr, "H_Health")
hydro_health_future = os.path.join(out_fldr, "H_Health" + str(forecast_yrs))

# Convert Inputs to Rasters
dss_r = arcpy.sa.Raster(dss)
pss_r = arcpy.sa.Raster(pss)
fss_r = arcpy.sa.Raster(fss)
risk_norm_r = arcpy.sa.Raster(risk_norm)

# Calculate Present and Future Hydrographic Gap
hydro_gap_present_term = dss_r - pss_r
hydro_gap_present_term.save(hydro_gap_present)
hydro_gap_future_term = dss_r - fss_r
hydro_gap_future_term.save(hydro_gap_future)

# Calculate Hydrographic Health
hydro_health_term = arcpy.sa.Raster(hydro_gap_present) * risk_norm
hydro_health_term.save(hydro_health_present)

hydro_health_fut_term = arcpy.sa.Raster(hydro_gap_future) * risk_norm
hydro_health_fut_term.save(hydro_health_future)'''
