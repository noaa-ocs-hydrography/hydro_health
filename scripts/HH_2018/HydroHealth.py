# Hydro Health
import arcpy
import os
import copy
from arcpy import sa
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values
hgap = Parameters("HGAP")
risk = ["RISK"]

# Input Parameters
forecast_years_list = [5, 10]  # User-Defined

hgap_fn = hgap.raster_final_filename()
risk_fn = risk.raster_final_filename()

# Output Parameters


hh_list = list(forecast_years_list)
hh_list.append("")

fgap_fn_list = []

# Calculate Hydro Health for all Future Survey Score Files
for num in forecast_years_list:
    fgap_fn = hgap.raster_final_filename("_%s" % (num))
    fgap_fn_list.append(fgap_fn)
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
