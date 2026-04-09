# Risk
import arcpy
import os
from arcpy import sa
from arcpy.sa import *
from HSTB.ArcExt.NHSP.globals import Parameters

ais = Parameters("AIS")
z = Parameters("Depth")
qual = Parameters("SurveyQuality")
re = Parameters("RepErr")
navhaz = Parameters("KnownHazards")
ground = Parameters("Groundings")
sar = Parameters("SAR")
aux = Parameters("AUX")
sbdare = Parameters("SBDARE")
reef = Parameters("Reefs")
port = Parameters("Ports")
risk = Parameters("RISK")

# Input Parameters
# Likelihood Terms
like_traffic_all = ais.raster_final_filename("a_ac_u")
like_traffic_pass = ais.raster_final_filename("passu")
like_traffic_nopass = ais.raster_final_filename("np_ac")
like_traffic_tank = ais.raster_final_filename("tanku")
like_traffic_notank = ais.raster_final_filename("ntk_ac")
like_depth = z.raster_final_filename("500m")
like_survey_quality = qual.raster_final_filename()
like_rep_err = re.raster_final_filename()
like_nav_haz = navhaz.raster_final_filename()
like_ground = ground.raster_final_filename()

# Consequence Terms
cons_sar_pass = sar.raster_final_filename("pass")
cons_sar_nopass = sar.raster_final_filename("npass")
cons_ship_dam = sbdare.raster_final_filename()
cons_reef_tank = reef.raster_final_filename("tk")
cons_reef_notank = reef.raster_final_filename("ntk")
cons_port = port.raster_final_filename()

eez = aux["eez_1_ras"]

# Output Variables
like_base_all = risk.working_rastername("lbase_a")
like_base_pass = risk.working_rastername("lbase_p2")
like_base_npass = risk.working_rastername("lbase_np")
like_base_tank = risk.working_rastername("lbase_t2")
like_base_ntank = risk.working_rastername("lbase_nt")

risk_ship_dam = risk.working_rastername("csbdare")
risk_port = risk.working_rastername("cport")
risk_sar_pass = risk.working_rastername("csar_p")
risk_sar_npass = risk.working_rastername("csar_np")
risk_reef_tank = risk.working_rastername("creef_t")
risk_reef_ntank = risk.working_rastername("creef_nt")

risk_rf = risk.raster_final_filename()

# Convert Terms to Arcpy Rasters
# Likelihood Terms
like_traffic_all_r = arcpy.sa.Raster(like_traffic_all)
like_traffic_pass_r = arcpy.sa.Raster(like_traffic_pass)
like_traffic_nopass_r = arcpy.sa.Raster(like_traffic_nopass)
like_traffic_tank_r = arcpy.sa.Raster(like_traffic_tank)
like_traffic_notank_r = arcpy.sa.Raster(like_traffic_notank)

like_depth_r = arcpy.sa.Raster(like_depth)
like_survey_quality_r = arcpy.sa.Raster(like_survey_quality)
like_rep_err_r = arcpy.sa.Raster(like_rep_err)
like_nav_haz_r = arcpy.sa.Raster(like_nav_haz)
like_ground_r = arcpy.sa.Raster(like_ground)

# Consequence Terms
cons_ship_dam_r = arcpy.sa.Raster(cons_ship_dam)
cons_port_r = arcpy.sa.Raster(cons_port)
cons_sar_pass_r = arcpy.sa.Raster(cons_sar_pass)
cons_sar_nopass_r = arcpy.sa.Raster(cons_sar_nopass)
cons_reef_tank_r = arcpy.sa.Raster(cons_reef_tank)
cons_reef_notank_r = arcpy.sa.Raster(cons_reef_notank)

# Generate Likelihood Base Terms
like_base_all_term = Log10(Power(10, like_traffic_all_r) + Power(10, like_depth_r) + Power(10, like_survey_quality_r) + Power(10, like_rep_err_r) + Power(10, like_nav_haz_r) + Power(10, like_ground_r))
like_base_all_term.save(like_base_all)
like_base_pass_term = Log10(Power(10, like_traffic_pass_r) + Power(10, like_depth_r) + Power(10, like_survey_quality_r) + Power(10, like_rep_err_r) + Power(10, like_nav_haz_r) + Power(10, like_ground_r))
like_base_pass_term.save(like_base_pass)
like_base_npass_term = Log10(Power(10, like_traffic_nopass_r) + Power(10, like_depth_r) + Power(10, like_survey_quality_r) + Power(10, like_rep_err_r) + Power(10, like_nav_haz_r) + Power(10, like_ground_r))
like_base_npass_term.save(like_base_npass)
like_base_tank_term = Log10(Power(10, like_traffic_tank_r) + Power(10, like_depth_r) + Power(10, like_survey_quality_r) + Power(10, like_rep_err_r) + Power(10, like_nav_haz_r) + Power(10, like_ground_r))
like_base_tank_term.save(like_base_tank)
like_base_ntank_term = Log10(Power(10, like_traffic_notank_r) + Power(10, like_depth_r) + Power(10, like_survey_quality_r) + Power(10, like_rep_err_r) + Power(10, like_nav_haz_r) + Power(10, like_ground_r))
like_base_ntank_term.save(like_base_ntank)

# Convert Likelihood Base Terms to Raster
like_base_all_term_r = arcpy.sa.Raster(like_base_all)
like_base_pass_term_r = arcpy.sa.Raster(like_base_pass)
like_base_npass_term_r = arcpy.sa.Raster(like_base_npass)
like_base_tank_term_r = arcpy.sa.Raster(like_base_tank)
like_base_ntank_term_r = arcpy.sa.Raster(like_base_ntank)

# Generate Individual Risk Terms
risk_ship_dam_term = cons_ship_dam_r * like_base_all_term_r
risk_ship_dam_term.save(risk_ship_dam)
risk_port_term = cons_port_r * like_base_all_term_r
risk_port_term.save(risk_port)
risk_sar_pass_term = cons_sar_pass_r * like_base_pass_term_r
risk_sar_pass_term.save(risk_sar_pass)
risk_sar_npass_term = cons_sar_nopass_r * like_base_npass_term_r
risk_sar_npass_term.save(risk_sar_npass)
risk_reef_tank_term = cons_reef_tank_r * like_base_tank_term_r
risk_reef_tank_term.save(risk_reef_tank)
risk_reef_ntk_term = cons_reef_notank_r * like_base_ntank_term_r
risk_reef_ntk_term.save(risk_reef_ntank)

# Sum Risk Terms
risk_terms_list = [risk_ship_dam, risk_port, risk_sar_pass, risk_sar_npass, risk_reef_tank, risk_reef_ntank]
risk_unnorm_term = arcpy.sa.CellStatistics(risk_terms_list, "SUM", "DATA")
risk_unnorm_term.save(risk_rf)
