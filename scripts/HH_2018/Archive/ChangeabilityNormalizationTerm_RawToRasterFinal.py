# Risk Normalization: Bounds of Risk Parameters
import arcpy
import os

'''
# The bounds of the changeability parameter are normalized by the number of change agents
# contributing to the changeability at any given location. The bounds of the change agents were defined as outlined below:

# Change Agents
# Human Debris: Model Bounds
# Storms: Extent of Final Storm Dataset
# Currents: Extent of Final Currents Dataset
'''

# Input Parameters
model_bounds = r'H:\GRID\Grid.gdb\Grid_Model_Bounds_Raster'
model_bounds_rc = r'H:\Christy\RISK_NORMALIZATION\model_bnd_rc'
storms_c_extents = r'Y:\Hydro_Health\2016\Raster_Final\storm_c_f'
storms_f_extents = r'Y:\Hydro_Health\2016\Raster_Final\storm_f_f'
curr_extents = r'Y:\Hydro_Health\2016\Raster_Final\curr_final'
out_fldr = r'H:\Christy\CHANGE_NORMALIZATION'

# Output Parameters
storm_change_norm = os.path.join(out_fldr, "storm_c_norm")
storm_fore_norm = os.path.join(out_fldr, "storm_f_norm")
curr_norm = os.path.join(out_fldr, "curr_norm")
change_decay = os.path.join(out_fldr, "change_c_norm")
forecast_decay = os.path.join(out_fldr, "change_f_norm")

arcpy.env.snapRaster = model_bounds
arcpy.env.extent = model_bounds

# Human Debris
hd_norm = model_bounds_rc

# Storm Change Extents
sc_min = float(arcpy.GetRasterProperties_management(storms_c_extents, "MINIMUM").getOutput(0))
sc_max = float(arcpy.GetRasterProperties_management(storms_c_extents, "MAXIMUM").getOutput(0))
sc_remap = arcpy.sa.RemapRange([[sc_min, sc_max, 1], ["NODATA", "NODATA", 0]])
sc_rc = arcpy.sa.Reclassify(storms_c_extents, "Value", sc_remap, "NODATA")
sc_rc.save(storm_change_norm)

# Storm Forecast Extents
sf_min = float(arcpy.GetRasterProperties_management(storms_f_extents, "MINIMUM").getOutput(0))
sf_max = float(arcpy.GetRasterProperties_management(storms_f_extents, "MAXIMUM").getOutput(0))
sf_remap = arcpy.sa.RemapRange([[sf_min, sf_max, 1], ["NODATA", "NODATA", 0]])
sf_rc = arcpy.sa.Reclassify(storms_f_extents, "Value", sf_remap, "NODATA")
sf_rc.save(storm_fore_norm)

# Currents Extents
c_min = float(arcpy.GetRasterProperties_management(curr_extents, "MINIMUM").getOutput(0))
c_max = float(arcpy.GetRasterProperties_management(curr_extents, "MAXIMUM").getOutput(0))
c_remap = arcpy.sa.RemapRange([[c_min, c_max, 1], ["NODATA", "NODATA", 0]])
c_rc = arcpy.sa.Reclassify(curr_extents, "Value", c_remap, "NODATA")
c_rc.save(curr_norm)

# Normalization
change_decay_list = [hd_norm, storm_change_norm, curr_norm]
forecast_decay_list = [hd_norm, storm_fore_norm, curr_norm]

change_decay_stat = arcpy.sa.CellStatistics(change_decay_list, "SUM", "DATA")
change_decay_stat.save(change_decay)
forecast_decay_stat = arcpy.sa.CellStatistics(forecast_decay_list, "SUM", "DATA")
forecast_decay_stat.save(forecast_decay)
