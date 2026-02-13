# Risk Normalization: Bounds of Risk Parameters
'''
# The bounds of the risk parameter are normalized by the number of consequence and likelihood parameters
# contributing to the risk at any given location. The bounds of the likelihood and consequence parameters
# were defined as outlined below:

# Likelihood Parameters
# Traffic Density: Model Bounds (notes: can not differenciate between no AIS data because no boats and no AIS data because outside of bounds of AIS transponder, assuming all AIS boats detected because no mechanism/data to differenciate between the two)
# Known Hazards: CATZOC B bounds clipped to ENC bounds buffered by 2 nautical miles (note: 2 nm buffer is because source ENC points were buffered by 2 nm)
# Groundings: Model Bounds
# Depth: ENC Bounds (unbuffered; notes: used depth raster because depth raster limited to ENC bounds)
# CATZOC: Model Bounds (notes: all areas within model extents were assigned a CATZOC value, if no CATZOC value, assigned CATZOC = 6 (5 in model binning))
# Reported Errors: ENC Bounds buffered by 4 nautical miles (note: 4 nm buffer is because source ENC points were buffered by 4 nm)

# Consequence Parameters
# Passenger Search and Rescue: Passenger Vessel Present Raster (note: only considered risk to passenger vessel if passenger vessel transiting in the area)
# Non-Passenger Search and Rescue: Model Bounds (note: areas where vessels not transitting were still factored into model, but only received a non-pass SAR consequence value if > 600 nm from SAR station).
# Damage to Ship: SBDARE Bounds
# Tanker Reefs: ENC Bounds buffered by 20 nautical miles (note: 2 nm buffer is because source ENC points were buffered by 2 nautical miles. Areas outside of this extent were not assigned a Tanker Reefs Consequence value).
# Non-Tanker Reefs: Model Bounds (notes: Although dataset is limited to ENC bounds, a non-tanker transiting anywhere within the model extents is at least assigned a non-tanker reef consequence value of "0")
# Tanker Sanctuaries: Model Bounds (notes: comprehensive dataset)
# Non-Tankers Sanctuaries: Model Bounds (notes: comprehensive dataset)
# Ports: Model Bounds

# Risk is normalized by taking the log base 5 of the Risk term and dividing by (1 + log base 5 (Number of Consequence Terms) + (Number of Likelihood Terms))
'''

# Import Modules
import arcpy
import os
from arcpy import sa
from arcpy import env

# Input Parameters
model_bounds = r'H:\NHSP_2_0\GRID\Grid.gdb\Grid_Model_Bounds_Raster'
catzoc_b = r'H:\Hydro_Health\2016\Raster_Auxiliary\catzoc_bcdu'
enc_bounds = r'H:\Hydro_Health\2016\Raster_Final\depth_final'  # Depth bounds used as ENC bounds because Depth Raster limited to extents of available ENC data
pass_length = r'H:\Hydro_Health\2016\Raster\sar_passl'
sbdare = r'H:\Hydro_Health\2016\Raster_Final\ship_dam_f'
out_fldr = r"H:\NHSP_2_0\Christy\RISK_NORMALIZATION"
out_gdb = r"H:\NHSP_2_0\Christy\RISK_NORMALIZATION\RISK_NORM.gdb"
cellsize = 500

# Variables
arcpy.env.snapRaster = model_bounds
arcpy.env.extent = model_bounds

buff_2nm = 3704
buff_4nm = 7408
buff_20nm = 37040

# Output Parameters
model_bounds_rc = os.path.join(out_fldr, "model_bnd_rc")
enc_bounds_p = os.path.join(out_gdb, "enc_bounds_poly")
enc_bounds_p_2buff = os.path.join(out_gdb, "enc_bounds_poly_2nmbuff")
enc_bounds_p_4buff = os.path.join(out_gdb, "enc_bounds_poly_4nmbuff")
enc_bounds_p_20buff = os.path.join(out_gdb, "enc_bounds_poly_20nmbuff")

enc_bound_r_2buff = os.path.join(out_fldr, "enc_2buf")
enc_bound_r_4buff = os.path.join(out_fldr, "enc_4buf")
enc_bound_r_20buff = os.path.join(out_fldr, "enc_20buf")
enc_bound_r_4buff_rc = os.path.join(out_fldr, "enc_4buf_rc")
enc_bound_r_4buff_mask = os.path.join(out_fldr, "rep_mask")
enc_bound_r_20buff_rc = os.path.join(out_fldr, "enc_20buf_rc")
enc_bound_r_20buff_mask = os.path.join(out_fldr, "reef_t_mask")

catzoc_b_enc_clip = os.path.join(out_fldr, "catzocb_enc")
catzoc_b_enc_clip_rc = os.path.join(out_fldr, "catzb_enc_rc")
catzoc_b_enc_clip_mask = os.path.join(out_fldr, "navhaz_mask")
enc_bounds_rc = os.path.join(out_fldr, "enc_bnd_rc")
enc_bounds_mask = os.path.join(out_fldr, "depth_mask")

sbdare_rc = os.path.join(out_fldr, "ship_dam_rc")
pass_length_rc = os.path.join(out_fldr, "passl_rc")
pass_length_mask = os.path.join(out_fldr, "sar_p_mask")

risk_norm = os.path.join(out_fldr, "Risk_Norm")


# Number of Likelihood Terms  (Variables Defined Below)
# Model Bounds: Traffic Density, Groundings, CATZOC
# Reclassify Model Bounds as Binary
mb_min = float(arcpy.GetRasterProperties_management(model_bounds, "MINIMUM").getOutput(0))
mb_max = float(arcpy.GetRasterProperties_management(model_bounds, "MAXIMUM").getOutput(0))
mb_remap = arcpy.sa.RemapRange([[mb_min, mb_max, 1], ["NODATA", "NODATA"]])
mb_reclass = arcpy.sa.Reclassify(model_bounds, "Value", mb_remap, "NODATA")
mb_reclass.save(model_bounds_rc)


# CATZOC B Clipped to ENC Bounds Buffered by 2 nm: Known Hazards
# Buffer ENC Bounds by 2 nm. Convert to Polygon and Buffer since known hazard features were buffered with a circle-like buffer,
# not a square buffer which is used in the Raster version of buffer, Euclidian Distance. Then clip CATOC by Buffered ENC Bounds
# and set equal to known hazards bounds
arcpy.RasterToPolygon_conversion(enc_bounds, enc_bounds_p, "NO_SIMPLIFY", "VALUE")
arcpy.RepairGeometry_management(enc_bounds_p)
arcpy.Buffer_analysis(enc_bounds_p, enc_bounds_p_2buff, str(buff_2nm) + " meters")
arcpy.RepairGeometry_management(enc_bounds_p_2buff)

field = "dis"
arcpy.AddField_management(enc_bounds_p_2buff, field, "DOUBLE")
arcpy.CalculateField_management(enc_bounds_p_2buff, field, 1, "PYTHON_9.3")
arcpy.PolygonToRaster_conversion(enc_bounds_p_2buff, field, enc_bound_r_2buff, "CELL_CENTER", field, cellsize)

ce = arcpy.sa.ExtractByMask(catzoc_b, enc_bound_r_2buff)
ce.save(catzoc_b_enc_clip)  # VERIFY 1 and NAN

cbe_remap = arcpy.sa.RemapRange([[1, 1, 1], ["NODATA", "NODATA", 0]])  # WILL THIS WORK????????
cbe = arcpy.sa.Reclassify(catzoc_b_enc_clip, "Value", cbe_remap, "NODATA")
cbe.save(catzoc_b_enc_clip_rc)

cbe_mask_remap = arcpy.sa.RemapRange([[1, 1, 1], [0, 0, "NODATA"]])  # WILL THIS WORK????????
cbe = arcpy.sa.Reclassify(catzoc_b_enc_clip_rc, "Value", cbe_mask_remap, "NODATA")
cbe.save(catzoc_b_enc_clip_mask)

# ENC Bounds: Depth
# Reclassify ENC Bounds as Binary
enc_min = float(arcpy.GetRasterProperties_management(enc_bounds, "MINIMUM").getOutput(0))
enc_max = float(arcpy.GetRasterProperties_management(enc_bounds, "MAXIMUM").getOutput(0))
enc_remap = arcpy.sa.RemapRange([[enc_min, enc_max, 1], ["NODATA", "NODATA", 0]])  # WILL THIS WORK????????
enc_reclass = arcpy.sa.Reclassify(enc_bounds, "Value", enc_remap, "NODATA")
enc_reclass.save(enc_bounds_rc)

enc_mask_remap = arcpy.sa.RemapRange([[1, 1, 1], [0, 0, "NODATA"]])  # WILL THIS WORK????????
enc_mask_reclass = arcpy.sa.Reclassify(enc_bounds_rc, "Value", enc_mask_remap, "NODATA")
enc_mask_reclass.save(enc_bounds_mask)

# ENC Bounds Buffered by 4 nm: Reported Errors
arcpy.Buffer_analysis(enc_bounds_p, enc_bounds_p_4buff, str(buff_4nm) + " meters")
arcpy.RepairGeometry_management(enc_bounds_p_4buff)

field = "dis"
arcpy.AddField_management(enc_bounds_p_4buff, field, "DOUBLE")
arcpy.CalculateField_management(enc_bounds_p_4buff, field, 1, "PYTHON_9.3")
arcpy.PolygonToRaster_conversion(enc_bounds_p_4buff, field, enc_bound_r_4buff, "CELL_CENTER", field, cellsize)

enc4_remap = arcpy.sa.RemapRange([[1, 1, 1], ["NODATA", "NODATA", 0]])
enc4 = arcpy.sa.Reclassify(enc_bound_r_4buff, "Value", enc4_remap, "NODATA")
enc4.save(enc_bound_r_4buff_rc)

enc4_mask_remap = arcpy.sa.RemapRange([[1, 1, 1], [0, 0, "NODATA"]])
enc4_mask = arcpy.sa.Reclassify(enc_bound_r_4buff_rc, "Value", enc4_mask_remap, "NODATA")
enc4_mask.save(enc_bound_r_4buff_mask)

# Number of Consequence Terms
# Passenger Vessel Present Raster: SAR Passenger
psar_min = float(arcpy.GetRasterProperties_management(pass_length, "MINIMUM").getOutput(0))
psar_max = float(arcpy.GetRasterProperties_management(pass_length, "MAXIMUM").getOutput(0))
psar_remap = arcpy.sa.RemapRange([[psar_min, psar_max, 1], ["NODATA", "NODATA", 0]])  # WILL THIS WORK????????
psar_reclass = arcpy.sa.Reclassify(pass_length, "Value", psar_remap, "NODATA")
psar_reclass.save(pass_length_rc)

psar_mask_remap = arcpy.sa.RemapRange([[1, 1, 1], [0, 0, "NODATA"]])
psar_mask = arcpy.sa.Reclassify(pass_length_rc, "Value", psar_mask_remap, "NODATA")
psar_mask.save(pass_length_mask)

# Damage to Ship
# Reclassify Damage to Ship Bounds as Binary
sb_min = float(arcpy.GetRasterProperties_management(sbdare, "MINIMUM").getOutput(0))
sb_max = float(arcpy.GetRasterProperties_management(sbdare, "MAXIMUM").getOutput(0))
sb_remap = arcpy.sa.RemapRange([[sb_min, sb_max, 1], ["NODATA", "NODATA", 0]])
mb_reclass = arcpy.sa.Reclassify(sbdare, "Value", sb_remap, "NODATA")
mb_reclass.save(sbdare_rc)


# Model Bounds: Non-Pass SAR, Non-Tank Reefs


# ENC Bounds: Damage to Ship

# ENC Bounds Buffered by 20 nm: Tanker Reefs
arcpy.Buffer_analysis(enc_bounds_p, enc_bounds_p_20buff, str(buff_20nm) + " meters")
arcpy.RepairGeometry_management(enc_bounds_p_20buff)

field = "dis"
arcpy.AddField_management(enc_bounds_p_20buff, field, "DOUBLE")
arcpy.CalculateField_management(enc_bounds_p_20buff, field, 1, "PYTHON_9.3")
arcpy.PolygonToRaster_conversion(enc_bounds_p_20buff, field, enc_bound_r_20buff, "CELL_CENTER", field, cellsize)

enc20_remap = arcpy.sa.RemapRange([[1, 1, 1], ["NODATA", "NODATA", 0]])
enc20 = arcpy.sa.Reclassify(enc_bound_r_20buff, "Value", enc20_remap, "NODATA")
enc20.save(enc_bound_r_20buff_rc)

enc20_mask_remap = arcpy.sa.RemapRange([[1, 1, 1], [0, 0, "NODATA"]])
enc20_mask = arcpy.sa.Reclassify(enc_bound_r_20buff_rc, "Value", enc20_mask_remap, "NODATA")
enc20_mask.save(enc_bound_r_20buff_mask)


# Define Variables
# Likelihood Parameters
traffic_den = model_bounds_rc
groundings = model_bounds_rc
catzoc = model_bounds_rc
known_hazards = catzoc_b_enc_clip_rc
depth = enc_bounds_rc
rep_err = enc_bound_r_4buff_rc

# Consequence Parameters
pass_sar = pass_length_rc
nonpass_sar = model_bounds_rc
tank_reef = enc_bound_r_20buff_rc
nontank_reef = model_bounds_rc
tank_sanc = model_bounds_rc
nontank_sanc = model_bounds_rc
ports = model_bounds_rc
damage_ship = sbdare_rc

risk_list = [traffic_den, groundings, catzoc, known_hazards, depth, rep_err, pass_sar, nonpass_sar, tank_reef, nontank_reef, tank_sanc, nontank_sanc, ports, damage_ship]
risk_norm_stat = arcpy.sa.CellStatistics(risk_list, "SUM", "NODATA")
risk_norm_stat.save(risk_norm)

'''
######## HOW TO ADD RASTERS ###########
import arcpy
from arcpy import env

# Set the current workspace
#
env.workspace = "C:/Data/DEMS"

#Check out ArcGIS Spatial Analyst extension license
arcpy.CheckOutExtension("Spatial")


# Get a list of ESRI GRIDs from the workspace and print
#

rasterList = arcpy.ListRasters("*", "GRID")

#Step through the list of raster names
for rasname in rasterList:
    #Cast rasname as raster before adding
    listras = Raster(rasname)
    #Add each raster to an output raster.
    #The previous step of casting as a Raster will invoke
    #the Spatial Analyst Addition function.
    outras += listras

#Save output Raster
outras.save("C:/temp/outras")

'''
