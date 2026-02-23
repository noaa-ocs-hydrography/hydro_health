# 2018 Model Run: Reclassify 30 m Depth Grid following the below rubric:
# > 100 m = 1
# 50 - 100 m = 2
# 20 - 49 m = 3
# 0 - 19 m = 4

# Define Variables
import os
import arcpy
import collections
import time
from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP.globals import Parameters, initial_values
z = Parameters("Depth")
aux = Parameters("AUX")

# 2018 Model Run
# Input Variables
z_30m = z.raster_filename("30m")
eez_r = aux["eez_1_ras"]

# Output Variables
z_rc = z.raster_classified_filename("30m")
z_rf_t = z.working_rastername("30m_rft")
z_rf = z.raster_final_filename("30m")

# Classify Depth Raster
z_max_t = arcpy.GetRasterProperties_management(z_30m, "MAXIMUM")
z_max = (float(z_max_t.getOutput(0))) + 100000  # Ensure maximum is not mis-classified.

arcpy.gp.Reclassify_sa(z_30m, "VALUE", "-100 19 4;20 49 3;50 99 2;100 %s 1" % (z_max), z_rc, "DATA")

# Mosaic Classified Depth Raster with EEZ = 1 to assign Likelihood = 1 to all areas where depth information is not available (Puerto Rico)
arcpy.MosaicToNewRaster_management([z_rc, eez_r], os.path.dirname(z_rf_t), os.path.basename(z_rf_t), z.projection_number, "32_BIT_SIGNED", z.cell_size, "1", "MAXIMUM", "FIRST")

# Clip Depth Output to EEZ and Export
e = arcpy.sa.ExtractByMask(z_rf_t, eez_r)
e.save(z_rf)

'''
# Depth Data and Complexity
# Input Variables
enc_sde_1 = r"C:\\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\postgresql_dev_encd_viewer.sde"
ub_sde_1 = r"C:\\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\NHSP2_0.sde"
output_gdb_1 = r"H:\HH_2018\Raw\Risk\Likelihood\Depth\Depth.gdb"
out_fldr = r"H:\HH_2018\Raw\Risk\Likelihood\Depth"
features_1 = '.soundg'
geoms_1 = '_point'
prj_1_str = 'North America Albers Equal Area Conic'
prj_1 = 102008
z_500m = r"H:\HH_2018\Raw\Risk\Likelihood\Depth\Depth.gdb\enc_depth500m"
# z_30m = r"
eez_v = r'H:\HH_2018\Auxiliary\Auxiliary.gdb\Shoreline_EEZ_Buffer707m'
eez_r = r'H:\HH_2018\Auxiliary\Auxiliary.gdb\Shoreline_EEZ_Buffer707m_Raster'
tin_field = "z"
max_edge = 2000
cell_size_500 = 500
cell_size_30 = 30
m = []

# Output Variables
raw = os.path.join(output_gdb_1, "soundg" + "_raw")
raw_prj_fn = "soundg" + "_raw_prj"
raw_prj = os.path.join(output_gdb_1, raw_prj_fn)
tin = os.path.join(out_fldr, "z_tin_raw")
tin_2k = tin + "_2k"  # 2000 m is the maximum edge length of a TIN triangle along the perimeter. 2000 m was selected after reviewing TIN outputs at varying edge lengths. 2000 m well-captures nearshore variability with out gridding across the land.
tin_2k_bounds = os.path.join(output_gdb_1, os.path.split(tin_2k)[1] + "_bounds")
tin_raster_500m = os.path.join(out_fldr, "z_tin_2k_500m")
tin_raster_30m = os.path.join(out_fldr, "z_tin_2k_30m")
z_500m_final = "z_500m"
z_30m_final = "z_30m"

# Extract Sounding Data
bands = ['overview', 'general', 'coastal', 'approach', 'harbor', 'berthing']
for band in bands:
    fn = (enc_sde_1 + '\\' + 'ocsdev.' + "%s" % (band) + features_1 + geoms_1)
    m.append(fn)
arcpy.Merge_management(m, raw)

# Project Dataset (if necessary)
Functions.check_spatial_reference(output_gdb_1, raw, raw_prj_fn, prj_1_str, prj_1)

#@todo
# Current Status
# TIN generated for nearshore areas - nearshore areas were defined by depth points not exceeding 2000 m
# Raster generated of nearshore TIN at 500 m resolution
# Action
# Iterate through grid bounds to
# Generate 30 m (25 m? - ask Patrick what resolution max draft will be at) Raster of nearshore TIN
# Merge 30 m nearshore TIN Raster with 30 m Lucy Grid (z_30m). Where the nearshore TIN Raster takes precedence, where it exists.
# Note, if any clipping necessary to merge 30 m TIN Raster/Lucy Raster, then consider buffering Lucy Grid to prevent data gaps
# Merge 500 m nearshore TIN Raster with 500 m Lucy Grid (z_500m). Where the nearshore TIN Raster takes precedence, where it exists.
# Note, if any clipping necessary to merge 30 m TIN Raster/Lucy Raster, then consider buffering Lucy Grid to prevent data gaps
# Note, the 500 m TIN Raster/Lucy Raster may be able to be completed outside of loop
# Merge all grids
# Convert from numpy to raster and export
# Reclassify and export - I think this should work (it would be smarter if we pulled the min and max values of the raster to replace them with -/+ 100,000
# r = arcpy.sa.Reclassify([insert final raster grid], "VALUE", "-100000 4 3;4 20 4;20 50 3;50 100 2;100 100000 5", "DATA")
# r.save(re_final)
# Talk to Christy about whether or not we want to export other depth-derived products
# Depth < 20 m
# Depth < 40 m
# Create TIN of merged data
try:
    # Create Tin
    arcpy.CreateTin_3d(tin, prj_1, "'%s' %s Mass_Points <None>" % (str(raw_prj), tin_field), "Delaunay")
    # Copy Tin
    arcpy.CopyTin_3d(tin, tin_2k)
    # Delineate Tin Data Area
    arcpy.DelineateTinDataArea_3d(tin_2k, max_edge, "PERIMETER_ONLY")
    # Export extents of delineated TIN
    arcpy.TinDomain_3d(tin_2k, tin_2k_bounds, "POLYGON")  # @todo- fix to output in geodatabase
    # Convert Tin to Raster (500 m adn 30 m resolution)
    arcpy.TinRaster_3d(tin_2k, tin_raster_500m, "FLOAT", "LINEAR", "CELLSIZE %s" % (cell_size_500), "1")
    arcpy.TinRaster_3d(tin_2k, tin_raster_30m, "FLOAT", "LINEAR", "CELLSIZE %s" % (cell_size_30), "1")
except arcpy.ExecuteError:
    print(arcpy.GetMessages())
except Exception as err:
    print(err)

# Generate Mask Files
# Remove extents of 2k max edge raster from shoreline
tt = time.time()
arcpy.TinRaster_3d(tin_2k, tin_raster_30m, "FLOAT", "LINEAR", "CELLSIZE %s" % (cell_size_30), "1")
print("Finished Tin to Raster at %.1f secs" % (time.time() - tt))
mask_2k = os.path.join(output_gdb_1, "mask_2k")
arcpy.Erase_analysis(eez_v, tin_2k_bounds, mask_2k)
# Buffer 2k Mask by 707 m

tt = time.time()
enc_500_erase = os.path.join(out_fldr, "enc_500m_e2k")
ebm_500 = arcpy.sa.ExtractByMask(z_500m, mask_2k)
print("Finished Extract at %.1f secs" % (time.time() - tt))
ebm_500.save(enc_500_erase)

enc_30_erase = os.path.join(out_fldr, "enc_30m_e2k")
ebm_30 = ExtractByMask(z_30m, mask_2k)
ebm_30.save(enc_30_erase)

# Mosaic to New Raster
arcpy.MosaicToNewRaster_management([enc_500_erase, tin_raster_500m], out_fldr, z_500m_final, prj_1, "32_BIT_SIGNED", str(cell_size_500), "1", "MINIMUM", "FIRST")
arcpy.MosaicToNewRaster_management([enc_30_erase, tin_raster_30m], out_fldr, z_30m_final, prj_1, "32_BIT_SIGNED", str(cell_size_30), "1", "MINIMUM", "FIRST")



# Output variables
z_20_out = os.path.join(out_fldr, "depth_LT20m")
z_40_out = os.path.join(out_fldr, "depth_LT40m")
z_like_out = os.path.join(out_fldr, "depth_final")

# Depth Processing
z_prj = OrderedDict()
z_list = [z_30m, z_500m]
for zz in z_list:
	if arcpy.Describe(zz).SpatialReference <> des_sprj_code:
		#arcpy.ProjectRaster_management(zz, zz+"_p", des_sprj_code, "", cellSize)
		z_prj[zz] = zz+"_p"
	else:
		z_prj[zz] = zz

z_rc_l = ["-15000 20 20;20 15000 NODATA", "-15000 40 40;40 15000 NODATA", "-15000 4 3; 4 20 4; 20 50 3; 50 100 2; 100 15000 1"]


# Reclassify Rasters
for value in z_prj.values():
	if value in [z_30m+"_p", z_30m]:
		z_20 = arcpy.sa.Reclassify(value, "VALUE", z_rc_l[0], "NODATA")
		z_20.save(z_20_out)
		z_40 = arcpy.sa.Reclassify(value, "VALUE", z_rc_l[1], "NODATA")
		z_40.save(z_40_out)
	elif value in [z_500m+"_p", z_500m]:
		z_like = arcpy.sa.Reclassify(value, "VALUE", z_rc_l[2], "NODATA")
		z_like.save(z_like_out)
'''
