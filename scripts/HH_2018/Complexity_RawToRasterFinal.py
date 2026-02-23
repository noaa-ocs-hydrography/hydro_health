# Complexity Processing
# This script assembles a complexity raster using geomorphological bounds (e.g. continental slope, abyss, and trenches) as well as the slope
# of the seafloor. The continental shelf is assessed for seafloor slope and defined as complex m > X and simple m < X, while trenches
# and the continental slope were defined as complex and the abyss was characterized as simple.

import arcpy
import os
import time
from arcpy.sa import *
arcpy.CheckOutExtension("Spatial")

# Input Data
out_gdb = r'H:\NHSP_2_0\Christy\COMPLEXITY\Complexity.gdb'
out_fldr = r'H:\NHSP_2_0\Christy\COMPLEXITY'
bounds_r = r"H:\NHSP_2_0\GRID\Grid.gdb\Grid_Model_Bounds_Raster"
bounds_p = r'H:\NHSP_2_0\Christy\Grid\Grid.gdb\Grid_Model_Bounds'
# shelf = r'H:\NHSP_2_0\Christy\COMPLEXITY\Complexity.gdb\Shelf' # Needed?
slope = r'H:\NHSP_2_0\Christy\COMPLEXITY\Complexity.gdb\Slope'
abyss = r'H:\NHSP_2_0\Christy\COMPLEXITY\Complexity.gdb\Abyss'
trench = r'H:\NHSP_2_0\Christy\COMPLEXITY\Complexity.gdb\Trenches'
depth = r'H:\NHSP_2_0\Christy\DEPTH\depth_30m'


# Input Variables
cellSize = 500
des_sprj_code = 102008
z_threshold = 0.5  # Pinet, Paul R. Invitation to Oceanography 4th edition. Sudbury, Massachusetts: Jones and Bartlett Publishers, 1993. (p. 38)

# Output variables
slope_p = os.path.join(out_gdb, "slope_p")  # Delete
abyss_p = os.path.join(out_gdb, "abyss_p")  # Delete
trench_p = os.path.join(out_gdb, "trench_p")  # Delete

slope_rt = os.path.join(out_fldr, "slope_rt")  # Delete
abyss_rt = os.path.join(out_fldr, "abyss_rt")  # Delete
trench_rt = os.path.join(out_fldr, "trench_rt")  # Delete

slope_r = os.path.join(out_fldr, "slope_r")  # Delete
abyss_r = os.path.join(out_fldr, "abyss_r")  # Delete
trench_r = os.path.join(out_fldr, "trench_r")  # Delete

bounds_rc = os.path.join(out_fldr, "bounds_rc")  # Delete
slope_rc = os.path.join(out_fldr, "slope_rc")  # Delete
abyss_rc = os.path.join(out_fldr, "abyss_rc")  # Delete
trench_rc = os.path.join(out_fldr, "trench_rc")  # Delete

abyss_slope_r = os.path.join(out_fldr, "abyss_slope")  # Delete
abyss_slope_p = os.path.join(out_gdb, "abyss_slope_p")  # Delete

shelf_new_p = os.path.join(out_gdb, "shelf_new")  # Delete
shelf_new_r = os.path.join(out_fldr, "shelf_new_r")  # Delete

depth_slope = os.path.join(out_fldr, "depth_30m_m")
depth_slope_shelf = os.path.join(out_fldr, "depth30m_shf")
depth_rc = os.path.join(out_fldr, "depth_m_rc")  # Get rid of zeros??

complexity = os.path.join(out_fldr, "complexity_f")
complexity_simple = os.path.join(out_fldr, "simple_fin")
complexity_complex = os.path.join(out_fldr, "complex_fin")

# Project Data and Convert to Raster
d_field = "dis"
unprjs = [slope, abyss, trench]
prjs = [slope_p, abyss_p, trench_p]
rasts = [slope_rt, abyss_rt, trench_rt]
rass = [slope_r, abyss_r, trench_r]
tt = time.time()
for unprj, prj, rast, ras in zip(unprjs, prjs, rasts, rass):
    t_start = time.time()
    #arcpy.Project_management(unprj, prj, des_sprj_code)
    #print("Finished Project Raster in %.1f secs"%(time.time()-t_start))
    arcpy.RepairGeometry_management(prj)
    arcpy.AddField_management(prj, d_field, "DOUBLE")
    arcpy.CalculateField_management(prj, d_field, 1, "PYTHON_9.3")
    print(("Finished Repair Geometry and Calculate Field in %.1f secs" % (time.time() - t_start)))
    arcpy.PolygonToRaster_conversion(prj, d_field, rast, "CELL_CENTER", d_field, cellSize)
    print(("Finished Polygon to Raster in %.1f secs" % (time.time() - t_start)))
    e = arcpy.sa.ExtractByMask(rast, bounds_r)
    e.save(ras)
    print(("Finished Extract in %.1f secs" % (time.time() - t_start)))
print(("Finished Full Process at %.1f secs" % (time.time() - tt)))


# Reclassify Rasters
# Bounds & Abyss = Simple
# Slope & Trench = Complex
s_min = min(float(arcpy.GetRasterProperties_management(bounds_r, "MINIMUM").getOutput(0)), float(arcpy.GetRasterProperties_management(abyss_r, "MINIMUM").getOutput(0)))
s_max = max(float(arcpy.GetRasterProperties_management(bounds_r, "MAXIMUM").getOutput(0)), float(arcpy.GetRasterProperties_management(abyss_r, "MAXIMUM").getOutput(0)))
s_remap = RemapRange([[s_min, s_max, 1]])
for s_unclass, s_class in ([bounds_r, bounds_rc], [abyss_r, abyss_rc]):
    s_reclass = arcpy.sa.Reclassify(s_unclass, "VALUE", s_remap, "NODATA")
    s_reclass.save(s_class)

c_min = min(float(arcpy.GetRasterProperties_management(slope_r, "MINIMUM").getOutput(0)), float(arcpy.GetRasterProperties_management(trench_r, "MINIMUM").getOutput(0)))
c_max = max(float(arcpy.GetRasterProperties_management(slope_r, "MAXIMUM").getOutput(0)), float(arcpy.GetRasterProperties_management(trench_r, "MAXIMUM").getOutput(0)))
c_remap = RemapRange([[c_min, c_max, 100]])
for c_unclass, c_class in ([slope_r, slope_rc], [trench_r, trench_rc]):
    c_reclass = arcpy.sa.Reclassify(c_unclass, "VALUE", c_remap, "NODATA")
    c_reclass.save(c_class)

# Merge Abyss and Slope Area, Convert to Polygon, and Erase from model Bounds to identify shelf areas
arcpy.MosaicToNewRaster_management([slope_rc, abyss_rc], os.path.split(abyss_slope_r)[0], os.path.split(abyss_slope_r)[1], des_sprj_code, "32_BIT_SIGNED", str(cellSize), "1", "MINIMUM", "FIRST")
arcpy.RasterToPolygon_conversion(abyss_slope_r, abyss_slope_p, "NO_SIMPLIFY", "VALUE")
arcpy.RepairGeometry_management(abyss_slope_p)
arcpy.Erase_analysis(bounds_p, abyss_slope_p, shelf_new_p)
arcpy.RepairGeometry_management(shelf_new_p)

# Create Depth Slope Raster and Extract Shelf_New Area
m = arcpy.sa.Slope(depth, "DEGREE", 1)  # May also be Slope())
m.save(depth_slope)
e_depth = arcpy.sa.ExtractByMask(depth_slope, shelf_new_p)
e_depth.save(depth_slope_shelf)

# Reclassify Depth Slope Shelf by user-input value after examining depth slope shelf raster
z_min = float(arcpy.GetRasterProperties_management(depth_slope_shelf, "MINIMUM").getOutput(0))
z_max = float(arcpy.GetRasterProperties_management(depth_slope_shelf, "MAXIMUM").getOutput(0))
z_remap = RemapRange([[z_min, z_threshold, 1], [z_threshold, z_max, 100]])
z_reclass = arcpy.sa.Reclassify(depth_slope_shelf, "Value", z_remap, "NODATA")
z_reclass.save(depth_rc)

# Mosaic raster classified bounds, abyss, slope, depth (slope) into one complexity raster
complexity_rasters = [bounds_rc, depth_rc, trench_rc, slope_rc, abyss_rc]
arcpy.MosaicToNewRaster_management(complexity_rasters, os.path.split(complexity)[0], os.path.split(complexity)[1], des_sprj_code, "32_BIT_SIGNED", str(cellSize), "1", "MAXIMUM", "FIRST")

# Export Simple and Complex rasters separately
s_remap_f = RemapRange([[1, 1, 1], [1, 100, "NODATA"]])
c_remap_f = RemapRange([[1, 1, "NODATA"], [1, 100, 100]])

c_simple = arcpy.sa.Reclassify(complexity, "Value", s_remap_f, "NODATA")
c_simple.save(complexity_simple)

c_complex = arcpy.sa.Reclassify(complexity, "Value", c_remap_f, "NODATA")
c_complex.save(complexity_complex)
