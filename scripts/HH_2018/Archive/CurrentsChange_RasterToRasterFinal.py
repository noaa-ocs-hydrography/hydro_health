# Currents: Raster to Raster Final
# This script follows Currents_RawToVectorProcessed
import arcpy
import os

# Input Files
out_fldr = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\CURRENTS"
mbounds = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\CURRENTS\Testing.gdb\grid_ext_all"
c_raster = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\CURRENTS\currents_r"
yrs_raster = "Path to superseded SURDEX"
f_yrs = 10

# Output Files
o_raster = os.path.join(out_fldr, "ones_R")
#f_yrs_raster = os.path.join(out_fldr, "forecast_R")
#c_out = os.path.join(out_fldr, "Currents_%03dyr")%f_yrs
c_f = os.path.join(out_fldr, "Cur_F_%03dyr_F") % f_yrs
c_c = os.path.join(out_fldr, "Cur_C_F")

# Set Environmental Parameters
r = (arcpy.sa.Raster(mbounds)).extent
arcpy.env.extent = arcpy.Extent(r.XMin, r.YMin, r.XMax, r.YMax)
arcpy.env.snapRaster = mbounds

# Create Model Bounds Raster with all values equal to 1
maxval = float(arcpy.GetRasterProperties_management(mbounds, "MAXIMUM")[0])
minval = float(arcpy.GetRasterProperties_management(mbounds, "MINIMUM")[0])

rc_val = "NODATA NODATA;%s %s 1" % (minval, maxval)
r = arcpy.sa.Reclassify(mbounds, "VALUE", rc_val, "DATA")
r.save(o_raster)

# Create Forecast Raster, Multiply by Current Raster and Export
o_ras = arcpy.Raster(o_raster)
c_ras = arcpy.Raster(c_raster)
f_yrs_ras = o_ras * f_yrs
c_f_yrs_ras = f_yrs_ras * c_ras
c_f_yrs_ras.save(c_f)

# Calculate Changeability Current Raster and Export
s_yrs_ras = arcpy.Raster(yrs_raster)
c_syrs_ras = c_ras * s_yrs_ras
c_syrs_ras.save(c_c)
