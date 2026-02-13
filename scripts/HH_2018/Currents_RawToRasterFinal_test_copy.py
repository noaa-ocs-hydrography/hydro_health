# Currents

import os
import arcpy
from arcpy import env

#from HSTB.ArcExt.NHSP import Functions
#from HSTB.ArcExt.NHSP.globals import Parameters

#curr = Parameters("Curr")
#aux = Parameters("AUX")

# FIGURE THIS OUT
#arcpy.env.workspace = r"H:\NHSP_2_0\Christy\CURRENTS"


'''
# Input Files
curr_raw_e = curr.raw_filename("E")  # user defined
curr_raw_w = curr.raw_filename("W")  # user defined

print(curr_raw_e)
print(curr_raw_w)


# Variables
tin_field = "MEANCU"  # user defined
max_edge = 4000  # user defined
sandy_lt20m = aux["sandy_lt20m_ras"]
mbounds = aux["grid_extents_ras"]

# Output variables
curr_e_prj_fn = curr.working_filename("e_prj")
curr_w_prj_fn = curr.working_filename("w_prj")
curr_vp = curr.vector_processed_filename()
c_tin = curr.working_filename("TIN", False)  # Delete
c_tin_bounds = curr.working_filename("Bounds", False)  # Delete
c_raster = curr.raster_filename()
c_raster_rct1 = curr.working_filename("RC_t", False)  # Delete
c_raster_rc = curr.raster_classified_filename()
c_raster_final = curr.raster_final_filename()

# Set Environmental Parameters
r = (arcpy.sa.Raster(mbounds)).extent
arcpy.env.extent = arcpy.Extent(r.XMin, r.YMin, r.XMax, r.YMax)
arcpy.env.snapRaster = mbounds

# Project data, if projected in desired coordinate system, set projected variable to raw variable

curr_raw_e_prj = Functions.check_spatial_reference(curr_raw_e, curr_e_prj_fn, curr.projection_name, curr.projection_number)
curr_raw_w_prj = Functions.check_spatial_reference(curr_raw_w, curr_w_prj_fn, curr.projection_name, curr.projection_number)
'''

# Input Files
#input_dir = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0"
input_dir = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files"
#output_dir = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HydroHealth_update_2019\HH"
output_dir = r"C:\Users\Hilina.Tarekegn\Documents\HydroHealth\Python_Files\HydroHealth_update_2019\HH"
curr_raw_e = os.path.join(input_dir, r"Currents\RAW\tide_data_east.shp")  # user defined
curr_raw_w = os.path.join(input_dir, r"Currents\RAW\tide_data_west.shp")  # user defined
output_prj = arcpy.SpatialReference(102008) # Desired projection of output
cell_size = "500"

# Set Workspace
arcpy.env.workspace = output_dir
arcpy.env.overwriteOutput = True

# Variables
tin_field = "MEANCU"  # user defined
max_edge = 4000  # user defined
base_dir = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0"
sandy_lt20m = os.path.join(base_dir, r"HH_2018\Auxiliary\sandy_lt40m") 
mbounds = os.path.join(base_dir, r"GRID\grid_ext_all")

# Output variables
curr_e_prj_fn = os.path.join(output_dir, r"Working\Currents\e_prj.shp")
curr_w_prj_fn = os.path.join(output_dir, r"Working\Currents\w_prj.shp")
curr_vp = os.path.join(output_dir, r"Vector_Processed\currents_vp.shp")
c_tin = os.path.join(output_dir, r"Working\Currents\curr_TIN") # Delete
c_tin_bounds = os.path.join(output_dir, r"Working\Currents\currents_Bounds.shp") # Delete
c_raster = os.path.join(output_dir, r"Raster\r_curr")
c_raster_rct1 = os.path.join(output_dir, r"Working\Currents\RC_t") # Delete
c_raster_rc = os.path.join(output_dir, r"Raster_Class\rc_curr")
c_raster_final = os.path.join(output_dir, r"Raster_Final\rf_curr")

# Create output directories for all variables if not already existing
if not os.path.exists(os.path.dirname(curr_e_prj_fn)): # Make Working\Currets folder in output directory
    os.makedirs(os.path.dirname(curr_e_prj_fn))
if not os.path.exists(os.path.dirname(curr_vp)): # Make Vector Processed folder in output directory
    os.makedirs(os.path.dirname(curr_vp))
if not os.path.exists(os.path.dirname(c_raster)): # Make Raster folder in output directory
    os.makedirs(os.path.dirname(c_raster))
if not os.path.exists(os.path.dirname(c_raster_rc)): # Make Raster_Class folder in output directory
    os.makedirs(os.path.dirname(c_raster_rc))
if not os.path.exists(os.path.dirname(c_raster_final)): # Make Final_Raster folder in output directory
    os.makedirs(os.path.dirname(c_raster_final)) 


# Set Environmental Parameters
r = (arcpy.sa.Raster(mbounds)).extent
arcpy.env.extent = arcpy.Extent(r.XMin, r.YMin, r.XMax, r.YMax)
arcpy.env.snapRaster = mbounds

# Project data, if projected in desired coordinate system, set projected variable to raw variable
curr_raw_e_prj = arcpy.Project_management(curr_raw_e, curr_e_prj_fn, output_prj)
curr_raw_w_prj = arcpy.Project_management(curr_raw_w, curr_w_prj_fn, output_prj)


# Merge East Coast and West Coast Current Data
arcpy.Merge_management([curr_raw_e_prj, curr_raw_w_prj], curr_vp)

# Create TIN of merged current data
try:
    # Create Tin
    arcpy.CreateTin_3d(c_tin, output_prj, "'%s' %s Mass_Points <None>" % (str(curr_vp), tin_field), "Delaunay")
    # Delineate Tin Data Area
    arcpy.DelineateTinDataArea_3d(c_tin, max_edge, "ALL")
    # Export extents of delineated TIN
    arcpy.TinDomain_3d(c_tin, c_tin_bounds, "POLYGON")
    # Convert Tin to Raster
    arcpy.TinRaster_3d(c_tin, c_raster, "FLOAT", "LINEAR", "CELLSIZE 500", "1")
except arcpy.ExecuteError:
    print((arcpy.GetMessages()))
except Exception as err:
    print(err)

# Classify Currents Raster
# r = arcpy.sa.Reclassify(c_raster, "VALUE", "0 0.1 1;0.1 0.2 2;0.2 0.5 3;0.5 1 4;1 100 5;NODATA NODATA", "NODATA")
arcpy.gp.Reclassify_sa(c_raster, "VALUE", "0 0.1 1;0.1 0.2 2;0.2 0.5 3;0.5 1 4;1 100 5;NODATA NODATA", c_raster_rct1, "NODATA")

# r.save(c_raster_rct1)
# r2 = arcpy.sa.Reclassify(c_raster_rct1, "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", "DATA")
arcpy.gp.Reclassify_sa(c_raster_rct1, "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", c_raster_rc, "NODATA")
# r2.save(c_raster_rc)

# Clip Current data to Depth (Z<20 m) and Sandy Bounds
m = arcpy.sa.ExtractByMask(c_raster_rc, sandy_lt20m)
m.save(c_raster_final)

# Delete Unnecessary Files
l = [curr_raw_e_prj, curr_raw_w_prj, c_tin, c_raster_rct1]
for ll in l:
    arcpy.Delete_management(ll)

