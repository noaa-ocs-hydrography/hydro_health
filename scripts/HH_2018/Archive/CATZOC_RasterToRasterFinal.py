# CATZOC
# Data Source
    # 1. MCD CATZOC Layer, compiled by Sean Legeer. (authoritative) and represents data applied to the chart
            # - This layer includes three sources
                    # 1. ENC
                    # 2. Source (HCELL --> Source), awaiting ENC application
                    # 3. 2013 Superseeded CATZOC layer.
                            # This layer does not include F, W, or D surveys and used the following classification
                                # Survey Type                                                           # CATZOC
                                # Complete MBES                                                         # A1
                                # 200% SSS with concurrent MBES                                         # A1
                                # 100 % SSS with complete MBES                                          # A1
                                # SSS (no MBES)                                                         # A2
                                # Bathy Lidar                                                           # B
                                # SBES Nearshore (1:40k or larger) since 1940                           # B
                                # SBES Offshore since 1940                                              # C
                                # 1920 - 1940 surveys on known horizontal and vertical datums           # C
                                # Pre 1940 surveys with unknown horizontal and vertical datums          # D
                                # All surveys pre-1920                                                  # D
    # 2. HSD OPS CATZOC Assessment
            # - This layer includes two sources and is only applied in those areas where the MCD CATZOC layer has no data
                    # HCELL
                    # TECSOU/Chart Scale/Year Algorithm * Only applied to H and F surveys, W surveys, if not captured in MCD CATZOC layer or HSD OPS HCell, then left as unassessed
                                # OD/CC	                                                                # 1
                                # year<1920	                                                            # 5
                                # TECSOU contains 1 and 3 with year > 2000	                            # 1
                                # TECSOU contains 1 and either 2 or (3 w/ year<2000)	                # 2
                                # year < 1940 or TECSOU contains 1 with scale >40000 or blank	        # 4
                                # TECSOU contains 7 or TECSOU contains 1 with scale <=40000	            # 3
                                # TECSOU contains 3	                                                    # 1
                                # Everything else	                                                    # 4

                                import arcpy, os

# Input Variables
c_MCD = r'H:\HH_2018\Working\SurveyQuality\CATZOC.gdb\Legeer_CATZOC'
eez_v = r'H:\HH_2018\Auxiliary\Auxiliary.gdb\Shoreline_EEZ_Buffer707m'

# Output Variables

# Vector Data
# Identify area within EEZ where MCD CATZOC classification does not exist

# Dissolve MCD CATZOC Layer
dis_field = "dis"
arcpy.AddField_management(c_MCD, dis_field, "DOUBLE")
arcpy.CalculateField_management(c_MCD, dis_field, 1, "PYTHON_9.3")
c_MCD_dis = c_MCD+"_dis" # Delete
arcpy.Dissolve_management(c_MCD, c_MCD_dis, dis_field)

# Erase MCD CATZOC Layer from EEZ


# Delete
c_MCD_dis
'''
# This script updates unassessed CATZOC data using the following age ranges:
# < 1920 = CATZOC Undefined
# 1920 - 1940 = CATZOC D
# 1940 - 1970 = CATZOC C
# 1970 - 2000 = CATZOC B
# > 2000 = CATZOC A2
# Input
c =  r'H:\NHSP_2_0\Christy\CATZOC\CATZOC.gdb\CATZOC_JJ_COREY_copy'
c_field = "catzoc" # Catzoc Field in input CATZOC data
bounds = r"H:\NHSP_2_0\GRID\Grid.gdb\Grid_Model_Bounds"
output_gdb = r"H:\NHSP_2_0\Christy\CATZOC\CATZOC.gdb"
out_fldr = r"H:\NHSP_2_0\Christy\CATZOC"
s_year = r'H:\Hydro_Health\2016\Raster_Auxiliary\s_year_r'
cellsize = 500
output_prj = 102008 # USER DEFINED: integer of projection keycode (http://pro.arcgis.com/en/pro-app/arcpy/classes/pdf/projected_coordinate_systems.pdf)


# Output
c_dis_r = os.path.join(out_fldr, "CATZOC_Dis_R") #Delete
c_dis_p = os.path.join(output_gdb, "CATZOC_Dis_P") #Delete
c_unassessed_bounds_p = os.path.join(output_gdb, "CATZOC_Unassessed_Bounds_P") #Delete
s_year_unassessed = os.path.join(out_fldr, "s_yr_nocatzoc") # Delete
c_unassessed_updated = os.path.join(out_fldr, "catzoc_upd") #Delete
c_r_old = os.path.join(out_fldr, "CATZOC_R_old") #Delete
catzoc_r = "CATZOC_R"
catzoc_final = os.path.join(out_fldr, "catzoc_final")
iss = os.path.join(out_fldr, "Initial_SS")
catzoc_BCDU = os.path.join(out_fldr, "catzoc_bcdu")

# Repair Geometry of input CATZOC layer
#arcpy.RepairGeometry_management(c)

# Dissolve catzoc file to identify bounds of unassessed catzoc area
field = "dis"
arcpy.AddField_management(c, field, "DOUBLE")
arcpy.CalculateField_management(c, field, 1, "PYTHON_9.3")
arcpy.PolygonToRaster_conversion(c, field, c_dis_r,"CELL_CENTER", field, cellsize)
arcpy.RasterToPolygon_conversion (c_dis_r, c_dis_p, "NO_SIMPLIFY", "VALUE")
arcpy.RepairGeometry_management(c_dis_p)

# Erase CATZOC bounds from Model Bounds to identify unassessed CATZOC Area
arcpy.Erase_analysis(bounds, c_dis_p, c_unassessed_bounds_p)
arcpy.RepairGeometry_management(c_unassessed_bounds_p)

# Extract Survey Year Data for areas of Unassessed CATZOC
m = arcpy.sa.ExtractByMask(s_year, c_unassessed_bounds_p)
m.save(s_year_unassessed)

# Reclassify Unassessed CATZOC Area by Survey Year
r = arcpy.sa.Reclassify(s_year_unassessed, "VALUE", "0 1900 6;1901 1919 5;1920 1969 4;1970 1999 3;2000 2016 2", "NODATA")
r.save(c_unassessed_updated)

# Mosaic Old CATZOC and Updated CATZOC to new raster using the minimum catzoc (best catzoc value)
arcpy.PolygonToRaster_conversion(c, c_field, c_r_old, "CELL_CENTER", c_field, cellsize)
arcpy.MosaicToNewRaster_management([c_r_old, c_unassessed_updated], out_fldr, catzoc_r, output_prj, "32_BIT_SIGNED", str(cellsize), "1", "MINIMUM", "FIRST")

# Reclassify CATZOC for CATZOC Likelihood and Initial Survey Score
r_like = arcpy.sa.Reclassify(os.path.join(out_fldr, catzoc_r), "VALUE", "5 6 5;4 4 4;3 3 3;1 2 1", "NODATA")
r_like.save(catzoc_final)
r_iss = arcpy.sa.Reclassify(os.path.join(out_fldr, catzoc_r), "VALUE", "5 6 0;4 4 30;3 3 80;1 2 100", "NODATA")
r_iss.save(iss)
r_bcdu = arcpy.sa.Reclassify(os.path.join(out_fldr, catzoc_r), "VALUE", "3 6 1;1 2 NODATA", "NODATA")
r_bcdu.save(catzoc_BCDU)

# Delete Unnecessary Files
del_fns = [c_dis_r, c_dis_p,c_unassessed_bounds_p, s_year_unassessed, c_unassessed_updated, c_r_old]
for fn in del_fns:
    arcpy.Delete_management(fn)
print("Finished Delete at %.1f secs"%(time.time()-t_start))
'''


