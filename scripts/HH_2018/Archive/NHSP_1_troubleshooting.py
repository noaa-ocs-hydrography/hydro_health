# NHSP Update 
# Notes #
    # 
# This script completes the following:
    # Projects the user-sepecified NHSP and SURDEX layers into WGS84 EASE
    # Adds prev_priority (string) and classify_yr (string) fields to NHSP layer, if not already present
    # Updates NHSP layer over user-specified time range. 
# Final Output is sdx_null_final named SURDEX_Superseded_Raster in the specified output geodatabase

import arcpy, os, numpy
#bDebug = True
import time
n = arcpy.GetParameter(0)
s = arcpy.GetParameter(1)
yr_min = arcpy.GetParameter(2)
yr_max = arcpy.GetParameter(3)
output_ws = str(arcpy.GetParameter(4))
temp = "n_prj_priority"
n_prj = os.path.join(output_ws, 'NHSP_WGS84EASE')
s_prj = os.path.join(output_ws, 'SURDEX_WGS84EASE')
n_rs_ns = os.path.join(output_ws, 'NHSP_RS_NS')
s_temp = "s_prj_yr"
n_priority_temp = "NHSP_Priority_RegionYr_Temp"
n_merge = os.path.join(output_ws, "NHSP_Priority_Updated_Merge")
n_update = os.path.join(output_ws, str(arcpy.GetParameter(5)))
regions = ['AK', 'EC', 'GL', 'GM', 'PI', 'PR', 'WC']

# Define range of SURDEX years to search and priority variable as a function of survey year 
yr_range = sorted(numpy.linspace(yr_min, yr_max, (yr_max - yr_min)+1).tolist())
n_priority = os.path.join(output_ws, "NHSP_Priority_%d"%(yr_range[0]-1))

t_start=time.time()
#if not bDebug: #for debugging the files already exist, for a real user create the files.
    # Project NHSP and SURDEX to WGS84 EASE and Repair Geometry
#if bDebug:
#    t=time.time()
#    arcpy.AddMessage("Starting %s repair took %.1f secs"%(n_prj, time.time()-t))
#    arcpy.Project_management('%s'%(n), '%s'%(n_prj), arcpy.SpatialReference(3975))
#    arcpy.AddMessage("%s projection took %.1f secs"%(n_prj, time.time()-t))
#    arcpy.RepairGeometry_management('%s'%(n_prj))
#    arcpy.AddMessage("%s repair took %.1f secs"%(n_prj, time.time()-t))
#else:    
for unprj, prj in ([n,n_prj], [s, s_prj]):
    t=time.time()
    arcpy.AddMessage("Starting %s repair took %.1f secs"%(n_prj, time.time()-t))
    arcpy.Project_management('%s'%(unprj), '%s'%(prj), arcpy.SpatialReference(3975))
    arcpy.AddMessage("%s projection took %.1f secs"%(n_prj, time.time()-t))
    arcpy.RepairGeometry_management('%s'%(prj))
    arcpy.AddMessage("%s repair took %.1f secs"%(n_prj, time.time()-t))
        
arcpy.AddMessage("Finished Repair at %.1f secs"%(time.time()-t_start))
        
    
# Add Prev_Priority and Classify_Yr Fields and populate with Priority and 1994, respectively (if necessary)
to_add = ["prev_priority", "classify_yr"]
fieldList = arcpy.ListFields(n_prj)
fieldName = [f.name for f in fieldList]

for field in to_add:
    if field in fieldName:
        pass
    else:
        arcpy.AddField_management(n_prj, field, "TEXT")
        if field == to_add[0]:
            arcpy.CalculateField_management(n_prj, to_add[0], "!priority!", "PYTHON_9.3")
            arcpy.AddMessage("I AM CALCULATING FIELD AND I SHOULDN'T BE!!")
        elif field == to_add[1]:
            arcpy.CalculateField_management(n_prj, to_add[1], "2009")
            arcpy.AddMessage("I AM CALCULATING FIELD AND I SHOULDNT BE!!")

arcpy.AddMessage("Finished CalulateFields at %.1f secs"%(time.time()-t_start))

# Generate and Save two separate NHSP layers. Layer 1 = P1 - P5, C, EC, and FBC and Layer 2 = NS and RS. (cond = ['1', '2', '3', 'EC']    " OR ".join(["priority = '%s'"%c for c in cond]))
p_rule = [" Priority = '1' OR Priority = '2' OR Priority = '3' OR Priority = '4' OR Priority = '5' OR Priority = 'C' OR Priority = 'EC' OR Priority = 'FBC' OR Priority = 'U'", " Priority = 'NS' OR Priority = 'RS'"]
arcpy.MakeFeatureLayer_management('%s'%(n_prj), temp)

for p_rules, f_out in [p_rule[0], n_priority], [p_rule[1], n_rs_ns]:
    arcpy.SelectLayerByAttribute_management('%s'%(temp), "NEW_SELECTION", p_rules)
    arcpy.CopyFeatures_management('%s'%(temp), f_out)
    arcpy.SelectLayerByAttribute_management ('%s'%(temp), "CLEAR_SELECTION")

arcpy.AddMessage("Finished Dividing up NHSP RNS and P at %.1f secs"%(time.time()-t_start))

# Supersession
# 1. Generate array with user-specified min to max years, separated by one  year
arcpy.MakeFeatureLayer_management('%s'%(s_prj), s_temp)
fields = ["Priority", to_add[0], to_add[1]]

for yr in yr_range:
    arcpy.MakeFeatureLayer_management(os.path.join(output_ws, "NHSP_Priority_%d"%(yr-1)), n_priority_temp) 
    arcpy.SelectLayerByAttribute_management('%s'%(s_temp), "NEW_SELECTION", "year" + '=' + '%s'%("'") + '%s'%(int(yr)) + '%s'%("'") + ' AND ("survey" LIKE' + " 'H%' " + 'OR "survey" LIKE' + " 'F%') ")
    #t=time.time()00    
    for region in regions:
        arcpy.SelectLayerByAttribute_management('%s'%(n_priority_temp), "NEW_SELECTION", "region" + '=' + '%s'%("'") + '%s'%(region) + '%s'%("'")) #If selection present, then do the rest...
        #desc = arcpy.Describe("NHSP_Priority_RegionYr_Temp")
        #val = desc.FIDSet
        #if val > 0:
            arcpy.Clip_analysis(n_priority_temp, '%s'%(s_temp), os.path.join(output_ws, "NHSP_Clip_%s_%d"%(region, yr)))
            arcpy.SelectLayerByAttribute_management('%s'%(n_priority_temp), "CLEAR_SELECTION")
            arcpy.Erase_analysis(n_priority_temp, os.path.join(output_ws, "NHSP_Clip_%s_%d"%(region, yr)), os.path.join(output_ws, "NHSP_Minus_%s_%d"%(region, yr)))
            arcpy.SelectLayerByAttribute_management('%s'%(s_temp), "CLEAR_SELECTION")
            #arcpy.RepairGeometry_management(os.path.join(output_ws, "NHSP_Clip_%d"%(yr)))
            with arcpy.da.UpdateCursor(os.path.join(output_ws, "NHSP_Clip_%s_%d"%(region, yr)), fields) as cursor:
                for row in cursor:
                    if float(yr) > float(row[2]):
                        row[2] = str(int(yr))
                        row[1] = row[0]
                        row[0] = "FBC"
                        cursor.updateRow(row)
            arcpy.Merge_management([os.path.join(output_ws, "NHSP_Minus_%s_%d"%(region, yr)), os.path.join(output_ws, "NHSP_Clip_%s_%d"%(region, yr))], os.path.join(output_ws, "NHSP_Priority_%s_%d"%(region,yr)))

# If selection was present, then do merge for those layers...
arcpy.Merge_management([os.path.join(output_ws, "NHSP_Priority_AK_%d"%(yr)), os.path.join(output_ws, "NHSP_Priority_EC_%d"%(yr)), os.path.join(output_ws, "NHSP_Priority_GL_%d"%(yr)), os.path.join(output_ws, "NHSP_Priority_GM_%d"%(yr)), os.path.join(output_ws, "NHSP_Priority_PI_%d"%(yr)), os.path.join(output_ws, "NHSP_Priority_PI_%d"%(yr)), os.path.join(output_ws, "NHSP_Priority_PR_%d"%(yr)), os.path.join(output_ws, "NHSP_Priority_WC_%d"%(yr))], os.path.join(output_ws, "NHSP_Priority_%d"%(yr)))
arcpy.RepairGeometry_management(os.path.join(output_ws, "NHSP_Priority_%d"%(yr)))
#" , ".join(["NHSP_Priority_%s_%d"%(r, yr) for r in regions])

# Merge final clipped NHSP layer (Priorities 1-5, C, EC, and FBC) with Resurvey/Nav-Sig Layer
arcpy.Merge_management([os.path.join(output_ws, "NHSP_Priority_%d"%(yr_max)), n_rs_ns],  n_merge)
arcpy.Dissolve_management(n_merge, n_update, ["Priority", "Region", to_add[0], to_add[1]]) 

# Add AREA_KM2 and AREA_SNM fields and populate with area metrics (if necessary)
area_add = ["AREA_KM2", "AREA_SNM"]
areaList = arcpy.ListFields(n_update)
areaName = [f.name for f in areaList]

for field in area_add:
    if field in areaName:
        pass 
    elif field.lower() in areaName:
        pass
    else:
        arcpy.AddField_management(n_update, field, "FLOAT")

arcpy.CalculateField_management(n_update, area_add[0], "!SHAPE.AREA@SQUAREKILOMETERS!", "PYTHON_9.3")
arcpy.CalculateField_management(n_update, area_add[1], '%s'%('!') + '%s'%(area_add[0]) + '%s'%('!') + '* 0.29155335', "PYTHON_9.3")

# Delete unnecessary files
#arcpy.Delete_management(n_prj)
#arcpy.Delete_management(s_prj)
#arcpy.Delete_management(n_rs_ns)
#arcpy.Delete_management(n_merge)

#delfiles = sorted(numpy.linspace(yr_min-1, yr_max, (yr_max - (yr_min-1))+1).tolist())
#for delfile in delfiles:
#    arcpy.Delete_management(os.path.join(output_ws, "NHSP_Clip_%d"%(delfile)))
#    arcpy.Delete_management(os.path.join(output_ws, "NHSP_Minus_%d"%(delfile)))
#    arcpy.Delete_management(os.path.join(output_ws, "NHSP_Priority_%d"%(delfile)))

