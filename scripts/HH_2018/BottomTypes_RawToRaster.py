# Bottom types

# Import Modules
import os
import arcpy
from arcpy import env

# Workspace
arcpy.env.workspace = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\Gordon\Data"

# Define Parameters
#@todo - add Raw, Vector, Vector Processed, Raster, Raster Classified, and Final default output geodatabases/folders
input_file_li = [
    "NCEI_Extract_lon_n180_180_lat_30_10.txt", "NCEI_Extract_lon_n180_180_lat_50_30.txt",
    "NCEI_Extract_lon_n180_180_lat_70_50.txt", "NCEI_Extract_lon_n180_180_lat_90_70.txt",
]
input_excel_bottom_type_template_file = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\Gordon\Data\Bottom_Type_Template_Final.xlsx"

# Temp gdb
temp_gdb = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\Gordon\Data\Temp\Temp.gdb"
# Auxiliary gdb
aux_gdb = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2018\Auxiliary\Auxiliary.gdb"
# Output gdb
out_gdb = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\Gordon\Data\BottomType.gdb"

cell_size = 500
des_sprj_code = 102008
input_excel_sheet_merge = "Merge"
in_field = "descrp4join"
join_field = "legend4join"
legend_field = "legend"
in_field_calc_express = "!descrp!.replace(' ', '').upper()"
join_field_calc_express = "!legend!.replace(' ', '').upper()"
f_descrp = "descrp"
f_class1 = "class1"
f_class2 = "class2"
f_class3 = "class3"
f_q_class1 = "QuantifiedClass1"
f_q_class2 = "QuantifiedClass2"
f_q_class3 = "QuantifiedClass3"
f_q_class1_calc_express = "getClass(!class1!.upper())"
f_q_class2_calc_express = "getClass(!class2!.upper())"
f_q_class3_calc_express = "getClass(!class3!.upper())"
remap_class3_2 = "2 2 2;5 5 NODATA"


sp_ref = r"Coordinate Systems\Geographic Coordinate Systems\North America\NAD 1983"

ref_cs_file = os.path.join(aux_gdb, "Shoreline_EEZ_Buffer707m")
c_bd = os.path.join(aux_gdb, "Shoreline_EEZ_Buffer707m")
mbounds = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\GRID\grid_ext_all"

# List
temp_c_li = [
    "Temp\NCEI_Extract_lon_n180_180_lat_30_10", "Temp\NCEI_Extract_lon_n180_180_lat_50_30",
    "Temp\NCEI_Extract_lon_n180_180_lat_70_50", "Temp\NCEI_Extract_lon_n180_180_lat_90_70",
]

# Output Parameters
c_merge = os.path.join(temp_gdb, "temp_BottomType_Merge_Raw")
c_merge_c_copy = os.path.join(temp_gdb, "BottomType_Merge_Classification_Copy")
t_merge = os.path.join(temp_gdb, "Merge")
t_merge_sort = os.path.join(temp_gdb, "Merge_Sort")
t_check_geo_result = os.path.join(temp_gdb, "Check_Geo_Result_Table")
c_merge_p = os.path.join(out_gdb, "BottomType_Merge_Raw")
c_merge_c = os.path.join(out_gdb, "BT_V")
c_merge_c_selected = os.path.join(out_gdb, "BottomType_Merge_Classification_Selected")
c_merge_c_thiessen = os.path.join(out_gdb, "BT_VP")
r_class1 = os.path.join(out_gdb, "BT_All_Ras")
r_class2 = os.path.join(out_gdb, "BT_MSR_Ras")
r_class3 = os.path.join(out_gdb, "BT_HS_Ras")
r_class1_extracted = os.path.join(out_gdb, "BT_All_F")
r_class2_extracted = os.path.join(out_gdb, "BT_MSR_F")
r_class3_extracted = os.path.join(out_gdb, "BT_HS_F")
r_class3_2_extracted_reclass = os.path.join(out_gdb, "BT_Sand")

# Classification Scheme from input_excel_bottom_type_template_file
f_q_class1_codeblock = """def getClass( class_1 ):
    global reVal
    if (class_1 == 'BOULDER'):
        reVal = 1
    elif (class_1 == 'CLAY'):
        reVal = 2
    elif (class_1 == 'COBBLES'):
        reVal = 3
    elif (class_1 == 'CORAL'):
        reVal = 4
    elif (class_1 == 'GRAVEL'):
        reVal = 5
    elif (class_1 == 'HARD'):
        reVal = 6
    elif (class_1 == 'MUD'):
        reVal = 7
    elif (class_1 == 'OOZE'):
        reVal = 8
    elif (class_1 == 'PEBBLES'):
        reVal = 9
    elif (class_1 == 'ROCKY'):
        reVal = 10
    elif (class_1 == 'SAND'):
        reVal = 11
    elif (class_1 == 'SHALE'):
        reVal = 12
    elif (class_1 == 'SHELL'):
        reVal = 13
    elif (class_1 == 'SILT'):
        reVal = 14
    elif (class_1 == 'STONE'):
        reVal = 15
    elif (class_1 == 'VOLCANIC'):
        reVal = 16
    elif (class_1 == 'CINDER'):
        reVal = 17
    elif (class_1 == 'UNKNOWN'):
        reVal = 0
    return reVal"""

f_q_class2_codeblock = """def getClass( class_2 ):
    global reVal
    if (class_2 == 'MUD'):
        reVal = 1
    elif (class_2 == 'SAND'):
        reVal = 2
    elif (class_2 == 'ROCK'):
        reVal = 5
    elif (class_2 == 'UNKNOWN'):
        reVal = 0
    return reVal"""

f_q_class3_codeblock = """def getClass( class_3 ):
    global reVal
    if (class_3 == 'HARD'):
        reVal = 5
    elif (class_3 == 'SOFT'):
        reVal = 2
    elif (class_3 == 'UNKNOWN'):
        reVal = 0
    return reVal"""

# Set Environmental Parameters
ext = (arcpy.sa.Raster(mbounds)).extent
arcpy.env.extent = arcpy.Extent(ext.XMin, ext.YMin, ext.XMax, ext.YMax)
arcpy.env.snapRaster = mbounds

try:
    # Create merged feature class
    x_coords = "lon"
    y_coords = "lat"
    # Set the spatial reference
    for idx, f in enumerate(input_file_li):
        in_Text = f
        out_Layer = temp_c_li[idx]
        print(in_Text)
        print(out_Layer)
        # Make the XY event layer...
        arcpy.MakeXYEventLayer_management(in_Text, x_coords, y_coords, out_Layer, sp_ref)
    # Merge feature classes
    arcpy.Merge_management(temp_c_li, c_merge)
    # Set output coordinate system
    desc = arcpy.Describe(ref_cs_file)
    outCS = desc.spatialReference
    # Run project tool
    arcpy.Project_management(c_merge, c_merge_p, outCS)

    # Copy feature class
    arcpy.CopyFeatures_management(c_merge_p, c_merge_c)

    # Load Excel sheet Classification and convert it into table
    arcpy.ExcelToTable_conversion(input_excel_bottom_type_template_file, t_merge, input_excel_sheet_merge)

    # VECTOR
    # Reclassification
    arcpy.Sort_management(t_merge, t_merge_sort, [[legend_field, "DESCENDING"]])
    if not arcpy.ListFields(c_merge_c, in_field):
        arcpy.AddField_management(c_merge_c, in_field, "TEXT", field_length=300)
    arcpy.CalculateField_management(c_merge_c, in_field, in_field_calc_express, "PYTHON_9.3")
    if not arcpy.ListFields(t_merge_sort, join_field):
        arcpy.AddField_management(t_merge_sort, join_field, "TEXT", field_length=300)
    arcpy.CalculateField_management(t_merge_sort, join_field, join_field_calc_express, "PYTHON_9.3")
    arcpy.JoinField_management(c_merge_c, in_field, t_merge_sort, join_field)

    # Quantify merged Bottom Type data
    # Quantify class1
    if not arcpy.ListFields(c_merge_c, f_q_class1):
        arcpy.AddField_management(c_merge_c, f_q_class1, "SHORT", field_length=10)
    arcpy.CalculateField_management(c_merge_c, f_q_class1, f_q_class1_calc_express, "PYTHON_9.3", f_q_class1_codeblock)
    # Quantify class2
    if not arcpy.ListFields(c_merge_c, f_q_class2):
        arcpy.AddField_management(c_merge_c, f_q_class2, "SHORT", field_length=10)
    arcpy.CalculateField_management(c_merge_c, f_q_class2, f_q_class2_calc_express, "PYTHON_9.3", f_q_class2_codeblock)
    # Quantify class3
    if not arcpy.ListFields(c_merge_c, f_q_class3):
        arcpy.AddField_management(c_merge_c, f_q_class3, "SHORT", field_length=10)
    arcpy.CalculateField_management(c_merge_c, f_q_class3, f_q_class3_calc_express, "PYTHON_9.3", f_q_class3_codeblock)

    # VECTOR PROCESSED
    # Create Thiessen Polygons
    # Copy feature class
    arcpy.CopyFeatures_management(c_merge_c, c_merge_c_copy)
    # Exclude NULL and UNKNOWN records
    where_clause = f_descrp + " IS NOT NULL AND UPPER(" + f_class3 + ") <> 'UNKNOWN'"
    arcpy.Select_analysis(c_merge_c_copy, c_merge_c_selected, where_clause)
    # Check geometry
    arcpy.CheckGeometry_management([c_merge_c_selected], t_check_geo_result)
    result = arcpy.GetCount_management(t_check_geo_result)
    count = int(result.getOutput(0))
    print(count)
    # Repair geometry if necessary
    if count > 0:
        arcpy.RepairGeometry_management(c_merge_c_selected, "DELETE_NULL")
        # Create Thiessen polygons
    arcpy.CreateThiessenPolygons_analysis(c_merge_c_selected, c_merge_c_thiessen, "ALL")

    # RASTER
    # # Class 1 rasterization
    # arcpy.FeatureToRaster_conversion(c_merge_c_thiessen, f_q_class1, r_class1, cell_size)
    # Class 2 rasterization
    arcpy.FeatureToRaster_conversion(c_merge_c_thiessen, f_q_class2, r_class2, cell_size)
    # Class 3 rasterization
    arcpy.FeatureToRaster_conversion(c_merge_c_thiessen, f_q_class3, r_class3, cell_size)

    # Raster FINAL
    # # Class 1 raster extraction
    # outExtractByMask = arcpy.sa.ExtractByMask(r_class1, c_bd)
    # outExtractByMask.save(r_class1_extracted)
    # Class 2 raster extraction
    outExtractByMask = arcpy.sa.ExtractByMask(r_class2, c_bd)
    outExtractByMask.save(r_class2_extracted)
    # Class 3 raster extraction
    outExtractByMask = arcpy.sa.ExtractByMask(r_class3, c_bd)
    outExtractByMask.save(r_class3_extracted)

    # Soft only for class3_2
    r3_2 = arcpy.sa.Reclassify(r_class3_extracted, "VALUE", remap_class3_2, "NODATA")
    r3_2.save(r_class3_2_extracted_reclass)

    # Delete
    for ll in temp_c_li:
        print(ll)
        if arcpy.Exists(ll):
            arcpy.Delete_management(ll)

except Exception as err:
    print((err.args[0]))
