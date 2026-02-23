# Proximity to Search and Rescue

def SAR_RawToVector(output_ws, db_name, raw_wkbk, raw_sheet, prj):
    import os, arcpy, xlrd
    from arcpy import env
    r'''This function creates a point feature dataset of Search and Rescue (SAR) locations from raw excel lat and lon data.

    NOTE: This script creates a file geodatabase named by the db_name variable in the output_ws location.
          This script will read in data as WGS84 geographic coordinate system.

    output_ws = string of output workspace with raw string annotation. e.g. r"H:\Christy\SAR"
    db_name = string of database name to be created. e.g. "SAR"
    raw_wkbk = string of excel workbook file location with raw string annotation. e.g. r"H:\Christy\SAR\SAR_RAW.xls"
    raw_sheet = string of excel workbook sheet name. e.g. "XY"
    prj = integer of projection keycode (http://pro.arcgis.com/en/pro-app/arcpy/classes/pdf/projected_coordinate_systems.pdf
    '''

    output_gdb = output_ws+"\\"+db_name+".gdb"
    raw_table = os.path.join(output_ws, db_name+".dbf")
    raw_table_lyr = os.path.join(output_ws, "SAR_Table_layer.lyr")
    raw_table_vector = os.path.join(output_gdb, "SAR_VECTOR")
    raw_table_vector_proc = os.path.join(output_gdb, "SAR_VECTOR_PROCESSED")
    try:
        arcpy.CreateFileGDB_management(output_ws, db_name+".gdb")
        arcpy.ExcelToTable_conversion(raw_wkbk, raw_table, raw_sheet)
        arcpy.MakeXYEventLayer_management(raw_table, "X", "Y", db_name, 4326)
        arcpy.SaveToLayerFile_management(db_name, raw_table_lyr)
        try:
            arcpy.CopyFeatures_management(raw_table_lyr, raw_table_vector)
        except:
            arcpy.CopyFeatures_management(raw_table_lyr+"x", raw_table_vector)
        arcpy.Project_management(raw_table_vector, raw_table_vector_proc, prj)
    except:
    # If conversion from raw to vector fails, print error messages
	    print("May be unable to create event layer, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")
        print(arcpy.GetMessages())


SAR_RawToVector(r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\SAR", "SAR", r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\SAR\SAR_RAW.xls", "XY", 102008)
