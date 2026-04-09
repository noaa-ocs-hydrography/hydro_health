# Groundings

def Make_Selection(gdb, input_filename, query, output_filename):
	'''Make Selection will query an input layer within a user-defined geodatabase and then export the queried layer to the same geodatabase
	gdb = string of the geodatabase where the input layer is stored as well as where the queried layer will be exported
	input_filename = string of the name of the input layer
	query = query command, e.g. "gs <> 'Fishing Vessel'"
	output_filename = string of the name of the exported layer
	'''
	import arcpy, os, arcgisscripting
	try:
		input = os.path.join(gdb, input_filename)
		output = os.path.join(gdb, output_filename)
		input_temp = "temp"
	
		arcpy.MakeFeatureLayer_management(input, input_temp)
		arcpy.SelectLayerByAttribute_management(input_temp, "NEW_SELECTION", query)
		arcpy.CopyFeatures_management(input_temp, output)
		arcpy.SelectLayerByAttribute_management(input_temp, "CLEAR_SELECTION")
	except arcgisscripting.ExecuteError:
		print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.") 
	
Make_Selection("H:\Christy\Groundings\Groundings.gdb", "Groundings_VECTOR", "gs <> 'Fishing Vessel' AND gs <> 'Recreational'  AND status = 'Grounding'", "Groundings_VECTOR_PROCESSED2")

