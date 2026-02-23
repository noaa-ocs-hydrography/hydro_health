# Density and Proximity to Navigational Hazards
# Navigational Hazards are defined as an OBSTRN, PILPNT, UWTROC, or WRECKS features that occur within CATZOC B, C, D, or U coverage and do not include BOOM or FISH HAVEN features.

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of navigational hazard. Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of Hazards w/i 2 nm          0            0                1               2-4                 4+
# Distance to Hazard                >2 nm       1 - 2 nm        0.5 - 1 nm      0.5 - 0.25 nm       < 0.25 nm

# This script formats the RAW data. This script extracts raw feature data from the ENC viewer database, merges like-geometries and clips data to best scale ENC usage band. Data are then projected to North America
# Albers Equal Area Conic projection and exported as RAW data to a user-specified database. 
    # The user has an sde connection file to the ENC Direct Viewer Database
	# The user has an sde connection file to the database where best usage band ENCs are stored 
	
# Notes:
	# The available features as of September 26, 2016 were used to generate this script, as described in the table below. Based on the data available at this time.


def extract_enc_data (enc_sde, ub_sde, output_gdb, features, geoms, output_fn, prj):
	r'''This function extracts user-specified features from the ENC Viewer database, merges all features of identical geometry, clips features to best scale ENC usage band, and then
	exports a merged feature layer for each geometry type.
	
	NOTE: This script assumes the data naming of the enc viewer database structure is as follows: ocsdev.ENCBAND.s57Attribute_geometry. e.g. ocsdev.approach.obstrn_point
		  This script assumes the best scale usage bands are named as follows: ENC_BestCoverage_UB1. If this is not the case adjust clip_out pathname. 
	
	enc_sde = string of file location for ENC Viewer Database Connection with read command. 
	ub_sde = string of file location for Best Scale Usage Band database connection with read command.  r"C:\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\postgresql_dev_encd_viewer.sde"
	output_gdb = string of output geodatabase location with read command. 
	features = list of strings of desired features as defined in ENC Viewer database. e.g. ['.obstrn', '.pilpnt', '.uwtroc', '.wrecks']
	geoms = list of strings of desired geometries as defined in ENC Viewer database. e.g. ['_point', '_line', '_polygon']
	output_fn = string of output filename root e.g. "KnownHaz" geometry and "RAW" will be appended
	prj = integer of projection keycode (http://pro.arcgis.com/en/pro-app/arcpy/classes/pdf/projected_coordinate_systems.pdf)
	'''
	
	import os, arcpy, arcgisscripting, collections

	OVERVIEW, GENERAL, COASTAL, APPROACH, HARBOR, BERTHING = ['overview', 'general', 'coastal', 'approach', 'harbor', 'berthing']
	bands = collections.OrderedDict([(OVERVIEW,'1'), (GENERAL,'2'), (COASTAL,'3'), (APPROACH,'4'), (HARBOR,'5'), (BERTHING,'6')])

	database_tables = {}
	merged_feature_layers = {}
	for geom in geoms: 
		merged_feature_layers[geom]=[]
	for band,uband in list(bands.items()):
		database_tables[band] = {}
		database_tables[band] = {}
		for geom in geoms:
			database_tables[band][geom]=[]
			for feat in features:
				geom_exist = enc_sde + '\\' + 'ocsdev.' + band + feat + geom
				if arcpy.Exists(geom_exist):
					database_tables[band][geom].append(geom_exist)
			if database_tables[band][geom]: 
				merge_out = os.path.join(output_gdb, band + geom + '_merge')
				arcpy.Merge_management(database_tables[band][geom], merge_out)
				band_geom_margename = os.path.join(output_gdb, band + geom + '_merge_'+ 'UB' + uband + 'Clip')
				clip_out = os.path.join(ub_sde, "ENC_BestCoverage_UB" + uband)
				arcpy.Clip_analysis(merge_out,clip_out, band_geom_margename)
				merged_feature_layers[geom].append(band_geom_margename)
			

	# Merge all bands together		
	for geom in geoms:
		arcpy.Merge_management(merged_feature_layers[geom],os.path.join(output_gdb, "All_UB" + geom + '_merge'))
		arcpy.Project_management(os.path.join(output_gdb, "All_UB" + geom + '_merge'), os.path.join(output_gdb, output_fn+geom+"_RAW"), prj)
	

extract_enc_data (r"C:\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\postgresql_dev_encd_viewer.sde", r"C:\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\NHSP2_0.sde", r"C:\Users\Christina.Fandel\Desktop\Delete\ENC_Testing.gdb", ['.obstrn', '.pilpnt', '.uwtroc', '.wrecks'], ['_point', '_line', '_polygon'], "KnownHaz", 102008)

	
