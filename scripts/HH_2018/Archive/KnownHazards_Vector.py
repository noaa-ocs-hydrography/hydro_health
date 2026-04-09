# Density and Proximity to Navigational Hazards
# Navigational Hazards are defined as an OBSTRN, PILPNT, UWTROC, or WRECKS features that occur within CATZOC B, C, D, or U coverage and do not include BOOM or FISH HAVEN features.

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of navigational hazard. Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of Hazards w/i 2 nm          0            0                1               2-4                 4+
# Distance to Hazard                >2 nm       1 - 2 nm        0.5 - 1 nm      0.5 - 0.25 nm       < 0.25 nm


# This script formats the VECTOR data. This script restrics the RAW Known Hazards data to not include fish haven or boom features and clips the data by the CATZOC B bounds. 
# NOTE: KnownHazards_Raw.py must be run before completing this script. The output from KnownHazards_Raw.py (KnownHaz_point_RAW, KnownHaz_line_RAW, and KnownHaz_polygon_RAW) are inputs to this script. 

import arcpy
# Define Variables
output_gdb = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\NAV_HAZARDS\NavHazards.gdb"
geoms = ['_point', '_line', '_polygon']

# Define Merged Usage Band Geometry Features 
raw_pt = os.path.join(output_ws, 'KnownHaz' + geoms [0] + '_RAW') #MUST CHANGE TO NHSP
raw_ln = os.path.join(output_ws, 'KnownHaz' + geoms [1] + '_RAW') #MUST CHANGE TO NHSP
raw_pg = os.path.join(output_ws, 'KnownHaz' + geoms [2] + '_RAW') #MUST CHANGE TO NHSP

vec_pt = os.path.join(output_ws, 'KnownHaz' + geoms [0] + '_VECTOR') #MUST CHANGE TO NHSP
vec_ln = os.path.join(output_ws, 'KnownHaz' + geoms [1] + '_VECTOR') #MUST CHANGE TO NHSP
vec_pg = os.path.join(output_ws, 'KnownHaz' + geoms [2] + '_VECTOR') #MUST CHANGE TO NHSP

catzoc_b = os.path.join(output_ws, 'Test') # GET PARAMETERS
eez = os.path.join(output_ws, 'EEZ_Polygon') # GET PARAMETERS

# 4. Remove boom and fish haven features
for fl, fl_out in [raw_pt, '%s'%(vec_pt)], [raw_ln, '%s'%(vec_ln)], [raw_pg, '%s'%(vec_pg)]:
	temp = '%s'%(fl)+'_temp'
	arcpy.MakeFeatureLayer_management('%s'%(fl), temp)
	arcpy.SelectLayerByAttribute_management(temp, "NEW_SELECTION", "catobs IS NULL OR catobs = '                         ' OR catobs <> '8                        ' AND catobs <> '10                       ' AND catobs <> '5                        '")
		#.ljust
	arcpy.Clip_analysis(temp, catzoc_b, fl_out+'_temp')
	arcpy.Clip_analysis(fl_out+'_temp', eez, fl_out)
	arcpy.SelectLayerByAttribute_management ('%s'%(temp), "CLEAR_SELECTION")

    
    
