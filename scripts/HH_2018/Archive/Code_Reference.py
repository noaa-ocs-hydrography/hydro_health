# Import new module
# Navigate to C:\Program Files\ArcGIS\Pro\bin\Python\Scripts
# Install previous version of pip
C:\Program Files\ArcGIS\Pro\bin\Python\Scripts>easy_install pip==7.1.2
# Install module to target location where openpyxl is the module (ignore-installed is a toggle that ignores any files within the module install that already exist within the target location)
C:\Program Files\ArcGIS\Pro\bin\Python\Scripts>..\python pip-script.py install --target="C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\Lib\\site-packages" --ignore-installed openpyxl
# If do not need to ignore-install, then...
C:\Program Files\ArcGIS\Pro\bin\Python\Scripts>..\python pip-script.py install --target="C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\Lib\\site-packages" openpyxl
# Or maybe try this...
"C:\Program Files\ArcGIS\Pro\bin\Python\Scripts\propy.bat" pip-script.py install openpyxl



importlib.reload(gridclass)
<module 'HSTB.ArcExt.NHSP.gridclass' from 'C:\\Program Files\\ArcGIS\\Pro\\bin\\Python\\envs\\arcgispro-py3\\lib\\site-packages\\ArcExt\\NHSP\\gridclass.py'>
d=gridclass.datagrid(input, 500, 7500, numpy.uint32)
d.MakeVectorGrid(r"H:\Christy\Grid\Grid_Testing.gdb\PR_Polygon_100mGen", r"H:\Christy\Grid\Grid_Testing.gdb", "PR_Grid_Clip_100mGen")
Finished Process at 596.3 secs

from HSTB.ArcExt.NHSP import gridclass

# OLD CODE
# Get array shape
d.gridarray.shape
# ############### #
def Func1():
    # 1. Merge Features
    # Point Features
    for band in bands.keys():
        #for feat in features:
            #Merge(enc_view, band, feat, POINT)
        pt = [enc_view + '\\' + 'ocsdev.' + band + feat + geoms[0] for feat in features]
        arcpy.Merge_management(pt, os.path.join(output_ws, band + geoms [0] + '_merge'))
    # Line Features
    for band in bands.keys():
        #if band not in (GENERAL, HARBOR):
        try:
            ln = [enc_view + '\\' + 'ocsdev.' + band + features[0] + geoms[1]]
            arcpy.Merge_management(ln, os.path.join(output_ws, band + geoms [1] + '_merge'))
        except arcgisscripting.ExecuteError:
            continue

    # Polygon Features
    for band in bands.keys():
        if band == OVERVIEW:
            pg = [enc_view + '\\' + 'ocsdev.' + band + features[0] + geoms [2]]
            arcpy.Merge_management(pg, os.path.join(output_ws, band + geoms[2] + '_merge'))
        else:
            pg = [enc_view + '\\' + 'ocsdev.' + band + features[0] + geoms[2], enc_view + '\\' + 'ocsdev.' + band + features[3] + geoms[2]]
            arcpy.Merge_management(pg, os.path.join(output_ws, band + geoms[2] + '_merge'))
# MUST RESUME TESTING ONCE NHSP DATABASE IS RUNNING
# PROJECT PROJECT PROJECT
def Func2():
    # 2. Clip Features by Usage Band
    for band, uband in bands.items():
        if band == 'general' or band == 'harbor':
            # Clip Point Features
            arcpy.Clip_analysis(os.path.join(output_ws, band + geoms [0] + '_merge'), os.path.join(nhsp2, "ENC_BestCoverage_UB" + '%s'%(uband)), os.path.join(output_ws, band + geoms [0] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')
            # Clip Polygon Features
            arcpy.Clip_analysis(os.path.join(output_ws, band + geoms [2] + '_merge'), os.path.join(nhsp2, "ENC_BestCoverage_UB" + '%s'%(uband)), os.path.join(output_ws, band + geoms [2] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')
        else:
        # Clip Point Features
        arcpy.Clip_analysis(os.path.join(output_ws, band + geoms [0] + '_merge'), os.path.join(nhsp2, "ENC_BestCoverage_UB" + '%s'%(uband)), os.path.join(output_ws, band + geoms [0] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')
        # Clip Line Features
        arcpy.Clip_analysis(os.path.join(output_ws, band + geoms [1] + '_merge'), os.path.join(nhsp2, "ENC_BestCoverage_UB" + '%s'%(uband)), os.path.join(output_ws, band + geoms [1] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')
        # Clip Polygon Features
        arcpy.Clip_analysis(os.path.join(output_ws, band + geoms [2] + '_merge'), os.path.join(nhsp2, "ENC_BestCoverage_UB" + '%s'%(uband)), os.path.join(output_ws, band + geoms [2] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')

def Func3():
    # 3. Merge all point, line, and polygon features and export to database as RAW # MUST RERUN TO VERIFY NEW NAMING CONVENTION KnownHaz_points_RAW) AND EXPORT TO DATABSE IS WORKING CORRECTLY
    # Point and Polygon Features
    pt_merge = []
    pg_merge = []
    for band, uband in bands.items():
        pt_merge += [os.path.join(output_ws, band + geoms [0] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')]
        pg_merge += [os.path.join(output_ws, band + geoms [2] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')]
    arcpy.Merge_management(pt_merge, ub_pt)
    arcpy.Merge_management(pg_merge, ub_pg)

    # Line Features
    ln_merge = []
    for band in ['general', 'coastal', 'harbor']:
        uband = bands[band]
        ln_merge += [os.path.join(output_ws, band + geoms [1] + '_merge_'+ 'UB' + '%s'%(uband) + 'Clip')]
    arcpy.Merge_management(ln_merge, ub_ln)