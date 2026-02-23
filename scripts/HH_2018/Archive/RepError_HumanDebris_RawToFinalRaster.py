# Reported Errors, Human Debris Changeability, and Human Debris Forecast .

# A REP or PA Feature is defined as an OBSTRN, PILPNT, or WRECKS feature with QUAPOS = 7 or 8 or 4
# A ED or PD Features is defined as an OBSTRN, PILPNT, or WRECKS feature with QUAPOS = 5 or STATUS = 18 and NULL
# Known Features are defined as an OBSTRN, PILPNT, or WRECKS feature with QUAPOS != 4, 5, 7, 8 and STATUS <> 18

# Human Debris: Sum of Known Hazards (within 2 nm) plus 3*PA/PD/Rep/ED (within 2 nm).
# Human Debris Changeability: Sum of "Known" (i.e. non PA/PD, Rep) OBSTRN PLUS 3(PA/PD OBSTRN) within 2 nm
# Human Debris Forecast: Sum of "Known" (i.e. non PA/PD, Rep) OBSTRN PLUS 3(PA/PD OBSTRN) within 2 nm (same as above)

# Human Debris Changeability and Forecast
# Objective: Classify risk within each grid cell based on density (within set distance) of PA/PD and/or Rep. Classify risk using the following table

# Risk Category                                          1            2                3                4                  5
# Density of Known Features + 3(PA/PD) within 2 nm       0            1              2 - 5            5 - 10              10 +

# Define Variables
import os
import arcpy
import time
import arcgisscripting
import numpy
from HSTB.ArcExt.NHSP import gridclass, Functions

enc_sde_1 = r"C:\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\postgresql_dev_encd_viewer.sde"
ub_sde_1 = r"C:\Users\Christina.Fandel\AppData\Roaming\ESRI\Desktop10.4\ArcCatalog\NHSP2_0.sde"
output_gdb_1 = r'H:\HH_2018\Working\HumanDebris\HumanDebris.gdb'
grid_ws = r'H:\GRID\Grid.gdb'  # USER DEFINED
out_fldr = r'H:\HH_2018\Working\HumanDebris'
features_1 = ['.obstrn', '.pilpnt', '.wrecks']  # UPDATE ## Is this correct? Check Mike's notes
geoms_1 = ['_point']
output_fn_1 = "RE_HD_RAW"
prj_1 = 102008
cell_size = 500  # USER DEFINED: integer  # NOTE: Must adapt script to adjust buffer (e.g. circle9) as a function of grid cell size!

papd_vp_fn = "RE_HD_PAPD_VP"
rep_vp_fn = "RE_Rep_VP"
kf_vp_fn = "HD_KnownFeat_VP"

# Final Outputs
'''papd_den_r_fn = "RE_PAPD_R" # PAPD Density
rep_den_r_fn = "RE_Rep_R" # Reported Density
papd_dis_c_fn = "RE_PAPD_C" # PAPD Distance Classified
rep_dis_c_fn = "RE_Rep_C" # Reported Distance Classified
papd_rep_sum_c_fn = "RE_PRSum_C" # Sum PAPD and Reported Classified
re_out_fn = "RE_Final" # Reported Error Final'''
papd_den_hd_r_fn = "HD_PAPD_R"  # PAPD Density (same as above with human debris prefix)
kf_den_r_fn = "HD_KnF_R"  # Known Features Density
hd_out_fn = "HD_Final"  # Human Debris Final

# Extract Raw Data: OBSTRN, PILPNT, and WRECKS Features
re_hd_raw = Functions.extract_enc_data(enc_sde_1, ub_sde_1, output_gdb_1, features_1, geoms_1, output_fn_1, prj_1)
# re_hd_raw = 'C:\\Users\\Christina.Fandel\\Documents\\Fandel\\NHSP\\HUMAN_DEBRIS\\HUMAN_DEBRIS.gdb\\RE_HD_point_RAW'

# Compute Vector Processed Data: Extract Known Features, PA/PD Features and Rep Features
# Add AND CATOBS = NULL or '   ' for PA/PD features

# UPDATE ### To include status and update code so Reported and PA go together and PD and ED go together

# papd_query = "quapos = '4  ' OR quapos = '5  '"
papd_query = "quapos = '4  ' OR quapos = '5  ' AND catobs IS NULL OR catobs = '                         ' OR catobs <> '1                        '"
# rep_query = "quapos = '7  ' OR quapos = '8  '"
rep_query = "quapos = '7  ' OR quapos = '8  ' AND catobs IS NULL OR catobs = '                         ' OR catobs <> '1                        '"
# kf_query = "quapos <> '7  ' AND quapos <> '8  ' AND quapos <> '4  ' AND quapos <> '5  ' OR quapos IS NULL" #Known Features
kf_query = "quapos <> '7  ' AND quapos <> '8  ' AND quapos <> '4  ' AND quapos <> '5  ' OR quapos IS NULL OR catobs = '                         '"  # Known Features  ## UPDATE ## Do not include CATOBS <> 1

papd_vp = Functions.Make_Selection(os.path.split(re_hd_raw)[0], os.path.split(re_hd_raw)[1], papd_query, papd_vp_fn)
rep_vp = Functions.Make_Selection(os.path.split(re_hd_raw)[0], os.path.split(re_hd_raw)[1], rep_query, rep_vp_fn)
kf_vp = Functions.Make_Selection(os.path.split(re_hd_raw)[0], os.path.split(re_hd_raw)[1], kf_query, kf_vp_fn)

# Lists to hold filenames of variable grid paths to later be deleted
# Reported Errors Exports
papd_den_r_fns = []
rep_den_r_fns = []
papd_dis_c_fns = []
rep_dis_c_fns = []
papd_rep_sum_c_fns = []
re_out_fns = []
kf_den_r_fns = []
hd_out_fns = []
del_fns = []

# Temp: Read in PAPD, Reported, and Known Features
papd_vp = r'C:\\Users\\Christina.Fandel\\Documents\\Fandel\\NHSP\\HUMAN_DEBRIS\\HUMAN_DEBRIS.gdb\\RE_HD_PAPD_VP'
rep_vp = r'C:\Users\Christina.Fandel\Documents\Fandel\NHSP\HUMAN_DEBRIS\HUMAN_DEBRIS.gdb\RE_Rep_VP'
kf_vp = r'C:\Users\Christina.Fandel\Documents\Fandel\NHSP\HUMAN_DEBRIS\HUMAN_DEBRIS.gdb\HD_KnownFeat_VP'

for igrid in range(1, 14):  # Improve: Find all matching grid extents in the geodatabase
    t_start = time.time()
    grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
    grid_path = os.path.join(grid_ws, grid_name)

    # Reported Errors Exports
    # Update with templayer code
    papd_den_r = os.path.join(out_fldr, "re_p_%02d_den" % igrid)  # gridclass.Templayer(out_fldr)
    rep_den_r = os.path.join(out_fldr, "re_r_%02d_den" % igrid)

    papd_dis_c = os.path.join(out_fldr, "re_p_%02d_rc" % igrid)
    rep_dis_c = os.path.join(out_fldr, "re_r_%02d_rc" % igrid)
    papd_rep_sum_c = os.path.join(out_fldr, "re_pr_%02d_rc" % igrid)

    re_out = os.path.join(out_fldr, "re_%02d_final" % igrid)

    # Human Debris Exports
    kf_den_r = os.path.join(out_fldr, "hd_kf_%02d_den" % igrid)
    hd_out = os.path.join(out_fldr, "hd_%02d_final" % igrid)
    hd_out_fns.append(hd_out)

    # Read Grid
    print(("Starting Grid %02d" % igrid))
    grid = gridclass.datagrid.FromRaster(grid_path, nodata_to_value=-99999)

    # REPORTED ERRORS

    # Calculate PAPD Density and Distance Rasters and Export Density Raster and Classified Distance Raster
    papd_den, papd_dis3, papd_dis5, papd_dis9, papd_dis17, papd_den_f = grid.ComputePointDensity(papd_vp, [gridclass.circle3, gridclass.circle5, gridclass.circle9, gridclass.circle17, gridclass.circle17])
    papd_den_2nm = papd_den_f
    grid.ExportMatchingRaster(papd_den_f, papd_den_r)

    papd_dis3[(papd_dis3 != 0)] = 5
    papd_dis5[(papd_dis5 != 0)] = 4
    papd_dis9[(papd_dis9 != 0)] = 3
    papd_dis17[(papd_dis17 != 0)] = 2
    papd_dis_pad1 = (papd_dis3 * 0) + 1

    papd_dis_max = numpy.array((papd_dis17, papd_dis9, papd_dis5, papd_dis3, papd_dis_pad1))
    papd_dis_max_f = numpy.max(papd_dis_max, 0)
    grid.ExportMatchingRaster(papd_dis_max_f, papd_dis_c)

    # Calculate Rep Density and Distance Rasters and Export Density Raster
    rep_den, rep_dis5, rep_dis9, rep_dis17, rep_dis31, rep_den_f = grid.ComputePointDensity(rep_vp, [gridclass.circle5, gridclass.circle9, gridclass.circle17, gridclass.circle31, gridclass.circle31])
    grid.ExportMatchingRaster(rep_den_f, rep_den_r)

    # Reclassify Point Rep Distance Histograms and Export Classified Raster
    rep_dis5[(rep_dis5 != 0)] = 5
    rep_dis9[(rep_dis9 != 0)] = 4
    rep_dis17[(rep_dis17 != 0)] = 3
    rep_dis31[(rep_dis31 != 0)] = 2
    rep_dis_pad1 = (rep_dis5 * 0) + 1

    rep_dis_max = numpy.array((rep_dis31, rep_dis17, rep_dis9, rep_dis5, rep_dis_pad1))
    rep_dis_max_f = numpy.max(rep_dis_max, 0)
    grid.ExportMatchingRaster(rep_dis_max_f, rep_dis_c)

    # Sum PAPD and Rep Density, Reclassify, and Export
    papd_rep_sum = papd_den_f + rep_den_f

    papd_rep_sum[(papd_rep_sum > 4)] = 5
    papd_rep_sum[(papd_rep_sum >= 2) & (papd_rep_sum <= 4)] = 4
    papd_rep_sum[(papd_rep_sum == 1)] = 2
    papd_rep_sum[(papd_rep_sum < 1)] = 1
    grid.ExportMatchingRaster(papd_rep_sum, papd_rep_sum_c)

    # Calculate maximum of Sum PAPD/Rep Density, Distance to Reported, or Distance to PAPD.
    re_max = numpy.array((papd_dis_max_f, rep_dis_max_f, papd_rep_sum))
    re_max_f = numpy.max(re_max, 0)
    grid.ExportMatchingRaster(re_max_f, re_out)
    print(("Finished Reported Errors at %.1f secs" % (time.time() - t_start)))

    # HUMAN DEBRIS (Changeability)

    # Calculate Known Features Density Raster
    kf_den, kf_den_2nm = grid.ComputePointDensity(kf_vp, [gridclass.circle17])
    hd = kf_den_2nm + (papd_den_2nm * 3)
    grid.ExportMatchingRaster(kf_den_2nm, kf_den_r)

    # Reclassify HD Raster (prevents overwriting of risk category)
    hd_5 = numpy.copy(hd)
    hd_4 = numpy.copy(hd)
    hd_3 = numpy.copy(hd)
    hd_2 = numpy.copy(hd)
    hd_1 = numpy.copy(hd) * 1

    hd_5[(hd_5 <= 9)] = 0
    hd_5[(hd_5 > 9)] = 5

    hd_4[(hd_4 < 6)] = 0
    hd_4[(hd_4 > 9)] = 0
    hd_4[(hd_4 >= 6) & (hd_4 <= 9)] = 4

    hd_3[(hd_3 < 2)] = 0
    hd_3[(hd_3 > 5)] = 0
    hd_3[(hd_3 >= 2) & (hd_3 <= 5)] = 3

    hd_2[(hd_2 < 1)] = 0
    hd_2[(hd_2 > 1)] = 0
    hd_2[(hd_2 == 1)] = 2

    # Find Max of Human Debris Raster
    hd_out_t = numpy.array((hd_5, hd_4, hd_3, hd_2, hd_1))
    hd_out_f = numpy.max(hd_out_t, 0)

    grid.ExportMatchingRaster(hd_out_f, hd_out)
    print(("Finished Human Debris at %.1f secs" % (time.time() - t_start)))

    # Create List of Distance, Density, and Max Rasters to be Combined and Deleted later
    papd_den_r_fns.append(papd_den_r)
    rep_den_r_fns.append(rep_den_r)
    papd_dis_c_fns.append(papd_dis_c)
    rep_dis_c_fns.append(rep_dis_c)
    papd_rep_sum_c_fns.append(papd_rep_sum_c)
    re_out_fns.append(re_out)
    kf_den_r_fns.append(kf_den_r)
    hd_out_fns.append(hd_out)

    del_fn = [papd_den_r, rep_den_r, papd_dis_c, rep_dis_c, papd_rep_sum_c, re_out, kf_den_r, hd_out_f]
    del_fns.extend(del_fn)
    print(("Total Time for Grid %02d = " % (igrid), time.time() - t_start))

# Combine All Rasters
t_start = time.time()
for filelist, outname in [[papd_den_r_fns, papd_den_r_fn],
                          [rep_den_r_fns, rep_den_r_fn],
                          [papd_dis_c_fns, papd_dis_c_fn],
                          [rep_dis_c_fns, rep_dis_c_fn],
                          [papd_rep_sum_c_fns, papd_rep_sum_c_fn],
                          [re_out_fns, re_out_fn + "_t"],
                          [papd_den_r_fns, papd_den_hd_r_fn],
                          [kf_den_r_fns, kf_den_r_fn],
                          [hd_out_fns, hd_out_fn]]:
    t_start2 = time.time()
    arcpy.MosaicToNewRaster_management(filelist, out_fldr, outname, prj_1, "32_BIT_SIGNED", str(cell_size), "1", "MAXIMUM", "FIRST")
    print(("Finished Mosaic to New Raster for %s" % (outname), time.time() - t_start2))
print(("Finished Mosaic to New Raster for %s" % (outname), time.time() - t_start))

# Reclassify Reported Errors Raster
re_final_t = os.path.join(out_fldr, re_out_fn + "_t")
re_final = os.path.join(out_fldr, re_out_fn)
r = arcpy.sa.Reclassify(re_out_fn + "_t", "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", "DATA")
r.save(re_final)
arcpy.Delete_management(re_out_fn + "_t")

for fn in del_fns:  # enumerate retrieves index number
    print((str(fn)))
    arcpy.Delete_mangagement(str(fn))

# Past Timing for Export Classified Density Rasters for all Grids Section (500 m grid cell resolution):
# Grid 1: 74 s
# Grid 2: 130 s
# Grid 3: 99 s
# Grid 4: 37 s
# Grid 5: 144 s
# Grid 6: 586 s
# Grid 7: 94 s
# Grid 8: 46 s
# Grid 9: 338 s
# Grid 10: 48 s
# Grid 11: 51 s
# Grid 12: 45 s
# Grid 13: 61 s
# Mosaic:  169 s


# Future Development: Consider creating dictionaries

# Mosaic or copy papd_2nm feature for reported errors and human debris raster
#d={'DENSITIES':{'PAPD':{"FILES":[], "OPTIONS":["1","MAXIMUM","FIRST"], "OUTPUT":papd_den_r_fn}, "REP":[rep_den_r_fns,"MINIMUM"]}, "DISTANCES":{}}
#d['DENSITIES']['PAPD']['FILES'] = []
# for data_type in d.keys(): #DENSITIES, DISTANCES, OUTPUTS
#    for data_set in ('PADP', 'KF'): #d[data_type]: #PAPD, REP, KF...
