# Reported Errors
# A REP or PA Feature is defined as an OBSTRN, PILPNT, UWTROC, or WRECKS feature with QUAPOS = 7 or 8 or 4
# A ED or PD Features is defined as an OBSTRN, PILPNT, UWTROC, or WRECKS feature with QUAPOS = 5 or STATUS = 18 and NULL
# Known Features are defined as an OBSTRN, PILPNT, UWTROC, or WRECKS feature with QUAPOS != 4, 5, 7, 8 and STATUS <> 18

# Reported Errors: Max of PA/Rep Count (within 2 nm) PLUS PD/ED (within 4 nm), or Distance to nearest PA/Rep, or Distance to nearest ED/PD
# Human Debris Changeability: Sum of "Known" (i.e. non PA/PD, Rep) OBSTRN PLUS 3(PA/PD OBSTRN) within 3 nm
# Human Debris Forecast: Sum of "Known" (i.e. non PA/PD, Rep) OBSTRN PLUS 3(PA/PD OBSTRN) within 3 nm (same as above)

# Reported Errors
# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of PA/PD/ED and/or Rep. Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of PA/Rep w/i 2 nm            0            1                 -               2-4                 4+
# Distance to PA/Rep w/i 2 nm        >2 nm       1 - 2 nm        0.5 - 1 nm      0.5 - 0.25 nm       < 0.25 nm
# Number of ED/PD w/i 4 nm              0            1                 -               2-4                 4+
# Distance to ED/PD w/i 4 nm        >4 nm       2 - 4 nm        1 - 2 nm           1 - 0.5 nm         < 0.5 nm

# Define Input Variables
import os
import time

import numpy
import arcpy
import arcgisscripting

from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP.reported_error import *

# Define Parameters
reperr = globals.Parameters("RepErr")
aux = globals.Parameters("AUX")
inputs = globals.Parameters("Input")
grid_ws = inputs["grid_ws"]  # "r'H:\GRID\Grid.gdb'  # USER DEFINED

repe_raw = reperr.vector_filename("point")

# Define Output Variables
repe_PaRep_vp = reperr.vector_processed_filename("PARep")
repe_EdPd_vp = reperr.vector_processed_filename("EDPD")
parep_den_r_fn = reperr.raster_filename("PARep")  # PARep Density
edpd_den_r_fn = reperr.raster_filename("EDPD")  # EDPD Density
parep_dis_c_fn = reperr.raster_classified_filename("PARep")  # PARep Distance Classified
edpd_dis_c_fn = reperr.raster_classified_filename("EDPD")  # ED/PD Distance Classified
papd_rep_ed_sum_c_fn = reperr.raster_classified_filename("PRSum")  # Sum PA/Rep and PD/ED Classified
re_final = reperr.raster_final_filename()  # Reported Error Final
re_final_t = reperr.raster_final_filename("_t")

# Lists to hold filenames of variable grid paths to later be deleted
# Reported Errors Exports
parep_den_r_fns = []
edpd_den_r_fns = []
parep_dis_c_fns = []
edpd_dis_c_fns = []
papd_rep_ed_sum_c_fns = []
re_out_fns = []
del_fns = []

extract_raw(reperr)  # pulls the raw data from the ENCs

# VECTOR PROCESSED
# @todo - output to Vector Processed geodatabase
# Compute Vector Processed Data: Extract Rep Features
arcpy.Select_analysis(repe_raw, repe_PaRep_vp, repe_PaRep_query)
arcpy.Select_analysis(repe_raw, repe_EdPd_vp, repe_EdPd_query)

# RASTER
for igrid in range(1, 14):  # Improve: Find all matching grid extents in the geodatabase
    t_start = time.time()
    grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
    grid_path = os.path.join(grid_ws, grid_name)

    # Reported Errors Exports
    # Update with templayer code
    parep_den_r = reperr.working_rastername("pr_%02d_den" % igrid)
    edpd_den_r = reperr.working_rastername("ep_%02d_den" % igrid)

    parep_dis_c = reperr.raster_classified_filename("pr_%02d" % igrid)
    edpd_dis_c = reperr.raster_classified_filename("ep_%02d" % igrid)
    papd_rep_ed_sum_c = reperr.raster_classified_filename("rpred_%02d" % igrid)

    re_out = reperr.raster_final_filename("%02d_final" % igrid)

    # Read Grid
    print(("Starting Grid %02d" % igrid))
    grid = gridclass.datagrid.FromRaster(grid_path, nodata_to_value=-99999)

    # REPORTED ERRORS

    # Calculate PA/Rep Density and Distance Rasters and Export Density Raster and Classified Distance Raster
    parep_den, parep_dis3, parep_dis5, parep_dis9, parep_dis17, parep_den_f = grid.ComputePointDensity(repe_PaRep_vp, [gridclass.circle3, gridclass.circle5, gridclass.circle9, gridclass.circle17, gridclass.circle17])
    grid.ExportMatchingRaster(parep_den_f, parep_den_r)

    parep_dis3[(parep_dis3 != 0)] = 5
    parep_dis5[(parep_dis5 != 0)] = 4
    parep_dis9[(parep_dis9 != 0)] = 3
    parep_dis17[(parep_dis17 != 0)] = 2
    parep_dis_pad1 = (parep_dis3 * 0) + 1

    parep_dis_max = numpy.array((parep_dis17, parep_dis9, parep_dis5, parep_dis3, parep_dis_pad1))
    parep_dis_max_f = numpy.max(parep_dis_max, 0)
    grid.ExportMatchingRaster(parep_dis_max_f, parep_dis_c)
    print(("Finished PA/Rep Density and Distance Rasters at %.1f secs" % (time.time() - t_start)))

    # Calculate PD/ED Density and Distance Rasters and Export Density Raster
    edpd_den, edpd_dis5, edpd_dis9, edpd_dis17, edpd_dis31, edpd_den_f = grid.ComputePointDensity(repe_EdPd_vp, [gridclass.circle5, gridclass.circle9, gridclass.circle17, gridclass.circle31, gridclass.circle31])
    grid.ExportMatchingRaster(edpd_den_f, edpd_den_r)

    # Reclassify Point ED Distance Histograms and Export Classified Raster
    edpd_dis5[(edpd_dis5 != 0)] = 5
    edpd_dis9[(edpd_dis9 != 0)] = 4
    edpd_dis17[(edpd_dis17 != 0)] = 3
    edpd_dis31[(edpd_dis31 != 0)] = 2
    edpd_dis_pad1 = (edpd_dis5 * 0) + 1

    edpd_dis_max = numpy.array((edpd_dis31, edpd_dis17, edpd_dis9, edpd_dis5, edpd_dis_pad1))
    edpd_dis_max_f = numpy.max(edpd_dis_max, 0)
    grid.ExportMatchingRaster(edpd_dis_max_f, edpd_dis_c)
    print(("Finished PA/Rep Density and Distance Rasters at %.1f secs" % (time.time() - t_start)))

    # Sum PAPDED and Rep Density, Reclassify, and Export
    papd_rep_ed_sum = parep_den_f + edpd_den_f

    papd_rep_ed_sum[(papd_rep_ed_sum > 4)] = 5
    papd_rep_ed_sum[(papd_rep_ed_sum >= 2) & (papd_rep_ed_sum <= 4)] = 4
    papd_rep_ed_sum[(papd_rep_ed_sum == 1)] = 2
    papd_rep_ed_sum[(papd_rep_ed_sum < 1)] = 1
    grid.ExportMatchingRaster(papd_rep_ed_sum, papd_rep_ed_sum_c)

    # Calculate maximum of Sum PAPD/Rep Density, Distance to Reported, or Distance to PAPD.
    re_max = numpy.array((parep_dis_max_f, edpd_dis_max_f, papd_rep_ed_sum))
    re_max_f = numpy.max(re_max, 0)
    grid.ExportMatchingRaster(re_max_f, re_out)

    # Create List of Distance, Density, and Max Rasters to be Combined and Deleted later
    parep_den_r_fns.append(parep_den_r)
    edpd_den_r_fns.append(edpd_den_r)
    parep_dis_c_fns.append(parep_dis_c)
    edpd_dis_c_fns.append(edpd_dis_c)
    papd_rep_ed_sum_c_fns.append(papd_rep_ed_sum_c)
    re_out_fns.append(re_out)

    del_fn = [parep_den_r, edpd_den_r, parep_dis_c, edpd_dis_c, papd_rep_ed_sum_c, re_out]
    del_fns.extend(del_fn)
    print(("Total Time for Grid %02d = " % (igrid), time.time() - t_start))

# Combine All Rasters
t_start = time.time()
for filelist, outpath in [[parep_den_r_fns, parep_den_r_fn],
                          [edpd_den_r_fns, edpd_den_r_fn],
                          [parep_dis_c_fns, parep_dis_c_fn],
                          [edpd_dis_c_fns, edpd_dis_c_fn],
                          [papd_rep_ed_sum_c_fns, papd_rep_ed_sum_c_fn],
                          [re_out_fns, re_final_t]]:
    t_start2 = time.time()
    outdir, outname = os.path.split(outpath)
    arcpy.MosaicToNewRaster_management(filelist, outdir, outname, reperr.projection_number, "32_BIT_SIGNED", str(reperr.cell_size), "1", "MAXIMUM", "FIRST")
    print(("Finished Mosaic to New Raster for %s" % (outname), time.time() - t_start2))
print(("Finished Mosaic to New Raster for %s" % (outname), time.time() - t_start))

# Reclassify Reported Errors Raster
r = arcpy.sa.Reclassify(re_final_t, "VALUE", "0 NODATA;0 1 1;1 2 2;2 3 3;3 4 4;4 5 5", "DATA")
r.save(re_final)


for fn in del_fns:  # enumerate retrieves index number
    print((str(fn)))
    arcpy.Delete_mangagement(str(fn))
arcpy.Delete_management(re_final_t)


# Past Timing for Export Classified Density Rasters for all Grids Section (500 m grid cell resolution):
# Timing in order: Sept 2016, Jan 2018;
# Grid 1: 66 s (78 s)
# Grid 2: 111 s (147 s)
# Grid 3: 84 s (155 s)
# Grid 4: 35 s (55 s)
# Grid 5: 127 s (175 s)
# Grid 6: 518 s (643 s)
# Grid 7: 96 s (154 s)
# Grid 8: 43 s (76 s)
# Grid 9: 317 s (494 s)
# Grid 10: 62 s (82 s)
# Grid 11: 48 s (66 s)
# Grid 12: 81 s (84 s)
# Grid 13: 43 s (77 s)
# Mosaic:  139 s (1500 s)


# Future Development: Consider creating dictionaries

# Mosaic or copy papd_2nm feature for reported errors and human debris raster
# d={'DENSITIES':{'PAPD':{"FILES":[], "OPTIONS":["1","MAXIMUM","FIRST"], "OUTPUT":papd_den_r_fn}, "REP":[rep_den_r_fns,"MINIMUM"]}, "DISTANCES":{}}
# d['DENSITIES']['PAPD']['FILES'] = []
# for data_type in d.keys(): #DENSITIES, DISTANCES, OUTPUTS
#    for data_set in ('PADP', 'KF'): #d[data_type]: #PAPD, REP, KF...
