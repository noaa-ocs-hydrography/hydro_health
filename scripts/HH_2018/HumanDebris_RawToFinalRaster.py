# Reported Errors, Human Debris Changeability, and Human Debris Forecast .

# A REP or PA Feature is defined as an OBSTRN, PILPNT, or WRECKS feature with QUAPOS = 7 or 8 or 4
# A ED or PD Features is defined as an OBSTRN, PILPNT, or WRECKS feature with QUAPOS = 5 or STATUS = 18 and NULL
# Known Features are defined as an OBSTRN, PILPNT, or WRECKS feature with QUAPOS != 4, 5, 7, 8 and STATUS <> 18

# Human Debris: Sum of Known Hazards (within 2 nm) plus 3*PA/PD/Rep/ED (within 2 nm).
# Human Debris Changeability: Sum of "Known" (i.e. non PA/PD, Rep) OBSTRN PLUS 3(PA/PD OBSTRN) within 2 nm
# Human Debris Forecast: Sum of "Known" (i.e. non PA/PD, Rep) OBSTRN PLUS 3(PA/PD OBSTRN) within 2 nm (same as above)

# Human Debris Changeability and Forecast
# Objective: Classify risk within each grid cell based on density (within set distance) of PA/PD and/or Rep. Classify risk using the follwing table

# Risk Category                                          1            2                3                4                  5
# Density of Known Features + 3(PA/PD) within 2 nm       0            1              2 - 5            5 - 10              10 +

# Define Variables
import os
import time

import numpy
import arcpy
import arcgisscripting

from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP.human_debris import *

# Define Parameters
debris = globals.Parameters("HumanDebris")
aux = globals.Parameters("AUX")

grid_ws = r'H:\GRID\Grid.gdb'  # USER DEFINED

# NOTE: Must adapt script to adjust buffer (e.g. circle9) as a function of grid cell size!
eez = aux["eez_0_ras"]

edpd_vp = debris.working_filename("EDPD_VP")
parep_vp = debris.working_filename("PARep_VP")
kf_vp = debris.working_filename("kf_VP")


# Final Outputs
papd_den_hd_r_fn = "HD_PAPD_R"  # PAPD Density (same as above with human debris prefix)'''
ukf_den_r_fn = "HD_ukf_R"  # Unknown Features (PA/PD/ED/Rep) Density
kf_den_r_fn = "HD_KnF_R"  # Known Features Density
ukf_kf_m_r_fn = "HD_Mult_R"  # Known Features + 3*(Unknown Features)
hd_rc_fn = os.path.split(debris.raster_classified_filename())[1]  # "HD_RC"  # Human Debris RC
hd_final_fn = os.path.split(debris.raster_final_filename())[1]  # "HD_F"  # Human Debris Final

# RAW DATA
# Extract Raw Data: OBSTRN, PILPNT, and WRECKS Features
hd_raw = extract_raw(debris)[0]

# VECTOR PROCESSED
# Compute Vector Processed Data: Extract Known Features, PA/PD Features and Rep Features and project

# papd_query = "quapos = '4  ' OR quapos = '5  ' AND catobs IS NULL OR catobs = '                         ' OR catobs <> '1                        '" # 2016 model version
# rep_query = "quapos = '7  ' OR quapos = '8  ' AND catobs IS NULL OR catobs = '                         ' OR catobs <> '1                        '" # 2016 model version
# kf_query = "quapos <> '7  ' AND quapos <> '8  ' AND quapos <> '4  ' AND quapos <> '5  ' OR quapos IS NULL OR catobs = '                         '" # 2016 model version


parep_query = "((quapos = '4' OR quapos = '7' OR quapos = '8') AND (status IS NULL OR status <> '18   ')) AND ((catobs <> '1                        ') OR (catobs IS NULL))"
kf_query = "(quapos IS NULL OR quapos = '10 ' OR quapos = '1  ') AND (catobs IS NULL OR catobs <> '1                        '  OR catobs = '                         ' )"
edpd_query = "(( quapos = '5  ' AND status IS NULL) OR (quapos = '5  ' AND status = '18   ') OR (quapos IS NULL AND status = '18   ')) AND (catobs IS NULL OR catobs <> '1                        '  OR catobs = '                         ' )"
arcpy.Select_analysis(hd_raw, edpd_vp, edpd_query)
arcpy.Select_analysis(hd_raw, parep_vp, parep_query)
arcpy.Select_analysis(hd_raw, kf_vp, kf_query)

# Merge PA/Rep and ED/PD data for processing
hd_ukf_vp = debris.working_filename("edpd_parep_vp")
arcpy.Merge_management([edpd_vp, parep_vp], hd_ukf_vp)

# Lists to hold filenames of variable grid paths to later be deleted
ukf_den_r_fns = []
kf_den_r_fns = []
ukf_kf_m_r_fns = []
hd_out_fns = []
del_fns = []

# RASTER / RASTER CLASSIFIED

for igrid in range(1, 14):  # Improve: Find all matching grid extents in the geodatabase
    t_start = time.time()
    grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
    grid_path = os.path.join(grid_ws, grid_name)

    # Human Debris Exports
    ukf_den_r = debris.working_rastername("ukf_%02d_den" % igrid)
    kf_den_r = debris.working_rastername("kf_%02d_den" % igrid)
    ukf_kf_m_r = debris.working_rastername("mult_%02d" % igrid)
    hd_out_r = debris.working_rastername("%02d_final" % igrid)
    hd_out_fns.append(hd_out_r)

    # Read Grid
    print(("Starting Grid %02d" % igrid))
    grid = gridclass.datagrid.FromRaster(grid_path, nodata_to_value=-99999)

    # Calculate PAPD Density and Distance Rasters and Export Density Raster and Classified Distance Raster
    papd_den, papd_den_f = grid.ComputePointDensity(hd_ukf_vp, [gridclass.circle17])
    ukf_den_2nm = papd_den_f
    '''grid.ExportMatchingRaster(papd_den_f, ukf_den_r)'''

    # Calculate Known Features Density Raster
    kf_den, kf_den_2nm = grid.ComputePointDensity(kf_vp, [gridclass.circle17])
    # hd = kf_den_2nm+(papd_den_2nm*3)
    hd = kf_den_2nm + (ukf_den_2nm * 3)
    '''grid.ExportMatchingRaster(kf_den_2nm, kf_den_r)
    grid.ExportMatchingRaster(hd, ukf_kf_m_r)'''

    # Reclassify HD Raster (prevents overwriting of risk category)
    hd_5 = numpy.copy(hd)
    hd_4 = numpy.copy(hd)
    hd_3 = numpy.copy(hd)
    hd_2 = numpy.copy(hd)
    hd_1 = (numpy.copy(hd) * 0) + 1

    hd_5[(hd_5 <= 9)] = 1
    hd_5[(hd_5 > 9)] = 5

    hd_4[(hd_4 < 6)] = 1
    hd_4[(hd_4 > 9)] = 1
    hd_4[(hd_4 >= 6) & (hd_4 <= 9)] = 4

    hd_3[(hd_3 < 2)] = 1
    hd_3[(hd_3 > 5)] = 1
    hd_3[(hd_3 >= 2) & (hd_3 <= 5)] = 3

    hd_2[(hd_2 > 1)] = 99999
    hd_2[(hd_2 == 1)] = 2
    hd_2[(hd_2 == 99999)] = 1
    hd_2[(hd_2 == 0)] = 1

    # Find Max of Human Debris Raster
    hd_out_t = numpy.array((hd_5, hd_4, hd_3, hd_2, hd_1))
    hd_out_f = numpy.max(hd_out_t, 0)

    grid.ExportMatchingRaster(hd_out_f, hd_out_r)
    print(("Finished Human Debris at %.1f secs" % (time.time() - t_start)))

    # Create List of Distance, Density, and Max Rasters to be Combined and Deleted later
    ukf_den_r_fns.append(ukf_den_r)
    kf_den_r_fns.append(kf_den_r)
    ukf_kf_m_r_fns.append(ukf_kf_m_r)
    hd_out_fns.append(hd_out_r)

    # del_fn = [papd_den_r, rep_den_r, papd_dis_c, rep_dis_c, papd_rep_sum_c, re_out,kf_den_r,hd_out_f]
    del_fn = [ukf_den_r, kf_den_r, ukf_kf_m_r, hd_out_f]
    del_fns.extend(del_fn)
    print(("Total Time for Grid %02d = " % (igrid), time.time() - t_start))

# Combine All Rasters
t_start = time.time()
for filelist, outdir, outname in [[ukf_den_r_fns, debris.raster_dir, ukf_den_r_fn],
                                  [kf_den_r_fns, debris.raster_dir, kf_den_r_fn],
                                  [ukf_kf_m_r_fns, debris.raster_dir, ukf_kf_m_r_fn],
                                  [hd_out_fns, debris.raster_classified_dir, hd_rc_fn]]:
    t_start2 = time.time()
    # @todo is this the right place?  raster final or classified
    arcpy.MosaicToNewRaster_management(filelist, outdir, outname, debris.projection_number, "32_BIT_SIGNED", str(debris.cell_size), "1", "MAXIMUM", "FIRST")
    print(("Finished Mosaic to New Raster for %s" % (outname), time.time() - t_start2))
print(("Finished Mosaic to New Raster for %s" % (outname), time.time() - t_start))

# Clip HD RC to EEZ Bounds
e = arcpy.sa.ExtractByMask(debris.raster_classified_filename(), eez)
e.save(debris.raster_final_filename())

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
