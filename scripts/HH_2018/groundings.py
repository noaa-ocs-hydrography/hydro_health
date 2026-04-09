# Density and Proximity to Groundings
# Groundings are defined as a grounding incident by a non-recreational or fishing vessel.

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of grounding. Classify risk using the following table

# Risk Category                       1            2                3                4                  5
# Number of Groundings w/i 2 nm       0            0                1               2-4                 4+
# Distance to Hazard                >2 km       1 - 2 km        0.5 - 1 km      0.5 - 0.25 km       < 0.25 km

# This script assumes the following
# All input layers are projected into the correct and similar coordinate system
# All input layers extend to the desired bounds (e.g. if analysis is only to be completed for CATZOC B area, input layers are clipped to CATZOC B extents)

# Import modules
import os
import time
import traceback

import numpy
import arcpy
import arcgisscripting

from HSTB.ArcExt.NHSP import Functions
from HSTB.ArcExt.NHSP import gridclass
from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP.globals import Parameters, timer
exec("from HSTB.ArcExt.NHSP.globals import print")


groundings_query = "LOWER(TRIM(BOTH ' ' FROM " + "IncidentSubTypeLookupName" + ")) = '" + "grounding" + "'"


def create_vp(raw_data, x, y, params):
    return Functions.query_csv(raw_data, x, y, params, groundings_query)


# FIX THIS
# Workspace
# arcpy.env.workspace = r"N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2018\Working\Groundings"


def get_vector_processed(ground):
    """ creates VECTOR PROCESSED
        Read raw grounding data and extract only grounding incidents
        """
    # User-Defined Input Variables
    x_coords = ground["x_coords"]  # User-Defined
    y_coords = ground["y_coords"]  # User-Defined
    ground_raw = os.path.join(ground.raw_dir, ground["csv_data"])  # User-Defined
    ground_vp = create_vp(ground_raw, x_coords, y_coords, ground)
    return ground_vp


def process_grid(ground, igrid):
    with timer("Finished Processing Grid %02d" % igrid, "Processing Grid %02d" % igrid) as tm:
        grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
        grid_ws = ground.ini["Input"]["grid_ws"]
        grid_path = os.path.join(grid_ws, grid_name)
        dens_c_out = ground.working_rastername("%02d_denC" % igrid)
        dis_c_out = ground.working_rastername("%02d_disC" % igrid)
        dens_dis_max_out = ground.working_rastername("%02d_max" % igrid)

        # Read Grid and Extract XY Coordinates from Groundings File and Generate histogram of point groundings features based on extents of grid raster
        grid = gridclass.datagrid.FromRaster(grid_path, nodata_to_value=-99999)
        h_den, h_dis3, h_dis5, h_dis7, h_dis9, h_den_f = grid.ComputePointDensity(ground.vector_processed_filename(), [gridclass.circle3, gridclass.circle5, gridclass.circle9, gridclass.circle17, gridclass.circle17])

        # Reclassify Numpy Array based on density of groundings and Export Classified Raster
        h_den_f[(h_den_f > 4)] = 5
        h_den_f[(h_den_f >= 2) & (h_den_f <= 4)] = 4
        h_den_f[(h_den_f == 1)] = 2
        h_den_f[(h_den_f < 1)] = 1

        try:
            grid.ExportMatchingRaster(h_den_f, dens_c_out)
            tm.msg("Finished Density Raster Classify Export")
        except RuntimeError:
            tm.msg("Export Reclassified Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

        # Reclassify Point Groundings Distance Histograms and Export Classified Raster
        h_dis3[(h_dis3 != 0)] = 5
        h_dis5[(h_dis5 != 0)] = 4
        h_dis7[(h_dis7 != 0)] = 3
        h_dis9[(h_dis9 != 0)] = 2

        h_dis_max = numpy.array((h_dis9, h_dis7, h_dis5, h_dis3))
        h_dis_max_f = numpy.max(h_dis_max, 0)
        try:
            grid.ExportMatchingRaster(h_dis_max_f, dis_c_out)
            tm.msg("Finished Distance Raster Classify Export at ")
        except RuntimeError:
            tm.msg("Export Reclassified Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

        # Calculate Maximum Density or Distance and Export Raster
        h_den_dis_array = numpy.array((h_den_f, h_dis_max_f))
        h_den_dis_max = numpy.max(h_den_dis_array, 0)
        try:
            grid.ExportMatchingRaster(h_den_dis_max, dens_dis_max_out)
            tm.msg("Finished Max Raster Classify Export at ")
        except RuntimeError:
            tm.msg("Export Reclassified Maximum Distance and Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")
    return dis_c_out, dens_c_out, dens_dis_max_out


def process_grids(ground, grids=list(range(1, 14))):
    dis_fns = []
    dens_fns = []
    max_fns = []
    for igrid in grids:
        dis_c_out, dens_c_out, dens_dis_max_out = process_grid(ground, igrid)
        # Create List of Distance, Density, and Max Rasters to be Combined and Deleted later
        for igrid in grids:
            dis_fns.append(dis_c_out)
            dens_fns.append(dens_c_out)
            max_fns.append(dens_dis_max_out)
    return dis_fns, dens_fns, max_fns


def execute(ini, grids=list(range(1, 14)), use_existing_vp=True):
    ground = Parameters("Groundings", ini)
    if not use_existing_vp or not arcpy.Exists(ground.vector_processed_filename()):
        ground_vp = get_vector_processed(ground)
        if not ground_vp:
            print("failed to make vector processed data for groundings, did csv file not exist?")
            return
    # Export Classified Distance and Density Rasters for all Grids (13 GRIDS TOTAL)
    dis_fns, dens_fns, max_fns = process_grids(ground, grids)

    # Output Variables
    ground_dis_rc = ground.raster_classified_filename("dis")
    ground_den_rc = ground.raster_classified_filename("den")
    ground_rc_t = ground.working_filename("rc_t", False)
    ground_rc = ground.raster_classified_filename()
    ground_f = ground.raster_final_filename()

    # Combine Distance, Density, and Max Rasters
    with timer("Finished Mosaic to New Raster at ", "Start Raster Mosaic") as _t:
        arcpy.MosaicToNewRaster_management(dis_fns, os.path.split(ground_dis_rc)[0], os.path.split(ground_dis_rc)[1], ground.projection_number, "32_BIT_SIGNED", str(ground.cell_size), "1", "MAXIMUM", "FIRST")
        arcpy.MosaicToNewRaster_management(dens_fns, os.path.split(ground_den_rc)[0], os.path.split(ground_den_rc)[1], ground.projection_number, "32_BIT_SIGNED", str(ground.cell_size), "1", "MAXIMUM", "FIRST")
        arcpy.MosaicToNewRaster_management(max_fns, os.path.split(ground_rc_t)[0], os.path.split(ground_rc_t)[1], ground.projection_number, "32_BIT_SIGNED", str(ground.cell_size), "1", "MAXIMUM", "FIRST")

    # Reclassify the mosaicked raster to remove zero values in no data areas
    with timer("Finished Raster Reclassification ", "Start Raster Reclassification") as _t:
        arcpy.gp.Reclassify_sa(ground_rc_t, "VALUE", "111 0;112 116 NODATA", ground_rc, "DATA")

    # Parameters
    eez_r = ini["AUX"]["eez_0_ras"]

    # Clip final raster to EEZ Bounds
    with timer("Finished Raster Extraction ", "Start Raster Extraction") as _t:
        outExtractByMask = arcpy.sa.ExtractByMask(ground_rc, eez_r)
        arcpy.Delete_management(ground_f)  # save fails if data on disk in the way
        outExtractByMask.save(ground_f)

    # Delete Unnecesary Files
    with timer("Finished Temp Layer Deletion ", "Deleting temp layers") as _t:
        for fns in (dis_fns, dens_fns, max_fns):
            for fn in fns:
                arcpy.Delete_management(fn)
        arcpy.Delete_management(str(ground_rc_t))


def main():
    execute(globals.initial_values, use_existing_vp=False)


# Past Timing for Export Classified Density Rasters for all Grids Section (500 m grid cell resolution;
# Grid 1: 51 s
# Grid 2: 73 s
# Grid 3: 47 s
# Grid 4: 32 s
# Grid 5: 71 s
# Grid 6: 205 s
# Grid 7: 58 s
# Grid 8: 44 s
# Grid 9: 130 s
# Grid 10: 48 s
# Grid 11: 43 s
# Grid 12: 48 s
# Grid 13: 32 s
# Mosaic: 124 s
