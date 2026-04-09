# Density and Proximity to Navigational Hazards
# Navigational Hazards are defined as an OBSTRN, PILPNT, UWTROC, or WRECKS features that occur within CATZOC B, C, D, or U coverage and do not include BOOM or FISH HAVEN features.

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of navigational hazard. Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of Hazards w/i 2 nm          0            0                1               2-4                 4+
# Distance to Hazard                >2 nm       1 - 2 nm        0.5 - 1 nm      0.5 - 0.25 nm       < 0.25 nm

# This script assumes the following
#    All input layers are projected into the correct and similar coordinate system
#    All input layers extend to the desired bounds (e.g. if analysis is only to be completed for CATZOC B area, input layers are clipped to CATZOC B extents)

# Import modules
import os
import time

import numpy
import scipy
import scipy.signal
import skimage
import skimage.draw
import arcpy
import arcgisscripting
from arcpy import env
from arcpy.sa import *

from HSTB.ArcExt.NHSP import gridclass, Functions
from HSTB.ArcExt.NHSP import globals

# Define Parameters
haz = globals.Parameters("KnownHazards")
aux = globals.Parameters("AUX")
# out_fldr = r'H:\HH_2018\Working\KnownHaz'

vec_pt_t = haz.vector_filename("KnownHaz_point")
vec_line_t = haz.vector_filename("KnownHaz_line")
vec_pg_t = haz.vector_filename("KnownHaz_polygon")

eez = aux["eez_buffered_ras"]

catzoc_b_r = r'H:\HH_2018\Auxiliary\bcdu_f'
grid_ws = r'H:\GRID\Grid.gdb'  # USER DEFINED

den_fns = []
den_c_fns = []
ln_clip_fns = []
pg_clip_fns = []
pg_clip_en_fns = []

# Output Variables
catzoc_b_p = haz.working_filename("catzoc_bcdu_p")
vec_pt = haz.working_filename("point_VP")
vec_line = haz.working_filename("line_VP")
vec_pg = haz.working_filename("polygon_VP")
den_final = "navhaz_r"
# den_c_temp_final = "NH_RC_t" #Remove comment and comment next line once back on old machine (shouldnt be RC though...)
den_c_temp_final = "navhaz_rct"  # Delete
den_c_rc = "navhaz_rct"  # Delete

den_c_final_temp = haz.working_rastername("rf")
den_c_final = haz.raster_final_filename()

# den_c_final = os.path.join(out_fldr, "navhaz_r")
# navhaz_final = os.path.join(out_fldr, "navhaz_final")


# Clip Vector Data by CATZOC B Area
arcpy.RasterToPolygon_conversion(catzoc_b_r, catzoc_b_p, "NO_SIMPLIFY", "VALUE")
bcdu = Functions.generate_polygon_bounds(catzoc_b_p, catzoc_b_p + "_bounds")
# arcpy.RepairGeometry_management(catzoc_b_p)
arcpy.Clip_analysis(vec_pt_t, catzoc_b_p + "_bounds", vec_pt)
arcpy.Clip_analysis(vec_line_t, catzoc_b_p + "_bounds", vec_line)
arcpy.Clip_analysis(vec_pg_t, catzoc_b_p + "_bounds", vec_pg)


tt = time.time()
for igrid in range(1, 14):  # Improve: Find all matching grid extents in the geodatabase
    vec_pg_en = vec_pg
    t_start = time.time()
    grid_name = "Grid_Model_Extents_%02d_Raster" % igrid
    grid_path = os.path.join(grid_ws, grid_name)
    model_bounds = os.path.join(grid_ws, "Model_Extents_%02d" % igrid)
    vec_line_clip = vec_line + "_%02d" % igrid
    vec_pg_clip = vec_pg + "_%02d" % igrid
    vec_pg_en = vec_pg_clip + "_envelope"
    dens_out = haz.raster_filename("NH_%02d_den" % igrid)
    dens_c_out = haz.raster_classified_filename("NH_%02d_denC" % igrid)

    ln_clip_fns.append(vec_line_clip)
    pg_clip_fns.append(vec_pg_clip)
    pg_clip_en_fns.append(vec_pg_en)

    # dens_pt_out = os.path.join(out_fldr, "NH_pt_%02d_den"%igrid) # Uncomment if desire point export for each grid
    # dens_ln_out = os.path.join(out_fldr, "NH_ln_%02d_den"%igrid)  # Uncomment if desire line export for each grid
    # dens_pg_out = os.path.join(out_fldr, "NH_pg_%02d_den"%igrid)  # Uncomment if desire polygon export for each grid

    ## Point Features ##

    # Read Grid and Extract XY Coordinates from point known features File
    print(("Starting Grid %02d" % igrid))
    grid = gridclass.datagrid.FromRaster(grid_path, nodata_to_value=-99999)

    # Generate histogram of point known features based on extents of grid raster
    # @todo: Future Development
    # For Point, Line, and Polygon features, add catzoc-driven buffer
    # 1. Add circle function to gridclass function for no buffer (just color in grid cell)
    # 2. Generate density grids based on CATZOC-driven buffer (CATZOC B = 50 m (gridclass.circle1); CATZOC C = 500 m (gridclass.circle3); CATZOC D/U = 2000 m (gridclass.circle9))
    # 3. Sum all point density grids
    h_den, h_den_f = grid.ComputePointDensity(vec_pt, [gridclass.circle17])

    # try: # Uncomment if desire point export for each grid
    #    grid.ExportMatchingRaster(h_den_f, dens_pt_out)
    #    print("Finished Point Density Raster Classify Export at %.1f secs"%(time.time()-t_start))
    # except RuntimeError:
    #    print("Export Reclassified Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    ## Line Features ##
    # @todo: Future Development
    # Add catzoc-driven buffer
    # Clip input polygons to grid extents
    print("Start clipping lines")
    arcpy.Clip_analysis(vec_line, model_bounds, vec_line_clip)
    print("Done clipping lines")

    # Generate list and array of x and y coordinates for each line known feature
    rows = [row[0] for row in arcpy.da.SearchCursor(vec_line_clip, ["SHAPE@", "*", ])]
    segs = []

    for row in rows:  # each row is a geometry object -- probably a polyline but could be multiple geometries
        for nPart in range(row.partCount):  # loop through all the geometries contained in the main "row" object
            segs.append([])
            polyseg = row.getPart(nPart)  # an individual polyline segment
            pt1 = None  # in each polysegment set the first point to None so it doesn't connect to the last polyline accidentally
            for pt2 in polyseg:
                if pt1 != None:  # connect the points of a polysegment together
                    segs[-1].append([[pt1.X, pt1.Y], [pt2.X, pt2.Y]])
                pt1 = pt2

    # Connect lines of same segment so single line is not counted twice if multiple segments and populate numpy array
    segs_v2 = []
    for s in segs:
        seg_i = grid.ArrayIndicesFromXY(numpy.array(s))
        segs_v2.append(seg_i)
    segs_indices = numpy.array(segs_v2)

    l_den_f = grid.zeros(dtype=numpy.uint32)
    temp_den = grid.zeros(dtype=numpy.uint32)
    for polysegment in segs_indices:
        temp_den *= 0  # clear the temp buffer
        for segment in polysegment:
            r, c = skimage.draw.line(segment[0][0], segment[0][1], segment[1][0], segment[1][1])  # Connects lines and populates raster with values where line intersects grid
            for ir, ic in zip(r, c):
                try:
                    temp_den[ir - 4:ir + 5, ic - 4:ic + 5] += gridclass.circle17
                except ValueError:
                    continue
        temp_den[temp_den > 0] = 1
        l_den_f += temp_den
    # try: # Uncomment if desire line export for each grid
    #    grid.ExportMatchingRaster(l_den_f, dens_ln_out)
    #    print("Finished Line Density Raster Export at %.1f secs"%(time.time()-t_start))
    # except RuntimeError:
    #    print("Export Line Density Raster Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    ## Polygon Features ##
    # @todo
    # Add catzoc-driven buffer
    t = time.time()

    # Clip input polygons to grid extents
    print("start clipping polygons")
    arcpy.Clip_analysis(vec_pg, model_bounds, vec_pg_clip)
    print("Done clipping Polygons")
    # Verify Clipped Polygon Feature Exists (clip analysis for polygon features will progress without error if empty output, but not output a file (unlike line features which will output an empty line file)).

    if arcpy.Exists(vec_pg_clip):
        vec_pg_en_geom = arcpy.MinimumBoundingGeometry_management(vec_pg_clip, vec_pg_en, "ENVELOPE", "NONE")

        # Make point features of polygon (5 points per polygon, first and last the same)
        pe_pts = arcpy.da.FeatureClassToNumPyArray(vec_pg_en_geom, ["SHAPE@XY"], explode_to_points=True)
        new_pts = numpy.array([pt[0] for pt in pe_pts])  # Generate array from points (if not, extra layer in data file and can not conduct vector math)
        pe_reshape = new_pts.reshape((-1, 5, 2))  # Make each polygon a separate array within array. Specify two inputs, otherwise read as one layer (no math)
        pe_grid_indices = grid.ArrayIndicesFromXY(pe_reshape)

        # Generate Polygon Navigational Hazards Density Histograms
        p_den_f = grid.zeros(dtype=numpy.uint32)  # numpy.zeros([len(grid.x_edges),len(grid.y_edges)]) #histogram for polygons that is the same size as the points histogram
        p_den = grid.zeros(dtype=numpy.uint32)  # numpy.zeros([len(grid.x_edges),len(grid.y_edges)]) #histogram for polygons that is the same size as the points histogram
        for i, envelope in enumerate(pe_grid_indices):
            p_den *= 0
            r1, c1 = envelope[0]
            r2, c2 = envelope[2]
            minr, maxr = min((r1, r2)), max((r1, r2))
            minc, maxc = min((c1, c2)), max((c1, c2))
            for ir in range(minr, maxr + 1):
                for ic in range(minc, maxc + 1):
                    try:
                        p_den[ir - 4:ir + 5, ic - 4:ic + 5] += gridclass.circle17
                    except ValueError:
                        # print("Polygon too close to edge - skipping")
                        # print(new_pts.reshape((-1,5,2))[i])
                        continue
            p_den[p_den > 0] = 1

            p_den_f += p_den  # gridclass.countOccurences(p_den, 17)
        print(("Finished Polygon Density Raster Classify Export at %.1f secs" % (time.time() - t)))

        # try: # Uncomment if desire polygon export for each grid
        #    grid.ExportMatchingRaster(p_den_f, dens_pg_out)
        #    print("Finished Polygon Density Raster Export at %.1f secs"%(time.time()-t_start))
        # except RuntimeError:
        #    print("Export Polygon Density Raster Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    else:
        # Sum point, line, and polygon density rasters, export summed raster, and store list of filepath.
        p_den_f = grid.zeros(dtype=numpy.uint32)

    # Make all numpy arrays the same size
    all_shape = numpy.array((h_den_f.shape, l_den_f.shape, p_den_f.shape))
    all_shape_max = numpy.amax(all_shape, axis=0)

    # Sum point, line, and polygon density rasters, export summed raster, and store list of filepath.
    nh_array = numpy.array((h_den_f, l_den_f, p_den_f))
    nh_array_sum = numpy.sum(nh_array, 0)

    try:
        grid.ExportMatchingRaster(nh_array_sum, dens_out)
        print(("Finished Summed Feature Export at %.1f secs" % (time.time() - t_start)))
    except RuntimeError:
        print("Export Sum Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    den_fns.append(dens_out)

    # Reclassify summed raster based on number of features within each grid cell
    nh_array_sum[(nh_array_sum > 4)] = 5
    nh_array_sum[(nh_array_sum >= 2) & (nh_array_sum <= 4)] = 4
    nh_array_sum[(nh_array_sum == 1)] = 2
    nh_array_sum[(nh_array_sum < 1)] = 1

    try:
        grid.ExportMatchingRaster(nh_array_sum, dens_c_out)
        print(("Finished Classified Summed Feature Export at %.1f secs" % (time.time() - t_start)))
    except RuntimeError:
        print("Export Reclassified Sum Density Numpy Array Process Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    den_c_fns.append(dens_c_out)

print(("Process took %.1f secs" % (time.time() - tt)))

# Combine Distance, Density, and Max Rasters
t_start = time.time()
arcpy.MosaicToNewRaster_management(den_fns, haz.raster_final_dir, den_final, haz.projection_number, "32_BIT_SIGNED", str(haz.cell_size), "1", "MAXIMUM", "FIRST")
arcpy.MosaicToNewRaster_management(den_c_fns, haz.raster_final_dir, den_c_rc, haz.projection_number, "32_BIT_SIGNED", str(haz.cell_size), "1", "MAXIMUM", "FIRST")
print(("Finished Mosaic to New Raster at %.1f secs" % (time.time() - t_start)))

# Reclassify Raster to get rid of '0' values
r_like = arcpy.sa.Reclassify(os.path.join(haz.raster_final_dir, den_c_rc), "VALUE", "0 0 1;1 1 1;2 2 2;4 4 4; 5 5 5; NODATA NODATA", "NODATA")
r_like.save(den_c_final_temp)

# Clip final raster to EEZ bounds
# Clip Final CATZOC_RC to EEZ Bounds
e = arcpy.sa.ExtractByMask(den_c_final_temp, eez)
e.save(den_c_final)

# Delete individual grid rasters
t_start = time.time()
for fn in range(0, 13):
    print(("Deleting Grid %02d Intermediate Rasters" % (fn + 1)))
    arcpy.Delete_management(str(den_fns[fn]))
    arcpy.Delete_management(str(den_c_fns[fn]))
    if arcpy.Exists(str(ln_clip_fns[fn])):
        arcpy.Delete_management(str(ln_clip_fns[fn]))
    if arcpy.Exists(str(pg_clip_fns[fn])):
        arcpy.Delete_management(str(pg_clip_fns[fn]))
    if arcpy.Exists(str(pg_clip_en_fns[fn])):
        arcpy.Delete_management(str(pg_clip_en_fns[fn]))
print(("Finished Process at %.1f secs" % (time.time() - t_start)))

# Past Timing for Export Classified Density Rasters for all Grids Section (500 m grid cell resolution;
# Grid 1: 74 s
# Grid 2: 164 s
# Grid 3: 70 s
# Grid 4: 46 s
# Grid 5: 96 s
# Grid 6: 774 s
# Grid 7: 52 s
# Grid 8: 33 s
# Grid 9: 154 s
# Grid 10: 33 s
# Grid 11: 33 s
# Grid 12: 25 s
# Grid 13: 33 s
# Mosaic: 124 s
