# Density and Proximity to Navigational Hazards
# Navigational Hazards are defined as an OBSTRN, PILPNT, UWTROC, or WRECKS features that occur within CATZOC B, C, D, or U coverage and do not include BOOM or FISH HAVEN features.

# Objective: Classify risk within each grid cell based on proximity and density (within set distance) of navigational hazard. Classify risk using the follwing table

# Risk Category                       1            2                3                4                  5
# Number of Hazards w/i 2 nm          0            0                1               2-4                 4+
# Distance to Hazard                >2 nm       1 - 2 nm        0.5 - 1 nm      0.5 - 0.25 nm       < 0.25 nm

# This script assumes the following
    # All input layers are projected into the correct and similar coordinate system 
    # All input layers extend to the desired bounds (e.g. if analysis is only to be completed for CATZOC B area, input layers are clipped to CATZOC B extents)

# Import modules
import os, arcpy
import numpy, scipy, scipy.signal
import matplotlib, matplotlib.pylab
import skimage, skimage.draw
import HSTB.ArcExt.NHSP
from HSTB.ArcExt.NHSP import gridclass

## To reimport class:
# import importlib
#importlib.reload(gridclass)

# Define Parameters
output_ws = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\Nav_Hazards\Testing.gdb"
input_ws = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\Nav_Hazards\NAV_HAZARDS_BG.gdb"
n = os.path.join(input_ws, "FL_NavHazards_Points")
lines = os.path.join(input_ws, "FL_NavHazards_Lines")
p = os.path.join(input_ws, "FL_NavHazards_Polygons")
pe_out = os.path.join(output_ws, "FL_NavHazards_Polygons_Envelope")
g = os.path.join(input_ws, "FL_Grid")
cellSize = 500
arcgrid = gridclass.grid(g, cellSize)
n_out = os.path.join(output_ws, "FL_KnownFeatures_Points_Max")
l_out = os.path.join(output_ws, "FL_KnownFeatures_Lines_Max")
p_out = os.path.join(output_ws, "FL_KnownFeatures_Polygons_Max")

## Identify spatial characteristics of input grid raster 
#descData = arcpy.Describe(g)
##cellSize = descData.meanCellHeight
#extent = descData.Extent
#sr = descData.spatialReference
#pnt = arcpy.Point(extent.XMin, extent.YMin) # Min x and y coordinate of grid raster; used to generate raster of nav hazards

## Define extents of Navigation Hazards Histogram based on extents of grid raster
## Note: If Must keep origin at same grid cell node, set orig_x and orig_y as min coords of grid (bottom left)
#minx = extent.XMin
#maxx = extent.XMax
#orig_x = int(minx/cellSize)*int(cellSize) # Desired grid cell size = 500 m
#x_edges = range(orig_x, int(maxx+cellSize), int(cellSize))

#miny = extent.YMin
#maxy = extent.YMax
#orig_y = int(miny/cellSize)*int(cellSize) # Desired grid cell size = 500 m
#y_edges = range(orig_y, int(maxy+int(cellSize)), int(cellSize))
#orig_new = arcpy.Point(orig_x, orig_y)

# #################################### POINT FEATURES #################################### #

# Generate list and array of x and y coordinates for each Point Navigational Hazards 
rows = [row[0] for row in arcpy.da.SearchCursor(n,[ "SHAPE@", "*",])]
xys= numpy.array([(row.lastPoint.X, row.lastPoint.Y) for row in rows])

# Generate histogram of Point Navigation Hazards Features based on extents of grid raster
h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [arcgrid.x_edges,arcgrid.y_edges])

# Reclassify Point Navigational Hazards Density Array  
#h_den, xs, ys = numpy.histogram2d(xys[:,0], xys[:,1], [10, 10]) # TEST
#h_den = scipy.array(h_den, scipy.int32) # TEST: Visual of h_den, can delete 
#h_den_f = numpy.array(h_den) # count of point occurrences in each cell

h_den_f = scipy.signal.convolve2d(h_den, numpy.ones((9,9), scipy.int32), 'same') # Count of point occurrences within 2 nm of grid cell 
#h_den_f = scipy.array(h_den_f, scipy.int32) # TEST: Visual of h_den, can delete 
h_den_f[(h_den_f > 4)] = 5
h_den_f[(h_den_f >= 2) & (h_den_f <= 4)] = 4
h_den_f[(h_den_f == 1)] = 3
h_den_f[(h_den_f < 1)] = 1

# Export Point Navigational Hazards Density Array
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_den_f)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Points_ClassifiedDensity9"))

# Generate Point Navigational Hazards Distance Histograms
h_dist3 = gridclass.countOccurences(h_den, 3)
h_dist5 = gridclass.countOccurences(h_den, 5)
h_dist7 = gridclass.countOccurences(h_den, 7)
h_dist9 = gridclass.countOccurences(h_den, 9)

# Reclassify Point Navigational Hazards Distance Histograms
h_dist3[(h_dist3 != 0)] = 5
h_dist5[(h_dist5 != 0)] = 4
h_dist7[(h_dist7 != 0)] = 3
h_dist9[(h_dist9 != 0)] = 2

# Export all distance point navigational hazard histograms
# Export Point Navigational Hazards Distance Array (3,3 = 500 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist3)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Points_ClassifiedDistance3"))

# Export Point Navigational Hazards Distance Array (5,5 = 1000 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist5)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Points_ClassifiedDistance5"))

# Export Point Navigational Hazards Distance Array (7,7 = 1500 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist7)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Points_ClassifiedDistance7"))

# Export Point Navigational Hazards Distance Array (9,9 = 2000 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(h_dist9)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Points_ClassifiedDistance9"))

# #################################### LINE FEATURES #################################### #

# Generate list and array of x and y coordinates for each Line Navigational Hazards 
rows = [row[0] for row in arcpy.da.SearchCursor(lines,[ "SHAPE@", "*",])]
segs=[]
for row in rows: #each row is a geometry object -- probably a polyline but could be multiple geometries
    for nPart in range(row.partCount): #loop thru all the geometries contained in the main "row" object
        segs.append([])
        polyseg=row.getPart(nPart) #an individual polyline segment
        pt1 = None #in each polysegment set the first point to None so it doesn't connect to the last polyline accidentally
        for pt2 in polyseg:
            if pt1 != None: #connect the points of a polysegment together
                segs[-1].append([[pt1.X, pt1.Y], [pt2.X, pt2.Y]])
            pt1=pt2
#all_segs = numpy.array(segs)
#segs_indices = numpy.array((all_segs-numpy.array((orig_x, orig_y)))/numpy.array((cellSize, cellSize)), dtype=numpy.int)

# Connect lines of same segment so single line is not counted twice if multiple segments
segs_v2 = []
for s in segs:
    seg_i = numpy.array((numpy.array(s)-numpy.array((arcgrid.orig_x, arcgrid.orig_y)))/numpy.array((cellSize, cellSize)), dtype=numpy.int)
    #seg_i = arcgrid.ArrayIndicesFromXY(numpy.array(s))
    segs_v2.append(seg_i)
segs_indices = numpy.array(segs_v2)

l_den = numpy.zeros([len(arcgrid.x_edges),len(arcgrid.y_edges)]) #histogram for polygons that is the same size as the points histogram
temp_den = numpy.zeros([len(arcgrid.x_edges),len(arcgrid.y_edges)]) #histogram for polygons that is the same size as the points histogram
for polysegment in segs_indices:
    temp_den*=0 #clear the temp buffer
    for segment in polysegment:
        r,c = skimage.draw.line(segment[0][0], segment[0][1], segment[1][0], segment[1][1])
        temp_den[r,c] = 1
    l_den+=temp_den


# Reclassify Line Navigational Hazards Density Array  
l_den_f = gridclass.countOccurences(l_den, 9) #scipy.signal.convolve2d(l_den, numpy.ones((9,9), scipy.int32), 'same') # Count of point occurrences within 2 nm of grid cell 
l_den_f[(l_den_f > 4)] = 5
l_den_f[(l_den_f >= 2) & (l_den_f <= 4)] = 4
l_den_f[(l_den_f == 1)] = 3
l_den_f[(l_den_f < 1)] = 1

# Export Line Navigational Hazards Density Array
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(l_den_f)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Lines_ClassifiedDensity9"))

# Generate Line Navigational Hazards Distance Histograms
l_dist3 = gridclass.countOccurences(l_den, 3) #scipy.signal.convolve2d(l_den, numpy.ones((3,3), scipy.int32), 'same')
l_dist5 = gridclass.countOccurences(l_den, 5) #scipy.signal.convolve2d(l_den, numpy.ones((5,5), scipy.int32), 'same')
l_dist7 = gridclass.countOccurences(l_den, 7) #scipy.signal.convolve2d(l_den, numpy.ones((7,7), scipy.int32), 'same')
l_dist9 = gridclass.countOccurences(l_den, 9) #scipy.signal.convolve2d(l_den, numpy.ones((9,9), scipy.int32), 'same')

# Reclassify Point Navigational Hazards Distance Histograms
l_dist3[(l_dist3 != 0)] = 5
l_dist5[(l_dist5 != 0)] = 4
l_dist7[(l_dist7 != 0)] = 3
l_dist9[(l_dist9 != 0)] = 2

# Export all distance line navigational hazard histograms
# Export Point Navigational Hazards Distance Array (3,3 = 500 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(l_dist3)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Lines_ClassifiedDistance3"))

# Export Point Navigational Hazards Distance Array (5,5 = 1000 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(l_dist5)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Lines_ClassifiedDistance5"))

# Export Point Navigational Hazards Distance Array (7,7 = 1500 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(l_dist7)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Lines_ClassifiedDistance7"))

# Export Point Navigational Hazards Distance Array (9,9 = 2000 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(l_dist9)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Lines_ClassifiedDistance9"))

# ################################### POLYGON FEATURES ################################### #
# Generate enveloped polygon features (max bounds)
pe = arcpy.MinimumBoundingGeometry_management(p, pe_out, "ENVELOPE", "NONE")

# Make point features of polygon (5 points per polygon, first and last the same)
pe_pts = arcpy.da.FeatureClassToNumPyArray(pe,["SHAPE@XY"], explode_to_points=True)
new_pts = numpy.array([pt[0] for pt in pe_pts]) # Generate array from points (if not, extra layer in data file and can not conduct vector math)
pe_reshape = new_pts.reshape((-1,5,2)) # Make each polygon a separate array within array. Specify two inputs, otherwise read as one layer (no math)
pe_grid_indices = (pe_reshape-numpy.array((arcgrid.orig_x, arcgrid.orig_y)))/numpy.array((cellSize, cellSize)) # Translating polygon enveloppe indices to grid indices

#def row_index(y): #deprecated
#    return (y-orig_y)/cell_size_y
#def col_index(x):
#    return (x-orig_x)/cell_size_x

# Generate Polygon Navigational Hazards Density Histograms 
p_den = numpy.zeros([len(arcgrid.x_edges),len(arcgrid.y_edges)]) #histogram for polygons that is the same size as the points histogram
for envelope in pe_grid_indices:
    r1,c1 = envelope[0]
    r2,c2 = envelope[2]
    minr, maxr = min((r1,r2)), max((r1,r2))
    minc, maxc = min((c1,c2)), max((c1,c2))
    p_den[minr:maxr+1, minc:maxc+1]+=1

# Reclassify Polygon Navigational Hazards Density Array  
p_den_f = scipy.signal.convolve2d(p_den, numpy.ones((9,9), scipy.int32), 'same') # Count of point occurrences within 2 nm of grid cell 
p_den_f[(p_den_f > 4)] = 5
p_den_f[(p_den_f >= 2) & (p_den_f <= 4)] = 4
p_den_f[(p_den_f == 1)] = 3
p_den_f[(p_den_f < 1)] = 1

# Export Polygon Navigational Hazards Density Array
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(p_den_f)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Polygons_ClassifiedDensity9"))

# Generate Line Navigational Hazards Distance Histograms
p_dist3 = gridclass.countOccurences(p_den, 3)  #scipy.signal.convolve2d(p_den, numpy.ones((3,3), scipy.int32), 'same')
p_dist5 = gridclass.countOccurences(p_den, 5)  #scipy.signal.convolve2d(p_den, numpy.ones((5,5), scipy.int32), 'same')
p_dist7 = gridclass.countOccurences(p_den, 7)  #scipy.signal.convolve2d(p_den, numpy.ones((7,7), scipy.int32), 'same')
p_dist9 = gridclass.countOccurences(p_den, 9)  #scipy.signal.convolve2d(p_den, numpy.ones((9,9), scipy.int32), 'same')

# Reclassify Point Navigational Hazards Distance Histograms
p_dist3[(p_dist3 != 0)] = 5
p_dist5[(p_dist5 != 0)] = 4
p_dist7[(p_dist7 != 0)] = 3
p_dist9[(p_dist9 != 0)] = 2

# Export all distance polygon navigational hazard histograms
# Export Point Navigational Hazards Distance Array (3,3 = 500 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(p_dist3)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Polygons_ClassifiedDistance3"))

# Export Point Navigational Hazards Distance Array (5,5 = 1000 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(p_dist5)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Polygons_ClassifiedDistance5"))

# Export Point Navigational Hazards Distance Array (7,7 = 1500 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(p_dist7)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Polygons_ClassifiedDistance7"))

# Export Point Navigational Hazards Distance Array (9,9 = 2000 m)
#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(p_dist9)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, "FL_NavHazards_Polygons_ClassifiedDistance9"))


# ################################## DEFINE FINAL ARRAY ################################## #

# Calculate Max value per grid cell as a function of all arrays (density, distance3, distance 5, distance 7 and distance 9) for each geometry type
pt_risk = numpy.array((h_den_f, h_dist9, h_dist7, h_dist5, h_dist3))
pt_risk_f = numpy.max(pt_risk, 0)

ln_risk = numpy.array((l_den_f, l_dist9, l_dist7, l_dist5, l_dist3))
ln_risk_f = numpy.max(ln_risk, 0)

pg_risk = numpy.array((p_den_f, p_dist9, p_dist7, p_dist5, p_dist3))
pg_risk_f = numpy.max(pg_risk, 0)

# Export max value raster per geometry type
arcgrid.ExportRaster((numpy.rot90(pt_risk_f)), os.path.join(output_ws, n_out))
arcgrid.ExportRaster((numpy.rot90(ln_risk_f)), os.path.join(output_ws, l_out))
arcgrid.ExportRaster((numpy.rot90(pg_risk_f)), os.path.join(output_ws, p_out))

#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(pt_risk_f)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, n_out))

#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(ln_risk_f)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, l_out))

#nav_raster = arcpy.NumPyArrayToRaster((numpy.rot90(pg_risk_f)), orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, sr)
#nav_raster.save(os.path.join(output_ws, p_out))

