# Standard Deviation of Depth Slope

# Objective: Classify complexity within each grid cell based on the standard deviation of surrounding depth slopes

# Complexity Category                      0                      1
# Slope                                   < X                    > X
# Distance to Hazard                  Less Complex           More Complex  
# Save to C:\Program Files\ArcGIS\Pro\bin\Python\Lib\site-packages\ArcExt\NHSP

# This script assumes the following
    # All input layers are projected into the correct and similar coordinate system 

# Import modules
import os, arcpy
import numpy, scipy, scipy.signal
#import matplotlib, matplotlib.pylab
#import skimage, skimage.draw
import HSTB.ArcExt.NHSP
from HSTB.ArcExt.NHSP import gridclass

# Define Parameters
#output_ws = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\Data_Processing\Depth_Slope\Depth_Slope.gdb"
output_ws = r"H:\NHSP_2_0\Christy\FLORIDA_TEST_CASE\FL_Grid.gdb"

#d_path = os.path.join(output_ws, "FL_ENCDepth_500m")
#d = arcpy.Raster(d_path)
#s_path = os.path.join(output_ws, "FL_ENCDepthSlope_500m")
s_path = os.path.join(output_ws, "FL_DepthSlope_30m_WGS84WM")
s = arcpy.Raster(s_path)
#g = os.path.join(output_ws, "FL_Grid")
cellSize = 500
#cellSize = 30
arcgrid = gridclass.grid(s, cellSize)

# If not running on Christy's machine...

descData = arcpy.Describe(g)
extent = descData.Extent
self.sr = descData.spatialReference
self.pnt = arcpy.Point(extent.XMin, extent.YMin) # Min x and y coordinate of grid raster; used to generate raster 
# Define extents of  Histogram based on extents of grid raster
# Note: If Must keep origin at same grid cell node, set orig_x and orig_y as min coords of grid (bottom left)
minx = extent.XMin
maxx = extent.XMax
self.orig_x = int(minx/cellSize)*int(cellSize) # Desired grid cell size = 500 m
self.x_edges = range(self.orig_x, int(maxx+cellSize), int(cellSize))

miny = extent.YMin
maxy = extent.YMax
self.cell_size_x = cellSize
self.cell_size_y = cellSize
self.orig_y = int(miny/cellSize)*int(cellSize) # Desired grid cell size = 500 m
self.y_edges = range(self.orig_y, int(maxy+int(cellSize)), int(cellSize))
        self.orig_new = arcpy.Point(self.orig_x, self.orig_y)


# Convert Raster to Numpy Array and change all no data values to NaN
#dd = arcpy.RasterToNumPyArray(d, nodata_to_value=-99999)
#dd[(dd == -99999)] = 'NaN'
ss = arcpy.RasterToNumPyArray(s, nodata_to_value=-99999)
ss[(ss == -99999)] = 'NaN'

# Iterate through raster and find standard deviation of depth slope within set distance
rad = (19-1)/2 # Search radius diameeter is 9500 m, 500 m grid cell
ss_std = numpy.zeros(ss.shape)
for r in range(ss.shape[0]):
    for c in range (ss.shape[1]):
        ss_std[r,c] = numpy.nanstd(ss[max(r-rad,0):r+rad+1, max(c-rad,0):c+rad+1]) #On left boundary of raster, there are no values. Set search radius to default to left boundary (0). On right side of raster, no values, but numpy will not return value by default Non inclusive, + 1 (0-2 would be 0 and 1)
        

# Export with origin of raster lower left


# Export with origin of grid lower left
arcgrid.ExportRaster(ss_std, os.path.join(output_ws, "FL_ENCDepthSlope_500m_STD5nm"), rotate=False)
#nav_raster = arcpy.NumPyArrayToRaster(dd, arcgrid.orig_new, cellSize, cellSize)
#arcpy.DefineProjection_management(nav_raster, arcgrid.sr)
#nav_raster.save(os.path.join(output_ws, "dd_raster_rasterorigin_norot"))