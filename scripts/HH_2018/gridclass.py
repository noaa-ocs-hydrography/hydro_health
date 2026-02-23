
import os
import time
import math
import random
import string

import scipy.signal
import numpy
import arcpy
import arcgisscripting
from arcpy import env
from arcpy import sa
# wrapping this import as eclipse parser uses python 2
exec("from HSTB.ArcExt.NHSP.globals import print")

try:
    import scipy.signal

    def countOccurences(hist, mat):
        if isinstance(mat, int):
            mat = numpy.ones((mat, mat), scipy.int32)
        return scipy.signal.convolve2d(hist, mat, 'same')
except:
    print("failed to load scipy")


class GridError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class grid(object):
    '''Class to handle an ESRI grid
        NOTE: MUST UPDATE OLDER SCRIPTS TO INCLUDE BUFFER DISTANCE (4 nm = ~7500 m)'''

    def __init__(self, g, cellSize, buffer_dist=0):
        '''cellSize is either a single number for a square grid or a 2-tuple for x_size, ysize
        !!!buffer_dist is the size to expand a grid -- not sure this should be here!!!
        '''
        try:
            self.cell_size_x, self.cell_size_y = cellSize
        except:
            self.cell_size_x = cellSize
            self.cell_size_y = cellSize
        # Identify spatial characteristics of input grid raster
        if g is not None:
            descData = arcpy.Describe(g)
            # cellSize = descData.meanCellHeight
            extent = descData.Extent
            self.sr = descData.spatialReference
            if math.isnan(extent.XMin):
                raise GridError("Layer is empty, Xmin is NaN")
            self.minx = extent.XMin - buffer_dist  # Desired grid cell size = 500 m ## BUFFER
            self.maxx = extent.XMax + buffer_dist
            self.miny = extent.YMin - buffer_dist  # Desired grid cell size = 500 m ## BUFFER
            self.maxy = extent.YMax + buffer_dist
        else:
            self.sr = None
            self.minx, self.maxx, self.miny, self.maxy = 0, 0, 0, 0

    def __getstate__(self):
        d = self.__dict__.copy()
        d.pop('sr', 0)   # remove the ESRI spatialReference which can't be pickled directly
        if self.sr:
            d['srstring'] = self.sr.exportToString()
        return d

    def __setstate__(self, obj):
        self.__dict__ = obj
        if obj.get('srstring', None):
            self.sr = arcpy.SpatialReference()
            self.sr.loadFromString(obj['srstring'])
        else:
            self.sr = None

    def clip(self, other_grid):
        new_grid = grid(None, [self.cell_size_x, self.cell_size_y])
        new_grid.minx = max(other_grid.minx, self.minx)
        new_grid.maxx = min(other_grid.maxx, self.maxx)
        new_grid.miny = max(other_grid.miny, self.miny)
        new_grid.maxy = min(other_grid.maxy, self.maxy)
        new_grid.sr = self.sr
        return new_grid

    @property
    def cell_size_x(self):
        return self._cell_size_y

    @cell_size_x.setter
    def cell_size_x(self, value):
        self._cell_size_y = value
        self.reset_bounds()

    @property
    def cell_size_y(self):
        return self._cell_size_y

    @cell_size_y.setter
    def cell_size_y(self, value):
        self._cell_size_y = value
        self.reset_bounds()

    @property
    def minx(self):
        return self._minx

    @minx.setter
    def minx(self, value):
        self._minx = value
        self.reset_bounds()

    @property
    def miny(self):
        return self._miny

    @miny.setter
    def miny(self, value):
        self._miny = value
        self.reset_bounds()

    @property
    def maxx(self):
        return self._maxx

    @maxx.setter
    def maxx(self, value):
        self._maxx = value
        self.reset_bounds()

    @property
    def maxy(self):
        return self._maxy

    @maxy.setter
    def maxy(self, value):
        self._maxy = value
        self.reset_bounds()

    @property
    def orig_x(self):
        return self.minx

    @property
    def orig_y(self):
        return self.miny

    @property
    def origin(self):
        return (self.orig_x, self.orig_y)

    def reset_bounds(self):
        try:
            self.x_edges = numpy.arange(self.orig_x, self.maxx + self.cell_size_x, self.cell_size_x)
            self.y_edges = numpy.arange(self.orig_y, self.maxy + self.cell_size_y, self.cell_size_y)
        except AttributeError:
            pass  # may not be set up yet

    # Define extents of Groundings Histogram based on extents of grid raster
    # Note: If Must keep origin at same grid cell node, set orig_x and orig_y as min coords of grid (bottom left)

    @property
    def orig(self):
        return arcpy.Point(self.orig_x, self.orig_y)  # Min x and y coordinate of grid raster; used to generate raster of Groundings

    @property
    def cellSize_area(self):
        return self.cell_size_x * self.cell_size_y

    @property
    def cellSize_areaL(self):
        return self.cellSize_area - 1  # Limit grid cell size to cell size area minus 1 m2

    def RowColFromXY(self, x, y):
        return self.row_index(y), self.col_index(x)
        raise Exception("This is backwards -- use ArrayIndicesFromXY")

    def row_index(self, y):  # deprecated
        return (y - self.orig_y) / self.cell_size_y

    def col_index(self, x):
        return (x - self.orig_x) / self.cell_size_x

    def ArrayIndicesFromXY(self, inputarray):
        '''Pass in a numpy array of XY values and returns the row,column indices as a numpy array'''
        output = numpy.array((inputarray - numpy.array(self.origin)) / numpy.array((self.cell_size_x, self.cell_size_y)), dtype=numpy.int32)  # Translating polygon enveloppe indices to grid indices
        return output

    def ExportMatchingRaster(self, data, filename, rot=False):
        '''Export a raster that has the same origin and cell sizes to an arc layer
           Useful if operations (adding/subtracting etc) are done to the gridarray of a datagrid instance.
        '''
#        if rot:
#            g = numpy.rot90(data)
#        else:
#            g = numpy.flipud(data.T)
#        nav_raster = arcpy.NumPyArrayToRaster(g, self.orig, self.cell_size_x, self.cell_size_y)
#        arcpy.DefineProjection_management(nav_raster, self.sr)
#        nav_raster.save(filename)
        try:
            if rot:
                g = numpy.rot90(data)
            else:
                g = numpy.flipud(data.T)
            if g.dtype == numpy.int64:
                g = g.astype(numpy.int32)
            nav_raster = arcpy.NumPyArrayToRaster(g, self.orig, self.cell_size_x, self.cell_size_y)
            arcpy.DefineProjection_management(nav_raster, self.sr)
            nav_raster.save(filename)
        except Exception as err:
            print(err.args[0])
            if "ERROR 010240" in err.args[0]:
                raise RuntimeError(filename)
            print("Error 999998 may be unsupported data type in grid")
        except RuntimeError:
            print("Export Raster to Numpy Array Failed. Filepath or Filename may be too long. Filepath is limited to 128 characters and filename is limited to 13 characters. Revise file naming structure and re-run.")

    def zeros(self, dtype=numpy.float64):
        return numpy.zeros([len(self.x_edges) - 1, len(self.y_edges) - 1], dtype=dtype)

    def MakeVectorGrid(self, clip_filepath, output_gdb, filename):
        t_start = time.time()
        out_vector = os.path.join(output_gdb, filename)
        out_raster = os.path.join(output_gdb, filename + "_Raster")
        # temp_raster = os.path.join(output_gdb, "temp_raster")
        temp_raster_clip = os.path.join(output_gdb, "temp_raster_clip")
        temp_vector = os.path.join(output_gdb, "temp_vector")
        query = "Shape_Area >=" + "%s" % (self.cellSize_areaL)

        r = numpy.arange((len(self.x_edges) - 1) * (len(self.y_edges) - 1), dtype=numpy.uint32)
        r = r.reshape([len(self.x_edges) - 1, len(self.y_edges) - 1])
        self.ExportMatchingRaster(r, out_raster)
        arcpy.BuildRasterAttributeTable_management(out_raster, "Overwrite")
        print("Finished Export Raster at %.1f secs" % (time.time() - t_start))

        # Leave Commented
        # mask = arcpy.sa.ExtractByMask(out_raster, clip_filepath)
        # mask.save(temp_raster_clip)

        arcpy.Clip_management(out_raster, "#", temp_raster_clip, clip_filepath, "-9999", "ClippingGeometry", "NO_MAINTAIN_EXTENT")
        print("Finished Clip at %.1f secs" % (time.time() - t_start))

        # Leave Commented
        # arcpy.RasterToPolygon_conversion(temp_raster_clip, out_vector, "SIMPLIFY")
        # print("Finished Raster to Polygon at %.1f secs"%(time.time()-t_start))

        arcpy.RasterToPolygon_conversion(temp_raster_clip, temp_vector, "SIMPLIFY")
        print("Finished Raster to Polygon at %.1f secs" % (time.time() - t_start))

        try:
            temp_select = "temp"
            arcpy.MakeFeatureLayer_management('%s' % (temp_vector), temp_select)
            arcpy.SelectLayerByAttribute_management(temp_select, "NEW_SELECTION", query)
            arcpy.CopyFeatures_management(temp_select, out_vector)
            arcpy.SelectLayerByAttribute_management(temp_select, "CLEAR_SELECTION")
        except arcgisscripting.ExecuteError:
            print("Unable to query input file, verify New Map is opened in ArcPro. e.g. Insert --> New Map.")

        print("Finished Select at %.1f secs" % (time.time() - t_start))

        arcpy.Delete_management(temp_raster_clip)
        arcpy.Delete_management(temp_vector)

    def Pickle(self, data, fname):
        import pickle
        outs = pickle.dumps(data, 2)
        open(fname, 'wb').write(outs)

    def ComputePointDensity(self, data, cell_densities=[], weights=None):
        ''' Read Grid and Extract XY Coordinates from Groundings File
        cell_densities will execute the countOccurences function for each item in the cell_densities list and append to the returned list (which starts with histogram)
        weights is a string of the data column to use for weighting the histogram OR an iterable list of weight values to pass to histogram directly
        '''
        fields = ["SHAPE@XY", ]
        if isinstance(weights, str):
            fields.append(weights)
        records = arcpy.da.FeatureClassToNumPyArray(data, fields, spatial_reference=self.sr)  # null_value=0 , explode_to_points=True)
        xys = records["SHAPE@XY"]

        # Generate histogram of point groundings features based on extents of grid raster
        if isinstance(weights, str):
            weighting_values = records[weights]
        else:
            weighting_values = weights
        h_den, xs, ys = numpy.histogram2d(xys[:, 0], xys[:, 1], [self.x_edges, self.y_edges], weights=weighting_values)
        ret = [h_den]
        for cd in cell_densities:
            ret.append(countOccurences(h_den, cd))  # Count of point occurrences within 2 nm of grid cell
        return ret


class datagrid(grid):
    '''Make a numpy array grid that covers an Arc layer'''

    def __init__(self, g, cellSize, buffer_dist=0, dtype=numpy.float64):
        grid.__init__(self, g, cellSize, buffer_dist)
        self.gridarray = None

    @classmethod
    def FromLayer(cls, g, cellSize, buffer_dist=0, dtype=numpy.float64):
        self = cls(g, cellSize, buffer_dist)
        self.gridarray = self.zeros(dtype)
        return self

    @classmethod
    def FromRaster(cls, raster, nodata_to_value, lower_left=None, ncols=None, nrows=None):
        # Note: nodata_to_value is not required because ESRI is not consistent in nodata_to_value default
        if isinstance(raster, str):
            raster = arcpy.Raster(raster)
        self = cls(raster, (raster.meanCellWidth, raster.meanCellHeight))
        if lower_left and not isinstance(lower_left, arcpy.Point):
            lower_left = arcpy.Point(*lower_left)
        g = arcpy.RasterToNumPyArray(raster, lower_left, ncols, nrows, nodata_to_value)
        # arcpy RasterToNumPyArray returns an array of rows,cols - which is backwards from x,y.
        # Also the rows (Ys) are reversed from what we'd expect -- row 0 is the upper limit rather than the lowerleft specified
        # So to correct this -- flip the rows then transpose to make it so indexing the data [i(x)][j(y)] is correct and [0][0] is lowerleft
        self.gridarray = numpy.flipud(g).T
        if lower_left:
            self.minx = lower_left.X
            self.miny = lower_left.Y
        if ncols:
            self.maxx = self.minx + raster.meanCellWidth * ncols
        if nrows:
            self.maxy = self.miny + raster.meanCellHeight * nrows
        return self

    def ExportRaster(self, filename):
        return self.ExportMatchingRaster(self.gridarray, filename, rot=False)

    def Pickle(self, fname):
        grid.Pickle(self, self.gridarray, fname)

    def clip(self, other_grid):
        # Create a copy of the data -- don't clip in place
        new_grid = datagrid(None, [self.cell_size_x, self.cell_size_y])
        # Determine coordinate conversions from s1 and year rasters to main grid
        new_grid.minx = max(other_grid.minx, self.minx)
        new_grid.maxx = min(other_grid.maxx, self.maxx)
        new_grid.miny = max(other_grid.miny, self.miny)
        new_grid.maxy = min(other_grid.maxy, self.maxy)
        new_grid.sr = self.sr
        min_row, min_col = self.ArrayIndicesFromXY([new_grid.minx, new_grid.miny])
        max_row, max_col = self.ArrayIndicesFromXY([new_grid.maxx, new_grid.maxy])
        # Create a copy of the data -- don't clip in place
        new_grid.gridarray = self.gridarray[min_row:max_row + 1, min_col:max_col + 1].copy()
        return new_grid


def randomword(length):
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(length))


class TempLayer(str):
    '''Class to create a temporary random+unique named layer within a database and delete it when finished
    To pass this into arc functions it must be derived from a string OR table or a featureclass
    '''
    def __new__(cls, gdb="in_memory/PydroTmp", suffix="_tmp"):
        rnd = randomword(16)
        name = "Py" + rnd + suffix  # must start with a letter, not number, so put Py in front
        layer_name = os.path.normpath(os.path.join(gdb, name))
        obj = str.__new__(cls, layer_name)
        obj.layer_name = layer_name
        return obj

    def __del__(self):
        # print("deleting")
        arcpy.Delete_management(self.layer_name)


# to save/display in matplotlib from iPython
'''
%matplotlib
import matplotlib
import numpy
import matplotlib.pyplot as plt
import pickle

fname="c:\\"
s=open(fname,'rb').read()
g = pickle.loads(s)
rate = 1.0/5.0 #downsample data in case it's too large -- 20% (1/5th) in this example
g2 = ndimage.interpolation.zoom(g,rate)
fig = plt.figure(figsize=(6, 3.2))
ax = fig.add_subplot(111)
plt.imshow(g2, origin='lower')
'''

circle3 = numpy.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=numpy.uint32)
circle5 = numpy.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=numpy.uint32)
circle7 = numpy.array([[0, 0, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 0, 0]], dtype=numpy.uint32)
circle9 = numpy.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 1, 1, 1, 1, 0],
                       [0, 0, 1, 1, 1, 1, 1, 0, 0],
                       [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=numpy.uint32)
circle17 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint32)
circle19 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint32)

circle31 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint32)

circle39 = numpy.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=numpy.uint32)
