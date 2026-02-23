# Color Band Percent Segmentation
# This script returns break points, segmented at the user-specified number of desired bands. For example, it the user would like 10 color bands (e.g. 10% segmentations),
# the output will be a list of raster values that denote 10% increments.

# Segment by Percentiles
import arcpy
import numpy
from HSTB.ArcExt.NHSP.globals import Parameters

# User Input Variable
param_str = "Risk"
num_bands = 10  # Number of color bands for raster

r = Parameters(param_str)
tol_val = 10**-9  # Make this the max precision val
r_rf = r.raster_final_filename()

r_a = arcpy.RasterToNumPyArray(r_rf)
r_nonan = r_a[r_a > 0]

a = numpy.arange(0, num_bands + 1, 1.0) * (100 / num_bands)
a[-1] = 100.0
r_nonan_div = numpy.percentile(r_nonan, a)

# Unit Test
# r_nonan = numpy.array([1, 1, 2, 2, 2, 4, 5, 6])
# r_nonan_div = [1, 2, 2, 2, 2, 5, 6]

print(r_nonan_div)
r_area = []
r_p_area = len(r_nonan[(r_nonan < r_nonan_div[1])])
r_area.append(r_p_area)
for p_val in range(1, len(r_nonan_div) - 2):
    if r_nonan_div[p_val - 1] == r_nonan_div[p_val + 1]:
        continue
    p_start = r_nonan_div[p_val]
    p_end = r_nonan_div[p_val + 1]
    if r_nonan_div[p_val - 1] == r_nonan_div[p_val]:  # Assumes raster is comprised of more than one value
        r_p_area = len(r_nonan[(r_nonan > p_start) & (r_nonan < p_end)])
    elif r_nonan_div[p_val + 1] == r_nonan_div[p_val]:  # Assumes raster is comprised of more than one value
        r_p_area = len(r_nonan[(r_nonan == p_start)])
    else:
        r_p_area = (len(r_nonan[(r_nonan >= p_start) & (r_nonan < p_end)]))
    r_area.append(r_p_area)
r_p_area = len(r_nonan[(r_nonan >= r_nonan_div[-2])])
r_area.append(r_p_area)
print(r_area)
r_area = numpy.array(r_area, dtype=numpy.float64) * (r.cell_size**2)
print(r_area)

# Unit Test Return
# [2, 3, 1, 2]
# [ 500000.  750000.  250000.  500000.]
