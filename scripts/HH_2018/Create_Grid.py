# Create Grid

import os
import arcpy
import numpy
import HSTB.ArcExt.NHSP
from HSTB.ArcExt.NHSP import gridclass
# wrapping this import as eclipse parser uses python 2
exec("from HSTB.ArcExt.NHSP.globals import print")


def create_grids(params):
    """input_params is a globals.Parameters object pointed at the desired IniFile
    Function makes a raster and vector representation of
    """
    # output_gdb = r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\GRID\Grid.gdb"
    output_gdb = params.ini["Input"]["grid_ws"]  # r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\GRID\Grid_Testing.gdb"
    files = ["Model_Extents_%02d"%i for i in range(1,14)]

    cellsize = int(params.ini["Input"]["cell_size"])  # 500
    buffer_dist = cellsize * 15  # 7500
    clipper = os.path.join(output_gdb, "USA_Shoreline_NAEAC_Explode_Gen100m_Buffer707m")  # "r"C:\Users\Christina.Fandel\Documents\Fandel\NHSP\GRID\Grid.gdb\USA_Shoreline_NAEAC_Explode_Gen100m_Buffer707m"

    for filename in files:
        try:
            n = gridclass.datagrid.FromLayer(os.path.join(output_gdb, filename), cellsize, buffer_dist, numpy.uint32)
            n.MakeVectorGrid(clipper, output_gdb, "Grid_" + filename)
        except gridclass.GridError as ge:
            print((filename, str(ge)))
