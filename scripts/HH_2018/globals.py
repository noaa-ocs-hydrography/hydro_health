
import os
import re
import types
import arcpy

try:  # python 2.7
    from io import StringIO
    basic_types = ((str,), int)
except:  # python 3.5+
    from io import StringIO
    from configparser import ConfigParser, ExtendedInterpolation
    basic_types = (str, int)
    # wrapping this import as eclipse parser uses python 2 and gets angry about print statement
    exec("from HSTB.ArcExt.base import print")

from HSTB.ArcExt.base import timer, IniFile, BaseToolbox
from HSTB.ArcExt import base
# @TODO temporary during debugging -- will fail in Python2
import importlib
importlib.reload(base)
timer = base.timer
IniFile = base.IniFile
BaseToolbox = base.BaseToolbox
# end @TODO


class RasterName(str):
    def __init__(self, val):
        if len(val) > 13:
            raise Exception("Raster names can not be more than 13 characters")
        super(RasterName, self).__init__(val)


initial_values = IniFile(os.path.join(os.path.dirname(__file__), "NHSP.ini"))  # a global singleton to house the paths to geodatabases etc.
grid_numbers = list(range(1, 14))


class BaseParameters(object):
    r"""
    Each directory potentially has a geodatabase of the same name as the working directory
    and also raster files that with base names of the section.ext and then the short identifiers
    ex: If the Output Root = c:\HH2018
        obj = BaseParameters()
        obj.vector_dir => c:\HH2018\vector
        obj.vector_gdb => c:\HH2018\vector\vector.gdb
        obj.cell_size = 500


    show all the values by printing obj.get_all_values()
    """

    def __init__(self, ini=initial_values):
        self.ini = ini

    @property
    def projection_number(self):
        return int(self.ini["Output"]["output_prj"])

    @property
    def projection_name(self):
        return self.ini["Output"]["output_prj_name"]

    @property
    def input_root(self):
        return self.ini["Input"]["root"]

    @property
    def grid_dir(self):
        return self.ini["Input"]["grid_dir"]

    @property
    def grid_gdb(self):
        return self.ini["Input"]["grid_ws"]

    @property
    def cell_size(self):
        return int(self.ini["Input"]["cell_size"])

    @property
    def output_root(self):
        return self.ini["Output"]["root"]

    @property
    def aux_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["aux"])

    @property
    def aux_gdb(self):
        return os.path.join(self.aux_dir, os.path.basename(self.aux_dir) + ".gdb")

    @property
    def working_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["working"])

    @property
    def vector_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["vector"])

    @property
    def vector_gdb(self):
        return os.path.join(self.vector_dir, os.path.basename(self.vector_dir) + ".gdb")

    @property
    def raw_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["raw"])

    @property
    def raw_gdb(self):
        return os.path.join(self.raw_dir, os.path.basename(self.raw_dir) + ".gdb")

    @property
    def raster_final_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["raster_final"])

    @property
    def raster_final_gdb(self):
        return os.path.join(self.raster_final_dir, os.path.basename(self.raster_final_dir) + ".gdb")

    @property
    def raster_classified_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["raster_classified"])

    @property
    def raster_classified_gdb(self):
        return os.path.join(self.raster_classified_dir, os.path.basename(self.raster_classified_dir) + ".gdb")

    @property
    def raster_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["raster"])

    @property
    def raster_gdb(self):
        return os.path.join(self.raster_dir, os.path.basename(self.raster_dir) + ".gdb")

    @property
    def vector_processed_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["vector_processed"])

    @property
    def vector_processed_gdb(self):
        return os.path.join(self.vector_processed_dir, os.path.basename(self.vector_processed_dir) + ".gdb")

    @property
    def vector_final_dir(self):
        return os.path.join(self.output_root, self.ini["Output"]["vector_final"])

    @property
    def vector_final_gdb(self):
        return os.path.join(self.vector_final_dir, os.path.basename(self.vector_final_dir) + ".gdb")

    @property
    def sde_dir(self):
        return self.ini["Input"]["sde_folder"]

    def get_all_values(self):
        for n in dir(self):
            if "__" not in n:
                v = eval("self.%s" % n)
                if isinstance(v, basic_types):
                    print(n, "=", v)
                else:
                    try:  # filename take two args
                        print(n, '("test", True) =', v("test", True))
                        print(n, '("test", False) =', v("test", False))
                    except:  # everything else takes one
                        try:
                            print(n, '("test") =', v("test"))
                        except:
                            pass


class Parameters(BaseParameters):
    r"""
    Each directory potentially has a geodatabase of the same name as the working directory
    and also files/layers that with base names of the section.ext and then the short identifiers.
    Rasters make files by default while everything else makes gdb layers by default,
    use the gdb argument to switch the behavior
    ex:  for ActiveCaptain (ext = AC) with the Output Root = c:\HH2018
        obj = Parameters("ActiveCaptain")
        obj.vector_dir => c:\HH2018\vector
        obj.vector_gdb => c:\HH2018\vector\vector.gdb
        obj.cell_size = 500
        obj.vector => AC_V
        obj.vector_filename("test") => c:\HH_2018\Vector\Vector.gdb\AC_V_test
        obj.vector_filename("test", gdb=False) => c:\HH_2018\Vector\AC_V_test
        obj.raster_filename("test") => c:\HH_2018\\Raster\\AC_R_test
        obj.raster_filename("test", gdb=True) => c:\HH_2018\\Raster\\Raster.gdb\\AC_R_test

    show all the values by printing obj.get_all_values()
    """

    def __init__(self, section_name, ini=initial_values):
        super().__init__(ini)
        self.section_name = section_name

    def __getitem__(self, val):
        return self.ini[self.section_name][val]

    def __setitem__(self, key, val):
        self.ini[self.section_name][key] = val

    @property
    def section_name(self):
        return self._section_name

    @section_name.setter
    def section_name(self, val):
        self._section_name = val

    @property
    def ext(self):
        return self.ini[self.section_name]["ext"]

    @ext.setter
    def ext(self, val):
        self.ini[self.section_name]["ext"] = val

    @property
    def gdb(self):
        return self.ini[self.section_name]["gdb"]

    @property
    def out_fldr(self):
        return self.ini["Output"]["root"]  # self.ini[self.section_name]["out_fldr"]

    # All the following code could be replaced by a class factory that generates the properties and filename functions.

    @property
    def raw(self):
        return "_".join([self.ini["Extensions"]["raw"], self.ext])

    def raw_filename(self, fname="", gdb=True):
        if gdb:
            path = self.raw_gdb
        else:
            path = self.raw_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.raw + fname)

    @property
    def raster(self):
        return "_".join([self.ini["Extensions"]["raster"], self.ext])

    def raster_filename(self, fname="", gdb=False):
        if gdb:
            path = self.raster_gdb
        else:
            path = self.raster_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.raster + fname)

    @property
    def raster_classified(self):
        return "_".join([self.ini["Extensions"]["raster"] + self.ini["Extensions"]["classified"], self.ext])

    def raster_classified_filename(self, fname="", gdb=False):
        if gdb:
            path = self.raster_classified_gdb
        else:
            path = self.raster_classified_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.raster_classified + fname)

    @property
    def raster_final(self):
        return "_".join([self.ini["Extensions"]["raster"] + self.ini["Extensions"]["final"], self.ext])

    def raster_final_filename(self, fname="", gdb=False):
        if gdb:
            path = self.raster_final_gdb
        else:
            path = self.raster_final_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.raster_final + fname)

    @property
    def vector(self):
        return "_".join([self.ini["Extensions"]["vector"], self.ext])

    def vector_filename(self, fname="", gdb=True):
        if gdb:
            path = self.vector_gdb
        else:
            path = self.vector_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.vector + fname)

    @property
    def vector_processed(self):
        return "_".join([self.ini["Extensions"]["vector"] + self.ini["Extensions"]["processed"], self.ext])

    def vector_processed_filename(self, fname="", gdb=True):
        if gdb:
            path = self.vector_processed_gdb
        else:
            path = self.vector_processed_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.vector_processed + fname)

    @property
    def vector_final(self):
        return "_".join([self.ini["Extensions"]["vector"] + self.ini["Extensions"]["final"], self.ext])

    def vector_final_filename(self, fname="", gdb=True):
        if gdb:
            path = self.vector_final_gdb
        else:
            path = self.vector_final_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.vector_final + fname)

#     @property
#     def prj(self):
#         return "_".join([self.raw, self.projection_name])
#
#     def prj_filename(self, fname, gdb=True):
#         if gdb:
#             path = self.gdb
#         else:
#             path = self.output_root
#         if fname:
#             fname = "_"+fname
#         return os.path.join(path, self.prj + fname)
#
#     @property
#     def final(self):
#         return "_".join([self.ext, self.ini["Extensions"]["final"]])
#
#     def final_filename(self, fname="", gdb=True):
#         if gdb:
#             path = self.gdb
#         else:
#             path = self.output_root
#         if fname:
#             fname = "_"+fname
#         return os.path.join(path, self.final + fname)

    def aux_filename(self, fname="", gdb=True):
        if gdb:
            path = self.aux_gdb
        else:
            path = self.aux_dir
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.ext + fname)

    def aux_rastername(self, fname=""):
        return self.aux_filename(fname, False)

    def working_filename(self, fname="", gdb=True):
        if gdb:
            path = self.working_gdb
        else:
            path = os.path.join(self.working_dir, self.section_name)
        if fname and fname[0] != "_":
            fname = "_" + fname
        return os.path.join(path, self.ext + fname)

    def working_rastername(self, fname=""):
        return self.working_filename(fname, False)

    @property
    def working_gdb(self):
        return os.path.join(self.working_dir, self.section_name, self.gdb)


class BaseNHSPToolbox(BaseToolbox):
    INI_FILE = "INI_FILE"
    INPUT_DIR = "INPUT_ROOT"
    OUTPUT_DIR = "OUTPUT_ROOT"
    GRIDS = "GRID_AREAS"
    # ALL_GRIDS = "ALL_GRIDS"

    def __init__(self, label, desc, toolbox_ini=""):
        if not toolbox_ini:
            toolbox_ini = os.path.join(os.path.dirname(__file__), "hhtoolbox.ini")
        super().__init__(label, desc, toolbox_ini)

    def getParameterInfo(self):
        ini_file = arcpy.Parameter(
            displayName="INI File to use",
            name=self.INI_FILE,
            # datatype="DETextfile",
            datatype="DEFile",
            parameterType="Required",  # "Derived",
            direction="Input")
        ini_file.filter.list = ['ini']
        ini_file.value = os.path.join(os.path.dirname(__file__), "NHSP.ini")

        # Derived Output Features parameter
        in_dir = arcpy.Parameter(
            displayName="Override Input Data Directory",
            name=self.INPUT_DIR,
            datatype="DEFolder",  # "DEFeatureClass",  # "GPFeatureLayer", "DEWorkspace", "DEFolder"
            parameterType="Optional",  # "Derived",
            direction="Input")

        # Derived Output Features parameter
        out_dir = arcpy.Parameter(
            displayName="Override Output Data Directory",
            name=self.OUTPUT_DIR,
            datatype="DEFolder",  # "DEFeatureClass",  # "GPFeatureLayer", "DEWorkspace", "DEFolder"
            parameterType="Optional",  # "Derived",
            direction="Input")

#         # Use the built in select/clear all functionality
#         chkbox = arcpy.Parameter(
#             displayName="Areas to Process",
#             name=self.ALL_GRIDS,
#             datatype="GPBoolean",  # "GPFeatureRecordSetLayer", #"DEFeatureDataset", #"GPLayer", #"DEFeatureClass", #"GPFeatureLayer",
#             parameterType="Optional",
#             direction="Input")
#         parameters[self.ALL_GRIDS] = chkbox
#         self.restore(parameters, self.ALL_GRIDS, bool)

        chklistbox = arcpy.Parameter(
            displayName="Individual Areas to Process",
            name=self.GRIDS,
            datatype="GPString",  # "GPFeatureRecordSetLayer", #"DEFeatureDataset", #"GPLayer", #"DEFeatureClass", #"GPFeatureLayer",
            parameterType="Optional",
            direction="Input",
            multiValue=True)
        chklistbox.filter.list = ["1 - Great Lakes", "2 - East Coast",
                                  "3 - Gulf Coast", "4 - Puerto Rico", "5 - West Coast",
                                  "6 - Alaska", "7 - Marianas/Guam", "8 - Wake Island",
                                  "9 - Hawaii/Johnston Atoll", "10 - Baker Island",
                                  "11 - Palmyra Atoll/Kingman Reef", "12 - American Samoa",
                                  "13 - Jarvis Island", ]

        parameters = [ini_file, in_dir, out_dir, chklistbox]

        self.restore(parameters, self.INI_FILE)
        self.restore(parameters, self.INPUT_DIR)
        self.restore(parameters, self.OUTPUT_DIR)
        self.restore(parameters, self.GRIDS)
        return parameters

    def get_selected_grids(self, parameters):
        parameters = self.parameters_to_ordereddict(parameters)
        grids = []
        if parameters[self.GRIDS].value:
            for s in str(parameters[self.GRIDS].value).split(";"):
                grids.append(int(re.search(r'\d+', s).group()))
        return grids

    def updateParameters(self, parameters):
        pass
#         parameters = self.parameters_to_ordereddict(parameters)
#         raise Exception(str([parameters[self.INI_FILE].altered, not parameters[self.INI_FILE].hasBeenValidated]))
#         if parameters[self.INI_FILE].altered and not parameters[self.INI_FILE].hasBeenValidated:
#             pth, fname = os.path.split(str(parameters[self.INI_FILE].value))
#             ini = IniFile(fname, pth)
#             parameters[self.INPUT_DIR].value = ini["Input"]["root"]
#             parameters[self.OUTPUT_DIR].value = ini["Output"]["root"]

#         if not parameters[self.ALL_GRIDS].hasBeenValidated:
#             if parameters[self.ALL_GRIDS].value:
#                 parameters[self.GRIDS].value = "'" + "';'".join(parameters[self.GRIDS].filter.list) + "'"
#                 parameters[self.ALL_GRIDS].value = False
    def get_modified_ini(self, parameters):
        parameters = self.parameters_to_ordereddict(parameters)
        ini = IniFile(str(parameters[self.INI_FILE].value))
        if parameters[self.INPUT_DIR].value and str(parameters[self.INPUT_DIR].value):  # catch None and ""
            ini["Input"]["root"] = str(parameters[self.INPUT_DIR].value) + "\\"
        if parameters[self.OUTPUT_DIR].value and str(parameters[self.OUTPUT_DIR].value):
            ini["Output"]["root"] = str(parameters[self.OUTPUT_DIR].value) + "\\"
        # print(ini.as_string())
        return ini

    def execute(self, parameters, messages):
        """The source code of the tool."""
        parameters = self.parameters_to_ordereddict(parameters)
        self.store(parameters, self.INI_FILE)
        self.store(parameters, self.INPUT_DIR)
        self.store(parameters, self.OUTPUT_DIR)
        try:
            self.store(parameters, self.GRIDS)
            print("Areas selected:", parameters[self.GRIDS].value)
        except:
            pass  # some tools don't have GRIDS as an option
        return parameters
