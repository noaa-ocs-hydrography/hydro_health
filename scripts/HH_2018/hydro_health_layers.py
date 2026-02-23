import os
import arcpy

try:
    import configparser  # python 2
except:
    import configparser as ConfigParser  # python 3

from HSTB.ArcExt.NHSP import globals
from HSTB.ArcExt.NHSP import gridclass
import importlib
importlib.reload(globals)
importlib.reload(gridclass)
from HSTB.ArcExt.NHSP import active_captain
from HSTB.ArcExt.NHSP import groundings
from HSTB.ArcExt.NHSP import known_hazards
from HSTB.ArcExt.NHSP import reported_error
from HSTB.ArcExt.NHSP import ports
from HSTB.ArcExt.NHSP import sar
from HSTB.ArcExt.NHSP import reef_sanctuary
from HSTB.ArcExt.NHSP import create_test_data

importlib.reload(groundings)
importlib.reload(active_captain)
importlib.reload(known_hazards)
importlib.reload(reported_error)
importlib.reload(ports)
importlib.reload(sar)
importlib.reload(reef_sanctuary)
importlib.reload(create_test_data)
# wrapping this import as eclipse parser uses python 2
exec("from HSTB.ArcExt.NHSP.globals import print")

arcpy.env.overwriteOutput = True


class Toolbox(object):
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Hydro Health"
        self.alias = "Build All Hydro Health Layers"
        # self.stylesheet = "hydro_health_layers.xsl"

        # List of tool classes associated with this toolbox
        self.tools = [BuildAll, BuildTestData,
                      BuildActiveCaptain, BuildGroundings, BuildHaz, BuildRepErr,
                      BuildPorts, BuildReefs, BuildSAR]


class BuildTestData(globals.BaseNHSPToolbox):
    IN_FC = "CLIP_FC"
    ORIG_INI = "ORIG_INI"
    CELL_SZ = "CELL_SZ"

    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create a test NHSP data set", "CreateTestData")
        # self.description = "Using data specified by the 'INI File to use' the tool copies necessary rasters and vectors to the 'Input Data Directory' to be used by the other process tools"
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

    def getParameterInfo(self):
        """Define parameter definitions"""
        parameters = super().getParameterInfo()  # gets the main parameters for ini, directory overrides
        parameters = self.parameters_to_ordereddict(parameters)

#         ini_file = arcpy.Parameter(
#             displayName="ORIGINAL RAW INI File to use",
#             name=self.ORIG_INI,
#             datatype="DEFile",
#             parameterType="Required",  # "Derived",
#             direction="Input")
#         ini_file.filter.list = ['ini']
#         ini_file.value = os.path.join(os.path.dirname(__file__), "NHSP.ini")

        fs = arcpy.Parameter(
            displayName="Polygon to create test data within -- rectangles preferred",
            name=self.IN_FC,
            datatype="DEFeatureClass",  # "GPFeatureRecordSetLayer", #"DEFeatureDataset", #"GPLayer", #"DEFeatureClass", #"GPFeatureLayer",
            parameterType="Optional",
            direction="Input"
        )

        cell_sz = arcpy.Parameter(
            displayName="Cell size in meters (500 was original size)",
            name=self.CELL_SZ,
            datatype="GPLong",  # "GPFeatureRecordSetLayer", #"DEFeatureDataset", #"GPLayer", #"DEFeatureClass", #"GPFeatureLayer",
            parameterType="Required",
            direction="Input"
        )
        cell_sz.value = 500

        parameters[self.IN_FC] = fs
        parameters[self.CELL_SZ] = cell_sz
        self.restore(parameters, self.IN_FC)
        self.restore(parameters, self.CELL_SZ, int)
#        self.restore(parameters, self.ORIG_INI)

        return list(parameters.values())

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = self.parameters_to_ordereddict(parameters)
        if not parameters[self.INPUT_DIR].value or not str(parameters[self.INPUT_DIR].value):
            raise Exception("Can't create test data in the same location as pointed to by the ini file")
        print((parameters[self.IN_FC].value))
        parameters = super().execute(parameters, messages)
        self.store(parameters, self.IN_FC)
        self.store(parameters, self.CELL_SZ)
#        self.store(parameters, self.ORIG_INI)
        self.ini.save()

        clipping_polygon = [row[0] for row in arcpy.da.SearchCursor(parameters[self.IN_FC].value, ["SHAPE@", ])][0]
        # raise Exception(str(clipping_polygon.extent))
        create_test_data.create_data(str(parameters[self.INI_FILE].value), str(parameters[self.INPUT_DIR].value) + "\\", clipping_polygon)
# C:\Data\ArcGIS\NHSP_TestData
# C:\Data\ArcGIS\NHSP_TestData\HH_2018


class BuildActiveCaptain(globals.BaseNHSPToolbox):
    FAKE = "FAKE_INPUT"

    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create the Active Captain layer", "ActiveCaptain")
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

#     def getParameterInfo(self):
#         """Define parameter definitions"""
#         parameters = super().getParameterInfo()  # gets the main parameters for ini, directory overrides
#         parameters = self.parameters_to_ordereddict(parameters)
#
#         fs = arcpy.Parameter(
#             displayName="Bogus string for Active Captain -- testing interface",
#             name=self.FAKE,
#             datatype="GPString",  # "GPFeatureRecordSetLayer", #"DEFeatureDataset", #"GPLayer", #"DEFeatureClass", #"GPFeatureLayer",
#             parameterType="Optional",
#             direction="Input"
#         )
#         parameters[self.FAKE] = fs
#         self.restore(parameters, self.FAKE)
#
#         return list(parameters.values())

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = super().execute(parameters, messages)
        self.ini.save()

        ini = self.get_modified_ini(parameters)

        grids = self.get_selected_grids(parameters)
        if grids:
            active_captain.execute(ini, grids)
        else:
            print("No grids selected, no processing to be done")


class BuildGroundings(globals.BaseNHSPToolbox):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create the Groundings layer", "Groundings")
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = super().execute(parameters, messages)
        self.ini.save()

        ini = self.get_modified_ini(parameters)

        grids = self.get_selected_grids(parameters)
        if grids:
            groundings.execute(ini, grids)
        else:
            print("No grids selected, no processing to be done")


class BuildHaz(globals.BaseNHSPToolbox):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create the Nav Hazards layer", "NavHazards")
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = super().execute(parameters, messages)
        self.ini.save()

        ini = self.get_modified_ini(parameters)

        grids = self.get_selected_grids(parameters)
        if grids:
            known_hazards.execute(ini, grids)
        else:
            print("No grids selected, no processing to be done")


class BuildRepErr(globals.BaseNHSPToolbox):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create the Reported Error layer", "RepErr")
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = super().execute(parameters, messages)
        self.ini.save()

        ini = self.get_modified_ini(parameters)

        grids = self.get_selected_grids(parameters)
        if grids:
            reported_error.execute(ini, grids)
        else:
            print("No grids selected, no processing to be done")


class BuildPorts(globals.BaseNHSPToolbox):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create the Ports layer", "Ports")
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

    def getParameterInfo(self):
        """Define parameter definitions"""
        parameters = super().getParameterInfo()  # gets the main parameters for ini, directory overrides
        parameters = self.parameters_to_ordereddict(parameters)
        for k, v in list(parameters.items()):
            if v.name == self.GRIDS:
                parameters.pop(k)

        return list(parameters.values())

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = super().execute(parameters, messages)
        self.ini.save()

        ini = self.get_modified_ini(parameters)

        ports.execute(ini)


class BuildReefs(globals.BaseNHSPToolbox):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create the Reefs layer", "Reefs")
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

    def getParameterInfo(self):
        """Define parameter definitions"""
        parameters = super().getParameterInfo()  # gets the main parameters for ini, directory overrides
        parameters = self.parameters_to_ordereddict(parameters)
        for k, v in list(parameters.items()):
            if v.name == self.GRIDS:
                parameters.pop(k)

        return list(parameters.values())

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = super().execute(parameters, messages)
        self.ini.save()

        ini = self.get_modified_ini(parameters)

        reef_sanctuary.execute(ini)


class BuildSAR(globals.BaseNHSPToolbox):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create the Search and Rescue layer", "SAR")
        # self.stylesheet = os.path.join(os.path.dirname(__file__), "hydro_health_layers.xsl")

    def getParameterInfo(self):
        """Define parameter definitions"""
        parameters = super().getParameterInfo()  # gets the main parameters for ini, directory overrides
        parameters = self.parameters_to_ordereddict(parameters)
        for k, v in list(parameters.items()):
            if v.name == self.GRIDS:
                parameters.pop(k)

        return list(parameters.values())

    def execute(self, parameters, messages):
        """The source code of the tool."""
        # ActiveCaptain_RawToRasterFinal.execute(self, parameters, messages)
        parameters = super().execute(parameters, messages)
        self.ini.save()

        ini = self.get_modified_ini(parameters)

        sar.execute(ini)


class BuildAll(globals.BaseNHSPToolbox):
    ACTIVE_CAPTAIN = "ACTIVE_CAPTAIN"
    GROUNDINGS = "GROUNDINGS"

    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        super().__init__("Create All Hydro Health Layers", "Hydro Health")

    @property
    def sub_tools(self):
        subs = []
        for tool in Toolbox().tools:
            if tool != BuildAll:
                d = tool().description
                subs.append((tool, d, "Process " + d))
        return subs
        # return ((BuildActiveCaptain, self.ACTIVE_CAPTAIN, "Process Active Captain"),
        #        (BuildGroundings, self.GROUNDINGS, "Process Groundings"),
        #        )

    def getParameterInfo(self):
        """Define parameter definitions"""
        parameters = super().getParameterInfo()  # gets the main parameters for ini, directory overrides
        for cls, name, displayname in self.sub_tools:
            chkbox = arcpy.Parameter(
                displayName=displayname,
                name=name,
                datatype="GPBoolean",  # "GPFeatureRecordSetLayer", #"DEFeatureDataset", #"GPLayer", #"DEFeatureClass", #"GPFeatureLayer",
                parameterType="Optional",
                direction="Input")
            parameters.insert(-1, chkbox)  # do this so the other tools come before the areas list
            self.restore(parameters, name, bool)

        # now add any extra parameters the other tools want individually
        # this way if the same tool is used by multiple sub-tools it will only show up once.
        parameters = self.parameters_to_ordereddict(parameters)

        for cls, name, displayname in self.sub_tools:
            tool = cls()
            more_params = tool.getParameterInfo()
            for p in more_params:
                if p.name not in parameters:
                    parameters[p.name] = p

        return list(parameters.values())

    def updateParameters(self, parameters):
        super().updateParameters(parameters)
        parameters = self.parameters_to_ordereddict(parameters)
        enabled = {}
        disabled = {}
        # enable all the base parameters
        more_params = super().getParameterInfo()
        for p in more_params:
            enabled[p.name] = None
        for cls, name, displayname in self.sub_tools:
            # if the tool is checked then enable all its parameters
            # otherwise list as postentially disabling
            if parameters[name].value:
                extend_dict = enabled
            else:
                extend_dict = disabled

            tool = cls()
            more_params = tool.getParameterInfo()
            for p in more_params:
                extend_dict[p.name] = None
        # make sure everything is enabled that will need to be
        for k in list(enabled.keys()):
            parameters[k].enabled = True
        # make sure the paramter didn't get used in more than one tool and needs to be on
        for k in list(disabled.keys()):
            if k not in enabled:
                parameters[k].enabled = False

    def execute(self, parameters, messages):
        """The source code of the tool."""
        parameters = super().execute(parameters, messages)
        for cls, name, displayname in self.sub_tools:
            self.store(parameters, name)
#        self.store(parameters, self.GROUNDINGS)
        self.ini.save()

        # find an unused temporary name to place polygons in for clipping.
        n = 1
        tmp_name = "in_memory/PydroTmp"
        while arcpy.Exists(tmp_name + str(n)):
            n += 1
        tmp_name += str(n)

        for cls, name, displayname in self.sub_tools:
            if parameters[name].value:
                tool = cls()
                print("")
                print(("v" * 30))
                print(("Running: ", displayname))
                tool.execute(parameters, messages)
                print(("^" * 30))
                print("")

        return parameters
