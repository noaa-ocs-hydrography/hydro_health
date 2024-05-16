import arcpy
from hydro_health.ags_tools.HHLayer import HHLayer
from hydro_health.engines.CreateReefsLayerEngine import CreateReefsLayerEngine
from hydro_health.engines.CreateActiveCaptainLayerEngine import CreateActiveCaptainLayerEngine
from hydro_health.engines.CreateGroundingsLayerEngine import CreateGroundingsLayerEngine


class RunHydroHealthModel(HHLayer):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Run the Hydro Health Model"
        self.description = ""
        self.param_lookup = {}

    def getParameterInfo(self):
        """Define the tool parameters."""
        params = self.get_params()
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return

    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""
        return

    def execute(self, parameters, messages):
        """The source code of the tool."""

        param_lookup = self.setup_param_lookup(parameters)
        reefs = CreateReefsLayerEngine(param_lookup)
        reefs.start()
        active_captain = CreateActiveCaptainLayerEngine(param_lookup)
        active_captain.start()
        groundings = CreateGroundingsLayerEngine(param_lookup)
        groundings.start()

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

    # Custom Python code ##############################
    def get_params(self):
        """
        Set up the tool parameters
        - Default: Input Directory, Output Directory
        - Override to add additional parameters or append to base method
        """

        params = super().get_params()

        slider_id = '{C8C46E43-3D27-4485-9B38-A49F3AC588D9}'
        slider_range = [1, 10]

        reef_slider = arcpy.Parameter(
            displayName="Reefs - Weighting",
            name="reefs_weighting",
            datatype="Long",
            parameterType="Required",
            direction="Input"
        )

        reef_slider.controlCLSID = slider_id
        reef_slider.value = 7
        reef_slider.filter.type = "Range"
        reef_slider.filter.list = slider_range
        params.append(reef_slider)
        
        active_captain_slider = arcpy.Parameter(
            displayName="Active Captain - Weighting",
            name="active_captain_weighting",
            datatype="Long",
            parameterType="Required",
            direction="Input"
        )
        active_captain_slider.controlCLSID = slider_id
        active_captain_slider.value = 5
        active_captain_slider.filter.type = "Range"
        active_captain_slider.filter.list = slider_range
        params.append(active_captain_slider)

        groundings_slider = arcpy.Parameter(
            displayName="Groundings - Weighting",
            name="groundings_weighting",
            datatype="Long",
            parameterType="Required",
            direction="Input"
        )
        groundings_slider.controlCLSID = slider_id
        groundings_slider.value = 5
        groundings_slider.filter.type = "Range"
        groundings_slider.filter.list = slider_range
        params.append(groundings_slider)

        return params
    
    @property
    def parameters(self):
        """Get a list of all parameter names"""

        return list(self.parameter_lookup.keys())
    
    def get_parameter(self, param):
        """Return a single parameter by key"""

        parameter = self.parameter_lookup.get(param)
        return parameter
    
    def setup_param_lookup(self, params):
        """Build key/value lookup for parameters"""

        param_names = super().get_param_names()

        lookup = {}
        for name, param in zip(param_names, params):
            lookup[name] = param
        self.param_lookup = lookup
        return lookup
        

    
    
