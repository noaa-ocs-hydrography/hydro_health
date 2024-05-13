import arcpy
from hydro_health.ags_tools.HHLayer import HHLayer
from hydro_health.engines.CreateGroundingsLayerEngine import CreateGroundingsLayerEngine


class CreateGroundingsLayer(HHLayer):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Create the Active Captain Layer"
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
        engine = CreateGroundingsLayerEngine(param_lookup)
        engine.start()
        return
        

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

        return super().get_params()
    
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
        

    
    
