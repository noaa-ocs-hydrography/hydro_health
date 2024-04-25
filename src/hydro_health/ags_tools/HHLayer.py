import arcpy


class HHLayer:
    def get_params(self):
        """Set up the default tool parameters"""
        
        input_directory = arcpy.Parameter(
            displayName="Input Data Directory",
            name="input_directory",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        )
        output_directoty = arcpy.Parameter(
            displayName="Output Data Directory",
            name="output_directoty",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        )

        return [
            input_directory,
            output_directoty
        ]
    
    def get_param_names(self):
        return [
            'input_directory', 
            'output_directoty'
            ]