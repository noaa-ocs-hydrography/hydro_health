import arcpy
import aiofiles
import pathlib


class HHLayerTool:
    def get_params(self):
        """Set up the default tool parameters"""
        
        input_directory = arcpy.Parameter(
            displayName="Input Data Directory",
            name="input_directory",
            datatype="DEFolder",
            parameterType="Optional",
            direction="Input"
        )
        output_directory = arcpy.Parameter(
            displayName="Output Data Directory",
            name="output_directory",
            datatype="DEFolder",
            parameterType="Required",
            direction="Input"
        )

        return [
            input_directory,
            output_directory
        ]
    
    def get_param_names(self):
        return [
            'input_directory', 
            'output_directory'
            ]
    