import arcpy
import pathlib
import time
import os


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
    
    def reset_log_file(self, param_lookup) -> None:
        """Archive the previous log file"""

        output = self.param_lookup['output_directory'].valueAsText
        log_path = pathlib.Path(output) / 'log_prints.txt'
        if log_path.exists():
            now = time.time()
            os.rename(log_path, pathlib.Path(output) / f'log_prints_{now}.txt')
    