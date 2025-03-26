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
    
    async def read_async_log(self, log_file_path: pathlib.Path):
        async with aiofiles.open(log_file_path, mode='r') as log_file:
            async for line in log_file:
                arcpy.AddMessage(line.strip())

    async def stream_log(self, output_folder: str):
        log_file_path = pathlib.Path(output_folder) / 'log_prints.txt'
        # Delete log file on startup
        if log_file_path.exists():
            log_file_path.unlink()
        open(log_file_path, 'a').close()
        await self.read_async_log(log_file_path)