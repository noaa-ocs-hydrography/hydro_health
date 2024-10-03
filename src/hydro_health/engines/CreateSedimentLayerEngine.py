import os
import pathlib
import subprocess

from osgeo import ogr, osr, gdal
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


osr.DontUseExceptions()
INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateGroundingsLayerEngine(Engine):
    """Class to hold the logic for processing the Groundings layer"""

    def __init__(self, param_lookup:dict=None):
        super().__init__()
        if param_lookup:
            self.param_lookup = param_lookup
            if self.param_lookup['input_directory'].valueAsText:
                global INPUTS
                INPUTS = pathlib.Path(self.param_lookup['input_directory'].valueAsText)
            if self.param_lookup['output_directoty'].valueAsText:
                global OUTPUTS
                OUTPUTS = pathlib.Path(self.param_lookup['output_directoty'].valueAsText)

    def start(self):
        """Entrypoint for processing Grouundings layer"""
        # TODO prep all the data