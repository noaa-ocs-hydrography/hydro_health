# -*- coding: utf-8 -*-

import pathlib
HH_MODULE = pathlib.Path(__file__).parents[1]
if HH_MODULE.name != 'src':
    raise ImportError('Hydro_Health_Toolbox.pyt must reside in "src/hydro_health" folder location!')

import sys
sys.path.append(str(HH_MODULE))

from hydro_health.ags_tools.CreateReefsLayerTool import CreateReefsLayerTool
from hydro_health.ags_tools.CreateActiveCaptainLayerTool import CreateActiveCaptainLayerTool
from hydro_health.ags_tools.CreateGroundingsLayerTool import CreateGroundingsLayerTool
from hydro_health.ags_tools.RunHydroHealthModelTool import RunHydroHealthModelTool


class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Hydro Health Toolbox"
        self.alias = "Hydro Health Toolbox"

        self.tools = [
            CreateReefsLayerTool,
            CreateActiveCaptainLayerTool,
            CreateGroundingsLayerTool,
            RunHydroHealthModelTool
        ]
