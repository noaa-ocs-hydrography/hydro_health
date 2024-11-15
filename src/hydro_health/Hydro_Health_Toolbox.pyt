# -*- coding: utf-8 -*-

import pathlib
HH_MODULE = pathlib.Path(__file__).parents[1]
if HH_MODULE.name != 'src':
    raise ImportError('Hydro_Health_Toolbox.pyt must reside in "src/hydro_health" folder location!')

import sys
sys.path.append(str(HH_MODULE))

from hydro_health.ags_tools.CreateReefsLayer import CreateReefsLayer
from hydro_health.ags_tools.CreateActiveCaptainLayer import CreateActiveCaptainLayer
from hydro_health.ags_tools.CreateGroundingsLayer import CreateGroundingsLayer
from hydro_health.ags_tools.RunHydroHealthModel import RunHydroHealthModel


class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Hydro Health Toolbox"
        self.alias = "Hydro Health Toolbox"

        self.tools = [
            CreateReefsLayer,
            CreateActiveCaptainLayer,
            CreateGroundingsLayer,
            RunHydroHealthModel
        ]
