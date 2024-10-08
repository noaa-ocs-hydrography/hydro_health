# -*- coding: utf-8 -*-

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
