# -*- coding: utf-8 -*-

from ags_tools.CreateReefsLayer import CreateReefsLayer


class Toolbox:
    def __init__(self):
        """Define the toolbox (the name of the toolbox is the name of the
        .pyt file)."""
        self.label = "Hydro Health Toolbox"
        self.alias = "Hydro Health Toolbox"

        self.tools = [CreateReefsLayer]
