# pyt files can't be as easily imported in python shell -- it can be done but is less natural
# so move the code to a .py file which can then be imported/reloaded in the python console.

# do a reload so that when Arc does a refresh that the code changes come through from the .py file
import importlib
from HSTB.ArcExt.NHSP import hydro_health_layers
importlib.reload(hydro_health_layers)

# import * seems to be the only way for the toolbox to work in the arc catalog
# but leads to the problem where the code changes won't update unless a reload happens above
from HSTB.ArcExt.NHSP.hydro_health_layers import *
