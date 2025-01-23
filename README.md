# Hydro Health

This application contains Python code for the workflows related to
the Hydro Health model.

## Location of Tools
The Hydro Health model current contains an ArcGIS Python Toolbox and associated Python Code

**The Toolbox file:** hydro_health/src/hydro_health/Hydro_Health_Toolbox.pyt

## Tools
1. hydro_health/src/hydro_health/CreateActiveCaptainTool.py
2. hydro_health/src/hydro_health/CreateGroundingsLayerTool.py
3. hydro_health/src/hydro_health/HHLayerTool.py
4. hydro_health/src/hydro_health/RunHydroHealthModelTool.py

### Add Folder Connection example
1. Open ArcGIS Pro
2. Open an existing project or create a new project
3. Access the Catalog Pane: **Click View, then click Catalog Pane**
4. Right click on **Folders**, then click on **Add Folder Connection**
5. Choose a folder that lets you access the tools; **ex: N:\path\to\Code\hydro_health...**
6. In the Catalog Pane, expand Folders and expand the new folder you just added
7. Navigate to the Hydro Health toolbox; **ex: N:\path\to\Code\hydro_health\src\hydro_health\Hydro_Health_Toolbox.pyt**

## Use of Tools
1. Double click on the Hydro_Health_Toolbox.pyt file to open it
2. Double click one of the tools to open the user interface
3. Choose an output folder location or leave blank for Code\hydro_health\outputs to be the default
5. Click run to start the tool <br>
6. Click **View Details** to see log messages that show the status of the tool
7. View the output folder you selected to see the output data
