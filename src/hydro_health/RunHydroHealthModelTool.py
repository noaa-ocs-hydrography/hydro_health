import arcpy
import os
import pathlib
import geopandas as gpd

from hydro_health.HHLayerTool import HHLayerTool
from hydro_health.helpers import tools


class RunHydroHealthModelTool(HHLayerTool):
    def __init__(self):
        """Define the tool (tool name is the name of the class)."""
        self.label = "Run the Hydro Health Model"
        self.description = ""
        self.param_lookup = {}

    def getParameterInfo(self):
        """Define the tool parameters."""
        params = self.get_params()
        return params

    def isLicensed(self):
        """Set whether the tool is licensed to execute."""
        return True

    def updateParameters(self, parameters):
        """Modify the values and properties of parameters before internal
        validation is performed.  This method is called whenever a parameter
        has been changed."""
        return 
       
    def updateMessages(self, parameters):
        """Modify the messages created by internal validation for each tool
        parameter. This method is called after internal validation."""

        eco_regions, tile_selector = parameters[2:4]
        if not tile_selector.value and not eco_regions.value:
            tile_selector.setErrorMessage('Select Shapefile or Draw a Polygon')
            eco_regions.setErrorMessage('Choose Eco Region(s)')

    def execute(self, parameters, messages):
        """The source code of the tool."""

        param_lookup = self.setup_param_lookup(parameters)

        if not param_lookup['tile_selector'].value and not param_lookup['eco_regions'].value:
            # Force choice of tile_selector or eco_regions
            return

        if param_lookup['tile_selector'].value:
            self.convert_tile_selector(param_lookup)

        arcpy.AddMessage(f'"""\n\nDownloading of BlueTopo tiles and Digital Coast data\nuses parallel processing for speed.\n')
        arcpy.AddMessage(f'Additional log messages are written here: {param_lookup["output_directory"].valueAsText}\\log_prints.txt\n\n"""')

        tiles = tools.get_ecoregion_tiles(param_lookup)
        arcpy.AddMessage(f'Selected tiles: {tiles.shape[0]}')
        
        self.reset_log_file(param_lookup)
        self.download_bluetopo_tiles(tiles)
        self.download_digital_coast_tiles(tiles)
        self.create_raster_masks()
        self.grid_vrt_files()
        arcpy.AddMessage('Done')

    def postExecute(self, parameters):
        """This method takes place after outputs are processed and
        added to the display."""
        return

    # Custom Python code ##############################
    def convert_tile_selector(self, param_lookup: dict) -> None:
        """Convert any drawn features and store them on the hidden parameter 'drawn_polygon'"""

        output_json = os.path.join(param_lookup['output_directory'].valueAsText, 'drawn_polygons.geojson')
        arcpy.conversion.FeaturesToJSON(
            param_lookup['tile_selector'].value,
            output_json,
            geoJSON="GEOJSON",
            outputToWGS84="WGS84",
        )
        param_lookup['drawn_polygon'].value = output_json

    def create_raster_masks(self) -> None:
        """Create prediction and training masks for all ecoregions"""

        arcpy.AddMessage(f'Creating prediction and training masks for ecoregions')
        tools.process_create_masks(self.param_lookup['output_directory'].valueAsText)

    def download_bluetopo_tiles(self, tiles: gpd.GeoDataFrame) -> None:
        """Download all bluetopo tiles"""

        tools.process_bluetopo_tiles(tiles, self.param_lookup['output_directory'].valueAsText)
        ecoregions = tools.get_ecoregion_folders(self.param_lookup)
        for ecoregion in ecoregions:
            arcpy.AddMessage(f" - {ecoregion} BlueTopo tiles: {len(next(os.walk(pathlib.Path(self.param_lookup['output_directory'].valueAsText) / ecoregion / 'BlueTopo'))[1])}")

        arcpy.AddMessage('Tile process completed')
        for ecoregion in ecoregions:
            for dataset in ['elevation', 'slope', 'rugosity', 'uncertainty']:
                arcpy.AddMessage(f'Building {ecoregion} - {dataset} VRT file')
                tools.create_raster_vrts(self.param_lookup['output_directory'].valueAsText, dataset, ecoregion, 'BlueTopo')

    def download_digital_coast_tiles(self, tiles: gpd.GeoDataFrame) -> None:
        """Download all digital coast tiles"""

        arcpy.AddMessage('Obtaining Digital Coast data for selected area')
        tools.process_digital_coast_files(tiles, self.param_lookup['output_directory'].valueAsText)
        for ecoregion in tools.get_ecoregion_folders(self.param_lookup):
            arcpy.AddMessage(f'Building {ecoregion} - Digital Coast VRT file')
            tools.create_raster_vrts(self.param_lookup['output_directory'].valueAsText, 'NCMP', ecoregion, 'DigitalCoast')
            digital_coast_folder = os.path.join(self.param_lookup['output_directory'].valueAsText, ecoregion, 'DigitalCoast')
            if os.path.exists(digital_coast_folder):
                files = len(next(os.walk(digital_coast_folder))[1])
                if files > 0:
                    arcpy.AddMessage(f" - {ecoregion} Digital Coast files: {files}")
                else:
                    arcpy.AddMessage(' - No Digital Coast files found')
            else:
                arcpy.AddMessage(' - No Digital Coast files found')
        
    def get_params(self):
        """
        Set up the tool parameters
        - Default: Input Directory, Output Directory
        - Override to add additional parameters or append to base method
        """

        params = super().get_params()

        # coastal_states = arcpy.Parameter(
        #     displayName="Pick coastal state(s) to run model",
        #     name="coastal_states",
        #     datatype="GPString",
        #     parameterType="Required",
        #     direction="Input",
        #     multiValue=True
        # )
        # coastal_states.filter.type = "ValueList"
        # coastal_states.filter.list = [
        #         "California",
        #         "Connecticut",
        #         "Delaware",
        #         "Florida",
        #         "Georgia",
        #         "Louisiana",
        #         "Maine",
        #         "Maryland",
        #         "Massachusetts",
        #         "New Hampshire",
        #         "New Jersey",
        #         "New York",
        #         "North Carolina",
        #         "Oregon",
        #         "Rhode Island",
        #         "South Carolina",
        #         "Virginia",
        #         "Washington"
        # ]
        # params.append(coastal_states)

        eco_regions = arcpy.Parameter(
            displayName="Pick eco region(s) to run model",
            name="eco_regions",
            datatype="GPString",
            parameterType="Optional",
            direction="Input",
            multiValue=True
        )
        eco_regions.filter.type = "ValueList"
        eco_regions.filter.list = [
            'ER_1-Texas',
            'ER_2-Louisiana',
            'ER_3-Florida-West',
            'ER_4-Florida-East',
            'ER_5-Mid-Atlantic',
            'ER_6-Upper-Atlantic'
        ]
        params.append(eco_regions)

        tile_selector = arcpy.Parameter(
            displayName="Draw a polygon or select a feature layer",
            name="tile_selector",
            datatype="GPFeatureRecordSetLayer",
            parameterType="Optional",
            direction="Input"
        )
        tile_selector.filter.list = ["Polygon"]
        params.append(tile_selector)
        drawn_polygon = arcpy.Parameter(
            displayName="Drawn polygon output geojson",
            name="drawn_polygon",
            datatype="GPString",
            parameterType="Derived",
            direction="Output"
        )
        params.append(drawn_polygon)

        # slider_id = '{C8C46E43-3D27-4485-9B38-A49F3AC588D9}'
        # slider_range = [1, 10]

        # reef_slider = arcpy.Parameter(
        #     displayName="Reefs - Weighting",
        #     name="reefs_weighting",
        #     datatype="Long",
        #     parameterType="Required",
        #     direction="Input"
        # )

        # reef_slider.controlCLSID = slider_id
        # reef_slider.value = 7
        # reef_slider.filter.type = "Range"
        # reef_slider.filter.list = slider_range
        # params.append(reef_slider)

        # active_captain_slider = arcpy.Parameter(
        #     displayName="Active Captain - Weighting",
        #     name="active_captain_weighting",
        #     datatype="Long",
        #     parameterType="Required",
        #     direction="Input"
        # )
        # active_captain_slider.controlCLSID = slider_id
        # active_captain_slider.value = 5
        # active_captain_slider.filter.type = "Range"
        # active_captain_slider.filter.list = slider_range
        # params.append(active_captain_slider)

        # groundings_slider = arcpy.Parameter(
        #     displayName="Groundings - Weighting",
        #     name="groundings_weighting",
        #     datatype="Long",
        #     parameterType="Required",
        #     direction="Input"
        # )
        # groundings_slider.controlCLSID = slider_id
        # groundings_slider.value = 5
        # groundings_slider.filter.type = "Range"
        # groundings_slider.filter.list = slider_range
        # params.append(groundings_slider)

        return params

    @property
    def parameters(self):
        """Get a list of all parameter names"""

        return list(self.parameter_lookup.keys())

    def get_parameter(self, param: str):
        """Return a single parameter by key"""

        parameter = self.parameter_lookup.get(param)
        return parameter
    
    def grid_vrt_files(self):
        """Clip DigitalCoast VRT files to BlueTopo grid"""

        arcpy.AddMessage('Gridding VRT file to Blue Topo tiles')
        tools.grid_vrt_files(self.param_lookup['output_directory'].valueAsText, 'DigitalCoast')

    def setup_param_lookup(self, params: list[str]):
        """Build key/value lookup for parameters"""

        param_names = super().get_param_names()
        # param_names.append('coastal_states')
        param_names.append('eco_regions')
        param_names.append('tile_selector')
        param_names.append('drawn_polygon')

        lookup = {}
        for name, param in zip(param_names, params):
            lookup[name] = param
        self.param_lookup = lookup
        return lookup
