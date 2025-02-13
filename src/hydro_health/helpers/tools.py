import yaml
import pathlib
import geopandas as gpd  # pip install geopandas;requires numpy==1.22.4 and activating cloned env in Pro

from hydro_health.engines.tiling.TileProcessor import TileProcessor


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class Param:
        def __init__(self, path):
            self.value = path

        @property
        def valueAsText(self):
            return self.value


def get_config_item(parent: str, child: str=False) -> tuple[str, int]:
    """Load config and return speciific key"""

    with open(str(INPUTS / 'lookups' / 'config.yaml'), 'r') as lookup:
        config = yaml.safe_load(lookup)
        parent_item = config[parent]
        if child:
            return parent_item[child]
        else:
            return parent_item
        

def get_state_tiles(param_lookup: dict[str]) -> gpd.GeoDataFrame:
    """Obtain a subset of tiles based on state names"""

    geopackage = INPUTS / get_config_item('SHARED', 'DATABASE')

    all_states = gpd.read_file(geopackage, layer=get_config_item('SHARED', 'STATES'), columns=['STATE_NAME'])
    coastal_states = param_lookup['coastal_states'].valueAsText.replace("'", "").split(';')
    selected_states = all_states[all_states['STATE_NAME'].isin(coastal_states)]

    all_tiles = gpd.read_file(geopackage, layer=get_config_item('SHARED', 'TILES'), columns=[get_config_item('SHARED', 'TILENAME')], mask=selected_states)
    state_tiles = all_tiles.sjoin(selected_states)  # needed to keep STATE_NAME
    state_tiles = state_tiles.drop(['index_right'], axis=1)

    coastal_boundary = gpd.read_file(geopackage, layer=get_config_item('SHARED', 'BOUNDARY'))
    tiles = state_tiles.sjoin(coastal_boundary)
    # tiles.to_file(OUTPUTS / 'state_tiles.shp', driver='ESRI Shapefile')

    return tiles


def get_ecoregion_tiles(param_lookup: dict[str]) -> gpd.GeoDataFrame:
    """Obtain a subset of tiles based on selected eco regions"""

    geopackage = INPUTS / get_config_item('SHARED', 'DATABASE')
    # get eco region from shapefile that matches drop down choices
    all_ecoregions = gpd.read_file(geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), columns=['EcoRegion'])
    eco_regions = param_lookup['eco_regions'].valueAsText.replace("'", "").split(';')
    eco_regions = [region.split('-')[0] for region in eco_regions]
    selected_regions = all_ecoregions[all_ecoregions['EcoRegion'].isin(eco_regions)]  # select eco_region polygons
    tiles = gpd.read_file(geopackage, layer=get_config_item('SHARED', 'TILES'), columns=[get_config_item('SHARED', 'TILENAME')], mask=selected_regions)
    tiles.to_file(OUTPUTS / 'selected_tiles.shp')  
    return tiles


def process_tiles(tiles: gpd.GeoDataFrame, outputs:str = False) -> None:
    # get environment (dev, prod)
    # if dev, use multiprocessing
    # if prod, send to API endpoint of listeners in kubernetes
        # pickle each tuple of engine and tile
        # unpickle the object
        # call the class method with the tile argument
        # log success of each call
        # notify the main caller of completion?!

    output_folder = OUTPUTS if not outputs else outputs
    processor = TileProcessor()
    processor.process(tiles, output_folder)
    