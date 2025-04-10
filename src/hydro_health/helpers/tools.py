import yaml
import pathlib
import geopandas as gpd  # pip install geopandas;requires numpy==1.22.4 and activating cloned env in Pro

from hydro_health.engines.tiling.BlueTopoProcessor import BlueTopoProcessor
from hydro_health.engines.tiling.DigitalCoastProcessor import DigitalCoastProcessor
from hydro_health.engines.tiling.ModelDataProcessor import ModelDataProcessor

from osgeo import gdal, osr

gdal.UseExceptions()

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class Param:
        def __init__(self, path):
            self.value = path

        @property
        def valueAsText(self):
            return self.value


def create_raster_vrt(output_folder: str, file_type: str, data_type: str) -> None:
    """Create an output VRT from found .tif files"""

    glob_lookup = {
        'elevation': '*[0-9].tiff',
        'slope': '*_slope.tiff',
        'rugosity': '*_rugosity.tiff',
        'NCMP': '*.tif'
    }

    outputs = pathlib.Path(output_folder) / data_type if output_folder else OUTPUTS / data_type
    geotiffs = list(outputs.rglob(glob_lookup[file_type]))

    output_geotiffs = {}
    for geotiff in geotiffs:
        geotiff_ds = gdal.Open(geotiff)
        projection_wkt = geotiff_ds.GetProjection()
        spatial_ref = osr.SpatialReference(wkt=projection_wkt)  
        projected_crs_string = spatial_ref.GetAuthorityCode('DATUM')
        clean_crs_string = projected_crs_string.replace('/', '').replace(' ', '_')
        provider_folder = geotiff.relative_to(outputs).parents[-2]
        # Handle BlueTopo and DigitalCoast differently
        clean_crs_key = f'{clean_crs_string}_{provider_folder}' if data_type == 'DigitalCoast' else clean_crs_string
        # Store tile and CRS
        if clean_crs_key not in output_geotiffs:
            output_geotiffs[clean_crs_key] = {'crs': None, 'tiles': []}
        output_geotiffs[clean_crs_key]['tiles'].append(geotiff)
        if output_geotiffs[clean_crs_key]['crs'] is None:
            output_geotiffs[clean_crs_key]['crs'] = spatial_ref
        geotiff_ds = None

    for crs, tile_dict in output_geotiffs.items():
        # Create VRT for each tile and set output CRS to fix heterogenous crs issue
        vrt_tiles = []
        for tile in tile_dict['tiles']:
            output_raster_vrt = str(tile.parents[0] / f"{tile.stem}.vrt")
            gdal.Warp(
                output_raster_vrt, 
                tile,
                format="VRT",
                dstSRS=output_geotiffs[crs]['crs']
            )
            vrt_tiles.append(output_raster_vrt)
        
        vrt_filename = str(outputs / f'mosaic_{file_type}_{crs}.vrt')
        gdal.BuildVRT(vrt_filename, vrt_tiles, callback=gdal.TermProgress_nocb)


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

    # get master_grid geopackage path
    master_grid_geopackage = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
    # if/else logic only allows one option of Eco Region selection or Draw Polygon
    if param_lookup['drawn_polygon'].value:
        drawn_layer_gdf = gpd.read_file(param_lookup['drawn_polygon'].value)
        selected_sub_grids = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=drawn_layer_gdf)
    else:
        # get eco region from shapefile that matches drop down choices
        all_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), columns=['EcoRegion'])
        eco_regions = param_lookup['eco_regions'].valueAsText.replace("'", "").split(';')
        eco_regions = [region.split('-')[0] for region in eco_regions]
        selected_ecoregions = all_ecoregions[all_ecoregions['EcoRegion'].isin(eco_regions)]  # select eco_region polygons
        selected_sub_grids = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=selected_ecoregions)

    mask_tiles = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=selected_sub_grids)
    # Clip to remove extra polygons not handled by mask property
    tiles = gpd.clip(mask_tiles, selected_sub_grids, keep_geom_type=True)
    tiles.to_file(OUTPUTS / 'selected_tiles.shp') 

    return tiles


def process_bluetopo_tiles(tiles: gpd.GeoDataFrame, outputs:str = False) -> None:
    """Entry point for parallel processing of BlueTopo tiles"""

    # get environment (dev, prod)
    # if dev, use multiprocessing
    # if prod, send to API endpoint of listeners in kubernetes
        # pickle each tuple of engine and tile
        # unpickle the object
        # call the class method with the tile argument
        # log success of each call
        # notify the main caller of completion?!
    processor = BlueTopoProcessor()
    processor.process(tiles, outputs)


def process_digital_coast_files(tiles: gpd.GeoDataFrame, outputs:str = False) -> None:
    """Entry point for parallel proccessing of Digital Coast data"""
    
    processor = DigitalCoastProcessor()
    processor.process(tiles, outputs)

def process_model_data(input_directory: str = False) -> None: 
    processor = ModelDataProcessor()
    processor.process(input_directory)  
