import yaml
import pathlib
import geopandas as gpd
import rioxarray as rxr
import rasterio

from socket import gethostname
from osgeo import gdal, osr, ogr


gdal.UseExceptions()


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class Param:
        def __init__(self, path):
            self.value = path

        @property
        def valueAsText(self):
            return self.value


def get_environment() -> str:
    """Determine current environment running code"""

    hostname = gethostname()
    if 'L' in hostname:
        return  'local'
    elif 'VS' in hostname:
        return 'remote'
    elif 'ip-10' in hostname:
        return 'aws'
    else:
        return 'remote'


def get_config_item(parent: str, child: str=False, env_string: str=False) -> str:
    """
    Load config and return speciific key
    :param str parent: Primary key in config
    :param str child: Secondary key in config
    :param str env_string: Optional explicit value of "local" or "remote"
    :returns str: Value from local or remote YAML config
    """

    # TODO add sample data and folders to input folder

    env = env_string if env_string else None
    if env is None:
        env = get_environment()
    
    with open(str(INPUTS / 'lookups' / f'{env}_path_config.yaml'), 'r') as lookup:
        config = yaml.safe_load(lookup)
        parent_item = config[parent]
        if child:
            return parent_item[child]
        else:
            return parent_item


def get_ecoregion_folders(param_lookup: dict[str]) -> gpd.GeoDataFrame:
    """Obtain the intersected EcoRegion folders"""

    output_folder = pathlib.Path(param_lookup['output_directory'].valueAsText)
    # get master_grid geopackage path
    master_grid_geopackage = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
    all_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), columns=['EcoRegion'])
    if param_lookup['env'] == 'local':
        drawn_layer_gdf = gpd.read_file(param_lookup['drawn_polygon'].value)
        selected_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), mask=drawn_layer_gdf)
        make_ecoregion_folders(selected_ecoregions, output_folder)
    else:
        # get eco region from shapefile that matches drop down choices
        eco_regions = param_lookup['eco_regions'].valueAsText.replace("'", "").split(';')
        eco_regions = [region.split('-')[0] for region in eco_regions]
        selected_ecoregions = all_ecoregions[all_ecoregions['EcoRegion'].isin(eco_regions)]  # select eco_region polygons
        make_ecoregion_folders(selected_ecoregions, output_folder)
    return list(selected_ecoregions['EcoRegion'].unique())



def get_ecoregion_tiles(param_lookup: dict[str]) -> gpd.GeoDataFrame:
    """Obtain a subset of tiles based on selected eco regions"""

    output_folder = pathlib.Path(param_lookup['output_directory'].valueAsText)
    # get master_grid geopackage path
    master_grid_geopackage = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

    # if/else logic only allows one option of Eco Region selection or Draw Polygon
    all_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), columns=['EcoRegion'])
    if param_lookup['env'] == 'local' or param_lookup['env'] == 'aws':
        drawn_layer_gdf = gpd.read_file(param_lookup['drawn_polygon'].value)
        selected_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), mask=drawn_layer_gdf)
        make_ecoregion_folders(selected_ecoregions, output_folder)
        selected_sub_grids = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=drawn_layer_gdf)
    else:
        # get eco region from shapefile that matches drop down choices
        eco_regions = param_lookup['eco_regions'].valueAsText.replace("'", "").split(';')
        eco_regions = [region.split('-')[0] for region in eco_regions]
        selected_ecoregions = all_ecoregions[all_ecoregions['EcoRegion'].isin(eco_regions)]  # select eco_region polygons
        make_ecoregion_folders(selected_ecoregions, output_folder)
        selected_sub_grids = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=selected_ecoregions)

    mask_tiles = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=selected_sub_grids)
    # Clip to remove extra polygons not handled by mask property
    tiles = gpd.clip(mask_tiles, selected_sub_grids, keep_geom_type=True)
    # Store EcoRegion ID with tiles
    tiles = tiles.sjoin(selected_ecoregions, how="left")[['tile', 'EcoRegion', 'geometry']]
    # selected_ecoregions.to_file(output_folder / 'selected_ecoregions.shp') 
    # tiles.to_file(output_folder / 'selected_tiles.shp') 

    return tiles


def grid_digital_coast_files(outputs: str, data_type: str) -> None:
    """Process for gridding Digital Coast files to BlueTopo grid"""

    print('Gridding Digital Coast files to BlueTopo grids')
    gpkg_ds = ogr.Open(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))
    blue_topo_layer = gpkg_ds.GetLayerByName(get_config_item('SHARED', 'TILES'))
    ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
    for ecoregion in ecoregions:
        blue_topo_folder = ecoregion / get_config_item('BLUETOPO', 'SUBFOLDER') / 'BlueTopo'
        bluetopo_grids = [folder.stem for folder in blue_topo_folder.iterdir() if folder.is_dir()]
        data_folder = ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / data_type
        vrt_files = list(data_folder.glob('*.vrt'))
        for vrt in vrt_files:
            vrt_ds = gdal.Open(str(vrt))
            vrt_data_folder = vrt.parents[0] / '_'.join(vrt.stem.split('_')[3:])
            vrt_tile_index = list(vrt_data_folder.rglob('*_dis.shp'))[0]
            shp_driver = ogr.GetDriverByName('ESRI Shapefile')
            vrt_tile_index_shp = shp_driver.Open(vrt_tile_index, 0)
            dissolve_layer = vrt_tile_index_shp.GetLayer(0)
            dissolve_feature = dissolve_layer.GetFeature(0)  # have to keep reference to feature or it will garbage collect
            dissolve_geom = dissolve_feature.GetGeometryRef()
            blue_topo_layer.ResetReading()
            for tile in blue_topo_layer:
                # Clip VRT by current polygon
                current_tile_geom = tile.GetGeometryRef()
                folder_name = tile.GetField('tile')
                if folder_name in bluetopo_grids:
                    output_path = ecoregion / get_config_item('DIGITALCOAST', 'TILED_SUBFOLDER') / folder_name
                    output_clipped_vrt = output_path / f'{vrt.stem}_{folder_name}.tiff'
                    if output_clipped_vrt.exists():
                        print(f' - Skipping {output_clipped_vrt.name}')
                        continue
                    if current_tile_geom.Intersects(dissolve_geom):
                        output_path.mkdir(parents=True, exist_ok=True)
                        print(f' - Creating {output_clipped_vrt.name}')
                        try:
                            polygon = current_tile_geom.ExportToWkt()
                            gdal.Warp(
                                str(output_clipped_vrt),
                                str(vrt),
                                format='GTiff',
                                cutlineDSName=polygon,
                                cropToCutline=True,
                                dstNodata=-9999,
                                cutlineSRS=vrt_ds.GetProjection(),
                                creationOptions=["COMPRESS=DEFLATE", "TILED=YES"]
                            )
                        except Exception as e:
                            print(f'failure: {vrt.name} - ', e)
                    current_tile_geom = None
            shp_driver = None
            dissolve_layer = None
            vrt_ds = None
    gpkg_ds = None
    blue_topo_layer = None


def make_ecoregion_folders(selected_ecoregions: gpd.GeoDataFrame, output_folder: pathlib.Path):
    """Create the main EcoRegion folders"""

    for _, row in selected_ecoregions.iterrows():
        ecoregion_folder = output_folder / row['EcoRegion']
        ecoregion_folder.mkdir(parents=True, exist_ok=True)


def project_raster_wgs84(raster_path: pathlib.Path, raster_ds: gdal.Dataset, wgs84_srs: osr.SpatialReference) -> pathlib.Path:
    """Project a raster/geotiff to WGS84 spatial reference for tiling"""

    raster_wgs84 = raster_path.parents[0] / f'{raster_path.stem}_wgs84.tif'
    gdal.Warp(
        raster_wgs84,
        raster_ds,
        dstSRS=wgs84_srs
    )
    return raster_wgs84

