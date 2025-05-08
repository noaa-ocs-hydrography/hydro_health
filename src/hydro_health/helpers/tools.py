import yaml
import pathlib
import tempfile
import geopandas as gpd

from hydro_health.engines.tiling.BlueTopoProcessor import BlueTopoProcessor
from hydro_health.engines.tiling.DigitalCoastProcessor import DigitalCoastProcessor
from hydro_health.engines.tiling.RasterMaskProcessor import RasterMaskProcessor
from hydro_health.engines.tiling.SurgeTideForecastProcessor import SurgeTideForecastProcessor
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


def process_create_masks(outputs:str) -> None:
    """Create prediction and training masks for found ecoregions"""

    processor = RasterMaskProcessor()
    processor.process(outputs)


def create_raster_vrt(output_folder: str, file_type: str, ecoregion: str, data_type: str) -> None:
    """Create an output VRT from found .tif files"""

    glob_lookup = {
        'elevation': '*[0-9].tiff',
        'slope': '*_slope.tiff',
        'rugosity': '*_rugosity.tiff',
        'NCMP': '*.tif'
    }

    # TODO data_type needs to be ecoregion folder
    outputs = pathlib.Path(output_folder) / ecoregion / data_type
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

    output_folder = pathlib.Path(param_lookup['output_directory'].valueAsText)
    # get master_grid geopackage path
    master_grid_geopackage = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

    # if/else logic only allows one option of Eco Region selection or Draw Polygon
    all_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), columns=['EcoRegion'])
    if param_lookup['drawn_polygon'].value:
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
    selected_ecoregions.to_file(output_folder / 'selected_ecoregions.shp') 
    tiles.to_file(output_folder / 'selected_tiles.shp') 

    return tiles


def get_ecoregion_folders(param_lookup: dict[str]) -> gpd.GeoDataFrame:
    """Obtain the intersected EcoRegion folders"""

    output_folder = pathlib.Path(param_lookup['output_directory'].valueAsText)
    # get master_grid geopackage path
    master_grid_geopackage = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
    all_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), columns=['EcoRegion'])
    if param_lookup['drawn_polygon'].value:
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


def grid_vrt_files(outputs: str, data_type: str) -> None:
    """Clip VRT files to BlueTopo grid"""

    gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetCacheMax(28000)

    ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
    for ecoregion in ecoregions:
        bluetopo_grids = [folder.stem for folder in pathlib.Path(ecoregion / 'BlueTopo').iterdir() if folder.is_dir()]
        data_folder = ecoregion / data_type
        vrt_files = data_folder.glob('*.vrt')
        for vrt in vrt_files:
            vrt_ds = gdal.Open(vrt)
            gpkg_ds = ogr.Open(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS'))
            blue_topo_layer = gpkg_ds.GetLayerByName(get_config_item('SHARED', 'TILES'))

            # create extent polygon of raster
            gt = vrt_ds.GetGeoTransform()
            raster_extent = (gt[0], gt[3], gt[0] + gt[1] * vrt_ds.RasterXSize, gt[3] + gt[5] * vrt_ds.RasterYSize)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(raster_extent[0], raster_extent[1])
            ring.AddPoint(raster_extent[2], raster_extent[1])
            ring.AddPoint(raster_extent[2], raster_extent[3])
            ring.AddPoint(raster_extent[0], raster_extent[3])
            ring.AddPoint(raster_extent[0], raster_extent[1])
            raster_geom = ogr.Geometry(ogr.wkbPolygon)
            raster_geom.AddGeometry(ring)

            for feature in blue_topo_layer:
                # Clip VRT by current polygon
                polygon = feature.GetGeometryRef()
                folder_name = feature.GetField('tile')
                output_path = ecoregion / data_type / 'tiled' / folder_name
                output_clipped_vrt = output_path / f'{vrt.stem}_{folder_name}.tiff'
                if output_clipped_vrt.exists():
                    if output_clipped_vrt.stat().st_size == 0:
                        try:
                            print(f're-warp empty raster: {output_clipped_vrt.name}')
                            gdal.Warp(
                                output_clipped_vrt,
                                vrt,
                                format='GTiff',
                                cutlineDSName=polygon,
                                cropToCutline=True,
                                dstNodata=vrt_ds.GetRasterBand(1).GetNoDataValue(),
                                cutlineSRS=vrt_ds.GetProjection()
                            )
                        except RuntimeError as e:
                            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                            print(f'Rerun-Error: {output_clipped_vrt} - {e}')
                            continue
                    else:
                        print(f'Skipping {output_clipped_vrt.name}')
                        continue
                elif folder_name in bluetopo_grids:
                    if polygon.Intersects(raster_geom):
                        output_path.mkdir(parents=True, exist_ok=True)
                        print(f'Creating {output_clipped_vrt.name}')
                        # Try to force clear temp directory to conserve space
                        # with tempfile.TemporaryDirectory() as temp:
                        #     gdal.SetConfigOption('CPL_TMPDIR', temp)
                        try:
                            gdal.Warp(
                                output_clipped_vrt,
                                vrt,
                                format='GTiff',
                                cutlineDSName=polygon,
                                cropToCutline=True,
                                dstNodata=vrt_ds.GetRasterBand(1).GetNoDataValue(),
                                cutlineSRS=vrt_ds.GetProjection()
                            )
                        except RuntimeError as e:
                            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                            print(f'Error: {output_clipped_vrt} - {e}')
                            continue

            raster_geom = None
            vrt_ds = None
            gpkg_ds = None
            blue_topo_layer = None


def make_ecoregion_folders(selected_ecoregions: gpd.GeoDataFrame, output_folder: pathlib.Path):
    """Create the main EcoRegion folders"""

    for _, row in selected_ecoregions.iterrows():
        ecoregion_folder = output_folder / row['EcoRegion']
        ecoregion_folder.mkdir(parents=True, exist_ok=True)


def process_bluetopo_tiles(tiles: gpd.GeoDataFrame, outputs:str) -> None:
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


def process_digital_coast_files(tiles: gpd.GeoDataFrame, outputs: str) -> None:
    """Entry point for parallel proccessing of Digital Coast data"""
    
    processor = DigitalCoastProcessor()
    processor.process(tiles, outputs)


def process_stofs_files(tiles: gpd.GeoDataFrame, outputs: str) -> None:
    """Entry point for parallel processing of STOFS data"""

    processor = SurgeTideForecastProcessor()
    processor.process(tiles, outputs)
