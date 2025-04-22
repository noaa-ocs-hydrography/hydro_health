import yaml
import pathlib
import json
import geopandas as gpd  # pip install geopandas;requires numpy==1.22.4 and activating cloned env in Pro

from hydro_health.engines.tiling.BlueTopoProcessor import BlueTopoProcessor
from hydro_health.engines.tiling.DigitalCoastProcessor import DigitalCoastProcessor
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


def create_prediction_masks() -> None:
    """Build prediction mask from all Digital Coast """
    return


def create_ecoregion_rasters() -> pathlib.Path:
    """Create a boolean mask of ecoregions"""

    ecoregions = ['ER_3']

    gpkg = INPUTS / 'Master_Grids.gpkg'
    gpkg_ds = ogr.Open(gpkg)
    ecoregions_50m = gpkg_ds.GetLayerByName('EcoRegions_50m')
    xmin, xmax, ymin, ymax = ecoregions_50m.GetExtent()
    # TODO need to create 8m grid size
    pixel_size = .00025
    nodata = 0.0
    x_res = int((xmax - xmin) / pixel_size)
    y_res = int((ymax - ymin) / pixel_size)

    ecoregions_50m.ResetReading()
    in_memory = ogr.GetDriverByName('Memory')
    for feature in ecoregions_50m:
        feature_json = json.loads(feature.ExportToJson())
        ecoregion_id = feature_json['properties']['EcoRegion']
        if ecoregion_id in ecoregions:
            in_memory_ds = in_memory.CreateDataSource(str(OUTPUTS / f'output_layer_{ecoregion_id}.shp'))
            in_memory_layer = in_memory_ds.CreateLayer(f'poly_{ecoregion_id}', srs=ecoregions_50m.GetSpatialRef(), geom_type=ogr.wkbPolygon)
            mem_poly = ogr.Feature(in_memory_layer.GetLayerDefn())
            mem_poly.SetGeometry(feature.GetGeometryRef())
            in_memory_layer.CreateFeature(mem_poly)

            mask_path = OUTPUTS / f'ecoregions_50m_mask_{ecoregion_id}.tif'
            with gdal.GetDriverByName("GTiff").Create(
                str(mask_path),
                x_res,
                y_res,
                1,
                gdal.GDT_Float32,
                options=["COMPRESS=LZW"],
            ) as target_ds:
                target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                target_ds.SetProjection(srs.ExportToWkt())
                band = target_ds.GetRasterBand(1)
                band.SetNoDataValue(nodata)

                # Rasterize
                gdal.RasterizeLayer(target_ds, [1], in_memory_layer, burn_values=[1])
                
            in_memory_layer = None


def create_training_masks() -> None:
    """Build training masks from Ecoregion polygons and VRT of any Digital Coast files"""
    return


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

def make_ecoregion_folders(selected_ecoregions: gpd.GeoDataFrame, output_folder: pathlib.Path):
    """Create the main EcoRegion folders"""

    for _, row in selected_ecoregions.iterrows():
        ecoregion_folder = output_folder / row['EcoRegion']
        ecoregion_folder.mkdir(parents=True, exist_ok=True)


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
