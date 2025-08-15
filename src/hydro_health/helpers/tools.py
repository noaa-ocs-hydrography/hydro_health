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


def create_raster_vrts(output_folder: str, file_type: str, ecoregion: str, data_type: str) -> None:
    """Create an output VRT from found .tif files"""

    glob_lookup = {
        'elevation': '*[0-9].tiff',
        'uncertainty': '*_unc.tiff',
        'slope': '*_slope.tiff',
        'rugosity': '*_rugosity.tiff',
        'NCMP': '*.tif'
    }

    outputs = pathlib.Path(output_folder) / ecoregion / data_type
    geotiffs = list(outputs.rglob(glob_lookup[file_type]))

    output_geotiffs = {}
    for geotiff_path in geotiffs:
        geotiff = str(geotiff_path)
        geotiff_ds = gdal.Open(geotiff)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        geotiff_srs = geotiff_ds.GetSpatialRef()
        # Project all geotiff to match BlueTopo tiles WGS84
        if data_type == 'DigitalCoast' and not geotiff_srs.IsSame(wgs84_srs):
            geotiff_ds = None  # close original dataset

            old_geotiff = geotiff_path.parents[0] / f'{geotiff_path.stem}_old.tif'
            geotiff_path.rename(old_geotiff)
            raster_wgs84 = geotiff_path.parents[0] / f'{geotiff_path.stem}_wgs84.tif'
            rasterio_wgs84 = rasterio.crs.CRS.from_epsg(4326)
            with rxr.open_rasterio(old_geotiff) as geotiff_raster:
                wgs84_geotiff_raster = geotiff_raster.rio.reproject(rasterio_wgs84)
                wgs84_geotiff_raster.rio.to_raster(raster_wgs84)

            wgs84_ds = gdal.Open(str(raster_wgs84))
            # Compress and overwrite original geotiff path

            # TODO run and see if standard XY will fix 2010 failure
            resolution = 0.000008983
            gdal.Warp(
                geotiff,
                wgs84_ds,
                srcNodata=wgs84_ds.GetRasterBand(1).GetNoDataValue(),
                dstNodata=-9999,
                xRes=resolution,
                yRes=resolution,
                resampleAlg="bilinear",
                creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"]
            )
            wgs84_ds = None
            
            # Delete intermediate files
            old_geotiff.unlink()
            raster_wgs84.unlink()

            # repoen new geotiff
            geotiff_ds = gdal.Open(geotiff)

        projection_wkt = geotiff_ds.GetProjection()
        spatial_ref = osr.SpatialReference(wkt=projection_wkt)  
        projected_crs_string = spatial_ref.GetAuthorityCode('DATUM')
        clean_crs_string = projected_crs_string.replace('/', '').replace(' ', '_')
        provider_folder = geotiff_path.parents[2].name
        # Handle BlueTopo and DigitalCoast differently
        clean_crs_key = f'{clean_crs_string}_{provider_folder}' if data_type == 'DigitalCoast' else clean_crs_string
        # Store tile and CRS
        if clean_crs_key not in output_geotiffs:
            output_geotiffs[clean_crs_key] = {'crs': None, 'tiles': []}
        output_geotiffs[clean_crs_key]['tiles'].append(geotiff_path)
        if output_geotiffs[clean_crs_key]['crs'] is None:
            output_geotiffs[clean_crs_key]['crs'] = spatial_ref
        
        projected_crs_string = None
        projection_wkt = None
        wgs84_srs = None
        geotiff_srs = None
        spatial_ref = None
        geotiff_ds = None
    
    for crs, tile_dict in output_geotiffs.items():
        # Create VRT for each tile and set output CRS to fix heterogenous crs issue
        vrt_tiles = []
        for tile in tile_dict['tiles']:
            output_raster_vrt = str(tile.parents[0] / f"{tile.stem}.vrt")
            gdal.Warp(
                output_raster_vrt, 
                str(tile),
                format="VRT",
                dstSRS=output_geotiffs[crs]['crs']
            )
            vrt_tiles.append(output_raster_vrt)
        
        vrt_filename = str(outputs / f'mosaic_{file_type}_{crs}.vrt')
        gdal.BuildVRT(vrt_filename, vrt_tiles, callback=gdal.TermProgress_nocb)


def get_environment() -> str:
    """Determine current environment running code"""

    hostname = gethostname()
    if 'L' in hostname:
        return  'local'
    elif 'VS' in hostname:
        return 'remote'
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


def grid_digital_coast_files(outputs: str, data_type: str) -> None:
    """Process for gridding Digital Coast files to BlueTopo grid"""

    print('Gridding Digital Coast files to BlueTopo grids')
    gpkg_ds = ogr.Open(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))
    blue_topo_layer = gpkg_ds.GetLayerByName(get_config_item('SHARED', 'TILES'))
    ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
    for ecoregion in ecoregions:
        blue_topo_folder = ecoregion / 'BlueTopo'
        bluetopo_grids = [folder.stem for folder in blue_topo_folder.iterdir() if folder.is_dir()]
        data_folder = ecoregion / data_type
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
                    if current_tile_geom.Intersects(dissolve_geom):
                        output_path = ecoregion / data_type / 'tiled' / folder_name
                        output_clipped_vrt = output_path / f'{vrt.stem}_{folder_name}.tiff'
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
                            print('failure:', e)
                    current_tile_geom = None
            shp_driver = None
            dissolve_layer = None
            vrt_ds = None
    gpkg_ds = None
    blue_topo_layer = None
    print('Finished Gridding Digital Coast')


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


def run_vrt_creation(param_lookup: dict[str]) -> None:
    """Entry point for building VRT files for BlueTopo and Digital Coast data"""

    for ecoregion in get_ecoregion_folders(param_lookup):
        for dataset in ['elevation', 'slope', 'rugosity', 'uncertainty']:
            print(f'Building {ecoregion} - {dataset} VRT file')
            create_raster_vrts(param_lookup['output_directory'].valueAsText, dataset, ecoregion, 'BlueTopo')
        create_raster_vrts(param_lookup['output_directory'].valueAsText, 'NCMP', ecoregion, 'DigitalCoast')

