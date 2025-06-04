import yaml
import pathlib
import geopandas as gpd
import rioxarray as rxr
import rasterio

from rio_vrt import build_vrt
from hydro_health.engines.tiling.BlueTopoProcessor import BlueTopoProcessor
from hydro_health.engines.tiling.DigitalCoastProcessor import DigitalCoastProcessor
from hydro_health.engines.tiling.RasterMaskProcessor import RasterMaskProcessor
from hydro_health.engines.tiling.SurgeTideForecastProcessor import SurgeTideForecastProcessor
from osgeo import gdal, osr


gdal.UseExceptions()
gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')
gdal.SetConfigOption("GDALWARP_IGNORE_BAD_CUTLINE", "YES")
gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
gdal.SetCacheMax(2684354560)  # 20gb RAM

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
        output_vrt = geotiff_path.parents[0] / f'{geotiff_path.stem}.vrt'
        if output_vrt.exists():
            print(f'Skipping VRT: {output_vrt.name}')
        else:
            geotiff_ds = gdal.Open(geotiff)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            geotiff_srs = geotiff_ds.GetSpatialRef()
            # Project all geotiff to match BlueTopo tiles WGS84
            if data_type == 'DigitalCoast' and not geotiff_srs.IsSame(wgs84_srs):
                geotiff_ds = None  # close dataset

                old_geotiff = geotiff_path.parents[0] / f'{geotiff_path.stem}_old.tif'
                geotiff_path.rename(old_geotiff)
                raster_wgs84 = geotiff_path.parents[0] / f'{geotiff_path.stem}_wgs84.tif'
                rasterio_wgs84 = rasterio.crs.CRS.from_epsg(4326)
                with rxr.open_rasterio(old_geotiff) as geotiff_raster:
                    wgs84_geotiff_raster = geotiff_raster.rio.reproject(rasterio_wgs84)
                    wgs84_geotiff_raster.rio.to_raster(raster_wgs84)

                wgs84_ds = gdal.Open(str(raster_wgs84))
                # Compress and overwrite original geotiff path
                gdal.Warp(
                    geotiff,
                    wgs84_ds,
                    creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"]
                )
                wgs84_ds = None
                
                # Delete intermediate files
                old_geotiff.unlink()
                raster_wgs84.unlink()

        geotiff_ds = gdal.Open(geotiff)  # reopen new projected geotiff path
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
    
    # TODO does this need to happen now that all files are wgs84?
    for crs, tile_dict in output_geotiffs.items():
        # Create VRT for each tile and set output CRS to fix heterogenous crs issue
        vrt_tiles = []
        for tile in tile_dict['tiles']:
            output_raster_vrt = tile.parents[0] / f"{tile.stem}.vrt"
            if not output_raster_vrt.exists():
                gdal.Warp(
                    str(output_raster_vrt), 
                    str(tile),
                    format="VRT",
                    dstSRS=output_geotiffs[crs]['crs']
                )
            vrt_tiles.append(output_raster_vrt)
        
        vrt_filename = outputs / f'mosaic_{file_type}_{crs}.vrt'
        # gdal.BuildVRT(vrt_filename, vrt_tiles, callback=gdal.TermProgress_nocb)
        if not vrt_filename.exists():
            build_vrt(str(vrt_filename), vrt_tiles)
    print('finished create_raster_vrts')


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


def get_digitalcoast_folders(data_folder) -> list[pathlib.Path]:
    """Get the lower level project folders for Digital Coast"""

    digitalcoast_projects = [folder for folder in data_folder.iterdir() if folder.is_dir() and folder.stem != 'tiled']
    digitalcoast_file_folders = []
    for project in digitalcoast_projects:
        dem_folder = project / 'dem'
        for file_folder in dem_folder.iterdir():
            if file_folder.is_dir():
                digitalcoast_file_folders.append(file_folder)
    return digitalcoast_file_folders


def grid_digitalcoast_files(outputs: str) -> None:
    """Clip VRT files to BlueTopo grid"""

    blue_topo_layer = gpd.read_file(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')), layer=get_config_item('SHARED', 'TILES'))
    ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
    for ecoregion in ecoregions:
        blue_topo_folder = ecoregion / 'BlueTopo'
        bluetopo_folders = [folder.stem for folder in blue_topo_folder.iterdir() if folder.is_dir()]

        data_folder = ecoregion / 'DigitalCoast'
        digitalcoast_file_folders = get_digitalcoast_folders(data_folder)

        for file_folder in digitalcoast_file_folders:
            tileindex_shp_path = list(file_folder.glob('tileindex*.shp'))[0]
            tileindex_gdf = gpd.read_file(tileindex_shp_path)
            tiled_folder = file_folder.parents[2] / 'tiled'
            for bluetopo_folder in bluetopo_folders:
                bluetopo_data = blue_topo_layer[blue_topo_layer['tile'] == bluetopo_folder]
                polygon_wkt = bluetopo_data['geometry'].iloc[0]
                tile_bluetopo_intersect = tileindex_gdf[tileindex_gdf.intersects(bluetopo_data.unary_union)]
                if len(tile_bluetopo_intersect) > 0:
                    intersected_images = [str(file_folder / pathlib.Path(image).name) for image in tile_bluetopo_intersect['location']]
                    bad_files, good_files = [], []
                    for image in intersected_images:
                        if not pathlib.Path(image).exists():
                            bad_files.append(image)
                        else:
                            good_files.append(image)
                    if bad_files:
                        with open(ecoregion.parents[0] / 'log_grid_tiling.txt', 'a') as writer:
                            print(f'bad files: {bad_files}')
                            for bad_file in bad_files:
                                writer.write(bad_file + '\n')
                            writer.write('\n')
                    if good_files:
                        output_path = tiled_folder / bluetopo_folder
                        output_path.mkdir(parents=True, exist_ok=True)
                        project_name = pathlib.Path(tileindex_shp_path).parents[2]
                        clipped_vrt = output_path / f'{project_name.stem}_{bluetopo_folder}.tif'
                        if clipped_vrt.exists():
                            continue
                        print(f'Creating {clipped_vrt.name}')
                        in_memory_vrt = gdal.BuildVRT('', good_files, callback=gdal.TermProgress_nocb)
                        gdal.Warp(
                            clipped_vrt,
                            in_memory_vrt,
                            format='GTiff',
                            cutlineDSName=polygon_wkt,
                            cropToCutline=True,
                            warpMemoryLimit=2684354560,
                            dstNodata=in_memory_vrt.GetRasterBand(1).GetNoDataValue(),
                            cutlineSRS=in_memory_vrt.GetProjection(),
                            creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=IF_NEEDED", "TILED=YES"]
                        )


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


def project_raster_wgs84(raster_path: pathlib.Path, raster_ds: gdal.Dataset, wgs84_srs: osr.SpatialReference) -> pathlib.Path:
    """Project a raster/geotiff to WGS84 spatial reference for tiling"""

    raster_wgs84 = raster_path.parents[0] / f'{raster_path.stem}_wgs84.tif'
    gdal.Warp(
        raster_wgs84,
        raster_ds,
        dstSRS=wgs84_srs
    )
    return raster_wgs84
