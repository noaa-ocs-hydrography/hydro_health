import yaml
import pathlib
import geopandas as gpd
import s3fs
import os
import tempfile
import json
import time

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
        eco_regions = param_lookup['eco_regions'].value
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
    if param_lookup['env'] == 'local':  # or param_lookup['env'] == 'aws':
        drawn_layer_gdf = gpd.read_file(param_lookup['drawn_polygon'].value)
        selected_ecoregions = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'ECOREGIONS'), mask=drawn_layer_gdf)
        selected_sub_grids = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=drawn_layer_gdf)
    else:
        # get eco region from shapefile that matches drop down choices
        eco_regions = param_lookup['eco_regions'].valueAsText   #.replace("'", "").split(';')
        selected_ecoregions = all_ecoregions[all_ecoregions['EcoRegion'].isin(eco_regions)]  # select eco_region polygons
        selected_sub_grids = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=selected_ecoregions)

    mask_tiles = gpd.read_file(master_grid_geopackage, layer=get_config_item('SHARED', 'TILES'), columns=['tile'], mask=selected_sub_grids)
    # Clip to remove extra polygons not handled by mask property
    tiles = gpd.clip(mask_tiles, selected_sub_grids, keep_geom_type=True)
    # Store EcoRegion ID with tiles
    tiles = tiles.sjoin(selected_ecoregions, how="left")[['tile', 'EcoRegion', 'geometry']]
    # selected_ecoregions.to_file(output_folder / 'selected_ecoregions.shp') 
    # tiles.to_file(output_folder / 'selected_tiles.shp') 

    return tiles


def grid_local_digital_coast_files(outputs: str) -> None:
    """Process for gridding Digital Coast files to BlueTopo grid"""

    print('Gridding Digital Coast files to BlueTopo grids')
    gpkg_ds = ogr.Open(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))
    blue_topo_layer = gpkg_ds.GetLayerByName(get_config_item('SHARED', 'TILES'))
    ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
    for ecoregion in ecoregions:
        blue_topo_folder = ecoregion / get_config_item('BLUETOPO', 'SUBFOLDER') / 'BlueTopo'
        bluetopo_grids = [folder.stem for folder in blue_topo_folder.iterdir() if folder.is_dir()]
        data_folder = ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
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


def grid_s3_digital_coast_files() -> None:
    """Process for gridding Digital Coast files to BlueTopo grid using GeoPandas and s3fs"""

    # All of these seem necessary for writing to S3
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'BASEDIR')
    gdal.SetConfigOption('CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE', 'YES')
    gdal.SetConfigOption('CPL_VSIL_S3_WRITE_SUPPORT', 'YES')
    gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '120') 
    gdal.SetConfigOption('GDAL_PAM_ENABLED', 'NO')
    gdal.SetConfigOption('AWS_REGION', 'us-east-2')
    gdal.SetConfigOption('GDAL_VSI_S3_CHUNK_SIZE', '64') # 64 MB chunks
    gdal.SetConfigOption('CPL_VSIL_S3_USE_PRE_SIGNED_URL', 'YES')
    gdal.SetConfigOption('GDAL_VSI_CACHE', 'NO')
    gdal.UseExceptions()

    s3_files = s3fs.S3FileSystem()
    
    print('Gridding Digital Coast files to BlueTopo grids')
    bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')
    
    master_grids_path = str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS'))
    tiles_layer_name = get_config_item('SHARED', 'TILES')
    blue_topo_gdf = gpd.read_file(master_grids_path, layer=tiles_layer_name)

    ecoregions = s3_files.glob(f"{bucket_name}/testing/ER_*")
    for ecoregion_prefix in ecoregions:
        bluetopo_subfolder = get_config_item('BLUETOPO', 'SUBFOLDER')
        blue_topo_search = f"{ecoregion_prefix}/{bluetopo_subfolder}/BlueTopo/"
        bluetopo_grids = [p.split('/')[-1] for p in s3_files.ls(blue_topo_search) if s3_files.isdir(p)]
        
        digital_coast_subfolder = get_config_item('DIGITALCOAST', 'SUBFOLDER')
        vrt_files = s3_files.glob(f"{ecoregion_prefix}/{digital_coast_subfolder}/DigitalCoast/*.vrt")
        for vrt_s3_path in vrt_files:
            vrt_stem = vrt_s3_path.split('/')[-1].replace('.vrt', '')
            vsi_vrt_path = f"/vsis3/{vrt_s3_path}"
            
            try:
                vrt_ds = gdal.Open(vsi_vrt_path)
                vrt_projection = vrt_ds.GetProjection()
            except Exception as e:
                print(f" - Error opening VRT {vsi_vrt_path}: {e}")
                continue
            
            vrt_data_suffix = '_'.join(vrt_stem.split('_')[3:])
            vrt_parent = vrt_s3_path.rsplit('/', 1)[0]
            shp_search_path = f"{vrt_parent}/{vrt_data_suffix}/**/*_dis.shp"
            shp_matches = s3_files.glob(shp_search_path)
            
            if not shp_matches:
                vrt_ds = None
                continue
            
            try:
                # Only downloading shp works for upload process to S3
                with tempfile.TemporaryDirectory() as tmpdir:
                    s3_shp_full = shp_matches[0]
                    s3_base = s3_shp_full.rsplit('.', 1)[0]
                    local_base = os.path.join(tmpdir, "tileindex")
                    
                    for ext in ['.shp', '.shx', '.dbf', '.prj']:
                        s3_target = f"{s3_base}{ext}"
                        if s3_files.exists(s3_target):
                            s3_files.get(s3_target, f"{local_base}{ext}")
                    
                    dissolve_gdf = gpd.read_file(f"{local_base}.shp")
                    if dissolve_gdf.crs != blue_topo_gdf.crs:
                        dissolve_gdf = dissolve_gdf.to_crs(blue_topo_gdf.crs)
                    
                    dissolve_geom = dissolve_gdf.union_all() 

                intersecting_tiles = blue_topo_gdf[
                    (blue_topo_gdf['tile'].isin(bluetopo_grids)) & 
                    (blue_topo_gdf.intersects(dissolve_geom))
                ]

                for _, tile_row in intersecting_tiles.iterrows():
                    folder_name = tile_row['tile']
                    tiled_sub = get_config_item('DIGITALCOAST', 'TILED_SUBFOLDER')
                    output_prefix = f"{ecoregion_prefix}/{tiled_sub}/{folder_name}/{vrt_stem}_{folder_name}.tiff"
                    
                    if s3_files.exists(output_prefix):
                        print(f' - Skipping {vrt_stem}_{folder_name}.tiff')
                        continue

                    print(f' - Creating local {vrt_stem}_{folder_name}.tiff')
                    
                    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp_file:
                        local_tmp_path = tmp_file.name

                    in_memory_geojson = f"/vsimem/cutline_{folder_name}.json"
                    
                    try:
                        # Use geojson to create cutline polygon
                        tile_geojson = {
                            "type": "FeatureCollection",
                            "features": [{
                                "type": "Feature",
                                "geometry": tile_row.geometry.__geo_interface__,
                                "properties": {"tile": folder_name}
                            }]
                        }
                        gdal.FileFromMemBuffer(in_memory_geojson, json.dumps(tile_geojson))
                        
                        gdal.ErrorReset()
                        dest_ds = gdal.Warp(
                            local_tmp_path,
                            vsi_vrt_path,
                            format='GTiff',
                            cutlineDSName=in_memory_geojson, 
                            cropToCutline=True,
                            dstNodata=-9999,
                            srcSRS=vrt_projection,
                            dstSRS=vrt_projection,
                            creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "NUM_THREADS=ALL_CPUS"]
                        )
                        
                        if dest_ds is not None:
                            dest_ds.FlushCache()
                            dest_ds = None # Explicitly close local file
                            s3_files.put(local_tmp_path, output_prefix)
                            print(f" - Successfully uploaded: {output_prefix}")
                        else:
                            print(f" - Warp failed for {folder_name}: {gdal.GetLastErrorMsg()}")

                    except Exception as e:
                        print(f'   ! Failure on {vrt_stem}_{folder_name}: {e}')
                    
                    finally:
                        if os.path.exists(local_tmp_path):
                            os.remove(local_tmp_path)
                        gdal.Unlink(in_memory_geojson)
                        dest_ds = None
            except Exception as e:
                print(f" - Critical error processing shapefile for {vrt_stem}: {e}")
            
            finally:
                vrt_ds = None 


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

