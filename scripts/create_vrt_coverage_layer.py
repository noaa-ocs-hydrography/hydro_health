import pathlib
import boto3
import gc
import yaml
import s3fs
import json
from osgeo import gdal, ogr, osr
import geopandas as gpd
from shapely.geometry import shape
from shapely.wkt import loads as load_wkt
from osgeo import gdal

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'

gdal.UseExceptions()


# Assuming get_config_item and self.write_message exist in your class context

def _set_gdal_s3_options():
    """Optimized GDAL settings for S3 VSI stability."""

    gdal.SetConfigOption('GDAL_CACHEMAX', '1024') # Bumped to 1GB if memory allows
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('VSI_CACHE_SIZE', '128000000') # 128MB cache for chunk reads
    
    # Error handling and retries
    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '10') # Increase retries
    gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '5') # Give S3 breathing room
    gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '120') # Increase timeout for slow tiles
    
    # Crucial for avoiding throttling/deadlocks on high-latency networks
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('CPL_VSIL_CURL_ALLOWED_FILENAME_EXTENSIONS', '.tif,.vrt')
    
    # AWS specifics
    gdal.SetConfigOption('AWS_REGION', 'us-east-2')

    gdal.SetConfigOption('GDAL_HTTP_USE_PROXIES', 'YES')
    gdal.SetConfigOption('AWS_DEFAULT_PROFILE', 'your-profile-if-applicable')



def get_config_item(parent: str, child: str=False, env_string: str=False, pilot_mode: bool=False) -> str:
    """
    Load config and return speciific key
    :param str parent: Primary key in config
    :param str child: Secondary key in config
    :param str env_string: Optional explicit value of "local" or "remote"
    :param str pilot_mode: Optional pilot model focused run
    :returns str: Value from local or remote YAML config
    """

    # TODO add sample data and folders to input folder

    env = env_string if env_string else None
    if env is None:
        env = 'aws'

    config_name = f'{env}_{pilot_mode}_path_config.yaml' if pilot_mode else f'{env}_path_config.yaml'
    file_path = str(INPUTS / 'lookups' / config_name) 

    with open(file_path, 'r') as lookup:
        config = yaml.safe_load(lookup)
        parent_item = config[parent]
        
        if child:
            return parent_item[child]
        else:
            return parent_item
        

def find_provider_vrts(ecoregion: str, manual_downloads: bool) -> list[str]:
    """Obtain list of VRT S3 paths"""

    s3 = s3fs.S3FileSystem()
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    dc_sub = get_config_item('DIGITALCOAST', 'SUBFOLDER')
    search_paths = [f"s3://{bucket}/{ecoregion}/{dc_sub}/DigitalCoast"]
    if manual_downloads: 
        search_paths.append(f"{search_paths[0]}_manual_downloads")
    
    found = []
    for path in search_paths: 
        found.extend(s3.glob(f"{path}/**/mosaic_*.vrt"))
    return found
    

def create_vector_attribution_mask(ecoregion: str, s3_vrt_paths: list[str], wkt: str, outputs: str) -> str:
    """
    Creates a new GeoPackage layer where polygons represent the exact spatial intersection 
    between each VRT file's actual data footprint and the starting ecoregion WKT boundary.
    """
    _set_gdal_s3_options()

    scratch_dir = pathlib.Path.home() / f"gdal_scratch_{ecoregion}_vector"
    scratch_dir.mkdir(exist_ok=True)
    
    final_output_path = scratch_dir / f"vrt_coverage_{ecoregion}.gpkg"

    s3_client = boto3.client('s3', region_name='us-east-2')
    
    # 1. Convert the incoming WKT string into the base GeoDataFrame
    # Adjust target_crs to match your data if it isn't EPSG:4326
    target_crs = "EPSG:4326" 
    ecoregion_geom = load_wkt(wkt)
    pred_gdf = gpd.GeoDataFrame(geometry=[ecoregion_geom], crs=target_crs)

    vrt_records = []
    
    for idx, s3_path in enumerate(s3_vrt_paths, start=1):
        print(idx, ': ', s3_path)
        vsi_path = s3_path.replace("s3://", "/vsis3/")
        
        try:
            # 2. Open the VRT dataset
            ds = gdal.Open(vsi_path)
            if ds is None:
                continue
                
            band = ds.GetRasterBand(1)
            ovr_count = band.GetOverviewCount()
            
            # Use a unique virtual file path for this iteration
            mem_vsi_path = f"/vsimem/footprint_{idx}.json"
            
            # Formulate the C++ CLI flags
            footprint_args = [
                '-of', 'GeoJSON',
                '-t_srs', 'EPSG:4326',
                '-max_points', '1000'
            ]
            
            # If overviews exist, use the lowest-resolution one
            if ovr_count > 0:
                footprint_args.extend(['-ovr', str(ovr_count - 1)])
                
            # Write directly to our in-memory virtual filesystem string destination
            out_ds = gdal.Footprint(mem_vsi_path, ds, options=footprint_args)
            
            if out_ds is not None:
                # Close the output dataset handle to flush bytes to /vsimem/
                out_ds = None 
                
                # Read the raw string payload from GDAL's virtual memory
                f = gdal.VSIFOpenL(mem_vsi_path, 'r')
                if f:
                    gdal.VSIFSeekL(f, 0, 2)
                    size = gdal.VSIFTellL(f)
                    gdal.VSIFSeekL(f, 0, 0)
                    geojson_str = gdal.VSIFReadL(1, size, f).decode('utf-8')
                    gdal.VSIFCloseL(f)
                    
                    footprint_json = json.loads(geojson_str)
                    if footprint_json.get('features'):
                        vrt_poly = shape(footprint_json['features'][0]['geometry'])
                    else:
                        vrt_poly = None
                else:
                    vrt_poly = None
            else:
                vrt_poly = None
                
            # Always clean up the virtual memory file to avoid memory leaks
            gdal.Unlink(mem_vsi_path)
            ds = None # Explicitly close the GDAL dataset asset
            
            if vrt_poly is None:
                continue
            
        except Exception as e:
            print(f"Failed to process footprint for {s3_path}: Unexpected exception: {e}")
            # Ensure cleanup happens even on exception
            if 'mem_vsi_path' in locals():
                gdal.Unlink(mem_vsi_path)
            continue

        # Track file metadata alongside the true data geometry
        vrt_records.append({
            "vrt_id": idx,
            "vrt_filename": pathlib.Path(s3_path).name,
            "s3_path": s3_path,
            "geometry": vrt_poly
        })

    if not vrt_records:
        return f"{ecoregion}: No valid VRT footprints found."

    # 3. Convert extracted VRT footprints into a GeoDataFrame
    vrt_gdf = gpd.GeoDataFrame(vrt_records, crs=target_crs)

    # 4. Handle any CRS mismatches prior to overlay intersection computation
    if vrt_gdf.crs != pred_gdf.crs:
        vrt_gdf = vrt_gdf.to_crs(pred_gdf.crs)

    # 5. Intersect the two vector layers using spatial overlay
    intersection_gdf = gpd.overlay(pred_gdf, vrt_gdf, how="intersection")

    # 6. Save the resulting attribution layer to a GeoPackage
    intersection_gdf.to_file(final_output_path, driver="GPKG", layer="vrt_attribution")

    # 7. Upload final product back to S3
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    mask_sub = get_config_item('MASK', 'SUBFOLDER')
    s3_key = f"{ecoregion}/{mask_sub}/{ecoregion}_vector_data.gpkg"
    s3_client.upload_file(str(final_output_path), bucket, s3_key)

    return f"{ecoregion}: Vector attribution mask completed"


if __name__ == '__main__':
    # 2. Read the prediction layer into a GeoDataFrame
    print('start')
    gpkg = str(INPUTS / 'Master_Grids.gpkg')
    gdf = gpd.read_file(gpkg, layer='Enhanced_EcoRegions').to_crs("EPSG:32617")
    for _, row in gdf.iterrows():
        er = row['EcoRegion']
        wkt = row['geometry'].wkt
        vrts = find_provider_vrts(er, manual_downloads=False)
        vrt_list = [f"s3://{v}" if not v.startswith('s3://') else v for v in vrts]
        create_vector_attribution_mask(er, vrt_list, wkt, OUTPUTS)
    print('done')