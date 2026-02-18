import pathlib
import rioxarray as rxr
import json
import s3fs
import tempfile
import boto3
import os

from pyproj.database import query_crs_info
from pyproj.enums import PJType
from osgeo import gdal, osr
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


def _process_single_bluetopo(params: list) -> tuple[str, str, str]:
    """Parallel process creating a Warped VRT for BlueTopo directly from S3"""

    geotiff_prefix, s3_bucket, _ = params

    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('AWS_S3_ENDPOINT', 's3.amazonaws.com')
    
    geotiff_stem = str(pathlib.Path(geotiff_prefix).stem)
    gdal_geotiff_path = f'/vsis3/{geotiff_prefix}'
    
    with tempfile.NamedTemporaryFile(suffix=f"_{geotiff_stem}.vrt", delete=False) as tmp:
        local_vrt_path = tmp.name

    src_ds = None
    try:
        src_ds = gdal.Open(gdal_geotiff_path)
        if src_ds is None:
            raise FileNotFoundError(f"GDAL could not open {gdal_geotiff_path}")
            
        warp_options = {
            'format': 'VRT',
            'dstSRS': 'EPSG:4326',
            'resampleAlg': gdal.GRA_Bilinear
        }

        # Create the local VRT XML
        warped_vrt_ds = gdal.Warp(local_vrt_path, src_ds, **warp_options)
            
        projection_wkt = warped_vrt_ds.GetProjection()
        spatial_ref = osr.SpatialReference(wkt=projection_wkt)
        datum_code = spatial_ref.GetAuthorityCode('DATUM')
        warped_vrt_ds = None 
        
        geotiff_parent = '/'.join(geotiff_prefix.split('/')[1:-1])
        s3_vrt_key = f"{geotiff_parent}/{geotiff_stem}.vrt"
        
        # Upload the individual VRT to S3
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_vrt_path, s3_bucket, s3_vrt_key)
        final_s3_vrt_path = f"/vsis3/{s3_bucket}/{s3_vrt_key}"

        clean_key = f"{datum_code}"

        return clean_key, final_s3_vrt_path, projection_wkt

    except Exception as e:
        raise RuntimeError(f'_process_single_bluetopo failed: {geotiff_prefix} - {str(e)}')
    finally:
        src_ds = None
        if os.path.exists(local_vrt_path):
            os.remove(local_vrt_path)

def _process_single_digitalcoast(params: list) -> list[str, str, str]:
    """Parallel process creating a Warped VRT for DigitalCoast"""

    geotiff_prefix, s3_bucket, all_crs_info = params

    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('AWS_S3_ENDPOINT', 's3.amazonaws.com')
    
    geotiff_stem = str(pathlib.Path(geotiff_prefix.split('/')[-1]).stem)
    gdal_geotiff_prefix = f'/vsis3/{geotiff_prefix}'
    
    with tempfile.NamedTemporaryFile(suffix=f"_{geotiff_stem}.vrt", delete=False) as tmp:
        local_vrt_path = tmp.name

    src_ds = None
    try:
        src_ds = gdal.Open(gdal_geotiff_prefix)
        if src_ds is None:
            raise FileNotFoundError(f"GDAL could not open {gdal_geotiff_prefix}")
            
        src_srs = src_ds.GetSpatialRef()
        src_srs_name = src_srs.GetName()
        
        override_srs = None
        if '+' in src_srs_name:
            try:
                srs_json = json.loads(src_srs.ExportToPROJJSON())
                comp_name = srs_json.get('components', [{}])[0].get('name', '')
                
                if '+' in comp_name:
                    horiz_name = comp_name.split(' + ')[0].lower().strip()
                    match = [cr.code for cr in all_crs_info if cr.name.lower() == horiz_name]
                    if match:
                        override_srs = f"EPSG:{match[0]}"
            except Exception:
                pass # Fall back to original CRS if resolution fails

        # Define Warp options
        warp_options = {
            'format': 'VRT',
            'dstSRS': 'EPSG:4326',
            'srcSRS': override_srs if override_srs is not None else src_srs.ExportToWkt(),
            'resampleAlg': gdal.GRA_Bilinear
        }

        # Create the local VRT XML
        warped_vrt_ds = gdal.Warp(local_vrt_path, src_ds, **warp_options)
            
        projection_wkt = warped_vrt_ds.GetProjection()
        spatial_ref = osr.SpatialReference(wkt=projection_wkt)
        datum_code = spatial_ref.GetAuthorityCode('DATUM') or "Unknown"
        warped_vrt_ds = None 
        
        # Upload the individual VRT to S3
        geotiff_parent = '/'.join(geotiff_prefix.split('/')[1:-1])
        s3_vrt_key = f"{geotiff_parent}/{geotiff_stem}.vrt"
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_vrt_path, s3_bucket, s3_vrt_key)
        final_s3_vrt_path = f"/vsis3/{s3_bucket}/{s3_vrt_key}"

        parts = geotiff_prefix.split('/')
        geotiff_folders = '/'.join(parts[1:])
        p_path = pathlib.Path(geotiff_folders)
        provider = p_path.parents[2].name if 'dem' in geotiff_folders else p_path.parents[3].name
        clean_key = f"{datum_code}_{provider}"

    finally:
        src_ds = None
        if os.path.exists(local_vrt_path):
            os.remove(local_vrt_path)

    return clean_key, final_s3_vrt_path, projection_wkt


class RasterVRTS3Engine(Engine):
    """Class for handling VRT creation of BlueTopo and DigitalCoast datasets"""

    def __init__(self, param_lookup) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.glob_lookup = {
            'elevation': '*[0-9].tiff',
            'uncertainty': '*_unc.tiff',
            'slope': '*_slope.tiff',
            'rugosity': '*_rugosity.tiff',
            'NCMP': '*.tif'
        }
        self.all_crs = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])

    def build_output_vrts(self, output_prefix: str, file_type: str, output_geotiffs: dict, temp_output_path: pathlib.Path) -> None:
        """Create main Master VRT file from individual S3 VRTs"""
        
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gdal.SetConfigOption('AWS_S3_ENDPOINT', 's3.amazonaws.com')
        gdal.SetConfigOption('GDAL_VRT_ENABLE_REPROJECTION', 'YES')

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')

        for crs_key, tile_dict in output_geotiffs.items():
            vrt_tiles = tile_dict['tiles'] 
            vrt_filename = temp_output_path / f'mosaic_{file_type}_{crs_key}.vrt'
            gdal.BuildVRT(str(vrt_filename), vrt_tiles)
  
            if not vrt_filename.exists():
                print(f"ERROR: Failed to create Master VRT at {vrt_filename}")
                continue

            s3_prefix = f'{output_prefix}/{vrt_filename.name}'
            print(f'Uploading Master VRT: {vrt_filename.name} to {s3_prefix}')
            s3_client.upload_file(str(vrt_filename), bucket_name, s3_prefix)

    def create_raster_vrts(self, file_type: str, ecoregion: str, data_type: str, skip_existing=False) -> None:
        """Create an output VRT from found .tif files"""

        output_prefix = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/testing/{ecoregion}/{get_config_item(data_type.upper(), 'SUBFOLDER')}/{data_type}"
        output_prefix_folder = f"testing/{ecoregion}/{get_config_item(data_type.upper(), 'SUBFOLDER')}/{data_type}"
        s3_files = s3fs.S3FileSystem()
        if data_type == 'BlueTopo':
            if skip_existing:
                if s3_files.glob(f'{output_prefix}/mosaic_{file_type}*.vrt'):
                    print(f'- skipping Bluetopo {file_type}')
                    return

            geotiffs = s3_files.glob(f"{output_prefix}/**/{self.glob_lookup[file_type]}")
            output_geotiffs = self.get_bluetopo_tifs(geotiffs)
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output_path = pathlib.Path(temp_dir) 
                self.build_output_vrts(output_prefix_folder, file_type, output_geotiffs, temp_output_path)
        else:
            provider_folders = s3_files.glob(f"{output_prefix}/*")
            for provider in provider_folders:
                provider_name = provider.split("/")[-1]
                if skip_existing:
                    if s3_files.glob(f'{output_prefix}/*{provider_name}.vrt'):
                        print(f'- skipping {provider_name}')
                        continue
                geotiffs = s3_files.glob(f"{provider}/**/{self.glob_lookup[file_type]}")
                output_geotiffs = self.get_digitalcoast_tifs(geotiffs)
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_output_path = pathlib.Path(temp_dir) 
                    self.build_output_vrts(output_prefix_folder, file_type, output_geotiffs, temp_output_path)

    def get_bluetopo_tifs(self, geotiffs: list[pathlib.Path]) -> dict[str]:
        """Dask processing to build BlueTopo tifs"""
        
        s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        params = [(gtif, s3_bucket, None) for gtif in geotiffs]

        futures = self.client.map(_process_single_bluetopo, params)
        results = self.client.gather(futures)

        output_geotiffs = {}
        for crs_code, s3_path, wkt in results:
            clean_key = str(crs_code).replace('/', '').replace(' ', '_')
            if clean_key not in output_geotiffs:
                output_geotiffs[clean_key] = {'crs': osr.SpatialReference(wkt=wkt), 'tiles': []}
            output_geotiffs[clean_key]['tiles'].append(s3_path)
            
        return output_geotiffs

    def get_digitalcoast_tifs(self, geotiffs: list) -> dict:
        """Dask processing to build DigitalCoast tifs"""

        s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        task_params = [(geotiff, s3_bucket, self.all_crs) for geotiff in geotiffs]
        futures = self.client.map(_process_single_digitalcoast, task_params)
        results = self.client.gather(futures)

        output_geotiffs = {}
        for clean_key, s3_vrt_path, wkt in results:
            if clean_key not in output_geotiffs:
                output_geotiffs[clean_key] = {
                    'crs': osr.SpatialReference(wkt=wkt), 
                    'tiles': []
                }
            output_geotiffs[clean_key]['tiles'].append(s3_vrt_path)
            
        return output_geotiffs
    
    def run(self, _: str, file_type: str, ecoregion: str, data_type: str, skip_existing=False) -> None:
        self.setup_dask(self.param_lookup['env'])
        self.create_raster_vrts(file_type, ecoregion, data_type, skip_existing)
        self.close_dask()

    def write_crs_to_raster(self, geotiff_raster: rxr.rioxarray, geotiff_srs: osr.SpatialReference) -> rxr.rioxarray:
        """Obtain the horizontal CRS of a compound CRS and write it to the raster"""

        proj_json = json.loads(geotiff_srs.ExportToPROJJSON())
        crs_name = proj_json['components'][0]['name']
        horizontal_name = crs_name.split(' + ')[0]  # Will this always work for compound CRS?
        clean_horiz_name = horizontal_name.lower().strip()
        epsg = [crs.code for crs in self.all_crs if crs.name.lower() == clean_horiz_name][0]
        geotiff_raster.rio.write_crs(f"EPSG:{epsg}", inplace=True)

        return geotiff_raster
