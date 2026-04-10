import pathlib
import s3fs
import tempfile
import boto3
import json
import os

from pyproj.database import query_crs_info
from pyproj.enums import PJType
from osgeo import gdal, osr
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


def _process_single_bluetopo(params: list) -> tuple[str, str, str]:
    """Original BlueTopo logic: Creates individual Warped VRTs (EPSG:4326) on S3"""

    geotiff_prefix, s3_bucket, _ = params
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    
    geotiff_stem = str(pathlib.Path(geotiff_prefix).stem)
    vsi_geotiff_path = f'/vsis3/{geotiff_prefix}'
    
    with tempfile.NamedTemporaryFile(suffix=f"_{geotiff_stem}.vrt", delete=False) as tmp:
        local_vrt_path = tmp.name

    src_ds = None
    try:
        src_ds = gdal.Open(vsi_geotiff_path)
        if src_ds is None:
            raise FileNotFoundError(f"GDAL could not open {vsi_geotiff_path}")
            
        warp_options = {
            'format': 'VRT',
            'dstSRS': 'EPSG:4326',
            'resampleAlg': gdal.GRA_Bilinear
        }

        warped_vrt_ds = gdal.Warp(local_vrt_path, src_ds, **warp_options)
        projection_wkt = warped_vrt_ds.GetProjection()
        spatial_ref = osr.SpatialReference(wkt=projection_wkt)
        datum_code = spatial_ref.GetAuthorityCode('DATUM')
        warped_vrt_ds = None 
        
        geotiff_parent = '/'.join(geotiff_prefix.split('/')[1:-1])
        s3_vrt_key = f"{geotiff_parent}/{geotiff_stem}.vrt"
        
        boto3.client('s3').upload_file(local_vrt_path, s3_bucket, s3_vrt_key)
        final_s3_vrt_path = f"/vsis3/{s3_bucket}/{s3_vrt_key}"

        return str(datum_code), final_s3_vrt_path, projection_wkt

    except Exception as e:
        raise RuntimeError(f'_process_single_bluetopo failed: {geotiff_prefix} - {str(e)}')
    finally:
        src_ds = None
        if os.path.exists(local_vrt_path):
            os.remove(local_vrt_path)


def _read_geotiff_metadata(params: list):
    """
    Dask Worker: Scout that handles NAD83(2011) and other pedantic CRS strings.
    Restored your original PROJJSON component matching logic!
    """

    geotiff_prefix, all_crs_info, data_folder = params
    vsi_path = f'/vsis3/{geotiff_prefix}'
    
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    
    ds = None
    try:
        ds = gdal.Open(vsi_path)
        if ds is None: 
            return None
            
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        
        src_srs = ds.GetSpatialRef()
        # Ensure we are using a consistent axis order (Long, Lat)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        bin_id = src_srs.GetAuthorityCode(None)
        if not bin_id:
            try:
                srs_json = json.loads(src_srs.ExportToPROJJSON())
                components = srs_json.get('components', [{}])
                comp_name = components[0].get('name', '')
                horizontal_name = comp_name.split(' + ')[0].lower().strip()
                match = [cr.code for cr in all_crs_info if cr.name.lower() == horizontal_name]
                if match:
                    bin_id = match[0]
            except:
                pass

        if not bin_id:
            bin_id = src_srs.GetAuthorityCode('DATUM') or src_srs.GetName().replace(" ", "_")
            
        parts = geotiff_prefix.split('/')
        try:
            dc_index = parts.index('DigitalCoast_manual_downloads') if data_folder else parts.index('DigitalCoast')
            provider = parts[dc_index + 1]
        except (ValueError, IndexError):
            provider = parts[-4]
            
        return {
            'bin_key': f"{bin_id}_{provider}",
            'vsi_path': vsi_path,
            'nodata': nodata,
            'wkt': src_srs.ExportToWkt()
        }
    except Exception as e:
        print(f" - Error obtaining metdata: {geotiff_prefix}: {e}")
        return None
    finally:
        ds = None


class RasterVRTS3Engine(Engine):
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

    def build_output_vrts(self, output_prefix: str, file_type: str, output_geotiffs: dict, temp_output_path: pathlib.Path, data_type: str) -> None:
        """Master VRT Builder: Custom logic per data_type"""

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')

        for bin_key, info in output_geotiffs.items():
            tifs = info['tiles'] 
            vrt_filename = temp_output_path / f'mosaic_{file_type}_{bin_key}.vrt'
            
            if data_type == 'DigitalCoast':
                # Native CRS of geotiffs
                options = gdal.BuildVRTOptions(
                    resampleAlg='near', 
                    srcNodata=info.get('nodata_val'),
                    allowProjectionDifference=True
                )
            else:
                # Mosaic of 4326 VRTs
                options = gdal.BuildVRTOptions(
                    resampleAlg='bilinear',
                    allowProjectionDifference=True
                )

            gdal.BuildVRT(str(vrt_filename), tifs, options=options)

            if vrt_filename.exists():
                s3_key = f'{output_prefix}/{vrt_filename.name}'
                print(f' - Uploading {data_type} Master VRT: {vrt_filename.name}')
                s3_client.upload_file(str(vrt_filename), bucket_name, s3_key)

    def get_bluetopo_tifs(self, geotiffs: list) -> dict:
        """Get all BlueTopo VRT files warped to 4326"""

        s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        params = [(gtif, s3_bucket, None) for gtif in geotiffs]
        results = self.client.gather(self.client.map(_process_single_bluetopo, params))

        output_geotiffs = {}
        for crs_code, s3_path, wkt in results:
            clean_key = str(crs_code).replace('/', '').replace(' ', '_')
            if clean_key not in output_geotiffs:
                output_geotiffs[clean_key] = {'crs': osr.SpatialReference(wkt=wkt), 'tiles': []}
            output_geotiffs[clean_key]['tiles'].append(s3_path)
        return output_geotiffs

    def get_digitalcoast_geotiffs(self, geotiffs: list, data_folder: str) -> dict:
        """Get all DigitalCoast tifs with original CRS"""

        task_params = [(gtif, self.all_crs, data_folder) for gtif in geotiffs]
        results = [r for r in self.client.gather(self.client.map(_read_geotiff_metadata, task_params)) if r is not None]

        output_geotiffs = {}
        for res in results:
            key = res['bin_key']
            if key not in output_geotiffs:
                output_geotiffs[key] = {'tiles': [], 'nodata_val': res['nodata']}
            output_geotiffs[key]['tiles'].append(res['vsi_path'])
        return output_geotiffs

    def run(self, outputs: str, file_type: str, ecoregion: str, data_type: str, data_folder=False, skip_existing=False) -> None:
        self.setup_dask(self.param_lookup['env'])
        
        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        sub = get_config_item(data_type.upper(), 'SUBFOLDER')
        base_s3 = f"s3://{bucket}/{ecoregion}/{sub}/{data_folder if data_folder else data_type}"
        output_prefix = f"{ecoregion}/{sub}/{data_folder if data_folder else data_type}"

        if data_type == 'BlueTopo':
            geotiffs = s3_files.glob(f"{base_s3}/**/{self.glob_lookup[file_type]}")
            if geotiffs:
                output_geotiffs = self.get_bluetopo_tifs(geotiffs)
                with tempfile.TemporaryDirectory() as td:
                    self.build_output_vrts(output_prefix, file_type, output_geotiffs, pathlib.Path(td), data_type)
        else:
            provider_folders = s3_files.glob(f"{base_s3}/*")
            for provider_path in provider_folders:
                geotiffs = s3_files.glob(f"{provider_path}/**/{self.glob_lookup[file_type]}")
                if not geotiffs: 
                    continue
                output_geotiffs = self.get_digitalcoast_geotiffs(geotiffs, data_folder)
                with tempfile.TemporaryDirectory() as td:
                    self.build_output_vrts(output_prefix, file_type, output_geotiffs, pathlib.Path(td), data_type)

        self.close_dask()