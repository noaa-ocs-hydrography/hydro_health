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


def _clean(s: str) -> str:
    """Helper to normalize strings for comparison (removes non-breaking spaces, etc.)"""

    if not s: 
        return ""
    # Replaces non-breaking spaces (\xa0) with standard spaces and strips whitespace
    return " ".join(s.split()).lower().strip()


def _check_equal_crs(wkt1: str, wkt2: str) -> bool:
    """Check if two WKT strings represent the exact same CRS."""

    if not wkt1 or not wkt2:
        return False
    srs1, srs2 = osr.SpatialReference(), osr.SpatialReference()
    srs1.ImportFromWkt(wkt1)
    srs2.ImportFromWkt(wkt2)
    return bool(srs1.IsSame(srs2))


def _process_single_bluetopo(params: list) -> tuple[str, str, str]:
    """Original BlueTopo logic: Creates individual Warped VRTs (EPSG:4326) on S3"""

    _set_gdal_s3_options()
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
            'resampleAlg': gdal.GRA_Bilinear,
            'srcNodata': -999999,
            'dstNodata': -999999,  # Ensures the "empty" space in the reprojected VRT is transparent
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE'] # Optional: helps with clean edges
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


def _read_geotiff_metadata(params: list) -> dict[str]:
    """Read the CRS metadata from each geotiff for use with VRT output"""

    _set_gdal_s3_options()

    geotiff_prefix, all_crs_info, data_type = params
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
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        
        bin_id = src_srs.GetAuthorityCode(None)
        if not bin_id:
            try:
                srs_json = json.loads(src_srs.ExportToPROJJSON())
                components = srs_json.get('components', [{}])
                comp_name = components[0].get('name', '')
                horizontal_name = _clean(comp_name.split(' + ')[0])
                match = [cr.code for cr in all_crs_info if _clean(cr.name) == horizontal_name]
                if match:
                    bin_id = match[0]
            except:
                pass

        if not bin_id:
            fallback_name = src_srs.GetName()
            bin_id = src_srs.GetAuthorityCode('DATUM') or _clean(fallback_name).replace(" ", "_")
            
        parts = geotiff_prefix.split('/')
        try:
            # Handle standard data types as well as Manual Downloads dynamically
            dc_index = [i for i, part in enumerate(parts) if part in (data_type, 'Digital_Coast_Manual_Downloads')][0]
            provider = parts[dc_index + 1]
        except (ValueError, IndexError):
            provider = parts[-4]
            
        return {
            'bin_key': provider, 
            'vsi_path': vsi_path,
            'nodata': nodata,
            'wkt': src_srs.ExportToWkt()
        }
    except Exception as e:
        print(f" - Error obtaining metadata: {geotiff_prefix}: {e}")
        return None
    finally:
        ds = None


def _set_gdal_s3_options() -> None:
    """Set the default S3 options for GDAL usage"""

    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE') # Depends on your S3 setup
    gdal.SetConfigOption('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
    gdal.SetConfigOption('GDAL_HTTP_MULTIPLEX', 'YES')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('VSI_CACHE_SIZE', '10000000') # 10MB cache
    

class RasterVRTS3Engine(Engine):
    """Class for handling S3 network-based VRT creation and re-uploads"""

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

    def build_output_vrts(self, s3_output_path: str, file_type: str, output_geotiffs: dict, temp_output_path: pathlib.Path, data_type: str) -> None:
        """Master VRT Builder: Custom logic per data_type"""

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')

        for bin_key, info in output_geotiffs.items():
            tifs = info['tiles'] 
            vrt_filename = temp_output_path / f'mosaic_{file_type}_{bin_key}.vrt'
            
            if data_type in ['DigitalCoast', 'Digital_Coast_Manual_Downloads']:
                # Force the VRT to use the first tile's WKT and allow differences
                options = gdal.BuildVRTOptions(
                    resampleAlg='near', 
                    srcNodata=info.get('nodata_val'),
                    VRTNodata=info.get('nodata_val'),
                    addAlpha=True,
                    outputSRS=info.get('primary_wkt')
                )
            else:
                # Mosaic of 4326 VRTs
                options = gdal.BuildVRTOptions(
                    resampleAlg='bilinear',
                    allowProjectionDifference=True
                )

            gdal.BuildVRT(str(vrt_filename), tifs, options=options)

            if vrt_filename.exists():
                s3_key = f'{s3_output_path}/{vrt_filename.name}'
                print(f' - Uploading {data_type} Master VRT to: {s3_key}')
                s3_client.upload_file(str(vrt_filename), bucket_name, s3_key)

    def get_bluetopo_tifs(self, geotiffs: list) -> dict:
        """Get all BlueTopo VRT files warped to 4326"""

        s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        params = [(gtif, s3_bucket, None) for gtif in geotiffs]
        results = self.client.gather(self.client.map(_process_single_bluetopo, params))

        output_geotiffs = {}
        for crs_code, s3_path, wkt in results:
            # Normalized key to ensure consistency
            clean_key = _clean(str(crs_code)).replace('/', '').replace(' ', '_')
            if clean_key not in output_geotiffs:
                output_geotiffs[clean_key] = {'crs': osr.SpatialReference(wkt=wkt), 'tiles': []}
            output_geotiffs[clean_key]['tiles'].append(s3_path)
        return output_geotiffs

    def get_digitalcoast_geotiffs(self, geotiffs: list, data_type: str) -> dict:
        """Get all DigitalCoast tifs and reproject mismatched CRS tiles into memory VRTs."""

        task_params = [(gtif, self.all_crs, data_type) for gtif in geotiffs]
        results = [r for r in self.client.gather(self.client.map(_read_geotiff_metadata, task_params)) if r is not None]

        output_geotiffs = {}
        
        for res in results:
            key = res['bin_key']
            tile_wkt = res['wkt']
            vsi_path = res['vsi_path']
            
            if key not in output_geotiffs:
                # Set the first tile's CRS (e.g., EPSG:6346) as the master projection for this bin
                output_geotiffs[key] = {
                    'tiles': [], 
                    'nodata_val': res['nodata'],
                    'primary_wkt': tile_wkt
                }
            
            primary_wkt = output_geotiffs[key]['primary_wkt']
            
            if _check_equal_crs(tile_wkt, primary_wkt):
                output_geotiffs[key]['tiles'].append(vsi_path)
            else:
                # Need to warp unique CRS files into their own temp VRT
                warped_vrt_path = f"/vsimem/reprojected_{os.path.basename(vsi_path)}.vrt"
                warp_options = gdal.WarpOptions(
                    format='VRT',
                    srcSRS=tile_wkt,
                    dstSRS=primary_wkt,
                    resampleAlg='near',
                    srcNodata=res['nodata'],
                    dstNodata=res['nodata']
                )
                gdal.Warp(warped_vrt_path, vsi_path, options=warp_options)
                
                # Pass the reprojected VRT path to the master list
                output_geotiffs[key]['tiles'].append(warped_vrt_path)

        return output_geotiffs
    
    def run(self, outputs: str, file_type: str, ecoregion: str, data_type: str, output_prefix: str="", manual_downloads: bool=False) -> None:
        """Main cloud execution method routing control using structural parameters"""

        _set_gdal_s3_options()
        self.setup_dask(self.param_lookup['env'])
        
        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        sub = get_config_item(data_type.upper(), 'SUBFOLDER')
        
        prefix_segment = f"{output_prefix}/" if output_prefix else ""
        base_s3 = f"s3://{bucket}/{prefix_segment}{ecoregion}/{sub}/{data_type}"
        s3_output_path = f"{prefix_segment}{ecoregion}/{sub}/{data_type}"

        if data_type == 'BlueTopo':
            geotiffs = s3_files.glob(f"{base_s3}/**/{self.glob_lookup[file_type]}")
            if geotiffs:
                output_geotiffs = self.get_bluetopo_tifs(geotiffs)
                with tempfile.TemporaryDirectory() as td:
                    self.build_output_vrts(s3_output_path, file_type, output_geotiffs, pathlib.Path(td), data_type)
        else:
            providers_to_process = [(folder, data_type, s3_output_path) for folder in s3_files.glob(f"{base_s3}/*")]
            if manual_downloads:
                manual_data_type = 'Digital_Coast_Manual_Downloads'
                manual_base_s3 = f"s3://{bucket}/{prefix_segment}{ecoregion}/{sub}/{manual_data_type}"
                manual_s3_output_path = f"{prefix_segment}{ecoregion}/{sub}/{manual_data_type}"
                
                manual_folders = s3_files.glob(f"{manual_base_s3}/*")
                providers_to_process.extend([(folder, manual_data_type, manual_s3_output_path) for folder in manual_folders])

            for provider_path, current_datatype, current_out_path in providers_to_process:
                geotiffs = s3_files.glob(f"{provider_path}/**/{self.glob_lookup[file_type]}")
                if not geotiffs: 
                    continue
                
                output_geotiffs = self.get_digitalcoast_geotiffs(geotiffs, current_datatype)
                with tempfile.TemporaryDirectory() as td:
                    self.build_output_vrts(current_out_path, file_type, output_geotiffs, pathlib.Path(td), current_datatype)

        self.close_dask()