import pathlib
import s3fs
import tempfile
import boto3
import shutil
import os
from collections import defaultdict

from pyproj.database import query_crs_info
from pyproj.enums import PJType
from osgeo import gdal, osr
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


def _set_gdal_s3_options() -> None:
    """Configure GDAL /vsis3/ driver to resolve AWS IAM Role credentials from EC2 IMDS."""

    gdal.SetConfigOption('AWS_NO_SIGN_REQUEST', 'NO')
    gdal.SetConfigOption('AWS_EC2_METADATA_DISABLED', 'FALSE')
    
    gdal.SetConfigOption('AWS_REGION', 'us-east-2')
    
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('VSI_CACHE', 'FALSE')  # Prevents stale VSI 404 cache hits
    gdal.SetConfigOption('GDAL_HTTP_MERGE_CONSECUTIVE_RANGES', 'YES')
    gdal.SetConfigOption('GDAL_HTTP_MULTIPLEX', 'YES')


def _read_geotiff_metadata(params: list) -> dict[str]:
    """Read CRS metadata and extract pure integer EPSG codes across parallel Dask tasks."""

    _set_gdal_s3_options()

    raw_prefix, all_crs_info, data_type = params
    s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    
    # Standardize relative key (strip leading s3:// or bucket name)
    clean_path = raw_prefix.replace('s3://', '').lstrip('/')
    parts = clean_path.split('/')
    if parts[0] == s3_bucket:
        parts = parts[1:]
    
    relative_s3_key = '/'.join(parts)
    vsi_path = f'/vsis3/{s3_bucket}/{relative_s3_key}'
    
    ds = None
    try:
        ds = gdal.Open(vsi_path)
        if ds is None: 
            return None
            
        band = ds.GetRasterBand(1)
        nodata = band.GetNoDataValue()
        
        src_srs = ds.GetSpatialRef()
        epsg_code = None
        
        if src_srs:
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            src_srs.AutoIdentifyEPSG()
            
            auth_code = (
                src_srs.GetAuthorityCode("PROJCS") or 
                src_srs.GetAuthorityCode("GEOGCS") or 
                src_srs.GetAuthorityCode(None)
            )
            
            if auth_code and auth_code.isdigit():
                epsg_code = int(auth_code)
            else:
                srs_name = src_srs.GetName()
                if "16N" in srs_name or "16" in srs_name:
                    epsg_code = 6345
                elif "17N" in srs_name or "17" in srs_name:
                    epsg_code = 6346

        # Robust provider folder extraction
        provider = None
        for i, part in enumerate(parts):
            if part in ('DigitalCoast', 'Digital_Coast_Manual_Downloads') and (i + 1) < len(parts):
                provider = parts[i + 1]
                break
        
        if not provider:
            provider = parts[-3] if len(parts) >= 3 else parts[0]
            
        return {
            'bin_key': provider, 
            'vsi_path': vsi_path,
            'relative_s3_key': relative_s3_key,
            'nodata': nodata,
            'epsg': epsg_code
        }
    except Exception as e:
        print(f" - Error obtaining metadata: {relative_s3_key}: {e}")
        return None
    finally:
        ds = None


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
        """Master VRT Builder: Constructs a unified VRT referencing standardized native and reprojected S3 GeoTIFFs."""

        s3_client = boto3.client('s3')
        bucket_name = get_config_item('SHARED', 'OUTPUT_BUCKET')

        for provider, info in output_geotiffs.items():
            tifs = info['tiles'] 
            vrt_filename = temp_output_path / f'mosaic_{file_type}_{provider}.vrt'
            nodata = info.get('nodata_val', -999999)
            
            if data_type in ['DigitalCoast', 'Digital_Coast_Manual_Downloads']:
                vrt_options = gdal.BuildVRTOptions(
                    resampleAlg='near',
                    allowProjectionDifference=True,
                    srcNodata=nodata,
                    VRTNodata=nodata
                )
            else:
                vrt_options = gdal.BuildVRTOptions(
                    resampleAlg='bilinear',
                    allowProjectionDifference=True
                )

            # Build Master VRT directly against persistent /vsis3/ GeoTIFF targets
            gdal.BuildVRT(str(vrt_filename), tifs, options=vrt_options)

            if vrt_filename.exists():
                s3_key = f'{s3_output_path}/{vrt_filename.name}'
                print(f' - Uploading Master VRT to: {s3_key}')
                s3_client.upload_file(str(vrt_filename), bucket_name, s3_key)

    def get_digitalcoast_geotiffs(self, geotiffs: list, data_type: str, temp_dir: pathlib.Path) -> dict:
        """Bins tiles by provider, selects majority CRS, reprojects ONLY non-matching tiles to 'reprojected/'."""

        _set_gdal_s3_options()
        s3_client = boto3.client('s3')
        s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        # Filter out existing reprojected files upfront
        clean_geotiffs = [gt for gt in geotiffs if '/reprojected/' not in gt]

        task_params = [(gtif, self.all_crs, data_type) for gtif in clean_geotiffs]
        results = [r for r in self.client.gather(self.client.map(_read_geotiff_metadata, task_params)) if r is not None]

        provider_bins = defaultdict(list)
        provider_nodata = defaultdict(lambda: None)

        for res in results:
            provider = res['bin_key']
            provider_bins[provider].append(res)
            
            if provider_nodata[provider] is None and res['nodata'] is not None:
                provider_nodata[provider] = res['nodata']

        output_geotiffs = {}

        for provider, tile_list in provider_bins.items():
            if not tile_list:
                continue

            # Tally EPSG integers
            epsg_counts = defaultdict(int)
            for t in tile_list:
                epsg_counts[t['epsg']] += 1
            
            # Select target EPSG integer by majority count
            primary_epsg = max(epsg_counts.keys(), key=lambda e: epsg_counts[e])
            primary_crs = f"EPSG:{primary_epsg}"
            nodata_val = provider_nodata[provider] if provider_nodata[provider] is not None else -999999
            
            print(f"\n -> [{provider}] Target Primary CRS set to: {primary_crs}")
            print(f" -> EPSG Counts: {dict(epsg_counts)}")

            target_srs = osr.SpatialReference()
            target_srs.ImportFromEPSG(primary_epsg)
            target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            print(f"\n[DIAGNOSTIC] Provider: {provider}")
            print(f"[DIAGNOSTIC] Total tiles in list: {len(tile_list)}")
            print(f"[DIAGNOSTIC] EPSG Counts: {dict(epsg_counts)}")
            print(f"[DIAGNOSTIC] Winning Primary EPSG: {primary_epsg}")
            
            # Print the first 3 native vs minority comparisons
            for t in tile_list[:3]:
                print(f"  Tile EPSG: {t['epsg']} | Target EPSG: {primary_epsg} | Match: {t['epsg'] == primary_epsg}")

            final_vsi_tiles = []
            for tile in tile_list:
                tile_epsg = tile['epsg']
                vsi_path = tile['vsi_path'] # Pure /vsis3/bucket/key URI
                # Fallback to 's3_prefix' if 'relative_s3_key' is not set
                relative_s3_key = tile.get('relative_s3_key') or tile.get('s3_prefix')
                
                parts = relative_s3_key.split('/')
                filename = parts[-1]
                parent_prefix = '/'.join(parts[:-1])

                # Integer check: 6345 == 6345
                if tile_epsg == primary_epsg:
                    # NATIVE MATCH: Use native /vsis3/ URI
                    final_vsi_tiles.append(vsi_path)
                else:
                    # MINORITY TILE: Point to /reprojected/ subfolder
                    reprojected_s3_key = f"{parent_prefix}/reprojected/{filename}"
                    reprojected_vsi_path = f"/vsis3/{s3_bucket}/{reprojected_s3_key}"
                    
                    try:
                        s3_client.head_object(Bucket=s3_bucket, Key=reprojected_s3_key)
                        print(f" - Found existing reprojected GeoTIFF: {reprojected_s3_key}")
                    except Exception:
                        print(f" - Reprojecting {filename} (EPSG:{tile_epsg} -> EPSG:{primary_epsg}) to S3: {reprojected_s3_key}")
                        
                        src_srs = osr.SpatialReference()
                        src_srs.ImportFromEPSG(tile_epsg)
                        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

                        local_tif_path = temp_dir / filename
                        warp_options = gdal.WarpOptions(
                            format='GTiff',
                            srcSRS=src_srs,
                            dstSRS=target_srs,
                            resampleAlg='near',
                            srcNodata=nodata_val,
                            dstNodata=nodata_val,
                            creationOptions=['COMPRESS=LZW', 'TILED=YES']
                        )
                        
                        warped_ds = gdal.Warp(str(local_tif_path), vsi_path, options=warp_options)
                        warped_ds = None
                        
                        s3_client.upload_file(str(local_tif_path), s3_bucket, reprojected_s3_key)
                        
                        if local_tif_path.exists():
                            os.remove(local_tif_path)
                    
                    final_vsi_tiles.append(reprojected_vsi_path)

            output_geotiffs[provider] = {
                'tiles': final_vsi_tiles,
                'nodata_val': provider_nodata[provider],
                'primary_crs': primary_crs
            }

        return output_geotiffs
    
    def run(self, outputs: str, file_type: str, ecoregion: str, data_type: str, output_prefix: str="", manual_downloads: bool=False) -> None:
        """Main cloud execution method routing control using structural parameters"""

        _set_gdal_s3_options()
        self.setup_dask(self.param_lookup['env'])
        local_tmp_path = pathlib.Path.home() / "local_tmp_dir"
        local_tmp_path.mkdir(parents=True, exist_ok=True)
        gdal.SetConfigOption('CPL_TMPDIR', str(local_tmp_path))
        
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
                # if '2019_DEM_NOAA_NGS_69338' in provider_path:
                print(f'running: {provider_path}')
                geotiffs = [
                    gt for gt in s3_files.glob(f"{provider_path}/**/{self.glob_lookup[file_type]}") 
                    if '/reprojected/' not in gt
                ]
                
                if not geotiffs: 
                    continue
                
                with tempfile.TemporaryDirectory(dir=local_tmp_path) as td:
                    temp_path = pathlib.Path(td)
                    output_geotiffs = self.get_digitalcoast_geotiffs(geotiffs, current_datatype, temp_path)
                    self.build_output_vrts(current_out_path, file_type, output_geotiffs, temp_path, current_datatype)

        shutil.rmtree(local_tmp_path)
        self.close_dask()