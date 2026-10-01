import pathlib
from collections import defaultdict
from pyproj.database import query_crs_info
from pyproj.enums import PJType
from osgeo import gdal, osr
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


class RasterVRTEngine(Engine):
    """Class for handling VRT creation keeping native GeoTIFF CRS locally"""

    def __init__(self, param_lookup) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.glob_lookup = {
            'elevation': '*[0-9].tiff',
            'uncertainty': '*_unc.tiff',
            'survey_end_date': '*survey_end_date.tiff',
            'slope': '*_slope.tiff',
            'catzoc_decay_all': '*ISS_all*.tiff',
            'catzoc_decay_latest': '*ISS_latest*.tiff',
            'NCMP': '*.tif'
        }
        self.all_crs = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])

    def get_local_bluetopo_geotiffs(self, geotiffs: list[pathlib.Path]) -> dict:
        """Groups BlueTopo tiles strictly by their native EPSG UTM CRS code."""

        utm_bins = defaultdict(list)

        for geotiff_path in geotiffs:
            ds = gdal.Open(str(geotiff_path))
            if ds is None:
                continue

            epsg_code = None
            try:
                src_srs = ds.GetSpatialRef()
                if src_srs:
                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    try:
                        src_srs.AutoIdentifyEPSG()
                        auth_code = (
                            src_srs.GetAuthorityCode("PROJCS") or 
                            src_srs.GetAuthorityCode("GEOGCS") or 
                            src_srs.GetAuthorityCode(None)
                        )
                        if auth_code and auth_code.isdigit():
                            epsg_code = int(auth_code)
                    except Exception:
                        pass

                    if not epsg_code:
                        raw_code = (
                            src_srs.GetAuthorityCode("PROJCS") or 
                            src_srs.GetAuthorityCode("GEOGCS") or 
                            src_srs.GetAuthorityCode(None)
                        )
                        if raw_code and raw_code.isdigit():
                            epsg_code = int(raw_code)

                crs_key = f"EPSG_{epsg_code}" if epsg_code else "UNKNOWN_CRS"
                utm_bins[crs_key].append(str(geotiff_path))

            except Exception as e:
                print(f" - Error obtaining SRS for {geotiff_path}: {e}")
            finally:
                ds = None

        output_dict = {}
        for crs_key, tile_paths in utm_bins.items():
            print(f" -> BlueTopo Local Bin [{crs_key}]: {len(tile_paths)} tiles")
            output_dict[crs_key] = {
                'tiles': tile_paths,
                'nodata_val': -9999.0
            }

        return output_dict

    def get_local_digitalcoast_geotiffs(self, geotiffs: list[pathlib.Path]) -> dict:
        """Reads metadata from local files to build DigitalCoast provider bins."""

        provider_bins = defaultdict(list)
        provider_nodata = defaultdict(lambda: None)

        for geotiff_path in geotiffs:
            ds = gdal.Open(str(geotiff_path))
            if ds is None: 
                continue
                
            try:
                band = ds.GetRasterBand(1)
                nodata = band.GetNoDataValue()
                
                src_srs = ds.GetSpatialRef()
                epsg_code = None
                raw_wkt = None

                if src_srs:
                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    raw_wkt = src_srs.ExportToWkt()
                    try:
                        src_srs.AutoIdentifyEPSG()
                        auth_code = (
                            src_srs.GetAuthorityCode("PROJCS") or 
                            src_srs.GetAuthorityCode("GEOGCS") or 
                            src_srs.GetAuthorityCode(None)
                        )
                        if auth_code and auth_code.isdigit():
                            epsg_code = int(auth_code)
                    except Exception:
                        pass

                parts = geotiff_path.parts
                try:
                    if 'Digital_Coast_Manual_Downloads' in parts:
                        dc_index = parts.index('Digital_Coast_Manual_Downloads')
                    elif 'DigitalCoast' in parts:
                        dc_index = parts.index('DigitalCoast')
                    else:
                        raise ValueError
                    provider = parts[dc_index + 1]
                except (ValueError, IndexError):
                    provider = parts[-4] if len(parts) >= 4 else "UnknownProvider"
                    
                provider_bins[provider].append({
                    'path': str(geotiff_path),
                    'epsg': epsg_code,
                    'wkt': raw_wkt,
                    'nodata': nodata
                })

                if provider_nodata[provider] is None and nodata is not None:
                    provider_nodata[provider] = nodata

            except Exception as e:
                print(f" - Error obtaining metadata for {geotiff_path}: {e}")
            finally:
                ds = None
                
        output_geotiffs = {}
        for provider, tile_list in provider_bins.items():
            if not tile_list:
                continue

            epsg_counts = defaultdict(int)
            for t in tile_list:
                key = t['epsg'] if t['epsg'] is not None else 'UNKNOWN_WKT'
                epsg_counts[key] += 1
            
            primary_key = max(epsg_counts.keys(), key=lambda e: epsg_counts[e])
            primary_crs = f"EPSG:{primary_key}" if isinstance(primary_key, int) else "CUSTOM_WKT"
            
            output_geotiffs[provider] = {
                'tiles': [t['path'] for t in tile_list],
                'nodata_val': provider_nodata[provider],
                'primary_crs': primary_crs
            }

        return output_geotiffs

    def build_output_vrts(self, outputs: pathlib.Path, file_type: str, output_geotiffs: dict, data_type: str) -> None:
        """Create Master VRT files with companion overview pyramids locally."""

        for bin_key, info in output_geotiffs.items():
            tifs = [str(t) for t in info['tiles']]
            if not tifs:
                continue

            vrt_filename = outputs / f'mosaic_{file_type}_{bin_key}.vrt'
            nodata = info.get('nodata_val', -9999.0)
            if nodata is None:
                nodata = -9999.0

            if data_type == 'BlueTopo':
                # Build native UTM VRT strictly within matched CRS group
                vrt_options = gdal.BuildVRTOptions(
                    resampleAlg='bilinear',
                    allowProjectionDifference=False,
                    srcNodata=nodata,
                    VRTNodata=nodata
                )
                gdal.BuildVRT(str(vrt_filename), tifs, options=vrt_options)

            elif data_type in ['DigitalCoast', 'Digital_Coast_Manual_Downloads']:
                vrt_options = gdal.BuildVRTOptions(
                    resampleAlg='near',
                    allowProjectionDifference=True,
                    srcNodata=nodata,
                    VRTNodata=nodata
                )
                gdal.BuildVRT(str(vrt_filename), tifs, options=vrt_options)

            else:
                vrt_options = gdal.BuildVRTOptions(
                    resampleAlg='bilinear',
                    allowProjectionDifference=True,
                    srcNodata=nodata,
                    VRTNodata=nodata
                )
                gdal.BuildVRT(str(vrt_filename), tifs, options=vrt_options)

            if vrt_filename.exists():
                print(f' - Building local VRT Overviews for {vrt_filename.name}...')
                vrt_ds = gdal.Open(str(vrt_filename), gdal.GA_Update)
                if vrt_ds is not None:
                    # Creates local .vrt.ovr sidecar file
                    vrt_ds.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64])
                    vrt_ds = None

            print(f'- Finished Master VRT & Overviews: {vrt_filename.name}')

    def run(self, output_folder: str, file_type: str, ecoregion: str, data_type: str, output_prefix: str="", data_folder: str="", manual_downloads: bool=False) -> None:
        """Main method for running VRT Engine locally"""
        
        sub = get_config_item(data_type.upper(), 'SUBFOLDER')
        
        if output_prefix:
            outputs = pathlib.Path(output_folder) / output_prefix / ecoregion / sub / (data_folder if data_folder else data_type)
        else:
            outputs = pathlib.Path(output_folder) / ecoregion / sub / (data_folder if data_folder else data_type)

        outputs.mkdir(parents=True, exist_ok=True)

        if data_type == 'BlueTopo':
            raw_geotiffs = list(outputs.rglob(self.glob_lookup[file_type]))
            
            if file_type == 'elevation':
                geotiffs = [
                    g for g in raw_geotiffs 
                    if not any(x in g.name.lower() for x in ['_unc', '_slope', 'decay', 'iss'])
                ]
            else:
                geotiffs = raw_geotiffs

            if geotiffs:
                output_geotiffs = self.get_local_bluetopo_geotiffs(geotiffs)
                self.build_output_vrts(outputs, file_type, output_geotiffs, data_type)
        else:
            provider_folders = [f for f in outputs.glob('*') if f.is_dir() and 'unused_providers' not in f.name]
            for provider_path in provider_folders:
                geotiffs = [g for g in provider_path.rglob(self.glob_lookup[file_type]) if not g.name.startswith('mask_')]
                if not geotiffs: 
                    continue
                output_geotiffs = self.get_local_digitalcoast_geotiffs(geotiffs)
                self.build_output_vrts(outputs, file_type, output_geotiffs, data_type)