import pathlib
import json
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
            'slope': '*_slope.tiff',
            'rugosity': '*_rugosity.tiff',
            'catzoc_decay_all': '*decay_all*.tiff',
            'catzoc_decay_latest': '*decay_latest*.tiff',
            'NCMP': '*.tif'
        }
        self.all_crs = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])

    def get_local_bluetopo_geotiffs(self, geotiffs: list[pathlib.Path]) -> dict:
        """Groups BlueTopo tiles by their native CRS/UTM Zone bin_key"""
        output_geotiffs = {}

        for geotiff_path in geotiffs:
            ds = gdal.Open(str(geotiff_path))
            if ds is None:
                continue

            try:
                src_srs = ds.GetSpatialRef()
                if src_srs:
                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                    bin_id = src_srs.GetAuthorityCode(None) or src_srs.GetAuthorityCode('DATUM') or "unknown_crs"
                else:
                    bin_id = "unknown_crs"

                if bin_id not in output_geotiffs:
                    output_geotiffs[bin_id] = {'tiles': []}

                output_geotiffs[bin_id]['tiles'].append(geotiff_path)

            except Exception as e:
                print(f" - Error obtaining metadata for {geotiff_path}: {e}")
            finally:
                ds = None

        return output_geotiffs

    def get_local_digitalcoast_geotiffs(self, geotiffs: list[pathlib.Path]) -> dict:
        """Reads metadata from local files to build DigitalCoast bins"""
        output_geotiffs = {}
        
        for geotiff_path in geotiffs:
            ds = gdal.Open(str(geotiff_path))
            if ds is None: 
                continue
                
            try:
                band = ds.GetRasterBand(1)
                nodata = band.GetNoDataValue()
                
                src_srs = ds.GetSpatialRef()
                if src_srs:
                    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                
                bin_id = src_srs.GetAuthorityCode(None) if src_srs else None
                if not bin_id and src_srs:
                    try:
                        srs_json = json.loads(src_srs.ExportToPROJJSON())
                        components = srs_json.get('components', [{}])
                        comp_name = components[0].get('name', '')
                        horizontal_name = " ".join(comp_name.split(' + ')[0].split()).lower().strip()
                        match = [cr.code for cr in self.all_crs if " ".join(cr.name.split()).lower().strip() == horizontal_name]
                        if match:
                            bin_id = match[0]
                    except:
                        pass

                if not bin_id and src_srs:
                    fallback_name = src_srs.GetName() or ""
                    clean_fallback = " ".join(fallback_name.split()).lower().strip().replace(" ", "_")
                    bin_id = src_srs.GetAuthorityCode('DATUM') or clean_fallback
                elif not bin_id:
                    bin_id = "unknown_provider"

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
                    
                if provider not in output_geotiffs:
                    output_geotiffs[provider] = {
                        'tiles': [], 
                        'nodata_val': nodata,
                        'wkt': src_srs.ExportToWkt() if src_srs else None
                    }
                output_geotiffs[provider]['tiles'].append(str(geotiff_path))
                
            except Exception as e:
                print(f" - Error obtaining metadata for {geotiff_path}: {e}")
            finally:
                ds = None
                
        return output_geotiffs

    def build_output_vrts(self, outputs: pathlib.Path, file_type: str, output_geotiffs: dict, data_type: str) -> None:
        """Create Master VRT files directly referencing native source files."""

        for bin_key, info in output_geotiffs.items():
            tifs = [str(t) for t in info['tiles']]
            vrt_filename = outputs / f'mosaic_{file_type}_{bin_key}.vrt'
            
            if data_type == 'DigitalCoast':
                options = gdal.BuildVRTOptions(
                    resampleAlg='bilinear',
                    resolution='highest',
                    srcNodata=info.get('nodata_val'),
                    VRTNodata=info.get('nodata_val'),
                    addAlpha=True,
                    allowProjectionDifference=True,
                    outputSRS=info.get('wkt')
                )
                gdal.BuildVRT(str(vrt_filename), tifs, options=options)
            else:
                # Warp the source GeoTIFF paths directly into a single EPSG:4326 VRT
                warp_options = gdal.WarpOptions(
                    format='VRT',
                    dstSRS='EPSG:4326',
                    resampleAlg=gdal.GRA_Bilinear,
                    srcNodata=-9999.0,
                    dstNodata=-9999.0
                )
                gdal.Warp(str(vrt_filename), tifs, options=warp_options)

            print(f'- Finished Master VRT: {vrt_filename.name}')

    def run(self, output_folder: str, file_type: str, ecoregion: str, data_type: str, output_prefix: str="", data_folder: str="", manual_downloads: bool=False) -> None:
        """Main method for running VRT Engine locally"""
        
        sub = get_config_item(data_type.upper(), 'SUBFOLDER')
        
        if output_prefix:
            outputs = pathlib.Path(output_folder) / output_prefix / ecoregion / sub / (data_folder if data_folder else data_type)
        else:
            outputs = pathlib.Path(output_folder) / ecoregion / sub / (data_folder if data_folder else data_type)

        if data_type == 'BlueTopo':
            raw_geotiffs = list(outputs.rglob(self.glob_lookup[file_type]))
            
            if file_type == 'elevation':
                geotiffs = [
                    g for g in raw_geotiffs 
                    if not any(x in g.name.lower() for x in ['_unc', '_slope', '_rugosity', 'decay', 'iss'])
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