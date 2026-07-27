import pathlib
import json
import tempfile
from pyproj.database import query_crs_info
from pyproj.enums import PJType
from osgeo import gdal, osr
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


def _clean(s: str) -> str:
    """Helper to normalize strings for comparison (removes non-breaking spaces, etc.)"""

    if not s: 
        return ""
    return " ".join(s.split()).lower().strip()


def _local_process_single_bluetopo(geotiff_path: pathlib.Path, output_dir: pathlib.Path) -> tuple[str, str, str]:
    """
    Local equivalent of the S3 BlueTopo logic.
    Creates an individual warped VRT (EPSG:4326) on the local disk.
    """

    gdal.UseExceptions()
    
    geotiff_stem = geotiff_path.stem
    # Save the intermediate warped VRT inside the provided temp directory
    warped_vrt_path = output_dir / f"{geotiff_stem}_warped.vrt"
    
    src_ds = None
    try:
        src_ds = gdal.Open(str(geotiff_path))
        if src_ds is None:
            raise FileNotFoundError(f"GDAL could not open {geotiff_path}")
            
        warp_options = {
            'format': 'VRT',
            'dstSRS': 'EPSG:4326',
            'resampleAlg': gdal.GRA_Bilinear,
            'srcNodata': -999999,
            'dstNodata': -999999,  # Ensures "empty" reprojected space is transparent
            'warpOptions': ['CUTLINE_ALL_TOUCHED=TRUE']
        }

        warped_vrt_ds = gdal.Warp(str(warped_vrt_path), src_ds, **warp_options)
        projection_wkt = warped_vrt_ds.GetProjection()
        spatial_ref = osr.SpatialReference(wkt=projection_wkt)
        datum_code = spatial_ref.GetAuthorityCode('DATUM')
        warped_vrt_ds = None 

        return str(datum_code), str(warped_vrt_path), projection_wkt

    except Exception as e:
        raise RuntimeError(f'_local_process_single_bluetopo failed: {geotiff_path} - {str(e)}')
    finally:
        src_ds = None


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
            'catzoc_decay_all': '*decay_all.tiff',
            'catzoc_decay_latest': '*decay_latest.tiff',
            'NCMP': '*.tif'
        }
        self.all_crs = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])

    def build_output_vrts(self, outputs: pathlib.Path, file_type: str, output_geotiffs: dict, data_type: str) -> None:
        """Create Master VRT files matching the original S3 options architecture"""

        for bin_key, info in output_geotiffs.items():
            tifs = [str(t) for t in info['tiles']]
            vrt_filename = outputs / f'mosaic_{file_type}_{bin_key}.vrt'
            
            if data_type == 'DigitalCoast':
                options = gdal.BuildVRTOptions(
                    resampleAlg='near', 
                    srcNodata=info.get('nodata_val'),
                    VRTNodata=info.get('nodata_val'),
                    addAlpha=True,
                    allowProjectionDifference=True,
                    outputSRS=info.get('wkt')
                )
            else:
                options = gdal.BuildVRTOptions(
                    resampleAlg='bilinear',
                    allowProjectionDifference=True
                )

            gdal.BuildVRT(str(vrt_filename), tifs, options=options)
            print(f'- Finished Master VRT: {vrt_filename}')

    def get_local_bluetopo_tifs(self, geotiffs: list[pathlib.Path], base_output_path: pathlib.Path) -> dict:
        """Processes BlueTopo files locally by warping them to 4326 first"""

        output_geotiffs = {}
        
        for gtif in geotiffs:
            try:
                crs_code, local_vrt_path, wkt = _local_process_single_bluetopo(gtif, base_output_path)
                clean_key = _clean(str(crs_code)).replace('/', '').replace(' ', '_')
                
                if clean_key not in output_geotiffs:
                    output_geotiffs[clean_key] = {
                        'crs': osr.SpatialReference(wkt=wkt), 
                        'tiles': [],
                        'nodata_val': -999999,
                        'wkt': wkt
                    }
                output_geotiffs[clean_key]['tiles'].append(local_vrt_path)
            except Exception as e:
                print(f" - Error processing BlueTopo file {gtif}: {e}")
                
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
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                
                bin_id = src_srs.GetAuthorityCode(None)
                if not bin_id:
                    try:
                        srs_json = json.loads(src_srs.ExportToPROJJSON())
                        components = srs_json.get('components', [{}])
                        comp_name = components[0].get('name', '')
                        horizontal_name = _clean(comp_name.split(' + ')[0])
                        match = [cr.code for cr in self.all_crs if _clean(cr.name) == horizontal_name]
                        if match:
                            bin_id = match[0]
                    except:
                        pass

                if not bin_id:
                    fallback_name = src_srs.GetName()
                    bin_id = src_srs.GetAuthorityCode('DATUM') or _clean(fallback_name).replace(" ", "_")
                    
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
                        'wkt': src_srs.ExportToWkt()
                    }
                output_geotiffs[provider]['tiles'].append(str(geotiff_path))
                
            except Exception as e:
                print(f" - Error obtaining metadata for {geotiff_path}: {e}")
            finally:
                ds = None
                
        return output_geotiffs
    
    def run(self, output_folder: str, file_type: str, ecoregion: str, data_type: str, output_prefix: str="", data_folder: str = "") -> None:
        """Main execution method mimicking the S3 engine's control routing entirely local"""
        
        sub = get_config_item(data_type.upper(), 'SUBFOLDER')
        
        if output_prefix:
            outputs = pathlib.Path(output_folder) / output_prefix / ecoregion / sub / (data_folder if data_folder else data_type)
        else:
            outputs = pathlib.Path(output_folder) / ecoregion / sub / (data_folder if data_folder else data_type)

        if data_type == 'BlueTopo':
            geotiffs = list(outputs.rglob(self.glob_lookup[file_type]))
            if geotiffs:
                # Isolate the temporary single-tile VRTs inside a Context Manager
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_vrt_path = pathlib.Path(temp_dir)
                    output_geotiffs = self.get_local_bluetopo_tifs(geotiffs, temp_vrt_path)
                    self.build_output_vrts(outputs, file_type, output_geotiffs, data_type)
        else:
            provider_folders = [f for f in outputs.glob('*') if f.is_dir() and 'unused_providers' not in f.name]
            for provider_path in provider_folders:
                geotiffs = [g for g in provider_path.rglob(self.glob_lookup[file_type]) if not g.name.startswith('mask_')]
                if not geotiffs: 
                    continue
                output_geotiffs = self.get_local_digitalcoast_geotiffs(geotiffs)
                self.build_output_vrts(outputs, file_type, output_geotiffs, data_type)