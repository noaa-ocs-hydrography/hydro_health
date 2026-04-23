import pathlib
import json
from pyproj.database import query_crs_info
from pyproj.enums import PJType
from osgeo import gdal, osr
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


class RasterVRTEngine(Engine):
    """Class for handling VRT creation keeping native GeoTIFF CRS"""

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
        # Cache projected CRS info for name matching
        self.all_crs = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])

    def build_output_vrts(self, outputs: pathlib.Path, file_type: str, output_geotiffs: dict) -> None:
        """Create Master VRT files for each CRS bin"""

        for bin_key, info in output_geotiffs.items():
            tifs = [str(t) for t in info['tiles']]
            vrt_filename = outputs / f'mosaic_{file_type}_{bin_key}.vrt'
            
            # Use BuildVRT to create a mosaic of the tiles in this CRS bin
            options = gdal.BuildVRTOptions(
                resampleAlg='near',
                srcNodata=info.get('nodata'),
                VRTNodata=info.get('nodata'),
                allowProjectionDifference=True
            )
            
            gdal.BuildVRT(str(vrt_filename), tifs, options=options)
            print(f'- finished VRT: {vrt_filename}')

    def create_raster_vrts(self, output_folder: str, file_type: str, ecoregion: str, data_type: str, skip_existing: False) -> None:
        """Find tiles and initiate VRT building"""

        outputs = pathlib.Path(output_folder) / ecoregion / get_config_item(data_type.upper(), 'SUBFOLDER') / data_type

        if data_type == 'BlueTopo':
            geotiffs = list(outputs.rglob(self.glob_lookup[file_type]))
            output_geotiffs = self.get_geotiff_metadata_bins(geotiffs, data_type, skip_existing)
            self.build_output_vrts(outputs, file_type, output_geotiffs)
        else:
            provider_folders = [f for f in outputs.glob('*') if f.is_dir() and 'unused_providers' not in f.name]
            for provider in provider_folders:
                geotiffs = [g for g in provider.rglob(self.glob_lookup[file_type]) if not g.name.startswith('mask_')]
                if not geotiffs: continue
                output_geotiffs = self.get_geotiff_metadata_bins(geotiffs, data_type, skip_existing)
                self.build_output_vrts(outputs, file_type, output_geotiffs)

    def get_geotiff_metadata_bins(self, geotiffs: list[pathlib.Path], data_type: str, skip_existing: False) -> dict:
        """Groups geotiffs by CRS and Provider without reprojecting"""

        output_bins = {}
        
        for geotiff_path in geotiffs:
            ds = gdal.Open(str(geotiff_path))
            if ds is None: 
                continue
            if skip_existing:
                output_vrt = geotiff_path.parents[0] / f'{geotiff_path.stem}.vrt'
                if output_vrt.exists():
                    print(f'Skipping VRT: {output_vrt.name}')
                    continue
            
            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            src_srs = ds.GetSpatialRef()
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

            # Extract CRS ID
            bin_id = src_srs.GetAuthorityCode(None)
            if not bin_id:
                # Fallback to your PROJJSON matching for NAD83(2011) cases
                try:
                    srs_json = json.loads(src_srs.ExportToPROJJSON())
                    comp_name = srs_json.get('components', [{}])[0].get('name', '')
                    horiz_name = comp_name.split(' + ')[0].lower().strip()
                    match = [cr.code for cr in self.all_crs if cr.name.lower() == horiz_name]
                    bin_id = match[0] if match else src_srs.GetName().replace(" ", "_")
                except:
                    bin_id = "UnknownCRS"

            # Determine Provider for the key
            parts = geotiff_path.parts
            try:
                # Logic to grab the provider folder name based on your directory structure
                provider = parts[-4] if data_type == 'BlueTopo' else parts[-3]
            except IndexError:
                provider = "UnknownProvider"

            clean_key = f"{bin_id}_{provider}" if data_type == 'DigitalCoast' else bin_id
            
            if clean_key not in output_bins:
                output_bins[clean_key] = {'tiles': [], 'nodata': nodata, 'wkt': src_srs.ExportToWkt()}
            
            output_bins[clean_key]['tiles'].append(geotiff_path)
            ds = None

        return output_bins

    def run(self, output_folder: str, file_type: str, ecoregion: str, data_type: str, skip_existing: False) -> None:
        self.create_raster_vrts(output_folder, file_type, ecoregion, data_type, skip_existing)