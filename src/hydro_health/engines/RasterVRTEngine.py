import pathlib
import geopandas as gpd
import rioxarray as rxr
import rasterio
import json

from pyproj.database import query_crs_info
from pyproj.enums import PJType
from osgeo import gdal, osr, ogr
from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine


class RasterVRTEngine(Engine):
    """Class for handling VRT creation of BlueTopo and DigitalCoast datasets"""

    def __init__(self) -> None:
        super().__init__()
        self.glob_lookup = {
            'elevation': '*[0-9].tiff',
            'uncertainty': '*_unc.tiff',
            'slope': '*_slope.tiff',
            'rugosity': '*_rugosity.tiff',
            'NCMP': '*.tif'
        }
        self.all_crs = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])

    def build_output_vrts(self, outputs: pathlib.Path, file_type: str, output_geotiffs: dict[str]) -> None:
        """Create main VRT files from list of outputs"""

        for crs, tile_dict in output_geotiffs.items():
            # Create VRT for each tile and set output CRS to fix heterogenous crs issue
            vrt_tiles = []
            for tile in tile_dict['tiles']:
                output_raster_vrt = str(tile.parents[0] / f"{tile.stem}.vrt")
                gdal.Warp(
                    output_raster_vrt, 
                    str(tile),
                    format="VRT",
                    dstSRS=output_geotiffs[crs]['crs']
                )
                vrt_tiles.append(output_raster_vrt)

            vrt_filename = str(outputs / f'mosaic_{file_type}_{crs}.vrt')
            gdal.BuildVRT(vrt_filename, vrt_tiles, callback=gdal.TermProgress_nocb)
            print(f'- finished VRT: {vrt_filename}')

    def create_raster_vrts(self, output_folder: str, file_type: str, ecoregion: str, data_type: str, skip_existing=False) -> None:
        """Create an output VRT from found .tif files"""

        outputs = pathlib.Path(output_folder) / ecoregion / get_config_item(data_type.upper(), 'SUBFOLDER') / data_type

        if data_type == 'BlueTopo':
            geotiffs = list(outputs.rglob(self.glob_lookup[file_type]))
            output_geotiffs = self.get_bluetopo_tifs(output_folder, outputs, geotiffs, data_type, skip_existing)
            self.build_output_vrts(outputs, file_type, output_geotiffs)
        else:
            provider_folders = [folder for folder in outputs.glob('*') if folder.is_dir() and 'unused_providers' not in str(folder)]
            for provider in provider_folders:
                geotiffs = list(provider.rglob(self.glob_lookup[file_type]))
                output_geotiffs = self.get_digitalcoast_tifs(output_folder, outputs, geotiffs, data_type, skip_existing)
                self.build_output_vrts(outputs, file_type, output_geotiffs)

    def get_bluetopo_tifs(self, output_folder: pathlib.Path, outputs: pathlib.Path, geotiffs: list[pathlib.Path], data_type: str, skip_existing: False) -> dict[str]:
        """Transform BlueTopo .tif to WGS84 and create individual VRT files for processing"""

        output_geotiffs = {}
        for geotiff_path in geotiffs:
            geotiff = str(geotiff_path)
            if skip_existing:
                output_vrt = geotiff_path.parents[0] / f'{geotiff_path.stem}.vrt'
                if output_vrt.exists():
                    print(f'Skipping VRT: {output_vrt.name}')
                    continue
            geotiff_ds = gdal.Open(geotiff)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            projection_wkt = geotiff_ds.GetProjection()
            spatial_ref = osr.SpatialReference(wkt=projection_wkt)  
            projected_crs_string = spatial_ref.GetAuthorityCode('DATUM')
            clean_crs_string = projected_crs_string.replace('/', '').replace(' ', '_')
            provider_folder = geotiff_path.parents[2].name if 'dem' in str(geotiff_path) else geotiff_path.parents[3].name
            # Handle BlueTopo and DigitalCoast differently
            clean_crs_key = f'{clean_crs_string}_{provider_folder}' if data_type == 'DigitalCoast' else clean_crs_string
            # Store tile and CRS
            if clean_crs_key not in output_geotiffs:
                output_geotiffs[clean_crs_key] = {'crs': None, 'tiles': []}
            output_geotiffs[clean_crs_key]['tiles'].append(geotiff_path)
            if output_geotiffs[clean_crs_key]['crs'] is None:
                output_geotiffs[clean_crs_key]['crs'] = spatial_ref

            projected_crs_string = None
            projection_wkt = None
            wgs84_srs = None
            spatial_ref = None
            geotiff_ds = None

        return output_geotiffs
    
    def get_digitalcoast_tifs(self, output_folder: pathlib.Path, outputs: pathlib.Path, geotiffs: list[pathlib.Path], data_type: str, skip_existing: False) -> dict[str]:
        """Transform DigitalCoast .tif to WGS84 and create individual VRT files for processing"""

        output_geotiffs = {}
        for geotiff_path in geotiffs:
            geotiff = str(geotiff_path)
            if skip_existing:
                output_vrt = geotiff_path.parents[0] / f'{geotiff_path.stem}.vrt'
                if output_vrt.exists():
                    print(f'Skipping VRT: {output_vrt.name}')
                    continue
            geotiff_ds = gdal.Open(geotiff)
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            geotiff_srs = geotiff_ds.GetSpatialRef()

            # Project all geotiff to match BlueTopo tiles WGS84
            if geotiff_srs.GetName() != wgs84_srs.GetName():
                unused_providers_folder = outputs / 'unused_providers'
                if unused_providers_folder.exists():
                    unused_providers_names = [folder.stem for folder in unused_providers_folder.iterdir() if folder.is_dir()]
                    provider_folder = geotiff_path.parents[2].stem
                    if provider_folder in unused_providers_names:
                        print(f' - Skipping unused provider: {provider_folder}')
                        continue

                geotiff_ds = None  # close original dataset
                old_geotiff = geotiff_path.parents[0] / f'{geotiff_path.stem}_old.tif'
                geotiff_path.rename(old_geotiff)
                raster_wgs84 = geotiff_path.parents[0] / f'{geotiff_path.stem}_wgs84.tif'
                rasterio_wgs84 = rasterio.crs.CRS.from_epsg(4326)
                with rxr.open_rasterio(old_geotiff) as geotiff_raster:
                    srs_json = json.loads(geotiff_srs.ExportToPROJJSON())

                    # Check for problematic compound CRS
                    try:
                        name = srs_json['components'][0]['name']
                        if '+' in name:
                            # Obtain EPSG and redefine raster CRS
                            print(f' - defining compound CRS by horiz. CRS: {geotiff_path}')
                            geotiff_raster = self.write_crs_to_raster(geotiff_raster, geotiff_srs)
                    except (KeyError, IndexError, TypeError):
                        # pass for missing keys, empty array, or no + sign
                        pass
                        
                    wgs84_geotiff_raster = geotiff_raster.rio.reproject(rasterio_wgs84)
                    wgs84_geotiff_raster.rio.to_raster(raster_wgs84)

                wgs84_ds = gdal.Open(str(raster_wgs84))
                input_resolution = wgs84_ds.GetGeoTransform()
                # Compress and overwrite original geotiff path
                gdal.Warp(
                    geotiff,
                    wgs84_ds,
                    srcNodata=wgs84_ds.GetRasterBand(1).GetNoDataValue(),
                    dstNodata=-9999,
                    xRes=input_resolution[1],
                    yRes=input_resolution[5],
                    creationOptions=["COMPRESS=ZSTD", "BIGTIFF=YES", "NUM_THREADS=ALL_CPUS"]
                )
                wgs84_ds = None

                # Delete intermediate files
                old_geotiff.unlink()
                raster_wgs84.unlink()

                # repoen new geotiff
                geotiff_ds = gdal.Open(geotiff)

            projection_wkt = geotiff_ds.GetProjection()
            spatial_ref = osr.SpatialReference(wkt=projection_wkt)  
            projected_crs_string = spatial_ref.GetAuthorityCode('DATUM')
            clean_crs_string = projected_crs_string.replace('/', '').replace(' ', '_')
            provider_folder = geotiff_path.parents[2].name if 'dem' in str(geotiff_path) else geotiff_path.parents[3].name
            # Handle BlueTopo and DigitalCoast differently
            clean_crs_key = f'{clean_crs_string}_{provider_folder}' if data_type == 'DigitalCoast' else clean_crs_string
            # Store tile and CRS
            if clean_crs_key not in output_geotiffs:
                output_geotiffs[clean_crs_key] = {'crs': None, 'tiles': []}
            output_geotiffs[clean_crs_key]['tiles'].append(geotiff_path)
            if output_geotiffs[clean_crs_key]['crs'] is None:
                output_geotiffs[clean_crs_key]['crs'] = spatial_ref

            projected_crs_string = None
            projection_wkt = None
            wgs84_srs = None
            geotiff_srs = None
            spatial_ref = None
            geotiff_ds = None

        return output_geotiffs
    
    def run(self, output_folder: str, file_type: str, ecoregion: str, data_type: str, skip_existing=False) -> None:
        self.create_raster_vrts(output_folder, file_type, ecoregion, data_type, skip_existing)


    def write_crs_to_raster(self, geotiff_raster: rxr.rioxarray, geotiff_srs: osr.SpatialReference) -> rxr.rioxarray:
        """Obtain the horizontal CRS of a compound CRS and write it to the raster"""

        proj_json = json.loads(geotiff_srs.ExportToPROJJSON())
        crs_name = proj_json['components'][0]['name']
        horizontal_name = crs_name.split(' + ')[0]  # Will this always work for compound CRS?
        clean_horiz_name = horizontal_name.lower().strip()
        epsg = [crs.code for crs in self.all_crs if crs.name.lower() == clean_horiz_name][0]
        geotiff_raster.rio.write_crs(f"EPSG:{epsg}", inplace=True)

        return geotiff_raster
