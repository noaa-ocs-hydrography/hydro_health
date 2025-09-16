import pathlib
import rasterio
import sys
import rioxarray as rxr

from osgeo import gdal, osr

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.helpers.tools import get_config_item, get_environment

gdal.UseExceptions()

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class RasterVRTEngine:
    """Class for creating VRT files"""

    def __init__(self) -> None:
        pass

    def run(self) -> None:
        pass


def create_raster_vrts(output_folder: str, ecoregion: str, data_type: str, provider: str) -> None:
    """Create an output VRT from found .tif files"""

    # Get DigitalCoast or BlueTopo folder
    outputs = pathlib.Path(output_folder) / ecoregion / get_config_item(data_type.upper(), 'SUBFOLDER') / data_type
    provider_folder = outputs / provider
    # TODO look for specific tifs for a provider
    geotiffs = list(provider_folder.rglob('*.tif'))
    total_tiffs = len(geotiffs)
    output_geotiffs = {}
    for i, geotiff_path in enumerate(geotiffs):
        print(f'{i+1} of {total_tiffs}')
        geotiff = str(geotiff_path)
        geotiff_ds = gdal.Open(geotiff)
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)
        geotiff_srs = geotiff_ds.GetSpatialRef()
        # Project all geotiff to match BlueTopo tiles WGS84
        if data_type == 'DigitalCoast' and geotiff_srs.GetName() != wgs84_srs.GetName():

            
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
                wgs84_geotiff_raster = geotiff_raster.rio.reproject(rasterio_wgs84)
                wgs84_geotiff_raster.rio.to_raster(raster_wgs84)

            wgs84_ds = gdal.Open(str(raster_wgs84))

            # Compress and overwrite original geotiff path
            resolution = 0.000008983
            gdal.Warp(
                geotiff,
                wgs84_ds,
                srcNodata=wgs84_ds.GetRasterBand(1).GetNoDataValue(),
                dstNodata=-9999,
                xRes=resolution,
                yRes=resolution,
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
        # Handle BlueTopo and DigitalCoast differently
        clean_crs_key = f'{clean_crs_string}_{provider}' if data_type == 'DigitalCoast' else clean_crs_string
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
    
    for provider_crs, tile_dict in output_geotiffs.items():
        # Create VRT for each tile and set output CRS to fix heterogenous crs issue
        vrt_tiles = []
        for tile in tile_dict['tiles']:
            output_raster_vrt = str(tile.parents[0] / f"{tile.stem}.vrt")
            gdal.Warp(
                output_raster_vrt, 
                str(tile),
                format="VRT",
                dstSRS=output_geotiffs[provider_crs]['crs']
            )
            vrt_tiles.append(output_raster_vrt)
        
        vrt_filename = str(outputs / f'mosaic_{provider_crs}.vrt')
        gdal.BuildVRT(vrt_filename, vrt_tiles, callback=gdal.TermProgress_nocb)
        print(f'- finished VRT: {vrt_filename}')


if __name__ == "__main__":
    # if get_environment() == 'local':
    #     output_folder = r'C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs'
    # else:
    stop = 'NOAA_NGS_2019_9709'
    output_folder = r'N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run'
    
    # provider = 'NOAA_NGS_2016_6365'  # 781mb file becomes 20gb WGS84?
    provider = 'NOAA_NGS_2016_8574'
    if provider != stop:
        create_raster_vrts(output_folder, 'ER_3', 'DigitalCoast', provider)