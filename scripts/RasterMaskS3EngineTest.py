"""
This is a sample class for testing individual VRT training mask output.
The production RasterMaskS3Engine creates a single final training mask and uploads it to S3.
"""

import sys
import os
import pathlib


HH_MODEL = pathlib.Path(__file__).parents[1] / 'src'
sys.path.append(str(HH_MODEL))

os.environ['PROJ_LIB_CACHE'] = 'OFF'
os.environ['PROJ_NETWORK'] = 'OFF'
os.environ['PROJ_USER_WRITABLE_DIRECTORY'] = '/tmp'

import time
import tempfile
import boto3
import s3fs
import numpy as np
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor  
from functools import partial
from osgeo import gdal, osr, ogr
from pyproj.database import query_crs_info
from pyproj.enums import PJType

from hydro_health.helpers.tools import Param
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

gdal.UseExceptions()

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'

def _set_gdal_s3_options():
    """Spike's special sauce for keeping S3 connections alive and silencing PROJ locks."""

    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '5')
    gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '3')
    gdal.SetConfigOption('AWS_REGION', 'us-east-2') 
    gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '30')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')

    gdal.SetConfigOption('PROJ_CACHE_DIR', '/tmp/proj_cache')
    gdal.SetConfigOption('GDAL_PROJ_THREAD_SAFE', 'YES')


def _scout_geotiff_metadata(params: list):
    _set_gdal_s3_options()
    geotiff_prefix, all_crs_info, data_folder = params
    vsi_path = f'/vsis3/{geotiff_prefix}'
    
    ds = None
    try:
        ds = gdal.Open(vsi_path)
        if ds is None: return None
            
        band = ds.GetRasterBand(1)
        srs = ds.GetSpatialRef()
        datum_code = srs.GetAuthorityCode('DATUM') or "Unknown"
        nodata = band.GetNoDataValue()
        
        parts = geotiff_prefix.split('/')
        try:
            dc_index = parts.index('DigitalCoast')
            provider = parts[dc_index + 1]
        except (ValueError, IndexError):
            provider = parts[-3]
            
        clean_key = f"{datum_code}_{provider}"
        
        return {
            'bin_key': clean_key,
            'vsi_path': vsi_path,
            'nodata': nodata
        }
    except Exception as e:
        print(f"!!! Spike tripped on a TIFF: {e}")
        return None
    finally:
        ds = None


def _process_tile_worker(tile_params, vsi_vrt_paths, geo_t, target_srs_wkt, outputs):
    _set_gdal_s3_options()
    x_off, y_off, x_size, y_size = tile_params
    
    # Calculate precise tile bounds 
    tile_xmin = geo_t[0] + (x_off * geo_t[1])
    tile_ymax = geo_t[3] + (y_off * geo_t[5])
    tile_xmax = tile_xmin + (x_size * geo_t[1])
    tile_ymin = tile_ymax + (y_size * geo_t[5])
    
    tile_mask = np.zeros((y_size, x_size), dtype=np.uint8)
    extreme_nodata = -3.402823e+38 # Found in your metadata 

    for vrt in vsi_vrt_paths:
        mem_path = f"/vsimem/tile_{x_off}_{y_off}_{time.time_ns()}.tif"
        
        try:
            ds = gdal.Warp(
                mem_path, 
                vrt, 
                outputBounds=[tile_xmin, tile_ymin, tile_xmax, tile_ymax], 
                width=x_size,
                height=y_size,
                dstSRS=target_srs_wkt,
                outputType=gdal.GDT_Float32,
                resampleAlg=gdal.GRA_Bilinear, # Smooths gaps from CRS transformation 
                srcNodata=extreme_nodata,
                dstNodata=-9999.0, # Unique float for background 
                xRes=8,
                yRes=8,
                targetAlignedPixels=True, # Force alignment to your grid 
                options=['-multi'] 
            )
            
            if ds:
                data = ds.GetRasterBand(1).ReadAsArray()
                # Validate imagery: Not the background, not the extreme float, and finite 
                valid_data_mask = (data != -9999.0) & (data > -1e30) & (np.isfinite(data))
                
                if valid_data_mask.any():
                    tile_mask[valid_data_mask] = 1
                ds = None 
        except Exception as e:
            print(f"!!! Warp Error: {e}")
        finally:
            gdal.Unlink(mem_path)

    return x_off, y_off, tile_mask


def _create_prediction_mask(param_inputs: list) -> None:
    ecoregion, wkt_geom = param_inputs
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    mask_sub = get_config_item('MASK', 'SUBFOLDER')
    s3_key = f"{ecoregion}/{mask_sub}/prediction_mask_{ecoregion}.tif"

    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(32617)
    target_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    
    mem_ds = ogr.GetDriverByName('Memory').CreateDataSource(f'mem_{ecoregion}')
    mem_layer = mem_ds.CreateLayer('selected_ecoregion', srs=target_srs, geom_type=ogr.wkbPolygon)
    feat = ogr.Feature(mem_layer.GetLayerDefn())
    feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt_geom))
    mem_layer.CreateFeature(feat)
    
    xmin, xmax, ymin, ymax = mem_layer.GetExtent()
    pixel_size = 8
    x_res = max(1, int((xmax - xmin) / pixel_size))
    y_res = max(1, int((ymax - ymin) / pixel_size))

    with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(tmp.name, x_res, y_res, 1, gdal.GDT_Byte, 
                                options=["TILED=YES", "SPARSE_OK=YES"])
        out_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
        out_ds.SetProjection(target_srs.ExportToWkt())
        out_ds.GetRasterBand(1).SetNoDataValue(0)
        gdal.RasterizeLayer(out_ds, [1], mem_layer, burn_values=[1])
        out_ds = None 
        boto3.client('s3').upload_file(tmp.name, bucket, s3_key)


def _create_training_mask(params: list) -> str:
    ecoregion, s3_vrt_paths, outputs = params
    engine = Engine()
    _set_gdal_s3_options()

    scratch_dir = pathlib.Path.home() / f"gdal_scratch_{ecoregion}"
    scratch_dir.mkdir(exist_ok=True)
    local_pred_path = scratch_dir / "base_pred.tif"
    
    s3_client = boto3.client('s3', region_name='us-east-2')
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    mask_sub = get_config_item('MASK', 'SUBFOLDER')
    
    # 1. Download base mask template
    s3_client.download_file(bucket, f"{ecoregion}/{mask_sub}/prediction_mask_{ecoregion}.tif", str(local_pred_path))
    
    # 2. Process VRTs individually for debugging
    for i, vrt_path in enumerate(s3_vrt_paths):
        vsi_vrt_path = vrt_path.replace("s3://", "/vsis3/")
        debug_output_path = scratch_dir / f"final_training_{ecoregion}_VRT_index_{i}.tif"
        
        engine.write_message(f"Checking VRT index {i}: {vsi_vrt_path}", outputs)

        base_ds = gdal.Open(str(local_pred_path))
        geo_t = base_ds.GetGeoTransform()
        target_srs_wkt = base_ds.GetProjection()
        cols, rows = base_ds.RasterXSize, base_ds.RasterYSize
        
        # Create uncompressed temp file for random writes
        driver = gdal.GetDriverByName('GTiff')
        temp_train_path = scratch_dir / f"temp_vrt_{i}.tif"
        train_ds = driver.Create(str(temp_train_path), cols, rows, 1, gdal.GDT_Byte, 
                                 options=["TILED=YES", "BIGTIFF=YES", "BLOCKXSIZE=512", "BLOCKYSIZE=512"])
        train_ds.SetGeoTransform(geo_t)
        train_ds.SetProjection(target_srs_wkt)
        train_band = train_ds.GetRasterBand(1)
        
        # Initialize with prediction mask
        base_array = base_ds.ReadAsArray()
        train_band.WriteArray(base_array)
        train_band.FlushCache() 
        base_ds = None 

        tiles = [(x, y, min(4096, cols - x), min(4096, rows - y)) 
                 for y in range(0, rows, 4096) for x in range(0, cols, 4096)]

        worker_func = partial(_process_tile_worker, vsi_vrt_paths=[vsi_vrt_path], 
                              geo_t=geo_t, target_srs_wkt=target_srs_wkt, outputs=outputs)

        with ProcessPoolExecutor(max_workers=10) as executor:
            for x_off, y_off, tile_mask in executor.map(worker_func, tiles):
                # If tile_mask is all 0s, skip entirely to prevent unnecessary IO
                if tile_mask is None or not np.any(tile_mask == 1):
                    continue

                # Read current chunk (contains original 1s from prediction mask)
                target_chunk = train_band.ReadAsArray(x_off, y_off, tile_mask.shape[1], tile_mask.shape[0])
                
                # DIAGNOSTIC: Create a temp mask where VRT data exists (value 2)
                vrt_presence = (tile_mask == 1).astype(np.uint8) * 2
                updated_chunk = np.maximum(target_chunk, vrt_presence)
                
                # Write back the merged result
                train_band.WriteArray(updated_chunk, x_off, y_off)

        # Finalize and Compress
        train_band.FlushCache()
        train_band = None
        train_ds = None 

        gdal.Translate(str(debug_output_path), str(temp_train_path), 
                       options=gdal.TranslateOptions(format="GTiff", 
                                                    creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]))
        
        if temp_train_path.exists():
            temp_train_path.unlink()
            
        engine.write_message(f"Debug file finished: {debug_output_path}", outputs)

    return f"{ecoregion}: Debug complete."


class RasterMaskS3Engine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup
        self.all_crs = query_crs_info(auth_name="EPSG", pj_types=[PJType.PROJECTED_CRS])
        self.glob_lookup = {'elevation': '*[0-9].tiff', 'NCMP': '*.tif'}

    def find_provider_vrts(self, ecoregion, manual_downloads) -> list[str]:
        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        dc_sub = get_config_item('DIGITALCOAST', 'SUBFOLDER')
        search_paths = [f"s3://{bucket}/{ecoregion}/{dc_sub}/DigitalCoast"]
        if manual_downloads: search_paths.append(f"{search_paths[0]}_manual_downloads")
        
        found = []
        for path in search_paths: 
            found.extend(s3.glob(f"{path}/**/mosaic_*.vrt"))
        return found

    def gather_provider_intel(self, geotiffs: list, data_folder) -> dict:
        task_params = [(gtif, self.all_crs, data_folder) for gtif in geotiffs]
        futures = self.client.map(_scout_geotiff_metadata, task_params)
        results = [r for r in self.client.gather(futures) if r is not None]
        
        binned = {}
        for res in results:
            key = res['bin_key']
            if key not in binned: binned[key] = {'tiles': [], 'nodata_val': res['nodata']}
            binned[key]['tiles'].append(res['vsi_path'])
        return binned

    def build_flat_mosaic_vrt(self, output_prefix: str, file_type: str, binned_data: dict, temp_path: pathlib.Path) -> None:
        s3_client = boto3.client('s3')
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        for bin_key, info in binned_data.items():
            vrt_file = temp_path / f'mosaic_{file_type}_{bin_key}.vrt'
            options = gdal.BuildVRTOptions(resampleAlg='near', srcNodata=info['nodata_val'])
            gdal.BuildVRT(str(vrt_file), info['tiles'], options=options)
            if vrt_file.exists():
                s3_client.upload_file(str(vrt_file), bucket, f'{output_prefix}/{vrt_file.name}')

    def run_vrt_prep(self, ecoregion: str, data_type: str, manual_downloads=False):
        """Main runner script for building masks"""

        self.setup_dask(self.param_lookup['env'])
        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        sub = get_config_item(data_type.upper(), 'SUBFOLDER')
        digital_coast_folder = f"s3://{bucket}/{ecoregion}/{sub}/{manual_downloads if manual_downloads else data_type}"
        
        for provider_path in s3.glob(f"{digital_coast_folder}/*"):
            geotiffs = s3.glob(f"{provider_path}/**/{self.glob_lookup['NCMP']}")
            if not geotiffs: 
                continue
            binned = self.gather_provider_intel(geotiffs, manual_downloads)
            with tempfile.TemporaryDirectory() as td:
                self.build_flat_mosaic_vrt(f"{ecoregion}/{sub}/{manual_downloads if manual_downloads else data_type}", 'NCMP', binned, pathlib.Path(td))
        self.close_dask()

    def run(self, outputs: str, manual_downloads=False) -> None:
        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        existing_ers = [f.split('/')[-1] for f in s3.glob(f"s3://{bucket}/ER*")]

        gpkg = str(INPUTS / 'Master_Grids.gpkg')
        gdf = gpd.read_file(gpkg, layer='EcoRegions_50m').to_crs("EPSG:32617")
        gdf = gdf[gdf['EcoRegion'].isin(existing_ers)]

        for _, row in gdf.iterrows():
            er = row['EcoRegion']
            self.run_vrt_prep(er, 'DigitalCoast', manual_downloads)
            self.write_message(f"Creating prediction mask for {er}", outputs)
            # _create_prediction_mask([er, row['geometry'].wkt])
            
            vrts = self.find_provider_vrts(er, manual_downloads)
            if vrts:
                vrts_to_process = vrts[5:10]
                # vrts_to_process = vrts
                self.write_message(f"Starting Training Mask Swarm for {er} and first vrt: {vrts_to_process[0]}", outputs)
                
                result = _create_training_mask([er, [f"s3://{v}" if not v.startswith('s3://') else v for v in vrts_to_process], outputs])
                self.write_message(result, outputs)


if __name__ == "__main__":
    param_lookup = {
        'output_directory': Param(str(OUTPUTS)),
        'env': 'aws'
    }
    engine = RasterMaskS3Engine(param_lookup)
    engine.run(str(OUTPUTS), manual_downloads=True)