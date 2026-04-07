import os
import gc
import pathlib
import tempfile
import boto3
import s3fs
import numpy as np
import geopandas as gpd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from osgeo import gdal, osr, ogr

# Force GDAL settings for multi-processing stability
os.environ['PROJ_LIB_CACHE'] = 'OFF'
os.environ['PROJ_NETWORK'] = 'OFF'
os.environ['PROJ_USER_WRITABLE_DIRECTORY'] = '/tmp'

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

gdal.UseExceptions()


INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


def _set_gdal_s3_options():
    """Optimized GDAL settings for S3 VSI stability."""

    gdal.SetConfigOption('GDAL_CACHEMAX', '512')
    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '5')
    gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '3')
    gdal.SetConfigOption('AWS_REGION', 'us-east-2') 
    gdal.SetConfigOption('GDAL_HTTP_TIMEOUT', '60')
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('PROJ_CACHE_DIR', '/tmp/proj_cache')
    gdal.SetConfigOption('GDAL_PROJ_THREAD_SAFE', 'YES')


def _vrt_to_mask_worker(vrt_path: str, scratch_dir: str, geo_t: list[float], cols: int, rows: int, target_srs_wkt: str) -> str:
    """Converts one VRT into a detailed binary mask (1=data, 0=nodata)."""

    _set_gdal_s3_options()

    pid = os.getpid()
    vrt_name = pathlib.Path(vrt_path).stem
    local_mask_path = scratch_dir / f"part_{vrt_name}.tif"
    
    xmin, res_x, _, ymax, _, res_y = geo_t
    xmax = xmin + (cols * res_x)
    ymin = ymax + (rows * res_y)

    try:
        # Read the source nodata value
        src_ds = gdal.Open(vrt_path)
        src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
        src_ds = None

        with tempfile.TemporaryDirectory(dir=scratch_dir) as tmp_work:
            tmp_warp = pathlib.Path(tmp_work) / "warp_with_alpha.tif"
            
            # Warp with nodata to find discrete areas
            warp_opts = {
                'format': 'GTiff',
                'outputBounds': [xmin, ymin, xmax, ymax],
                'width': cols,
                'height': rows,
                'dstSRS': target_srs_wkt,
                'dstAlpha': True, # Band 2 becomes our high-res binary mask
                'resampleAlg': gdal.GRA_NearestNeighbour,
                'creationOptions': ['COMPRESS=LZW', 'TILED=YES']
            }
            
            if src_nodata is not None:
                warp_opts['srcNodata'] = src_nodata

            gdal.Warp(str(tmp_warp), vrt_path, **warp_opts)
            
            # Convert band 2 to local partial mask
            ds = gdal.Open(str(tmp_warp))
            gdal.Translate(
                str(local_mask_path),
                ds,
                bandList=[2], 
                format='GTiff',
                creationOptions=['COMPRESS=LZW', 'TILED=YES', 'SPARSE_OK=YES']
            )
            ds = None
            
        print(f"[Worker {pid}] Detailed mask complete: {vrt_name}")
        return str(local_mask_path)
    except Exception as e:
        print(f"!!! [Worker {pid}] Failed {vrt_name}: {e}")
        return None


class RasterMaskS3Engine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def create_prediction_mask(self, ecoregion: str, wkt_geom: str) -> None:
        """Build the base prediction mask (Value 1) for the ecoregion polygon."""

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
        cols = max(1, int((xmax - xmin) / pixel_size))
        rows = max(1, int((ymax - ymin) / pixel_size))

        with tempfile.NamedTemporaryFile(suffix='.tif') as tmp:
            driver = gdal.GetDriverByName("GTiff")
            out_ds = driver.Create(tmp.name, cols, rows, 1, gdal.GDT_Byte, 
                                   options=["TILED=YES", "SPARSE_OK=YES"])
            out_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
            out_ds.SetProjection(target_srs.ExportToWkt())
            out_ds.GetRasterBand(1).SetNoDataValue(0)
            gdal.RasterizeLayer(out_ds, [1], mem_layer, burn_values=[1])
            out_ds = None 
            
            boto3.client('s3').upload_file(tmp.name, bucket, s3_key)

    def create_training_mask(self, ecoregion: str, s3_vrt_paths: list[str], outputs: str) -> str:
        """Tile-based merge using bitwise OR to combine 58 bathymetry masks."""

        _set_gdal_s3_options()

        scratch_dir = pathlib.Path.home() / f"gdal_scratch_{ecoregion}"
        scratch_dir.mkdir(exist_ok=True)
        
        local_pred_path = scratch_dir / "base_pred.tif"
        local_raw_path = scratch_dir / "raw_training_merge.tif"
        final_compressed_path = scratch_dir / f"training_mask_{ecoregion}.tif"

        s3_client = boto3.client('s3', region_name='us-east-2')
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        mask_sub = get_config_item('MASK', 'SUBFOLDER')

        self.write_message(f"Downloading base prediction mask for {ecoregion}...", outputs)
        s3_client.download_file(bucket, f"{ecoregion}/{mask_sub}/prediction_mask_{ecoregion}.tif", str(local_pred_path))

        base_ds = gdal.Open(str(local_pred_path))
        geo_t = base_ds.GetGeoTransform()
        target_srs_wkt = base_ds.GetProjection()
        cols, rows = base_ds.RasterXSize, base_ds.RasterYSize
        base_ds = None 

        vsi_vrt_paths = [v.replace("s3://", "/vsis3/") for v in s3_vrt_paths]
        worker_func = partial(_vrt_to_mask_worker, scratch_dir=scratch_dir, 
                             geo_t=geo_t, cols=cols, rows=rows, target_srs_wkt=target_srs_wkt)

        self.write_message(f"Starting worker swarm for {len(vsi_vrt_paths)} detailed VRT masks...", outputs)
        gc.collect()
        
        with ProcessPoolExecutor(max_workers=6) as executor:
            mask_files = list(executor.map(worker_func, vsi_vrt_paths))
        
        valid_masks = [f for f in mask_files if f is not None]

        # Initialize training mask from prediction mask
        gdal.Translate(str(local_raw_path), str(local_pred_path), 
                       options=gdal.TranslateOptions(format="GTiff", creationOptions=["TILED=YES", "BIGTIFF=YES"]))
        
        train_ds = gdal.Open(str(local_raw_path), gdal.GA_Update)
        train_band = train_ds.GetRasterBand(1)
        mask_dss = [gdal.Open(m) for m in valid_masks]
        mask_bands = [ds.GetRasterBand(1) for ds in mask_dss]

        self.write_message(f"Merging {len(valid_masks)} partial masks with exhaustive OR logic...", outputs)
        tile_size = 4096  # Cloud optimized largest block size; Could also use 2048
        for y in range(0, rows, tile_size):
            win_y = min(tile_size, rows - y)
            for x in range(0, cols, tile_size):
                win_x = min(tile_size, cols - x)
                prediction_chunk = train_band.ReadAsArray(x, y, win_x, win_y)
                starting_mask = np.zeros((win_y, win_x), dtype=np.uint8)

                for mask_band in mask_bands:
                    mask_block = mask_band.ReadAsArray(x, y, win_x, win_y)
                    # Set to mask_block or keep starting_mask values
                    starting_mask |= (mask_block > 0).astype(np.uint8)
                
                # If inside Ecoregion (1) AND has bathymetry (presence > 0) -> Value 2
                mask_indices = (prediction_chunk == 1) & (starting_mask > 0)
                
                if np.any(mask_indices):
                    prediction_chunk[mask_indices] = 2
                    train_band.WriteArray(prediction_chunk, x, y)

        train_band.FlushCache()
        train_ds = None
        mask_dss = None 

        # Final Compress and Upload
        gdal.Translate(
            str(final_compressed_path),
            str(local_raw_path),
            options=gdal.TranslateOptions(
                format="GTiff", creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
            ),
        )

        s3_key = f"{ecoregion}/{mask_sub}/training_mask_{ecoregion}.tif"
        s3_client.upload_file(str(final_compressed_path), bucket, s3_key)

        # Clean up EBS
        # for f in scratch_dir.glob("part_*.tif"): f.unlink()
        # local_pred_path.unlink()
        # local_raw_path.unlink()
        # final_compressed_path.unlink()
        # scratch_dir.rmdir()

        return f"{ecoregion}: Training mask completed"

    def find_provider_vrts(self, ecoregion: str, manual_downloads: bool) -> list[str]:
        """Obtain list of VRT S3 paths"""

        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        dc_sub = get_config_item('DIGITALCOAST', 'SUBFOLDER')
        search_paths = [f"s3://{bucket}/{ecoregion}/{dc_sub}/DigitalCoast"]
        if manual_downloads: 
            search_paths.append(f"{search_paths[0]}_manual_downloads")
        
        found = []
        for path in search_paths: 
            found.extend(s3.glob(f"{path}/**/mosaic_*.vrt"))
        return found

    def remerge_training_mask(self, ecoregion: str, outputs: str) -> str:
        """
        Standalone 'Merge-Only' function. 
        Use this if you have manually modified/deleted part_*.tif files 
        in the scratch directory and want to rebuild the final output.
        """

        _set_gdal_s3_options()
        
        # Path setup
        scratch_dir = pathlib.Path.home() / f"gdal_scratch_{ecoregion}"
        local_pred_path = scratch_dir / "base_pred.tif"
        local_raw_path = scratch_dir / "raw_training_merge.tif"
        final_compressed_path = scratch_dir / f"training_mask_{ecoregion}.tif"
        
        if not local_pred_path.exists():
            return f"Error: {local_pred_path} not found. Need the base prediction mask to continue."

        # Collect all remaining part mask files
        valid_masks = [str(f) for f in scratch_dir.glob("part_*.tif")]
        
        if not valid_masks:
            return f"Error: No part_*.tif files found in {scratch_dir}"

        self.write_message(f"Remerging {len(valid_masks)} files for {ecoregion}...", outputs)

        # Initialize the training mask from prediction mask
        gdal.Translate(str(local_raw_path), str(local_pred_path), 
                       options=gdal.TranslateOptions(format="GTiff", creationOptions=["TILED=YES", "BIGTIFF=YES"]))
        
        # Open training dataset and read values
        train_ds = gdal.Open(str(local_raw_path), gdal.GA_Update)
        train_band = train_ds.GetRasterBand(1)
        cols, rows = train_ds.RasterXSize, train_ds.RasterYSize
        
        mask_dss = [gdal.Open(partial_mask) for partial_mask in valid_masks]
        mask_bands = [ds.GetRasterBand(1) for ds in mask_dss]

        # Build output band by reading tile blocks
        tile_size = 4096 
        for y in range(0, rows, tile_size):
            win_y = min(tile_size, rows - y)
            for x in range(0, cols, tile_size):
                win_x = min(tile_size, cols - x)
                
                prediction_chunk = train_band.ReadAsArray(x, y, win_x, win_y)
                combined_presence = np.zeros((win_y, win_x), dtype=np.uint8)

                for mb in mask_bands:
                    m_tile = mb.ReadAsArray(x, y, win_x, win_y)
                    combined_presence |= (m_tile > 0).astype(np.uint8)
                
                # Ecoregion (1) + Bathy Presence (True) = Value 2
                mask_indices = (prediction_chunk == 1) & (combined_presence > 0)
                
                if np.any(mask_indices):
                    prediction_chunk[mask_indices] = 2
                    train_band.WriteArray(prediction_chunk, x, y)

        # Cleanup handles
        train_band.FlushCache()
        train_ds = None
        mask_dss = None 

        # Compress and Upload
        self.write_message(f"Compressing and uploading rebuilt mask...", outputs)
        gdal.Translate(
            str(final_compressed_path),
            str(local_raw_path),
            options=gdal.TranslateOptions(
                format="GTiff", creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"]
            ),
        )

        s3_client = boto3.client('s3', region_name='us-east-2')
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        mask_sub = get_config_item('MASK', 'SUBFOLDER')
        s3_key = f"{ecoregion}/{mask_sub}/training_mask_{ecoregion}.tif"
        
        s3_client.upload_file(str(final_compressed_path), bucket, s3_key)

        return f"{ecoregion}: Remerge complete. S3 updated."
    
    def run(self, outputs: str, manual_downloads=False) -> None:
        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        existing_ers = [f.split('/')[-1] for f in s3.glob(f"s3://{bucket}/ER*")]

        gpkg = str(INPUTS / 'Master_Grids.gpkg')
        gdf = gpd.read_file(gpkg, layer='EcoRegions_50m').to_crs("EPSG:32617")
        gdf = gdf[gdf['EcoRegion'].isin(existing_ers)]

        for _, row in gdf.iterrows():
            er = row['EcoRegion']
            vrts = self.find_provider_vrts(er, manual_downloads)
            self.create_prediction_mask(er, row['geometry'].wkt)
            if vrts:
                vrt_list = [f"s3://{v}" if not v.startswith('s3://') else v for v in vrts]
                result_string = self.create_training_mask(er, vrt_list, outputs)
                self.write_message(result_string, outputs)
