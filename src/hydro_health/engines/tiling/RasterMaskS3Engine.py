import boto3
import pathlib
import s3fs
import tempfile
import time

import geopandas as gpd
from osgeo import ogr, osr, gdal

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

gdal.UseExceptions()


INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


def _create_prediction_mask(param_inputs: list) -> None:
    """Creates the ecoregion prediction mask with (Value 1)."""

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
        tmp_path = tmp.name
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(tmp_path, x_res, y_res, 1, gdal.GDT_Byte, 
                               options=["COMPRESS=LZW", "TILED=YES", "SPARSE_OK=YES"])
        out_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
        out_ds.SetProjection(target_srs.ExportToWkt())
        out_ds.GetRasterBand(1).SetNoDataValue(0)
        gdal.RasterizeLayer(out_ds, [1], mem_layer, burn_values=[1])
        
        out_ds = None 
        s3_client = boto3.client('s3', region_name='us-east-2')
        s3_client.upload_file(tmp_path, bucket, s3_key)
    
    return f' - uploaded prediction mask'


def _create_training_mask(params: list) -> str:
    """
    Training mask logic that loads the 'mosaic' VRT files, 
    reads the valid data areas and writes them to a geotiff to upload to S3
    - VRT files are binary byte-based masks in CRS 32617
    """

    ecoregion, s3_vrt_paths, outputs = params
    engine = Engine()
    
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_INGESTED_BYTES_AT_OPEN', '32768')
    gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY', '10')
    gdal.SetConfigOption('GDAL_HTTP_RETRY_DELAY', '5')
    gdal.SetConfigOption('VSI_CACHE', 'TRUE')
    gdal.SetConfigOption('VSI_CACHE_SIZE', '1073741824')  # 1GB Cache
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    gdal.SetConfigOption('CHECK_DISK_FREE_SPACE', 'FALSE')

    # EC2 /home directory has +100 GB of available storage
    # previouse use of /tmp actually uses RAM, which limits storage to 16 GB
    scratch_dir = pathlib.Path.home() / f"gdal_scratch_{ecoregion}"
    scratch_dir.mkdir(exist_ok=True)
    
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    mask_sub = get_config_item('MASK', 'SUBFOLDER')
    s3_pred_key = f"{ecoregion}/{mask_sub}/prediction_mask_{ecoregion}.tif"
    s3_train_key = f"{ecoregion}/{mask_sub}/training_mask_{ecoregion}.tif"

    s3_client = boto3.client('s3', region_name='us-east-2')

    try:
        local_pred_path = scratch_dir / "base_pred.tif"
        local_train_path = scratch_dir / "final_training.tif"
        
        # Download base - if this fails, the whole region is a bust
        s3_client.download_file(bucket, s3_pred_key, str(local_pred_path))
        base_ds = gdal.Open(str(local_pred_path))
        
        geo_t = base_ds.GetGeoTransform()
        cols, rows = base_ds.RasterXSize, base_ds.RasterYSize
        target_bounds = [geo_t[0], geo_t[3] + (rows * geo_t[5]), geo_t[0] + (cols * geo_t[1]), geo_t[3]]

        driver = gdal.GetDriverByName('GTiff')
        train_ds = driver.CreateCopy(str(local_train_path), base_ds, 
                                    options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES"])
        train_band = train_ds.GetRasterBand(1)

        vsi_vrt_paths = [v.replace("s3://", "/vsis3/") for v in s3_vrt_paths]

        for i, vrt_path in enumerate(vsi_vrt_paths):
            vrt_name = pathlib.Path(vrt_path).stem 
            engine.write(f"{time.ctime()}: Starting {i+1} of {len(vsi_vrt_paths)} - {vrt_name}", outputs)
            part_path = str(scratch_dir / f"part_{i+1}_{vrt_name}.tif")
            
            success = False
            for attempt in range(3):
                try:
                    gdal.Warp(part_path, vrt_path, 
                              format='GTiff',
                              outputBounds=target_bounds, 
                              xRes=abs(geo_t[1]), yRes=abs(geo_t[5]),
                              resampleAlg=gdal.GRA_NearestNeighbour,
                              dstNodata=0,
                              creationOptions=['COMPRESS=LZW', 'BIGTIFF=YES', 'SPARSE_OK=YES'])
                    success = True
                    break
                except Exception as e:
                    engine.write_message(f" - failed (attempt {attempt+1}): {e}", outputs)
                    time.sleep(10 * (attempt + 1))
            
            if not success:
                raise RuntimeError(f"Failed to Warp {vrt_name} after multiple tries.")

            part_ds = gdal.Open(part_path)
            part_band = part_ds.GetRasterBand(1)

            # Block process valid areas from current provider mosaic VRT to training local geotiff
            block_size = 2048 
            for y in range(0, rows, block_size):
                win_y = min(block_size, rows - y)
                for x in range(0, cols, block_size):
                    win_x = min(block_size, cols - x)

                    t_chunk = train_band.ReadAsArray(x, y, win_x, win_y)
                    p_chunk = part_band.ReadAsArray(x, y, win_x, win_y)

                    mask_indices = (p_chunk > 0) & (t_chunk == 1)
                    if mask_indices.any():
                        t_chunk[mask_indices] = 2
                        train_band.WriteArray(t_chunk, x, y)
            
            engine.write_message(f"{time.ctime()}: Finished {vrt_name}", outputs)

            part_ds = None 
            pathlib.Path(part_path).unlink()

        train_ds.FlushCache()
        train_ds = None 
        base_ds = None

        s3_client.upload_file(str(local_train_path), bucket, s3_train_key)
        
        for f in scratch_dir.glob("*"): f.unlink()
        scratch_dir.rmdir()
        
        return f"{ecoregion}: Training mask finished."

    except Exception as e:
        if scratch_dir.exists():
             print(f"Cleaning up failed run for {ecoregion}...")
        return f"{ecoregion}: Failed - {str(e)}"
    

class RasterMaskS3Engine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def find_provider_vrts(self, ecoregion, manual_downloads) -> list[str]:
        """Obtain full list of DigitalCoast mosaic VRT files"""

        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        dc_sub = get_config_item('DIGITALCOAST', 'SUBFOLDER')
        found_vrts = []
        
        search_paths = [f"s3://{bucket}/{ecoregion}/{dc_sub}/DigitalCoast"]
        if manual_downloads:
            search_paths.append(f"s3://{bucket}/{ecoregion}/{dc_sub}/DigitalCoast_manual_downloads")
            
        for path in search_paths:
            vrts = s3.glob(f"{path}/**/mosaic_*.vrt")
            found_vrts.extend(vrts)
        return found_vrts


    def run(self, outputs: str, manual_downloads=False) -> None:
        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        s3_folders = s3.glob(f"s3://{bucket}/ER*")
        existing_ers = [f.split('/')[-1] for f in s3_folders]

        gpkg_path = str(INPUTS / 'Master_Grids.gpkg')
        gdf_all = gpd.read_file(gpkg_path, layer='EcoRegions_50m').to_crs("EPSG:32617")
        gdf_filtered = gdf_all[gdf_all['EcoRegion'].isin(existing_ers)]

        self.write_message("Creating prediction mask", outputs)
        geom_payload = [[row['EcoRegion'], row['geometry'].wkt] for _, row in gdf_filtered.iterrows()]
        for payload in geom_payload:
            result = _create_prediction_mask(payload)
            self.write_message(result, outputs)
        
        self.write_message("Creating training mask", outputs)
        train_payload = []
        for _, row in gdf_filtered.iterrows():
            er = row['EcoRegion']
            vrts = self.find_provider_vrts(er, manual_downloads)
            standardized_vrts = [v if v.startswith('s3://') else f"s3://{v}" for v in vrts]
            if standardized_vrts:
                train_payload.append([er, standardized_vrts, outputs])
        
        for training_payload in train_payload:
            self.write_message(f'Starting: {training_payload[0]}', outputs)
            result = _create_training_mask(training_payload)
            self.write_message(result, outputs)
