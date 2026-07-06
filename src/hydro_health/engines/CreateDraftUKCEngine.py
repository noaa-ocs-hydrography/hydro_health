import os
import sys
import shutil
import tempfile
import fiona
import boto3
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import concurrent.futures
from botocore.exceptions import ClientError
from rasterio.windows import Window
from rasterio.features import rasterize
from rasterio.enums import MergeAlg, Resampling
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds, reproject
from rasterio.io import MemoryFile
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
from shapely.geometry import box
from urllib.parse import urlparse

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_environment

# ==============================================================================
# PARALLEL WORKER FUNCTIONS (Top-level for multiprocessing pickling)
# ==============================================================================

def get_wkt(crs):
    """Safely converts a CRS object to its WKT or string representation."""
    if crs is None: return None
    return crs.to_wkt() if hasattr(crs, 'to_wkt') else str(crs)

def _worker_draft(layer, input_gdb, orig_bounds, orig_crs_wkt, final_crs_wkt, temp_dir, out_meta, height, width, transform, block_size):
    """Worker process for reading, filtering, and rasterizing a single month's max draft data."""
    try:
        print(f"  [Worker Started] Processing Max Draft for Month: {layer}")
        with fiona.open(input_gdb, layer=layer) as src:
            layer_crs = src.crs
        
        if orig_crs_wkt and layer_crs:
            orig_box_gdf = gpd.GeoDataFrame({'geometry': [box(*orig_bounds)]}, crs=orig_crs_wkt)
            read_bounds = tuple(orig_box_gdf.to_crs(layer_crs).total_bounds)
        else:
            read_bounds = orig_bounds

        gdf = gpd.read_file(input_gdb, layer=layer, bbox=read_bounds)
        
        if gdf.empty:
            return None

        if gdf.crs != final_crs_wkt:
            gdf = gdf.to_crs(final_crs_wkt)

        if 'Draft' in gdf.columns:
            if gdf['Draft'].dtype == object or pd.api.types.is_string_dtype(gdf['Draft']):
                gdf['Draft'] = gdf['Draft'].astype(str).str.extract(r'(\d+\.?\d*)', expand=False)
            
            gdf['Draft'] = pd.to_numeric(gdf['Draft'], errors='coerce').fillna(0)
            gdf = gdf[gdf['Draft'] > 0]
        else:
            return None

        if gdf.empty:
            return None

        gdf = gdf.sort_values(by='Draft', ascending=True)
        sindex = gdf.sindex
        monthly_tif_path = os.path.join(temp_dir, f"{layer}_max_draft.tiff")
        
        with rasterio.open(monthly_tif_path, "w", **out_meta) as dest:
            for row in range(0, height, block_size):
                for col in range(0, width, block_size):
                    win_height = min(block_size, height - row)
                    win_width = min(block_size, width - col)
                    window = Window(col, row, win_width, win_height)
                    win_transform = rasterio.windows.transform(window, transform)
                    win_bounds = rasterio.windows.bounds(window, transform)
                    
                    possible_matches_index = list(sindex.intersection(win_bounds))
                    if not possible_matches_index:
                        dest.write(np.zeros((win_height, win_width), dtype=np.float32), 1, window=window)
                        continue
                        
                    possible_matches = gdf.iloc[possible_matches_index]
                    precise_matches = possible_matches[possible_matches.intersects(box(*win_bounds))]
                    
                    if precise_matches.empty:
                        dest.write(np.zeros((win_height, win_width), dtype=np.float32), 1, window=window)
                        continue
                        
                    shapes = ((geom, val) for geom, val in zip(precise_matches.geometry, precise_matches['Draft']))
                    burned = rasterize(
                        shapes=shapes,
                        out_shape=(win_height, win_width),
                        transform=win_transform,
                        fill=0,
                        all_touched=True,
                        merge_alg=MergeAlg.replace, 
                        dtype=np.float32
                    )
                    dest.write(burned, 1, window=window)
                    
        return monthly_tif_path

    except Exception as e:
        print(f"  > [Worker {layer}] FATAL ERROR: {e}")
        return None


class CreateDraftUKCEngine(Engine):
    """Class to hold the logic for processing the Max Draft and UKC rasters using S3 storage"""

    def __init__(self):
        super().__init__()
        # Configuration setup
        self.s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        # Local input (EC2 path)
        # Using an absolute path (~ resolves to /home/aubrey.mccutchen.lx) to prevent directory execution errors
        self.input_dir = os.path.abspath(os.path.expanduser("~/Repos/hydro_health/inputs/AIS_data"))
        self.input_gdb = f"{self.input_dir}/AIS_2024.gdb"
        
        # Processing parameters
        self.er_regions = ["ER_1", "ER_2", "ER_3", "ER_4", "ER_5", "ER_6"]
        
        # New Processing Configurations Array (100m processing ordered first)
        self.processing_configs = [
            {"res": 100, "suffix": "", "bathy_tag": "", "upload_regional": False, "master_bathy": True},
            {"res": 100, "suffix": "_offshore", "bathy_tag": "_Offshore", "upload_regional": True, "master_bathy": True},
            {"res": 20, "suffix": "_nearshore", "bathy_tag": "", "upload_regional": True, "master_bathy": False}
        ]
        
        self.max_workers = 8
        self.target_crs = "EPSG:6350"
        self.nodata_val = -9999.0
        
        # S3 Clients
        self.s3_client = boto3.client('s3')

    def _get_s3_uri(self, key: str) -> str:
        """Helper to format S3 URI"""
        return f"s3://{self.s3_bucket}/{key}"

    def generate_annual_max_draft(self, er_name: str, resolution: int, mask_local_path: str, draft_s3_key: str) -> bool:
        """Creates the Max Draft raster in memory/temp and pushes directly to S3"""
        
        print(f"\n--- Processing Max Draft for {er_name} at {resolution}m ---")
        
        # Pre-flight check: See if Draft raster already exists on S3 to prevent duplicate work
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=draft_s3_key)
            print(f"  > SUCCESS: Draft raster already exists on S3. Skipping generation -> s3://{self.s3_bucket}/{draft_s3_key}")
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                pass # Doesn't exist, proceed with generation
            else:
                print(f"  > S3 Error checking draft {draft_s3_key}: {e}")
                return False

        # Pre-flight check: Ensure the local mask exists before proceeding
        if not os.path.exists(mask_local_path):
            print(f"  > SKIP: Mask file not found locally -> {mask_local_path}")
            return False
        
        # Utilize TemporaryDirectory to ensure no files persist on local storage
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # 1. Setup Grid using local Mask
                with rasterio.open(mask_local_path) as mask_src:
                    orig_crs = mask_src.crs
                    orig_bounds = tuple(mask_src.bounds)
                    mask_nodata = mask_src.nodata

                target_crs_obj = rasterio.crs.CRS.from_string(self.target_crs)
                orig_crs_wkt = get_wkt(orig_crs)
                final_crs_wkt = get_wkt(target_crs_obj)

                if orig_crs and orig_crs != target_crs_obj:
                    minx, miny, maxx, maxy = transform_bounds(orig_crs, target_crs_obj, *orig_bounds)
                else:
                    minx, miny, maxx, maxy = orig_bounds

                # Dynamic Resolution Snapping
                minx = np.floor(minx / resolution) * resolution
                miny = np.floor(miny / resolution) * resolution
                maxx = np.ceil(maxx / resolution) * resolution
                maxy = np.ceil(maxy / resolution) * resolution
                width = int((maxx - minx) / resolution)
                height = int((maxy - miny) / resolution)
                transform = from_bounds(minx, miny, maxx, maxy, width, height)

                out_meta = {
                    "driver": "GTiff",
                    "height": height, "width": width,
                    "transform": transform, "crs": final_crs_wkt,
                    "count": 1, "dtype": 'float32', "nodata": self.nodata_val,
                    "compress": "lzw", "tiled": True,
                    "blockxsize": 1024, "blockysize": 1024
                }

                # 2. Dispatch multiprocessing workers
                if not os.path.exists(self.input_gdb):
                    print(f"  > ERROR: GDB not found at {self.input_gdb}. Check the path!")
                    return False
                
                try:
                    all_layers = fiona.listlayers(self.input_gdb)
                except Exception as e:
                    print(f"  > ERROR: Failed to open GDB at {self.input_gdb}. Is the .gdb file nested? Error: {e}")
                    return False

                ais_layers = [layer for layer in all_layers if layer.startswith("AIS_2024_")]
                valid_monthly_tifs = []

                if not ais_layers:
                    print("No AIS layers found in GDB.")
                    return False

                with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [executor.submit(
                        _worker_draft, layer, self.input_gdb, orig_bounds, orig_crs_wkt, final_crs_wkt, 
                        temp_dir, out_meta, height, width, transform, 4096
                    ) for layer in ais_layers]
                    
                    for future in concurrent.futures.as_completed(futures):
                        res_path = future.result()
                        if res_path: valid_monthly_tifs.append(res_path)

                if not valid_monthly_tifs:
                    print("No valid draft features found.")
                    return False

                # 3. Aggregate Final Raster and Upload directly to S3
                src_datasets = [rasterio.open(tif) for tif in valid_monthly_tifs]
                
                print("Aggregating blocks & writing final Draft to temp file...")
                final_draft_path = os.path.join(temp_dir, "final_draft_temp.tiff")
                with rasterio.open(final_draft_path, 'w', **out_meta) as dest:
                    with rasterio.open(mask_local_path) as mask_src_again:
                        for row in range(0, height, 4096):
                            for col in range(0, width, 4096):
                                win_height = min(4096, height - row)
                                win_width = min(4096, width - col)
                                window = Window(col, row, win_width, win_height)
                                win_transform = rasterio.windows.transform(window, transform)
                                
                                chunk_max = np.zeros((win_height, win_width), dtype=np.float32)
                                for src in src_datasets:
                                    chunk_max = np.maximum(chunk_max, src.read(1, window=window))
                                chunk_max[chunk_max == 0] = self.nodata_val

                                # Masking logic
                                mask_chunk = np.zeros((win_height, win_width), dtype=np.float32)
                                reproject(
                                    source=rasterio.band(mask_src_again, 1),
                                    destination=mask_chunk,
                                    src_transform=mask_src_again.transform, src_crs=mask_src_again.crs,
                                    dst_transform=win_transform, dst_crs=target_crs_obj,
                                    resampling=Resampling.nearest
                                )
                                
                                is_valid = (mask_chunk == 1) | ((mask_chunk != mask_nodata) & (mask_chunk > 0)) if mask_nodata else (mask_chunk == 1) | (mask_chunk > 0)
                                chunk_max[~is_valid] = self.nodata_val
                                dest.write(chunk_max, 1, window=window)

                # Upload from temp file to S3
                self.s3_client.upload_file(final_draft_path, self.s3_bucket, draft_s3_key)
                print(f"Max Draft successfully uploaded to s3://{self.s3_bucket}/{draft_s3_key}")

                for src in src_datasets:
                    src.close()
                return True

            except Exception as e:
                print(f"Error during Max Draft generation: {e}")
                return False


    def calculate_regional_ukc(self, er_name: str, resolution: int, draft_s3_key: str, bathy_s3_key: str, ukc_s3_key: str) -> bool:
        """Calculates UKC purely in-memory using S3 buffers"""

        print(f"\n--- Processing UKC for {er_name} at {resolution}m ---")
        draft_uri = self._get_s3_uri(draft_s3_key)
        bathy_uri = self._get_s3_uri(bathy_s3_key)
        
        # Pre-flight check for Bathy mosaic
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=bathy_s3_key)
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                print(f"  > SKIP: Bathy mosaic not found on S3 -> {bathy_uri}")
                return False
            else:
                print(f"  > S3 Error checking bathy {bathy_uri}: {e}")
                return False

        try:
            with rasterio.open(draft_uri) as draft_src:
                with rasterio.open(bathy_uri) as bathy_src:
                    
                    vrt_options = {
                        'resampling': Resampling.average, 
                        'crs': draft_src.crs,
                        'transform': draft_src.transform,
                        'height': draft_src.height,
                        'width': draft_src.width,
                    }
                    
                    out_profile = draft_src.profile.copy()
                    out_profile.update({
                        "nodata": self.nodata_val,
                        "BIGTIFF": "YES"
                    })

                    print(f"Calculating UKC and writing to temp file before S3 upload...")
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        final_ukc_path = os.path.join(temp_dir, "final_ukc_temp.tiff")
                        with WarpedVRT(bathy_src, **vrt_options) as vrt_bathy:
                            with rasterio.open(final_ukc_path, 'w', **out_profile) as dest:
                                for _, window in draft_src.block_windows(1):
                                    
                                    draft_chunk = draft_src.read(1, window=window)
                                    bathy_chunk = vrt_bathy.read(1, window=window)
                                    
                                    draft_nodata = np.isnan(draft_chunk) if np.isnan(draft_src.nodata) else (draft_chunk == draft_src.nodata)
                                    valid_draft = ~draft_nodata
                                    
                                    if bathy_src.nodata is not None:
                                        valid_bathy = ~np.isnan(bathy_chunk) if np.isnan(bathy_src.nodata) else (bathy_chunk != bathy_src.nodata)
                                    else:
                                        valid_bathy = np.ones_like(bathy_chunk, dtype=bool)
                                        
                                    valid_combined = valid_draft & valid_bathy
                                    
                                    ukc_chunk = np.full(draft_chunk.shape, self.nodata_val, dtype=np.float32)
                                    if np.any(valid_combined):
                                        ukc_chunk[valid_combined] = np.abs(bathy_chunk[valid_combined]) - draft_chunk[valid_combined]
                                        
                                    dest.write(ukc_chunk, window=window, indexes=1)
                                    
                        # Upload from temp file to S3
                        self.s3_client.upload_file(final_ukc_path, self.s3_bucket, ukc_s3_key)
                        print(f"UKC Mosaic successfully uploaded to s3://{self.s3_bucket}/{ukc_s3_key}")
                        return True
                            
        except Exception as e:
            print(f"Failed to process UKC. Error: {e}")
            return False

    def merge_regional_ukc(self, res: int, suffix: str, regional_ukc_keys: list) -> None:
        """Merges regional UKC rasters into a single master global mosaic and uploads to S3"""
        print(f"\n{'='*80}")
        print(f"STARTING GLOBAL MERGE FOR ALL ER REGIONS AT {res}m{suffix}")
        print(f"{'='*80}")
        
        if not regional_ukc_keys:
            print("No regional UKC mosaics generated to merge.")
            return

        regional_uris = [self._get_s3_uri(key) for key in regional_ukc_keys]
        sources = []

        for uri in regional_uris:
            try:
                src = rasterio.open(uri)
                sources.append(src)
                print(f"-> Successfully queued {uri} for merging.")
            except Exception as e:
                print(f"-> Warning: Could not open {uri}. Error: {e}")

        if len(sources) > 0:
            print("\nMerging all regional datasets in RAM... (This may take a moment)")
            mosaic_data, mosaic_transform = merge(sources, nodata=self.nodata_val)
            
            out_meta = sources[0].meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": mosaic_data.shape[1],
                "width": mosaic_data.shape[2],
                "transform": mosaic_transform,
                "nodata": self.nodata_val,
                "compress": "lzw",
                "tiled": True,
                "blockxsize": 1024,
                "blockysize": 1024,
                "BIGTIFF": "YES"
            })

            master_s3_key = f"low_res/ukc/UKC_Mosaic_{res}m{suffix}.tiff"
            
            print(f"Constructing master mosaic buffer in temp file and streaming directly to S3...")
            with tempfile.TemporaryDirectory() as temp_dir:
                master_temp_path = os.path.join(temp_dir, "master_mosaic.tiff")
                with rasterio.open(master_temp_path, 'w', **out_meta) as dest:
                    dest.write(mosaic_data)
                
                self.s3_client.upload_file(master_temp_path, self.s3_bucket, master_s3_key)
            
            print(f"-> Global mosaic successfully overwritten directly to S3: s3://{self.s3_bucket}/{master_s3_key}")

            for src in sources:
                src.close()
        else:
            print(f"\nError: No regional TIFFs were successfully opened. Master mosaic aborted.")

    def run(self) -> None:
        """Main Orchestration Loop"""
        print("Starting Draft and UKC Processing Engine...")
        
        for config in self.processing_configs:
            res = config["res"]
            suffix = config["suffix"]
            bathy_tag = config["bathy_tag"]
            master_bathy = config.get("master_bathy", False)
            
            print(f"\n{'#'*80}")
            print(f"STARTING BATCH RUN FOR: {res}m {suffix.upper().strip('_')}")
            print(f"{'#'*80}")
            
            regional_ukc_keys = []
            
            for er_name in self.er_regions:
                # S3 Key & Local Path Definitions (Mask is local)
                mask_local_path = os.path.join(self.input_dir, f"{er_name}_binary_mask_EPSG6350_{res}m.tiff")
                draft_s3_key = f"low_res/drafts/{er_name}_AIS_2024_Annual_Max_Draft_{res}m.tiff"
                
                # Dynamic bathy string formatting based on config rules
                if master_bathy:
                    bathy_s3_key = f"low_res/{res}m/BlueTopo_Bathy_Mosaic{bathy_tag}_{res}m.tiff"
                else:
                    bathy_s3_key = f"low_res/{res}m/{er_name}/{er_name}_BlueTopo_Bathy_Mosaic{bathy_tag}_{res}m.tiff"
                    
                ukc_s3_key = f"low_res/ukc/{er_name}_UKC_Mosaic_{res}m{suffix}.tiff"

                # 1. Draft Processing
                success = self.generate_annual_max_draft(er_name, res, mask_local_path, draft_s3_key)
                
                # 2. UKC Processing (Runs if Draft succeeds)
                if success:
                    ukc_success = self.calculate_regional_ukc(er_name, res, draft_s3_key, bathy_s3_key, ukc_s3_key)
                    if ukc_success:
                        regional_ukc_keys.append(ukc_s3_key)
                else:
                    print(f"Skipping UKC calculation for {er_name} at {res}m due to Draft generation failure.")

            # 3. Master Mosaic Merge
            self.merge_regional_ukc(res, suffix, regional_ukc_keys)

        print("Engine run completed successfully.")