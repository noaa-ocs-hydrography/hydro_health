import pathlib
import boto3
import re
import os
import sys
import psutil
import tempfile
import shutil
import concurrent.futures
import gc
import numpy as np
import rasterio
from botocore.config import Config
from osgeo import gdal, ogr, osr
# Make sure to import or define get_config_item in your new engine
# from hydro_health.helpers.tools import get_config_item

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class HHMSimpleEngine:
    def __init__(self):
        self.target_crs = "EPSG:6350"

    def crop_to_ecoregion(self, tiff_path: pathlib.Path, ecoregion_id: str, resolution: int) -> None:
        """
        Crop a single tile's TIFF to its ecoregion boundary if resolution is 20m.
        
        NOTE: When implementing your file loop, call this method like so:
        if target_res == 20:
            stem = tiff_file_path.stem
            for local_file in tile_folder.glob(f"{stem}*.tiff"):
                self.crop_to_ecoregion(local_file, ecoregion_id, target_res)
        """

        if resolution != 20:
            return

        master_gpkg_path = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

        # Find cutline layer name (Enhanced_EcoRegions)
        cutline_layer_name = None
        er_field_name = None
        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            for name in layer_names:
                if name.lower().replace("_", "").replace(" ", "") == "enhancedecoregions":
                    cutline_layer_name = name
                    break
            if cutline_layer_name:
                layer = ds.GetLayerByName(cutline_layer_name)
                layer_defn = layer.GetLayerDefn()
                field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
                for expected in ['ecoregion_id', 'ecoregion', 'er', 'region', 'name', 'id']:
                    for field in field_names:
                        if field.lower() == expected:
                            er_field_name = field
                            break
                    if er_field_name:
                        break
            ds = None

        if not cutline_layer_name:
            print(f"[{tiff_path.name}] Enhanced_EcoRegions layer not found in Master_Grids.gpkg, skipping crop.")
            return

        temp_cropped = tiff_path.parent / f"crop_{tiff_path.name}"

        # Build query robustly matching various formats (e.g., 'ER_1', 'ER1', '1')
        match = re.search(r'\d+', str(ecoregion_id))
        clean_er_num = match.group(0) if match else str(ecoregion_id)
        er_prefix_val = f"ER_{clean_er_num}"
        er_val_no_underscore = f"ER{clean_er_num}"

        where_clauses = []
        if er_field_name:
            where_clauses.extend([
                f"{er_field_name} = '{er_prefix_val}'",
                f"{er_field_name} = '{er_val_no_underscore}'",
                f"{er_field_name} = '{ecoregion_id}'",
                f"{er_field_name} = '{clean_er_num}'"
            ])
            if clean_er_num.isdigit():
                where_clauses.append(f"{er_field_name} = {clean_er_num}")
        else:
            print(f"[{tiff_path.name}] Warning: Ecoregion ID field name not found. Attempting crop without filter...")

        cutline_where = " OR ".join(where_clauses) if where_clauses else None

        # Pass srcSRS explicitly so GDAL always aligns the source data and cutline CRS correctly
        warp_kwargs = {
            "format": "GTiff",
            "srcSRS": self.target_crs,
            "dstSRS": self.target_crs,
            "dstNodata": -9999,
            "cutlineDSName": str(master_gpkg_path),
            "cutlineLayer": cutline_layer_name,
            "cropToCutline": True,
            "creationOptions": ["COMPRESS=DEFLATE"]
        }
        if cutline_where:
            warp_kwargs["cutlineWhere"] = cutline_where

        warp_options = gdal.WarpOptions(**warp_kwargs)

        try:
            ds_crop = gdal.Warp(str(temp_cropped), str(tiff_path), options=warp_options)
            ds_crop = None # EXPLICIT CLEANUP
            
            if temp_cropped.exists():
                tiff_path.unlink()
                temp_cropped.rename(tiff_path)
        except Exception as e:
            print(f"[{tiff_path.name}] Failed to crop to ecoregion: {e}")
            if temp_cropped.exists():
                temp_cropped.unlink()

    def create_hurricane_tile(self, tiff_file_path: pathlib.Path, resolution: int) -> None:
        """
        Generate a hurricane count raster by evaluating the survey year and 
        accumulating the corresponding yearly hurricane counts for each cell.
        """

        print(f"[{tiff_file_path.name}] Creating cumulative hurricane count tile...")
        stem = tiff_file_path.stem
        hurricane_tile_path = tiff_file_path.parents[0] / f'{stem}_hurricane.tiff'

        # Local survey path always exists because we are enforcing unconditional local reprocessing
        local_survey_path = tiff_file_path.parents[0] / f'{stem}_survey_end_date.tiff'
        survey_ds_path = str(local_survey_path)

        try:
            with rasterio.open(survey_ds_path) as src:
                survey_years = src.read(1)
                transform = src.transform
                nodata = -9999
                width = src.width
                height = src.height
        except rasterio.errors.RasterioIOError:
            print(f"[{stem}] Error: Survey end date raster required for Hurricane calculation could not be loaded from {survey_ds_path}")
            return

        minx = transform.c
        maxy = transform.f
        maxx = minx + transform.a * width
        miny = maxy + transform.e * height

        # Create a rigorous valid mask specifically ignoring 0, nodata, or nan
        valid_mask = (survey_years != nodata) & (survey_years > 0) & ~np.isnan(survey_years)

        unique_years = np.unique(survey_years[valid_mask])
        valid_years = [y for y in unique_years]

        hurricane_path_prefix = get_config_item('HURRICANE', 'COUNT_RASTER_PATH')

        # Initialize everything to explicit nodata.
        result_array = np.full(survey_years.shape, nodata, dtype=np.float32)

        if not valid_years:
            with rasterio.open(
                hurricane_tile_path,
                "w",
                driver="GTiff",
                count=1,
                width=width,
                height=height,
                dtype=rasterio.float32,
                compress="lzw",
                tiled=True,
                blockxsize=512,
                blockysize=512,
                crs=self.target_crs,
                transform=transform,
                nodata=nodata,
            ) as dst:
                dst.write(result_array, 1)
            return

        def get_year_count_array(target_year):
            # Clamp year to valid data range
            target_year = max(1851, min(2023, int(target_year)))

            # Construct both the .tiff and the .tif fallbacks
            s3_bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            s3_uri_tiff = f"/vsis3/{s3_bucket}/{hurricane_path_prefix}/cumulative_count_{target_year}.tiff"
            s3_uri_tif = f"/vsis3/{s3_bucket}/{hurricane_path_prefix}/cumulative_count_{target_year}.tif"

            warp_options = gdal.WarpOptions(
                format="MEM",
                outputBounds=(minx, miny, maxx, maxy),
                width=width,   
                height=height,  
                dstSRS=self.target_crs,
                resampleAlg=gdal.GRA_NearestNeighbour, # Counts should not be interpolated
                dstNodata=-9999
            )

            mem_ds = None
            gdal.PushErrorHandler('CPLQuietErrorHandler')

            # 1. Attempt the .tiff version first (catching RuntimeError if UseExceptions is active)
            try:
                mem_ds = gdal.Warp('', s3_uri_tiff, options=warp_options)
            except Exception:
                mem_ds = None

            # 2. If the .tiff version failed, try the .tif version as a fallback
            if mem_ds is None:
                try:
                    mem_ds = gdal.Warp('', s3_uri_tif, options=warp_options)
                except Exception:
                    mem_ds = None

            gdal.PopErrorHandler()

            if mem_ds is None:
                print(f"[{stem}] Warning: Could not load hurricane raster for year {target_year} at either {s3_uri_tiff} or {s3_uri_tif}. Returning zeros.")
                return np.zeros((height, width), dtype=np.float32)

            arr = mem_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
            arr[arr == -9999] = 0.0 
            
            mem_ds = None # EXPLICIT CLEANUP TO PREVENT LEAK
            return arr

        # Start hurricane count at 0.0 ONLY for cells with a valid survey year
        result_array[valid_mask] = 0.0

        min_survey_year = max(1851, int(min(valid_years)))
        max_hurricane_year = 2023

        print(f"[{stem}] Aggregating yearly hurricane counts from {min_survey_year} to {max_hurricane_year}...")
        for target_year in range(min_survey_year, max_hurricane_year + 1):
            year_arr = get_year_count_array(target_year)

            # Add this year's count to any valid pixel whose survey year is <= target_year
            add_mask = valid_mask & (survey_years <= target_year)
            result_array[add_mask] += year_arr[add_mask]

        # FINAL SAFETY NET: forcefully reset any invalid cells (like survey_year = 0) back to nodata
        result_array[~valid_mask] = nodata

        with rasterio.open(
            hurricane_tile_path,
            "w",
            driver="GTiff",
            count=1,
            width=width,
            height=height,
            dtype=rasterio.float32,
            compress="lzw",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            crs=self.target_crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(result_array, 1)
            factors = [2, 4, 8, 16]
            dst.build_overviews(factors, rasterio.enums.Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')

    def create_combined_isobaths(self, ecoregion_ids: list[str], s3_bucket: str, current_res: int, output_prefix: str|bool) -> None:
        """
        Creates solid polygon isobaths for specified depth bands (0-20m and 0-40m)
        and appends them directly to the local isobath_layers.gpkg.
        
        Uses an efficient tile-by-tile memory strategy to prevent VRT rendering dropouts,
        S3 connection thrashing, and massive disk usage.
        """

        if current_res == 20:
            print(f"[All Regions] Target resolution ({current_res}m) is 20m. Skipping isobath generation.")
            return

        # Changed to save to a separate newly created GPKG
        isobath_gpkg_path = pathlib.Path("/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/isobath_layers.gpkg")

        print(f"[All Regions] Generating combined polygon isobaths and saving to local {isobath_gpkg_path}...")
        s3_client = boto3.client('s3')

        # Optimize GDAL for S3 reads: allows dynamic block chunking directly from the web
        gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
        gdal.SetConfigOption('VSI_CACHE', 'YES')

        driver = ogr.GetDriverByName("GPKG")
        if isobath_gpkg_path.exists():
            isobath_ds = driver.Open(str(isobath_gpkg_path), 1)  # 1 indicates update mode
        else:
            isobath_ds = driver.CreateDataSource(str(isobath_gpkg_path))

        if isobath_ds is None:
            print(f"Error: Could not open or create {isobath_gpkg_path}")
            return

        srs = osr.SpatialReference()
        srs.SetFromUserInput(self.target_crs)

        # We process negative depths. 0 to -20 is the first band, 0 to -40 is the second.
        depth_bands = [
            ((0.0, -20.0), 'isobath_0_20m'),
            ((0.0, -40.0), 'isobath_0_40m')
        ]

        # Check if layers already exist to skip recreation
        existing_layers = [isobath_ds.GetLayerByIndex(i).GetName() for i in range(isobath_ds.GetLayerCount())]
        expected_layers = [layer_name for _, layer_name in depth_bands]

        # Skip if they exist
        if all(layer in existing_layers for layer in expected_layers):
            print(f"[All Regions] Isobath layers already exist in {isobath_gpkg_path}. Skipping creation.")
            isobath_ds = None
            return

        # Ensure layers exist in the isobath_layers.gpkg as POLYGONS (unconditionally overwrite if we reach here)
        layers = {}
        for _, layer_name in depth_bands:
            # Safely delete layer if it already exists by checking integer indices
            for i in range(isobath_ds.GetLayerCount()):
                if isobath_ds.GetLayerByIndex(i).GetName() == layer_name:
                    isobath_ds.DeleteLayer(i)
                    break

            layer = isobath_ds.CreateLayer(layer_name, srs=srs, geom_type=ogr.wkbPolygon)
            field_defn = ogr.FieldDefn("elevation", ogr.OFTReal)
            layer.CreateField(field_defn)
            layers[layer_name] = layer

        # Process one ecoregion at a time
        for ecoregion_id in ecoregion_ids:
            print(f"[{ecoregion_id}] Gathering S3 links...")
            search_prefix = f"low_res/{current_res}m/{ecoregion_id}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo"
            if output_prefix and output_prefix.strip('/') != "low_res":
                search_prefix = f"{output_prefix}/{search_prefix}"

            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix)

            vsis3_paths = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        # Strictly look for files ending with digits + .tiff (case insensitive)
                        if re.search(r'_[0-9]{8}\.tiff?$', key, re.IGNORECASE):
                            vsis3_paths.append(f"/vsis3/{s3_bucket}/{key}")

            if not vsis3_paths:
                print(f"[{ecoregion_id}] No base tiles found. Skipping.")
                continue

            print(f"[{ecoregion_id}] Processing {len(vsis3_paths)} tiles individually via memory to guarantee 100% data extraction...")

            for path in vsis3_paths:
                src_ds = gdal.Open(path)
                if src_ds is None:
                    continue

                src_band = src_ds.GetRasterBand(1)

                # Fetch full tile from S3 instantly into memory (BlueTopo tiles are typically small enough for this)
                data = src_band.ReadAsArray()

                if data is None:
                    src_band = None
                    src_ds = None
                    continue

                # Skip entirely nodata/empty tiles quickly
                if not np.any(data != -9999):
                    src_band = None
                    src_ds = None
                    continue

                for depth_range, layer_name in depth_bands:
                    upper, lower = depth_range

                    # Create the boolean mask array
                    mask_data = np.zeros_like(data, dtype=np.uint8)
                    mask_data[(data <= upper) & (data >= lower)] = 1

                    # Only polygonize if there are actually matches in this tile
                    if np.any(mask_data):
                        # Use GDAL memory driver for rapid, disk-free mask polygonization
                        drv = gdal.GetDriverByName('MEM')
                        mask_ds = drv.Create('', src_ds.RasterXSize, src_ds.RasterYSize, 1, gdal.GDT_Byte)
                        mask_ds.SetGeoTransform(src_ds.GetGeoTransform())
                        mask_ds.SetProjection(src_ds.GetProjection())

                        mask_band = mask_ds.GetRasterBand(1)
                        # Setting NoDataValue to 0 instructs Polygonize to only outline the '1' values
                        mask_band.SetNoDataValue(0)
                        mask_band.WriteArray(mask_data)

                        # Polygonize directly to the active layer (GDAL Polygonize internally manages transactions for each operation safely)
                        gdal.Polygonize(mask_band, mask_band, layers[layer_name], -1, [], callback=None)

                        mask_band = None
                        mask_ds = None

                src_band = None
                src_ds = None

        print(f"[All Regions] Saving and securely closing {isobath_gpkg_path}...")
        layers.clear()
        isobath_ds = None

    def _download_vsis3_paths_concurrently(self, vsis3_paths: list[str], temp_dir: pathlib.Path) -> list[str]:
        """
        [OPTION 1 FIX] Bulk download tiles from S3 to local disk before GDAL processing.
        Significantly reduces network thrashing inside the GDAL Warp operations.
        """
        local_paths = []
        download_tasks = []

        for vsi_path in vsis3_paths:
            parts = vsi_path.replace("/vsis3/", "").split("/", 1)
            if len(parts) == 2:
                bucket, key = parts
                # Flatten the path to prevent folder creation issues and guarantee uniqueness
                safe_filename = key.replace("/", "_")
                local_path = temp_dir / safe_filename
                download_tasks.append((bucket, key, local_path))
        
        # FIXED: Initialize one single globally shared Boto3 client context pool
        s3_pool_client = boto3.client('s3', config=Config(max_pool_connections=32))

        def _download_task(task_info):
            bucket, key, local_path = task_info
            if not local_path.exists():
                s3_pool_client.download_file(bucket, key, str(local_path))
            return str(local_path)

        mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print(f"      [S3 Pre-Download] FAST FETCH: Fetching {len(download_tasks)} files locally to eliminate S3 warp latency... (RAM: {mem_mb:.1f}MB)")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(_download_task, task) for task in download_tasks]
            for future in concurrent.futures.as_completed(futures):
                try:
                    local_paths.append(future.result())
                except Exception as e:
                    print(f"      [S3 Pre-Download] WARNING: File download failed: {e}")

        print(f"      [S3 Pre-Download] Complete. {len(local_paths)} files perfectly staged locally.")
        
        # Clear out Boto3 client connections to free RAM
        del s3_pool_client
        gc.collect()
        
        return local_paths

    def _build_local_mosaic(self, target_path: pathlib.Path, paths: list[str], warp_kwargs: dict) -> None:
        """
        Process a massive list of inputs locally using a highly-optimized Two-Step Translate/Warp Pipeline.
        Step 1: Translate VRT -> Stitched Temp TIFF (Very Fast, no vector intersection math)
        Step 2: Warp Temp TIFF -> Final TIFF (Applies cutline precisely on a single file, rather than 700+)
        """
        gdal.UseExceptions()

        with tempfile.TemporaryDirectory(dir=target_path.parent) as temp_dir:
            temp_dir_path = pathlib.Path(temp_dir)
            
            # 1. PULL EVERYTHING TO LOCAL FAST STORAGE FIRST
            local_paths = self._download_vsis3_paths_concurrently(paths, temp_dir_path)

            if not local_paths:
                raise RuntimeError("No files were successfully downloaded to build the mosaic.")

            print(f"      [Mosaic Final Render] Building Virtual Raster (VRT) for {len(local_paths)} local files...")
            final_vrt = str(temp_dir_path / "final_merge.vrt")
            
            # Use the provided warp arguments directly (now handles clipping + mosaicking simultaneously)
            warp_kwargs_base = warp_kwargs.copy()
            
            # Extract final-stage only parameters for vector masking so they don't break the Translate phase
            cutline_ds = warp_kwargs_base.pop("cutlineDSName", None)
            cutline_layer = warp_kwargs_base.pop("cutlineLayer", None)
            cutline_where = warp_kwargs_base.pop("cutlineWhere", None)
            crop_to_cutline = warp_kwargs_base.pop("cropToCutline", False)

            vrt_options = gdal.BuildVRTOptions(
                resampleAlg=warp_kwargs_base.get("resampleAlg", gdal.GRA_NearestNeighbour),
                srcNodata=warp_kwargs_base.get("srcNodata", -9999),
                VRTNodata=warp_kwargs_base.get("dstNodata", -9999)
            )
            vrt_ds = gdal.BuildVRT(final_vrt, local_paths, options=vrt_options)
            vrt_ds = None # Must flush to disk, or VRT remains cached indefinitely in RAM

            # Custom progress callback function for GDAL
            def custom_progress_callback(complete, message, cb_data):
                percent = int(complete * 100)
                if percent > cb_data['last_printed']:
                    print(f"        -> {cb_data['step_name']} Progress: {percent}%")
                    sys.stdout.flush() # Force output to console immediately so it doesn't look hung
                    cb_data['last_printed'] = percent
                return 1

            solid_tiff = str(temp_dir_path / "solid_mosaic.tiff")

            print(f"      [Mosaic Final Render] Executing Step 1/2: Stitching 700+ tiles into a single master raster... (Fast Translate)")
            try:
                # Step 1: Rapidly stitch without worrying about vector math bounds
                cb_data_1 = {'last_printed': -1, 'step_name': 'Stitching'}
                trans_opts = gdal.TranslateOptions(
                    format="GTiff",
                    creationOptions=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"],
                    noData=-9999,
                    callback=custom_progress_callback,
                    callback_data=cb_data_1
                )
                trans_ds = gdal.Translate(solid_tiff, final_vrt, options=trans_opts)
                trans_ds = None
                
                print(f"      [Mosaic Final Render] Executing Step 2/2: Applying Vector Mask... (Single File Warp)")
                # Step 2: Now that it's a single file, safely apply the vector mask without crashing GDAL threading
                if cutline_ds:
                    mask_kwargs = warp_kwargs_base.copy()
                    mask_kwargs["cutlineDSName"] = cutline_ds
                    if cutline_layer:
                        mask_kwargs["cutlineLayer"] = cutline_layer
                    if cutline_where:
                        mask_kwargs["cutlineWhere"] = cutline_where
                    mask_kwargs["cropToCutline"] = crop_to_cutline
                    
                    # Prevent I/O thread lockup, the single file warp is natively fast enough
                    mask_kwargs["multithread"] = False 
                    
                    cb_data_2 = {'last_printed': -1, 'step_name': 'Masking'}
                    warp_ds = gdal.Warp(
                        str(target_path), 
                        solid_tiff, 
                        options=gdal.WarpOptions(**mask_kwargs), 
                        callback=custom_progress_callback, 
                        callback_data=cb_data_2
                    )
                    warp_ds = None
                else:
                    # Fallback if no mask was provided
                    shutil.copy(solid_tiff, str(target_path))

            except Exception as e:
                warn_msg = f"        -> FATAL: Mosaic stitch/mask failed. Error: {e}"
                print(warn_msg)
                raise e
                
        # Force Python garbage collection
        gc.collect()

    def create_and_upload_er_mosaics(self, ecoregion_ids: list[str], s3_bucket: str, current_res: int, output_prefix: str|bool, increased_scale: bool=False) -> None:
        """
        Creates isolated mosaics for each individual ecoregion and precisely masks them
        to their specific boundary polygon from the EcoRegions layer.
        """

        low_res_dir = OUTPUTS / "low_res" / f'{current_res}m'
        low_res_dir.mkdir(parents=True, exist_ok=True)

        master_gpkg_path = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

        s3_client = boto3.client('s3')

        # Flattened S3 Pagination: Page bucket ONCE instead of inside loop
        print(f"[ER Mosaic] Paginating S3 to gather all keys upfront to minimize API calls...")
        search_prefix_base = f"low_res/{current_res}m/"
        if output_prefix and output_prefix.strip('/') != "low_res":
            search_prefix_base = f"{output_prefix}/{search_prefix_base}"
        
        all_s3_keys = []
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix_base)
        
        for page_num, page in enumerate(pages, start=1):
            if 'Contents' in page:
                for obj in page['Contents']:
                    all_s3_keys.append(obj['Key'])
            
            # Print S3 pagination progress
            if page_num % 10 == 0:
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                print(f"      [S3 Paginator] Scanned {page_num} pages, found {len(all_s3_keys)} keys so far... (RAM: {mem_mb:.1f}MB)")

        print(f"[ER Mosaic] Finished scanning S3. Total keys retrieved: {len(all_s3_keys)}")
        all_s3_keys_set = set(all_s3_keys)

        # Locate the EcoRegions layer once for masking
        cutline_layer_name = None
        er_field_name = None

        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            for name in layer_names:
                if name.lower().replace("_", "").replace(" ", "") == "enhancedecoregions":
                    cutline_layer_name = name
                    break

            # Intelligently discover the column name containing the Ecoregion ID
            if cutline_layer_name:
                layer = ds.GetLayerByName(cutline_layer_name)
                layer_defn = layer.GetLayerDefn()
                field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]

                # Prioritize field names to avoid grabbing a generic 'id' if 'ecoregion' exists
                for expected in ['ecoregion_id', 'ecoregion', 'er', 'region', 'name', 'id']:
                    for field in field_names:
                        if field.lower() == expected:
                            er_field_name = field
                            break
                    if er_field_name:
                        break
            ds = None
        else:
            print(f"[ER Mosaic] WARNING: Could not open {master_gpkg_path}. Proceeding without mask...")

        # Base warp arguments allowing GDAL to naturally unify projections perfectly
        # Set targetAlignedPixels to perfectly align the grids eliminating edge artifact grid lines
        warp_kwargs_base = {
            "format": "GTiff",
            "dstSRS": self.target_crs,
            "xRes": current_res,
            "yRes": current_res,
            "srcNodata": -9999,
            "dstNodata": -9999,
            "targetAlignedPixels": True, 
            "multithread": False, # Stop complex Vector/Thread lockups here
            "warpOptions": ["NUM_THREADS=4"], # Safe chunk-based threading (prevents vector deadlock)
            "warpMemoryLimit": 512 * 1024 * 1024, # 512MB Limit to brutally stop OS RAM Thrashing
            "creationOptions": ["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
        }

        for ecoregion_id in ecoregion_ids:
            # -------------------------------------------------------------------------------------
            # SKIP LOGIC: Only generate ER mosaics for ER 6 as requested
            # -------------------------------------------------------------------------------------
            match = re.search(r'\d+', str(ecoregion_id))
            clean_er_num = match.group(0) if match else str(ecoregion_id)

            if clean_er_num not in ["6"]:
                print(f"[ER Mosaic] Skipping mosaic generation for {ecoregion_id} as only ER 6 is requested.")
                continue
            # -------------------------------------------------------------------------------------

            er_prefix = ecoregion_id if str(ecoregion_id).startswith("ER_") else f"ER_{ecoregion_id}"

            # Apply a specific SQL WHERE clause to the vector file to grab ONLY this Ecoregion's polygon
            er_warp_kwargs = warp_kwargs_base.copy()
            temp_mask_path = str(low_res_dir / f"{ecoregion_id}_mask.geojson")

            if cutline_layer_name and er_field_name:
                # Robust extraction of the ecoregion number to prevent ER_1/ER1 split matching errors
                er_prefix_val = f"ER_{clean_er_num}"
                er_val_no_underscore = f"ER{clean_er_num}"

                # Build a robust SQL query that perfectly matches any possible GPKG ID format
                where_clauses = [
                    f"{er_field_name} = '{er_prefix_val}'",
                    f"{er_field_name} = '{er_val_no_underscore}'",
                    f"{er_field_name} = '{ecoregion_id}'",
                    f"{er_field_name} = '{clean_er_num}'"
                ]
                if clean_er_num.isdigit():
                    where_clauses.append(f"{er_field_name} = {clean_er_num}")
                    
                print(f"[{ecoregion_id}] Extracting vector mask to memory-safe GeoJSON to prevent SQLite thread locking...")
                ds_gpkg = ogr.Open(str(master_gpkg_path))
                if ds_gpkg:
                    layer = ds_gpkg.GetLayerByName(cutline_layer_name)
                    layer.SetAttributeFilter(" OR ".join(where_clauses))
                    
                    if os.path.exists(temp_mask_path):
                        os.remove(temp_mask_path)
                        
                    drv = ogr.GetDriverByName("GeoJSON")
                    out_ds = drv.CreateDataSource(temp_mask_path)
                    out_layer = out_ds.CreateLayer("mask", srs=layer.GetSpatialRef(), geom_type=layer.GetGeomType())
                    
                    for feat in layer:
                        out_layer.CreateFeature(feat)
                        
                    out_ds = None
                    ds_gpkg = None
                    
                    er_warp_kwargs["cutlineDSName"] = temp_mask_path
                    er_warp_kwargs["cropToCutline"] = True

            # Map configuration properties dictating regex mapping
            # REMOVED Slope from base config to calculate at the mosaic level
            base_mosaics_config = {
                "Bathy": re.compile(r'_[0-9]{8}\.tiff?$', re.IGNORECASE),
                "ISS": re.compile(rf"_ISS_all{'_110' if increased_scale else ''}\.tiff?$", re.IGNORECASE),
                "Survey_Date": re.compile(r'_survey_end_date\.tiff?$', re.IGNORECASE),
                "Hurricane": re.compile(r'_hurricane\.tiff?$', re.IGNORECASE),
            }

            mosaics_config = {}
            for name, regex in base_mosaics_config.items():
                # Omit BlueTopo prefix if we are generating Hurricane mosaics
                bt_prefix = "" if name.lower() == "hurricane" else "BlueTopo_"

                mosaics_config[name] = {
                    "regex": regex,
                    "filename": f"{er_prefix}_{bt_prefix}{name}_Mosaic_{current_res}m{'_110' if increased_scale and name == 'ISS' else ''}.tiff",
                    "paths": [],
                    "exclude_nearshore_bands": False
                }

            mosaics_to_process = mosaics_config
            for m_key, config in mosaics_to_process.items():
                s3_key = f"low_res/{current_res}m/{ecoregion_id}/{config['filename']}"
                if output_prefix and output_prefix.strip('/') != "low_res":
                    s3_key = f"{output_prefix}/{s3_key}"
                config['s3_key'] = s3_key

            print(f"[{ecoregion_id}] Mapping local S3 keys for ER specific mosaics: {list(mosaics_to_process.keys())}")
            
            # Use dynamic folder checking to ensure robust mapping whether it's '1' or 'ER_1'
            valid_folders = [f"/{ecoregion_id}/", f"/ER_{clean_er_num}/", f"/ER{clean_er_num}/"]

            for key in all_s3_keys:
                if not any(f in f"/{key}" for f in valid_folders):
                    continue
                if f"/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo/" not in f"/{key}":
                    continue
                    
                is_nearshore = bool(re.search(r'/BH[45]', key, re.IGNORECASE))
                for m_key, config in mosaics_to_process.items():
                    if config['regex'].search(key):
                        if config.get('exclude_nearshore_bands') and is_nearshore:
                            continue
                        config['paths'].append(f"/vsis3/{s3_bucket}/{key}")

            # Process standard mosaics before offshore mosaics
            ordered_mosaics = sorted(mosaics_to_process.items(), key=lambda item: 1 if "Offshore" in item[0] else 0)
            for m_key, config in ordered_mosaics:
                if not config['paths']:
                    print(f"[{ecoregion_id}] No base tiles found for {m_key}. Skipping.")
                    continue

                local_mosaic_path = low_res_dir / config['filename']
                s3_key = config['s3_key']

                # Categorical components require nearest neighbor resampling
                m_key_lower = m_key.lower()
                
                # Check if mosaics already exist to skip heavy processing
                if m_key_lower == "bathy" or m_key_lower == "bathy_offshore":
                    slope_s3_key = s3_key.replace("Bathy", "Slope")
                    if s3_key in all_s3_keys_set and slope_s3_key in all_s3_keys_set:
                        print(f"[{ecoregion_id}] {m_key} and its Slope mosaic already exist in S3. Skipping.")
                        continue
                else:
                    if s3_key in all_s3_keys_set:
                        print(f"[{ecoregion_id}] {m_key} mosaic already exists in S3. Skipping.")
                        continue

                if "bathy" in m_key_lower or "unc" in m_key_lower:
                    resample_alg = gdal.GRA_Bilinear
                else:
                    resample_alg = gdal.GRA_NearestNeighbour

                current_warp_kwargs = er_warp_kwargs.copy()
                current_warp_kwargs["resampleAlg"] = resample_alg

                print(f"[{ecoregion_id}] Warping and precisely masking {m_key} ER mosaic dynamically from {len(config['paths'])} source files...")
                
                # Single pass directly to output
                try:
                    self._build_local_mosaic(local_mosaic_path, config['paths'], current_warp_kwargs)
                except Exception as e:
                    error_msg = f"[{ecoregion_id}] ERROR: Mosaic generation failed for {m_key}. Exception details: {e}"
                    print(error_msg)
                    continue # Safely skip to the next mosaic type instead of crashing the whole pipeline

                print(f"[{ecoregion_id}] Generating robust overviews for {m_key}...")
                ds = gdal.Open(str(local_mosaic_path), gdal.GA_Update)
                if ds is not None:
                    ds.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                    ds = None

                print(f"[{ecoregion_id}] Uploading {m_key} ER mosaic to s3://{s3_bucket}/{s3_key}...")
                s3_client.upload_file(
                    str(local_mosaic_path), 
                    s3_bucket, 
                    s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'resolution': str(current_res),
                            'crs': self.target_crs
                        }
                    }
                )
                print(f"[{ecoregion_id}] {m_key} ER mosaic creation and upload perfectly complete.")

                # Generate Slope directly from Bathy to prevent tile-edge artifacts
                if m_key_lower == "bathy":
                    slope_key = m_key.replace("Bathy", "Slope")
                    slope_filename = config['filename'].replace("Bathy", "Slope")
                    slope_local_path = low_res_dir / slope_filename
                    slope_s3_key = config['s3_key'].replace("Bathy", "Slope")
                    
                    print(f"[{ecoregion_id}] Generating {slope_key} mosaic directly from {m_key} mosaic to prevent edge artifacts...")
                    slope_ds = gdal.DEMProcessing(
                        str(slope_local_path), 
                        str(local_mosaic_path), 
                        'slope', 
                        creationOptions=["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
                    )
                    slope_ds = None # FIXED: Close dataset immediately
                    
                    print(f"[{ecoregion_id}] Generating robust overviews for {slope_key}...")
                    ds_slope = gdal.Open(str(slope_local_path), gdal.GA_Update)
                    if ds_slope is not None:
                        ds_slope.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                        ds_slope = None
                        
                    print(f"[{ecoregion_id}] Uploading completely masked {slope_key} mosaic to s3://{s3_bucket}/{slope_s3_key}...")
                    s3_client.upload_file(
                        str(slope_local_path), 
                        s3_bucket, 
                        slope_s3_key,
                        ExtraArgs={
                            'Metadata': {
                                'resolution': str(current_res),
                                'crs': self.target_crs
                            }
                        }
                    )
                    print(f"[{ecoregion_id}] {slope_key} ER mosaic creation and upload perfectly complete.")
                    if slope_local_path.exists():
                        slope_local_path.unlink()

                if local_mosaic_path.exists():
                    local_mosaic_path.unlink() # Save disk space
            
            # Clean up the temporary extracted mask for this ER
            if os.path.exists(temp_mask_path):
                os.remove(temp_mask_path)
                
            gc.collect()

    def create_and_upload_mosaics(self, ecoregion_ids: list[str], s3_bucket: str, current_res: int, output_prefix: str|bool, increased_scale: bool=False) -> None:
        """
        Creates massive mosaics from all base bathy tiles and derivatives across all processed ecoregions,
        masks them securely with the EcoRegions layer, saves locally to low_res, and uploads to S3.
        """

        if current_res < 50:
            print(f"[Mosaic] Target resolution ({current_res}m) is under 50m. Skipping massive mosaic generation to prevent extreme file sizes.")
            return

        low_res_dir = OUTPUTS / "low_res" / f'{current_res}m'
        low_res_dir.mkdir(parents=True, exist_ok=True)

        master_gpkg_path = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')

        s3_client = boto3.client('s3')

        # Configuration holding the regex mapping (All variables uncommented)
        # REMOVED Slope from base config to calculate at the mosaic level
        base_mosaics_config = {
            "Bathy": re.compile(r'_[0-9]{8}\.tiff?$', re.IGNORECASE),
            "ISS": re.compile(rf"_ISS_all{'_110' if increased_scale else ''}\.tiff?$", re.IGNORECASE),
            "Survey_Date": re.compile(r'_survey_end_date\.tiff?$', re.IGNORECASE),
            "Hurricane": re.compile(r'_hurricane\.tiff?$', re.IGNORECASE),
        }

        mosaics_config = {}
        for name, regex in base_mosaics_config.items():
            # Drop BlueTopo prefix if we are processing Hurricane
            bt_prefix = "" if name.lower() == "hurricane" else "BlueTopo_"

            # Commented out the standard national 100m mosaics as requested
            if current_res != 100:
                mosaics_config[name] = {
                    "regex": regex,
                    "filename": f"{bt_prefix}{name}_Mosaic_{current_res}m{'_110' if increased_scale and name == 'ISS' else ''}.tiff",
                    "paths": [],
                    "exclude_nearshore_bands": False
                }
            # Create a secondary non-Band4/5 version specifically for 100m runs
            if current_res == 100:
                mosaics_config[f"{name}_Offshore"] = {
                    "regex": regex,
                    "filename": f"{bt_prefix}{name}_Mosaic_Offshore_{current_res}m{'_110' if increased_scale and name == 'ISS' else ''}.tiff",
                    "paths": [],
                    "exclude_nearshore_bands": True
                }

        # Unconditionally process all configured mosaics
        mosaics_to_process = mosaics_config
        for m_key, config in mosaics_to_process.items():
            s3_key = f"low_res/{current_res}m/{config['filename']}"
            if output_prefix and output_prefix.strip('/') != "low_res":
                s3_key = f"{output_prefix}/{s3_key}"
            config['s3_key'] = s3_key

        print(f"[Mosaic] Gathering S3 links across {len(ecoregion_ids)} ecoregions for: {list(mosaics_to_process.keys())}")

        # Flattened S3 Pagination: Page bucket ONCE instead of inside loop
        search_prefix_base = f"low_res/{current_res}m/"
        if output_prefix and output_prefix.strip('/') != "low_res":
            search_prefix_base = f"{output_prefix}/{search_prefix_base}"
        
        all_s3_keys = []
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=search_prefix_base)
        print(f"[Mosaic] Paginating S3 to gather all keys upfront to minimize API calls...")
        
        for page_num, page in enumerate(pages, start=1):
            if 'Contents' in page:
                for obj in page['Contents']:
                    all_s3_keys.append(obj['Key'])
                    
            # Print S3 pagination progress
            if page_num % 10 == 0:
                mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
                print(f"      [S3 Paginator] Scanned {page_num} pages, found {len(all_s3_keys)} keys so far... (RAM: {mem_mb:.1f}MB)")

        print(f"[Mosaic] Finished scanning S3. Total keys retrieved: {len(all_s3_keys)}")
        all_s3_keys_set = set(all_s3_keys)

        # 2. Gather source file paths in a single efficient pass through S3 per ecoregion
        for ecoregion_id in ecoregion_ids:
            
            # Use dynamic folder checking to ensure robust mapping whether it's '1' or 'ER_1'
            match = re.search(r'\d+', str(ecoregion_id))
            clean_er_num = match.group(0) if match else str(ecoregion_id)
            valid_folders = [f"/{ecoregion_id}/", f"/ER_{clean_er_num}/", f"/ER{clean_er_num}/"]

            for key in all_s3_keys:
                if not any(f in f"/{key}" for f in valid_folders):
                    continue
                if f"/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo/" not in f"/{key}":
                    continue

                is_nearshore = bool(re.search(r'/BH[45]', key, re.IGNORECASE))
                for m_key, config in mosaics_to_process.items():
                    if config['regex'].search(key):
                        if config.get('exclude_nearshore_bands') and is_nearshore:
                            continue
                        config['paths'].append(f"/vsis3/{s3_bucket}/{key}")

        # 3. Locate the EcoRegions layer once for masking
        cutline_layer_name = None
        ds = ogr.Open(str(master_gpkg_path))
        if ds is not None:
            layer_names = [ds.GetLayerByIndex(i).GetName() for i in range(ds.GetLayerCount())]
            for name in layer_names:
                # Match case-insensitively, allowing for underscores or spaces
                if name.lower().replace("_", "").replace(" ", "") == "enhancedecoregions":
                    cutline_layer_name = name
                    break

            if not cutline_layer_name:
                print(f"[Mosaic] WARNING: Could not find an 'Enhanced_EcoRegions' layer in {master_gpkg_path}.")
                print(f"[Mosaic] Available layers: {layer_names}")
                print("[Mosaic] Proceeding without vector mask to prevent crash...")

            ds = None
        else:
            print(f"[Mosaic] WARNING: Could not open {master_gpkg_path} to check layers. Proceeding without mask...")

        temp_mask_path = str(low_res_dir / "full_mask.geojson")
        if cutline_layer_name:
            print(f"[Mosaic] Extracting full vector mask to memory-safe GeoJSON to prevent SQLite thread locking...")
            ds_gpkg = ogr.Open(str(master_gpkg_path))
            if ds_gpkg:
                layer = ds_gpkg.GetLayerByName(cutline_layer_name)
                
                if os.path.exists(temp_mask_path):
                    os.remove(temp_mask_path)
                    
                drv = ogr.GetDriverByName("GeoJSON")
                out_ds = drv.CreateDataSource(temp_mask_path)
                out_layer = out_ds.CreateLayer("mask", srs=layer.GetSpatialRef(), geom_type=layer.GetGeomType())
                
                for feat in layer:
                    out_layer.CreateFeature(feat)
                    
                out_ds = None
                ds_gpkg = None

        # Base warp arguments allowing GDAL to naturally unify projections perfectly
        # Set targetAlignedPixels to perfectly align the grids eliminating edge artifact grid lines
        warp_kwargs_base = {
            "format": "GTiff",
            "dstSRS": self.target_crs,
            "xRes": current_res,
            "yRes": current_res,
            "srcNodata": -9999,
            "dstNodata": -9999,
            "targetAlignedPixels": True,
            "multithread": False, # Stop complex Vector/Thread lockups here
            "warpOptions": ["NUM_THREADS=4"], # Safe chunk-based threading (prevents vector deadlock)
            "warpMemoryLimit": 512 * 1024 * 1024, # 512MB Limit to brutally stop OS RAM Thrashing
            "creationOptions": ["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
        }

        if cutline_layer_name:
            warp_kwargs_base["cutlineDSName"] = temp_mask_path
            warp_kwargs_base["cropToCutline"] = True
            # For the combined 100m all-tiles mosaic, we intentionally DO NOT pass a cutlineWhere clause.
            # This forces GDAL to utilize every single polygon inside the Enhanced_EcoRegions layer 
            # to mask the entire dataset simultaneously, avoiding SQL OR-clause limit truncations 
            # which cause some tiles to miss their mask.

        # 4. Generate & Upload each required mosaic dynamically
        # Process standard mosaics before offshore mosaics
        ordered_mosaics = sorted(mosaics_to_process.items(), key=lambda item: 1 if "Offshore" in item[0] else 0)
        for m_key, config in ordered_mosaics:
            if not config['paths']:
                print(f"[Mosaic] No base tiles found for {m_key}. Skipping.")
                continue

            local_mosaic_path = low_res_dir / config['filename']
            s3_key = config['s3_key']

            # Categorical components require nearest neighbor resampling
            m_key_lower = m_key.lower()
            
            # Check if mosaics already exist to skip heavy processing
            if m_key_lower == "bathy" or m_key_lower == "bathy_offshore":
                slope_s3_key = s3_key.replace("Bathy", "Slope")
                if s3_key in all_s3_keys_set and slope_s3_key in all_s3_keys_set:
                    print(f"[Mosaic] {m_key} and its Slope mosaic already exist in S3. Skipping.")
                    continue
            else:
                if s3_key in all_s3_keys_set:
                    print(f"[Mosaic] {m_key} mosaic already exists in S3. Skipping.")
                    continue

            if "bathy" in m_key_lower or "unc" in m_key_lower:
                resample_alg = gdal.GRA_Bilinear
            else:
                resample_alg = gdal.GRA_NearestNeighbour

            current_warp_kwargs = warp_kwargs_base.copy()
            current_warp_kwargs["resampleAlg"] = resample_alg

            print(f"[Mosaic] Warping and masking {m_key} mosaic directly from {len(config['paths'])} source files...")

            # Single pass directly to output
            try:
                self._build_local_mosaic(local_mosaic_path, config['paths'], current_warp_kwargs)
            except Exception as e:
                error_msg = f"[Mosaic] ERROR: Massive mosaic generation failed for {m_key}. Exception details: {e}"
                print(error_msg)
                continue # Safely skip to the next mosaic type instead of crashing the whole pipeline

            print(f"[Mosaic] Generating robust overviews for fast performance...")
            ds = gdal.Open(str(local_mosaic_path), gdal.GA_Update)
            if ds is not None:
                ds.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                ds = None

            print(f"[Mosaic] Uploading completely masked {m_key} mosaic to s3://{s3_bucket}/{s3_key}...")
            s3_client.upload_file(
                str(local_mosaic_path), 
                s3_bucket, 
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'resolution': str(current_res),
                        'crs': self.target_crs
                    }
                }
            )
            print(f"[Mosaic] {m_key} mosaic creation and upload perfectly complete.")

            # Generate Slope directly from Bathy to prevent tile-edge artifacts
            if m_key_lower == "bathy" or m_key_lower == "bathy_offshore":
                slope_key = m_key.replace("Bathy", "Slope")
                slope_filename = config['filename'].replace("Bathy", "Slope")
                slope_local_path = low_res_dir / slope_filename
                slope_s3_key = config['s3_key'].replace("Bathy", "Slope")
                
                print(f"[Mosaic] Generating {slope_key} mosaic directly from {m_key} mosaic to prevent edge artifacts...")
                slope_ds = gdal.DEMProcessing(
                    str(slope_local_path), 
                    str(local_mosaic_path), 
                    'slope', 
                    creationOptions=["COMPRESS=DEFLATE", "ZLEVEL=3", "TILED=YES", "BIGTIFF=YES", "NUM_THREADS=4"]
                )
                slope_ds = None # FIXED: Close dataset immediately
                
                print(f"[Mosaic] Generating robust overviews for {slope_key}...")
                ds_slope = gdal.Open(str(slope_local_path), gdal.GA_Update)
                if ds_slope is not None:
                    ds_slope.BuildOverviews("BILINEAR", [2, 4, 8, 16], callback=gdal.TermProgress_nocb)
                    ds_slope = None
                    
                print(f"[Mosaic] Uploading completely masked {slope_key} mosaic to s3://{s3_bucket}/{slope_s3_key}...")
                s3_client.upload_file(
                    str(slope_local_path), 
                    s3_bucket, 
                    slope_s3_key,
                    ExtraArgs={
                        'Metadata': {
                            'resolution': str(current_res),
                            'crs': self.target_crs
                        }
                    }
                )
                print(f"[Mosaic] {slope_key} mosaic creation and upload perfectly complete.")
                if slope_local_path.exists():
                    slope_local_path.unlink()

            if local_mosaic_path.exists():
                local_mosaic_path.unlink() # Save disk space
        
        # Cleanup full mask
        if os.path.exists(temp_mask_path):
            os.remove(temp_mask_path)
            
        gc.collect()