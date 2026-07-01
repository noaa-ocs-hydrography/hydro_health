import pathlib
import boto3
import re
import numpy as np
from osgeo import gdal, ogr, osr
# Make sure to import or define get_config_item in your new engine
# from hydro_health.helpers.tools import get_config_item

class IsobathEngine:
    def __init__(self):
        self.target_crs = "EPSG:6350"

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