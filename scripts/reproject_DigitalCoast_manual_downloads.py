#!/usr/bin/env python3
"""
Process manual DigitalCoast downloads on EC2 using s3fs:
1. Discovers GeoTIFFs under the specified manual downloads prefix.
2. Copies raw GeoTIFFs to an 'original/' sub-prefix in S3.
3. Reprojects to EPSG:32617 locally with native GDAL C-streaming.
4. Overwrites the root provider key in S3 with the reprojected GeoTIFF.
"""

import ctypes
import gc
import os
import pathlib
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import s3fs
from osgeo import gdal

# --- Configuration Constants ---
S3_MANUAL_DOWNLOADS_PREFIX = (
    "s3://ocs-dev-csdl-hydrohealth/ER_3/model_variables/Prediction/raw/Digital_Coast_Manual_Downloads"
)
TARGET_CRS = "EPSG:32617"
MAX_WORKERS = 4  # Adjust according to available vCPUs / RAM
OVERWRITE = False  # Set to True to reprocess files even if 'original/' exists

# RedHat / Linux Environment configuration
os.environ["SHAPE_RESTORE_SHX"] = "YES"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"


def _trim_memory() -> None:
    """Forces Linux glibc to release unmanaged C/GDAL memory arenas back to the OS."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _init_worker(scratch_dir_str: str) -> None:
    """Initialize GDAL cache and temp directories per worker process."""
    gdal.SetCacheMax(256 * 1024 * 1024)
    gdal.SetConfigOption("GDAL_TMPDIR", scratch_dir_str)
    gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "TRUE")


def _reproject_raster_gdal(
    src_path: str | pathlib.Path,
    dst_path: str | pathlib.Path,
    target_crs: str = "EPSG:32617",
) -> None:
    """
    Native C++ streaming reprojection and COG conversion using gdal.Warp.
    Streams block-by-block with a strict 128MB warp cache limit.
    """
    src_path_str = str(src_path)
    dst_path_str = str(dst_path)

    src_ds = gdal.Open(src_path_str, gdal.GA_ReadOnly)
    if not src_ds:
        raise RuntimeError(f"Could not open source raster: {src_path_str}")

    band = src_ds.GetRasterBand(1)
    band_type = gdal.GetDataTypeName(band.DataType)
    predictor = "3" if "Float" in band_type else "2"
    src_ds = None

    warp_options = gdal.WarpOptions(
        dstSRS=target_crs,
        resampleAlg="bilinear",
        format="GTiff",
        warpMemoryLimit=128 * 1024 * 1024,  # 128 MB
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "COMPRESS=DEFLATE",
            f"PREDICTOR={predictor}",
            "BIGTIFF=IF_SAFER",
        ],
    )

    dst_ds = gdal.Warp(dst_path_str, src_path_str, options=warp_options)
    if dst_ds is None:
        raise RuntimeError(f"gdal.Warp failed for {src_path_str}")

    min_dim = min(dst_ds.RasterXSize, dst_ds.RasterYSize)
    potential_ov = [2, 4, 8, 16]
    valid_ov = [o for o in potential_ov if (min_dim // o) >= 4]
    if valid_ov:
        dst_ds.BuildOverviews("AVERAGE", valid_ov)

    dst_ds = None


def process_geotiff(
    raw_s3_path: str,
    target_crs: str,
    scratch_dir_str: str,
    overwrite: bool = False,
) -> str:
    """
    Processes a single S3 GeoTIFF:
    1. Server-side copies raw file to <provider_dir>/original/<file>.tif.
    2. Downloads raw file locally.
    3. Streams GDAL warp to target_crs with internal overviews.
    4. Uploads reprojected raster back to raw_s3_path.
    """
    fs = s3fs.S3FileSystem()
    scratch_dir = pathlib.Path(scratch_dir_str)

    parent_dir, filename = raw_s3_path.rsplit("/", 1)
    original_s3_path = f"{parent_dir}/original/{filename}"

    # Check if this raster was already archived/processed
    if fs.exists(original_s3_path) and not overwrite:
        return f"SKIPPED (already processed): {raw_s3_path}"

    unique_id = uuid.uuid4().hex[:8]
    tmp_raw_path = scratch_dir / f"raw_{unique_id}_{filename}"
    tmp_rep_path = scratch_dir / f"rep_{unique_id}_{filename}"

    try:
        # Step 1: Server-side S3 copy to preserve the unprojected raw file
        if not fs.exists(original_s3_path) or overwrite:
            fs.copy(raw_s3_path, original_s3_path)

        # Step 2: Download raw file from the original archive to scratch
        fs.get(original_s3_path, str(tmp_raw_path))

        # Step 3: Stream reprojection and build pyramids
        _reproject_raster_gdal(tmp_raw_path, tmp_rep_path, target_crs=target_crs)

        # Step 4: Overwrite the provider root path with the reprojected raster
        fs.put(str(tmp_rep_path), raw_s3_path)

        return f"SUCCESS: {raw_s3_path} (archived to {original_s3_path})"

    except Exception as exc:
        raise exc
    finally:
        if tmp_raw_path.exists():
            tmp_raw_path.unlink()
        if tmp_rep_path.exists():
            tmp_rep_path.unlink()
        _trim_memory()


def main() -> None:
    clean_prefix = S3_MANUAL_DOWNLOADS_PREFIX.replace("s3://", "").rstrip("/")
    scratch_dir = pathlib.Path.home() / "temp_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    fs = s3fs.S3FileSystem()

    # List all files recursively
    all_s3_paths = fs.find(clean_prefix)

    # Filter for GeoTIFFs while ignoring any already located inside 'original/'
    candidate_paths = [
        path
        for path in all_s3_paths
        if path.lower().endswith((".tif", ".tiff")) and "/original/" not in path
    ]

    if not candidate_paths:
        return


    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(str(scratch_dir),),
    ) as executor:
        futures = {
            executor.submit(
                process_geotiff,
                raw_s3_path=path,
                target_crs=TARGET_CRS,
                scratch_dir_str=str(scratch_dir),
                overwrite=OVERWRITE,
            ): path
            for path in candidate_paths
        }

        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print('Failure:', e)

if __name__ == "__main__":
    main()