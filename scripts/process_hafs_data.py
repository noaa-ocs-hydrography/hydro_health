import boto3
import botocore
import xarray as xr
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.crs import CRS
from pathlib import Path
from datetime import datetime, timedelta

bucket = 'noaa-nws-hafs-pds'
prefix_template = 'hfsa/{date}/00/'
local_root = Path(r"C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\outputs\hafs_data\hafs_raw_data")
years = [2023, 2024, 2025]

anon_session = boto3.session.Session()
s3 = anon_session.client('s3', config=botocore.client.Config(signature_version=botocore.UNSIGNED))


def list_matching_keys(year):
    """Yield matching S3 keys for the specified year."""
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)

    paginator = s3.get_paginator("list_objects_v2")

    while start < end:
        prefix = prefix_template.format(date=start.strftime('%Y%m%d'))
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if (
                    "parent" in key and "atm" in key and "f000" in key and key.endswith(".grb2")
                ):
                    yield key
        start += timedelta(days=1)

def download_files(keys):
    """Download files from S3 if not already present."""
    downloaded_files = []

    for key in keys:
        local_path = local_root / key
        local_path.parent.mkdir(parents=True, exist_ok=True)

        if local_path.exists():
            print(f"[SKIP] Already downloaded: {key}")
        else:
            print(f"[DOWNLOAD] {key}")
            try:
                s3.download_file(bucket, key, str(local_path))
            except Exception as e:
                print(f"[ERROR] Failed to download {key}: {e}")
                continue

        downloaded_files.append(local_path)

    return downloaded_files


def process_files(files, year):
    """Process local GRIB2 files into cumulative gust raster."""
    cumulative = None
    meta = None

    for f in files:
        print(f" - Processing {f.name}")
        try:
            ds = xr.open_dataset(f, engine="cfgrib",
                                 filter_by_keys={'stepType': 'instant', 'typeOfLevel': 'surface'})
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')
            gust = ds['gust'].values
            gust = np.flipud(gust)

            if cumulative is None:
                cumulative = gust
                lat = ds.latitude.values
                lon = ds.longitude.values
                res_lat = abs(lat[1] - lat[0])
                res_lon = abs(lon[1] - lon[0])
                transform = from_origin(west=lon.min(), north=lat.max(), xsize=res_lon, ysize=res_lat)
                meta = {
                    'height': gust.shape[0],
                    'width': gust.shape[1],
                    'count': 1,
                    'dtype': gust.dtype,
                    'crs': CRS.from_epsg(4326),
                    'transform': transform
                }
            else:
                cumulative += gust

        except Exception as e:
            print(f"Error - Failed to process {f.name}: {e}")

    if cumulative is not None:
        output_file = local_root / f"gust_cumulative_{year}.tif"
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(cumulative, 1)
        print(f" - Saved cumulative raster for {year}: {output_file}")
    else:
        print(f" - skip - No data processed for {year}")


for year in years:
    print(f" - Processing {year} data...")
    keys = list(list_matching_keys(year))
    # files = download_files(keys)
    files = list(local_root.rglob("*.grb2"))
    process_files(files, year)
