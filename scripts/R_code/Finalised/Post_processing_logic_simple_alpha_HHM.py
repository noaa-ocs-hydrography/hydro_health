"""
process_bluetopo_tiles.py

Python translation of the R post-processing workflow.

Outputs per BlueTopo tile:
  1. ISS_alpha_<alpha>.tif
  2. PSS_alpha_<alpha>.tif
  3. DSS_UKC_slope.tif
  4. survey_age_ISS_PSS_alpha_<alpha>.png
  5. DSS_PSS_alpha_<alpha>.png

Requires:
  boto3
  rasterio
  numpy
  matplotlib
  dask
  distributed
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import reproject
from rasterio.shutil import copy as rio_copy
import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster, as_completed


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------

BUCKET = "ocs-dev-csdl-hydrohealth"

PARENT_PREFIX = (
    "low_res/ER_3/model_variables/Prediction/"
    "pre_processed/BlueTopo/"
)

ANALYSIS_YEAR = 2026
ALPHA = 0.022

SLOPE_THRESHOLD = 0.5

N_WORKERS = 4
THREADS_PER_WORKER = 1


# -----------------------------------------------------------------------------
# S3 helpers
# -----------------------------------------------------------------------------

def get_s3_client():
    return boto3.client("s3")


def list_tile_prefixes(bucket: str, parent_prefix: str) -> list[str]:
    """
    List tile folders below parent_prefix.

    Equivalent to the R list_tile_prefixes() function.
    """
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=parent_prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    tile_prefixes = set()

    for key in keys:
        rest = key.replace(parent_prefix, "", 1)
        parts = rest.split("/")
        if len(parts) >= 2 and parts[0]:
            tile_prefixes.add(f"{parent_prefix}{parts[0]}/")

    return sorted(tile_prefixes)


def list_keys(bucket: str, prefix: str) -> list[str]:
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])

    return keys


def find_object(bucket: str, tile_prefix: str, pattern: str) -> str:
    """
    Find exactly one object in a tile folder matching a regex pattern.
    """
    keys = list_keys(bucket, tile_prefix)
    regex = re.compile(pattern, flags=re.IGNORECASE)

    matches = [key for key in keys if regex.search(key)]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one match for pattern `{pattern}` in `{tile_prefix}`, "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def s3_uri(bucket: str, key: str) -> str:
    """
    Rasterio/GDAL S3 URI.
    """
    return f"s3://{bucket}/{key}"


def download_s3_file(bucket: str, key: str, local_file: Path) -> None:
    s3 = get_s3_client()
    local_file.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_file))


def upload_s3_file(bucket: str, key: str, local_file: Path) -> None:
    s3 = get_s3_client()
    s3.upload_file(str(local_file), bucket, key)


# -----------------------------------------------------------------------------
# Raster helpers
# -----------------------------------------------------------------------------

def read_raster(path: str) -> tuple[np.ndarray, dict]:
    """
    Read a single-band raster and return array + raster profile.
    """
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)

    return arr, profile


def clean_survey_year(year_arr: np.ndarray, analysis_year: int) -> np.ndarray:
    """
    Match the R clean_survey_year() logic.
    """
    x = np.round(year_arr).astype("float32")

    x = np.where(x >= 10000000, np.floor(x / 10000), x)
    x = np.where((x >= 10000) & (x < 10000000), np.floor(x / 100), x)

    x = np.where((x < 1900) | (x > analysis_year), np.nan, x)

    return x.astype("float32")


def profiles_match(profile_a: dict, profile_b: dict) -> bool:
    """
    Basic geometry comparison.
    """
    return (
        profile_a["crs"] == profile_b["crs"]
        and profile_a["transform"] == profile_b["transform"]
        and profile_a["width"] == profile_b["width"]
        and profile_a["height"] == profile_b["height"]
    )


def resample_to_match(
    source_arr: np.ndarray,
    source_profile: dict,
    target_profile: dict,
    method: Resampling = Resampling.nearest,
) -> np.ndarray:
    """
    Resample source array to match target profile.
    """
    destination = np.full(
        (target_profile["height"], target_profile["width"]),
        np.nan,
        dtype="float32",
    )

    reproject(
        source=source_arr,
        destination=destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        resampling=method,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return destination


def write_cog_to_s3(
    arr: np.ndarray,
    profile: dict,
    bucket: str,
    key: str,
    nodata: float = -999999.0,
) -> str:
    """
    Write array as a local GeoTIFF, convert to COG, upload to S3.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        raw_tif = tmp / f"raw_{Path(key).name}"
        cog_tif = tmp / Path(key).name

        out_profile = profile.copy()
        out_profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=nodata,
            compress="deflate",
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

        out_arr = np.where(np.isnan(arr), nodata, arr).astype("float32")

        with rasterio.open(raw_tif, "w", **out_profile) as dst:
            dst.write(out_arr, 1)

        rio_copy(
            raw_tif,
            cog_tif,
            driver="COG",
            compress="DEFLATE",
            overview_resampling="average",
        )

        upload_s3_file(bucket, key, cog_tif)

    return key


# -----------------------------------------------------------------------------
# Scoring logic
# -----------------------------------------------------------------------------

def make_pss(
    iss: np.ndarray,
    survey_end_year: np.ndarray,
    analysis_year: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create survey age and PSS.
    """
    survey_age = analysis_year - survey_end_year
    survey_age = np.clip(survey_age, 0, 200)

    pss = iss * np.exp(-alpha * survey_age)
    pss = np.clip(pss, 0, 100)

    pss = np.where(np.isnan(iss) | np.isnan(survey_age), np.nan, pss)

    return survey_age.astype("float32"), pss.astype("float32")


def make_dss_from_ukc_slope(
    ukc: np.ndarray,
    slope: np.ndarray,
    slope_threshold: float = 0.5,
) -> np.ndarray:
    """
    Desired Survey Score.

    SIMPLE seafloor, slope < 0.5:
      - UKC [-1, 1]       -> 100
      - UKC (1, 20]       -> 80
      - UKC (20, 50]      -> 30
      - UKC > 50          -> 10

    COMPLEX seafloor, slope >= 0.5:
      - UKC [-1, 1]       -> 100
      - UKC (1, 40]       -> 80
      - UKC (40, 100]     -> 30
      - UKC > 100         -> 10
    """
    dss = np.full(ukc.shape, np.nan, dtype="float32")

    valid = ~np.isnan(ukc) & ~np.isnan(slope)

    simple = valid & (slope < slope_threshold)
    complex_ = valid & (slope >= slope_threshold)

    # SIMPLE bins
    dss[simple & (ukc >= -1) & (ukc <= 1)] = 100
    dss[simple & (ukc > 1) & (ukc <= 20)] = 80
    dss[simple & (ukc > 20) & (ukc <= 50)] = 30
    dss[simple & (ukc > 50)] = 10

    # COMPLEX bins
    dss[complex_ & (ukc >= -1) & (ukc <= 1)] = 100
    dss[complex_ & (ukc > 1) & (ukc <= 40)] = 80
    dss[complex_ & (ukc > 40) & (ukc <= 100)] = 30
    dss[complex_ & (ukc > 100)] = 10

    return dss


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_raster(ax, arr: np.ndarray, title: str):
    masked = np.ma.masked_invalid(arr)
    image = ax.imshow(masked)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def make_three_panel_plot(
    survey_age: np.ndarray,
    iss: np.ndarray,
    pss: np.ndarray,
    output_file: Path,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    plot_raster(axes[0], survey_age, "Survey age")
    plot_raster(axes[1], iss, "Initial Survey Score (ISS)")
    plot_raster(axes[2], pss, "Present Survey Score (PSS)")

    plt.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return output_file


def make_dss_pss_plot(
    dss: np.ndarray,
    pss: np.ndarray,
    output_file: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    plot_raster(axes[0], dss, "Desired Survey Score (DSS)")
    plot_raster(axes[1], pss, "Present Survey Score (PSS)")

    plt.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

    return output_file


# -----------------------------------------------------------------------------
# Tile processing
# -----------------------------------------------------------------------------

def process_tile(
    bucket: str,
    tile_prefix: str,
    analysis_year: int = 2026,
    alpha: float = 0.022,
    slope_threshold: float = 0.5,
) -> Optional[dict]:
    tile_id = Path(tile_prefix.rstrip("/")).name
    print(f"Processing {tile_id}")

    survey_key = find_object(bucket, tile_prefix, r"survey_end_date\.tiff?$")

    # Raw initial score / supersession score / ISS.
    # This should NOT point to catzoc decay.
    iss_key = find_object(
        bucket,
        tile_prefix,
        r"supersession.*score.*all\.tiff?$|ISS.*\.tiff?$",
    )

    ukc_key = find_object(
        bucket,
        tile_prefix,
        r"UKC.*\.tiff?$|under.*keel.*clearance.*\.tiff?$",
    )

    slope_key = find_object(
        bucket,
        tile_prefix,
        r"slope.*\.tiff?$",
    )

    survey_end_year, survey_profile = read_raster(s3_uri(bucket, survey_key))
    iss, iss_profile = read_raster(s3_uri(bucket, iss_key))
    ukc, ukc_profile = read_raster(s3_uri(bucket, ukc_key))
    slope, slope_profile = read_raster(s3_uri(bucket, slope_key))

    survey_end_year = clean_survey_year(survey_end_year, analysis_year)

    # Match ISS to survey year grid.
    if not profiles_match(iss_profile, survey_profile):
        iss = resample_to_match(
            iss,
            iss_profile,
            survey_profile,
            method=Resampling.nearest,
        )
        iss_profile = survey_profile.copy()

    survey_age, pss = make_pss(
        iss=iss,
        survey_end_year=survey_end_year,
        analysis_year=analysis_year,
        alpha=alpha,
    )

    # Match UKC and slope to the PSS / survey grid.
    if not profiles_match(ukc_profile, survey_profile):
        ukc = resample_to_match(
            ukc,
            ukc_profile,
            survey_profile,
            method=Resampling.bilinear,
        )
        ukc_profile = survey_profile.copy()

    if not profiles_match(slope_profile, survey_profile):
        slope = resample_to_match(
            slope,
            slope_profile,
            survey_profile,
            method=Resampling.bilinear,
        )
        slope_profile = survey_profile.copy()

    dss = make_dss_from_ukc_slope(
        ukc=ukc,
        slope=slope,
        slope_threshold=slope_threshold,
    )

    alpha_label = str(alpha).replace(".", "_")

    iss_out_key = f"{tile_prefix}ISS_alpha_{alpha_label}.tif"
    pss_out_key = f"{tile_prefix}PSS_alpha_{alpha_label}.tif"
    dss_out_key = f"{tile_prefix}DSS_UKC_slope.tif"

    fig_out_key = f"{tile_prefix}survey_age_ISS_PSS_alpha_{alpha_label}.png"
    dss_pss_fig_out_key = f"{tile_prefix}DSS_PSS_alpha_{alpha_label}.png"

    write_cog_to_s3(iss, survey_profile, bucket, iss_out_key)
    write_cog_to_s3(pss, survey_profile, bucket, pss_out_key)
    write_cog_to_s3(dss, survey_profile, bucket, dss_out_key)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        fig_file = tmp / Path(fig_out_key).name
        make_three_panel_plot(survey_age, iss, pss, fig_file)
        upload_s3_file(bucket, fig_out_key, fig_file)

        dss_pss_fig_file = tmp / Path(dss_pss_fig_out_key).name
        make_dss_pss_plot(dss, pss, dss_pss_fig_file)
        upload_s3_file(bucket, dss_pss_fig_out_key, dss_pss_fig_file)

    return {
        "tile_id": tile_id,
        "iss": iss_out_key,
        "pss": pss_out_key,
        "dss": dss_out_key,
        "figure_age_iss_pss": fig_out_key,
        "figure_dss_pss": dss_pss_fig_out_key,
    }


def process_all_tiles(
    bucket: str,
    parent_prefix: str,
    analysis_year: int,
    alpha: float,
    slope_threshold: float = 0.5,
    n_workers: int = 4,
    threads_per_worker: int = 1,
) -> list[dict]:
    tile_prefixes = list_tile_prefixes(bucket, parent_prefix)

    print(f"Found {len(tile_prefixes)} tile folders.")

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=True,
    )

    client = Client(cluster)

    results = []

    try:
        futures = [
            client.submit(
                process_tile,
                bucket,
                tile_prefix,
                analysis_year,
                alpha,
                slope_threshold,
            )
            for tile_prefix in tile_prefixes
        ]

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(f"Finished {result['tile_id']}")
            except Exception as error:
                print(f"Failed tile: {error}")

    finally:
        client.close()
        cluster.close()

    return results


if __name__ == "__main__":
    results = process_all_tiles(
        bucket=BUCKET,
        parent_prefix=PARENT_PREFIX,
        analysis_year=ANALYSIS_YEAR,
        alpha=ALPHA,
        slope_threshold=SLOPE_THRESHOLD,
        n_workers=N_WORKERS,
        threads_per_worker=THREADS_PER_WORKER,
    )

    print("Processing complete.")
    print(f"Successful tiles: {len(results)}")