# Post-processing logic, using a fixed depreciation constant (alpha) value of 0.022 for the simplistic HHM approach 


from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from dask.distributed import Client, LocalCluster
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.io import MemoryFile


@dataclass
class Config:
    bucket: str = "ocs-dev-csdl-hydrohealth"
    parent_prefix: str = (
        "low_res/ER_3/model_variables/Prediction/"
        "pre_processed/BlueTopo/"
    )
    analysis_year: int = 2026
    alpha: float = 0.022


def list_tile_prefixes(config: Config) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    prefixes = set()

    for page in paginator.paginate(
        Bucket=config.bucket,
        Prefix=config.parent_prefix,
        Delimiter="/",
    ):
        for item in page.get("CommonPrefixes", []):
            prefixes.add(item["Prefix"])

    return sorted(prefixes)


def list_tile_objects(config: Config, tile_prefix: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    keys = []

    for page in paginator.paginate(Bucket=config.bucket, Prefix=tile_prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))

    return keys


def find_object(keys: list[str], pattern: str, tile_prefix: str) -> str:
    matches = [
        key for key in keys
        if re.search(pattern, key, flags=re.IGNORECASE)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one match for `{pattern}` in `{tile_prefix}`, "
            f"found {len(matches)}."
        )

    return matches[0]


def s3_uri(config: Config, key: str) -> str:
    return f"s3://{config.bucket}/{key}"


def clean_survey_year(values: np.ndarray, analysis_year: int) -> np.ndarray:
    years = np.rint(values).astype("float32")

    yyyymmdd = years >= 10_000_000
    yyyymm = (years >= 10_000) & (years < 10_000_000)

    years[yyyymmdd] = np.floor(years[yyyymmdd] / 10_000)
    years[yyyymm] = np.floor(years[yyyymm] / 100)

    invalid = (years < 1900) | (years > analysis_year)
    years[invalid] = np.nan

    return years


def read_raster(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        profile = src.profile.copy()

        if src.nodata is not None:
            data[data == src.nodata] = np.nan

    return data, profile


def resample_to_match(
    source_data: np.ndarray,
    source_profile: dict,
    target_profile: dict,
    resampling: Resampling = Resampling.nearest,
) -> np.ndarray:
    destination = np.empty(
        (target_profile["height"], target_profile["width"]),
        dtype="float32",
    )

    reproject(
        source=source_data,
        destination=destination,
        src_transform=source_profile["transform"],
        src_crs=source_profile["crs"],
        dst_transform=target_profile["transform"],
        dst_crs=target_profile["crs"],
        resampling=resampling,
    )

    return destination


def write_cog_to_s3(
    data: np.ndarray,
    profile: dict,
    config: Config,
    output_key: str,
) -> str:
    s3 = boto3.client("s3")

    output_profile = profile.copy()
    output_profile.update(
        driver="COG",
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="deflate",
        blocksize=512,
    )

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with rasterio.open(tmp_path, "w", **output_profile) as dst:
            dst.write(data.astype("float32"), 1)

        s3.upload_file(tmp_path, config.bucket, output_key)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return output_key


def write_three_panel_figure_to_s3(
    survey_age: np.ndarray,
    iss: np.ndarray,
    pss: np.ndarray,
    config: Config,
    output_key: str,
) -> str:
    s3 = boto3.client("s3")

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)

        panels = [
            (survey_age, "Survey age", "YlGnBu"),
            (iss, "Initial Survey Score (ISS)", "Spectral"),
            (pss, "Present Survey Score (PSS)", "Spectral"),
        ]

        for ax, (array, title, cmap) in zip(axes, panels):
            image = ax.imshow(array, cmap=cmap)
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        fig.savefig(tmp_path, dpi=150)
        plt.close(fig)

        s3.upload_file(tmp_path, config.bucket, output_key)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return output_key


def process_tile(tile_prefix: str, config: Config) -> dict:
    tile_id = Path(tile_prefix.rstrip("/")).name
    print(f"Processing {tile_id}")

    keys = list_tile_objects(config, tile_prefix)

    survey_key = find_object(
        keys,
        pattern=r"survey_end_date\.tiff?$",
        tile_prefix=tile_prefix,
    )

    iss_key = find_object(
        keys,
        pattern=r"catzoc.*decay.*all\.tiff?$",
        tile_prefix=tile_prefix,
    )

    survey_year_raw, survey_profile = read_raster(s3_uri(config, survey_key))
    iss, iss_profile = read_raster(s3_uri(config, iss_key))

    survey_year = clean_survey_year(
        survey_year_raw,
        analysis_year=config.analysis_year,
    )

    same_grid = (
        iss_profile["height"] == survey_profile["height"]
        and iss_profile["width"] == survey_profile["width"]
        and iss_profile["transform"] == survey_profile["transform"]
        and iss_profile["crs"] == survey_profile["crs"]
    )

    if not same_grid:
        iss = resample_to_match(
            iss,
            source_profile=iss_profile,
            target_profile=survey_profile,
            resampling=Resampling.nearest,
        )

    survey_age = config.analysis_year - survey_year
    survey_age = np.clip(survey_age, 0, 200)

    pss = iss * np.exp(-config.alpha * survey_age)  # we use the minus alpha value here so that its depreciated, ie. scores should decline with age
    pss = np.clip(pss, 0, 100)

    alpha_label = str(config.alpha).replace(".", "_")

    iss_out_key = f"{tile_prefix}ISS_alpha_{alpha_label}.tif"
    pss_out_key = f"{tile_prefix}PSS_alpha_{alpha_label}.tif"
    fig_out_key = f"{tile_prefix}survey_age_ISS_PSS_alpha_{alpha_label}.png"

    write_cog_to_s3(iss, survey_profile, config, iss_out_key)
    write_cog_to_s3(pss, survey_profile, config, pss_out_key)

    write_three_panel_figure_to_s3(
        survey_age,
        iss,
        pss,
        config,
        fig_out_key,
    )

    return {
        "tile_id": tile_id,
        "iss": iss_out_key,
        "pss": pss_out_key,
        "figure": fig_out_key,
    }


def process_all_tiles(config: Config, workers: int = 4) -> list[dict]:
    tile_prefixes = list_tile_prefixes(config)

    cluster = LocalCluster(n_workers=workers, threads_per_worker=1)
    client = Client(cluster)

    try:
        futures = client.map(
            lambda prefix: process_tile(prefix, config),
            tile_prefixes,
        )

        results = client.gather(futures)
    finally:
        client.close()
        cluster.close()

    return results


if __name__ == "__main__":
    config = Config()

    results = process_all_tiles(
        config,
        workers=8,
    )

    for result in results:
        print(result)
