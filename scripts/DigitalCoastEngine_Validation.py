
import boto3
import gc
import logging
import os
import pathlib
import requests
import shutil
import tempfile
import zipfile
import geopandas as gpd
import rasterio
import rasterio.shutil

from botocore import UNSIGNED
from botocore.client import Config
from rasterio.enums import Resampling
from typing import Any, Dict, List
from collections import defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item
from hydro_health.helpers.validation_reports import DigitalCoastValidationReport


# Environment & GDAL Flags
os.environ["SHAPE_RESTORE_SHX"] = "YES"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["LANG"] = "en_US.UTF-8"

logging.getLogger("pyproj").setLevel(logging.ERROR)


def _download_tile_index(param_inputs: list) -> None:
    """Parallel process and download tile index shapefiles."""

    download_link, provider_folder, param_lookup, outputs = param_inputs
    engine = DigitalCoastEngine(param_lookup)

    if get_config_item("DIGITALCOAST", "BUCKET") in download_link and ("dem" in download_link):
        _, data_file = download_link.replace("/index.html", "").split(".com")
        lidar_bucket = engine.get_bucket()
        for obj_summary in lidar_bucket.objects.filter(Prefix=f"{data_file[1:]}"):
            if "tileindex" in obj_summary.key and obj_summary.key.endswith(".zip"):
                output_zip_file = provider_folder / obj_summary.key
                file_parent_folder = output_zip_file.parents[0]
                shp_path = output_zip_file.parents[0] / pathlib.Path(
                    str(output_zip_file.stem) + ".shp"
                )
                provider_and_file = str(pathlib.Path(*shp_path.parts[-4:]))

                if os.path.exists(shp_path):
                    engine.write_message(f" - Skipping index: {provider_and_file}", outputs)
                    continue
                else:
                    engine.write_message(f" - Downloading index: {provider_and_file}", outputs)

                file_parent_folder.mkdir(parents=True, exist_ok=True)
                with open(output_zip_file, "wb") as tile_index:
                    lidar_bucket.download_fileobj(obj_summary.key, tile_index)


def _download_intersected_datasets_local(param_inputs: list) -> Dict[str, Any]:
    """Spatial filter and download datasets to local storage, tracking success/failure counts."""

    tile_gdf, shp_path, param_lookup, outputs = param_inputs
    shp_path_obj = pathlib.Path(shp_path)
    engine = DigitalCoastEngine(param_lookup)

    parts = shp_path_obj.parts
    provider = (
        parts[parts.index("DigitalCoast") + 1]
        if "DigitalCoast" in parts
        else shp_path_obj.parent.name
    )

    metrics = {
        "provider": provider,
        "shp": shp_path_obj.name,
        "expected": 0,
        "downloaded": 0,
        "failed": 0,
        "errors": [],
    }

    try:
        shp_df = gpd.read_file(shp_path).to_crs(4326)
        shp_df.columns = shp_df.columns.str.lower()
        if "tile" in shp_df.columns:
            shp_df.drop("tile", axis=1, inplace=True)

        df_joined = shp_df.sjoin(df=tile_gdf, how="left")
        df_joined.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

        if not df_joined["url"].any():
            return metrics

        df_joined = df_joined.loc[df_joined["tile"].notnull()]
        urls = [
            engine.cleansed_url(u)
            for u in df_joined["url"].unique()
            if engine.cleansed_url(u).endswith(".tif")
        ]
        metrics["expected"] = len(urls)

        shp_folder = shp_path_obj.parents[0]
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[404, 429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session = requests.Session()
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        for i, cleansed_url in enumerate(urls):
            dataset_name = cleansed_url.split("/")[-1]
            output_file = shp_folder / dataset_name

            if os.name == "nt":
                abs_str = str(output_file.resolve())
                file_target = abs_str if abs_str.startswith("\\\\?\\") else "\\\\?\\" + abs_str
            else:
                file_target = output_file

            if os.path.exists(file_target):
                metrics["downloaded"] += 1
                continue

            try:
                response = session.get(cleansed_url, timeout=15)
                if response.status_code == 200:
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(file_target, "wb") as f:
                        f.write(response.content)
                    metrics["downloaded"] += 1
                else:
                    metrics["failed"] += 1
                    metrics["errors"].append(f"{dataset_name} HTTP {response.status_code}")
            except Exception as e:
                metrics["failed"] += 1
                metrics["errors"].append(f"{dataset_name}: {type(e).__name__}")

        session.close()

    except Exception as e:
        metrics["errors"].append(f"Fatal Shapefile Error: {str(e)}")

    return metrics


def _download_intersected_datasets_s3(param_inputs: list) -> Dict[str, Any]:
    """Spatial filter, convert GeoTIFF to COG locally, upload to S3, and track counts."""

    ecoregion, tile_gdf, shp_path, outputs, param_lookup, temp_root = param_inputs
    shp_path_obj = pathlib.Path(shp_path)
    engine = DigitalCoastEngine(param_lookup)

    parts = shp_path_obj.parts
    provider = (
        parts[parts.index("DigitalCoast") + 1]
        if "DigitalCoast" in parts
        else shp_path_obj.parent.name
    )

    metrics = {
        "provider": provider,
        "shp": shp_path_obj.name,
        "expected": 0,
        "downloaded": 0,
        "failed": 0,
        "errors": [],
    }

    scratch_dir = pathlib.Path.home() / "temp_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    gdal_config = {"GDAL_TMPDIR": str(scratch_dir), "GDAL_CACHEMAX": 512}

    try:
        shp_df = gpd.read_file(shp_path, engine="pyogrio")
        shp_df["geometry"] = shp_df.geometry.make_valid()
        shp_df = shp_df.to_crs(4326)
        shp_df.columns = shp_df.columns.str.lower()

        cols_to_clear = ["tile", "idx_right", "index_right", "index_left"]
        laundered_cols = [c for c in shp_df.columns if c.startswith("ecoregio_")]
        shp_df = shp_df.drop(columns=cols_to_clear + laundered_cols, errors="ignore")

        df_joined = shp_df.sjoin(df=tile_gdf, how="left")
        if "index_right" in df_joined.columns:
            df_joined = df_joined.rename(columns={"index_right": "idx_right"})

        df_joined = df_joined.loc[:, ~df_joined.columns.duplicated()].copy()
        df_joined.to_file(shp_path, driver="ESRI Shapefile")

        urls = []
        if not df_joined.empty and "url" in df_joined.columns:
            urls = [
                engine.cleansed_url(u)
                for u in df_joined.loc[df_joined["tile"].notnull(), "url"].unique().tolist()
            ]
            urls = [u for u in urls if u.endswith(".tif")]

        metrics["expected"] = len(urls)

        del shp_df, df_joined
        gc.collect()

        if urls:
            retry_strategy = Retry(
                total=3,
                backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("https://", adapter)

            with rasterio.Env(**gdal_config):
                for url in urls:
                    dataset_name = url.split("/")[-1]
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".tif", dir=scratch_dir, delete=True
                        ) as tmp_raw:
                            with session.get(url, timeout=20, stream=True) as r:
                                r.raise_for_status()
                                for chunk in r.iter_content(chunk_size=1024 * 1024):
                                    tmp_raw.write(chunk)
                                tmp_raw.flush()

                            with rasterio.open(tmp_raw.name) as src:
                                dst_profile = src.profile.copy()
                                dst_profile.update(
                                    {
                                        "driver": "GTiff",
                                        "tiled": True,
                                        "blockxsize": 512,
                                        "blockysize": 512,
                                        "compress": "deflate",
                                        "predictor": 3,
                                        "interleave": "pixel",
                                    }
                                )

                                with tempfile.NamedTemporaryFile(
                                    suffix=".tif", dir=scratch_dir, delete=True
                                ) as tmp_cog:
                                    rasterio.shutil.copy(src, tmp_cog.name, **dst_profile)

                                    with rasterio.open(tmp_cog.name, "r+") as dst:
                                        min_dim = min(dst.width, dst.height)
                                        valid_ov = [
                                            o for o in [2, 4, 8, 16] if (min_dim // o) >= 4
                                        ]
                                        if valid_ov:
                                            dst.build_overviews(valid_ov, Resampling.average)

                                    local_parent_dir = shp_path_obj.parent
                                    relative_s3_path = local_parent_dir.relative_to(temp_root)
                                    s3_prefix = f"{relative_s3_path.as_posix()}/{dataset_name}"
                                    engine.upload_file_to_s3(tmp_cog.name, s3_prefix)

                                    metrics["downloaded"] += 1
                        gc.collect()

                    except Exception as e:
                        metrics["failed"] += 1
                        metrics["errors"].append(f"{dataset_name}: {str(e)}")

            session.close()

    except Exception as e:
        metrics["errors"].append(f"Fatal index processing error: {str(e)}")

    return metrics


def _download_intersected_datasets(param_inputs: list) -> Dict[str, Any]:
    """Parallel process dispatcher returning operational metrics."""

    if len(param_inputs) == 6:
        return _download_intersected_datasets_s3(param_inputs)
    return _download_intersected_datasets_local(param_inputs)


class DigitalCoastEngine(Engine):
    """Parallel engine for downloading and validating Digital Coast datasets."""

    def __init__(self, param_lookup: dict):
        """Initialize the engine with configuration parameters and metrics container."""

        super().__init__()
        self.param_lookup = param_lookup
        self.approved_size = 0
        self.run_metrics: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"expected": 0, "downloaded": 0, "failed": 0, "errors": []}
        )

    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> None:
        """Merge worker output metrics into engine tracking dictionary."""

        for res in results:
            if not isinstance(res, dict):
                continue
            prov = res.get("provider", "unknown")
            self.run_metrics[prov]["expected"] += res.get("expected", 0)
            self.run_metrics[prov]["downloaded"] += res.get("downloaded", 0)
            self.run_metrics[prov]["failed"] += res.get("failed", 0)
            self.run_metrics[prov]["errors"].extend(res.get("errors", []))

    def breakup_cudem(self, digital_coast_folder: pathlib.Path) -> None:
        """Organize CUDEM tiles into dedicated year folders."""

        for folder in digital_coast_folder.glob("NOAA_NCEI_0*"):
            year_tifs = defaultdict(list)
            for tif_file in folder.rglob("*.tif"):
                year = str(tif_file.stem)[-6:-2]
                year_tifs[year].append(tif_file)

            for year in year_tifs:
                project_folder = year_tifs[year][0].parents[0]
                year_folder = (
                    digital_coast_folder
                    / f"NOAA_NCEI_{year}_{str(folder.stem)[-4:]}"
                    / "dem"
                    / project_folder.stem
                )
                year_folder.mkdir(parents=True, exist_ok=True)
                for tif_file in year_tifs[year]:
                    out_tif = year_folder / tif_file.name
                    if out_tif.exists():
                        out_tif.unlink()
                    shutil.move(tif_file, year_folder)

                for tile_index in project_folder.glob("*index*.shp"):
                    tile_index_df = gpd.read_file(tile_index)
                    tile_index_df.to_file(year_folder / tile_index.name)

                feature_json = project_folder.parents[1] / "feature.json"
                if feature_json.exists():
                    shutil.copy(feature_json, year_folder.parents[1])

            shutil.rmtree(folder)

    def check_tile_index_areas(
        self,
        digital_coast_folder: pathlib.Path | list[pathlib.Path],
        outputs: str,
    ) -> list[pathlib.Path] | None:
        """Filter out providers whose total footprint does not satisfy threshold."""

        self.write_message("Checking area size of tileindex files", outputs)
        if isinstance(digital_coast_folder, (list, tuple)):
            tile_index_shapefiles = digital_coast_folder
            is_list_mode = True
        else:
            tile_index_shapefiles = [
                f for f in digital_coast_folder.rglob("*index*.shp")
                if "unused_providers" not in str(f)
            ]
            is_list_mode = False

        providers = []
        for shp_path in tile_index_shapefiles:
            shp_df = gpd.read_file(shp_path).to_crs(9822)
            shp_df["area"] = shp_df["geometry"].area
            total_area = shp_df["area"].sum()
            if total_area < self.approved_size:
                self.write_message(f" - provider too small: {total_area}\n{shp_path}", outputs)
                if not is_list_mode:
                    shutil.rmtree(shp_path.parents[2])
            else:
                providers.append(shp_path)

        return providers if is_list_mode else None

    def delete_unused_folder(self, digital_coast_folder: pathlib.Path, outputs: str) -> None:
        """Prune empty provider directories."""

        if digital_coast_folder.exists():
            self.write_message("Deleting empty provider folders", outputs)
            for provider in [f for f in digital_coast_folder.glob("*") if f.is_dir()]:
                if provider.stem != "unused_providers":
                    data_types = os.listdir(provider)
                    if not provider.suffix and "dem" not in data_types and "laz" not in data_types:
                        self.write_message(f" - removing empty provider: {provider}", outputs)
                        shutil.rmtree(provider)

    def download_support_files(
        self,
        digital_coast_folder: pathlib.Path | gpd.GeoDataFrame,
        tile_gdf: gpd.GeoDataFrame | str,
        ecoregion: str | pathlib.Path,
        outputs: pathlib.Path | str,
    ) -> None:
        """Download provider tile index shapefiles."""

        if isinstance(digital_coast_folder, gpd.GeoDataFrame):
            tile_gdf, ecoregion, digital_coast_folder = digital_coast_folder, tile_gdf, ecoregion

        outputs_str = str(outputs)
        self.write_message("Download Support Files", outputs_str)
        ecoregion_geom_strings = self.get_ecoregion_geometry_strings(tile_gdf, ecoregion)
        for geometry_coords in ecoregion_geom_strings:
            tile_index_links = self.get_available_datasets(geometry_coords, digital_coast_folder)
            bulk_download_params = [
                [link_dict["link"], link_dict["provider_path"], self.param_lookup, outputs_str]
                for link_dict in tile_index_links
                if link_dict["label"] == "Bulk Download"
            ]
            future_tiles = self.client.map(_download_tile_index, bulk_download_params)
            _ = self.client.gather(future_tiles)
            self.unzip_all_files(digital_coast_folder)

    def get_bucket(self) -> boto3.resource:
        """Establish anonymous connection to NOAA S3 bucket."""

        creds = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "config": Config(signature_version=UNSIGNED),
        }
        s3 = boto3.resource("s3", **creds)
        return s3.Bucket(get_config_item("DIGITALCOAST", "BUCKET"))

    def unzip_all_files(self, digital_coast_folder: pathlib.Path | str) -> None:
        """Uncompress all archive downloads."""

        for zipped_file in pathlib.Path(digital_coast_folder).rglob("*.zip"):
            with zipfile.ZipFile(zipped_file, "r") as z:
                z.extractall(str(zipped_file.parents[0]))
            zipped_file.unlink()

    def upload_file_to_s3(self, temp_file_path: str, s3_prefix: str) -> None:
        """Push local file directly to target S3 prefix."""

        s3_client = boto3.client("s3", region_name="us-east-2")
        bucket_name = get_config_item("SHARED", "OUTPUT_BUCKET")
        s3_client.upload_file(str(temp_file_path), bucket_name, s3_prefix.lstrip("/"))

    def upload_files_to_s3(
        self, provider_folder: pathlib.Path, temp_root: pathlib.Path
    ) -> None:
        """Recursively upload folder tree to S3."""

        s3_client = boto3.client("s3", region_name="us-east-2")
        bucket_name = get_config_item("SHARED", "OUTPUT_BUCKET")
        for found_file in provider_folder.rglob("*"):
            if found_file.is_file():
                relative_path = found_file.relative_to(temp_root)
                s3_client.upload_file(str(found_file), bucket_name, relative_path.as_posix())

    def run(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str | bool = False) -> None:
        """Primary execution harness."""

        print("Starting DigitalCoastEngine")
        if self.param_lookup["env"] == "aws":
            self._run_s3(tile_gdf, str(output_prefix) if output_prefix else "")
        else:
            self._run_local(tile_gdf, output_prefix)

    def _run_local(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str | bool) -> None:
        """Local pipeline sequence with metric collection."""

        outputs = self.param_lookup["output_directory"].valueAsText
        self.setup_dask(self.param_lookup["env"])
        ecoregions = list(tile_gdf["EcoRegion"].unique())

        for ecoregion in ecoregions:
            if isinstance(ecoregion, str):
                digital_coast_subfolder = (
                    pathlib.Path(ecoregion)
                    / get_config_item("DIGITALCOAST", "SUBFOLDER")
                    / "DigitalCoast"
                )
                if output_prefix:
                    digital_coast_subfolder = pathlib.Path(output_prefix) / digital_coast_subfolder
                digital_coast_folder = pathlib.Path(outputs) / digital_coast_subfolder
                ecoregion_tile_gdf = tile_gdf.loc[tile_gdf["EcoRegion"] == ecoregion]

                self.download_support_files(
                    digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs
                )
                self.check_tile_index_areas(digital_coast_folder, outputs)

                tile_index_shapefiles = [
                    f for f in digital_coast_folder.rglob("*index*.shp")
                    if "unused_providers" not in str(f)
                ]
                param_inputs = [
                    [ecoregion_tile_gdf, shp, self.param_lookup, outputs]
                    for shp in tile_index_shapefiles
                ]
                future_tiles = self.client.map(_download_intersected_datasets, param_inputs)
                results = self.client.gather(future_tiles)
                self._aggregate_metrics(results)

                if digital_coast_folder.exists():
                    self.delete_unused_folder(digital_coast_folder, outputs)

        self.close_dask()

    def _run_s3(self, tile_gdf: gpd.GeoDataFrame, output_prefix: str) -> None:
        """Cloud S3 ingestion harness with batching, deduplication, and automated PDF audit generation."""

        outputs = self.param_lookup["output_directory"].valueAsText
        provider_log = pathlib.Path(outputs) / "processed_providers.log"
        provider_log.touch(exist_ok=True)
        self.setup_dask(self.param_lookup["env"])
        ecoregions = list(tile_gdf["EcoRegion"].unique())

        for ecoregion in ecoregions:
            if not isinstance(ecoregion, str):
                continue

            ecoregion_tile_gdf = tile_gdf.loc[tile_gdf["EcoRegion"] == ecoregion]

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_digital_coast_outputs = pathlib.Path(temp_dir)
                digital_coast_subfolder = (
                    pathlib.Path(ecoregion)
                    / get_config_item("DIGITALCOAST", "SUBFOLDER")
                    / "DigitalCoast"
                )
                if output_prefix:
                    digital_coast_subfolder = pathlib.Path(output_prefix) / digital_coast_subfolder

                digital_coast_folder = temp_digital_coast_outputs / digital_coast_subfolder
                self.download_support_files(
                    digital_coast_folder, ecoregion_tile_gdf, ecoregion, outputs
                )
                tile_index_shapefiles = [
                    shp for shp in digital_coast_folder.rglob("*index*.shp")
                    if "unused_providers" not in str(shp)
                ]
                approved_shapefiles = self.check_tile_index_areas(tile_index_shapefiles, outputs)

                chunk_size = 5
                for i in range(0, len(approved_shapefiles), chunk_size):
                    current_files = approved_shapefiles[i:i + chunk_size]
                    completed_providers = set(provider_log.read_text().splitlines())

                    pending_files = []
                    for shp in current_files:
                        prov_name = shp.parents[0].name
                        if prov_name in completed_providers:
                            print(f"Skipping already processed provider: {prov_name}")
                        else:
                            pending_files.append(shp)

                    if not pending_files:
                        continue

                    param_inputs = [
                        [
                            ecoregion,
                            ecoregion_tile_gdf,
                            shp,
                            outputs,
                            self.param_lookup,
                            temp_digital_coast_outputs,
                        ]
                        for shp in pending_files
                    ]

                    future_tasks = self.client.map(_download_intersected_datasets, param_inputs)
                    results = self.client.gather(future_tasks)
                    self._aggregate_metrics(results)

                    with open(provider_log, "a") as f:
                        for shp in pending_files:
                            prov = shp.parents[0].name
                            f.write(f"{prov}\n")

                    for provider_shp in pending_files:
                        provider_parts = provider_shp.parts
                        dc_idx = provider_parts.index("DigitalCoast")
                        provider_folder = pathlib.Path(*provider_parts[:dc_idx + 2])
                        self.upload_files_to_s3(provider_folder, temp_digital_coast_outputs)

        self.close_dask()

        self.run_validation_report(output_prefix)

    def run_validation_report(self, output_prefix: str) -> None:
        """Run post-download validation audit, generate summary PDF, and store report in S3."""

        print("Generating DigitalCoast validation and auditing report...")
        bucket_name = get_config_item("SHARED", "OUTPUT_BUCKET")
        s3_audit_prefix = f"s3://{bucket_name}/{output_prefix}".rstrip("/") + "/"

        validator = DigitalCoastValidationReport(
            s3_prefix=s3_audit_prefix,
            execution_metrics=dict(self.run_metrics),
        )

        try:
            pdf_name = "DigitalCoast_Validation_Report.pdf"
            local_pdf = validator.generate_pdf(output_pdf_filename=pdf_name)
            s3_url = validator.upload_to_s3(local_pdf)
            print(f"Audit report uploaded successfully to: {s3_url}")
        except Exception as e:
            print(f"Validation report generation encountered an error: {e}")
        finally:
            validator.cleanup()
