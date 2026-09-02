"""Class engine for isolating rasterio and GDAL logic, converting tables back to Cloud Optimized GeoTIFFs."""

import os
import gc
import math
import shutil
import pathlib
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Tuple, Optional

import s3fs
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.features import geometry_mask
from shapely.geometry import mapping
from upath import UPath

try:
    from osgeo import gdal  # type: ignore
except Exception:  
    gdal = None

from hydro_health.helpers.tools import get_config_item
from hydro_health.engines.Engine import Engine

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'

class LocalizedFile:
    """Helper to download an S3 template raster to local disk for rasterio to access quickly."""
    def __init__(self, uri: str, is_aws: bool):
        self.uri = uri
        self.is_aws = is_aws
        self.tempdir = None
        self.path = None

    def __enter__(self) -> str:
        if not self.is_aws or not self.uri.startswith("s3://"):
            self.path = self.uri
            return self.uri
        self.tempdir = tempfile.TemporaryDirectory(prefix="seabed_template_")
        self.path = str(Path(self.tempdir.name) / "template.tif")
        s3fs.S3FileSystem().get(self.uri, self.path)
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.tempdir:
            self.tempdir.cleanup()

def _dataframe_to_grid(
    df: pd.DataFrame, value_col: str, template_uri: str, is_aws: bool,
    tile_geom_wkb: bytes, crs: str, nodata: float = -9999.0
) -> Tuple[np.ndarray, dict]:
    """Rasterize point predictions onto a seam-safe, template-aligned bounding box."""
    
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    xs = pd.to_numeric(df['X'], errors="coerce").to_numpy(dtype=float)
    ys = pd.to_numeric(df['Y'], errors="coerce").to_numpy(dtype=float)
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    
    if not keep.any():
        raise ValueError(f"No finite X/Y/{value_col} rows available for rasterization")

    tile_geom = gpd.GeoSeries.from_wkb([tile_geom_wkb], crs=crs).iloc[0]

    with LocalizedFile(template_uri, is_aws) as local_template:
        with rasterio.open(local_template) as template:
            # Snap out bounding box to template transform
            raw_window = from_bounds(*tile_geom.bounds, transform=template.transform)
            window = raw_window.round_offsets(op="floor").round_lengths(op="ceil")
            window = window.intersection(Window(0, 0, template.width, template.height))
            
            height, width = int(window.height), int(window.width)
            if height <= 0 or width <= 0:
                raise ValueError("Tile polygon does not intersect the template raster")

            local_transform = template.window_transform(window)
            rows, cols = rasterio.transform.rowcol(local_transform, xs[keep], ys[keep])
            rows, cols = np.asarray(rows), np.asarray(cols)
            
            inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            vals = values[keep][inside]
            rows, cols = rows[inside], cols[inside]

            accum = np.zeros((height, width), dtype=np.float64)
            counts = np.zeros((height, width), dtype=np.uint32)
            if len(vals):
                np.add.at(accum, (rows, cols), vals)
                np.add.at(counts, (rows, cols), 1)
                
            grid = np.full((height, width), nodata, dtype=np.float32)
            populated = counts > 0
            grid[populated] = (accum[populated] / counts[populated]).astype(np.float32)

            # Mask out cells that fall completely outside the official polygon geometry
            valid_polygon = geometry_mask(
                [mapping(tile_geom)], out_shape=(height, width), transform=local_transform,
                invert=True, all_touched=False,
            )
            grid[~valid_polygon] = nodata

            profile = template.profile.copy()
            profile.update(
                driver="GTiff", count=1, width=width, height=height,
                transform=local_transform, dtype="float32", nodata=nodata,
            )
            return grid, profile

def _build_overviews_and_cog(raw_tif: str, cog_tif: str, cfg: dict) -> None:
    """Builds overviews and translates to COG format via GDAL."""
    levels = cfg.get("overview_levels", [2, 4, 8, 16, 32])
    resampling = cfg.get("overview_resampling", "average")
    compress = cfg.get("compress", "DEFLATE")
    blocksize = cfg.get("blocksize", 512)
    
    # Internal overviews for raw
    with rasterio.open(raw_tif, "r+") as dst:
        valid_levels = [l for l in levels if min(dst.width, dst.height) // l >= 1]
        if valid_levels:
            dst.build_overviews(valid_levels, rasterio.enums.Resampling[resampling])
            dst.update_tags(ns="rio_overview", resampling=resampling)

    # Translate to COG
    if gdal is not None:
        options = gdal.TranslateOptions(
            format="COG",
            creationOptions=[f"COMPRESS={compress}", f"BLOCKSIZE={blocksize}", "BIGTIFF=IF_SAFER", "OVERVIEWS=AUTO"],
        )
        gdal.Translate(cog_tif, raw_tif, options=options)
    elif shutil.which("gdal_translate"):
        subprocess.run(["gdal_translate", "-of", "COG", "-co", f"COMPRESS={compress}", 
                        "-co", f"BLOCKSIZE={blocksize}", "-co", "BIGTIFF=IF_SAFER", raw_tif, cog_tif], check=True)
    else:
        shutil.copy2(raw_tif, cog_tif)

def _export_cog_task(params: list) -> dict:
    """Worker task reading a summary table and generating multiple COG rasters."""
    tile_id, pair, table_uri, template_uri, out_dir, tile_geom_wkb, crs, raster_cfg, is_aws, local_tmp_dir, current, total, verbose = params

    if verbose:
        Engine.write_message_dask(f"Rendering COGs for tile {tile_id} - Pair {pair} ({current}/{total})...", OUTPUTS)

    try:
        df = pd.read_parquet(table_uri, storage_options={"anon": False} if is_aws else None)
        
        # Determine the target columns and map them to their output prefix
        products = {
            "bathy_t": "LOCAL_Pred_Start_Bathy_t",
            "mean_predicted_bathy_t1": "LOCAL_Pred_Mean_Bathy_t1",
            "mean_predicted_change": "LOCAL_Pred_Mean_Change",
            "uncertainty_sd_bathy_t1": "LOCAL_Pred_SD_Bathy_t1"
        }

        with tempfile.TemporaryDirectory(dir=local_tmp_dir, prefix=f"raster_{tile_id}_") as tmp:
            
            for value_col, prefix in products.items():
                if value_col not in df:
                    continue
                    
                # Build Numpy Grid
                grid, profile = _dataframe_to_grid(df, value_col, template_uri, is_aws, tile_geom_wkb, crs)
                
                # Format blocksize
                profile.update(tiled=True, BIGTIFF="IF_SAFER", compress=raster_cfg.get("compress", "DEFLATE"))
                profile["blockxsize"] = min(profile["width"], max(16, int(math.ceil(raster_cfg.get("blocksize", 512)/16)*16)))
                profile["blockysize"] = min(profile["height"], max(16, int(math.ceil(raster_cfg.get("blocksize", 512)/16)*16)))

                raw_path = str(Path(tmp) / f"raw_{prefix}.tif")
                cog_path = str(Path(tmp) / f"{prefix}_{pair}.tif")
                
                with rasterio.open(raw_path, "w", **profile) as dst:
                    dst.write(grid, 1)
                    
                # Translate to COG
                _build_overviews_and_cog(raw_path, cog_path, raster_cfg)
                
                # Upload to S3
                final_uri = str(UPath(out_dir) / f"{prefix}_{pair}.tif")
                if is_aws and final_uri.startswith("s3://"):
                    s3fs.S3FileSystem().put(cog_path, final_uri)
                else:
                    UPath(final_uri).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(cog_path, final_uri)

        if verbose:
            Engine.write_message_dask(f" [{current}/{total}] [SUCCESS] Tile '{tile_id}' COGs uploaded.", OUTPUTS)
            
        return {"ok": True, "tile_id": tile_id, "pair": pair}

    except Exception as e:
        Engine.write_message_dask(f"ERROR: RasterExport failed for tile {tile_id} - pair {pair}: {e}", OUTPUTS)
        return {"ok": False, "tile_id": tile_id, "pair": pair, "reason": str(e)}
    finally:
        gc.collect()

class MLRasterExportEngine(Engine):
    """Class for converting XGBoost point prediction tables into Cloud Optimized GeoTIFFs."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix

        # EC2 Temp Storage mapping
        self.local_tmp_dir = pathlib.Path(str(Path.home() / "hydro_health_local_tmp" / "raster_tmp"))
        self.local_tmp_dir.mkdir(parents=True, exist_ok=True)
        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']

    def _resolve_paths(self, region: str) -> None:
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        
        bucket = get_config_item('S3', 'BUCKET_NAME')
        s3_dir_base = f"s3://{bucket}/{region}"

        # Get Inputs from XGBoost Modeling Engine
        xgb_output_dir = get_config_item('MODEL', 'XGB_OUTPUT_DIR')
        self.xgb_in_dir = UPath(f"{s3_dir_base}/{xgb_output_dir}") if self.is_aws else UPath(self.outputs_dir / xgb_output_dir)
        
        # Get Template Raster and Grids
        self.template_uri = get_config_item('MODEL', 'TEMPLATE_RASTER_URI')
        self.template_uri = f"{s3_dir_base}/{self.template_uri}" if self.is_aws else str(self.outputs_dir / self.template_uri)

        prediction_subgrid_path = get_config_item('MODEL', 'PREDICTION_SUB_GRIDS')
        subgrid_full_path = UPath(f"{s3_dir_base}/{prediction_subgrid_path}") if self.is_aws else UPath(self.outputs_dir / prediction_subgrid_path)
        self.subgrids = gpd.read_file(str(subgrid_full_path))

        # Raster Output Directories
        raster_output_dir = get_config_item('MODEL', 'RASTER_OUTPUT_DIR')
        self.raster_out_dir = UPath(f"{s3_dir_base}/{raster_output_dir}") if self.is_aws else UPath(self.outputs_dir / raster_output_dir)

    def run(self) -> None:
        """Main execution method pulling tables and distributing Raster rendering tasks."""
        env = self.param_lookup.get('env', 'local')
        model_cfg = self.param_lookup.get('model_config', {})
        raster_cfg = self.param_lookup.get('raster_config', {})
        crs = model_cfg.get('crs', "EPSG:32617")
        year_pairs = model_cfg.get('year_pairs', [])
        verbose_workers = model_cfg.get('verbose_logging', False)

        try:
            # Rendering is CPU/memory heavy, limit workers accordingly
            self.setup_dask(env, n_workers=2, threads_per_worker=2, memory_limit="10GB")
            
            for eco_region in self.param_lookup['eco_regions'].value:
                self._resolve_paths(eco_region)
                
                # Pre-fetch geometries from subgrids for boundary clipping
                geometries = {row['tile_id']: row['geometry'].wkb for _, row in self.subgrids.iterrows() if row['geometry'] is not None}

                if self.is_aws:
                    fs = s3fs.S3FileSystem(anon=False)
                    processed_tile_dirs = [d.split('/')[-1] for d in fs.ls(str(self.xgb_in_dir)) if fs.isdir(d)]
                else:
                    processed_tile_dirs = [d.name for d in self.xgb_in_dir.iterdir() if d.is_dir()]

                params_list = []
                total_tasks = len(processed_tile_dirs) * len(year_pairs)
                idx = 1
                
                for tile_id in processed_tile_dirs:
                    if tile_id not in geometries:
                        continue
                        
                    for pair in year_pairs:
                        table_uri = str(self.xgb_in_dir / tile_id / f"LOCAL_prediction_summary_{pair}.parquet")
                        out_dir = str(self.raster_out_dir / tile_id)
                        
                        fs_check = s3fs.S3FileSystem(anon=False) if self.is_aws else None
                        exists = fs_check.exists(table_uri) if self.is_aws else Path(table_uri).exists()
                        
                        if exists:
                            params_list.append([
                                tile_id, pair, table_uri, self.template_uri, out_dir, 
                                geometries[tile_id], crs, raster_cfg, self.is_aws, 
                                str(self.local_tmp_dir), idx, total_tasks, verbose_workers
                            ])
                            idx += 1

                self.write_message(f"Submitting {len(params_list)} COG export tasks to Dask...", OUTPUTS)
                futures = self.client.map(_export_cog_task, params_list)
                results = self.client.gather(futures)

                valid_results = [r for r in results if r.get('ok')]
                self.write_message(f"Successfully generated COGs for {len(valid_results)} / {len(params_list)} tables.", OUTPUTS)

        finally:
            self.cleanup_resources(OUTPUTS)