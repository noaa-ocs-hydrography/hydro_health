import copy
import gc
import json
import math
import os
import re
import shutil
import tempfile
from itertools import islice
from pathlib import Path
from uuid import uuid4

import cmocean
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from dask.distributed import as_completed

from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.shutil import copy as raster_copy
from rasterio.windows import Window, from_bounds
from rasterio.windows import transform as window_transform
from shapely import wkb
from shapely.errors import ShapelyError
from shapely.validation import explain_validity
from shapely.geometry import mapping
from upath import UPath
matplotlib.use("Agg")

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


OUTPUTS = Path(__file__).parents[3] / "outputs"

TRAIN_PRODUCTS = {
    "bathy_t": "LOCAL_Train_Start_Bathy_t",
    "mean_predicted_bathy_t1": "LOCAL_Train_Mean_Predicted_Bathy_t1",
    "delta_bathy": "LOCAL_Train_Surveyed_Change",
    "mean_predicted_change": "LOCAL_Train_Mean_Predicted_Change",
    "uncertainty_sd_bathy_t1": "LOCAL_Train_SD_Bathy_t1",
    "mean_predicted_rate": "LOCAL_Train_Mean_Annualized_Rate",
}
PRED_PRODUCTS = {
    "bathy_t": "LOCAL_Pred_Start_Bathy_t",
    "mean_predicted_bathy_t1": "LOCAL_Pred_Mean_Bathy_t1",
    "mean_predicted_change": "LOCAL_Pred_Mean_Change",
    "uncertainty_sd_bathy_t1": "LOCAL_Pred_SD_Bathy_t1",
    "mean_predicted_rate": "LOCAL_Pred_Mean_Annualized_Rate",
    "predictor_extrapolation_count": "LOCAL_Predictor_Extrapolation_Count",
    "UC_t": "LOCAL_Pred_BlueTopo_UC_t",
    "survey_year_bin_code": "LOCAL_Pred_Survey_Year_Bin",
}
OPTIONAL = {"UC_t", "survey_year_bin_code"}
SURVEY_LABELS = ["2025+", "2024", "2023", "2022", "2021", "2020",
                 "2015-2019", "2010-2014", "2000-2009", "1990-1999", "<1990"]

def _copy_file(source: str, destination: str) -> None:
    """Copy one object; local destinations use an atomic sibling-file replacement."""

    target = UPath(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if str(target).startswith("s3://"):
        with UPath(source).open("rb") as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    else:
        temporary = Path(str(target)).with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            with UPath(source).open("rb") as src, temporary.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            os.replace(temporary, str(target))
        finally:
            temporary.unlink(missing_ok=True)

class LocalizedFile:
    """Provide a local path for GDAL-backed readers and clean up S3 downloads."""

    def __init__(self, uri: str) -> None:
        """Retain the source URI until the context is entered."""

        self.uri = str(uri)
        self.tempdir = None

    def __enter__(self) -> str:
        """Download S3 assets only when a local GDAL path is needed."""

        if not self.uri.startswith("s3://"):
            return self.uri
        self.tempdir = tempfile.TemporaryDirectory(prefix="hh_asset_")
        path = str(Path(self.tempdir.name) / UPath(self.uri).name)
        try:
            _copy_file(self.uri, path)
        except Exception:
            self.tempdir.cleanup()
            raise
        return path

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Remove downloaded assets while leaving caller-owned local files intact."""

        if self.tempdir is not None:
            self.tempdir.cleanup()

def _dataframe_to_grid(df: pd.DataFrame, value_col: str, template_metadata: dict, tile_geom_wkb: bytes, crs: str, nodata: float = -9999.0, max_grid_cells: int = 25000000, diagnostics: dict | None = None) -> tuple[np.ndarray, dict]:
    """Mean finite point values in each cell, snap outward, and mask cell centres.

    The official polygon determines extent, never the surviving data rows.
    Template values/NoData are not used as a second mask, matching the R writer.
    Each layer filters only its own X/Y/value; driver NAs do not remove rows.
    Survey classes use the modal class per cell (ties favour newer classes).
    """

    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.clear()
    diagnostics.update(input_rows=len(df), value_column=value_col, target_crs=str(crs),
                       dataframe_geometry_crs=str(getattr(df, "crs", None)),
                       template_crs=str(template_metadata["crs"]), template_uri=template_metadata.get("source_uri"),
                       template_shape=[template_metadata["height"], template_metadata["width"]],
                       template_transform=list(template_metadata["transform"]), nodata=nodata)
    missing = [column for column in ("X", "Y", value_col) if column not in df]
    if missing:
        diagnostics["missing_columns"] = missing
        raise ValueError(f"Missing raster input columns: {missing}")
    geom = wkb.loads(tile_geom_wkb)
    diagnostics["polygon_bounds"] = list(geom.bounds)
    if not geom.is_valid or geom.is_empty or geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError("Tile geometry must be a valid nonempty polygon")
    if rasterio.crs.CRS.from_user_input(template_metadata["crs"]) != rasterio.crs.CRS.from_user_input(crs):
        raise ValueError(f"Template CRS must match {crs}; no implicit reprojection")
    t = template_metadata["transform"]
    if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
        raise ValueError("Use a north-up, unrotated template")
    raw = from_bounds(*geom.bounds, transform=t)
    c0, r0 = math.floor(raw.col_off + 1e-8), math.floor(raw.row_off + 1e-8)
    c1 = math.ceil(raw.col_off + raw.width - 1e-8)
    r1 = math.ceil(raw.row_off + raw.height - 1e-8)
    if c0 < 0 or r0 < 0 or c1 > template_metadata["width"] or r1 > template_metadata["height"]:
        raise ValueError("Template must cover the complete tile bounds; refusing silent clipping")
    height, width = r1 - r0, c1 - c0
    if height <= 0 or width <= 0 or height * width > max_grid_cells:
        raise ValueError("Invalid or oversized tile grid; check template/tile or max_grid_cells")
    transform = window_transform(Window(c0, r0, width, height), t)
    diagnostics.update(tile_grid_shape=[height, width], pixel_size=[t.a, abs(t.e)],
                       tile_grid_bounds=list(rasterio.transform.array_bounds(height, width, transform)))
    arrays = [pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in ("X", "Y", value_col)]
    xs, ys, values = arrays
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    for column, array in zip(("X", "Y", value_col), arrays):
        finite = np.isfinite(array)
        diagnostics[f"{column}_nonfinite_or_nonnumeric_rows"] = int((~finite).sum())
        diagnostics[f"{column}_finite_range"] = [float(array[finite].min()), float(array[finite].max())] if finite.any() else None
    diagnostics["finite_xy_value_rows"] = int(keep.sum())
    diagnostics["source_nodata_rows"] = int((keep & (values == nodata)).sum())
    rows, cols = rasterio.transform.rowcol(transform, xs[keep], ys[keep])
    rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows, cols, vals = rows[inside], cols[inside], values[keep][inside]
    diagnostics["rows_inside_tile_grid"] = len(vals)
    diagnostics["rows_outside_tile_grid"] = int(keep.sum()) - len(vals)
    grid = np.full((height, width), nodata, dtype="float32")
    if value_col == "survey_year_bin_code" and len(vals):
        votes = pd.DataFrame({"r": rows, "c": cols, "v": vals}).groupby(["r", "c", "v"]).size().reset_index(name="n")
        winners = votes.sort_values(["n", "v"], ascending=[False, True]).drop_duplicates(["r", "c"])
        grid[winners.r.to_numpy(), winners.c.to_numpy()] = winners.v.to_numpy()
    elif len(vals):
        accum = np.zeros(grid.shape, dtype="float64")
        counts = np.zeros(grid.shape, dtype="uint32")
        np.add.at(accum, (rows, cols), vals)
        np.add.at(counts, (rows, cols), 1)
        populated = counts > 0
        grid[populated] = (accum[populated] / counts[populated]).astype("float32")
    valid_polygon = geometry_mask([mapping(geom)], out_shape=grid.shape,
                                  transform=transform, invert=True, all_touched=False)
    diagnostics["polygon_center_cells"] = int(valid_polygon.sum())
    diagnostics["populated_cells_before_polygon_mask"] = int(np.count_nonzero(grid != nodata))
    diagnostics["input_rows_in_polygon_center_cells"] = int(valid_polygon[rows, cols].sum())
    grid[~valid_polygon] = nodata
    diagnostics["populated_cells_after_polygon_mask"] = int(np.count_nonzero(grid != nodata))
    diagnostics["nonfinite_output_cells"] = int(np.count_nonzero(~np.isfinite(grid)))
    if np.any(grid[valid_polygon] == nodata) and np.any(vals == nodata):
        raise ValueError("A finite source value equals the configured NoData sentinel")
    profile = dict(driver="GTiff", count=1, width=width, height=height,
                   transform=transform, crs=crs, dtype="float32", nodata=nodata)
    return grid, profile

def _build_overviews_and_cog(raw_tif: Path, cog_tif: Path, cfg: dict) -> None:
    """Use Rasterio's bundled GDAL COG driver; never label a plain TIFF as a COG."""

    levels = cfg.get("overview_levels", [2, 4, 8, 16, 32])
    resampling = cfg.get("overview_resampling", "average")
    with rasterio.open(raw_tif, "r+") as dst:
        usable = [int(v) for v in levels if v > 1 and min(dst.width, dst.height) // v >= 1]
        if usable:
            dst.build_overviews(usable, rasterio.enums.Resampling[resampling])
    with rasterio.Env() as env:
        if "COG" not in env.drivers():
            raise RuntimeError("Installed Rasterio/GDAL lacks the COG driver")
    raster_copy(raw_tif, cog_tif, driver="COG", COMPRESS=cfg.get("compress", "DEFLATE"),
                BLOCKSIZE=cfg.get("blocksize", 512), BIGTIFF="IF_SAFER",
                OVERVIEWS="FORCE_USE_EXISTING", RESAMPLING=resampling.upper(), NUM_THREADS="1")
    with rasterio.open(cog_tif) as src:
        if src.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") != "COG":
            raise RuntimeError("Output does not advertise COG layout")

def _survey_codes(series: pd.Series, legacy_offset: bool) -> np.ndarray:
    """Translate the R survey-year bins; optionally preserve its +800 legacy fix.

    YYYY and YYYY.fraction are accepted. ISO strings are decoded as dates.
    Numeric epoch dates are deliberately not guessed.
    """

    year = pd.to_numeric(series, errors="coerce").astype(float)
    if not pd.api.types.is_numeric_dtype(series):
        iso = series.astype(str).str.match(r"^\d{4}-\d{2}-\d{2}", na=False)
        year.loc[iso] = pd.to_datetime(series.loc[iso], errors="coerce").dt.year
    if legacy_offset:
        legacy = (year >= 1000) & (year < 1800)
        year.loc[legacy] += 800
    year = np.floor(year.to_numpy())
    valid = np.isfinite(year) & (year >= 1800) & (year <= 2100)
    code = np.full(len(year), np.nan)
    for i, condition in enumerate([year >= 2025, year == 2024, year == 2023,
        year == 2022, year == 2021, year == 2020,
        (year >= 2015) & (year <= 2019), (year >= 2010) & (year <= 2014),
        (year >= 2000) & (year <= 2009), (year >= 1990) & (year <= 1999), year < 1990]):
        code[valid & condition] = i
    return code

def add_extrapolation_flags(training: pd.DataFrame, prediction: pd.DataFrame, predictors: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Replicate R A14 finite training-range checks, with no predictor imputation."""

    out = prediction.copy()
    count = np.zeros(len(out), dtype=int)
    ranges = []
    for p in predictors:
        if p not in training or p not in out:
            raise ValueError(f"Locked model predictor missing from summary: {p}")
        train = pd.to_numeric(training[p], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(out[p], errors="coerce").to_numpy(dtype=float)
        finite = train[np.isfinite(train)]
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (None, None)
        if len(finite):
            count += np.isfinite(pred) & ((pred < lo) | (pred > hi))
        ranges.append({"predictor": p, "min_training": lo, "max_training": hi})
    out["predictor_extrapolation_count"] = count
    out["predictor_extrapolation_any"] = count > 0
    return out, ranges

class RasterExportDiagnostic(ValueError):
    """Report a tile diagnostic to the worker without failing the export queue."""


def _export_cog_products(df: pd.DataFrame, role: str, cfg: dict) -> dict:
    """Rasterize one role into a private staging directory at full resolution."""

    pair = cfg["pair"]
    products = TRAIN_PRODUCTS if role == "training" else PRED_PRODUCTS
    outputs, skipped = {}, []
    with tempfile.TemporaryDirectory(prefix="hh_raster_") as tmp:
        for column in products:
            if column not in df:
                if column in OPTIONAL:
                    skipped.append({"column": column, "reason": "not supplied by model summary"})
                    continue
                raise RasterExportDiagnostic(f"Required {role} output column is absent: {column}")
            diagnostics = {"region": cfg["region"], "tile_id": cfg["tile_id"], "pair": pair,
                           "role": role, "column": column, "summary_path": cfg[f"{role}_summary"]}
            grid_diagnostics = {}
            try:
                grid, profile = _dataframe_to_grid(
                    df, column, cfg["template_metadata"], cfg["geometry"], cfg["crs"],
                    cfg["raster_config"]["nodata"], cfg["raster_config"]["max_grid_cells"], grid_diagnostics)
            except (ValueError, KeyError, ShapelyError) as exc:
                diagnostics.update(grid_diagnostics)
                raise RasterExportDiagnostic(f"Rasterization failed for {role}/{column} on {cfg['tile_id']}: {exc}\nDiagnostics: {json.dumps(diagnostics, indent=2, default=str)}") from exc
            diagnostics.update(grid_diagnostics)
            if column not in OPTIONAL and not np.any(grid != profile["nodata"]):
                if not diagnostics["input_rows"]:
                    reason = "The summary contains no rows."
                elif not diagnostics["finite_xy_value_rows"]:
                    reason = "No row has finite numeric X, Y and value together; inspect the per-column invalid counts."
                elif not diagnostics["rows_inside_tile_grid"]:
                    reason = "All usable coordinates fall outside the tile grid; compare coordinate ranges, polygon bounds and CRS. The geometry CRS does not verify the X/Y columns."
                elif not diagnostics["polygon_center_cells"]:
                    reason = "No template cell centers fall inside the polygon; inspect pixel size and grid alignment."
                elif not diagnostics["input_rows_in_polygon_center_cells"]:
                    reason = "All usable points map to cells removed by the polygon center mask; inspect point/tile overlap and grid alignment."
                else:
                    reason = "Points reach retained cells, but every aggregated output equals NoData; inspect source_nodata_rows and the value range."
                raise RasterExportDiagnostic(f"No populated {role}/{column} cells for {cfg['tile_id']}: {reason}\nDiagnostics: {json.dumps(diagnostics, indent=2, default=str)}")
            raw = Path(tmp) / "raw.tif"
            cog = Path(cfg["raster_paths"][role][column])
            raster_cfg = dict(cfg["raster_config"])
            profile.update(tiled=True, blockxsize=raster_cfg["blocksize"],
                           blockysize=raster_cfg["blocksize"], compress=raster_cfg["compress"])
            with rasterio.open(raw, "w", **profile) as dst:
                dst.write(grid, 1)
                dst.update_tags(value_column=column, pair=pair, role=role,
                    aggregation="mode" if column == "survey_year_bin_code" else "mean",
                    survey_classes=json.dumps(SURVEY_LABELS) if column == "survey_year_bin_code" else "")
            del grid
            if column == "survey_year_bin_code":
                raster_cfg["overview_resampling"] = "mode"
            _build_overviews_and_cog(raw, cog, raster_cfg)
            outputs[column] = str(cog)
    return {"ok": True, "role": role, "pair": pair, "outputs": outputs, "skipped": skipped}

def _read_layer(path: str, max_dimension: int, categorical: bool = False) -> tuple[np.ma.MaskedArray, tuple[float, float, float, float]]:
    """Read a bounded float32 preview; exported COG resolution stays unchanged."""

    with rasterio.open(path) as src:
        scale = min(1.0, max_dimension / max(src.height, src.width))
        shape = (max(1, math.ceil(src.height * scale)), max(1, math.ceil(src.width * scale)))
        method = Resampling.mode if categorical else Resampling.average
        data = src.read(1, out_shape=shape, out_dtype="float32", masked=True, resampling=method)
        return data, (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

def _plot_summary_pages(results: dict, train_out: Path, pair: str, tile_id: str, max_dimension: int) -> list[str]:
    """Plot comparison PDFs from saved COG values using shared colour scales."""

    plot_dir = Path(train_out) / "diagnostic_plots"
    plot_dir.mkdir(exist_ok=True)
    layers = {}
    plotted = {"bathy_t", "mean_predicted_bathy_t1", "mean_predicted_change", "delta_bathy",
               "uncertainty_sd_bathy_t1", "UC_t", "survey_year_bin_code"}
    for role, result in results.items():
        for col, path in result["outputs"].items():
            if col in plotted:
                layers[role, col] = _read_layer(path, max_dimension, col == "survey_year_bin_code")
    # Plot-only water mask, as in R A15. Exported rasters retain all valid cells.
    for role in ("training", "prediction"):
        water = layers[role, "bathy_t"][0]
        water_mask = np.ma.getmaskarray(water) | (water.filled(0) >= 0)
        for (r, col), (data, extent) in list(layers.items()):
            if r == role:
                mask = np.ma.getmaskarray(data) | water_mask
                layers[r, col] = (np.ma.array(data.data, mask=mask, copy=False), extent)
    def finite_values(keys: list[tuple[str, str]]) -> np.ndarray:
        """Collect finite water-cell values so related panels use the same scale."""

        parts = [layers[k][0].compressed() for k in keys if k in layers]
        v = np.concatenate(parts) if parts else np.array([])
        return v[np.isfinite(v)]
    depths = finite_values([(r, c) for r in results for c in ("bathy_t", "mean_predicted_bathy_t1")])
    changes = finite_values([(r, "mean_predicted_change") for r in results] + [("training", "delta_bathy")])
    sd = finite_values([(r, "uncertainty_sd_bathy_t1") for r in results] + [("prediction", "UC_t")])
    lo, hi = (float(depths.min()), float(depths.max())) if len(depths) else (-1., 0.)
    if hi <= lo:
        hi = lo + 1
    dmax = max(float(np.abs(changes).max()) if len(changes) else 1., 1e-8)
    smax = max(float(sd.max()) if len(sd) else 1., 1e-8)
    norms = {"depth": Normalize(lo, hi), "change": TwoSlopeNorm(0, -dmax, dmax),
             "uncertainty": Normalize(0, smax), "survey": BoundaryNorm(np.arange(-.5, 11.5), 11)}
    cmaps = {"depth": "viridis", "change": cmocean.cm.curl, "uncertainty": "viridis",
             "survey": plt.get_cmap("viridis", 11)}
    def panel(ax: Axes, role: str, col: str, title: str, kind: str) -> None:
        """Draw a saved raster with its shared normalization or a clearly labeled no-data panel."""

        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        if (role, col) not in layers or not layers[role, col][0].count():
            ax.text(.5, .5, "No data supplied", ha="center", va="center", transform=ax.transAxes)
            return
        data, extent = layers[role, col]
        artist = ax.imshow(data, extent=extent, origin="upper", cmap=cmaps[kind],
                           norm=norms[kind], interpolation="nearest")
        ax.set_aspect("equal")
        if kind == "survey":
            codes = sorted(set(data.compressed().astype(int)))
            ax.legend(handles=[Patch(color=cmaps[kind](i), label=SURVEY_LABELS[i]) for i in codes],
                      loc="lower left", fontsize=7, framealpha=.9)
        else:
            label = {"depth": "Bathymetry (m)", "change": "Change (m)\n- erosion / + deposition",
                     "uncertainty": "UC (m)" if col == "UC_t" else "Bootstrap SD (m)"}[kind]
            bar = ax.figure.colorbar(artist, ax=ax, shrink=.72, pad=.025)
            bar.set_label(label, fontsize=8)
            bar.ax.tick_params(labelsize=7)
    common = [("training", "bathy_t", "Train start bathymetry", "depth"),
              ("training", "mean_predicted_bathy_t1", "Train mean prediction (t1)", "depth"),
              ("training", "mean_predicted_change", "Train predicted change [t1 - t]", "change"),
              ("training", "delta_bathy", "Train surveyed change [t1 - t]", "change"),
              ("prediction", "bathy_t", "Prediction start bathymetry", "depth"),
              ("prediction", "mean_predicted_bathy_t1", "Prediction mean bathymetry (t1)", "depth"),
              ("prediction", "mean_predicted_change", "Prediction change [t1 - t]", "change"),
              ("prediction", "survey_year_bin_code", "Prediction survey end year", "survey")]
    files = []
    for uncertainty in (False, True):
        spec = list(common)
        if uncertainty:
            spec[3] = ("training", "uncertainty_sd_bathy_t1", "Train bootstrap uncertainty", "uncertainty")
            col = "UC_t" if ("prediction", "UC_t") in layers else "uncertainty_sd_bathy_t1"
            spec[7] = ("prediction", col, "BlueTopo uncertainty (UC_t)" if col == "UC_t" else "Prediction bootstrap uncertainty", "uncertainty")
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), layout="constrained")
        for ax, args in zip(axes.flat, spec):
            panel(ax, *args)
        fig.suptitle(f"{tile_id} | {pair} | Training and prediction" + (" - uncertainty" if uncertainty else ""), fontsize=16)
        name = f"LOCAL_Summary_Training_vs_Prediction{'_Uncertainty' if uncertainty else ''}_{pair}.pdf"
        file = plot_dir / name
        fig.savefig(file, dpi=180)
        plt.close(fig)
        files.append(str(file))
    # Additional page exposes prediction MODEL SD even when the R panel chooses UC.
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout="constrained")
    panel(axes[0], "training", "uncertainty_sd_bathy_t1", "Training bootstrap SD", "uncertainty")
    panel(axes[1], "prediction", "uncertainty_sd_bathy_t1", "Prediction bootstrap SD", "uncertainty")
    panel(axes[2], "prediction", "UC_t", "BlueTopo uncertainty (UC_t)", "uncertainty")
    fig.suptitle(f"{tile_id} | {pair} | Model SD and input UC are different quantities", fontsize=13)
    file = plot_dir / f"LOCAL_Uncertainty_Comparison_{pair}.pdf"
    fig.savefig(file, dpi=180)
    plt.close(fig)
    files.append(str(file))
    return files

def _plot_fit_diagnostics(train: pd.DataFrame, train_out: Path, pair: str, tile_id: str, max_points: int) -> list[str]:
    """Plot apparent training fit and optional SHAP/gain summaries."""

    plot_dir = train_out / "diagnostic_plots"
    plot_dir.mkdir(exist_ok=True)
    files = []
    actual = pd.to_numeric(train.bathy_t1, errors="coerce").to_numpy(float)
    fitted = pd.to_numeric(train.mean_predicted_bathy_t1, errors="coerce").to_numpy(float)
    keep = np.isfinite(actual) & np.isfinite(fitted)
    if keep.sum() > 2:
        a, p = actual[keep], fitted[keep]
        rmse = np.sqrt(np.mean((p - a)**2))
        denom = np.sum((a - a.mean())**2)
        r2 = 1 - np.sum((p - a)**2) / denom if denom else np.nan
        fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
        step = max(1, math.ceil(len(a) / max_points))
        ax.scatter(a[::step], p[::step], s=2, alpha=.15, rasterized=True)
        bounds = [min(a.min(), p.min()), max(a.max(), p.max())]
        ax.plot(bounds, bounds, "r--")
        ax.set_aspect("equal")
        ax.set(xlabel="Surveyed bathymetry (t1, m)", ylabel="Mean predicted bathymetry (t1, m)",
               title=f"{tile_id} | {pair}\nApparent training fit: RMSE={rmse:.3f} m; predictive R²={r2:.3f}")
        file = plot_dir / f"LOCAL_model_fit_t1_{pair}.png"
        fig.savefig(file, dpi=180)
        plt.close(fig)
        files.append(str(file))
    for source, column, label, name in [("shap_summary", "mean_abs_shap", "Mean absolute SHAP (m)", "shap_importance"),
                                     ("importance", "gain_fraction", "Mean gain fraction", "gain_importance")]:
        path = Path(train_out) / f"LOCAL_{source}_{pair}.parquet"
        if not path.exists():
            continue
        data = pd.read_parquet(path)
        if data.empty or column not in data:
            continue
        values = data.groupby("predictor")[column].mean().dropna().nlargest(20).sort_values()
        if values.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, max(5, .3 * len(values) + 2)), layout="constrained")
        ax.barh(values.index, values.values, color="steelblue")
        ax.set(xlabel=label, title=f"{tile_id} | {pair} | {name.replace('_', ' ')}")
        file = plot_dir / f"LOCAL_{name}_{pair}.png"
        fig.savefig(file, dpi=180)
        plt.close(fig)
        files.append(str(file))
    return files

def _make_visual_outputs(train: pd.DataFrame, results: dict, cfg: dict) -> list[str]:
    """Render bounded previews within this tile's single-threaded worker process."""

    root = Path(cfg["training_output_dir"])
    for name in ("importance", "shap_summary"):
        uri = cfg[name]
        if UPath(uri).exists():
            _copy_file(uri, str(root / UPath(uri).name))
    try:
        plots = _plot_summary_pages(results, root, cfg["pair"], cfg["tile_id"], cfg["plot_max_dimension"])
        plots += _plot_fit_diagnostics(train, root, cfg["pair"], cfg["tile_id"], cfg["plot_max_points"])
        return plots
    finally:
        plt.close("all")

def _load_summary(path: str, role: str, cfg: dict, predictors: list[str]) -> pd.DataFrame:
    """Read a spatial model summary and check the fields needed for export."""

    with UPath(path).open("rb") as stream:
        spatial = gpd.read_parquet(stream)
    if spatial.crs is None or spatial.crs != cfg["crs"]:
        raise ValueError(f"{role} summary must already use {cfg['crs']}")
    df = pd.DataFrame(spatial.drop(columns=spatial.geometry.name)).reset_index(drop=True)
    products = TRAIN_PRODUCTS if role == "training" else PRED_PRODUCTS
    required = (set(products) - OPTIONAL - {"predictor_extrapolation_count"}) | {"X", "Y", "FID"} | set(predictors)
    required |= {"bathy_t1"} if role == "training" else {"prediction_row_usable"}
    if not df.columns.is_unique or df.empty:
        raise ValueError(f"{role} summary is empty or has duplicate columns")
    if missing := required - set(df.columns):
        raise ValueError(f"Missing {role} summary fields: {sorted(missing)}")
    if df.FID.isna().any() or df.FID.duplicated().any():
        raise ValueError(f"{role} summary FIDs must be nonmissing and unique")
    for column, expected in (("tile_id", cfg["tile_id"]), ("pair_id", cfg["pair"])):
        if column in df and not df[column].eq(expected).all():
            raise ValueError(f"{role} summary {column} differs from {expected}")
    usable = np.ones(len(df), dtype=bool)
    if role == "prediction":
        if not df.prediction_row_usable.isin([True, False]).all():
            raise ValueError("prediction_row_usable must contain boolean values")
        usable = df.prediction_row_usable.to_numpy(dtype=bool)
    numeric = ["X", "Y", "bathy_t", "mean_predicted_bathy_t1", "mean_predicted_change", "mean_predicted_rate", "uncertainty_sd_bathy_t1"]
    if not usable.any() or not np.isfinite(df.loc[usable, numeric].to_numpy(dtype=float)).all():
        raise ValueError(f"Usable {role} rows must have finite coordinates and model outputs")
    return df

def _load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Read the current model handoff without requiring test reports or manifests."""

    metadata = json.loads(UPath(cfg["model_metadata"]).read_text(encoding="utf-8"))
    if any(metadata.get(key) != cfg[key] for key in ("tile_id", "pair", "crs")):
        raise ValueError("Model metadata does not match the configured tile, pair and CRS")
    predictors = metadata.get("predictors")
    if not predictors or "bathy_t" not in predictors or len(predictors) != len(set(predictors)):
        raise ValueError("Model metadata needs the locked predictor schema including bathy_t")
    if metadata.get("target") != "bathy_t1" or metadata.get("prediction_baseline_source") != "bathy_t":
        raise ValueError("Expected bathy_t1 target and bathy_t prediction baseline")
    train = _load_summary(cfg["training_summary"], "training", cfg, predictors)
    prediction = _load_summary(cfg["prediction_summary"], "prediction", cfg, predictors)
    for role, frame, key in (("training", train, "n_fit"), ("prediction", prediction, "n_prediction_rows")):
        if key in metadata and len(frame) != metadata[key]:
            raise ValueError(f"{role} row count differs from model metadata")
    prediction, ranges = add_extrapolation_flags(train, prediction, predictors)
    if "survey_end_date" in prediction:
        prediction["survey_year_bin_code"] = _survey_codes(prediction.survey_end_date, cfg["legacy_survey_year_offset"])
    return train, prediction, ranges

def _existing_outputs(cfg: dict) -> list[str]:
    """Find this tile/year pair's previous outputs without deleting them."""

    paths = [path for role in cfg["raster_paths"].values() for path in role.values()]
    paths += [cfg["extrapolation_path"], cfg["ranges_path"]]
    paths += [str(path) for path in UPath(cfg["plot_output_dir"]).glob(f"LOCAL_*_{cfg['pair']}.*")]
    return [path for path in paths if UPath(path).exists()]

def _stage_config(cfg: dict, root: Path) -> dict:
    """Redirect generated products to private local folders until the tile succeeds."""

    staged = dict(cfg)
    staged["raster_paths"] = {}
    for role in ("training", "prediction"):
        folder = root / role.capitalize()
        folder.mkdir(parents=True)
        staged[f"{role}_output_dir"] = str(folder)
        staged["raster_paths"][role] = {key: str(folder / UPath(path).name)
                                         for key, path in cfg["raster_paths"][role].items()}
    staged["plot_output_dir"] = str(root / "Training" / "diagnostic_plots")
    for key in ("extrapolation_path", "ranges_path"):
        staged[key] = str(root / "Prediction" / UPath(cfg[key]).name)
    return staged

def _publish_outputs(files: list[tuple[str, str]], cfg: dict) -> None:
    """Publish a finished tile and restore previous files if publication raises."""

    previous = _existing_outputs(cfg)
    if previous and not cfg["overwrite"]:
        raise FileExistsError(f"Outputs exist for {cfg['tile_id']}, {cfg['pair']}")
    backup_dir = Path(tempfile.mkdtemp(prefix="hh_export_backup_"))
    backups = {}
    try:
        for index, path in enumerate(previous):
            backup = str(backup_dir / str(index))
            _copy_file(path, backup)
            backups[path] = backup
    except Exception:
        shutil.rmtree(backup_dir)
        raise
    attempted = []
    try:
        for source, destination in files:
            attempted.append(destination)
            _copy_file(source, destination)
        generated = {destination for _, destination in files}
        for path in previous:
            if path not in generated:
                UPath(path).unlink()
    except Exception:
        try:
            for path in attempted:
                if path not in backups and UPath(path).exists():
                    UPath(path).unlink()
            for path, backup in backups.items():
                _copy_file(backup, path)
        except Exception as rollback_error:
            raise RuntimeError(f"Publication rollback failed; backups retained at {backup_dir}") from rollback_error
        shutil.rmtree(backup_dir)
        raise
    shutil.rmtree(backup_dir)

def _export_tile_task(cfg: dict) -> dict:
    """Load, stage and publish one complete tile in an isolated Dask process."""

    Engine.write_message_dask(f"Exporting {cfg['tile_id']} | {cfg['pair']}", cfg["log_dir"])
    try:
        if not cfg["overwrite"] and _existing_outputs(cfg):
            raise FileExistsError(f"Outputs exist for {cfg['tile_id']}, {cfg['pair']}")
        train, prediction, ranges = _load_inputs(cfg)
        with tempfile.TemporaryDirectory(prefix="hh_export_tile_") as tmp:
            staged = _stage_config(cfg, Path(tmp))
            results = {role: _export_cog_products(frame, role, staged)
                       for role, frame in (("training", train), ("prediction", prediction))}
            prediction[["X", "Y", "predictor_extrapolation_count", "predictor_extrapolation_any"]].to_parquet(
                staged["extrapolation_path"], index=False)
            Path(staged["ranges_path"]).write_text(json.dumps(ranges, indent=2), encoding="utf-8")
            visuals = _make_visual_outputs(train, results, staged)
            files = []
            for role in ("training", "prediction"):
                folder = Path(staged[f"{role}_output_dir"])
                for path in folder.rglob("*"):
                    if path.is_file() and path.name not in (UPath(cfg["importance"]).name, UPath(cfg["shap_summary"]).name):
                        destination = str(UPath(cfg[f"{role}_output_dir"]) / str(path.relative_to(folder)))
                        files.append((str(path), destination))
            _publish_outputs(files, cfg)
            published = dict(files)
            for result in results.values():
                result["outputs"] = {key: published[path] for key, path in result["outputs"].items()}
            Engine.write_message_dask(f"Successfully exported {cfg['region']}/{cfg['tile_id']} | {cfg['pair']}", cfg["log_dir"])
            return {"region": cfg["region"], "tile_id": cfg["tile_id"], "pair": cfg["pair"],
                    "results": results, "visual_outputs": [published[path] for path in visuals]}
    except RasterExportDiagnostic as exc:
        Engine.write_message_dask(f"Skipping raster export for {cfg['region']}/{cfg['tile_id']} | {cfg['pair']}: {exc}", cfg["log_dir"])
        return {"region": cfg["region"], "tile_id": cfg["tile_id"], "pair": cfg["pair"],
                "skipped": True, "reason": str(exc), "results": {}, "visual_outputs": []}
    finally:
        gc.collect()

class MLRasterExportEngine(Engine):
    """Export model summaries as COGs and diagnostic plots for each ecoregion."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize shared engine state and raster-export settings."""

        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        if param_lookup["env"] not in ("local", "aws"):
            raise ValueError("env must be local or aws")
        self.is_aws = param_lookup["env"] == "aws"
        self.settings = {
            "tile_id": None, "pair": None, "tile_id_column": "tile_id",
            "legacy_survey_year_offset": True,
            "raster_workers": 8,
            "memory_limit": "5GB", "raster_processes": True,
            "plot_max_dimension": 1000, "plot_max_points": 50000,
            "raster_config": {
                "nodata": -9999.0, "compress": "DEFLATE", "blocksize": 512,
                "overview_levels": [2, 4, 8, 16, 32], "overview_resampling": "average",
                "max_grid_cells": 25000000,
            },
        }
        overrides = copy.deepcopy(param_lookup.get("raster_settings", {}))
        self.settings["raster_config"].update(overrides.pop("raster_config", {}))
        self.settings.update(overrides)
        if self.settings["raster_processes"] is not True:
            raise ValueError("Raster plotting requires process workers with one thread each")
        for key in ("raster_workers", "plot_max_dimension", "plot_max_points"):
            value = self.settings[key]
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            self.settings[key] = int(value)
        nodata = self.settings["raster_config"]["nodata"]
        if not np.isfinite(nodata) or float(np.float32(nodata)) != nodata:
            raise ValueError("NoData must be finite and exactly representable as float32")
        block = self.settings["raster_config"]["blocksize"]
        if not isinstance(block, int) or block < 128 or block % 16:
            raise ValueError("COG blocksize must be an integer >=128 and divisible by 16")

    def _resolve_paths(self, region: str) -> None:
        """Resolve local/S3 directories and assets for the given ecoregion."""

        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.write_message(f"MLRasterExportEngine resolved outputs_dir for {region}: {self.outputs_dir}", OUTPUTS)
        bucket = get_config_item("SHARED", "OUTPUT_BUCKET")
        s3_dir_base = f"s3://{bucket}/{self.output_prefix}/{region}" if self.output_prefix else f"s3://{bucket}/{region}"

        model_dir = get_config_item("MODEL", "XGB_OUTPUT_DIR")
        self.model_out_dir = UPath(f"{s3_dir_base}/{model_dir}") if self.is_aws else UPath(self.outputs_dir / model_dir)

        raster_dir = get_config_item("MODEL", "RASTER_OUTPUT_DIR")
        self.raster_out_dir = UPath(f"{s3_dir_base}/{raster_dir}") if self.is_aws else UPath(self.outputs_dir / raster_dir)

        template = get_config_item("MASK", "MASK_PRED_PATH")
        self.template_path = UPath(f"{s3_dir_base}/{template}") if self.is_aws else UPath(self.outputs_dir / template)

        tile_grid = get_config_item("MODEL", "SUBGRIDS")
        self.tile_grid_path = UPath(f"{s3_dir_base}/{tile_grid}") if self.is_aws else UPath(self.outputs_dir / tile_grid)

    def load_template_metadata(self) -> dict:
        """Read the template header once; workers never download the template raster."""

        path = str(self.template_path).replace("s3://", "/vsis3/", 1)
        with rasterio.open(path) as template:
            if template.crs != rasterio.crs.CRS.from_user_input(self.target_crs):
                raise ValueError("Template CRS must match the target CRS")
            return {"transform": template.transform, "width": template.width,
                    "height": template.height, "crs": str(template.crs), "source_uri": str(self.template_path)}

    def collect_training_files(self) -> list[UPath]:
        """Find spatial model summaries under the resolved modeling directory."""

        return sorted(self.model_out_dir.rglob("LOCAL_training_summary_*.geoparquet"), key=str)

    def _tile_mask_skip_reason(self, tile_geom_wkb: bytes) -> str | None:
        """Skip tiles without usable prediction-mask cells under the center rule.

        Valid prediction-mask pixels are unmasked, finite and nonzero.
        This eligibility check does not change the exported polygon mask.
        """

        geom = wkb.loads(tile_geom_wkb)
        metadata = self.template_metadata
        t = metadata["transform"]
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError("Use a north-up, unrotated prediction mask")
        raw = from_bounds(*geom.bounds, transform=t)
        c0 = max(0, math.floor(raw.col_off + 1e-8))
        r0 = max(0, math.floor(raw.row_off + 1e-8))
        c1 = min(metadata["width"], math.ceil(raw.col_off + raw.width - 1e-8))
        r1 = min(metadata["height"], math.ceil(raw.row_off + raw.height - 1e-8))
        height, width = r1 - r0, c1 - c0
        if height <= 0 or width <= 0:
            return "tile lies outside the prediction-mask raster extent"
        if height * width > self.settings["raster_config"]["max_grid_cells"]:
            raise ValueError(f"Prediction-mask overlap window is oversized ({height} x {width}); check tile bounds and max_grid_cells")
        window = Window(c0, r0, width, height)
        transform = window_transform(window, t)
        polygon_cells = geometry_mask([mapping(geom)], out_shape=(height, width),
                                      transform=transform, invert=True, all_touched=False)
        if not polygon_cells.any():
            return "no prediction-mask cell centers fall inside the tile polygon"
        path = str(self.template_path).replace("s3://", "/vsis3/", 1)
        with rasterio.open(path) as mask_raster:
            data = mask_raster.read(1, window=window, masked=True)
        valid = ~np.ma.getmaskarray(data) & np.isfinite(data.data) & (data.data != 0)
        if not np.any(polygon_cells & valid):
            return "tile contains no valid prediction-mask cells (valid = unmasked, finite and nonzero)"
        return None

    def load_tile_grid(self) -> gpd.GeoDataFrame:
        """Read the official tile polygons once per region in the target CRS."""

        uri = str(self.tile_grid_path)
        if Path(uri).suffix.lower() in (".parquet", ".geoparquet"):
            with self.tile_grid_path.open("rb") as stream:
                grid = gpd.read_parquet(stream)
        else:
            layer = get_config_item("MODEL", "PREDICTION_SUB_GRIDS_LAYER")
            options = {"layer": layer} if layer else {}
            with LocalizedFile(uri) as local:
                grid = gpd.read_file(local, **options)
        if grid.crs is None or self.settings["tile_id_column"] not in grid:
            raise ValueError("Tile grid needs a CRS and the configured tile ID column")
        return grid.to_crs(self.target_crs)

    def prepare_tasks(self, region: str, grid: gpd.GeoDataFrame) -> None:
        """Match model summary pairs to their official polygons and output folders."""

        self.task_configs = []
        seen = set()
        mask_skip_reasons = {}
        for training_file in self.collect_training_files():
            match = re.fullmatch(r"LOCAL_training_summary_(\d{4}_\d{4})\.geoparquet", training_file.name)
            if match is None:
                continue
            tile_id, pair = training_file.parent.parent.name, match[1]
            if (self.settings["tile_id"] and tile_id != self.settings["tile_id"]) or (self.settings["pair"] and pair != self.settings["pair"]):
                continue
            if (tile_id, pair) in seen:
                raise ValueError(f"Duplicate model summaries for {tile_id}, {pair}")
            if int(pair[:4]) >= int(pair[5:]):
                raise ValueError(f"Year pair must be increasing: {pair}")
            seen.add((tile_id, pair))
            selected = grid.loc[grid[self.settings["tile_id_column"]].astype(str) == tile_id]
            if len(selected) != 1:
                raise ValueError(f"Expected one official polygon for {tile_id}; found {len(selected)}")
            try:
                geom = selected.geometry.iloc[0]
                if geom is None:
                    raise ValueError("Missing geometry")
                if geom.is_empty:
                    raise ValueError("Empty geometry")
                if geom.geom_type not in ("Polygon", "MultiPolygon"):
                    raise ValueError(f"Expected Polygon or MultiPolygon; got {geom.geom_type}")
                if not geom.is_valid:
                    raise ValueError(explain_validity(geom))
            except (ValueError, ShapelyError) as exc:
                self.write_message(f"Skipping raster export for {region}/{tile_id}, pair {pair}: invalid official tile polygon ({exc}).", OUTPUTS)
                continue
            if tile_id not in mask_skip_reasons:
                mask_skip_reasons[tile_id] = self._tile_mask_skip_reason(geom.wkb)
            skip_reason = mask_skip_reasons[tile_id]
            if skip_reason is not None:
                self.write_message(f"Skipping raster export for {region}/{tile_id}, pair {pair}: {skip_reason}.", OUTPUTS)
                continue
            model_train = training_file.parent
            model_pred = self.model_out_dir / tile_id / "Prediction"
            train_out = self.raster_out_dir / tile_id / "Training"
            pred_out = self.raster_out_dir / tile_id / "Prediction"
            raster_paths = {
                role: {column: str(folder / f"{stem}_{pair}.tif") for column, stem in products.items()}
                for role, folder, products in (("training", train_out, TRAIN_PRODUCTS), ("prediction", pred_out, PRED_PRODUCTS))
            }
            self.task_configs.append({
                "region": region, "tile_id": tile_id, "pair": pair, "crs": str(self.target_crs),
                "training_summary": str(training_file),
                "prediction_summary": str(model_pred / f"LOCAL_prediction_summary_{pair}.geoparquet"),
                "model_metadata": str(model_train / f"LOCAL_model_metadata_{pair}.json"),
                "importance": str(model_train / f"LOCAL_importance_{pair}.parquet"),
                "shap_summary": str(model_train / f"LOCAL_shap_summary_{pair}.parquet"),
                "training_output_dir": str(train_out), "prediction_output_dir": str(pred_out),
                "plot_output_dir": str(train_out / "diagnostic_plots"), "raster_paths": raster_paths,
                "extrapolation_path": str(pred_out / f"LOCAL_extrapolation_diagnostics_{pair}.parquet"),
                "ranges_path": str(pred_out / f"LOCAL_predictor_ranges_{pair}.json"),
                "template_metadata": self.template_metadata, "geometry": geom.wkb,
                "legacy_survey_year_offset": self.settings["legacy_survey_year_offset"],
                "plot_max_dimension": self.settings["plot_max_dimension"],
                "plot_max_points": self.settings["plot_max_points"],
                "raster_config": self.settings["raster_config"], "log_dir": str(self.outputs_dir),
            })

    def execute_tasks(self, overwrite: bool = False) -> list[dict]:
        """Keep a bounded rolling queue of tile jobs across all process workers."""

        jobs = iter(dict(cfg, overwrite=overwrite) for cfg in self.task_configs)
        initial = list(islice(jobs, 2 * self.settings["raster_workers"]))
        pending = set(self.client.map(_export_tile_task, initial, pure=False))
        completed = as_completed(pending)
        records = []
        try:
            for future in completed:
                pending.remove(future)
                try:
                    records.append(future.result())
                finally:
                    future.release()
                job = next(jobs, None)
                if job is not None:
                    future = self.client.submit(_export_tile_task, job, pure=False)
                    pending.add(future)
                    completed.add(future)
        except Exception:
            self.client.cancel(list(pending))
            raise
        finally:
            for future in pending:
                future.release()
        return records

    def run(self, overwrite: bool = False) -> list[dict]:
        """Set up Dask, process ecoregions and always clean up shared resources."""

        records = []
        try:
            settings = self.settings
            self.setup_dask(self.param_lookup["env"], processes=settings["raster_processes"],
                            n_workers=settings["raster_workers"], threads_per_worker=1,
                            memory_limit=settings["memory_limit"])
            for eco_region in self.param_lookup["eco_regions"].value:
                self._resolve_paths(eco_region)
                self.template_metadata = self.load_template_metadata()
                grid = self.load_tile_grid()
                self.prepare_tasks(eco_region, grid)
                self.write_message(f"Queuing {len(self.task_configs)} raster exports for {eco_region}.", OUTPUTS)
                if self.task_configs:
                    records.extend(self.execute_tasks(overwrite))
            return records
        finally:
            self.cleanup_resources(OUTPUTS)
