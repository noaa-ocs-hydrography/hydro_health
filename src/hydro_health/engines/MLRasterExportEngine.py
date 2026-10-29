import copy
import csv
import textwrap
import gc
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from io import BytesIO
from itertools import islice
from urllib.parse import urlencode
from urllib.request import Request, urlopen
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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
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

REPORT_PRODUCTS = {
    "bathy_t": ("Starting bathymetry", "depth", "Bathymetry (m)"),
    "mean_predicted_bathy_t1": ("Mean predicted bathymetry", "depth", "Bathymetry (m)"),
    "delta_bathy": ("Surveyed bathymetric change", "change", "Change (m)"),
    "mean_predicted_change": ("Mean predicted change", "change", "Change (m)"),
    "uncertainty_sd_bathy_t1": ("Bootstrap uncertainty", "uncertainty", "Uncertainty (m)"),
    "mean_predicted_rate": ("Mean annualized change", "rate", "Change (m/year)"),
    "predictor_extrapolation_count": ("Predictors outside training range", "count", "Predictor count"),
    "UC_t": ("BlueTopo uncertainty", "uncertainty", "Uncertainty (m)"),
    "survey_year_bin_code": ("Survey end year", "survey", "Survey year"),
}


def _report_raster_path(uri: str) -> str:
    """Read remote GeoTIFFs by range requests rather than downloading full tiles."""

    return str(uri).replace("s3://", "/vsis3/", 1)


def _read_report_grid(uri: str, profile: dict, categorical: bool = False) -> np.ma.MaskedArray:
    """Read one tile onto the bounded report grid, preserving NoData coverage."""

    method = Resampling.mode if categorical else Resampling.average
    with rasterio.open(_report_raster_path(uri)) as source:
        if source.crs is None:
            raise ValueError("Source raster has no CRS")
        with WarpedVRT(source, crs=profile["crs"], transform=profile["transform"],
                       width=profile["width"], height=profile["height"], dtype="float32",
                       nodata=float("nan"), resampling=method) as overview:
            return np.ma.masked_invalid(overview.read(1, masked=True))


def _report_color_scale(family: str, limits: list[float]) -> tuple[object, Normalize]:
    """Use common units and one scale for all comparable mosaics in a region."""

    lo, hi = limits
    if family == "survey":
        return plt.get_cmap("viridis", len(SURVEY_LABELS)), BoundaryNorm(np.arange(-.5, len(SURVEY_LABELS) + .5), len(SURVEY_LABELS))
    if family in ("change", "rate"):
        bound = max(abs(lo), abs(hi), 1e-6)
        return cmocean.cm.curl, TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    if family in ("uncertainty", "count"):
        return plt.get_cmap("viridis"), Normalize(0, max(hi, 1e-6))
    return cmocean.cm.deep_r, Normalize(lo, hi if hi > lo else lo + 1)


def _fetch_report_basemap(bounds: tuple, crs: str, width: int, height: int, settings: dict) -> tuple[np.ndarray, tuple]:
    """Fetch one USGS map image per region, shared by every PDF page."""

    spatial = rasterio.crs.CRS.from_user_input(crs)
    reference = str(spatial.to_epsg()) if spatial.to_epsg() else json.dumps({"wkt": spatial.to_wkt()})
    params = {"bbox": ",".join(str(value) for value in bounds), "bboxSR": reference,
              "imageSR": reference, "size": f"{width},{height}", "format": "png32",
              "transparent": "false", "f": "json"}
    endpoint = settings["report_basemap_service"].rstrip("/") + "/export?" + urlencode(params)
    headers = {"User-Agent": "HydroHealth-mosaic-report/1.0"}
    with urlopen(Request(endpoint, headers=headers), timeout=settings["report_basemap_timeout"]) as response:
        info = json.load(response)
    if "error" in info or "href" not in info:
        raise ValueError(f"Basemap service returned: {info.get('error', info)}")
    with urlopen(Request(info["href"], headers=headers), timeout=settings["report_basemap_timeout"]) as response:
        pixels = plt.imread(BytesIO(response.read()), format="png")
    extent = info["extent"]
    return pixels, (extent["xmin"], extent["xmax"], extent["ymin"], extent["ymax"])


def _render_mosaic_page(pdf: PdfPages, entry: dict, grid: np.ma.MaskedArray, profile: dict, mask: np.ndarray, basemap: tuple | None, cmap: object, norm: Normalize, region: str, number: int, total: int) -> None:
    """Draw one labelled mosaic and auditable tile counts on each report page."""

    left, bottom, right, top = rasterio.transform.array_bounds(profile["height"], profile["width"], profile["transform"])
    extent = (left, right, bottom, top)
    title, family, units = REPORT_PRODUCTS[entry["column"]]
    fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
    try:
        fig.text(.055, .954, "HYDRO HEALTH  /  REGIONAL MOSAICS", color="#426078", fontsize=10, weight="bold")
        fig.text(.055, .903, title, color="#152f42", fontsize=23, weight="bold")
        fig.text(.055, .865, f"{region}  |  {entry['role'].title()}  |  {entry['pair'].replace('_', ' - ')}", fontsize=12, color="#426078")
        included, skipped = len(entry["included_tiles"]), len(entry["skipped_tiles"])
        fig.text(.055, .817, f"INCLUDED  {included}     SKIPPED  {skipped}     TOTAL  {included + skipped}",
                 fontsize=11, weight="bold", color="#17685b")
        reasons = Counter(item["category"] for item in entry["skipped_tiles"])
        summary = "; ".join(f"{label}: {count}" for label, count in sorted(reasons.items())) or "No tiles skipped"
        fig.text(.055, .787, textwrap.shorten(summary, width=140, placeholder=" ... (see export audit)"), fontsize=9, color="#52616c")
        ax = fig.add_axes([.065, .175, .77, .575])
        ax.set_facecolor("#e5edf1")
        if basemap is not None:
            ax.imshow(basemap[0], extent=basemap[1], origin="upper", zorder=0)
        image = ax.imshow(grid, extent=extent, origin="upper", cmap=cmap, norm=norm,
                          interpolation="nearest", alpha=.90 if basemap is not None else 1., zorder=2)
        if mask.any():
            # Padding also draws the boundary when the valid mask fills the image.
            t = profile["transform"]
            xs = t.c + (np.arange(-1, profile["width"] + 1) + .5) * t.a
            ys = t.f + (np.arange(-1, profile["height"] + 1) + .5) * t.e
            padded = np.pad(mask.astype("uint8"), 1)
            ax.contour(xs, ys, padded, levels=[.5], colors="white", linewidths=2.2, zorder=3)
            ax.contour(xs, ys, padded, levels=[.5], colors="#122c45", linewidths=.8, zorder=4)
        if not grid.count():
            ax.text(.5, .5, "No available tile data for this mosaic", transform=ax.transAxes,
                    ha="center", va="center", bbox=dict(facecolor="white", alpha=.9, edgecolor="none"))
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Easting (m)", fontsize=9)
        ax.set_ylabel("Northing (m)", fontsize=9)
        ax.set_aspect("equal")
        ax.legend(handles=[Line2D([0], [0], color="#122c45", lw=1.5, label="Prediction mask outline (overview)")],
                  loc="upper right", fontsize=8, framealpha=.95)
        cax = fig.add_axes([.87, .24, .018, .43])
        bar = fig.colorbar(image, cax=cax)
        bar.set_label(units, fontsize=10)
        bar.ax.tick_params(labelsize=9)
        if family == "survey":
            bar.set_ticks(range(len(SURVEY_LABELS)))
            bar.set_ticklabels(SURVEY_LABELS)
        if family == "count":
            bar.locator = MaxNLocator(integer=True)
            bar.update_ticks()
        fig.text(.055, .11, f"Mosaic: {Path(entry['styled_path']).name}", fontsize=8, color="#31495b")
        fig.text(.055, .085, f"{profile['crs']}  |  {profile['width']} x {profile['height']} pixels  |  Shared {family} scale across all pages", fontsize=8, color="#52616c")
        fig.text(.055, .06, f"Prior-run TIFFs included: {len(entry.get('previous_export_tiles', []))}. This run: see export audit pages. Full mosaic details: mosaic_inventory.json", fontsize=8, color="#52616c")
        attribution = "Map services and data available from U.S. Geological Survey, National Geospatial Program." if basemap is not None else "Basemap unavailable; see the export log."
        fig.text(.055, .032, attribution, fontsize=7, color="#52616c")
        fig.text(.945, .032, f"{number} / {total}", ha="right", fontsize=8, color="#52616c")
        pdf.savefig(fig, dpi=150)
    finally:
        plt.close(fig)



def _export_problem_category(reason: str) -> str:
    """Classify tile-level eligibility or raster diagnostics for report counts."""

    text = reason.lower()
    if "polygon" in text and any(word in text for word in ("invalid", "expected one", "missing geometry", "empty geometry")):
        return "Invalid tile polygon"
    if any(word in text for word in ("prediction-mask", "mask overlap", "mask cell centers")):
        return "No mask overlap"
    if any(word in text for word in ("no populated", "no usable", "no rows", "no row has", "no finite")):
        return "No usable raster data"
    if "missing" in text and any(word in text for word in ("column", "field")):
        return "Missing summary fields"
    if "outputs already exist" in text:
        return "Existing outputs"
    return "Raster export problem"


def _build_export_audit(candidates: dict, records: list[dict], prep_reasons: dict) -> dict:
    """Keep export outcomes separate from availability of older mosaic TIFFs."""

    outcomes = {(item["tile_id"], item["pair"]): item for item in records}
    tiles = []
    for pair, tile_ids in sorted(candidates.items()):
        for tile_id in sorted(tile_ids):
            outcome = outcomes.get((tile_id, pair))
            reason = prep_reasons.get((tile_id, pair), "")
            if outcome is not None and not outcome.get("skipped", False):
                status, category = "Exported", "Exported successfully"
                reason = ""
            elif outcome is not None or reason:
                status = "Skipped"
                reason = outcome.get("reason", reason) if outcome is not None else reason
                category = outcome.get("category", _export_problem_category(reason)) if outcome is not None else _export_problem_category(reason)
            else:
                status, category, reason = "Not run", "Prior outputs only", "No export job was discovered for this tile/year pair; existing TIFFs may still appear in mosaics."
            details = outcome.get("details", {}) if outcome is not None else {}
            tiles.append(dict(tile_id=tile_id, pair=pair, status=status, category=category, reason=reason, details=details))
    pairs = []
    for pair in sorted(candidates):
        selected = [item for item in tiles if item["pair"] == pair]
        counts = Counter(item["status"] for item in selected)
        categories = Counter(item["category"] for item in selected if item["status"] == "Skipped")
        issue_counts = Counter()
        for item in selected:
            issue_counts.update(set(issue["category"] for issue in item["details"].get("issues", [])))
        pairs.append(dict(pair=pair, total=len(selected), exported=counts["Exported"], skipped=counts["Skipped"],
                          not_run=counts["Not run"], skip_categories=dict(categories), input_issue_tiles=dict(issue_counts)))
    return dict(pairs=pairs, tiles=tiles)


def _render_export_audit(pdf: PdfPages, region: str, audit: dict) -> int:
    """Add export totals plus readable per-tile reasons ahead of the mosaic maps."""

    pages = 0
    for summary in audit["pairs"]:
        fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        try:
            fig.text(.055, .95, "HYDRO HEALTH  /  ML RASTER EXPORT AUDIT", fontsize=10, color="#426078", weight="bold")
            fig.text(.055, .89, f"{region}  |  {summary['pair'].replace('_', ' - ')}", fontsize=23, color="#152f42", weight="bold")
            fig.text(.055, .83, f"EXPORTED  {summary['exported']}     SKIPPED  {summary['skipped']}     NOT RUN  {summary['not_run']}     TOTAL  {summary['total']}", fontsize=12, color="#17685b", weight="bold")
            lines = ["WHY EXPORTS WERE SKIPPED"]
            lines.extend(f"{category}: {count} tile(s)" for category, count in sorted(summary["skip_categories"].items()))
            if not summary["skip_categories"]:
                lines.append("No export skips for this year pair.")
            if summary["input_issue_tiles"]:
                lines.extend(["", "INPUT PROBLEMS FOUND (one tile can have several problems)"])
                lines.extend(f"{category}: {count} tile(s)" for category, count in sorted(summary["input_issue_tiles"].items()))
            fig.text(.065, .76, "\n".join(lines), va="top", fontsize=min(11, 330 / max(1, len(lines) * 1.65)), linespacing=1.65, color="#31495b")
            fig.text(.055, .12, "These are this run's export outcomes. Mosaic pages separately count available TIFFs, including older outputs.", fontsize=9, color="#52616c")
            fig.text(.055, .085, "Exact paths, missing fields and data counts: export_audit.json and export_audit.csv.", fontsize=9, color="#52616c")
            pdf.savefig(fig)
            pages += 1
        finally:
            plt.close(fig)
        skipped = [item for item in audit["tiles"] if item["pair"] == summary["pair"] and item["status"] == "Skipped"]
        for offset in range(0, len(skipped), 4):
            fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
            try:
                fig.text(.055, .95, "HYDRO HEALTH  /  TILE EXPORT PROBLEMS", fontsize=10, color="#426078", weight="bold")
                fig.text(.055, .90, f"{region}  |  {summary['pair'].replace('_', ' - ')}", fontsize=20, color="#152f42", weight="bold")
                for index, item in enumerate(skipped[offset:offset + 4]):
                    y = .82 - index * .18
                    fig.text(.055, y, f"{item['tile_id']}  |  {item['category']}", fontsize=11, weight="bold", color="#17685b")
                    wrapped = textwrap.wrap(item["reason"].replace("\n", " "), width=143)
                    if len(wrapped) > 6:
                        wrapped = wrapped[:5] + ["... full details in export_audit.json / export_audit.csv"]
                    fig.text(.065, y - .025, "\n".join(wrapped), fontsize=8, va="top", linespacing=1.25, color="#31495b")
                fig.text(.055, .04, "Each problem was logged and skipped so other tiles could continue.", fontsize=8, color="#52616c")
                pdf.savefig(fig)
                pages += 1
            finally:
                plt.close(fig)
    return pages


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
    """Mean finite point values per cell and retain cells touched by the polygon.

    The official polygon determines extent, never the surviving data rows.
    The polygon mask uses all_touched=True to preserve narrow tile fragments.
    Template values/NoData are not used as a second mask during rasterization.
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
                                  transform=transform, invert=True, all_touched=True)
    diagnostics["polygon_mask_rule"] = "all_touched"
    diagnostics["polygon_touching_cells"] = int(valid_polygon.sum())
    diagnostics["populated_cells_before_polygon_mask"] = int(np.count_nonzero(grid != nodata))
    diagnostics["input_rows_in_polygon_touching_cells"] = int(valid_polygon[rows, cols].sum())
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
    """Report a structured tile diagnostic without failing the export queue."""

    def __init__(self, message: str, category: str | None = None, details: dict | None = None) -> None:
        super().__init__(message)
        self.category = category or _export_problem_category(message)
        self.details = details if details is not None else {}


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
                raise RasterExportDiagnostic(f"Rasterization failed for {role}/{column} on {cfg['tile_id']}: {exc}\nDiagnostics: {json.dumps(diagnostics, indent=2, default=str)}", details=diagnostics) from exc
            diagnostics.update(grid_diagnostics)
            if column not in OPTIONAL and not np.any(grid != profile["nodata"]):
                if not diagnostics["input_rows"]:
                    reason = "The summary contains no rows."
                elif not diagnostics["finite_xy_value_rows"]:
                    reason = "No row has finite numeric X, Y and value together; inspect the per-column invalid counts."
                elif not diagnostics["rows_inside_tile_grid"]:
                    reason = "All usable coordinates fall outside the tile grid; compare coordinate ranges, polygon bounds and CRS. The geometry CRS does not verify the X/Y columns."
                elif not diagnostics["polygon_touching_cells"]:
                    reason = "No template cells intersect the polygon; inspect polygon bounds and grid alignment."
                elif not diagnostics["input_rows_in_polygon_touching_cells"]:
                    reason = "All usable points map to cells removed by the polygon intersection mask; inspect point/tile overlap and grid alignment."
                else:
                    reason = "Points reach retained cells, but every aggregated output equals NoData; inspect source_nodata_rows and the value range."
                raise RasterExportDiagnostic(f"No populated {role}/{column} cells for {cfg['tile_id']}: {reason}\nDiagnostics: {json.dumps(diagnostics, indent=2, default=str)}", details=diagnostics)
            raw = Path(tmp) / "raw.tif"
            cog = Path(cfg["raster_paths"][role][column])
            raster_cfg = dict(cfg["raster_config"])
            profile.update(tiled=True, blockxsize=raster_cfg["blocksize"],
                           blockysize=raster_cfg["blocksize"], compress=raster_cfg["compress"])
            with rasterio.open(raw, "w", **profile) as dst:
                dst.write(grid, 1)
                dst.update_tags(value_column=column, pair=pair, role=role,
                    aggregation="mode" if column == "survey_year_bin_code" else "mean", polygon_mask_rule="all_touched",
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

def _load_summary(path: str, role: str, cfg: dict, predictors: list[str], audit: dict | None = None) -> pd.DataFrame:
    """Inspect all required fields and report row-level counts for this summary."""

    audit = audit if audit is not None else {}
    audit.update(path=path, role=role)
    with UPath(path).open("rb") as stream:
        spatial = gpd.read_parquet(stream)
    audit["crs"] = str(spatial.crs)
    df = pd.DataFrame(spatial.drop(columns=spatial.geometry.name)).reset_index(drop=True)
    audit["rows"] = len(df)
    errors = []
    if spatial.crs is None or spatial.crs != cfg["crs"]:
        errors.append(f"CRS is {spatial.crs}; expected {cfg['crs']}")
    if not df.columns.is_unique:
        errors.append("duplicate column names")
    if df.empty:
        errors.append("summary has zero rows")
    products = TRAIN_PRODUCTS if role == "training" else PRED_PRODUCTS
    required = (set(products) - OPTIONAL - {"predictor_extrapolation_count"}) | {"X", "Y", "FID"} | set(predictors)
    required |= {"bathy_t1"} if role == "training" else {"prediction_row_usable"}
    audit["missing_fields"] = sorted(required - set(df.columns))
    if audit["missing_fields"]:
        errors.append(f"missing fields: {audit['missing_fields']}")
    if not df.columns.is_unique:
        raise ValueError(f"{role} summary: " + "; ".join(errors))
    if "FID" in df:
        audit["missing_fids"] = int(df.FID.isna().sum())
        audit["duplicate_fids"] = int(df.FID.duplicated().sum())
        if audit["missing_fids"] or audit["duplicate_fids"]:
            errors.append(f"FID missing={audit['missing_fids']}, duplicate={audit['duplicate_fids']}")
    for column, expected in (("tile_id", cfg["tile_id"]), ("pair_id", cfg["pair"])):
        if column in df and not df[column].eq(expected).all():
            errors.append(f"{column} differs from expected {expected}")
    usable = np.ones(len(df), dtype=bool)
    if role == "prediction":
        if "prediction_row_usable" not in df:
            usable[:] = False
        else:
            invalid = ~df.prediction_row_usable.isin([True, False])
            if invalid.any():
                errors.append(f"prediction_row_usable has {int(invalid.sum())} nonboolean rows")
            usable = df.prediction_row_usable.eq(True).fillna(False).to_numpy(dtype=bool)
    audit["usable_rows"] = int(usable.sum())
    if not usable.any():
        errors.append("no usable rows")
    numeric = ["X", "Y", "bathy_t", "mean_predicted_bathy_t1", "mean_predicted_change", "mean_predicted_rate", "uncertainty_sd_bathy_t1"]
    audit["nonfinite_rows_by_field"] = {}
    for column in numeric:
        if column in df:
            values = pd.to_numeric(df.loc[usable, column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            count = int((~np.isfinite(values)).sum())
            if count:
                audit["nonfinite_rows_by_field"][column] = count
    if audit["nonfinite_rows_by_field"]:
        errors.append(f"usable rows with nonfinite/nonnumeric fields: {audit['nonfinite_rows_by_field']}")
    if errors:
        audit["problems"] = errors
        raise ValueError(f"{role} summary: " + "; ".join(errors))
    return df


def _load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Audit metadata and both summaries, collecting independent input failures."""

    checks, issues, frames = {}, [], {}
    metadata = None
    for key in ("model_metadata", "training_summary", "prediction_summary"):
        path = cfg[key]
        checks[key] = dict(path=path, exists=UPath(path).exists())
        if not checks[key]["exists"]:
            label = {"model_metadata": "model metadata", "training_summary": "training summary", "prediction_summary": "prediction summary"}[key]
            issues.append(dict(category=f"Missing {label}", message=f"Missing required {label}: {path}"))
    if checks["model_metadata"]["exists"]:
        try:
            metadata = json.loads(UPath(cfg["model_metadata"]).read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("Metadata JSON must contain an object")
            checks["model_metadata"]["expected"] = {key: cfg[key] for key in ("tile_id", "pair", "crs")}
            for key in ("tile_id", "pair", "crs"):
                if metadata.get(key) != cfg[key]:
                    issues.append(dict(category="Invalid model metadata", message=f"Metadata {key}: found {metadata.get(key)!r}; expected {cfg[key]!r}"))
            predictors = metadata.get("predictors")
            if not isinstance(predictors, list) or not all(isinstance(item, str) for item in predictors) or not predictors or "bathy_t" not in predictors or len(predictors) != len(set(predictors)):
                issues.append(dict(category="Invalid model metadata", message="Metadata needs a unique predictor-name list including bathy_t"))
            if metadata.get("target") != "bathy_t1" or metadata.get("prediction_baseline_source") != "bathy_t":
                issues.append(dict(category="Invalid model metadata", message=f"Expected target=bathy_t1 and prediction_baseline_source=bathy_t; found {metadata.get('target')!r}, {metadata.get('prediction_baseline_source')!r}"))
        except (OSError, ValueError, TypeError) as exc:
            issues.append(dict(category="Unreadable model metadata", message=f"{cfg['model_metadata']}: {exc}"))
            metadata = None
    predictors = metadata.get("predictors", []) if metadata is not None else []
    if not isinstance(predictors, list) or not all(isinstance(item, str) for item in predictors):
        predictors = []
    for role in ("training", "prediction"):
        key = f"{role}_summary"
        if not checks[key]["exists"]:
            continue
        try:
            frames[role] = _load_summary(cfg[key], role, cfg, predictors, checks[key])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            audit = checks[key]
            category = "Missing summary fields" if audit.get("missing_fields") else "Invalid summary"
            if not audit.get("missing_fields") and (audit.get("rows") == 0 or audit.get("usable_rows") == 0 or audit.get("nonfinite_rows_by_field")):
                category = "No usable ML data"
            issues.append(dict(category=category, message=f"{cfg[key]}: {exc}"))
        count_key = "n_fit" if role == "training" else "n_prediction_rows"
        if metadata is not None and count_key in metadata and "rows" in checks[key] and checks[key]["rows"] != metadata[count_key]:
            issues.append(dict(category="Row count mismatch", message=f"{role} summary rows={checks[key]['rows']}; metadata {count_key}={metadata[count_key]}"))
    if issues:
        category = issues[0]["category"] if len({item["category"] for item in issues}) == 1 else "Multiple ML input problems"
        reason = "Model input checks failed: " + " | ".join(item["message"] for item in issues)
        raise RasterExportDiagnostic(reason, category=category, details=dict(input_checks=checks, issues=issues))
    cfg["_input_audit"] = checks
    train, prediction = frames["training"], frames["prediction"]
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
        raise RasterExportDiagnostic(f"Outputs already exist for {cfg['tile_id']}, {cfg['pair']}; overwrite=False")
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
            raise RasterExportDiagnostic(f"Outputs already exist for {cfg['tile_id']}, {cfg['pair']}; overwrite=False")
        try:
            train, prediction, ranges = _load_inputs(cfg)
        except RasterExportDiagnostic:
            raise
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise RasterExportDiagnostic(f"Incomplete or invalid model inputs: {type(exc).__name__}: {exc}") from exc
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
                    "status": "Exported", "details": {"input_checks": cfg.get("_input_audit", {}),
                    "optional_outputs_not_supplied": {role: result.get("skipped", []) for role, result in results.items()}},
                    "results": results, "visual_outputs": [published[path] for path in visuals]}
    except RasterExportDiagnostic as exc:
        Engine.write_message_dask(f"Skipping raster export for {cfg['region']}/{cfg['tile_id']} | {cfg['pair']}: {exc}", cfg["log_dir"])
        return {"region": cfg["region"], "tile_id": cfg["tile_id"], "pair": cfg["pair"],
                "skipped": True, "status": "Skipped", "category": exc.category, "details": exc.details,
                "reason": str(exc), "results": {}, "visual_outputs": []}
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
            "mosaic_report": True, "report_max_dimension": 1600,
            "report_basemap_service": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer",
            "report_basemap_timeout": 20,
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
        for key in ("raster_workers", "plot_max_dimension", "plot_max_points", "report_max_dimension", "report_basemap_timeout"):
            value = self.settings[key]
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(f"{key} must be a positive integer")
            self.settings[key] = int(value)
        if self.settings["report_max_dimension"] > 2048:
            raise ValueError("report_max_dimension must be <=2048 to bound report memory and basemap requests")
        self.mosaic_reports = []
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
        """Resolve expected training paths from any discovered ML handoff artifact."""

        expected = set()
        for stem, extension in (("training_summary", "geoparquet"), ("prediction_summary", "geoparquet"), ("model_metadata", "json")):
            for path in self.model_out_dir.rglob(f"LOCAL_{stem}_*.{extension}"):
                match = re.fullmatch(r"LOCAL_" + stem + r"_(\d{4}_\d{4})\." + extension, path.name)
                if match is not None:
                    tile_root = path.parent.parent
                    expected.add(str(tile_root / "Training" / f"LOCAL_training_summary_{match[1]}.geoparquet"))
        return [UPath(path) for path in sorted(expected)]

    def _tile_mask_skip_reason(self, tile_geom_wkb: bytes) -> str | None:
        """Skip tiles without a valid prediction-mask pixel touched by the polygon.

        Valid prediction-mask pixels are unmasked, finite and nonzero.
        Eligibility and export both use all_touched=True for the polygon mask.
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
                                      transform=transform, invert=True, all_touched=True)
        if not polygon_cells.any():
            return "no prediction-mask pixels intersect the tile polygon"
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
        self.report_candidates = {}
        self.report_skip_reasons = {}
        for training_file in self.collect_training_files():
            match = re.fullmatch(r"LOCAL_training_summary_(\d{4}_\d{4})\.geoparquet", training_file.name)
            if match is None:
                continue
            tile_id, pair = training_file.parent.parent.name, match[1]
            if (self.settings["tile_id"] and tile_id != self.settings["tile_id"]) or (self.settings["pair"] and pair != self.settings["pair"]):
                continue
            self.report_candidates.setdefault(pair, set()).add(tile_id)
            if (tile_id, pair) in seen:
                raise ValueError(f"Duplicate model summaries for {tile_id}, {pair}")
            if int(pair[:4]) >= int(pair[5:]):
                raise ValueError(f"Year pair must be increasing: {pair}")
            seen.add((tile_id, pair))
            selected = grid.loc[grid[self.settings["tile_id_column"]].astype(str) == tile_id]
            if len(selected) != 1:
                reason = f"Expected one official polygon for {tile_id}; found {len(selected)}"
                self.report_skip_reasons[tile_id, pair] = reason
                self.write_message(f"Skipping raster export for {region}/{tile_id}, pair {pair}: {reason}.", OUTPUTS)
                continue
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
                self.report_skip_reasons[tile_id, pair] = f"Invalid official tile polygon: {exc}"
                self.write_message(f"Skipping raster export for {region}/{tile_id}, pair {pair}: invalid official tile polygon ({exc}).", OUTPUTS)
                continue
            if tile_id not in mask_skip_reasons:
                mask_skip_reasons[tile_id] = self._tile_mask_skip_reason(geom.wkb)
            skip_reason = mask_skip_reasons[tile_id]
            if skip_reason is not None:
                self.report_skip_reasons[tile_id, pair] = skip_reason
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

    def generate_mosaic_report(self, region: str, records: list[dict]) -> str | None:
        """Build all regional previews after exports, including existing tile TIFFs."""

        if not self.settings["mosaic_report"]:
            return None
        self.write_message(f"Building regional mosaic report for {region}.", OUTPUTS)
        candidates = {pair: set(tiles) for pair, tiles in self.report_candidates.items()}
        available = {}
        # List each role once, avoiding an S3 existence request per product/tile.
        for role, products in (("training", TRAIN_PRODUCTS), ("prediction", PRED_PRODUCTS)):
            folder = "Training" if role == "training" else "Prediction"
            for path in self.raster_out_dir.glob(f"*/{folder}/*.tif"):
                tile_id = path.parent.parent.name
                if self.settings["tile_id"] and tile_id != self.settings["tile_id"]:
                    continue
                for column, stem in products.items():
                    match = re.fullmatch(re.escape(stem) + r"_(\d{4}_\d{4})\.tif", path.name)
                    if match is None:
                        continue
                    pair = match[1]
                    if self.settings["pair"] and pair != self.settings["pair"]:
                        continue
                    candidates.setdefault(pair, set()).add(tile_id)
                    available[tile_id, pair, role, column] = str(path)
                    break
        if not candidates:
            self.write_message(f"Skipping regional mosaic report for {region}: no tile/year pairs found.", OUTPUTS)
            return None
        meta = self.template_metadata
        factor = max(1., max(meta["height"], meta["width"]) / self.settings["report_max_dimension"])
        height, width = max(1, math.ceil(meta["height"] / factor)), max(1, math.ceil(meta["width"] / factor))
        transform = meta["transform"] * Affine.scale(meta["width"] / width, meta["height"] / height)
        profile = dict(driver="GTiff", count=1, dtype="float32", crs=meta["crs"], transform=transform,
                       width=width, height=height, nodata=-9999.0, compress="DEFLATE", tiled=True,
                       blockxsize=256, blockysize=256)
        bounds = rasterio.transform.array_bounds(height, width, transform)
        # Read the mask directly at the bounded overview size using categorical sampling.
        with rasterio.open(_report_raster_path(self.template_path)) as source:
            mask_data = source.read(1, out_shape=(height, width), masked=True, resampling=Resampling.nearest)
        mask = ~np.ma.getmaskarray(mask_data) & np.isfinite(mask_data.data) & (mask_data.data != 0)
        basemap = None
        try:
            basemap = _fetch_report_basemap(bounds, profile["crs"], width, height, self.settings)
        except (OSError, ValueError, KeyError) as exc:
            self.write_message(f"Basemap unavailable for {region} mosaic report: {exc}. Report pages will be labelled accordingly.", OUTPUTS)
        outcomes = {(record["tile_id"], record["pair"]): record for record in records}
        export_audit = _build_export_audit(candidates, records, self.report_skip_reasons)
        audit_by_tile = {(item["tile_id"], item["pair"]): item for item in export_audit["tiles"]}
        destination = self.raster_out_dir / "mosaic_report"
        entries, ranges = [], {}
        with tempfile.TemporaryDirectory(prefix="hh_mosaic_report_") as tmp:
            root = Path(tmp)
            for pair, tiles in sorted(candidates.items()):
                for role, products in (("training", TRAIN_PRODUCTS), ("prediction", PRED_PRODUCTS)):
                    for column in products:
                        title, family, units = REPORT_PRODUCTS[column]
                        name = f"MOSAIC_{role}_{column}_{pair}"
                        entry = dict(pair=pair, role=role, column=column, included_tiles=[], skipped_tiles=[],
                                     data_path=f"{name}.tif", styled_path=f"{name}_colored.tif")
                        mosaic = np.ma.masked_all((height, width), dtype="float32")
                        for tile_id in sorted(tiles):
                            uri = available.get((tile_id, pair, role, column))
                            if uri is None:
                                folder = "Training" if role == "training" else "Prediction"
                                expected_path = str(self.raster_out_dir / tile_id / folder / f"{products[column]}_{pair}.tif")
                                if UPath(expected_path).exists():
                                    uri = expected_path
                                    self.write_message(f"Mosaic listing missed an existing TIFF; reading exact path: {uri}", OUTPUTS)
                                else:
                                    outcome = outcomes.get((tile_id, pair), {})
                                    audit_item = audit_by_tile[tile_id, pair]
                                    optional = outcome.get("results", {}).get(role, {}).get("skipped", [])
                                    omitted = next((item for item in optional if item.get("column") == column), None)
                                    if omitted is not None:
                                        category, reason = "Optional data not supplied", omitted["reason"]
                                    elif audit_item["status"] == "Skipped":
                                        category, reason = audit_item["category"], audit_item["reason"]
                                    elif audit_item["status"] == "Exported":
                                        category, reason = "Missing output after export", "Export reported success but this TIFF is absent."
                                    else:
                                        category, reason = "Missing output", "No current export job or output TIFF was found."
                                    reason = f"{reason} Expected TIFF: {expected_path}"
                                    entry["skipped_tiles"].append(dict(tile_id=tile_id, category=category, reason=reason))
                                    continue
                            try:
                                data = _read_report_grid(uri, profile, family in ("survey", "count"))
                            except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
                                entry["skipped_tiles"].append(dict(tile_id=tile_id, category="Read error", reason=f"{uri}: {exc}"))
                                self.write_message(f"Skipping {region}/{tile_id} from {name}: {exc}", OUTPUTS)
                                continue
                            usable = ~np.ma.getmaskarray(data)
                            if not usable.any():
                                entry["skipped_tiles"].append(dict(tile_id=tile_id, category="No overview data", reason="No finite cells remain at report resolution."))
                                continue
                            # A mosaic uses the first valid tile in sorted ID order at overlaps.
                            take = usable & np.ma.getmaskarray(mosaic)
                            if not take.any():
                                entry["skipped_tiles"].append(dict(tile_id=tile_id, category="Covered by earlier tiles", reason="All overview cells are already covered by an earlier tile."))
                                continue
                            mosaic.data[take] = data.data[take]
                            mosaic.mask[take] = False
                            entry["included_tiles"].append(tile_id)
                        entry["previous_export_tiles"] = [tile for tile in entry["included_tiles"] if audit_by_tile[tile, pair]["status"] != "Exported"]
                        entry["included_count"] = len(entry["included_tiles"])
                        entry["skipped_count"] = len(entry["skipped_tiles"])
                        entry["total_count"] = len(tiles)
                        if mosaic.count():
                            lo, hi = float(mosaic.min()), float(mosaic.max())
                            previous = ranges.get(family, [lo, hi])
                            ranges[family] = [min(previous[0], lo), max(previous[1], hi)]
                        with rasterio.open(root / entry["data_path"], "w", **profile) as output:
                            output.write(mosaic.filled(profile["nodata"]), 1)
                            output.update_tags(region=region, pair=pair, role=role, value_column=column,
                                               included_tiles=entry["included_count"], skipped_tiles=entry["skipped_count"],
                                               resampling="mode" if family in ("survey", "count") else "average",
                                               overlap_rule="first valid tile in sorted tile ID order")
                        entries.append(entry)
                        self.write_message(f"Mosaic {region}/{name}: included {entry['included_count']}, skipped {entry['skipped_count']} of {entry['total_count']} tiles.", OUTPUTS)
            report_name = f"{region}_mosaic_report.pdf"
            scale_metadata = {}
            with PdfPages(root / report_name) as pdf:
                pdf.infodict().update(Title=f"{region} model output mosaics", Author="Hydro Health")
                audit_pages = _render_export_audit(pdf, region, export_audit)
                for number, entry in enumerate(entries, 1):
                    family = REPORT_PRODUCTS[entry["column"]][1]
                    cmap, norm = _report_color_scale(family, ranges.get(family, [0., 1.]))
                    scale_metadata[family] = dict(min=float(norm.vmin), max=float(norm.vmax), cmap=cmap.name)
                    if family == "survey":
                        scale_metadata[family]["classes"] = SURVEY_LABELS
                    with rasterio.open(root / entry["data_path"], "r+") as source:
                        grid = source.read(1, masked=True)
                        source.update_tags(display_min=norm.vmin, display_max=norm.vmax, display_cmap=cmap.name,
                                           display_note="Use the matching _colored.tif for fixed RGB rendering")
                    rgba = cmap(norm(grid.filled(0)), bytes=True)
                    rgba[np.ma.getmaskarray(grid)] = 0
                    color_profile = dict(profile, count=4, dtype="uint8", nodata=None, photometric="RGB", alpha="YES")
                    with rasterio.open(root / entry["styled_path"], "w", **color_profile) as output:
                        output.write(np.moveaxis(rgba, -1, 0))
                        output.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
                        output.update_tags(display_min=norm.vmin, display_max=norm.vmax, display_cmap=cmap.name,
                                           source_values=entry["data_path"])
                    _render_mosaic_page(pdf, entry, grid, profile, mask, basemap, cmap, norm, region, number + audit_pages, len(entries) + audit_pages)
            inventory = dict(region=region, mask_uri=str(self.template_path),
                             counts_note="Per mosaic: included tiles contribute finite overview cells; skipped tiles are missing, unreadable or contribute no cells. Existing exports count as included.",
                             mask_validity="unmasked, finite, nonzero", mask_outline="nearest sampled at report resolution",
                             scale_note="Shared scales use all finite numeric mosaic values in each unit family across this region's roles and year pairs.",
                             basemap_available=basemap is not None, export_summary=export_audit["pairs"], color_scales=scale_metadata, mosaics=entries)
            (root / "mosaic_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            (root / "export_audit.json").write_text(json.dumps(export_audit, indent=2, default=str), encoding="utf-8")
            with (root / "export_audit.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_id", "pair", "status", "category", "reason", "input_checks"])
                writer.writeheader()
                for tile in export_audit["tiles"]:
                    writer.writerow({key: tile[key] for key in ("tile_id", "pair", "status", "category", "reason")} | {"input_checks": json.dumps(tile["details"], default=str)})
            # Publish the PDF last, after its accompanying TIFFs and inventory succeed.
            files = sorted(path for path in root.iterdir() if path.name != report_name)
            files.append(root / report_name)
            for path in files:
                _copy_file(str(path), str(destination / path.name))
        result = str(destination / report_name)
        self.write_message(f"Successfully generated mosaic report for {region}: {result} ({len(entries) + audit_pages} pages).", OUTPUTS)
        return result


    def execute_tasks(self, overwrite: bool = True) -> list[dict]:
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

    def run(self, overwrite: bool = True) -> list[dict]:
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
                region_records = self.execute_tasks(overwrite) if self.task_configs else []
                records.extend(region_records)
                try:
                    report = self.generate_mosaic_report(eco_region, region_records)
                    if report is not None:
                        self.mosaic_reports.append(report)
                except Exception as exc:
                    self.write_message(f"Mosaic report failed for {eco_region}: {type(exc).__name__}: {exc}", OUTPUTS)
            return records
        finally:
            self.cleanup_resources(OUTPUTS)
