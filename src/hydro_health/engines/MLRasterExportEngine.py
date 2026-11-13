import csv
import textwrap
import traceback
import gc
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter, deque
from io import BytesIO
from collections.abc import Iterator
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
from dask.utils import SerializableLock

from matplotlib.collections import PolyCollection
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.warp import transform_bounds
from rasterio.warp import transform as transform_coordinates
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


_PLOT_LOCK = SerializableLock("hydro_health_raster_plotting")

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
OPTIONAL = {"UC_t", "survey_year_bin_code", "predictor_extrapolation_count"}
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


def _probe_report_gaps(source, data: np.ma.MaskedArray, profile: dict, limit: int = 32) -> list[tuple]:
    """Compare blank overview cells with native centers and their 3x3 neighborhoods."""

    valid = ~np.ma.getmaskarray(data)
    adjacent = np.zeros(valid.shape, dtype=bool)
    adjacent[1:, :] |= valid[:-1, :]
    adjacent[:-1, :] |= valid[1:, :]
    adjacent[:, 1:] |= valid[:, :-1]
    adjacent[:, :-1] |= valid[:, 1:]
    rows, cols = np.where(adjacent & ~valid)
    if not len(rows):
        return []
    xs, ys = rasterio.transform.xy(profile["transform"], rows, cols)
    xs, ys = transform_coordinates(profile["crs"], source.crs, xs, ys)
    native_rows, native_cols = rasterio.transform.rowcol(source.transform, xs, ys)
    inside = [i for i, (r, c) in enumerate(zip(native_rows, native_cols))
              if 0 <= r < source.height and 0 <= c < source.width]
    if not inside:
        return []
    selected = np.linspace(0, len(inside) - 1, min(limit, len(inside)), dtype=int)
    probes = []
    for index in selected:
        i = inside[index]
        pixel = source.read(1, window=Window(int(native_cols[i]), int(native_rows[i]), 1, 1), masked=True)
        native_valid = bool(np.ma.masked_invalid(pixel).count())
        r, c = int(native_rows[i]), int(native_cols[i])
        r0, c0 = max(0, r - 1), max(0, c - 1)
        neighborhood = source.read(1, window=Window(c0, r0, min(source.width, c + 2) - c0,
                                                   min(source.height, r + 2) - r0), masked=True)
        nearby_valid = bool(np.ma.masked_invalid(neighborhood).count())
        edge = min(r, c, source.height - 1 - r, source.width - 1 - c) <= 1
        probes.append((int(rows[i]), int(cols[i]), native_valid, nearby_valid, edge))
    return probes


def _read_dataset_coverage(uri: str, profile: dict) -> tuple:
    """Read only the tile-sized preview grid; respect source band NoData and masks."""

    try:
        with rasterio.open(_report_raster_path(uri)) as source:
            if source.crs is None:
                return None, "Missing CRS", None, None
            left, bottom, right, top = transform_bounds(source.crs, profile["crs"], *source.bounds, densify_pts=21)
            x0, y0, x1, y1 = profile["bounds"]
            if right <= x0 or left >= x1 or top <= y0 or bottom >= y1:
                return None, "Outside tile", None, None
            with WarpedVRT(source, crs=profile["crs"], transform=profile["transform"],
                           width=profile["width"], height=profile["height"], dtype="float32",
                           nodata=float("nan"), resampling=Resampling.nearest) as vrt:
                values = np.ma.masked_invalid(vrt.read(1, masked=True))
            valid = ~np.ma.getmaskarray(values)
            valid &= values.filled(-9999) != -9999
            if source.crs.is_projected:
                factor = source.crs.linear_units_factor[1]
                pixel_note = f"{source.res[0]*factor:.4g} x {source.res[1]*factor:.4g} m native"
            else:
                # Geographic pixels: estimate ground size locally in the tile projection.
                cx, cy = (x0+x1)/2, (y0+y1)/2
                sx, sy = transform_coordinates(profile["crs"], source.crs, [cx], [cy])
                xx, yy = transform_coordinates(source.crs, profile["crs"],
                    [sx[0], sx[0]+source.transform.a, sx[0]+source.transform.b],
                    [sy[0], sy[0]+source.transform.d, sy[0]+source.transform.e])
                factor = rasterio.crs.CRS.from_user_input(profile["crs"]).linear_units_factor[1]
                pixel_note = f"~{math.hypot(xx[1]-xx[0], yy[1]-yy[0])*factor:.4g} x {math.hypot(xx[2]-xx[0], yy[2]-yy[0])*factor:.4g} m locally"
            return valid, None, pixel_note, values
    except Exception as exc:
        return None, f"Read error: {type(exc).__name__}", None, None


def _render_coverage_year_thumbnails(fig, year_grids: dict, profile: dict, colors: dict, depth_limits: tuple) -> None:
    """Fit all overlapping survey years on the right of the coverage page."""

    years = sorted(year_grids)
    if not years:
        fig.text(.56, .55, "No yearly mosaic data available", fontsize=10)
        return
    ncols = max(1, math.ceil(math.sqrt(len(years) * .8)))
    nrows = math.ceil(len(years) / ncols)
    layout = fig.add_gridspec(nrows, ncols, left=.55, right=.97, bottom=.27, top=.86, wspace=.12, hspace=.25)
    left, bottom, right, top = profile["bounds"]
    artist = None
    for i, year in enumerate(years):
        ax = fig.add_subplot(layout[i // ncols, i % ncols])
        artist = ax.imshow(year_grids[year], extent=(left, right, bottom, top), origin="upper",
                           cmap="viridis", vmin=depth_limits[0], vmax=depth_limits[1], interpolation="nearest")
        ax.set_title(year, color=colors[year], fontsize=max(6, min(10, 24 / nrows)), fontweight="bold", pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    cax = fig.add_axes([.60, .225, .32, .015])
    bar = fig.colorbar(artist, cax=cax, orientation="horizontal", extend="both")
    bar.set_label("Survey mosaic bathymetry (m) — shared scale", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.text(.55, .145, "One map per survey year; white = no valid coverage.\nSame-year overlaps: first valid dataset in filename order.\nValues outside the depth scale use the end colors.", fontsize=7, va="top")


def _check_native_raster_values(source) -> None:
    """Reject empty or all-zero TIFFs using native blocks; stop at the first finite nonzero value."""

    has_valid = False
    for _, window in source.block_windows(1):
        values = np.ma.masked_invalid(source.read(1, window=window, masked=True)).compressed()
        if values.size:
            has_valid = True
            if np.any(values != 0):
                return
    if has_valid:
        raise ValueError("All-zero TIFF: all finite, unmasked source values are zero")
    raise ValueError("No valid raster values: all source cells are NoData, NaN or infinite")


def _read_report_grid(uri: str, profile: dict, categorical: bool = False) -> tuple:
    """Read one tile onto the bounded report grid, preserving NoData coverage."""

    method = Resampling.mode if categorical else Resampling.average
    with rasterio.open(_report_raster_path(uri), OVERVIEW_LEVEL="NONE") as source:
        if source.crs is None:
            raise ValueError("Source raster has no CRS")
        _check_native_raster_values(source)
        with WarpedVRT(source, crs=profile["crs"], transform=profile["transform"],
                       width=profile["width"], height=profile["height"], dtype="float32",
                       nodata=float("nan"), resampling=method) as overview:
            data = np.ma.masked_invalid(overview.read(1, masked=True))
        # Recover only blank cells whose exact map-center native pixel is valid.
        # This is source sampling, not interpolation or neighborhood gap filling.
        rows, cols = np.where(np.ma.getmaskarray(data))
        # Restrict candidates before coordinate transformation to the TIFF bounds.
        bounds = transform_bounds(source.crs, profile["crs"], *source.bounds, densify_pts=21)
        xs, ys = rasterio.transform.xy(profile["transform"], rows, cols)
        xs, ys = np.asarray(xs), np.asarray(ys)
        inside = (xs >= bounds[0]) & (xs < bounds[2]) & (ys > bounds[1]) & (ys <= bounds[3])
        rows, cols, xs, ys = rows[inside], cols[inside], xs[inside], ys[inside]
        restored = 0
        for start in range(0, len(rows), 4096):
            stop = start + 4096
            sx, sy = transform_coordinates(profile["crs"], source.crs, xs[start:stop].tolist(), ys[start:stop].tolist())
            samples = np.ma.vstack(list(source.sample(zip(sx, sy), indexes=1, masked=True)))[:, 0]
            samples = np.ma.masked_invalid(samples)
            valid = ~np.ma.getmaskarray(samples)
            rr, cc = rows[start:stop][valid], cols[start:stop][valid]
            data.data[rr, cc] = samples.data[valid]
            data.mask[rr, cc] = False
            restored += int(valid.sum())
        try:
            probes = _probe_report_gaps(source, data, profile)
            probe_error = None
        except Exception as exc:
            probes, probe_error = [], f"{type(exc).__name__}: {exc}"
        return data, probes, probe_error, restored



def _read_report_tile(tile_id: str, uri: str | None, expected_path: str, profile: dict, categorical: bool) -> tuple:
    """Resolve and read one independent TIFF; return failures for per-tile reporting."""

    try:
        if uri is None:
            if not UPath(expected_path).exists():
                return tile_id, None, None, None
            uri = expected_path
        return tile_id, uri, _read_report_grid(uri, profile, categorical), None
    except (OSError, ValueError, rasterio.errors.RasterioError) as exc:
        return tile_id, uri or expected_path, None, exc


def _iter_report_tiles(jobs: Iterator[tuple], profile: dict, categorical: bool, client, limit: int) -> Iterator[tuple]:
    """Use the shared Dask workers, bounding reads and preserving tile overlap order."""

    jobs = iter(jobs)
    pending = deque()
    try:
        for _ in range(limit):
            job = next(jobs, None)
            if job is None:
                break
            pending.append(client.submit(_read_report_tile, *job, profile, categorical, pure=False))
        while pending:
            future = pending.popleft()
            try:
                result = future.result()
            finally:
                future.release()
            yield result
            del result
            job = next(jobs, None)
            if job is not None:
                pending.append(client.submit(_read_report_tile, *job, profile, categorical, pure=False))
    finally:
        if pending:
            try:
                client.cancel(list(pending))
            finally:
                for future in pending:
                    future.release()


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
    """Fetch a quiet gray basemap and separate city labels once per region."""

    spatial = rasterio.crs.CRS.from_user_input(crs)
    reference = str(spatial.to_epsg()) if spatial.to_epsg() else json.dumps({"wkt": spatial.to_wkt()})
    params = {"bbox": ",".join(str(value) for value in bounds), "bboxSR": reference,
              "imageSR": reference, "size": f"{width},{height}", "format": "png32",
              "transparent": "false", "f": "json"}
    headers = {"User-Agent": "HydroHealth-mosaic-report/1.0"}

    def fetch(service, transparent):
        query = dict(params, transparent="true" if transparent else "false")
        endpoint = service.rstrip("/") + "/export?" + urlencode(query)
        with urlopen(Request(endpoint, headers=headers), timeout=settings["report_basemap_timeout"]) as response:
            info = json.load(response)
        if "error" in info or "href" not in info:
            raise ValueError(f"Basemap service returned: {info.get('error', info)}")
        with urlopen(Request(info["href"], headers=headers), timeout=settings["report_basemap_timeout"]) as response:
            pixels = plt.imread(BytesIO(response.read()), format="png")
        extent = info["extent"]
        return pixels, (extent["xmin"], extent["xmax"], extent["ymin"], extent["ymax"])

    pixels, extent = fetch(settings["report_basemap_service"], False)
    labels = fetch(settings["report_basemap_reference_service"], True)
    return pixels, extent, labels



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
        reasons = _mosaic_skip_counts(entry)
        summary = "; ".join(f"{label}: {count}" for label, count in sorted(reasons.items())) or "No tiles skipped"
        fig.text(.055, .787, textwrap.shorten(summary, width=140, placeholder=" ... (see variable count summary)"), fontsize=9, color="#52616c")
        ax = fig.add_axes([.065, .175, .77, .575])
        ax.set_facecolor("#e5edf1")
        if basemap is not None:
            ax.imshow(basemap[0], extent=basemap[1], origin="upper", zorder=0)
            ax.imshow(basemap[2][0], extent=basemap[2][1], origin="upper", zorder=4)
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
        bar = fig.colorbar(image, cax=cax, extend="neither" if family in ("survey", "count") else "both")
        bar.set_label(units, fontsize=10)
        bar.ax.tick_params(labelsize=9)
        if family == "survey":
            bar.set_ticks(range(len(SURVEY_LABELS)))
            bar.set_ticklabels(SURVEY_LABELS)
        if family == "count":
            bar.locator = MaxNLocator(integer=True)
            bar.update_ticks()
        stats = entry.get("value_summary")
        if stats:
            value_caption = (f"Values ({units}): min {stats['min']:.4g} | P02 {stats['p02']:.4g} | "
                             f"median {stats['median']:.4g} | P98 {stats['p98']:.4g} | max {stats['max']:.4g}")
            zero_caption = (f"Zero cells: {stats['zero_cells']:,} / {stats['valid_cells']:,} valid "
                            f"({100 * stats['zero_cells'] / max(stats['valid_cells'], 1):.2f}%). "
                            "Statistics describe the report-resolution mosaic; P02/P98 are the 2nd/98th percentiles.")
        else:
            value_caption = "Value check: no valid mosaic cells; minimum, percentiles, median and maximum unavailable."
            zero_caption = "Zero cells: 0 / 0 valid."
        fig.text(.055, .112, value_caption, fontsize=7.5, color="#31495b")
        fig.text(.055, .094, zero_caption, fontsize=7, color="#52616c")
        valid_values = grid.compressed()
        outside = "" if family in ("survey", "count") else f" | Below/above scale: {np.count_nonzero(valid_values < norm.vmin):,}/{np.count_nonzero(valid_values > norm.vmax):,}"
        scale_note = "full range" if family in ("survey", "count") else "P02–P98 envelope; symmetric changes; uncertainty starts at 0"
        fig.text(.055, .072, f"Shared scale: {norm.vmin:.4g} to {norm.vmax:.4g} ({scale_note}){outside}", fontsize=7, color="#52616c")
        fig.text(.055, .050, f"Prior-run TIFFs included: {len(entry.get('previous_export_tiles', []))}. See the variable summary for availability counts.", fontsize=8, color="#52616c")
        attribution = "Basemap: Esri Light Gray Canvas | Sources: Esri, HERE, Garmin, OpenStreetMap contributors." if basemap is not None else "Basemap unavailable; see the export log."
        fig.text(.055, .025, attribution, fontsize=7, color="#52616c")
        fig.text(.945, .025, f"{number} / {total}", ha="right", fontsize=8, color="#52616c")
        pdf.savefig(fig, dpi=150)
    finally:
        plt.close(fig)



def _mosaic_skip_counts(entry: dict) -> Counter:
    """Count one reason per skipped tile without putting tile IDs or paths in the PDF."""

    counts = Counter()
    for item in entry["skipped_tiles"]:
        category = item["category"].lower()
        if category in ("missing output", "missing output after export"):
            label = "Missing TIFF"
        elif category == "all-zero tiff":
            label = "All-zero TIFF"
        elif "crs" in category:
            label = "CRS issue"
        elif category in ("no valid raster values", "no usable raster data"):
            label = "NoData / no valid values"
        elif category == "no overview data":
            label = "No data at report resolution"
        elif category == "covered by earlier tiles":
            label = "Covered by earlier tiles"
        elif category == "product unavailable":
            label = "Variable unavailable"
        else:
            label = "Other issue"
        counts[label] += 1
    return counts


def _render_mosaic_skip_details(pdf: PdfPages, region: str, entries: list[dict]) -> int:
    """Summarize counts by year pair, role and variable; never list individual tiles."""

    lines = []
    for entry in entries:
        lines.append(f"{entry['pair']} | {entry['role']} / {entry['column']}")
        lines.append(f"Included {entry['included_count']:,}; skipped {entry['skipped_count']:,}; total {entry['total_count']:,}")
        counts = _mosaic_skip_counts(entry)
        summary = "; ".join(f"{label}: {count:,}" for label, count in sorted(counts.items())) or "No tiles skipped"
        lines.extend(textwrap.wrap(summary, width=140))
        gap = entry.get("gap_check")
        if gap is None:
            lines.append("GAP TEST: NOT AVAILABLE — no diagnostic result recorded.")
        else:
            blank, total = gap["blank_inside_mask"], gap["mask_cells"]
            lines.append(f"GAP CHECK: {blank:,}/{total:,} report cells blank inside mask ({100 * blank / max(total, 1):.2f}%).")
            lines.append(f"GAP SOURCE TEST: {gap['sampled_blank_cells']:,} sampled blanks; valid native centers {gap['native_valid']:,}; "
                         f"valid neighbors only {gap['nearby_only']:,}; no valid 3x3 pixels {gap['native_nodata']:,}.")
            lines.append(f"Near TIFF edge: {gap['source_edge_samples']:,}; failed tile checks: {gap['failed_tile_checks']:,}. "
                         "No samples means inconclusive, not no gaps.")
        lines.append(f"Native-center recovery: {entry.get('native_center_restored', 0):,} per-tile cells (overlaps possible); stored overviews disabled.")
        lines.append("")
    pages = 0
    for offset in range(0, len(lines), 40):
        fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        try:
            fig.text(.055, .95, f"{region} | TIFF AVAILABILITY AND GAP DIAGNOSTICS", fontsize=16, weight="bold")
            fig.text(.055, .90, "\n".join(lines[offset:offset + 40]), va="top", fontsize=9, linespacing=1.4)
            fig.text(.055, .035, "Blank coverage is measured on the report grid. Source samples are not area percentages; valid centers flag a discrepancy, not a confirmed cause.", fontsize=8)
            pdf.savefig(fig)
            pages += 1
        finally:
            plt.close(fig)
    return pages


def _export_problem_category(reason: str) -> str:
    """Classify tile-level eligibility or raster diagnostics for report counts."""

    text = reason.lower()
    if "no crs" in text or "crs is none" in text:
        return "No CRS"
    if "crs mismatch" in text or "template crs must match" in text:
        return "CRS mismatch"
    if "all-zero tiff" in text:
        return "All-zero TIFF"
    if "no valid raster values" in text:
        return "No valid raster values"
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


def _concise_export_problems(item: dict) -> list[str]:
    """Explain each actual failure without embedding paths or diagnostic JSON."""

    details = item.get("details", {})
    lines = []
    for key, check in details.get("input_checks", {}).items():
        label = key.replace("_", " ")
        if check.get("exists") is False:
            lines.append(f"Missing {label}.")
        lines.extend(f"{label}: {problem}" for problem in check.get("problems", []))
    for issue in details.get("issues", []):
        message = issue["message"]
        for check in details.get("input_checks", {}).values():
            if check.get("path"):
                message = message.replace(str(check["path"]) + ": ", "")
        if not any(message in line or line in message for line in lines):
            lines.append(message)
    if not lines:
        reason = item.get("reason", item.get("category", "Unknown failure"))
        reason = reason.split("\nDiagnostics:", 1)[0]
        reason = reason.split(" Expected TIFF:", 1)[0].split(" Source TIFF:", 1)[0]
        lines = [reason]
    if "finite_xy_value_rows" in details:
        label = f"{details.get('role', '')}/{details.get('column', details.get('value_column', ''))}"
        lines.append(f"{label}: {details.get('input_rows', 0)} rows; "
                     f"{details['finite_xy_value_rows']} with finite X/Y/value; "
                     f"{details.get('rows_inside_tile_grid', 0)} inside grid; "
                     f"{details.get('input_rows_in_polygon_touching_cells', 0)} inside polygon cells.")
    return list(dict.fromkeys(lines))


def _render_export_audit(pdf: PdfPages, region: str, audit: dict) -> int:
    """Summarize current-run export outcomes without listing affected tiles."""

    lines = []
    for summary in audit["pairs"]:
        lines.append(f"{summary['pair']} | EXPORTED {summary['exported']:,}   SKIPPED {summary['skipped']:,}   NOT RUN {summary['not_run']:,}   TOTAL {summary['total']:,}")
        counts = summary["skip_categories"]
        if counts:
            text = "; ".join(f"{category}: {count:,}" for category, count in sorted(counts.items()))
            lines.extend(textwrap.wrap(text, width=140))
        elif summary["not_run"]:
            lines.append("No exports attempted for these tiles; existing TIFFs are checked in the variable summary.")
        else:
            lines.append("No export problems found.")
        lines.append("")
    pages = 0
    for offset in range(0, len(lines), 40):
        fig = plt.figure(figsize=(11.7, 8.3), facecolor="white")
        try:
            fig.text(.055, .95, f"{region} | EXPORT SUMMARY", fontsize=16, weight="bold")
            fig.text(.055, .89, "\n".join(lines[offset:offset + 40]), va="top", fontsize=9, linespacing=1.35)
            fig.text(.055, .035, "Current-run counts only. TIFF availability and skip reasons are counted separately for each variable.", fontsize=8)
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
    finite_rows = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    keep = finite_rows & (values != nodata)
    for column, array in zip(("X", "Y", value_col), arrays):
        finite = np.isfinite(array)
        diagnostics[f"{column}_nonfinite_or_nonnumeric_rows"] = int((~finite).sum())
        diagnostics[f"{column}_finite_range"] = [float(array[finite].min()), float(array[finite].max())] if finite.any() else None
    diagnostics["finite_xy_value_rows"] = int(keep.sum())
    diagnostics["source_nodata_rows"] = int((finite_rows & (values == nodata)).sum())
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

def _source_extrapolation(prediction: pd.DataFrame, training: pd.DataFrame | None, metadata: dict | None, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Use the recorded source ensemble's predictors and training ranges."""

    notes = cfg["_export_notes"]
    if "predictor_extrapolation_count" in prediction and pd.to_numeric(prediction["predictor_extrapolation_count"], errors="coerce").notna().any():
        notes.append("Using extrapolation counts already saved in the prediction summary.")
        return prediction, []
    try:
        deployment_path = UPath(cfg["prediction_summary"]).parent / f"LOCAL_prediction_metadata_{cfg['pair']}.json"
        deployment = json.loads(deployment_path.read_text()) if deployment_path.exists() else {}
        if deployment and (deployment.get("tile_id") != cfg["tile_id"] or deployment.get("pair") != cfg["pair"]):
            raise ValueError("Prediction metadata tile/year pair does not match this export")
        sources = set()
        if "local_source_model_tile" in prediction:
            sources.update(prediction["local_source_model_tile"].dropna().astype(str).unique())
        if deployment.get("source_model_tile"):
            sources.add(str(deployment["source_model_tile"]))
        if not sources and training is not None:
            sources.add(cfg["tile_id"])
        if len(sources) != 1:
            raise ValueError("Source model tile is missing or ambiguous")
        source = sources.pop()
        if source in (".", "..") or "/" in source or "\\" in source:
            raise ValueError("Invalid source model tile identifier")
        root = UPath(cfg.get("model_out_dir", str(UPath(cfg["training_summary"]).parent.parent.parent)))
        folder = root / source / "Training"
        source_path = UPath(deployment["source_metadata_path"]) if deployment.get("source_metadata_path") else folder / f"LOCAL_model_metadata_{cfg['pair']}.json"
        source_meta = json.loads(source_path.read_text()) if source_path.exists() else (metadata if source == cfg["tile_id"] else None)
        if not source_meta or source_meta.get("tile_id") != source or source_meta.get("pair") != cfg["pair"]:
            raise ValueError(f"Missing or mismatched source model metadata for {source}/{cfg['pair']}")
        predictors = source_meta.get("predictors")
        if not isinstance(predictors, list) or not predictors or not all(isinstance(v, str) for v in predictors) or len(set(predictors)) != len(predictors):
            raise ValueError("Source model predictor schema is invalid")
        missing = sorted(set(predictors) - set(prediction.columns))
        if missing:
            raise ValueError(f"Prediction summary lacks source predictors: {missing}")
        folder = source_path.parent if source_path.exists() else folder
        range_path = folder / f"training_predictor_ranges_{cfg['pair']}.csv"
        if range_path.exists():
            with range_path.open("rb") as stream:
                table = pd.read_csv(stream)
            if not {"predictor", "min_training", "max_training"}.issubset(table.columns) or table.predictor.duplicated().any():
                raise ValueError("Source range table has invalid columns or duplicate predictors")
            table = table.set_index("predictor").loc[predictors]
            ranges = table.reset_index().to_dict("records")
            range_source = str(range_path)
        else:
            training_path = folder / f"LOCAL_training_summary_{cfg['pair']}.geoparquet"
            if source == cfg["tile_id"] and training is not None:
                source_training = training
            else:
                with training_path.open("rb") as stream:
                    source_training = pd.read_parquet(stream, columns=predictors)
            ranges = []
            for predictor in predictors:
                values = pd.to_numeric(source_training[predictor], errors="coerce").to_numpy(dtype=float)
                finite = values[np.isfinite(values)]
                ranges.append(dict(predictor=predictor, min_training=float(finite.min()) if len(finite) else None,
                                   max_training=float(finite.max()) if len(finite) else None))
            range_source = str(training_path)
        counts = np.zeros(len(prediction), dtype=int)
        for item in ranges:
            lo, hi = item["min_training"], item["max_training"]
            if lo is None or hi is None or not np.isfinite(float(lo)) or not np.isfinite(float(hi)):
                raise ValueError(f"No finite source training range for {item['predictor']}")
            if float(lo) > float(hi):
                raise ValueError(f"Reversed source training range for {item['predictor']}")
            values = pd.to_numeric(prediction[item["predictor"]], errors="coerce").to_numpy(dtype=float)
            counts += np.isfinite(values) & ((values < float(lo)) | (values > float(hi)))
        out = prediction.copy()
        out["predictor_extrapolation_count"] = counts
        out["predictor_extrapolation_any"] = counts > 0
        cfg["_range_source"] = dict(tile_id=source, pair=cfg["pair"], path=range_source)
        notes.append(f"Extrapolation checked against source model {source}/{cfg['pair']}: {range_source}")
        return out, ranges
    except Exception as exc:
        notes.append(f"Predictor extrapolation unavailable: {type(exc).__name__}: {exc}. Other raster products will still export.")
        return prediction.drop(columns=["predictor_extrapolation_count", "predictor_extrapolation_any"], errors="ignore"), []


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
            try:
                if column not in df:
                    if column in OPTIONAL:
                        skipped.append({"column": column, "reason": "No local training ranges and no extrapolation count supplied by prediction summary" if column == "predictor_extrapolation_count" else "not supplied by model summary"})
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
                if not np.any(grid != profile["nodata"]):
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
            except Exception as exc:
                failed = Path(cfg["raster_paths"][role][column])
                failed.unlink(missing_ok=True)
                reason = str(exc).split("\nDiagnostics:", 1)[0]
                skipped.append(dict(column=column, reason=f"{type(exc).__name__}: {reason}"))
    return {"ok": bool(outputs), "role": role, "pair": pair, "outputs": outputs, "skipped": skipped}

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
        if (role, "bathy_t") not in layers:
            continue
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

def _make_visual_outputs(train: pd.DataFrame | None, results: dict, cfg: dict) -> list[str]:
    """Render previews with Matplotlib access serialized inside each worker."""

    root = Path(cfg["training_output_dir"])
    for name in ("importance", "shap_summary"):
        uri = cfg[name]
        if UPath(uri).exists():
            _copy_file(uri, str(root / UPath(uri).name))
    with _PLOT_LOCK:
        try:
            plots = _plot_summary_pages(results, root, cfg["pair"], cfg["tile_id"], cfg["plot_max_dimension"])
            if train is not None:
                plots += _plot_fit_diagnostics(train, root, cfg["pair"], cfg["tile_id"], cfg["plot_max_points"])
            return plots
        finally:
            plt.close("all")

def _diagnostic_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(float) if column in frame else np.full(len(frame), np.nan)


def _render_tile_diagnostics(pdf: PdfPages, frames: dict, shap: pd.DataFrame, notes: list[str], region: str, tile: str, pair: str, settings: dict) -> None:
    """Port the supplied LOCAL fit, SHAP and eight-panel raster diagnostics."""

    heading = f"{region} | {tile} | {pair.replace('_', ' - ')}"

    def finish(fig, title, caption):
        fig.suptitle(title, x=.06, ha="left", fontsize=19, color="#173247", fontweight="bold")
        fig.text(.06, .92, heading, fontsize=11, color="#375b7a")
        fig.text(.06, .035, caption, fontsize=8, color="#375b7a", parse_math=False)
        try:
            pdf.savefig(fig, dpi=160)
        finally:
            plt.close(fig)

    # Clean visual copies only; summary files and model outputs remain untouched.
    frames = {role: frame.copy() for role, frame in frames.items()}
    nodata_values = {-9999.0, float(settings["raster_config"]["nodata"])}
    for role, frame in frames.items():
        rejected = np.zeros(len(frame), dtype=bool)
        for column in set(REPORT_PRODUCTS) | {"bathy_t1"}:
            if column not in frame:
                continue
            values = _diagnostic_values(frame, column)
            invalid = np.isin(values, list(nodata_values))
            if column == "bathy_t":
                rejected |= invalid
            if column == "bathy_t1" and "delta_bathy" in frame:
                frame.loc[invalid, "delta_bathy"] = np.nan
            frame.loc[invalid, column] = np.nan
        # Derived changes from an invalid baseline are not usable diagnostics.
        if rejected.any():
            frames[role] = frame.loc[~rejected].copy()
            notes.append(f"{role.title()}: excluded {int(rejected.sum()):,} rows with NoData starting bathymetry, including their derived predictions/changes.")

    train = frames["training"]
    actual, predicted = _diagnostic_values(train, "bathy_t1"), _diagnostic_values(train, "mean_predicted_bathy_t1")
    good = np.isfinite(actual) & np.isfinite(predicted)
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 8.3))
    fig.subplots_adjust(left=.09, right=.97, bottom=.15, top=.83, wspace=.65)
    ax = axes[0]
    if good.sum() > 2:
        a, b = actual[good], predicted[good]
        rmse = np.sqrt(np.mean((b-a)**2))
        denominator = np.sum((a-a.mean())**2)
        r2 = 1-np.sum((b-a)**2)/denominator if denominator else np.nan
        ax.scatter(a, b, s=3, alpha=.25, color="darkblue", rasterized=True)
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        if lo == hi:
            lo, hi = lo-1, hi+1
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Apparent training fit\nN={len(a):,}; RMSE={rmse:.3g} m; R²={r2:.3g}", fontsize=10)
    else:
        ax.text(.5, .5, "No finite observed/predicted pairs", ha="center", transform=ax.transAxes, wrap=True)
    ax.set(xlabel="Surveyed bathymetry t1 (m)", ylabel="Predicted bathymetry t1 (m)")
    ax = axes[1]
    if {"predictor", "mean_abs_shap"}.issubset(shap.columns):
        shap = shap.copy()
        shap["mean_abs_shap"] = pd.to_numeric(shap["mean_abs_shap"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        values = shap.groupby("predictor")["mean_abs_shap"].mean().dropna().nlargest(20).sort_values()
    else:
        values = pd.Series(dtype=float)
    if len(values):
        ax.barh([textwrap.fill(str(name), 24) for name in values.index], values.values, color="#367c92")
        ax.tick_params(axis="y", labelsize=7)
    else:
        ax.text(.5, .5, "SHAP summary unavailable / no finite values", ha="center", transform=ax.transAxes, wrap=True)
    bootstrap_count = shap.loc[np.isfinite(shap["mean_abs_shap"]), "bootstrap_iteration"].nunique() if {"mean_abs_shap", "bootstrap_iteration"}.issubset(shap.columns) else None
    ax.set(title="LOCAL SHAP importance" + (f"\nAcross {bootstrap_count} bootstrap models" if bootstrap_count is not None else ""), xlabel="Mean absolute SHAP (m)")
    finish(fig, "Model fit and SHAP importance", "Fit metrics use all finite training pairs (not held-out validation). All finite pairs plotted; SHAP averages bootstrap summaries.")

    water = {}
    crs_labels = []
    for role, frame in frames.items():
        crs = getattr(frame, "crs", None)
        if crs is None:
            crs_label = "CRS unavailable"
        else:
            authority = crs.to_authority()
            crs_label = f"{authority[0]}:{authority[1]}" if authority else str(crs.name)[:100]
        crs_labels.append(f"{role}: {crs_label}")
        depth = _diagnostic_values(frame, "bathy_t")
        water[role] = frame.loc[np.isfinite(depth) & (depth < 0)].copy()
        if "survey_end_date" in water[role]:
            water[role]["survey_year_bin_code"] = _survey_codes(water[role]["survey_end_date"], settings["legacy_survey_year_offset"])

    def limits(columns, symmetric=False):
        arrays = [_diagnostic_values(frame, col) for frame in water.values() for col in columns]
        values = np.concatenate(arrays)
        values = values[np.isfinite(values)]
        if not len(values):
            return (-1., 1.) if symmetric else (0., 1.)
        if symmetric:
            bound = max(float(np.percentile(np.abs(values), 98)), 1e-6)
            return -bound, bound
        lo, hi = (float(value) for value in np.percentile(values, [2, 98]))
        return (lo, hi) if lo < hi else (lo-1., hi+1.)

    depth_limits = settings["diagnostic_depth_limits"]
    change_limits = settings["diagnostic_change_limits"]
    panel_titles = {
        ("training", "bathy_t"): "Train Start Bathy (t) – Survey data",
        ("training", "mean_predicted_bathy_t1"): "Train Mean Prediction (t1)",
        ("training", "mean_predicted_change"): "Train Predicted Change [t1 – t]",
        ("training", "delta_bathy"): "Train Surveyed Change [t1 – t]",
        ("prediction", "bathy_t"): "Pred Start Bathy (t) – BlueTopo",
        ("prediction", "mean_predicted_bathy_t1"): "Pred Mean Prediction (t1)",
        ("prediction", "mean_predicted_change"): "Predicted Change [t1 – t]",
        ("prediction", "survey_year_bin_code"): "Prediction Survey End Date",
        ("training", "uncertainty_sd_bathy_t1"): "Train Uncertainty (SD from bootstraps)",
        ("prediction", "uncertainty_sd_bathy_t1"): "Prediction Uncertainty (SD from bootstraps)",
        ("prediction", "UC_t"): "BlueTopo Uncertainty (UC_t)",
    }
    uncertainty_max = max(limits(["uncertainty_sd_bathy_t1", "UC_t"])[1], 1e-6)
    for uncertainty in (False, True):
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.subplots_adjust(left=.055, right=.94, bottom=.19, top=.83, hspace=.65, wspace=.5)
        for row, role in enumerate(("training", "prediction")):
            frame = water[role]
            fourth = ("uncertainty_sd_bathy_t1" if role == "training" or "UC_t" not in frame else "UC_t") if uncertainty else ("delta_bathy" if role == "training" else "survey_year_bin_code")
            for ax, column in zip(axes[row], ["bathy_t", "mean_predicted_bathy_t1", "mean_predicted_change", fourth]):
                title, family, label = REPORT_PRODUCTS[column]
                ax.set_title(panel_titles[role, column], fontsize=9, fontweight="bold", loc="left")
                x, y, z = (_diagnostic_values(frame, name) for name in ("X", "Y", column))
                valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
                if not valid.any():
                    ax.text(.5, .5, "No finite water-only data\n(or field/coordinates missing)", ha="center", va="center", transform=ax.transAxes, fontsize=8)
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                indices = np.flatnonzero(valid)
                if family == "survey":
                    cmap, norm = plt.get_cmap("viridis", len(SURVEY_LABELS)), BoundaryNorm(np.arange(-.5, len(SURVEY_LABELS)+.5), len(SURVEY_LABELS))
                else:
                    bounds = depth_limits if family == "depth" else change_limits if family == "change" else (0., uncertainty_max)
                    cmap, norm = _report_color_scale(family, list(bounds))
                if family == "depth":
                    cmap = plt.get_cmap("viridis")
                # geom_raster equivalent: draw every finite cell at its coordinate
                # spacing, without point thinning or interpolating across holes.
                spacing = []
                for coordinates in (x, y):
                    differences = np.diff(np.unique(coordinates[np.isfinite(coordinates)]))
                    spacing.append(float(differences.min()) if len(differences) else 1.)
                centers = np.column_stack((x[indices], y[indices]))
                offsets = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * spacing
                artist = PolyCollection(centers[:, None, :] + offsets, array=z[indices], cmap=cmap, norm=norm, edgecolors="none", antialiaseds=False, rasterized=True)
                ax.add_collection(artist)
                coordinates_valid = np.isfinite(x) & np.isfinite(y)
                ax.set_xlim(x[coordinates_valid].min()-spacing[0]/2, x[coordinates_valid].max()+spacing[0]/2)
                ax.set_ylim(y[coordinates_valid].min()-spacing[1]/2, y[coordinates_valid].max()+spacing[1]/2)
                if family == "change":
                    ax.text(.97, .97, "Deposition", ha="right", va="top", transform=ax.transAxes, fontsize=8, fontweight="bold")
                    ax.text(.97, .03, "Erosion", ha="right", va="bottom", transform=ax.transAxes, fontsize=8, fontweight="bold")
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([])
                ax.set_yticks([])
                bar = fig.colorbar(artist, ax=ax, fraction=.045, pad=.03, extend="neither" if family == "survey" else "both")
                bar.ax.tick_params(labelsize=7)
                if family == "survey":
                    bar.set_ticks(range(len(SURVEY_LABELS)))
                    bar.set_ticklabels(SURVEY_LABELS)
                    bar.ax.invert_yaxis()
                    bar.ax.set_title("Survey\nYear Bin", fontsize=8, loc="left")
                else:
                    bar.ax.set_title("Depth (m)" if family == "depth" else "Delta (m)" if family == "change" else "UC (m)" if column == "UC_t" else "SD (m)", fontsize=8, loc="left")
                    if family == "depth":
                        bar.set_ticks(np.arange(math.ceil(norm.vmin), math.floor(norm.vmax) + 1, 1))
                    elif family == "change":
                        bar.set_ticks(np.arange(math.ceil(norm.vmin), math.floor(norm.vmax) + 1, 1))
                values = z[valid]
                p02, median, p98 = np.percentile(values, [2, 50, 98])
                clipped = "" if family == "survey" else f"; below/above scale={np.count_nonzero(values < norm.vmin):,}/{np.count_nonzero(values > norm.vmax):,}"
                ax.annotate(f"N={len(values):,}{clipped}; zeros={np.count_nonzero(values == 0):,}\nmin / p02 / median / p98 / max\n{values.min():.3g} / {p02:.3g} / {median:.3g} / {p98:.3g} / {values.max():.3g}", xy=(0, 0), xycoords="axes fraction", xytext=(0, -27), textcoords="offset points", va="top", fontsize=6.5)
        finish(fig, "Tile uncertainty" if uncertainty else "Training and prediction summary", "Water-only: finite bathy_t < 0. All finite cells plotted at X/Y spacing; missing cells stay blank. No interpolation.\n" + " | ".join(crs_labels) + f"\nShared scales: depth {depth_limits[0]:g} to {depth_limits[1]:g} m; change {change_limits[0]:g} to {change_limits[1]:g} m; uncertainty 0–P98. End colors saturate outside limits; counts shown under plots.")
    if notes:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.text(.06, .82, "\n\n".join(textwrap.fill(note, 110) for note in notes), va="top", fontsize=10, parse_math=False)
        finish(fig, "Diagnostic input notes", "Missing inputs do not stop the regional reports.")


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
        errors.append(f"No CRS; expected {cfg['crs']}" if spatial.crs is None else f"CRS mismatch: {spatial.crs}; expected {cfg['crs']}")
    if not df.columns.is_unique:
        errors.append("duplicate column names")
    if df.empty:
        errors.append("summary has zero rows")
    products = TRAIN_PRODUCTS if role == "training" else PRED_PRODUCTS
    required = {"X", "Y", "FID"}
    if role == "prediction":
        required.add("prediction_row_usable")
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
    # Product values are filtered independently during rasterization.
    if errors:
        audit["problems"] = errors
        raise ValueError(f"{role} summary: " + "; ".join(errors))
    return df.loc[usable].copy()


def _load_inputs(cfg: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None, list[dict]]:
    """Validate prediction exports even when no local training summary exists."""

    checks, issues, frames = {}, [], {}
    notes = []
    metadata = None
    for key in ("model_metadata", "training_summary", "prediction_summary"):
        path = cfg[key]
        checks[key] = dict(path=path, exists=UPath(path).exists())
    for key in ("model_metadata", "training_summary", "prediction_summary"):
        if checks[key]["exists"]:
            continue
        if key == "training_summary":
            notes.append("No local training summary: training rasters and training-fit diagnostics are unavailable; exporting prediction products.")
            continue
        if key == "model_metadata" and not checks["training_summary"]["exists"]:
            notes.append("No local model metadata: validating prediction summary fields and CRS directly.")
            continue
        label = "model metadata" if key == "model_metadata" else "prediction summary"
        issues.append(dict(category=f"Missing {label}", message=f"Missing required {label}: {cfg[key]}"))
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
    cfg["_export_notes"] = notes
    train, prediction = frames.get("training"), frames.get("prediction")
    ranges = []
    if prediction is not None:
        prediction, ranges = _source_extrapolation(prediction, train, metadata, cfg)
    if prediction is not None and "survey_end_date" in prediction:
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
                       for role, frame in (("training", train), ("prediction", prediction)) if frame is not None}
            for role, frame, products in (("training", train, TRAIN_PRODUCTS), ("prediction", prediction, PRED_PRODUCTS)):
                if frame is None:
                    results[role] = dict(ok=False, role=role, pair=cfg["pair"], outputs={},
                                         skipped=[dict(column=column, reason=f"No {role} summary; exporting the available role only.") for column in products])
            for role, result in results.items():
                for item in result["skipped"]:
                    cfg["_export_notes"].append(f"NOT EXPORTED {role}/{item['column']}: {item['reason']}")
            if not any(result["outputs"] for result in results.values()):
                raise RasterExportDiagnostic("No raster products could be exported: " + "; ".join(cfg["_export_notes"]), category="No raster products")
            try:
                if prediction is not None and "predictor_extrapolation_count" in prediction:
                    diagnostic_columns = [column for column in ("X", "Y", "predictor_extrapolation_count", "predictor_extrapolation_any") if column in prediction]
                    prediction[diagnostic_columns].to_parquet(staged["extrapolation_path"], index=False)
                if ranges:
                    Path(staged["ranges_path"]).write_text(json.dumps(ranges, indent=2), encoding="utf-8")
            except Exception as exc:
                Path(staged["extrapolation_path"]).unlink(missing_ok=True)
                Path(staged["ranges_path"]).unlink(missing_ok=True)
                cfg["_export_notes"].append(f"Extrapolation sidecar files unavailable: {type(exc).__name__}: {exc}; successful rasters retained.")
            try:
                visuals = _make_visual_outputs(train, results, staged)
            except Exception as exc:
                visuals = []
                shutil.rmtree(staged["plot_output_dir"], ignore_errors=True)
                cfg["_export_notes"].append(f"Diagnostic plots unavailable: {type(exc).__name__}: {exc}; successful rasters retained.")
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
                    "status": "Exported", "notes": cfg.get("_export_notes", []), "details": {"input_checks": cfg.get("_input_audit", {}),
                    "extrapolation_source": cfg.get("_range_source"),
                    "optional_outputs_not_supplied": {role: result.get("skipped", []) for role, result in results.items()}},
                    "results": results, "visual_outputs": [published[path] for path in visuals]}
    except RasterExportDiagnostic as exc:
        Engine.write_message_dask(f"Skipping raster export for {cfg['region']}/{cfg['tile_id']} | {cfg['pair']}: {exc}", cfg["log_dir"])
        return {"region": cfg["region"], "tile_id": cfg["tile_id"], "pair": cfg["pair"],
                "skipped": True, "status": "Skipped", "category": exc.category, "details": exc.details,
                "reason": str(exc), "results": {}, "visual_outputs": []}
    except Exception as exc:
        details = {"exception_type": type(exc).__name__, "traceback": traceback.format_exc(),
                   "input_checks": cfg.get("_input_audit", {}),
                   "training_summary": cfg["training_summary"], "prediction_summary": cfg["prediction_summary"]}
        reason = f"{type(exc).__name__}: {exc}"
        Engine.write_message_dask(f"Raster export failed for {cfg['region']}/{cfg['tile_id']} | {cfg['pair']}: {reason}\n{details['traceback']}", cfg["log_dir"])
        return {"region": cfg["region"], "tile_id": cfg["tile_id"], "pair": cfg["pair"],
                "skipped": True, "status": "Skipped", "category": "Export exception", "details": details,
                "reason": reason, "results": {}, "visual_outputs": []}
    finally:
        gc.collect()

class MLRasterExportEngine(Engine):
    """Export model summaries as COGs and diagnostic plots for each ecoregion."""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        """Initialize shared engine state and raster-export settings."""

        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        self.is_aws = param_lookup["env"] == "aws"
        self.settings = {
            "tile_id": None, "pair": None, "tile_id_column": "tile_id",
            "diagnostic_depth_limits": (-10.0, 0.0),  # Shared across diagnostic depth panels and year pairs.
            "diagnostic_change_limits": (-6.0, 6.0),  # Symmetric deposition/erosion scale for diagnostics.
            "diagnostic_tile_id": "BH4S556X_3",  # None disables the separate tile PDF; does not filter regional reports.
            "create_model_tifs": False,  # False: skip tile exports and rebuild the PDF from existing TIFFs.
            "legacy_survey_year_offset": True,
            "raster_workers": 16,
            "raster_threads_per_worker": 2,  # In-flight limit = workers * threads (32).
            "mosaic_report": True, "report_max_dimension": 1600,
            "report_progress_interval": 250,  # Log aggregate progress every 250 files, including skipped tiles.
            "report_basemap_service": "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer",
            "report_basemap_reference_service": "https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Reference/MapServer",
            "report_basemap_timeout": 20,
            "memory_limit": "5GB", "raster_processes": True,
            "plot_max_dimension": 1000, "plot_max_points": 50000,
            "raster_config": {
                "nodata": -9999.0, "compress": "DEFLATE", "blocksize": 512,
                "overview_levels": [2, 4, 8, 16, 32], "overview_resampling": "average",
                "max_grid_cells": 25000000,
            },
        }
        if self.settings["raster_processes"] is not True:
            raise ValueError("Raster export requires process workers")
        for key in ("report_progress_interval", "raster_workers", "raster_threads_per_worker", "plot_max_dimension", "plot_max_points", "report_max_dimension", "report_basemap_timeout"):
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

        resampled_dir = str(get_config_item("LIDAR_PLOTS", "RESAMPLED_MOSAICS"))
        self.resampled_mosaics_dir = (UPath(resampled_dir) if "://" in resampled_dir or Path(resampled_dir).is_absolute()
                                   else UPath(f"{s3_dir_base}/{resampled_dir}") if self.is_aws
                                   else UPath(self.outputs_dir / resampled_dir))

        self.write_message(f"{region}: raster output root: {self.raster_out_dir}", OUTPUTS)
        self.write_message(f"{region}: training rasters: {self.raster_out_dir / '{tile_id}' / 'Training'}", OUTPUTS)
        self.write_message(f"{region}: prediction rasters: {self.raster_out_dir / '{tile_id}' / 'Prediction'}", OUTPUTS)
        self.write_message(f"{region}: diagnostic plots: {self.raster_out_dir / '{tile_id}' / 'Training' / 'diagnostic_plots'}", OUTPUTS)
        if self.settings["mosaic_report"]:
            self.write_message(f"{region}: mosaics and PDF report: {self.raster_out_dir / 'mosaic_report'}", OUTPUTS)

        template = get_config_item("MASK", "MASK_PRED_PATH")
        self.template_path = UPath(f"{s3_dir_base}/{template}") if self.is_aws else UPath(self.outputs_dir / template)

        subgrids = get_config_item("MODEL", "SUBGRIDS")
        self.subgrids_path = UPath(f"{s3_dir_base}/{subgrids}") if self.is_aws else UPath(self.outputs_dir / subgrids)
        self.subgrids_layer = get_config_item("MODEL", "SUBGRIDS_LAYER")

    def load_template_metadata(self) -> dict:
        """Read grid alignment and extent; template values never control export eligibility."""

        path = str(self.template_path).replace("s3://", "/vsis3/", 1)
        with rasterio.open(path) as template:
            if template.crs != rasterio.crs.CRS.from_user_input(self.target_crs):
                raise ValueError("Template CRS must match the target CRS")
            return {"transform": template.transform, "width": template.width,
                    "height": template.height, "crs": str(template.crs), "source_uri": str(self.template_path)}

    def collect_training_files(self) -> Iterator[UPath]:
        """Yield unique training paths incrementally from ML handoff artifacts."""

        expected = set()
        for stem, extension in (("training_summary", "geoparquet"), ("prediction_summary", "geoparquet"), ("model_metadata", "json")):
            for path in self.model_out_dir.rglob(f"LOCAL_{stem}_*.{extension}"):
                match = re.fullmatch(r"LOCAL_" + stem + r"_(\d{4}_\d{4})\." + extension, path.name)
                if match is not None:
                    tile_root = path.parent.parent
                    training_path = str(tile_root / "Training" / f"LOCAL_training_summary_{match[1]}.geoparquet")
                    if training_path not in expected:
                        expected.add(training_path)
                        yield UPath(training_path)

    def load_tile_grid(self) -> gpd.GeoDataFrame:
        """Read the official tile polygons once per region in the target CRS."""

        uri = str(self.subgrids_path)
        if Path(uri).suffix.lower() in (".parquet", ".geoparquet"):
            with self.subgrids_path.open("rb") as stream:
                grid = gpd.read_parquet(stream)
        else:
            layer = self.subgrids_layer
            options = {"layer": layer} if layer else {}
            with LocalizedFile(uri) as local:
                grid = gpd.read_file(local, **options)
        if grid.crs is None or self.settings["tile_id_column"] not in grid:
            raise ValueError("Tile grid needs a CRS and the configured tile ID column")
        return grid.to_crs(self.target_crs)

    def prepare_tasks(self, region: str, grid: gpd.GeoDataFrame) -> Iterator[dict]:
        """Prepare each job only when the bounded submission queue requests it."""
        seen = set()
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
            model_train = training_file.parent
            model_pred = self.model_out_dir / tile_id / "Prediction"
            train_out = self.raster_out_dir / tile_id / "Training"
            pred_out = self.raster_out_dir / tile_id / "Prediction"
            raster_paths = {
                role: {column: str(folder / f"{stem}_{pair}.tif") for column, stem in products.items()}
                for role, folder, products in (("training", train_out, TRAIN_PRODUCTS), ("prediction", pred_out, PRED_PRODUCTS))
            }
            yield {
                "region": region, "tile_id": tile_id, "pair": pair, "crs": str(self.target_crs),
                "model_out_dir": str(self.model_out_dir),
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
            }

    def render_tile_dataset_coverage(self, pdf: PdfPages, region: str, tile: str) -> None:
        """First diagnostic page: individual survey footprints, colored by survey year."""

        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        fig.subplots_adjust(left=.07, right=.48, bottom=.27, top=.86)
        fig.suptitle(f"Dataset coverage | {region} | {tile}", x=.06, ha="left", fontsize=18, color="#173247")
        records, counts = [], Counter()
        try:
            grid = self.load_tile_grid()
            selected = grid.loc[grid[self.settings["tile_id_column"]].astype(str) == tile]
            if len(selected) != 1 or selected.geometry.iloc[0].geom_type not in ("Polygon", "MultiPolygon"):
                raise ValueError("Expected one official tile polygon")
            geom = selected.geometry.iloc[0]
            left, bottom, right, top = geom.bounds
            resolution = max(right-left, top-bottom) / self.settings["plot_max_dimension"]
            width, height = max(1, math.ceil((right-left)/resolution)), max(1, math.ceil((top-bottom)/resolution))
            transform = Affine(resolution, 0, left, 0, -resolution, top)
            profile = dict(crs=str(grid.crs), transform=transform, width=width, height=height,
                           bounds=(left, top-height*resolution, left+width*resolution, top))
            inside = geometry_mask([mapping(geom)], out_shape=(height, width), transform=transform, invert=True)
            meters_per_unit = rasterio.crs.CRS.from_user_input(profile["crs"]).linear_units_factor[1]
            background_note = "BlueTopo background unavailable (no usable prediction starting-bathymetry TIFF)."
            backgrounds = sorted((self.raster_out_dir / tile / "Prediction").glob("LOCAL_Pred_Start_Bathy_t_*.tif"), key=lambda path: path.name, reverse=True)
            for background_path in backgrounds:
                try:
                    with rasterio.open(_report_raster_path(str(background_path))) as source:
                        if source.crs is None:
                            continue
                        with WarpedVRT(source, crs=profile["crs"], transform=transform, width=width, height=height,
                                       dtype="float32", nodata=float("nan"), resampling=Resampling.nearest) as vrt:
                            background = np.ma.masked_invalid(vrt.read(1, masked=True))
                    background = np.ma.masked_where(~inside | (background.filled(-9999) == -9999), background)
                    if not background.count():
                        continue
                    lo, hi = np.percentile(background.compressed(), [2, 98])
                    ax.imshow(background, extent=(left, left+width*resolution, top-height*resolution, top),
                              origin="upper", cmap="Greys_r", vmin=lo, vmax=hi if hi > lo else lo+1,
                              alpha=.4, interpolation="nearest", zorder=0)
                    background_note = f"Muted gray: Pred Start Bathy (t) – BlueTopo; {background_path.name} (latest usable pair)."
                    break
                except Exception as exc:
                    self.write_message(f"Coverage background skipped: {background_path.name}: {type(exc).__name__}", OUTPUTS)
            pixel_notes = set()
            year_grids = {}
            paths = sorted(self.resampled_mosaics_dir.rglob("mosaic_*_resampled.tif"), key=str)
            jobs = []
            for path in paths:
                years = re.findall(r"(?<![0-9])(?:19|20)[0-9]{2}(?![0-9])", path.stem)
                if len(years) != 1:
                    counts["Missing/ambiguous year"] += 1
                    continue
                jobs.append((path, years[0]))
            years = sorted({year for _, year in jobs})
            # Deliberately separated hues rather than neighboring colors on a gradient.
            palette = ["#E6194B", "#0057D9", "#008A35", "#A020F0", "#F58200",
                       "#00A6B2", "#C0008B", "#756000", "#001970", "#7D2418",
                       "#387800", "#6B4C9A", "#007A72", "#A65A00", "#303030"]
            # Extend without repeating colors if more years than the base palette occur.
            candidates = [matplotlib.colors.hsv_to_rgb((h, sat, val)) for h in np.linspace(0, 1, 72, endpoint=False)
                          for sat in (.65, 1.) for val in (.55, .85)]
            while len(palette) < len(years):
                chosen = np.array([matplotlib.colors.to_rgb(color) for color in palette])
                distances = np.min(np.sum((np.asarray(candidates)[:, None, :] - chosen[None, :, :])**2, axis=2), axis=1)
                palette.append(matplotlib.colors.to_hex(candidates.pop(int(np.argmax(distances)))))
            colors = dict(zip(years, palette))
            xs = left + (np.arange(-1, width+1)+.5)*resolution
            ys = top - (np.arange(-1, height+1)+.5)*resolution
            pending, iterator = deque(), iter(jobs)
            limit = self.settings["raster_workers"] * self.settings["raster_threads_per_worker"]

            def submit_next():
                item = next(iterator, None)
                if item is not None:
                    path, year = item
                    pending.append((path, year, self.client.submit(_read_dataset_coverage, str(path), profile, pure=False)))

            try:
                for _ in range(limit):
                    submit_next()
                checked = 0
                while pending:
                    path, year, future = pending.popleft()
                    try:
                        coverage, error, pixel_note, dataset_values = future.result()
                    except Exception as exc:
                        coverage, error, pixel_note, dataset_values = None, f"Worker error: {type(exc).__name__}", None, None
                    finally:
                        future.release()
                    checked += 1
                    if error:
                        counts[error] += 1
                    else:
                        coverage &= inside
                        if coverage.any():
                            if year not in year_grids:
                                year_grids[year] = np.ma.masked_all((height, width), dtype="float32")
                            combined = year_grids[year]
                            take = coverage & np.ma.getmaskarray(combined)
                            combined.data[take] = dataset_values.data[take]
                            combined.mask[take] = False
                            pixel_notes.add(pixel_note)
                            records.append((year, path.name, pixel_note))
                            counts["Included"] += 1
                        else:
                            counts["No valid data inside tile"] += 1
                    if checked % 25 == 0 or checked == len(jobs):
                        self.write_message(f"COVERAGE {region}/{tile}: checked {checked}/{len(jobs)} datasets; {counts['Included']} included.", OUTPUTS)
                    submit_next()
            finally:
                for _, _, future in pending:
                    future.cancel()
                    future.release()
            # Union of all valid footprints for each year: no internal dataset seams.
            for year, combined in sorted(year_grids.items()):
                union = np.pad((~np.ma.getmaskarray(combined)).astype("uint8"), 1)
                ax.contour(xs, ys, union, levels=[.5], colors="white", linewidths=3.6, linestyles="solid", zorder=2)
                ax.contour(xs, ys, union, levels=[.5], colors=[colors[year]], linewidths=2.0, linestyles="solid", zorder=3)
            selected.boundary.plot(ax=ax, color="black", linewidth=1.4)
            used_years = sorted({year for year, _, _ in records})
            handles = [Line2D([0], [0], color=colors[year], linestyle="solid", linewidth=2.0, label=year) for year in used_years]
            if handles:
                ax.legend(handles=handles, title="Survey year", loc="upper left", bbox_to_anchor=(0, -.1), fontsize=7, ncol=min(5, len(handles)))
            else:
                ax.text(.5, .5, "No usable dataset coverage found", transform=ax.transAxes, ha="center")
            ax.set(xlim=(left, right), ylim=(bottom, top), xlabel="Easting", ylabel="Northing", aspect="equal")
            ax.ticklabel_format(style="plain", useOffset=False)
            _render_coverage_year_thumbnails(fig, year_grids, profile, colors, self.settings["diagnostic_depth_limits"])
            resolution_note = "; ".join(sorted(pixel_notes)) if len(pixel_notes) <= 3 else "multiple native pixel sizes; see dataset index"
            fig.text(.06, .095, textwrap.fill(background_note, 130), fontsize=7, parse_math=False)
            fig.text(.06, .055, f"Outlines: same-year union of mosaic_*_resampled.tif from LIDAR_PLOTS.RESAMPLED_MOSAICS; source pixels: {resolution_note or 'unavailable'}.\nDisplay grid: {resolution*meters_per_unit:.4g} x {resolution*meters_per_unit:.4g} m; nearest resampling. Black: tile boundary. Small features may be omitted; outlines may overlap.", fontsize=7, parse_math=False)

        except Exception as exc:
            counts["Coverage page error"] += 1
            ax.text(.05, .5, textwrap.fill(f"Coverage unavailable: {type(exc).__name__}: {str(exc)[:350]}", 85), transform=ax.transAxes, parse_math=False)
        finally:
            pdf.savefig(fig)
            plt.close(fig)
        self.write_message(f"DATASET COVERAGE {region}/{tile}: {dict(counts)}; source: {self.resampled_mosaics_dir}", OUTPUTS)
        # Index identifies datasets individually even where outlines overlap.
        lines = [f"{year} | {name} | {pixel_note}" for year, name, pixel_note in sorted(records)]
        lines += [f"{reason}: {count}" for reason, count in sorted(counts.items())]
        for start in range(0, len(lines), 14):
            fig = plt.figure(figsize=(11.7, 8.3))
            fig.text(.06, .93, f"Dataset coverage index | {tile}", fontsize=17, color="#173247")
            fig.text(.06, .86, "\n\n".join(textwrap.fill(line, 115) for line in lines[start:start+14]), va="top", fontsize=8, parse_math=False)
            pdf.savefig(fig)
            plt.close(fig)

    def generate_tile_diagnostic_report(self, region: str) -> str | None:
        """Publish one tile's all-pairs diagnostic PDF before regional mosaics."""

        tile = self.settings["diagnostic_tile_id"]
        if not tile:
            return None
        root = self.model_out_dir / tile
        sources = {}
        for role, folder in (("training", "Training"), ("prediction", "Prediction")):
            for path in (root / folder).glob(f"LOCAL_{role}_summary_*.geoparquet"):
                match = re.fullmatch(rf"LOCAL_{role}_summary_(\d{{4}}_\d{{4}})\.geoparquet", path.name)
                if match:
                    sources.setdefault(match[1], {})[role] = path
        destination = self.raster_out_dir / "mosaic_report" / f"{region}_{tile}_diagnostics.pdf"
        self.write_message(f"Creating tile diagnostic PDF: {destination} ({len(sources)} year pairs).", OUTPUTS)
        with tempfile.TemporaryDirectory(prefix="hh_tile_diagnostic_") as tmp:
            local = Path(tmp) / destination.name
            with _PLOT_LOCK, PdfPages(local) as pdf:
                self.render_tile_dataset_coverage(pdf, region, tile)
                if not sources:
                    fig = plt.figure(figsize=(11.7, 8.3))
                    fig.text(.06, .88, f"{region} | {tile} | No model summaries found", fontsize=18)
                    fig.text(.06, .75, "No training or prediction year pairs are available for this tile.", fontsize=11)
                    pdf.savefig(fig)
                    plt.close(fig)
                for pair, paths in sorted(sources.items()):
                    frames, notes = {}, []
                    for role in ("training", "prediction"):
                        frames[role] = pd.DataFrame()
                        if role not in paths:
                            notes.append(f"{role.title()} summary: missing.")
                            continue
                        try:
                            with LocalizedFile(str(paths[role])) as source:
                                frames[role] = gpd.read_parquet(source)
                        except Exception as exc:
                            notes.append(f"{role.title()} summary unreadable: {type(exc).__name__}: {str(exc)[:250]}")
                    shap = pd.DataFrame()
                    try:
                        path = root / "Training" / f"LOCAL_shap_summary_{pair}.parquet"
                        if path.exists():
                            with LocalizedFile(str(path)) as source:
                                shap = pd.read_parquet(source)
                        else:
                            notes.append("SHAP summary: missing.")
                    except Exception as exc:
                        notes.append(f"SHAP summary unreadable: {type(exc).__name__}: {str(exc)[:250]}")
                    _render_tile_diagnostics(pdf, frames, shap, notes, region, tile, pair, self.settings)
                    self.write_message(f"Tile diagnostics {region}/{tile} | {pair}: pages completed; {len(notes)} input issue(s).", OUTPUTS)
            _copy_file(str(local), str(destination))
        self.write_message(f"Published tile diagnostic PDF: {destination}", OUTPUTS)
        return str(destination)

    def generate_mosaic_report(self, region: str, records: list[dict]) -> list[str]:
        """Build one PDF per role and variable, covering all year pairs."""

        if not self.settings["mosaic_report"]:
            return []
        read_limit = self.settings["raster_workers"] * self.settings["raster_threads_per_worker"]
        self.write_message(f"Building regional mosaic report for {region}: shared Dask workers, up to {read_limit} TIFF reads in flight; progress every {self.settings['report_progress_interval']} tiles.", OUTPUTS)
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
            return []
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
        entries = []
        with tempfile.TemporaryDirectory(prefix="hh_mosaic_report_") as tmp:
            root = Path(tmp)
            report_names = []
            scale_metadata = {}
            # Exclude unwanted regional PDFs; tile raster exports remain available.
            excluded_reports = {
                ("training", "bathy_t"),
                ("training", "mean_predicted_bathy_t1"),
                ("training", "delta_bathy"),
                ("training", "mean_predicted_change"),
                ("training", "uncertainty_sd_bathy_t1"),
                ("training", "mean_predicted_rate"),
                ("prediction", "survey_year_bin_code"),
            }
            report_groups = [(role, column, products)
                             for role, products in (("training", TRAIN_PRODUCTS), ("prediction", PRED_PRODUCTS))
                             for column in products if (role, column) not in excluded_reports]
            for variable_number, (role, column, products) in enumerate(report_groups, 1):
                variable_entries, ranges = [], {}
                report_name = f"{region}_mosaic_report_{role}_{column}.pdf"
                scale_key = f"{role}/{column}"
                self.write_message(f"Preparing report {variable_number}/{len(report_groups)}: {region}/{role}/{column}; all year pairs. PDF destination: {destination / report_name}", OUTPUTS)
                for pair, tiles in sorted(candidates.items()):
                    title, family, units = REPORT_PRODUCTS[column]
                    name = f"MOSAIC_{role}_{column}_{pair}"
                    entry = dict(pair=pair, role=role, column=column, included_tiles=[], skipped_tiles=[],
                                 data_path=f"{name}.tif", styled_path=f"{name}_colored.tif")
                    mosaic = np.ma.masked_all((height, width), dtype="float32")
                    gap_probes, gap_probe_errors = [], 0

                    def log_tile_progress(tile_id: str, skipped: dict | None = None) -> None:
                        included_count = len(entry["included_tiles"])
                        skipped_count = len(entry["skipped_tiles"])
                        checked = included_count + skipped_count
                        if checked != len(tiles) and checked % self.settings["report_progress_interval"]:
                            return
                        missing_count = sum(item["category"] in ("Missing output", "Missing output after export")
                                            for item in entry["skipped_tiles"])
                        other_count = skipped_count - missing_count
                        self.write_message(
                            f"PDF progress {region}/{role}/{column} | {pair}"
                            f" | checked {checked:,}/{len(tiles):,} files"
                            f" | used {included_count:,}, missing {missing_count:,}, other skipped {other_count:,}", OUTPUTS)

                    folder = "Training" if role == "training" else "Prediction"
                    jobs = ((tile_id, available.get((tile_id, pair, role, column)),
                             str(self.raster_out_dir / tile_id / folder / f"{products[column]}_{pair}.tif"))
                            for tile_id in sorted(tiles))
                    for tile_id, uri, data, read_error in _iter_report_tiles(
                            jobs, profile, family in ("survey", "count"), self.client, read_limit):
                        if uri is None:
                            expected_path = str(self.raster_out_dir / tile_id / folder / f"{products[column]}_{pair}.tif")
                            outcome = outcomes.get((tile_id, pair), {})
                            audit_item = audit_by_tile[tile_id, pair]
                            optional = outcome.get("results", {}).get(role, {}).get("skipped", [])
                            omitted = next((item for item in optional if item.get("column") == column), None)
                            if omitted is not None:
                                category, reason = "Product unavailable", omitted["reason"]
                            elif audit_item["status"] == "Skipped":
                                category, reason = audit_item["category"], audit_item["reason"]
                            elif audit_item["status"] == "Exported":
                                category, reason = "Missing output after export", "Export reported success but this TIFF is absent."
                            else:
                                category, reason = "Missing output", "No current export job or output TIFF was found."
                            reason = f"{reason} Expected TIFF: {expected_path}"
                            entry["skipped_tiles"].append(dict(tile_id=tile_id, category=category, reason=reason))
                            log_tile_progress(tile_id, entry["skipped_tiles"][-1])
                            continue
                        if read_error is not None:
                            entry["skipped_tiles"].append(dict(tile_id=tile_id, category=_export_problem_category(str(read_error)), reason=str(read_error)))
                            log_tile_progress(tile_id, entry["skipped_tiles"][-1])
                            continue
                        data, probes, probe_error, restored = data
                        entry["native_center_restored"] = entry.get("native_center_restored", 0) + restored
                        gap_probes.extend(probes)
                        gap_probe_errors += int(probe_error is not None)
                        usable = ~np.ma.getmaskarray(data)
                        if not usable.any():
                            entry["skipped_tiles"].append(dict(tile_id=tile_id, category="No overview data", reason=f"No finite cells remain at report resolution ({width} x {height}). A small valid footprint may disappear during resampling. Source TIFF: {uri}"))
                            log_tile_progress(tile_id, entry["skipped_tiles"][-1])
                            continue
                        # A mosaic uses the first valid tile in sorted ID order at overlaps.
                        take = usable & np.ma.getmaskarray(mosaic)
                        if not take.any():
                            entry["skipped_tiles"].append(dict(tile_id=tile_id, category="Covered by earlier tiles", reason=f"All {int(usable.sum())} usable overview cells are already covered by earlier tiles in sorted ID order; the TIFF exists and is readable. Source TIFF: {uri}"))
                            log_tile_progress(tile_id, entry["skipped_tiles"][-1])
                            continue
                        mosaic.data[take] = data.data[take]
                        mosaic.mask[take] = False
                        entry["included_tiles"].append(tile_id)
                        log_tile_progress(tile_id)
                    self.write_message(f"REPORT COVERAGE REPAIR {region}/{role}/{column} | {pair}: "
                                       f"{entry.get('native_center_restored', 0):,} per-tile blank report cells restored from valid native center pixels (may overlap across tiles); stored overviews disabled.", OUTPUTS)
                    # Only count sampled gaps still blank after ALL tiles have been merged.
                    blank = np.ma.getmaskarray(mosaic)
                    gap_cells = {}
                    for row, col, native_valid, nearby_valid, edge in gap_probes:
                        if blank[row, col] and mask[row, col]:
                            previous = gap_cells.get((row, col), (False, False, False))
                            gap_cells[row, col] = tuple(a or b for a, b in zip(previous, (native_valid, nearby_valid, edge)))
                    native_valid_count = sum(item[0] for item in gap_cells.values())
                    nearby_only_count = sum(not item[0] and item[1] for item in gap_cells.values())
                    native_nodata_count = sum(not item[1] for item in gap_cells.values())
                    edge_count = sum(item[2] for item in gap_cells.values())
                    inside_count = int(np.count_nonzero(mask))
                    blank_inside = int(np.count_nonzero(blank & mask))
                    entry["gap_check"] = dict(sampled_blank_cells=len(gap_cells), native_valid=native_valid_count,
                                              native_nodata=native_nodata_count, nearby_only=nearby_only_count,
                                              source_edge_samples=edge_count, blank_inside_mask=blank_inside,
                                              mask_cells=inside_count, failed_tile_checks=gap_probe_errors)
                    self.write_message(
                        f"GAP CHECK {region}/{role}/{column} | {pair}: {blank_inside:,}/{inside_count:,} "
                        f"report cells inside the mask are blank ({100 * blank_inside / max(inside_count, 1):.2f}%). "
                        "This is report-grid coverage, not native missing area.", OUTPUTS)
                    self.write_message(
                        f"GAP SOURCE TEST: {len(gap_cells)} sampled final blanks inside the mask: "
                        f"{native_valid_count} have valid native centers (resampling/overview/grid alignment loss); "
                        f"{nearby_only_count} have blank centers but valid 3x3 neighbors (narrow native gaps or edges); "
                        f"{native_nodata_count} have no valid native 3x3 pixels (source NoData at sampled locations). "
                        f"{edge_count} samples lie within one native pixel of a TIFF edge; "
                        f"{gap_probe_errors} tile checks failed. Samples are not area percentages. "
                        "A missing source outside sampled TIFF footprints is not tested; zero samples is inconclusive.", OUTPUTS)
                    entry["previous_export_tiles"] = [tile for tile in entry["included_tiles"] if audit_by_tile[tile, pair]["status"] != "Exported"]
                    entry["included_count"] = len(entry["included_tiles"])
                    entry["skipped_count"] = len(entry["skipped_tiles"])
                    entry["total_count"] = len(tiles)
                    if mosaic.count():
                        values = mosaic.compressed()
                        lo, hi = float(values.min()), float(values.max())
                        p02, median, p98 = np.percentile(values, [2, 50, 98])
                        zero_count = int(np.count_nonzero(values == 0))
                        entry["value_summary"] = dict(min=lo, max=hi, p02=float(p02), median=float(median),
                                                     p98=float(p98), zero_cells=zero_count, valid_cells=int(values.size))
                        self.write_message(
                            f"VALUE CHECK {region}/{role}/{column} | {pair}: min={lo:g}, p02={p02:g}, "
                            f"median={median:g}, p98={p98:g}, max={hi:g}; zero cells={zero_count:,}/{values.size:,}. "
                            "These are report-resolution mosaic values. A much wider min/max than p02/p98 can compress map colors.", OUTPUTS)
                        del values
                        # Envelope of each pair's percentiles keeps year-pair pages comparable
                        # without allowing isolated extreme values to flatten every map.
                        display_lo, display_hi = (lo, hi) if family in ("survey", "count") else (float(p02), float(p98))
                        previous = ranges.get(family, [display_lo, display_hi])
                        ranges[family] = [min(previous[0], display_lo), max(previous[1], display_hi)]
                    with rasterio.open(root / entry["data_path"], "w", **profile) as output:
                        output.write(mosaic.filled(profile["nodata"]), 1)
                        output.update_tags(region=region, pair=pair, role=role, value_column=column,
                                           included_tiles=entry["included_count"], skipped_tiles=entry["skipped_count"],
                                           resampling="mode" if family in ("survey", "count") else "average",
                                           overlap_rule="first valid tile in sorted tile ID order")
                    entry["skip_categories"] = dict(Counter(item["category"] for item in entry["skipped_tiles"]))
                    entries.append(entry)
                    variable_entries.append(entry)
                    self.write_message(f"Mosaic {region}/{name}: included {entry['included_count']}, skipped {entry['skipped_count']} of {entry['total_count']} tiles.", OUTPUTS)
                    for category, count in sorted(entry["skip_categories"].items()):
                        self.write_message(f"  {name}: {category}: {count} tile(s)", OUTPUTS)

                for checked_entry in variable_entries:
                    gap = checked_entry.get("gap_check")
                    context = f"{region}/{role}/{column} | {checked_entry['pair']}"
                    if gap is None:
                        print(f"PDF GAP SUMMARY {context}: NOT AVAILABLE — no diagnostic result recorded.", flush=True)
                    else:
                        print(f"PDF GAP SUMMARY {context}: sampled={gap['sampled_blank_cells']}; "
                              f"native-valid={gap['native_valid']}; neighbors-only={gap['nearby_only']}; "
                              f"native-3x3-empty={gap['native_nodata']}; TIFF-edge={gap['source_edge_samples']}; "
                              f"failed-checks={gap['failed_tile_checks']}. Full counts are in the PDF summary.", flush=True)
                self.write_message(f"Creating {region}/{role}/{column} PDF from {len(variable_entries)} completed mosaics.", OUTPUTS)
                with PdfPages(root / report_name) as pdf:
                    pdf.infodict().update(Title=f"{region} {role.title()} {REPORT_PRODUCTS[column][0]} - all year pairs", Author="Hydro Health")
                    summary_pages = _render_mosaic_skip_details(pdf, region, variable_entries)
                    for number, entry in enumerate(variable_entries, 1):
                        family = REPORT_PRODUCTS[entry["column"]][1]
                        cmap, norm = _report_color_scale(family, ranges.get(family, [0., 1.]))
                        scale_metadata[scale_key] = dict(family=family, min=float(norm.vmin), max=float(norm.vmax), cmap=cmap.name, method="full_range" if family in ("survey", "count") else "per_pair_p02_p98_envelope")
                        if family == "survey":
                            scale_metadata[scale_key]["classes"] = SURVEY_LABELS
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
                        _render_mosaic_page(pdf, entry, grid, profile, mask, basemap, cmap, norm, region,
                                            number + summary_pages, len(variable_entries) + summary_pages)
                # Close the complete PDF, then publish it before starting the next variable.
                report_path = str(destination / report_name)
                _copy_file(str(root / report_name), report_path)
                report_names.append(report_name)
                self.write_message(f"Published {region}/{role}/{column} PDF: {report_path} ({len(variable_entries)} mosaic pages across all year pairs, plus {summary_pages} count-summary pages).", OUTPUTS)
            inventory = dict(region=region, mask_uri=str(self.template_path),
                             counts_note="Per mosaic: included tiles contribute finite overview cells; skipped tiles are missing, unreadable or contribute no cells. Existing exports count as included.",
                             mask_used_for_export_eligibility=False, mask_validity="report outline only: unmasked, finite, nonzero", mask_outline="nearest sampled at report resolution",
                             scale_note="Each role/variable uses one shared scale across all of its year pairs.",
                             basemap_available=basemap is not None, reports=report_names, export_summary=export_audit["pairs"], color_scales=scale_metadata, mosaics=entries)
            (root / "mosaic_inventory.json").write_text(json.dumps(inventory, indent=2), encoding="utf-8")
            with (root / "mosaic_skips.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["region", "pair", "role", "column", "tile_id", "category", "reason"])
                writer.writeheader()
                for entry in entries:
                    for item in entry["skipped_tiles"]:
                        writer.writerow(dict(region=region, pair=entry["pair"], role=entry["role"], column=entry["column"], **item))
            # PDFs are already published individually; copy remaining supporting files.
            report_set = set(report_names)
            files = sorted(path for path in root.iterdir() if path.name not in report_set)
            for path in files:
                target = destination / path.name
                if target.exists():
                    continue
                _copy_file(str(path), str(target))
        results = [str(destination / name) for name in report_names]
        self.write_message(f"Successfully generated {len(results)} variable PDFs for {region}: {destination}.", OUTPUTS)
        return results


    def execute_tasks(self) -> list[dict]:
        """Use inherited self.overwrite, with jobs bounded by worker/thread capacity."""

        self.write_message(f"Raster export: overwrite={self.overwrite}; existing tile outputs will be {'replaced' if self.overwrite else 'skipped'}. Final variable PDFs will always be replaced.", OUTPUTS)
        jobs = iter(self.task_configs)
        limit = self.settings["raster_workers"] * self.settings["raster_threads_per_worker"]
        pending = set()
        job_by_future = {}
        completed = as_completed([])
        records = []
        submitted = exported = skipped = 0

        def submit_next() -> bool:
            nonlocal submitted
            cfg = next(jobs, None)
            if cfg is None:
                return False
            job = dict(cfg, overwrite=self.overwrite)
            future = self.client.submit(_export_tile_task, job, pure=False)
            pending.add(future)
            job_by_future[future] = job
            completed.add(future)
            submitted += 1
            return True

        try:
            for _ in range(limit):
                if not submit_next():
                    break
                if submitted == 1:
                    self.write_message(f"Raster export: first job submitted; discovering more (in-flight limit {limit}).", OUTPUTS)
            self.write_message(f"Raster export: submitted {submitted}; in flight {len(pending)}/{limit}. More jobs are discovered as slots open.", OUTPUTS)
            for future in completed:
                pending.remove(future)
                try:
                    record = future.result()
                except Exception as exc:
                    cfg = job_by_future[future]
                    reason = f"{type(exc).__name__}: {exc}"
                    self.write_message(f"Export worker failed for {cfg['region']}/{cfg['tile_id']} | {cfg['pair']}: {reason}", OUTPUTS)
                    record = dict(region=cfg["region"], tile_id=cfg["tile_id"], pair=cfg["pair"],
                                  skipped=True, status="Skipped", category="Worker failure", reason=reason,
                                  details={"traceback": traceback.format_exc()}, results={}, visual_outputs=[])
                finally:
                    job_by_future.pop(future, None)
                    future.release()
                records.append(record)
                for note in record.get("notes", []):
                    self.write_message(f"EXPORT NOTE {record['region']}/{record['tile_id']} | {record['pair']}: {note}", OUTPUTS)
                if record.get("skipped", False):
                    skipped += 1
                    reason = "; ".join(" ".join(problem.split()) for problem in _concise_export_problems(record))
                    self.write_message(f"SKIPPED {record['region']}/{record['tile_id']} | {record['pair']} | {record.get('category', 'Raster export problem')}: {reason}", OUTPUTS)
                else:
                    exported += 1
                submit_next()
                self.write_message(f"Raster export: processed {len(records)} ({exported} completed, {skipped} skipped); submitted {submitted}; in flight {len(pending)}/{limit}.", OUTPUTS)
        except BaseException:
            self.client.cancel(list(pending))
            raise
        finally:
            for future in pending:
                future.release()
        self.write_message(f"Raster export finished: {len(records)} total ({exported} completed, {skipped} skipped).", OUTPUTS)
        return records

    def run(self) -> list[dict]:
        """Build reports from existing TIFFs, optionally creating model exports first."""

        records = []
        try:
            settings = self.settings
            if settings["create_model_tifs"] or settings["mosaic_report"] or settings["diagnostic_tile_id"]:
                self.setup_dask(self.param_lookup["env"], processes=settings["raster_processes"],
                                n_workers=settings["raster_workers"], threads_per_worker=settings["raster_threads_per_worker"],
                                memory_limit=settings["memory_limit"])
            for eco_region in self.param_lookup["eco_regions"].value:
                self._resolve_paths(eco_region)
                self.template_metadata = self.load_template_metadata()
                self.report_candidates = {}
                self.report_skip_reasons = {}
                region_records = []
                if settings["create_model_tifs"]:
                    grid = self.load_tile_grid()
                    self.task_configs = self.prepare_tasks(eco_region, grid)
                    self.write_message(f"Discovering and submitting raster exports incrementally for {eco_region}; at most {settings['raster_workers'] * settings['raster_threads_per_worker']} jobs in flight.", OUTPUTS)
                    region_records = self.execute_tasks()
                    records.extend(region_records)
                else:
                    self.write_message(f"{eco_region}: create_model_tifs=False; skipping model TIFF creation and building the PDF from existing TIFFs.", OUTPUTS)
                try:
                    diagnostic = self.generate_tile_diagnostic_report(eco_region)
                    if diagnostic:
                        self.mosaic_reports.append(diagnostic)
                except Exception as exc:
                    self.write_message(f"Tile diagnostic report failed for {eco_region}: {type(exc).__name__}: {exc}", OUTPUTS)
                try:
                    reports = self.generate_mosaic_report(eco_region, region_records)
                    self.mosaic_reports.extend(reports)
                except Exception as exc:
                    self.write_message(f"Mosaic report failed for {eco_region}: {type(exc).__name__}: {exc}", OUTPUTS)
            return records
        finally:
            if self.settings["create_model_tifs"] or self.settings["mosaic_report"] or self.settings["diagnostic_tile_id"]:
                self.cleanup_resources(OUTPUTS)
