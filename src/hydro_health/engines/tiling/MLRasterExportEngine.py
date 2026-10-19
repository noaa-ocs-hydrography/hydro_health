"""Export validated repository model summaries as aligned COGs and comparison PDFs.

Retain the tested R-style mean rasterization, outward template snapping, polygon
masking, native missing values and diagnostic layouts. Temporary repository IO
replaces S3 routing while the real project Engine and optional Dask tasks remain.
No model fitting, feature selection or conversion is performed. Search
TEMPORARY REPOSITORY TEST before restoring cloud production configuration/IO.
"""
import gc
import hashlib
import json
import math
import re
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.shutil import copy as raster_copy
from shapely import wkb
from shapely.geometry import mapping

from hydro_health.engines.Engine import Engine
# TEMPORARY REPOSITORY TEST: reinstate cloud dependencies/config before production merge.
# import s3fs
# from upath import UPath
# from hydro_health.helpers.tools import get_config_item

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
INPUTS = REPOSITORY_ROOT / "inputs"
OUTPUTS = REPOSITORY_ROOT / "outputs"

TRAIN_PRODUCTS = {
    'bathy_t': 'LOCAL_Train_Start_Bathy_t',
    'mean_predicted_bathy_t1': 'LOCAL_Train_Mean_Predicted_Bathy_t1',
    'delta_bathy': 'LOCAL_Train_Surveyed_Change',
    'mean_predicted_change': 'LOCAL_Train_Mean_Predicted_Change',
    'uncertainty_sd_bathy_t1': 'LOCAL_Train_SD_Bathy_t1',
    'mean_predicted_rate': 'LOCAL_Train_Mean_Annualized_Rate',
}
PRED_PRODUCTS = {
    'bathy_t': 'LOCAL_Pred_Start_Bathy_t',
    'mean_predicted_bathy_t1': 'LOCAL_Pred_Mean_Bathy_t1',
    'mean_predicted_change': 'LOCAL_Pred_Mean_Change',
    'uncertainty_sd_bathy_t1': 'LOCAL_Pred_SD_Bathy_t1',
    'mean_predicted_rate': 'LOCAL_Pred_Mean_Annualized_Rate',
    'predictor_extrapolation_count': 'LOCAL_Predictor_Extrapolation_Count',
    'UC_t': 'LOCAL_Pred_BlueTopo_UC_t',
    'survey_year_bin_code': 'LOCAL_Pred_Survey_Year_Bin',
}
OPTIONAL = {'UC_t', 'survey_year_bin_code'}
SURVEY_LABELS = ['2025+', '2024', '2023', '2022', '2021', '2020',
                 '2015-2019', '2010-2014', '2000-2009', '1990-1999', '<1990']


class LocalizedFile:
    """Keep the production context-manager interface, accepting local files only."""
    def __init__(self, uri, is_aws=False):
        """Keep the production localizer interface but reject cloud URIs in this stage."""
        if is_aws or str(uri).startswith('s3://'):
            raise ValueError('This test package accepts local paths only')
        self.path = str(uri)
        # TEMPORARY REPOSITORY TEST: restore S3 template localization before merge.
        # self.tempdir = tempfile.TemporaryDirectory(prefix="seabed_template_")
        # self.path = str(Path(self.tempdir.name) / "template.tif")
        # s3fs.S3FileSystem().get(self.uri, self.path)

    def __enter__(self):
        """Return the local template without downloading or changing its pixels."""
        return self.path

    def __exit__(self, *args):
        """Leave the caller-owned local file untouched; a restored S3 cache must self-clean."""
        pass


def _dataframe_to_grid(df, value_col, template_uri, is_aws, tile_geom_wkb,
                       crs, nodata=-9999.0, max_grid_cells=25000000):
    """Mean finite point values in each cell, snap outward, and mask cell centres.

    The official polygon determines extent, never the surviving data rows.
    Template values/NoData are not used as a second mask, matching the R writer.
    Each layer filters only its own X/Y/value; driver NAs do not remove rows.
    Survey classes use the modal class per cell (ties favour newer classes).
    """
    geom = wkb.loads(tile_geom_wkb)
    if not geom.is_valid or geom.is_empty or geom.geom_type not in ('Polygon', 'MultiPolygon'):
        raise ValueError('Tile geometry must be a valid nonempty polygon')
    with LocalizedFile(template_uri, is_aws) as path, rasterio.open(path) as template:
        if template.crs != rasterio.crs.CRS.from_user_input(crs):
            raise ValueError('Template CRS must match EPSG:32617; no implicit reprojection')
        t = template.transform
        if t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0:
            raise ValueError('Use a north-up, unrotated template')
        raw = from_bounds(*geom.bounds, transform=t)
        # Snap BOTH endpoints: rounding length alone can lose the final row/column.
        c0, r0 = math.floor(raw.col_off + 1e-8), math.floor(raw.row_off + 1e-8)
        c1 = math.ceil(raw.col_off + raw.width - 1e-8)
        r1 = math.ceil(raw.row_off + raw.height - 1e-8)
        if c0 < 0 or r0 < 0 or c1 > template.width or r1 > template.height:
            raise ValueError('Template must cover the complete tile bounds; refusing silent clipping')
        height, width = r1-r0, c1-c0
        if height <= 0 or width <= 0 or height*width > max_grid_cells:
            raise ValueError('Invalid or oversized tile grid; check template/tile or max_grid_cells')
        transform = template.window_transform(Window(c0, r0, width, height))
    arrays = [pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float) for c in ('X', 'Y', value_col)]
    xs, ys, values = arrays
    keep = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(values)
    rows, cols = rasterio.transform.rowcol(transform, xs[keep], ys[keep])
    rows, cols = np.asarray(rows, dtype=int), np.asarray(cols, dtype=int)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows, cols, vals = rows[inside], cols[inside], values[keep][inside]
    grid = np.full((height, width), nodata, dtype='float32')
    if value_col == 'survey_year_bin_code' and len(vals):
        votes = pd.DataFrame({'r': rows, 'c': cols, 'v': vals}).groupby(['r','c','v']).size().reset_index(name='n')
        winners = votes.sort_values(['n','v'], ascending=[False,True]).drop_duplicates(['r','c'])
        grid[winners.r.to_numpy(), winners.c.to_numpy()] = winners.v.to_numpy()
    elif len(vals):
        accum = np.zeros(grid.shape, dtype='float64')
        counts = np.zeros(grid.shape, dtype='uint32')
        np.add.at(accum, (rows, cols), vals)
        np.add.at(counts, (rows, cols), 1)
        populated = counts > 0
        grid[populated] = (accum[populated]/counts[populated]).astype('float32')
    valid_polygon = geometry_mask([mapping(geom)], out_shape=grid.shape,
                                  transform=transform, invert=True, all_touched=False)
    grid[~valid_polygon] = nodata
    if np.any(grid[valid_polygon] == nodata) and np.any(vals == nodata):
        raise ValueError('A finite source value equals the configured NoData sentinel')
    profile = dict(driver='GTiff', count=1, width=width, height=height,
                   transform=transform, crs=crs, dtype='float32', nodata=nodata)
    return grid, profile


def _build_overviews_and_cog(raw_tif, cog_tif, cfg):
    """Use Rasterio's bundled GDAL COG driver; never label a plain TIFF as a COG."""
    levels = cfg.get('overview_levels', [2,4,8,16,32])
    resampling = cfg.get('overview_resampling', 'average')
    with rasterio.open(raw_tif, 'r+') as dst:
        usable = [int(v) for v in levels if v > 1 and min(dst.width, dst.height)//v >= 1]
        if usable:
            dst.build_overviews(usable, rasterio.enums.Resampling[resampling])
    with rasterio.Env() as env:
        if 'COG' not in env.drivers():
            raise RuntimeError('Installed Rasterio/GDAL lacks the COG driver')
    raster_copy(raw_tif, cog_tif, driver='COG', COMPRESS=cfg.get('compress','DEFLATE'),
                BLOCKSIZE=cfg.get('blocksize',512), BIGTIFF='IF_SAFER',
                OVERVIEWS='FORCE_USE_EXISTING', RESAMPLING=resampling.upper())
    with rasterio.open(cog_tif) as src:
        if src.tags(ns='IMAGE_STRUCTURE').get('LAYOUT') != 'COG':
            raise RuntimeError('Output does not advertise COG layout')


def _survey_codes(series, legacy_offset):
    """Translate the R survey-year bins; optionally preserve its +800 legacy fix.

    YYYY and YYYY.fraction are accepted. ISO strings are decoded as dates.
    Numeric epoch dates are deliberately not guessed.
    """
    year = pd.to_numeric(series, errors='coerce').astype(float)
    if not pd.api.types.is_numeric_dtype(series):
        iso = series.astype(str).str.match(r'^\d{4}-\d{2}-\d{2}', na=False)
        year.loc[iso] = pd.to_datetime(series.loc[iso], errors='coerce').dt.year
    if legacy_offset:
        legacy = (year >= 1000) & (year < 1800)
        year.loc[legacy] += 800
    year = np.floor(year.to_numpy())
    valid = np.isfinite(year) & (year >= 1800) & (year <= 2100)
    code = np.full(len(year), np.nan)
    for i, condition in enumerate([year >= 2025, year == 2024, year == 2023,
        year == 2022, year == 2021, year == 2020,
        (year >= 2015)&(year <= 2019), (year >= 2010)&(year <= 2014),
        (year >= 2000)&(year <= 2009), (year >= 1990)&(year <= 1999), year < 1990]):
        code[valid & condition] = i
    return code


def add_extrapolation_flags(training, prediction, predictors):
    """Replicate R A14 finite training-range checks, with no predictor imputation."""
    out = prediction.copy()
    count = np.zeros(len(out), dtype=int)
    ranges = []
    for p in predictors:
        if p not in training or p not in out:
            raise ValueError(f'Locked model predictor missing from summary: {p}')
        train = pd.to_numeric(training[p], errors='coerce').to_numpy(dtype=float)
        pred = pd.to_numeric(out[p], errors='coerce').to_numpy(dtype=float)
        finite = train[np.isfinite(train)]
        lo, hi = (float(finite.min()),float(finite.max())) if len(finite) else (None,None)
        if len(finite):
            count += np.isfinite(pred) & ((pred < lo) | (pred > hi))
        ranges.append({'predictor':p, 'min_training':lo, 'max_training':hi})
    out['predictor_extrapolation_count'] = count
    out['predictor_extrapolation_any'] = count > 0
    return out, ranges


def _export_cog_task(params):
    """Keep the worker-task boundary: export one role/table and return an audit."""
    df, role, pair, template, out_dir, geometry, crs, cfg = params
    Engine.write_message_dask(f"Exporting {role} COG rasters | {pair}", out_dir)
    products = TRAIN_PRODUCTS if role == 'training' else PRED_PRODUCTS
    outputs, skipped = {}, []
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hh_raster_') as tmp:
        for column, prefix in products.items():
            if column not in df:
                if column in OPTIONAL:
                    skipped.append({'column':column,'reason':'not supplied by model summary'})
                    continue
                raise ValueError(f'Required {role} output column is absent: {column}')
            grid, profile = _dataframe_to_grid(df, column, template, False, geometry, crs,
                cfg.get('nodata',-9999.0), cfg.get('max_grid_cells',25000000))
            if column not in OPTIONAL and not np.any(grid != profile['nodata']):
                raise ValueError(f'No populated {role}/{column} cells; check canonical grid/template and summary coordinates')
            raw, cog = Path(tmp)/'raw.tif', Path(tmp)/'cog.tif' 
            # GeoTIFF block dimensions must be multiples of 16, including tiny grids.
            profile.update(tiled=True, blockxsize=cfg.get('blocksize',512),
                           blockysize=cfg.get('blocksize',512), compress=cfg.get('compress','DEFLATE'))
            with rasterio.open(raw,'w',**profile) as dst:
                dst.write(grid,1)
                dst.update_tags(value_column=column, pair=pair, role=role,
                    aggregation='mode' if column == 'survey_year_bin_code' else 'mean',
                    survey_classes=json.dumps(SURVEY_LABELS) if column == 'survey_year_bin_code' else '')
            product_cfg = dict(cfg)
            if column == 'survey_year_bin_code':
                product_cfg['overview_resampling'] = 'mode'
            _build_overviews_and_cog(raw,cog,product_cfg)
            final = Path(out_dir)/f'{prefix}_{pair}.tif'
            # TEMPORARY REPOSITORY TEST: delete local copy and reinstate S3 upload before merge.
            # s3fs.S3FileSystem().put(str(cog), final_uri)
            shutil.copyfile(cog,final)
            outputs[column] = str(final)
    gc.collect()
    return {'ok':True,'role':role,'pair':pair,'outputs':outputs,'skipped':skipped}


def _read_layer(path):
    """Read one completed raster for plotting; honour the saved NoData mask."""
    with rasterio.open(path) as src:
        data = src.read(1,masked=True).astype(float)
        return data, (src.bounds.left,src.bounds.right,src.bounds.bottom,src.bounds.top)


def _make_visual_outputs(train, prediction, results, train_out, pair, tile_id):
    """Restore A15 side-by-side PDFs and fitting/SHAP PNGs from saved rasters."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm, BoundaryNorm
    from matplotlib.patches import Patch
    import cmocean
    plot_dir = Path(train_out)/'diagnostic_plots'
    plot_dir.mkdir(exist_ok=True)
    layers = {}
    for role, result in results.items():
        for col,path in result['outputs'].items():
            layers[role,col] = _read_layer(path)
    # Plot-only water mask, as in R A15. Exported rasters retain all valid cells.
    for role in ('training','prediction'):
        water = layers[role,'bathy_t'][0]
        for (r,col),(data,extent) in list(layers.items()):
            if r == role:
                layers[r,col] = (np.ma.masked_where(np.ma.getmaskarray(water) | (water.filled(0)>=0),data),extent)
    def finite_values(keys):
        """Collect finite water-cell values so related panels use the same scale."""
        parts = [layers[k][0].compressed() for k in keys if k in layers]
        v = np.concatenate(parts) if parts else np.array([])
        return v[np.isfinite(v)]
    depths = finite_values([(r,c) for r in results for c in ('bathy_t','mean_predicted_bathy_t1')])
    changes = finite_values([(r,'mean_predicted_change') for r in results]+[('training','delta_bathy')])
    sd = finite_values([(r,'uncertainty_sd_bathy_t1') for r in results]+[('prediction','UC_t')])
    lo,hi = (float(depths.min()),float(depths.max())) if len(depths) else (-1.,0.)
    if hi <= lo: hi = lo+1
    dmax = max(float(np.abs(changes).max()) if len(changes) else 1.,1e-8)
    smax = max(float(sd.max()) if len(sd) else 1.,1e-8)
    norms = {'depth':Normalize(lo,hi), 'change':TwoSlopeNorm(0,-dmax,dmax),
             'uncertainty':Normalize(0,smax),'survey':BoundaryNorm(np.arange(-.5,11.5),11)}
    cmaps = {'depth':'viridis','change':cmocean.cm.curl,'uncertainty':'viridis',
             'survey':plt.get_cmap('viridis',11)}
    def panel(ax,role,col,title,kind):
        """Draw a saved raster with its shared normalization or a clearly labeled no-data panel."""
        ax.set_title(title,fontsize=10,fontweight='bold',pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        if (role,col) not in layers or not layers[role,col][0].count():
            ax.text(.5,.5,'No data supplied',ha='center',va='center',transform=ax.transAxes)
            return
        data,extent = layers[role,col]
        artist = ax.imshow(data,extent=extent,origin='upper',cmap=cmaps[kind],
                           norm=norms[kind],interpolation='nearest')
        ax.set_aspect('equal')
        if kind == 'survey':
            codes = sorted(set(data.compressed().astype(int)))
            ax.legend(handles=[Patch(color=cmaps[kind](i),label=SURVEY_LABELS[i]) for i in codes],
                      loc='lower left',fontsize=7,framealpha=.9)
        else:
            label = {'depth':'Bathymetry (m)','change':'Change (m)\n- erosion / + deposition',
                     'uncertainty':'UC (m)' if col == 'UC_t' else 'Bootstrap SD (m)'}[kind]
            bar = ax.figure.colorbar(artist,ax=ax,shrink=.72,pad=.025)
            bar.set_label(label,fontsize=8); bar.ax.tick_params(labelsize=7)
    common = [('training','bathy_t','Train start bathymetry','depth'),
              ('training','mean_predicted_bathy_t1','Train mean prediction (t1)','depth'),
              ('training','mean_predicted_change','Train predicted change [t1 - t]','change'),
              ('training','delta_bathy','Train surveyed change [t1 - t]','change'),
              ('prediction','bathy_t','Prediction start bathymetry','depth'),
              ('prediction','mean_predicted_bathy_t1','Prediction mean bathymetry (t1)','depth'),
              ('prediction','mean_predicted_change','Prediction change [t1 - t]','change'),
              ('prediction','survey_year_bin_code','Prediction survey end year','survey')]
    files = []
    for uncertainty in (False,True):
        spec = list(common)
        if uncertainty:
            spec[3] = ('training','uncertainty_sd_bathy_t1','Train bootstrap uncertainty','uncertainty')
            col = 'UC_t' if ('prediction','UC_t') in layers else 'uncertainty_sd_bathy_t1'
            spec[7] = ('prediction',col,'BlueTopo uncertainty (UC_t)' if col == 'UC_t' else 'Prediction bootstrap uncertainty','uncertainty')
        fig, axes = plt.subplots(2,4,figsize=(20,10),layout='constrained')
        for ax, args in zip(axes.flat,spec): panel(ax,*args)
        fig.suptitle(f'{tile_id} | {pair} | Training and prediction'+(' - uncertainty' if uncertainty else ''),fontsize=16)
        name = f'LOCAL_Summary_Training_vs_Prediction{"_Uncertainty" if uncertainty else ""}_{pair}.pdf'
        file = plot_dir/name; fig.savefig(file,dpi=180); plt.close(fig); files.append(str(file))
    # Additional page exposes prediction MODEL SD even when the R panel chooses UC.
    fig, axes = plt.subplots(1,3,figsize=(15,6),layout='constrained')
    panel(axes[0],'training','uncertainty_sd_bathy_t1','Training bootstrap SD','uncertainty')
    panel(axes[1],'prediction','uncertainty_sd_bathy_t1','Prediction bootstrap SD','uncertainty')
    panel(axes[2],'prediction','UC_t','BlueTopo uncertainty (UC_t)','uncertainty')
    fig.suptitle(f'{tile_id} | {pair} | Model SD and input UC are different quantities',fontsize=13)
    file = plot_dir/f'LOCAL_Uncertainty_Comparison_{pair}.pdf'
    fig.savefig(file,dpi=180); plt.close(fig); files.append(str(file))
    actual = pd.to_numeric(train.bathy_t1,errors='coerce').to_numpy(float)
    fitted = pd.to_numeric(train.mean_predicted_bathy_t1,errors='coerce').to_numpy(float)
    keep = np.isfinite(actual)&np.isfinite(fitted)
    if keep.sum() > 2:
        a,p = actual[keep],fitted[keep]
        rmse = np.sqrt(np.mean((p-a)**2)); denom = np.sum((a-a.mean())**2)
        r2 = 1-np.sum((p-a)**2)/denom if denom else np.nan
        fig,ax = plt.subplots(figsize=(7,7),layout='constrained')
        ax.scatter(a,p,s=2,alpha=.15,rasterized=True)
        bounds = [min(a.min(),p.min()),max(a.max(),p.max())]
        ax.plot(bounds,bounds,'r--'); ax.set_aspect('equal')
        ax.set(xlabel='Surveyed bathymetry (t1, m)',ylabel='Mean predicted bathymetry (t1, m)',
               title=f'{tile_id} | {pair}\nApparent training fit: RMSE={rmse:.3f} m; predictive R²={r2:.3f}')
        file = plot_dir/f'LOCAL_model_fit_t1_{pair}.png'
        fig.savefig(file,dpi=180); plt.close(fig); files.append(str(file))
    for source,column,label,name in [('shap_summary','mean_abs_shap','Mean absolute SHAP (m)','shap_importance'),
                                     ('importance','gain_fraction','Mean gain fraction','gain_importance')]:
        path = Path(train_out)/f'LOCAL_{source}_{pair}.parquet'
        if not path.exists(): continue
        data = pd.read_parquet(path)
        if data.empty or column not in data: continue
        values = data.groupby('predictor')[column].mean().dropna().nlargest(20).sort_values()
        if values.empty: continue
        fig,ax = plt.subplots(figsize=(8,max(5,.3*len(values)+2)),layout='constrained')
        ax.barh(values.index,values.values,color='steelblue')
        ax.set(xlabel=label,title=f'{tile_id} | {pair} | {name.replace("_"," ")}')
        file = plot_dir/f'LOCAL_{name}_{pair}.png'
        fig.savefig(file,dpi=180); plt.close(fig); files.append(str(file))
    return files


def _file_hash(path):
    """Fingerprint immutable input tables/reports so export and replay cannot mix runs."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _export_contract(cfg, identity, predictors):
    """Bind export-affecting settings while allowing later reference/tolerance checks."""
    validation_only = {"r_reference_dirs", "r_comparison_atol", "r_comparison_rtol",
                       "pixel_check_atol", "pixel_check_rtol"}
    return {"identity":identity,"configuration":{k:v for k,v in cfg.items() if k not in validation_only},
            "predictors":predictors}


class MLRasterExportEngine(Engine):
    """Production-shaped final stage with explicit repository-local input and output controls."""

    def __init__(self, param_lookup: dict | None = None, output_prefix: str | bool = False):
        """Keep all temporary routing and export settings together, using the real base Engine."""
        super().__init__()
        self.param_lookup = param_lookup or {}
        self.output_prefix = output_prefix
        self.repository_root = Path(self.param_lookup.get("repository_root", REPOSITORY_ROOT)).resolve()
        # --- perhaps move to config file? ---
        self.settings = {
            "tile_id": "BH4S556X_3",
            "pair": "2004_2006",  # A list also works once each model pair has been validated.
            "region": "ER_3",
            "crs": "EPSG:32617",
            "training_output_dir": "outputs/local_test_outputs/Training_data_grid_tiles/{tile_id}",
            "prediction_output_dir": "outputs/local_test_outputs/Prediction_data_grid_tiles/{tile_id}",
            "template_raster": "inputs/local_test_inputs/prediction.mask.UTM17_8m.tif",
            "tile_grid": "inputs/local_test_inputs/master_grid_canonical.gpkg",
            "tile_grid_layer": None,
            "tile_id_column": "tile_id",
            "legacy_survey_year_offset": True,
            "raster_workers": 1,
            "raster_memory_limit": "4GB",
            "raster_processes": True,
            "raster_config": {
                "nodata": -9999.0, "compress": "DEFLATE", "blocksize": 512,
                "overview_levels": [2, 4, 8, 16, 32], "overview_resampling": "average",
                "max_grid_cells": 25000000,
            },
            "pixel_check_atol": 0.00002,
            "pixel_check_rtol": 0.000001,
            "r_reference_dirs": {"training": None, "prediction": None},
            "r_comparison_atol": 0.00002,
            "r_comparison_rtol": 0.000001,
        }
        # --- end perhaps move to config file? ---
        import copy
        overrides = copy.deepcopy(self.param_lookup.get("raster_settings", {}))
        for key in ("raster_config", "r_reference_dirs"):
            self.settings[key].update(overrides.pop(key, {}))
        self.settings.update(overrides)
        # TEMPORARY REPOSITORY TEST: remove local-only guard with restored S3 routing before merge.
        # self.is_aws = self.param_lookup.get('env', 'local') in ['remote', 'aws']
        if self.param_lookup.get("env", "local") != "local":
            raise ValueError("S3 execution is temporarily disabled for repository testing")
        self.is_aws = False
        self.outputs_dir = self.repository_root / "outputs/local_test_outputs"

    def _repository_path(self, value, pair):
        """Resolve explicit asset/output paths from the checkout and configured tile/pair."""
        if not value or str(value).startswith("s3://"):
            raise ValueError("Supply an explicit local path; S3 is disabled in this stage")
        path = Path(str(value).format(tile_id=self.settings["tile_id"], pair=pair))
        return str(path if path.is_absolute() else self.repository_root / path)

    def configurations(self, pair=None):
        """Build per-pair routes from one grouped setting block, with no JSON test config."""
        import copy
        pairs = self.settings["pair"] if isinstance(self.settings["pair"], list) else [self.settings["pair"]]
        if pair is not None:
            if pair not in pairs:
                raise ValueError("Requested pair is not in the engine settings")
            pairs = [pair]
        if not pairs or len(pairs) != len(set(pairs)):
            raise ValueError("Configure a nonempty list of unique year pairs")
        configurations = []
        for label in pairs:
            match = re.fullmatch(r"(\d{4})_(\d{4})", label)
            if not match or int(match[2]) <= int(match[1]):
                raise ValueError("pair must be an increasing YYYY_YYYY label")
            cfg = copy.deepcopy(self.settings); cfg["pair"] = label
            if cfg["crs"] != "EPSG:32617":
                raise ValueError("This integration stage requires EPSG:32617")
            for key in ("training_output_dir", "prediction_output_dir", "template_raster", "tile_grid"):
                cfg[key] = self._repository_path(cfg[key], label)
            train = Path(cfg["training_output_dir"])
            pred = Path(cfg["prediction_output_dir"])
            cfg.update(training_summary=str(train/f"LOCAL_training_summary_{label}.geoparquet"),
                prediction_summary=str(pred/f"LOCAL_prediction_summary_{label}.geoparquet"),
                locked_predictors=str(train/f"locked_predictors_{label}.parquet"),
                model_metadata=str(train/f"LOCAL_model_metadata_{label}.json"),
                model_report=str(train/f"XGBoost_Modeling_Engine_DV_test_report_{label}.json"),
                model_manifest=str(train/f"LOCAL_test_run_manifest_{label}.json"))
            for role, directory in cfg["r_reference_dirs"].items():
                if directory:
                    cfg["r_reference_dirs"][role] = self._repository_path(directory, label)
            raster = cfg["raster_config"]
            block = raster["blocksize"]
            if not isinstance(block, int) or block < 128 or block % 16:
                raise ValueError("COG blocksize must be an integer >=128 and divisible by 16")
            if not np.isfinite(raster["nodata"]) or float(np.float32(raster["nodata"])) != raster["nodata"]:
                raise ValueError("NoData must be finite and exactly representable as float32")
            if not isinstance(raster["max_grid_cells"],int) or raster["max_grid_cells"] < 1:
                raise ValueError("max_grid_cells must be a positive integer")
            if raster["overview_resampling"] not in rasterio.enums.Resampling.__members__:
                raise ValueError("Invalid overview resampling method")
            levels = raster["overview_levels"]
            if levels != sorted(set(levels)) or any(not isinstance(v,int) or v < 2 for v in levels):
                raise ValueError("overview_levels must be distinct increasing integers >=2")
            for key in ("pixel_check_atol","pixel_check_rtol","r_comparison_atol","r_comparison_rtol"):
                if not np.isfinite(cfg[key]) or cfg[key] < 0:
                    raise ValueError(f"{key} must be finite and nonnegative")
            if cfg["raster_workers"] != 1:
                raise ValueError("Use one worker for this initial single-tile integration stage")
            configurations.append(cfg)
        return configurations

    def _resolve_paths(self, cfg):
        """Use explicit repository tables/assets and the authoritative tile polygon.

        DELETE TEMPORARY REPOSITORY ROUTING BEFORE MERGING TO PRODUCTION;
        restore S3 configuration, localization and tile/region discovery together.
        """
        self.training_out = Path(cfg["training_output_dir"])
        self.prediction_out = Path(cfg["prediction_output_dir"])
        self.outputs_dir = self.training_out
        self.training_out.mkdir(parents=True,exist_ok=True)
        self.prediction_out.mkdir(parents=True,exist_ok=True)
        for key in ("template_raster", "tile_grid"):
            if not Path(cfg[key]).is_file():
                raise FileNotFoundError(f"{key}: {cfg[key]}. Copy the same asset used in the successful local export, or edit this class setting.")
        # CLOUD resolver/localizer to reinstate before production merge:
        # bucket = get_config_item('S3', 'BUCKET_NAME')
        # base = f"s3://{bucket}/{region}"
        # self.template_uri = f"{base}/{get_config_item('MODEL','TEMPLATE_RASTER_URI')}"
        # grid_uri = f"{base}/{get_config_item('MODEL','PREDICTION_SUB_GRIDS')}"
        # self.raster_out_dir = UPath(f"{base}/{get_config_item('MODEL','RASTER_OUTPUT_DIR')}")
        if Path(cfg["tile_grid"]).suffix.lower() in (".parquet", ".geoparquet"):
            grid = gpd.read_parquet(cfg["tile_grid"])
        else:
            kwargs = {"layer":cfg["tile_grid_layer"]} if cfg["tile_grid_layer"] else {}
            grid = gpd.read_file(cfg["tile_grid"],**kwargs)
        field = cfg["tile_id_column"]
        if field not in grid or grid.crs is None:
            raise ValueError("Canonical tile grid needs a CRS and configured tile ID column")
        selected = grid.loc[grid[field].astype(str) == str(cfg["tile_id"])]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one polygon for {cfg['tile_id']}; found {len(selected)}")
        geom = selected.to_crs(cfg["crs"]).geometry.iloc[0]
        if geom is None or geom.is_empty or not geom.is_valid or geom.geom_type not in ("Polygon","MultiPolygon"):
            raise ValueError("Use a valid nonempty canonical polygon, including its holes")
        self.geometry = geom.wkb

    def load_inputs(self, cfg):
        """Bind saved summaries to the exact successful model/schema and unchanged DataPrep inputs.

        No Booster replay or fitting is performed here. The upstream model report
        remains the authority for those tests; this stage checks its handoff and
        converts the current saved summary values without changing source files.
        """
        self._resolve_paths(cfg)
        train_dir = Path(cfg["training_output_dir"]); pred_dir = Path(cfg["prediction_output_dir"])
        sources = {key:Path(cfg[key]) for key in ("training_summary","prediction_summary","locked_predictors",
                   "model_metadata","model_report","model_manifest","tile_grid")}
        for role, directory in (("training",train_dir),("prediction",pred_dir)):
            sources[f"{role}_ordinary"] = directory/f"LOCAL_{role}_summary_{cfg['pair']}.parquet"
        prepared = {"training_geoparquet": train_dir/f"ML_ready_training_{cfg['pair']}.geoparquet",
                    "prediction_geoparquet": pred_dir/f"ML_ready_prediction_{cfg['pair']}.geoparquet",
                    "locked_predictors_parquet": Path(cfg["locked_predictors"]),
                    "data_prep_report": train_dir/f"Data_Prep_Engine_Unit_Test_report_{cfg['pair']}.json"}
        sources.update({f"prepared_{k}":p for k,p in prepared.items()})
        for name in ("shap_summary","importance"):
            path = train_dir/f"LOCAL_{name}_{cfg['pair']}.parquet"
            if path.exists(): sources[name] = path
        for path in sources.values():
            if not path.is_file(): raise FileNotFoundError(f"Required model handoff input missing: {path}")
        identity = {key:{"path":str(path),"sha256":_file_hash(path)} for key,path in sources.items()}
        report = json.loads(Path(cfg["model_report"]).read_text())
        meta = json.loads(Path(cfg["model_metadata"]).read_text())
        manifest = json.loads(Path(cfg["model_manifest"]).read_text())
        predictors = pd.read_parquet(cfg["locked_predictors"]).predictor.tolist()
        for document in (report,meta):
            if any(document.get(k) != cfg[k] for k in ("tile_id","pair","crs")):
                raise ValueError("Model report/metadata identity does not match this tile, pair and CRS")
        if report.get("status") != "PASS" or report.get("contract_status") != "PASS":
            raise ValueError("Run the repository XGBoost test successfully before raster export")
        if manifest.get("model_execution_status") != "COMPLETED":
            raise ValueError("Model execution manifest is not COMPLETED")
        contract = manifest.get("contract",{})
        if predictors != meta.get("predictors") or predictors != report.get("schema") or predictors != contract.get("schema"):
            raise ValueError("Locked schema, fitted model, test report and manifest disagree")
        if report.get("input_sha256") != contract.get("identity") or report.get("model_config") != contract.get("model_config"):
            raise ValueError("Model report and completed run manifest describe different inputs/settings")
        for key,path in prepared.items():
            if identity[f"prepared_{key}"]["sha256"] != contract.get("identity",{}).get(key):
                raise ValueError(f"DataPrep changed after the model run: {path}; rerun the model")
        if meta.get("target") != "bathy_t1" or meta.get("prediction_baseline_source") != "bathy_t":
            raise ValueError("Expected fitted bathy_t1 target and bathy_t prediction baseline")
        if meta.get("n_boot") != report.get("validation",{}).get("n_boot") or meta.get("best_iteration") != report.get("validation",{}).get("best_nrounds"):
            raise ValueError("Model metadata rounds/bootstrap count differs from the PASS report")
        frames = {}
        required = {"training":set(TRAIN_PRODUCTS)|{"X","Y","FID","bathy_t1"},
                    "prediction":(set(PRED_PRODUCTS)-OPTIONAL-{"predictor_extrapolation_count"})|{"X","Y","FID","prediction_row_usable"}}
        for role in ("training","prediction"):
            spatial = gpd.read_parquet(cfg[f"{role}_summary"])
            if spatial.crs is None or spatial.crs.to_epsg() != 32617:
                raise ValueError(f"{role} summary must already use EPSG:32617")
            df = pd.DataFrame(spatial.drop(columns=spatial.geometry.name)).reset_index(drop=True)
            pd.testing.assert_frame_equal(df,pd.read_parquet(sources[f"{role}_ordinary"]).reset_index(drop=True),check_dtype=False,check_exact=True)
            if missing := (required[role]|set(predictors))-set(df.columns):
                raise ValueError(f"Missing required {role} summary fields: {sorted(missing)}")
            if len(df) != report["validation"]["summaries"][role]["rows"]:
                raise ValueError(f"{role} summary row count differs from the PASS model report")
            for col, expected in (("tile_id",cfg["tile_id"]),("pair_id",cfg["pair"])):
                if col not in df or df[col].isna().any() or not df[col].eq(expected).all():
                    raise ValueError(f"{role} summary {col} differs from the configured identity")
            if df.FID.isna().any() or df.FID.duplicated().any():
                raise ValueError("Summary FIDs must remain unique and nonmissing")
            x = pd.to_numeric(df.X,errors="coerce"); y = pd.to_numeric(df.Y,errors="coerce")
            located = np.isfinite(x)&np.isfinite(y)
            np.testing.assert_array_equal(spatial.geometry.isna(),~located)
            points = spatial.loc[located]
            if not points.geometry.geom_type.eq("Point").all(): raise ValueError("Summary geometry must be points")
            np.testing.assert_allclose(points.geometry.x,x[located],rtol=0,atol=1e-7)
            np.testing.assert_allclose(points.geometry.y,y[located],rtol=0,atol=1e-7)
            usable = np.ones(len(df),dtype=bool) if role == "training" else df.prediction_row_usable.to_numpy(dtype=bool)
            if int(usable.sum()) != report["validation"]["summaries"][role]["usable"]:
                raise ValueError("Prediction usability changed after model validation")
            if not usable.any() or not np.isfinite(df.loc[usable,["mean_predicted_bathy_t1","mean_predicted_change","mean_predicted_rate","uncertainty_sd_bathy_t1"]].to_numpy(dtype=float)).all():
                raise ValueError("Usable summary rows must contain finite model outputs")
            frames[role] = df
        prediction,ranges = add_extrapolation_flags(frames["training"],frames["prediction"],predictors)
        if "survey_end_date" in prediction:
            prediction["survey_year_bin_code"] = _survey_codes(prediction.survey_end_date,cfg["legacy_survey_year_offset"])
        with rasterio.open(cfg["template_raster"]) as template:
            t = template.transform
            template_identity = {"path":cfg["template_raster"],"size":Path(cfg["template_raster"]).stat().st_size,
                "mtime_ns":Path(cfg["template_raster"]).stat().st_mtime_ns,"crs":str(template.crs),
                "transform":list(t),"width":template.width,"height":template.height}
        identity["template"] = template_identity
        identity["geometry_wkb_sha256"] = hashlib.sha256(self.geometry).hexdigest()
        return frames["training"],prediction,predictors,ranges,identity,report

    def _generated_output_paths(self, cfg):
        """Enumerate only this pair's raster-stage artifacts, including optional stale products."""
        pair = cfg["pair"]; train = Path(cfg["training_output_dir"]); pred = Path(cfg["prediction_output_dir"])
        files = [folder/f"{prefix}_{pair}.tif" for folder, products in ((train,TRAIN_PRODUCTS),(pred,PRED_PRODUCTS)) for prefix in products.values()]
        files += [pred/f"LOCAL_extrapolation_diagnostics_{pair}.parquet", pred/f"LOCAL_predictor_ranges_{pair}.json"]
        plots = train/"diagnostic_plots"
        files += [plots/f"{prefix}_{pair}{suffix}" for prefix,suffix in (
            ("LOCAL_Summary_Training_vs_Prediction",".pdf"),
            ("LOCAL_Summary_Training_vs_Prediction_Uncertainty",".pdf"),
            ("LOCAL_Uncertainty_Comparison",".pdf"), ("LOCAL_model_fit_t1",".png"),
            ("LOCAL_shap_importance",".png"),("LOCAL_gain_importance",".png"))]
        files += [train/f"LOCAL_raster_export_manifest_{pair}.json"]
        return files

    def close_dask(self):
        """Retire allocated workers before closing the scheduler, tolerating partial setup."""
        try:
            if self.client is not None and self.client.status != "closed":
                try:
                    workers = list(self.client.scheduler_info().get("workers",{}))
                    if workers: self.client.retire_workers(workers=workers,close_workers=True)
                finally:
                    self.client.close()
        finally:
            if self.cluster is not None: self.cluster.close()

    def run(self, overwrite=False, execution="dask", pair=None):
        """Export roles through Dask, then build PDFs only in the caller's main process.

        Source model/DataPrep tables are immutable. Overwrite removes only this
        pair's raster-stage products, including stale optional plots/rasters.
        """
        if execution not in ("dask","inline"): raise ValueError("execution must be dask or inline")
        records = []
        for cfg in self.configurations(pair):
            train,pred,predictors,ranges,identity,model_report = self.load_inputs(cfg)
            generated = self._generated_output_paths(cfg)
            if not overwrite and any(path.exists() for path in generated):
                raise FileExistsError("Raster/visual outputs already exist; use --overwrite deliberately")
            protected = {Path(v["path"]).resolve() for v in identity.values() if isinstance(v,dict) and "path" in v}
            if any(path.resolve() in protected for path in generated):
                raise ValueError("Raster output would overwrite a protected model/preparation input")
            if overwrite:
                for path in generated: path.unlink(missing_ok=True)
            self.local_tmp_dir = Path(tempfile.mkdtemp(prefix="hhm_raster_repository_"))
            manifest_path = self.training_out/f"LOCAL_raster_export_manifest_{cfg['pair']}.json"
            contract = _export_contract(cfg,identity,predictors)
            manifest_path.write_text(json.dumps({"status":"STARTED","contract":contract},indent=2))
            tasks = [[df,role,cfg["pair"],cfg["template_raster"],folder,self.geometry,cfg["crs"],cfg["raster_config"]]
                for role,df,folder in (("training",train,self.training_out),("prediction",pred,self.prediction_out))]
            try:
                print(f"Exporting rasters/PDFs | {cfg['tile_id']} | {cfg['pair']} | {execution}",flush=True)
                if execution == "dask":
                    self.setup_dask("local",processes=cfg["raster_processes"],n_workers=cfg["raster_workers"],
                                    threads_per_worker=1,memory_limit=cfg["raster_memory_limit"])
                    task_results = self.client.gather(self.client.map(_export_cog_task,tasks))
                else:
                    task_results = [_export_cog_task(task) for task in tasks]
                if execution == "dask":
                    self.close_dask()  # Release raster workers before main-process PDF rendering.
                results = {result["role"]:result for result in task_results}
                if set(results) != {"training","prediction"} or not all(r["ok"] for r in results.values()):
                    raise RuntimeError("Not all raster export tasks completed")
                pred[["X","Y","predictor_extrapolation_count","predictor_extrapolation_any"]].to_parquet(
                    self.prediction_out/f"LOCAL_extrapolation_diagnostics_{cfg['pair']}.parquet",index=False)
                (self.prediction_out/f"LOCAL_predictor_ranges_{cfg['pair']}.json").write_text(json.dumps(ranges,indent=2))
                # Matplotlib remains in the main process; workers only rasterize/write COGs.
                visuals = _make_visual_outputs(train,pred,results,self.training_out,cfg["pair"],cfg["tile_id"])
                record = {"pair":cfg["pair"],"results":results,"visual_outputs":visuals,
                          "skipped_plots":[str(path) for path in generated if path.suffix == ".png" and not path.exists()]}
                # Re-read hashes/asset geometry to reject mixed exports if upstream data changed.
                current = self.load_inputs(cfg)[4]
                if current != identity: raise ValueError("Source inputs or template/grid changed during export")
                record["output_sha256"] = {str(path):_file_hash(path) for path in generated if path != manifest_path and path.is_file()}
                manifest_path.write_text(json.dumps({"status":"COMPLETED","contract":contract,"exports":record},indent=2))
                records.append(record)
            except Exception as exc:
                manifest_path.write_text(json.dumps({"status":"FAILED","contract":contract,"error":str(exc)},indent=2))
                raise
            finally:
                self.cleanup_resources(self.outputs_dir)
        return records
