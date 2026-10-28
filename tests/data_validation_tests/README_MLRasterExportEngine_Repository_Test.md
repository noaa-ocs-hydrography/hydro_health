# MLRasterExportEngine repository integration test

The final repository-stage update follows the same three-file structure as
DataPrep and XGBoost. It reads your successfully validated model outputs and
writes training/prediction COG rasters, comparison PDFs and diagnostic PNGs.
It does not fit a model, choose predictors, change sample weights, fill missing
values or modify the model tables. Your 1869-round/five-bootstrap result remains
the input, not a setting this engine needs to recreate.

Every function has a purpose docstring. Temporary S3/EC2 replacement sections are
clearly marked; the real repository base `Engine` is retained. All controls are
inside **perhaps move to config file?** in the engine. No extra JSON config,
requirements, support, conversion or R-export scripts, or validation document
are included. Existing R rasters can still be compared without invoking R.

## 1. Place the three files on your integration branch

The ZIP contains repository-relative paths, with **no enclosing package folder**.
Extract to a temporary folder and copy these files into the existing checkout;
do not replace entire src/tests folders.

| File | Action |
|---|---|
| `src/hydro_health/engines/tiling/MLRasterExportEngine.py` | Replace the existing tracked engine |
| `tests/data_validation_tests/ML_Raster_Export_Engine_DV_tests.py` | Add the consolidated test |
| `README_MLRasterExportEngine_Repository_Test.md` | Add this guide |

The test locates the ancestor containing `src/hydro_health`, so moving it into a
nested test folder will not repeat the earlier fixed-parent path problem.
`--repo-root "C:/path/to/hydro_health"` is also available.

## 2. Verify the two spatial assets before running

Your screenshots identify the model-table folders, but do not establish that
`inputs/Master_Grids.gpkg` is identical to the canonical tile grid used in the
successful raster test. This package deliberately does not substitute that file.
Use the **same template and canonical polygon file that produced your successful
local rasters**.

Default repository paths in the grouped settings:

| Setting | Default |
|---|---|
| `template_raster` | `inputs/local_test_inputs/prediction.mask.UTM17_8m.tif` |
| `tile_grid` | `inputs/local_test_inputs/master_grid_canonical.gpkg` |
| `tile_grid_layer` | `None` (default layer; specify the correct name if needed) |
| `tile_id_column` | `tile_id` |

Copy those assets to the two default locations, or change **only those settings**
to their current local/network paths. There is no need to copy a large template
merely to test the export. The original R call supplied these paths; verify they
are still the ones you actually used successfully:

```python
"template_raster": "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif",
"tile_grid": "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/master_grid_canonical.gpkg",
```

Absolute local/network paths and repository-relative paths are supported. The
template must already be north-up EPSG:32617 and cover the complete selected
tile bounds. Its values are not read as an additional mask; its shape/transform
set alignment. No resolution/origin is guessed from point spacing. Polygon CRS
is explicitly transformed to EPSG:32617; exactly one valid polygon must match
BH4S556X_3. Polygon holes are retained. Grid formats include GeoPackage and
GeoParquet; inspect layer names if a GeoPackage contains several layers.

## 3. The current model outputs are the input

Defaults use the existing repository folders:

- `outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3/`
- `outputs/local_test_outputs/Prediction_data_grid_tiles/BH4S556X_3/`

For pair `2004_2006`, the engine reads `LOCAL_training_summary_2004_2006.geoparquet`
and `LOCAL_prediction_summary_2004_2006.geoparquet`, and checks each against its
ordinary `.parquet` companion. It also reads:

- `locked_predictors_2004_2006.parquet`
- `LOCAL_model_metadata_2004_2006.json`
- `LOCAL_test_run_manifest_2004_2006.json`
- `XGBoost_Modeling_Engine_DV_test_report_2004_2006.json`
- The original ML-ready training/prediction tables and DataPrep report whose
  hashes were recorded in that model run.

The model report must be PASS with contract_status PASS and the manifest must
be COMPLETED. Tile, pair, CRS, ordered schema, round/bootstrap counts, summary
row/usability counts and current prepared-input hashes must agree. If DataPrep
has changed since model fitting, rerun the model first. Export does not silently
retrain or substitute earlier copied fixtures.

The schema stays exactly as locked by the model; there is no Boruta text-file
lookup here. Extrapolation counts use those predictors' finite training-summary
ranges and do not treat missing predictor values as outside-range values.

The upstream report remains the authority for saved-Booster replay: this export
stage checks the handoff and rasterizes the **current saved summary values**;
it does not replay Boosters itself or prove new R/Python model parity. Source
hashes are recorded and checked throughout export/revalidation.

## 4. Environment and run command

Use the same dedicated hydro_health environment that just passed the model test.
From the repository root:

```bat
conda activate hydro_health
python -c "import rasterio, matplotlib, cmocean, pypdf; print('Raster and PDF packages available')"
python tests\data_validation_tests\ML_Raster_Export_Engine_DV_tests.py --overwrite
```

The package-check command installs nothing. If it reports a missing module, add
that package to the repository environment using your established package process
and record it in the maintained environment manifest. Dependencies specific to
this stage are Rasterio, Matplotlib, cmocean and pypdf; GeoPandas, pandas, NumPy,
PyArrow and Dask were used by the earlier stages. Pillow is used for PNG checks
and is also a Matplotlib dependency. Do not create a new `.venv` or separately
install `osgeo/gdal` just for this package. Rasterio must expose GDAL's COG driver;
a missing driver causes an explicit failure, not a plain TIFF mislabeled COG.

`--overwrite` replaces **only this pair's raster-stage products**: COGs,
extrapolation diagnostics/ranges, the six known plot products and raster manifest.
This includes stale optional UC/survey rasters or SHAP/gain PNGs that a new run
cannot produce. It leaves all model files, model summary tables, prepared inputs,
schema, model report/manifest, R reference folders and other pairs untouched.
The changed output files must be closed in viewers if Windows reports a file lock.

COG tasks run through one local Dask worker (default 4GB, one thread, processes
on). Matplotlib PDF/PNG rendering runs only in the calling main process. Raster
workers are released after COG creation and before plotting. Temporary folders
are run/task-owned and cleanup tolerates partial setup. The main guard in the
test script is required for Windows worker spawning.

To deliberately debug without Dask:

```bat
python tests\data_validation_tests\ML_Raster_Export_Engine_DV_tests.py --execution inline --overwrite
```

Inline mode does not validate Dask execution. No silent fallback is used.

## 5. Outputs and where to open them

COGs go directly into the same training/prediction folders above, preserving the
names used by the successful local export. Each ends in `_2004_2006.tif`:

| Training prefix | Field |
|---|---|
| LOCAL_Train_Start_Bathy_t | bathy_t |
| LOCAL_Train_Mean_Predicted_Bathy_t1 | mean_predicted_bathy_t1 |
| LOCAL_Train_Surveyed_Change | delta_bathy |
| LOCAL_Train_Mean_Predicted_Change | mean_predicted_change |
| LOCAL_Train_SD_Bathy_t1 | uncertainty_sd_bathy_t1 |
| LOCAL_Train_Mean_Annualized_Rate | mean_predicted_rate |

| Prediction prefix | Field |
|---|---|
| LOCAL_Pred_Start_Bathy_t | bathy_t |
| LOCAL_Pred_Mean_Bathy_t1 | mean_predicted_bathy_t1 |
| LOCAL_Pred_Mean_Change | mean_predicted_change |
| LOCAL_Pred_SD_Bathy_t1 | uncertainty_sd_bathy_t1 |
| LOCAL_Pred_Mean_Annualized_Rate | mean_predicted_rate |
| LOCAL_Predictor_Extrapolation_Count | computed range count |
| LOCAL_Pred_BlueTopo_UC_t | optional supplied UC_t |
| LOCAL_Pred_Survey_Year_Bin | optional survey-date category |

The training folder's **diagnostic_plots** receives:

1. `LOCAL_Summary_Training_vs_Prediction_2004_2006.pdf` - 2 x 4 maps of training and
   prediction starting/fitted bathymetry and change, plus surveyed change/survey year.
2. `LOCAL_Summary_Training_vs_Prediction_Uncertainty_2004_2006.pdf` - R-style layout
   replacing the final panels with training bootstrap SD and BlueTopo UC when
   supplied, otherwise prediction bootstrap SD.
3. `LOCAL_Uncertainty_Comparison_2004_2006.pdf` - training SD, prediction SD and
   BlueTopo UC shown separately. They are different quantities, not interchangeable
   uncertainty estimates.
4. `LOCAL_model_fit_t1_2004_2006.png` - fitting-row actual versus predicted bathymetry;
   these are apparent training metrics, not held-out accuracy.
5. Optional `LOCAL_shap_importance_2004_2006.png` and `LOCAL_gain_importance_2004_2006.png`
   from the saved model diagnostic tables. Absent/empty sources produce explicit
   skipped-plot entries; old plots are removed during deliberate overwrite.

Open that directory in VS Code or your PDF viewer. From the terminal:

```bat
explorer "outputs\local_test_outputs\Training_data_grid_tiles\BH4S556X_3\diagnostic_plots"
```

Open the TIFFs in ArcGIS Pro or QGIS. Full-resolution rasters are retained even
though the PDF is a page-sized display. The PDF reads those saved rasters, not
an independently transformed point table. Related panels share full finite
bathymetry/change scales; change is symmetric about zero. Negative elevation
change is erosion, positive is deposition. SD/UC use a shared meter scale but
remain labeled distinctly. No percentile clipping, calibration, smoothing,
new predictions or uncertainty fusion is added.

Only the map displays mask non-water cells where starting bathymetry is missing
or >=0. Exported rasters retain all finite values allowed by the polygon mask.
Your model report preserved 40672 unusable prediction rows; their derived model
outputs remain NA rather than being filled for the maps. Missing **drivers**
alone do not remove prediction rows; native XGBoost NA handling is retained.

The prediction folder also receives `LOCAL_extrapolation_diagnostics_2004_2006.parquet`
and `LOCAL_predictor_ranges_2004_2006.json`. The training folder receives
`LOCAL_raster_export_manifest_2004_2006.json` and
`ML_Raster_Export_Engine_DV_test_report_2004_2006.json`.

## 6. Validation/replay and year-pair routing

The consolidated test independently groups source points by cell and compares
all populated pixels/NoData masks, full outward-snapped extent, CRS/alignment,
float32 storage, block sizes and COG layout metadata. Continuous duplicate
values are averaged per cell, as in R; survey categories use modal class, with
ties favouring the newer class and mode overviews. Each layer filters only its
own nonfinite X/Y/value rows. No valid pixels for a mandatory product fail with
a tile/template diagnostic; an all-NoData optional layer is written and labeled.

It checks all three single-page PDFs contain the selected pair title and checks
PNG integrity. Completed export manifests fingerprint every generated product.
Revalidate without rewriting rasters or plots:

```bat
python tests\data_validation_tests\ML_Raster_Export_Engine_DV_tests.py --validate-only
```

Changed upstream model/preparation inputs, export settings, geometry/template
metadata or output fingerprints fail replay. The large template is identified
by file size/time plus CRS/dimensions/transform, not by reading all its pixels;
its pixel values are not used. Tolerance/reference-only settings can be changed
for subsequent checks without forcing a new export.

Change `pair` in the grouped class settings after the corresponding validated
model outputs exist. A list such as `["2004_2006", "2006_2010"]` exports each pair
with its own report/PDFs and filenames. This does not add multi-pair training to
the model stage. Scales are shared within a pair's maps, not between different
pairs' PDF files. The survey-bin boundary 2025+ and legacy +800 survey-year repair
are retained from R; `legacy_survey_year_offset` makes that repair explicit.
Numeric years/decimal years and ISO dates are supported; epoch units are not guessed.

## 7. Optional existing R raster comparison

Set `r_reference_dirs.training` and/or `.prediction` in the engine to **separate**
folders containing R rasters with matching product filenames. Revalidate existing
exports with --validate-only; no R process or converter is invoked.

Comparisons require identical CRS, shape, transform and NoData masks. Full raster
values use the explicit `r_comparison_atol`/`rtol` settings (defaults 2e-5/1e-6 for
float32 storage); review those against scientific precision needs. No resampling
is used to make different grids appear equal. Investigate R/Rasterio edge-cell
mask differences rather than relaxing masks silently.

Report statuses are separate: export `contract_status` and optional
`r_raster_comparison`. Missing references are listed as NOT_AVAILABLE, no actual
reference comparison is NOT_RUN, and partial supplied-product coverage is PARTIAL.
A failed R comparison can leave export contract PASS while overall status FAIL.
Only your provided references establish raster parity, and this is not R/Python
model-training parity. Verify that references represent equivalent fitted
summary values; an older R model using another predictor list is not equivalent.

## Changes to review with your developer

Line references below are for the delivered production-path engine; they move
when edited. Keep this file at its existing tracked path for GitHub red/green
review, alongside the new test/README on your integration branch.

1. **Temporary cloud boundary / real base** - `MLRasterExportEngine.py`, lines 29-37; `MLRasterExportEngine.py`, lines 491-525; `MLRasterExportEngine.py`, lines 643-698. Restore S3 configuration, template localization, uploads and region/tile discovery together before production merge. Do not just uncomment imports. The real base Engine is used; no standalone support file is merged.
2. **Grouped settings and explicit assets** - `MLRasterExportEngine.py`, lines 392-419; `MLRasterExportEngine.py`, lines 439-489. Decide later which controls belong in YAML. Confirm the exact successful template/canonical-grid assets; never substitute a similarly named grid or invent a resolution from point spacing.
3. **Model handoff and source identity** - `MLRasterExportEngine.py`, lines 527-615; `MLRasterExportEngine.py`, lines 366-372; `MLRasterExportEngine.py`, lines 375-380. Require matching PASS model report, COMPLETED model manifest, metadata/schema/rounds/counts, and unchanged DataPrep hashes. Record current summary/asset identity and preserve model/preparation files. This is export of saved summary values, not a new saved-Booster replay or accuracy test.
4. **Outward snapping, full extent and finite means** - `MLRasterExportEngine.py`, lines 83-138. Retain floor/ceil endpoints, template-aligned origin, polygon centre masking/holes and per-layer finite means. Do not reduce extent to surviving prediction points or apply an extra template pixel mask. No interpolation/imputation is added.
5. **Genuine COGs and small-grid blocks** - `MLRasterExportEngine.py`, lines 141-157; `MLRasterExportEngine.py`, lines 204-241. Retain GDAL COG CreateCopy and internal overviews; fail if COG support/layout metadata is absent. Block dimensions stay valid multiples of 16 even for tiny grids. Do not restore the original ordinary-TIFF fallback labeled as COG.
6. **Mandatory/optional layers and stale products** - `MLRasterExportEngine.py`, lines 204-241; `MLRasterExportEngine.py`, lines 617-629. Missing mandatory fields or no populated mandatory cells fail, exposing wrong-grid/no-overlap problems. Optional UC/survey layers are reported as absent or all-NoData, never invented. Deliberate overwrite removes stale optional rasters/plots from the same pair only.
7. **Extrapolation and survey dates** - `MLRasterExportEngine.py`, lines 184-201; `MLRasterExportEngine.py`, lines 160-181. Use the exact fitted predictor order and finite training ranges, retaining missing drivers. Keep +800 repair configurable; ISO/null string handling is explicit. Modal categories and mode overviews avoid fractional survey classes; continuous fields retain means.
8. **R-style visual outputs and uncertainty interpretation** - `MLRasterExportEngine.py`, lines 251-363; `MLRasterExportEngine.py`, lines 244-248. Read the saved rasters for all PDF maps, use shared scales and plot-only water masks, and expose training SD, prediction SD and input UC separately. Do not equate bootstrap SD with BlueTopo uncertainty or turn apparent training fit into held-out accuracy.
9. **Dask versus plotting / cleanup** - `MLRasterExportEngine.py`, lines 631-641; `MLRasterExportEngine.py`, lines 643-698. Raster tasks may use the base's local Dask pool; release workers before main-process Matplotlib plotting. Preserve main guards and private temporary paths. Review moving common defensive close_dask into base Engine alongside the previous two engines.
10. **Output protection and manifest replay** - `MLRasterExportEngine.py`, lines 617-629; `MLRasterExportEngine.py`, lines 643-698; `MLRasterExportEngine.py`, lines 375-380. Delete only same-pair raster-stage outputs on explicit overwrite. Fingerprint completed outputs and bind them to inputs/settings; ignore reference/tolerance-only changes when deciding whether rerendering is necessary. Keep model files and other pairs untouched.
11. **CRS and canonical geometry provenance** - `MLRasterExportEngine.py`, lines 491-525; `MLRasterExportEngine.py`, lines 527-615. Validate point CRS/coordinates and explicitly transform only the official polygon. Require the entire tile inside the template and retain its holes. These checks should survive cloud reintegration even though local asset paths are temporary.
12. **Reference comparison scope** - `MLRasterExportEngine.py`, lines 643-698; `MLRasterExportEngine.py`, lines 375-380. Retain independent pixel/reference checks in the consolidated test, with separate export and R statuses. R references must be separate provided products. No raster comparison can establish R/Python fitting parity or compensate for different predictors/models. S3 raster/PDF upload and deployment testing still need your developer's review.


## Software validation completed before delivery

Six synthetic regression tests cover hand-calculated outward snapping/duplicate
means and layer NAs; polygon holes, empty layers and CRS/coverage guards; legacy
and ISO/null dates, categorical ties and finite extrapolation; actual Dask COG
export, source/output fingerprints and replay; absent optional panels and stale
DataPrep detection; alternate pair routing and independent reference-comparison
mechanics, including rejection of changed raster values.

A separate synthetic preview run produced all three PDFs, which were rendered
and visually inspected for alignment, readable titles/legends and consistent
scales. The fixture is explicitly synthetic model-like data, not an independently
trained model or R reference. No real Windows model table/template/grid was
available in this workspace. Your default run uses the actual base Engine;
software tests used the explicit in-script base double and real threaded Dask.
A container-only process-monitoring adapter was needed for tests and is not
included in either delivered script. Windows worker processes remain for your
actual repository integration run.

Tested here with Python 3.12.14, NumPy 2.3.5, pandas 2.2.3 and 3.0.6,
GeoPandas 1.2.0, PyArrow 26.0.0, Rasterio 1.5.2, Dask/distributed
2026.8.0, Matplotlib 3.10.8, cmocean 4.0.3, pypdf 6.10.0 and Pillow
12.3.0. The final six-test suite passed with pandas 3.0.6. Your exact
Windows/Python 3.11 environment and real data still require the repository run.

No GitHub branch, pull request or merge was created by packaging these files.
Review the same tracked engine path on your existing branch before restoring
S3 execution; cloud operation remains the next deployment stage.
