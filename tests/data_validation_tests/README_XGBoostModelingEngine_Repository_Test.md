# XGBoostModelingEngine repository integration test

This is the same three-file structure as the successful DataPrep repository
update. It carries forward the tested model logic and output checks, uses the
real project base `Engine`, and temporarily reads repository-local DataPrep
outputs while the production S3 inputs are unavailable.

No separate JSON config, support module, conversion script, requirements file,
inspection helper, validation document or `.diff` file is needed. Every function
has a purpose docstring. Settings are grouped inside the engine class under
**perhaps move to config file?** for later decisions with your developer.

## 1. Place the three files

The ZIP has repository-relative paths with **no enclosing package folder**.
Extract to a temporary folder, then copy to the matching locations in your
existing checkout. Do not replace whole src/tests folders.

| File | Action |
|---|---|
| `src/hydro_health/engines/tiling/XGBoostModelingEngine.py` | Replace the tracked engine on your integration branch |
| `tests/data_validation_tests/XGBoost_Modeling_Engine_DV_tests.py` | Add the consolidated model test |
| `README_XGBoostModelingEngine_Repository_Test.md` | Add this guide |

The test searches its ancestors for `src/hydro_health`, avoiding the fixed-parent
index issue from moving the previous test under `tests/data_validation_tests`.
An explicit `--repo-root "C:/path/to/hydro_health"` is also available. Replace the
existing tracked engine file at its production path so your pull request shows
modifications to that file, rather than an unrelated new copy.

## 2. Inputs: use the fresh DataPrep outputs

The default model inputs deliberately come from **outputs/local_test_outputs**,
not the earlier immutable fixtures under inputs/local_test_inputs. This tests
the actual DataPrep-to-model handoff you just validated.

| Input | Repository-relative path |
|---|---|
| Training | `outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3/ML_ready_training_2004_2006.geoparquet` |
| Prediction | `outputs/local_test_outputs/Prediction_data_grid_tiles/BH4S556X_3/ML_ready_prediction_2004_2006.geoparquet` |
| Ordered schema | `outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3/locked_predictors_2004_2006.parquet` |
| DataPrep PASS report | `outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3/Data_Prep_Engine_Unit_Test_report_2004_2006.json` |

Your DataPrep report confirmed 100815 training rows, 248744 prediction rows and
this ordered schema:

```text
bathy_t
tsm_mean_2004_2006
distance_from_shore
bpi_broad_t
grain_size_layer
bpi_fine_t
curv_total_t
flowdir_sin_t
flowdir_cos_t
gradmag_t
```

Those names are documented here, **not redefined as a competing model list**.
The engine reads the locked Parquet list and requires it to match the DataPrep
report exactly. It checks tile, pair, CRS, source row counts, geometry/X/Y,
unique FIDs, essential fitting fields and positive prepared weights. Missing
predictors or invalid inputs fail; they do not trigger feature selection or
silent row removal. An entirely NA driver stays in the locked schema.

No Boruta text file needs to be loaded again at this step. DataPrep already
resolved it. No additional data preparation or weight normalization occurs here.
The DataPrep report's original coverage note is retained in the model report;
a successful model run does not expand prepared-fixture preparation coverage.

## 3. Run in your existing environment

From your repository root:

```bat
conda activate hydro_health
python -c "import sys; print(sys.executable)"
python tests\data_validation_tests\XGBoost_Modeling_Engine_DV_tests.py --overwrite
```

Use the dedicated hydro_health interpreter that passed DataPrep. No new `.venv`
or separate package installation is required if its model dependencies are
already present. Keep the repository environment manifest as the shared record;
this package does not install or update anything in your Windows environment.

Normal execution uses the real base Engine and local Dask worker processes.
Keep the main guard in the test for Windows process spawning. Run this test
script rather than launching the engine module directly.

`--overwrite` deliberately replaces matching model-stage outputs for the selected
pair. Old matching bootstrap and CV files are removed before retraining, so a
previous larger ensemble cannot leave convincing stale artifacts. It leaves
DataPrep files, schema, preparation report, R references, other year pairs,
rasters, PDFs and diagnostic image folders untouched. Rebuild the visual
products later; existing ones still describe the previous model run.

The engine emits a start message and the consolidated test prints the progress
log path. With your actual base Engine, worker/bootstrap progress is appended
to training output `log_prints.txt` rather than necessarily printed live.

### Optional smoke run, isolated from full outputs

```bat
python tests\data_validation_tests\XGBoost_Modeling_Engine_DV_tests.py --profile smoke --overwrite
```

Defaults: 100 fixed rounds, 2 bootstraps, 100 SHAP rows; **all input rows and the
same locked predictors remain**. Both model-output folders receive an
`xgb_smoke` subfolder. A smoke PASS checks the reduced run; use the full profile
for your actual model comparison. Smoke controls are in the same class settings.

### Replay existing outputs without retraining

```bat
python tests\data_validation_tests\XGBoost_Modeling_Engine_DV_tests.py --validate-only
```

For smoke output replay, also supply `--profile smoke`. The manifest must show
completed execution and match the current input-file hashes, ordered schema,
model parameters and profile. A changed DataPrep report also invalidates replay.
This prevents validating saved models against a different preparation run.
Models saved by the old standalone package need one new repository run to
create the new manifest/report contract.

`--execution inline` is an explicit debugging alternative. It executes the same
worker task, but does not test Dask. No silent fallback is used.

## 4. Model controls retained from the successful local package

The full defaults retain depth 4, eta 0.01, gamma/lambda/alpha 1,
subsample 0.7, colsample_bytree 0.8, histogram training, max_bin 256, 5 bootstraps,
5 spatial folds, 200 m blocks, CV limit 2000, early stopping 15 and the 1000-round
minimum/fallback under `cv_with_floor`. Global seed 12345, 1000 SHAP sample rows
and the 2-year standard horizon are retained. XGBoost uses one thread per model
and one local Dask worker for this single-tile integration test.

The selected rounds are recomputed, not pinned to the previous terminal value.
Recorded seeds and bootstrap indices support reproducibility, but exact agreement
between different package versions/environments is not assumed.

Changing `pair` in the grouped settings updates paths/titles/metadata routing.
The corresponding DataPrep outputs/report/schema must already exist. There is
no remaining 2004/2006-only input assertion; valid DataPrep per-row intervals
are preserved. For example a 2006_2010 pair can use four-year change/rate inputs
while the independent standard horizon remains two years unless you change it.
Only one configured tile/pair is run here; production multi-tile/region discovery
is one of the temporary cloud sections to reinstate with your developer.

## 5. Outputs and checks

Full outputs go to the existing Training_data_grid_tiles/BH4S556X_3 and
Prediction_data_grid_tiles/BH4S556X_3 folders under outputs/local_test_outputs.
Filename conventions remain compatible with the tested raster exporter:

- Training `models/model_boot_001_2004_2006.json` through the configured bootstrap count.
- Training `bootstrap_indices/bootstrap_001_2004_2006.parquet` etc. when enabled.
- Training `LOCAL_spatial_cv_folds_2004_2006.parquet`, `...metrics...`, `...oof...`.
- Training `LOCAL_importance_2004_2006.parquet`, `LOCAL_shap_summary_2004_2006.parquet`.
- Training `LOCAL_model_metadata_2004_2006.json` and `LOCAL_apparent_training_metrics_2004_2006.json`.
- Training `LOCAL_training_summary_2004_2006.parquet` and `.geoparquet`.
- Prediction `LOCAL_prediction_summary_2004_2006.parquet` and `.geoparquet`.
- Training `LOCAL_test_run_manifest_2004_2006.json` and
  `XGBoost_Modeling_Engine_DV_test_report_2004_2006.json`.

The test loads **every saved Booster**, checks feature order and round count,
then predicts by row chunks and independently computes NumPy mean/sample SD.
It compares ordinary and spatial summary tables, original source columns/order,
geometry, prediction usability, derived change/rate/standard-horizon formulas,
residuals, importance, optional SHAP, bootstrap indices and CV block membership.
Unusable prediction rows remain in output with NA model-derived values. Native
missing predictors and prepared fitting weights remain unchanged.

`best_iteration` in model metadata retains the previous package's naming but
stores the **number of fitted boosting rounds**, not a zero-based index.
Sample SD uses n_boot-1; one bootstrap gives zero SD on usable rows, which is
not evidence of zero scientific uncertainty. SHAP summaries average sampled
absolute contributions, and gain fraction is normalized total split gain;
neither is a new feature-selection step.

The apparent training metrics describe fitting rows and are optimistic relative
to held-out performance. Use the spatial-CV metrics/OOF predictions for held-out
checks. Occupied square blocks stay intact, but nearby blocks can be assigned
to different folds: this is not buffered CV or R blockCV's optimized assignment.
The existing early-stopping selection and minimum-round controls are unchanged.
Fixed/fallback cases now write explicit empty audit tables where no holdout
fits occurred; inspect cv_status before interpreting those tables.

## 6. Optional R model-output comparison

The default report correctly says `NOT_RUN` for both R comparisons. Contract
PASS is not evidence that independently trained R and Python models agree.

To compare already available R outputs, edit the class settings:

- `r_references.training` / `.prediction`: local CSV or Parquet/GeoParquet paths.
- `r_comparison_tolerances`: explicit scientifically chosen thresholds for maximum
  prediction difference, prediction RMSE difference and/or maximum SD difference.

References need unique FID, X/Y, bathy_t, interval_years,
mean_predicted_bathy_t1 and uncertainty_sd_bathy_t1. Available tile/pair fields
must agree; rows are aligned by FID and checked by coordinates/baselines.
Geometry, survey-date metadata or different numerical predictions cannot be
used to hide a mismatched source grid.

With reference paths but no tolerances the comparison is DESCRIPTIVE_ONLY, not
a numerical parity PASS. With supplied tolerances it can PASS or FAIL; the
report preserves that result separately from contract_status. Failures raise
an error. You must establish equivalent training data/settings, seeds, folds,
package versions and R reference provenance separately. No R invocation,
conversion helper, guessed tolerance or automatic retraining is included.

## Changes to review with your developer

Line references are for the delivered production-path engine and will move
when you edit it. The test and README are new files; the existing engine should
appear as modified in your PR's Files changed view.

1. **Temporary cloud boundary and real base Engine** - `XGBoostModelingEngine.py`, lines 21-33; `XGBoostModelingEngine.py`, lines 708-719; `XGBoostModelingEngine.py`, lines 826-877. Delete temporary repository routing and restore S3 loading/upload, configuration and tile/region discovery together before production merge. The real base is retained; no LocalModelEngineSupport module is used.
2. **Grouped settings** - `XGBoostModelingEngine.py`, lines 624-664; `XGBoostModelingEngine.py`, lines 684-706. Decide later which paths/scientific/profile controls return to YAML. There is no separate JSON file or duplicated normal-run hyperparameter list in the test.
3. **Fresh DataPrep handoff** - `XGBoostModelingEngine.py`, lines 731-790. Read regenerated output tables, exact ordered schema and the new DataPrep report filename. Require matching PASS/tile/pair/CRS/counts/schema; preserve fitting weights and original prediction rows. Report/source hashes bind this run, but a preparation-only PASS does not prove model accuracy.
4. **Strict predictor and target construction** - `XGBoostModelingEngine.py`, lines 200-256. Keep `bathy_t` as a feature, `bathy_t1` as target, and all native/all-NA driver fields. Reject missing columns, target/metadata leakage, bad prepared weights and infinite predictors. No importance-driven re-selection, intersected schema, baseline substitution or fitting-row drop is introduced.
5. **Spatial CV limitations and pandas 3** - `XGBoostModelingEngine.py`, lines 74-95; `XGBoostModelingEngine.py`, lines 121-178. Preserve occupied-square block assignment and existing early-stopping controls. These are not buffered folds or R blockCV. `unique_blocks = ...to_numpy(copy=True)` owns writable storage before shuffling under pandas 3 Copy-on-Write; it does not alter labels, shuffle seed or assignment logic.
6. **Fallback audit bug fix** - `XGBoostModelingEngine.py`, lines 134-137. Initialize audit tables as DataFrames rather than lists. Fixed/one-block/no-valid-fold modes now produce explicit empty artifacts, avoiding a false missing-file failure. Round selection formulas and fitted model behavior are unchanged.
7. **Prepared weights and bootstrap streams** - `XGBoostModelingEngine.py`, lines 353-355; lines 392-403. Carry weights unchanged into sampled training matrices; use separate deterministic bootstrap/model/SHAP streams. Diagnostic sampling must not change later fitted models.
8. **Ensemble mean, sample SD and derived change** - `XGBoostModelingEngine.py`, lines 259-268; `XGBoostModelingEngine.py`, lines 181-197. Retain streamed float64 statistics and later-minus-earlier change, per-row annualization and configured standard horizon. One-bootstrap zero SD is a computational convention, not a quantified absence of uncertainty.
9. **Model artifact IO** - `XGBoostModelingEngine.py`, lines 270-283; `XGBoostModelingEngine.py`, lines 285-299; `XGBoostModelingEngine.py`, lines 301-306; `XGBoostModelingEngine.py`, lines 457-476. Preserve native Booster JSON and ordinary/spatial summaries; reinstate the commented S3 model/table upload paths along with the resolver. Local GeoParquet output also needs the production cloud IO strategy.
10. **Reproducible validation and safe overwrite** - `XGBoostModelingEngine.py`, lines 600-606; `XGBoostModelingEngine.py`, lines 792-806; `XGBoostModelingEngine.py`, lines 826-877. Match input hashes/settings/schema/profile to a COMPLETED manifest before replay. Explicit overwrite removes only same-pair model-owned outputs so stale bootstrap/CV files do not survive. Do not delete prepared inputs, R references, other pairs or visual artifacts.
11. **Resource cleanup** - `XGBoostModelingEngine.py`, lines 808-824; `XGBoostModelingEngine.py`, lines 826-877. Retire allocated workers before scheduler shutdown, close partial resources safely and clean only the run-owned temporary directory. Review moving common defensive cleanup into the shared base alongside the DataPrep fix; this update does not replace base Engine.py.
12. **Diagnostic interpretation** - `XGBoostModelingEngine.py`, lines 97-118; `XGBoostModelingEngine.py`, lines 541-545. Keep apparent training metrics distinct from held-out CV and contract PASS distinct from R model-output parity. Model metadata records round selection, seeds, feature order and package versions. Unchanged SHAP/gain summaries do not prove feature causality.


## Validation completed before delivery

Six synthetic regression tests passed, using actual XGBoost and real local
Dask with threaded workers. Tests cover native/all-NA predictors, target/metadata
leakage, positive weights, schema handoff, occupied-block folds and round modes,
an independent refit of a recorded bootstrap, replay of every saved model,
SHAP independence, alternate pair duration, single-bootstrap SD, one-block CV
fallback audits, isolated smoke outputs, stale artifact cleanup, changed-input
manifest rejection, synthetic reference-comparison mechanics and corrupted
output detection.

The small synthetic model parameters are confined to --self-test. Normal full
runs use the class settings above. No real network-drive dataset, actual R model
reference or Windows repository worker-process run was available here. The base
Engine dependencies are not present in this package-only workspace, so the
synthetic run explicitly used the in-script base test double; your normal run
uses the real base. A host-only process-monitoring adapter was needed in the
container and is not part of the delivered code.

Tested with Python 3.12.14, XGBoost 3.4.1, NumPy 2.3.5, pandas 2.2.3 and a separate compatibility run using pandas 3.0.6, GeoPandas 1.2.0, PyArrow 26.0.0, Dask/distributed 2026.8.0. Both six-test suites passed. Your reported Windows Python 3.11.17 / NumPy 2.4.6 / PyArrow 25.0.0 environment remains for your actual repository integration run.

The predictor/data-preparation, seed, XGBoost parameter, metric, change/rate and
running-statistic scientific bodies were checked against the tested standalone
version. Material changes are local repository IO/validation, audit completeness,
owned-output lifecycle and an owned writable block-label copy for pandas 3.

This package does not create a GitHub branch, PR or merge. Add these three paths
to the integration branch you already created, test, and review the actual diff
against your developer's branch. Do not include local data/model artifacts in
the code commit. Cloud execution and raster-export repository cleanup remain
separate next stages.

Reference: [pandas Copy-on-Write documentation](https://pandas.pydata.org/docs/user_guide/copy_on_write.html).
