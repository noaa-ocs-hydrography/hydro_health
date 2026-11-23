# DataPrepEngine repository integration test

This update is **three files only**: the replacement production-path engine,
one consolidated unit/integration-test script, and this README. It preserves the
scientific fixes tested locally while temporarily using repository GeoParquet
fixtures. It does not alter XGBoost or raster export yet. No conversion scripts,
external JSON test config, standalone engine-support module, requirements file,
validation document or patch/diff file are included.

## Place the files in the existing hydro_health checkout

The ZIP contains repository-relative paths, with **no enclosing package folder**:

| File | Action |
|---|---|
| `src/hydro_health/engines/tiling/DataPrepEngine.py` | Replace the existing tracked file on your working branch |
| `tests/Data_Prep_Engine_Unit_Test.py` | Add the consolidated runner |
| `README_DataPrepEngine_Repository_Test.md` | Add this guide without replacing your main repository README |

Extract into a temporary folder first, then copy these three files to the
matching locations in your actual `hydro_health` checkout. Do not delete or
replace its existing `src`, `tests`, `inputs`, `outputs` or base `Engine.py`.
Do not create a second DataPrepEngine under a new local-test package: replacing
the **same tracked path** is what allows GitHub to show red/green changes.

## Inputs and outputs from your screenshots

All paths resolve from the repository root, not the terminal's current folder.
The default tile is `BH4S556X_3`, pair `2004_2006`, CRS `EPSG:32617`.

| Purpose | Repository-relative path |
|---|---|
| Immutable training fixture | `inputs/local_test_inputs/Training_data_grid_tiles/ML_ready_training_2004_2006.geoparquet` |
| Immutable prediction fixture | `inputs/local_test_inputs/Prediction_data_grid_tiles/ML_ready_prediction_2004_2006.geoparquet` |
| Training outputs | `outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3/` |
| Prediction outputs | `outputs/local_test_outputs/Prediction_data_grid_tiles/BH4S556X_3/` |
| Successful Boruta schema, read before writing | `outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3/locked_predictors_2004_2006.parquet` |

The last file is visible in your screenshot. Its existing ordered predictor list
is reused, including `distance_from_shore` and the pair-specific TSM field. There
is no automatic fallback to the old broad list and no inference from importance
ranking. The engine loads the schema before rewriting that same schema output.
Keep that successful file available for the first run. Alternatively set
`master_predictors` in the grouped settings to your exact ordered Boruta names;
base names `tsm_mean` and `hurr_strength_mean` receive the selected year suffix.
Use `bathy_t` first and include `distance_from_shore`.

Every test setting from the old config that still applies is now in the class's
**perhaps move to config file?** section. Conversion-only settings are removed.
The runner does not contain a second production configuration or predictor list.
Synthetic-only settings in its self-test do not control your scientific run.

## What can be tested with already-ready fixtures?

Default `input_stage="prepared_fixture"` re-executes the preparation task on the
copied `ML_ready` inputs. The test checks unchanged fixture values, row order,
predictor NAs, geometry, targets, weights, prediction usability and saved schema,
in addition to independent preparation formulas. The input files remain read
only; their SHA256 hashes are recorded and checked after the run.

This is a **prepared-fixture regression test**, not a repeat of the original
raw-data preparation parity test. Previously removed training rows are absent,
so original training eligibility and pre-filter missingness cannot be recovered.
Missingness now describes the fixture that was actually supplied. If you change
scientific settings, a frozen prepared-fixture comparison may correctly fail;
use the original raw batches to evaluate such changes.

For a full preparation test, use the existing raw GeoParquet batches (the `raw`
folders are visible in your screenshots):

```bat
python tests\Data_Prep_Engine_Unit_Test.py --input-stage raw_batch --overwrite
```

The expected default paths are:

- `outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3/raw/BH4S556X_3/BH4S556X_3_2004_2006_training_batch.parquet`
- `outputs/local_test_outputs/Prediction_data_grid_tiles/BH4S556X_3/raw/BH4S556X_3/BH4S556X_3_2004_2006_prediction_batch.parquet`

These are already-converted GeoParquet files even though their extension is
`.parquet`. Verify those filenames in the raw folders; edit `raw_training_input`
and `raw_prediction_input` in the grouped class settings if yours differ. The
runner does not guess another file or silently switch stages when one is absent.

## Run in the dedicated hydro_health environment

Use your existing `hydro_health` interpreter (Python 3.11.17 in the screenshot).
There is no need to create another `.venv` or reinstall packages already present.
The selected VS Code interpreter and the terminal's Python must both be correct.
From the **repository root**, in an activated terminal:

```bat
conda activate hydro_health
python -c "import sys; print(sys.executable)"
python tests\data_validation_tests\Data_Prep_Engine_DV_tests.py --overwrite
```

The interpreter printed should be the `...Local\ESRI\conda\envs\hydro_health\python.exe`
shown in your screenshot. It should not be Anaconda base or the previous `.venv`.
If activation is unavailable in that shell, select the hydro_health interpreter
in VS Code and create a new terminal, then verify the printed executable.

`--overwrite` is appropriate here because your screenshot already contains the
locked schema and preparation diagnostics. It replaces **preparation outputs**,
not input fixtures, trained models, rasters or PDFs. Those downstream outputs
will still describe their previous run until their engines are updated/rerun.
Without `--overwrite`, existing preparation outputs cause an explicit failure.

Normal execution uses the repository's real `Engine` and its local Dask setup.
The single runner's `if __name__ == "__main__"` guard supports Windows workers.
For deliberate debugging only, `--execution inline` bypasses Dask; that is not
validation of Dask execution. Production/base import errors are not hidden by a
standalone support class.

**Does the environment supersede requirements.txt?** It supersedes the separate
standalone-test installation step. A working environment does not document
itself for colleagues or EC2 rebuilds: retain the repository's environment YAML
or other chosen dependency manifest as the maintained source of truth. This
package adds no separate requirements file and does not modify your environment.

## Outputs and interpretation

Training folder receives:

- `ML_ready_training_2004_2006.geoparquet`
- `locked_predictors_2004_2006.parquet`
- `predictor_missingness_training_2004_2006.parquet`
- `predictor_ranges_training_2004_2006.parquet`
- `Data_Prep_Engine_Unit_Test_report_2004_2006.json` and `log_prints.txt`

Prediction folder receives:

- `ML_ready_prediction_2004_2006.geoparquet`
- `predictor_missingness_prediction_2004_2006.parquet`

The report records the input stage, hashes, versions, settings, row counts,
locked schema and PASS/FAIL. A missing requested predictor, invalid input CRS,
inconsistent geometry, failed/skipped tile or output mismatch is a failure.
It does not train a model or claim cloud validation.

Optional comparison with your previously saved R-prepared reference tables:

```bat
python tests\Data_Prep_Engine_Unit_Test.py --overwrite --compare-r-reference
```

For a raw-input preparation comparison:

```bat
python tests\Data_Prep_Engine_Unit_Test.py --input-stage raw_batch --overwrite --compare-r-reference
```

Both output folders must contain the corresponding
`R_reference_training_2004_2006.parquet` / `R_reference_prediction_2004_2006.parquet`
from the **matching predictor list, pair and preparation settings**. No R code is
invoked. Full fields, row order and NA positions are compared against those saved
reference values; their provenance must be confirmed by you. Missing references
fail an explicitly requested comparison. Without this flag the R comparison is
`NOT_RUN`. This does not imply R/Python model-training parity.

## GitHub review: red/green changes within the whole file

Changes are reviewable in a pull request **before merging**. Start on the branch
agreed with your developer, create your test branch, and copy the files into the
existing tracked locations. Example commands from the repository root:

```bat
git switch -c data-prep-repository-test
git diff -- src/hydro_health/engines/tiling/DataPrepEngine.py
git status --short
git add src/hydro_health/engines/tiling/DataPrepEngine.py tests/Data_Prep_Engine_Unit_Test.py README_DataPrepEngine_Repository_Test.md
git commit -m "Integrate tested data preparation into repository"
git push -u origin data-prep-repository-test
```

Then open the pull request against the branch selected by your developer.
In **Files changed**, the engine should be **modified**, not an entirely new
file. GitHub shows additions/deletions with unchanged context; expand the
collapsed unchanged sections to inspect the rest of the file. Choose unified
or split view as preferred. The new test and README naturally appear as new
files. You do not need a separate `_changes.diff` file, or to merge first.

Do not add the local input/output data, generated reports or temporary files to
the code commit. No remote branch, pull request or repository merge was created
by generating this package. The current checkout/branch was not available here;
review the resulting diff against your actual repository version before merging.

## Changes to review with your developer

Line references below refer to this delivered `DataPrepEngine.py`; they move
if the file is edited. Every function has a purpose/why docstring, and the
script summary states the temporary integration scope.

1. **Temporary imports and local/cloud boundary** - `DataPrepEngine.py`, lines 19-34; `DataPrepEngine.py`, lines 469-492; `DataPrepEngine.py`, lines 507-520. The real base `Engine` is used, while S3/config lookup is commented. Search `TEMPORARY REPOSITORY TEST`. Delete the temporary fixture routing and restore S3 loading, upload, region/batch discovery and configuration **together before merging into cloud production**. Keep the tested scientific helpers. Merely uncommenting `import s3fs` does not restore a cloud-ready engine.
2. **Settings migration** - `DataPrepEngine.py`, lines 398-427. These class settings replace the old local JSON. Decide later which belong in YAML and which are engine-specific. The default stage is explicitly prepared-fixture regression; use `raw_batch` for complete row-filter testing.
3. **Boruta source and exact order** - `DataPrepEngine.py`, lines 443-467. Read the successful locked schema before it is rewritten; never infer predictors from importance ranking or reintroduce the broad list. The temporary dependency on a previous schema output should become an explicit Boruta/config input for a fresh production deployment. `master_predictors` permits that explicit list now.
4. **Correct time direction and target spelling** - `DataPrepEngine.py`, lines 68-98. Retain `bathy_t1 - bathy_t`, `year_t1 - year_t` and the positive pair-duration fallback. Legacy `bathy_ti` / `year_ti` names are excluded from predictors, not silently treated as valid new targets.
5. **Schema/leakage checks and no silent Boruta omission** - `DataPrepEngine.py`, lines 106-136; `DataPrepEngine.py`, lines 301-302. Keep ordered predictors, dynamic-year resolution and training-first prediction compatibility. Missing requested training fields now fail rather than quietly shrinking your selected model.
6. **Preserve native predictor NAs** - `DataPrepEngine.py`, lines 324-344. Filter training only on essential fields in `native_missing` mode; complete-case filtering is an explicit alternative. Already-ready fixtures cannot prove which original rows were excluded.
7. **Retain the tested R weighting formulas** - `DataPrepEngine.py`, lines 139-165. Power weights normalize to mean one and capped-linear weights use the original quantile cap. Recompute training weights without adding them to prediction inputs.
8. **Retain every prediction row** - `DataPrepEngine.py`, lines 365-370. Preserve `prediction_row_usable` and null geometries for invalid XY, so downstream extent is not silently reduced.
9. **Coordinate provenance** - `DataPrepEngine.py`, lines 235-259; `DataPrepEngine.py`, lines 196-215. Check EPSG:32617, tile/pair identity, point type, null positions and numeric X/Y agreement. Geometry assignment is not reprojection. Revisit how these checks map onto upstream production Parquet rather than discarding them when removing local IO.
10. **Diagnostics scope** - `DataPrepEngine.py`, lines 168-176; `DataPrepEngine.py`, lines 179-193. Missingness is measured before this run's filter, ranges among fitting rows. Fixture-stage missingness is already downstream of the earlier raw filtering; do not label it original raw missingness.
11. **Resource cleanup and failures** - `DataPrepEngine.py`, lines 522-533; `DataPrepEngine.py`, lines 535-574. Unique temporary folders, deterministic client/cluster shutdown, explicit task failure and overwrite guards stay. Consider moving the defensive `close_dask` fix into the shared base in a separate reviewed change; this package does not edit the base. Windows/Dask worker behaviour must be verified locally.
12. **Overwrite boundaries and downstream handoff** - `DataPrepEngine.py`, lines 469-492; `DataPrepEngine.py`, lines 535-574. Local inputs and outputs must be different files; only preparation products are replaced. Existing models/rasters remain from their prior run until updated downstream. Restore per-region/multi-tile discovery when reintegrating the S3 resolver.


## Validation included in this consolidated runner

Run optional synthetic software tests without your scientific inputs:

```bat
python tests\Data_Prep_Engine_Unit_Test.py --self-test
```

Before delivery, six regression tests passed: signed targets/interval fallback,
R weighting formulas and target exclusion; raw inputs through real local Dask;
complete-case versus native-NA row counts; full prepared-fixture regression and
locked schema; missing predictors, invalid CRS and input/output collision guards;
and failure propagation/cleanup before Dask resources exist.

The synthetic test used a temporary repository layout and an isolated base-class
test double because this workspace does not contain the actual project base or
its GDAL/provider dependencies. It exercised real Dask with threaded workers;
Windows worker processes and your dedicated repository environment remain for
your local integration run. A host-only /proc mapping was needed for this
container's process monitoring; it is not included in either delivered script.
Scientific helper AST checks confirmed target calculation, numeric coercion,
weighting, NA diagnostics and range formulas remain unchanged from the tested
local package. Actual network-drive fixtures and saved R references were not
available here and were not claimed as revalidated.

Useful references:

- [GitHub: reviewing proposed changes](https://docs.github.com/en/pull-requests/how-tos/review-pull-requests/reviewing-proposed-changes-in-a-pull-request)
- [Conda: managing environments](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)
