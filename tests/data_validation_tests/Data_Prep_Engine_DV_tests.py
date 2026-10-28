"""Exercise the repository DataPrepEngine without conversion helpers or JSON config.

Default: load the real project Engine and process the configured GeoParquet
fixtures through local Dask. Check every field, row order, NA pattern, schema,
geometry and diagnostics, saving a candid preparation-only report. Already-ready
fixtures provide a regression check; use raw_batch to test original filtering.
Optional --self-test uses synthetic inputs and an isolated base-class test double
when the real repository base is unavailable. It is never the default fallback.
"""
import argparse
import hashlib
import importlib
import json
import platform
import sys
import tempfile
import types
import unittest
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


DEFAULT_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def load_engine(repository_root: Path, self_test: bool = False):
    """Import the replacement from src, using the actual repository base by default.

    Synthetic software tests may supply a minimal base test double only when
    explicitly requested and no Engine.py exists. Never hide an import failure
    in the user's real environment by silently replacing its base Engine.
    """
    source = repository_root / "src"
    sys.path.insert(0, str(source))
    if self_test and not (source / "hydro_health/engines/Engine.py").is_file():
        base = types.ModuleType("hydro_health.engines.Engine")

        class Engine:
            """Test double for this script's synthetic checks, not a production dependency."""
            def __init__(self):
                """Match the production base resource handles without provider services."""
                self.client = None
                self.cluster = None

            def setup_dask(self, env, **kwargs):
                """Exercise actual local Dask; synthetic tests use threads for portability."""
                from dask.distributed import Client, LocalCluster
                self.cluster = LocalCluster(**kwargs, dashboard_address=None)
                self.client = Client(self.cluster)

            @staticmethod
            def write_message_dask(message, output_folder):
                """Record worker messages in the same log used by the real base."""
                folder = Path(output_folder); folder.mkdir(parents=True, exist_ok=True)
                with (folder / "log_prints.txt").open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")

            write_message = write_message_dask

            def cleanup_resources(self, output_folder):
                """Mirror base cleanup, dispatching the subclass's defensive close_dask."""
                import shutil
                self.close_dask()
                if hasattr(self, "local_tmp_dir"):
                    shutil.rmtree(self.local_tmp_dir, ignore_errors=True)

        base.Engine = Engine
        sys.modules[base.__name__] = base
    return importlib.import_module("hydro_health.engines.tiling.DataPrepEngine")


def file_digest(path: Path) -> str:
    """Hash fixture bytes so the report can identify inputs and prove no mutation."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_tables(training, prediction, predictors, cfg):
    """Independently calculate expected values using the agreed R preparation rules.

    This reference does not call production target/filter/weight helpers. It
    cannot recover rows absent from an already-prepared training fixture.
    """
    train, pred = training.copy(), prediction.copy()
    years = [int(y) for y in cfg["pair"].split("_")]
    fallback = float(years[1] - years[0])
    numeric = set(predictors) | {"X", "Y", "FID", "bathy_t", "bathy_t1", "year_t", "year_t1",
                                  "delta_bathy", "delta_rate", "interval_years"}
    for frame in (train, pred):
        for col in numeric.intersection(frame.columns):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["pair_id"] = cfg["pair"]
        frame["tile_id"] = cfg["tile_id"]
    if "year_t" in train and "year_t1" in train:
        interval = train.year_t1.to_numpy(float) - train.year_t.to_numpy(float)
    elif "interval_years" in train:
        interval = train.interval_years.to_numpy(float)
    else:
        interval = np.full(len(train), fallback)
    interval[~np.isfinite(interval) | (interval <= 0)] = fallback
    train["interval_years"] = interval
    train["delta_bathy"] = train.bathy_t1 - train.bathy_t
    train["delta_rate"] = train.delta_bathy / interval
    essential = np.isfinite(train[["X", "Y", "bathy_t", "bathy_t1", "delta_bathy", "interval_years"]].to_numpy(float)).all(axis=1)
    if cfg["pre_xgb_data_mode"] == "legacy_complete":
        essential &= np.isfinite(train[predictors].to_numpy(float)).all(axis=1)
    ready = train.loc[essential].copy().reset_index(drop=True)
    magnitude = np.abs(ready.delta_bathy.to_numpy(float))
    weights = np.ones(len(ready))
    if cfg["use_weighted_loss"] and len(ready):
        if cfg["weight_method"] == "power":
            raw = (magnitude + 1e-6) ** cfg["weight_alpha"]
            total = raw.sum()
            if np.isfinite(total) and total > 0:
                weights = raw * (len(raw) / total)
        elif cfg["weight_method"] == "capped_linear":
            cap = np.quantile(magnitude, cfg["weight_cap_quantile"])
            if np.isfinite(cap) and cap > 0:
                weights = 1 + (cfg["max_weight"] - 1) * (np.minimum(magnitude, cap) / cap) ** cfg["weight_alpha"]
        else:
            raise ValueError("Unknown reference weighting method")
    ready["sample_weight"] = weights
    pred["interval_years"] = fallback
    pred["prediction_row_usable"] = np.isfinite(pred[["X", "Y", "bathy_t"]].to_numpy(float)).all(axis=1)
    return ready, pred.reset_index(drop=True), train, pred


def check_outputs(engine, sources, compare_r_reference=False):
    """Verify full saved tables, schema, pre-filter missingness, finite ranges and geometry."""
    cfg = engine.settings; pair = cfg["pair"]
    predictors = [f"{p}_{pair}" if p in ("hurr_strength_mean", "tsm_mean") else p for p in engine.master_predictors]
    schema_path = engine.ml_prep_train_dir / f"locked_predictors_{pair}.parquet"
    locked = pd.read_parquet(schema_path).predictor.tolist()
    if locked != predictors:
        raise AssertionError("Saved predictor order differs from the successful Boruta schema")
    tables = {role: pd.DataFrame(frame.drop(columns=frame.geometry.name)) for role, frame in sources.items()}
    expected_train, expected_pred, normalized_train, normalized_pred = expected_tables(tables["training"], tables["prediction"], predictors, cfg)
    checks = {}; r_checks = {}; counts = {}
    for role, expected, normalized, directory in (
        ("training", expected_train, normalized_train, engine.ml_prep_train_dir),
        ("prediction", expected_pred, normalized_pred, engine.ml_prep_pred_dir),
    ):
        spatial = gpd.read_parquet(directory / f"ML_ready_{role}_{pair}.geoparquet")
        actual = pd.DataFrame(spatial.drop(columns=spatial.geometry.name)).reset_index(drop=True)
        if set(actual.columns) != set(expected.columns):
            raise AssertionError(f"{role} output columns differ: {set(actual.columns) ^ set(expected.columns)}")
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False,
                                      check_exact=False, atol=1e-8, rtol=1e-10)
        assert spatial.crs.to_epsg() == 32617
        finite_xy = np.isfinite(actual.X) & np.isfinite(actual.Y)
        np.testing.assert_array_equal(spatial.geometry.isna(), ~finite_xy)
        np.testing.assert_allclose(spatial.loc[finite_xy].geometry.x, actual.loc[finite_xy, "X"], atol=1e-8, rtol=0)
        np.testing.assert_allclose(spatial.loc[finite_xy].geometry.y, actual.loc[finite_xy, "Y"], atol=1e-8, rtol=0)
        missing = pd.read_parquet(directory / f"predictor_missingness_{role}_{pair}.parquet")
        missing_ref = pd.DataFrame({"predictor": predictors, "n_rows": len(normalized),
            "n_na": [normalized[p].isna().sum() for p in predictors],
            "proportion_na": [normalized[p].isna().mean() for p in predictors],
            "n_finite": [np.isfinite(normalized[p].to_numpy(float)).sum() for p in predictors]})
        pd.testing.assert_frame_equal(missing, missing_ref, check_dtype=False, atol=1e-12, rtol=1e-10)
        if role == "prediction":
            assert len(actual) == len(sources[role]), "Prediction rows were dropped"
            assert "sample_weight" not in actual, "Prediction tables must not inherit training weights"
        if cfg["input_stage"] == "prepared_fixture":
            # This is a frozen-reference regression check, separate from formula checks.
            baseline = tables[role].reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[baseline.columns], baseline, check_dtype=False,
                                          check_exact=False, atol=1e-8, rtol=1e-10)
        if compare_r_reference:
            reference = directory / f"R_reference_{role}_{pair}.parquet"
            if not reference.is_file():
                raise FileNotFoundError(f"Requested R reference not found: {reference}")
            r = pd.read_parquet(reference).reset_index(drop=True)
            if set(r.columns) != set(actual.columns):
                raise AssertionError(f"{role} R reference columns differ")
            pd.testing.assert_frame_equal(actual[r.columns], r, check_dtype=False,
                                          check_exact=False, atol=1e-8, rtol=1e-10)
            r_checks[role] = "PASS: saved R reference values, row order and NA pattern"
        checks[role] = "PASS: all values, row order, NA positions, geometry and missingness"
        counts[role] = {"input_rows": len(sources[role]), "output_rows": len(actual),
                        "null_geometry_rows": int(spatial.geometry.isna().sum())}
    ranges = pd.read_parquet(engine.ml_prep_train_dir / f"predictor_ranges_training_{pair}.parquet")
    assert ranges.predictor.tolist() == predictors
    for row in ranges.itertuples(index=False):
        v = expected_train[row.predictor].to_numpy(float); v = v[np.isfinite(v)]
        expected = [v.min(), v.max(), v.mean()] if len(v) else [np.nan] * 3
        np.testing.assert_allclose([row.min_training, row.max_training, row.mean_training], expected, equal_nan=True)
    return {"checks": checks, "counts": counts, "locked_predictors": locked,
            "range_check": "PASS", "r_reference_comparison": r_checks or "NOT_RUN"}


def run_test(module, repository_root, overrides=None, overwrite=False, execution="dask", compare_r_reference=False):
    """Run the real engine, preserve immutable fixtures, and write a scoped PASS/FAIL report."""
    engine = module.DataPrepEngine({"env": "local", "repository_root": str(repository_root),
                                    "data_prep_settings": overrides or {}})
    cfg = engine.settings
    report_dir = engine._repository_path(cfg["training_output"])
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"Data_Prep_Engine_Unit_Test_report_{cfg['pair']}.json"
    report = {"status": "RUNNING", "tile_id": cfg["tile_id"], "pair": cfg["pair"], "crs": cfg["crs"],
              "input_stage": cfg["input_stage"], "execution": execution, "settings": cfg,
              "python": sys.version, "platform": platform.platform(),
              "versions": {m: version(m) for m in ("numpy", "pandas", "geopandas", "pyarrow", "dask", "distributed")},
              "coverage": "Prepared-fixture regression; original excluded rows and pre-filter missingness cannot be recovered"
                if cfg["input_stage"] == "prepared_fixture" else "Raw GeoParquet preparation: original row eligibility and NA handling exercised",
              "r_reference_comparison": "NOT_RUN", "scope": "Data preparation only; no model training or cloud validation"}
    try:
        engine._resolve_paths(cfg["region"])
        train_path, pred_path = engine.raw_train_uri, engine.raw_pred_uri
        inputs = {"training": train_path, "prediction": pred_path}
        digests = {role: file_digest(path) for role, path in inputs.items()}
        sources = {role: gpd.read_parquet(path) for role, path in inputs.items()}
        report["source_files"] = {role: {"path": str(path), "sha256": digests[role]} for role, path in inputs.items()}
        # Snapshot the schema before the engine writes its output at the same legacy location.
        report["input_predictors"] = engine._load_master_predictors()
        report["results"] = engine.run(overwrite=overwrite, execution=execution)
        report.update(check_outputs(engine, sources, compare_r_reference))
        for role, path in inputs.items():
            if file_digest(path) != digests[role]:
                raise AssertionError(f"Input fixture was modified: {path}")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {report['coverage']}. Report: {report_path}")
    print("R reference comparison:", "PASS" if compare_r_reference else "NOT_RUN")
    return report


def write_fixture(path, frame):
    """Write synthetic EPSG:32617 points, preserving null geometry for invalid XY."""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(frame.X) & np.isfinite(frame.Y)
    points = gpd.points_from_xy(frame.X.fillna(0), frame.Y.fillna(0), crs=32617)
    points[~valid] = None
    gpd.GeoDataFrame(frame, geometry=points).to_parquet(path, index=False)


def make_fixture(root):
    """Create raw and already-ready fixtures with known removed rows and native NAs."""
    n = 140; tile = "BH4S556X_3"; pair = "2004_2006"
    train = pd.DataFrame({"FID": np.arange(n), "X": 340000. + np.arange(n) * 8,
        "Y": np.full(n, 3024000.), "bathy_t": np.full(n, -8.),
        "bathy_t1": -8 + np.linspace(-1, 1, n), "year_t": 2004., "year_t1": 2006.,
        "distance_from_shore": np.linspace(100, 500, n), "tsm_mean_2004_2006": np.full(n, 3.)})
    train.loc[0, "distance_from_shore"] = np.nan
    train.loc[1, "X"] = np.nan
    train.loc[2, "bathy_t1"] = np.nan
    train.loc[3, "bathy_t"] = np.nan
    train.loc[4, ["year_t", "year_t1"]] = np.nan
    pred = train.drop(columns=["bathy_t1", "year_t", "year_t1"]).copy()
    overrides = {"master_predictors": ["bathy_t", "distance_from_shore", "tsm_mean"],
                 "data_prep_processes": False, "data_prep_memory_limit": "1GB", "verbose_logging": False}
    raw_train = root / f"outputs/local_test_outputs/Training_data_grid_tiles/{tile}/raw/{tile}/{tile}_{pair}_training_batch.parquet"
    raw_pred = root / f"outputs/local_test_outputs/Prediction_data_grid_tiles/{tile}/raw/{tile}/{tile}_{pair}_prediction_batch.parquet"
    write_fixture(raw_train, train); write_fixture(raw_pred, pred)
    cfg = {"pair": pair, "tile_id": tile, "pre_xgb_data_mode": "native_missing", "use_weighted_loss": True,
           "weight_method": "power", "weight_alpha": .5, "weight_cap_quantile": .95, "max_weight": 2.}
    ready_train, ready_pred, _, _ = expected_tables(train, pred, ["bathy_t", "distance_from_shore", "tsm_mean_2004_2006"], cfg)
    write_fixture(root / f"inputs/local_test_inputs/Training_data_grid_tiles/ML_ready_training_{pair}.geoparquet", ready_train)
    write_fixture(root / f"inputs/local_test_inputs/Prediction_data_grid_tiles/ML_ready_prediction_{pair}.geoparquet", ready_pred)
    return overrides


def run_self_tests(module):
    """Test actual task code with controlled fixtures; never label synthetic data as R parity."""
    class PreparationRegressionTests(unittest.TestCase):
        """Cover scientific corrections, input stages, Dask execution and failure propagation."""
        def setUp(self):
            """Give every test a private repository-layout fixture tree."""
            self.temp = tempfile.TemporaryDirectory(prefix="hhm_repository_self_test_")
            self.root = Path(self.temp.name)
            self.overrides = make_fixture(self.root)

        def tearDown(self):
            """Remove only this test's private synthetic files."""
            self.temp.cleanup()

        def test_targets_weights_and_leakage(self):
            """Check exact signed changes, pair fallback, R weighting and target exclusion."""
            source = pd.DataFrame({"bathy_t": [-8., -8.], "bathy_t1": [-7., -10.],
                                   "year_t": [2004., np.nan], "year_t1": [2006., np.nan]})
            targets = module._ensure_training_targets(source, "2004_2006")
            np.testing.assert_allclose(targets.delta_bathy, [1., -2.])
            np.testing.assert_allclose(targets.delta_rate, [.5, -1.])
            np.testing.assert_allclose(targets.interval_years, [2., 2.])
            changes = np.array([-4., 0., 1., 9.])
            power = np.sqrt(np.abs(changes) + 1e-6)
            np.testing.assert_allclose(module._calculate_sample_weights(changes,.5,"power",.95,2),power / power.mean())
            cap = np.quantile(np.abs(changes),.95)
            np.testing.assert_allclose(module._calculate_sample_weights(changes,.5,"capped_linear",.95,2),1+np.sqrt(np.minimum(np.abs(changes),cap)/cap))
            resolved, _, _ = module._resolve_predictors("2004_2006", ["bathy_t1", "year_t1", "distance_from_shore"],
                ["bathy_t", "bathy_t1", "year_t1", "distance_from_shore"], ["bathy_t", "distance_from_shore"], "og_training_first")
            self.assertEqual(resolved,["bathy_t", "distance_from_shore"])

        def test_raw_dask_and_complete_case(self):
            """Use real local Dask to check raw filtering, retained prediction extent and native NAs."""
            overrides = dict(self.overrides, input_stage="raw_batch")
            report = run_test(module,self.root,overrides,execution="dask")
            self.assertEqual(report["counts"]["training"]["output_rows"],137)
            self.assertEqual(report["counts"]["prediction"]["output_rows"],140)
            self.assertEqual(report["counts"]["prediction"]["null_geometry_rows"],1)
            self.assertEqual(report["r_reference_comparison"],"NOT_RUN")
            overrides["pre_xgb_data_mode"] = "legacy_complete"
            report = run_test(module,self.root,overrides,overwrite=True,execution="inline")
            self.assertEqual(report["counts"]["training"]["output_rows"],136)
            self.assertEqual(report["counts"]["prediction"]["output_rows"],140)

        def test_prepared_regression_exact_schema_and_overwrite(self):
            """Exercise saved-schema loading, exact fixture regression and deliberate overwrite."""
            overrides = dict(self.overrides)
            run_test(module,self.root,overrides,execution="inline")
            overrides["master_predictors"] = None
            report = run_test(module,self.root,overrides,overwrite=True,execution="inline")
            self.assertEqual(report["locked_predictors"],["bathy_t", "distance_from_shore", "tsm_mean_2004_2006"])
            with self.assertRaises(FileExistsError):
                run_test(module,self.root,overrides,execution="inline")

        def test_missing_predictor_and_insufficient_rows_fail(self):
            """Never silently shrink the requested Boruta list or treat a skipped tile as PASS."""
            overrides = dict(self.overrides,master_predictors=self.overrides["master_predictors"]+["not_in_data"])
            with self.assertRaisesRegex(RuntimeError,"Requested predictors missing"):
                run_test(module,self.root,overrides,execution="inline")
            overrides = dict(self.overrides,minimum_training_rows=1000)
            engine = module.DataPrepEngine({"repository_root":str(self.root),"data_prep_settings":overrides})
            with self.assertRaises(RuntimeError): engine.run(execution="inline",overwrite=True)
            self.assertFalse(engine.local_tmp_dir.exists())

        def test_crs_and_input_output_collision_fail(self):
            """Reject mislocated geometry and refuse to overwrite an input fixture."""
            path = self.root/"inputs/local_test_inputs/Training_data_grid_tiles/ML_ready_training_2004_2006.geoparquet"
            frame = gpd.read_parquet(path).set_crs(4326,allow_override=True); frame.to_parquet(path,index=False)
            with self.assertRaisesRegex(RuntimeError,"EPSG:32617"):
                run_test(module,self.root,self.overrides,execution="inline")
            overrides = dict(self.overrides,training_output="inputs/local_test_inputs/Training_data_grid_tiles")
            with self.assertRaisesRegex(ValueError,"input and output"):
                run_test(module,self.root,overrides,execution="inline")

        def test_setup_failure_preserves_error_and_cleanup(self):
            """A Dask setup exception must survive cleanup even when client/cluster are None."""
            engine = module.DataPrepEngine({"repository_root":str(self.root),"data_prep_settings":self.overrides})
            def fail_setup(*args, **kwargs):
                """Inject an allocation failure before resources exist."""
                raise RuntimeError("deliberate Dask setup failure")
            engine.setup_dask = fail_setup
            with self.assertRaisesRegex(RuntimeError,"deliberate Dask setup failure"):
                engine.run(execution="dask")
            self.assertFalse(engine.local_tmp_dir.exists())

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PreparationRegressionTests))
    if not result.wasSuccessful():
        raise SystemExit(1)


def main():
    """Provide one Windows-safe entry point for repository checks and optional self-tests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--input-stage",choices=["prepared_fixture","raw_batch"])
    parser.add_argument("--execution",choices=["dask","inline"],default="dask")
    parser.add_argument("--compare-r-reference",action="store_true",help="Compare saved R reference tables already in output folders")
    parser.add_argument("--self-test",action="store_true",help="Run synthetic software tests, not the real repository data")
    args = parser.parse_args()
    module = load_engine(args.repo_root.resolve(),self_test=args.self_test)
    if args.self_test:
        run_self_tests(module)
    else:
        overrides = {"input_stage":args.input_stage} if args.input_stage else {}
        run_test(module,args.repo_root.resolve(),overrides,args.overwrite,args.execution,args.compare_r_reference)


if __name__ == "__main__":
    # Required before Dask starts Windows worker processes.
    main()
