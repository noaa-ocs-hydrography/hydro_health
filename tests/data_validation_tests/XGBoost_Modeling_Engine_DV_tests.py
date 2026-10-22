"""Run and validate the repository model engine using the verified DataPrep outputs.

Full model execution/output-contract checks replay every saved Booster in chunks,
check ensemble mean/SD independently and preserve all source rows and fields.
This PASS is distinct from optional R model-output comparison. Settings live
only in XGBoostModelingEngine.py; no conversion, support or JSON config is used.
Synthetic --self-test never substitutes its tiny parameters in your normal run.
"""
import argparse
import copy
import importlib
import json
import platform
import sys
import tempfile
import traceback
import types
import unittest
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xgboost as xgb

DERIVED = [
    "mean_predicted_bathy_next", "mean_predicted_change", "mean_predicted_rate",
    "mean_predicted_standard_change", "mean_predicted_standard_bathy",
    "uncertainty_sd_bathy_next", "mean_predicted_bathy_t1", "uncertainty_sd_bathy_t1",
    "uncertainty_sd_change", "uncertainty_sd_rate",
]


def find_repository_root(script_path: Path) -> Path:
    """Find the ancestor containing src/hydro_health even if this test folder moves."""
    for parent in script_path.resolve().parents:
        if (parent / "src/hydro_health").is_dir():
            return parent
    raise FileNotFoundError("Cannot find repository src/hydro_health above the test script")


def load_engine(repository_root: Path, self_test: bool = False):
    """Import the production-path replacement and real base without a silent fallback.

    Only an explicit synthetic test in a package without the project Engine.py
    installs this script's isolated base test double. Normal runs always use
    the actual repository base and surface missing dependencies/import failures.
    """
    sys.path.insert(0,str(repository_root / "src"))
    if self_test and not (repository_root / "src/hydro_health/engines/Engine.py").is_file():
        base = types.ModuleType("hydro_health.engines.Engine")

        class Engine:
            """Synthetic-test base double; not a production module or normal-run fallback."""
            def __init__(self):
                """Match the production resource handles without its provider dependencies."""
                self.client = None
                self.cluster = None

            def setup_dask(self, env, **kwargs):
                """Run actual local Dask with threads in the portable synthetic test."""
                from dask.distributed import Client, LocalCluster
                self.cluster = LocalCluster(**kwargs,dashboard_address=None)
                self.client = Client(self.cluster)

            @staticmethod
            def write_message_dask(message, output_folder):
                """Record task messages under the synthetic output directory."""
                directory = Path(output_folder); directory.mkdir(parents=True,exist_ok=True)
                with (directory / "log_prints.txt").open("a",encoding="utf-8") as stream:
                    stream.write(str(message)+"\n")

            write_message = write_message_dask

            def cleanup_resources(self, output_folder):
                """Mirror base cleanup and dispatch the subclass's safe close_dask override."""
                import shutil
                self.close_dask()
                if hasattr(self,"local_tmp_dir"):
                    shutil.rmtree(self.local_tmp_dir,ignore_errors=True)

        base.Engine = Engine
        sys.modules[base.__name__] = base
    return importlib.import_module("hydro_health.engines.tiling.XGBoostModelingEngine")


def verify_saved_outputs(cfg, train, pred, tx, px, predictors):
    """Independently replay every saved Booster and verify output means/SDs by row chunks."""
    pair = cfg["pair"]
    base = Path(cfg["training_output_dir"])
    metadata = json.loads((base / f"LOCAL_model_metadata_{pair}.json").read_text())
    if metadata["target"] != "bathy_t1" or metadata["training_baseline_source"] != "bathy_t":
        raise AssertionError("Incorrect saved model target or baseline")
    if metadata["predictors"] != predictors or metadata["n_boot"] != cfg["model_config"]["n_boot"]:
        raise AssertionError("Saved schema or bootstrap count differs from configuration")
    n_boot = metadata["n_boot"]
    models = []
    for uri in metadata["model_uris"]:
        model = xgb.Booster()
        model.load_model(uri)
        model.set_param({"nthread": 1})
        if model.feature_names != predictors or model.num_boosted_rounds() != metadata["best_iteration"]:
            raise AssertionError("Saved Booster schema/round count differs from metadata")
        models.append(model)
    if len(models) != n_boot:
        raise AssertionError("Missing saved bootstrap models")
    summaries = {}
    for role, source, features in (("training", train, tx), ("prediction", pred, px)):
        directory = Path(cfg[f"{role}_output_dir"])
        spatial = gpd.read_parquet(directory / f"LOCAL_{role}_summary_{pair}.geoparquet")
        summary = pd.DataFrame(spatial.drop(columns="geometry")).reset_index(drop=True)
        ordinary = pd.read_parquet(directory / f"LOCAL_{role}_summary_{pair}.parquet")
        pd.testing.assert_frame_equal(summary, ordinary, check_dtype=False, check_exact=True)
        # Every original source field and row must survive the model step.
        pd.testing.assert_frame_equal(summary[source.columns], source.reset_index(drop=True), check_dtype=False, check_exact=True)
        if spatial.crs.to_epsg() != 32617:
            raise AssertionError("Output CRS changed")
        finite_xy = np.isfinite(summary.X) & np.isfinite(summary.Y)
        if not spatial.geometry.isna().equals(~finite_xy):
            raise AssertionError("Invalid-XY rows must have null geometry")
        np.testing.assert_allclose(spatial.loc[finite_xy].geometry.x, summary.loc[finite_xy, "X"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(spatial.loc[finite_xy].geometry.y, summary.loc[finite_xy, "Y"], rtol=0, atol=1e-8)
        usable = np.ones(len(source), dtype=bool) if role == "training" else source.prediction_row_usable.to_numpy(dtype=bool)
        positions = np.flatnonzero(usable)
        chunk_size = int(cfg["verification_chunk_rows"])
        for start in range(0, len(positions), chunk_size):
            rows = positions[start:start + chunk_size]
            matrix = xgb.DMatrix(features.iloc[rows].to_numpy(dtype=np.float32), feature_names=predictors, missing=np.nan)
            # Direct NumPy mean/std are independent of the worker's streaming Welford formula.
            predictions = np.stack([model.predict(matrix).astype(np.float64) for model in models])
            mean = predictions.mean(axis=0)
            sd = predictions.std(axis=0, ddof=1) if n_boot > 1 else np.zeros(len(rows))
            np.testing.assert_allclose(summary.mean_predicted_bathy_next.iloc[rows], mean, rtol=0, atol=1e-7)
            np.testing.assert_allclose(summary.uncertainty_sd_bathy_next.iloc[rows], sd, rtol=0, atol=1e-7)
        if (~usable).any() and not summary.loc[~usable, DERIVED].isna().all().all():
            raise AssertionError("Unusable prediction rows must retain NA model-derived fields")
        selected = summary.loc[usable]
        np.testing.assert_allclose(selected.mean_predicted_change, selected.mean_predicted_bathy_next - selected.bathy_t, rtol=0, atol=1e-8)
        np.testing.assert_allclose(selected.mean_predicted_rate, selected.mean_predicted_change / selected.interval_years, rtol=0, atol=1e-8)
        np.testing.assert_allclose(selected.mean_predicted_standard_change, selected.mean_predicted_rate * cfg["model_config"]["standard_horizon_years"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(selected.mean_predicted_standard_bathy, selected.bathy_t + selected.mean_predicted_standard_change, rtol=0, atol=1e-8)
        np.testing.assert_allclose(selected.mean_predicted_bathy_t1, selected.mean_predicted_bathy_next, rtol=0, atol=0)
        np.testing.assert_allclose(selected.uncertainty_sd_bathy_t1, selected.uncertainty_sd_bathy_next, rtol=0, atol=0)
        np.testing.assert_allclose(selected.uncertainty_sd_change, selected.uncertainty_sd_bathy_next, rtol=0, atol=0)
        np.testing.assert_allclose(selected.uncertainty_sd_rate, selected.uncertainty_sd_bathy_next / selected.interval_years, rtol=0, atol=1e-8)
        if (selected.uncertainty_sd_bathy_next < 0).any() or not np.isfinite(selected[DERIVED].to_numpy()).all():
            raise AssertionError("Usable predictions/SDs must be finite, with nonnegative SD")
        if role == "training":
            np.testing.assert_allclose(summary.prediction_residual, source.bathy_t1 - summary.mean_predicted_bathy_next, rtol=0, atol=1e-8)
            np.testing.assert_allclose(summary.delta_residual, source.delta_bathy - summary.mean_predicted_change, rtol=0, atol=1e-8)
        summaries[role] = {"rows": len(summary), "usable": int(usable.sum()), "preserved_unusable": int((~usable).sum()),
                           "source_fields": "PASS", "model_replay": "PASS", "change_rate_sd_formulas": "PASS"}
    importance = pd.read_parquet(base / f"LOCAL_importance_{pair}.parquet")
    if len(importance) != len(predictors) * n_boot or not np.isfinite(importance[["gain", "total_gain", "gain_fraction"]]).all().all():
        raise AssertionError("Missing or invalid importance rows")
    for _, group in importance.groupby("bootstrap_iteration"):
        fraction = float(group.gain_fraction.sum())
        np.testing.assert_allclose(fraction, 1.0 if group.total_gain.sum() > 0 else 0.0, rtol=0, atol=1e-8)
    if cfg["model_config"]["shap_sample_rows"] > 0:
        shap = pd.read_parquet(base / f"LOCAL_shap_summary_{pair}.parquet")
        if len(shap) != len(predictors) * n_boot or not np.isfinite(shap.mean_abs_shap).all() or (shap.mean_abs_shap < 0).any():
            raise AssertionError("Invalid SHAP summaries")
    if cfg["model_config"]["save_bootstrap_indices"]:
        for b in range(1, n_boot + 1):
            draws = pd.read_parquet(base / "bootstrap_indices" / f"bootstrap_{b:03d}_{pair}.parquet")
            idx = draws.training_row_index.to_numpy(dtype=int)
            if len(idx) != len(train) or (idx < 0).any() or (idx >= len(train)).any():
                raise AssertionError("Invalid bootstrap row indices")
            np.testing.assert_array_equal(draws.FID, train.FID.to_numpy()[idx])
    if cfg["model_config"]["round_selection_mode"] == "cv_with_floor" and metadata["best_iteration"] < cfg["model_config"]["minimum_nrounds"]:
        raise AssertionError("Round floor was not honored")
    if cfg["model_config"]["round_selection_mode"] != "fixed":
        folds = pd.read_parquet(base / f"LOCAL_spatial_cv_folds_{pair}.parquet")
        np.testing.assert_array_equal(folds.row_index, np.arange(len(train)))
        np.testing.assert_array_equal(folds.FID, train.FID)
        size = cfg["model_config"]["block_size_m"]
        blocks = pd.DataFrame({"gx": np.floor((train.X - train.X.min()) / size),
                               "gy": np.floor((train.Y - train.Y.min()) / size), "fold": folds.fold})
        if blocks.groupby(["gx", "gy"]).fold.nunique().max() != 1:
            raise AssertionError("A spatial block was split across folds")
        oof = pd.read_parquet(base / f"LOCAL_spatial_cv_oof_{pair}.parquet")
        if len(oof):
            if oof.row_index.duplicated().any():
                raise AssertionError("Duplicate CV holdout predictions")
            np.testing.assert_allclose(oof.observed_bathy_t1, train.bathy_t1.to_numpy()[oof.row_index.to_numpy(dtype=int)], rtol=1e-6, atol=1e-6)
            np.testing.assert_array_equal(oof.fold, folds.fold.to_numpy()[oof.row_index.to_numpy(dtype=int)])
    return {"summaries": summaries, "best_nrounds": metadata["best_iteration"], "n_boot": n_boot,
            "cv_status": metadata["cv_status"], "model_replay": "PASS", "schema": predictors}


def compare_r_outputs(cfg):
    """Compare user-provided R summaries on matching FIDs/coordinates without inventing tolerance thresholds."""
    results = {}
    limits = cfg.get("r_comparison_tolerances", {})
    known_limits = ("max_abs_prediction_difference_m", "rmse_prediction_difference_m", "max_abs_sd_difference_m")
    supplied_limits = {k: limits[k] for k in known_limits if limits.get(k) is not None}
    for key, value in supplied_limits.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"R comparison tolerance {key} must be finite and nonnegative")
    for role in ("training", "prediction"):
        reference_path = cfg.get("r_references", {}).get(role)
        if not reference_path:
            results[role] = {"status": "NOT_RUN", "reason": "No R model-output reference supplied"}
            continue
        python_path = Path(cfg[f"{role}_output_dir"]) / f'LOCAL_{role}_summary_{cfg["pair"]}.parquet'
        if Path(reference_path).resolve() == python_path.resolve():
            raise ValueError("An R reference cannot be the Python output being compared")
        suffix = Path(reference_path).suffix.lower()
        if suffix == ".csv":
            ref = pd.read_csv(reference_path)
        elif suffix in (".parquet", ".geoparquet"):
            ref = pd.read_parquet(reference_path)
        else:
            raise ValueError("Model-output references must already be CSV/Parquet/GeoParquet")
        py = pd.read_parquet(python_path)
        required = {"FID", "X", "Y", "bathy_t", "interval_years"}
        if not required.issubset(ref.columns) or ref.FID.isna().any() or ref.FID.duplicated().any():
            raise ValueError(f"{role} R reference requires unique FID and X/Y/bathy_t/interval_years")
        for key, expected in (("tile_id", cfg["tile_id"]), ("pair_id", cfg["pair"])):
            if key in ref and not ref[key].eq(expected).all():
                raise ValueError(f"R reference {key} does not match this tile/pair")
        if "year_pair" in ref and not ref.year_pair.eq(cfg["pair"]).all():
            raise ValueError("R reference year_pair mismatch")
        indexed = ref.set_index("FID")
        if not py.FID.isin(indexed.index).all():
            raise ValueError(f"R reference is missing {int((~py.FID.isin(indexed.index)).sum())} Python output FIDs")
        aligned = indexed.loc[py.FID].reset_index()
        for col in ("X", "Y", "bathy_t", "interval_years"):
            np.testing.assert_allclose(py[col], aligned[col], rtol=0, atol=1e-6, equal_nan=True)
        field_map = {"prediction": "mean_predicted_bathy_t1", "sd": "uncertainty_sd_bathy_t1"}
        record = {"reference": str(reference_path), "matched_rows": len(py), "reference_rows": len(ref),
                  "reference_only_rows": int((~ref.FID.isin(py.FID)).sum()), "fields": {}}
        measurements = {}
        nan_mismatch = False
        for name, col in field_map.items():
            if col not in aligned:
                raise ValueError(f"R reference missing {col}")
            a = pd.to_numeric(py[col], errors="raise").to_numpy(dtype=float)
            b = pd.to_numeric(aligned[col], errors="raise").to_numpy(dtype=float)
            usable = np.ones(len(py), dtype=bool) if role == "training" else py.prediction_row_usable.to_numpy(dtype=bool)
            mismatch = int(np.sum(usable & (np.isfinite(a) != np.isfinite(b))))
            nan_mismatch |= mismatch > 0
            finite = usable & np.isfinite(a) & np.isfinite(b)
            d = a[finite] - b[finite]
            maximum = float(np.max(np.abs(d))) if len(d) else None
            rmse = float(np.sqrt(np.mean(d**2))) if len(d) else None
            record["fields"][name] = {"n_finite_compared": int(len(d)), "finite_mask_mismatches_on_usable_rows": mismatch,
                                      "max_abs_difference_m": maximum, "rmse_difference_m": rmse,
                                      "bias_python_minus_r_m": float(np.mean(d)) if len(d) else None,
                                      "unusable_rows_excluded_from_numeric_comparison": int((~usable).sum())}
            if name == "prediction":
                measurements.update(max_abs_prediction_difference_m=maximum, rmse_prediction_difference_m=rmse)
            else:
                measurements["max_abs_sd_difference_m"] = maximum
        record["tolerances"] = supplied_limits
        if not supplied_limits:
            record["status"] = "DESCRIPTIVE_ONLY"
        else:
            record["status"] = "PASS" if not nan_mismatch and all(measurements[k] is not None and measurements[k] <= value for k, value in supplied_limits.items()) else "FAIL"
        record["comparison_scope"] = "matching rows of supplied R/Python summaries; same training settings/seeds/folds/version must be established separately"
        results[role] = record
    return results


def run_test(module, repository_root, overrides=None, overwrite=False, validate_only=False,
             profile="full", execution="dask"):
    """Train or replay the actual engine and write independent contract/R statuses.

    --validate-only verifies current input hashes and scientific configuration
    against the completed run manifest before replay; it does not retrain.
    """
    engine = module.XGBoostModelingEngine({"env":"local", "repository_root":str(repository_root),
                                         "modeling_settings":overrides or {}})
    cfg = engine.configuration(profile)
    directory = Path(cfg["training_output_dir"]); directory.mkdir(parents=True,exist_ok=True)
    pair = cfg["pair"]
    report_path = directory / f"XGBoost_Modeling_Engine_DV_test_report_{pair}.json"
    manifest_path = directory / f"LOCAL_test_run_manifest_{pair}.json"
    report = {"tile_id":cfg["tile_id"],"pair":pair,"crs":cfg["crs"],"profile":profile,
        "execution":"validate_only" if validate_only else execution,
        "python":sys.version,"platform":platform.platform(),"contract_status":"NOT_RUN",
        "scope":"Saved-model execution/output-contract checks; not R model parity or held-out accuracy",
        "versions":{p:version(p) for p in ("xgboost","numpy","pandas","geopandas","pyarrow","dask","distributed")}}
    try:
        train,pred,tx,px,schema,identity = engine.load_inputs(cfg)
        report["data_prep_coverage"] = json.loads(Path(cfg["data_prep_report"]).read_text()).get("coverage", "Not specified in DataPrep report")
        report.update(input_rows={"training":len(train),"prediction":len(pred)},input_sha256=identity,
            schema=schema,model_config=cfg["model_config"],
            predictor_na_counts={p:int(train[p].isna().sum()) for p in schema},
            training_sample_weight_summary={"min":float(train.sample_weight.min()),
                                           "mean":float(train.sample_weight.mean()),"max":float(train.sample_weight.max())},
            source_files={key:cfg[key] for key in identity})
        expected_contract = {"identity":identity,"model_config":cfg["model_config"],"schema":schema,"profile":profile}
        if validate_only:
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("contract") != expected_contract or manifest.get("model_execution_status") != "COMPLETED":
                raise ValueError("Saved run does not match current inputs/settings, or execution did not complete")
        else:
            print(f"Starting {profile} XGBoost: {cfg['tile_id']} | {pair} | {len(train)} fitting rows | {len(schema)} predictors", flush=True)
            print(f"CV/bootstrap progress is recorded in: {directory / 'log_prints.txt'}", flush=True)
            report["task_results"] = engine.run(overwrite=overwrite, execution=execution, profile=profile)
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("contract") != expected_contract or manifest.get("model_execution_status") != "COMPLETED":
                raise ValueError("Input/configuration changed between preflight and completed model run")
        report["validation"] = verify_saved_outputs(cfg,train,pred,tx,px,schema)
        for key,digest in identity.items():
            if module._file_hash(cfg[key]) != digest:
                raise ValueError(f"DataPrep input changed during validation: {cfg[key]}")
        report["contract_status"] = "PASS"
        report["r_comparison"] = compare_r_outputs(cfg)
        if any(r["status"] == "FAIL" for r in report["r_comparison"].values()):
            raise AssertionError("R model-output comparison exceeds your supplied tolerance(s)")
        report["status"] = "PASS"
    except Exception as exc:
        report.update(status="FAIL",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
        if report["contract_status"] != "PASS":
            report["contract_status"] = "FAIL"
        raise
    finally:
        report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"PASS: {profile} model execution/output-contract checks. Report: {report_path}")
    for role,outcome in report["r_comparison"].items():
        print(f"R/Python {role} model-output comparison: {outcome['status']}")
    return report


def write_fixture(path, frame):
    """Write synthetic point GeoParquet while retaining null geometry for invalid XY."""
    path.parent.mkdir(parents=True,exist_ok=True)
    valid = np.isfinite(frame.X)&np.isfinite(frame.Y)
    points = gpd.points_from_xy(frame.X.fillna(0),frame.Y.fillna(0),crs=32617)
    points[~valid] = None
    gpd.GeoDataFrame(frame.copy(),geometry=points).to_parquet(path,index=False)


def make_fixture(root, pair="2004_2006"):
    """Create small DataPrep-shaped inputs and an explicitly synthetic PASS report.

    Missing drivers, an entirely missing driver, extra source uncertainty and
    unusable prediction rows expose accidental imputation/schema/coverage changes.
    """
    tile="BH4S556X_3"; rows=np.arange(256); years=[int(y) for y in pair.split("_")]
    interval=float(years[1]-years[0]); dynamic=f"tsm_mean_{pair}"
    train=pd.DataFrame({"X":340000.+(rows%16)*8,"Y":3024000.+(rows//16)*8,
        "FID":rows+1,"tile_id":tile,"pair_id":pair,"year_t":years[0],"year_t1":years[1],
        "bathy_t":-18+np.cos(rows/12)*2,"distance_from_shore":rows.astype(float)*10,
        dynamic:np.nan,"interval_years":interval,"survey_end_date":2022.})
    train["bathy_t1"]=train.bathy_t+1.5*np.sin(rows/20)+.3
    train["delta_bathy"]=train.bathy_t1-train.bathy_t
    train["delta_rate"]=train.delta_bathy/interval
    weights=np.sqrt(abs(train.delta_bathy)+1e-6); train["sample_weight"]=weights/weights.mean()
    train.loc[0:10,"distance_from_shore"]=np.nan
    pred=train.drop(columns=["bathy_t1","delta_bathy","delta_rate","sample_weight","year_t","year_t1"]).copy()
    pred["UC_t"]=1.15
    pred.loc[0,"X"]=np.nan; pred.loc[1,"bathy_t"]=np.nan
    pred["prediction_row_usable"]=np.isfinite(pred.X)&np.isfinite(pred.Y)&np.isfinite(pred.bathy_t)
    schema=["bathy_t",dynamic,"distance_from_shore"]
    for role,frame in (("training",train),("prediction",pred)):
        directory=root/f"outputs/local_test_outputs/{role.title()}_data_grid_tiles/{tile}"
        write_fixture(directory/f"ML_ready_{role}_{pair}.geoparquet",frame)
    directory=root/f"outputs/local_test_outputs/Training_data_grid_tiles/{tile}"
    pd.DataFrame({"predictor":schema}).to_parquet(directory/f"locked_predictors_{pair}.parquet",index=False)
    (directory/f"Data_Prep_Engine_Unit_Test_report_{pair}.json").write_text(json.dumps({
        "status":"PASS","tile_id":tile,"pair":pair,"crs":"EPSG:32617","synthetic_fixture":True,
        "locked_predictors":schema,"counts":{"training":{"output_rows":len(train)},"prediction":{"output_rows":len(pred)}}}))
    overrides={"pair":pair,"verification_chunk_rows":37,"model_config":{
        "round_selection_mode":"cv_with_floor","minimum_nrounds":8,"cv_nrounds":24,
        "cv_fallback_nrounds":8,"fixed_nrounds":20,"early_stopping_rounds":4,
        "n_boot":3,"n_folds":3,"block_size_m":32.,"eta":.2,"gamma":0.,"reg_alpha":0.,
        "shap_sample_rows":24,"xgb_memory_limit":"1GB","xgb_processes":False,"verbose_logging":False}}
    return overrides,train,pred,schema


def run_self_tests(module):
    """Run small real XGBoost/actual Dask checks without using scientific input files."""
    class ModelingRegressionTests(unittest.TestCase):
        """Cover locked features, native NAs, model replay, bootstrap fit and lifecycle safeguards."""
        def setUp(self):
            """Create a private repository-layout fixture for each regression case."""
            self.temp=tempfile.TemporaryDirectory(prefix="hhm_xgb_repository_test_")
            self.root=Path(self.temp.name)
            self.overrides,self.train,self.pred,self.schema=make_fixture(self.root)

        def tearDown(self):
            """Remove only this test's private fixture and model files."""
            self.temp.cleanup()

        def test_schema_native_na_and_input_report(self):
            """Keep all-NA features and row coverage; reject leakage, bad weights and stale schema."""
            a,b,tx,px,schema,_=module._prepare_model_data(self.train,self.pred,self.schema)
            self.assertEqual(schema,self.schema)
            self.assertTrue(tx["tsm_mean_2004_2006"].isna().all())
            self.assertEqual(len(b),256); self.assertEqual(b.prediction_row_usable.sum(),254)
            self.assertNotIn("UC_t",tx)
            for leak in ("bathy_t1","year_t1","sample_weight"):
                with self.assertRaises(ValueError):module._prepare_model_data(self.train,self.pred,self.schema+[leak])
            for column,value in (("sample_weight",0),("distance_from_shore",np.inf)):
                bad=self.train.copy();bad.loc[0,column]=value
                with self.assertRaises(ValueError):module._prepare_model_data(bad,self.pred,self.schema)
            with self.assertRaises(ValueError):module._prepare_model_data(self.train,self.pred.drop(columns="distance_from_shore"),self.schema)
            engine=module.XGBoostModelingEngine({"repository_root":str(self.root),"modeling_settings":self.overrides})
            cfg=engine.configuration()
            report=Path(cfg["data_prep_report"]); data=json.loads(report.read_text()); data["locked_predictors"]=list(reversed(self.schema));report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,"differ"):engine.load_inputs(cfg)

        def test_blocks_round_modes_and_cleanup(self):
            """Keep blocks intact, honour fixed/fallback rounds and preserve setup errors."""
            coords=pd.DataFrame({"X":np.repeat(np.arange(4)*20,20),"Y":np.tile(np.arange(20),4)})
            folds=module._spatial_fold_ids(coords,20,3,11)
            for x in coords.X.unique():self.assertEqual(len(set(folds[coords.X==x])),1)
            audit={}
            self.assertEqual(module._choose_rounds_spatial_cv(coords,coords,coords.X.to_numpy(),np.ones(len(coords)),
                {"round_selection_mode":"fixed","fixed_nrounds":17},11,audit),17)
            tiny=pd.DataFrame({"X":np.zeros(30),"Y":np.zeros(30)})
            audit={}
            self.assertEqual(module._choose_rounds_spatial_cv(tiny,tiny,np.ones(30),np.ones(30),
                {"round_selection_mode":"cv_with_floor","cv_fallback_nrounds":4,"minimum_nrounds":12,"block_size_m":200,"n_folds":3},11,audit),12)
            self.assertIsInstance(audit["oof"],pd.DataFrame)
            engine=module.XGBoostModelingEngine({"repository_root":str(self.root),"modeling_settings":self.overrides})
            def fail_setup(*args,**kwargs):
                """Inject a setup error before client/cluster resources exist."""
                raise RuntimeError("deliberate model Dask setup failure")
            engine.setup_dask=fail_setup
            with self.assertRaisesRegex(RuntimeError,"deliberate model Dask setup failure"):engine.run(execution="dask")
            self.assertFalse(engine.local_tmp_dir.exists())

        def test_dask_replay_bootstrap_and_shap_independence(self):
            """Independently refit a saved bootstrap and confirm SHAP cannot change the ensemble."""
            report=run_test(module,self.root,self.overrides,execution="dask")
            self.assertEqual(report["contract_status"],"PASS")
            self.assertEqual(report["r_comparison"]["training"]["status"],"NOT_RUN")
            self.assertEqual(report["validation"]["summaries"]["prediction"]["preserved_unusable"],2)
            engine=module.XGBoostModelingEngine({"repository_root":str(self.root),"modeling_settings":self.overrides})
            cfg=engine.configuration();base=Path(cfg["training_output_dir"]);pair=cfg["pair"]
            meta=json.loads((base/f"LOCAL_model_metadata_{pair}.json").read_text())
            idx=pd.read_parquet(base/"bootstrap_indices"/f"bootstrap_001_{pair}.parquet").training_row_index.to_numpy(int)
            mc=cfg["model_config"]
            parameters={"objective":"reg:squarederror","eval_metric":["rmse","mae"],"max_depth":mc["max_depth"],
                "eta":mc["eta"],"gamma":mc["gamma"],"lambda":mc["reg_lambda"],"alpha":mc["reg_alpha"],
                "subsample":mc["subsample"],"colsample_bytree":mc["colsample_bytree"],"tree_method":mc["tree_method"],
                "max_bin":mc["max_bin"],"nthread":1,"seed":meta["bootstrap_seeds"][0]["xgb_seed"]}
            values=self.train[self.schema].to_numpy(dtype=np.float32)
            matrix=xgb.DMatrix(values[idx],label=self.train.bathy_t1.to_numpy(dtype=np.float32)[idx],
                weight=self.train.sample_weight.to_numpy()[idx],feature_names=self.schema,missing=np.nan)
            reference=xgb.train(parameters,matrix,num_boost_round=meta["best_iteration"],verbose_eval=False)
            saved=xgb.Booster();saved.load_model(meta["model_uris"][0]);saved.set_param({"nthread":1})
            all_rows=xgb.DMatrix(values,feature_names=self.schema,missing=np.nan)
            np.testing.assert_array_equal(saved.predict(all_rows),reference.predict(all_rows))
            run_test(module,self.root,self.overrides,validate_only=True)
            before=pd.read_parquet(base/f"LOCAL_training_summary_{pair}.parquet").mean_predicted_bathy_next
            second=copy.deepcopy(self.overrides);second["model_config"]["shap_sample_rows"]=0
            second["training_output_dir"]="outputs/local_test_outputs/second_training"
            second["prediction_output_dir"]="outputs/local_test_outputs/second_prediction"
            run_test(module,self.root,second,execution="inline")
            other=self.root/second["training_output_dir"]/f"LOCAL_training_summary_{pair}.parquet"
            np.testing.assert_array_equal(before,pd.read_parquet(other).mean_predicted_bathy_next)
            with self.assertRaises(FileExistsError):run_test(module,self.root,self.overrides,execution="inline")
            source=Path(cfg["training_geoparquet"]);changed=gpd.read_parquet(source);changed.loc[0,"sample_weight"]*=1.01;changed.to_parquet(source,index=False)
            with self.assertRaisesRegex(ValueError,"Saved run does not match"):run_test(module,self.root,self.overrides,validate_only=True)

        def test_alternate_pair_one_bootstrap_and_fallback(self):
            """Use another interval and single-block CV fallback; one bootstrap must give zero SD."""
            overrides,train,pred,schema=make_fixture(self.root,"2006_2010")
            overrides["model_config"].update(n_boot=1,block_size_m=10000.,minimum_nrounds=4,cv_fallback_nrounds=4)
            report=run_test(module,self.root,overrides,execution="inline")
            self.assertEqual(report["validation"]["cv_status"],"fewer_than_two_folds_using_fallback")
            base=self.root/"outputs/local_test_outputs/Training_data_grid_tiles/BH4S556X_3"
            output=pd.read_parquet(base/"LOCAL_training_summary_2006_2010.parquet")
            np.testing.assert_allclose(output.uncertainty_sd_bathy_t1,0)
            np.testing.assert_allclose(output.mean_predicted_rate,output.mean_predicted_change/4)

        def test_smoke_isolation_and_stale_artifact_cleanup(self):
            """Keep smoke models separate and remove only stale same-pair model-owned files."""
            changes=copy.deepcopy(self.overrides)
            changes["smoke_model_overrides"]={"fixed_nrounds":4,"n_boot":1,"shap_sample_rows":4}
            report=run_test(module,self.root,changes,execution="inline",profile="smoke")
            self.assertEqual(report["validation"]["n_boot"],1)
            engine=module.XGBoostModelingEngine({"repository_root":str(self.root),"modeling_settings":changes})
            full=engine.configuration("full");smoke=engine.configuration("smoke")
            self.assertNotEqual(full["training_output_dir"],smoke["training_output_dir"])
            self.assertEqual(full["training_geoparquet"],smoke["training_geoparquet"])
            root=Path(smoke["training_output_dir"])
            stale=root/"models/model_boot_999_2004_2006.json";stale.write_text("stale")
            another=root/"models/model_boot_999_2010_2015.json";another.write_text("keep another pair")
            cv=root/"LOCAL_spatial_cv_oof_2004_2006.parquet"
            cv.write_text("stale CV artifact")
            protected=module._file_hash(smoke["locked_predictors_parquet"])
            run_test(module,self.root,changes,execution="inline",profile="smoke",overwrite=True)
            self.assertFalse(stale.exists());self.assertTrue(another.exists())
            self.assertTrue(pd.read_parquet(cv).empty)
            self.assertEqual(protected,module._file_hash(smoke["locked_predictors_parquet"]))

        def test_comparator_and_corruption_detection(self):
            """Exercise optional reference mechanics and reject a corrupted ensemble mean."""
            run_test(module,self.root,self.overrides,execution="inline")
            engine=module.XGBoostModelingEngine({"repository_root":str(self.root),"modeling_settings":self.overrides})
            cfg=engine.configuration();pair=cfg["pair"]
            for role in ("training","prediction"):
                summary=pd.read_parquet(Path(cfg[f"{role}_output_dir"])/f"LOCAL_{role}_summary_{pair}.parquet")
                reference=self.root/f"synthetic_{role}_reference.parquet";summary.to_parquet(reference,index=False)
                cfg["r_references"][role]=str(reference)
            self.assertEqual(compare_r_outputs(cfg)["training"]["status"],"DESCRIPTIVE_ONLY")
            cfg["r_comparison_tolerances"].update(max_abs_prediction_difference_m=0.,max_abs_sd_difference_m=0.)
            self.assertEqual(compare_r_outputs(cfg)["prediction"]["status"],"PASS")
            path=Path(cfg["r_references"]["training"]);data=pd.read_parquet(path);data.loc[0,"mean_predicted_bathy_t1"]+=1;data.to_parquet(path,index=False)
            self.assertEqual(compare_r_outputs(cfg)["training"]["status"],"FAIL")
            prepared=engine.load_inputs(cfg)
            path=Path(cfg["training_output_dir"])/f"LOCAL_training_summary_{pair}.parquet"
            data=pd.read_parquet(path);data.loc[0,"mean_predicted_bathy_next"]+=1;data.to_parquet(path,index=False)
            with self.assertRaises(AssertionError):verify_saved_outputs(cfg,*prepared[:5])

    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ModelingRegressionTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)


def main():
    """Provide a Windows-safe, repo-location-independent terminal entry point."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=None)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--profile",choices=("full","smoke"),default="full")
    parser.add_argument("--execution",choices=("dask","inline"),default="dask")
    parser.add_argument("--self-test",action="store_true",help="Synthetic software regression, not your scientific run")
    args=parser.parse_args()
    root=args.repo_root.resolve() if args.repo_root else find_repository_root(Path(__file__))
    module=load_engine(root,self_test=args.self_test)
    if args.self_test:run_self_tests(module)
    else:run_test(module,root,overwrite=args.overwrite,validate_only=args.validate_only,profile=args.profile,execution=args.execution)


if __name__ == "__main__":
    # Required when Dask spawns Windows worker processes.
    main()
