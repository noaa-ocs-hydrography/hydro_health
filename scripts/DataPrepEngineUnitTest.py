"""Run the actual local DataPrepEngine on one tile/pair and compare with R.

# move everything thats in the config file into the class, we can decide what goes into the master config later
# update paths 
# make paths relative, to find other files in the same directory / src directory 
HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))


# 
This is an integration/parity test, despite its requested UnitTest filename.
It prepares data only; it does not fit or evaluate a machine learning model.
"""
import argparse
import json
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from importlib.metadata import version

import geopandas as gpd
import numpy as np
import pandas as pd
from DataPrepEngine import DataPrepEngine
from FstToGeoParquet import load_config, convert, conversion_identity, batch_paths, master_predictors


def check_outputs(cfg):
    """Compare saved tables with independently prepared R references and verify geometry."""
    checks = {}
    counts = {}
    for role in ("training", "prediction"):
        base = Path(cfg[f"{role}_out"])
        output = base / f'ML_ready_{role}_{cfg["pair"]}.geoparquet'
        spatial = gpd.read_parquet(output)
        actual = pd.DataFrame(spatial.drop(columns="geometry")).reset_index(drop=True)
        expected = pd.read_parquet(base / f'R_reference_{role}_{cfg["pair"]}.parquet').reset_index(drop=True)
        # Column order is not scientifically relevant, but row order and every value are.
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-8)
        if set(actual.columns) != set(expected.columns):
            raise AssertionError(f"Unexpected output columns: {set(actual.columns) ^ set(expected.columns)}")
        assert spatial.crs.to_epsg() == 32617
        finite_xy = np.isfinite(actual.X) & np.isfinite(actual.Y)
        assert spatial.geometry.isna().equals(~finite_xy)
        located = spatial.loc[finite_xy]
        np.testing.assert_allclose(located.geometry.x, located.X, rtol=0, atol=1e-8)
        np.testing.assert_allclose(located.geometry.y, located.Y, rtol=0, atol=1e-8)
        miss = pd.read_parquet(base / f'predictor_missingness_{role}_{cfg["pair"]}.parquet')
        miss_r = pd.read_parquet(base / f'R_reference_missingness_{role}_{cfg["pair"]}.parquet')
        pd.testing.assert_frame_equal(miss.reset_index(drop=True), miss_r.reset_index(drop=True), check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-12)
        if role == "training":
            np.testing.assert_allclose(actual.delta_bathy, actual.bathy_t1 - actual.bathy_t, rtol=0, atol=1e-8)
            np.testing.assert_allclose(actual.delta_rate, actual.delta_bathy / actual.interval_years, rtol=0, atol=1e-8)
            assert np.isfinite(actual.sample_weight).all() and (actual.sample_weight > 0).all()
        else:
            assert len(actual) == len(gpd.read_parquet(batch_paths(cfg, role)))
            assert "sample_weight" not in actual.columns
            usable = finite_xy & np.isfinite(actual.bathy_t)
            assert actual.prediction_row_usable.equals(usable)
        checks[role] = "PASS: all fields, row order, NA positions, geometry, CRS, and missingness match reference"
        counts[role] = {"ready_rows": len(actual), "rows_with_null_geometry": int(spatial.geometry.isna().sum())}
    schema = pd.read_parquet(Path(cfg["training_out"]) / f'locked_predictors_{cfg["pair"]}.parquet').predictor.tolist()
    manifest = json.loads((Path(cfg["training_out"]) / "conversion_manifest.json").read_text())
    assert schema == manifest["predictors"] and "distance_from_shore" in schema
    ranges = pd.read_parquet(Path(cfg["training_out"]) / f'predictor_ranges_training_{cfg["pair"]}.parquet')
    train = gpd.read_parquet(Path(cfg["training_out"]) / f'ML_ready_training_{cfg["pair"]}.geoparquet')
    for row in ranges.itertuples(index=False):
        values = train[row.predictor].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        expected = [np.min(values), np.max(values), np.mean(values)] if len(values) else [np.nan] * 3
        np.testing.assert_allclose([row.min_training, row.max_training, row.mean_training], expected, equal_nan=True)
    return {"checks": checks, "counts": counts, "locked_predictors": schema, "range_check": "PASS"}


def run_test(cfg, do_convert=False, overwrite=False):
    """Convert on request, run the real class through Dask, then save an honest pass/fail report."""
    report_path = Path(cfg["training_out"]) / "DataPrepEngineUnitTest_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"tile_id": cfg["tile_id"], "pair": cfg["pair"], "crs": cfg["crs"],
              "python": sys.version, "platform": platform.platform(),
              "versions": {m: version(m) for m in ("pandas", "numpy", "geopandas", "pyarrow", "dask", "distributed")},
              "model_config": cfg["model_config"], "scope": "preparation only; no ML fitting"}
    engine = None
    try:
        if do_convert:
            convert(cfg, overwrite=overwrite)
        manifest_path = Path(cfg["training_out"]) / "conversion_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["identity"] != conversion_identity(cfg):
            raise ValueError("Prepared data are stale: config, predictor list, or source file metadata changed. Rerun conversion.")
        for role in ("training", "prediction"):
            # Restrict discovery to precisely one tile and pair, even if an out folder has other tests.
            discovered = sorted((Path(cfg[f"{role}_out"]) / "raw").rglob(f"*_{role}_batch.parquet"))
            if discovered != [batch_paths(cfg, role)]:
                raise ValueError(f"Expected exactly one {role} batch for the configured tile/pair; found {discovered}")
            if not overwrite and (Path(cfg[f"{role}_out"]) / f'ML_ready_{role}_{cfg["pair"]}.geoparquet').exists():
                raise FileExistsError("Prepared ML outputs already exist. Use --overwrite for a deliberate rerun.")
        engine = DataPrepEngine({
            "env": "local", "eco_regions": SimpleNamespace(value=["ER_3"]),
            "model_config": {**cfg["model_config"], "crs": cfg["crs"]},
            "local_paths": {
                "training_tiles_dir": str(Path(cfg["training_out"]) / "raw"),
                "prediction_tiles_dir": str(Path(cfg["prediction_out"]) / "raw"),
                "ml_prep_train_dir": cfg["training_out"], "ml_prep_pred_dir": cfg["prediction_out"],
                "log_dir": cfg["training_out"],
            },
        })
        engine.master_predictors = master_predictors(cfg)
        results = engine.run()
        if len(results) != 1 or results[0]["tile_id"] != cfg["tile_id"] or results[0]["pair"] != cfg["pair"]:
            raise AssertionError(f"Unexpected tasks: {results}")
        report.update(check_outputs(cfg))
        report["results"] = results
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if engine is not None:
            engine.cleanup_resources(cfg["training_out"])
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: preparation matches the R reference. Report: {report_path}")
    return report


if __name__ == "__main__":
    # Required for multiprocessing when starting local Dask workers on Windows.
    # parser = argparse.ArgumentParser(description=__doc__)
    # parser.add_argument("--config", default=str(Path(__file__).with_name("local_test_config.json")))
    # parser.add_argument("--convert", action="store_true", help="Convert the FST inputs before testing")
    # parser.add_argument("--overwrite", action="store_true", help="Replace this test's previously generated outputs")
    # args = parser.parse_args()
    # run_test(load_config(args.config), args.convert, args.overwrite)
    print("stating engine")
    # engine = DataPrepEngine() # this makes the engine varaible a class 
    # engine.run()
    print("done")
# add print statemnts for proessing and done
# update the envinronment to point to the branch environment 