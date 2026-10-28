"""Validate the repository raster/PDF exporter without additional support/config scripts.

Normal runs read the successful model handoff, execute COG export through local
Dask, verify raster cells against independent source grouping and check PDF/PNG
files. Export PASS is distinct from model accuracy and optional R raster parity.
--self-test uses explicitly synthetic model-like tables, never the user's data.
"""
import argparse
import copy
import importlib
import json
import platform
import shutil
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
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from shapely import wkb
from shapely.geometry import mapping, box, Polygon
from pypdf import PdfReader
from PIL import Image


def find_repository_root(script_path):
    """Find src/hydro_health above this script, including when test folders are rearranged."""
    for parent in Path(script_path).resolve().parents:
        if (parent/"src/hydro_health").is_dir(): return parent
    raise FileNotFoundError("Cannot find repository src/hydro_health above this test script")


def load_engine(root, self_test=False):
    """Import the real project engine; only explicit synthetic runs can use the base double."""
    sys.path.insert(0,str(Path(root)/"src"))
    if self_test and not (Path(root)/"src/hydro_health/engines/Engine.py").is_file():
        base = types.ModuleType("hydro_health.engines.Engine")

        class Engine:
            """Isolated synthetic-test base, never a fallback for normal repository runs."""
            def __init__(self):
                """Match the base resource handles without provider/GDAL services."""
                self.client = None; self.cluster = None

            def setup_dask(self, env, **kwargs):
                """Exercise actual local Dask with threaded workers in the portable fixture."""
                from dask.distributed import Client,LocalCluster
                self.cluster = LocalCluster(**kwargs,dashboard_address=None)
                self.client = Client(self.cluster)

            @staticmethod
            def write_message_dask(message,output_folder):
                """Append messages to the same synthetic-folder log as the production base."""
                folder = Path(output_folder);folder.mkdir(parents=True,exist_ok=True)
                with (folder/"log_prints.txt").open("a",encoding="utf-8") as stream:stream.write(str(message)+"\n")

            write_message = write_message_dask

            def cleanup_resources(self, output_folder):
                """Mirror base cleanup and dispatch the exporter defensive close_dask override."""
                self.close_dask()
                if hasattr(self,"local_tmp_dir"):shutil.rmtree(self.local_tmp_dir,ignore_errors=True)

        base.Engine = Engine
        sys.modules[base.__name__] = base
    return importlib.import_module("hydro_health.engines.tiling.MLRasterExportEngine")


def verify_raster(path,source,column,template,geometry,crs,nodata,rtol=1e-6,atol=2e-5):
    """Independently group source points by cell and compare all populated pixels."""
    with rasterio.open(path) as src, rasterio.open(template) as ref:
        actual = src.read(1,masked=True)
        assert src.crs==ref.crs==rasterio.crs.CRS.from_user_input(crs), 'Raster CRS mismatch'
        assert src.count==1 and src.dtypes[0]=='float32' and src.nodata==nodata
        assert src.tags(ns='IMAGE_STRUCTURE').get('LAYOUT')=='COG', 'COG layout missing'
        t, base = src.transform,ref.transform
        assert t.a==base.a and t.e==base.e and t.b==t.d==0
        offsets = ((t.c-base.c)/base.a, (t.f-base.f)/base.e)
        assert np.allclose(offsets,np.round(offsets),atol=1e-7), 'Template alignment mismatch'
        assert src.width>0 and src.height>0
        minx,miny,maxx,maxy = wkb.loads(geometry).bounds
        c0 = int(np.floor((minx-base.c)/base.a+1e-8))
        c1 = int(np.ceil((maxx-base.c)/base.a-1e-8))
        r0 = int(np.floor((maxy-base.f)/base.e+1e-8))
        r1 = int(np.ceil((miny-base.f)/base.e-1e-8))
        assert src.shape == (r1-r0,c1-c0), 'Output extent does not cover the full outward-snapped tile'
        np.testing.assert_allclose([t.c,t.f],[base.c+c0*base.a,base.f+r0*base.e],rtol=0,atol=1e-7)
        assert all(a%16==b%16==0 for a,b in src.block_shapes)
        xyv = source[['X','Y',column]].apply(pd.to_numeric,errors='coerce')
        finite = np.isfinite(xyv.to_numpy()).all(axis=1)
        d = xyv.loc[finite].copy()
        d['c'] = np.floor((d.X-t.c)/t.a).astype(int)
        d['r'] = np.floor((d.Y-t.f)/t.e).astype(int)
        d = d[(d.r>=0)&(d.r<src.height)&(d.c>=0)&(d.c<src.width)]
        if column=='survey_year_bin_code':
            expected = d.groupby(['r','c'])[column].agg(lambda s:s.mode().iloc[0])
        else:
            expected = d.groupby(['r','c'])[column].mean()
        inside = geometry_mask([mapping(wkb.loads(geometry))],out_shape=actual.shape,
                               transform=t,invert=True,all_touched=False)
        expected_grid = np.full(actual.shape,np.nan)
        if len(expected):
            r,c = np.array(expected.index.tolist()).T
            expected_grid[r,c] = expected.to_numpy()
        expected_grid[~inside] = np.nan
        expected_mask = ~np.isfinite(expected_grid)
        np.testing.assert_array_equal(np.ma.getmaskarray(actual),expected_mask,
                                      err_msg='Raster NoData mask does not match finite source cells/polygon')
        np.testing.assert_allclose(actual.data[~expected_mask],expected_grid[~expected_mask],
                                   rtol=rtol,atol=atol,err_msg='Raster values differ from per-cell source means')
        return {'file':str(path),'column':column,'finite_source_rows':int(finite.sum()),
                'populated_cells':int((~expected_mask).sum()),'width':src.width,'height':src.height,
                'transform':list(t),'crs':str(src.crs),'overviews':src.overviews(1),'cog_layout':True}


def compare_r_reference(python_path,reference_path,atol,rtol):
    """Compare an explicitly supplied R raster; do not resample to conceal misalignment."""
    with rasterio.open(python_path) as py,rasterio.open(reference_path) as r:
        assert py.crs==r.crs and py.shape==r.shape and py.transform==r.transform, 'R grid mismatch'
        a,b = py.read(1,masked=True),r.read(1,masked=True)
        np.testing.assert_array_equal(np.ma.getmaskarray(a),np.ma.getmaskarray(b),err_msg='R NoData mask mismatch')
        keep = ~np.ma.getmaskarray(a)
        np.testing.assert_allclose(a.data[keep],b.data[keep],atol=atol,rtol=rtol)
        difference = a.data[keep].astype(float)-b.data[keep].astype(float)
        return {'status':'PASS','reference':str(reference_path),'cells':int(keep.sum()),
                'max_abs_difference':float(np.abs(difference).max()) if len(difference) else None}


def run_test(module,root,overrides=None,overwrite=False,execution="dask",validate_only=False):
    """Run or revalidate each configured pair and record separate export/R comparison statuses."""
    engine = module.MLRasterExportEngine({"env":"local","repository_root":str(root),"raster_settings":overrides or {}})
    reports = []
    for cfg in engine.configurations():
        out = Path(cfg["training_output_dir"]);out.mkdir(parents=True,exist_ok=True)
        report_path = out/f"ML_Raster_Export_Engine_DV_test_report_{cfg['pair']}.json"
        manifest_path = out/f"LOCAL_raster_export_manifest_{cfg['pair']}.json"
        report = {"status":"RUNNING","contract_status":"NOT_RUN","tile_id":cfg["tile_id"],"pair":cfg["pair"],
            "crs":cfg["crs"],"execution":"validate_only" if validate_only else execution,"configuration":cfg,
            "scope":"Rasterization/export and PDF/PNG contract checks; not model accuracy or R model parity",
            "r_raster_comparison":{"status":"NOT_RUN"},"python":sys.version,"platform":platform.platform(),
            "versions":{p:version(p) for p in ("numpy","pandas","geopandas","pyarrow","rasterio","matplotlib","cmocean","pypdf","dask","distributed")}}
        try:
            train,pred,predictors,ranges,identity,upstream = engine.load_inputs(cfg)
            report.update(model_predictors=predictors,source_identity=identity,
                source_rows={"training":len(train),"prediction":len(pred)},
                upstream_model_contract=upstream["contract_status"],upstream_model_r_comparison=upstream.get("r_comparison",{}),
                upstream_data_prep_coverage=upstream.get("data_prep_coverage","Not specified"))
            expected = module._export_contract(cfg,identity,predictors)
            if validate_only:
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "COMPLETED" or manifest.get("contract") != expected:
                    raise ValueError("Export does not match current model inputs/settings, or did not complete")
                record = manifest["exports"]
            else:
                record = engine.run(overwrite=overwrite,execution=execution,pair=cfg["pair"])[0]
                manifest = json.loads(manifest_path.read_text())
                if manifest.get("status") != "COMPLETED" or manifest.get("contract") != expected:
                    raise ValueError("Model inputs/settings changed between preflight and completed export")
            if not record.get("output_sha256"): raise ValueError("Export manifest has no output fingerprints")
            for path,digest in record["output_sha256"].items():
                if module._file_hash(path) != digest: raise ValueError(f"Exported output changed after rendering: {path}")
            checks=[];comparisons=[];reference_requests=[]
            for role,result in record["results"].items():
                source = train if role == "training" else pred
                for column,path in result["outputs"].items():
                    checks.append(verify_raster(path,source,column,cfg["template_raster"],engine.geometry,cfg["crs"],
                        cfg["raster_config"]["nodata"],cfg["pixel_check_rtol"],cfg["pixel_check_atol"]))
                    directory=cfg["r_reference_dirs"].get(role)
                    if directory:reference_requests.append((Path(directory)/Path(path).name,Path(path)))
            visual_checks=[]
            for path in record["visual_outputs"]:
                if Path(path).suffix == ".pdf":
                    reader=PdfReader(path,strict=True)
                    if len(reader.pages) != 1:raise AssertionError("Expected one comparison page per PDF")
                    if cfg["pair"] not in (reader.pages[0].extract_text() or ""):
                        raise AssertionError("PDF title does not show the configured pair")
                elif Path(path).suffix == ".png":
                    with Image.open(path) as img:img.verify()
                if Path(path).stat().st_size == 0:raise AssertionError("Empty visual output")
                visual_checks.append({"file":str(path),"status":"PASS"})
            if engine.load_inputs(cfg)[4] != identity:raise ValueError("Source inputs changed during verification")
            report.update(contract_status="PASS",raster_checks=checks,visual_checks=visual_checks,exports=record)
            for reference,path in reference_requests:
                if reference.resolve() == path.resolve():raise ValueError("R reference must be separate from the Python output")
                if not reference.exists():comparisons.append({"file":str(reference),"status":"NOT_AVAILABLE"})
                else:
                    try:comparisons.append(compare_r_reference(path,reference,cfg["r_comparison_atol"],cfg["r_comparison_rtol"]))
                    except Exception as exc:
                        report["r_raster_comparison"]={"status":"FAIL","reference":str(reference),"error":str(exc),"files":comparisons}
                        raise
            report["r_raster_comparison"]={"status":"PASS" if comparisons and all(r["status"]=="PASS" for r in comparisons)
                else "PARTIAL" if any(r["status"]=="PASS" for r in comparisons) else "NOT_RUN","files":comparisons}
            report["status"]="PASS"
        except Exception as exc:
            report.update(status="FAIL",error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc())
            if report["contract_status"] != "PASS":report["contract_status"]="FAIL"
            raise
        finally:
            report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"PASS: raster/PDF export checks. Report: {report_path}")
        print("R raster comparison:",report["r_raster_comparison"]["status"])
        reports.append(report)
    return reports


def write_spatial(path,frame):
    """Write synthetic point tables preserving null geometry for invalid XY."""
    path.parent.mkdir(parents=True,exist_ok=True)
    finite=np.isfinite(frame.X)&np.isfinite(frame.Y)
    geom=gpd.points_from_xy(frame.X.fillna(0),frame.Y.fillna(0),crs=32617);geom[~finite]=None
    gpd.GeoDataFrame(frame.copy(),geometry=geom).to_parquet(path,index=False)


def make_fixture(module,root,pair="2004_2006",optional=True):
    """Build explicitly synthetic model-like summaries and a hash-consistent model handoff.

    These are known raster values, not fitted XGBoost or R outputs. Prepared files
    and model/PASS metadata serve only to exercise the export-stage handoff checks.
    """
    root=Path(root);tile="SYNTHETIC_TILE";years=[int(y) for y in pair.split("_")];interval=float(years[1]-years[0])
    assets=root/"inputs/local_test_inputs";assets.mkdir(parents=True,exist_ok=True)
    template=assets/"synthetic_template.tif";grid=assets/"synthetic_grid.geoparquet"
    transform=from_origin(500000,4500000,8,8)
    with rasterio.open(template,"w",driver="GTiff",width=40,height=32,count=1,dtype="float32",crs=32617,transform=transform,nodata=-9999) as dst:
        dst.write(np.ones((32,40),dtype="float32"),1)
    polygon=box(500000+8*2.2,4500000-8*29.2,500000+8*37.2,4500000-8*2.2)
    gpd.GeoDataFrame({"tile_id":[tile]},geometry=[polygon],crs=32617).to_parquet(grid,index=False)
    rows,cols=np.meshgrid(np.arange(3,29),np.arange(3,37),indexing="ij")
    x,y=rasterio.transform.xy(transform,rows.ravel(),cols.ravel());n=len(x)
    bathy=-8-3*np.sin(cols.ravel()/6)-rows.ravel()/12
    change=.5*np.sin(rows.ravel()/4)*np.cos(cols.ravel()/8)
    dynamic=f"tsm_mean_{pair}"
    train=pd.DataFrame({"FID":np.arange(n),"X":x,"Y":y,"tile_id":tile,"pair_id":pair,
        "bathy_t":bathy,"bathy_t1":bathy+change,"delta_bathy":change,"delta_rate":change/interval,
        "interval_years":interval,"sample_weight":1.,"mean_predicted_bathy_t1":bathy+change*.9,
        "mean_predicted_change":change*.9,"mean_predicted_rate":change*.9/interval,
        "uncertainty_sd_bathy_t1":.05+np.abs(change)*.2,
        "distance_from_shore":cols.ravel()*100.,dynamic:rows.ravel()/32.})
    train.loc[0,dynamic]=np.nan
    duplicate=train.iloc[[20]].copy();duplicate["FID"]=n;duplicate["bathy_t1"]+=.1
    duplicate["delta_bathy"]+=.1;duplicate["delta_rate"]=duplicate.delta_bathy/interval
    train=pd.concat([train,duplicate],ignore_index=True)
    pred=train.iloc[:n].drop(columns=["bathy_t1","delta_bathy","delta_rate","sample_weight"]).copy()
    pred.loc[0,"distance_from_shore"]=10000.
    pred.loc[6,"bathy_t"]=np.nan;pred.loc[8,"X"]=np.nan
    pred["prediction_row_usable"]=np.isfinite(pred.X)&np.isfinite(pred.Y)&np.isfinite(pred.bathy_t)
    pred.loc[~pred.prediction_row_usable,["mean_predicted_bathy_t1","mean_predicted_change","mean_predicted_rate","uncertainty_sd_bathy_t1"]]=np.nan
    if optional:
        pred["UC_t"]=.2+np.abs(change)*.3
        pred["survey_end_date"]=np.where(rows.ravel()<15,2024.5,2017.2)
    overrides={"tile_id":tile,"pair":pair,"template_raster":str(template),"tile_grid":str(grid),
        "raster_processes":False,"raster_memory_limit":"4GB","raster_config":{"blocksize":128,"overview_levels":[2,4,8]}}
    engine=module.MLRasterExportEngine({"repository_root":str(root),"raster_settings":overrides})
    cfg=engine.configurations()[0];train_dir=Path(cfg["training_output_dir"]);pred_dir=Path(cfg["prediction_output_dir"])
    schema=["bathy_t",dynamic,"distance_from_shore"]
    for role,df,folder in (("training",train,train_dir),("prediction",pred,pred_dir)):
        write_spatial(folder/f"LOCAL_{role}_summary_{pair}.geoparquet",df)
        df.to_parquet(folder/f"LOCAL_{role}_summary_{pair}.parquet",index=False)
        derived=[c for c in df if c.startswith("mean_predicted_") or c.startswith("uncertainty_sd_")]
        write_spatial(folder/f"ML_ready_{role}_{pair}.geoparquet",df.drop(columns=derived))
    pd.DataFrame({"predictor":schema}).to_parquet(cfg["locked_predictors"],index=False)
    prep=train_dir/f"Data_Prep_Engine_Unit_Test_report_{pair}.json"
    prep.write_text(json.dumps({"status":"PASS","synthetic_fixture":True}))
    prepared={"training_geoparquet":train_dir/f"ML_ready_training_{pair}.geoparquet",
        "prediction_geoparquet":pred_dir/f"ML_ready_prediction_{pair}.geoparquet",
        "locked_predictors_parquet":Path(cfg["locked_predictors"]),"data_prep_report":prep}
    identity={key:module._file_hash(path) for key,path in prepared.items()}
    mc={"n_boot":5,"standard_horizon_years":2.}
    meta={"tile_id":tile,"pair":pair,"crs":"EPSG:32617","predictors":schema,"target":"bathy_t1",
        "prediction_baseline_source":"bathy_t","n_boot":5,"best_iteration":10,"synthetic_fixture":True}
    Path(cfg["model_metadata"]).write_text(json.dumps(meta))
    report=dict(meta,status="PASS",contract_status="PASS",schema=schema,input_sha256=identity,model_config=mc,
        scope="Synthetic model-like summaries; no model fitted",data_prep_coverage="Synthetic fixture",
        validation={"n_boot":5,"best_nrounds":10,"summaries":{"training":{"rows":len(train),"usable":len(train)},
            "prediction":{"rows":len(pred),"usable":int(pred.prediction_row_usable.sum())}}},
        r_comparison={"training":{"status":"NOT_RUN"},"prediction":{"status":"NOT_RUN"}})
    Path(cfg["model_report"]).write_text(json.dumps(report))
    Path(cfg["model_manifest"]).write_text(json.dumps({"model_execution_status":"COMPLETED",
        "contract":{"identity":identity,"schema":schema,"model_config":mc}}))
    pd.DataFrame({"predictor":schema*2,"mean_abs_shap":[1.5,.3,.2,1.6,.4,.1],
        "bootstrap_iteration":[1]*3+[2]*3}).to_parquet(train_dir/f"LOCAL_shap_summary_{pair}.parquet",index=False)
    pd.DataFrame({"predictor":schema,"gain_fraction":[.7,.2,.1]}).to_parquet(train_dir/f"LOCAL_importance_{pair}.parquet",index=False)
    return overrides


def small_template(folder):
    """Return a six-by-six grid for hand-calculated alignment and polygon regression checks."""
    path=Path(folder)/"small.tif"
    with rasterio.open(path,"w",driver="GTiff",width=6,height=6,count=1,dtype="float32",crs=32617,transform=from_origin(0,60,10,10)) as dst:
        dst.write(np.zeros((6,6),dtype="float32"),1)
    return path


def run_self_tests(module):
    """Exercise edge cases and actual local Dask without the user's scientific files."""
    class RasterRegressionTests(unittest.TestCase):
        """Keep numerical rasterization, audit routing, optional plots and source immutability testable."""
        def setUp(self):
            """Give each test a private repository-layout fixture tree."""
            self.temp=tempfile.TemporaryDirectory(prefix="hhm_repository_raster_test_")
            self.root=Path(self.temp.name)

        def tearDown(self):
            """Remove only private synthetic files after each regression case."""
            self.temp.cleanup()

        def test_outward_snap_means_and_layer_na(self):
            """Check known duplicate means, complete snapped bounds and per-layer missingness."""
            df=pd.DataFrame({"X":[15,15,25,35],"Y":[45,45,35,25],"v":[2,4,np.nan,8],"unrelated":[np.nan]*4})
            array,profile=module._dataframe_to_grid(df,"v",small_template(self.root),False,box(12,12,48,48).wkb,"EPSG:32617")
            self.assertEqual(array.shape,(4,4));self.assertEqual(profile["transform"],from_origin(10,50,10,10))
            self.assertEqual(array[0,0],3);self.assertEqual(array[2,2],8);self.assertEqual(array[1,1],-9999)

        def test_polygon_holes_empty_layer_and_guards(self):
            """Retain holes, represent an empty helper layer and reject incompatible geometry/template."""
            polygon=Polygon([(0,0),(60,0),(60,60),(0,60)],holes=[[(10,30),(30,30),(30,50),(10,50)]])
            df=pd.DataFrame({"X":[15,35],"Y":[45,25],"v":[1,2]});template=small_template(self.root)
            array,_=module._dataframe_to_grid(df,"v",template,False,polygon.wkb,"EPSG:32617")
            self.assertEqual(array[1,1],-9999);self.assertEqual(array[3,3],2)
            df.v=np.nan
            array,_=module._dataframe_to_grid(df,"v",template,False,polygon.wkb,"EPSG:32617")
            self.assertTrue(np.all(array==-9999))
            with self.assertRaises(ValueError):module._dataframe_to_grid(df,"v",template,False,polygon.wkb,"EPSG:4326")
            with self.assertRaises(ValueError):module._dataframe_to_grid(df,"v",template,False,box(-1,0,60,60).wkb,"EPSG:32617")

        def test_survey_categories_and_extrapolation(self):
            """Check legacy/ISO dates, category tie rules and finite training-range flags."""
            train=pd.DataFrame({"p":[1,3,np.nan],"all_na":[np.nan]*3});pred=pd.DataFrame({"p":[0,2,4,np.nan],"all_na":[5]*4})
            out,_=module.add_extrapolation_flags(train,pred,["p","all_na"])
            self.assertEqual(out.predictor_extrapolation_count.tolist(),[1,0,1,0]);self.assertNotIn("predictor_extrapolation_count",pred)
            np.testing.assert_allclose(module._survey_codes(pd.Series(["2024.5","2017","1224","2023-01-02","bad"]),True),[1,6,1,2,np.nan],equal_nan=True)
            np.testing.assert_allclose(module._survey_codes(pd.Series(["2024-03-01",None,"unknown","1224.1"]),True),[1,np.nan,np.nan,1],equal_nan=True)
            df=pd.DataFrame({"X":[15,15,15,15],"Y":[45]*4,"survey_year_bin_code":[1,6,6,1]})
            array,_=module._dataframe_to_grid(df,"survey_year_bin_code",small_template(self.root),False,box(0,0,60,60).wkb,"EPSG:32617")
            self.assertEqual(array[1,1],1)

        def test_dask_export_validate_only_and_output_tampering(self):
            """Export through actual Dask, verify 14 layers/three PDFs and detect modified outputs."""
            overrides=make_fixture(module,self.root)
            reports=run_test(module,self.root,overrides,execution="dask")
            self.assertEqual(reports[0]["status"],"PASS");self.assertEqual(len(reports[0]["raster_checks"]),14)
            self.assertEqual(sum(Path(v["file"]).suffix==".pdf" for v in reports[0]["visual_checks"]),3)
            self.assertEqual(reports[0]["r_raster_comparison"]["status"],"NOT_RUN")
            run_test(module,self.root,overrides,validate_only=True)
            with self.assertRaises(FileExistsError):run_test(module,self.root,overrides,execution="inline")
            path=Path(reports[0]["exports"]["results"]["training"]["outputs"]["bathy_t"])
            content=path.read_bytes();path.write_bytes(content+b"changed")
            with self.assertRaisesRegex(ValueError,"changed after rendering"):run_test(module,self.root,overrides,validate_only=True)

        def test_missing_optional_and_stale_model_handoff(self):
            """Show optional no-data panels and reject DataPrep changes after the model PASS."""
            overrides=make_fixture(module,self.root,optional=False)
            report=run_test(module,self.root,overrides,execution="inline")[0]
            skipped=report["exports"]["results"]["prediction"]["skipped"]
            self.assertEqual({v["column"] for v in skipped},{"UC_t","survey_year_bin_code"})
            self.assertEqual(len(report["raster_checks"]),12)
            engine=module.MLRasterExportEngine({"repository_root":str(self.root),"raster_settings":overrides})
            cfg=engine.configurations()[0]
            prepared=Path(cfg["training_output_dir"])/"ML_ready_training_2004_2006.geoparquet"
            frame=gpd.read_parquet(prepared);frame.loc[0,"bathy_t"]+=.1;frame.to_parquet(prepared,index=False)
            with self.assertRaisesRegex(ValueError,"DataPrep changed"):run_test(module,self.root,overrides,validate_only=True)

        def test_alternate_pair_and_optional_r_comparator(self):
            """Update all pair labels dynamically and compare only explicit separate reference rasters."""
            overrides=make_fixture(module,self.root,pair="2006_2010")
            report=run_test(module,self.root,overrides,execution="inline")[0]
            self.assertTrue(all("2006_2010" in p for p in report["exports"]["visual_outputs"]))
            raster=Path(report["exports"]["results"]["training"]["outputs"]["bathy_t"])
            reference_dir=self.root/"synthetic_raster_reference";reference_dir.mkdir()
            reference=reference_dir/raster.name;shutil.copyfile(raster,reference)
            overrides["r_reference_dirs"]={"training":str(reference_dir)}
            check=run_test(module,self.root,overrides,validate_only=True)[0]
            self.assertEqual(check["r_raster_comparison"]["status"],"PARTIAL")
            with rasterio.open(reference) as src:array=src.read(1);profile=src.profile
            array[array!=-9999]+=1
            with rasterio.open(reference,"w",**profile) as dst:dst.write(array,1)
            with self.assertRaises(AssertionError):run_test(module,self.root,overrides,validate_only=True)
            report_file=Path(report["exports"]["results"]["training"]["outputs"]["bathy_t"]).parent/"ML_Raster_Export_Engine_DV_test_report_2006_2010.json"
            failed=json.loads(report_file.read_text())
            self.assertEqual(failed["contract_status"],"PASS");self.assertEqual(failed["r_raster_comparison"]["status"],"FAIL")

    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RasterRegressionTests))
    if not result.wasSuccessful():raise SystemExit(1)


def main():
    """Provide a Windows-safe repository entry point for export, replay and synthetic tests."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=None)
    parser.add_argument("--overwrite",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    parser.add_argument("--execution",choices=("dask","inline"),default="dask")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args();root=args.repo_root.resolve() if args.repo_root else find_repository_root(Path(__file__))
    module=load_engine(root,self_test=args.self_test)
    if args.self_test:run_self_tests(module)
    else:run_test(module,root,overwrite=args.overwrite,execution=args.execution,validate_only=args.validate_only)


if __name__ == "__main__":
    # Required for local Dask worker spawning on Windows.
    main()
