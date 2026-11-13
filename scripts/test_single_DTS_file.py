import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
import geopandas as gpd

HYDRO_HEALTH = pathlib.Path(__file__).parents[1] / 'src'

print(HYDRO_HEALTH)
import sys
sys.path.append(str(HYDRO_HEALTH))

from hydro_health.engines.DistanceToShoreEngine import (
    DistanceToShoreEngine,
    _process_tile,
    INPUTS,
    OUTPUTS,
)
from hydro_health.helpers.tools import get_config_item, Param


def get_rss_gb() -> tuple[float, float]:
    """Returns (current_rss_gb, peak_rss_gb) directly from Linux /proc/self/status."""
    rss_kb = hwm_kb = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss_kb = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kb = int(line.split()[1])
    return rss_kb / (1024**2), hwm_kb / (1024**2)


def test_single_water_tile(test_tif_path: str, simulate_threads: int = 1) -> None:
    ecoregion = "ER_3"
    data_type = "DigitalCoast"
    param_lookup = {
        "env": Param("aws"),  # Change to 'local' if testing on local files
        "output_directory": Param(OUTPUTS),
        "eco_regions": Param([ecoregion]),
    }

    engine = DistanceToShoreEngine(param_lookup, data_type=data_type)
    scratch_cache = pathlib.Path.home() / "local_tmp_dir" / "support_cache"
    scratch_cache.mkdir(parents=True, exist_ok=True)

    # 1. Prepare X-sorted .npy and FlatGeobuf boundary once (same as run())
    local_npy_path = str(
        engine.ensure_local_shoreline_npy(
            ecoregion=ecoregion,
            crs_name="EPSG_32617",
            cache_dir=scratch_cache,
        )
    )

    ecoregion_gdf_path = scratch_cache / f"{ecoregion}_{data_type}_boundary.fgb"
    if not ecoregion_gdf_path.exists():
        master_gpkg = INPUTS / get_config_item("SHARED", "MASTER_GRIDS")
        gdf = gpd.read_file(str(master_gpkg), layer="Enhanced_EcoRegions_50m")
        gdf = gdf[gdf["EcoRegion"] == ecoregion].to_crs("EPSG:32617")
        gdf.to_file(ecoregion_gdf_path, driver="FlatGeobuf")
        del gdf

    temp_path = pathlib.Path.home() / "local_tmp_dir" / "test_run"
    task_args = [
        test_tif_path,
        data_type,
        ecoregion,
        str(ecoregion_gdf_path),
        str(temp_path),
        param_lookup,
        "test_prefix",  # Isolates any S3 test upload from production outputs
        local_npy_path,
    ]

    rss, peak = get_rss_gb()
    print(f"[Before] Current RSS: {rss:.2f} GiB | Peak RSS: {peak:.2f} GiB")

    t0 = time.perf_counter()
    if simulate_threads == 1:
        results = [_process_tile(task_args)]
    else:
        # Simulates N Dask worker threads hitting water tiles at the exact same time
        with ThreadPoolExecutor(max_workers=simulate_threads) as pool:
            results = list(pool.map(_process_tile, [task_args] * simulate_threads))

    elapsed = time.perf_counter() - t0
    rss, peak = get_rss_gb()
    print(f"[After {elapsed:.2f}s] Current RSS: {rss:.2f} GiB | Peak RSS: {peak:.2f} GiB")
    print("Results:", results)


if __name__ == "__main__":
    CONFIRMED_WATER_TILE = (
        "/vsis3/ocs-dev-csdl-hydrohealth/ER_3/model_variables/Prediction/raw/"
        "DigitalCoast/2016_DEM_NOAA_NGS_53125/dem/"
        "NGS_FL_Topobathy_DEM_sugarloaf_2016_8583/"
        "2017_441500e_2720000n_dem.tif"
    )
    test_single_water_tile(CONFIRMED_WATER_TILE, simulate_threads=2)