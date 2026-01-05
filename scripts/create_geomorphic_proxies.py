
# Creates geomorphic flow-proxy predictors in raster format, using input raster bathymetry.
# This has to be performed at the raster level, instad of the dataframe level, to ensure spatial reference to other data points.



import os
import re
import tempfile
import rasterio
import numpy as np
from rasterio.plot import reshape_as_image
from rasterio.enums import Resampling
from whitebox.whitebox_tools import WhiteboxTools

wbt = WhiteboxTools()

# --- Directories -----------------------------------------------------------

bathy_dir = r"N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/prediction/processed"

# Find bathy files (modify patterns as needed)
bathy_files = [
    os.path.join(bathy_dir, f)
    for f in os.listdir(bathy_dir)
    if re.match(r"^bt\.bathy\.tif$", f)
]

# --- Utility ----------------------------------------------------------------

def write_array_as_tif(template_path, out_path, arr):
    """Write a numpy array to GeoTIFF using template raster for geotransform."""
    with rasterio.open(template_path) as src:
        meta = src.meta.copy()
        meta.update(dtype="float32", count=1)

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(arr.astype("float32"), 1)

# --- Terrain metrics using whitebox or numpy ---------------------------------

def compute_aspect_slope(bathy_path):
    """Compute slope (deg/rad) and aspect via Whitebox."""
    base = os.path.dirname(bathy_path)
    name = os.path.basename(bathy_path)

    tmp = tempfile.mktemp(suffix=".tif")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)

    # Whitebox requires writing DEM separately
    wbt.copy_raster(bathy_path, tmp)

    aspect = tempfile.mktemp(suffix=".tif")
    slope_deg = tempfile.mktemp(suffix=".tif")
    slope_rad = tempfile.mktemp(suffix=".tif")

    wbt.aspect(tmp, aspect, units="degrees")
    wbt.slope(tmp, slope_rad, units="radians")
    wbt.slope(tmp, slope_deg, units="degrees")

    return aspect, slope_deg, slope_rad


def compute_curvature(dem_path):
    prof = tempfile.mktemp(suffix=".tif")
    plan = tempfile.mktemp(suffix=".tif")
    total = tempfile.mktemp(suffix=".tif")

    wbt.profile_curvature(dem_path, prof)
    wbt.plan_curvature(dem_path, plan)
    wbt.total_curvature(dem_path, total)

    return prof, plan, total


def compute_tci(dem_path):
    """TCI or fallback local convergence."""
    out_tmp = tempfile.mktemp(suffix=".tif")

    try:
        wbt.convergence_index(dem_path, out_tmp)
        return out_tmp
    except:
        # fallback: mean(neighbors) - cell
        with rasterio.open(dem_path) as src:
            arr = src.read(1)
            kernel = np.ones((3, 3), dtype=float)
            kernel[1, 1] = 0

            # local 8-neighbor mean
            from scipy.ndimage import convolve
            neigh_mean = convolve(arr, kernel / 8, mode="nearest")

            tci = neigh_mean - arr

        write_array_as_tif(dem_path, out_tmp, tci)
        return out_tmp


def compute_flow_accumulation(dem_path):
    inv = tempfile.mktemp(suffix=".tif")
    acc = tempfile.mktemp(suffix=".tif")

    # invert DEM: marine flow accumulation flows "down" toward deeper values
    with rasterio.open(dem_path) as src:
        arr = src.read(1)
    write_array_as_tif(dem_path, inv, -arr)

    wbt.d8_flow_accumulation(inv, acc, out_type="cells")
    return acc

# --- MAIN PROCESSING --------------------------------------------------------

def process_bathy_raster(bathy_path):
    filename = os.path.basename(bathy_path)

    # Extract year (if filename contains bathy_YYYY_filled.tif)
    match = re.search(r"(?<=bathy_)\d{4}(?=_filled)", filename)
    year = match.group(0) if match else "unknown"

    print(f"Processing bathy year: {year} — {bathy_path}")

    # Aspect, Slope
    aspect, slope_deg, slope_rad = compute_aspect_slope(bathy_path)
    wbt.copy_raster(aspect, os.path.join(bathy_dir, f"flowdir_{year}.tif"))
    wbt.copy_raster(slope_rad, os.path.join(bathy_dir, f"gradmag_{year}.tif"))
    wbt.copy_raster(slope_deg, os.path.join(bathy_dir, f"slope_deg_{year}.tif"))

    # Curvature
    prof, plan, total = compute_curvature(bathy_path)
    wbt.copy_raster(prof,  os.path.join(bathy_dir, f"curv_profile_{year}.tif"))
    wbt.copy_raster(plan,  os.path.join(bathy_dir, f"curv_plan_{year}.tif"))
    wbt.copy_raster(total, os.path.join(bathy_dir, f"curv_total_{year}.tif"))

    # TCI
    tci = compute_tci(bathy_path)
    wbt.copy_raster(tci, os.path.join(bathy_dir, f"tci_{year}.tif"))

    # Flow Accumulation
    flowacc = compute_flow_accumulation(bathy_path)
    wbt.copy_raster(flowacc, os.path.join(bathy_dir, f"flowacc_{year}.tif"))

    # Shear proxy = slope_deg × |plan curvature|
    with rasterio.open(slope_deg) as s, rasterio.open(plan) as p:
        slope_arr = s.read(1)
        plan_arr  = p.read(1)

    shear = slope_arr * np.abs(plan_arr)
    write_array_as_tif(bathy_path,
                       os.path.join(bathy_dir, f"shearproxy_{year}.tif"),
                       shear)

    print(f"Completed year {year}")


# Run on all input bathy files
for f in bathy_files:
    process_bathy_raster(f)
