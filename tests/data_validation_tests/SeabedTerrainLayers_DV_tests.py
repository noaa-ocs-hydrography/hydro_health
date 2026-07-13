"""
S3 Output Validation Test Suite for Seabed Terrain Engine.

Instructions for running:
1. Install dependencies: pip install pytest rasterio s3fs universal-pathlib boto3
2. Set your AWS credentials in your environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
3. Set the target S3 path as an environment variable:
   export TEST_S3_BTM_PATH="s3://your-bucket-name/path/to/BTM_outputs"
4. Run the tests:
   pytest test_s3_outputs.py -v
"""

import os
import random
import pytest
import rasterio
from upath import UPath
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
# Default to an empty string to force the user to set it, or fallback for local dev
S3_BTM_PATH = os.environ.get("TEST_S3_BTM_PATH", "s3://my-hydro-bucket/BTM_outputs")
S3_DICT_PATH = os.environ.get("TEST_S3_DICT_PATH", "s3://my-hydro-bucket/dictionaries")

# The exact 13 derivatives created by the CreateSeabedTerrainLayerEngine
EXPECTED_DERIVATIVE_SUFFIXES = [
    "_slope_deg.tif",
    "_gradmag.tif",
    "_flowdir.tif",
    "_curv_profile.tif",
    "_curv_plan.tif",
    "_curv_total.tif",
    "_flowacc.tif",
    "_tci.tif",
    "_shearproxy.tif",
    "_rugosity_tri.tif",
    "_bpi_fine_std.tif",
    "_bpi_broad_std.tif",
    "_terrain_classification.tif"
]

TARGET_CRS = "EPSG:32617"
TARGET_RES = 8.0

# Thread count for parallelizing S3 metadata requests
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 16))


# --- FIXTURES ---

@pytest.fixture(scope="session")
def s3_all_files():
    """Fetches all file paths in the output directory once for the entire test session."""
    print(f"\nScanning S3 Path: {S3_BTM_PATH} ... This might take a moment.")
    base_path = UPath(S3_BTM_PATH)
    
    # Check if path exists to prevent immediate failure on empty environments
    if not base_path.exists():
        pytest.skip(f"S3 Path does not exist or is inaccessible: {S3_BTM_PATH}")
        
    all_files = list(base_path.rglob("*.tif"))
    return all_files


@pytest.fixture(scope="session")
def base_file_groups(s3_all_files):
    """Groups derivative files back to their original base bathy file name."""
    groups = defaultdict(list)
    
    for f in s3_all_files:
        filename = f.name
        # Find which suffix this file has
        matched_suffix = None
        for suffix in EXPECTED_DERIVATIVE_SUFFIXES:
            if filename.endswith(suffix):
                matched_suffix = suffix
                break
        
        if matched_suffix:
            # Strip the suffix to get the base name (e.g., 'combined2_bathy_B1234567_2015')
            base_name = filename.replace(matched_suffix, "")
            groups[base_name].append(matched_suffix)
            
    return groups


# --- TESTS ---

def test_files_exist_in_bucket(s3_all_files):
    """Ensure that the engine actually wrote files to the bucket."""
    assert len(s3_all_files) > 0, f"No .tif files found in {S3_BTM_PATH}!"


def test_derivative_counts(base_file_groups):
    """
    For every unique base bathymetry file processed, ensure exactly 
    13 derivative files were created and output.
    """
    assert len(base_file_groups) > 0, "No valid derivative groups found to test."
    
    failed_bases = []
    for base_name, found_suffixes in base_file_groups.items():
        missing_suffixes = set(EXPECTED_DERIVATIVE_SUFFIXES) - set(found_suffixes)
        if missing_suffixes:
            failed_bases.append(f"{base_name} is missing: {missing_suffixes}")
            
    assert not failed_bases, f"Some base files did not generate all 13 derivatives:\n" + "\n".join(failed_bases)


def test_total_derivative_count(s3_all_files, base_file_groups):
    """
    Calculates the theoretical total number of derivative files that should exist
    based on the number of unique base bathymetry files, and verifies the 
    bucket contains exactly that number.
    """
    num_base_files = len(base_file_groups)
    num_suffixes = len(EXPECTED_DERIVATIVE_SUFFIXES)
    
    expected_total_derivatives = num_base_files * num_suffixes
    
    # Count actual derivatives found in the bucket that match our known suffixes
    actual_derivatives = [
        f for f in s3_all_files 
        if any(f.name.endswith(suffix) for suffix in EXPECTED_DERIVATIVE_SUFFIXES)
    ]
    
    assert len(actual_derivatives) == expected_total_derivatives, (
        f"Expected {expected_total_derivatives} total derivatives "
        f"({num_base_files} base files * {num_suffixes} suffixes), "
        f"but found {len(actual_derivatives)}."
    )


def test_no_rogue_files(s3_all_files):
    """
    Ensures that ONLY valid derivatives exist in the output directory.
    Catches stray temporary files, .aux.xml, or accidentally exported intermediate steps.
    """
    rogue_files = []
    for f in s3_all_files:
        if not any(f.name.endswith(suffix) for suffix in EXPECTED_DERIVATIVE_SUFFIXES):
            rogue_files.append(f.name)
            
    assert not rogue_files, f"Found {len(rogue_files)} unexpected rogue files in the bucket:\n" + "\n".join(rogue_files)


def _check_raster_metadata(file_path):
    """Helper function to run all metadata checks on a single file."""
    errors = []
    try:
        # 1. Size check (do it here to save an S3 call)
        size = file_path.stat().st_size
        if size <= 1000:
            errors.append(f"Suspiciously small file size ({size} bytes)")

        with rasterio.open(str(file_path)) as src:
            # 2. CRS Check
            if src.crs is None:
                errors.append("No CRS defined")
            elif src.crs.to_string() != TARGET_CRS:
                errors.append(f"Wrong CRS: {src.crs.to_string()} (expected {TARGET_CRS})")
            
            # 3. Resolution Check
            res_x, res_y = src.res
            if round(res_x, 2) != TARGET_RES or round(res_y, 2) != TARGET_RES:
                errors.append(f"Wrong resolution: ({res_x}, {res_y}) expected {TARGET_RES}")
            
            # 4. Bounds Check (UTM 17N US EEZ Approximation)
            bounds = src.bounds
            if bounds.left <= 100000 or bounds.right >= 900000:
                errors.append(f"X bounds out of expected UTM 17N range: ({bounds.left}, {bounds.right})")
            if bounds.bottom <= 2000000 or bounds.top >= 6000000:
                errors.append(f"Y bounds out of expected UTM 17N range: ({bounds.bottom}, {bounds.top})")
                
            # 5. Type and Band Check
            if src.count < 1:
                errors.append("No raster bands found")
            else:
                # 6. Dtype check (Ensuring we aren't wasting space with float64)
                band_dtype = src.dtypes[0]
                if band_dtype not in ['float32', 'int32', 'int16', 'uint8']:
                    errors.append(f"Unexpected data type: {band_dtype} (Expected float32 or smaller)")
                
            # 7. NoData Value Check
            if src.nodata is None:
                errors.append("No 'nodata' value defined in raster header")
                
            # 8. S3 Optimization Check (Tiling and Compression)
            # LZW is specifically used in your Engine class
            if src.profile.get('compress', '').lower() != 'lzw':
                errors.append(f"Raster is not LZW compressed (Current: {src.profile.get('compress')})")
            
            # Warn/Fail if not internally tiled (essential for S3 cloud-native reading)
            if not src.profile.get('tiled') and not src.is_tiled:
                errors.append("Raster is not internally tiled (Not optimized for cloud/S3 reads)")
                
    except Exception as e:
        errors.append(f"Failed to open or read file: {e}")
        
    if errors:
        return f"{file_path.name}:\n  - " + "\n  - ".join(errors)
    return None


def test_all_raster_metadata_and_sizes(s3_all_files):
    """
    Concurrently opens ALL output files to check sizes, CRS, Resolution, and Bounds.
    No sampling - every single file is verified concurrently to speed up network requests.
    """
    assert len(s3_all_files) > 0, "No files to test."
    
    failed_files = []
    
    # Use a ThreadPoolExecutor to parallelize S3 reads
    print(f"\nChecking metadata for all {len(s3_all_files)} files using {MAX_WORKERS} threads...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(_check_raster_metadata, fp): fp for fp in s3_all_files}
        for future in as_completed(future_to_file):
            error_msg = future.result()
            if error_msg:
                failed_files.append(error_msg)
                
    assert not failed_files, f"{len(failed_files)} files failed metadata checks:\n" + "\n".join(failed_files)


def test_dictionaries_created(base_file_groups):
    """Ensure the regional classification dictionaries were output for all processed years."""
    dict_dir = UPath(S3_DICT_PATH)
    if not dict_dir.exists():
        pytest.skip(f"Dictionary directory does not exist: {S3_DICT_PATH}")
        
    csv_files = list(dict_dir.glob("*.csv"))
    assert len(csv_files) > 0, f"No classification dictionaries (*.csv) found in {S3_DICT_PATH}"
    
    csv_names = [f.name for f in csv_files]
    
    # Extract unique years from the base file names
    expected_years = set()
    for base in base_file_groups.keys():
        if "BlueTopo" in base or "bluetopo" in base.lower():
            expected_years.add("bluetopo")
        else:
            # E.g., combined2_bathy_B1234567_2015
            parts = base.split("_")
            for part in parts:
                if part.isdigit() and len(part) == 4:
                    expected_years.add(part)
    
    missing_dicts = []
    for year in expected_years:
        expected_csv = f"dictionary_{year}.csv"
        # Fallback for lowercase/uppercase variations
        if expected_csv not in csv_names and expected_csv.lower() not in [n.lower() for n in csv_names]:
            missing_dicts.append(expected_csv)
            
    assert not missing_dicts, f"Missing classification dictionaries for expected years: {missing_dicts}"
    
    # Spot check that they have some size
    for csv in csv_files:
        assert csv.stat().st_size > 50, f"Dictionary {csv.name} is empty or invalid."