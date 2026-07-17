"""
S3 Output Validation Test Suite for Seabed Terrain Engine.

Instructions for running:
1. Install dependencies: pip install pytest rasterio s3fs universal-pathlib boto3 fpdf
2. Ensure you have access to the 'hydro_health' module in your PYTHONPATH.
3. Set your AWS credentials in your environment (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
4. Run the tests:
   python test_seabed_terrain_outputs.py
"""

import os
import random
import pytest
import rasterio
from upath import UPath
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Import the project's configuration tools
try:
    from hydro_health.helpers.tools import get_config_item
except ImportError:
    raise ImportError("Could not import get_config_item. Make sure 'hydro_health' is in your PYTHONPATH.")


# --- GLOBAL REPORT TRACKER ---
REPORT_DATA = {}
REPORT_FILENAME = "Seabed_Terrain_Validation_Report"

# --- PDF REPORT GENERATOR ---
def generate_pdf_report(data, filename=REPORT_FILENAME):
    """
    Generates a PDF using fpdf if available. 
    Falls back to a Markdown (.md) file if fpdf is not installed.
    Silently overwrites the file each time it's called so progress is saved.
    """
    try:
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Seabed Terrain Validation Test Report", ln=True, align='C')
        pdf.ln(10)
        
        for test_name, errors in data.items():
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, txt=f"Test: {test_name}", ln=True)
            pdf.set_font("Arial", size=10)
            
            if not errors:
                pdf.set_text_color(0, 150, 0) # Green
                pdf.cell(0, 10, txt="Status: PASSED", ln=True)
                pdf.set_text_color(0, 0, 0)
            elif errors and errors[0].startswith("SKIPPED"):
                pdf.set_text_color(200, 120, 0) # Orange
                pdf.cell(0, 10, txt=errors[0], ln=True)
                pdf.set_text_color(0, 0, 0)
            else:
                pdf.set_text_color(200, 0, 0) # Red
                pdf.cell(0, 10, txt=f"Status: FAILED ({len(errors)} errors)", ln=True)
                pdf.set_text_color(0, 0, 0)
                for err in errors:
                    # encode to handle special characters gracefully
                    safe_txt = f"- {err}".encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 8, txt=safe_txt)
            pdf.ln(5)
        
        pdf_path = f"{filename}.pdf"
        pdf.output(pdf_path)
        
    except ImportError:
        # Fallback to Markdown
        md_path = f"{filename}.md"
        with open(md_path, "w") as f:
            f.write("# Seabed Terrain Validation Test Report\n\n")
            for test_name, errors in data.items():
                f.write(f"### Test: {test_name}\n")
                if not errors:
                    f.write("**Status:** PASSED\n\n")
                elif errors and errors[0].startswith("SKIPPED"):
                    f.write(f"**Status:** {errors[0]}\n\n")
                else:
                    f.write(f"**Status:** FAILED ({len(errors)} errors)\n\n")
                    for err in errors:
                        f.write(f"- {err}\n")
                    f.write("\n")

def fail_with_errors(errors, test_name="Test"):
    """Helper to fail tests cleanly and record all full paths for the PDF."""
    REPORT_DATA[test_name] = errors
    generate_pdf_report(REPORT_DATA)  # Write to PDF immediately
    
    if errors:
        error_msg = f"{test_name} found {len(errors)} error(s):\n" + "\n".join(errors[:50])
        if len(errors) > 50:
            error_msg += f"\n...and {len(errors) - 50} more errors omitted from console. See generated PDF report."
        pytest.fail(error_msg)

def skip_test(reason, test_name="Test"):
    """Helper to skip tests and record the skip reason for the PDF."""
    REPORT_DATA[test_name] = [f"SKIPPED: {reason}"]
    generate_pdf_report(REPORT_DATA)  # Write to PDF immediately
    pytest.skip(reason)

def log_error(errors_list, msg):
    """Helper to instantly print out a failure to the console and track it."""
    print(f"\n[FAIL] {msg}")
    errors_list.append(msg)


# --- CONFIGURATION ---
# Dynamically fetch the paths using the original project configuration
bucket = get_config_item('S3', 'BUCKET_NAME').strip('/')
raw_output_dir = get_config_item('TERRAIN', 'OUTPUTS')
clean_out_dir = str(raw_output_dir).replace("s3://", "").strip('/')

if not clean_out_dir.startswith(bucket):
    main_output_dir = f"s3://{bucket}/{clean_out_dir}"
else:
    main_output_dir = f"s3://{clean_out_dir}"

default_btm_path = f"{main_output_dir}/BTM_outputs"
default_dict_path = f"{main_output_dir}/dictionaries"

# We still allow env var overrides for flexible CI/CD testing, but default to the dynamic config
S3_BTM_PATH = os.environ.get("TEST_S3_BTM_PATH", default_btm_path)
S3_DICT_PATH = os.environ.get("TEST_S3_DICT_PATH", default_dict_path)

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
        skip_test(f"S3 Path does not exist or is inaccessible: {S3_BTM_PATH}", "Fixture: s3_all_files")
        
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
    errors = []
    if len(s3_all_files) == 0:
        log_error(errors, f"No .tif files found in {S3_BTM_PATH}!")
        
    fail_with_errors(errors, "Files Exist in Bucket")


def test_derivative_counts(base_file_groups):
    """
    For every unique base bathymetry file processed, ensure exactly 
    13 derivative files were created and output.
    """
    if len(base_file_groups) == 0:
        skip_test("No valid derivative groups found to test.", "Derivative Counts")
        
    errors = []
    for base_name, found_suffixes in base_file_groups.items():
        missing_suffixes = set(EXPECTED_DERIVATIVE_SUFFIXES) - set(found_suffixes)
        if missing_suffixes:
            log_error(errors, f"{base_name} is missing: {missing_suffixes}")
            
    fail_with_errors(errors, "Derivative Counts")


def test_total_derivative_count(s3_all_files, base_file_groups):
    """
    Calculates the theoretical total number of derivative files that should exist
    based on the number of unique base bathymetry files, and verifies the 
    bucket contains exactly that number.
    """
    if len(base_file_groups) == 0:
        skip_test("No valid derivative groups found to test.", "Total Derivative Count")
        
    num_base_files = len(base_file_groups)
    num_suffixes = len(EXPECTED_DERIVATIVE_SUFFIXES)
    
    expected_total_derivatives = num_base_files * num_suffixes
    
    # Count actual derivatives found in the bucket that match our known suffixes
    actual_derivatives = [
        f for f in s3_all_files 
        if any(f.name.endswith(suffix) for suffix in EXPECTED_DERIVATIVE_SUFFIXES)
    ]
    
    errors = []
    if len(actual_derivatives) != expected_total_derivatives:
        log_error(errors, f"Expected {expected_total_derivatives} total derivatives "
                          f"({num_base_files} base files * {num_suffixes} suffixes), "
                          f"but found {len(actual_derivatives)}.")
                          
    fail_with_errors(errors, "Total Derivative Count")


def test_no_rogue_files(s3_all_files):
    """
    Ensures that ONLY valid derivatives exist in the output directory.
    Catches stray temporary files, .aux.xml, or accidentally exported intermediate steps.
    """
    errors = []
    for f in s3_all_files:
        if not any(f.name.endswith(suffix) for suffix in EXPECTED_DERIVATIVE_SUFFIXES):
            log_error(errors, f"Unexpected rogue file found: {f.name}")
            
    fail_with_errors(errors, "No Rogue Files")


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
    if len(s3_all_files) == 0:
        skip_test("No files to test.", "Raster Metadata and Sizes")
    
    errors = []
    total_files = len(s3_all_files)
    
    # Use a ThreadPoolExecutor to parallelize S3 reads
    print(f"\nChecking metadata for all {total_files} files using {MAX_WORKERS} threads...")
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {executor.submit(_check_raster_metadata, fp): fp for fp in s3_all_files}
        for future in as_completed(future_to_file):
            completed += 1
            print(f"\rTesting metadata {completed}/{total_files}...   ", end="", flush=True)
            
            error_msg = future.result()
            if error_msg:
                log_error(errors, error_msg)
                
    print("\nFinished checking metadata.")
    fail_with_errors(errors, "Raster Metadata and Sizes")


def test_dictionaries_created(base_file_groups):
    """Ensure the regional classification dictionaries were output for all processed years."""
    dict_dir = UPath(S3_DICT_PATH)
    if not dict_dir.exists():
        skip_test(f"Dictionary directory does not exist: {S3_DICT_PATH}", "Dictionaries Created")
        
    errors = []
    csv_files = list(dict_dir.glob("*.csv"))
    if len(csv_files) == 0:
        log_error(errors, f"No classification dictionaries (*.csv) found in {S3_DICT_PATH}")
        fail_with_errors(errors, "Dictionaries Created")
        return
    
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
    
    for year in expected_years:
        expected_csv = f"dictionary_{year}.csv"
        # Fallback for lowercase/uppercase variations
        if expected_csv not in csv_names and expected_csv.lower() not in [n.lower() for n in csv_names]:
            log_error(errors, f"Missing classification dictionary for expected year: {expected_csv}")
            
    # Spot check that they have some size
    for csv in csv_files:
        if csv.stat().st_size <= 50:
            log_error(errors, f"Dictionary {csv.name} is empty or invalid (<= 50 bytes).")
            
    fail_with_errors(errors, "Dictionaries Created")


# --- CUSTOM TEST RUNNER ---
# Run via `python test_seabed_terrain_outputs.py` to use this block!

if __name__ == "__main__":
    import sys

    # Comment out any test in this list that you want to skip.
    tests_to_run = [
        "test_files_exist_in_bucket",
        "test_derivative_counts",
        "test_total_derivative_count",
        "test_no_rogue_files",
        "test_all_raster_metadata_and_sizes",
        "test_dictionaries_created"
    ]

    print("--- custom test runner ---")
    
    if not tests_to_run:
        print("No tests selected to run. Uncomment tests in the `tests_to_run` list.")
        sys.exit(0)

    # Build the pytest arguments
    file_path = str(Path(__file__).absolute())
    pytest_args = ["-v", "-s"] + [f"{file_path}::{test}" for test in tests_to_run]
    
    print(f"Executing {len(tests_to_run)} test(s)...\n")
    
    # Pass control to pytest (runs tests in the same process, populating REPORT_DATA)
    exit_code = pytest.main(pytest_args)
    
    print(f"\n[+] Testing finished. Final report generated at {REPORT_FILENAME}.pdf (or .md)")
    
    sys.exit(exit_code)