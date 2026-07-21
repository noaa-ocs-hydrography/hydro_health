"""
Tests to validate the outputs of the ModelDataPreProcessor on S3.
Run this using: python test_s3_outputs.py
"""
import pytest
import s3fs
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
import numpy as np
import warnings
from rasterio.session import AWSSession
from rasterio.features import geometry_mask
from rasterio.warp import transform_bounds
from upath import UPath
from shapely.geometry import box

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

# --- GLOBAL REPORT TRACKER ---
REPORT_DATA = {}
REPORT_S3_DIR = "s3://ocs-dev-csdl-hydrohealth/ER_3"

# --- PDF REPORT GENERATOR ---
def generate_pdf_report(data, filename="S3_Validation_Report"):
    """
    Generates a PDF using fpdf if available. 
    Falls back to a Markdown (.md) file if fpdf is not installed.
    Silently overwrites the file locally and uploads to S3 each time it's called so progress is saved.
    """
    s3 = s3fs.S3FileSystem(anon=False)
    
    try:
        from fpdf import FPDF
        
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "S3 Validation Test Report", ln=True, align='C')
        pdf.ln(10)
        
        for test_name, test_info in data.items():
            errors = test_info.get("errors", [])
            desc = test_info.get("description", "")
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, txt=f"Test: {test_name}", ln=True)
            
            if desc:
                pdf.set_font("Arial", 'I', 10)
                # handle encoding for special characters in description
                safe_desc = desc.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 8, txt=f"Description: {safe_desc}")
                
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
                    # encode to handle special characters (like degrees symbol) gracefully
                    safe_txt = f"- {err}".encode('latin-1', 'replace').decode('latin-1')
                    pdf.multi_cell(0, 8, txt=safe_txt)
            pdf.ln(5)
        
        pdf_path = f"{filename}.pdf"
        pdf.output(pdf_path)
        
        # Upload to S3
        s3.put(pdf_path, f"{REPORT_S3_DIR}/{pdf_path}")
        
    except ImportError:
        # Fallback to Markdown
        md_path = f"{filename}.md"
        with open(md_path, "w") as f:
            f.write("# S3 Validation Test Report\n\n")
            for test_name, test_info in data.items():
                errors = test_info.get("errors", [])
                desc = test_info.get("description", "")
                
                f.write(f"### Test: {test_name}\n")
                if desc:
                    f.write(f"**Description:** {desc}\n\n")
                    
                if not errors:
                    f.write("**Status:** PASSED\n\n")
                elif errors and errors[0].startswith("SKIPPED"):
                    f.write(f"**Status:** {errors[0]}\n\n")
                else:
                    f.write(f"**Status:** FAILED ({len(errors)} errors)\n\n")
                    for err in errors:
                        f.write(f"- {err}\n")
                    f.write("\n")
            
        # Upload to S3
        s3.put(md_path, f"{REPORT_S3_DIR}/{md_path}")

def fail_with_errors(errors, test_name="Test", description=""):
    """Helper to fail tests cleanly and record all full paths for the PDF."""
    REPORT_DATA[test_name] = {"errors": errors, "description": description}
    generate_pdf_report(REPORT_DATA)  # Write to PDF immediately
    
    if errors:
        error_msg = f"{test_name} found {len(errors)} error(s):\n" + "\n".join(errors[:50])
        if len(errors) > 50:
            error_msg += f"\n...and {len(errors) - 50} more errors omitted from console. See generated PDF report."
        pytest.fail(error_msg)

def skip_test(reason, test_name="Test", description=""):
    """Helper to skip tests and record the skip reason for the PDF."""
    REPORT_DATA[test_name] = {"errors": [f"SKIPPED: {reason}"], "description": description}
    generate_pdf_report(REPORT_DATA)  # Write to PDF immediately
    pytest.skip(reason)

def log_error(errors_list, msg):
    """Helper to instantly print out a failure to the console and track it."""
    print(f"\n[FAIL] {msg}")
    errors_list.append(msg)

def is_excluded(filepath, excluded_list):
    """Helper to check if a file path falls under any excluded directories."""
    clean_path = str(filepath).replace("s3://", "")
    return any(clean_path.startswith(excl) for excl in excluded_list)

# --- CONFIGURATION & FIXTURES ---

@pytest.fixture(scope="module")
def s3_fs():
    """Returns an authenticated s3fs file system instance."""
    return s3fs.S3FileSystem(anon=False)

@pytest.fixture(scope="module")
def s3_prefix():
    """Dynamically reads the S3 bucket name from config."""
    bucket = get_config_item('S3', 'BUCKET_NAME', pilot_mode=False)
    return f"s3://{bucket}/"

@pytest.fixture(scope="module")
def engine_instance():
    """Provides an instance of the Engine to read base configuration."""
    return Engine()

@pytest.fixture(scope="module")
def year_pairs(engine_instance):
    """The expected year ranges read dynamically from the Engine."""
    return engine_instance.year_ranges

@pytest.fixture(scope="module")
def target_crs(engine_instance):
    """The target CRS read dynamically from the Engine."""
    return engine_instance.target_crs

@pytest.fixture(scope="module")
def target_res(engine_instance):
    """The target resolution read dynamically from the Engine."""
    return engine_instance.target_res

@pytest.fixture(scope="module")
def raster_directories(s3_prefix):
    """List of dictionaries containing output rasters and their recursive search flag."""
    return [
        {
            "path": str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR', pilot_mode=False)}")),
            "recursive": True
        },
        {
            "path": str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'TRAINING_OUTPUT_DIR', pilot_mode=False)}")),
            "recursive": True
        },
        {
            "path": str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'UNCOMBINED_LIDAR_DIR', pilot_mode=False)}")),
            "recursive": False
        }
    ]

@pytest.fixture(scope="module")
def excluded_prefixes(s3_prefix):
    """Directories to exclude from recursive searches (like intermediate terrain/filled)."""
    excludes = []
    
    try:
        filled = get_config_item('TERRAIN', 'FILLED_DIR', pilot_mode=False)
        excludes.append(str(UPath(f"{s3_prefix}{filled}")).replace("s3://", ""))
    except KeyError:
        pass
        
    return excludes

@pytest.fixture(scope="module")
def prediction_tiles_dir(s3_prefix):
    return str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'PREDICTION_TILES_DIR', pilot_mode=False)}"))

@pytest.fixture(scope="module")
def training_tiles_dir(s3_prefix):
    return str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'TRAINING_TILES_DIR', pilot_mode=False)}"))

@pytest.fixture(scope="module")
def tile_directories(prediction_tiles_dir, training_tiles_dir):
    """List of directories containing output tile parquet files."""
    return [prediction_tiles_dir, training_tiles_dir]


# --- 1. DIRECTORY AND FILE PRESENCE TESTS ---

def test_output_directories_exist(s3_fs, raster_directories, tile_directories):
    """Test that all expected output directories were created on S3."""
    desc = "Verifies that all expected S3 directories for model outputs and parquet tiles were successfully created."
    all_dirs = [d["path"] for d in raster_directories] + tile_directories
    errors = []
    for d in all_dirs:
        try:
            # s3fs.ls() throws FileNotFoundError if the prefix doesn't exist at all
            contents = s3_fs.ls(d)
            if len(contents) == 0:
                log_error(errors, f"Directory exists but is empty: {d}")
        except FileNotFoundError:
            log_error(errors, f"Directory missing: {d}")
        except Exception as e:
            log_error(errors, f"Error checking directory {d}: {str(e)}")
            
    fail_with_errors(errors, "Output Directories", desc)

def test_csv_stats_generated(s3_fs, training_tiles_dir, prediction_tiles_dir):
    """Test that the summary NaN stats CSVs were generated in the parent dirs."""
    desc = "Checks that the summary CSV files containing missing data (NaN) statistics were generated correctly."
    
    # Use UPath to safely navigate S3 paths to get the parent folder
    train_parent = UPath(training_tiles_dir).parent
    pred_parent = UPath(prediction_tiles_dir).parent
    
    expected_csvs = [
        str(train_parent / "year_pair_nan_counts_training.csv"),
        str(pred_parent / "year_pair_nan_counts_prediction.csv")
    ]
    
    errors = []
    for csv_path in expected_csvs:
        if not s3_fs.exists(csv_path):
            log_error(errors, f"Missing stats CSV: {csv_path}")
            
    fail_with_errors(errors, "CSV Stats Generation", desc)


# --- 2. RASTER PROPERTY TESTS ---

def test_raster_crs_and_resolution(s3_fs, raster_directories, target_crs, target_res, excluded_prefixes):
    """
    Test all rasters from each directory to ensure:
    - Target CRS matches Engine configuration
    - Target Resolution matches Engine configuration
    - Output type is Float32
    - Compression is LZW
    """
    desc = "Validates that all generated TIFFs have the correct Coordinate Reference System (CRS), target resolution, float32 datatype, and LZW compression."
    
    # 1. Collect all TIFFs first to get the total count
    all_tifs = []
    for d in raster_directories:
        if d["recursive"]:
            found = s3_fs.find(d["path"])
            all_tifs.extend([f for f in found if f.lower().endswith(('.tif', '.tiff')) and not is_excluded(f, excluded_prefixes)])
        else:
            found = s3_fs.ls(d["path"])
            all_tifs.extend([f for f in found if f.lower().endswith(('.tif', '.tiff')) and not is_excluded(f, excluded_prefixes)])
            
    if not all_tifs:
        skip_test(f"No TIFFs found in any of the provided raster directories: {[d['path'] for d in raster_directories]}.", "Raster CRS and Resolution", desc)
            
    total_tifs = len(all_tifs)
    sampled_tifs = all_tifs[::10]
    print(f"\nFound {total_tifs} rasters. Testing {len(sampled_tifs)} (every 10th file) for CRS and resolution.")
    
    errors = []
    # 2. Iterate with a progress counter
    for i, tif in enumerate(sampled_tifs):
        actual_index = (i * 10) + 1
        print(f"\rTesting raster {actual_index}/{total_tifs}: {Path(tif).name[:30]}...   ", end="", flush=True)
        
        # Check for 0-byte/corrupted files before rasterio tries to open them
        if s3_fs.size(tif) == 0:
            log_error(errors, f"Raster is 0 bytes (corrupt): {tif}")
            continue
            
        # safely handle path formatting depending on if s3:// prefix exists in the s3fs output
        tif_clean = tif.replace("s3://", "")
        vsi_path = f"/vsis3/{tif_clean}"
        
        try:
            # Added GDAL_DISABLE_READDIR_ON_OPEN to prevent S3 API throttling
            with rasterio.Env(session=AWSSession(), GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(vsi_path) as src:
                    # Attempt to read the first data block to ensure the TIFF has readable underlying data structures
                    try:
                        for ji, window in src.block_windows(1):
                            _ = src.read(1, window=window)
                            break
                    except Exception as read_err:
                        log_error(errors, f"Cannot read data block in {tif}: {str(read_err)}")
                        continue

                    if src.crs.to_string() != target_crs:
                        log_error(errors, f"Incorrect CRS in {tif}")
                    
                    res = src.res
                    if res[0] != target_res:
                        log_error(errors, f"Incorrect X resolution in {tif}: {res[0]}")
                    if res[1] != target_res:
                        log_error(errors, f"Incorrect Y resolution in {tif}: {res[1]}")
                    
                    if src.dtypes[0] != 'float32':
                        log_error(errors, f"Datatype is not float32 in {tif}")
                    if src.profile.get('compress') != 'lzw':
                        log_error(errors, f"Not LZW compressed: {tif}")
                    if src.profile.get('tiled') is not True:
                        log_error(errors, f"File is not tiled: {tif}")
                    if src.profile.get('blockxsize') != 512:
                        log_error(errors, f"Block size X is not 512: {tif}")
        except Exception as e:
            log_error(errors, f"Failed to open/read {tif}: {str(e)}")
                
    print("\nFinished checking CRS and resolution.")
    fail_with_errors(errors, "Raster CRS and Resolution", desc)

def test_raster_approximate_location(s3_fs, s3_prefix, excluded_prefixes):
    """
    Test that all prediction rasters fall within an approximate valid bounding box for all US Coasts.
    Transforms native raster bounds to EPSG:4326 (Lat/Lon) to perform a universal geographic check.
    """
    desc = "Ensures prediction rasters are geographically located roughly within the bounding box of the United States and its territories."
    prediction_dir = str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR', pilot_mode=False)}"))
    
    # Use find to capture files nested in subdirectories, excluding intermediate files
    tifs = [f for f in s3_fs.find(prediction_dir) if f.lower().endswith(('.tif', '.tiff')) and not is_excluded(f, excluded_prefixes)]
    
    if not tifs:
        skip_test(f"No prediction TIFFs found in {prediction_dir} to test location.", "Raster Approximate Location", desc)
        
    total_tifs = len(tifs)
    sampled_tifs = tifs[::10]
    print(f"\nFound {total_tifs} rasters. Testing {len(sampled_tifs)} (every 10th file) for approximate location.")
    
    errors = []
    for i, tif in enumerate(sampled_tifs):
        actual_index = (i * 10) + 1
        print(f"\rTesting location {actual_index}/{total_tifs}: {Path(tif).name[:30]}...   ", end="", flush=True)
        
        tif_clean = tif.replace("s3://", "")
        vsi_path = f"/vsis3/{tif_clean}"
        
        try:
            with rasterio.Env(session=AWSSession()):
                with rasterio.open(vsi_path) as src:
                    # Dynamically get bounds in Lat/Lon regardless of native CRS
                    min_lon, min_lat, max_lon, max_lat = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                    
                    # Broad check: US states and territories fall roughly between -20 and 75 Latitude
                    # (Captures everything from American Samoa up to Northern Alaska)
                    if not (-20 <= min_lat <= 75 and -20 <= max_lat <= 75):
                        log_error(errors, f"Latitude ({min_lat:.2f}, {max_lat:.2f}) seemingly outside US bounds in {tif}")
                        
                    # Broad check: Exclude Prime Meridian, Africa, Europe, and mainland Asia 
                    # This ensures it isnt dumped at exactly 0,0 or completely mangled during reprojection.
                    if (-50 < min_lon < 130) or (-50 < max_lon < 130):
                        log_error(errors, f"Longitude ({min_lon:.2f}, {max_lon:.2f}) seemingly outside US bounds in {tif}")
                        
        except Exception as e:
            log_error(errors, f"Failed to open/read {tif}: {str(e)}")
            
    print("\nFinished checking approximate locations.")
    fail_with_errors(errors, "Raster Approximate Location", desc)


# --- 3. PARQUET DERIVATIVE & TILE TESTS ---

def test_derivative_counts_per_tile(s3_fs, tile_directories, year_pairs):
    """
    Test that exactly 13 long-format files are generated for EVERY tile processed.
    This ensures the `batch_long_format_transformation` loop worked correctly.
    """
    desc = "Confirms that the correct number of long-format parquet derivatives (one for each year pair) are generated for every processed tile."
    all_tile_folders = []
    errors = []
    
    for directory in tile_directories:
        try:
            all_tile_folders.extend(s3_fs.ls(directory))
        except Exception as e:
            log_error(errors, f"Failed to list directory {directory}: {str(e)}")
            
    if not all_tile_folders:
        skip_test(f"No tile folders found in directories: {tile_directories}", "Derivative Counts Per Tile", desc)
            
    total_folders = len(all_tile_folders)
    sampled_folders = all_tile_folders[::10]
    print(f"\nFound {total_folders} tile folders. Checking {len(sampled_folders)} (every 10th tile) for derivative counts.")
    
    # Format expected year pairs for report details
    expected_pairs_str = ", ".join([f"{y0}_{y1}" for y0, y1 in year_pairs])
    
    for i, tile_folder in enumerate(sampled_folders):
        actual_index = (i * 10) + 1
        tile_name = Path(tile_folder).name
        print(f"\rChecking tile {actual_index}/{total_folders}: {tile_name[:30]}...   ", end="", flush=True)
        
        try:
            files = s3_fs.ls(tile_folder)
            long_format_files = [f for f in files if f.endswith("_long.parquet")]
            
            if len(long_format_files) > 0:
                if len(long_format_files) != len(year_pairs):
                    found_names = [Path(f).name for f in long_format_files]
                    log_error(
                        errors, 
                        f"Tile folder '{tile_folder}' has derivative count mismatch. "
                        f"Expected exactly {len(year_pairs)} derivative files matching '*_long.parquet' (one for each configured year pair: [{expected_pairs_str}]). "
                        f"Instead, found {len(long_format_files)} files: {found_names}"
                    )
        except Exception as e:
            log_error(errors, f"Failed to check tile folder {tile_folder}: {str(e)}")
                
    print("\nFinished checking derivative counts.")
    fail_with_errors(errors, "Derivative Counts Per Tile", desc)


def test_primary_training_parquet_schema(s3_fs, year_pairs, training_tiles_dir):
    """
    Test that primary (wide format) training parquets contain all the dynamic
    and static columns correctly generated for the specified year pairs.
    """
    desc = "Checks that wide-format primary training parquet files contain all the correct static columns and dynamic environmental columns generated for every unique year and year pair configuration."
    
    # Get all primary training parquets (not the _long ones)
    all_files = [f for f in s3_fs.glob(f"{training_tiles_dir}/**/*.parquet") if not f.endswith("_long.parquet")]
    
    if not all_files:
        skip_test(f"No primary training parquet files found in {training_tiles_dir}.", "Primary Training Parquet Schema", desc)
        
    total_files = len(all_files)
    sampled_files = all_files[::10]
    print(f"\nFound {total_files} primary training parquets. Checking {len(sampled_files)} (every 10th file) for full wide schema.")
    
    # Build the expected schema dynamically based on year_pairs
    unique_years = sorted(list(set([y for pair in year_pairs for y in pair])))
    
    static_cols = {'X', 'Y', 'grain_size_layer', 'prim_sed_layer', 'survey_end_date', 'FID', 'tile_id'}
    single_year_templates = [
        'bathy_{}_filled', 'bpi_broad_{}', 'bpi_fine_{}', 'curv_plan_{}',
        'curv_profile_{}', 'curv_total_{}', 'flowacc_{}', 'flowdir_{}',
        'gradmag_{}', 'rugosity_{}', 'shearproxy_{}', 'slope_{}',
        'slope_deg_{}', 'tci_{}', 'terrain_classification_{}'
    ]
    pair_year_templates = [
        'hurr_strength_mean_{}_{}', 'tsm_mean_{}_{}', 'delta_bathy_{}_{}'
    ]
    
    expected_cols = set(static_cols)
    for y in unique_years:
        for template in single_year_templates:
            expected_cols.add(template.format(y))
            
    for y1, y2 in year_pairs:
        for template in pair_year_templates:
            expected_cols.add(template.format(y1, y2))
            
    errors = []
    sample_actual_cols = None
    
    for i, file_path in enumerate(sampled_files):
        actual_index = (i * 10) + 1
        print(f"\rChecking primary schema {actual_index}/{total_files}: {Path(file_path).name[:30]}...   ", end="", flush=True)
        
        s3_path = f"s3://{file_path}"
        
        try:
            # Use pyarrow to read schema quickly without loading all data into memory
            import pyarrow.parquet as pq
            with s3_fs.open(s3_path, 'rb') as f:
                schema = pq.read_schema(f)
                actual_cols = set(schema.names)
                
            missing_cols = expected_cols - actual_cols
            if missing_cols:
                log_error(
                    errors,
                    f"Primary schema violation in {s3_path}.\n"
                    f"  Missing required columns: {sorted(list(missing_cols))}"
                )
            elif sample_actual_cols is None:
                sample_actual_cols = sorted(list(actual_cols))
                
        except Exception as e:
            # Fallback to pandas if pyarrow import fails
            try:
                df = pd.read_parquet(s3_path, storage_options={"anon": False})
                actual_cols = set(df.columns)
                missing_cols = expected_cols - actual_cols
                if missing_cols:
                    log_error(
                        errors,
                        f"Primary schema violation in {s3_path}.\n"
                        f"  Missing required columns: {sorted(list(missing_cols))}"
                    )
                elif sample_actual_cols is None:
                    sample_actual_cols = sorted(list(actual_cols))
            except Exception as inner_e:
                log_error(errors, f"Failed to process {s3_path}: {str(inner_e)}")
             
    print("\nFinished checking primary training schemas.")
    
    # Append examples to the description for the report
    desc += f"\n\n--- Validation Details ---\nGenerated {len(expected_cols)} expected columns dynamically from {len(unique_years)} unique years and {len(year_pairs)} year pairs."
    desc += f"\nExpected Schema (Total {len(expected_cols)} cols): {sorted(list(expected_cols))}"
    if sample_actual_cols:
        desc += f"\nExample Found Schema (Total {len(sample_actual_cols)} cols): {sample_actual_cols}"
        
    fail_with_errors(errors, "Primary Training Parquet Schema", desc)


def test_primary_prediction_parquet_schema(s3_fs, year_pairs, prediction_tiles_dir):
    """
    Test that primary prediction parquets contain all the static (BlueTopo)
    and dynamic (year pair) columns correctly generated for the specified year pairs.
    """
    desc = "Checks that primary prediction parquet files contain all the correct static BlueTopo columns and dynamic environmental columns generated for every unique year pair configuration."
    
    # Get all prediction parquets
    all_files = list(s3_fs.glob(f"{prediction_tiles_dir}/**/*_prediction_clipped_data.parquet"))
    
    if not all_files:
        skip_test(f"No primary prediction parquet files found in {prediction_tiles_dir}.", "Primary Prediction Parquet Schema", desc)
        
    total_files = len(all_files)
    sampled_files = all_files[::10]
    print(f"\nFound {total_files} primary prediction parquets. Checking {len(sampled_files)} (every 10th file) for full wide schema.")
    
    static_cols = {
        'X', 'Y', 'bt.bathy', 'bt.bpi_broad', 'bt.bpi_fine', 'bt.curv_plan',
        'bt.curv_profile', 'bt.curv_total', 'bt.flowacc', 'bt.flowdir',
        'bt.gradmag', 'bt.rugosity', 'bt.shearproxy', 'bt.slope', 'bt.slope_deg',
        'bt.tci', 'bt.terrain_classification', 'bt.unc',
        'grain_size_layer', 'prim_sed_layer', 'survey_end_date', 'FID', 'tile_id'
    }
    
    pair_year_templates = [
        'hurr_strength_mean_{}_{}', 'tsm_mean_{}_{}'
    ]
    
    expected_cols = set(static_cols)
    for y1, y2 in year_pairs:
        for template in pair_year_templates:
            expected_cols.add(template.format(y1, y2))
            
    errors = []
    sample_actual_cols = None
    
    for i, file_path in enumerate(sampled_files):
        actual_index = (i * 10) + 1
        print(f"\rChecking prediction schema {actual_index}/{total_files}: {Path(file_path).name[:30]}...   ", end="", flush=True)
        
        s3_path = f"s3://{file_path}"
        
        try:
            # Use pyarrow to read schema quickly without loading all data into memory
            import pyarrow.parquet as pq
            with s3_fs.open(s3_path, 'rb') as f:
                schema = pq.read_schema(f)
                actual_cols = set(schema.names)
                
            missing_cols = expected_cols - actual_cols
            if missing_cols:
                log_error(
                    errors,
                    f"Prediction schema violation in {s3_path}.\n"
                    f"  Missing required columns: {sorted(list(missing_cols))}"
                )
            elif sample_actual_cols is None:
                sample_actual_cols = sorted(list(actual_cols))
                
        except Exception as e:
            # Fallback to pandas if pyarrow import fails
            try:
                df = pd.read_parquet(s3_path, storage_options={"anon": False})
                actual_cols = set(df.columns)
                missing_cols = expected_cols - actual_cols
                if missing_cols:
                    log_error(
                        errors,
                        f"Prediction schema violation in {s3_path}.\n"
                        f"  Missing required columns: {sorted(list(missing_cols))}"
                    )
                elif sample_actual_cols is None:
                    sample_actual_cols = sorted(list(actual_cols))
            except Exception as inner_e:
                log_error(errors, f"Failed to process {s3_path}: {str(inner_e)}")
             
    print("\nFinished checking prediction schemas.")
    
    # Append examples to the description for the report
    desc += f"\n\n--- Validation Details ---\nGenerated {len(expected_cols)} expected columns dynamically from {len(year_pairs)} year pairs."
    desc += f"\nExpected Schema (Total {len(expected_cols)} cols): {sorted(list(expected_cols))}"
    if sample_actual_cols:
        desc += f"\nExample Found Schema (Total {len(sample_actual_cols)} cols): {sample_actual_cols}"
        
    fail_with_errors(errors, "Primary Prediction Parquet Schema", desc)


def test_parquet_schema_and_contents(s3_fs, year_pairs, training_tiles_dir, prediction_tiles_dir):
    """
    Test that EVERY generated training and prediction parquet file contains the exact required schema:
    - X, Y
    - For training tiles specifically: year_t, year_t1, delta_bathy and bathy_t1
    - Verify years match one of our 13 pairs
    """
    desc = "Checks that all long-format training and prediction parquet files contain the required data columns and that their embedded year pairs match the configuration."
    
    train_files = s3_fs.glob(f"{training_tiles_dir}/**/*_long.parquet")
    pred_files = s3_fs.glob(f"{prediction_tiles_dir}/**/*_prediction_clipped_data.parquet")
    
    all_files = train_files + pred_files
    
    if not all_files:
        skip_test(f"No parquet files found in {training_tiles_dir} or {prediction_tiles_dir}.", "Parquet Schema and Contents", desc)
        
    total_files = len(all_files)
    sampled_files = all_files[::10]
    print(f"\nFound {total_files} parquet files. Checking {len(sampled_files)} (every 10th file) for schema and contents.")
    
    expected_pairs_str = ", ".join([f"{y0}_{y1}" for y0, y1 in year_pairs])
    errors = []
    
    # Store an example of valid schemas for the final report
    sample_train_cols = None
    sample_pred_cols = None
    
    for i, file_path in enumerate(sampled_files):
        actual_index = (i * 10) + 1
        print(f"\rChecking schema {actual_index}/{total_files}: {Path(file_path).name[:30]}...   ", end="", flush=True)
        
        s3_path = f"s3://{file_path}"
        is_prediction = file_path in pred_files
        
        try:
            df = pd.read_parquet(s3_path, storage_options={"anon": False})
            actual_cols = list(df.columns)
            
            # Setup specific expectations
            if is_prediction:
                expected_cols = {'X', 'Y'}
                missing_cols = expected_cols - set(actual_cols)
                if missing_cols:
                    log_error(
                        errors,
                        f"Schema violation in prediction file {s3_path}.\n"
                        f"  Expected columns: {sorted(list(expected_cols))}\n"
                        f"  Actual columns  : {actual_cols}\n"
                        f"  Missing columns : {sorted(list(missing_cols))}"
                    )
                elif sample_pred_cols is None:
                    sample_pred_cols = actual_cols # Capture a successful example
            else:
                # For training files, we must check coordinates, years, and bathymetry indicators
                base_expected = {'X', 'Y', 'year_t', 'year_t1'}
                missing_base = base_expected - set(actual_cols)
                
                has_bathy = 'delta_bathy' in actual_cols or 'bathy_t1' in actual_cols
                
                if missing_base or not has_bathy:
                    # Construct a clear display set of expected columns
                    expected_display = sorted(list(base_expected))
                    if 'delta_bathy' in actual_cols:
                        expected_display.append('delta_bathy')
                    elif 'bathy_t1' in actual_cols:
                        expected_display.append('bathy_t1')
                    else:
                        expected_display.append('delta_bathy (or bathy_t1)')
                        
                    missing_details = sorted(list(missing_base))
                    if not has_bathy:
                        missing_details.append("delta_bathy (or bathy_t1)")
                        
                    log_error(
                        errors,
                        f"Schema violation in training file {s3_path}.\n"
                        f"  Expected columns: {expected_display}\n"
                        f"  Actual columns  : {actual_cols}\n"
                        f"  Missing columns : {missing_details}"
                    )
                elif sample_train_cols is None:
                    sample_train_cols = actual_cols # Capture a successful example
                
                # Year value validation
                if 'year_t' in df.columns and 'year_t1' in df.columns:
                    unique_years = df[['year_t', 'year_t1']].drop_duplicates()
                    for _, row in unique_years.iterrows():
                        pair = (int(row['year_t']), int(row['year_t1']))
                        if pair not in year_pairs:
                            log_error(
                                errors, 
                                f"Value validation failure in training file {s3_path}.\n"
                                f"  Found year pair: {pair}\n"
                                f"  Expected one of configured year pairs: [{expected_pairs_str}]"
                            )
        except Exception as e:
             log_error(errors, f"Failed to process {s3_path}: {str(e)}")
             
    print("\nFinished checking parquet schemas.")
    
    # Append examples to the description for the report
    desc += "\n\n--- Validation Details ---"
    desc += "\nExpected Prediction Columns: ['X', 'Y']"
    if sample_pred_cols:
        desc += f"\nExample Found Prediction Schema: {sample_pred_cols}"
        
    desc += "\n\nExpected Training Columns: ['X', 'Y', 'year_t', 'year_t1'] AND ['delta_bathy' OR 'bathy_t1']"
    if sample_train_cols:
        desc += f"\nExample Found Training Schema: {sample_train_cols}"
        
    fail_with_errors(errors, "Parquet Schema and Contents", desc)


def test_long_format_training_parquet_schema(s3_fs, year_pairs, training_tiles_dir):
    """
    Test that long-format training parquet files contain the exact 29 required columns,
    including the 27 static variables and the 2 dynamic variables (hurr_strength_mean and tsm_mean)
    specific to the file's year pair.
    """
    desc = "Checks that long-format training parquet files strictly contain all 27 expected static variables and exactly 2 dynamic year-pair variables."
    
    training_path = UPath(training_tiles_dir)
    long_format_files = list(training_path.rglob("*_long.parquet"))
    
    if not long_format_files:
        skip_test(f"No long-format parquet files found in {training_tiles_dir}", "Long Format Training Parquet Schema", desc)

    total_files = len(long_format_files)
    sample_files = long_format_files[::10]
    print(f"\nFound {total_files} long-format parquets. Checking {len(sample_files)} (every 10th file) for full long schema.")
    
    expected_static_cols = {
        'X', 'Y', 'FID', 'tile_id', 'year_t', 'year_t1', 'bathy_t1', 'bathy_t', 
        'bpi_broad_t', 'bpi_fine_t', 'curv_plan_t', 'curv_profile_t', 'curv_total_t', 
        'flowacc_t', 'flowdir_cos_t', 'flowdir_sin_t', 'gradmag_t', 'rugosity_t', 
        'shearproxy_t', 'slope_t', 'slope_deg_t', 'tci_t', 'terrain_classification_t', 
        'delta_bathy', 'grain_size_layer', 'prim_sed_layer', 'survey_end_date'
    }
    
    errors = []
    sample_actual_cols = None
    
    for i, file_path in enumerate(sample_files):
        actual_index = (i * 10) + 1
        print(f"\rChecking long schema {actual_index}/{total_files}: {file_path.name[:30]}...   ", end="", flush=True)
        
        s3_path = f"s3://{file_path}"
        try:
            # Use pyarrow to read schema quickly
            try:
                import pyarrow.parquet as pq
                with s3_fs.open(s3_path, 'rb') as f:
                    schema = pq.read_schema(f)
                    actual_cols = set(schema.names)
            except Exception:
                # Fallback to pandas
                df = pd.read_parquet(s3_path, storage_options={"anon": False})
                actual_cols = set(df.columns)
                
            missing_static = expected_static_cols - actual_cols
            if missing_static:
                log_error(errors, f"File {file_path.name} is missing static columns: {sorted(list(missing_static))}")
            
            # Check dynamic columns
            dynamic_hurr = [c for c in actual_cols if c.startswith('hurr_strength_mean_')]
            dynamic_tsm = [c for c in actual_cols if c.startswith('tsm_mean_')]
            
            if len(dynamic_hurr) != 1:
                log_error(errors, f"File {file_path.name} expects exactly 1 hurr_strength_mean column, found {len(dynamic_hurr)}: {dynamic_hurr}")
            if len(dynamic_tsm) != 1:
                log_error(errors, f"File {file_path.name} expects exactly 1 tsm_mean column, found {len(dynamic_tsm)}: {dynamic_tsm}")
                
            # Verify they belong to configured year pairs
            if len(dynamic_hurr) == 1 and len(dynamic_tsm) == 1:
                try:
                    hurr_pair = dynamic_hurr[0].replace('hurr_strength_mean_', '').split('_')
                    h_pair_tuple = (int(hurr_pair[0]), int(hurr_pair[1]))
                    if h_pair_tuple not in year_pairs:
                        log_error(errors, f"File {file_path.name} has hurr_strength_mean for unconfigured pair {h_pair_tuple}")
                except Exception:
                    log_error(errors, f"Could not parse year pair from column {dynamic_hurr[0]}")
                        
                try:
                    tsm_pair = dynamic_tsm[0].replace('tsm_mean_', '').split('_')
                    t_pair_tuple = (int(tsm_pair[0]), int(tsm_pair[1]))
                    if t_pair_tuple not in year_pairs:
                        log_error(errors, f"File {file_path.name} has tsm_mean for unconfigured pair {t_pair_tuple}")
                except Exception:
                    log_error(errors, f"Could not parse year pair from column {dynamic_tsm[0]}")
                    
            # Check for unexpected extra columns
            extra_cols = actual_cols - expected_static_cols - set(dynamic_hurr) - set(dynamic_tsm)
            if extra_cols:
                log_error(errors, f"File {file_path.name} has unexpected extra columns: {sorted(list(extra_cols))}")

            if sample_actual_cols is None:
                sample_actual_cols = sorted(list(actual_cols))
                
        except Exception as e:
            log_error(errors, f"Failed to read schema for {file_path.name}: {str(e)}")
                
    print("\nFinished checking long format training schemas.")
    
    desc += f"\n\n--- Validation Details ---\nExpected {len(expected_static_cols)} static columns: {sorted(list(expected_static_cols))}"
    desc += "\nExpected exactly 2 dynamic columns matching: ['hurr_strength_mean_Y0_Y1', 'tsm_mean_Y0_Y1']"
    desc += f"\nTotal Expected Columns per file: {len(expected_static_cols) + 2}"
    if sample_actual_cols:
        desc += f"\nExample Found Schema (Total {len(sample_actual_cols)} cols): {sample_actual_cols}"
        
    fail_with_errors(errors, "Long Format Training Parquet Schema", desc)


def test_long_format_prediction_parquet_schema(s3_fs, year_pairs, prediction_tiles_dir):
    """
    Test that long-format prediction parquet files contain the exact 26 required columns,
    including the 24 static variables and the 2 dynamic variables (hurr_strength_mean and tsm_mean)
    specific to the file's year pair.
    """
    desc = "Checks that long-format prediction parquet files strictly contain all 24 expected static variables and exactly 2 dynamic year-pair variables."
    
    prediction_path = UPath(prediction_tiles_dir)
    long_format_files = list(prediction_path.rglob("*_long.parquet"))
    
    if not long_format_files:
        skip_test(f"No long-format prediction parquet files found in {prediction_tiles_dir}", "Long Format Prediction Parquet Schema", desc)

    total_files = len(long_format_files)
    sample_files = long_format_files[::10]
    print(f"\nFound {total_files} prediction long-format parquets. Checking {len(sample_files)} (every 10th file) for full long schema.")
    
    expected_static_cols = {
        'X', 'Y', 'FID', 'tile_id', 'bathy_t', 'bpi_broad_t', 'bpi_fine_t', 
        'curv_plan_t', 'curv_profile_t', 'curv_total_t', 'flowacc_t', 'gradmag_t', 
        'rugosity_t', 'shearproxy_t', 'slope_t', 'slope_deg_t', 'tci_t', 
        'terrain_classification_t', 'unc_t', 'flowdir_sin_t', 'flowdir_cos_t',
        'grain_size_layer', 'prim_sed_layer', 'survey_end_date'
    }
    
    errors = []
    sample_actual_cols = None
    
    for i, file_path in enumerate(sample_files):
        actual_index = (i * 10) + 1
        print(f"\rChecking long prediction schema {actual_index}/{total_files}: {file_path.name[:30]}...   ", end="", flush=True)
        
        s3_path = f"s3://{file_path}"
        try:
            # Use pyarrow to read schema quickly
            try:
                import pyarrow.parquet as pq
                with s3_fs.open(s3_path, 'rb') as f:
                    schema = pq.read_schema(f)
                    actual_cols = set(schema.names)
            except Exception:
                # Fallback to pandas
                df = pd.read_parquet(s3_path, storage_options={"anon": False})
                actual_cols = set(df.columns)
                
            missing_static = expected_static_cols - actual_cols
            if missing_static:
                log_error(errors, f"File {file_path.name} is missing static columns: {sorted(list(missing_static))}")
            
            # Check dynamic columns
            dynamic_hurr = [c for c in actual_cols if c.startswith('hurr_strength_mean_')]
            dynamic_tsm = [c for c in actual_cols if c.startswith('tsm_mean_')]
            
            if len(dynamic_hurr) != 1:
                log_error(errors, f"File {file_path.name} expects exactly 1 hurr_strength_mean column, found {len(dynamic_hurr)}: {dynamic_hurr}")
            if len(dynamic_tsm) != 1:
                log_error(errors, f"File {file_path.name} expects exactly 1 tsm_mean column, found {len(dynamic_tsm)}: {dynamic_tsm}")
                
            # Verify they belong to configured year pairs
            if len(dynamic_hurr) == 1 and len(dynamic_tsm) == 1:
                try:
                    hurr_pair = dynamic_hurr[0].replace('hurr_strength_mean_', '').split('_')
                    h_pair_tuple = (int(hurr_pair[0]), int(hurr_pair[1]))
                    if h_pair_tuple not in year_pairs:
                        log_error(errors, f"File {file_path.name} has hurr_strength_mean for unconfigured pair {h_pair_tuple}")
                except Exception:
                    log_error(errors, f"Could not parse year pair from column {dynamic_hurr[0]}")
                        
                try:
                    tsm_pair = dynamic_tsm[0].replace('tsm_mean_', '').split('_')
                    t_pair_tuple = (int(tsm_pair[0]), int(tsm_pair[1]))
                    if t_pair_tuple not in year_pairs:
                        log_error(errors, f"File {file_path.name} has tsm_mean for unconfigured pair {t_pair_tuple}")
                except Exception:
                    log_error(errors, f"Could not parse year pair from column {dynamic_tsm[0]}")
                    
            # Check for unexpected extra columns
            extra_cols = actual_cols - expected_static_cols - set(dynamic_hurr) - set(dynamic_tsm)
            if extra_cols:
                log_error(errors, f"File {file_path.name} has unexpected extra columns: {sorted(list(extra_cols))}")

            if sample_actual_cols is None:
                sample_actual_cols = sorted(list(actual_cols))
                
        except Exception as e:
            log_error(errors, f"Failed to read schema for {file_path.name}: {str(e)}")
                
    print("\nFinished checking long format prediction schemas.")
    
    desc += f"\n\n--- Validation Details ---\nExpected {len(expected_static_cols)} static columns: {sorted(list(expected_static_cols))}"
    desc += "\nExpected exactly 2 dynamic columns matching: ['hurr_strength_mean_Y0_Y1', 'tsm_mean_Y0_Y1']"
    desc += f"\nTotal Expected Columns per file: {len(expected_static_cols) + 2}"
    if sample_actual_cols:
        desc += f"\nExample Found Schema (Total {len(sample_actual_cols)} cols): {sample_actual_cols}"
        
    fail_with_errors(errors, "Long Format Prediction Parquet Schema", desc)


def test_parquet_spatial_integrity(s3_fs, prediction_tiles_dir):
    """
    Test that the X and Y coordinates inside ALL Parquet tabular data
    align with the target resolution rounding (e.g. they should not have 
    crazy floating points if resolution is exactly 8m).
    """
    desc = "Verifies that the X and Y coordinates inside the parquet files align perfectly with the target resolution grid without unintended floating-point decimals."
    all_files = s3_fs.glob(f"{prediction_tiles_dir}/**/*_prediction_clipped_data.parquet")
    
    if not all_files:
        skip_test(f"No prediction parquet files found in {prediction_tiles_dir}.", "Parquet Spatial Integrity", desc)
        
    total_files = len(all_files)
    sampled_files = all_files[::10]
    print(f"\nFound {total_files} parquet files. Checking {len(sampled_files)} (every 10th file) for spatial integrity.")
    
    errors = []
    for i, file_path in enumerate(sampled_files):
        actual_index = (i * 10) + 1
        print(f"\rChecking coordinates {actual_index}/{total_files}: {Path(file_path).name[:30]}...   ", end="", flush=True)
        
        s3_path = f"s3://{file_path}"
        try:
            df = pd.read_parquet(s3_path, storage_options={"anon": False}, columns=['X', 'Y'])
            
            if df.empty:
                log_error(errors, f"Parquet file {s3_path} is empty.")
                continue
                
            # Check X coordinates
            x_rounded = df['X'].round(3)
            if not np.allclose(df['X'], x_rounded, atol=1e-4, equal_nan=True):
                # Isolate a bad value to print in the report for easier debugging
                bad_x_idx = ~np.isclose(df['X'], x_rounded, atol=1e-4, equal_nan=True)
                bad_x = df['X'][bad_x_idx].iloc[0]
                log_error(errors, f"X coordinates have >3 decimal places in {s3_path} (e.g., {bad_x})")
                
            # Check Y coordinates
            y_rounded = df['Y'].round(3)
            if not np.allclose(df['Y'], y_rounded, atol=1e-4, equal_nan=True):
                bad_y_idx = ~np.isclose(df['Y'], y_rounded, atol=1e-4, equal_nan=True)
                bad_y = df['Y'][bad_y_idx].iloc[0]
                log_error(errors, f"Y coordinates have >3 decimal places in {s3_path} (e.g., {bad_y})")
                
        except Exception as e:
             log_error(errors, f"Failed to process {s3_path}: {str(e)}")
             
    print("\nFinished checking spatial integrity.")
    fail_with_errors(errors, "Parquet Spatial Integrity", desc)


def test_parquet_prediction_training_parity(s3_fs, prediction_tiles_dir, training_tiles_dir):
    """
    Test that corresponding prediction and training parquet files for each tile have the same number of rows,
    identical bounding box coordinates, and matching data in shared columns where training data is valid.
    """
    desc = "Verifies that each tile's prediction and primary training parquet files have matching row counts, identical bounding box spatial extents, and perfectly aligned data in shared columns where training values exist."
    
    try:
        pred_tile_folders = s3_fs.ls(prediction_tiles_dir)
    except FileNotFoundError:
        pred_tile_folders = []
        
    if not pred_tile_folders:
        skip_test(f"No tile folders found in {prediction_tiles_dir}", "Parquet Prediction/Training Parity", desc)
        
    total_folders = len(pred_tile_folders)
    sampled_folders = pred_tile_folders[::10]
    print(f"\nFound {total_folders} tile folders. Checking {len(sampled_folders)} (every 10th tile) for prediction vs training parquet parity.")
    
    errors = []
    
    for i, pred_folder in enumerate(sampled_folders):
        actual_index = (i * 10) + 1
        tile_name = Path(pred_folder).name
        print(f"\rChecking parity {actual_index}/{total_folders}: {tile_name[:30]}...   ", end="", flush=True)
        
        # Locate the prediction parquet
        pred_files = [f for f in s3_fs.ls(pred_folder) if f.endswith("_prediction_clipped_data.parquet")]
        if not pred_files:
            continue
        pred_file = pred_files[0]
        
        # Locate the matching primary training folder
        train_folder = f"{training_tiles_dir.rstrip('/')}/{tile_name}"
        
        try:
            # Look for the primary training parquet (ignoring long format derivatives and csvs)
            train_files = [f for f in s3_fs.ls(train_folder) if f.endswith(".parquet") and not f.endswith("_long.parquet")]
            if not train_files:
                log_error(errors, f"Missing primary training parquet for tile {tile_name} in {train_folder}")
                continue
            train_file = train_files[0]
            
            # Load both files
            df_pred = pd.read_parquet(f"s3://{pred_file}", storage_options={"anon": False})
            df_train = pd.read_parquet(f"s3://{train_file}", storage_options={"anon": False})
            
            # Check 1: Row count parity
            if len(df_pred) != len(df_train):
                log_error(errors, f"Row count mismatch for tile {tile_name}: Prediction ({len(df_pred)} rows) vs Training ({len(df_train)} rows)")
                continue
                
            # Check 2: Bounding Box / Extent Parity
            pred_bbox = (df_pred['X'].min(), df_pred['Y'].min(), df_pred['X'].max(), df_pred['Y'].max())
            train_bbox = (df_train['X'].min(), df_train['Y'].min(), df_train['X'].max(), df_train['Y'].max())
            
            if not np.allclose(pred_bbox, train_bbox, equal_nan=True):
                log_error(errors, f"Bounding Box mismatch for tile {tile_name}. Pred: {pred_bbox}, Train: {train_bbox}")
                
            # Check 3: Shared column data alignment
            shared_cols = set(df_pred.columns).intersection(set(df_train.columns))
            
            # Ensure index alignment by sorting on coordinates if they don't perfectly match inherently
            if not np.allclose(df_pred['X'], df_train['X'], equal_nan=True) or not np.allclose(df_pred['Y'], df_train['Y'], equal_nan=True):
                df_pred = df_pred.sort_values(by=['Y', 'X']).reset_index(drop=True)
                df_train = df_train.sort_values(by=['Y', 'X']).reset_index(drop=True)
                
            for col in shared_cols:
                # Get the mask of where the training data actually has valid values
                valid_mask = df_train[col].notna()
                
                train_valid = df_train.loc[valid_mask, col]
                pred_valid = df_pred.loc[valid_mask, col]
                
                if len(train_valid) == 0:
                    continue
                    
                # Float columns use allclose, object/string columns use equals
                if pd.api.types.is_numeric_dtype(train_valid):
                    if not np.allclose(train_valid, pred_valid, equal_nan=True):
                        log_error(errors, f"Value parity mismatch in shared column '{col}' for tile {tile_name}")
                else:
                    if not train_valid.equals(pred_valid):
                        log_error(errors, f"Value parity mismatch in shared column '{col}' for tile {tile_name}")
                        
        except Exception as e:
            log_error(errors, f"Failed processing prediction/training parity for {tile_name}: {str(e)}")
            
    print("\nFinished checking prediction and training parity.")
    fail_with_errors(errors, "Parquet Prediction/Training Parity", desc)


# --- 4. GEOMETRY / MASK TESTS ---

def test_mask_and_subgrid_crs(s3_fs, target_crs, s3_prefix):
    """
    Test that the training mask and subgrids were saved with the correct CRS.
    """
    desc = "Ensures that the spatial geometries for training masks and regional subgrids were saved using the correct target CRS."
    mask_pq = str(UPath(f"{s3_prefix}{get_config_item('MASK', 'PREDICTION_MASK_PQ', pilot_mode=False)}"))
    subgrid_gpkg = str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'SUBGRIDS', pilot_mode=False)}"))
    
    errors = []
    
    # Check Mask Parquet
    if s3_fs.exists(mask_pq):
        gdf = gpd.read_parquet(f"s3://{mask_pq}", storage_options={"anon": False})
        if gdf.crs.to_string() != target_crs:
             log_error(errors, f"Mask Parquet CRS is incorrect in {mask_pq}: {gdf.crs.to_string()}")
        
    # Check GPKG
    if s3_fs.exists(subgrid_gpkg):
        with s3_fs.open(subgrid_gpkg, "rb") as f:
            gdf = gpd.read_file(f)
            if gdf.crs.to_string() != target_crs:
                log_error(errors, f"Subgrid GPKG CRS is incorrect in {subgrid_gpkg}: {gdf.crs.to_string()}")

    fail_with_errors(errors, "Mask and Subgrid CRS", desc)


# --- 5. DATA VALIDITY & INTEGRITY TESTS ---

def test_training_tiffs_are_masked(s3_fs, s3_prefix, target_crs, excluded_prefixes):
    """
    Test that training TIFFs truly no longer have data where training masks should be masking data.
    Reads rasters in chunks to prevent EC2 OOM / stalling.
    """
    desc = "Validates that pixels in the training TIFFs falling outside of the designated valid training geometries are properly set to nodata (NaN)."
    train_dir = str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'TRAINING_OUTPUT_DIR', pilot_mode=False)}"))
    
    try:
        mask_pq_suffix = get_config_item('MASK', 'TRAINING_MASK_PQ', pilot_mode=False)
        mask_pq = str(UPath(f"{s3_prefix}{mask_pq_suffix}"))
    except KeyError:
        skip_test("Config key MASK.TRAINING_MASK_PQ not found.", "Training TIFF Masking", desc)

    if not s3_fs.exists(mask_pq):
        skip_test(f"Training mask parquet not found at {mask_pq}", "Training TIFF Masking", desc)

    mask_gdf = gpd.read_parquet(f"s3://{mask_pq}", storage_options={"anon": False})
    mask_gdf = mask_gdf.to_crs(target_crs)
    
    if mask_gdf.empty:
        skip_test("Training mask geometry is empty.", "Training TIFF Masking", desc)
        
    valid_geometries = mask_gdf.geometry.tolist()
    
    # Update to support recursive TIFFs within the training directory hierarchy, excluding intermediates
    train_tifs = [f for f in s3_fs.find(train_dir) if f.lower().endswith(('.tif', '.tiff')) and not is_excluded(f, excluded_prefixes)]
    if not train_tifs:
        skip_test(f"No training tiffs found in {train_dir} to validate masking.", "Training TIFF Masking", desc)

    total_tifs = len(train_tifs)
    sampled_tifs = train_tifs[::10]
    print(f"\nFound {total_tifs} training tiffs. Checking {len(sampled_tifs)} (every 10th file) for proper masking.")
    
    errors = []
    with rasterio.Env(session=AWSSession()):
        for i, train_tif in enumerate(sampled_tifs):
            actual_index = (i * 10) + 1
            print(f"\rChecking masking {actual_index}/{total_tifs}: {Path(train_tif).name[:30]}...   ", end="", flush=True)
            
            tif_clean = train_tif.replace("s3://", "")
            
            try:
                with rasterio.open(f"/vsis3/{tif_clean}") as src:
                    nodata_val = src.nodata
                    is_nodata_nan = pd.isna(nodata_val)
                    
                    # Pre-filter geometries to just the raster's bounding box
                    # This prevents geometry_mask from choking on massive data sets
                    raster_bounds = box(*src.bounds)
                    local_geometries = [geom for geom in valid_geometries if geom.intersects(raster_bounds)]
                    
                    # Chunked evaluation
                    for ji, window in src.block_windows(1):
                        data = src.read(1, window=window)
                        
                        # If there are no geometries in the bounding box, the whole block should be masked
                        if not local_geometries:
                            out_of_bounds_mask = np.ones(data.shape, dtype=bool)
                        else:
                            out_of_bounds_mask = geometry_mask(
                                local_geometries,
                                out_shape=data.shape,
                                transform=src.window_transform(window),
                                invert=False 
                            )
                        
                        masked_pixels = data[out_of_bounds_mask]
                        if len(masked_pixels) == 0:
                            continue
                        
                        if is_nodata_nan:
                            if not np.all(np.isnan(masked_pixels)):
                                log_error(errors, f"Found unmasked data outside bounds in {train_tif} at window {window}")
                                break # Stop checking chunks for this file
                        else:
                            if not np.all(masked_pixels == nodata_val):
                                log_error(errors, f"Found unmasked data outside bounds in {train_tif} at window {window}")
                                break # Stop checking chunks for this file
                                
            except Exception as e:
                log_error(errors, f"Failed to process {train_tif}: {str(e)}")
                
    print("\nFinished checking training tiff masking.")
    fail_with_errors(errors, "Training TIFF Masking", desc)

def test_training_and_prediction_values_match(s3_fs, s3_prefix, excluded_prefixes):
    """
    Test that where training TIFFs have valid data, their values match 
    the values in the corresponding prediction TIFF.
    Reads rasters in chunks to prevent EC2 OOM / stalling.
    """
    desc = "Confirms that pixel values in the training TIFFs perfectly match their corresponding prediction TIFFs where valid data exists."
    train_dir_str = str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'TRAINING_OUTPUT_DIR', pilot_mode=False)}"))
    pred_dir_str = str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR', pilot_mode=False)}"))

    # Use find to locate files in subdirectories too, excluding intermediate folders, and skipping TSM/Hurricane files
    train_tifs = [
        f for f in s3_fs.find(train_dir_str) 
        if f.lower().endswith(('.tif', '.tiff')) 
        and not is_excluded(f, excluded_prefixes)
        and "tsm" not in f.lower()
        and "hurricane" not in f.lower()
    ]
    
    if not train_tifs:
        skip_test(f"No training tiffs found in {train_dir_str} to compare values.", "Training/Prediction Values Match", desc)

    total_tifs = len(train_tifs)
    sampled_tifs = train_tifs[::10]
    print(f"\nFound {total_tifs} training tiffs. Checking {len(sampled_tifs)} (every 10th file) against prediction tiffs.")

    errors = []
    for i, train_tif_path in enumerate(sampled_tifs):
        actual_index = (i * 10) + 1
        filename = Path(train_tif_path).name
        print(f"\rChecking values {actual_index}/{total_tifs}: {filename[:30]}...   ", end="", flush=True)
        
        # Reconstruct the relative path to locate the prediction raster in the exact same subdirectory
        train_clean = train_tif_path.replace("s3://", "")
        train_dir_clean = train_dir_str.replace("s3://", "")
        rel_path = train_clean.replace(train_dir_clean, "").lstrip("/")
        
        pred_s3_path = f"{pred_dir_str}/{rel_path}"
        pred_vsi_path = f"/vsis3/{pred_s3_path.replace('s3://', '')}"
        train_vsi_path = f"/vsis3/{train_clean}"
        
        # If the prediction file doesnt exist, just skip it instead of logging an error
        if not s3_fs.exists(pred_s3_path):
            continue
            
        # Check for 0-byte/corrupted files before rasterio tries to open them
        if s3_fs.size(train_tif_path) == 0:
            log_error(errors, f"Training file is 0 bytes (corrupt): {train_tif_path}")
            continue
        if s3_fs.size(pred_s3_path) == 0:
            log_error(errors, f"Prediction file is 0 bytes (corrupt): {pred_s3_path}")
            continue

        try:
            # Move AWSSession inside the loop to prevent token expiration on long runs
            # and add GDAL_DISABLE_READDIR_ON_OPEN to prevent S3 API throttling
            with rasterio.Env(session=AWSSession(), GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
                with rasterio.open(train_vsi_path) as src_train, \
                     rasterio.open(pred_vsi_path) as src_pred:
                    
                    if src_train.shape != src_pred.shape:
                        log_error(errors, f"Shape mismatch: train {train_tif_path} {src_train.shape} vs pred {pred_s3_path} {src_pred.shape}")
                        continue
                        
                    train_nodata = src_train.nodata
                    is_nodata_nan = pd.isna(train_nodata)
                    
                    # Chunked evaluation
                    for ji, window in src_train.block_windows(1):
                        train_data = src_train.read(1, window=window)
                        pred_data = src_pred.read(1, window=window)
                        
                        if is_nodata_nan:
                            valid_mask = ~np.isnan(train_data)
                        else:
                            valid_mask = train_data != train_nodata
                            
                        train_valid_pixels = train_data[valid_mask]
                        pred_valid_pixels = pred_data[valid_mask]
                        
                        if len(train_valid_pixels) == 0:
                            continue
                            
                        if not np.allclose(train_valid_pixels, pred_valid_pixels, equal_nan=True):
                            log_error(errors, f"Pixel values mismatched between {train_tif_path} and {pred_s3_path} at window {window}")
                            break # Stop checking chunks for this file
                            
        except Exception as e:
            log_error(errors, f"Failed to process {train_tif_path}: {str(e)}")
            
    print("\nFinished checking train vs prediction values.")
    fail_with_errors(errors, "Training/Prediction Values Match", desc)

def test_year_range_tiffs_exist(s3_fs, s3_prefix, year_pairs, excluded_prefixes):
    """
    Test that files for each year range (like hurr_count_cumulative, tsm_mean, etc.)
    exist for both prediction and training output directories.
    """
    desc = "Checks that all expected environmental derivative TIFFs (like hurricane counts and TSM) exist for each specified year range configuration."
    train_dir = str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'TRAINING_OUTPUT_DIR', pilot_mode=False)}"))
    pred_dir = str(UPath(f"{s3_prefix}{get_config_item('MODEL', 'PREDICTION_OUTPUT_DIR', pilot_mode=False)}"))

    expected_prefixes = ['hurr_count_cumulative', 'hurr_count_mean', 'tsm_mean', 'tsm_cumulative']
    
    # Pull recursive lists so sub-directories are fully searched, excluding intermediate folders
    train_files = [f for f in s3_fs.find(train_dir) if not is_excluded(f, excluded_prefixes)]
    pred_files = [f for f in s3_fs.find(pred_dir) if not is_excluded(f, excluded_prefixes)]

    errors = []
    for directory, all_files in [(train_dir, train_files), (pred_dir, pred_files)]:
        for year_t, year_t1 in year_pairs:
            for prefix in expected_prefixes:
                if prefix.startswith('tsm_'):
                    expected_filename = f"{year_t}_{year_t1}_{prefix}.tif"
                else:
                    expected_filename = f"{prefix}_{year_t}_{year_t1}.tif"
                
                # Verify that at least one file ending with the expected filename exists
                if not any(f.endswith(expected_filename) for f in all_files):
                    log_error(errors, f"Missing expected TIFF matching '{expected_filename}' in {directory}")

    fail_with_errors(errors, "Year Range TIFF Existence", desc)


# --- CUSTOM TEST RUNNER ---
# Run via `python test_s3_outputs.py` to use this block!

if __name__ == "__main__":
    import sys

    # Comment out any test in this list that you want to skip.
    tests_to_run = [
        # "test_output_directories_exist",
        # "test_year_range_tiffs_exist",
        # "test_raster_approximate_location",
        # "test_raster_crs_and_resolution",
        # "test_training_tiffs_are_masked",
        "test_training_and_prediction_values_match",
        "test_derivative_counts_per_tile",
        "test_primary_training_parquet_schema",
        "test_primary_prediction_parquet_schema",
        "test_parquet_schema_and_contents",
        "test_long_format_training_parquet_schema",
        "test_long_format_prediction_parquet_schema",
        "test_parquet_spatial_integrity",
        "test_parquet_prediction_training_parity",
        "test_mask_and_subgrid_crs",
        "test_csv_stats_generated"
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
    
    print(f"\n[+] Testing finished. Final report uploaded to {REPORT_S3_DIR}/S3_Validation_Report.pdf (or .md)")
    
    sys.exit(exit_code)