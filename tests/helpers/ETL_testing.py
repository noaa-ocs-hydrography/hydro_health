import os
import pytest
import rasterio
import geopandas as gpd
import numpy as np
import s3fs
import logging
import glob
from datetime import datetime
from rasterio.features import rasterize

# =====================================================================
# LOGGING CONFIGURATION
# =====================================================================
# Create a 'logs' directory in the same folder as this script
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(log_dir, exist_ok=True)

# --- Keep a maximum of 5 log files ---
# Find all existing log files matching the pattern and sort them (oldest first)
existing_logs = sorted(glob.glob(os.path.join(log_dir, "etl_testing_*.log")))

# Keep only the newest 4, delete the rest so the new file makes exactly 5 total
if len(existing_logs) >= 5:
    logs_to_delete = existing_logs[:-4]
    for old_log in logs_to_delete:
        try:
            os.remove(old_log)
        except OSError:
            pass # Skip if the file is locked or currently in use

# Generate a timestamped log file name
log_file = os.path.join(log_dir, f"etl_testing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Prevent adding multiple handlers if pytest re-runs the module
if not logger.handlers:
    # File handler (writes to the log file)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler (prints to terminal)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Define log message formatting
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

logger.info("=== ETL Test Suite Initialized ===")
logger.info(f"Writing logs to: {log_file}")


# =====================================================================
# 1. PLACEHOLDERS: Fill these in with your S3 paths and expected values
# =====================================================================

# Directories for prediction and training TIFFs
S3_PREDICTION_DIR = "s3://ocs-dev-csdl-hydrohealth/ER_3/pilot/model_variables/Prediction/processed"
S3_TRAINING_DIR = "s3://ocs-dev-csdl-hydrohealth/ER_3/pilot/model_variables/Training/processed"

# Directories for prediction and training GeoParquets
S3_PRED_PARQUET_DIR = "s3://ocs-dev-csdl-hydrohealth/ER_3/pilot/model_variables/Prediction/prediction_tiles"
S3_TRAIN_PARQUET_DIR = "s3://ocs-dev-csdl-hydrohealth/ER_3/pilot/model_variables/Training/training_tiles"

# NEW: Single masks for the entire ecoregion/dataset
S3_PREDICTION_MASK_PATH = "s3://ocs-dev-csdl-hydrohealth/ER_3/pilot/masks/prediction_mask_pilot.parquet"
S3_TRAINING_MASK_PATH = "s3://ocs-dev-csdl-hydrohealth/ER_3/pilot/masks/training_mask_pilot.parquet"

EXPECTED_CRS = "EPSG:32617"     # Expected Coordinate Reference System (e.g., EPSG:4326, EPSG:32630)
EXPECTED_RESOLUTION = (8, 8)    # Expected pixel resolution (x, y) in map units, this should be meters

# =====================================================================
# 2. FIXTURES: Load data once per test session to save time and network
# =====================================================================

@pytest.fixture(scope="module")
def tiff_inventories():
    """
    Lists all TIFFs in the prediction and training directories.
    Returns two dictionaries mapping the base filename (no extension) to its full S3 path.
    """
    fs = s3fs.S3FileSystem(anon=False)
    
    # glob returns paths without the 's3://' protocol prefix, so we add it back
    # We grab both .tif and .tiff extensions just to be safe
    pred_files = fs.glob(f"{S3_PREDICTION_DIR}/*.tif") + fs.glob(f"{S3_PREDICTION_DIR}/*.tiff")
    train_files = fs.glob(f"{S3_TRAINING_DIR}/*.tif") + fs.glob(f"{S3_TRAINING_DIR}/*.tiff")
    
    # os.path.splitext removes the extension so we can match purely on the base name
    pred_dict = {os.path.splitext(os.path.basename(f))[0]: f"s3://{f}" for f in pred_files}
    train_dict = {os.path.splitext(os.path.basename(f))[0]: f"s3://{f}" for f in train_files}
    
    return pred_dict, train_dict

@pytest.fixture(scope="module")
def parquet_inventories():
    """
    Lists all GeoParquets in the prediction and training directories.
    Returns two dictionaries mapping the base filename (no extension) to its full S3 path.
    """
    fs = s3fs.S3FileSystem(anon=False)
    
    pred_files = fs.glob(f"{S3_PRED_PARQUET_DIR}/*.parquet")
    train_files = fs.glob(f"{S3_TRAIN_PARQUET_DIR}/*.parquet")
    
    # os.path.splitext removes the extension so we can match purely on the base name
    pred_dict = {os.path.splitext(os.path.basename(f))[0]: f"s3://{f}" for f in pred_files}
    train_dict = {os.path.splitext(os.path.basename(f))[0]: f"s3://{f}" for f in train_files}
    
    return pred_dict, train_dict

@pytest.fixture(scope="module")
def pred_mask_gdf():
    """Reads the single Prediction GeoParquet mask file from S3."""
    return gpd.read_parquet(S3_PREDICTION_MASK_PATH)

@pytest.fixture(scope="module")
def train_mask_gdf():
    """Reads the single Training GeoParquet mask file from S3."""
    return gpd.read_parquet(S3_TRAINING_MASK_PATH)

# =====================================================================
# 3. THE TESTS
# =====================================================================

def test_tiff_metadata(tiff_inventories):
    """Checks if all TIFFs have the exact CRS, resolution, and data type (float32) expected."""
    pred_dict, train_dict = tiff_inventories
    
    mismatches = []

    def check_metadata(tiff_dict, dataset_name):
        logger.info(f"Validating Metadata for {len(tiff_dict)} {dataset_name} TIFFs: Expected CRS {EXPECTED_CRS}, Resolution {EXPECTED_RESOLUTION}, Dtype float32")
        with rasterio.Env():
            for name, path in tiff_dict.items():
                try:
                    with rasterio.open(path) as src:
                        # Test CRS
                        actual_crs = src.crs.to_string() if src.crs else "None"
                        if actual_crs != EXPECTED_CRS:
                            mismatches.append(f"[{dataset_name}] '{name}' CRS mismatch: Expected {EXPECTED_CRS}, got {actual_crs}")
                        
                        # Test Resolution
                        actual_res = src.res
                        if actual_res != EXPECTED_RESOLUTION:
                            mismatches.append(f"[{dataset_name}] '{name}' Resolution mismatch: Expected {EXPECTED_RESOLUTION}, got {actual_res}")

                        # Test Data Type (Dtype)
                        # src.dtypes returns a tuple of data types for each band (e.g., ('float32',))
                        for band_idx, dtype in enumerate(src.dtypes):
                            if dtype != 'float32':
                                mismatches.append(f"[{dataset_name}] '{name}' Band {band_idx + 1} dtype mismatch: Expected float32, got {dtype}")

                except Exception as e:
                    mismatches.append(f"[{dataset_name}] '{name}' FAILED to read metadata: {str(e)}")

    # Run for both prediction and training sets
    check_metadata(pred_dict, "Prediction")
    check_metadata(train_dict, "Training")

    if mismatches:
        logger.error(f"Found {len(mismatches)} files with metadata mismatches.")
        for mismatch in mismatches:
            logger.error(mismatch)
                
    assert not mismatches, (
        f"Found {len(mismatches)} files with metadata mismatches:\n" + 
        "\n".join(mismatches)
    )


def test_all_nan_masks_match_parquet_extents(tiff_inventories, pred_mask_gdf, train_mask_gdf):
    """
    Validates that the valid pixels in all TIFFs exactly match the boundaries 
    of the single GeoParquet mask for their respective dataset.
    """
    pred_tiffs, train_tiffs = tiff_inventories
    
    mismatches = []

    def check_masks(tiff_dict, mask_gdf, dataset_name):
        logger.info(f"Checking {len(tiff_dict)} mask overlays against the single {dataset_name} mask...")
        
        for name, tiff_path in tiff_dict.items():
            try:
                with rasterio.Env():
                    with rasterio.open(tiff_path) as tiff_src:
                        # 1. Ensure the vector mask is in the same projection as the TIFF
                        current_mask_gdf = mask_gdf
                        if current_mask_gdf.crs != tiff_src.crs:
                            current_mask_gdf = current_mask_gdf.to_crs(tiff_src.crs)
                        
                        # 2. Read the TIFF's actual valid data mask (0 = NoData/NaN, 255 = Valid Pixel)
                        actual_tiff_mask = tiff_src.read_masks(1)
                        
                        # 3. Create an in-memory reference mask to test against. 
                        # This DOES NOT modify the TIFF file on S3. It uses the global polygon 
                        # to draw a temporary reference grid in RAM, matching the TIFF's exact 
                        # dimensions, to prove that the ETL pipeline correctly masked the actual file.
                        expected_geom_mask = rasterize(
                            shapes=current_mask_gdf.geometry,
                            out_shape=tiff_src.shape,
                            transform=tiff_src.transform,
                            fill=0,
                            default_value=255,
                            dtype='uint8'
                        )
                        
                        # 4. Compare the two masks to ensure the pipeline's masking worked perfectly
                        if not np.array_equal(actual_tiff_mask, expected_geom_mask):
                            total_pixels = actual_tiff_mask.size
                            matching_pixels = np.sum(actual_tiff_mask == expected_geom_mask)
                            match_percentage = (matching_pixels / total_pixels) * 100
                            
                            mismatches.append(
                                f"[{dataset_name}] '{name}' mask match failed. They align on {match_percentage:.3f}% of pixels."
                            )
            except Exception as e:
                mismatches.append(f"[{dataset_name}] '{name}' FAILED to process mask validation: {str(e)}")

    # Run for both prediction and training sets using their respective single masks
    check_masks(pred_tiffs, pred_mask_gdf, "Prediction")
    check_masks(train_tiffs, train_mask_gdf, "Training")

    if mismatches:
        logger.error(f"Found {len(mismatches)} files with NaN mask alignment issues.")
        for mismatch in mismatches:
            logger.error(mismatch)
            
    assert not mismatches, (
        f"Found {len(mismatches)} files with NaN mask alignment issues:\n" + 
        "\n".join(mismatches)
    )


def test_prediction_tiffs_exist_in_training(tiff_inventories):
    """
    Checks if every prediction TIFF has a file with a matching name in the training output.
    """
    pred_dict, train_dict = tiff_inventories
    
    logger.info(f"Found {len(pred_dict)} prediction TIFFs and {len(train_dict)} training TIFFs.")
    
    assert len(pred_dict) > 0, f"No prediction TIFFs found in {S3_PREDICTION_DIR}."
    
    missing_files = []
    for pred_name in pred_dict.keys():
        if pred_name not in train_dict:
            missing_files.append(pred_name)
            
    if missing_files:
        logger.error(f"Missing {len(missing_files)} files in training output: {missing_files}")
            
    assert not missing_files, (
        f"Found {len(missing_files)} prediction TIFFs missing from training output. "
        f"Missing files: {missing_files}"
    )


def test_matching_tiffs_have_same_dimensions_and_extents(tiff_inventories):
    """
    Looks at matching named TIFFs in both directories and ensures they have the 
    same number of rows and columns (shape) and identical bounding box extents.
    """
    pred_dict, train_dict = tiff_inventories
    
    # Get filenames that exist in both directories
    matching_names = set(pred_dict.keys()).intersection(set(train_dict.keys()))
    
    logger.info(f"Found {len(matching_names)} matching TIFFs. Checking dimensions and extents...")
    
    assert len(matching_names) > 0, "No matching TIFF names found between prediction and training directories."
    
    mismatches = []
    
    # Check the shape and bounds of each matched pair
    with rasterio.Env():
        for name in matching_names:
            pred_path = pred_dict[name]
            train_path = train_dict[name]
            
            with rasterio.open(pred_path) as p_src, rasterio.open(train_path) as t_src:
                # 1. Check dimensions (rows, cols)
                if p_src.shape != t_src.shape:
                    mismatches.append(
                        f"'{name}' SHAPE MISMATCH: Prediction {p_src.shape} != Training {t_src.shape}"
                    )
                
                # 2. Check bounding box extents (left, bottom, right, top)
                if p_src.bounds != t_src.bounds:
                    mismatches.append(
                        f"'{name}' BOUNDS MISMATCH: Prediction {p_src.bounds} != Training {t_src.bounds}"
                    )
                    
    if mismatches:
        logger.error(f"Found {len(mismatches)} files with mismatched dimensions or bounds.")
        for mismatch in mismatches:
            logger.error(mismatch)
                
    assert not mismatches, (
        f"Found {len(mismatches)} files with mismatched dimensions or bounds:\n" + 
        "\n".join(mismatches)
    )

def test_matching_parquets_have_same_dimensions_and_extents(parquet_inventories):
    """
    Looks at matching named GeoParquets in both directories and ensures they have the 
    same number of rows and columns (shape) and identical bounding box extents.
    """
    pred_dict, train_dict = parquet_inventories
    
    # Get filenames that exist in both directories
    matching_names = set(pred_dict.keys()).intersection(set(train_dict.keys()))
    
    logger.info(f"Found {len(matching_names)} matching GeoParquets. Checking dimensions and extents...")
    
    assert len(matching_names) > 0, "No matching GeoParquet names found between prediction and training directories."
    
    mismatches = []
    
    for name in matching_names:
        pred_path = pred_dict[name]
        train_path = train_dict[name]
        
        try:
            # Read both GeoParquets into GeoDataFrames
            p_gdf = gpd.read_parquet(pred_path)
            t_gdf = gpd.read_parquet(train_path)
            
            # 1. Check dimensions (rows, cols)
            if p_gdf.shape != t_gdf.shape:
                mismatches.append(
                    f"'{name}' SHAPE MISMATCH: Prediction {p_gdf.shape} != Training {t_gdf.shape}"
                )
            
            # 2. Check bounding box extents (minx, miny, maxx, maxy)
            # We use np.allclose instead of == to avoid false failures from microscopic floating point rounding differences
            if not np.allclose(p_gdf.total_bounds, t_gdf.total_bounds, atol=1e-6):
                mismatches.append(
                    f"'{name}' BOUNDS MISMATCH: Prediction {p_gdf.total_bounds} != Training {t_gdf.total_bounds}"
                )
                
        except Exception as e:
            mismatches.append(f"'{name}' FAILED TO LOAD OR PROCESS: {str(e)}")
                
    if mismatches:
        logger.error(f"Found {len(mismatches)} GeoParquet files with mismatched dimensions or bounds.")
        for mismatch in mismatches:
            logger.error(mismatch)
            
    assert not mismatches, (
        f"Found {len(mismatches)} GeoParquet files with mismatched dimensions or bounds:\n" + 
        "\n".join(mismatches)
    )

if __name__ == "__main__":
    # This block allows you to run the script directly with standard Python.
    # It will automatically pass the current file to pytest along with verbosity and console-output flags.
    pytest.main(["-v", "-s", __file__])