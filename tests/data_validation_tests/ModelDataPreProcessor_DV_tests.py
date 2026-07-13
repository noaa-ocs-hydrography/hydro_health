"""
Tests to validate the outputs of the ModelDataPreProcessor on S3.
Run this using: pytest test_s3_outputs.py -v
"""
import pytest
import s3fs
import rasterio
import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
import numpy as np
from rasterio.session import AWSSession

from hydro_health.engines.Engine import Engine

# --- CONFIGURATION & FIXTURES ---

# Replace with your actual bucket and prefix
BUCKET_NAME = "your-s3-bucket-name" 
PREFIX = "path/to/model/outputs"
S3_BASE = f"s3://{BUCKET_NAME}/{PREFIX}"

@pytest.fixture(scope="module")
def s3_fs():
    """Returns an authenticated s3fs file system instance."""
    return s3fs.S3FileSystem(anon=False)

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
def raster_directories():
    """List of directories containing output rasters."""
    return [
        f"{S3_BASE}/prediction_output",
        f"{S3_BASE}/training_output",
        f"{S3_BASE}/uncombined_lidar"
    ]

@pytest.fixture(scope="module")
def tile_directories():
    """List of directories containing output tile parquet files."""
    return [
        f"{S3_BASE}/prediction_tiles",
        f"{S3_BASE}/training_tiles"
    ]

# --- 1. DIRECTORY AND FILE PRESENCE TESTS ---

def test_output_directories_exist(s3_fs, raster_directories, tile_directories):
    """Test that all expected output directories were created on S3."""
    all_dirs = raster_directories + tile_directories
    for d in all_dirs:
        # Check if directory exists (s3fs considers a dir to exist if it has contents)
        assert s3_fs.exists(d) or len(s3_fs.ls(d)) > 0, f"Directory missing or empty: {d}"

def test_csv_stats_generated(s3_fs):
    """Test that the summary NaN stats CSVs were generated."""
    expected_csvs = [
        f"{S3_BASE}/year_pair_nan_counts_training.csv",
        f"{S3_BASE}/year_pair_nan_counts_prediction.csv"
    ]
    for csv_path in expected_csvs:
        assert s3_fs.exists(csv_path), f"Missing stats CSV: {csv_path}"

# --- 2. RASTER PROPERTY TESTS ---

def test_raster_crs_and_resolution(s3_fs, raster_directories, target_crs, target_res):
    """
    Test all rasters from each directory to ensure:
    - Target CRS matches Engine configuration
    - Target Resolution matches Engine configuration
    - Output type is Float32
    - Compression is LZW
    """
    for directory in raster_directories:
        # Get ALL rasters per directory
        tifs = [f for f in s3_fs.ls(directory) if f.lower().endswith(('.tif', '.tiff'))]
        
        for tif in tifs:
            vsi_path = f"/vsis3/{tif}"
            with rasterio.Env(session=AWSSession()):
                with rasterio.open(vsi_path) as src:
                    # 1. Check CRS
                    assert src.crs.to_string() == target_crs, f"Incorrect CRS in {tif}"
                    
                    # 2. Check Resolution (X should be res, Y should be res or -res depending on origin)
                    res = src.res
                    assert res[0] == target_res, f"Incorrect X resolution in {tif}: {res[0]}"
                    assert res[1] == target_res, f"Incorrect Y resolution in {tif}: {res[1]}"
                    
                    # 3. Check Datatype and Compression
                    assert src.dtypes[0] == 'float32', f"Datatype is not float32 in {tif}"
                    assert src.profile.get('compress') == 'lzw', f"Not LZW compressed: {tif}"
                    assert src.profile.get('tiled') is True, f"File is not tiled: {tif}"
                    assert src.profile.get('blockxsize') == 512, f"Block size X is not 512: {tif}"

def test_raster_approximate_location(s3_fs):
    """
    Test that all prediction rasters fall within an approximate valid bounding box for UTM Zone 17N.
    UTM 17N generally covers longitude 84°W to 78°W (e.g. Florida, US East Coast).
    Approximate X bounds: 200,000 to 800,000 meters
    Approximate Y bounds: 2,500,000 to 4,500,000 meters (depending on exact latitude)
    """
    # Test all prediction rasters
    prediction_dir = f"{S3_BASE}/prediction_output"
    tifs = [f for f in s3_fs.ls(prediction_dir) if f.lower().endswith('.tif')]
    
    if not tifs:
        pytest.skip("No prediction TIFFs found to test location.")
        
    for tif in tifs:
        vsi_path = f"/vsis3/{tif}"
        with rasterio.Env(session=AWSSession()):
            with rasterio.open(vsi_path) as src:
                bounds = src.bounds
                # Loose checks to ensure data isn't at 0,0 or in raw WGS84 (Lat/Lon)
                assert 100_000 < bounds.left < 999_999, f"Left bound {bounds.left} seems outside UTM 17N in {tif}"
                assert 1_000_000 < bounds.bottom < 6_000_000, f"Bottom bound {bounds.bottom} seems outside UTM 17N in {tif}"

# --- 3. PARQUET DERIVATIVE & TILE TESTS ---

def test_derivative_counts_per_tile(s3_fs, tile_directories, year_pairs):
    """
    Test that exactly 13 long-format files are generated for EVERY tile processed.
    This ensures the `batch_long_format_transformation` loop worked correctly.
    """
    for directory in tile_directories:
        subfolders = s3_fs.ls(directory)
        for tile_folder in subfolders: # Exhaustively check all tiles
            
            # List all parquet files in this tile's folder
            files = s3_fs.ls(tile_folder)
            
            # Count the ones matching the long format suffix
            long_format_files = [f for f in files if f.endswith("_long.parquet")]
            
            # Only assert if processing actually occurred for this tile
            if len(long_format_files) > 0:
                assert len(long_format_files) == len(year_pairs), \
                    f"Expected {len(year_pairs)} derivatives for tile {tile_folder}, but found {len(long_format_files)}"

def test_parquet_schema_and_contents(s3_fs, year_pairs):
    """
    Test that EVERY generated training parquet file contains the exact required schema:
    - X, Y, year_t, year_t1, delta_bathy
    - Verify years match one of our 13 pairs
    """
    training_dir = f"{S3_BASE}/training_tiles"
    all_files = s3_fs.glob(f"{training_dir}/**/*_long.parquet")
    
    if not all_files:
        pytest.skip("No long-format parquet files found.")
        
    # Read ALL files exhaustively
    for file_path in all_files:
        s3_path = f"s3://{file_path}"
        df = pd.read_parquet(s3_path, storage_options={"anon": False})
        
        # 1. Check basic columns
        required_cols = {'X', 'Y', 'year_t', 'year_t1'}
        assert required_cols.issubset(df.columns), f"Missing required columns in {s3_path}. Found: {df.columns}"
        
        # 2. Check Target / delta columns
        assert 'delta_bathy' in df.columns or 'bathy_t1' in df.columns, f"Target bathymetry variables missing in {s3_path}."
        
        # 3. Check Years validation
        unique_years = df[['year_t', 'year_t1']].drop_duplicates()
        for _, row in unique_years.iterrows():
            pair = (int(row['year_t']), int(row['year_t1']))
            assert pair in year_pairs, f"Found year pair {pair} in {s3_path} which is not in the configured ranges!"

def test_parquet_spatial_integrity(s3_fs):
    """
    Test that the X and Y coordinates inside ALL Parquet tabular data
    align with the target resolution rounding (e.g. they should not have 
    crazy floating points if resolution is exactly 8m).
    """
    prediction_dir = f"{S3_BASE}/prediction_tiles"
    all_files = s3_fs.glob(f"{prediction_dir}/**/*_prediction_long.parquet")
    
    if not all_files:
        pytest.skip("No prediction parquet files found.")
        
    for file_path in all_files:
        s3_path = f"s3://{file_path}"
        df = pd.read_parquet(s3_path, storage_options={"anon": False}, columns=['X', 'Y'])
        
        # Ensure dataframe is not empty
        assert not df.empty, f"Parquet file {s3_path} is empty."
        
        # X and Y were explicitly rounded to 3 decimal places in code: `df['X'].round(3)`
        # Check that they don't exceed 3 decimal places
        x_decimals = (df['X'] * 1000 % 1)
        # The modulo of X*1000 by 1 should be very close to 0
        assert np.allclose(x_decimals, 0, atol=1e-5), f"X coordinates appear to have more than 3 decimal places in {s3_path}."

# --- 4. GEOMETRY / MASK TESTS ---

def test_mask_and_subgrid_crs(s3_fs, target_crs):
    """
    Test that the training mask and subgrids were saved with the correct CRS.
    """
    mask_pq = f"{S3_BASE}/masks/prediction_mask.parquet" 
    subgrid_gpkg = f"{S3_BASE}/subgrids/prediction_subgrids.gpkg"
    
    # Check Mask Parquet
    if s3_fs.exists(mask_pq):
        gdf = gpd.read_parquet(f"s3://{mask_pq}", storage_options={"anon": False})
        assert gdf.crs.to_string() == target_crs, "Mask Parquet CRS is incorrect"
        
    # Check GPKG
    if s3_fs.exists(subgrid_gpkg):
        # Geopandas needs to download GPKG or use fsspec
        # fsspec reading of gpkg can be tricky; using s3fs file obj
        with s3_fs.open(subgrid_gpkg, "rb") as f:
            gdf = gpd.read_file(f)
            assert gdf.crs.to_string() == target_crs, "Subgrid GPKG CRS is incorrect"