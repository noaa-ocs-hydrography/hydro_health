import pytest
import rasterio
import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

# =====================================================================
# 1. PLACEHOLDERS: Fill these in with your S3 paths and expected values
# =====================================================================

S3_GEOPARQUET_PATH = "s3://ocs-dev-csdl-hydrohealth/ER_3/masks/prediction_mask_ecoregion.parquet"
S3_TIFF_PATH = "s3://ocs-dev-csdl-hydrohealth/ER_3/rasters/output_image.tif"

EXPECTED_ROWS = 1500           # Expected number of polygons/rows in the parquet
EXPECTED_COLS = 12             # Expected number of attribute columns in the parquet
EXPECTED_CRS = "EPSG:32617"     # Expected Coordinate Reference System (e.g., EPSG:4326, EPSG:32630)
EXPECTED_RESOLUTION = (8, 8) # Expected pixel resolution (x, y) in map units, this should be meters

# =====================================================================
# 2. FIXTURES: Load data once per test session to save time and network
# =====================================================================

@pytest.fixture(scope="module")
def mask_gdf():
    """Reads the GeoParquet mask file from S3."""
    # Note: Requires 's3fs' to be installed to read 's3://' paths directly
    gdf = gpd.read_parquet(S3_GEOPARQUET_PATH)
    return gdf

@pytest.fixture(scope="module")
def tiff_src():
    """
    Opens the GeoTIFF from S3. 
    Yields the rasterio dataset object, ensuring it safely closes after tests.
    """
    # rasterio.Env() activates GDAL's cloud-optimized reading capabilities
    with rasterio.Env():
        with rasterio.open(S3_TIFF_PATH) as src:
            yield src

# =====================================================================
# 3. THE TESTS
# =====================================================================

def test_geoparquet_schema(mask_gdf):
    """Checks if the GeoParquet file has the correct number of rows and columns."""
    actual_rows, actual_cols = mask_gdf.shape
    
    assert actual_rows == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, but got {actual_rows}."
    assert actual_cols == EXPECTED_COLS, f"Expected {EXPECTED_COLS} columns, but got {actual_cols}."


def test_tiff_metadata(tiff_src):
    """Checks if the TIFF has the exact CRS and resolution expected."""
    # Test CRS
    actual_crs = tiff_src.crs.to_string() if tiff_src.crs else "None"
    assert actual_crs == EXPECTED_CRS, f"CRS mismatch. Expected {EXPECTED_CRS}, got {actual_crs}."
    
    # Test Resolution
    actual_res = tiff_src.res
    assert actual_res == EXPECTED_RESOLUTION, f"Resolution mismatch. Expected {EXPECTED_RESOLUTION}, got {actual_res}."


def test_tiff_nan_mask_matches_parquet_extent(mask_gdf, tiff_src):
    """
    Validates that the valid pixels in the TIFF exactly match the polygons 
    in the GeoParquet mask, and NaN/NoData pixels fall outside it.
    """
    # 1. Ensure the vector mask is in the same projection as the TIFF
    if mask_gdf.crs != tiff_src.crs:
        mask_gdf = mask_gdf.to_crs(tiff_src.crs)
    
    # 2. Read the TIFF's valid data mask (0 = NoData/NaN, 255 = Valid Pixel)
    # read_masks(1) gets the mask for the first band.
    actual_tiff_mask = tiff_src.read_masks(1)
    
    # 3. Create a simulated mask by rasterizing the GeoParquet polygons into the TIFF's grid.
    # We burn '255' into the pixels inside the polygons, and '0' for the background.
    expected_geom_mask = rasterize(
        shapes=mask_gdf.geometry,
        out_shape=tiff_src.shape,
        transform=tiff_src.transform,
        fill=0,
        default_value=255,
        dtype='uint8'
    )
    
    # 4. Compare the two masks
    # Strict matching: Are the arrays 100% identical?
    is_exact_match = np.array_equal(actual_tiff_mask, expected_geom_mask)
    
    if not is_exact_match:
        # If it fails, let's calculate the percentage of overlap to give a helpful error message.
        # Vector rasterization sometimes differs from clipping tools by 1 pixel on the borders
        # due to sub-pixel boundary logic (e.g. "all_touched").
        total_pixels = actual_tiff_mask.size
        matching_pixels = np.sum(actual_tiff_mask == expected_geom_mask)
        match_percentage = (matching_pixels / total_pixels) * 100
        
        # You can change this test to allow a 99.9% match if exact pixel boundary math is slightly fuzzy:
        # assert match_percentage > 99.9, f"Masks only match {match_percentage:.3f}%"
        
        pytest.fail(
            f"The TIFF's NaN mask does not exactly match the Parquet extent. "
            f"They match on {match_percentage:.3f}% of pixels."
        )