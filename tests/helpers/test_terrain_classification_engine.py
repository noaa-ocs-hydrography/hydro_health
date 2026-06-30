import os
import sys

# --- FIX: Add the project's 'src' directory to the python path ---
# This ensures pytest can find and import the 'hydro_health' package
# which is located at .../hydro_health/src/hydro_health/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "../../src"))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from types import ModuleType

# Mock 'whitebox' if it's not installed in the test environment 
# to prevent tests from crashing on import.
if "whitebox" not in sys.modules:
    mock_wbt = ModuleType("whitebox")
    mock_wbt.WhiteboxTools = MagicMock()
    sys.modules["whitebox"] = mock_wbt

# --------------------------------------------------------

@pytest.fixture
def engine():
    """Fixture to initialize the engine with mocked external dependencies."""
    # We patch the environment and filesystem during initialization
    with patch('hydro_health.helpers.tools.get_environment', return_value='local'), \
         patch('s3fs.S3FileSystem'), \
         patch('whitebox.WhiteboxTools'):
        
        from hydro_health.engines.CreateSeabedTerrainLayerEngine import CreateSeabedTerrainLayerEngine
        
        engine_instance = CreateSeabedTerrainLayerEngine()
        
        yield engine_instance


# ==============================================================================
#  TESTS FOR STRING / REGEX PARSING LOGIC
# ==============================================================================

def test_get_tile_id(engine):
    """Test extracting 8-character alphanumeric Tile ID."""
    assert engine._get_tile_id("bathy_BH4SD56H_2020.tif") == "BH4SD56H"
    assert engine._get_tile_id("bluetopo_12345678_2020.tif") is None  # all digits
    assert engine._get_tile_id("bluetopo_Z2345678_2020.tif") is None  # doesnt start with 'B'

def test_get_variable_type(engine):
    """Test target variable matching."""
    engine.target_vars = ["slope", "rugosity"]
    assert engine._get_variable_type("bt_slope_2020.tif") == "slope"
    assert engine._get_variable_type("bt_bathy_2020.tif") is None

def test_get_year(engine):
    """Test year extraction from filename."""
    assert engine._get_year("bathy_BH4SD56H_2015.tif") == 2015
    assert engine._get_year("bathy_BH4SD56H_1999.tif") == 1999
    assert engine._get_year("bathy_BH4SD56H_nodate.tif") is None


# ==============================================================================
#  TESTS FOR PUBLIC MATH / LOGIC FUNCTIONS
# ==============================================================================

def test_calculate_bpi(engine):
    """Test Bathymetric Position Index calculation."""
    arr = np.ones((100, 100))
    arr[5, 5] = 10  # A spike
    bpi = engine.calculate_bpi(arr, cell_size=1, inner_radius=8, outer_radius=32)
    assert bpi.shape == (100, 100)
    assert bpi[5, 5] > 0  # Spike should have positive BPI

def test_calculate_slope_and_tri(engine):
    """Test slope and TRI calculation."""
    arr = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
    slope, tri = engine.calculate_slope_and_tri(arr, cell_size=1)
    assert slope.shape == (3, 3)
    assert tri.shape == (3, 3)
    assert np.nanmean(slope) > 0

def test_create_classification_dictionary(engine):
    """Test dataframe creation for classification rules."""
    broad = np.array([0.1, 0.5, 0.9])
    fine = np.array([0.2, 0.6, 0.8])
    slope = np.array([1.0, 5.0, 10.0])
    
    df = engine.create_classification_dictionary(broad, fine, slope)
    assert isinstance(df, pd.DataFrame)
    assert 'Class_ID' in df.columns
    assert len(df) == 8

def test_focal_fill_block(engine):
    """Test NaN focal filling on blocks."""
    arr = np.array([
        [1.0, 1.0, 1.0],
        [1.0, np.nan, 1.0],
        [1.0, 1.0, 1.0]
    ])
    filled = engine.focal_fill_block(arr, w=3)
    assert not np.isnan(filled[1, 1])
    assert filled[1, 1] == 1.0

def test_standardize_raster_array(engine):
    """Test zero-mean unit-variance standardization."""
    arr = np.array([[1, 2], [3, 4]], dtype=float)
    std_arr = engine.standardize_raster_array(arr)
    assert np.isclose(np.mean(std_arr), 0.0)
    assert np.isclose(np.std(std_arr), 1.0)