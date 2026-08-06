import os
import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box
from unittest.mock import patch, MagicMock

from hydro_health.engines.tiling.ModelDataPreProcessor import ModelDataPreProcessor

# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture
def mock_dependencies():
    """
    This fixture mocks out all the external dependencies that reach out to the 
    network, database, or config files so we can test the class logic in isolation.
    """
    with patch('hydro_health.engines.tiling.ModelDataPreProcessor.get_environment') as mock_get_env, \
         patch('hydro_health.engines.tiling.ModelDataPreProcessor.get_config_item') as mock_get_config, \
         patch('hydro_health.engines.tiling.ModelDataPreProcessor.s3fs.S3FileSystem') as mock_s3fs, \
         patch('hydro_health.engines.tiling.ModelDataPreProcessor.ModelDataPreProcessor._load_exclusion_config', return_value=set()):
        
        # Default mock returns
        mock_get_env.return_value = 'local'
        # Make get_config_item just return the key name so we can verify it easily
        mock_get_config.side_effect = lambda section, key: f"mock_{key.lower()}"
        
        yield {
            'get_env': mock_get_env,
            'get_config': mock_get_config,
            's3fs': mock_s3fs
        }

@pytest.fixture
def dummy_raster(tmp_path):
    """Creates a tiny 2x2 dummy raster (.tif) file for testing spatial extractions."""
    raster_path = tmp_path / "dummy_bathy_filled_2010.tif"
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    # Create a simple transform (Top Left X, Top Left Y, Pixel Width, Pixel Height)
    # This creates bounds: left=0, right=10, bottom=0, top=10
    transform = from_origin(0, 10, 5, 5) 
    
    with rasterio.open(
        str(raster_path), 'w', driver='GTiff',
        height=data.shape[0], width=data.shape[1],
        count=1, dtype=data.dtype,
        crs='EPSG:32617', transform=transform, nodata=-9999
    ) as dst:
        dst.write(data, 1)
        
    return raster_path

@pytest.fixture
def dummy_training_gdf():
    """Creates a dummy GeoDataFrame simulating a pre-transformation tile."""
    df = pd.DataFrame({
        'X': [1.0, 2.0],
        'Y': [1.0, 2.0],
        'tile_id': ['test_tile', 'test_tile'],
        'bathy_filled_2010': [10.5, 12.0],
        'bathy_filled_2015': [11.0, 11.5],
        '1998_2004_hurricane_force': [5, 6], # Forcing col
        'grain_size_layer': [0.5, 0.6]       # Static col
    })
    gdf = gpd.GeoDataFrame(df, geometry=[Point(1, 1), Point(2, 2)], crs="EPSG:32617")
    return gdf

@pytest.fixture
def dummy_prediction_gdf():
    """Creates a dummy GeoDataFrame simulating a pre-transformation prediction tile."""
    df = pd.DataFrame({
        'X': [1.0, 2.0],
        'Y': [1.0, 2.0],
        'tile_id': ['test_tile', 'test_tile'],
        'bt.bathy_filled_2010': [10.5, 12.0], # Has the bt. prefix
        'bt.tsm_mean_2010': [1.1, 1.2],
        '2010_2015_hurricane_force': [5, 6],  # Forcing col
        'grain_size_layer': [0.5, 0.6]        # Static col
    })
    gdf = gpd.GeoDataFrame(df, geometry=[Point(1, 1), Point(2, 2)], crs="EPSG:32617")
    return gdf


# ==========================================
# TESTS
# ==========================================

@pytest.mark.parametrize("env_name, expected_is_aws", [
    ('local', False),
    ('aws', True)
])
def test_init_aws_mode(mock_dependencies, env_name, expected_is_aws):
    """Test that the class correctly identifies if it is in AWS mode based on the environment."""
    mock_dependencies['get_env'].return_value = env_name
    
    processor = ModelDataPreProcessor()
    
    assert processor.is_aws is expected_is_aws
    # Ensure S3FileSystem was initialized
    mock_dependencies['s3fs'].assert_called_once_with(anon=False)

@pytest.mark.parametrize("env_name, expected_prefix", [
    ('local', ''),
    ('aws', 's3://mock_bucket_name/')
])
def test_create_file_paths(mock_dependencies, env_name, expected_prefix):
    """Test that UPaths are created correctly with or without the s3:// prefix."""
    mock_dependencies['get_env'].return_value = env_name
    
    # If we are in AWS mode, we need get_config_item to return a bucket name
    def custom_config(section, key):
        if key == 'BUCKET_NAME': return 'mock_bucket_name'
        return f"mock_{key.lower()}"
    
    mock_dependencies['get_config'].side_effect = custom_config
    
    processor = ModelDataPreProcessor()
    processor.create_file_paths()
    
    # Verify paths have the correct prefix
    assert str(processor.mask_prediction_pq).startswith(expected_prefix)
    assert str(processor.preprocessed_dir).startswith(expected_prefix)
    
    # Verify the specific path construction for one item
    expected_mask_path = f"{expected_prefix}mock_prediction_mask_pq"
    assert str(processor.mask_prediction_pq) == expected_mask_path

def test_transform_flowdir_cols_inplace(mock_dependencies):
    """Test the math transformation for flow directions (degrees to sin/cos)."""
    processor = ModelDataPreProcessor()
    
    # Create a dummy dataframe with flowdir angles (0, 90, 180 degrees)
    df = pd.DataFrame({
        'other_col': [1, 2, 3],
        'flowdir_1': [0, 90, 180],
        'bt.flowdir_2': [90, 180, 270]
    })
    
    processor._transform_flowdir_cols_inplace(df)
    
    # Ensure original flowdir cols are dropped
    assert 'flowdir_1' not in df.columns
    assert 'bt.flowdir_2' not in df.columns
    
    # Ensure other cols are untouched
    assert 'other_col' in df.columns
    
    # Verify math (using np.isclose due to floating point precision)
    # Sin of 0, 90, 180
    assert np.allclose(df['flowdir_1_sin'], [0.0, 1.0, 0.0], atol=1e-10)
    # Cos of 0, 90, 180
    assert np.allclose(df['flowdir_1_cos'], [1.0, 0.0, -1.0], atol=1e-10)

def test_get_column_metadata(mock_dependencies):
    """Test the regex parsing of column names to extract year and variable base."""
    processor = ModelDataPreProcessor()
    
    columns = [
        "bathy_filled_2010", 
        "tsm_mean_2022", 
        "ignore_this_column", 
        "bt.flowdir_1998"
    ]
    
    meta_df = processor._get_column_metadata(columns)
    
    # Should only process columns ending in _YYYY or just a year (based on your regex logic)
    assert len(meta_df) == 3
    
    # Check extraction for bathy_filled_2010
    bathy_row = meta_df[meta_df['colname'] == 'bathy_filled_2010'].iloc[0]
    assert bathy_row['year'] == 2010
    assert bathy_row['var_base'] == 'bathy_filled'
    
    # Check extraction for bt.flowdir_1998
    flow_row = meta_df[meta_df['colname'] == 'bt.flowdir_1998'].iloc[0]
    assert flow_row['year'] == 1998
    assert 'bt.flowdir' in flow_row['var_base']

def test_extract_raster_to_df(mock_dependencies, dummy_raster):
    """Test reading a raster window and converting valid pixels to a pandas DataFrame."""
    processor = ModelDataPreProcessor()
    
    # Extent covering the whole 10x10 dummy raster
    tile_extent = (0, 0, 10, 10)
    
    df = processor._extract_raster_to_df(str(dummy_raster), tile_extent)
    
    assert not df.empty
    assert list(df.columns) == ['X', 'Y', 'Value', 'Raster']
    assert len(df) == 4 # We created a 2x2 grid, so 4 valid pixels
    assert df['Raster'].iloc[0] == 'dummy_bathy_filled_2010'
    
    # Test an extent that does not overlap with the raster at all
    empty_extent = (20, 20, 30, 30)
    empty_df = processor._extract_raster_to_df(str(dummy_raster), empty_extent)
    assert empty_df.empty

def test_process_prediction_raster_intersection(mock_dependencies, dummy_raster):
    """Test that rasters only proceed to GDAL Warp if they intersect the mask."""
    processor = ModelDataPreProcessor()
    processor.target_crs = 'EPSG:32617'
    processor.target_res = 8
    
    # The dummy raster spans X: 0->10, Y: 0->10.
    # Create a mask that intersects this bounding box.
    intersecting_mask = box(5, 5, 15, 15)
    
    with patch.object(processor, '_warp_to_cutline') as mock_warp:
        processor.process_prediction_raster(str(dummy_raster), intersecting_mask, "mock_out.tif")
        mock_warp.assert_called_once()
        
    # Test a mask completely outside the dummy raster bounds
    non_intersecting_mask = box(20, 20, 30, 30)
    
    with patch.object(processor, '_warp_to_cutline') as mock_warp:
        processor.process_prediction_raster(str(dummy_raster), non_intersecting_mask, "mock_out.tif")
        mock_warp.assert_not_called()

def test_create_nan_stats_csv(mock_dependencies):
    """Test the calculation of NaN percentages for specific columns."""
    processor = ModelDataPreProcessor()
    
    df = pd.DataFrame({
        'other_col': [1, 2, 3, 4],
        'b.change.2010_2015': [1.0, np.nan, 3.0, np.nan], # 50% NaNs
        'b.change.2015_2020': [1.0, 2.0, 3.0, 4.0]        # 0% NaNs
    })
    
    stats_df = processor.create_nan_stats_csv(df, tile_id="tile_XYZ")
    
    assert not stats_df.empty
    assert stats_df['tile_id'].iloc[0] == "tile_XYZ"
    assert stats_df['2010_2015_nan_percent'].iloc[0] == 50.0
    assert stats_df['2015_2020_nan_percent'].iloc[0] == 0.0

def test_process_and_save_training_tile(mock_dependencies, dummy_training_gdf, tmp_path):
    """Test the wide-to-long DataFrame transformation specific to training data."""
    processor = ModelDataPreProcessor()
    
    # Temporarily override year ranges to match our dummy data
    processor.year_ranges = [(2010, 2015)] 
    
    output_dir = str(tmp_path)
    tile_name = "test_tile"
    
    # Run the processor
    saved_files = processor._process_and_save_training_tile(dummy_training_gdf, output_dir, tile_name)
    
    assert len(saved_files) == 1
    assert saved_files[0] == "test_tile_2010_2015_long.parquet"
    
    # Read the output and verify the transformation worked
    output_path = os.path.join(output_dir, saved_files[0])
    result_df = pd.read_parquet(output_path)
    
    # Verify the target / temporal renaming worked correctly
    assert 'bathy_t' in result_df.columns
    assert 'bathy_t1' in result_df.columns
    assert 'delta_bathy' in result_df.columns
    assert 'year_t' in result_df.columns
    
    # Verify delta bathy math was calculated accurately (t1 - t)
    # Row 1: 11.0 - 10.5 = 0.5. Row 2: 11.5 - 12.0 = -0.5
    assert result_df['delta_bathy'].iloc[0] == 0.5
    assert result_df['delta_bathy'].iloc[1] == -0.5
    
    # Ensure static vars persisted
    assert 'grain_size_layer' in result_df.columns

def test_process_and_save_prediction_tile(mock_dependencies, dummy_prediction_gdf, tmp_path):
    """Test the wide-to-long DataFrame transformation specific to prediction data (handling bt. prefixes)."""
    processor = ModelDataPreProcessor()
    
    # Temporarily override year ranges to match our dummy data
    processor.year_ranges = [(2010, 2015)] 
    
    output_dir = str(tmp_path)
    tile_name = "test_pred_tile"
    
    # Run the processor
    saved_files = processor._process_and_save_prediction_tile(dummy_prediction_gdf, output_dir, tile_name)
    
    assert len(saved_files) == 1
    assert saved_files[0] == "test_pred_tile_2010_2015_prediction_long.parquet"
    
    output_path = os.path.join(output_dir, saved_files[0])
    result_df = pd.read_parquet(output_path)
    
    # Verify the 'bt.' prefix was successfully removed and '_t' was appended
    assert 'bathy_filled_2010_t' in result_df.columns
    assert 'tsm_mean_2010_t' in result_df.columns
    
    # Ensure the old 'bt.' columns are gone
    assert 'bt.bathy_filled_2010' not in result_df.columns
    
    # Ensure forcing and static vars persisted
    assert '2010_2015_hurricane_force' in result_df.columns
    assert 'grain_size_layer' in result_df.columns

def test_save_combined_data(mock_dependencies, tmp_path):
    """Test merging gridded and ungridded dataframes and saving to parquet."""
    processor = ModelDataPreProcessor()
    processor.is_aws = False # Force local mode to avoid UPath S3 issues during mkdir
    
    gridded_df = pd.DataFrame({
        'X': [1.0, 2.0],
        'Y': [1.0, 2.0],
        'b.change.2010_2015': [1.5, np.nan] # Required for nan stats
    })
    
    ungridded_df = pd.DataFrame({
        'X': [1.0, 2.0],
        'Y': [1.0, 2.0],
        'static_var': [99, 100]
    })
    
    output_folder = tmp_path / "output_tiles"
    tile_id = "tile_001"
    
    # Run the delayed function synchronously for testing using .compute()
    from dask import compute
    task = processor.save_combined_data(gridded_df, ungridded_df, str(output_folder), "training", tile_id)
    stats_df = compute(task)[0]
    
    # 1. Verify Parquet file was created
    expected_file = output_folder / f"{tile_id}_training_clipped_data.parquet"
    assert expected_file.exists()
    
    # 2. Verify merge was successful
    saved_df = pd.read_parquet(expected_file)
    assert 'static_var' in saved_df.columns
    assert saved_df['static_var'].iloc[0] == 99
    
    # 3. Verify stats df was returned
    assert not stats_df.empty
    assert stats_df['tile_id'].iloc[0] == "tile_001"
    assert stats_df['2010_2015_nan_percent'].iloc[0] == 50.0

@patch('hydro_health.engines.tiling.ModelDataPreProcessor.rasterio.open')
@patch('hydro_health.engines.tiling.ModelDataPreProcessor.shapes')
def test_raster_to_spatial_df(mock_shapes, mock_rio_open, mock_dependencies, tmp_path):
    """Test converting a mask raster to a GeoDataFrame using mocked rasterio shapes."""
    processor = ModelDataPreProcessor()
    processor.is_aws = False
    
    # Mock the rasterio src object
    mock_src = MagicMock()
    mock_src.read.return_value = np.array([[1, 0], [1, 2]], dtype='uint8')
    mock_src.transform = MagicMock()
    mock_src.crs = "EPSG:4326"
    mock_rio_open.return_value.__enter__.return_value = mock_src
    
    # Mock the rasterio.features.shapes generator to return a dummy geometry
    # shape format: (geojson_dict, value)
    mock_shapes.return_value = [
        ({"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]]}, 1)
    ]
    
    # We need to mock get_config_item specifically for this test's MASKS_DIR call
    def custom_config(section, key):
        if key == 'MASKS_DIR': return str(tmp_path / 'masks')
        return f"mock_{key.lower()}"
    mock_dependencies['get_config'].side_effect = custom_config
    
    dummy_raster_path = tmp_path / "dummy_mask.tif"
    
    # Execute for prediction (looks for 1)
    gdf = processor.raster_to_spatial_df(dummy_raster_path, process_type='prediction')
    
    # Verify the GDF was created and reprojected
    assert not gdf.empty
    assert gdf.crs == "EPSG:32617"
    assert len(gdf) == 1
    
    # Verify the parquet file was saved
    expected_parquet = tmp_path / "masks" / "prediction_mask_pilot.parquet"
    assert expected_parquet.exists()

def test_load_exclusion_config_missing_file():
    """Test loading the exclusion config when the file doesn't exist."""
    # We bypass the mock_dependencies fixture here because it actively mocks out _load_exclusion_config
    with patch('hydro_health.engines.tiling.ModelDataPreProcessor.pathlib.Path.exists', return_value=False):
        with patch('hydro_health.engines.tiling.ModelDataPreProcessor.s3fs.S3FileSystem'), patch('hydro_health.engines.tiling.ModelDataPreProcessor.get_environment', return_value='local'):
            processor = ModelDataPreProcessor()
            # Should gracefully return an empty set if yaml is missing
            assert processor.excluded_keys == set()

@patch('hydro_health.engines.tiling.ModelDataPreProcessor.yaml.safe_load')
def test_load_exclusion_config_with_data(mock_yaml_load):
    """Test loading the exclusion config with valid YAML data."""
    mock_yaml_load.return_value = {
        'EcoRegion-3': {
            'dataset_A': {'use': False},
            'dataset_B': {'use': True},
            'dataset_C': {'use': False}
        }
    }
    
    with patch('hydro_health.engines.tiling.ModelDataPreProcessor.pathlib.Path.exists', return_value=True), \
         patch('builtins.open', new_callable=MagicMock), \
         patch('hydro_health.engines.tiling.ModelDataPreProcessor.s3fs.S3FileSystem'), \
         patch('hydro_health.engines.tiling.ModelDataPreProcessor.get_environment', return_value='local'):
        
        processor = ModelDataPreProcessor()
        
        # Should only grab keys where 'use' is False
        assert processor.excluded_keys == {'dataset_A', 'dataset_C'}

def test_process_training_raster_intersection(mock_dependencies, dummy_raster):
    """Test that training rasters check intersection and call warp with np.nan dst_nodata."""
    processor = ModelDataPreProcessor()
    processor.target_crs = 'EPSG:32617'
    processor.target_res = 8
    
    intersecting_mask = box(5, 5, 15, 15)
    
    with patch.object(processor, '_warp_to_cutline') as mock_warp:
        processor.process_training_raster(str(dummy_raster), intersecting_mask, "mock_out.tif")
        mock_warp.assert_called_once()
        
        # Verify training-specific kwarg (dst_nodata=np.nan) was passed
        kwargs = mock_warp.call_args.kwargs
        assert 'dst_nodata' in kwargs
        assert np.isnan(kwargs['dst_nodata'])

@patch('hydro_health.engines.tiling.ModelDataPreProcessor.gdal.Warp')
@patch('hydro_health.engines.tiling.ModelDataPreProcessor.gpd.GeoDataFrame.to_file')
def test_warp_to_cutline(mock_to_file, mock_warp, mock_dependencies):
    """Test that gdal.Warp is configured and called correctly."""
    processor = ModelDataPreProcessor()
    processor.is_aws = False
    
    mask_geom = box(0, 0, 10, 10)
    
    # Run the warp function
    processor._warp_to_cutline(
        src_path="dummy_in.tif",
        dst_path="dummy_out.tif",
        mask_geometry=mask_geom,
        dst_crs="EPSG:32617",
        x_res=8,
        y_res=8,
        crop_to_cutline=True
    )
    
    # Ensure a GeoJSON was temporarily generated for the cutline
    mock_to_file.assert_called_once()
    
    # Ensure gdal.Warp was called with the correct structural arguments
    mock_warp.assert_called_once()
    args, kwargs = mock_warp.call_args
    
    assert args[0] == "dummy_out.tif"  # Destination
    assert args[1] == "dummy_in.tif"   # Source
    assert kwargs['dstSRS'] == "EPSG:32617"
    assert kwargs['xRes'] == 8
    assert kwargs['cropToCutline'] is True
    assert 'COMPRESS=LZW' in kwargs['creationOptions']

@patch.object(ModelDataPreProcessor, '_extract_raster_to_df')
def test_subtile_process_gridded(mock_extract, mock_dependencies):
    """Test pivoting multiple raster arrays into a wide format DataFrame by coordinate."""
    processor = ModelDataPreProcessor()
    processor.target_crs = "EPSG:32617"
    
    # Setup mock sub_grid boundary series
    sub_grid = pd.Series({
        'original_tile': 'tile_A',
        'geometry': box(0, 0, 10, 10)
    })
    
    # Mock extract to simulate reading two different rasters for the same spatial extent
    mock_extract.side_effect = [
        pd.DataFrame({'X': [1, 2], 'Y': [1, 2], 'Value': [10.5, 20.5], 'Raster': 'bathy_filled'}),
        pd.DataFrame({'X': [1, 2], 'Y': [1, 2], 'Value': [0.1, 0.2], 'Raster': 'tsm_mean'})
    ]
    
    # Mock UPath rglob to return our two dummy rasters
    with patch('hydro_health.engines.tiling.ModelDataPreProcessor.UPath.rglob') as mock_rglob:
        mock_f1 = MagicMock(); mock_f1.suffix = '.tif'; mock_f1.name = 'tile_A_bathy.tif'
        mock_f2 = MagicMock(); mock_f2.suffix = '.tif'; mock_f2.name = 'tile_A_tsm.tif'
        mock_rglob.return_value = [mock_f1, mock_f2]
        
        result_gdf = processor.subtile_process_gridded(sub_grid, "dummy_dir")
        
        # Verify pivot logic happened correctly
        assert not result_gdf.empty
        assert isinstance(result_gdf, gpd.GeoDataFrame)
        assert 'bathy_filled' in result_gdf.columns
        assert 'tsm_mean' in result_gdf.columns
        assert result_gdf['bathy_filled'].iloc[0] == 10.5
        assert result_gdf['tsm_mean'].iloc[0] == 0.1

@patch('hydro_health.engines.tiling.ModelDataPreProcessor.gpd.sjoin')
@patch('hydro_health.engines.tiling.ModelDataPreProcessor.gpd.read_file')
@patch('hydro_health.engines.tiling.ModelDataPreProcessor.gpd.read_parquet')
def test_create_subgrids(mock_read_parquet, mock_read_file, mock_sjoin, mock_dependencies):
    """Test generating a spatial join between the grid and a unioned mask."""
    processor = ModelDataPreProcessor()
    processor.is_aws = False
    
    # Mock the geometries
    mock_mask_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(10, 10, 20, 20)], crs="EPSG:32617")
    mock_grid_gdf = gpd.GeoDataFrame(geometry=[box(0, 0, 5, 5)], crs="EPSG:32617")
    
    # Mock the file reads
    mock_read_parquet.return_value = mock_mask_gdf
    mock_read_file.return_value = mock_grid_gdf
    
    # Mock the result of the sjoin
    mock_sjoin_result = gpd.GeoDataFrame(geometry=[box(0, 0, 5, 5)], crs="EPSG:32617")
    mock_sjoin_result.to_file = MagicMock()
    mock_sjoin.return_value = mock_sjoin_result
    
    processor.create_subgrids("dummy_mask.parquet", "dummy_out.gpkg", "prediction")
    
    # Verify sjoin was called
    mock_sjoin.assert_called_once()
    
    # Verify the results were written to the output file
    mock_sjoin_result.to_file.assert_called_once_with("dummy_out.gpkg", driver="GPKG")

@patch.object(ModelDataPreProcessor, '_extract_raster_to_df')
def test_subtile_process_ungridded(mock_extract, mock_dependencies):
    """Test extracting ungridded (static) rasters and pivoting them."""
    processor = ModelDataPreProcessor()
    
    # Override static patterns to simplify the test
    processor.static_patterns = ['tsm_mean', 'hurr']
    
    # Setup mock sub_grid boundary series
    sub_grid = pd.Series({'geometry': box(0, 0, 10, 10)})
    
    # Mock extract to simulate reading the two different static rasters
    mock_extract.side_effect = [
        pd.DataFrame({'X': [1, 2], 'Y': [1, 2], 'Value': [0.5, 0.6], 'Raster': 'tsm_mean_raster'}),
        pd.DataFrame({'X': [1, 2], 'Y': [1, 2], 'Value': [5, 6], 'Raster': 'hurr_raster'})
    ]
    
    with patch('hydro_health.engines.tiling.ModelDataPreProcessor.UPath.rglob') as mock_rglob:
        # rglob is called once per pattern in the loop, so we return lists of fake files accordingly
        mock_rglob.side_effect = [
            [MagicMock(suffix='.tif', name='tsm_mean_raster.tif')],
            [MagicMock(suffix='.tif', name='hurr_raster.tif')]
        ]
        
        result_df = processor.subtile_process_ungridded(sub_grid, "dummy_dir")
        
        # Verify pivot logic happened correctly for the ungridded data
        assert not result_df.empty
        assert 'tsm_mean_raster' in result_df.columns
        assert 'hurr_raster' in result_df.columns
        assert result_df['tsm_mean_raster'].iloc[0] == 0.5
        assert result_df['hurr_raster'].iloc[0] == 5.0

@patch('hydro_health.engines.tiling.ModelDataPreProcessor.gpd.read_parquet')
@patch.object(ModelDataPreProcessor, '_process_and_save_training_tile')
@patch.object(ModelDataPreProcessor, '_process_and_save_prediction_tile')
def test_transform_tile_task_routing(mock_pred, mock_train, mock_read_pq, mock_dependencies):
    """Test that the Dask worker wrapper properly routes to training or prediction based on mode."""
    processor = ModelDataPreProcessor()
    
    # Mock reading the dataframe so it doesn't look for a real file
    mock_read_pq.return_value = MagicMock()
    
    # 1. Test Training Mode
    mock_train.return_value = ["file1.parquet", "file2.parquet"]  # Returns a list of 2 saved files
    
    res_training = processor._transform_tile_task("fake_dir/tileA_clipped_data.parquet", mode="training")
    
    mock_train.assert_called_once()
    mock_pred.assert_not_called()
    assert "Success: tileA" in res_training
    assert "(2 pairs)" in res_training
    
    # 2. Test Prediction Mode
    mock_train.reset_mock()
    mock_pred.return_value = ["file3.parquet"] # Returns 1 saved file
    
    res_prediction = processor._transform_tile_task("fake_dir/tileB_clipped_data.parquet", mode="prediction")
    
    mock_pred.assert_called_once()
    mock_train.assert_not_called()
    assert "Success: tileB" in res_prediction
    assert "(1 pairs)" in res_prediction
    
    # 3. Test Failure Mode (Exception caught)
    mock_pred.side_effect = Exception("Mocked processing error")
    res_failure = processor._transform_tile_task("fake_dir/tileC_clipped_data.parquet", mode="prediction")
    
    assert "Failed: tileC_clipped_data.parquet" in res_failure
    assert "Mocked processing error" in res_failure