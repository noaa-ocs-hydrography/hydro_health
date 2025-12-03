import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from typing import Tuple

# --- Raster Alignment Functions ---

def check_raster_alignment(path1: str, path2: str) -> bool:
    """Checks if two GeoTIFFs have the same extent, CRS, and shape.
    param str path1: File path to the first GeoTIFF.
    param str path2: File path to the second GeoTIFF.
    return: True if extent, CRS, and shape match, False otherwise.
    """
    with rasterio.open(path1) as ds1:
        with rasterio.open(path2) as ds2:

            is_aligned = True

            if ds1.bounds != ds2.bounds:
                print(f"Error: Extents do not match.")
                is_aligned = False

            if ds1.crs != ds2.crs:
                print(f"Error: Coordinate Reference Systems do not match.")
                is_aligned = False

            if ds1.shape != ds2.shape:
                print(f"Error: Raster dimensions (shape) do not match.")
                is_aligned = False

            return is_aligned

def compare_cell_values(path1: str, path2: str) -> Tuple[np.ndarray, float]:
    """Calculates cell-by-cell difference and percentage of non-matching cells.
    param str path1: File path to the first GeoTIFF.
    param str path2: File path to the second GeoTIFF.
    return: A tuple (diff_array, percent_mismatch).
             diff_array: 2D numpy array (float) of (raster1 - raster2).
             percent_mismatch: float percentage of differing cells.
    """
    with rasterio.open(path1) as ds1:
        with rasterio.open(path2) as ds2:

            data1 = ds1.read(1).astype(float)
            data2 = ds2.read(1).astype(float)

            ndv1 = ds1.nodata
            ndv2 = ds2.nodata

            if ndv1 is not None:
                mask1 = (data1 == ndv1)
            else:
                mask1 = np.zeros_like(data1, dtype=bool)

            if ndv2 is not None:
                mask2 = (data2 == ndv2)
            else:
                mask2 = np.zeros_like(data2, dtype=bool)

            nan_mask1 = np.isnan(data1)
            nan_mask2 = np.isnan(data2)

            combined_nodata_mask = mask1 | mask2 | nan_mask1 | nan_mask2

            valid_mask = ~combined_nodata_mask

            total_valid_cells = np.count_nonzero(valid_mask)

            if total_valid_cells == 0:
                print("Warning: No valid cells (all cells are nodata or nan).")
                return (np.full_like(data1, np.nan), 0.0)

            data1_valid = data1[valid_mask]
            data2_valid = data2[valid_mask]

            mismatch_count = np.count_nonzero(~np.isclose(data1_valid, data2_valid))

            percent_mismatch = (mismatch_count / total_valid_cells) * 100

            diff_array = data1 - data2

            diff_array[combined_nodata_mask] = np.nan

            return (diff_array, percent_mismatch)

# --- GeoParquet Alignment Functions ---

def check_parquet_alignment(path1: str, path2: str, value_col: str) -> bool:
    """Checks if two GeoParquet files have the same X/Y coordinates and a specific value column.
    
    This function prioritizes comparison of 'X' and 'Y' columns, assuming they represent 
    the point coordinates, and compares the values in the target value_col.

    param str path1: File path to the first GeoParquet file.
    param str path2: File path to the second GeoParquet file.
    param str value_col: The column name whose values will be compared (e.g., 'sed_type_raster_100m').
    return: True if coordinates and values match (within tolerance), False otherwise.
    """
    cols_to_load = ['X', 'Y', value_col]
    
    try:
        df1 = pd.read_parquet(path1, columns=cols_to_load)
        df2 = pd.read_parquet(path2, columns=cols_to_load)
    except KeyError as e:
        print(f"Error: One or both GeoParquet files is missing required column(s): {e}")
        return False
    
    is_aligned = True
        
    if len(df1) != len(df2):
        print(f"Error: Number of records do not match ({len(df1)} vs {len(df2)}).")
        return False

    sort_cols = ['X', 'Y']
    df1 = df1.sort_values(by=sort_cols, ignore_index=True)
    df2 = df2.sort_values(by=sort_cols, ignore_index=True)

    # Coordinate comparison (alignment check)
    x_match = np.allclose(df1['X'], df2['X'])
    y_match = np.allclose(df1['Y'], df2['Y'])

    if not x_match or not y_match:
        print(f"Error: X/Y coordinates do not match after sorting.")
        is_aligned = False

    # Value comparison
    v1 = df1[value_col].to_numpy()
    v2 = df2[value_col].to_numpy()
    
    total_valid_cells = len(v1)
    
    mismatch_count = np.count_nonzero(~np.isclose(v1, v2))
    
    percent_mismatch = (mismatch_count / total_valid_cells) * 100
    percent_match = 100 - percent_mismatch
    
    print(f"\n--- GeoParquet Comparison Results for '{value_col}' ---")
    print(f"Percentage of non-matching values (by coordinate): {percent_match:.2f}%")

    if percent_mismatch > 0:
        diff_vals = v1 - v2
        min_diff = np.nanmin(diff_vals)
        max_diff = np.nanmax(diff_vals)
        mean_diff = np.nanmean(diff_vals)
        print(f"  Min difference: {min_diff}")
        print(f"  Max difference: {max_diff}")
        print(f"  Mean difference: {mean_diff}")

    return is_aligned and (percent_mismatch == 0.0)

# --- Main Execution ---

def main():
    """Main execution function to compare two geospatial files."""

    # Set file paths and the column to compare for GeoParquet files
    file_path_1 = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Training\training_tiles\BH4SD56H_1\BH4SD56H_1_training_clipped_data.parquet"
    file_path_2 = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\prediction_tiles\BH4SD56H_1\BH4SD56H_1_prediction_clipped_data.parquet"
    parquet_value_column = "sed_size_raster_100m" # Column to compare if files are parquet

    ext1 = os.path.splitext(file_path_1)[1].lower()
    ext2 = os.path.splitext(file_path_2)[1].lower()
    
    print(f"Comparing '{file_path_1}' and '{file_path_2}'...")
    print("---")

    if ext1 == '.tiff' and ext2 == '.tiff':
        
        # Raster Comparison Pipeline
        print("Detected GeoTIFF files for comparison.")
        print("Checking alignment (extent, CRS, shape)...")
        if not check_raster_alignment(file_path_1, file_path_2):
            print("\nAnalysis stopped: Rasters are not aligned.")
            return

        print("Rasters are aligned.")
        print("---")

        print("Calculating cell-by-cell differences...")
        diff_array, percent_mismatch = compare_cell_values(file_path_1, file_path_2)

        print(f"\n--- Comparison Results ---")
        print(f"Percentage of non-matching cells: {percent_mismatch:.6f}%")

        if percent_mismatch > 0:
            min_diff = np.nanmin(diff_array)
            max_diff = np.nanmax(diff_array)
            mean_diff = np.nanmean(diff_array)

            print(f"  Min difference: {min_diff}")
            print(f"  Max difference: {max_diff}")
            print(f"  Mean difference: {mean_diff}")

    elif ext1 == '.parquet' and ext2 == '.parquet':
        
        # GeoParquet Comparison Pipeline
        print(f"Detected GeoParquet files for comparison on column '{parquet_value_column}'.")
        print("Checking alignment (X/Y coordinates) and value match...")
        
        if not check_parquet_alignment(file_path_1, file_path_2, parquet_value_column):
            print("\nAnalysis stopped: GeoParquet files are not aligned or values do not match.")
            return

        print("GeoParquet files are aligned and values match (within tolerance).")

    else:
        print(f"Error: File types '{ext1}' and '{ext2}' are not both .tiff or both .parquet.")


if __name__ == "__main__":
    main()