import rasterio
import numpy as np
import pandas as pd
import os
import sys

# --- RASTER FUNCTIONS ---

def compare_rasters(path1: str, path2: str):
    """
    Performs a two-stage check on GeoTIFFs:
    1. Grid Alignment (CRS, Bounds, Shape)
    2. Cell Values (Pixel-by-pixel comparison)
    """
    print(f"\n--- Comparing Rasters ---")
    print(f"File 1: {os.path.basename(path1)}")
    print(f"File 2: {os.path.basename(path2)}")

    with rasterio.open(path1) as ds1, rasterio.open(path2) as ds2:
        
        # --- STAGE 1: GRID ALIGNMENT ---
        print("\n[Stage 1] Checking Grid Alignment...")
        
        # Check 1: CRS
        if ds1.crs != ds2.crs:
            print(f"Error: CRS mismatch.\n   F1: {ds1.crs}\n   F2: {ds2.crs}")
            return
        
        # Check 2: Dimensions (Shape)
        if ds1.shape != ds2.shape:
            print(f"Error: Shape mismatch (Height/Width).\n   F1: {ds1.shape}\n   F2: {ds2.shape}")
            return
        
        # Check 3: Borders (Bounds) - Rounded to 6 decimals to allow for tiny float diffs
        b1 = [round(x, 6) for x in ds1.bounds]
        b2 = [round(x, 6) for x in ds2.bounds]
        if b1 != b2:
            print(f"Error: Spatial Bounds (Borders) do not match.")
            print(f"   F1: {b1}")
            print(f"   F2: {b2}")
            return
            
        print("Success: Grids align perfectly (CRS, Shape, and Bounds match).")

        # --- STAGE 2: VALUE COMPARISON ---
        print("\n[Stage 2] Checking Cell Values...")
        
        data1 = ds1.read(1).astype(float)
        data2 = ds2.read(1).astype(float)
        
        # Standardize NoData to np.nan for comparison
        if ds1.nodata is not None:
            data1[data1 == ds1.nodata] = np.nan
        if ds2.nodata is not None:
            data2[data2 == ds2.nodata] = np.nan

        # Compare values (equal_nan=True allows NaN==NaN to pass)
        # We use a small tolerance (atol=1e-5) for floating point numbers
        mismatch_mask = ~np.isclose(data1, data2, equal_nan=True, atol=1e-5)
        mismatch_count = np.count_nonzero(mismatch_mask)
        
        if mismatch_count == 0:
            print("Success: All cell values match exactly.")
        else:
            total = data1.size
            percent = (mismatch_count / total) * 100
            print(f"Failure: Values differ in {mismatch_count} cells ({percent:.4f}%).")
            
            # Show stats of differences
            diffs = data1[mismatch_mask] - data2[mismatch_mask]
            print(f"   Max Difference: {np.nanmax(diffs)}")
            print(f"   Mean Difference: {np.nanmean(diffs)}")


# --- PARQUET FUNCTIONS ---

def compare_parquets(path1: str, path2: str, value_col: str):
    """
    Performs a two-stage check on GeoParquet:
    1. Grid Alignment (Row counts, Bounding Box, Exact X/Y Coordinate matching)
    2. Cell Values (Comparing the specific value column)
    """
    print(f"\n--- Comparing Parquet Files ---")
    print(f"File 1: {os.path.basename(path1)}")
    print(f"File 2: {os.path.basename(path2)}")
    print(f"Target Column: {value_col}")

    # Explicit column check list
    cols = ['X', 'Y', value_col]
    
    try:
        # We attempt to load only the required columns.
        # If the column is missing, this will fail immediately.
        df1 = pd.read_parquet(path1, columns=cols)
        df2 = pd.read_parquet(path2, columns=cols)
    except Exception as e:
        # Check if the error is related to missing columns
        error_msg = str(e).lower()
        if "not in the index" in error_msg or "field not found" in error_msg:
            print(f"Error: The specific column '{value_col}' (or X/Y) was not found in one of the files.")
            print(f"Details: {e}")
        else:
            print(f"Error: Could not read parquet files. {e}")
        return

    # --- STAGE 1: GRID ALIGNMENT ---
    print("\n[Stage 1] Checking Grid Alignment...")

    # Check 1: Row Counts
    if len(df1) != len(df2):
        print(f"Error: Row count mismatch ({len(df1)} vs {len(df2)}).")
        return

    # Check 2: Borders (Bounding Box)
    # This ensures the overall area is the same before we check individual points
    bbox1 = [df1.X.min(), df1.X.max(), df1.Y.min(), df1.Y.max()]
    bbox2 = [df2.X.min(), df2.X.max(), df2.Y.min(), df2.Y.max()]
    
    # Use allclose for float safety
    if not np.allclose(bbox1, bbox2, atol=1e-6):
        print(f"Error: Borders (Extent) do not match.")
        print(f"   F1: {bbox1}")
        print(f"   F2: {bbox2}")
        return

    print("   -> Borders match. Sorting data to check internal grid...")

    # Sort both dataframes by coordinates to align them row-by-row
    df1 = df1.sort_values(by=['X', 'Y'], ignore_index=True)
    df2 = df2.sort_values(by=['X', 'Y'], ignore_index=True)

    # Check 3: Internal Coordinates (The actual Grid)
    # This ensures that point 1 in File A is spatially identical to point 1 in File B
    x_match = np.allclose(df1['X'], df2['X'], atol=1e-6)
    y_match = np.allclose(df1['Y'], df2['Y'], atol=1e-6)

    if not x_match or not y_match:
        print("Error: Internal grid misalignment. The X/Y coordinates do not match row-for-row.")
        return
    
    print("Success: Grid alignment verified (Borders and all X/Y coordinates match).")

    # --- STAGE 2: VALUE COMPARISON ---
    print("\n[Stage 2] Checking Values...")

    v1 = df1[value_col].to_numpy()
    v2 = df2[value_col].to_numpy()

    # Check matches (NaN == NaN is considered a match here)
    mismatch_mask = ~np.isclose(v1, v2, equal_nan=True, atol=1e-5)
    mismatch_count = np.count_nonzero(mismatch_mask)

    if mismatch_count == 0:
        print(f"Success: All values in '{value_col}' match exactly.")
    else:
        total = len(v1)
        percent = (mismatch_count / total) * 100
        print(f"Failure: Values differ in {mismatch_count} rows ({percent:.4f}%).")
        
        diffs = v1[mismatch_mask] - v2[mismatch_mask]
        print(f"   Max Difference: {np.nanmax(diffs)}")
        print(f"   Mean Difference: {np.nanmean(diffs)}")


def main():
    # --- CONFIGURATION ---
    # Update these paths to your actual files
    f1 = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Training\training_tiles\BH4SD56H_1\BH4SD56H_1_training_clipped_data.parquet"
    f2 = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\prediction_tiles\BH4SD56H_1\BH4SD56H_1_prediction_clipped_data.parquet"
    parquet_col = "combined1_bathy_BH4SD56H_2017"

    # --- RUNNER ---
    if not os.path.exists(f1) or not os.path.exists(f2):
        print("Error: Files not found.")
        return

    ext1 = os.path.splitext(f1)[1].lower()
    
    if ext1 in ['.tif', '.tiff']:
        compare_rasters(f1, f2)
    elif ext1 == '.parquet':
        compare_parquets(f1, f2, parquet_col)
    else:
        print("Unsupported file extension.")

if __name__ == "__main__":
    main()