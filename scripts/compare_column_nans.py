import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict

def compare_nan_distribution(path: str, col1: str, col2: str) -> None:
    print(f"Loading columns ['{col1}', '{col2}'] using fastparquet...")

    print(f"All columns: {pd.read_parquet(path, engine='fastparquet').columns.tolist()}")

    # Direct load using fastparquet engine
    df = pd.read_parquet(path, columns=[col1, col2], engine='fastparquet')

    mask1 = df[col1].isna()
    mask2 = df[col2].isna()
    
    total_rows = len(df)
    
    nans_1 = mask1.sum()
    nans_2 = mask2.sum()
    
    pct_1 = (nans_1 / total_rows) * 100
    pct_2 = (nans_2 / total_rows) * 100

    both_nan = (mask1 & mask2).sum()
    c1_nan_only = (mask1 & ~mask2).sum()
    c2_nan_only = (~mask1 & mask2).sum()
    mismatch_count = (mask1 != mask2).sum()

    print(f"\n--- NaN Comparison Results ---")
    print(f"Total Rows: {total_rows:,}")
    print(f"-" * 30)
    
    print(f"Column 1 ('{col1}'):")
    print(f"  Total NaNs: {nans_1:,} ({pct_1:.4f}%)")
    
    print(f"Column 2 ('{col2}'):")
    print(f"  Total NaNs: {nans_2:,} ({pct_2:.4f}%)")
    
    print(f"-" * 30)
    print(f"Comparison Logic:")
    
    print(f"  [Intersection] Both columns are NaN:       {both_nan:,}")
    
    if mismatch_count == 0:
        print(f"  [Perfect Match] The NaN patterns are identical.")
    else:
        print(f"  [Mismatch] Total rows with different NaN status: {mismatch_count:,}")
        print(f"     -> '{col1}' is NaN, but '{col2}' has data: {c1_nan_only:,}")
        print(f"     -> '{col2}' is NaN, but '{col1}' has data: {c2_nan_only:,}")

    if c1_nan_only > 0 or c2_nan_only > 0:
        print(f"\nWarning: The missing data patterns imply these variables are not perfectly synonymous.")
        if nans_1 > nans_2:
            print(f"Observation: '{col1}' is missing more data than '{col2}'.")
        elif nans_2 > nans_1:
            print(f"Observation: '{col2}' is missing more data than '{col1}'.")
    else:
        print(f"\nSuccess: These columns share the exact same missing data structure.")

def main():
    file_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Training\training_tiles\BH4S5574_4\BH4S5574_4_training_clipped_data.parquet"
    # file_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\prediction_tiles\BH4SD56H_1\BH4SD56H_1_prediction_clipped_data.parquet"

    
    column_a = "combined1_bathy_BH4S5574_2020"
    column_b = "combined1_bathy_BH4S5574_2020_flowacc"

    print(f"Analyzing file: {os.path.basename(file_path)}")
    compare_nan_distribution(file_path, column_a, column_b)

if __name__ == "__main__":
    main()