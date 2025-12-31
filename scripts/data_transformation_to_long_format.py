# Aubrey, there are 3 parts to the scirpt - 1 The main transformation functions, 2- test the functionality on a single tile, and -3 run the function on all tiles
# I also asked AI to insert some additional context explaining the proceses a bit more -  pelase delete whats redundant to you\
    
 """
Data Transformation for Time-Series Modeling (Python / Parquet version)

This module converts wide-format grid tile data (per-tile .parquet files)
into long-format, year-pair data structures suitable for time-series
modeling. It preserves the core logic of the original R code, while
adopting Parquet as the storage format and using pandas for tabular
operations (Dask-compatible).

Key features
-----------
- Angle transformation for flow-direction columns (degrees → sin/cos).
- Training-mode transformation: builds (year_t, year_t1) pairs, target and
  predictor sets, and delta_* variables.
- Prediction-mode transformation: builds matching predictor-only year-pair
  tables.
- Batch runner that:
    * finds all tiles in a base directory,
    * processes them in parallel (optional),
    * writes Parquet outputs per tile and per year-pair, and
    * verifies written files (read metadata) and deletes any corrupt outputs.

All functions are designed to work cleanly with pandas. For Dask usage,
you can apply the same logic at the partition level or use this code as a
per-file worker in a Dask bag/delayed pipeline.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Literal, Optional

import pandas as pd
import numpy as np
import pyarrow.parquet as pq  # for verification only


# =============================================================================
#  Angle Transformation Helper
# =============================================================================

def transform_flowdir_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace flow direction angle columns (0–360°) with sin/cos components.

    Purpose
    -------
    Many hydrodynamic predictors include flow direction as an angle in
    degrees (0–360). These angles are circular: 0° and 360° represent the
    same direction, and 350° is close to 10°. Using raw degrees in machine
    learning can cause artificial discontinuities around the wrap point.

    This helper:
      - Detects any columns whose names contain the substring "flowdir".
      - For each such column:
          * creates `<col>_sin` = sin(angle in radians)
          * creates `<col>_cos` = cos(angle in radians)
      - Drops the original degree columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing possible flow direction columns.

    Returns
    -------
    pandas.DataFrame
        The same dataframe (modified in-place) with angle columns replaced
        by their sin/cos representations.
    """
    flow_cols = [c for c in df.columns if "flowdir" in c]

    if not flow_cols:
        return df

    # vectorized trig; works for pandas and is friendly to Dask if used
    radians = {c: np.deg2rad(df[c]) for c in flow_cols}

    for col in flow_cols:
        df[f"{col}_sin"] = np.sin(radians[col])
        df[f"{col}_cos"] = np.cos(radians[col])

    df.drop(columns=flow_cols, inplace=True)
    return df


# =============================================================================
#  Training Data Transformation
# =============================================================================

def process_training_df(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Transform a single training data tile into long-format year-pair tables.

    Purpose
    -------
    Given a wide-format tile (one row per cell, many columns for different
    years and variables), this function:

      1. Converts flow direction angles to sin/cos (see transform_flowdir_cols).
      2. Identifies per-year "state" variables that follow the pattern:
           <var_base>_<year>
         but excludes pair-year columns like <var>_2004_2006.
      3. For each consecutive year pair (2004–2006, 2006–2010, etc.):
           - Finds the set of variables present at both years.
           - Keeps only variables shared across both years.
           - Renames them to suffixes `_t` (time t) and `_t1` (time t+1).
           - Gathers:
                * ID columns (X, Y, FID, tile_id)
                * state variables at t and t1
                * forcing columns specific to the year-pair
                * static columns (e.g. sediment classes, survey_end_date)
           - Optionally computes bathymetric change (delta_bathy) if not
             already supplied as a delta column.
      4. Outputs one dataframe per year pair, suitable for time-series ML.

    Parameters
    ----------
    df : pandas.DataFrame
        A single tile of training data in wide format.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary keyed by `"year_t_year_t1"` (e.g. `"2004_2006"`),
        each value being a long-format dataframe ready for modeling.
    """
    df = df.copy()
    df = transform_flowdir_cols(df)

    year_pairs = [
        ("2004", "2006"),
        ("2006", "2010"),
        ("2010", "2015"),
        ("2015", "2022"),
    ]

    all_cols = df.columns.tolist()
    # columns like <var>_YYYY but not <var>_YYYY_YYYY
    potential_state_cols = [
        c for c in all_cols
        if re.search(r"_\d{4}", c) and not re.search(r"_\d{4}_\d{4}$", c)
    ]

    col_meta = pd.DataFrame({"colname": potential_state_cols})
    col_meta["year"] = col_meta["colname"].str.extract(r"_(\d{4})").astype(int)
    # remove "_YYYY" from the end to get the "base" variable name
    col_meta["var_base"] = [
        re.sub(f"_{int(y)}$", "", c)
        for c, y in zip(col_meta["colname"], col_meta["year"])
    ]

    year_pair_dfs: Dict[str, pd.DataFrame] = {}

    for y0_str, y1_str in year_pairs:
        y0 = int(y0_str)
        y1 = int(y1_str)
        pair_name = f"{y0_str}_{y1_str}"

        cols_t_meta = col_meta[col_meta["year"] == y0]
        cols_t1_meta = col_meta[col_meta["year"] == y1]

        common_vars = sorted(
            set(cols_t_meta["var_base"]).intersection(cols_t1_meta["var_base"])
        )

        # ensure we have bathymetry for this pair
        target_base_var = "bathy_filled"
        if target_base_var not in common_vars:
            continue

        # map base var -> column name at year y0 / y1
        t_map = dict(zip(cols_t_meta["var_base"], cols_t_meta["colname"]))
        t1_map = dict(zip(cols_t1_meta["var_base"], cols_t1_meta["colname"]))

        cols_t_exist = [t_map[v] for v in common_vars]
        cols_t1_exist = [t1_map[v] for v in common_vars]

        new_names_t = [f"{v}_t" for v in common_vars]
        new_names_t1 = [f"{v}_t1" for v in common_vars]

        forcing_pattern = f"({y0_str}_{y1_str})$"
        forcing_cols = [
            c for c in df.columns if re.search(forcing_pattern, c)
        ]

        delta_bathy_col = f"delta_bathy_{pair_name}"
        forcing_cols = [c for c in forcing_cols if c != delta_bathy_col]

        static_vars = ["grain_size_layer", "prim_sed_layer", "survey_end_date"]
        static_cols = [c for c in static_vars if c in df.columns]

        id_cols = ["X", "Y", "FID", "tile_id"]

        cols_to_grab = id_cols + cols_t_exist + cols_t1_exist + forcing_cols + static_cols
        if delta_bathy_col in df.columns:
            cols_to_grab.append(delta_bathy_col)

        missing = [c for c in cols_to_grab if c not in df.columns]
        if missing:
            # Skip this pair if required columns are missing
            continue

        pair_df = df[cols_to_grab].drop_duplicates()

        # rename *_YYYY -> *_t / *_t1
        rename_map = dict(zip(cols_t_exist + cols_t1_exist,
                              new_names_t + new_names_t1))
        pair_df = pair_df.rename(columns=rename_map)

        # If we already have a delta_bathy_<pair>, rename it to delta_bathy
        if delta_bathy_col in pair_df.columns:
            pair_df = pair_df.rename(columns={delta_bathy_col: "delta_bathy"})

        # add year_t and year_t1
        pair_df["year_t"] = y0
        pair_df["year_t1"] = y1

        # Compute delta from bathy_t / bathy_t1 if available
        # Note: depending on your original R version, you may or may not
        # have these columns at this stage. This mimics the intended behavior.
        if {"bathy_t", "bathy_t1"}.issubset(pair_df.columns):
            pair_df["delta_bathy"] = pair_df["bathy_t1"] - pair_df["bathy_t"]

        # Strip "_filled" suffix from any columns after everything else
        filled_cols = [c for c in pair_df.columns if "_filled" in c]
        if filled_cols:
            rename_filled = {
                c: c.replace("_filled", "")
                for c in filled_cols
            }
            pair_df = pair_df.rename(columns=rename_filled)

        # Model target and predictor sets
        target_col = "bathy_t1"
        predictor_cols_t = [c for c in pair_df.columns if c.endswith("_t")]
        predictor_cols_delta = [c for c in pair_df.columns if c.startswith("delta_")]

        id_cols_final = ["X", "Y", "FID", "tile_id", "year_t", "year_t1"]

        final_cols = (
            id_cols_final
            + [target_col]
            + predictor_cols_t
            + predictor_cols_delta
            + forcing_cols
            + static_cols
        )

        final_cols = [c for c in final_cols if c in pair_df.columns]
        final_pair_df = pair_df[final_cols]

        year_pair_dfs[pair_name] = final_pair_df

    return year_pair_dfs


# =============================================================================
#  Prediction Data Transformation
# =============================================================================

def process_prediction_df(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Transform a single prediction data tile into long-format year-pair tables.

    Purpose
    -------
    For prediction, we typically have a single "current state" of the seabed
    (e.g. `bt.bathy` and other bt.* predictors) and separate forcing layers
    defined for different future time windows (e.g. 2004_2006, 2006_2010…).

    This function:
      1. Converts flow direction angles to sin/cos.
      2. Renames all `bt.*` columns to `<basename>_t`, so they match the
         training feature convention for time t.
      3. For each year pair (2004–2006, 2006–2010, ...):
          - Gathers:
              * ID columns (X, Y, FID, tile_id)
              * All predictors at time t (`*_t`)
              * Forcing columns specific to that year pair
              * Static columns (grain size, sediment, survey date)
          - Produces a dataframe with predictors aligned to the training
            schema, but without bathy_t1 (the target), since we are predicting it.

    Parameters
    ----------
    df : pandas.DataFrame
        A single tile of prediction data in wide format.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Dictionary of year-pair → prediction dataframe suitable for
        feeding into a trained time-series model.
    """
    df = df.copy()
    df = transform_flowdir_cols(df)

    df_copy = df.copy()
    bt_cols = [c for c in df_copy.columns if re.match(r"^bt\.", c)]

    # bt.<var> -> <var>_t
    new_t_names = [re.sub(r"^bt\.", "", c) + "_t" for c in bt_cols]
    rename_bt = dict(zip(bt_cols, new_t_names))
    df_copy = df_copy.rename(columns=rename_bt)

    year_pairs = [
        ("2004", "2006"),
        ("2006", "2010"),
        ("2010", "2015"),
        ("2015", "2022"),
    ]

    year_pair_dfs: Dict[str, pd.DataFrame] = {}

    for y0_str, y1_str in year_pairs:
        pair_name = f"{y0_str}_{y1_str}"

        predictor_cols_t = new_t_names

        forcing_pattern = f"({y0_str}_{y1_str})$"
        forcing_cols = [
            c for c in df_copy.columns if re.search(forcing_pattern, c)
        ]

        static_vars = ["grain_size_layer", "prim_sed_layer", "survey_end_date"]
        static_cols = [c for c in static_vars if c in df_copy.columns]

        id_cols = ["X", "Y", "FID", "tile_id"]

        final_cols = id_cols + predictor_cols_t + forcing_cols + static_cols
        final_cols = [c for c in final_cols if c in df_copy.columns]

        pred_df_pair = df_copy[final_cols].drop_duplicates()

        year_pair_dfs[pair_name] = pred_df_pair

    return year_pair_dfs


# =============================================================================
#  File Verification Helper (Parquet)
# =============================================================================

def verify_parquet_file(path: str) -> bool:
    """
    Quick integrity check for a Parquet file.

    Purpose
    -------
    After writing each per-tile, per-year-pair output, we want to ensure
    the file is not corrupt (e.g., truncated write, disk error, or process
    interruption). Instead of reading the full dataset, we:

      - Attempt to open it with pyarrow.parquet.ParquetFile.
      - If metadata can be read successfully, we treat the file as valid.
      - If an exception is raised, we treat it as corrupt and delete it.

    Parameters
    ----------
    path : str
        Path to the Parquet file to verify.

    Returns
    -------
    bool
        True if the file appears valid, False if corrupt.
    """
    try:
        _ = pq.ParquetFile(path)
        return True
    except Exception:
        return False


# =============================================================================
#  Core Batch Transformation + Save
# =============================================================================

def _process_single_tile_file(
    f_path: str,
    mode: Literal["training", "prediction"]
) -> None:
    """
    Internal helper to process a single tile file and write per-year-pair outputs.
    """
    tile_name = os.path.basename(f_path).split("_")[0]
    output_tile_dir = os.path.dirname(f_path)

    # read input (.parquet)
    df = pd.read_parquet(f_path)

    process_func = process_training_df if mode == "training" else process_prediction_df
    processed_dict = process_func(df)

    for pair_name, pair_df in processed_dict.items():
        suffix = "_long.parquet" if mode == "training" else "_prediction_long.parquet"
        out_name = f"{tile_name}_{pair_name}{suffix}"
        out_path = os.path.join(output_tile_dir, out_name)

        pair_df.to_parquet(out_path, index=False)

        # verify and delete if corrupt
        if not verify_parquet_file(out_path):
            print(
                f"CRITICAL WARNING: Corrupt file detected and deleted -> "
                f"{out_name} in tile folder {tile_name}"
            )
            try:
                os.remove(out_path)
            except FileNotFoundError:
                pass


def transform_and_save_tiles(
    base_dir: str,
    mode: Literal["training", "prediction"] = "training",
    parallel_run: bool = True,
    max_workers: Optional[int] = None,
) -> None:
    """
    Process and save all tiles in a directory, with Parquet verification.

    Purpose
    -------
    This is the high-level orchestrator for your tile-based workflow.
    It:

      1. Recursively scans `base_dir` for files matching the pattern:
           * `_training_clipped_data.parquet` for mode="training"
           * `_prediction_clipped_data.parquet` for mode="prediction"
      2. For each file (tile):
           - Reads the wide-format parquet.
           - Calls either `process_training_df` or `process_prediction_df`.
           - Writes one Parquet file per year pair to the same folder.
           - Verifies each written file and deletes it if corrupt.
      3. Optionally runs in parallel using a process pool.

    This mirrors the R version (which used fst + doParallel) while using
    Parquet and Python’s concurrent.futures.

    Parameters
    ----------
    base_dir : str
        Base directory containing tile subdirectories with *_clipped_data.parquet.
    mode : {"training", "prediction"}, default "training"
        Whether to run the training or prediction transformation.
    parallel_run : bool, default True
        If True, use ProcessPoolExecutor for parallel tile processing.
    max_workers : int or None, default None
        Maximum number of worker processes. If None, uses `os.cpu_count() - 1`
        or 1, whichever is larger.

    Returns
    -------
    None
    """
    print(f"\n🚀 Starting data transformation for mode: {mode}")

    # file pattern: e.g., "_training_clipped_data.parquet"
    file_suffix = f"_{mode}_clipped_data.parquet"
    files_to_process: List[str] = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.endswith(file_suffix):
                files_to_process.append(os.path.join(root, fname))

    if not files_to_process:
        print("⚠️ No files found to process. Exiting.")
        return

    print(f"Found {len(files_to_process)} tiles to process.")

    if not parallel_run:
        print("Running in sequential mode.")
        for f_path in files_to_process:
            tile_name = os.path.basename(f_path).split("_")[0]
            print(f"Processing tile: {tile_name}")
            _process_single_tile_file(f_path, mode)
        print("\n🎉 Transformation complete (sequential).")
        return

    # Parallel execution
    if max_workers is None:
        cpu = os.cpu_count() or 2
        max_workers = max(1, cpu - 1)

    print(f"Setting up parallel processing with {max_workers} workers.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_single_tile_file, f_path, mode): f_path
            for f_path in files_to_process
        }

        for future in as_completed(futures):
            f_path = futures[future]
            try:
                future.result()
            except Exception as e:
                tile_name = os.path.basename(f_path).split("_")[0]
                print(f"❌ Error processing tile {tile_name}: {e}")

    print("\n🎉 Transformation complete (parallel).")


# =============================================================================
#  Test Utilities: Single-Tile Transformation
# =============================================================================

def test_single_tile_transformation(
    tile_id: str,
    training_base_dir: str,
    prediction_base_dir: str,
    training_output_dir: str,
    prediction_output_dir: str,
) -> None:
    """
    Run a focused test on a single tile for both training and prediction modes.

    Purpose
    -------
    Before launching the full batch job on all tiles, it is often safer to:
      - Pick one representative tile ID (with "good" data)

    This function:
      1. Locates the training and prediction clipped Parquet files for the
         specified tile.
      2. Applies `process_training_df` and `process_prediction_df` directly.
      3. Writes the outputs into separate `_test_output` directories to keep
         them isolated from production results.

    Parameters
    ----------
    tile_id : str
        Tile identifier, e.g. "BH4S556X_3".
    training_base_dir : str
        Base directory containing training tiles.
    prediction_base_dir : str
        Base directory containing prediction tiles.
    training_output_dir : str
        Directory where training test outputs will be written.
    prediction_output_dir : str
        Directory where prediction test outputs will be written.

    Returns
    -------
    None
    """
    os.makedirs(training_output_dir, exist_ok=True)
    os.makedirs(prediction_output_dir, exist_ok=True)

    # --- Training test ---
    print(f"\nTesting TRAINING data transformation for tile: {tile_id}")

    training_file_path = os.path.join(
        training_base_dir, tile_id, f"{tile_id}_training_clipped_data.parquet"
    )

    if os.path.exists(training_file_path):
        train_df = pd.read_parquet(training_file_path)
        processed_train = process_training_df(train_df)

        for pair_name, df_pair in processed_train.items():
            out_name = f"{tile_id}_{pair_name}_long.parquet"
            out_path = os.path.join(training_output_dir, out_name)
            df_pair.to_parquet(out_path, index=False)
            print(f"  ✅ Saved training output: {out_name}")
    else:
        print(f"  ⚠️ WARNING: Training file not found at: {training_file_path}")

    # --- Prediction test ---
    print(f"\nTesting PREDICTION data transformation for tile: {tile_id}")

    prediction_file_path = os.path.join(
        prediction_base_dir, tile_id, f"{tile_id}_prediction_clipped_data.parquet"
    )

    if os.path.exists(prediction_file_path):
        pred_df = pd.read_parquet(prediction_file_path)
        processed_pred = process_prediction_df(pred_df)

        for pair_name, df_pair in processed_pred.items():
            out_name = f"{tile_id}_{pair_name}_prediction_long.parquet"
            out_path = os.path.join(prediction_output_dir, out_name)
            df_pair.to_parquet(out_path, index=False)
            print(f"  ✅ Saved prediction output: {out_name}")
    else:
        print(f"  ⚠️ WARNING: Prediction file not found at: {prediction_file_path}")

    print("\n🎉 Test complete. Check the '_test_output' folders for results.")


# =============================================================================
#  Example "main" usage (similar to R execution script)
# =============================================================================
if __name__ == "__main__":
    # Example paths — adjust to your environment
    training_base_dir = r"N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles"
    prediction_base_dir = r"N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"

    # Example: run test on a single tile
    tile_to_test = "BH4S556X_3"
    training_output_dir = os.path.join(training_base_dir, "_test_output")
    prediction_output_dir = os.path.join(prediction_base_dir, "_test_output")

    # Uncomment if you want to run the single-tile test:
    # test_single_tile_transformation(
    #     tile_id=tile_to_test,
    #     training_base_dir=training_base_dir,
    #     prediction_base_dir=prediction_base_dir,
    #     training_output_dir=training_output_dir,
    #     prediction_output_dir=prediction_output_dir,
    # )

    # Full batch transformations (parallel)
    # transform_and_save_tiles(training_base_dir, mode="training", parallel_run=True)
    # transform_and_save_tiles(prediction_base_dir, mode="prediction", parallel_run=True)

    print("\n✅ Module loaded. Call functions from your own driver script as needed.\n")
   