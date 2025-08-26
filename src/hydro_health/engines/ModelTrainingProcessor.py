import os
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.transform import from_origin
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from shapely.geometry import box
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import partial_dependence
import warnings
import cProfile
import pstats
import json
import dask
from dask import delayed, compute
from dask.distributed import Client, print

# Ignore specific warnings from libraries for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='geopandas')
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
#
#                           GLOBAL PARAMETERS
#
# ==============================================================================
# LOAD PARAMETERS CREATED FROM PREPROCESSING STAGE
# update to ER3 directories
output_dir = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\prediction_tiles"
# TODO update this to use the global parameter for the year pairs
year_pairs = [
            ('2006_2010'),
            ('2016_2020')]
block_size_m = 200 # The 200m block size that worked well for the scale of our sub-grid tile size - to support cross validation and k folds
grid_gpkg_path = (r"N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg") # from Blue topo
#
training_sub_grids_utm_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_subgrids\training_intersecting_subgrids.gpkg"
prediction_sub_grids_utm_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_subgrids\prediction_intersecting_subgrids.gpkg"
#
# training_mask_df = gpd.read_parquet(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\masks\training_mask.parquet")# spatial DF of extent
# prediction_mask_df = gpd.read_parquet("N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\masks\prediction_mask.parquet")# spatial DF of extent
#
output_dir_train = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Training\training_tiles"


# ==============================================================================
#
#                            PRE-TRAINING FUNCTIONS
# --- note - the current boruta predictor selection method is tile by tile approach, but for the pilot model it takes about 48 hrs to run. This is
# not feasable for the ER size model. So we need to modify the methodlogy to work on a global scale by selecting a 'representative' sub sample of tiles
# for the boruta predictor selection to run on - which will then be applied generally accross all tiles. ---
#
# ==============================================================================

#' BORUTA PREDICTOR SELECTION
# Perform Predictor Selection before we run the model using Boruta
#'
#' This function takes training data and uses the Boruta algorithm to identify
#' important predictors for a given response variable. It saves the results,
#' including a list of confirmed predictors and performance metrics, to be used
#' in downstream modeling.
#'
#' @param training_sub_grids_utm_path A data frame or sf object with tile IDs.
#' @param output_dir_train The base directory where training data is stored and results will be saved.
#' @param year_pairs A list of year pairs to process (e.g., "2000_2005").
#' @param max_runs The maximum number of iterations for the Boruta algorithm.
#'
#' @return A list containing the paths to the selection results for each tile and year pair.

# ------------------------------------------------------------------------------
# 1. BORUTA PREDICTOR SELECTION
# ------------------------------------------------------------------------------
def select_predictors_boruta_py(output_dir_train, year_pairs, max_runs=100):
    """
    Perform Predictor Selection before we run the model using Boruta.

    This function takes training data and uses the Boruta algorithm to identify
    important predictors for a given response variable. It saves the results,
    including a list of confirmed predictors and performance metrics, to be used
    in downstream modeling.

    Args:
        output_dir_train (str): The base directory where training data is stored and results will be saved.
        year_pairs (list): A list of year pairs to process (e.g., "2004_2006").
        max_runs (int): The maximum number of iterations for the Boruta algorithm.
    """
    # -------------------------------------------------------
    # 1. INITIALIZATION & PARALLEL SETUP----
    # -------------------------------------------------------
    print("\nStarting Predictor Selection with Boruta...\n")

    log_file = os.path.join(output_dir_train, "predictor_selection_log.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename=log_file, filemode='w')
    logging.info("Log - Boruta Predictor Selection")

    for pair in year_pairs:
        csv_path = os.path.join(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\year_pair_nan_counts.csv")
        df = pd.read_csv(csv_path)

        nan_percent_col = f'{pair}_nan_percent'

        # Filter rows where the value in the column is < 100, then select 'tile_id'
        # TODO which percent limit to use?
        tile_ids = df.loc[df[nan_percent_col] < 95, 'tile_id'].tolist()

        print(f"Running Boruta selection for {len(tile_ids)} tiles for {pair}...")

        # -------------------------------------------------------
        # 2. ITERATE THROUGH TILES AND YEAR PAIRS----
        # -------------------------------------------------------
        tasks = [delayed(_worker_boruta_tile)(tile_id, output_dir_train, pair, max_runs)
                for tile_id in tile_ids]

        # print("Submitting tasks to Dask scheduler...")
        compute(*tasks)

        print("\n Boruta Predictor Selection Complete! Check `predictor_selection_log.txt` for details.\n")

def _worker_boruta_tile(tile_id, output_dir_train, pair, max_runs)-> None:
    """Helper worker function for running Boruta on a single tile."""
    try:
        training_data_path = os.path.join(output_dir_train, tile_id, f"{tile_id}_training_clipped_data.parquet")
        if not os.path.exists(training_data_path):
            logging.warning(f"Missing training tile: {tile_id}, skipping.")
            return
        training_data = pd.read_parquet(training_data_path)
        # print(training_data.columns)

        start_year, end_year = pair.split('_')
        response_var = f"b.change.{pair}".strip() # Use strip() as a safeguard

        if response_var not in training_data.columns:
            logging.error(f"ERROR: Response variable '{response_var}' missing for Tile: {tile_id}, skipping.")
            return

        static_predictors = ["sed_size_raster_100m", "sed_type_raster_100m"]
        # dynamic_predictors = [
        #     f"bathy_{start_year}", f"bathy_{end_year}",
        #     f"slope_{start_year}", f"slope_{end_year}",
        #     f"hurr_count_{pair}", f"hurr_strength_{pair}", f"tsm_{pair}"
        # ]
        # TODO need to fix the naming for the sediment and tsm variables
        # TODO need to think more about which rugosity and slope years are used but the base process appears to work
        dynamic_predictors = [
            f"bathy_{start_year}", f"bathy_{end_year}",
            f"slope", f"slope", f"{pair}_tsm_mean"
        ]
        # print(f"dynamic predictors: {dynamic_predictors}")
        rugosity_start = [col for col in training_data.columns if col.startswith(f"rugosity") and col.endswith(start_year)]
        rugosity_end = [col for col in training_data.columns if col.startswith(f"rugosity") and col.endswith(end_year)]

        all_predictors = list(set(static_predictors + dynamic_predictors + rugosity_start + rugosity_end) & set(training_data.columns))
        print(f"all predictors: {all_predictors}")

        # print(f'full: {training_data}')
        nan_counts = training_data[all_predictors + [response_var]].isnull().sum()

        # print("---------------------------------------")
        # print("Number of Missing Values (NaNs) per Column:")
        # print("---------------------------------------")
        # print(nan_counts.sort_values(ascending=False).to_string())

        all_predictors=['bathy_2006', 'bathy_2010', 'sed_size_raster_100m']

        sub_data = training_data[all_predictors + [response_var]].dropna()
        print(f"dropped date remaining: {sub_data}")

        if len(sub_data) < 50 or sub_data[response_var].nunique() <= 1:
            logging.warning(f"Skipping Boruta for Tile: {tile_id} | Pair: {pair} - Insufficient data.")
            return

        logging.info(f"Running Boruta for Tile: {tile_id} | Pair: {pair}")

        rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=1)
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, max_iter=max_runs, random_state=1)

        X = sub_data[all_predictors].values
        y = sub_data[response_var].values

        feat_selector.fit(X, y)

        # --- Store Results ---
        confirmed_preds = list(np.array(all_predictors)[feat_selector.support_])
        print(f"Confirmed predictors for Tile: {tile_id} | Pair: {pair}: {confirmed_preds}")

        decision_map = {True: 'Confirmed', False: 'Rejected'}
        tentative_indices = np.where(feat_selector.support_weak_)[0]

        # The .feature_importances_ attribute has one score for each predictor
        stats_df = pd.DataFrame({
            'predictor': all_predictors,
            'meanImp': feat_selector.cv_scores_,
            'decision': [decision_map.get(val, 'Tentative') for val in feat_selector.support_]
        })
        for idx in tentative_indices:
            stats_df.loc[idx, 'decision'] = 'Tentative'

        stats_df = stats_df.sort_values(by='meanImp', ascending=False)

        output_list = {
            'confirmed_predictors': confirmed_preds,
            'boruta_statistics': stats_df
        }

        tile_dir = os.path.join(output_dir_train, tile_id)
        os.makedirs(tile_dir, exist_ok=True)
        # Use pickle for complex list/dict objects. This is the right tool for the job.
        save_path = os.path.join(tile_dir, f"boruta_selection_{pair}.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(output_list, f)

    except Exception as e:
        logging.error(f"ERROR in Boruta selection for Tile: {tile_id} | Pair: {pair} | {e}", exc_info=True)

def create_boruta_summary_report(output_dir_train, overall_report_filename="boruta_summary_report.png", by_year_report_filename="boruta_importance_by_year.png", top_n=10):
    """
    Create Visual Reports for Boruta Predictor Selection.

    This function scans a directory for Boruta selection results (`.pkl` files),
    aggregates them, and generates two separate summary plots:
    1. An overall summary of predictor confirmation, rejection, and importance.
    2. A breakdown of the most important predictors for each year-pair.

    Args:
        output_dir_train (str): The base directory where the tile folders are located.
        overall_report_filename (str): The name of the output PNG for the overall summary.
        by_year_report_filename (str): The name of the output PNG for the year-pair breakdown.
        top_n (int): The number of top predictors to display in the plots.
    """
    # -------------------------------------------------------
    # 1. FIND FILES
    # -------------------------------------------------------
    print("📊 Starting Boruta summary report generation...")

    selection_files = [os.path.join(root, name)
                       for root, _, files in os.walk(output_dir_train)
                       for name in files if name.startswith("boruta_selection_") and name.endswith(".pkl")]

    if not selection_files:
        print("No 'boruta_selection_*.pkl' files found. Please run the selection script first.")
        return
    print(f"Found {len(selection_files)} Boruta result files.")

    # -------------------------------------------------------
    # 2. LOAD and PROCESS ALL RESULTS
    # -------------------------------------------------------
    all_results = []
    for fp in selection_files:
        try:
            with open(fp, 'rb') as f:
                result_list = pickle.load(f)
            stats_df = result_list['boruta_statistics']
            path_parts = fp.replace('\\', '/').split('/')
            stats_df['tile_id'] = path_parts[-2]
            stats_df['year_pair'] = path_parts[-1].split('_')[-1].replace('.pkl', '')
            all_results.append(stats_df)
        except Exception as e:
            print(f"Could not process file {fp}: {e}")
            continue

    if not all_results:
        print("No valid Boruta results could be loaded.")
        return

    combined_df = pd.concat(all_results, ignore_index=True)
    print("Successfully processed all result files.")

    # -------------------------------------------------------
    # 3. PREPARE DATA FOR PLOTTING (OVERALL SUMMARY)
    # -------------------------------------------------------
    confirmed_counts = combined_df[combined_df['decision'] == 'Confirmed']['predictor'].value_counts().nlargest(top_n)
    rejected_counts = combined_df[combined_df['decision'] == 'Rejected']['predictor'].value_counts().nlargest(top_n)
    decision_summary = combined_df['decision'].value_counts()
    importance_summary = combined_df.groupby('predictor')['meanImp'].mean().nlargest(top_n)

    # -------------------------------------------------------
    # 4. GENERATE OVERALL 4-PANEL PLOT
    # -------------------------------------------------------
    print("Generating overall summary plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Boruta Predictor Selection Overall Summary", fontsize=16)

    # --- PLOT 1: Most Confirmed ---
    sns.barplot(x=confirmed_counts.values, y=confirmed_counts.index, ax=axes[0, 0], color='darkgreen', orient='h')
    axes[0, 0].set_title(f"Top {top_n} Most Confirmed Predictors")
    axes[0, 0].set_xlabel("Times Confirmed")

    # --- PLOT 2: Most Rejected ---
    sns.barplot(x=rejected_counts.values, y=rejected_counts.index, ax=axes[0, 1], color='firebrick', orient='h')
    axes[0, 1].set_title(f"Top {top_n} Most Rejected Predictors")
    axes[0, 1].set_xlabel("Times Rejected")

    # --- PLOT 3: Overall Decisions ---
    sns.barplot(x=decision_summary.index, y=decision_summary.values, ax=axes[1, 0], palette={"Confirmed": "darkgreen", "Tentative": "darkorange", "Rejected": "firebrick"})
    axes[1, 0].set_title("Overall Decision Frequency")
    axes[1, 0].set_ylabel("Total Count")

    # --- PLOT 4: Top Importance Score ---
    sns.barplot(x=importance_summary.values, y=importance_summary.index, ax=axes[1, 1], color='steelblue', orient='h')
    axes[1, 1].set_title(f"Top {top_n} Predictors by Avg. Importance")
    axes[1, 1].set_xlabel("Mean Importance (Gain)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir_train, overall_report_filename), dpi=100)
    plt.close()
    print(f"Report saved to: {os.path.join(output_dir_train, overall_report_filename)}")

    # -------------------------------------------------------
    # 5. PREPARE DATA FOR YEAR-PAIR SPECIFIC PLOT
    # -------------------------------------------------------
    print("Preparing data for year-pair specific importance plot...")
    year_pairs = sorted(combined_df['year_pair'].unique())

    # -------------------------------------------------------
    # 6. GENERATE YEAR-PAIR SPECIFIC IMPORTANCE PLOT
    # -------------------------------------------------------
    print("Generating year-pair specific importance plot...")
    n_pairs = len(year_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), squeeze=False)
    fig.suptitle("Top Predictor Importance by Year Pair", fontsize=16, y=0.98)

    for i, pair in enumerate(year_pairs):
        ax = axes.flatten()[i]
        plot_data = combined_df[combined_df['year_pair'] == pair].groupby('predictor')['meanImp'].mean().nlargest(top_n)
        if not plot_data.empty:
            sns.barplot(x=plot_data.values, y=plot_data.index, ax=ax, color='darkcyan', orient='h')
            ax.set_title(f"Year Pair: {pair}")
            ax.set_xlabel("Mean Importance (Gain)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir_train, by_year_report_filename), dpi=100)
    plt.close()
    print(f"Year-pair report saved to: {os.path.join(output_dir_train, by_year_report_filename)}")
    print("Process complete.")

# ==============================================================================
#
#                            MAIN TRAINING FUNCTION
#
# ==============================================================================

def generate_cv_diagnostic_plot(sf_data, block_geom, max_k, tile_id, pair, output_dir_train):
    """
    Creates a plot to visualize the training data and the spatial blocks.

    Args:
        sf_data (gpd.GeoDataFrame): The GeoDataFrame containing the training data points.
        block_geom (gpd.GeoDataFrame): The GeoDataFrame of the grid blocks created for probing.
        max_k (int): The maximum number of folds possible for the geometry.
        tile_id (str): The character ID of the current tile.
        pair (str): The character ID of the current year pair.
        output_dir_train (str): The main output directory.
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))

        # Plot the raw data points
        sf_data.plot(ax=ax, color="grey", markersize=1.5, alpha=0.5, label="Data Points")

        # Overlay the spatial block boundaries
        block_geom.boundary.plot(ax=ax, color="blue", linewidth=0.7, label="CV Blocks")

        # Get block size from grid for subtitle
        if not block_geom.empty:
            block_width_est = np.sqrt(block_geom.geometry.area.mean())
            subtitle = f"Approx. Block Size: {block_width_est:.0f}m | Max Possible Folds (k): {max_k}"
            fig.suptitle(subtitle, y=0.92, fontsize=10)

        ax.set_title(f"Diagnostic Plot for Tile: {tile_id} | Pair: {pair}", fontsize=14)
        ax.set_xlabel("X (UTM)")
        ax.set_ylabel("Y (UTM)")
        ax.ticklabel_format(useOffset=False, style='plain')
        ax.legend()
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Define the output path and save the plot
        plot_dir = os.path.join(output_dir_train, tile_id, "diagnostic_plots")
        os.makedirs(plot_dir, exist_ok=True)

        figure_name = f"spatial_cv_map_{pair}.png"
        plt.savefig(os.path.join(plot_dir, figure_name), dpi=150)
        print(f"Diagnostic plot saved to: {os.path.join(plot_dir, figure_name)}")

    except Exception as e:
        logging.error(f"Could not generate CV diagnostic plot for {tile_id}/{pair}: {e}")
    finally:
        plt.close('all')

def model_train_full_spatial_cv_py(training_sub_grids_utm_path, output_dir_train, year_pairs, block_size_m, n_boot=20, n_folds=5):
    """
    This script contains a complete, refactored set of functions to train
    the XGBoost models. It is designed to:
        1. Run a robust, parallelized spatial cross-validation to find the optimal model parameters.
        2. Train a final model and run multiple bootstrap replicates to capture uncertainty.
        3. Save all necessary outputs for the prediction workflow.
    """
    # -------------------------------------------------------
    # 1. MODEL INITIALIZATION & ERROR LOGGING
    # -------------------------------------------------------
    print("\n🚀 Starting Full XGBoost Model Training with Robust Parallel Spatial CV...\n")

    master_log_file = os.path.join(output_dir_train, "training_log_final.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=master_log_file, filemode='w')
    logging.info(f"Error Log - XGBoost Full Training (run started at {pd.Timestamp.now()})")

    # --- Setup Parallel Processing ---
    training_sub_grids_utm = gpd.read_file(training_sub_grids_utm_path)
    tile_ids = list(training_sub_grids_utm['tile_id'])
    grid_crs = training_sub_grids_utm.crs # Extract CRS for raster saving

    # -------------------------------------------------------
    # 2. MAIN PARALLEL PROCESSING LOOP
    # -------------------------------------------------------
    # Create a list of delayed tasks for Dask

        #     prediction_tasks = []
        # for file in prediction_files:
        #     input_path = os.path.join(input_directory, file)
        #     output_path = os.path.join(prediction_out, file.name)
        #     prediction_tasks.append(dask.delayed(self.process_prediction_raster)(input_path, mask_gdf, output_path))

        # dask.compute(*prediction_tasks)  
    tasks = []    
    for pair in year_pairs:
        csv_path = os.path.join(r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\year_pair_nan_counts.csv")
        df = pd.read_csv(csv_path)

        nan_percent_col = f'{pair}_nan_percent'

        # Filter rows where the value in the column is < 100, then select 'tile_id'
        # TODO which percent limit to use?
        tile_ids = df.loc[df[nan_percent_col] < 95, 'tile_id'].tolist()
        print(tile_ids)
        for tile_id in tile_ids:
            tasks.append(dask.delayed(_worker_train_tile)(tile_id, output_dir_train, pair, block_size_m, n_boot, n_folds, grid_crs))

    # Execute all tasks in parallel using Dask's scheduler
    print("Submitting tasks to Dask scheduler...")
    dask.compute(*tasks)

    # -------------------------------------------------------
    # 10. CONSOLIDATE LOGS
    # -------------------------------------------------------
    print("\nParallel processing complete. Consolidating logs...")
    with open(master_log_file, 'a') as master:
        for tile_id in tile_ids:
            worker_log_path = os.path.join(output_dir_train, tile_id, f"log_worker_train_{tile_id}.txt")
            if os.path.exists(worker_log_path):
                try:
                    with open(worker_log_path, 'r') as worker_log:
                        master.write(f"\n--- Log for Tile: {tile_id} ---\n")
                        master.write(worker_log.read())
                    os.remove(worker_log_path) # Clean up individual worker logs
                except IOError as e:
                    print(f"Could not read or remove worker log {worker_log_path}: {e}")

    print(f"\n✅ Model Training Complete! Check `{master_log_file}` for details.\n")

def _worker_train_tile(tile_id, output_dir_train, pair, block_size_m, n_boot, n_folds, grid_crs):
    """Helper worker function for training on a single tile."""
    # Inner Loop - over year pairs runs sequentially
    tile_dir = os.path.join(output_dir_train, tile_id)
    # worker_log_file = os.path.join(tile_dir, f"log_worker_train_{tile_id}.txt")
    # log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler(worker_log_file, mode='w')
    # file_handler.setFormatter(log_formatter)
    # logger = logging.getLogger(f'worker_train_{tile_id}')
    # logger.setLevel(logging.INFO)
    # if not logger.handlers:
    #     logger.addHandler(file_handler)

    # logger.info(f"Worker log for tile: {tile_id} started at {pd.Timestamp.now()}")

    # for pair in year_pairs:
    try:
        # --- a. Load Data---
        print(f"\nProcessing Tile: {tile_id} | Year Pair: {pair}")
        training_data_path = os.path.join(tile_dir, f"{tile_id}_training_clipped_data.parquet")
        print(training_data_path)
        boruta_results_path = os.path.join(tile_dir, f"boruta_selection_{pair}.pkl")

        if not os.path.exists(training_data_path) or not os.path.exists(boruta_results_path):
            print("DIAGNOSTIC: Missing input file(s). Skipping.")
            return

        full_training_gdf = gpd.read_parquet(training_data_path)
        with open(boruta_results_path, 'rb') as f:
            boruta_results = pickle.load(f)
        predictors = boruta_results['confirmed_predictors']
        response_var = f"b.change.{pair}"

        if not predictors or response_var not in full_training_gdf.columns:
            print("DIAGNOSTIC: No predictors or response variable found. Skipping.")
            return
        print(f"DIAGNOSTIC: Loaded {len(full_training_gdf)} total rows from input data.")

        # -------------------------------------------------------
        # 3. FILTER & PREPARE TRAINING DATA----
        # -------------------------------------------------------
        cols_to_keep = predictors + [response_var, "X", "Y", "FID", "geometry"]
        subgrid_gdf = full_training_gdf[list(set(cols_to_keep) & set(full_training_gdf.columns))].copy()
        for col in predictors + [response_var]:
            subgrid_gdf[col] = pd.to_numeric(subgrid_gdf[col], errors='coerce')
        subgrid_gdf.dropna(subset=(predictors + [response_var]), inplace=True)
        subgrid_gdf.reset_index(drop=True, inplace=True)

        print(f"DIAGNOSTIC: Filtered data to {len(subgrid_gdf)} rows with finite values for training.")
        if len(subgrid_gdf) < 100:
            print("DIAGNOSTIC: Insufficient data (<100 rows) after filtering. Skipping.")
            return

        subgrid_data = pd.DataFrame(subgrid_gdf.drop(columns='geometry'))

        predictor_ranges = pd.DataFrame({
            'Predictor': predictors,
            'Min_Value': [subgrid_data[p].min() for p in predictors],
            'Max_Value': [subgrid_data[p].max() for p in predictors]
        })
        predictor_ranges['Range_Width'] = predictor_ranges['Max_Value'] - predictor_ranges['Min_Value']

        # -------------------------------------------------------
        # 4. INITIALIZE METRIC OUTPUT STORAGE ARRAYS----
        # -------------------------------------------------------
        deviance_mat = np.full((n_boot, 3), np.nan, dtype=float)
        influence_mat = np.full((len(predictors), n_boot), np.nan, dtype=float)
        boot_array = np.full((len(subgrid_data), 1, n_boot), np.nan, dtype=float)

        # -------------------------------------------------------
        # 5. SETUP for PARTIAL DEPENDENCE PLOT (PDP) DATA----
        # -------------------------------------------------------
        pred_mins = subgrid_data[predictors].min().to_dict()
        pred_maxs = subgrid_data[predictors].max().to_dict()

        pdp_env_ranges_list = []
        for pred in predictors:
            min_val, max_val = pred_mins.get(pred), pred_maxs.get(pred)
            vals = np.linspace(min_val, max_val, 100) if pd.notna(min_val) and pd.notna(max_val) and min_val < max_val else np.repeat(min_val, 100)
            pdp_env_ranges_list.append(pd.DataFrame({'Env_Value': vals, 'Predictor': pred}))
        pdp_env_ranges_df = pd.concat(pdp_env_ranges_list, ignore_index=True)

        if len(subgrid_data) >= 100: # Sample 100 spatial points
            sampled_coords = subgrid_data[['X', 'Y', 'FID']].head(100)
            pdp_env_ranges_df['X'] = np.tile(sampled_coords['X'], len(predictors))
            pdp_env_ranges_df['Y'] = np.tile(sampled_coords['Y'], len(predictors))
            pdp_env_ranges_df['FID'] = np.tile(sampled_coords['FID'], len(predictors))

        pd_array = np.full((100, len(predictors), n_boot), np.nan, dtype=float)
        all_pdp_long_list = []

        # -------------------------------------------------------
        # 6. SETUP ADAPTIVE SPATIAL CROSS VALIDATION & MODEL TRAINING
        # -------------------------------------------------------
        best_iteration = 100 # Default fallback value
        cv_results_df = None

        bounds = subgrid_gdf.total_bounds
        grid_cells = [box(x0, y0, x0 + block_size_m, y0 + block_size_m) for x0 in np.arange(bounds[0], bounds[2], block_size_m) for y0 in np.arange(bounds[1], bounds[3], block_size_m)]
        grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs=grid_crs)
        points_in_blocks = gpd.sjoin(subgrid_gdf, grid_gdf, how="inner", predicate="within")
        active_blocks_gdf = grid_gdf.iloc[points_in_blocks['index_right'].unique()]
        max_k = len(active_blocks_gdf)

        generate_cv_diagnostic_plot(subgrid_gdf, active_blocks_gdf, max_k, tile_id, pair, output_dir_train)

        k_final = min(n_folds, max_k)

        try:
            if k_final < 2: raise ValueError(f"CV not possible. Only {max_k} spatial block(s) found.")
            subgrid_gdf['block_id'] = gpd.sjoin(subgrid_gdf, active_blocks_gdf, how='left', predicate='within')['index_right'].fillna(-1)
            gkf = GroupKFold(n_splits=k_final)
            fold_splits = gkf.split(subgrid_gdf, groups=subgrid_gdf['block_id'])
            best_nrounds_per_fold, rmse_per_fold, mae_per_fold = [], [], []

            for k, (train_idx, test_idx) in enumerate(fold_splits):
                train_set, test_set = subgrid_data.iloc[train_idx], subgrid_data.iloc[test_idx]
                if test_set[response_var].nunique() < 2:
                    print(f"INFO: Skipping fold {k+1} due to zero variance in the test set response.")
                    continue

                dtrain_fold = xgb.DMatrix(train_set[predictors], label=train_set[response_var])
                dtest_fold = xgb.DMatrix(test_set[predictors], label=test_set[response_var])
                watchlist_fold = [(dtest_fold, 'test')]
                evals_result = {}

                fold_model = xgb.train(
                    params={'max_depth': 4, 'eta': 0.01, 'gamma': 1, 'objective': 'reg:squarederror', 'eval_metric': ['rmse', 'mae']},
                    dtrain=dtrain_fold, num_boost_round=1000,
                    evals=watchlist_fold, early_stopping_rounds=10,
                    evals_result=evals_result, verbose_eval=False
                )
                if fold_model.best_iteration > 0:
                    best_nrounds_per_fold.append(fold_model.best_iteration)
                    rmse_per_fold.append(evals_result['test']['rmse'][fold_model.best_iteration -1])
                    mae_per_fold.append(evals_result['test']['mae'][fold_model.best_iteration -1])

            if best_nrounds_per_fold:
                best_iteration = int(np.mean(best_nrounds_per_fold))
                cv_results_df = pd.DataFrame({'tile_id': [tile_id], 'year_pair': [pair], 'best_iteration': [best_iteration], 'test_rmse_mean': [np.mean(rmse_per_fold)], 'test_rmse_std': [np.std(rmse_per_fold)], 'test_mae_mean': [np.mean(mae_per_fold)], 'test_mae_std': [np.std(mae_per_fold)]})
                print(f"DIAGNOSTIC: CV successful. Optimal iteration: {best_iteration}")
            else: raise ValueError("Manual CV loop failed to find any best iterations.")
        except Exception as e:
            print(f"WARNING: CV SKIPPED for Tile: {tile_id} | Pair: {pair} with error: {e}. Using default {best_iteration} rounds.")
            cv_results_df = pd.DataFrame({'tile_id': [tile_id], 'year_pair': [pair], 'best_iteration': [best_iteration], 'test_rmse_mean': [np.nan], 'test_rmse_std': [np.nan], 'test_mae_mean': [np.nan], 'test_mae_std': [np.nan]})

        # FINAL TRAINING MODEL
        dtrain_full = xgb.DMatrix(subgrid_data[predictors], label=subgrid_data[response_var])
        xgb_params = {
            'max_depth': 4, 'eta': 0.01, 'gamma': 1, 'subsample': 0.7,
            'colsample_bytree': 0.8, 'objective': 'reg:squarederror'
        }

        print(f"DIAGNOSTIC: Starting bootstrap loop for {n_boot} iterations...")

        for b in range(n_boot):
            xgb_model = xgb.train(
                params=xgb_params, dtrain=dtrain_full,
                num_boost_round=best_iteration, nthreads=1
            )

            # -------------------------------------------------------
            # 7. STORE MODEL METRICS ----
            # -------------------------------------------------------
            predictions = xgb_model.predict(dtrain_full)
            boot_array[:, 0, b] = predictions

            importance_matrix = xgb_model.get_score(importance_type='gain')
            if importance_matrix:
                for i, pred in enumerate(predictors):
                    influence_mat[i, b] = importance_matrix.get(pred, 0)

            r2 = r2_score(subgrid_data[response_var], predictions)
            deviance_mat[b, 0] = r2 # Dev.Exp
            deviance_mat[b, 1] = np.sqrt(mean_squared_error(subgrid_data[response_var], predictions)) # RMSE
            deviance_mat[b, 2] = r2 # R2

            # -------------------------------------------------------
            # 8. STORE PARTIAL DEPENDENCE PLOT DATA ----
            # -------------------------------------------------------
            pdp_storage_boot = []
            predictor_means = subgrid_data[predictors].mean().to_dict()

            for j, pred_name in enumerate(predictors):
                pdp_grid = pd.DataFrame([predictor_means] * 100)
                pdp_grid[pred_name] = pdp_env_ranges_df[pdp_env_ranges_df['Predictor'] == pred_name]['Env_Value'].values
                pdp_predictions = xgb_model.predict(xgb.DMatrix(pdp_grid[predictors]))

                pd_array[:, j, b] = pdp_predictions

                pdp_df = pd.DataFrame({
                    'Predictor': pred_name, 'Env_Value': pdp_grid[pred_name],
                    'Replicate': f"Rep_{b+1}", 'PDP_Value': pdp_predictions
                })
                if 'X' in pdp_env_ranges_df.columns:
                    pdp_df['X'] = pdp_env_ranges_df[pdp_env_ranges_df['Predictor'] == pred_name]['X'].values
                    pdp_df['Y'] = pdp_env_ranges_df[pdp_env_ranges_df['Predictor'] == pred_name]['Y'].values
                    pdp_df['FID'] = pdp_env_ranges_df[pdp_env_ranges_df['Predictor'] == pred_name]['FID'].values
                pdp_storage_boot.append(pdp_df)

            all_pdp_long_list.append(pd.concat(pdp_storage_boot, ignore_index=True))

        print("DIAGNOSTIC: Bootstrap loop finished.")
        pdp_long_df = pd.concat(all_pdp_long_list, ignore_index=True)

        # -------------------------------------------------------
        # 8.5. PROCESS BOOTSTRAP PREDICTIONS & CALCULATE STATISTICS
        # -------------------------------------------------------
        print(f"DIAGNOSTIC: Processing bootstrap results. Array dimensions: {boot_array.shape}")

        mean_prediction = np.mean(boot_array, axis=2).flatten()
        uncertainty_sd = np.std(boot_array, axis=2).flatten()
        if n_boot == 1: uncertainty_sd = np.zeros_like(uncertainty_sd)

        boot_df = pd.DataFrame({
            'FID': subgrid_data['FID'], 'X': subgrid_data['X'], 'Y': subgrid_data['Y'],
            'b.change_actual': subgrid_data[response_var],
            'Mean_Prediction': mean_prediction, 'Uncertainty_SD': uncertainty_sd
        })
        print(f"DIAGNOSTIC: Created final bootstrap data frame with {len(boot_df)} rows.")

        # -------------------------------------------------------
        # 9. SAVE OUTPUTS----
        # -------------------------------------------------------
        os.makedirs(tile_dir, exist_ok=True)
        print(f"DIAGNOSTIC: Writing outputs to {tile_dir}")

        cv_results_df.to_parquet(os.path.join(tile_dir, f"cv_results_{pair}.parquet"))
        pd.DataFrame(deviance_mat, columns=["Dev.Exp", "RMSE", "R2"]).to_parquet(os.path.join(tile_dir, f"deviance_{pair}.parquet"))
        pd.DataFrame(influence_mat, index=predictors, columns=[f"Rep_{i+1}" for i in range(n_boot)]).reset_index().rename(columns={'index': 'Predictor'}).to_parquet(os.path.join(tile_dir, f"influence_{pair}.parquet"), index=False)
        predictor_ranges.to_parquet(os.path.join(tile_dir, f"predictor_ranges_{pair}.parquet"))
        boot_df.to_parquet(os.path.join(tile_dir, f"bootstraps_{pair}.parquet"))
        pdp_long_df.to_parquet(os.path.join(tile_dir, f"pdp_data_long_{pair}.parquet"))
        pdp_env_ranges_df.to_parquet(os.path.join(tile_dir, f"pdp_env_ranges_{pair}.parquet"))

        # Save Rasters
        res = 10 # Assuming 10m resolution
        min_x, min_y, max_x, max_y = subgrid_gdf.total_bounds
        out_shape = (int(np.ceil((max_y - min_y) / res)), int(np.ceil((max_x - min_x) / res)))
        transform = from_origin(min_x, max_y, res, res)
        for col, filename in [('Mean_Prediction', f"Mean_Boots_Prediction_{pair}.tif"), ('Uncertainty_SD', f"Uncertainty_SD_{pair}.tif")]:
            raster_data = np.full(out_shape, np.nan, dtype=np.float32)
            for _, row in boot_df.dropna(subset=[col]).iterrows():
                r, c = rasterio.transform.rowcol(transform, row['X'], row['Y'])
                if 0 <= r < out_shape[0] and 0 <= c < out_shape[1]: raster_data[r, c] = row[col]
            with rasterio.open(os.path.join(tile_dir, filename), 'w', driver='GTiff', height=out_shape[0], width=out_shape[1], count=1, dtype=raster_data.dtype, crs=grid_crs, transform=transform, nodata=np.nan) as dst:
                dst.write(raster_data, 1)

        final_model = xgb.train(params=xgb_params, dtrain=dtrain_full, num_boost_round=best_iteration)
        final_model.save_model(os.path.join(tile_dir, f"xgb_model_{pair}.json"))

        # GENERATE DIAGNOSTIC PLOT OF MODEL FIT ---
        plot_data = boot_df.dropna(subset=['Mean_Prediction'])
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.scatterplot(data=plot_data, x='b.change_actual', y='Mean_Prediction', alpha=0.3, color="darkblue", ax=ax)
        ax.axline((0, 0), slope=1, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"Model Fit for Tile: {tile_id} | Pair: {pair}")
        ax.set_xlabel("Actual Change (m)")
        ax.set_ylabel("Mean Predicted Change (m)")
        subtitle = f"Mean R-squared = {np.nanmean(deviance_mat[:, 2]):.3f} | Mean RMSE = {np.nanmean(deviance_mat[:, 1]):.3f}"
        fig.suptitle(subtitle, y=0.92, fontsize=10)
        plot_dir = os.path.join(output_dir_train, tile_id, "diagnostic_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"model_fit_{pair}.png"), dpi=150)
        plt.close()
        print("DIAGNOSTIC: All outputs saved successfully.")

    except Exception as e:
        print(f"FATAL ERROR in training for Tile: {tile_id} | Pair: {pair} | {e}", exc_info=True)


# ==============================================================================
#
#                           POST-TRAINING FUNCTIONS
#
# ==============================================================================
def diagnose_xgb_run(output_dir_train, tile_id, year_pair):
    """Diagnoses an XGBoost model run by inspecting its inputs and outputs."""
    print(f"--- Starting Diagnosis for Tile: {tile_id} | Year Pair: {year_pair} ---\n")
    base_path = os.path.join(output_dir_train, tile_id)
    if not os.path.exists(base_path):
        print(f"Directory for the specified tile_id does not exist: {base_path}")
        return

    paths = {
        'training_data': os.path.join(base_path, f"{tile_id}_training_clipped_data.geoparquet"),
        'boruta_results': os.path.join(base_path, f"boruta_selection_{year_pair}.pkl"),
        'xgb_model': os.path.join(base_path, f"xgb_model_{year_pair}.json"),
        'deviance': os.path.join(base_path, f"deviance_{year_pair}.parquet"),
        'influence': os.path.join(base_path, f"influence_{year_pair}.parquet")
    }

    missing_files = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing_files:
        print(f"One or more required files are missing: {', '.join(missing_files)}")
        return

    print("1. Loading and preparing data...")
    training_data = pd.read_parquet(paths['training_data'])
    with open(paths['boruta_results'], 'rb') as f: boruta_results = pickle.load(f)
    predictors = boruta_results['confirmed_predictors']
    response_var = f"b.change.{year_pair}"

    subgrid_data = training_data[predictors + [response_var]].copy()
    for col in subgrid_data.columns: subgrid_data[col] = pd.to_numeric(subgrid_data[col], errors='coerce')
    subgrid_data.dropna(inplace=True)

    print("2. Performing data diagnostics...")
    predictor_variances = subgrid_data[predictors].var()
    data_summary = {
        'n_rows': len(subgrid_data), 'n_predictors': len(predictors),
        'response_variable_summary': subgrid_data[response_var].describe().to_dict(),
        'predictor_variances': predictor_variances.sort_values(ascending=False).to_dict(),
        'predictors_with_zero_variance': predictor_variances[predictor_variances == 0].index.tolist(),
        'unique_value_counts': subgrid_data[predictors].nunique().sort_values().to_dict()
    }

    print("3. Performing model diagnostics...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(paths['xgb_model'])
    deviance_df = pd.read_parquet(paths['deviance'])
    influence_df = pd.read_parquet(paths['influence'])

    mean_influence = influence_df.set_index('Predictor').mean(axis=1).sort_values(ascending=False)

    model_summary = {
        'model_class': str(type(xgb_model)),
        'number_of_trees': xgb_model.num_boosted_rounds(),
        'top_5_influential_predictors': mean_influence.head(5).to_dict(),
        'mean_deviance_explained': deviance_df['Dev.Exp'].mean(),
        'mean_rmse': deviance_df['RMSE'].mean(),
        'mean_r_squared': deviance_df['R2'].mean()
    }

    print("4. Compiling final report...\n")
    final_report = {'data_diagnostics': data_summary, 'model_diagnostics': model_summary}
    # Using json.dumps to pretty-print the dictionary
    print(json.dumps(final_report, indent=2))
    print("\n--- Diagnosis Complete ---\n")
    return final_report

def create_xgboost_performance_report(output_dir_train, tile_ids=None, cv_report_filename="xgboost_cv_summary_report.png", perf_report_filename="xgboost_performance_summary_report.png"):
    """Creates visual reports for XGBoost model performance."""
    print("📊 Starting XGBoost performance report generation...")

    def find_files(pattern):
        all_files = [os.path.join(root, name) for root, _, files in os.walk(output_dir_train) for name in files if name.startswith(pattern) and name.endswith(".parquet")]
        if tile_ids:
            # Correctly filter files based on tile_ids in their path
            return [f for f in all_files if any(f'/{tid}/' in f.replace('\\', '/') for tid in tile_ids)]
        return all_files

    cv_files = find_files("cv_results_")
    deviance_files = find_files("deviance_")

    if cv_files:
        cv_df = pd.concat([pd.read_parquet(f) for f in cv_files], ignore_index=True)
        if not cv_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle("XGBoost Cross-Validation Summary", fontsize=16)
            sns.histplot(cv_df['best_iteration'], ax=axes[0, 0], color="cornflowerblue")
            axes[0, 0].set_title("Distribution of Best Training Iterations")
            sns.barplot(data=cv_df, x='year_pair', y='test_rmse_mean', ax=axes[0, 1], color="darkseagreen", errorbar=None)
            axes[0, 1].set_title("Average Test RMSE by Year Pair")
            sns.barplot(data=cv_df, x='year_pair', y='test_mae_mean', ax=axes[1, 0], color="darkkhaki", errorbar=None)
            axes[1, 0].set_title("Average Test MAE by Year Pair")
            sns.scatterplot(data=cv_df, x='test_rmse_mean', y='test_mae_mean', ax=axes[1, 1], alpha=0.3)
            axes[1, 1].set_title("CV Error: RMSE vs. MAE")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(output_dir_train, cv_report_filename))
            plt.close()
            print(f"Saved CV report to: {cv_report_filename}")

    if deviance_files:
        deviance_df = pd.concat([pd.read_parquet(f).assign(year_pair=f.split('_')[-1].split('.')[0], tile_id=f.split(os.sep)[-2]) for f in deviance_files], ignore_index=True)
        if not deviance_df.empty:
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            fig.suptitle("XGBoost Final Model Performance Summary", fontsize=16)
            sns.histplot(deviance_df['Dev.Exp'], ax=axes[0, 0], color="slateblue")
            axes[0, 0].set_title("Distribution of Deviance Explained")
            sns.histplot(deviance_df['RMSE'], ax=axes[0, 1], color="tomato")
            axes[0, 1].set_title("Distribution of Final Model RMSE")
            sns.histplot(deviance_df['R2'], ax=axes[1, 0], color="gold")
            axes[1, 0].set_title("Distribution of Final Model R-squared")
            sns.boxplot(data=deviance_df, x='year_pair', y='R2', ax=axes[1, 1], color="lightblue")
            axes[1, 1].set_title("R-squared Performance by Year Pair")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(os.path.join(output_dir_train, perf_report_filename))
            plt.close()
            print(f"Saved performance report to: {perf_report_filename}")

def create_pdp_report_magnitude(output_dir, year_pairs, tile_id=None, exclude_predictors=None, plot_output_dir=None, n_bins=50):
    """Creates a PDP magnitude plot with a fixed y-axis across facets."""
    print("📊 Starting PDP Magnitude report generation...")
    if plot_output_dir is None: plot_output_dir = output_dir

    for pair in year_pairs:
        pdp_files = [os.path.join(root, name) for root, _, files in os.walk(output_dir) for name in files if name == f"pdp_data_long_{pair}.parquet"]
        if tile_id: pdp_files = [f for f in pdp_files if f'/{tile_id}/' in f.replace('\\','/')]
        if not pdp_files: continue

        df = pd.concat([pd.read_parquet(f) for f in pdp_files], ignore_index=True)
        if exclude_predictors:
            pattern = '|'.join(exclude_predictors)
            df = df[~df['Predictor'].str.contains(pattern)]
        if df.empty: continue

        binned_df = df.groupby('Predictor').apply(lambda x: x.assign(Env_Bin=pd.cut(x['Env_Value'], bins=n_bins))).reset_index(drop=True)
        pdp_summary = binned_df.groupby(['Predictor', 'Env_Bin'], observed=False).agg(
            Env_Value_Mid=('Env_Value', 'mean'), PDP_Mean=('PDP_Value', 'mean'),
            PDP_Min=('PDP_Value', 'min'), PDP_Max=('PDP_Value', 'max')
        ).dropna().reset_index()

        g = sns.FacetGrid(pdp_summary, col="Predictor", col_wrap=3, sharey=True, aspect=1.5, height=4)
        g.map_dataframe(lambda data, color: plt.fill_between(data['Env_Value_Mid'], data['PDP_Min'], data['PDP_Max'], alpha=0.3, color='grey'))
        g.map_dataframe(sns.lineplot, x='Env_Value_Mid', y='PDP_Mean', color='black')
        g.set_titles("{col_name}")
        g.set_axis_labels("Model Predictor Value", "Mean Elevation Change (m)")
        plt.savefig(os.path.join(plot_output_dir, f"Overall_PDP_Magnitude_Report_{pair}.png"), dpi=150)
        plt.close()
        print(f"Saved magnitude plot for {pair}")

def create_pdp_report_shape(output_dir, year_pairs, tile_id=None, plot_output_dir=None, n_bins=50):
    """Creates a PDP shape plot with a free y-axis for each facet."""
    print("📊 Starting PDP Shape report generation...")
    if plot_output_dir is None: plot_output_dir = output_dir

    for pair in year_pairs:
        pdp_files = [os.path.join(root, name) for root, _, files in os.walk(output_dir) for name in files if name == f"pdp_data_long_{pair}.parquet"]
        if tile_id: pdp_files = [f for f in pdp_files if f'/{tile_id}/' in f.replace('\\','/')]
        if not pdp_files: continue

        df = pd.concat([pd.read_parquet(f) for f in pdp_files], ignore_index=True)
        if df.empty: continue

        binned_df = df.groupby('Predictor').apply(lambda x: x.assign(Env_Bin=pd.cut(x['Env_Value'], bins=n_bins))).reset_index(drop=True)
        pdp_summary = binned_df.groupby(['Predictor', 'Env_Bin'], observed=False).agg(
            Env_Value_Mid=('Env_Value', 'mean'), PDP_Mean=('PDP_Value', 'mean'),
            PDP_Min=('PDP_Value', 'min'), PDP_Max=('PDP_Value', 'max')
        ).dropna().reset_index()

        g = sns.FacetGrid(pdp_summary, col="Predictor", col_wrap=3, sharey=False, sharex=False, aspect=1.5, height=4)
        g.map_dataframe(lambda data, color: plt.fill_between(data['Env_Value_Mid'], data['PDP_Min'], data['PDP_Max'], alpha=0.3, color='grey'))
        g.map_dataframe(sns.lineplot, x='Env_Value_Mid', y='PDP_Mean', color='black')
        g.set_titles("{col_name}")
        g.set_axis_labels("Model Predictor Value", "Mean Elevation Change (m)")
        plt.savefig(os.path.join(plot_output_dir, f"Overall_PDP_Shape_Report_{pair}.png"), dpi=150)
        plt.close()
        print(f"Saved shape plot for {pair}")

# ==============================================================================
#
#                                 MAIN EXECUTION
#
# ==============================================================================
def main_workflow():
    """
    Main function to execute the entire modeling workflow.
    """

    specific_tile_ids = ["BH4RZ577_2"] 

    # Step 1: Run Predictor Selection
    select_predictors_boruta_py(output_dir_train, year_pairs)
    create_boruta_summary_report(output_dir_train)

    # # Step 2: Run the Main Model Training
    # model_train_full_spatial_cv_py(training_sub_grids_utm_path, output_dir_train, year_pairs, block_size_m, n_boot=5, n_folds=3)

    # # Step 3: Run Post-Training Reporting and Diagnostics
    # create_xgboost_performance_report(output_dir_train, tile_ids=specific_tile_ids)
    # diagnose_xgb_run(output_dir_train, "BH4RZ577_2", "2004_2006")
    # create_pdp_report_magnitude(output_dir_train, years)
    # create_pdp_report_shape(output_dir_train, years)

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    client = Client(n_workers=7, threads_per_worker=2, memory_limit="32GB")
    print(f"Dask Client started: {client.dashboard_link}")

    main_workflow()

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(10) 
    stats.dump_stats('workflow_performance.prof')
    print("\nPerformance profile saved to 'workflow_performance.prof'")
    print("To view the full report, you can use a profiler viewer like snakeviz: `snakeviz workflow_performance.prof`")

    client.close()
