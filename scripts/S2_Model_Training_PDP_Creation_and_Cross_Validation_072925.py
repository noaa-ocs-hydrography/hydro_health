import os
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import xgboost as xgb
import joblib
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
from joblib import Parallel, delayed
import warnings

# Ignore specific warnings from libraries for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module='geopandas')
warnings.filterwarnings("ignore", category=FutureWarning)

# LOAD PARAMETERS CREATED FROM  PREPROCESSING STAGE if not already loaded:----
# update to ER3 directories
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
years <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
block_size <- 200 # The 200m block size that worked well for the scale of our sub-grid tile size
grid_gpkg <- st_read("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg") # from Blue topo
#
training_sub_grids_UTM <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
prediction_sub_grids_UTM <-st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
#
training.mask.UTM <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif") # for reference CRS of training grid
prediction.mask.UTM <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif")
#
training.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.021425.Rds")# spatial DF of extent
prediction.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/prediction.mask.df.021425.Rds")# spatial DF of extent
#
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"


# ==============================================================================
#
#                            PRE-TRAINING FUNCTIONS
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
#' @param training_sub_grids_UTM A data frame or sf object with tile IDs.
#' @param output_dir_train The base directory where training data is stored and results will be saved.
#' @param year_pairs A list of year pairs to process (e.g., "2000_2005").
#' @param max_runs The maximum number of iterations for the Boruta algorithm.
#'
#' @return A list containing the paths to the selection results for each tile and year pair.

# ------------------------------------------------------------------------------
# 1. BORUTA PREDICTOR SELECTION
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
def select_predictors_boruta_py(training_sub_grids_utm_path, output_dir_train, year_pairs, max_runs=100):
    """
    Performs predictor selection using the Boruta algorithm in parallel.
    This function identifies important predictors for a given response variable.
    """
    print("\n🚀 Starting Predictor Selection with Boruta...\n")
    
    log_file = os.path.join(output_dir_train, "predictor_selection_log.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_file, filemode='w')

    training_sub_grids_utm = gpd.read_file(training_sub_grids_utm_path)
    tile_ids = list(training_sub_grids_utm['tile_id'])
    
    num_cores = joblib.cpu_count() - 1
    if num_cores < 1: num_cores = 1

    print(f"Running Boruta on {num_cores} cores for {len(tile_ids)} tiles...")

    Parallel(n_jobs=num_cores, verbose=10)(
        delayed(_worker_boruta_tile)(tile_id, output_dir_train, year_pairs, max_runs)
        for tile_id in tile_ids
    )
    
    print("\n✅ Boruta Predictor Selection Complete! Check `predictor_selection_log.txt` for details.\n")

def _worker_boruta_tile(tile_id, output_dir_train, year_pairs, max_runs):
    """Helper worker function for running Boruta on a single tile."""
    for pair in year_pairs:
        try:
            training_data_path = os.path.join(output_dir_train, tile_id, f"{tile_id}_training_clipped_data.geoparquet")
            if not os.path.exists(training_data_path):
                logging.warning(f"Missing training data for tile: {tile_id}, skipping.")
                continue

            training_data = pd.read_parquet(training_data_path)
            
            start_year, end_year = pair.split('_')
            response_var = f"b.change.{pair}"

            if response_var not in training_data.columns:
                logging.error(f"Response variable '{response_var}' missing for Tile: {tile_id}, skipping.")
                continue

            static_predictors = ["grain_size_layer", "prim_sed_layer"]
            dynamic_predictors = [
                f"bathy_{start_year}", f"bathy_{end_year}",
                f"slope_{start_year}", f"slope_{end_year}",
                f"hurr_count_{pair}", f"hurr_strength_{pair}", f"tsm_{pair}"
            ]
            rugosity_start = [col for col in training_data.columns if col.startswith(f"Rugosity_nbh") and col.endswith(start_year)]
            rugosity_end = [col for col in training_data.columns if col.startswith(f"Rugosity_nbh") and col.endswith(end_year)]
            
            all_predictors = list(set(static_predictors + dynamic_predictors + rugosity_start + rugosity_end) & set(training_data.columns))
            
            sub_data = training_data[all_predictors + [response_var]].dropna().reset_index(drop=True)

            if len(sub_data) < 50 or sub_data[response_var].nunique() <= 1:
                logging.warning(f"Skipping Boruta for Tile: {tile_id} | Pair: {pair} - Insufficient data.")
                continue
            
            logging.info(f"🏃 Running Boruta for Tile: {tile_id} | Pair: {pair}")
            
            rf = RandomForestRegressor(n_jobs=1, max_depth=5, random_state=1)
            feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, max_iter=max_runs, random_state=1)
            
            X = sub_data[all_predictors].values
            y = sub_data[response_var].values
            
            feat_selector.fit(X, y)
            
            confirmed_preds = list(np.array(all_predictors)[feat_selector.support_])
            
            decision_map = {True: 'Confirmed', False: 'Rejected'}
            tentative_indices = np.where(feat_selector.support_weak_)[0]
            
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
            save_path = os.path.join(tile_dir, f"boruta_selection_{pair}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(output_list, f)

        except Exception as e:
            logging.error(f"❌ ERROR in Boruta for Tile: {tile_id} | Pair: {pair} | {e}", exc_info=True)


#BORUTA PREDICTOR SELECTION SUMMARY REPORT (before running model examine) ----
#' Create Visual Reports for Boruta Predictor Selection
#'
#' This function scans a directory for Boruta selection results (`.rds` files),
#' aggregates them, and generates two separate summary plots using R's base
#' plotting functions:
#' 1. An overall summary of predictor confirmation, rejection, and importance.
#' 2. A breakdown of the most important predictors for each year-pair.
#'
#' @param output_dir_train The base directory where the tile folders are located.
#' @param overall_report_filename The name of the output PNG for the overall summary.
#' @param by_year_report_filename The name of the output PNG for the year-pair breakdown.
#' @param top_n The number of top predictors to display in the plots.
#'
#' @return None. Two PNG files are saved to the `output_dir_train`.

# ------------------------------------------------------------------------------
# 2. BORUTA SUMMARY REPORT
# ------------------------------------------------------------------------------
def create_boruta_summary_report(output_dir_train, overall_report_filename="boruta_summary_report.png", by_year_report_filename="boruta_importance_by_year.png", top_n=10):
    """Creates visual reports summarizing Boruta predictor selection results."""
    print("📊 Starting Boruta summary report generation...")
    
    selection_files = [os.path.join(root, name)
                       for root, _, files in os.walk(output_dir_train)
                       for name in files if name.startswith("boruta_selection_") and name.endswith(".pkl")]

    if not selection_files:
        print("No 'boruta_selection_*.pkl' files found. Please run selection first.")
        return

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
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Boruta Predictor Selection Overall Summary", fontsize=16)

    confirmed_counts = combined_df[combined_df['decision'] == 'Confirmed']['predictor'].value_counts().nlargest(top_n)
    sns.barplot(x=confirmed_counts.values, y=confirmed_counts.index, ax=axes[0, 0], color='darkgreen', orient='h')
    axes[0, 0].set_title(f"Top {top_n} Most Confirmed Predictors")

    rejected_counts = combined_df[combined_df['decision'] == 'Rejected']['predictor'].value_counts().nlargest(top_n)
    sns.barplot(x=rejected_counts.values, y=rejected_counts.index, ax=axes[0, 1], color='firebrick', orient='h')
    axes[0, 1].set_title(f"Top {top_n} Most Rejected Predictors")

    decision_summary = combined_df['decision'].value_counts()
    sns.barplot(x=decision_summary.index, y=decision_summary.values, ax=axes[1, 0], palette={"Confirmed": "darkgreen", "Tentative": "darkorange", "Rejected": "firebrick"})
    axes[1, 0].set_title("Overall Decision Frequency")

    importance_summary = combined_df.groupby('predictor')['meanImp'].mean().nlargest(top_n)
    sns.barplot(x=importance_summary.values, y=importance_summary.index, ax=axes[1, 1], color='steelblue', orient='h')
    axes[1, 1].set_title(f"Top {top_n} Predictors by Avg. Importance")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir_train, overall_report_filename), dpi=100)
    plt.close()
    print(f"Saved overall report to: {overall_report_filename}")

    year_pairs = sorted(combined_df['year_pair'].unique())
    n_pairs = len(year_pairs)
    n_cols = 2
    n_rows = (n_pairs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 7 * n_rows), squeeze=False)
    fig.suptitle("Top Predictor Importance by Year Pair", fontsize=16)
    
    for i, pair in enumerate(year_pairs):
        ax = axes.flatten()[i]
        plot_data = combined_df[combined_df['year_pair'] == pair].groupby('predictor')['meanImp'].mean().nlargest(top_n)
        if not plot_data.empty:
            sns.barplot(x=plot_data.values, y=plot_data.index, ax=ax, color='darkcyan', orient='h')
            ax.set_title(f"Year Pair: {pair}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir_train, by_year_report_filename), dpi=100)
    plt.close()
    print(f"Saved by-year report to: {by_year_report_filename}")


# ==============================================================================
#
#                            MAIN TRAINING FUNCTION
#
# ==============================================================================

#' Generate CV Diagnostic Plot -- Helper Function: Generate and Save a Diagnostic Plot for Spatial CV
#' -----------------------------------------------------------------------------
#' Creates a plot to visualize the training data and the spatial blocks.
#'
#' @param sf_data The sf object containing the training data points.
#' @param block_geom The sf object of the grid blocks created for probing.
#' @param max_k The maximum number of folds possible for the geometry.
#' @param tile_id The character ID of the current tile.
#' @param pair The character ID of the current year pair.
#' @param output_dir_train The main output directory.
#' @return Invisibly returns the ggplot object.


# ==============================================================================
# ==============================================================================

def generate_cv_diagnostic_plot(sf_data, block_geom, max_k, tile_id, pair, output_dir_train):
    """
    Creates a plot to visualize the training data and the spatial blocks.
    """
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
        
        # Plot the raw data points
        sf_data.plot(ax=ax, color="grey", markersize=1.5, alpha=0.5, label="Data Points")
        
        # Overlay the spatial block boundaries
        block_geom.boundary.plot(ax=ax, color="blue", linewidth=0.7, label="CV Blocks")
        
        # Get block size from grid for subtitle
        if not block_geom.empty:
            # Estimate an approximate side length for the subtitle
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
        
        plt.savefig(os.path.join(plot_dir, f"spatial_cv_map_{pair}.png"), dpi=150)
    except Exception as e:
        logging.error(f"Could not generate CV diagnostic plot for {tile_id}/{pair}: {e}")
    finally:
        plt.close('all')

# Model Train Full Spatial with Cross Validation - This script contains a complete, refactored set of functions to train
# the XGBoost models. It is designed to:
#   1. Run a robust, parallelized spatial cross-validation to find the optimal model parameters.
#   2. Train a final model and run multiple bootstrap replicates to capture uncertainty.
#   3. Save all necessary outputs for the prediction workflow, including:
#      - The final trained model object.
#      - Raw, unsmoothed Partial Dependence Plot (PDP) data.
#      - A complete bootstrap prediction file with Mean and Standard Deviation calculated.
#      - Raster outputs for the Mean Bootstrap Prediction and its uncertainty (SD).

def model_train_full_spatial_cv_py(training_sub_grids_utm_path, output_dir_train, year_pairs, block_size_m, n_boot=20, n_folds=5):
    """Main function to run the XGBoost model training with parallel spatial CV."""
    print("\n🚀 Starting Full XGBoost Model Training with Robust Parallel Spatial CV...\n")
    
    master_log_file = os.path.join(output_dir_train, "training_log_final.txt")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=master_log_file, filemode='w')

    training_sub_grids_utm = gpd.read_file(training_sub_grids_utm_path)
    tile_ids = list(training_sub_grids_utm['tile_id'])
    grid_crs = training_sub_grids_utm.crs
    
    num_cores = joblib.cpu_count() - 1
    if num_cores < 1: num_cores = 1
    print(f"Executing training on {num_cores} cores for {len(tile_ids)} tiles...")

    Parallel(n_jobs=num_cores, verbose=10)(
        delayed(_worker_train_tile)(tile_id, output_dir_train, year_pairs, block_size_m, n_boot, n_folds, grid_crs)
        for tile_id in tile_ids
    )
    
    print("\n✅ Model Training Complete! Check `training_log_final.txt` for details.\n")

def _worker_train_tile(tile_id, output_dir_train, year_pairs, block_size_m, n_boot, n_folds, grid_crs):
    """Helper worker function for training on a single tile."""
    tile_dir = os.path.join(output_dir_train, tile_id)
    worker_log_file = os.path.join(tile_dir, f"log_worker_train_{tile_id}.txt")
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(worker_log_file, mode='w')
    file_handler.setFormatter(log_formatter)
    logger = logging.getLogger(f'worker_train_{tile_id}')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(file_handler)
    
    for pair in year_pairs:
        try:
            logger.info(f"Processing Tile: {tile_id} | Year Pair: {pair}")
            training_data_path = os.path.join(tile_dir, f"{tile_id}_training_clipped_data.geoparquet")
            boruta_results_path = os.path.join(tile_dir, f"boruta_selection_{pair}.pkl")

            if not os.path.exists(training_data_path) or not os.path.exists(boruta_results_path):
                logger.warning("Data or Boruta file missing. Skipping.")
                continue

            full_training_gdf = gpd.read_parquet(training_data_path)
            with open(boruta_results_path, 'rb') as f:
                boruta_results = pickle.load(f)
            predictors = boruta_results['confirmed_predictors']
            response_var = f"b.change.{pair}"

            if not predictors or response_var not in full_training_gdf.columns:
                logger.warning("No predictors or response var found. Skipping.")
                continue

            cols_to_keep = predictors + [response_var, "X", "Y", "FID", "geometry"]
            subgrid_gdf = full_training_gdf[list(set(cols_to_keep) & set(full_training_gdf.columns))].copy()
            for col in predictors + [response_var]:
                subgrid_gdf[col] = pd.to_numeric(subgrid_gdf[col], errors='coerce')
            subgrid_gdf.dropna(subset=(predictors + [response_var]), inplace=True)
            subgrid_gdf.reset_index(drop=True, inplace=True)

            if len(subgrid_gdf) < 100:
                logger.warning("Insufficient data. Skipping.")
                continue
            
            subgrid_data = pd.DataFrame(subgrid_gdf.drop(columns='geometry'))

            deviance_mat = np.full((n_boot, 3), np.nan, dtype=float)
            influence_mat = np.full((len(predictors), n_boot), np.nan, dtype=float)
            boot_mat = np.full((len(subgrid_data), n_boot), np.nan, dtype=float)

            pdp_env_ranges_list = [pd.DataFrame({'Env_Value': np.linspace(subgrid_data[pred].min(), subgrid_data[pred].max(), 100) if subgrid_data[pred].nunique() > 1 else np.repeat(subgrid_data[pred].min(), 100), 'Predictor': pred}) for pred in predictors]
            pdp_env_ranges_df = pd.concat(pdp_env_ranges_list, ignore_index=True)
            if len(subgrid_data) >= 100:
                sampled_coords = subgrid_data[['X', 'Y', 'FID']].head(100)
                pdp_env_ranges_df['X'] = np.tile(sampled_coords['X'], len(predictors))
                pdp_env_ranges_df['Y'] = np.tile(sampled_coords['Y'], len(predictors))
                pdp_env_ranges_df['FID'] = np.tile(sampled_coords['FID'], len(predictors))

            best_iteration = 100
            bounds = subgrid_gdf.total_bounds
            grid_cells = [box(x0, y0, x0 + block_size_m, y0 + block_size_m) for x0 in np.arange(bounds[0], bounds[2], block_size_m) for y0 in np.arange(bounds[1], bounds[3], block_size_m)]
            grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs=grid_crs)
            points_in_blocks = gpd.sjoin(subgrid_gdf, grid_gdf, how="inner", predicate="within")
            active_blocks_gdf = grid_gdf.iloc[points_in_blocks['index_right'].unique()]
            max_k = len(active_blocks_gdf)
            k_final = min(n_folds, max_k)

            try:
                if k_final < 2: raise ValueError(f"CV not possible. Only {max_k} spatial block(s) found.")
                subgrid_gdf['block_id'] = gpd.sjoin(subgrid_gdf, active_blocks_gdf, how='left', predicate='within')['index_right'].fillna(-1)
                gkf = GroupKFold(n_splits=k_final)
                fold_splits = gkf.split(subgrid_gdf, groups=subgrid_gdf['block_id'])
                best_nrounds_per_fold, rmse_per_fold, mae_per_fold = [], [], []
                
                for train_idx, test_idx in fold_splits:
                    train_set, test_set = subgrid_data.iloc[train_idx], subgrid_data.iloc[test_idx]
                    if test_set[response_var].nunique() < 2: continue
                    dtrain_fold = xgb.DMatrix(train_set[predictors], label=train_set[response_var])
                    dtest_fold = xgb.DMatrix(test_set[predictors], label=test_set[response_var])
                    evals_result = {}
                    fold_model = xgb.train(
                        params={'max_depth': 4, 'eta': 0.01, 'gamma': 1, 'objective': 'reg:squarederror', 'eval_metric': ['rmse', 'mae']},
                        dtrain=dtrain_fold, num_boost_round=1000,
                        evals=[(dtest_fold, 'test')], early_stopping_rounds=10,
                        evals_result=evals_result, verbose_eval=False
                    )
                    if fold_model.best_iteration > 0:
                        best_nrounds_per_fold.append(fold_model.best_iteration)
                        rmse_per_fold.append(evals_result['test']['rmse'][fold_model.best_iteration -1])
                        mae_per_fold.append(evals_result['test']['mae'][fold_model.best_iteration -1])
                
                if best_nrounds_per_fold:
                    best_iteration = int(np.mean(best_nrounds_per_fold))
                    cv_results_df = pd.DataFrame({'tile_id': [tile_id], 'year_pair': [pair], 'best_iteration': [best_iteration], 'test_rmse_mean': [np.mean(rmse_per_fold)], 'test_rmse_std': [np.std(rmse_per_fold)], 'test_mae_mean': [np.mean(mae_per_fold)], 'test_mae_std': [np.std(mae_per_fold)]})
                    logger.info(f"CV successful. Optimal iteration: {best_iteration}")
                else: raise ValueError("Manual CV loop failed to find any best iterations.")
            except Exception as e:
                logger.warning(f"CV SKIPPED for Tile: {tile_id} | Pair: {pair} | Error: {e}. Using default {best_iteration} rounds.")
                cv_results_df = pd.DataFrame({'tile_id': [tile_id], 'year_pair': [pair], 'best_iteration': [best_iteration], 'test_rmse_mean': [np.nan], 'test_rmse_std': [np.nan], 'test_mae_mean': [np.nan], 'test_mae_std': [np.nan]})

            dtrain_full = xgb.DMatrix(subgrid_data[predictors], label=subgrid_data[response_var])
            xgb_params = {'max_depth': 4, 'eta': 0.01, 'gamma': 1, 'subsample': 0.7, 'colsample_bytree': 0.8, 'objective': 'reg:squarederror', 'nthread': 1}
            all_pdp_long_list = []
            for b in range(n_boot):
                xgb_model = xgb.train(params=xgb_params, dtrain=dtrain_full, num_boost_round=best_iteration)
                preds = xgb_model.predict(dtrain_full)
                boot_mat[:, b] = preds
                r2 = r2_score(subgrid_data[response_var], preds)
                deviance_mat[b, 0], deviance_mat[b, 2] = r2, r2
                deviance_mat[b, 1] = np.sqrt(mean_squared_error(subgrid_data[response_var], preds))
                importance = xgb_model.get_score(importance_type='gain')
                for i, pred_name in enumerate(predictors): influence_mat[i, b] = importance.get(pred_name, 0)
                
                predictor_means = subgrid_data[predictors].mean().to_dict()
                pdp_storage_boot = []
                for pred_name in predictors:
                    pdp_grid = pd.DataFrame([predictor_means] * 100)
                    pdp_grid[pred_name] = pdp_env_ranges_df[pdp_env_ranges_df['Predictor'] == pred_name]['Env_Value'].values
                    pdp_predictions = xgb_model.predict(xgb.DMatrix(pdp_grid[predictors]))
                    pdp_storage_boot.append(pd.DataFrame({'Predictor': pred_name, 'Env_Value': pdp_grid[pred_name], 'Replicate': f"Rep_{b+1}", 'PDP_Value': pdp_predictions}))
                all_pdp_long_list.append(pd.concat(pdp_storage_boot, ignore_index=True))
            
            pdp_long_df = pd.concat(all_pdp_long_list, ignore_index=True)
            logger.info("Writing outputs...")
            cv_results_df.to_parquet(os.path.join(tile_dir, f"cv_results_{pair}.parquet"))
            pd.DataFrame(deviance_mat, columns=["Dev.Exp", "RMSE", "R2"]).to_parquet(os.path.join(tile_dir, f"deviance_{pair}.parquet"))
            pd.DataFrame(influence_mat, index=predictors, columns=[f"Rep_{i+1}" for i in range(n_boot)]).reset_index().rename(columns={'index': 'Predictor'}).to_parquet(os.path.join(tile_dir, f"influence_{pair}.parquet"), index=False)
            pdp_long_df.to_parquet(os.path.join(tile_dir, f"pdp_data_{pair}.parquet"))
            pdp_env_ranges_df.to_parquet(os.path.join(tile_dir, f"pdp_env_ranges_{pair}.parquet"))
            
            boot_df = pd.DataFrame(boot_mat, columns=[f"Boot_{i+1}" for i in range(n_boot)])
            boot_df['FID'] = subgrid_data['FID']
            full_boot_df = pd.merge(full_training_gdf[['X', 'Y', 'FID']], boot_df, on='FID', how='left')
            boot_cols = [f"Boot_{i+1}" for i in range(n_boot)]
            full_boot_df['Mean_Boots_Prediction'] = full_boot_df[boot_cols].mean(axis=1)
            full_boot_df['Uncertainty_SD'] = full_boot_df[boot_cols].std(axis=1)
            gpd.GeoDataFrame(full_boot_df, geometry=gpd.points_from_xy(full_boot_df.X, full_boot_df.Y), crs=grid_crs).to_parquet(os.path.join(tile_dir, f"bootstraps_{pair}.geoparquet"))
            
            final_model = xgb.train(params=xgb_params, dtrain=dtrain_full, num_boost_round=best_iteration)
            final_model.save_model(os.path.join(tile_dir, f"xgb_model_{pair}.json"))
            
            res = 10
            min_x, min_y, max_x, max_y = full_training_gdf.total_bounds
            out_shape = (int(np.ceil((max_y - min_y) / res)), int(np.ceil((max_x - min_x) / res)))
            transform = from_origin(min_x, max_y, res, res)
            for col, filename in [('Mean_Boots_Prediction', f"Mean_Boots_Prediction_{pair}.tif"), ('Uncertainty_SD', f"Uncertainty_SD_{pair}.tif")]:
                raster_data = np.full(out_shape, np.nan, dtype=np.float32)
                for _, row in full_boot_df.dropna(subset=[col]).iterrows():
                    r, c = rasterio.transform.rowcol(transform, row['X'], row['Y'])
                    if 0 <= r < out_shape[0] and 0 <= c < out_shape[1]: raster_data[r, c] = row[col]
                with rasterio.open(os.path.join(tile_dir, filename), 'w', driver='GTiff', height=out_shape[0], width=out_shape[1], count=1, dtype=raster_data.dtype, crs=grid_crs, transform=transform, nodata=np.nan) as dst:
                    dst.write(raster_data, 1)
        except Exception as e:
            logger.critical(f"FATAL ERROR in training for Tile: {tile_id} | Pair: {pair} | {e}", exc_info=True)


# ==============================================================================
#
#                           POST-TRAINING FUNCTIONS
#
# ==============================================================================
# ==============================================================================

#' Diagnose XGBoost Model and Input Data (Diagnostic and error checking tool)
#'
#' This function loads all the final outputs for a specific tile and year-pair
#' and compiles a diagnostic summary. It is designed to help troubleshoot
#' issues, particularly with the cross-validation step, by providing a
#' snapshot of the data that was fed into the model and how the model was built.
#'
#' @param output_dir_train The base directory where the tile folders are located.
#' @param tile_id The specific tile ID you want to diagnose (e.g., "BH4RZ577_2").
#' @param year_pair The specific year pair you want to diagnose (e.g., "2004_2006").
#'
#' @return A detailed list object containing diagnostic information about the
#'   data and the trained model. This list is also printed to the console.
#'

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
    print(json.dumps(final_report, indent=2))
    print("\n--- Diagnosis Complete ---\n")
    return final_report

#' Create Visual Reports for XGBoost Model Performance
#'
#' This function scans a directory for XGBoost model outputs, including
#' cross-validation (`cv_results_*.fst`) and deviance (`deviance_*.fst`) files.
#' It can process all tiles or a specified subset and generates two separate
#' PNG images with summary plots using R's base plotting functions.
#'
#' @param output_dir_train The base directory where the tile folders are located.
#' @param tile_ids (Optional) A character vector of specific tile IDs to process.
#'   If NULL (default), all tiles in the output directory will be processed.
#' @param cv_report_filename The name of the output PNG file for CV results.
#' @param perf_report_filename The name of the output PNG file for performance results.
#'
#' @return None. Two PNG files are saved to the `output_dir_train`.

def create_xgboost_performance_report(output_dir_train, tile_ids=None, cv_report_filename="xgboost_cv_summary_report.png", perf_report_filename="xgboost_performance_summary_report.png"):
    """Creates visual reports for XGBoost model performance."""
    print("📊 Starting XGBoost performance report generation...")
    
    def find_files(pattern):
        all_files = [os.path.join(root, name) for root, _, files in os.walk(output_dir_train) for name in files if name.startswith(pattern) and name.endswith(".parquet")]
        if tile_ids:
            tile_pattern = '|'.join([f"/{tid}/" for tid in tile_ids])
            return [f for f in all_files if any(sub in f.replace('\\','/') for sub in tile_pattern.split('|'))]
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


# Create and Save Partial Dependence Plots ----
#'
#' This function aggregates PDP results from model outputs and generates a
#' multi-panel plot for each year-pair. It uses a binning strategy to create
#' a smooth, interpretable summary of the PDP trends when aggregating across
#' multiple tiles.
#'
#' @param output_dir The base directory where the tile folders are located.
#' @param year_pairs A character vector of year pairs to process (e.g., "2004_2006").
#' @param tile_id (Optional) A character string for a single tile_id to process.
#'   If NULL (default), all tiles in the output directory will be processed.
#' @param plot_output_dir The directory where the final PNG plots will be saved.
#' @param n_bins The number of bins to use for summarizing the continuous predictors.
#' @param exclude_predictors (Optional) A character vector of predictor names to
#'   exclude from the plot (e.g., c("bathy_2015", "bathy_2022")).
#'
#' @return None. PNG plot files are saved to the `plot_output_dir`.

# The MAGNITUDE PLOT - standardized Y-axis (scales = "free_x"). It accurately reflects the relative importance of the predictors
def create_pdp_report_magnitude(output_dir, year_pairs, tile_id=None, exclude_predictors=None, plot_output_dir=None, n_bins=50):
    """Creates a PDP magnitude plot with a fixed y-axis across facets."""
    print("📊 Starting PDP Magnitude report generation...")
    if plot_output_dir is None: plot_output_dir = output_dir
    
    for pair in year_pairs:
        pdp_files = [os.path.join(root, name) for root, _, files in os.walk(output_dir) for name in files if name == f"pdp_data_{pair}.parquet"]
        if tile_id: pdp_files = [f for f in pdp_files if f"/{tile_id}/" in f.replace('\\','/')]
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

# The SHAPE PLOT -  the same plot as above but with scales = "free" - to understand the nuances of each predictor's effect.
def create_pdp_report_shape(output_dir, year_pairs, tile_id=None, plot_output_dir=None, n_bins=50):
    """Creates a PDP shape plot with a free y-axis for each facet."""
    print("📊 Starting PDP Shape report generation...")
    if plot_output_dir is None: plot_output_dir = output_dir

    for pair in year_pairs:
        pdp_files = [os.path.join(root, name) for root, _, files in os.walk(output_dir) for name in files if name == f"pdp_data_{pair}.parquet"]
        if tile_id: pdp_files = [f for f in pdp_files if f"/{tile_id}/" in f.replace('\\','/')]
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



#
if __

    # --- 3. Execute the Full Workflow ---
    start_time = pd.Timestamp.now()
    print(f"\nStarting full workflow at: {start_time}")

    # Step 1: Run Predictor Selection
    select_predictors_boruta_py(grid_geopackage_path, output_dir, years)
    
    # Step 2: Run the Main Model Training
    model_train_full_spatial_cv_py(grid_geopackage_path, output_dir, years, block_size, n_boot=5, n_folds=3)
    
    # Step 3: Run Post-Training Reporting and Diagnostics
    create_boruta_summary_report(output_dir)
    create_xgboost_performance_report(output_dir, tile_ids=specific_tile_ids)
    diagnose_xgb_run(output_dir, "BH4RZ577_2", "2004_2006")
    create_pdp_report_magnitude(output_dir, years)
    create_pdp_report_shape(output_dir, years)

    end_time = pd.Timestamp.now()
    print(f"\nFinished full workflow at: {end_time}\nTotal execution time: {end_time - start_time}")
