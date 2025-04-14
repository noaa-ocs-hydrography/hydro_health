
# ********** the below code is for stage 2 only**********
# MODEL TRAINING, GENERATING TILE PDPS AND  EVALUATION METRICS (STAGE 2): (This will be its own engine)
# 1. Train the model over all sub grids
# 2. Create Partial Dependance plots from the model data - WE HAVE THEM ALL FOR EACH TILE, BUT THE AVERAGE PDP OVER STUDY AREA NEEDs COMPUTED
# 3. Evaluate model performance from Ranger RF model summary data - PERFORMANCE METRICS ARE SAVED TO CSV, BUT MODEL AND METRICS NEED INTEROGATED TO INFORM CHANGES BEFORE PREDICTION CODE RUN 



# Load Packages
require(raster); require(terra);require(xml2)
require(dplyr); require(sp); require(ranger) 
require(sf); require(mgcv); require(stringr)
library(progress) # For a progress bar
library(pdp)
library(data.table)
library(ggplot2)
library(tidyr)
library(readr)
library(purrr)
library(terra)
library(future)
library(pbapply)



# LOAD PARAMETERS FROM CREATED FROM STAGE 1 PREPROCESSING:
grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg" # from Blue topo
training.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.tif") # for reference CRS of training grid
prediction.mask <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.tif")
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"
output_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles"
input_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed" # raster data 
input_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # raster data                                                                                                                 training_sub_grids <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/intersecting_sub_grids.gpkg")
prediction_sub_grids <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/intersecting_sub_grids.gpkg")
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
# GPKG tile grids
training_sub_grids <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/intersecting_sub_grids.gpkg")
prediction_sub_grids <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/intersecting_sub_grids.gpkg")


# i = 1

# 1. Model training over all subgrids ----
process_tiles <- function(tiles_df, output_dir_train, year_pairs, use_leaps_subset = FALSE, 
                          x_label_size = 0.8, y_label_size = 0.9, tick_label_size = 0.8) {
  cat("Starting processing of all tiles...\n")
  
  # Static predictors (used across all year pairs)
  static_predictors <- c("prim_sed_layer", "grain_size_layer", "survey_end_date")
  
  results_summary <- list()
  
  for (i in seq_len(nrow(tiles_df))) {
    tile <- tiles_df[i, ]
    tile_id <- tile$tile_id
    tile_dir <- file.path(output_dir_train, tile_id)
    
    if (!dir.exists(tile_dir)) {
      dir.create(tile_dir, showWarnings = FALSE, recursive = TRUE)
    }
    
    cat("Processing tile:", tile_id, "\n")
    
    # Load corresponding training data
    training_data_path <- file.path(tile_dir, paste0(tile_id, "_training_clipped_data.rds"))
    if (!file.exists(training_data_path)) {
      cat("  Training data missing for tile:", tile_id, "\n")
      next
    }
    training_data <- readRDS(training_data_path)
    
    for (pair in year_pairs) {
      # Parse year pair
      start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
      end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
      
      # Select dynamic predictors matching the year pair
      dynamic_predictors <- c(
        paste0("bathy_", start_year), paste0("bathy_", end_year),
        paste0("slope_", start_year), paste0("slope_", end_year),
        paste0("hurr_count_", pair), paste0("hurr_strength_", pair),
        paste0("tsm_", pair)
      )
      
      # Combine dynamic and static predictors
      predictors <- intersect(c(dynamic_predictors, static_predictors), names(training_data))
      response_var <- paste0("b.change.", pair)
      
      # Filter and select data for the year pair
      subgrid_data <- training_data %>%
        filter(
          !is.na(!!sym(paste0("bathy_", start_year))) & 
            !is.na(!!sym(paste0("bathy_", end_year)))
        ) %>%
        select(all_of(c(predictors, response_var))) %>%
        drop_na()
      
      if (nrow(subgrid_data) == 0) {
        cat("  No valid data for year pair:", pair, "\n")
        next
      }
      
      # Optional: Subset selection using leaps
      if (use_leaps_subset) {
        formula <- as.formula(paste0(response_var, " ~ ."))
        leaps_model <- leaps::regsubsets(formula, data = subgrid_data, nvmax = length(predictors))
        best_predictors <- summary(leaps_model)$which[which.max(summary(leaps_model)$adjr2), -1]
        predictors <- names(best_predictors[best_predictors])
        cat("  Selected predictors:", paste(predictors, collapse = ", "), "\n")
      }
      
      # Train Random Forest
      formula <- as.formula(paste0(response_var, " ~ ", paste(predictors, collapse = " + ")))
      rf_model <- ranger(
        formula = formula,
        data = subgrid_data,
        num.trees = 500,
        mtry = floor(sqrt(length(all.vars(formula)))),
        importance = "impurity"
      )
      
      # Save model
      model_path <- file.path(tile_dir, paste0("model_", pair, ".rds"))
      saveRDS(rf_model, model_path)
      cat("  Model saved for year pair:", pair, "\n")
      
      # Initialize PDP data
      pdp_data <- list()
      tryCatch({
        for (pred in predictors) {
          pred_range <- seq(
            min(subgrid_data[[pred]], na.rm = TRUE),
            max(subgrid_data[[pred]], na.rm = TRUE),
            length.out = 50
          )
          pdp_df <- subgrid_data[1, , drop = FALSE][rep(1, 50), ]
          pdp_df[[pred]] <- pred_range
          predictions <- predict(rf_model, data = pdp_df)$predictions
          pdp_data[[pred]] <- data.frame(Predictor = pred, Value = pred_range, Prediction = predictions)
        }
      }, error = function(e) {
        cat("  Warning: Issue generating PDP data for year pair:", pair, "\n", "   Message:", e$message, "\n")
      })
      
      # Save PDP data
      pdp_path <- file.path(tile_dir, paste0("pdp_", pair, ".rds"))
      tryCatch({
        saveRDS(do.call(rbind, pdp_data), pdp_path)
        cat("  PDP saved for year pair:", pair, "\n")
      }, error = function(e) {
        cat("  Error saving PDP data for year pair:", pair, "\n", "   Message:", e$message, "\n")
      })
      
      # Plot PDP
      if (length(pdp_data) > 0) {
        pdp_plot_path <- file.path(tile_dir, paste0("pdp_", pair, ".jpeg"))
        jpeg(pdp_plot_path, width = 2400, height = 1800, res = 300)
        n_predictors <- length(predictors)
        rows <- ceiling(sqrt(n_predictors))
        cols <- ceiling(n_predictors / rows)
        par(mfrow = c(rows, cols), mar = c(4, 4, 2, 2), cex.lab = y_label_size, cex.axis = tick_label_size)
        
        for (pred in names(pdp_data)) {
          pdp <- pdp_data[[pred]]
          tryCatch({
            plot(
              pdp$Value, pdp$Prediction,
              type = "l", col = "blue", lwd = 2,
              xlab = "",  # Predictor name will be added below
              ylab = "Bathy Change (m)",
              main = "",  # Prevent large text overlap
              ylim = range(sapply(pdp_data, function(d) range(d$Prediction, na.rm = TRUE)))
            )
            mtext(
              side = 1, text = pdp$Predictor, line = 2.5, cex = x_label_size  # Adjustable X-axis label size
            )
          }, error = function(e) {
            cat("  Warning: Issue plotting predictor:", pred, "\n", "   Message:", e$message, "\n")
          })
        }
        dev.off()
        cat("  PDP plot saved as JPEG for year pair:", pair, "\n")
      } else {
        cat("  Warning: PDP plotting skipped for year pair:", pair, "\n")
      }
      
      # Collect performance metrics
      adjusted_r2 <- 1 - (1 - rf_model$r.squared) * (nrow(subgrid_data) - 1) / (nrow(subgrid_data) - length(predictors) - 1)
      residual_error <- sqrt(mean((rf_model$predictions - subgrid_data[[response_var]])^2, na.rm = TRUE))
      importance_df <- data.frame(
        Variable = names(rf_model$variable.importance),
        Importance = rf_model$variable.importance
      )
      importance_df <- importance_df[order(-importance_df$Importance), ]
      
      results_summary[[paste0(tile_id, "_", pair)]] <- data.frame(
        Tile = tile_id,
        YearPair = pair,
        R2 = rf_model$r.squared,
        AdjustedR2 = adjusted_r2,
        ResidualError = residual_error,
        Importance = importance_df$Importance,
        Variable = importance_df$Variable
      )
    }
  }
  
  # Combine and save results summary
  results_df <- do.call(rbind, results_summary)
  results_csv_path <- file.path(output_dir_train, "model_performance_summary.csv")
  tryCatch({
    write.csv(results_df, results_csv_path, row.names = FALSE)
    cat("Processing complete. Summary saved.\n")
  }, error = function(e) {
    cat("  Error saving results summary CSV\n", "   Message:", e$message, "\n")
  })
}



#Function call
Sys.time() # low memory utilisation, ~5GB, approx 17hrs to run through 51 tiles
process_tiles(
  tiles_df = training_sub_grids,
  output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles",
  year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022"),
  x_label_size = 0.7,
  y_label_size = 0.9,
  tick_label_size = 0.8
)
Sys.time()


# Test model output 
mod1 <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/Tile_BH4S2577_4/model_2015_2022.rds")

# 2. Create an averaged Partial Dependence Plot for the whole training area----
# JPEGS and .rds files (pdp_2015_2022.rds) are saved to each training tile folder
# by applying the below we should end up with 5 outputs, one for each year pair, and one overall. 
# (i) take an average across all tiles, for the same year
# (ii) take an average across all years and tiles


#3. Evaluate performance metrics----
# when the model para-maters are tweeked, the model performance changes. We want to evaluate the model performance, to make changes to
# to the model training stage, before we progress to the model prediction.

# in the main tile directory, a csv has been saved which summarize the following metrics: R2, Adjusted R2, Residual Error, and Variable importance. 
# we want to examine these metrics, by creating acceptable thresholds to identify tiles that fall below this threshold. 
# we wan to to look at the variable importance and remove model variables with poor contribution, then re-run the model training again. 

# we are trying to achieve a parsimonious model - the best performance for the fewest predictors. 


# ( Stage 3 is next)