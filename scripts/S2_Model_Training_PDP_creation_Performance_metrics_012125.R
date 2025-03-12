
# ********** the below code is for stage 2 only**********
# MODEL TRAINING, GENERATING TILE PDPS AND  EVALUATION METRICS (STAGE 2): (This will be its own engine)
# 1. Train the model over all sub grids
# 2. Create Partial Dependance plots from the model data - WE HAVE THEM ALL FOR EACH TILE, BUT THE AVERAGE PDP OVER STUDY AREA NEEDs COMPUTED
# 3. Evaluate model performance from Ranger RF model summary data - PERFORMANCE METRICS ARE SAVED TO CSV, BUT MODEL AND METRICS NEED INTEROGATED TO INFORM CHANGES BEFORE PREDICTION CODE RUN 



# ------------STAGE 2 - MODELLING STEPS--------------------# ----
## MODEL TEST TYEP 1 (Multi Varaible Spatial Random Forest)
# 1 run the models separately for each change year as is --- examine results.
# 2 then create new predictor features (time gap) and (historical change metrics) and see how this changes results 
# 3 Aggregate predicted changes with existing bathy data for all model years (may want to minus prediction from existing bathy to get just change, or add them together.... )
# 4 Fit temproal trend for cumulative change across all years 

## MODEL TEST TYPE 2 (Temporal Spatial Random Forest)
# xxxxx 


                          ## Things to consider in additional iterations ###
# 1 - Machine learning tracker, efficiency, performace score of model output tracking (Aubrey to investigate application / package to support this)
# 2 - Realise of predictions in the neatshore to be broadly reflective of realworld conditions / capturing change relationships well enough
# 3 - Bootstrapping - models will ideally be run 50-100 times and we take the mean predictions to increase certainty - note, that all model prediction
# ranges are stored, and this is how we create the associated uncertainty ( SD around the mean) sop we know the areas where the model predicts well consistanty
# and not
# 4 Better model validation - in SDM modelling, you would training your models on 75-80% of your training data, withold 20% and see how the predictions 
# compare back to your witheld data as a form of validation - several metrics use this form of data splitting for validation
# 5 - we still need to consider what we want to do for post processing, and how we make it realtable to HHM 1.0. Glen sugested that we could perhaps use Kalman Filters

#**** SD needs to be added in next iteration



# 6. Model training over all sub grids ----
# This script loops through grid tiles intersecting the training data boundary, runs models for each year pair, and stores results as .rds files.


# LOAD PARAMETERS FROM CREATED FROM PREPROCESSING if not already loaded from step 4:----
grid_gpkg <- st_read("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg") # from Blue topo
#
training_sub_grids_UTM <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/intersecting_sub_grids_UTM.gpkg")
prediction_sub_grids_UTM <-st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/intersecting_sub_grids_UTM.gpkg")
#
training.mask.UTM <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif") # for reference CRS of training grid
prediction.mask.UTM <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif")
#
training.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.021425.Rds")# spatial DF of extent
prediction.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/prediction.mask.df.021425.Rds")# spatial DF of extent
#
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed" # raster data 

# Load additional packages 
library(xgboost)
library(foreach)
library(doParallel)
library(dplyr)
library(tidyr)
library(sf)
library(data.table)
library(fst)
library(BBmisc)  # Normalization for PDP
library(ggplot2) # For saving PDP plots

# Safe rbind Function**
rbind_safe <- function(...) {
  args <- list(...)
  args <- args[!sapply(args, is.null)]  # Remove NULL results
  if (length(args) == 0) return(data.frame())  # Ensure it returns a dataframe
  return(bind_rows(args))
}

registerDoParallel(detectCores() - 1)

Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 2) { 
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  total_tiles <- length(tiles_df)
  pb <- txtProgressBar(min = 0, max = total_tiles, style = 3)
  
  log_file <- file.path(output_dir_train, "error_log.txt")
  cat("Error Log - XGBoost Training\n", Sys.time(), "\n", file = log_file, append = FALSE)  # Overwrite on new run
  
  # -------------------------------------------------------
  # 2. IDENTIFY PREDICTORS & VALIDATE TRAINING DATA
  # -------------------------------------------------------
  static_predictors <- c("grain_size_layer", "prim_sed_layer")  # Static predictors
  
  results_list <- foreach(i = seq_len(total_tiles), .combine = rbind, 
                          .packages = c("xgboost", "dplyr", "data.table", "fst", "foreach")) %dopar% {
                            tryCatch({
                              tile_id <- tiles_df[[i]]
                              training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
                              
                              if (!file.exists(training_data_path)) {
                                cat(Sys.time(), " Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
                                return(NULL)
                              }
                              
                              training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                              training_data <- as.data.frame(training_data)
                              
                              missing_static <- setdiff(static_predictors, names(training_data))
                              if (length(missing_static) > 0) {
                                cat(Sys.time(), " ERROR: Static predictors missing for Tile", tile_id, ":", paste(missing_static, collapse = ", "), "\n", file = log_file, append = TRUE)
                                return(NULL)
                              }
                              
                              foreach(pair = year_pairs, .combine = rbind, .packages = "foreach") %dopar% {  #  open `foreach`
                                start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                
                                rugosity_cols_start <- grep(paste0("^Rugosity_nbh\\d+_", start_year), names(training_data), value = TRUE)
                                rugosity_cols_end <- grep(paste0("^Rugosity_nbh\\d+_", end_year), names(training_data), value = TRUE)
                                
                                dynamic_predictors <- c(
                                  paste0("bathy_", start_year), paste0("bathy_", end_year),
                                  paste0("slope_", start_year), paste0("slope_", end_year),
                                  rugosity_cols_start, rugosity_cols_end,
                                  paste0("hurr_count_", pair), paste0("hurr_strength_", pair),
                                  paste0("tsm_", pair)
                                )
                                
                                predictors <- dplyr::intersect(c(static_predictors, dynamic_predictors), names(training_data))
                                response_var <- paste0("b.change.", pair)
                                
                                if (!response_var %in% names(training_data)) {
                                  cat(Sys.time(), " ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  return(NULL)
                                }
                                
                                if (length(predictors) == 0) {
                                  cat(Sys.time(), " No matching predictors found for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  return(NULL)
                                }
                                
                                # -------------------------------------------------------
                                # 3. FILTER & PREPARE TRAINING DATA
                                # -------------------------------------------------------
                                # Filter out NA values before training
                                subgrid_data <- training_data %>%
                                  dplyr::select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                  tidyr::drop_na()
                                
                                # If no valid data remains, log and skip
                                if (nrow(subgrid_data) == 0) {
                                  cat(Sys.time(), " No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  return(NULL)
                                }
                                
                                # Convert to data.table and replace NA with the mean (or zero)
                                setDT(subgrid_data)
                                for (col in predictors) {
                                  if (any(is.na(subgrid_data[[col]]))) {
                                    cat(Sys.time(), " WARNING: NA values found in Predictor:", col, "| Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                    subgrid_data[[col]][is.na(subgrid_data[[col]])] <- mean(subgrid_data[[col]], na.rm = TRUE)  # Replace NA with mean
                                  }
                                }
                                
                                # Ensure numeric conversion
                                subgrid_data[, (predictors) := lapply(.SD, as.numeric), .SDcols = predictors]
                                subgrid_data[, (response_var) := as.numeric(get(response_var))]
                                
                                # Final check: Stop if any predictor contains NaN or Inf
                                if (any(!is.finite(as.matrix(subgrid_data[, ..predictors])))) {
                                  cat(Sys.time(), " ERROR: NaN or Inf detected after conversion for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  return(NULL)
                                }
                                
                                # Check if response variable has no variability
                                if (length(unique(subgrid_data[[response_var]])) == 1) {
                                  cat(Sys.time(), " ERROR: Response variable has only one unique value for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  return(NULL)
                                }
                                
                                # -------------------------------------------------------
                                # 4. FINALIZE METRIC OUTPUT STORAGE ARRAYS 
                                # -------------------------------------------------------
                                deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot)
                                colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
                                influence_mat <- array(NA, dim = c(length(predictors), n.boot))
                                boot_mat <- matrix(NA, nrow = nrow(subgrid_data), ncol = n.boot)
                                
                                # -------------------------------------------------------
                                # 5. MODEL TRAINING & BOOTSTRAPPING
                                # -------------------------------------------------------
                                dtrain <- xgb.DMatrix(
                                  data = as.matrix(subgrid_data[, ..predictors]),  
                                  label = subgrid_data[[response_var]]
                                )
                                
                                # Model Training with Error Handling
                                for (b in seq_len(n.boot)) {
                                  tryCatch({
                                    xgb_model <- xgb.train(
                                      data = dtrain,
                                      max_depth = 6,
                                      eta = 0.01,
                                      nrounds = 500,
                                      subsample = 0.7,
                                      colsample_bytree = 0.8,
                                      objective = "reg:squarederror",
                                      eval_metric = "rmse",
                                      nthread = 1
                                    )
                                    
                                    # Store predictions for bootstrapping
                                    boot_mat[, b] <- predict(xgb_model, newdata = as.matrix(subgrid_data[, ..predictors]))
                                  }, error = function(e) {
                                    cat(Sys.time(), " ERROR: XGBoost training failed for Tile:", tile_id, "| Year Pair:", pair, "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
                                    return(NULL)
                                  })
                                }
                                
        # -------------------------------------------------------
        # 6. PARTIAL DEPENDENCE PLOTS
        # -------------------------------------------------------
        EnvRanges <- setNames(data.frame(matrix(NA, nrow = 100, ncol = length(predictors))), predictors)
        for (pred in predictors) {
          EnvRanges[[pred]] <- seq(min(subgrid_data[[pred]], na.rm = TRUE), 
                                   max(subgrid_data[[pred]], na.rm = TRUE), 
                                   length.out = 100)
        }
        
        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
        
        # Compute PDP using foreach for efficiency
        foreach(j = seq_along(predictors), .packages = "xgboost") %dopar% { 
          tryCatch({
            grid <- data.frame(x = EnvRanges[[predictors[j]]])
            grid$y <- predict(xgb_model, newdata = as.matrix(grid$x))
            
            loess_fit <- tryCatch(stats::loess(y ~ x, data = grid, span = 0.75), error = function(e) NULL)
            
            for (b in seq_len(n.boot)) {
              PD[, j, b] <- if (!is.null(loess_fit)) predict(loess_fit, newdata = grid$x) else NA_real_
            }
          }, error = function(e) {
            
            cat(Sys.time(), "ERROR in Predictor::", predictors[j], " | ", conditionMessage(e), "\n", file = log_file, append = TRUE)
          return(NULL)
          })
        }

        # -------------------------------------------------------
        # 7. PREDICTIONS & VARIABLE IMPORTANCE
        # -------------------------------------------------------
        # VARIABLE IMPORTANCE
        importance_matrix <- xgb.importance(model = xgb_model)
        importance_values <- importance_matrix$Gain[match(predictors, importance_matrix$Feature)]
        influence_mat[, b] <- ifelse(is.na(importance_values), 0, importance_values)  
        
        # -------------------------------------------------------
        # 8-11. COMPUTE METRICS
        # -------------------------------------------------------
        # Compute bootstrapped statistics (mean, residuals, sd, cv)
        boot_mean <- rowMeans(boot_mat, na.rm = TRUE)
        boot_sd <- apply(boot_mat, 1, sd, na.rm = TRUE)
        boot_cv <- boot_sd / boot_mean  # Coefficient of variation
        residuals <- abs(subgrid_data[[response_var]] - boot_mean)
        
        # Store in separate matrices to avoid altering boot_mat structure
        metrics_mat <- cbind(boot_mean, residuals, boot_sd, boot_cv)
        colnames(metrics_mat) <- c("boot_mean", "residuals", "boot_sd", "boot_cv")
        
        # Compute model fit metrics (for each bootstrap iteration)
        for (b in seq_len(n.boot)) {
          deviance_mat[b, "Dev.Exp"] <- cor(boot_mat[, b], subgrid_data[[response_var]], use = "complete.obs")^2
          deviance_mat[b, "RMSE"] <- sqrt(mean((boot_mat[, b] - subgrid_data[[response_var]])^2, na.rm = TRUE))
          deviance_mat[b, "R2"] <- summary(lm(subgrid_data[[response_var]] ~ boot_mat[, b]))$r.squared
        }
        
        # Mean and standard deviation of variable importance
        preds_influences <- apply(influence_mat, 1, function(x) c(mean = mean(x, na.rm = TRUE), sd = sd(x, na.rm = TRUE)))
        colnames(preds_influences) <- c("Mean_Influence", "SD_Influence")
        rownames(preds_influences) <- predictors
        
        # -------------------------------------------------------
        # 12. SAVE OUTPUTS
        # -------------------------------------------------------
        tile_dir <- file.path(output_dir_train, tile_id)
        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
        
        # save model outputs
        write_fst(as.data.table(metrics_mat), file.path(tile_dir, paste0("predictions_", pair, ".fst")))  
        write_fst(as.data.table(deviance_mat), file.path(tile_dir, paste0("deviance_", pair, ".fst")))  
        write_fst(as.data.table(influence_mat), file.path(tile_dir, paste0("importance_", pair, ".fst")))  
        write_fst(as.data.table(preds_influences), file.path(tile_dir, "mean_var_influence.fst"))  
        
        # save PDP outputs
        write_fst(as.data.table(PD), file.path(tile_dir, paste0("pdp_data_", pair, ".fst")))  
        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))  
        
        # Save the model 
        saveRDS(xgb_model, file.path(tile_dir, paste0("xgb_model_", pair, ".rds")))
        
}  
}, error = function(e) {
  cat(Sys.time(), "ERROR in Tile:", tile_id, "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
  return(NULL)
})  #  
}  #  

# -------------------------------------------------------
# 6. CLOSE PARALLEL PROCESSING
# -------------------------------------------------------
stopCluster(cl)
close(pb)
cat("\n Model Training Complete! Check `error_log.txt` for any issues.\n")
}  #  Function  closed

# **Run Model Training**
Sys.time()
Model_Train_XGBoost(
  training_sub_grids_UTM,
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles",
  year_pairs = c("2004_2006", "2006_2010", "2010_2015", "2015_2022"),
  n.boot = 2
)
Sys.time()

# Test model changes on a single tile
Test_Single_Tile_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 5, test_tile = "BH4S656W_4") {
  cat("\n🛠 Running Single-Tile Test for:", test_tile, "\n")

  tile_row <- training_sub_grids_UTM[training_sub_grids_UTM$tile_id == test_tile, , drop = FALSE]
  if (nrow(tile_row) == 0) stop(" ERROR: Tile ", test_tile, " not found in dataset!")

  tile_dir <- file.path(output_dir_train, test_tile)
  if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)

  training_data_path <- file.path(tile_dir, paste0(test_tile, "_training_clipped_data.fst"))
  if (!file.exists(training_data_path)) stop(" Missing training data for tile: ", test_tile)

  training_data <- read_fst(training_data_path, as.data.table = TRUE)
  training_data <- as.data.frame(training_data)

  #  Update Static Predictors to Correct Column Names
  static_predictors <- c("grain_size_layer", "prim_sed_layer")

  #  Ensure Static Predictors Exist in `training_data`
  missing_static_in_data <- setdiff(static_predictors, names(training_data))
  if (length(missing_static_in_data) > 0) {
    stop(" ERROR: Static predictors missing from training data: ", paste(missing_static_in_data, collapse = ", "))
  }

  for (pair in year_pairs) {
    start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
    end_year <- as.numeric(strsplit(pair, "_")[[1]][2])

    rugosity_cols_start <- grep(paste0("^Rugosity_nbh\\d+_", start_year), names(training_data), value = TRUE)
    rugosity_cols_end <- grep(paste0("^Rugosity_nbh\\d+_", end_year), names(training_data), value = TRUE)

    dynamic_predictors <- c(
      paste0("bathy_", start_year), paste0("bathy_", end_year),
      paste0("slope_", start_year), paste0("slope_", end_year),
      rugosity_cols_start, rugosity_cols_end,
      paste0("hurr_count_", pair), paste0("hurr_strength_", pair),
      paste0("tsm_", pair)
    )

    response_var <- paste0("b.change.", pair)

    # 🔹 **Ensure Static Predictors Are Included**
    predictors <- union(static_predictors, intersect(dynamic_predictors, names(training_data)))

    if (length(predictors) == 0) {
      message(" No matching predictors found for Tile: ", test_tile, " | Year Pair: ", pair)
      next
    }

    subgrid_data <- training_data %>%
      select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
      drop_na()

    #  Impute NA Values in Static Predictors
    subgrid_data[static_predictors] <- lapply(subgrid_data[static_predictors], function(x) {
      ifelse(is.na(x), median(x, na.rm = TRUE), x)
    })

    #  Check if Static Predictors Were Dropped
    missing_static_after_filter <- setdiff(static_predictors, names(subgrid_data))
    if (length(missing_static_after_filter) > 0) {
      message(" Static predictors dropped after filtering: ", paste(missing_static_after_filter, collapse = ", "))
    }

    if (nrow(subgrid_data) == 0) {
      message(" No valid data for Tile: ", test_tile, " | Year Pair: ", pair)
      next
    }

    message("\n Running Model on Tile: ", test_tile, " | Year Pair: ", pair, " | Rows: ", nrow(subgrid_data))

    dtrain <- xgb.DMatrix(data = as.matrix(subgrid_data[predictors]), label = subgrid_data[[response_var]])

    #  Ensure Static Predictors Are in Model
    missing_static_in_model <- setdiff(static_predictors, colnames(dtrain))
    if (length(missing_static_in_model) > 0) {
      message(" Static predictors missing from model: ", paste(missing_static_in_model, collapse = ", "))
    }

    #  Train Model
    xgb_model <- xgb.train(
      data = dtrain,
      max_depth = 6, eta = 0.01, nrounds = 500, subsample = 0.7, colsample_bytree = 0.8,
      objective = "reg:squarederror", eval_metric = "rmse", nthread = 1
    )

    #  Save Outputs
    predictions_df <- cbind(subgrid_data, pred.mean.b.change = predict(xgb_model, newdata = as.matrix(subgrid_data[predictors])))
    write_fst(as.data.table(predictions_df), file.path(tile_dir, paste0("predictions_", pair, ".fst")))

    message(" Finished Processing Tile: ", test_tile, " | Year Pair: ", pair)
  }

  cat("\n Single-Tile Test Complete!\n")
}

Test_Single_Tile_XGBoost(
  training_sub_grids_UTM,
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles",
  year_pairs = c("2015_2022"),
  n.boot = 3  # Reduce for quick testing
)


# Inspect Model Outputs 
predictions <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/predictions_2015_2022.fst")
glimpse(predictions)
var.imp <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/variable_importance_2015_2022.fst")
glimpse(var.imp)
deviance <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/deviance_2015_2022.fst")
glimpse(deviance)
pdp <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/pdp_env_ranges_2015_2022.fst")
glimpse(pdp)
training.data <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/BH4S656W_4_training_clipped_data.fst")
glimpse(training.data)

# 2. Create an averaged Partial Dependence Plot for the whole training area----
library(ggplot2)
library(data.table)
library(dplyr)

# 🚀 Process PDPs One Year Pair at a Time (Like Your Example)
generate_pdp_plot <- function(pair, pdp_data, output_dir) {
  cat("\n Generating PDP plot for Year Pair:", pair, "\n")
  
  #  Ensure Predictors & Values Are Ordered Correctly
  setorder(pdp_data, Predictor, Value)
  
  #  Fix Approximation Errors by Checking for At Least 2 Non-NA Values
  pdp_data[, `:=`(
    min_yhat_smooth = if (.N > 1 && sum(!is.na(min_yhat)) > 1) 
      approx(Value, min_yhat, xout = Value, rule = 2, ties = mean)$y else min_yhat,
    max_yhat_smooth = if (.N > 1 && sum(!is.na(max_yhat)) > 1) 
      approx(Value, max_yhat, xout = Value, rule = 2, ties = mean)$y else max_yhat
  ), by = Predictor]
  
  #  Ensure No NA Values Before Plotting
  pdp_data <- na.omit(pdp_data, cols = c("Value", "min_yhat_smooth", "max_yhat_smooth"))
  
  #  Debugging: Print lengths to verify
  cat("     Debugging: Value Length:", length(pdp_data$Value),
      "min_yhat_smooth Length:", length(pdp_data$min_yhat_smooth),
      "max_yhat_smooth Length:", length(pdp_data$max_yhat_smooth), "\n")
  
  #  Use Polygon-Based Shading (Your Previous Method)
  gg <- ggplot(pdp_data, aes(x = Value, y = mean_yhat, group = Predictor)) +  
    geom_polygon(aes(x = c(Value, rev(Value)), 
                     y = c(max_yhat_smooth, rev(min_yhat_smooth))), 
                 fill = "grey70", alpha = 0.4, na.rm = TRUE) +  #  Fix confidence interval shading
    geom_smooth(color = "black", size = 1.2, method = "loess", se = FALSE, na.rm = TRUE) +  #  Smoothed Mean Trend
    facet_wrap(~ Predictor, scales = "free", ncol = 4) +  #  Keep Grid Consistent
    theme_minimal() +
    labs(title = paste("📊 Averaged PDPs for Year Pair:", pair),
         x = "Predictor Value", y = "Mean Prediction") +
    theme(strip.text = element_text(size = 12))  #  Make Predictor Titles Readable
  
  #  Save Output
  ggsave(file.path(output_dir, paste0("averaged_pdp_", pair, ".png")), gg, width = 12, height = 9)
  cat(" PDP plot saved for year pair:", pair, "\n")
}


# 🚀 Main Function to Process PDP Data & Generate Plots
average_fast_pdps_all_tiles <- function(intersecting_tiles, year_pairs, input_dir, output_dir) {
  cat("\n Generating averaged PDPs across all intersecting tiles...\n")
  
  for (pair in year_pairs) {
    cat("\n🔵 Processing year pair:", pair, "\n")
    temp_pdp <- data.table()
    
    for (tile in intersecting_tiles$tile) {
      cat("   Processing Tile ID:", tile, "\n")
      
      pdp_path <- file.path(input_dir, tile, paste0("pdp_", pair, ".fst"))
      if (!file.exists(pdp_path)) {
        cat("     WARNING: Missing PDP data for Tile ID:", tile, "\n")
        next
      }
      
      pdp_data <- read_fst(pdp_path, as.data.table = TRUE)
      cat("    📌 Available columns in PDP file:", paste(names(pdp_data), collapse = ", "), "\n")
      
      if (!"Predictor" %in% names(pdp_data)) {
        predictor_col <- names(pdp_data)[grepl("predictor", names(pdp_data), ignore.case = TRUE)]
        if (length(predictor_col) == 1) {
          setnames(pdp_data, old = predictor_col, new = "Predictor")
        } else {
          cat("     ERROR: No 'Predictor' column found in PDP file for Tile:", tile, "\n")
          next
        }
      }
      
      temp_pdp <- rbind(temp_pdp, pdp_data, use.names = TRUE, fill = TRUE)
    }
    
    if (nrow(temp_pdp) > 0) {
      # ✅ Compute Min/Max & Mean Across All Tiles
      avg_pdp <- temp_pdp[, .(
        mean_yhat = mean(Prediction, na.rm = TRUE),
        min_yhat = min(Prediction, na.rm = TRUE),
        max_yhat = max(Prediction, na.rm = TRUE)
      ), by = .(Predictor, Value)]
      
      # ✅ Generate PDP Plot Using Your Old Approach
      generate_pdp_plot(pair, avg_pdp, output_dir)
      
    } else {
      cat("  ⚠️ No data available for year pair:", pair, "\n")
    }
  }
}


#Run generate_fast_pdps_for_tile
training_sub_grids_UTM <- st_read(file.path(training_dir, "intersecting_sub_grids_UTM.gpkg"))
# Extract valid tile IDs
tile_list <- training_sub_grids_UTM$tile_id
# Define year pairs
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
# training directory

# ✅ Call the function
for (tile_id in tile_list) {
  tile <- data.table(tile = tile_id)
  generate_fast_pdps_for_tile(tile, year_pairs, training_dir)
}

cat("\n✅ PDP generation completed for all tiles!\n")

training.data.clipped <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S456Z_3/BH4S456Z_3_training_clipped_data.rds")


#----#----#
# Run average_fast_pdps_all_tiles function
# ✅ Define required parameters
intersecting_tiles <- data.table(tile = c("BH4S656W_4", "BH4S656X_1", "BH4S2577_2", "BH4S2575_1"))  # Example tile list
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")  # Year pairs to process
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"  # Directory containing PDP files
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"  # Directory to save results

# ✅ Call the function
average_fast_pdps_all_tiles(intersecting_tiles, year_pairs, input_dir, output_dir)