
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
Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 20) { # change n.boot iterations as desired
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)  #  Enable parallel processing
  # registerDoSEQ()  # Run sequentially for debugging
  on.exit({
    if (exists("cl") && inherits(cl, "cluster")) {
      stopCluster(cl)
    }
    closeAllConnections()
  }, add = TRUE)
  
  log_file <- file.path(output_dir_train, "error_log.txt")
  cat("Error Log - XGBoost Training\n", Sys.time(), "\n", file = log_file, append = FALSE)  # Overwrite on new run
  
  # -------------------------------------------------------
  # 2. IDENTIFY PREDICTORS & VALIDATE TRAINING DATA
  # -------------------------------------------------------
  static_predictors <- c("grain_size_layer", "prim_sed_layer")  # Static predictors
  
  results_list <- foreach(i = seq_len(length(tiles_df)), .combine = rbind, 
                          .packages = c("xgboost", "dplyr", "data.table", "fst", "tidyr", "foreach")) %dopar% { 
                            
                            tile_id <- tiles_df[[i]] # MUST START COUNTER OUTSIDE LOOP
                            
                            foreach(pair = year_pairs, .combine = rbind, 
                                    .packages = c("xgboost", "dplyr", "data.table", "fst", "tidyr", "foreach")) %dopar% { 
                                      
                                      tryCatch({
                                        training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
                                        
                                        if (!file.exists(training_data_path)) {
                                          cat(Sys.time(), " Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                                        training_data <- as.data.frame(training_data)
                                        
                                        
                                        start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                        end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                        response_var <- paste0("b.change.", pair)
                                        
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), " ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                              file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        dynamic_predictors <- c(
                                          paste0("bathy_", start_year), paste0("bathy_", end_year),
                                          paste0("slope_", start_year), paste0("slope_", end_year),
                                          grep(paste0("^Rugosity_nbh\\d+_", start_year), names(training_data), value = TRUE),
                                          grep(paste0("^Rugosity_nbh\\d+_", end_year), names(training_data), value = TRUE),
                                          paste0("hurr_count_", pair), paste0("hurr_strength_", pair),
                                          paste0("tsm_", pair)
                                        )
                                        
                                        predictors <- intersect(c(static_predictors, dynamic_predictors), names(training_data))
                                        # response_var <- paste0("b.change.", pair)
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), " ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        # -------------------------------------------------------
                                        # 3. FILTER & PREPARE TRAINING DATA
                                        # -------------------------------------------------------
                                        cat(Sys.time(), " Tile:", tile_id, "| Year Pair:", pair, "| Available Predictors:", 
                                            paste(predictors, collapse = ", "), "| Response:", response_var, "\n", 
                                            file = log_file, append = TRUE)
                                        
                                        subgrid_data <- training_data %>%
                                          dplyr::select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                          drop_na()
                                        
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), " No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), " ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                              file = log_file, append = TRUE)
                                          next  # Move to next year pair
                                        }
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), " No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                              file = log_file, append = TRUE)
                                          next
                                        }

                                        
                                        # -------------------------------------------------------
                                        # 4. INITIALIZE METRIC OUTPUT STORAGE ARRAYS [outside of the boostrap loop!]
                                        # -------------------------------------------------------
                                        # Model Metrics 
                                        deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot)
                                        colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
                                        
                                        # Initialize influence matrix with NAs
                                        influence_mat <- array(NA, dim = c(length(predictors), n.boot))
                                        rownames(influence_mat) <- predictors  # Assign predictor names to rows
                                        colnames(influence_mat) <- paste0("Rep_", seq_len(n.boot))  # Assign boot iteration names
                                        
                                        
                                        boot_mat <- matrix(NA, nrow = nrow(subgrid_data), ncol = n.boot)
                                        
                                        # -------------------------------------------------------
                                        # 5. SETUP for PARTIAL DEPENDENCE PLOT (PDP) DATA
                                        # -------------------------------------------------------
                                        
                                        PredMins <- apply(subgrid_data[, predictors, drop = FALSE], 2, min, na.rm = TRUE)
                                        PredMaxs <- apply(subgrid_data[, predictors, drop = FALSE], 2, max, na.rm = TRUE)
                                        
                                        EnvRanges <- as.data.frame(matrix(NA, nrow = 100, ncol = length(predictors)))
                                        colnames(EnvRanges) <- predictors
                                        
                                        for (i in seq_along(predictors)) {
                                          EnvRanges[, i] <- seq(PredMins[i], PredMaxs[i], length.out = 100)
                                        }

                                        # set dimension names for the array 
                                        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
                                        dimnames(PD)[[2]] <- predictors
                                        dimnames(PD)[[3]] <- paste("Rep_", seq_len(n.boot), sep = "")

                                        # -------------------------------------------------------
                                        # 6. MODEL TRAINING & BOOTSTRAPPING
                                        # -------------------------------------------------------
                                        if (length(predictors) == 0) {
                                          cat(Sys.time(), " ERROR: No predictors found for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        
                                        if (any(!is.finite(as.matrix(subgrid_data[, predictors, drop = FALSE])))) {
                                          cat(Sys.time(), " ERROR: NA/NaN/Inf detected in predictors\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        cat(Sys.time(), " Training XGBoost model for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                        
                                        for (b in seq_len(n.boot)) { # open model boostrap loop
                                          
                                          dtrain <- xgb.DMatrix(data = as.matrix(subgrid_data[, predictors]), # turn training data into matrix to  
                                                                label = subgrid_data[[response_var]])           # be compatible with xgboost
                                          
                                          xgb_model <- xgb.train(
                                            data = dtrain,
                                            max_depth = 6,  # Controls tree depth (higher values capture more interactions but risk overfitting). Common range: 3-10
                                            eta = 0.01,  # Learning rate (lower values prevent overfitting, but require more trees). Common range: 0.001 - 0.3
                                            nrounds = 500,  # Number of boosting iterations (higher values improve performance but increase training time). Common range: 100-1000
                                            subsample = 0.7,  # Fraction of data used per boosting round (lower values prevent overfitting). Common range: 0.5-1
                                            colsample_bytree = 0.8,  # Fraction of predictors used per tree (lower values prevent overfitting). Common range: 0.5-1
                                            objective = "reg:squarederror",  # Regression objective function
                                            eval_metric = "rmse",  # RMSE measures model performance (lower is better)
                                            nthread = 1  # Number of CPU threads (set based on available computing resources)
                                            
                                          )

                                          # -------------------------------------------------------
                                          # 7. STORE MODEL METRICS - [still in bootstep loop] 
                                          # -------------------------------------------------------
                                          # store model results in an array for each iteration 
                                          boot_mat[, b] <- predict(xgb_model, newdata = as.matrix(subgrid_data[, predictors]))
                                          # retain spatial identifiers, and append actual elevation change 
                                          boot_df <- as.data.frame(boot_mat)
                                          colnames(boot_df) <- paste0("Boot_", seq_len(n.boot))  # Rename boot iterations
                                          boot_df <- cbind(subgrid_data[, c("X", "Y", "FID")], 
                                                           Actual_Change = subgrid_data[[response_var]], # aka b.change
                                                           boot_df)  # Append spatial & actual change
                                          
                                          
                                          # Extract importance scores from the model
                                          importance_matrix <- xgb.importance(model = xgb_model)
                                          
                                          # Map importance values to predictor names (some predictors may not appear in importance_matrix)
                                          importance_values <- setNames(importance_matrix$Gain, importance_matrix$Feature)
                                          
                                          # Ensure alignment: Only assign values where predictor names match
                                          matching_indices <- match(names(importance_values), rownames(influence_mat))
                                          valid_indices <- !is.na(matching_indices)
                                          
                                          # Assign values only for existing predictors
                                          influence_mat[matching_indices[valid_indices], b] <- importance_values[valid_indices]
                                          
                                          # Fill remaining NAs with 0
                                          influence_mat[is.na(influence_mat)] <- 0
                                          
                                          
                                          
                                          deviance_mat[b, "Dev.Exp"] <- cor(boot_mat[, b], subgrid_data[[response_var]], use = "complete.obs")^2
                                          deviance_mat[b, "RMSE"] <- sqrt(mean((boot_mat[, b] - subgrid_data[[response_var]])^2, na.rm = TRUE))
                                          deviance_mat[b, "R2"] <- summary(lm(subgrid_data[[response_var]] ~ boot_mat[, b]))$r.squared

                                          # -------------------------------------------------------
                                          # 8. STORE PARTIAL DEPENDENCE PLOT DATA - [still in bootstrap loop] 
                                          # -------------------------------------------------------
                                          
                                          
                                          for (j in seq_along(predictors)) { 
                                            grid <- data.frame(x = EnvRanges[[predictors[j]]])
                                            grid$y <- predict(xgb_model, newdata = as.matrix(grid$x))
                                            
                                            # Remove NAs and check unique values
                                            grid <- grid[complete.cases(grid), ]
                                            if (length(unique(grid$x)) < 5) {
                                              cat(Sys.time(), " Skipping PDP for Predictor:", predictors[j], "- Too few unique values\n", file = log_file, append = TRUE)
                                              next
                                            }
                                            
                                            # Try loess smoothing with a larger span
                                            loess_fit <- tryCatch(stats::loess(y ~ x, data = grid, span = 1), error = function(e) NULL)
                                            
                                            # Store PDP values (fallback if loess fails)
                                            PD[, j, b] <- if (!is.null(loess_fit)) predict(loess_fit, newdata = grid$x) else rep(NA_real_, 100)
                                            
                                            # stored in a conveinant format for post model plotting 
                                            PD_long <- as.data.frame.table(PD, responseName = "PDP_Value")
                                            colnames(PD_long) <- c("Index", "Predictor", "Replicate", "PDP_Value")  
                                            PD_long$Index <- NULL  # Remove redundant column
                                          } 
                                        } #close model boostrap
  
                                        # -------------------------------------------------------
                                        # 9. SAVE OUTPUTS
                                        # -------------------------------------------------------
                                        tile_dir <- file.path(output_dir_train, tile_id)
                                        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
                                        
                                        cat(Sys.time(), "💾 Writing outputs for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                        
                                        # save model outputs
                                        write_fst(as.data.table(deviance_mat), file.path(tile_dir, paste0("deviance_", pair, ".fst"))) 
                                        
                                        # write_fst(as.data.table(influence_mat), file.path(tile_dir, paste0("influence_", pair, ".fst"))) 
                                        influence_df <- as.data.frame(influence_mat)
                                        influence_df$Predictor <- rownames(influence_mat)
                                        influence_df <- influence_df[, c("Predictor", paste0("Rep_", seq_len(n.boot)))] # Reorder columns
                                        write_fst(as.data.table(influence_df), file.path(tile_dir, paste0("influence_", pair, ".fst")))

                                        # write_fst(as.data.table(boot_mat), file.path(tile_dir, paste0("bootstraps_", pair, ".fst"))) 
                                        write_fst(as.data.table(boot_df), file.path(tile_dir, paste0("bootstraps_", pair, ".fst")))
                                        saveRDS(xgb_model, file.path(tile_dir, paste0("xgb_model_", pair, ".rds")))  
                                        
                                        # save PDP outputs
                                        write_fst(as.data.table(PD_long), file.path(tile_dir, paste0("pdp_data_", pair, ".fst")))  
                                        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))  
                                        
                                        writeLines(paste(Sys.time(), "📁 Saved Model for Tile:", tile_id, "| Year Pair:", pair), log_file, append = TRUE)
                                        
                                        # }  # End for-loop over year_pairs
                                        
                                      }, error = function(e) {
                                        cat(Sys.time(), " ERROR in Tile:", tiles_df[[i]], "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
                                        return(NULL)
                                      })
                                    }
                          }
  # -------------------------------------------------------
  # 10. CLOSE PARALLEL PROCESSING
  # -------------------------------------------------------
  # stopCluster(cl)
  
  cat("\n Model Training Complete! Check `error_log.txt` for any issues.\n")
}

#  **Run Model Training**
Sys.time()
Model_Train_XGBoost(
  training_sub_grids_UTM,
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles",
  year_pairs = c("2004_2006", "2006_2010", "2010_2015", "2015_2022"),
  n.boot = 20
)
Sys.time()




showConnections()

closeAllConnections()

# Verify model outputs look good
deviance <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/deviance_2004_2006.fst")
influence <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/influence_2004_2006.fst")
boots <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/bootstraps_2004_2006.fst")
envranges <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/pdp_env_ranges_2004_2006.fst")
pdp <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/pdp_data_2004_2006.fst")

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