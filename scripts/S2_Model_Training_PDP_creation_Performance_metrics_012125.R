
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

Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 20) { 
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n🚀 Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)  # Enable parallel processing
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
                            
                            tile_id <- tiles_df[[i]] # MUST START COUNTER
                            
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
                                        
                                        # missing_static <- setdiff(static_predictors, names(training_data))
                                        # if (length(missing_static) > 0) {
                                        #   cat(Sys.time(), " ERROR: Static predictors missing for Tile", tile_id, ":", paste(missing_static, collapse = ", "), "\n", file = log_file, append = TRUE)
                                        #   return(NULL)
                                        # }
                                        # 
                                        # for (pair in year_pairs) {
                                        #   cat(Sys.time(), "Starting Year Pair:", pair, "for Tile:", tile_id, "\n", file = log_file, append = TRUE)
                                        
                                        start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                        end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                        response_var <- paste0("b.change.", pair)
                                        
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), "ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                        
                                        # Store Environmental Ranges in Long Format Immediately
                                        # Update with Actual Ranges
                                        # for (pred in predictors) {
                                        #   EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(PredMins[pred], PredMaxs[pred], length.out = 100)
                                        # }
                                        EnvRanges <- expand.grid(
                                          Env_Value = seq(0, 1, length.out = 100),  # Placeholder (Will be replaced)
                                          Predictor = predictors
                                        )
                                        
                                        for (pred in predictors) {
                                          min_val <- PredMins[pred]
                                          max_val <- PredMaxs[pred]
                                        
                                          # Ensure valid range before setting values
                                          if (!is.finite(min_val) || !is.finite(max_val) || min_val == max_val) {
                                            cat(Sys.time(), " Skipping predictor:", pred, "- Invalid range\n", file = log_file, append = TRUE)
                                            next
                                          }
                                          
                                          EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(min_val, max_val, length.out = 100)
                                        }
                                        
                                        # Store X, Y, FID in EnvRanges
                                        EnvRanges$X <- rep(subgrid_data$X[1:100], length(predictors))  # Sample 100 spatial points
                                        EnvRanges$Y <- rep(subgrid_data$Y[1:100], length(predictors))
                                        EnvRanges$FID <- rep(subgrid_data$FID[1:100], length(predictors))
                                        
                                        # Set up PDP storage: [100 (Env Values) x N Predictors x N Bootstraps]
                                        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
                                        dimnames(PD)[[2]] <- predictors
                                        dimnames(PD)[[3]] <- paste0("Rep_", seq_len(n.boot))
                                        
                                        
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
                                          # 6. STORE MODEL METRICS - [still in bootstep loop] 
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
                                          # 7. STORE PARTIAL DEPENDENCE PLOT DATA - [still in bootstrap loop] 
                                          # -------------------------------------------------------
                                          
                                          # Store PDP values for each predictor
                                          PDP_Storage <- list()  # Collect all PDP data
                                          
                                          # for (j in seq_along(predictors)) { 
                                          #   grid <- data.frame(Env_Value = EnvRanges$Env_Value[EnvRanges$Predictor == predictors[j]])
                                          #   grid$PDP_Value <- predict(xgb_model, newdata = as.matrix(grid$Env_Value))
                                          for (j in seq_along(predictors)) { 
                                            grid <- data.frame(
                                              Env_Value = EnvRanges$Env_Value[EnvRanges$Predictor == predictors[j]],
                                              X = EnvRanges$X[EnvRanges$Predictor == predictors[j]], 
                                              Y = EnvRanges$Y[EnvRanges$Predictor == predictors[j]], 
                                              FID = EnvRanges$FID[EnvRanges$Predictor == predictors[j]]
                                            )
                                            
                                            # Ensure correct input shape for predict()
                                            if (nrow(grid) == 0 || !all(is.finite(grid$Env_Value))) {
                                              cat(Sys.time(), " Skipping PDP for Predictor:", predictors[j], "- No valid values\n", file = log_file, append = TRUE)
                                              next
                                            }
                                            
                                            # Predict PDP Values (Ensure correct matrix shape)
                                            grid$PDP_Value <- predict(xgb_model, newdata = matrix(grid$Env_Value, ncol = 1))
                                            
                                            # Remove NAs and check unique values
                                            grid <- grid[complete.cases(grid), ]
                                            if (length(unique(grid$Env_Value)) < 5) {
                                              cat(Sys.time(), " Skipping PDP for Predictor:", predictors[j], "- Too few unique values\n", file = log_file, append = TRUE)
                                              next
                                            }
                                            
                                            # Apply loess smoothing (fallback to raw if needed)
                                            loess_fit <- tryCatch(loess(PDP_Value ~ Env_Value, data = grid, span = 1), error = function(e) NULL)
                                            PD[, j, b] <- if (!is.null(loess_fit)) predict(loess_fit, newdata = grid$Env_Value) else grid$PDP_Value
                                            
                                            # Store results in long format
                                            PDP_Storage[[j]] <- data.frame(
                                              Predictor = predictors[j],
                                              Env_Value = grid$Env_Value,
                                              Replicate = paste0("Rep_", b),
                                              PDP_Value = PD[, j, b],
                                              X = grid$X, Y = grid$Y, FID = grid$FID  # Ensure spatial IDs are saved
                                            )
                                          }
                                        } #close model boostrap
                                        
                                        # Convert to a single dataframe
                                        PDP_Long <- bind_rows(PDP_Storage)
                                        
                                        # -------------------------------------------------------
                                        # 8. SAVE OUTPUTS
                                        # -------------------------------------------------------
                                        tile_dir <- file.path(output_dir_train, tile_id)
                                        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
                                        
                                        cat(Sys.time(), " Writing outputs for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                        
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
                                        
                                        # Save PDP and Env Values Together
                                        write_fst(as.data.table(PDP_Long), file.path(tile_dir, paste0("pdp_data_", pair, ".fst")))  
                                        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))  
                                        write_fst(as.data.table(bind_rows(PDP_Storage)), file.path(tile_dir, paste0("pdp_data_raw_", pair, ".fst")))
                                        
                                        
                                        writeLines(paste(Sys.time(), " Saved Model for Tile:", tile_id, "| Year Pair:", pair), log_file, append = TRUE)
                                        
                                        # }  # End for-loop over year_pairs
                                        
                                      }, error = function(e) {
                                        cat(Sys.time(), " ERROR in Tile:", tiles_df[[i]], "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
                                        return(NULL)
                                      })
                                    }
                          }
  # -------------------------------------------------------
  # 9. CLOSE PARALLEL PROCESSING
  # -------------------------------------------------------
  # stopCluster(cl)
  
  cat("\n✅ Model Training Complete! Check `error_log.txt` for any issues.\n")
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

# 2. Create an averaged Partial Dependence Plot for the whole training area (one for each year pair)----
# in the future we will create one pdp plot per year pair, per Eco Region Tile. 
library(ggplot2)
library(data.table)
library(dplyr)

# Directories
tiles_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"

# Year pairs
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")

# List all tile folders
tile_folders <- list.dirs(tiles_dir, recursive = FALSE)

# Storage for overall PDPs
all_pdp_list <- list()

# Loop through each year pair
for (pair in year_pairs) {
  
  pdp_all_tiles <- list()  # Store PDPs from all tiles
  
  # Loop through tiles
  for (tile in tile_folders) {
    
    # File path
    pdp_file <- file.path(tile, paste0("pdp_data_", pair, ".fst"))
    
    if (file.exists(pdp_file)) {
      
      #  Load PDP data
      pdp_data <- read_fst(pdp_file)
      
      # 🔍 Debug: Check column names
      if (!"Env_Value" %in% colnames(pdp_data)) {
        cat(" ERROR: `Env_Value` is missing in:", pdp_file, "\n")
        cat("Available columns:", paste(colnames(pdp_data), collapse = ", "), "\n")
        next  # Skip this tile
      }
      
      #  Ensure `Env_Value` is numeric
      pdp_data <- pdp_data %>%
        mutate(Env_Value = as.numeric(Env_Value))
      
      #  Skip invalid PDP files
      if (any(is.na(pdp_data$Env_Value))) {
        cat("🚨 Skipping invalid PDP file (contains NAs):", pdp_file, "\n")
        next
      }
      

      # Aggregate PDP Data Using Its Own `Env_Value`
      pdp_data_agg <- pdp_data %>%
        select(Predictor, Env_Value, PDP_Value)  # ✅ Keep PDP_Value!
      
 
      
      # Store aggregated data
      pdp_all_tiles[[tile]] <- pdp_data_agg
    } else {
      cat("Skipping missing PDP file:", pdp_file, "\n")
    }
  }
  
  # ✅ Merge all available tiles into a single dataframe
  if (length(pdp_all_tiles) > 0) {
    overall_pdp_data <- bind_rows(pdp_all_tiles)
    
    # Compute overall mean but keep min/max as actual min/max
    pdp_summary <- overall_pdp_data %>%
      group_by(Predictor, Env_Value) %>%
      summarise(
        PDP_Mean = mean(PDP_Value, na.rm = TRUE),  
        PDP_Min = min(PDP_Value, na.rm = TRUE),
        PDP_Max = max(PDP_Value, na.rm = TRUE),
        .groups = "drop"
      )

    pdp_summary <- pdp_summary %>%
      mutate(
        PDP_Max = ifelse(PDP_Min == PDP_Max, PDP_Max + 1e-5, PDP_Max) # Add small variation
      )
    
    
    #  Print a Sample to Check for NA Issues
    print(paste0(" Processed PDP for ", pair, ":"))
    print(head(pdp_summary))
    
    #  Store for plotting
    all_pdp_list[[pair]] <- pdp_summary
  } else {
    cat(" No available PDP data for", pair, "- Skipping this year pair.\n")
  }
}

#  Create PDP Plots (Only for available year pairs)
#  Create PDP Plots (Only for available year pairs)
for (pair in names(all_pdp_list)) {
  
  # Ensure `PDP_Min` and `PDP_Max` are valid before plotting
  pdp_filtered <- all_pdp_list[[pair]] %>%
    filter(!is.na(PDP_Min), !is.na(PDP_Max))
  
  
  
  # If no valid data remains after filtering, skip
  if (nrow(pdp_filtered) == 0) {
    cat(" Skipping PDP plot for", pair, "as no valid data remains.\n")
    next
  }
  
  plot_pdp <- ggplot(pdp_filtered, aes(x = Env_Value, y = PDP_Mean)) +
    # Shaded confidence region (Min to Max range)
    geom_ribbon(aes(ymin = PDP_Min, ymax = PDP_Max), fill = "grey70") +  
    # Mean trend line (LOESS smooth)
    geom_smooth(method = "loess", se = FALSE, color = "black", linewidth = 1) +  
    facet_wrap(~Predictor, scales = "free", ncol = 3) +  # Separate plots for each predictor
    labs(
      title = paste("Averaged PDPs for Year Pair:", pair),
      x = "Model Predictor Value",
      y = "Mean elevation change (m)"
    ) +
    theme_minimal(base_size = 14) +  
    theme(
      strip.background = element_rect(fill = "lightgray"),
      strip.text = element_text(size = 12, face = "bold"),
      panel.grid.major = element_line(linewidth = 0.2, linetype = "dotted"),
      panel.grid.minor = element_blank(),
      legend.position = "none"
    )
  
  # Save the final PDP plot
  ggsave(filename = file.path(output_dir, paste0("Overall_PDP_", pair, ".pdf")), 
         plot = plot_pdp, width = 12, height = 6, dpi = 300) # saved colors in a vectorised format to store detail
  
}