
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



#  Updated Model Training Function with PDP, Performance Tracking, & Predictions Fixes
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

#  **New Safe rbind Function**
rbind_safe <- function(...) {
  args <- list(...)
  args <- args[!sapply(args, is.null)]  # Remove NULL results
  if (length(args) == 0) return(NULL)  
  return(bind_rows(args))
}

# 🚀 Updated Model Training Function with Robust Checks & Fixes
Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 50) {
  cat("\n🚀 Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  
  num_cores <- detectCores() - 1  
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  
  total_tiles <- length(tiles_df)
  pb <- txtProgressBar(min = 0, max = total_tiles, style = 3)
  
  #  **Step 1: Identify All Possible Predictors Across Tiles**
  all_possible_predictors <- c("grain_size_layer", "prim_sed_layer")
  
  for (tile_id in tiles_df) {
    training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
    if (file.exists(training_data_path)) {
      training_data <- read_fst(training_data_path, as.data.table = TRUE)  
      training_data <- as.data.frame(training_data)
      
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
        
        all_possible_predictors <- union(all_possible_predictors, dynamic_predictors)
      }
    }
  }
  
  #  **Step 2: Parallel Processing for Training**
  results_list <- foreach(i = seq_len(total_tiles), .combine = rbind_safe,
                          .packages = c("xgboost", "dplyr", "tidyr", "data.table", "fst", "BBmisc", "ggplot2")) %dopar% {  
                            
                            tile_id <- tiles_df[[i]]  
                            tile_results <- list()
                            tile_dir <- file.path(output_dir_train, tile_id)
                            if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
                            
                            training_data_path <- file.path(tile_dir, paste0(tile_id, "_training_clipped_data.fst"))
                            if (!file.exists(training_data_path)) {
                              message(" Missing training data for tile: ", tile_id)
                              return(NULL)
                            }
                            
                            training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                            training_data <- as.data.frame(training_data)  
                            
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
                              
                              #  **Use Global Predictor List**
                              predictors <- intersect(all_possible_predictors, names(training_data))
                              
                              if (length(predictors) == 0) {
                                message(" No matching predictors found for Tile: ", tile_id, " | Year Pair: ", pair)
                                next
                              }
                              
                              #  **Impute Static Predictors Before Filtering**
                              training_data[all_possible_predictors] <- lapply(training_data[all_possible_predictors], function(x) {
                                ifelse(is.na(x), median(x, na.rm = TRUE), x)
                              })
                              
                              subgrid_data <- training_data %>%
                                select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                drop_na()
                              
                              if (nrow(subgrid_data) == 0) {
                                message(" No valid data for Tile: ", tile_id, " | Year Pair: ", pair)
                                next
                              }
                              
                              message("\n📊 Processing Tile: ", tile_id, " | Year Pair: ", pair, " | Rows: ", nrow(subgrid_data))
                              
                              #  **Ensure Predictors are in the Same Order**
                              predictors <- sort(predictors)
                              subgrid_data <- subgrid_data[, c(predictors, response_var)]
                              dtrain <- xgb.DMatrix(data = as.matrix(subgrid_data[predictors]), label = subgrid_data[[response_var]])
                              
                              boot_mat <- array(NA_real_, c(nrow(subgrid_data), n.boot))
                              
                              for (b in seq_len(n.boot)) {
                                xgb_model <- xgb.train(
                                  data = dtrain,
                                  max_depth = 6, eta = 0.01, nrounds = 500, subsample = 0.7, colsample_bytree = 0.8,
                                  objective = "reg:squarederror", eval_metric = "rmse", nthread = 1
                                )
                                
                                newdata <- subgrid_data[predictors]
                                newdata <- newdata[, order(colnames(newdata))]  # **Force Matching Column Order**
                                
                                predictions <- predict(xgb_model, newdata = as.matrix(newdata))
                                boot_mat[, b] <- predictions
                              }
                              
                              message("📁 Saved All Outputs for Tile: ", tile_id, " | Year Pair: ", pair)
                            }
                            
                            setTxtProgressBar(pb, i)  
                            return(bind_rows(tile_results))
                          }
  
  stopCluster(cl)
  close(pb)
  cat("\n✅ Model Training Complete!\n")
}



# ✅ **Run Model Training**
Sys.time()
Model_Train_XGBoost(
  training_sub_grids_UTM,
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles",
  year_pairs = c("2004_2006", "2006_2010", "2010_2015", "2015_2022"),
  n.boot = 50
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