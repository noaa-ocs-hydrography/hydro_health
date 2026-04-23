# XGB paramater search automatic tuning 

# ==============================================================================
# XGB PARAMETER SEARCH - AUTOMATED MINI WORKFLOW
#   - Runs a single-model (no bootstrap) XGBoost for each parameter set
#   - Saves rasters: mean_predicted_change, prediction_residual, delta_residual
#   - includes weighted-loss tuning also 
#   - Computes metrics + scores per run
#   - Identifies:
#       * best_for_accuracy
#       * best_for_change ( best to predict delta)
#       * best_compromise
# ==============================================================================

# ==============================================================================
# XGB PARAMETER SEARCH - AUTOMATED MINI WORKFLOW (V2: includes weighted-loss tuning)
# ==============================================================================

XGB_ParamSearch_Auto <- function(
    tile_data_path,
    output_dir,
    predictor_list_path,
    year_pair,
    param_grid,              # NOW ALLOWED TO CONTAIN weight_alpha
    global_seed = 2025,
    grid_crs_epsg = 32617
) {
  # -------------------------------------------------------
  # LOAD LIBRARIES
  # -------------------------------------------------------
  library(xgboost)
  library(data.table)
  library(fst)
  library(dplyr)
  library(sf)
  library(raster)
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  log_file <- file.path(output_dir, "param_search_log.txt")
  
  cat("XGB PARAMETER SEARCH (V2: weighted-loss tuning enabled)\n",
      "Run started: ", as.character(Sys.time()), "\n",
      "Global seed: ", global_seed, "\n\n",
      file = log_file)
  
  set.seed(global_seed)
  
  # -------------------------------------------------------
  # READ TILE DATA
  # -------------------------------------------------------
  if (!file.exists(tile_data_path)) stop("Tile data not found: ", tile_data_path)
  if (!file.exists(predictor_list_path)) stop("Predictor list not found.")
  
  full_tile <- as.data.frame(read_fst(tile_data_path))
  preds_raw <- readLines(predictor_list_path)
  
  # -------------------------------------------------------
  # TRANSLATE NEW PREDICTOR NAMES
  # -------------------------------------------------------
  add_year_suffix <- c("hurr_count", "hurr_strength", "tsm")
  
  predictors <- sapply(preds_raw, function(p) {
    if (p %in% add_year_suffix) paste0(p, "_", year_pair) else p
  }, USE.NAMES = FALSE)
  
  if ("flowdir_sin_t" %in% predictors || "flowdir_cos_t" %in% predictors)
    predictors <- union(predictors, c("flowdir_sin_t", "flowdir_cos_t"))
  
  if ("bpi_fine_mean3_t" %in% predictors || "bpi_fine_sd3_t" %in% predictors)
    predictors <- union(predictors, "bpi_fine_t")
  
  if ("bpi_broad_mean3_t" %in% predictors || "bpi_broad_sd3_t" %in% predictors)
    predictors <- union(predictors, "bpi_broad_t")
  
  predictors <- predictors[predictors != "terrain_classification_t"]
  predictors <- union("bathy_t", predictors)
  predictors <- intersect(predictors, names(full_tile))
  
  if (length(predictors) == 0) stop("No predictors matched tile dataset!")
  
  # -------------------------------------------------------
  # RESPONSE
  # -------------------------------------------------------
  response_var <- "bathy_t1"
  if (!response_var %in% names(full_tile))
    stop("Tile dataset missing bathy_t1 (response).")
  
  if ("bathy_t" %in% names(full_tile) && "bathy_t1" %in% names(full_tile)) {
    suppressWarnings({ full_tile$delta_bathy <- full_tile$bathy_t1 - full_tile$bathy_t })
  }
  
  # -------------------------------------------------------
  # CLEAN TYPES
  # -------------------------------------------------------
  full_tile <- full_tile %>%
    mutate(across(all_of(c(predictors, response_var, "delta_bathy")),
                  ~ suppressWarnings(as.numeric(as.character(.)))))
  
  # -------------------------------------------------------
  # TRAINING SUBSET
  # -------------------------------------------------------
  train <- full_tile %>%
    filter(is.finite(.data[[response_var]]),
           is.finite(delta_bathy)) %>%
    filter(complete.cases(.[, predictors]))
  
  if (nrow(train) < 50)
    stop("Not enough valid training rows (<50).")
  
  cat("Data ready.\n",
      "Total rows: ", nrow(full_tile), "\n",
      "Train rows: ", nrow(train), "\n",
      "Predictors: ", paste(predictors, collapse = ", "), "\n\n",
      file = log_file, append = TRUE)
  
  grid_crs_proj4 <- sf::st_crs(grid_crs_epsg)$proj4string
  
  # -------------------------------------------------------
  # PARAMETER SEARCH LOOP
  # -------------------------------------------------------
  results_list <- vector("list", nrow(param_grid))
  
  for (i in seq_len(nrow(param_grid))) {
    params <- param_grid[i, ]
    run_id <- sprintf("run_%03d", i)
    run_seed <- global_seed + i
    set.seed(run_seed)
    
    # Extract weight_alpha if present, otherwise default = 1.0
    weight_alpha <- if ("weight_alpha" %in% names(params)) params$weight_alpha else 1.0
    
    cat("----------------------------------------------------\n",
        "RUN: ", run_id, "\n",
        "Seed: ", run_seed, "\n",
        "weight_alpha: ", weight_alpha, "\n",
        file = log_file, append = TRUE)
    
    cat("Parameters:\n",
        paste(capture.output(print(params)), collapse = "\n"),
        "\n",
        file = log_file, append = TRUE)
    
    # ---------------------------------------------------
    # WEIGHTED LOSS (NEW)
    # ---------------------------------------------------
    if (weight_alpha == 0) {
      sample_weights <- rep(1, nrow(train))
    } else {
      sample_weights <- abs(train$delta_bathy) ^ weight_alpha
      sample_weights <- sample_weights / mean(sample_weights, na.rm = TRUE)
    }
    
    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------
    dtrain <- xgb.DMatrix(
      data   = as.matrix(train[, predictors]),
      label  = train[[response_var]],
      weight = sample_weights,
      missing = NA
    )
    
    xgb_params <- list(
      max_depth        = params$max_depth,
      gamma            = params$gamma,
      lambda           = params$lambda,
      alpha            = params$alpha,
      eta              = params$eta %||% 0.01,
      subsample        = params$subsample %||% 0.7,
      colsample_bytree = params$colsample_bytree %||% 0.8,
      objective        = "reg:squarederror"
    )
    
    nrounds_use <- params$nrounds %||% 300
    
    model <- xgb.train(
      params  = xgb_params,
      data    = dtrain,
      nrounds = nrounds_use,
      verbose = 0
    )
    
    # ---------------------------------------------------
    # PREDICT
    # ---------------------------------------------------
    dpred <- xgb.DMatrix(as.matrix(full_tile[, predictors]))
    pred_t1 <- predict(model, dpred)
    
    # ---------------------------------------------------
    # RESIDUAL METRICS (unchanged)
    # ---------------------------------------------------
    actual_t1 <- full_tile[[response_var]]
    bathy_t   <- full_tile$bathy_t
    delta_act <- full_tile$delta_bathy
    
    pred_change <- pred_t1 - bathy_t
    pred_res    <- actual_t1 - pred_t1
    delta_res   <- delta_act - pred_change
    
    valid_change <- which(
      is.finite(actual_t1) &
        is.finite(pred_t1) &
        is.finite(bathy_t) &
        is.finite(delta_act)
    )
    
    RMSE_bathy <- sqrt(mean(pred_res[valid_change]^2, na.rm = TRUE))
    R2_bathy   <- suppressWarnings(cor(actual_t1[valid_change],
                                       pred_t1[valid_change],
                                       use = "complete.obs")^2)
    
    MAE_delta  <- mean(abs(delta_res[valid_change]), na.rm = TRUE)
    SD_delta   <- sd(delta_res[valid_change], na.rm = TRUE)
    Bias_delta <- mean(delta_res[valid_change], na.rm = TRUE)
    
    cat("Metrics:\n",
        "RMSE_bathy = ", RMSE_bathy, "\n",
        "R2_bathy   = ", R2_bathy, "\n",
        "MAE_delta  = ", MAE_delta, "\n",
        "SD_delta   = ", SD_delta, "\n",
        "Bias_delta = ", Bias_delta, "\n\n",
        file = log_file, append = TRUE)
    
    # ---------------------------------------------------
    # SAVE RASTERS
    # ---------------------------------------------------
    r_change   <- rasterFromXYZ(data.frame(x = full_tile$X, y = full_tile$Y, z = pred_change),
                                crs = grid_crs_proj4)
    r_deltares <- rasterFromXYZ(data.frame(x = full_tile$X, y = full_tile$Y, z = delta_res),
                                crs = grid_crs_proj4)
    r_predres  <- rasterFromXYZ(data.frame(x = full_tile$X, y = full_tile$Y, z = pred_res),
                                crs = grid_crs_proj4)
    
    writeRaster(r_change,   file.path(output_dir, paste0(run_id, "_mean_predicted_change.tif")), overwrite=TRUE)
    writeRaster(r_predres,  file.path(output_dir, paste0(run_id, "_prediction_residual.tif")),     overwrite=TRUE)
    writeRaster(r_deltares, file.path(output_dir, paste0(run_id, "_delta_residual.tif")),         overwrite=TRUE)
    
    # Store results
    results_list[[i]] <- data.table(
      run_id         = run_id,
      seed           = run_seed,
      max_depth      = params$max_depth,
      gamma          = params$gamma,
      lambda         = params$lambda,
      alpha          = params$alpha,
      weight_alpha   = weight_alpha,
      nrounds        = nrounds_use,
      
      RMSE_bathy = RMSE_bathy,
      R2_bathy   = R2_bathy,
      MAE_delta  = MAE_delta,
      SD_delta   = SD_delta,
      Bias_delta = Bias_delta
    )
  }
  
  # -------------------------------------------------------
  # SCORING (unchanged)
  # -------------------------------------------------------
  
  results_dt <- rbindlist(results_list, fill = TRUE)
  
  .norm_minmax <- function(x) {
    rng <- range(x, na.rm = TRUE)
    if (rng[1] == rng[2]) return(rep(0.5, length(x)))
    (x - rng[1]) / (rng[2] - rng[1])
  }
  
  results_dt[, RMSE_norm      := .norm_minmax(RMSE_bathy)]
  results_dt[, MAE_delta_norm := .norm_minmax(MAE_delta)]
  results_dt[, SD_delta_norm  := .norm_minmax(SD_delta)]
  results_dt[, Bias_abs_norm  := .norm_minmax(abs(Bias_delta))]
  
  r2min <- min(results_dt$R2_bathy)
  r2max <- max(results_dt$R2_bathy)
  
  if (r2min != r2max) {
    results_dt[, R2_penalty_norm := (r2max - R2_bathy) / (r2max - r2min)]
  } else {
    results_dt[, R2_penalty_norm := 0.5]
  }
  
  results_dt[, accuracy_score :=
               0.6 * RMSE_norm +
               0.4 * R2_penalty_norm]
  
  results_dt[, change_score :=
               0.4 * MAE_delta_norm +
               0.4 * SD_delta_norm +
               0.2 * Bias_abs_norm]
  
  results_dt[, compromise_score :=
               0.5 * accuracy_score + 0.5 * change_score]
  
  # Save outputs
  fwrite(results_dt, file.path(output_dir, "param_search_results.csv"))
  write_fst(results_dt, file.path(output_dir, "param_search_results.fst"))
  
  return(invisible(results_dt))
}




# 1) Build a parameter grid
param_grid <- expand.grid(
  max_depth    = c(3, 4, 5),
  gamma        = c(0, 0.5, 1),
  lambda       = c(0, 1),
  alpha        = c(0, 1),
  weight_alpha = c(0.5, 1, 1.5),
  nrounds      = c(250),     # or a vector if you want to vary it too
  KEEP.OUT.ATTRS = FALSE,
  stringsAsFactors = FALSE
)

# 2) Run the automated search
results <- XGB_ParamSearch_Auto(
  tile_data_path      = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Training_data_grid_tiles/BH4S556X_3/BH4S556X_3_2004_2006_long.fst",
  output_dir          = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Training_data_grid_tiles/BH4S556X_3",
  predictor_list_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Training_data_grid_tiles/final_master_predictor_list.txt",
  year_pair           = "2004_2006",
  param_grid          = param_grid,
  global_seed         = 2025
)

# 3) Inspect the best runs - # but you  can also ask AI to inspect output CSV to interpret the best run accross all metrics for balance
results[order(compromise_score)][1:5]



