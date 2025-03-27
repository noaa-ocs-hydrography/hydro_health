# V5 - WF 2.0 - New prediction function set---- below is the XGB model version
pdp_data <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/pdp_data_2004_2006.fst")
glimpse(pdp_data1)
training_data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4S656W_4/BH4S656W_4_training_clipped_data.fst")
prediction_data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/BH4S656W_4/BH4S656W_4_prediction_clipped_data.fst")
glimpse(training_data)
glimpse(prediction_data)

# prediction function set
process_tile <- function(tile_id, year_pair, training_dir, prediction_dir, output_dir, mask) {
  message("\n Processing Tile: ", tile_id, " | Year Pair: ", year_pair)
  
  training_file   <- file.path(training_dir, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
  prediction_file <- file.path(prediction_dir, tile_id, paste0(tile_id, "_prediction_clipped_data.fst"))
  pdp_file        <- file.path(training_dir, tile_id, paste0("pdp_data_", year_pair, ".fst"))
  
  if (!all(file.exists(training_file, prediction_file, pdp_file))) {
    message(" Missing required file(s) for ", tile_id, " | ", year_pair)
    return(NULL)
  }
  
  # Read and defensively copy
  training_data   <- copy(read_fst(training_file, as.data.table = TRUE))
  prediction_data <- copy(read_fst(prediction_file, as.data.table = TRUE))
  pdp_data        <- read_fst(pdp_file, as.data.table = TRUE)
  
  # Step 1: Align predictors
  processed <- tryCatch({
    align_predictors(training_data, prediction_data, year_pair)
  }, error = function(e) {
    message("align_predictors failed: ", e$message)
    return(NULL)
  })
  if (is.null(processed)) return(NULL)
  
  # Step 2: Get PDP ranges
  env_ranges <- tryCatch({
    extract_env_variable_ranges(pdp_data)
  }, error = function(e) {
    message("extract_env_variable_ranges failed: ", e$message)
    return(NULL)
  })
  if (is.null(env_ranges)) return(NULL)
  
  # Step 3: Match PDP
  pdp_enriched <- tryCatch({
    match_pdp_conditions(processed$prediction, pdp_data, env_ranges, year_pair)
  }, error = function(e) {
    message("match_pdp_conditions failed: ", e$message)
    return(NULL)
  })
  if (is.null(pdp_enriched)) return(NULL)
  
  # Step 4: Hybrid KNN Adjustments
  final <- tryCatch({
    apply_trend_adjustments_hybrid(pdp_enriched, processed$training, year_pair)
    
  }, error = function(e) {
    message("apply_trend_adjustments_hybrid failed: ", e$message)
    return(NULL)
  })
  if (is.null(final)) return(NULL)
  
  # Step 5: Save Results
  tryCatch({
    save_predictions(final, output_dir, tile_id, year_pair)
   
    # Save prediction output as raster 
     if (exists("mask")) {
      save_prediction_raster(final, output_dir, tile_id, year_pair, raster::crs(mask))
    }
    message("Saved predictions for ", tile_id, " | ", year_pair)
    
  }, error = function(e) {
    message("Failed saving predictions for ", tile_id, " | ", year_pair, ": ", e$message)
  })
  message("PDP Adjusted Change: ", sum(!is.na(prediction_data$pdp_adjusted_change)), "/", nrow(prediction_data))
  
  return(tile_id)
}
align_predictors <- function(training_data, prediction_data, year_pair) {
  setDT(training_data)
  setDT(prediction_data)
  
  start_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][1])
  
  # Define how predictors map from training to prediction
  train_vars <- list(
    bathy     = paste0("bathy_", start_year),
    slope     = paste0("slope_", start_year),
    rugosity  = paste0("Rugosity_nbh9_", start_year),
    tsm       = paste0("tsm_", year_pair),
    hurr_cnt  = paste0("hurr_count_", year_pair),
    hurr_str  = paste0("hurr_strength_", year_pair),
    response  = paste0("b.change.", year_pair)
  )
  
  pred_vars <- list(
    bathy     = "bt.bathy",
    slope     = "bt.slope",
    rugosity  = "bt.bathy.Rugosity_nbh_9",
    tsm       = train_vars$tsm,
    hurr_cnt  = train_vars$hurr_cnt,
    hurr_str  = train_vars$hurr_str
  )
  
  # Ensure all exist
  required_train <- unlist(train_vars)
  required_pred  <- unlist(pred_vars)
  
  missing_train <- setdiff(required_train, names(training_data))
  missing_pred  <- setdiff(required_pred, names(prediction_data))
  
  if (length(missing_train) > 0) stop("Missing training columns: ", paste(missing_train, collapse = ", "))
  if (length(missing_pred) > 0) stop("Missing prediction columns: ", paste(missing_pred, collapse = ", "))
  
  # Common columns (spatial, static)
  static_cols <- c("X", "Y", "FID", "grain_size_layer", "prim_sed_layer", "survey_end_date")
  
  # Rename and slice training data
  training_out <- training_data[, c(static_cols, required_train), with = FALSE]
  setnames(training_out, required_train, c(
    "starting_bathy", "starting_slope", "starting_rugosity",
    "tsm", "hurr_count", "hurr_strength", "b_change"
  ))
  
  # Rename and slice prediction data
  prediction_out <- prediction_data[, c(static_cols, required_pred), with = FALSE]
  setnames(prediction_out, required_pred, c(
    "starting_bathy", "starting_slope", "starting_rugosity",
    "tsm", "hurr_count", "hurr_strength"
  ))
  
  return(list(
    training         = training_out,
    prediction       = prediction_out,
    prediction_orig  = prediction_data
  ))
}
extract_env_variable_ranges <- function(pdp_data) {
  setDT(pdp_data)
  
  if (!"Predictor" %in% names(pdp_data) || !"Env_Value" %in% names(pdp_data)) {
    stop("PDP data must contain columns 'Predictor' and 'Env_Value'")
  }
  
  env_ranges <- pdp_data[
    ,
    .(
      min_val = min(Env_Value, na.rm = TRUE),
      max_val = max(Env_Value, na.rm = TRUE)
    ),
    by = Predictor
  ]
  
  env_ranges[, range_width := max_val - min_val]
  
  return(env_ranges)
}
match_pdp_conditions <- function(prediction_data, pdp_data, env_ranges, year_pair) {
  setDT(prediction_data)
  setDT(pdp_data)
  
  # Sanity Check
  if (!"Predictor" %in% names(pdp_data) || !"Env_Value" %in% names(pdp_data)) {
    stop("PDP data must contain 'Predictor' and 'Env_Value' columns.")
  }
  
  start_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][1])
  
  # Ensure year-based predictor names exist
  predictors <- list(
    bathy     = paste0("bathy_", start_year),
    slope     = paste0("slope_", start_year),
    rugosity  = paste0("Rugosity_nbh9_", start_year)
  )
  
  # Check that predictors are actually present in PDP
  for (pred in predictors) {
    if (!is.character(pred)) stop("Predictor names must be characters.")
    if (!(pred %in% pdp_data$Predictor)) {
      warning(" PDP predictor not found: ", pred)
    }
  }
  
  # Grab numeric environmental values
  bathy_vals    <- prediction_data$starting_bathy
  slope_vals    <- prediction_data$starting_slope
  rugosity_vals <- prediction_data$starting_rugosity
  
  # Grab ranges safely
  get_range <- function(var) {
    val <- env_ranges[Predictor == var, range_width]
    if (length(val) == 0) return(NA_real_) else return(val)
  }
  
  bathy_rng    <- get_range(predictors$bathy)
  slope_rng    <- get_range(predictors$slope)
  rugosity_rng <- get_range(predictors$rugosity)
  
  if (any(is.na(c(bathy_rng, slope_rng, rugosity_rng)))) {
    stop("Missing range width(s) for one or more predictors.")
  }
  
  # Apply row-wise matching
  prediction_data[, pdp_adjusted_change := {
    bathy_val    <- starting_bathy
    slope_val    <- starting_slope
    rugosity_val <- starting_rugosity
    
    matches <- pdp_data[
      (Predictor == predictors$bathy     & between(Env_Value, bathy_val    - 0.3 * bathy_rng,    bathy_val    + 0.3 * bathy_rng)) |
        (Predictor == predictors$slope     & between(Env_Value, slope_val    - 0.1 * slope_rng,    slope_val    + 0.1 * slope_rng)) |
        (Predictor == predictors$rugosity  & between(Env_Value, rugosity_val - 0.1 * rugosity_rng, rugosity_val + 0.1 * rugosity_rng)),
      .(PDP_Value)
    ]
    
    if (nrow(matches) > 0) mean(matches$PDP_Value, na.rm = TRUE) else NA_real_
  }, by = .(X, Y, FID)]
  
  message("PDP Adjusted Change matched on: ", sum(!is.na(prediction_data$pdp_adjusted_change)), "/", nrow(prediction_data))
  
  return(prediction_data)
} # environmental matching 
compare_prediction_methods <- function(prediction_data, tile_id, year_pair, output_dir, mask_crs, raster_diff = TRUE) {
  message(" Comparing PDP vs KNN for: ", tile_id, " | ", year_pair)
  
  # Summary of availability
  total_points <- nrow(prediction_data)
  matched_pdp  <- sum(!is.na(prediction_data$pdp_adjusted_change))
  matched_knn  <- sum(!is.na(prediction_data$pred_avg_change))
  matched_both <- sum(!is.na(prediction_data$pdp_adjusted_change) & !is.na(prediction_data$pred_avg_change))
  matched_none <- sum(is.na(prediction_data$pdp_adjusted_change) & is.na(prediction_data$pred_avg_change))
  
  # Correlation and diff stats
  overlap_df <- prediction_data[!is.na(pdp_adjusted_change) & !is.na(pred_avg_change)]
  corr <- if (nrow(overlap_df) > 2) cor(overlap_df$pdp_adjusted_change, overlap_df$pred_avg_change) else NA
  mean_diff <- if (nrow(overlap_df) > 2) mean(overlap_df$pdp_adjusted_change - overlap_df$pred_avg_change) else NA
  
  # Create hybrid column (use PDP where available, fallback to KNN)
  prediction_data[, hybrid_change := fifelse(
    !is.na(pdp_adjusted_change), pdp_adjusted_change,
    fifelse(!is.na(pred_avg_change), pred_avg_change, NA_real_)
  )]
  
  # Save summary log
  diag_summary <- data.table(
    Tile_ID        = tile_id,
    Year_Pair      = year_pair,
    Total_Points   = total_points,
    PDP_Matched    = matched_pdp,
    KNN_Matched    = matched_knn,
    Both_Matched   = matched_both,
    Neither_Matched = matched_none,
    PDP_KNN_Cor    = corr,
    PDP_KNN_MeanDiff = mean_diff
  )
  
  diag_log_file <- file.path(output_dir, "prediction_method_comparison_log.csv")
  if (!file.exists(diag_log_file)) {
    fwrite(diag_summary, diag_log_file)
  } else {
    fwrite(diag_summary, diag_log_file, append = TRUE)
  }
  
  # Create raster of PDP - KNN difference (where both exist)
  if (raster_diff && "X" %in% names(prediction_data) && "Y" %in% names(prediction_data)) {
    diff_df <- prediction_data[!is.na(pdp_adjusted_change) & !is.na(pred_avg_change)]
    if (nrow(diff_df) > 0) {
      diff_raster <- raster::rasterFromXYZ(
        data.frame(
          x = diff_df$X,
          y = diff_df$Y,
          z = diff_df$pdp_adjusted_change - diff_df$pred_avg_change
        ),
        crs = mask_crs
      )
      diff_out <- file.path(output_dir, tile_id, paste0(tile_id, "_PDP_KNN_diff_", year_pair, ".tif"))
      raster::writeRaster(diff_raster, diff_out, format = "GTiff", overwrite = TRUE)
      message("🗺️  Difference raster written: ", diff_out)
    }
  }
  
  return(prediction_data)  # Now includes hybrid_change
}
apply_trend_adjustments_hybrid <- function(prediction_data, training_data, year_pair, k = 5) {
  setDT(prediction_data)
  setDT(training_data)
  
  env_vars <- intersect(
    c("starting_bathy", "starting_slope", "starting_rugosity", "tsm", "hurr_count", "hurr_strength"),
    intersect(names(prediction_data), names(training_data))
  )
  
  if (length(env_vars) == 0) stop("No environmental variables available for KNN matching.")
  
  message(" KNN Matching on variables: ", paste(env_vars, collapse = ", "))
  
  for (var in env_vars) {
    prediction_data[[var]] <- suppressWarnings(as.numeric(prediction_data[[var]]))
    training_data[[var]] <- suppressWarnings(as.numeric(training_data[[var]]))
  }
  
  pred_clean <- prediction_data[complete.cases(prediction_data[, ..env_vars])]
  train_clean <- training_data[complete.cases(training_data[, ..env_vars])]
  
  message(" Dropped from prediction: ", nrow(prediction_data) - nrow(pred_clean), " rows")
  message(" Dropped from training: ", nrow(training_data) - nrow(train_clean), " rows")
  
  # Fixed response variable handling
  response_var <- "b_change"
  message("🔍 Using response variable: ", response_var)
  message("📋 Available training columns: ", paste(names(train_clean), collapse = ", "))
  
  if (!response_var %in% names(train_clean)) {
    stop(paste0(" Missing response variable in training_data: ", response_var))
  }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]]
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  weights <- 1 / (knn_result$nn.dist + 1e-6)
  weight_sums <- rowSums(weights, na.rm = TRUE)
  pred_avg <- rowSums(train_resp[knn_result$nn.index] * weights, na.rm = TRUE) / weight_sums
  
  prediction_data[, pred_avg_change := NA_real_]
  prediction_data[complete.cases(prediction_data[, ..env_vars]), pred_avg_change := pred_avg]
  
  message(" Assigned KNN predictions to ", sum(!is.na(prediction_data$pred_avg_change)), " rows.")
  
  return(prediction_data)
}
save_predictions <- function(prediction_data, output_dir, tile_id, year_pair) {
  out_file <- file.path(output_dir, paste0(tile_id, "_prediction_", year_pair, ".fst"))
  write_fst(prediction_data, out_file)
}
# Save raster TIFF from predicted change
save_prediction_raster <- function(prediction_data, output_dir, tile_id, year_pair, crs_obj) {
  if (!"X" %in% names(prediction_data) || !"Y" %in% names(prediction_data) || !"pred_avg_change" %in% names(prediction_data)) {
    message("Cannot save raster — required columns missing.")
    return(NULL)
  }
  
  raster_df <- data.frame(
    x = prediction_data[["X"]],
    y = prediction_data[["Y"]],
    z = prediction_data[["pred_avg_change"]]
  )
  
  crs_obj <- raster::projection(mask)
  pred_rast <- raster::rasterFromXYZ(raster_df, crs = crs_obj)
  
  out_raster_path <- file.path(output_dir, tile_id, paste0(tile_id, "_predicted_change_", year_pair, ".tif"))
  
  raster::writeRaster(pred_rast, out_raster_path, format = "GTiff", overwrite = TRUE)
  message("🗺️  Raster written: ", out_raster_path)
}
# run functions
run_predictions <- function(tile_ids, year_pairs, mask, training_dir, prediction_dir, output_dir, num_cores = parallel::detectCores() - 1) {
  message("\n Running Predictions with ", num_cores, " cores...")
  
  plan(multisession, workers = num_cores)
  handlers(global = TRUE)
  handlers("progress")
  
  p <- progressor(along = expand.grid(tile_ids, year_pairs))
  
  results <- with_progress({
    future_lapply(
      split(expand.grid(tile_ids, year_pairs), seq_len(nrow(expand.grid(tile_ids, year_pairs)))),
      function(params) {
        tile <- params[[1]]
        year <- params[[2]]
        p(sprintf("Tile: %s | Year: %s", tile, year))
        process_tile(tile, year, training_dir, prediction_dir, output_dir, mask = mask) 
      },
      future.seed = TRUE
    )
  })
  
  
  message("\n All predictions complete.")
  return(results)
}

# Run prediction in parallel over multiple tiles / all tiles----
tile_ids <- c("BH4S656W_4", "BH4S656X_1")
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")

training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles"
# run on all tiles
run_predictions(tile_ids, year_pairs, training_dir, prediction_dir, output_dir, num_cores = 1)


# Testing prediction on a single tile(s)----- 
# Mask reference
mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif")

# Only this training tile will be available for reference
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles"
prediction_output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles"

# Custom wrapper to only read training data from specific tiles
run_prediction_cross_tile <- function(training_tiles, prediction_tiles, year_pairs, mask, training_dir, prediction_dir, prediction_output_dir) {
  
  for (pred_tile in prediction_tiles) {
    for (year in year_pairs) {
      
      message("\n Predicting for tile ", pred_tile, " using training from: ", paste(training_tiles, collapse = ", "))
      
      # Merge training data from all training tiles
      training_data <- lapply(training_tiles, function(tile) {
        tf <- file.path(training_dir, tile, paste0(tile, "_training_clipped_data.fst"))
        if (file.exists(tf)) read_fst(tf, as.data.table = TRUE) else NULL
      }) %>% rbindlist(fill = TRUE)
      
      # Path to prediction file
      prediction_file <- file.path(prediction_dir, pred_tile, paste0(pred_tile, "_prediction_clipped_data.fst"))
      pdp_file        <- file.path(training_dir, training_tiles[1], paste0("pdp_data_", year, ".fst"))  # From any training tile
      
      if (!file.exists(prediction_file) || !file.exists(pdp_file)) {
        message("Required file missing.")
        next
      }
      
      prediction_data <- read_fst(prediction_file, as.data.table = TRUE)
      pdp_data        <- read_fst(pdp_file, as.data.table = TRUE)
      
      #  Align, enrich, predict
      processed <- align_predictors(training_data, prediction_data, year)
      env_ranges <- extract_env_variable_ranges(pdp_data)
      enriched   <- match_pdp_conditions(processed$prediction, pdp_data, env_ranges, year)
      final      <- apply_trend_adjustments_hybrid(enriched, processed$training, year)
      
      # Log diagnostics
      diag_log <- data.table(
        Tile_ID         = pred_tile,
        Year_Pair       = year,
        Total_Points    = nrow(final),
        PDP_Matched     = sum(!is.na(final$pdp_adjusted_change)),
        # Match_Quality = ifelse(KNN_Ratio < 0.5, " Sparse Match", "Good Match"),
        KNN_Matched     = sum(!is.na(final$pred_avg_change)),
        Output_Raster   = file.path(prediction_output_dir, pred_tile, paste0(pred_tile, "_predicted_change_", year, ".tif"))
      )
      
      # Append to CSV log
      log_file <- file.path(prediction_output_dir, "prediction_diagnostics.csv")
      if (!file.exists(log_file)) {
        fwrite(diag_log, log_file)
      } else {
        fwrite(diag_log, log_file, append = TRUE)
      }
      
      
      
      # 🔄 Save both tabular and raster
      save_predictions(final, prediction_output_dir, pred_tile, year)
      save_prediction_raster(final, prediction_output_dir, pred_tile, year, raster::projection(mask))
      
      message("✅ Prediction complete for ", pred_tile)
    }
  }
}

run_prediction_cross_tile(
  training_tiles   = c("BH4S2574_1", "BH4S2574_2", "BH4S2572_3"),          
  prediction_tiles = c("BH4S2574_1", "BH4S2572_4", "BH4S2572_1"),         
  year_pairs       = c("2004_2006"),
  mask             = mask,
  training_dir     = training_dir,
  prediction_dir   = prediction_dir,
  prediction_output_dir       = prediction_output_dir
)


# Post Processing functions (still in development)

extract_final_trend <- function(prediction_data_list) {
  combined_predictions <- rbindlist(prediction_data_list, fill = TRUE)
  combined_predictions[, final_trend := rowMeans(.SD, na.rm = TRUE), .SDcols = "pred_avg_change"]
  return(combined_predictions)
}


# apply depth attenuation
apply_depth_attenuation <- function(prediction_data, coeff = 0.05) {
  setDT(prediction_data)
  
  message("🌊 Applying Depth Attenuation...")
  
  prediction_data[, depth_attenuated_change := trend_adjusted_change * exp(-coeff * abs(starting_bathy))]
  
  return(prediction_data)
}
