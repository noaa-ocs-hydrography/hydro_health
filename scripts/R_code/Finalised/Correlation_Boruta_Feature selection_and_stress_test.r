

run_predictor_correlation_check <- function(
    training_dir,
    predictors,
    year_pair = NULL,
    file_pattern = "_long\\.fst$",
    cutoff = 0.9,
    output_png = NULL
) {
  # ---------------------------------------------------------------------------
  # Purpose:
  # Load one training *_long.fst dataset, calculate predictor correlations,
  # save a correlation plot, and return highly correlated predictor pairs.
  # ---------------------------------------------------------------------------
  
  # Required packages
  required_pkgs <- c("data.table", "fst", "corrplot")
  missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  
  if (length(missing_pkgs) > 0) {
    stop(
      "Missing required package(s): ",
      paste(missing_pkgs, collapse = ", "),
      call. = FALSE
    )
  }
  
  # ---------------------------------------------------------------------------
  # 1. Find the training file
  # ---------------------------------------------------------------------------
  files_found <- list.files(
    path = training_dir,
    pattern = file_pattern,
    full.names = TRUE
  )
  
  if (length(files_found) == 0) {
    stop("No *_long.fst files found in: ", training_dir, call. = FALSE)
  }
  
  if (!is.null(year_pair)) {
    pair_pattern <- paste0("_", year_pair, "_long\\.fst$")
    files_found <- files_found[grepl(pair_pattern, basename(files_found))]
    
    if (length(files_found) == 0) {
      stop(
        "No *_", year_pair, "_long.fst file found in: ",
        training_dir,
        call. = FALSE
      )
    }
  }
  
  if (length(files_found) > 1) {
    message("Multiple matching files found. Using the first one: ", basename(files_found[1]))
  }
  
  target_file <- files_found[1]
  
  # Default output filename
  if (is.null(output_png)) {
    file_stub <- tools::file_path_sans_ext(basename(target_file))
    output_png <- file.path(
      training_dir,
      paste0(file_stub, "_predictor_correlation.png")
    )
  }
  
  # ---------------------------------------------------------------------------
  # 2. Load data and convert to data.table
  # ---------------------------------------------------------------------------
  train <- fst::read_fst(target_file, as.data.table = TRUE)
  data.table::setDT(train)
  
  # ---------------------------------------------------------------------------
  # 3. Check predictor availability
  # ---------------------------------------------------------------------------
  missing_predictors <- setdiff(predictors, names(train))
  
  if (length(missing_predictors) > 0) {
    stop(
      "These predictors were not found in the dataset: ",
      paste(missing_predictors, collapse = ", "),
      call. = FALSE
    )
  }
  
  # Keep only complete rows for the selected predictors
  corr_data <- stats::na.omit(train[, ..predictors])
  
  if (nrow(corr_data) < 2) {
    stop("Not enough complete rows to compute a correlation matrix.", call. = FALSE)
  }
  
  # ---------------------------------------------------------------------------
  # 4. Calculate correlation matrix
  # ---------------------------------------------------------------------------
  cor_matrix <- stats::cor(
    corr_data,
    use = "complete.obs"
  )
  
  # ---------------------------------------------------------------------------
  # 5. Save correlation plot
  # ---------------------------------------------------------------------------
  grDevices::png(
    filename = output_png,
    width = 1600,
    height = 1400,
    res = 150
  )
  
  corrplot::corrplot(
    cor_matrix,
    method = "color",
    type = "upper",
    order = "original",
    tl.col = "red",
    tl.cex = 0.8,
    tl.srt = 90,
    addCoef.col = NULL
  )
  
  graphics::title(
    main = paste0(
      "Predictor correlation matrix\n",
      basename(target_file)
    )
  )
  
  grDevices::dev.off()
  
  # ---------------------------------------------------------------------------
  # 6. Extract highly correlated pairs
  # Only inspect the upper triangle so each pair appears once.
  # ---------------------------------------------------------------------------
  cor_upper <- cor_matrix
  cor_upper[lower.tri(cor_upper, diag = TRUE)] <- NA_real_
  
  high_idx <- which(abs(cor_upper) >= cutoff, arr.ind = TRUE)
  
  if (nrow(high_idx) > 0) {
    high_corr_pairs <- data.table::data.table(
      predictor_1 = rownames(cor_upper)[high_idx[, 1]],
      predictor_2 = colnames(cor_upper)[high_idx[, 2]],
      correlation = cor_upper[high_idx]
    )
    
    high_corr_pairs <- high_corr_pairs[
      order(-abs(correlation), predictor_1, predictor_2)
    ]
  } else {
    high_corr_pairs <- data.table::data.table(
      predictor_1 = character(),
      predictor_2 = character(),
      correlation = numeric()
    )
  }
  
  # ---------------------------------------------------------------------------
  # 7. Suggest predictors for review
  # This is intentionally conservative:
  # if a variable appears in any high-correlation pair, flag it for review.
  # ---------------------------------------------------------------------------
  predictors_to_review <- unique(c(
    high_corr_pairs$predictor_1,
    high_corr_pairs$predictor_2
  ))
  
  # Optional frequency summary: which variables appear most often
  if (length(predictors_to_review) > 0) {
    review_frequency <- sort(table(c(
      high_corr_pairs$predictor_1,
      high_corr_pairs$predictor_2
    )), decreasing = TRUE)
  } else {
    review_frequency <- integer(0)
  }
  
  # ---------------------------------------------------------------------------
  # 8. Return results
  # ---------------------------------------------------------------------------
  list(
    file_used = target_file,
    output_png = output_png,
    predictors_checked = predictors,
    cutoff = cutoff,
    n_rows_used = nrow(corr_data),
    correlation_matrix = cor_matrix,
    high_correlation_pairs = high_corr_pairs,
    predictors_to_review = predictors_to_review,
    review_frequency = review_frequency
  )
}

#Function Call 
predictors <- c(
  "bathy_t",
  "tci_t",
  "grain_size_layer",
  "prim_sed_layer",
  "hurr_strength_mean_2004_2006",
  "tsm_mean_2004_2006",
  "flowdir_cos_t",
  "flowdir_sin_t",
  "bpi_broad_t",
  "curv_total_t",
  "slope_t",
  "rugosity_t",
  "flowacc_t",
  "shearproxy_t",
  "gradmag_t"
)

corr_results <- run_predictor_correlation_check(
  training_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Training_data_grid_tiles/BH4S656W_2",
  predictors = predictors,
  year_pair = "2004_2006",
  cutoff = 0.9
)


# check outputs
corr_results$file_used
plot(corr_results$output_png)
corr_results$high_correlation_pairs
corr_results$predictors_to_review
# [1] "slope_t"          "rugosity_t"       "bathy_t"          "grain_size_layer" "gradmag_t"        "tci_t"            "prim_sed_layer"  


# plot 
library(png)
library(grid)

img <- readPNG(corr_results$output_png)
grid::grid.raster(img)

# Now, remove redundant predictors, after considering the list above, that are highly correlated so they are not included in the Boruta script 
# add variable name to 'alwas exclude'


# New Streamlined Boruta Predictor Selection, and stress testing  



# ==============================================================================
# Feature selection + stress testing pipeline
# Single-script refactor
# ------------------------------------------------------------------------------
# What this script does:
# 1. Samples training data for each year-pair.
# 2. Runs Boruta feature selection with consistency rules.
# 3. Builds grouped predictor configurations.
# 4. Runs a spatial cross-validation stress test on those configurations.
# 5. Saves per-pair outputs to disk.
# 6. Produces cross-pair summary outputs at the end.
#
# This is designed to run start-to-finish in one script.
# ==============================================================================

# ------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------
library(Boruta)
library(data.table)
library(doParallel)
library(dplyr)
library(foreach)
library(fst)
library(parallel)
library(ranger)
library(spdep)
library(stringr)
library(tibble)


# ------------------------------------------------------------------------------
# Global defaults and helper constants
# ------------------------------------------------------------------------------
response_var <- "bathy_t1"

id_vars <- c(
  "X", "Y", "FID", "tile_id", "year_t", "year_t1"
)

always_exclude <- c(
  response_var,
  id_vars,
  "survey_end_date",
  "delta_bathy", # causes data leakage if included, we want to model to learn this, and we dont have it for prediction data 
  "rugosity_t",
  "slope_t",
  "curve_plan_t",
  "slope_deg_t",
  "tci_t"
)

default_seed <- 123L

# ------------------------------------------------------------------------------
# Logging helper
# Simple timestamped console messages so progress is easier to follow.
# ------------------------------------------------------------------------------
log_message <- function(..., .time = TRUE) {
  msg <- paste0(...)
  if (.time) {
    msg <- paste0(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", msg)
  }
  message(msg)
}

# ------------------------------------------------------------------------------
# Flow direction transformation
# Converts directional variables to sin/cos representations and removes the
# original raw flow direction columns.
# ------------------------------------------------------------------------------
transform_flowdir_cols <- function(dt) {
  dt
}

# ------------------------------------------------------------------------------
# Predictor consistency rules
# Enforces:
# 1. Sin/cos pairing for flow direction.
# 2. Parent-child consistency for selected derived geomorph variables.
# 3. Generic neighborhood-stat parent rule.
# ------------------------------------------------------------------------------
enforce_predictor_consistency <- function(
    confirmed_predictors,
    all_predictors
) {
  # Flow direction pairing
  flow_pairs <- list(
    flowdir_sin_t = "flowdir_cos_t",
    flowdir_cos_t = "flowdir_sin_t"
  )
  
  for (pred in names(flow_pairs)) {
    pair_pred <- flow_pairs[[pred]]
    
    if (pred %in% confirmed_predictors && pair_pred %in% all_predictors) {
      confirmed_predictors <- unique(c(confirmed_predictors, pair_pred))
    }
  }
  
  # Explicit parent-child hierarchy
  parent_map <- list(
    curv_total_t = "slope_t",
    curv_plan_t = "slope_t",
    curv_profile_t = "slope_t",
    gradmag_t = "slope_t",
    bpi_fine_mean3_t = "bpi_fine_t",
    bpi_fine_sd3_t = "bpi_fine_t",
    bpi_broad_mean3_t = "bpi_broad_t",
    bpi_broad_sd3_t = "bpi_broad_t",
    slope_mean3_t = "slope_t",
    slope_sd3_t = "slope_t",
    rugosity_mean3_t = "rugosity_t",
    rugosity_sd3_t = "rugosity_t",
    tci_t = "gradmag_t",
    flowacc_t = "gradmag_t"
  )
  
  for (child in names(parent_map)) {
    parent <- parent_map[[child]]
    
    if (child %in% confirmed_predictors && parent %in% all_predictors) {
      confirmed_predictors <- unique(c(confirmed_predictors, parent))
    }
  }
  
  # Generic neighborhood parent rule
  neighborhood_stats <- confirmed_predictors[
    str_detect(confirmed_predictors, "_mean\\d|_sd\\d")
  ]
  
  if (length(neighborhood_stats) > 0) {
    parent_vars <- str_replace(neighborhood_stats, "_mean\\d|_sd\\d", "")
    parents_to_add <- intersect(parent_vars, all_predictors)
    confirmed_predictors <- unique(c(confirmed_predictors, parents_to_add))
  }
  
  confirmed_predictors
}

# ------------------------------------------------------------------------------
# File discovery helper
# Finds all training files for a single year-pair.
# ------------------------------------------------------------------------------
find_training_files <- function(training_base_dir, year_pair) {
  file_pattern <- paste0("_", year_pair, "_long\\.fst$")
  
  list.files(
    path = training_base_dir,
    pattern = file_pattern,
    recursive = TRUE,
    full.names = TRUE
  )
}

# ------------------------------------------------------------------------------
# Shared sampling function
# This replaces the duplicated chunked sampling logic from the original code.
# It reads files in chunks, samples each chunk, combines the results, and
# performs any common preprocessing needed by downstream steps.
# ------------------------------------------------------------------------------
sample_training_data <- function(
    training_base_dir,
    year_pair,
    sample_size = 100000,
    chunk_size = 5,
    seed = default_seed,
    transform_flowdir = TRUE
) {
  files_to_process <- find_training_files(
    training_base_dir = training_base_dir,
    year_pair = year_pair
  )
  
  if (length(files_to_process) == 0) {
    stop("No long-format training files found for pair: ", year_pair)
  }
  
  file_chunks <- split(
    files_to_process,
    ceiling(seq_along(files_to_process) / chunk_size)
  )
  
  samples_per_chunk <- max(100, ceiling(sample_size / length(file_chunks)))
  sampled_chunks <- vector("list", length(file_chunks))
  
  for (i in seq_along(file_chunks)) {
    chunk_files <- file_chunks[[i]]
    
    chunk_list <- lapply(chunk_files, function(path) {
      tryCatch(
        fst::read_fst(path, as.data.table = TRUE),
        error = function(e) NULL
      )
    })
    
    chunk_list <- chunk_list[!vapply(chunk_list, is.null, logical(1))]
    
    if (length(chunk_list) == 0) {
      next
    }
    
    chunk_dt <- data.table::rbindlist(chunk_list, use.names = TRUE, fill = TRUE)
    
    if (nrow(chunk_dt) > 0) {
      set.seed(seed + i)
      n_sample <- min(nrow(chunk_dt), samples_per_chunk)
      sampled_chunks[[i]] <- chunk_dt[sample(.N, n_sample)]
    }
    
    rm(chunk_list, chunk_dt)
    invisible(gc())
  }
  
  sampled_chunks <- sampled_chunks[!vapply(sampled_chunks, is.null, logical(1))]
  
  if (length(sampled_chunks) == 0) {
    stop("No valid sampled data could be created for pair: ", year_pair)
  }
  
  dt <- data.table::rbindlist(sampled_chunks, use.names = TRUE, fill = TRUE)
  
  if (!"tile_id" %in% names(dt)) {
    stop("tile_id column is required for spatial CV.")
  }
  
  if (transform_flowdir) {
    dt <- transform_flowdir_cols(dt)
  }
  
  dt
}

# ------------------------------------------------------------------------------
# Boruta data preparation
# Keeps predictor-prep logic in one place.
# ------------------------------------------------------------------------------
prepare_boruta_data <- function(
    dt,
    response = response_var,
    vars_to_exclude = always_exclude
) {
  all_potential_predictors <- setdiff(names(dt), vars_to_exclude)
  
  # Remove delta neighborhood summaries, keeping delta "core" variables only
  deltas_to_remove <- all_potential_predictors[
    str_detect(all_potential_predictors, "^delta_.*(_mean\\d|_sd\\d)$")
  ]
  
  potential_predictors <- setdiff(all_potential_predictors, deltas_to_remove)
  
  boruta_dt <- na.omit(dt[, c(response, potential_predictors), with = FALSE])
  
  list(
    boruta_dt = boruta_dt,
    potential_predictors = potential_predictors,
    removed_delta_predictors = deltas_to_remove
  )
}

# ------------------------------------------------------------------------------
# Boruta runner for a single year-pair
# Returns confirmed predictors plus the full Boruta importance table.
# ------------------------------------------------------------------------------
run_boruta_for_pair <- function(
    dt,
    year_pair,
    response = response_var,
    max_runs = 100
) {
  prepared <- prepare_boruta_data(
    dt = dt,
    response = response
  )
  
  boruta_dt <- prepared$boruta_dt
  potential_predictors <- prepared$potential_predictors
  
  if (nrow(boruta_dt) < 100) {
    stop("Insufficient non-missing rows for Boruta in pair: ", year_pair)
  }
  
  x_data <- as.data.frame(boruta_dt[, potential_predictors, with = FALSE])
  y_data <- boruta_dt[[response]]
  
  boruta_fit <- Boruta::Boruta(
    x = x_data,
    y = y_data,
    maxRuns = max_runs,
    doTrace = 0
  )
  
  confirmed_predictors <- Boruta::getSelectedAttributes(
    boruta_fit,
    withTentative = FALSE
  )
  
  confirmed_predictors <- enforce_predictor_consistency(
    confirmed_predictors = confirmed_predictors,
    all_predictors = potential_predictors
  )
  
  boruta_stats <- Boruta::attStats(boruta_fit) |>
    as.data.frame() |>
    tibble::rownames_to_column(var = "predictor") |>
    dplyr::arrange(desc(meanImp))
  
  list(
    year_pair = year_pair,
    confirmed_predictors = confirmed_predictors,
    boruta_statistics = boruta_stats,
    potential_predictors = potential_predictors,
    n_rows_used = nrow(boruta_dt),
    n_predictors_considered = length(potential_predictors)
  )
}

# ------------------------------------------------------------------------------
# Predictor grouping
# Used to create meaningful model configurations for the stress test.
# ------------------------------------------------------------------------------
categorise_predictors <- function(predictors) {
  data.table(
    predictor = predictors
  )[
    ,
    group := dplyr::case_when(
      predictor == "bathy_t" ~ "bathy_base",
      grepl("^bathy_", predictor) ~ "bathy_other",
      grepl("hurr_|tsm_", predictor) ~ "storms",
      grepl("grain_size_layer|prim_sed_layer", predictor) ~ "sediment",
      grepl(
        "bpi_|rugosity|slope_|curv_|gradmag_|flowacc_|tci_|shearproxy|flowdir_.*_(sin|cos)",
        predictor
      ) ~ "geomorph",
      grepl("^delta_", predictor) ~ "delta",
      TRUE ~ "other"
    )
  ]
}

# ------------------------------------------------------------------------------
# Model configuration builder
# Creates a small, interpretable set of grouped model recipes.
# ------------------------------------------------------------------------------
build_model_configs <- function(groups_present) {
  has_groups <- function(groups) {
    all(groups %in% groups_present)
  }
  
  configs <- list()
  
  if (has_groups("bathy_base")) {
    configs[["bathy_only"]] <- c("bathy_base")
  }
  
  if (has_groups(c("bathy_base", "geomorph"))) {
    configs[["bathy_geomorph"]] <- c("bathy_base", "geomorph")
  }
  
  if (has_groups(c("bathy_base", "storms"))) {
    configs[["bathy_storms"]] <- c("bathy_base", "storms")
  }
  
  if (has_groups("geomorph")) {
    configs[["geomorph_only"]] <- c("geomorph")
  }
  
  if (has_groups(c("storms", "geomorph"))) {
    configs[["storms_geomorph"]] <- c("storms", "geomorph")
  }
  
  if (has_groups(c("bathy_base", "geomorph", "storms"))) {
    configs[["bathy_geomorph_storms"]] <- c("bathy_base", "geomorph", "storms")
  }
  
  full_groups <- setdiff(groups_present, "other")
  configs[["full_model"]] <- full_groups
  
  configs
}

# ------------------------------------------------------------------------------
# Spatial cross-validation
# Splits by tile_id so validation is spatially grouped rather than random.
# ------------------------------------------------------------------------------
run_spatial_cv <- function(
    dt,
    predictors,
    response = response_var,
    k_folds = 5,
    seed = default_seed,
    compute_moran = TRUE
) {
  required_cols <- c(response, "tile_id")
  
  if (!all(required_cols %in% names(dt))) {
    stop("Response or tile_id is missing from the dataset.")
  }
  
  cols_needed <- c(response, predictors, "tile_id", "X", "Y")
  cols_needed <- intersect(cols_needed, names(dt))
  
  dt_use <- dt[, ..cols_needed]
  dt_use <- na.omit(dt_use)
  
  if (nrow(dt_use) < 200) {
    stop("Not enough complete rows for CV.")
  }
  
  set.seed(seed)
  unique_tiles <- unique(dt_use$tile_id)
  
  if (length(unique_tiles) >= k_folds) {
    fold_ids <- sample(rep(seq_len(k_folds), length.out = length(unique_tiles)))
    
    tile_lookup <- data.table(
      tile_id = unique_tiles,
      fold = fold_ids
    )
    
    dt_use <- merge(dt_use, tile_lookup, by = "tile_id")
  } else {
    warning(
      "Fewer unique tile_id values than k_folds. ",
      "Using row-based folds instead of tile-based spatial folds."
    )
    
    dt_use[, fold := sample(rep(seq_len(k_folds), length.out = .N))]
  }
  
  predictions_all <- numeric(0)
  observations_all <- numeric(0)
  coords_all <- NULL
  
  for (fold_k in seq_len(k_folds)) {
    train_dt <- dt_use[fold != fold_k]
    valid_dt <- dt_use[fold == fold_k]
    
    if (nrow(valid_dt) < 10 || nrow(train_dt) < 10) {
      next
    }
    
    x_train <- train_dt[, ..predictors]
    y_train <- train_dt[[response]]
    x_valid <- valid_dt[, ..predictors]
    y_valid <- valid_dt[[response]]
    
    fit <- ranger::ranger(
      x = x_train,
      y = y_train,
      num.trees = 500,
      mtry = max(1, floor(sqrt(length(predictors)))),
      importance = "permutation"
    )
    
    pred_valid <- predict(fit, data = x_valid)$predictions
    
    predictions_all <- c(predictions_all, pred_valid)
    observations_all <- c(observations_all, y_valid)
    
    if (all(c("X", "Y") %in% names(valid_dt))) {
      coords_all <- rbind(coords_all, as.matrix(valid_dt[, .(X, Y)]))
    }
  }
  
  if (length(predictions_all) == 0) {
    stop("No predictions were generated during CV.")
  }
  
  residuals <- observations_all - predictions_all
  
  rmse <- sqrt(mean((observations_all - predictions_all)^2))
  mae <- mean(abs(observations_all - predictions_all))
  r2 <- 1 - sum((observations_all - predictions_all)^2) /
    sum((observations_all - mean(observations_all))^2)
  
  moran_i <- NA_real_
  
  if (compute_moran && !is.null(coords_all) && requireNamespace("spdep", quietly = TRUE)) {
    max_moran_n <- 10000
    
    if (nrow(coords_all) > max_moran_n) {
      set.seed(seed)
      idx <- sample(seq_len(nrow(coords_all)), max_moran_n)
      coords_m <- coords_all[idx, , drop = FALSE]
      residuals_m <- residuals[idx]
    } else {
      coords_m <- coords_all
      residuals_m <- residuals
    }
    
    nb <- spdep::knearneigh(coords_m, k = 8)
    nb <- spdep::knn2nb(nb)
    lw <- spdep::nb2listw(nb, style = "W")
    
    moran_test <- spdep::moran.test(residuals_m, lw, zero.policy = TRUE)
    moran_i <- as.numeric(moran_test$estimate[["Moran I statistic"]])
  }
  
  list(
    rmse = rmse,
    mae = mae,
    r2 = r2,
    moran_I = moran_i
  )
}

# ------------------------------------------------------------------------------
# Stress test for a single year-pair
# Uses Boruta-selected predictors, groups them, builds model configs, and
# evaluates each config with spatial CV.
# ------------------------------------------------------------------------------
run_stress_test_for_pair <- function(
    dt,
    year_pair,
    confirmed_predictors,
    k_folds = 5,
    compute_moran = TRUE,
    response = response_var
) {
  confirmed_predictors <- unique(c(confirmed_predictors, "bathy_t"))
  confirmed_predictors <- intersect(confirmed_predictors, names(dt))
  
  predictor_metadata <- categorise_predictors(confirmed_predictors)
  groups_present <- unique(predictor_metadata$group)
  
  model_configs <- build_model_configs(groups_present)
  results_list <- list()
  
  for (config_name in names(model_configs)) {
    config_groups <- model_configs[[config_name]]
    
    config_predictors <- predictor_metadata[
      group %in% config_groups,
      predictor
    ]
    
    config_predictors <- unique(config_predictors)
    
    if ("bathy_base" %in% config_groups && !"bathy_t" %in% config_predictors) {
      config_predictors <- c("bathy_t", config_predictors)
    }
    
    if (length(config_predictors) == 0) {
      next
    }
    
    metrics <- tryCatch(
      run_spatial_cv(
        dt = dt,
        predictors = config_predictors,
        response = response,
        k_folds = k_folds,
        compute_moran = compute_moran
      ),
      error = function(e) {
        warning(
          "CV failed for pair ", year_pair,
          ", config ", config_name,
          ": ", e$message
        )
        NULL
      }
    )
    
    if (!is.null(metrics)) {
      results_list[[config_name]] <- data.table(
        year_pair = year_pair,
        model_name = config_name,
        n_predictors = length(config_predictors),
        rmse = metrics$rmse,
        mae = metrics$mae,
        r2 = metrics$r2,
        moran_I = metrics$moran_I
      )
    }
  }
  
  if (length(results_list) == 0) {
    return(NULL)
  }
  
  stress_results <- rbindlist(results_list, use.names = TRUE)
  
  if ("bathy_only" %in% stress_results$model_name) {
    base_rmse <- stress_results[model_name == "bathy_only", rmse][1]
    base_moran <- stress_results[model_name == "bathy_only", moran_I][1]
    
    stress_results[
      ,
      delta_rmse_vs_bathy := if (!is.na(base_rmse)) rmse - base_rmse else NA_real_
    ]
    
    stress_results[
      ,
      delta_moran_vs_bathy := if (!is.na(base_moran)) moran_I - base_moran else NA_real_
    ]
  } else {
    stress_results[
      ,
      `:=`(
        delta_rmse_vs_bathy = NA_real_,
        delta_moran_vs_bathy = NA_real_
      )
    ]
  }
  
  stress_results[]
}

# ------------------------------------------------------------------------------
# Per-pair output saver
# Keeps disk writing separate from the modeling logic.
# ------------------------------------------------------------------------------
save_pair_outputs <- function(
    training_base_dir,
    year_pair,
    boruta_result,
    stress_result
) {
  boruta_path <- file.path(
    training_base_dir,
    paste0("boruta_results_", year_pair, ".rds")
  )
  
  saveRDS(
    list(
      confirmed_predictors = boruta_result$confirmed_predictors,
      boruta_statistics = boruta_result$boruta_statistics
    ),
    file = boruta_path
  )
  
  if (!is.null(stress_result)) {
    stress_path <- file.path(
      training_base_dir,
      paste0("stress_test_results_", year_pair, ".fst")
    )
    
    fst::write_fst(stress_result, stress_path)
  }
  
  invisible(TRUE)
}

# ------------------------------------------------------------------------------
# Boruta summary report
# Builds the same kind of end-of-run summary you had before, but uses the
# in-memory results from the pipeline rather than scanning the folder again.
# ------------------------------------------------------------------------------
create_boruta_summary_report <- function(
    pair_results,
    training_base_dir,
    output_plot_filename = "boruta_summary_plot.png",
    output_list_filename = "boruta_master_predictor_list.txt",
    top_n_plot = 15,
    top_n_list = 15
) {
  boruta_tables <- lapply(pair_results, function(x) {
    if (is.null(x$boruta)) {
      return(NULL)
    }
    
    confirmed <- x$boruta$confirmed_predictors
    stats_df <- x$boruta$boruta_statistics |>
      dplyr::filter(predictor %in% confirmed) |>
      dplyr::mutate(year_pair = x$year_pair)
    
    stats_df
  })
  
  boruta_tables <- boruta_tables[!vapply(boruta_tables, is.null, logical(1))]
  
  if (length(boruta_tables) == 0) {
    warning("No Boruta results available for summary report.")
    return(invisible(NULL))
  }
  
  combined_df <- dplyr::bind_rows(boruta_tables)
  
  # Master list across pairs
  master_predictors <- combined_df |>
    dplyr::mutate(
      predictor_generic = dplyr::if_else(
        stringr::str_detect(predictor, "_\\d{4}_\\d{4}$"),
        stringr::str_remove(predictor, "_\\d{4}_\\d{4}$"),
        predictor
      )
    ) |>
    dplyr::group_by(predictor_generic) |>
    dplyr::summarise(
      confirmation_count = dplyr::n_distinct(year_pair),
      overall_importance = mean(meanImp),
      .groups = "drop"
    ) |>
    dplyr::arrange(desc(confirmation_count), desc(overall_importance)) |>
    dplyr::slice_head(n = top_n_list)
  
  writeLines(
    master_predictors$predictor_generic,
    file.path(training_base_dir, output_list_filename)
  )
  
  # Plot top predictors by pair
  year_pairs <- sort(unique(combined_df$year_pair))
  
  png(
    filename = file.path(training_base_dir, output_plot_filename),
    width = 1600,
    height = 1200,
    res = 100
  )
  
  n_panels <- length(year_pairs)
  n_rows <- ceiling(sqrt(n_panels))
  n_cols <- ceiling(n_panels / n_rows)
  
  par(
    mfrow = c(n_rows, n_cols),
    mar = c(5, 10, 4, 2),
    oma = c(0, 0, 3, 0)
  )
  
  for (pair in year_pairs) {
    plot_data <- combined_df |>
      dplyr::filter(year_pair == pair) |>
      dplyr::arrange(desc(meanImp)) |>
      dplyr::slice_head(n = top_n_plot)
    
    if (nrow(plot_data) > 0) {
      barplot(
        height = rev(plot_data$meanImp),
        names.arg = rev(plot_data$predictor),
        horiz = TRUE,
        las = 1,
        main = paste("Year Pair:", pair),
        xlab = "Mean importance (Z-score)",
        col = "darkcyan",
        cex.names = 0.8
      )
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "", main = pair)
      text(1, 1, "No confirmed predictors found.", cex = 1.1)
    }
  }
  
  mtext(
    paste("Top", top_n_plot, "confirmed predictor importance by year pair"),
    outer = TRUE,
    cex = 1.5,
    font = 2
  )
  
  dev.off()
  
  invisible(master_predictors)
}

# ------------------------------------------------------------------------------
# Stress test summary export
# Saves one combined table for all year-pairs and one simple diagnostic plot.
# ------------------------------------------------------------------------------
create_stress_test_summary <- function(
    pair_results,
    training_base_dir,
    output_table_filename = "stress_test_summary.csv",
    output_plot_filename = "stress_test_rmse_plot.png"
) {
  stress_tables <- lapply(pair_results, function(x) x$stress_test)
  stress_tables <- stress_tables[!vapply(stress_tables, is.null, logical(1))]
  
  if (length(stress_tables) == 0) {
    warning("No stress test results available for summary.")
    return(invisible(NULL))
  }
  
  combined_stress <- data.table::rbindlist(stress_tables, use.names = TRUE)
  
  fwrite(
    combined_stress,
    file = file.path(training_base_dir, output_table_filename)
  )
  
  year_pairs <- unique(combined_stress$year_pair)
  
  png(
    filename = file.path(training_base_dir, output_plot_filename),
    width = 1600,
    height = 1200,
    res = 100
  )
  
  n_panels <- length(year_pairs)
  n_rows <- ceiling(sqrt(n_panels))
  n_cols <- ceiling(n_panels / n_rows)
  
  par(
    mfrow = c(n_rows, n_cols),
    mar = c(5, 10, 4, 2),
    oma = c(0, 0, 3, 0)
  )
  
  for (pair in year_pairs) {
    plot_data <- combined_stress[year_pair == pair][order(rmse)]
    
    if (nrow(plot_data) > 0) {
      barplot(
        height = plot_data$rmse,
        names.arg = plot_data$model_name,
        horiz = TRUE,
        las = 1,
        main = paste("RMSE by model:", pair),
        xlab = "RMSE",
        col = "steelblue"
      )
    } else {
      plot(1, type = "n", axes = FALSE, xlab = "", ylab = "", main = pair)
      text(1, 1, "No stress-test results found.", cex = 1.1)
    }
  }
  
  mtext(
    "Stress test RMSE by model configuration",
    outer = TRUE,
    cex = 1.5,
    font = 2
  )
  
  dev.off()
  
  invisible(combined_stress)
}

# ------------------------------------------------------------------------------
# Full analysis for one year-pair
# This is the key refactor: one function owns the full workflow for a single
# pair from data sampling through Boruta, stress testing, and saving outputs.
# ------------------------------------------------------------------------------
run_pipeline_for_pair <- function(
    training_base_dir,
    year_pair,
    sample_size = 100000,
    chunk_size = 5,
    max_runs = 100,
    k_folds = 5,
    compute_moran = TRUE,
    seed = default_seed,
    save_outputs = TRUE
) {
  log_message("Starting year-pair: ", year_pair)
  
  dt <- sample_training_data(
    training_base_dir = training_base_dir,
    year_pair = year_pair,
    sample_size = sample_size,
    chunk_size = chunk_size,
    seed = seed,
    transform_flowdir = TRUE
  )
  
  log_message(
    "Sample created for ", year_pair,
    " with ", format(nrow(dt), big.mark = ","), " rows and ",
    ncol(dt), " columns."
  )
  
  boruta_result <- run_boruta_for_pair(
    dt = dt,
    year_pair = year_pair,
    response = response_var,
    max_runs = max_runs
  )
  
  log_message(
    "Boruta complete for ", year_pair,
    ". Confirmed predictors: ",
    length(boruta_result$confirmed_predictors), "."
  )
  
  stress_result <- run_stress_test_for_pair(
    dt = dt,
    year_pair = year_pair,
    confirmed_predictors = boruta_result$confirmed_predictors,
    k_folds = k_folds,
    compute_moran = compute_moran,
    response = response_var
  )
  
  if (save_outputs) {
    save_pair_outputs(
      training_base_dir = training_base_dir,
      year_pair = year_pair,
      boruta_result = boruta_result,
      stress_result = stress_result
    )
  }
  
  log_message("Finished year-pair: ", year_pair)
  
  list(
    year_pair = year_pair,
    boruta = boruta_result,
    stress_test = stress_result
  )
}

# ------------------------------------------------------------------------------
# Top-level pipeline runner
# Runs all year-pairs, optionally in parallel, then produces final summaries.
# ------------------------------------------------------------------------------
run_feature_selection_pipeline <- function(
    training_base_dir,
    year_pairs,
    sample_size = 100000,
    chunk_size = 5,
    max_runs = 100,
    k_folds = 5,
    max_cores = parallel::detectCores() - 1,
    compute_moran = TRUE,
    save_outputs = TRUE
) {
  log_message("Starting full feature selection pipeline.")
  
  num_cores <- min(length(year_pairs), max_cores)
  
  if (num_cores > 1) {
    log_message("Running in parallel with ", num_cores, " workers.")
    
    cl <- parallel::makeCluster(num_cores)
    doParallel::registerDoParallel(cl)
    
    on.exit(
      {
        parallel::stopCluster(cl)
      },
      add = TRUE
    )
    
    clusterExport(
      cl,
      varlist = c(
        "response_var",
        "id_vars",
        "always_exclude",
        "default_seed",
        "log_message",
        "transform_flowdir_cols",
        "enforce_predictor_consistency",
        "find_training_files",
        "sample_training_data",
        "prepare_boruta_data",
        "run_boruta_for_pair",
        "categorise_predictors",
        "build_model_configs",
        "run_spatial_cv",
        "run_stress_test_for_pair",
        "save_pair_outputs",
        "run_pipeline_for_pair"
      ),
      envir = environment()
    )
    
    pair_results <- foreach::foreach(
      pair = year_pairs,
      .packages = c(
        "Boruta",
        "data.table",
        "dplyr",
        "fst",
        "ranger",
        "spdep",
        "stringr",
        "tibble"
      ),
      .errorhandling = "pass"
    ) %dopar% {
      tryCatch(
        run_pipeline_for_pair(
          training_base_dir = training_base_dir,
          year_pair = pair,
          sample_size = sample_size,
          chunk_size = chunk_size,
          max_runs = max_runs,
          k_folds = k_folds,
          compute_moran = compute_moran,
          seed = default_seed,
          save_outputs = save_outputs
        ),
        error = function(e) {
          list(
            year_pair = pair,
            boruta = NULL,
            stress_test = NULL,
            error = e$message
          )
        }
      )
    }
  } else {
    log_message("Running sequentially on one core.")
    
    pair_results <- lapply(year_pairs, function(pair) {
      tryCatch(
        run_pipeline_for_pair(
          training_base_dir = training_base_dir,
          year_pair = pair,
          sample_size = sample_size,
          chunk_size = chunk_size,
          max_runs = max_runs,
          k_folds = k_folds,
          compute_moran = compute_moran,
          seed = default_seed,
          save_outputs = save_outputs
        ),
        error = function(e) {
          list(
            year_pair = pair,
            boruta = NULL,
            stress_test = NULL,
            error = e$message
          )
        }
      )
    })
  }
  
  names(pair_results) <- year_pairs
  
  # Final reporting
  create_boruta_summary_report(
    pair_results = pair_results,
    training_base_dir = training_base_dir
  )
  
  stress_summary <- create_stress_test_summary(
    pair_results = pair_results,
    training_base_dir = training_base_dir
  )
  
  log_message("Pipeline complete.")
  
  list(
    pair_results = pair_results,
    stress_summary = stress_summary
  )
}

# ==============================================================================
# FUNCTION CALL 
# Edit these values for your project, then run the script.
# ==============================================================================

training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles"

year_intervals <- c(
  "2004_2006",
  "2006_2010",
  "2010_2015",
  "2015_2022"
)

start_time <- Sys.time()
print(start_time)

pipeline_results <- run_feature_selection_pipeline(
  training_base_dir = training_dir,
  year_pairs = year_intervals,
  sample_size = 100000,
  chunk_size = 5,
  max_runs = 100,
  k_folds = 5,
  max_cores = max(1, parallel::detectCores() - 1),
  # max_cores = 7,
  compute_moran = TRUE,
  save_outputs = TRUE
)

end_time <- Sys.time()
print(end_time)

# ------------------------------------------------------------------------------
# Optional quick look at the combined stress-test summary in memory
# ------------------------------------------------------------------------------
if (!is.null(pipeline_results$stress_summary)) {
  dplyr::glimpse(pipeline_results$stress_summary)
}




# Function Call for single tile ----

# Run on a Single Tile Folder 
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Training_data_grid_tiles/BH4S656W_2"

year_intervals <- c(
  "2004_2006",
  "2006_2010",
  "2010_2015",
  "2015_2022"
)

pipeline_results <- run_feature_selection_pipeline(
  training_base_dir = training_dir,
  year_pairs = year_intervals,
  sample_size = 100000,
  chunk_size = 5,
  max_runs = 100,
  k_folds = 5,
  max_cores = 4,
  compute_moran = TRUE,
  save_outputs = TRUE
)











