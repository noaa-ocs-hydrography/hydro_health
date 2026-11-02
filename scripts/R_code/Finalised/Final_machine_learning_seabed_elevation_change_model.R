# ==============================================================================
# FINAL MASTER SEABED ELEVATION-CHANGE MODELLING WORKFLOW - ACCURACY PARITY REVISION
# Local tile modelling first; optional full-extent local deployment and global modelling second
# Accuracy-parity revision: OG-style pre-XGBoost structure + robust 1000-round floor
# ==============================================================================
#
# PRIMARY MODEL FORMULATION
#   bathy_t + environmental drivers -> predicted bathy_t1
#
# DERIVED PRODUCTS
#   predicted_delta_bathy = predicted_bathy_t1 - bathy_t
#   predicted_delta_rate  = predicted_delta_bathy / interval_years
#
# ARCHITECTURE
#   PART A: Shared configuration and helper functions
#   PART B: Primary LOCAL tile training + immediate prediction
#   PART C: Optional full-extent LOCAL deployment + VRT creation
#   PART D: Optional GLOBAL training-shard preparation
#   PART E: Secondary GLOBAL model training by year pair
#   PART F: Secondary GLOBAL prediction across all selected prediction tiles
#   PART G: Example function calls
#
# IMPORTANT DESIGN DECISIONS
#   1. Model A (bathy_t1 response) remains the production model.
#   2. Predictor NA values are retained. XGBoost handles missing predictor values.
#      Rows are only excluded when essential response, coordinates, interval, or
#      starting bathymetry fields are not usable for the requested operation.
#   3. Extreme erosion/deposition weighting affects MODEL FITTING. Prediction rows
#      are not assigned weights because prediction has no observed response. The
#      influence of weighting is carried into prediction through each fitted model.
#   4. Local and global workflows are separate and can be tested independently.
#   5. Global modelling uses standardized per-tile FST shards and balanced sampling
#      rather than constructing one unlimited in-memory data frame.
#
# ACCURACY-PARITY REVISION
#   * Preserves a separate full_tile_data and model-fitting subgrid_data, as in
#     the original master.
#   * Defaults to native XGBoost missing-value handling because controlled tests
#     showed it outperformed legacy complete-case fitting.
#   * Defaults to OG training-first predictor resolution; deployment schemas are
#     validated rather than silently shrinking the feature set.
#   * Spatial CV now uses current XGBoost API, recovers missing best_iteration
#     from evaluation_log, and cannot silently fall back below 1000 rounds when
#     round_selection_mode = "cv_with_floor".
#   * The original power weighting remains available and is the default method.
# ==============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(fst)
  library(sf)
  library(xgboost)
  library(raster)
  library(ggplot2)
  library(gridExtra)
  library(cmocean)
  library(blockCV)
  library(foreach)
  library(doParallel)
  library(stringr)
})

# ==============================================================================
# PART A. SHARED CONFIGURATION AND HELPER FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# A1. Logging helpers
# ------------------------------------------------------------------------------
log_line <- function(log_file, ..., .sep = "") {
  dir.create(dirname(log_file), recursive = TRUE, showWarnings = FALSE)
  msg <- paste0(
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    " | ",
    paste(..., collapse = .sep)
  )
  cat(msg, "\n", file = log_file, append = TRUE)
}

log_section <- function(log_file, title) {
  log_line(log_file, "------------------------------------------------------------")
  log_line(log_file, title)
  log_line(log_file, "------------------------------------------------------------")
}


# ------------------------------------------------------------------------------
# A?. Required LOCAL summary source-data fields
# ------------------------------------------------------------------------------
#
# These fields describe the spatial identity and invariant environmental
# information of each source row. They must survive from the original
# year-pair FST into every LOCAL summary regardless of whether they are
# currently used as XGBoost predictors.
#
# Keeping this requirement separate from the model predictor list prevents
# older/stale summary files from being treated as current simply because they
# satisfy an older model schema.
# ------------------------------------------------------------------------------

required_local_summary_fields <- c(
  "X",
  "Y",
  "tile_id",
  "distance_from_shore"
)
# ------------------------------------------------------------------------------
# A2. Deterministic seed helper
# ------------------------------------------------------------------------------
make_seed <- function(global_seed, ...) {
  key <- paste(..., collapse = "|")
  as.integer(global_seed + sum(utf8ToInt(key)) %% 1000000L)
}

# ------------------------------------------------------------------------------
# A3. Select one tile, several tiles, or all tiles
# ------------------------------------------------------------------------------
select_grid_tiles <- function(
    grid_sf,
    tile_ids = "all",
    allowed_sources = NULL,
    tile_field = "tile_id",
    source_field = "source"
) {
  stopifnot(inherits(grid_sf, "sf"))

  out <- grid_sf

  if (!is.null(allowed_sources) && source_field %in% names(out)) {
    out <- out[out[[source_field]] %in% allowed_sources, , drop = FALSE]
  }

  if (!(length(tile_ids) == 1L && identical(tile_ids, "all"))) {
    out <- out[out[[tile_field]] %in% tile_ids, , drop = FALSE]
  }

  if (nrow(out) == 0L) {
    stop("Tile selection returned zero grid tiles.")
  }

  out
}

# ------------------------------------------------------------------------------
# A4. File discovery helpers
# ------------------------------------------------------------------------------
find_tile_file <- function(tile_dir, pair, type = c("train", "pred")) {
  type <- match.arg(type)
  if (!dir.exists(tile_dir)) return(NA_character_)

  patterns <- if (type == "train") {
    c(
      paste0("_", pair, "_long\\.fst$"),
      paste0("_", pair, "_train_long\\.fst$")
    )
  } else {
    c(
      paste0("_", pair, "_prediction_long\\.fst$"),
      paste0("_", pair, "_pred_long\\.fst$")
    )
  }

  hits <- unique(unlist(lapply(patterns, function(pat) {
    list.files(tile_dir, pattern = pat, full.names = TRUE)
  })))

  if (length(hits) == 0L) return(NA_character_)

  bn <- basename(hits)
  hits[order(nchar(bn), bn)][1]
}

# ------------------------------------------------------------------------------
# A5. Numeric coercion without deleting NA values
# ------------------------------------------------------------------------------
safe_numeric_cols <- function(dt, cols) {
  data.table::setDT(dt)
  cols <- intersect(cols, names(dt))
  if (length(cols) == 0L) return(dt)

  dt[, (cols) := lapply(
    .SD,
    function(x) suppressWarnings(as.numeric(as.character(x)))
  ), .SDcols = cols]

  dt
}

# ------------------------------------------------------------------------------
# A6. Construct and verify training targets
# ------------------------------------------------------------------------------
ensure_training_targets <- function(dt, tolerance = 1e-8) {
  data.table::setDT(dt)

  required <- c("bathy_t", "bathy_t1")
  missing_required <- setdiff(required, names(dt))
  if (length(missing_required) > 0L) {
    stop("Missing required training fields: ", paste(missing_required, collapse = ", "))
  }

  dt <- safe_numeric_cols(dt, intersect(
    c("year_t", "year_t1", "bathy_t", "bathy_t1", "delta_bathy", "delta_rate"),
    names(dt)
  ))

  # Determine interval years from year_t/year_t1 where available, otherwise from
  # the year-pair label supplied by the caller later.
  if (all(c("year_t", "year_t1") %in% names(dt))) {
    dt[, interval_years := year_t1 - year_t]
  } else if (!("interval_years" %in% names(dt))) {
    dt[, interval_years := NA_real_]
  }

  recalculated_delta <- dt$bathy_t1 - dt$bathy_t

  if ("delta_bathy" %in% names(dt)) {
    mismatch <- is.finite(dt$delta_bathy) & is.finite(recalculated_delta) &
      abs(dt$delta_bathy - recalculated_delta) > tolerance
    if (any(mismatch)) {
      warning(
        sum(mismatch),
        " delta_bathy values differed from bathy_t1 - bathy_t and were replaced."
      )
    }
  }

  dt[, delta_bathy := bathy_t1 - bathy_t]
  dt[, delta_rate := fifelse(
    is.finite(interval_years) & interval_years > 0,
    delta_bathy / interval_years,
    NA_real_
  )]

  dt
}

pair_duration_years <- function(pair) {
  yy <- suppressWarnings(as.numeric(strsplit(pair, "_", fixed = TRUE)[[1]]))
  if (length(yy) != 2L || any(!is.finite(yy)) || yy[2] <= yy[1]) {
    stop("Unable to derive a valid interval duration from year pair: ", pair)
  }
  yy[2] - yy[1]
}

# ------------------------------------------------------------------------------
# A7. Translate Model A predictions into common downstream products
# ------------------------------------------------------------------------------
translate_model_a_predictions <- function(
    pred_bathy_t1,
    bathy_t,
    interval_years,
    standard_horizon_years = 2
) {
  pred_delta_bathy <- pred_bathy_t1 - bathy_t
  pred_delta_rate <- pred_delta_bathy / interval_years
  pred_delta_standard <- pred_delta_rate * standard_horizon_years
  pred_bathy_standard <- bathy_t + pred_delta_standard

  list(
    pred_bathy_t1 = pred_bathy_t1,
    pred_delta_bathy = pred_delta_bathy,
    pred_delta_rate = pred_delta_rate,
    pred_delta_standard = pred_delta_standard,
    pred_bathy_standard = pred_bathy_standard
  )
}

# ------------------------------------------------------------------------------
# A8. Dynamic predictor translation and deployment-safe predictor resolution
# ------------------------------------------------------------------------------
# These are base names in the master predictor list. They are translated to
# interval-specific columns such as hurr_count_2004_2006.
pair_dynamic_bases <- c(
  "hurr_count",
  "hurr_strength",
  "tsm_mean"
)

build_pair_specific_predictors <- function(master_predictors, pair) {
  vapply(master_predictors, function(p) {
    if (p %in% pair_dynamic_bases) paste0(p, "_", pair) else p
  }, character(1))
}

resolve_pair_predictors <- function(
    pair,
    master_predictors,
    train_columns,
    prediction_columns = NULL,
    force_include = "bathy_t",
    exclude = c(
      "X", "Y", "FID", "tile_id", "source", "dataset_role",
      "year_t", "year_t1", "interval_years", "pair_id",
      "bathy_t1", "delta_bathy", "delta_rate", "survey_end_date"
    ),
    require_prediction_compatibility = TRUE
) {
  candidates <- unique(c(
    force_include,
    build_pair_specific_predictors(master_predictors, pair)
  ))

  candidates <- setdiff(candidates, exclude)
  in_training <- intersect(candidates, train_columns)

  if (require_prediction_compatibility && !is.null(prediction_columns)) {
    resolved <- intersect(in_training, prediction_columns)
  } else {
    resolved <- in_training
  }

  if (!("bathy_t" %in% resolved)) {
    stop("bathy_t must exist in both the training and deployment predictor schemas.")
  }

  list(
    all_predictors = resolved,
    unavailable_in_training = setdiff(candidates, train_columns),
    unavailable_in_prediction = if (is.null(prediction_columns)) {
      character()
    } else {
      setdiff(in_training, prediction_columns)
    }
  )
}

# ------------------------------------------------------------------------------
# A9. Predictor NA diagnostics
# ------------------------------------------------------------------------------
summarize_predictor_missingness <- function(dt, predictors) {
  data.table(
    predictor = predictors,
    n_rows = nrow(dt),
    n_na = vapply(dt[, ..predictors], function(x) sum(is.na(x)), numeric(1)),
    proportion_na = vapply(dt[, ..predictors], function(x) mean(is.na(x)), numeric(1)),
    n_finite = vapply(dt[, ..predictors], function(x) sum(is.finite(x)), numeric(1))
  )
}

# ------------------------------------------------------------------------------
# A10. Sample weighting for extreme erosion and deposition
# ------------------------------------------------------------------------------
# Weighting is applied when fitting the model. The fitted relationship therefore
# gives more influence to stronger observed changes, and that learned influence is
# carried into every prediction produced by the model.
calculate_sample_weights <- function(
    change_vec,
    alpha = 1,
    method = c("capped_linear", "power"),
    cap_quantile = 0.95,
    max_weight = 2,
    epsilon = 1e-6
) {
  method <- match.arg(method)
  x <- abs(change_vec)

  if (method == "power") {
    w <- (x + epsilon)^alpha
    finite_sum <- sum(w[is.finite(w)], na.rm = TRUE)
    if (!is.finite(finite_sum) || finite_sum <= 0) return(rep(1, length(x)))
    w <- w / finite_sum * length(w)
  } else {
    cap <- stats::quantile(x, cap_quantile, na.rm = TRUE, names = FALSE)
    if (!is.finite(cap) || cap <= 0) return(rep(1, length(x)))

    scaled <- pmin(x, cap) / cap
    w <- 1 + (max_weight - 1) * scaled^alpha
  }

  w[!is.finite(w)] <- 1
  w
}

# ------------------------------------------------------------------------------
# A11. Common model-performance metrics
# ------------------------------------------------------------------------------
safe_metric <- function(x) if (length(x) == 0L || all(!is.finite(x))) NA_real_ else x

calculate_common_metrics <- function(
    obs_bathy_t1,
    pred_bathy_t1,
    bathy_t,
    interval_years
) {
  obs_delta <- obs_bathy_t1 - bathy_t
  pred_delta <- pred_bathy_t1 - bathy_t
  obs_rate <- obs_delta / interval_years
  pred_rate <- pred_delta / interval_years

  finite_state <- is.finite(obs_bathy_t1) & is.finite(pred_bathy_t1)
  finite_delta <- is.finite(obs_delta) & is.finite(pred_delta)
  finite_rate <- is.finite(obs_rate) & is.finite(pred_rate)

  predictive_r2 <- function(obs, pred, keep) {
    if (sum(keep) < 2L) return(NA_real_)
    sse <- sum((obs[keep] - pred[keep])^2)
    sst <- sum((obs[keep] - mean(obs[keep]))^2)
    if (!is.finite(sst) || sst <= 0) return(NA_real_)
    1 - sse / sst
  }

  hotspot_cut <- if (sum(finite_delta) > 10L) {
    quantile(abs(obs_delta[finite_delta]), 0.90, na.rm = TRUE, names = FALSE)
  } else {
    NA_real_
  }
  hotspot <- finite_delta & is.finite(hotspot_cut) & abs(obs_delta) >= hotspot_cut

  data.table(
    n_state = sum(finite_state),
    n_delta = sum(finite_delta),
    rmse_bathy_t1 = if (sum(finite_state) > 0) sqrt(mean((pred_bathy_t1[finite_state] - obs_bathy_t1[finite_state])^2)) else NA_real_,
    mae_bathy_t1 = if (sum(finite_state) > 0) mean(abs(pred_bathy_t1[finite_state] - obs_bathy_t1[finite_state])) else NA_real_,
    bias_bathy_t1 = if (sum(finite_state) > 0) mean(pred_bathy_t1[finite_state] - obs_bathy_t1[finite_state]) else NA_real_,
    r2_predictive_bathy_t1 = predictive_r2(obs_bathy_t1, pred_bathy_t1, finite_state),
    r2_correlation_bathy_t1 = if (sum(finite_state) > 2) cor(obs_bathy_t1[finite_state], pred_bathy_t1[finite_state])^2 else NA_real_,
    rmse_delta = if (sum(finite_delta) > 0) sqrt(mean((pred_delta[finite_delta] - obs_delta[finite_delta])^2)) else NA_real_,
    mae_delta = if (sum(finite_delta) > 0) mean(abs(pred_delta[finite_delta] - obs_delta[finite_delta])) else NA_real_,
    bias_delta = if (sum(finite_delta) > 0) mean(pred_delta[finite_delta] - obs_delta[finite_delta]) else NA_real_,
    delta_correlation = if (sum(finite_delta) > 2) cor(obs_delta[finite_delta], pred_delta[finite_delta]) else NA_real_,
    rmse_rate = if (sum(finite_rate) > 0) sqrt(mean((pred_rate[finite_rate] - obs_rate[finite_rate])^2)) else NA_real_,
    mae_rate = if (sum(finite_rate) > 0) mean(abs(pred_rate[finite_rate] - obs_rate[finite_rate])) else NA_real_,
    bias_rate = if (sum(finite_rate) > 0) mean(pred_rate[finite_rate] - obs_rate[finite_rate]) else NA_real_,
    sign_accuracy_delta = if (sum(finite_delta) > 0) mean(sign(pred_delta[finite_delta]) == sign(obs_delta[finite_delta])) else NA_real_,
    hotspot_rmse_delta = if (sum(hotspot) > 10L) sqrt(mean((pred_delta[hotspot] - obs_delta[hotspot])^2)) else NA_real_,
    predicted_erosion_proportion = if (sum(finite_delta) > 0) mean(pred_delta[finite_delta] < 0) else NA_real_,
    predicted_deposition_proportion = if (sum(finite_delta) > 0) mean(pred_delta[finite_delta] > 0) else NA_real_
  )
}

# ------------------------------------------------------------------------------
# A12. Spatial CV diagnostic plot
# ------------------------------------------------------------------------------
generate_cv_diagnostic_plot <- function(
    sf_data,
    block_geom,
    max_k,
    tile_id,
    pair,
    output_dir
) {
  diag_plot <- ggplot() +
    geom_sf(data = sf_data, color = "grey50", size = 0.1, alpha = 0.5) +
    geom_sf(data = block_geom, color = "blue", fill = "transparent", linewidth = 0.5) +
    labs(
      title = paste("Spatial CV | Tile:", tile_id, "| Pair:", pair),
      subtitle = paste("Occupied spatial blocks:", max_k),
      x = "X", y = "Y"
    ) +
    theme_minimal()

  plot_dir <- file.path(output_dir, tile_id, "diagnostic_plots")
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

  ggsave(
    file.path(plot_dir, paste0("spatial_cv_map_", pair, ".png")),
    plot = diag_plot,
    width = 8,
    height = 7,
    dpi = 150
  )
}

# ------------------------------------------------------------------------------
# A13. Template-aligned, grid-clipped raster writing
# ------------------------------------------------------------------------------
# IMPORTANT:
# The prediction FST can contain points extending slightly beyond the nominal
# model-grid polygon. If rasters are written from the point extent alone,
# neighboring tile GeoTIFFs overlap. A VRT does not average those overlaps; one
# source receives drawing priority and can conceal the correct neighboring tile.
#
# The master grid polygon is therefore the authoritative spatial footprint:
#
#   1. Crop the common template to the tile polygon extent.
#   2. Rasterize all finite prediction values on that aligned template.
#   3. Mask cells outside the exact tile polygon.
#   4. Write an explicit numeric NoData value.
#
# This produces non-overlapping, template-aligned tile rasters suitable for a
# seamless VRT mosaic.
write_prediction_tif <- function(
    pred_dt,
    value_col,
    out_tif,
    template_raster,
    tile_geometry = NULL,
    aggregate_fun = mean,
    trim_output = FALSE,
    nodata_value = -9999
) {
  stopifnot(
    all(
      c(
        "X",
        "Y",
        value_col
      ) %in% names(pred_dt)
    )
  )

  if (!inherits(template_raster, "Raster")) {
    stop(
      "template_raster must be a Raster* object from the raster package."
    )
  }

  dt <- data.table::copy(
    pred_dt[
      ,
      .(
        X,
        Y,
        value = get(value_col)
      )
    ]
  )

  dt <- dt[
    is.finite(X) &
      is.finite(Y) &
      is.finite(value)
  ]

  if (nrow(dt) == 0L) {
    return(NA_character_)
  }

  sp::coordinates(dt) <- ~ X + Y
  sp::proj4string(dt) <- raster::crs(
    template_raster
  )

  tile_sp <- NULL

  if (!is.null(tile_geometry)) {
    if (!inherits(tile_geometry, "sf")) {
      stop(
        "tile_geometry must be an sf object when supplied."
      )
    }

    if (nrow(tile_geometry) != 1L) {
      stop(
        "tile_geometry must contain exactly one grid feature."
      )
    }

    tile_geometry <- sf::st_transform(
      tile_geometry,
      crs = raster::projection(
        template_raster
      )
    )

    tile_sp <- methods::as(
      tile_geometry,
      "Spatial"
    )

    # Use the official model-grid polygon extent, not the prediction-point
    # extent. snap='out' keeps the crop aligned to the master template.
    local_template <- raster::crop(
      template_raster,
      raster::extent(tile_sp),
      snap = "out"
    )
  } else {
    # Backward-compatible fallback for non-tiled uses.
    prediction_extent <- raster::extent(
      dt
    )

    local_template <- raster::crop(
      template_raster,
      prediction_extent,
      snap = "out"
    )
  }

  r_out <- raster::rasterize(
    x = dt,
    y = local_template,
    field = "value",
    fun = aggregate_fun,
    background = NA_real_
  )

  if (!is.null(tile_sp)) {
    # The polygon mask is the critical seam-safe step. Cells outside the master
    # grid tile become NoData even when prediction points extend into them.
    r_out <- raster::mask(
      r_out,
      tile_sp,
      updatevalue = NA
    )
  }

  if (isTRUE(trim_output)) {
    r_out <- raster::trim(
      r_out
    )
  }

  dir.create(
    dirname(out_tif),
    recursive = TRUE,
    showWarnings = FALSE
  )

  raster::writeRaster(
    r_out,
    filename = out_tif,
    format = "GTiff",
    datatype = "FLT4S",
    NAflag = nodata_value,
    options = c(
      "COMPRESS=DEFLATE",
      "PREDICTOR=3",
      "TILED=YES",
      "BIGTIFF=IF_SAFER"
    ),
    overwrite = TRUE
  )

  out_tif
}


# ------------------------------------------------------------------------------
# A14. Predictor extrapolation diagnostics
# ------------------------------------------------------------------------------
make_predictor_range_table <- function(dt, predictors) {
  rbindlist(lapply(predictors, function(p) {
    x <- dt[[p]]
    data.table(
      predictor = p,
      min_training = if (any(is.finite(x))) min(x, na.rm = TRUE) else NA_real_,
      max_training = if (any(is.finite(x))) max(x, na.rm = TRUE) else NA_real_,
      mean_training = if (any(is.finite(x))) mean(x, na.rm = TRUE) else NA_real_
    )
  }))
}

add_extrapolation_flags <- function(pred_dt, predictor_ranges) {
  out <- copy(pred_dt)
  flags <- matrix(FALSE, nrow = nrow(out), ncol = nrow(predictor_ranges))

  for (j in seq_len(nrow(predictor_ranges))) {
    p <- predictor_ranges$predictor[j]
    lo <- predictor_ranges$min_training[j]
    hi <- predictor_ranges$max_training[j]
    if (p %in% names(out) && is.finite(lo) && is.finite(hi)) {
      flags[, j] <- is.finite(out[[p]]) & (out[[p]] < lo | out[[p]] > hi)
    }
  }

  out[, predictor_extrapolation_count := rowSums(flags, na.rm = TRUE)]
  out[, predictor_extrapolation_any := predictor_extrapolation_count > 0]
  out
}


# ------------------------------------------------------------------------------
# A15. Local raster outputs and diagnostic plots
# ------------------------------------------------------------------------------
# These helpers restore the visual products from the earlier master workflow,
# while using the finalized LOCAL_* object names and template-aligned raster
# writer. Predictor NA values remain untouched; only non-finite X/Y/value cells
# are omitted from each individual raster or plot layer.
save_local_visual_outputs <- function(
    training_summary,
    prediction_summary = NULL,
    shap_summary_dt = NULL,
    importance_dt = NULL,
    training_metrics = NULL,
    tile_id,
    pair,
    tile_dir_train,
    tile_dir_pred,
    template_raster,
    tile_geometry,
    worker_log_file,
    standard_horizon_years = 2
) {
  plot_dir <- file.path(tile_dir_train, "diagnostic_plots")
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

  safe_write_visual_raster <- function(dt, value_col, out_file) {
    if (is.null(dt) || !value_col %in% names(dt)) {
      log_line(worker_log_file, "VISUAL OUTPUT SKIPPED: missing column ", value_col)
      return(NA_character_)
    }

    out <- tryCatch(
      write_prediction_tif(
        pred_dt = dt,
        value_col = value_col,
        out_tif = out_file,
        template_raster = template_raster,
        tile_geometry = tile_geometry,
        trim_output = FALSE,
        nodata_value = -9999
      ),
      error = function(e) {
        log_line(
          worker_log_file,
          "VISUAL RASTER FAILED: ", basename(out_file), " | ",
          conditionMessage(e)
        )
        NA_character_
      }
    )

    if (is.character(out) && length(out) == 1L && !is.na(out)) {
      log_line(worker_log_file, "VISUAL RASTER WRITTEN: ", out)
    } else {
      log_line(worker_log_file, "VISUAL RASTER EMPTY/SKIPPED: ", basename(out_file))
    }
    out
  }

  # -------------------------------------------------------
  # 9A. WRITE TEMPLATE-ALIGNED TRAINING RASTER OUTPUTS
  # -------------------------------------------------------
  training_raster_products <- c(
    bathy_t = "LOCAL_Train_Start_Bathy_t",
    mean_predicted_bathy_t1 = "LOCAL_Train_Mean_Predicted_Bathy_t1",
    delta_bathy = "LOCAL_Train_Surveyed_Change",
    mean_predicted_change = "LOCAL_Train_Mean_Predicted_Change",
    uncertainty_sd_bathy_t1 = "LOCAL_Train_SD_Bathy_t1"
  )

  for (value_col in names(training_raster_products)) {
    safe_write_visual_raster(
      training_summary,
      value_col,
      file.path(
        tile_dir_train,
        paste0(training_raster_products[[value_col]], "_", pair, ".tif")
      )
    )
  }

  # -------------------------------------------------------
  # 9B. WRITE TEMPLATE-ALIGNED PREDICTION RASTER OUTPUTS
  # -------------------------------------------------------
  if (!is.null(prediction_summary) && nrow(prediction_summary) > 0L) {
    prediction_raster_products <- c(
      bathy_t = "LOCAL_Pred_Start_Bathy_t",
      mean_predicted_bathy_t1 = "LOCAL_Pred_Mean_Bathy_t1",
      mean_predicted_change = "LOCAL_Pred_Mean_Change",
      uncertainty_sd_bathy_t1 = "LOCAL_Pred_SD_Bathy_t1",
      predictor_extrapolation_count = "LOCAL_Predictor_Extrapolation_Count"
    )

    if ("UC_t" %in% names(prediction_summary)) {
      prediction_raster_products <- c(
        prediction_raster_products,
        UC_t = "LOCAL_Pred_BlueTopo_UC_t"
      )
    }

    for (value_col in names(prediction_raster_products)) {
      safe_write_visual_raster(
        prediction_summary,
        value_col,
        file.path(
          tile_dir_pred,
          paste0(prediction_raster_products[[value_col]], "_", pair, ".tif")
        )
      )
    }
  }

  # -------------------------------------------------------
  # 9C. MODEL-FIT DIAGNOSTIC PNG
  # -------------------------------------------------------
  fit_dt <- training_summary[
    is.finite(bathy_t1) & is.finite(mean_predicted_bathy_t1),
    .(Actual = bathy_t1, Predicted = mean_predicted_bathy_t1)
  ]

  if (nrow(fit_dt) > 2L) {
    metric_subtitle <- ""
    if (!is.null(training_metrics) && nrow(training_metrics) > 0L) {
      r2_col <- intersect(
        c("r2_bathy_t1_predictive", "r2_predictive_bathy_t1", "r2_bathy_t1"),
        names(training_metrics)
      )
      rmse_col <- intersect(c("rmse_bathy_t1"), names(training_metrics))

      r2_text <- if (length(r2_col)) {
        paste0("Predictive R² = ", round(training_metrics[[r2_col[1]]][1], 3))
      } else {
        NULL
      }
      rmse_text <- if (length(rmse_col)) {
        paste0("RMSE = ", round(training_metrics[[rmse_col[1]]][1], 3), " m")
      } else {
        NULL
      }
      metric_subtitle <- paste(c(r2_text, rmse_text), collapse = " | ")
    }

    fit_plot <- ggplot2::ggplot(
      fit_dt,
      ggplot2::aes(x = Actual, y = Predicted)
    ) +
      ggplot2::geom_point(alpha = 0.25, color = "darkblue", size = 0.8) +
      ggplot2::geom_abline(
        slope = 1,
        intercept = 0,
        color = "red",
        linetype = "dashed",
        linewidth = 1
      ) +
      ggplot2::coord_equal() +
      ggplot2::labs(
        title = paste("LOCAL Model Fit (t1):", tile_id, "|", pair),
        subtitle = metric_subtitle,
        x = "Surveyed Bathymetry (t1, m)",
        y = "Mean Predicted Bathymetry (t1, m)"
      ) +
      ggplot2::theme_minimal()

    ggplot2::ggsave(
      filename = file.path(plot_dir, paste0("LOCAL_model_fit_t1_", pair, ".png")),
      plot = fit_plot,
      width = 7,
      height = 7,
      dpi = 180
    )
    log_line(worker_log_file, "MODEL-FIT PLOT WRITTEN: ", pair)
  }

  # -------------------------------------------------------
  # 9D. SHAP IMPORTANCE PNG
  # -------------------------------------------------------
  if (!is.null(shap_summary_dt) && nrow(shap_summary_dt) > 0L) {
    shap_plot_dt <- data.table::as.data.table(shap_summary_dt)[
      is.finite(mean_abs_shap),
      .(
        overall_mean_abs_shap = mean(mean_abs_shap, na.rm = TRUE),
        shap_sd = stats::sd(mean_abs_shap, na.rm = TRUE),
        n_bootstrap_models = data.table::uniqueN(bootstrap_iteration)
      ),
      by = predictor
    ][order(-overall_mean_abs_shap)]

    n_show <- min(20L, nrow(shap_plot_dt))
    shap_plot_dt <- shap_plot_dt[seq_len(n_show)]

    shap_plot <- ggplot2::ggplot(
      shap_plot_dt,
      ggplot2::aes(
        x = stats::reorder(predictor, overall_mean_abs_shap),
        y = overall_mean_abs_shap
      )
    ) +
      ggplot2::geom_col(fill = "steelblue") +
      ggplot2::coord_flip() +
      ggplot2::labs(
        title = paste("LOCAL SHAP Importance:", tile_id, "|", pair),
        subtitle = paste(
          "Mean absolute SHAP across",
          max(shap_plot_dt$n_bootstrap_models, na.rm = TRUE),
          "bootstrap models"
        ),
        x = "Predictor",
        y = "Mean absolute SHAP"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(axis.text.y = ggplot2::element_text(size = 8))

    ggplot2::ggsave(
      filename = file.path(plot_dir, paste0("LOCAL_shap_importance_", pair, ".png")),
      plot = shap_plot,
      width = 8,
      height = max(5, 0.30 * n_show + 2),
      dpi = 180,
      limitsize = FALSE
    )
    log_line(worker_log_file, "SHAP PLOT WRITTEN: ", pair)
  } else {
    log_line(worker_log_file, "SHAP PLOT SKIPPED: no usable SHAP summary.")
  }

  # -------------------------------------------------------
  # 9E. COMBINED TRAINING + PREDICTION SUMMARY PDFS
  # -------------------------------------------------------
  if (is.null(prediction_summary) || nrow(prediction_summary) < 10L) {
    log_line(worker_log_file, "SUMMARY PDF SKIPPED: prediction summary unavailable/too small.")
    return(invisible(NULL))
  }

  train_dt <- data.table::copy(training_summary)
  pred_dt <- data.table::copy(prediction_summary)

  train_dt[, water_mask := is.finite(bathy_t) & bathy_t < 0]
  pred_dt[, water_mask := is.finite(bathy_t) & bathy_t < 0]

  # Retain all predictor NA values. The water mask is applied only to these
  # visual frames so that land/non-depth cells do not obscure the maps.
  train_plot <- train_dt[
    water_mask == TRUE,
    .(
      X, Y, bathy_t, bathy_t1, delta_bathy,
      mean_predicted_bathy_t1, mean_predicted_change,
      uncertainty_sd_bathy_t1
    )
  ]

  pred_keep_cols <- intersect(
    c(
      "X", "Y", "bathy_t", "mean_predicted_bathy_t1",
      "mean_predicted_change", "uncertainty_sd_bathy_t1",
      "survey_end_date", "UC_t"
    ),
    names(pred_dt)
  )
  pred_plot <- pred_dt[water_mask == TRUE, ..pred_keep_cols]

  if (nrow(train_plot) < 10L || nrow(pred_plot) < 10L) {
    log_line(
      worker_log_file,
      "SUMMARY PDF SKIPPED: insufficient water rows | train=",
      nrow(train_plot), " | pred=", nrow(pred_plot)
    )
    return(invisible(NULL))
  }

  # Survey-end-date cleanup and categorical legend.
  if ("survey_end_date" %in% names(pred_plot)) {
    pred_plot[
      ,
      survey_end_date_num := suppressWarnings(
        as.numeric(as.character(survey_end_date))
      )
    ]
    pred_plot[
      !is.finite(survey_end_date_num) | survey_end_date_num <= 0,
      survey_end_date_num := NA_real_
    ]
    pred_plot[
      !is.na(survey_end_date_num) &
        survey_end_date_num >= 1000 &
        survey_end_date_num < 1800,
      survey_end_date_num := survey_end_date_num + 800
    ]
    pred_plot[, survey_year := as.integer(floor(survey_end_date_num))]
    pred_plot[
      !is.na(survey_year) & (survey_year < 1800 | survey_year > 2100),
      survey_year := NA_integer_
    ]

    pred_plot[, survey_year_bin := data.table::fcase(
      is.na(survey_year), NA_character_,
      survey_year >= 2025, "2025+",
      survey_year == 2024, "2024",
      survey_year == 2023, "2023",
      survey_year == 2022, "2022",
      survey_year == 2021, "2021",
      survey_year == 2020, "2020",
      survey_year >= 2015 & survey_year <= 2019, "2015-2019",
      survey_year >= 2010 & survey_year <= 2014, "2010-2014",
      survey_year >= 2000 & survey_year <= 2009, "2000-2009",
      survey_year >= 1990 & survey_year <= 1999, "1990-1999",
      default = "<1990"
    )]

    pred_plot[, survey_year_bin := factor(
      survey_year_bin,
      levels = c(
        "2025+", "2024", "2023", "2022", "2021", "2020",
        "2015-2019", "2010-2014", "2000-2009", "1990-1999", "<1990"
      )
    )]
  } else {
    pred_plot[, survey_year_bin := factor(NA_character_)]
  }

  depth_values <- c(
    train_plot$bathy_t,
    train_plot$mean_predicted_bathy_t1,
    pred_plot$bathy_t,
    pred_plot$mean_predicted_bathy_t1
  )
  depth_values <- depth_values[is.finite(depth_values)]
  depth_min <- if (length(depth_values)) min(depth_values) else -10
  depth_max <- min(0, if (length(depth_values)) max(depth_values) else 0)
  if (!is.finite(depth_max) || depth_max <= depth_min) depth_max <- 0

  delta_values <- c(
    train_plot$mean_predicted_change,
    train_plot$delta_bathy,
    pred_plot$mean_predicted_change
  )
  delta_values <- delta_values[is.finite(delta_values)]
  max_abs_delta <- if (length(delta_values)) max(abs(delta_values)) else 1
  if (!is.finite(max_abs_delta) || max_abs_delta <= 0) max_abs_delta <- 1

  uncertainty_values <- c(
    train_plot$uncertainty_sd_bathy_t1,
    if ("UC_t" %in% names(pred_plot)) pred_plot$UC_t else numeric()
  )
  uncertainty_values <- uncertainty_values[is.finite(uncertainty_values)]
  uncertainty_max <- if (length(uncertainty_values)) max(uncertainty_values) else 1
  if (!is.finite(uncertainty_max) || uncertainty_max <= 0) uncertainty_max <- 1

  depth_breaks <- pretty(c(depth_min, depth_max), n = 12)
  delta_breaks <- pretty(c(-max_abs_delta, max_abs_delta), n = 12)
  uncertainty_breaks <- pretty(c(0, uncertainty_max), n = 10)

  theme_map <- ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(
      axis.title = ggplot2::element_blank(),
      axis.text = ggplot2::element_blank(),
      axis.ticks = ggplot2::element_blank(),
      panel.grid = ggplot2::element_blank(),
      panel.border = ggplot2::element_rect(
        color = "black", fill = NA, linewidth = 0.6
      ),
      plot.title = ggplot2::element_text(size = 12, face = "bold"),
      legend.position = "right",
      legend.title = ggplot2::element_text(size = 12),
      legend.text = ggplot2::element_text(size = 10),
      legend.key.height = grid::unit(0.85, "cm"),
      legend.key.width = grid::unit(0.75, "cm"),
      plot.margin = ggplot2::margin(6, 6, 6, 6)
    )

  guide_bar <- ggplot2::guide_colorbar(
    barheight = grid::unit(8.5, "cm"),
    barwidth = grid::unit(0.8, "cm"),
    ticks.colour = "black",
    frame.colour = "black"
  )

  add_dep_ero_labels <- function(p) {
    p +
      ggplot2::annotate(
        "text", x = Inf, y = Inf, label = "Deposition",
        hjust = 1.05, vjust = 1.2, fontface = "bold", size = 3.3
      ) +
      ggplot2::annotate(
        "text", x = Inf, y = -Inf, label = "Erosion",
        hjust = 1.05, vjust = -0.6, fontface = "bold", size = 3.3
      )
  }

  plot_depth <- function(dt, z_col, title_text) {
    ggplot2::ggplot(
      dt,
      ggplot2::aes(x = X, y = Y, fill = .data[[z_col]])
    ) +
      ggplot2::geom_raster() +
      ggplot2::coord_equal() +
      ggplot2::scale_fill_viridis_c(
        name = "Depth (m)",
        limits = c(depth_min, depth_max),
        breaks = depth_breaks,
        na.value = "transparent"
      ) +
      ggplot2::guides(fill = guide_bar) +
      ggplot2::labs(title = title_text) +
      theme_map
  }

  plot_delta <- function(dt, z_col, title_text) {
    p <- ggplot2::ggplot(
      dt,
      ggplot2::aes(x = X, y = Y, fill = .data[[z_col]])
    ) +
      ggplot2::geom_raster() +
      ggplot2::coord_equal() +
      ggplot2::scale_fill_gradientn(
        colours = cmocean::cmocean("curl")(100),
        limits = c(-max_abs_delta, max_abs_delta),
        breaks = delta_breaks,
        name = "Delta (m)",
        na.value = "transparent"
      ) +
      ggplot2::guides(fill = guide_bar) +
      ggplot2::labs(title = title_text) +
      theme_map

    add_dep_ero_labels(p)
  }

  plot_uncertainty <- function(dt, z_col, title_text, legend_title) {
    if (!z_col %in% names(dt) || all(!is.finite(dt[[z_col]]))) {
      return(
        ggplot2::ggplot() +
          ggplot2::theme_void() +
          ggplot2::ggtitle(paste0(title_text, " (No data)"))
      )
    }

    ggplot2::ggplot(
      dt,
      ggplot2::aes(x = X, y = Y, fill = .data[[z_col]])
    ) +
      ggplot2::geom_raster() +
      ggplot2::coord_equal() +
      ggplot2::scale_fill_viridis_c(
        name = legend_title,
        limits = c(0, uncertainty_max),
        breaks = uncertainty_breaks,
        na.value = "transparent"
      ) +
      ggplot2::guides(fill = guide_bar) +
      ggplot2::labs(title = title_text) +
      theme_map
  }

  plot_survey <- function(dt, title_text) {
    if (!"survey_year_bin" %in% names(dt) || all(is.na(dt$survey_year_bin))) {
      return(
        ggplot2::ggplot() +
          ggplot2::theme_void() +
          ggplot2::ggtitle(paste0(title_text, " (No data)"))
      )
    }

    ggplot2::ggplot(
      dt,
      ggplot2::aes(x = X, y = Y, fill = survey_year_bin)
    ) +
      ggplot2::geom_raster() +
      ggplot2::coord_equal() +
      ggplot2::scale_fill_viridis_d(
        name = "Survey\nYear Bin",
        drop = TRUE,
        na.value = "grey90"
      ) +
      ggplot2::labs(title = title_text) +
      theme_map
  }

  p1 <- plot_depth(
    train_plot, "bathy_t",
    "Train Start Bathy (t) - Survey data"
  )
  p2 <- plot_depth(
    train_plot, "mean_predicted_bathy_t1",
    "Train Mean Prediction (t1)"
  )
  p3 <- plot_delta(
    train_plot, "mean_predicted_change",
    "Train Predicted Change [t1 - t]"
  )
  p4 <- plot_delta(
    train_plot, "delta_bathy",
    "Train Surveyed Change [t1 - t]"
  )
  p5 <- plot_depth(
    pred_plot, "bathy_t",
    "Pred Start Bathy (t) - BlueTopo"
  )
  p6 <- plot_depth(
    pred_plot, "mean_predicted_bathy_t1",
    "Pred Mean Prediction (t1)"
  )
  p7 <- plot_delta(
    pred_plot, "mean_predicted_change",
    "Predicted Change [t1 - t]"
  )
  p8 <- plot_survey(
    pred_plot,
    "Prediction Survey End Date"
  )

  p4_uncertainty <- plot_uncertainty(
    train_plot,
    "uncertainty_sd_bathy_t1",
    "Train Uncertainty (SD from bootstraps)",
    "SD (m)"
  )

  p8_uncertainty <- if ("UC_t" %in% names(pred_plot)) {
    plot_uncertainty(
      pred_plot,
      "UC_t",
      "BlueTopo Uncertainty (UC_t)",
      "UC (m)"
    )
  } else {
    plot_uncertainty(
      pred_plot,
      "uncertainty_sd_bathy_t1",
      "Prediction Uncertainty (SD from bootstraps)",
      "SD (m)"
    )
  }

  summary_pdf <- file.path(
    plot_dir,
    paste0("LOCAL_Summary_Training_vs_Prediction_", pair, ".pdf")
  )
  uncertainty_pdf <- file.path(
    plot_dir,
    paste0("LOCAL_Summary_Training_vs_Prediction_Uncertainty_", pair, ".pdf")
  )

  grDevices::pdf(summary_pdf, width = 20, height = 10, onefile = TRUE)
  gridExtra::grid.arrange(
    p1, p2, p3, p4,
    p5, p6, p7, p8,
    ncol = 4
  )
  grDevices::dev.off()

  grDevices::pdf(uncertainty_pdf, width = 20, height = 10, onefile = TRUE)
  gridExtra::grid.arrange(
    p1, p2, p3, p4_uncertainty,
    p5, p6, p7, p8_uncertainty,
    ncol = 4
  )
  grDevices::dev.off()

  log_line(worker_log_file, "SUMMARY PDF WRITTEN: ", summary_pdf)
  log_line(worker_log_file, "UNCERTAINTY PDF WRITTEN: ", uncertainty_pdf)

  invisible(
    list(
      summary_pdf = summary_pdf,
      uncertainty_pdf = uncertainty_pdf
    )
  )
}

# ==============================================================================
# PART B. PRIMARY LOCAL TILE TRAINING + IMMEDIATE PREDICTION
# ==============================================================================
# testing with #run 010 
.Run_Local_Model_Training_And_Prediction_Batch <- function(
    grid_sf,
    output_dir_train,
    output_dir_pred,
    year_pairs,
    master_predictor_list_path,
    template_raster,
    tile_ids = "all",
    block_size_m = 200,
    n_boot = 5, # should this be 250? 
    n_folds = 5,
    use_weighted_loss = TRUE,
    weight_alpha = 0.5,
    weight_method = "power",
    weight_cap_quantile = 0.95,
    max_weight = 2,
    max_depth = 4,
    eta = 0.01,
    gamma = 1.0,
    lambda = 1,
    alpha = 1,

    # Accuracy-parity controls
    pre_xgb_data_mode = c("native_missing", "legacy_complete"),
    predictor_resolution_mode = c("og_training_first", "deployment_intersection"),
    round_selection_mode = c("cv_with_floor", "fixed", "cv_only"),
    minimum_nrounds = 1000,
    fixed_nrounds = 1000,
    cv_nrounds = 2000,
    early_stopping_rounds = 15,
    cv_fallback_nrounds = 1000,

    standard_horizon_years = 2,
    minimum_training_rows = 100,
    require_prediction_compatible_predictors = TRUE,
    write_global_training_shards = TRUE,
    global_shard_dir = file.path(output_dir_train, "_GLOBAL_TRAINING_SHARDS"),

    # Full-extent LOCAL deployment controls
    deploy_local_full_extent = TRUE,
    overwrite_existing_local = FALSE,
    write_full_extent_spatial_outputs = TRUE,

    n_workers = max(1, parallel::detectCores() - 1),
    global_seed = 12345
) {
  # =======================================================
  # LOCAL HYBRID XGBOOST SPATIAL-CV MODEL
  # Final production formulation: predict bathy_t1
  # =======================================================

  if (!file.exists(master_predictor_list_path)) {
    stop("Master predictor list not found: ", master_predictor_list_path)
  }
  if (!inherits(template_raster, "Raster")) {
    stop("template_raster must be loaded with raster::raster().")
  }

  pre_xgb_data_mode <- match.arg(pre_xgb_data_mode)
  predictor_resolution_mode <- match.arg(predictor_resolution_mode)
  round_selection_mode <- match.arg(round_selection_mode)

  if (minimum_nrounds < 1L || fixed_nrounds < 1L || cv_nrounds < 1L) {
    stop("All XGBoost round controls must be positive integers.")
  }

  master_predictors <- unique(trimws(readLines(master_predictor_list_path)))
  master_predictors <- master_predictors[nzchar(master_predictors)]

  selected_grid <- select_grid_tiles(
    grid_sf = grid_sf,
    tile_ids = tile_ids,
    allowed_sources = c("both", "training", "prediction")
  )
  tiles_df <- unique(as.character(selected_grid$tile_id))

  master_log_file <- file.path(output_dir_train, "LOCAL_MODEL_MASTER_LOG.txt")
  log_section(master_log_file, "LOCAL MODEL RUN START")
  log_line(master_log_file, "Tiles selected: ", length(tiles_df))
  log_line(master_log_file, "Tile IDs: ", paste(tiles_df, collapse = ", "))
  log_line(master_log_file, "Year pairs: ", paste(year_pairs, collapse = ", "))
  log_line(master_log_file, "Pre-XGB data mode: ", pre_xgb_data_mode)
  log_line(master_log_file, "Predictor resolution mode: ", predictor_resolution_mode)
  log_line(master_log_file, "Round selection mode: ", round_selection_mode,
           " | minimum rounds: ", minimum_nrounds,
           " | fallback rounds: ", cv_fallback_nrounds)

  # -------------------------------------------------------
  # 1. PARALLEL SETUP & LOGGING
  # -------------------------------------------------------
  cl <- parallel::makeCluster(n_workers)
  doParallel::registerDoParallel(cl)
  on.exit({
    if (!is.null(cl)) {
      try(parallel::stopCluster(cl), silent = TRUE)
    }
    try(closeAllConnections(), silent = TRUE)
  }, add = TRUE)

  # -------------------------------------------------------
  # 2. MAIN PARALLEL TILE PROCESSING LOOP
  # -------------------------------------------------------
  results_list <- foreach(
    tile_id = tiles_df,
    .packages = c(
      "data.table", "dplyr", "fst", "sf", "xgboost", "raster",
      "ggplot2", "gridExtra", "cmocean", "blockCV", "sp", "grid"
    ),
    .export = c(
      "log_line", "log_section", "make_seed", "find_tile_file",
      "safe_numeric_cols", "ensure_training_targets", "pair_duration_years",
      "translate_model_a_predictions", "build_pair_specific_predictors",
      "resolve_pair_predictors", "summarize_predictor_missingness",
      "calculate_sample_weights", "calculate_common_metrics",
      "generate_cv_diagnostic_plot", "write_prediction_tif",
      "make_predictor_range_table", "add_extrapolation_flags",
      "save_local_visual_outputs", "pair_dynamic_bases",
      "required_local_summary_fields"
    ),
    .errorhandling = "pass"
  ) %dopar% {
    tile_dir_train <- file.path(output_dir_train, tile_id)
    tile_dir_pred <- file.path(output_dir_pred, tile_id)
    dir.create(tile_dir_train, recursive = TRUE, showWarnings = FALSE)
    dir.create(tile_dir_pred, recursive = TRUE, showWarnings = FALSE)

    tile_geometry <- selected_grid[
      as.character(selected_grid$tile_id) == tile_id,
      ,
      drop = FALSE
    ]

    if (nrow(tile_geometry) != 1L) {
      stop(
        "Expected exactly one model-grid geometry for tile ",
        tile_id,
        "; found ",
        nrow(tile_geometry),
        "."
      )
    }

    worker_log_file <- file.path(tile_dir_train, paste0("LOCAL_LOG_", tile_id, ".txt"))
    cat("", file = worker_log_file)
    log_section(worker_log_file, paste0("LOCAL TILE START: ", tile_id))

    tile_results <- list()

    for (pair in year_pairs) {
      pair_result <- tryCatch({
        log_section(worker_log_file, paste0("YEAR PAIR: ", pair))
        interval_years <- pair_duration_years(pair)
        pair_seed <- make_seed(global_seed, "local", tile_id, pair)

        # -------------------------------------------------------
        # 2A. LOAD TRAINING AND PREDICTION DATA
        # -------------------------------------------------------
        training_data_path <- find_tile_file(tile_dir_train, pair, type = "train")
        prediction_data_path <- find_tile_file(tile_dir_pred, pair, type = "pred")

        if (is.na(training_data_path)) {
          log_line(worker_log_file, "SKIP: no training file for ", tile_id, " / ", pair)
          return(list(ok = FALSE, reason = "missing_training_file"))
        }

        training_raw <- fst::read_fst(training_data_path, as.data.table = TRUE)
        setDT(training_raw)
        training_raw <- ensure_training_targets(training_raw)
        training_raw[
          !is.finite(interval_years) | interval_years <= 0,
          interval_years := pair_duration_years(pair)
        ]
        training_raw[, pair_id := pair]
        training_raw[, tile_id := tile_id]

        prediction_available <- !is.na(prediction_data_path)
        prediction_raw <- if (prediction_available) {
          fst::read_fst(prediction_data_path, as.data.table = TRUE)
        } else {
          data.table()
        }
        setDT(prediction_raw)
        if (prediction_available) {
          prediction_raw[, interval_years := interval_years]
          prediction_raw[, pair_id := pair]
          prediction_raw[, tile_id := tile_id]
        }

        # -------------------------------------------------------
        # 2B. RESOLVE AND LOCK PREDICTORS
        # -------------------------------------------------------
        # OG parity mode resolves predictors from the master list + TRAINING
        # columns first. It does not silently reduce the model merely because a
        # predictor is absent from prediction. Missing deployment predictors are
        # treated as an explicit schema error so the model definition remains
        # scientifically transparent.
        require_intersection <- identical(
          predictor_resolution_mode,
          "deployment_intersection"
        )

        resolved <- resolve_pair_predictors(
          pair = pair,
          master_predictors = master_predictors,
          train_columns = names(training_raw),
          prediction_columns = if (prediction_available) names(prediction_raw) else NULL,
          require_prediction_compatibility =
            require_intersection && prediction_available
        )
        predictors <- resolved$all_predictors

        if (length(predictors) == 0L) {
          stop("No predictors resolved for tile/pair.")
        }

        if (
          identical(predictor_resolution_mode, "og_training_first") &&
          prediction_available
        ) {
          missing_prediction_predictors <- setdiff(
            predictors,
            names(prediction_raw)
          )
          if (length(missing_prediction_predictors) > 0L) {
            stop(
              "OG training-first predictor schema cannot be deployed because ",
              "prediction data are missing: ",
              paste(missing_prediction_predictors, collapse = ", ")
            )
          }
        }

        numeric_fields <- unique(c(
          predictors, "X", "Y", "FID", "bathy_t", "bathy_t1",
          "delta_bathy", "delta_rate", "interval_years"
        ))
        training_raw <- safe_numeric_cols(training_raw, numeric_fields)
        if (prediction_available) {
          prediction_raw <- safe_numeric_cols(
            prediction_raw,
            unique(c(predictors, "X", "Y", "FID", "bathy_t", "interval_years"))
          )
        }

        log_line(
          worker_log_file,
          "LOCKED predictors (", length(predictors), ") [",
          predictor_resolution_mode, "]: ",
          paste(predictors, collapse = ", ")
        )
        if (length(resolved$unavailable_in_prediction) > 0L) {
          log_line(worker_log_file, "Predictors unavailable in prediction: ",
                   paste(resolved$unavailable_in_prediction, collapse = ", "))
        }

        missingness_train <- summarize_predictor_missingness(training_raw, predictors)
        fwrite(
          missingness_train,
          file.path(tile_dir_train, paste0("predictor_missingness_training_", pair, ".csv"))
        )
        if (prediction_available) {
          fwrite(
            summarize_predictor_missingness(prediction_raw, predictors),
            file.path(tile_dir_pred, paste0("predictor_missingness_prediction_", pair, ".csv"))
          )
        }

        # -------------------------------------------------------
        # 3. FILTER & PREPARE TRAINING WORKFLOW DATA
        # -------------------------------------------------------
        # Preserve the original master architecture:
        #   full_tile_data = full selected/coerced tile used for prediction and
        #                    diagnostic reconstruction;
        #   subgrid_data   = rows eligible to fit XGBoost.
        #
        # native_missing (recommended from lean tests):
        #   retain predictor NA values during fitting; only essential target,
        #   coordinate, bathy and interval fields must be finite.
        #
        # legacy_complete:
        #   reproduce the OG complete-predictor training filter, while still
        #   predicting back over full_tile_data as the OG workflow did.
        full_tile_data <- data.table::copy(training_raw)

        essential_keep <- is.finite(full_tile_data$X) &
          is.finite(full_tile_data$Y) &
          is.finite(full_tile_data$bathy_t) &
          is.finite(full_tile_data$bathy_t1) &
          is.finite(full_tile_data$delta_bathy) &
          is.finite(full_tile_data$interval_years) &
          full_tile_data$interval_years > 0

        if (identical(pre_xgb_data_mode, "legacy_complete")) {
          predictor_matrix_for_filter <- as.matrix(
            full_tile_data[, ..predictors]
          )
          complete_predictors <- rowSums(
            is.finite(predictor_matrix_for_filter)
          ) == length(predictors)
          train_keep <- essential_keep & complete_predictors
        } else {
          train_keep <- essential_keep
        }

        subgrid_data <- full_tile_data[train_keep]

        log_line(
          worker_log_file,
          "PRE-XGB mode=", pre_xgb_data_mode,
          " | full rows=", nrow(full_tile_data),
          " | essential rows=", sum(essential_keep),
          " | model rows=", nrow(subgrid_data)
        )

        if (nrow(subgrid_data) < minimum_training_rows) {
          log_line(worker_log_file, "SKIP: fewer than ", minimum_training_rows,
                   " eligible training rows. N=", nrow(subgrid_data))
          return(list(ok = FALSE, reason = "insufficient_training_rows"))
        }

        # -------------------------------------------------------
        # 3B. PREPARE PREDICTION WORKFLOW DATA
        # -------------------------------------------------------
        # No complete.cases filtering is applied to predictors.
        # Rows with unusable coordinates or starting bathymetry cannot produce a
        # spatially located change product and are retained in the FST output but
        # will have NA derived values where calculation is not possible.
        if (prediction_available) {
          prediction_raw[, prediction_row_usable :=
                           is.finite(X) & is.finite(Y) & is.finite(bathy_t)]
        }

        # -------------------------------------------------------
        # 4. INITIALIZE TRAINING WORKFLOW STORAGE
        # -------------------------------------------------------
        boot_array <- matrix(
          NA_real_,
          nrow = nrow(full_tile_data),
          ncol = n_boot,
          dimnames = list(NULL, paste0("Rep_", seq_len(n_boot)))
        )
        influence_mat <- matrix(
          0,
          nrow = length(predictors),
          ncol = n_boot,
          dimnames = list(predictors, paste0("Rep_", seq_len(n_boot)))
        )
        predictor_boot_ranges_list <- vector("list", n_boot)
        shap_summary_list <- vector("list", n_boot)
        all_pdp_long_list <- vector("list", n_boot)

        # -------------------------------------------------------
        # 4B. INITIALIZE PREDICTION STORAGE
        # -------------------------------------------------------
        if (prediction_available) {
          pred_boot_array <- matrix(
            NA_real_,
            nrow = nrow(prediction_raw),
            ncol = n_boot,
            dimnames = list(NULL, paste0("Rep_", seq_len(n_boot)))
          )
        }

        # -------------------------------------------------------
        # 5. SETUP PDP DATA
        # -------------------------------------------------------
        predictor_means <- vapply(subgrid_data[, ..predictors], function(x) {
          if (any(is.finite(x))) mean(x, na.rm = TRUE) else NA_real_
        }, numeric(1))

        pdp_ranges <- rbindlist(lapply(predictors, function(p) {
          x <- subgrid_data[[p]]
          finite_x <- x[is.finite(x)]
          data.table(
            predictor = p,
            min_value = if (length(finite_x)) min(finite_x) else NA_real_,
            max_value = if (length(finite_x)) max(finite_x) else NA_real_,
            mean_value = predictor_means[[p]]
          )
        }))

        # -------------------------------------------------------
        # 6. SPATIAL CROSS VALIDATION
        # -------------------------------------------------------
        subgrid_sf <- st_as_sf(
          subgrid_data,
          coords = c("X", "Y"),
          remove = FALSE,
          crs = sf::st_crs(raster::crs(template_raster))
        )

        grid_blocks <- st_make_grid(subgrid_sf, cellsize = block_size_m)
        if (inherits(grid_blocks, "sfc")) grid_blocks <- st_sf(geometry = grid_blocks)
        points_in_blocks <- st_intersects(grid_blocks, subgrid_sf)
        occupied <- which(lengths(points_in_blocks) > 0L)
        max_k <- length(occupied)
        k_final <- min(n_folds, max_k)

        if (max_k > 0L) {
          generate_cv_diagnostic_plot(
            subgrid_sf,
            grid_blocks[occupied, ],
            max_k,
            tile_id,
            pair,
            output_dir_train
          )
        }

        sample_weights <- if (use_weighted_loss) {
          calculate_sample_weights(
            subgrid_data$delta_bathy,
            alpha = weight_alpha,
            method = weight_method,
            cap_quantile = weight_cap_quantile,
            max_weight = max_weight
          )
        } else {
          rep(1, nrow(subgrid_data))
        }

        weight_summary <- data.table(
          tile_id = tile_id,
          year_pair = pair,
          minimum = min(sample_weights, na.rm = TRUE),
          median = median(sample_weights, na.rm = TRUE),
          q95 = quantile(sample_weights, 0.95, na.rm = TRUE),
          maximum = max(sample_weights, na.rm = TRUE)
        )
        fwrite(weight_summary, file.path(tile_dir_train, paste0("weight_summary_", pair, ".csv")))

        # Round selection defaults to a 1000-tree floor because lean testing
        # showed that silent 100-tree fallback caused the severe magnitude error.
        best_iteration <- if (identical(round_selection_mode, "fixed")) {
          as.integer(fixed_nrounds)
        } else {
          as.integer(cv_fallback_nrounds)
        }
        cv_status <- if (identical(round_selection_mode, "fixed")) {
          "fixed"
        } else {
          "not_run"
        }
        cv_metrics_list <- list()
        best_nrounds <- integer()
        cv_errors <- character()

        if (!identical(round_selection_mode, "fixed") && k_final >= 2L) {
          set.seed(make_seed(pair_seed, "spatial_cv"))
          scv <- tryCatch(
            blockCV::cv_spatial(
              x = subgrid_sf,
              size = block_size_m,
              k = k_final,
              iteration = 200,
              progress = FALSE
            ),
            error = function(e) e
          )

          if (inherits(scv, "error")) {
            cv_status <- paste0("cv_creation_failed: ", conditionMessage(scv))
          } else if (is.null(scv$folds_list) || length(scv$folds_list) < 2L) {
            cv_status <- "cv_no_valid_folds"
          } else {
            cv_params <- list(
              objective = "reg:squarederror",
              eval_metric = c("rmse", "mae"),
              max_depth = max_depth,
              eta = eta,
              gamma = gamma,
              lambda = lambda,
              alpha = alpha,
              subsample = 0.7,
              colsample_bytree = 0.8,
              nthread = 1
            )

            for (k in seq_len(min(k_final, length(scv$folds_list)))) {
              # Keep OG fold-index extraction semantics.
              train_idx <- unlist(scv$folds_list[[k]][1])
              test_idx <- unlist(scv$folds_list[[k]][2])
              train_idx <- train_idx[train_idx >= 1L & train_idx <= nrow(subgrid_data)]
              test_idx <- test_idx[test_idx >= 1L & test_idx <= nrow(subgrid_data)]

              if (length(train_idx) < 20L || length(test_idx) < 10L) {
                cv_errors <- c(cv_errors, paste0("fold_", k, ": insufficient rows"))
                next
              }

              fold_result <- tryCatch({
                dtrain_fold <- xgb.DMatrix(
                  data = as.matrix(subgrid_data[train_idx, ..predictors]),
                  label = subgrid_data$bathy_t1[train_idx],
                  weight = sample_weights[train_idx],
                  missing = NA
                )
                dtest_fold <- xgb.DMatrix(
                  data = as.matrix(subgrid_data[test_idx, ..predictors]),
                  label = subgrid_data$bathy_t1[test_idx],
                  missing = NA
                )

                fold_model <- xgb.train(
                  params = cv_params,
                  data = dtrain_fold,
                  nrounds = cv_nrounds,
                  evals = list(train = dtrain_fold, test = dtest_fold),
                  early_stopping_rounds = early_stopping_rounds,
                  verbose = 0
                )

                fold_best <- suppressWarnings(as.integer(fold_model$best_iteration))
                if (
                  length(fold_best) != 1L || is.na(fold_best) ||
                  !is.finite(fold_best) || fold_best <= 0L
                ) {
                  elog <- fold_model$evaluation_log
                  if (
                    !is.null(elog) && "test_rmse" %in% names(elog) &&
                    any(is.finite(elog$test_rmse))
                  ) {
                    fold_best <- which.min(elog$test_rmse)
                  } else {
                    fold_best <- NA_integer_
                  }
                }

                fold_pred <- predict(fold_model, dtest_fold)
                fold_metrics <- calculate_common_metrics(
                  obs_bathy_t1 = subgrid_data$bathy_t1[test_idx],
                  pred_bathy_t1 = fold_pred,
                  bathy_t = subgrid_data$bathy_t[test_idx],
                  interval_years = subgrid_data$interval_years[test_idx]
                )
                fold_metrics[, `:=`(
                  tile_id = tile_id, year_pair = pair, fold = k,
                  best_iteration = fold_best
                )]

                list(best = fold_best, metrics = fold_metrics)
              }, error = function(e) e)

              if (inherits(fold_result, "error")) {
                cv_errors <- c(
                  cv_errors,
                  paste0("fold_", k, ": ", conditionMessage(fold_result))
                )
              } else {
                if (
                  length(fold_result$best) == 1L &&
                  !is.na(fold_result$best) &&
                  is.finite(fold_result$best) &&
                  fold_result$best > 0L
                ) {
                  best_nrounds <- c(best_nrounds, fold_result$best)
                }
                cv_metrics_list[[length(cv_metrics_list) + 1L]] <-
                  fold_result$metrics
              }
            }

            if (length(best_nrounds) > 0L) {
              cv_selected <- round(mean(best_nrounds, na.rm = TRUE))
              best_iteration <- if (identical(round_selection_mode, "cv_with_floor")) {
                max(as.integer(minimum_nrounds), as.integer(cv_selected))
              } else {
                max(50L, as.integer(cv_selected))
              }
              cv_status <- "cv_success"
            } else {
              cv_status <- "cv_failed_using_fallback"
              best_iteration <- if (identical(round_selection_mode, "cv_with_floor")) {
                max(as.integer(minimum_nrounds), as.integer(cv_fallback_nrounds))
              } else {
                as.integer(cv_fallback_nrounds)
              }
            }
          }
        }

        if (identical(round_selection_mode, "cv_with_floor")) {
          best_iteration <- max(as.integer(minimum_nrounds), best_iteration)
        }

        log_line(
          worker_log_file,
          "ROUND SELECTION: mode=", round_selection_mode,
          " | status=", cv_status,
          " | fold best=", paste(best_nrounds, collapse = ","),
          " | final rounds=", best_iteration
        )
        if (length(cv_errors) > 0L) {
          log_line(worker_log_file, "CV messages: ", paste(cv_errors, collapse = " | "))
        }

        cv_metrics <- if (length(cv_metrics_list)) {
          rbindlist(cv_metrics_list, fill = TRUE)
        } else {
          data.table(
            tile_id = tile_id, year_pair = pair, fold = NA_integer_,
            best_iteration = best_iteration, cv_status = cv_status
          )
        }
        cv_metrics[, cv_status := cv_status]
        fwrite(cv_metrics, file.path(tile_dir_train, paste0("spatial_cv_metrics_", pair, ".csv")))

        # -------------------------------------------------------
        # 7. BOOTSTRAP LOOP
        # -------------------------------------------------------
        model_save_dir <- file.path(tile_dir_train, "bootstrap_models")
        dir.create(model_save_dir, recursive = TRUE, showWarnings = FALSE)

        full_train_matrix <- xgb.DMatrix(
          as.matrix(full_tile_data[, ..predictors]),
          missing = NA
        )
        pred_matrix <- if (prediction_available) {
          xgb.DMatrix(as.matrix(prediction_raw[, ..predictors]), missing = NA)
        } else {
          NULL
        }

        xgb_params <- list(
          objective = "reg:squarederror",
          eval_metric = c("rmse", "mae"),
          max_depth = max_depth,
          eta = eta,
          gamma = gamma,
          lambda = lambda,
          alpha = alpha,
          subsample = 0.7,
          colsample_bytree = 0.8,
          nthread = 1
        )

        for (b in seq_len(n_boot)) {
          set.seed(make_seed(pair_seed, "bootstrap", b))
          boot_indices <- sample(seq_len(nrow(subgrid_data)), replace = TRUE)

          dtrain_boot <- xgb.DMatrix(
            data = as.matrix(subgrid_data[boot_indices, ..predictors]),
            label = subgrid_data$bathy_t1[boot_indices],
            weight = sample_weights[boot_indices],
            missing = NA
          )

          xgb_model <- xgb.train(
            params = xgb_params,
            data = dtrain_boot,
            nrounds = best_iteration,
            verbose = 0
          )

          saveRDS(
            xgb_model,
            file.path(model_save_dir, paste0("model_boot_", sprintf("%03d", b), "_", pair, ".rds"))
          )

          # -------------------------------------------------------
          # 7A. TRAINING PREDICTION INSIDE BOOTSTRAP
          # -------------------------------------------------------
          boot_array[, b] <- predict(xgb_model, full_train_matrix)

          # -------------------------------------------------------
          # 7B. PREDICTION WORKFLOW INSIDE TRAINING BOOTSTRAP
          # -------------------------------------------------------
          if (prediction_available) {
            pred_boot_array[, b] <- predict(xgb_model, pred_matrix)
          }

          importance_matrix <- xgb.importance(model = xgb_model)
          if (nrow(importance_matrix) > 0L) {
            imp_vals <- setNames(importance_matrix$Gain, importance_matrix$Feature)
            matched <- intersect(names(imp_vals), predictors)
            influence_mat[matched, b] <- imp_vals[matched]
          }

          boot_sample <- subgrid_data[boot_indices]
          predictor_boot_ranges_list[[b]] <- make_predictor_range_table(
            boot_sample,
            predictors
          )[, bootstrap_iteration := b]

          # -------------------------------------------------------
          # 8. STORE PDP DATA (as CHANGE)
          # -------------------------------------------------------
          pdp_list <- list()
          for (p in predictors) {
            rr <- pdp_ranges[predictor == p]
            if (!nrow(rr) || !is.finite(rr$min_value) || !is.finite(rr$max_value)) next

            pdp_grid <- as.data.table(as.list(predictor_means))[rep(1L, 100L)]
            pdp_grid[, (p) := seq(rr$min_value, rr$max_value, length.out = 100)]
            pdp_pred_t1 <- predict(
              xgb_model,
              xgb.DMatrix(as.matrix(pdp_grid[, ..predictors]), missing = NA)
            )
            # PDP is reported as change relative to mean starting bathymetry.
            start_bathy_mean <- predictor_means[["bathy_t"]]
            pdp_list[[p]] <- data.table(
              predictor = p,
              predictor_value = pdp_grid[[p]],
              bootstrap_iteration = b,
              predicted_change = pdp_pred_t1 - start_bathy_mean
            )
          }
          all_pdp_long_list[[b]] <- rbindlist(pdp_list, fill = TRUE)

          # -------------------------------------------------------
          # 8A. STORE SHAP DATA
          # -------------------------------------------------------
          shap_n <- min(1000L, nrow(subgrid_data))
          set.seed(make_seed(pair_seed, "shap", b))
          shap_idx <- sample(seq_len(nrow(subgrid_data)), shap_n)
          shap_matrix <- as.matrix(subgrid_data[shap_idx, ..predictors])

          shap_values <- tryCatch(
            predict(xgb_model, shap_matrix, predcontrib = TRUE),
            error = function(e) NULL
          )

          shap_summary_list[[b]] <- if (!is.null(shap_values)) {
            shap_no_bias <- shap_values[, seq_along(predictors), drop = FALSE]
            data.table(
              predictor = predictors,
              mean_abs_shap = colMeans(abs(shap_no_bias), na.rm = TRUE),
              bootstrap_iteration = b
            )
          } else {
            data.table()
          }
        }

        # -------------------------------------------------------
        # 8.5 PROCESS TRAINING BOOTSTRAP
        # -------------------------------------------------------
        mean_pred_t1 <- rowMeans(boot_array, na.rm = TRUE)
        mean_pred_t1[rowSums(is.finite(boot_array)) == 0L] <- NA_real_
        sd_pred_t1 <- apply(boot_array, 1, sd, na.rm = TRUE)
        if (n_boot == 1L) sd_pred_t1[is.finite(mean_pred_t1)] <- 0

        translated_train <- translate_model_a_predictions(
          pred_bathy_t1 = mean_pred_t1,
          bathy_t = training_raw$bathy_t,
          interval_years = ifelse(
            is.finite(training_raw$interval_years) & training_raw$interval_years > 0,
            training_raw$interval_years,
            interval_years
          ),
          standard_horizon_years = standard_horizon_years
        )

        training_summary <- copy(training_raw)
        training_summary[, mean_predicted_bathy_t1 := translated_train$pred_bathy_t1]
        training_summary[, uncertainty_sd_bathy_t1 := sd_pred_t1]
        training_summary[, mean_predicted_change := translated_train$pred_delta_bathy]
        training_summary[, mean_predicted_rate := translated_train$pred_delta_rate]
        training_summary[, mean_predicted_standard_change := translated_train$pred_delta_standard]
        training_summary[, mean_predicted_standard_bathy := translated_train$pred_bathy_standard]
        training_summary[, prediction_residual := bathy_t1 - mean_predicted_bathy_t1]
        training_summary[, delta_residual := delta_bathy - mean_predicted_change]

        training_metrics <- calculate_common_metrics(
          obs_bathy_t1 = training_summary$bathy_t1,
          pred_bathy_t1 = training_summary$mean_predicted_bathy_t1,
          bathy_t = training_summary$bathy_t,
          interval_years = ifelse(
            is.finite(training_summary$interval_years) & training_summary$interval_years > 0,
            training_summary$interval_years,
            interval_years
          )
        )
        training_metrics[, `:=`(tile_id = tile_id, year_pair = pair, evaluation = "apparent_full_tile")]

        # -------------------------------------------------------
        # 8.5b PROCESS PREDICTION BOOTSTRAP & SAVE OUTPUTS
        # -------------------------------------------------------
        prediction_summary <- NULL
        if (prediction_available) {
          pred_mean_t1 <- rowMeans(pred_boot_array, na.rm = TRUE)
          pred_mean_t1[rowSums(is.finite(pred_boot_array)) == 0L] <- NA_real_
          pred_sd_t1 <- apply(pred_boot_array, 1, sd, na.rm = TRUE)
          if (n_boot == 1L) pred_sd_t1[is.finite(pred_mean_t1)] <- 0

          translated_pred <- translate_model_a_predictions(
            pred_bathy_t1 = pred_mean_t1,
            bathy_t = prediction_raw$bathy_t,
            interval_years = interval_years,
            standard_horizon_years = standard_horizon_years
          )

          prediction_summary <- copy(prediction_raw)
          prediction_summary[, mean_predicted_bathy_t1 := translated_pred$pred_bathy_t1]
          prediction_summary[, uncertainty_sd_bathy_t1 := pred_sd_t1]
          prediction_summary[, mean_predicted_change := translated_pred$pred_delta_bathy]
          prediction_summary[, mean_predicted_rate := translated_pred$pred_delta_rate]
          prediction_summary[, mean_predicted_standard_change := translated_pred$pred_delta_standard]
          prediction_summary[, mean_predicted_standard_bathy := translated_pred$pred_bathy_standard]

          predictor_ranges <- make_predictor_range_table(subgrid_data, predictors)
          prediction_summary <- add_extrapolation_flags(prediction_summary, predictor_ranges)
          
          # -------------------------------------------------------
          # VERIFY SOURCE-DATA FIELDS SURVIVED INTO LOCAL SUMMARY
          # -------------------------------------------------------
          #
          # prediction_summary is deliberately constructed as a full copy of
          # prediction_raw. Therefore invariant spatial/source fields must survive
          # unchanged into the saved LOCAL summary.
          # -------------------------------------------------------
          
          missing_required_summary_fields <- setdiff(
            required_local_summary_fields,
            names(prediction_summary)
          )
          
          if (length(missing_required_summary_fields) > 0L) {
            
            stop(
              "LOCAL prediction summary schema failure for ",
              tile_id,
              " / ",
              pair,
              ": required source-data field(s) were lost: ",
              paste(
                missing_required_summary_fields,
                collapse = ", "
              )
            )
          }

          fst::write_fst(
            prediction_summary,
            file.path(tile_dir_pred, paste0("LOCAL_prediction_summary_", pair, ".fst"))
          )
          fwrite(
            predictor_ranges,
            file.path(tile_dir_train, paste0("training_predictor_ranges_", pair, ".csv"))
          )

          # Spatial rasters are written once in Section 9A through
          # save_local_visual_outputs(), avoiding duplicate disk I/O.
        }

        # -------------------------------------------------------
        # VERIFY SOURCE-DATA FIELDS SURVIVED INTO TRAINING SUMMARY
        # -------------------------------------------------------
        
        missing_required_training_summary_fields <- setdiff(
          required_local_summary_fields,
          names(training_summary)
        )
        
        if (length(missing_required_training_summary_fields) > 0L) {
          
          stop(
            "LOCAL training summary schema failure for ",
            tile_id,
            " / ",
            pair,
            ": required source-data field(s) were lost: ",
            paste(
              missing_required_training_summary_fields,
              collapse = ", "
            )
          )
        }
        # -------------------------------------------------------
        # 9. SAVE TRAINING OUTPUTS
        # -------------------------------------------------------
        fst::write_fst(
          training_summary,
          file.path(tile_dir_train, paste0("LOCAL_training_summary_", pair, ".fst"))
        )
        fwrite(
          training_metrics,
          file.path(tile_dir_train, paste0("LOCAL_training_metrics_", pair, ".csv"))
        )

        importance_dt <- as.data.table(influence_mat, keep.rownames = "predictor")
        fwrite(importance_dt, file.path(tile_dir_train, paste0("bootstrap_importance_", pair, ".csv")))
        fwrite(
          rbindlist(predictor_boot_ranges_list, fill = TRUE),
          file.path(tile_dir_train, paste0("bootstrap_predictor_ranges_", pair, ".csv"))
        )
        fwrite(
          rbindlist(shap_summary_list, fill = TRUE),
          file.path(tile_dir_train, paste0("bootstrap_SHAP_summary_", pair, ".csv"))
        )
        fwrite(
          rbindlist(all_pdp_long_list, fill = TRUE),
          file.path(tile_dir_train, paste0("bootstrap_PDP_as_change_", pair, ".csv"))
        )

        model_metadata <- list(
          model_formulation = "Model A: bathy_t + environmental predictors -> bathy_t1",
          tile_id = tile_id,
          year_pair = pair,
          interval_years = interval_years,
          predictors = predictors,
          best_iteration = best_iteration,
          n_boot = n_boot,
          weighting = list(
            enabled = use_weighted_loss,
            method = weight_method,
            alpha = weight_alpha,
            cap_quantile = weight_cap_quantile,
            max_weight = max_weight
          ),
          template_crs = raster::crs(template_raster),
          created = Sys.time()
        )
        saveRDS(model_metadata, file.path(tile_dir_train, paste0("LOCAL_model_metadata_", pair, ".rds")))


        # -------------------------------------------------------
        # 9A. WRITE LOCAL RASTERS + DIAGNOSTIC VISUAL OUTPUTS
        # -------------------------------------------------------
        shap_summary_dt <- rbindlist(shap_summary_list, fill = TRUE)
        importance_dt_for_plot <- as.data.table(
          influence_mat,
          keep.rownames = "predictor"
        )

        tryCatch(
          save_local_visual_outputs(
            training_summary = training_summary,
            prediction_summary = prediction_summary,
            shap_summary_dt = shap_summary_dt,
            importance_dt = importance_dt_for_plot,
            training_metrics = training_metrics,
            tile_id = tile_id,
            pair = pair,
            tile_dir_train = tile_dir_train,
            tile_dir_pred = tile_dir_pred,
            template_raster = template_raster,
            tile_geometry = tile_geometry,
            worker_log_file = worker_log_file,
            standard_horizon_years = standard_horizon_years
          ),
          error = function(e) {
            log_line(
              worker_log_file,
              "VISUAL OUTPUT BLOCK FAILED: ",
              conditionMessage(e)
            )
          }
        )

        # -------------------------------------------------------
        # 9B. WRITE OPTIONAL GLOBAL TRAINING SHARD
        # -------------------------------------------------------
        # This is intentionally only a standardized training shard, not a fitted
        # global model. It lets the later global workflow operate independently.
        if (write_global_training_shards) {
          pair_shard_dir <- file.path(global_shard_dir, pair)
          dir.create(pair_shard_dir, recursive = TRUE, showWarnings = FALSE)

          shard_cols <- unique(c(
            "tile_id", "pair_id", "X", "Y", "FID", "bathy_t", "bathy_t1",
            "delta_bathy", "delta_rate", "interval_years", predictors
          ))
          shard <- subgrid_data[, ..shard_cols]
          fst::write_fst(
            shard,
            file.path(pair_shard_dir, paste0(tile_id, "_", pair, "_global_training_shard.fst"))
          )

          manifest_row <- data.table(
            tile_id = tile_id,
            year_pair = pair,
            shard_path = file.path(pair_shard_dir, paste0(tile_id, "_", pair, "_global_training_shard.fst")),
            n_rows = nrow(shard),
            predictors = paste(predictors, collapse = "|"),
            created = as.character(Sys.time())
          )
          fwrite(
            manifest_row,
            file.path(pair_shard_dir, paste0(tile_id, "_", pair, "_manifest.csv"))
          )
        }

        log_line(worker_log_file, "SUCCESS: ", tile_id, " / ", pair,
                 " | training rows=", nrow(subgrid_data),
                 " | prediction available=", prediction_available)

        list(
          ok = TRUE,
          tile_id = tile_id,
          year_pair = pair,
          n_training_rows = nrow(subgrid_data),
          prediction_available = prediction_available,
          best_iteration = best_iteration
        )
      }, error = function(e) {
        log_line(worker_log_file, "ERROR: ", conditionMessage(e))
        list(ok = FALSE, tile_id = tile_id, year_pair = pair, reason = conditionMessage(e))
      })

      tile_results[[pair]] <- pair_result
    }

    log_section(worker_log_file, paste0("LOCAL TILE END: ", tile_id))
    tile_results
  }

  # The tile-processing cluster is no longer needed. Stop it now rather than
  # leaving it alive while full-extent deployment creates another cluster.
  try(parallel::stopCluster(cl), silent = TRUE)
  cl <- NULL
  foreach::registerDoSEQ()
  gc(verbose = FALSE)

  log_section(master_log_file, "LOCAL MODEL TRAINING/TILE-NATIVE PREDICTION END")

  batch_result_filename <- if (length(year_pairs) == 1L) {
    paste0("LOCAL_MODEL_RUN_RESULTS_", year_pairs[[1]], ".rds")
  } else {
    "LOCAL_MODEL_RUN_RESULTS_BATCH.rds"
  }

  saveRDS(
    results_list,
    file.path(output_dir_train, batch_result_filename)
  )

  # -------------------------------------------------------
  # 11. AUTOMATIC FULL-EXTENT LOCAL DEPLOYMENT
  # -------------------------------------------------------
  # Run only after all available local models have been trained. This fills
  # prediction-only tiles using the geographically nearest trained LOCAL model
  # ensemble, while retaining tile-native predictions where they already exist.
  full_extent_results <- NULL

  if (isTRUE(deploy_local_full_extent)) {
    log_section(master_log_file, "AUTOMATIC FULL-EXTENT LOCAL DEPLOYMENT")

    full_extent_results <- Predict_Full_Extent_With_Nearest_Local_Models(
      grid_sf = grid_sf,
      output_dir_train = output_dir_train,
      output_dir_pred = output_dir_pred,
      year_pairs = year_pairs,
      template_raster = template_raster,
      tile_ids = tile_ids,
      overwrite_existing_local = overwrite_existing_local,
      write_spatial_outputs = write_full_extent_spatial_outputs,
      n_workers = n_workers
    )
  }

  log_section(master_log_file, "LOCAL MODEL RUN END")

  invisible(list(
    tile_native_results = results_list,
    full_extent_results = full_extent_results
  ))
}



# ==============================================================================
# PART B2. MEMORY-SAFE MULTI-YEAR LOCAL WORKFLOW ENTRY POINT
# ==============================================================================
#
# Why this wrapper exists
# -----------------------
# A PSOCK message such as:
#
#   Error in unserialize(node$con): error reading from connection
#
# means that a parallel worker process terminated before returning its result.
# With several year pairs, one long-lived worker can accumulate large external
# XGBoost matrices and other objects across intervals. Processing one year pair
# per fresh cluster prevents that accumulation and makes the run restartable.
#
# The scientific workflow is unchanged. Only execution batching differs.
# ------------------------------------------------------------------------------
# using run #010
Run_Local_Model_Training_And_Prediction <- function(
    grid_sf,
    output_dir_train,
    output_dir_pred,
    year_pairs,
    master_predictor_list_path,
    template_raster,
    tile_ids = "all",
    block_size_m = 200,
    n_boot = 5,
    n_folds = 5,
    use_weighted_loss = TRUE,
    weight_alpha = 0.5,
    weight_method = "power",
    weight_cap_quantile = 0.95,
    max_weight = 2,
    max_depth = 4,
    eta = 0.01,
    gamma = 1.0,
    lambda = 1,
    alpha = 1,

    pre_xgb_data_mode = c(
      "native_missing",
      "legacy_complete"
    ),

    predictor_resolution_mode = c(
      "og_training_first",
      "deployment_intersection"
    ),

    round_selection_mode = c(
      "cv_with_floor",
      "fixed",
      "cv_only"
    ),

    minimum_nrounds = 1000,
    fixed_nrounds = 1000,
    cv_nrounds = 2000,
    early_stopping_rounds = 15,
    cv_fallback_nrounds = 1000,

    standard_horizon_years = 2,
    minimum_training_rows = 100,
    require_prediction_compatible_predictors = TRUE,
    write_global_training_shards = TRUE,
    global_shard_dir = file.path(
      output_dir_train,
      "_GLOBAL_TRAINING_SHARDS"
    ),

    deploy_local_full_extent = TRUE,
    overwrite_existing_local = FALSE,
    write_full_extent_spatial_outputs = TRUE,

    # Large XGBoost/raster jobs should not normally use every logical core.
    # Four to eight PSOCK workers is usually safer on a shared workstation.
    n_workers = max(
      1,
      min(
        6,
        parallel::detectCores() - 1
      )
    ),

    continue_after_pair_failure = FALSE,
    global_seed = 12345
) {
  pre_xgb_data_mode <- match.arg(
    pre_xgb_data_mode
  )

  predictor_resolution_mode <- match.arg(
    predictor_resolution_mode
  )

  round_selection_mode <- match.arg(
    round_selection_mode
  )

  year_pairs <- unique(
    as.character(year_pairs)
  )

  if (length(year_pairs) == 0L) {
    stop("No year pairs supplied.")
  }

  master_log_file <- file.path(
    output_dir_train,
    "LOCAL_MULTI_YEAR_MASTER_LOG.txt"
  )

  log_section(
    master_log_file,
    "MEMORY-SAFE MULTI-YEAR LOCAL RUN START"
  )

  log_line(
    master_log_file,
    "Year pairs: ",
    paste(
      year_pairs,
      collapse = ", "
    )
  )

  log_line(
    master_log_file,
    "Workers per year-pair batch: ",
    n_workers
  )

  all_pair_results <- list()
  failed_pairs <- list()

  for (pair_index in seq_along(year_pairs)) {
    pair <- year_pairs[[pair_index]]

    log_section(
      master_log_file,
      paste0(
        "YEAR-PAIR BATCH ",
        pair_index,
        "/",
        length(year_pairs),
        ": ",
        pair
      )
    )

    message(
      "\n============================================================"
    )
    message(
      "LOCAL MODEL YEAR-PAIR BATCH ",
      pair_index,
      " / ",
      length(year_pairs),
      ": ",
      pair
    )
    message(
      "============================================================"
    )

    pair_result <- tryCatch(
      .Run_Local_Model_Training_And_Prediction_Batch(
        grid_sf = grid_sf,
        output_dir_train = output_dir_train,
        output_dir_pred = output_dir_pred,
        year_pairs = pair,
        master_predictor_list_path =
          master_predictor_list_path,
        template_raster = template_raster,
        tile_ids = tile_ids,
        block_size_m = block_size_m,
        n_boot = n_boot,
        n_folds = n_folds,
        use_weighted_loss =
          use_weighted_loss,
        weight_alpha = weight_alpha,
        weight_method = weight_method,
        weight_cap_quantile =
          weight_cap_quantile,
        max_weight = max_weight,
        max_depth = max_depth,
        eta = eta,
        gamma = gamma,
        lambda = lambda,
        alpha = alpha,
        pre_xgb_data_mode =
          pre_xgb_data_mode,
        predictor_resolution_mode =
          predictor_resolution_mode,
        round_selection_mode =
          round_selection_mode,
        minimum_nrounds =
          minimum_nrounds,
        fixed_nrounds = fixed_nrounds,
        cv_nrounds = cv_nrounds,
        early_stopping_rounds =
          early_stopping_rounds,
        cv_fallback_nrounds =
          cv_fallback_nrounds,
        standard_horizon_years =
          standard_horizon_years,
        minimum_training_rows =
          minimum_training_rows,
        require_prediction_compatible_predictors =
          require_prediction_compatible_predictors,
        write_global_training_shards =
          write_global_training_shards,
        global_shard_dir =
          global_shard_dir,

        # Full-extent deployment is run inside this pair-specific batch, after
        # its tile-processing cluster has been explicitly stopped.
        deploy_local_full_extent =
          deploy_local_full_extent,
        overwrite_existing_local =
          overwrite_existing_local,
        write_full_extent_spatial_outputs =
          write_full_extent_spatial_outputs,

        n_workers = n_workers,
        global_seed = global_seed
      ),
      error = function(e) {
        structure(
          list(
            ok = FALSE,
            year_pair = pair,
            reason =
              conditionMessage(e)
          ),
          class = c(
            "local_pair_batch_error",
            "list"
          )
        )
      }
    )

    if (
      inherits(
        pair_result,
        "local_pair_batch_error"
      )
    ) {
      failed_pairs[[pair]] <- pair_result

      log_line(
        master_log_file,
        "FAILED year pair ",
        pair,
        ": ",
        pair_result$reason
      )

      saveRDS(
        pair_result,
        file.path(
          output_dir_train,
          paste0(
            "LOCAL_MODEL_PAIR_FAILURE_",
            pair,
            ".rds"
          )
        )
      )

      if (
        !isTRUE(
          continue_after_pair_failure
        )
      ) {
        stop(
          "Local modelling failed for ",
          pair,
          ": ",
          pair_result$reason,
          "\nCompleted pair results remain on disk. ",
          "Rerun with only the failed pair after correcting the cause."
        )
      }
    } else {
      all_pair_results[[pair]] <-
        pair_result

      saveRDS(
        pair_result,
        file.path(
          output_dir_train,
          paste0(
            "LOCAL_MODEL_COMPLETE_RESULT_",
            pair,
            ".rds"
          )
        )
      )

      log_line(
        master_log_file,
        "COMPLETED year pair ",
        pair
      )
    }

    # Release master-process references and encourage external-pointer cleanup
    # before the next cluster is created.
    try(foreach::registerDoSEQ(), silent = TRUE)
    closeAllConnections()
    gc(verbose = FALSE)
    Sys.sleep(1)
  }

  combined_result <- list(
    pair_results =
      all_pair_results,
    failed_pairs =
      failed_pairs,
    year_pairs_requested =
      year_pairs,
    n_workers_per_batch =
      n_workers,
    completed_at =
      Sys.time()
  )

  saveRDS(
    combined_result,
    file.path(
      output_dir_train,
      "LOCAL_MODEL_ALL_YEAR_PAIRS_RESULTS.rds"
    )
  )

  log_section(
    master_log_file,
    "MEMORY-SAFE MULTI-YEAR LOCAL RUN END"
  )

  log_line(
    master_log_file,
    "Completed pairs: ",
    paste(
      names(all_pair_results),
      collapse = ", "
    )
  )

  if (length(failed_pairs) > 0L) {
    log_line(
      master_log_file,
      "Failed pairs: ",
      paste(
        names(failed_pairs),
        collapse = ", "
      )
    )
  }

  invisible(
    combined_result
  )
}


# ==============================================================================
# PART C. OPTIONAL FULL-EXTENT LOCAL MODEL DEPLOYMENT
# ==============================================================================
#
# Purpose
# -------
# The tile-native local workflow can only fit a model where a training tile
# exists. This secondary LOCAL deployment fills prediction-only tiles by applying
# the geographically nearest successfully trained local model ensemble.
#
# This is NOT nearest-neighbour interpolation between raster cells. Each cell is
# predicted independently by XGBoost from its own predictor vector. Geographic
# proximity is used only once: to choose which trained LOCAL model ensemble is
# assigned to a prediction tile that lacks its own local training model.
#
# Existing tile-native local outputs are retained by default. Only missing local
# prediction tiles are filled unless overwrite_existing_local = TRUE.
# ------------------------------------------------------------------------------

Predict_Full_Extent_With_Nearest_Local_Models <- function(
    grid_sf,
    output_dir_train,
    output_dir_pred,
    year_pairs,
    template_raster,
    tile_ids = "all",
    overwrite_existing_local = TRUE,
    write_spatial_outputs = TRUE,
    n_workers = max(1, parallel::detectCores() - 1)
) {
  if (!inherits(grid_sf, "sf")) {
    stop("grid_sf must be an sf object.")
  }

  prediction_grid <- select_grid_tiles(
    grid_sf = grid_sf,
    tile_ids = tile_ids,
    allowed_sources = c("both", "prediction")
  )

  prediction_grid$tile_id <- as.character(prediction_grid$tile_id)

  log_file <- file.path(
    output_dir_pred,
    "_LOCAL_FULL_EXTENT_LOGS",
    "LOCAL_FULL_EXTENT_DEPLOYMENT_LOG.txt"
  )

  log_section(log_file, "FULL-EXTENT LOCAL DEPLOYMENT START")
  log_line(log_file, "Prediction tiles selected: ", nrow(prediction_grid))

  all_results <- list()

  for (pair in year_pairs) {
    # -------------------------------------------------------
    # C1. IDENTIFY SUCCESSFULLY TRAINED LOCAL MODEL TILES
    # -------------------------------------------------------
    candidate_model_tiles <- unique(as.character(grid_sf$tile_id))

    model_inventory <- data.table::rbindlist(
      lapply(candidate_model_tiles, function(model_tile_id) {
        metadata_path <- file.path(
          output_dir_train,
          model_tile_id,
          paste0("LOCAL_model_metadata_", pair, ".rds")
        )

        model_dir <- file.path(
          output_dir_train,
          model_tile_id,
          "bootstrap_models"
        )

        model_paths <- if (dir.exists(model_dir)) {
          list.files(
            model_dir,
            pattern = paste0(
              "^model_boot_[0-9]+_",
              pair,
              "\\.rds$"
            ),
            full.names = TRUE
          )
        } else {
          character()
        }

        if (
          file.exists(metadata_path) &&
          length(model_paths) > 0L
        ) {
          data.table::data.table(
            source_model_tile = model_tile_id,
            metadata_path = metadata_path,
            model_paths = list(model_paths)
          )
        } else {
          NULL
        }
      }),
      fill = TRUE
    )

    if (nrow(model_inventory) == 0L) {
      log_line(log_file, "SKIP ", pair, ": no trained local models found.")
      next
    }

    model_grid <- grid_sf[
      as.character(grid_sf$tile_id) %in%
        model_inventory$source_model_tile,
      ,
      drop = FALSE
    ]

    model_grid$source_model_tile <-
      as.character(model_grid$tile_id)

    # Use geometry-only interior points for assigning a model to a prediction
    # tile. Working on st_geometry() avoids the harmless sf warning that
    # attributes are assumed constant over polygon geometries.
    prediction_centroids <- sf::st_point_on_surface(
      sf::st_geometry(prediction_grid)
    )

    model_centroids <- sf::st_point_on_surface(
      sf::st_geometry(model_grid)
    )

    nearest_index <- sf::st_nearest_feature(
      prediction_centroids,
      model_centroids
    )

    assignment <- data.table::data.table(
      tile_id = prediction_grid$tile_id,
      source_model_tile =
        model_grid$source_model_tile[nearest_index],
      model_distance_map_units = as.numeric(
        sf::st_distance(
          prediction_centroids,
          model_centroids[nearest_index, ],
          by_element = TRUE
        )
      )
    )

    # A tile that has its own trained model must always use itself.
    assignment[
      tile_id %in% model_inventory$source_model_tile,
      `:=`(
        source_model_tile = tile_id,
        model_distance_map_units = 0
      )
    ]

    assignment <- merge(
      assignment,
      model_inventory,
      by = "source_model_tile",
      all.x = TRUE,
      sort = FALSE
    )

    data.table::fwrite(
      assignment[
        ,
        .(
          tile_id,
          source_model_tile,
          model_distance_map_units,
          metadata_path
        )
      ],
      file.path(
        output_dir_pred,
        paste0(
          "LOCAL_FULL_EXTENT_MODEL_ASSIGNMENT_",
          pair,
          ".csv"
        )
      )
    )

    # -------------------------------------------------------
    # C2. APPLY ASSIGNED LOCAL MODEL TO EACH PREDICTION TILE
    # -------------------------------------------------------
    cl <- parallel::makeCluster(n_workers)
    doParallel::registerDoParallel(cl)

    pair_results <- foreach::foreach(
      row_index = seq_len(nrow(assignment)),
      .packages = c(
        "data.table",
        "fst",
        "xgboost",
        "raster",
        "sp",
        "sf"
      ),
      .export = c(
        "find_tile_file",
        "safe_numeric_cols",
        "pair_duration_years",
        "translate_model_a_predictions",
        "make_predictor_range_table",
        "add_extrapolation_flags",
        "write_prediction_tif",
        "required_local_summary_fields" ## Required by the LOCAL summary schema checks, object defined in Part A but must be explicitly exported to the PSOCK workers used for full-extent deployment
      ),
      .errorhandling = "pass"
    ) %dopar% {
      assignment_row <- assignment[row_index]

      tryCatch({
        tile_id <- assignment_row$tile_id
        source_model_tile <-
          assignment_row$source_model_tile

        tile_geometry <- prediction_grid[
          as.character(prediction_grid$tile_id) == tile_id,
          ,
          drop = FALSE
        ]

        if (nrow(tile_geometry) != 1L) {
          stop(
            "Expected exactly one prediction-grid geometry for tile ",
            tile_id,
            "; found ",
            nrow(tile_geometry),
            "."
          )
        }

        tile_dir_pred <- file.path(
          output_dir_pred,
          tile_id
        )

        dir.create(
          tile_dir_pred,
          recursive = TRUE,
          showWarnings = FALSE
        )

        # -------------------------------------------------------
        # C?. CHECK WHETHER EXISTING LOCAL OUTPUTS ARE CURRENT
        # -------------------------------------------------------
        #
        # IMPORTANT:
        # A raster existing does NOT necessarily mean the associated
        # LOCAL_prediction_summary FST was produced with the current
        # predictor schema.
        #
        # This became important after adding predictors such as
        # distance_from_shore. Older prediction rasters may still exist
        # while their summary FSTs contain an older set of columns.
        #
        # We therefore skip a tile only when:
        #
        #   1. the primary LOCAL prediction raster exists;
        #   2. the LOCAL prediction summary exists; AND
        #   3. that summary contains every predictor required by the
        #      currently assigned LOCAL model.
        #
        # Otherwise the tile is rebuilt from the original raw prediction
        # FST, preserving all predictor columns and native NA values.
        # -------------------------------------------------------
        
        existing_change_tif <- file.path(
          tile_dir_pred,
          paste0(
            "LOCAL_Pred_Mean_Change_",
            pair,
            ".tif"
          )
        )
        
        existing_summary_fst <- file.path(
          tile_dir_pred,
          paste0(
            "LOCAL_prediction_summary_",
            pair,
            ".fst"
          )
        )
        
        # Read the assigned model metadata now so that we know the
        # CURRENT predictor schema before deciding whether to skip.
        metadata <- readRDS(
          assignment_row$metadata_path
        )
        
        predictors <- metadata$predictors
        
        summary_schema_current <- FALSE
        summary_schema_reason <- "summary_not_present"
        
        if (file.exists(existing_summary_fst)) {
          
          summary_meta <- tryCatch(
            fst::metadata_fst(
              existing_summary_fst
            ),
            error = function(e) NULL
          )
          
          if (!is.null(summary_meta)) {
            
            existing_summary_columns <- summary_meta$columnNames
            
            # --------------------------------------------------------------------------
            # Validate BOTH:
            #
            #   1. the predictor schema required by the assigned LOCAL model; and
            #   2. invariant source-data fields that every LOCAL summary must preserve.
            #
            # This prevents an older summary from being treated as current merely because
            # it satisfies an older model's predictor list.
            # --------------------------------------------------------------------------
            
            missing_summary_predictors <- setdiff(
              predictors,
              existing_summary_columns
            )
            
            missing_required_summary_fields <- setdiff(
              required_local_summary_fields,
              existing_summary_columns
            )
            
            summary_schema_current <-
              length(missing_summary_predictors) == 0L &&
              length(missing_required_summary_fields) == 0L
            
            
            if (summary_schema_current) {
              
              summary_schema_reason <- "current"
              
            } else {
              
              schema_problems <- character()
              
              if (length(missing_summary_predictors) > 0L) {
                schema_problems <- c(
                  schema_problems,
                  paste0(
                    "missing model predictors: ",
                    paste(
                      missing_summary_predictors,
                      collapse = ", "
                    )
                  )
                )
              }
              
              if (length(missing_required_summary_fields) > 0L) {
                schema_problems <- c(
                  schema_problems,
                  paste0(
                    "missing required summary fields: ",
                    paste(
                      missing_required_summary_fields,
                      collapse = ", "
                    )
                  )
                )
              }
              
              summary_schema_reason <- paste(
                schema_problems,
                collapse = " | "
              )
            }
            
          } else {
            
            summary_schema_reason <-
              "unable_to_read_summary_metadata"
          }
        }
        
        outputs_current <-
          file.exists(existing_change_tif) &&
          file.exists(existing_summary_fst) &&
          summary_schema_current
        
        
        if (
          outputs_current &&
          !isTRUE(overwrite_existing_local)
        ) {
          
          return(
            list(
              ok = TRUE,
              tile_id = tile_id,
              year_pair = pair,
              source_model_tile =
                assignment_row$source_model_tile,
              deployment =
                "existing_local_outputs_current",
              skipped_existing = TRUE,
              n_rasters_written = 0L
            )
          )
        }
        
        
        # If the raster exists but the summary is stale, report why the
        # supposedly-existing tile is intentionally being rebuilt.
        if (
          file.exists(existing_change_tif) &&
          !summary_schema_current
        ) {
          
          message(
            "REBUILD ",
            pair,
            " | ",
            tile_id,
            " | existing raster found but LOCAL summary schema is stale: ",
            summary_schema_reason
          )
        }

        prediction_path <- find_tile_file(
          tile_dir_pred,
          pair,
          type = "pred"
        )

        if (is.na(prediction_path)) {
          return(list(
            ok = FALSE,
            tile_id = tile_id,
            year_pair = pair,
            reason = "missing_prediction_file"
          ))
        }

        model_paths <- unlist(
          assignment_row$model_paths
        )

        prediction_dt <- fst::read_fst(
          prediction_path,
          as.data.table = TRUE
        )

        data.table::setDT(
          prediction_dt
        )

        missing_predictors <- setdiff(
          predictors,
          names(prediction_dt)
        )

        if (length(missing_predictors) > 0L) {
          stop(
            "Assigned LOCAL model ",
            source_model_tile,
            " requires missing predictors: ",
            paste(
              missing_predictors,
              collapse = ", "
            )
          )
        }

        interval_years <- pair_duration_years(
          pair
        )

        prediction_dt <- safe_numeric_cols(
          prediction_dt,
          unique(c(
            predictors,
            "X",
            "Y",
            "FID",
            "bathy_t"
          ))
        )

        prediction_dt[
          ,
          `:=`(
            tile_id = tile_id,
            pair_id = pair,
            interval_years =
              interval_years,
            local_source_model_tile =
              source_model_tile,
            local_source_model_distance =
              assignment_row$model_distance_map_units
          )
        ]

        prediction_matrix <- xgboost::xgb.DMatrix(
          as.matrix(
            prediction_dt[, ..predictors]
          ),
          missing = NA
        )

        bootstrap_predictions <- matrix(
          NA_real_,
          nrow = nrow(prediction_dt),
          ncol = length(model_paths)
        )

        for (
          bootstrap_index in
          seq_along(model_paths)
        ) {
          local_model <- readRDS(
            model_paths[[bootstrap_index]]
          )

          bootstrap_predictions[
            ,
            bootstrap_index
          ] <- predict(
            local_model,
            prediction_matrix
          )

          rm(local_model)
        }

        mean_predicted_t1 <- rowMeans(
          bootstrap_predictions,
          na.rm = TRUE
        )

        mean_predicted_t1[
          rowSums(
            is.finite(
              bootstrap_predictions
            )
          ) == 0L
        ] <- NA_real_

        sd_predicted_t1 <- apply(
          bootstrap_predictions,
          1,
          stats::sd,
          na.rm = TRUE
        )

        if (
          ncol(
            bootstrap_predictions
          ) == 1L
        ) {
          sd_predicted_t1[
            is.finite(
              mean_predicted_t1
            )
          ] <- 0
        }

        translated <- translate_model_a_predictions(
          pred_bathy_t1 =
            mean_predicted_t1,
          bathy_t =
            prediction_dt$bathy_t,
          interval_years =
            interval_years
        )

        output_dt <- data.table::copy(
          prediction_dt
        )

        output_dt[
          ,
          mean_predicted_bathy_t1 :=
            translated$pred_bathy_t1
        ]

        output_dt[
          ,
          uncertainty_sd_bathy_t1 :=
            sd_predicted_t1
        ]

        output_dt[
          ,
          mean_predicted_change :=
            translated$pred_delta_bathy
        ]

        predictor_range_path <- file.path(
          output_dir_train,
          source_model_tile,
          paste0(
            "training_predictor_ranges_",
            pair,
            ".csv"
          )
        )

        if (
          file.exists(
            predictor_range_path
          )
        ) {
          predictor_ranges <-
            data.table::fread(
              predictor_range_path
            )

          output_dt <-
            add_extrapolation_flags(
              output_dt,
              predictor_ranges
            )
        } else {
          output_dt[
            ,
            `:=`(
              predictor_extrapolation_count =
                NA_integer_,
              predictor_extrapolation_any =
                NA
            )
          ]
        }

        # -------------------------------------------------------
        # VERIFY CURRENT MODEL PREDICTORS SURVIVED INTO SUMMARY
        # -------------------------------------------------------
        
        # -------------------------------------------------------
        # VERIFY CURRENT MODEL + SOURCE-DATA SCHEMA SURVIVED
        # -------------------------------------------------------
        
        missing_output_predictors <- setdiff(
          predictors,
          names(output_dt)
        )
        
        missing_required_output_fields <- setdiff(
          required_local_summary_fields,
          names(output_dt)
        )
        
        
        if (
          length(missing_output_predictors) > 0L ||
          length(missing_required_output_fields) > 0L
        ) {
          
          schema_problems <- character()
          
          if (length(missing_output_predictors) > 0L) {
            schema_problems <- c(
              schema_problems,
              paste0(
                "lost model predictor(s): ",
                paste(
                  missing_output_predictors,
                  collapse = ", "
                )
              )
            )
          }
          
          if (length(missing_required_output_fields) > 0L) {
            schema_problems <- c(
              schema_problems,
              paste0(
                "lost required source-data field(s): ",
                paste(
                  missing_required_output_fields,
                  collapse = ", "
                )
              )
            )
          }
          
          stop(
            "FULL-EXTENT LOCAL schema failure for ",
            tile_id,
            " / ",
            pair,
            ": ",
            paste(
              schema_problems,
              collapse = " | "
            )
          )
        }
        
        fst::write_fst(
          output_dt,
          file.path(
            tile_dir_pred,
            paste0(
              "LOCAL_prediction_summary_",
              pair,
              ".fst"
            )
          )
        )

        raster_paths <- character()

        if (
          isTRUE(
            write_spatial_outputs
          )
        ) {
          raster_products <- c(
            bathy_t =
              "LOCAL_Pred_Start_Bathy_t",
            mean_predicted_bathy_t1 =
              "LOCAL_Pred_Mean_Bathy_t1",
            mean_predicted_change =
              "LOCAL_Pred_Mean_Change",
            uncertainty_sd_bathy_t1 =
              "LOCAL_Pred_SD_Bathy_t1",
            predictor_extrapolation_count =
              "LOCAL_Predictor_Extrapolation_Count"
          )

          for (
            value_column in
            names(raster_products)
          ) {
            raster_path <-
              write_prediction_tif(
                pred_dt = output_dt,
                value_col = value_column,
                out_tif = file.path(
                  tile_dir_pred,
                  paste0(
                    raster_products[[
                      value_column
                    ]],
                    "_",
                    pair,
                    ".tif"
                  )
                ),
                template_raster =
                  template_raster,
                tile_geometry =
                  tile_geometry,
                trim_output = FALSE,
                nodata_value = -9999
              )

            if (
              is.character(
                raster_path
              ) &&
              !is.na(
                raster_path
              )
            ) {
              raster_paths <- c(
                raster_paths,
                raster_path
              )
            }
          }
        }

        list(
          ok = TRUE,
          tile_id = tile_id,
          year_pair = pair,
          source_model_tile =
            source_model_tile,
          model_distance_map_units =
            assignment_row$model_distance_map_units,
          deployment =
            ifelse(
              identical(
                tile_id,
                source_model_tile
              ),
              "tile_native_local_rebuilt",
              "nearest_local_model"
            ),
          skipped_existing = FALSE,
          n_rasters_written =
            length(raster_paths),
          raster_paths =
            raster_paths
        )
      }, error = function(e) {
        list(
          ok = FALSE,
          tile_id =
            assignment_row$tile_id,
          year_pair = pair,
          source_model_tile =
            assignment_row$source_model_tile,
          reason =
            conditionMessage(e)
        )
      })
    }

    parallel::stopCluster(cl)

    all_results[[pair]] <- pair_results

    successful_tiles <- sum(
      vapply(
        pair_results,
        function(x) is.list(x) && isTRUE(x$ok),
        logical(1)
      )
    )

    failure_rows <- data.table::rbindlist(
      lapply(pair_results, function(x) {
        if (is.list(x) && !isTRUE(x$ok)) {
          data.table::data.table(
            tile_id = if (!is.null(x$tile_id)) x$tile_id else NA_character_,
            year_pair = pair,
            source_model_tile = if (!is.null(x$source_model_tile)) {
              x$source_model_tile
            } else {
              NA_character_
            },
            reason = if (!is.null(x$reason)) x$reason else "unknown_failure"
          )
        } else {
          NULL
        }
      }),
      fill = TRUE
    )

    failed_tiles <- nrow(failure_rows)

    if (failed_tiles > 0L) {
      failure_csv <- file.path(
        output_dir_pred,
        paste0("LOCAL_FULL_EXTENT_FAILURES_", pair, ".csv")
      )
      data.table::fwrite(failure_rows, failure_csv)

      reason_counts <- failure_rows[, .N, by = reason][order(-N)]
      log_line(
        log_file,
        "Failure reasons for ", pair, ": ",
        paste0(reason_counts$reason, "=", reason_counts$N, collapse = " | ")
      )
      log_line(log_file, "Failure detail CSV: ", failure_csv)
    }

    log_line(
      log_file,
      "Completed ", pair,
      " | successful tiles=", successful_tiles,
      " / ", length(pair_results),
      " | failed tiles=", failed_tiles
    )
  }

  saveRDS(
    all_results,
    file.path(
      output_dir_pred,
      "LOCAL_FULL_EXTENT_DEPLOYMENT_RESULTS.rds"
    )
  )

  invisible(
    all_results
  )
}


# ==============================================================================
# PART D. OPTIONAL GLOBAL TRAINING-SHARD PREPARATION
# ==============================================================================
#
# The local workflow can write shards while it runs. This avoids later differences
# in target construction, predictor naming, and numeric coercion.
#
# This retroactive function is provided for cases where local models have already
# run or shard writing was disabled. It reads each original training file, applies
# the same shared helpers, and writes one standardized FST shard per tile/pair.
# ==============================================================================

Build_Global_Training_Shards_Retroactively <- function(
    grid_sf,
    output_dir_train,
    output_dir_pred,
    year_pairs,
    master_predictor_list_path,
    tile_ids = "all",
    global_shard_dir = file.path(output_dir_train, "_GLOBAL_TRAINING_SHARDS"),
    require_prediction_compatible_predictors = TRUE
) {
  master_predictors <- unique(trimws(readLines(master_predictor_list_path)))
  master_predictors <- master_predictors[nzchar(master_predictors)]

  selected <- select_grid_tiles(
    grid_sf,
    tile_ids = tile_ids,
    allowed_sources = c("both", "training", "prediction")
  )
  selected_ids <- unique(as.character(selected$tile_id))

  manifest_all <- list()

  for (tile_id in selected_ids) {
    for (pair in year_pairs) {
      train_path <- find_tile_file(file.path(output_dir_train, tile_id), pair, "train")
      if (is.na(train_path)) next

      pred_path <- find_tile_file(file.path(output_dir_pred, tile_id), pair, "pred")
      train_dt <- fst::read_fst(train_path, as.data.table = TRUE)
      train_dt <- ensure_training_targets(train_dt)
      train_dt[
        !is.finite(interval_years) | interval_years <= 0,
        interval_years := pair_duration_years(pair)
      ]
      train_dt[, `:=`(tile_id = tile_id, pair_id = pair)]

      pred_cols <- NULL
      if (!is.na(pred_path)) {
        pred_cols <- names(fst::read_fst(pred_path, as.data.table = TRUE, columns = NULL))
      }

      resolved <- resolve_pair_predictors(
        pair,
        master_predictors,
        names(train_dt),
        pred_cols,
        require_prediction_compatibility = require_prediction_compatible_predictors && !is.null(pred_cols)
      )
      predictors <- resolved$all_predictors
      train_dt <- safe_numeric_cols(
        train_dt,
        unique(c(predictors, "X", "Y", "FID", "bathy_t", "bathy_t1", "delta_bathy", "delta_rate", "interval_years"))
      )

      keep <- is.finite(train_dt$X) & is.finite(train_dt$Y) &
        is.finite(train_dt$bathy_t) & is.finite(train_dt$bathy_t1) &
        is.finite(train_dt$delta_bathy) & is.finite(train_dt$interval_years) &
        train_dt$interval_years > 0
      shard <- train_dt[keep, unique(c(
        "tile_id", "pair_id", "X", "Y", "FID", "bathy_t", "bathy_t1",
        "delta_bathy", "delta_rate", "interval_years", predictors
      )), with = FALSE]

      pair_dir <- file.path(global_shard_dir, pair)
      dir.create(pair_dir, recursive = TRUE, showWarnings = FALSE)
      shard_path <- file.path(pair_dir, paste0(tile_id, "_", pair, "_global_training_shard.fst"))
      fst::write_fst(shard, shard_path)

      manifest_all[[length(manifest_all) + 1L]] <- data.table(
        tile_id = tile_id,
        year_pair = pair,
        shard_path = shard_path,
        n_rows = nrow(shard),
        predictors = paste(predictors, collapse = "|"),
        created = as.character(Sys.time())
      )
    }
  }

  manifest <- rbindlist(manifest_all, fill = TRUE)
  dir.create(global_shard_dir, recursive = TRUE, showWarnings = FALSE)
  fwrite(manifest, file.path(global_shard_dir, "GLOBAL_SHARD_MANIFEST.csv"))
  manifest
}

# ==============================================================================
# PART E. SECONDARY GLOBAL MODEL TRAINING BY YEAR PAIR
# ==============================================================================
#
# MEMORY-SAFE GLOBAL ARCHITECTURE
#   - Each tile/pair remains a separate FST shard.
#   - The global trainer reads shards one at a time.
#   - It takes a balanced sample from each tile, preventing large tiles from
#     overwhelming the global relationship.
#   - max_global_rows places a firm upper bound on the final in-memory table.
#   - The sampled global cache is saved so the exact training data are reproducible.
#
# This is a pooled global relationship. It does not search for the nearest tile at
# prediction time. A future regional/nearest-neighbour ensemble can be added as a
# third modelling strategy without changing the local or global workflows here.
# ==============================================================================

Train_Global_Models_ByYearPair <- function(
    output_dir_train,
    year_pairs,
    global_shard_dir = file.path(output_dir_train, "_GLOBAL_TRAINING_SHARDS"),
    tile_ids = "all",
    n_boot = 5,
    n_folds = 5,
    block_size_m = 200,
    max_rows_per_tile = 100000,
    max_global_rows = 2000000,
    use_weighted_loss = TRUE,
    weight_alpha = 1,
    weight_method = "capped_linear",
    weight_cap_quantile = 0.95,
    max_weight = 2,
    max_depth = 4,
    eta = 0.01,
    gamma = 0.5,
    lambda = 0,
    alpha = 0,
    global_seed = 12345,
    xgb_nthread = max(1, parallel::detectCores() - 1)
) {
  global_model_dir <- file.path(output_dir_train, "_GLOBAL_MODELS")
  dir.create(global_model_dir, recursive = TRUE, showWarnings = FALSE)
  log_file <- file.path(global_model_dir, "GLOBAL_TRAIN_LOG.txt")
  log_section(log_file, "GLOBAL TRAINING START")

  results <- list()

  for (pair in year_pairs) {
    pair_result <- tryCatch({
      log_section(log_file, paste0("GLOBAL YEAR PAIR: ", pair))
      pair_dir <- file.path(global_shard_dir, pair)
      shard_paths <- list.files(
        pair_dir,
        pattern = paste0("_", pair, "_global_training_shard\\.fst$"),
        full.names = TRUE
      )
      if (length(shard_paths) == 0L) stop("No global training shards found for ", pair)

      shard_tile_ids <- sub(paste0("_", pair, "_global_training_shard\\.fst$"), "", basename(shard_paths))
      if (!(length(tile_ids) == 1L && identical(tile_ids, "all"))) {
        keep_shards <- shard_tile_ids %in% tile_ids
        shard_paths <- shard_paths[keep_shards]
        shard_tile_ids <- shard_tile_ids[keep_shards]
      }
      if (length(shard_paths) == 0L) stop("No shards remain after tile selection for ", pair)

      # Determine a predictor schema common to every selected shard.
      shard_columns <- lapply(shard_paths, function(p) fst::metadata_fst(p)$columnNames)
      common_columns <- Reduce(intersect, shard_columns)
      non_predictors <- c(
        "tile_id", "pair_id", "X", "Y", "FID", "bathy_t1",
        "delta_bathy", "delta_rate", "interval_years"
      )
      predictors <- setdiff(common_columns, non_predictors)
      if (!("bathy_t" %in% predictors)) stop("bathy_t is missing from the common global schema.")

      # Allocate a balanced row allowance per tile and read one shard at a time.
      rows_per_tile_budget <- min(
        max_rows_per_tile,
        max(1L, floor(max_global_rows / length(shard_paths)))
      )

      sampled_list <- vector("list", length(shard_paths))
      sampling_manifest <- vector("list", length(shard_paths))

      for (i in seq_along(shard_paths)) {
        p <- shard_paths[[i]]
        tid <- shard_tile_ids[[i]]
        dt <- fst::read_fst(
          p,
          as.data.table = TRUE,
          columns = unique(c(
            "tile_id", "pair_id", "X", "Y", "FID", "bathy_t", "bathy_t1",
            "delta_bathy", "delta_rate", "interval_years", predictors
          ))
        )
        setDT(dt)

        n_before <- nrow(dt)
        if (n_before > rows_per_tile_budget) {
          set.seed(make_seed(global_seed, "global_sampling", pair, tid))
          dt <- dt[sample.int(n_before, rows_per_tile_budget)]
        }

        sampled_list[[i]] <- dt
        sampling_manifest[[i]] <- data.table(
          tile_id = tid,
          year_pair = pair,
          rows_available = n_before,
          rows_sampled = nrow(dt),
          shard_path = p
        )
        rm(dt)
      }

      global_dt <- rbindlist(sampled_list, fill = FALSE, use.names = TRUE)
      rm(sampled_list)
      gc(verbose = FALSE)

      # Predictor NAs remain in the global table and are handled by XGBoost.
      essential_keep <- is.finite(global_dt$X) & is.finite(global_dt$Y) &
        is.finite(global_dt$bathy_t) & is.finite(global_dt$bathy_t1) &
        is.finite(global_dt$delta_bathy) & is.finite(global_dt$interval_years) &
        global_dt$interval_years > 0
      global_dt <- global_dt[essential_keep]

      global_cache_path <- file.path(global_model_dir, paste0("GLOBAL_training_sample_", pair, ".fst"))
      fst::write_fst(global_dt, global_cache_path)
      fwrite(
        rbindlist(sampling_manifest),
        file.path(global_model_dir, paste0("GLOBAL_sampling_manifest_", pair, ".csv"))
      )
      fwrite(
        summarize_predictor_missingness(global_dt, predictors),
        file.path(global_model_dir, paste0("GLOBAL_predictor_missingness_", pair, ".csv"))
      )

      sample_weights <- if (use_weighted_loss) {
        calculate_sample_weights(
          global_dt$delta_bathy,
          alpha = weight_alpha,
          method = weight_method,
          cap_quantile = weight_cap_quantile,
          max_weight = max_weight
        )
      } else {
        rep(1, nrow(global_dt))
      }

      # -------------------------------------------------------
      # D1. GLOBAL SPATIAL CROSS VALIDATION
      # -------------------------------------------------------
      global_sf <- st_as_sf(global_dt, coords = c("X", "Y"), remove = FALSE, crs = NA)
      best_iteration <- 100L
      cv_metrics_list <- list()

      set.seed(make_seed(global_seed, "global_cv", pair))
      scv <- tryCatch(
        blockCV::cv_spatial(
          x = global_sf,
          size = block_size_m,
          k = n_folds,
          iteration = 200,
          progress = FALSE
        ),
        error = function(e) NULL
      )

      if (!is.null(scv) && length(scv$folds_list) >= 2L) {
        best_nrounds <- numeric()
        for (k in seq_along(scv$folds_list)) {
          train_idx <- unlist(scv$folds_list[[k]][[1]])
          test_idx <- unlist(scv$folds_list[[k]][[2]])
          train_idx <- train_idx[train_idx <= nrow(global_dt)]
          test_idx <- test_idx[test_idx <= nrow(global_dt)]
          if (length(train_idx) < 100L || length(test_idx) < 20L) next

          dtrain <- xgb.DMatrix(
            as.matrix(global_dt[train_idx, ..predictors]),
            label = global_dt$bathy_t1[train_idx],
            weight = sample_weights[train_idx],
            missing = NA
          )
          dtest <- xgb.DMatrix(
            as.matrix(global_dt[test_idx, ..predictors]),
            label = global_dt$bathy_t1[test_idx],
            missing = NA
          )

          model_cv <- xgb.train(
            params = list(
              objective = "reg:squarederror",
              eval_metric = c("rmse", "mae"),
              max_depth = max_depth,
              eta = eta,
              gamma = gamma,
              lambda = lambda,
              alpha = alpha,
              subsample = 0.7,
              colsample_bytree = 0.8
            ),
            data = dtrain,
            nrounds = 1500,
            watchlist = list(train = dtrain, test = dtest),
            early_stopping_rounds = 30,
            nthread = 1,
            verbose = 0
          )

          if (!is.null(model_cv$best_iteration)) {
            best_nrounds <- c(best_nrounds, model_cv$best_iteration)
          }
          pred_cv <- predict(model_cv, dtest)
          mm <- calculate_common_metrics(
            global_dt$bathy_t1[test_idx],
            pred_cv,
            global_dt$bathy_t[test_idx],
            global_dt$interval_years[test_idx]
          )
          mm[, `:=`(year_pair = pair, fold = k, best_iteration = model_cv$best_iteration)]
          cv_metrics_list[[length(cv_metrics_list) + 1L]] <- mm
        }

        if (length(best_nrounds)) {
          best_iteration <- max(50L, round(mean(best_nrounds, na.rm = TRUE)))
        }
      }

      fwrite(
        rbindlist(cv_metrics_list, fill = TRUE),
        file.path(global_model_dir, paste0("GLOBAL_spatial_cv_metrics_", pair, ".csv"))
      )

      # -------------------------------------------------------
      # D2. GLOBAL BOOTSTRAP MODEL ENSEMBLE
      # -------------------------------------------------------
      boot_dir <- file.path(global_model_dir, pair, "bootstrap_models")
      dir.create(boot_dir, recursive = TRUE, showWarnings = FALSE)

      model_paths <- character(n_boot)
      importance_list <- vector("list", n_boot)

      for (b in seq_len(n_boot)) {
        set.seed(make_seed(global_seed, "global_boot", pair, b))

        # Balanced tile-aware bootstrap: sample tiles with replacement, then sample
        # rows within the selected tiles. This better respects spatial grouping than
        # a simple pixel-level bootstrap.
        available_tiles <- unique(global_dt$tile_id)
        sampled_tiles <- sample(available_tiles, length(available_tiles), replace = TRUE)
        boot_parts <- lapply(sampled_tiles, function(tid) {
          tile_rows <- which(global_dt$tile_id == tid)
          global_dt[sample(tile_rows, length(tile_rows), replace = TRUE)]
        })
        boot_dt <- rbindlist(boot_parts, fill = FALSE)

        boot_weights <- if (use_weighted_loss) {
          calculate_sample_weights(
            boot_dt$delta_bathy,
            alpha = weight_alpha,
            method = weight_method,
            cap_quantile = weight_cap_quantile,
            max_weight = max_weight
          )
        } else {
          rep(1, nrow(boot_dt))
        }

        dtrain_boot <- xgb.DMatrix(
          as.matrix(boot_dt[, ..predictors]),
          label = boot_dt$bathy_t1,
          weight = boot_weights,
          missing = NA
        )

        model <- xgb.train(
          params = list(
            objective = "reg:squarederror",
            eval_metric = c("rmse", "mae"),
            max_depth = max_depth,
            eta = eta,
            gamma = gamma,
            lambda = lambda,
            alpha = alpha,
            subsample = 0.7,
            colsample_bytree = 0.8
          ),
          data = dtrain_boot,
          nrounds = best_iteration,
          nthread = xgb_nthread,
          verbose = 0
        )

        model_paths[b] <- file.path(boot_dir, paste0("GLOBAL_model_boot_", sprintf("%03d", b), "_", pair, ".rds"))
        saveRDS(model, model_paths[b])

        imp <- xgb.importance(model = model)
        if (nrow(imp)) {
          importance_list[[b]] <- as.data.table(imp)[, bootstrap_iteration := b]
        }
        rm(boot_dt, dtrain_boot, model)
        gc(verbose = FALSE)
      }

      predictor_ranges <- make_predictor_range_table(global_dt, predictors)
      fwrite(predictor_ranges, file.path(global_model_dir, paste0("GLOBAL_predictor_ranges_", pair, ".csv")))
      fwrite(rbindlist(importance_list, fill = TRUE), file.path(global_model_dir, paste0("GLOBAL_importance_", pair, ".csv")))

      global_metadata <- list(
        model_formulation = "Model A: pooled global bathy_t1 model",
        year_pair = pair,
        predictors = predictors,
        best_iteration = best_iteration,
        n_boot = n_boot,
        model_paths = model_paths,
        global_training_cache = global_cache_path,
        n_global_rows = nrow(global_dt),
        n_tiles = uniqueN(global_dt$tile_id),
        max_rows_per_tile = max_rows_per_tile,
        max_global_rows = max_global_rows,
        weighting = list(
          enabled = use_weighted_loss,
          method = weight_method,
          alpha = weight_alpha,
          cap_quantile = weight_cap_quantile,
          max_weight = max_weight
        ),
        created = Sys.time()
      )
      metadata_path <- file.path(global_model_dir, paste0("GLOBAL_model_metadata_", pair, ".rds"))
      saveRDS(global_metadata, metadata_path)

      log_line(log_file, "SUCCESS: global model ", pair,
               " | rows=", nrow(global_dt),
               " | tiles=", uniqueN(global_dt$tile_id),
               " | predictors=", length(predictors),
               " | boot models=", n_boot)

      list(ok = TRUE, year_pair = pair, metadata_path = metadata_path)
    }, error = function(e) {
      log_line(log_file, "ERROR ", pair, ": ", conditionMessage(e))
      list(ok = FALSE, year_pair = pair, reason = conditionMessage(e))
    })

    results[[pair]] <- pair_result
  }

  saveRDS(results, file.path(global_model_dir, "GLOBAL_TRAIN_RESULTS.rds"))
  log_section(log_file, "GLOBAL TRAINING END")
  results
}

# ==============================================================================
# PART F. SECONDARY GLOBAL PREDICTION ACROSS SELECTED TILES
# ==============================================================================

Predict_With_Global_Models <- function(
    grid_sf,
    output_dir_pred,
    output_dir_train,
    year_pairs,
    template_raster,
    tile_ids = "all",
    standard_horizon_years = 2,

    # Explicit spatial-output controls
    write_spatial_outputs = TRUE,
    write_global_start_bathy = TRUE,

    n_workers = max(1, parallel::detectCores() - 1)
) {
  global_model_dir <- file.path(output_dir_train, "_GLOBAL_MODELS")
  if (!dir.exists(global_model_dir)) stop("Global model directory not found: ", global_model_dir)

  selected <- select_grid_tiles(
    grid_sf,
    tile_ids = tile_ids,
    allowed_sources = c("both", "prediction")
  )
  pred_tile_ids <- unique(as.character(selected$tile_id))

  log_file <- file.path(output_dir_pred, "_GLOBAL_PRED_LOGS", "GLOBAL_PREDICT_LOG.txt")
  log_section(log_file, "GLOBAL PREDICTION START")
  log_line(log_file, "Tiles selected: ", length(pred_tile_ids))

  cl <- parallel::makeCluster(n_workers)
  doParallel::registerDoParallel(cl)
  on.exit({
    try(parallel::stopCluster(cl), silent = TRUE)
    try(closeAllConnections(), silent = TRUE)
  }, add = TRUE)

  all_results <- list()

  for (pair in year_pairs) {
    metadata_path <- file.path(global_model_dir, paste0("GLOBAL_model_metadata_", pair, ".rds"))
    if (!file.exists(metadata_path)) {
      log_line(log_file, "SKIP ", pair, ": metadata not found.")
      next
    }

    metadata <- readRDS(metadata_path)
    predictors <- metadata$predictors
    model_paths <- metadata$model_paths
    interval_years <- pair_duration_years(pair)
    predictor_ranges <- fread(
      file.path(global_model_dir, paste0("GLOBAL_predictor_ranges_", pair, ".csv"))
    )

    pair_results <- foreach(
      tile_id = pred_tile_ids,
      .packages = c(
        "data.table",
        "fst",
        "xgboost",
        "raster",
        "sp",
        "sf"
      ),
      .export = c(
        "find_tile_file", "safe_numeric_cols", "translate_model_a_predictions",
        "add_extrapolation_flags", "write_prediction_tif",
        "write_spatial_outputs", "write_global_start_bathy"
      ),
      .errorhandling = "pass"
    ) %dopar% {
      tryCatch({
        tile_geometry <- selected[
          as.character(selected$tile_id) == tile_id,
          ,
          drop = FALSE
        ]

        if (nrow(tile_geometry) != 1L) {
          stop(
            "Expected exactly one prediction-grid geometry for tile ",
            tile_id,
            "; found ",
            nrow(tile_geometry),
            "."
          )
        }

        tile_dir <- file.path(output_dir_pred, tile_id)
        pred_path <- find_tile_file(tile_dir, pair, "pred")
        if (is.na(pred_path)) return(list(ok = FALSE, tile_id = tile_id, reason = "missing_prediction_file"))

        pred_dt <- fst::read_fst(pred_path, as.data.table = TRUE)
        setDT(pred_dt)
        missing_predictors <- setdiff(predictors, names(pred_dt))
        if (length(missing_predictors)) {
          stop("Missing global predictors: ", paste(missing_predictors, collapse = ", "))
        }

        pred_dt <- safe_numeric_cols(
          pred_dt,
          unique(c(predictors, "X", "Y", "FID", "bathy_t"))
        )
        pred_dt[, `:=`(
          tile_id = tile_id,
          pair_id = pair,
          interval_years = interval_years
        )]

        dp <- xgb.DMatrix(as.matrix(pred_dt[, ..predictors]), missing = NA)
        pred_boot <- matrix(NA_real_, nrow(pred_dt), length(model_paths))

        for (b in seq_along(model_paths)) {
          model <- readRDS(model_paths[b])
          pred_boot[, b] <- predict(model, dp)
          rm(model)
        }

        mean_t1 <- rowMeans(pred_boot, na.rm = TRUE)
        mean_t1[rowSums(is.finite(pred_boot)) == 0L] <- NA_real_
        sd_t1 <- apply(pred_boot, 1, sd, na.rm = TRUE)
        if (ncol(pred_boot) == 1L) sd_t1[is.finite(mean_t1)] <- 0

        translated <- translate_model_a_predictions(
          mean_t1,
          pred_dt$bathy_t,
          interval_years,
          standard_horizon_years
        )

        out_dt <- copy(pred_dt)
        out_dt[, mean_predicted_bathy_t1 := translated$pred_bathy_t1]
        out_dt[, uncertainty_sd_bathy_t1 := sd_t1]
        out_dt[, mean_predicted_change := translated$pred_delta_bathy]
        out_dt[, mean_predicted_rate := translated$pred_delta_rate]
        out_dt[, mean_predicted_standard_change := translated$pred_delta_standard]
        out_dt[, mean_predicted_standard_bathy := translated$pred_bathy_standard]
        out_dt <- add_extrapolation_flags(out_dt, predictor_ranges)

        fst::write_fst(out_dt, file.path(tile_dir, paste0("GLOBAL_prediction_summary_", pair, ".fst")))

        # -------------------------------------------------------
        # 5B-1. WRITE GLOBAL SPATIAL OUTPUTS
        # -------------------------------------------------------
        # The earlier version wrote the FST summary successfully even when a
        # raster failed. This block makes raster creation explicit, validates
        # every file, and returns the paths/reasons to the parent process.
        raster_paths <- character()
        raster_failures <- character()

        if (isTRUE(write_spatial_outputs)) {
          raster_products <- c(
            mean_predicted_bathy_t1 =
              "GLOBAL_Pred_Mean_Bathy_t1",
            uncertainty_sd_bathy_t1 =
              "GLOBAL_Pred_SD_Bathy_t1",
            mean_predicted_change =
              "GLOBAL_Pred_Mean_Change",
            predictor_extrapolation_count =
              "GLOBAL_Predictor_Extrapolation_Count"
          )

          # Starting bathymetry is not model-derived, but writing it with the
          # GLOBAL prefix makes local/global VRT comparisons straightforward.
          if (isTRUE(write_global_start_bathy)) {
            raster_products <- c(
              raster_products,
              bathy_t = "GLOBAL_Pred_Start_Bathy_t"
            )
          }

          for (value_col in names(raster_products)) {
            output_tif <- file.path(
              tile_dir,
              paste0(
                raster_products[[value_col]],
                "_",
                pair,
                ".tif"
              )
            )

            raster_result <- tryCatch(
              write_prediction_tif(
                pred_dt = out_dt,
                value_col = value_col,
                out_tif = output_tif,
                template_raster = template_raster,
                tile_geometry = tile_geometry,
                trim_output = FALSE,
                nodata_value = -9999
              ),
              error = function(e) {
                structure(
                  NA_character_,
                  raster_error = conditionMessage(e)
                )
              }
            )

            if (
              is.character(raster_result) &&
              length(raster_result) == 1L &&
              !is.na(raster_result) &&
              file.exists(raster_result)
            ) {
              raster_paths <- c(
                raster_paths,
                raster_result
              )
            } else {
              failure_reason <- attr(
                raster_result,
                "raster_error"
              )

              if (
                is.null(failure_reason) ||
                !nzchar(failure_reason)
              ) {
                failure_reason <- paste0(
                  "No finite cells or output file not created: ",
                  basename(output_tif)
                )
              }

              raster_failures <- c(
                raster_failures,
                paste0(
                  value_col,
                  " => ",
                  failure_reason
                )
              )
            }
          }
        }

        list(
          ok = TRUE,
          tile_id = tile_id,
          year_pair = pair,
          prediction_summary_fst = file.path(
            tile_dir,
            paste0(
              "GLOBAL_prediction_summary_",
              pair,
              ".fst"
            )
          ),
          n_rasters_written = length(raster_paths),
          raster_paths = raster_paths,
          raster_failures = raster_failures
        )
      }, error = function(e) {
        list(ok = FALSE, tile_id = tile_id, year_pair = pair, reason = conditionMessage(e))
      })
    }

    all_results[[pair]] <- pair_results

    n_success <- sum(vapply(
      pair_results,
      function(x) is.list(x) && isTRUE(x$ok),
      logical(1)
    ))

    n_rasters <- sum(vapply(
      pair_results,
      function(x) {
        if (
          is.list(x) &&
          !is.null(x$n_rasters_written)
        ) {
          as.numeric(x$n_rasters_written)
        } else {
          0
        }
      },
      numeric(1)
    ))

    n_raster_failures <- sum(vapply(
      pair_results,
      function(x) {
        if (
          is.list(x) &&
          !is.null(x$raster_failures)
        ) {
          length(x$raster_failures)
        } else {
          0L
        }
      },
      integer(1)
    ))

    log_line(
      log_file,
      "Completed global prediction pair ", pair,
      " | successful tiles=", n_success,
      " | rasters written=", n_rasters,
      " | raster failures=", n_raster_failures
    )
  }

  saveRDS(all_results, file.path(output_dir_pred, "_GLOBAL_PRED_LOGS", "GLOBAL_PREDICT_RESULTS.rds"))
  log_section(log_file, "GLOBAL PREDICTION END")
  all_results
}

# ==============================================================================
# PART F. OPTIONAL TEMPORAL HOLD-ONE-PAIR-OUT DIAGNOSTIC
# ==============================================================================
# This diagnostic uses only predictor names common across all held-in and held-out
# pairs. It is not used to fit the production pair-specific models.

Run_Temporal_Holdout_Diagnostic <- function(
    output_dir_train,
    year_pairs,
    global_shard_dir = file.path(output_dir_train, "_GLOBAL_TRAINING_SHARDS"),
    max_rows_per_pair = 250000,
    nrounds = 250,
    global_seed = 12345
) {
  out_dir <- file.path(output_dir_train, "_TEMPORAL_VALIDATION")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  pair_data <- list()
  for (pair in year_pairs) {
    paths <- list.files(
      file.path(global_shard_dir, pair),
      pattern = "global_training_shard\\.fst$",
      full.names = TRUE
    )
    if (!length(paths)) next

    per_path <- max(1L, floor(max_rows_per_pair / length(paths)))
    parts <- lapply(seq_along(paths), function(i) {
      dt <- fst::read_fst(paths[i], as.data.table = TRUE)
      if (nrow(dt) > per_path) {
        set.seed(make_seed(global_seed, "temporal_pair_sample", pair, i))
        dt <- dt[sample.int(nrow(dt), per_path)]
      }
      dt[, source_pair := pair]
      dt
    })
    pair_data[[pair]] <- rbindlist(parts, fill = TRUE)
  }

  metrics_all <- list()

  for (held_out_pair in names(pair_data)) {
    train_pairs <- setdiff(names(pair_data), held_out_pair)
    train_dt <- rbindlist(pair_data[train_pairs], fill = TRUE)
    test_dt <- pair_data[[held_out_pair]]

    common_cols <- Reduce(intersect, lapply(c(pair_data[train_pairs], list(test_dt)), names))
    predictors <- setdiff(
      common_cols,
      c("tile_id", "pair_id", "source_pair", "X", "Y", "FID", "bathy_t1",
        "delta_bathy", "delta_rate", "interval_years", "year_t", "year_t1")
    )
    if (!("bathy_t" %in% predictors)) next

    dtrain <- xgb.DMatrix(as.matrix(train_dt[, ..predictors]), label = train_dt$bathy_t1, missing = NA)
    dtest <- xgb.DMatrix(as.matrix(test_dt[, ..predictors]), label = test_dt$bathy_t1, missing = NA)

    model <- xgb.train(
      params = list(
        objective = "reg:squarederror",
        max_depth = 4,
        eta = 0.03,
        subsample = 0.7,
        colsample_bytree = 0.8
      ),
      data = dtrain,
      nrounds = nrounds,
      verbose = 0
    )

    pred <- predict(model, dtest)
    mm <- calculate_common_metrics(
      test_dt$bathy_t1,
      pred,
      test_dt$bathy_t,
      test_dt$interval_years
    )
    mm[, `:=`(
      held_out_pair = held_out_pair,
      training_pairs = paste(train_pairs, collapse = "|"),
      n_predictors = length(predictors)
    )]
    metrics_all[[held_out_pair]] <- mm
  }

  metrics <- rbindlist(metrics_all, fill = TRUE)
  fwrite(metrics, file.path(out_dir, "TEMPORAL_HOLDOUT_METRICS.csv"))
  metrics
}

# ==============================================================================
# PART G. OPTIONAL LOCAL + GLOBAL VRT CREATION
# ==============================================================================

# ==============================================================================
# SECTION 2C — CORRECTED LOCAL + GLOBAL VRT BUILDER
# Fixes ArcGIS displaying Float32 NoData (-3.4e38) as a real raster value
# ==============================================================================

#
# Output layout:
#   <output_dir_train>/VRT/          LOCAL training mosaics
#   <output_dir_pred>/VRT/           LOCAL prediction mosaics
#   <output_dir_pred>/VRT/GLOBAL/    GLOBAL prediction mosaics
#
# Per-year mosaics and optional all-year multiband VRTs are created for each
# requested product. The inherited source NoData metadata and statistics checks
# are retained from the corrected local-only builder.

# Detect source NoData.
detect_source_nodata <- function(raster_file) {
  if (!file.exists(raster_file)) return(NA_real_)
  r <- terra::rast(raster_file)
  out <- tryCatch(terra::NAflag(r), error = function(e) NA_real_)
  suppressWarnings(as.numeric(out)[1])
}

# Build one VRT without forcing NaN as NoData.
build_one_vrt_fixed <- function(
    input_files,
    output_vrt,
    separate_bands = FALSE,
    overwrite = TRUE,
    nodata_mode = c("inherit", "detect_and_set"),
    build_statistics = TRUE
) {
  nodata_mode <- match.arg(nodata_mode)

  input_files <- unique(normalizePath(
    input_files[file.exists(input_files)],
    winslash = "/",
    mustWork = TRUE
  ))

  if (!length(input_files)) {
    warning("No input rasters found for: ", output_vrt)
    return(NA_character_)
  }

  dir.create(dirname(output_vrt), recursive = TRUE, showWarnings = FALSE)

  if (overwrite) {
    unlink(c(output_vrt, paste0(output_vrt, ".aux.xml")), force = TRUE)
  } else if (file.exists(output_vrt)) {
    return(output_vrt)
  }

  vrt_options <- c("-resolution", "highest")
  if (separate_bands) vrt_options <- c("-separate", vrt_options)

  if (nodata_mode == "detect_and_set") {
    source_nodata <- detect_source_nodata(input_files[[1]])
    if (is.finite(source_nodata)) {
      nodata_text <- format(
        source_nodata,
        scientific = TRUE,
        digits = 17,
        trim = TRUE
      )
      vrt_options <- c(
        vrt_options,
        "-srcnodata", nodata_text,
        "-vrtnodata", nodata_text
      )
      message("Using detected source NoData value: ", nodata_text)
    }
  }

  sf::gdal_utils(
    util = "buildvrt",
    source = input_files,
    destination = normalizePath(output_vrt, winslash = "/", mustWork = FALSE),
    options = vrt_options,
    quiet = TRUE
  )

  if (!file.exists(output_vrt)) stop("VRT was not created: ", output_vrt)

  vrt_raster <- terra::rast(output_vrt)
  range_table <- terra::global(vrt_raster, c("min", "max"), na.rm = TRUE)
  finite_min <- suppressWarnings(min(range_table$min, na.rm = TRUE))
  finite_max <- suppressWarnings(max(range_table$max, na.rm = TRUE))

  message(
    "Created VRT: ", output_vrt,
    " | sources=", length(input_files),
    " | finite range=", signif(finite_min, 7),
    " to ", signif(finite_max, 7)
  )

  if (build_statistics) {
    try(
      sf::gdal_utils(
        util = "info",
        source = output_vrt,
        options = c("-stats"),
        quiet = TRUE
      ),
      silent = TRUE
    )
  }

  output_vrt
}

find_product_tiles_fixed <- function(parent_dir, product_prefix, year_pair) {
  expected_filename <- paste0(product_prefix, "_", year_pair, ".tif")
  all_tifs <- list.files(
    parent_dir,
    pattern = "\\.tif$",
    recursive = TRUE,
    full.names = TRUE,
    ignore.case = TRUE
  )
  hits <- all_tifs[basename(all_tifs) == expected_filename]
  normalized <- gsub("\\\\", "/", hits)
  hits[
    !grepl("/VRT/", normalized, fixed = TRUE) &
      !grepl("/_GLOBAL_", normalized, fixed = TRUE) &
      !grepl("/_TEMPORAL_", normalized, fixed = TRUE)
  ]
}

Build_Local_and_Global_Output_VRTs <- function(
    output_dir_train,
    output_dir_pred,
    year_pairs,
    training_products = c(
      "LOCAL_Train_Mean_Predicted_Bathy_t1",
      "LOCAL_Train_Mean_Predicted_Change",
      "LOCAL_Train_Surveyed_Change",
      "LOCAL_Train_SD_Bathy_t1"
    ),
    prediction_products = c(
      "LOCAL_Pred_Mean_Bathy_t1",
      "LOCAL_Pred_Mean_Change",
      "LOCAL_Pred_Start_Bathy_t",
      "LOCAL_Pred_SD_Bathy_t1",
      "LOCAL_Predictor_Extrapolation_Count"
    ),
    global_prediction_products = c(
      "GLOBAL_Pred_Mean_Bathy_t1",
      "GLOBAL_Pred_Mean_Change",
      "GLOBAL_Pred_Start_Bathy_t",
      "GLOBAL_Pred_SD_Bathy_t1",
      "GLOBAL_Predictor_Extrapolation_Count"
    ),
    build_all_year_band_vrts = TRUE,
    nodata_mode = "detect_and_set",
    build_statistics = TRUE,
    overwrite = TRUE
) {
  train_vrt_dir <- file.path(output_dir_train, "VRT")
  pred_vrt_dir <- file.path(output_dir_pred, "VRT")
  global_vrt_dir <- file.path(pred_vrt_dir, "GLOBAL")

  dir.create(train_vrt_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(pred_vrt_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(global_vrt_dir, recursive = TRUE, showWarnings = FALSE)

  process_group <- function(parent_dir, vrt_dir, products, group_name) {
    result <- list()

    for (product in products) {
      pair_vrts <- character()
      manifest_rows <- list()

      for (i in seq_along(year_pairs)) {
        pair <- year_pairs[[i]]
        source_tifs <- find_product_tiles_fixed(parent_dir, product, pair)

        if (!length(source_tifs)) {
          warning("No source GeoTIFFs found for ", product, " / ", pair)
          next
        }

        pair_vrt <- file.path(vrt_dir, paste0(product, "_", pair, "_MOSAIC.vrt"))

        built <- build_one_vrt_fixed(
          input_files = source_tifs,
          output_vrt = pair_vrt,
          separate_bands = FALSE,
          overwrite = overwrite,
          nodata_mode = nodata_mode,
          build_statistics = build_statistics
        )

        if (!is.na(built)) {
          pair_vrts <- c(pair_vrts, built)
          manifest_rows[[length(manifest_rows) + 1L]] <- data.table::data.table(
            group = group_name,
            product = product,
            year_pair = pair,
            pair_order = i,
            n_tile_rasters = length(source_tifs),
            pair_vrt_path = built
          )
        }
      }

      manifest <- data.table::rbindlist(manifest_rows, fill = TRUE)
      all_year_vrt <- NA_character_

      if (build_all_year_band_vrts && nrow(manifest)) {
        data.table::setorder(manifest, pair_order)
        all_year_vrt <- file.path(vrt_dir, paste0(product, "_ALL_YEAR_PAIRS_BANDS.vrt"))

        build_one_vrt_fixed(
          input_files = manifest$pair_vrt_path,
          output_vrt = all_year_vrt,
          separate_bands = TRUE,
          overwrite = overwrite,
          nodata_mode = nodata_mode,
          build_statistics = build_statistics
        )

        manifest[, vrt_band := seq_len(.N)]
        data.table::fwrite(
          manifest[, .(
            vrt_band,
            year_pair,
            product,
            n_tile_rasters,
            pair_vrt_path
          )],
          file.path(vrt_dir, paste0(product, "_ALL_YEAR_PAIRS_BAND_MANIFEST.csv"))
        )
      }

      result[[product]] <- list(
        pair_vrts = pair_vrts,
        all_year_vrt = all_year_vrt,
        manifest = manifest
      )
    }

    result
  }

  results <- list(
    local_training = process_group(
      output_dir_train,
      train_vrt_dir,
      training_products,
      "local_training"
    ),
    local_prediction = process_group(
      output_dir_pred,
      pred_vrt_dir,
      prediction_products,
      "local_prediction"
    ),
    global_prediction = process_group(
      output_dir_pred,
      global_vrt_dir,
      global_prediction_products,
      "global_prediction"
    )
  )

  saveRDS(
    results,
    file.path(
      train_vrt_dir,
      "LOCAL_AND_GLOBAL_VRT_BUILD_RESULTS.rds"
    )
  )

  message("\nLocal and global VRT creation complete.")
  message("Local training VRTs: ", train_vrt_dir)
  message("Local prediction VRTs: ", pred_vrt_dir)
  message("Global prediction VRTs: ", global_vrt_dir)

  invisible(results)
}

# ==============================================================================
# PART H. EXAMPLE FUNCTION CALLS
# ==============================================================================
# Edit paths below before running. Keep execution calls commented until ready.

 # --- 1. Define paths and parameters ---
master_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/master_grid_canonical.gpkg"
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles"
output_dir_pred  <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"
master_predictor_file <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles/boruta_master_predictor_list_new.txt" ##### MAKE SURE THIS IS NEW
template_raster_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif"

grid_sf <- sf::st_read(master_grid_gpkg)
template_raster <- raster::raster(template_raster_path)
 # year_pairs <- c("2004_2006")
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")

# --- 2A. Run ONE TEST TILE through the primary local workflow ---
# local_test_results <- Run_Local_Model_Training_And_Prediction(
#   grid_sf = grid_sf,
#   output_dir_train = output_dir_train,
#   output_dir_pred = output_dir_pred,
#   year_pairs = year_pairs,
#   master_predictor_list_path = master_predictor_file,
#   template_raster = template_raster,
#   tile_ids = "BH4S556X_3",
#   block_size_m = 200,
#   n_boot = 1,
#   n_folds = 5,
#   use_weighted_loss = TRUE,
#   weight_method = "capped_linear",
#   weight_alpha = 0.5,
#   weight_cap_quantile = 0.95,
#   max_weight = 2,
#   write_global_training_shards = TRUE,
#   n_workers = 1
# )

# --- 2B. Run ALL TILES through the primary local workflow ---
Sys.time()
local_all_results <- Run_Local_Model_Training_And_Prediction(
  grid_sf = grid_sf,
  output_dir_train = output_dir_train,
  output_dir_pred = output_dir_pred,
  year_pairs = year_pairs,
  master_predictor_list_path = master_predictor_file,
  template_raster = template_raster,
  tile_ids = "all",
  block_size_m = 200,
  n_boot = 5,
  n_folds = 5,
  use_weighted_loss = TRUE,
  write_global_training_shards = TRUE,
  # n_workers = max(1, parallel::detectCores() - 1)
  n_workers = 4
)
Sys.time()
# Full-extent LOCAL deployment is now integrated into Step 2B by default:
#
#   deploy_local_full_extent = TRUE
#   overwrite_existing_local = FALSE
#   write_full_extent_spatial_outputs = TRUE
#
# Set deploy_local_full_extent = FALSE only when you deliberately want to run
# tile-native model training without filling prediction-only tiles.
#
# --- 2C. Optionally build LOCAL and GLOBAL VRT mosaics ---
# Run after LOCAL outputs exist; rerun after GLOBAL outputs are created to add
# or refresh GLOBAL mosaics. Hybrid products are intentionally excluded.
#
vrt_results <- Build_Local_and_Global_Output_VRTs(
  output_dir_train = output_dir_train,
  output_dir_pred = output_dir_pred,
  year_pairs = year_pairs,
  build_all_year_band_vrts = TRUE,
  nodata_mode = "inherit",
  build_statistics = TRUE,
  overwrite = TRUE
)

# --- 3. Build shards retroactively only when needed ---
# retro_manifest <- Build_Global_Training_Shards_Retroactively(
#   grid_sf = grid_sf,
#   output_dir_train = output_dir_train,
#   output_dir_pred = output_dir_pred,
#   year_pairs = year_pairs,
#   master_predictor_list_path = master_predictor_file,
#   tile_ids = "all"
# )

# --- 4A. Train GLOBAL models using ONE OR SEVERAL tiles for testing ---
# global_test_models <- Train_Global_Models_ByYearPair(
#   output_dir_train = output_dir_train,
#   year_pairs = year_pairs,
#   tile_ids = c("BH4S556X_3", "BH4S557X_3"),
#   n_boot = 2,
#   max_rows_per_tile = 50000,
#   max_global_rows = 200000,
#   xgb_nthread = 2
# )

# --- 4B. Train GLOBAL models using all available training shards ---
# global_models <- Train_Global_Models_ByYearPair(
#   output_dir_train = output_dir_train,
#   year_pairs = year_pairs,
#   tile_ids = "all",
#   n_boot = 5,
#   max_rows_per_tile = 100000,
#   max_global_rows = 2000000
# )

# --- 5A. Apply GLOBAL models to ONE TEST prediction tile ---
# global_test_predictions <- Predict_With_Global_Models(
#   grid_sf = grid_sf,
#   output_dir_pred = output_dir_pred,
#   output_dir_train = output_dir_train,
#   year_pairs = year_pairs,
#   template_raster = template_raster,
#   tile_ids = "BH4S556X_3",
#   write_spatial_outputs = TRUE,
#   write_global_start_bathy = TRUE,
#   n_workers = 1
# )

# --- 5B. Apply GLOBAL models to ALL prediction tiles ---
# global_all_predictions <- Predict_With_Global_Models(
#   grid_sf = grid_sf,
#   output_dir_pred = output_dir_pred,
#   output_dir_train = output_dir_train,
#   year_pairs = year_pairs,
#   template_raster = template_raster,
#   tile_ids = "all",
#   write_spatial_outputs = TRUE,
#   write_global_start_bathy = TRUE,
#   n_workers = max(1, parallel::detectCores() - 1)
# )

# --- 6. Optional temporal hold-one-pair-out diagnostic ---
# temporal_metrics <- Run_Temporal_Holdout_Diagnostic(
#   output_dir_train = output_dir_train,
#   year_pairs = year_pairs
# )

# ==============================================================================
# SEAM-SAFE RASTER CONTRACT
# ==============================================================================
#
# Every tiled LOCAL and GLOBAL raster is:
#   * snapped to template_raster;
#   * cropped using the official grid polygon;
#   * masked outside that polygon;
#   * written with NoData = -9999.
#
# Prediction FST rows may extend beyond the polygon, but those extra rows cannot
# create valid raster cells outside the tile. Consequently, VRT source rasters
# no longer overlap with valid values across neighboring grid boundaries.
# ==============================================================================

# ==============================================================================
# FINAL RETAINED SPATIAL OUTPUT SET
# ==============================================================================
#
# LOCAL training:
#   LOCAL_Train_Start_Bathy_t
#   LOCAL_Train_Mean_Predicted_Bathy_t1
#   LOCAL_Train_Surveyed_Change
#   LOCAL_Train_Mean_Predicted_Change
#   LOCAL_Train_SD_Bathy_t1
#
# LOCAL prediction:
#   LOCAL_Pred_Start_Bathy_t
#   LOCAL_Pred_Mean_Bathy_t1
#   LOCAL_Pred_Mean_Change
#   LOCAL_Pred_SD_Bathy_t1
#   LOCAL_Predictor_Extrapolation_Count
#   LOCAL_Pred_BlueTopo_UC_t (when available)
#
# GLOBAL prediction:
#   GLOBAL_Pred_Start_Bathy_t
#   GLOBAL_Pred_Mean_Bathy_t1
#   GLOBAL_Pred_Mean_Change
#   GLOBAL_Pred_SD_Bathy_t1
#   GLOBAL_Predictor_Extrapolation_Count
#
# Annualized rates remain available in the saved FST summaries but are no longer
# written as routine GeoTIFF products.
# ==============================================================================

# ==============================================================================
# MULTI-YEAR PARALLEL TROUBLESHOOTING
# ==============================================================================
#
# "Error in unserialize(node$con): error reading from connection" indicates that
# a PSOCK worker terminated. It is not an error in base::unserialize itself.
#
# This script now:
#   * creates a fresh cluster for each year pair;
#   * stops the tile cluster before full-extent deployment begins;
#   * saves pair-specific result files;
#   * caps the default worker count at six;
#   * forces cleanup between year pairs.
#
# For very large tiles, reduce n_workers to 2-4. More workers can be slower when
# each worker holds large XGBoost matrices and bootstrap arrays.
# ==============================================================================

# ==============================================================================
# END OF FINAL MASTER SCRIPT
# ==============================================================================
