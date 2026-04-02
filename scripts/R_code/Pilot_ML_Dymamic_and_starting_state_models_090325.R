
# Model training and prediction test workflow # 2 - 9/4/25 

# This script will contain 2 model trainings, and one prediction 

# Model 1 = The Dynamic Model 

# the dynamic model will train information on the Delta variables, how much the bathymetry has changed from 1 year to another, and all the change elements between all 
# the other environmental variables too. 

# Model 2 = Single State Model ---- which will then lead to prediction 

# First steps
# - 
# - standardise all CORE prediction rasters
# - generate additional neighberhood stats on bathy derrived data (spatially in raster formate first) over the prediction extent
# - check alignment of prediction rasters with mask 
# -standardise all training rasters
# - check alignment 
# - Grid out raster data  -all columns - 
# - make two version of the data now (1) keeep only starting year columns and save long format data and (2) Delta variables 




# Neighborhood Analysis (creating the SD, and Mean in a focal window of 3 cells in a focal window)----
# slope_t_mean3 = the mean of the surrounding data in a focal window of 3 cells
# bathy_t_sd3 = 		○ The mean bathy in a local window is redundant (strongly collinear with the pixel itself).
#But the SD of bathy within a window captures “roughness” and context: is the point in a flat patch or near a ridge/edge?

# GENERATE NEIGHBERHOOD STATS (SD and MEAN within a focal window of 3 cells)  - SPATIALLY - ON THE RASTER
# the neighberhood width of the bathy derrived features is created, so we try and give context, within a focal window, of the main variable of how the nearby features changes
# if our resolution is 8m, a neighberhood width of 3 is 24m focal window around our cell value. * This can be modified at different scales, depending on the scale of features
# which influence seabed change 



## STANDARDISE RASTERS FUNCTION ##
library(raster)
library(tools)

# Simple logger (prints to console; adapt to write to file if you want)
log_message <- function(msg) {
  ts <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s\n", ts, msg))
}

standardize_rasters_improved <- function(input_dir,
                                         mask_path,
                                         output_dir,
                                         sequential = TRUE,
                                         overwrite = FALSE,
                                         method_cont = "bilinear",
                                         method_cat = "ngb") {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Load mask reference
  mask <- tryCatch({
    raster(mask_path)
  }, error = function(e) stop("Cannot load mask raster: ", conditionMessage(e)))
  ref_res <- res(mask)
  
  # List input and output files
  all_input_files <- list.files(input_dir, pattern = "\\.tif$", full.names = TRUE, ignore.case = TRUE)
  if (length(all_input_files) == 0) {
    log_message("No .tif files found in input_dir; nothing to do.")
    return(invisible(list(processed = character(0), skipped = character(0), failed = character(0))))
  }
  
  existing_out_basenames <- list.files(output_dir, pattern = "\\.tif$", full.names = FALSE, ignore.case = TRUE)
  
  # Determine which files to process (skip those already present in output_dir)
  input_basenames <- basename(all_input_files)
  to_process_idx <- which(!input_basenames %in% existing_out_basenames)
  files_to_process <- all_input_files[to_process_idx]
  skipped_files <- input_basenames[!seq_along(input_basenames) %in% to_process_idx]
  
  log_message(sprintf("Found %d input file(s); %d already exist in output_dir and will be skipped; %d to process.",
                      length(all_input_files), length(skipped_files), length(files_to_process)))
  
  make_tmp_tif <- function(prefix = "tmp_") {
    tf <- raster::rasterTmpFile()
    if (!grepl("\\.tif$", tf, ignore.case = TRUE)) tf <- paste0(tf, ".tif")
    return(file.path(dirname(tf), paste0(prefix, basename(tf))))
  }
  
  process_one <- function(fpath) {
    fname <- basename(fpath)
    out_path <- file.path(output_dir, fname)
    log_message(paste0("-> Processing: ", fname))
    res <- tryCatch({
      r <- raster(fpath)
      
      # choose interpolation method depending on file name (categorical detection)
      is_categorical <- grepl("grain|sed|terrain|prim_sed|category", fname, ignore.case = TRUE)
      method_use <- if (is_categorical) method_cat else method_cont
      
      # project/resample directly to mask grid
      tmp_proj <- make_tmp_tif("proj_")
      r_proj <- projectRaster(from = r, to = mask, method = method_use, filename = tmp_proj, overwrite = TRUE)
      
      # mask to final shape
      tmp_mask <- make_tmp_tif("mask_")
      r_masked <- mask(r_proj, mask, filename = tmp_mask, overwrite = TRUE)
      
      # bathy-specific rule: convert positive depths to NA for main bathy files
      is_main_bathy_file <- grepl("^bathy_\\d{4}_filled\\.tif$", fname, ignore.case = TRUE)
      if (is_main_bathy_file) {
        log_message("   - Applying bathy rule: values > 0 -> NA")
        tmp_calc <- make_tmp_tif("calc_")
        r_masked <- calc(r_masked, fun = function(v) { v[v > 0] <- NA; v }, filename = tmp_calc, overwrite = TRUE)
      }
      
      # final write (unless file exists and overwrite == FALSE, but we already skipped pre-existing files)
      writeRaster(r_masked, filename = out_path, overwrite = overwrite, format = "GTiff",
                  options = c("COMPRESS=LZW", "TILED=YES"))
      
      # quick post-check alignment (resolution)
      r_check <- raster(out_path)
      if (!isTRUE(all.equal(res(r_check), ref_res, tolerance = 1e-6))) {
        log_message(sprintf("   ! Resolution mismatch after write for %s (expected: %s ; got: %s)",
                            fname, paste(ref_res, collapse = ","), paste(res(r_check), collapse = ",")))
      } else {
        log_message("   ✓ Written and aligned.")
      }
      
      # cleanup
      rm(r, r_proj, r_masked, r_check); gc()
      TRUE
    }, error = function(e) {
      log_message(sprintf("   ✗ ERROR processing %s : %s", fname, conditionMessage(e)))
      FALSE
    })
    return(list(file = fname, ok = isTRUE(res)))
  }
  
  results <- list(processed = character(0), failed = character(0))
  if (length(files_to_process) == 0) {
    log_message("No new files to process.")
  } else {
    if (sequential) {
      for (f in files_to_process) {
        r <- process_one(f)
        if (r$ok) results$processed <- c(results$processed, r$file) else results$failed <- c(results$failed, r$file)
      }
    } else {
      # If you want parallel, replace with mclapply or foreach + doParallel (not done here to keep portability)
      for (f in files_to_process) {
        r <- process_one(f)
        if (r$ok) results$processed <- c(results$processed, r$file) else results$failed <- c(results$failed, r$file)
      }
    }
  }
  
  summary_msg <- sprintf("Standardization complete: %d processed, %d skipped, %d failed.",
                         length(results$processed), length(skipped_files), length(results$failed))
  log_message(summary_msg)
  
  return(invisible(list(processed = results$processed, skipped = skipped_files, failed = results$failed)))
}


# Prediction
standardize_rasters_improved(input_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/part_processed",
                             mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif",
                             output_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/processed",
                             sequential = TRUE)
# Training 
standardize_rasters_improved(input_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/processed",
                             mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/training.mask.UTM17_8m.tif",
                             output_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Training/processed",
                             sequential = TRUE)


# run the below check after standardising the rasters 
# ==============================================================================
#
#           Raster Alignment Diagnostic Tool
#
# ==============================================================================
#
# Purpose:
# This script diagnoses issues with raster alignment by checking a directory of
# raster files against a reference "mask" raster. It identifies any rasters
# that have a different extent, resolution, or origin, which is a common
# cause of errors when creating a raster stack.
#
# Instructions:
# 1. Verify the file paths in the "Configuration" section below.
# 2. Run the script.
# 3. Review the console output for a summary of any mismatched files.
#
# ==============================================================================

# --- Load Necessary Library ---
library(raster)

# --- Configuration ---
# Set the paths to your mask and the directory of rasters to be checked.
# These are taken from the error message you provided.

# Training 
mask_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/training.mask.UTM17_8m.tif"
raster_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/training/processed"

#Prediction
mask_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif"
raster_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/prediction/processed"


# --- Main Diagnostic Function ---

#' Checks a list of rasters against a reference mask for alignment.
#' @param mask_raster_path The file path to the reference raster.
#' @param rasters_to_check_dir The directory containing the .tif files to check.
check_raster_alignment <- function(mask_raster_path, rasters_to_check_dir) {
  
  message("--- Starting Raster Alignment Check ---")
  
  # 1. Load the reference mask and get its properties
  tryCatch({
    mask <- raster(mask_raster_path)
    ref_extent <- extent(mask)
    ref_res <- res(mask)
    ref_origin <- origin(mask)
    ref_crs <- crs(mask)
    message(paste("Reference Mask Loaded:", basename(mask_raster_path)))
    message(paste("  - Extent:", paste(round(as.vector(ref_extent), 2), collapse=", ")))
    message(paste("  - Resolution:", paste(ref_res, collapse=", ")))
  }, error = function(e) {
    stop("Fatal Error: Could not load the reference mask. Please check the path. Error: ", e$message)
  })
  
  # 2. Get the list of raster files to check
  raster_files <- list.files(
    rasters_to_check_dir,
    pattern = "\\.tif$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  if (length(raster_files) == 0) {
    warning("No .tif files were found in the specified directory.")
    return(invisible(NULL))
  }
  
  message(paste("\nFound", length(raster_files), "rasters to check. Beginning comparison..."))
  
  # 3. Loop through files and check for mismatches
  mismatched_files <- data.frame(
    filename = character(),
    reason = character(),
    details = character(),
    stringsAsFactors = FALSE
  )
  
  for (r_file in raster_files) {
    basename_r <- basename(r_file)
    cat(".") # Simple progress indicator
    
    tryCatch({
      current_raster <- raster(r_file)
      
      # Use all.equal for a robust comparison, especially for floating point numbers
      extent_match <- isTRUE(all.equal(ref_extent, extent(current_raster)))
      res_match <- isTRUE(all.equal(ref_res, res(current_raster)))
      origin_match <- isTRUE(all.equal(ref_origin, origin(current_raster)))
      
      if (!extent_match) {
        mismatched_files <- rbind(mismatched_files, data.frame(
          filename = basename_r,
          reason = "Extent Mismatch",
          details = paste(round(as.vector(extent(current_raster)), 2), collapse=", ")
        ))
      }
      if (!res_match) {
        mismatched_files <- rbind(mismatched_files, data.frame(
          filename = basename_r,
          reason = "Resolution Mismatch",
          details = paste(res(current_raster), collapse=", ")
        ))
      }
      if (!origin_match) {
        mismatched_files <- rbind(mismatched_files, data.frame(
          filename = basename_r,
          reason = "Origin Mismatch",
          details = paste(origin(current_raster), collapse=", ")
        ))
      }
      
    }, error = function(e) {
      mismatched_files <<- rbind(mismatched_files, data.frame(
        filename = basename_r,
        reason = "Error Loading Raster",
        details = e$message
      ))
    })
  }
  
  # 4. Report the results
  cat("\n\n--- Diagnostic Complete ---\n\n")
  if (nrow(mismatched_files) == 0) {
    message("✅ SUCCESS: All rasters in the directory are perfectly aligned with the mask.")
  } else {
    message(paste("❌ PROBLEM: Found", nrow(mismatched_files), "issues with the following raster(s):"))
    print(mismatched_files, row.names = FALSE)
  }
  
  return(invisible(mismatched_files))
}

# --- Run the Diagnostic ---
# Check both prediction and training!
check_raster_alignment(mask_path, raster_dir)



#------
# if there are discrepanices between the training and prediction gpkgs this aligns them
build_canonical_master_grid <- function(
    training_grid_gpkg,
    prediction_grid_gpkg,
    out_master_grid_gpkg,
    out_training_tile_csv,
    out_prediction_tile_csv,
    out_diag_plot_png = NULL
) {
  library(sf)
  library(dplyr)
  library(ggplot2)
  
  log_msg <- function(...) cat("[", format(Sys.time(), "%H:%M:%S"), "]", ..., "\n")
  
  # ------------------------------------------------------------
  # 1. Read the two existing grids
  # ------------------------------------------------------------
  log_msg("Reading training grid:", training_grid_gpkg)
  g_train <- st_read(training_grid_gpkg, quiet = TRUE)
  log_msg(paste("Training tiles:", nrow(g_train)))
  
  log_msg("Reading prediction grid:", prediction_grid_gpkg)
  g_pred  <- st_read(prediction_grid_gpkg, quiet = TRUE)
  log_msg(paste("Prediction tiles:", nrow(g_pred)))
  
  # Normalise tile_id column name if needed
  if (!"tile_id" %in% names(g_train)) {
    if ("tile" %in% names(g_train)) {
      g_train <- g_train %>% rename(tile_id = tile)
    } else stop("Training grid has no 'tile_id' or 'tile' column.")
  }
  if (!"tile_id" %in% names(g_pred)) {
    if ("tile" %in% names(g_pred)) {
      g_pred <- g_pred %>% rename(tile_id = tile)
    } else stop("Prediction grid has no 'tile_id' or 'tile' column.")
  }
  
  # ------------------------------------------------------------
  # 2. Bring both into the same CRS
  # ------------------------------------------------------------
  log_msg("CRS (training):"); print(st_crs(g_train))
  log_msg("CRS (prediction):"); print(st_crs(g_pred))
  
  crs_target <- st_crs(g_train)
  if (is.na(crs_target)) stop("Training grid CRS is NA.")
  
  if (!identical(st_crs(g_pred), crs_target)) {
    log_msg("Reprojecting prediction grid to training CRS...")
    g_pred <- st_transform(g_pred, crs_target)
  }
  
  # ------------------------------------------------------------
  # 3. Build full tile_id set and canonical geometry per tile
  # ------------------------------------------------------------
  all_ids <- sort(unique(c(g_train$tile_id, g_pred$tile_id)))
  log_msg(paste("Total unique tile_ids:", length(all_ids)))
  
  geom_for_id <- function(sf_obj, id) {
    sf_obj[sf_obj$tile_id == id, ]
  }
  
  canon_list <- vector("list", length(all_ids))
  
  for (i in seq_along(all_ids)) {
    tid <- all_ids[i]
    tr <- geom_for_id(g_train, tid)
    pr <- geom_for_id(g_pred,  tid)
    
    has_tr <- nrow(tr) > 0
    has_pr <- nrow(pr) > 0
    
    # parent_tile (prefer training if available)
    parent_tile <- NA_character_
    if (has_tr && "parent_tile" %in% names(tr)) {
      parent_tile <- tr$parent_tile[1]
    } else if (has_pr && "parent_tile" %in% names(pr)) {
      parent_tile <- pr$parent_tile[1]
    }
    
    if (has_tr && has_pr) {
      # union each to a single geom
      tr_u <- st_make_valid(st_union(st_geometry(tr)))
      pr_u <- st_make_valid(st_union(st_geometry(pr)))
      
      # try intersection
      g_int <- tryCatch(st_intersection(tr_u, pr_u),
                        error = function(e) NULL)
      
      if (!is.null(g_int) &&
          length(g_int) > 0 &&
          !all(st_is_empty(g_int))) {
        geom_final <- g_int
      } else {
        # fallback to union of both
        geom_final <- st_make_valid(st_union(tr_u, pr_u))
      }
      
      source_val <- "both"
      
    } else if (has_tr) {
      geom_final <- st_make_valid(st_union(st_geometry(tr)))
      source_val <- "training"
      
    } else if (has_pr) {
      geom_final <- st_make_valid(st_union(st_geometry(pr)))
      source_val <- "prediction"
      
    } else {
      next  # shouldn't happen
    }
    
    # Enforce consistent schema: tile_id, parent_tile, source, geometry
    canon_list[[i]] <- st_sf(
      tile_id     = tid,
      parent_tile = parent_tile,
      source      = source_val,
      geometry    = geom_final,
      crs         = crs_target
    )
  }
  
  canon_list <- canon_list[!sapply(canon_list, is.null)]
  master_grid <- do.call(rbind, canon_list)
  master_grid <- st_make_valid(master_grid)
  
  log_msg(paste("Canonical master grid tiles:", nrow(master_grid)))
  
  # ------------------------------------------------------------
  # 4. Write master grid and tile lists
  # ------------------------------------------------------------
  log_msg("Writing master grid GPKG...")
  st_write(master_grid, out_master_grid_gpkg, delete_layer = TRUE, quiet = TRUE)
  
  pred_ids  <- sort(unique(g_pred$tile_id))
  train_ids <- sort(unique(g_train$tile_id))
  
  write.csv(data.frame(tile_id = pred_ids),
            out_prediction_tile_csv, row.names = FALSE)
  write.csv(data.frame(tile_id = train_ids),
            out_training_tile_csv, row.names = FALSE)
  
  log_msg(paste("Prediction tile list written to:", out_prediction_tile_csv))
  log_msg(paste("Training tile list written to:", out_training_tile_csv))
  log_msg(paste("Master grid written to:", out_master_grid_gpkg))
  
  # ------------------------------------------------------------
  # 5. Optional diagnostic plot
  # ------------------------------------------------------------
  if (!is.null(out_diag_plot_png)) {
    log_msg("Creating diagnostic plot...")
    
    p <- ggplot() +
      geom_sf(data = master_grid, aes(colour = source), fill = NA, linewidth = 0.3) +
      labs(
        title    = "Canonical Master Grid",
        subtitle = "purple = both, blue = training only, red = prediction only"
      ) +
      scale_colour_manual(values = c(
        prediction = "red",
        training   = "blue",
        both       = "purple"
      )) +
      theme_minimal()
    
    ggsave(out_diag_plot_png, p, width = 8, height = 6, dpi = 300)
    log_msg(paste("Diagnostic plot written to:", out_diag_plot_png))
  }
  
  log_msg("=== DONE (build_canonical_master_grid) ===")
  invisible(master_grid)
}


master_grid <- build_canonical_master_grid(
  training_grid_gpkg =
    "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg",
  prediction_grid_gpkg =
    "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg",
  out_master_grid_gpkg =
    "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/master_grid_canonical.gpkg",
  out_training_tile_csv =
    "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/training_tile_list_canonical.csv",
  out_prediction_tile_csv =
    "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/prediction_tile_list_canonical.csv",
  out_diag_plot_png =
    "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/master_grid_canonical_diag.png"
)





#' Grid Out Raster Data (Alignment-Safe, Parallel-Safe)   # WORKS GREAT BUT INTRODUCES NA VALUES 
#'
#' Splits large rasters into sub-grids based on a master grid,
#' ensuring perfect alignment between training and prediction outputs.
#'
#' For training runs, the training mask is preprocessed ONCE outside
#' the parallel loop, and each tile gets a precomputed vector of mask
#' values, so no heavy raster operations happen inside workers.
#'
#' @param master_grid_gpkg Path to GeoPackage file containing master grid tiles.
#'        Must have a 'tile_id' column.
#' @param raster_dir Directory containing rasters to process (.tif).
#' @param output_dir Directory for output .fst and logs.
#' @param mask_path Path to mask raster (normally prediction.mask).
#' @param template_mask_path Path to template raster that defines grid geometry.
#' @param data_type "training" or "prediction".
#' @param relevant_tile_ids Optional vector of tile IDs to process (subset).
#' @param parallel_tiles Logical; process tiles in parallel?
#' @return Invisibly returns a list of tile summaries.
#' @import raster sf dplyr foreach doParallel fst sp
grid_out_raster_data <- function(master_grid_gpkg,
                                 raster_dir,
                                 output_dir,
                                 mask_path,
                                 template_mask_path,
                                 data_type = c("training", "prediction"),
                                 relevant_tile_ids = NULL,
                                 parallel_tiles = TRUE) {
  
  ## --- Load packages ---
  library(sf); library(raster); library(dplyr)
  library(foreach); library(doParallel); library(fst); library(sp)
  
  data_type <- match.arg(data_type)
  log_message <- function(msg) {
    cat(sprintf("[%s] %s\n",
                format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
  }
  
  ## --- Read grid & template mask ---
  log_message("=== INITIAL SETUP & CRS VERIFICATION ===")
  grid_all <- sf::st_read(master_grid_gpkg, quiet = TRUE)
  
  if (!"tile_id" %in% names(grid_all)) {
    stop("master_grid_gpkg must contain a 'tile_id' column.")
  }
  
  # template grid for alignment
  template_r <- raster(template_mask_path)
  mask_r     <- raster(mask_path)  # currently unused, but kept for flexibility
  
  # align grid to template CRS if needed
  if (!compareCRS(grid_all, template_r)) {
    log_message("Grid CRS differs from template; reprojecting grid to template CRS.")
    grid_all <- st_transform(grid_all, crs = crs(template_r))
  }
  
  # subset tiles if requested
  if (!is.null(relevant_tile_ids)) {
    grid_all <- grid_all[grid_all$tile_id %in% relevant_tile_ids, ]
  }
  
  # --- Automatically filter to true training tiles ---
  if (data_type == "training") {
    training_tiles_gpkg <- gsub("master_grid_canonical.gpkg",
                                "Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg",
                                master_grid_gpkg)
    
    if (file.exists(training_tiles_gpkg)) {
      orig_train <- sf::st_read(training_tiles_gpkg, quiet = TRUE)
      true_train_ids <- orig_train$tile_id
      
      log_message(sprintf(
        "Filtering training run to original %d training tiles.",
        length(true_train_ids)
      ))
      
      # restrict grid_all BEFORE processing
      grid_all <- grid_all[grid_all$tile_id %in% true_train_ids, ]
      
      if (nrow(grid_all) == 0)
        stop("Training mode: No matching tiles found in the canonical grid.")
    } else {
      log_message("⚠ Could not locate original training grid — NOT filtering.")
    }
  }
  
  if (nrow(grid_all) == 0) stop("No matching tiles to process.")
  
  ## --- File list ---
  all_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  if (length(all_files) == 0) stop("No .tif files found in raster_dir.")
  log_message(sprintf("Processing %d tile(s) for '%s' using %d rasters.",
                      nrow(grid_all), data_type, length(all_files)))
  
  ## --- Preprocess training mask OUTSIDE parallel (critical for stability) ---
  # training_mask_tiles: named list, one logical vector per tile_id
  training_mask_tiles <- NULL
  
  if (data_type == "training") {
    training_mask_path <- gsub("prediction", "training", mask_path)
    if (file.exists(training_mask_path)) {
      log_message("Preprocessing training mask (once, outside parallel).")
      
      # load and align to template grid
      tm <- raster(training_mask_path)
      if (!compareCRS(tm, template_r)) {
        log_message("Reprojecting training mask to template CRS...")
        tm <- projectRaster(tm, template_r, method = "ngb")
      } else {
        # still resample to guarantee same grid if needed
        tm <- resample(tm, template_r, method = "ngb")
      }
      
      training_mask_tiles <- list()
      
      for (i in seq_len(nrow(grid_all))) {
        tile    <- grid_all[i, ]
        tname   <- as.character(tile$tile_id[1])
        tile_sp <- as(tile, "Spatial")
        
        # tile's template subset
        tile_template <- crop(template_r, extent(tile_sp))
        
        # crop training mask to same tile extent and resample onto tile_template
        tm_crop <- crop(tm, extent(tile_template))
        tm_tile <- resample(tm_crop, tile_template, method = "ngb")
        
        # store logical vector: TRUE for inside-training, FALSE elsewhere
        vals <- tm_tile[]
        training_mask_tiles[[tname]] <- !is.na(vals) & vals != 0
      }
      
      log_message("Training mask preprocessed for all tiles.")
      
    } else {
      log_message("⚠ No training mask raster found; proceeding without it.")
    }
  }
  
  ## --- Per-tile processor ---
  process_tile <- function(tile_row) {
    tile     <- tile_row
    tile_name <- as.character(tile$tile_id[1])
    
    tile_dir <- file.path(output_dir, tile_name)
    dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
    logf <- file.path(tile_dir, paste0("tile_log_", tile_name, ".txt"))
    cat(sprintf("[%s] === PROCESSING TILE %s ===\n",
                format(Sys.time(), "%H:%M:%S"), tile_name),
        file = logf, append = TRUE)
    
    tile_sp      <- as(tile, "Spatial")
    template_crop <- crop(template_r, extent(tile_sp))
    
    # --- Load and align rasters for this tile ---
    layer_list <- lapply(all_files, function(rf) {
      nm <- tools::file_path_sans_ext(basename(rf))
      tryCatch({
        r <- raster(rf)
        if (!compareCRS(r, template_r)) {
          r <- projectRaster(r, template_r, method = "bilinear")
        }
        r_crop  <- crop(r, extent(template_crop))
        
        # r_align <- resample(r_crop, template_crop, method = "bilinear") - possible cause of introduced NAs
        
        interp_method <- if (grepl("^bathy_|^bt\\.bathy$", nm)) {
          "bilinear"
        } else {
          "ngb"
        }
        
        r_align <- resample(r_crop, template_crop, method = interp_method)
        
        # Diagnostic check on the number of NAs
        na_frac <- mean(is.na(values(r_align)))
        
        cat(sprintf(
          "%s | NA fraction: %.3f\n",
          nm, na_frac
        ), file = logf, append = TRUE)
        
        
        # Apply training mask softly (set NA where mask == 0), but no raster ops here
        if (!is.null(training_mask_tiles) && data_type == "training") {
          m_vals <- training_mask_tiles[[tile_name]]
          if (!is.null(m_vals)) {
            # ensure same length
            if (length(m_vals) == ncell(r_align)) {
              vals_r <- r_align[]
              vals_r[!m_vals] <- NA
              r_align[] <- vals_r
            } else {
              cat(sprintf("WARNING: mask length mismatch for tile %s (mask: %d, raster: %d)\n",
                          tile_name, length(m_vals), ncell(r_align)),
                  file = logf, append = TRUE)
            }
          }
        }
        
        names(r_align) <- nm
        return(r_align)
      }, error = function(e) {
        cat(sprintf("Error in %s: %s\n", nm, e$message), file = logf, append = TRUE)
        NULL
      })
    })
    
    layer_list <- layer_list[!sapply(layer_list, is.null)]
    if (length(layer_list) == 0) {
      cat("No valid rasters for this tile.\n", file = logf, append = TRUE)
      return(NULL)
    }
    
    # --- Stack and extract all cells (including NA) ---
    r_stack <- stack(layer_list)
    names(r_stack) <- sapply(layer_list, names)
    
    xy   <- xyFromCell(r_stack, 1:ncell(r_stack))
    vals <- as.data.frame(getValues(r_stack))
    combined <- cbind(data.frame(X = xy[,1], Y = xy[,2]), vals)
    
    ## Data-type specific cleanup
    if (data_type == "training") {
      # drop any bt.* columns if present
      combined <- combined %>% dplyr::select(-dplyr::starts_with("bt."))
      
      needed <- c("bathy_2004_filled", "bathy_2006_filled", "bathy_2010_filled",
                  "bathy_2015_filled", "bathy_2022_filled")
      if (all(needed %in% names(combined))) {
        combined <- combined %>%
          mutate(
            delta_bathy_2004_2006 = bathy_2006_filled - bathy_2004_filled,
            delta_bathy_2006_2010 = bathy_2010_filled - bathy_2006_filled,
            delta_bathy_2010_2015 = bathy_2015_filled - bathy_2010_filled,
            delta_bathy_2015_2022 = bathy_2022_filled - bathy_2015_filled
          )
      }
    } else if (data_type == "prediction") {
      pattern   <- "^(bathy|bpi_broad|bpi_fine|slope|terrain_classification|rugosity|curv_plan|curv_profile|curv_total|flowacc|flowdir|gradmag|shearproxy|slope_deg|tci)_\\d{4}"
      drop_cols <- grep(pattern, names(combined), value = TRUE)
      if (length(drop_cols) > 0) {
        combined <- combined %>% dplyr::select(-dplyr::all_of(drop_cols))
      }
    }
    
    combined$FID     <- seq_len(nrow(combined))
    combined$tile_id <- tile_name
    
    out_path <- file.path(tile_dir,
                          sprintf("%s_%s_clipped_data.fst", tile_name, data_type))
    write.fst(combined, out_path)
    
    cat(sprintf("[%s] ✅ Wrote %s (%d rows, %d cols)\n",
                format(Sys.time(), "%H:%M:%S"), basename(out_path),
                nrow(combined), ncol(combined)),
        file = logf, append = TRUE)
    
    return(list(tile_id = tile_name, n_points = nrow(combined)))
  }
  
  ## --- Run tiles (optionally in parallel) ---
  results <- list()
  
  if (parallel_tiles) {
    cores <- max(1, parallel::detectCores() - 1)
    log_message(sprintf("Running tiles in parallel using %d cores.", cores))
    cl <- makeCluster(cores)
    on.exit(stopCluster(cl), add = TRUE)
    
    registerDoParallel(cl)
    
    # make sure needed packages & objects are visible in workers
    clusterEvalQ(cl, {
      library(raster); library(sf); library(sp)
      library(dplyr); library(fst)
      NULL
    })
    
    results <- foreach(i = seq_len(nrow(grid_all)),
                       .packages = c("raster","sf","sp","dplyr","fst","tools"),
                       .export   = c("process_tile", "training_mask_tiles",
                                     "template_r", "all_files")) %dopar% {
                                       process_tile(grid_all[i, ])
                                     }
  } else {
    log_message("Running tiles sequentially (no parallel).")
    for (i in seq_len(nrow(grid_all))) {
      results[[i]] <- process_tile(grid_all[i, ])
      gc()
    }
  }
  
  ## --- Summarise ---
  results <- results[!sapply(results, is.null)]
  if (length(results) > 0) {
    result_df <- do.call(rbind, lapply(results, as.data.frame))
    write.csv(result_df,
              file.path(output_dir, "tile_processing_summary.csv"),
              row.names = FALSE)
    log_message("=== ALL TILES COMPLETE ===")
    log_message(sprintf("Summary written to %s",
                        file.path(output_dir, "tile_processing_summary.csv")))
  } else {
    log_message("No tiles processed successfully.")
  }
  
  invisible(results)
}


grid_out_raster_data <- function(master_grid_gpkg, # V2 SWAPS THE ORDER OF CROPPING AND RESAMPLING 
                                 raster_dir,
                                 output_dir,
                                 mask_path,
                                 template_mask_path,
                                 data_type = c("training", "prediction"),
                                 relevant_tile_ids = NULL,
                                 parallel_tiles = TRUE) {
  
  ## --- Load packages ---
  library(sf); library(raster); library(dplyr)
  library(foreach); library(doParallel); library(fst); library(sp)
  
  data_type <- match.arg(data_type)
  log_message <- function(msg) {
    cat(sprintf("[%s] %s\n",
                format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
  }
  
  ## --- Read grid & template mask ---
  log_message("=== INITIAL SETUP & CRS VERIFICATION ===")
  grid_all <- sf::st_read(master_grid_gpkg, quiet = TRUE)
  
  if (!"tile_id" %in% names(grid_all)) {
    stop("master_grid_gpkg must contain a 'tile_id' column.")
  }
  
  # template grid for alignment
  template_r <- raster(template_mask_path)
  mask_r     <- raster(mask_path)  # currently unused, but kept for flexibility
  
  # align grid to template CRS if needed
  if (!compareCRS(grid_all, template_r)) {
    log_message("Grid CRS differs from template; reprojecting grid to template CRS.")
    grid_all <- st_transform(grid_all, crs = crs(template_r))
  }
  
  # subset tiles if requested
  if (!is.null(relevant_tile_ids)) {
    grid_all <- grid_all[grid_all$tile_id %in% relevant_tile_ids, ]
  }
  
  # --- Automatically filter to true training tiles ---
  if (data_type == "training") {
    training_tiles_gpkg <- gsub("master_grid_canonical.gpkg",
                                "Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg",
                                master_grid_gpkg)
    
    if (file.exists(training_tiles_gpkg)) {
      orig_train <- sf::st_read(training_tiles_gpkg, quiet = TRUE)
      true_train_ids <- orig_train$tile_id
      
      log_message(sprintf(
        "Filtering training run to original %d training tiles.",
        length(true_train_ids)
      ))
      
      # restrict grid_all BEFORE processing
      grid_all <- grid_all[grid_all$tile_id %in% true_train_ids, ]
      
      if (nrow(grid_all) == 0)
        stop("Training mode: No matching tiles found in the canonical grid.")
    } else {
      log_message("⚠ Could not locate original training grid — NOT filtering.")
    }
  }
  
  if (nrow(grid_all) == 0) stop("No matching tiles to process.")
  
  ## --- File list ---
  all_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  if (length(all_files) == 0) stop("No .tif files found in raster_dir.")
  log_message(sprintf("Processing %d tile(s) for '%s' using %d rasters.",
                      nrow(grid_all), data_type, length(all_files)))
  
  ## --- Preprocess training mask OUTSIDE parallel (critical for stability) ---
  # training_mask_tiles: named list, one logical vector per tile_id
  training_mask_tiles <- NULL
  
  if (data_type == "training") {
    training_mask_path <- gsub("prediction", "training", mask_path)
    if (file.exists(training_mask_path)) {
      log_message("Preprocessing training mask (once, outside parallel).")
      
      # load and align to template grid
      tm <- raster(training_mask_path)
      if (!compareCRS(tm, template_r)) {
        log_message("Reprojecting training mask to template CRS...")
        tm <- projectRaster(tm, template_r, method = "ngb")
      } else {
        # still resample to guarantee same grid if needed
        tm <- resample(tm, template_r, method = "ngb")
      }
      
      training_mask_tiles <- list()
      
      for (i in seq_len(nrow(grid_all))) {
        tile    <- grid_all[i, ]
        tname   <- as.character(tile$tile_id[1])
        tile_sp <- as(tile, "Spatial")
        
        # tile's template subset
        tile_template <- crop(template_r, extent(tile_sp))
        
        # crop training mask to same tile extent and resample onto tile_template
        tm_crop <- crop(tm, extent(tile_template))
        tm_tile <- resample(tm_crop, tile_template, method = "ngb")
        
        # store logical vector: TRUE for inside-training, FALSE elsewhere
        vals <- tm_tile[]
        training_mask_tiles[[tname]] <- !is.na(vals) & vals != 0
      }
      
      log_message("Training mask preprocessed for all tiles.")
      
    } else {
      log_message("⚠ No training mask raster found; proceeding without it.")
    }
  }
  
  ## --- Per-tile processor ---
  process_tile <- function(tile_row) {
    tile     <- tile_row
    tile_name <- as.character(tile$tile_id[1])
    
    tile_dir <- file.path(output_dir, tile_name)
    dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
    logf <- file.path(tile_dir, paste0("tile_log_", tile_name, ".txt"))
    cat(sprintf("[%s] === PROCESSING TILE %s ===\n",
                format(Sys.time(), "%H:%M:%S"), tile_name),
        file = logf, append = TRUE)
    
    tile_sp      <- as(tile, "Spatial")
    template_crop <- crop(template_r, extent(tile_sp), snap = "out")
    
    # --- Load and align rasters for this tile ---
    layer_list <- lapply(all_files, function(rf) {
      nm <- tools::file_path_sans_ext(basename(rf))
      
      tryCatch({
        r <- raster(rf)
        
        if (!compareCRS(r, template_r)) {
          r <- projectRaster(r, template_r, method = "bilinear")
        }
        # ORDER IS IMPORTANT - INTERPOLATION BEFORE CROPPING! 
        # define interp method (only bathy bilinear, everything else ngb)
        interp_method <- if (grepl("^bathy_|^bt\\.bathy$", nm)) "bilinear" else "ngb"
        
        # resample directly onto the tile template grid (no pre-crop)
        r_align <- resample(r, template_crop, method = interp_method)
        
        # training mask (unchanged)
        if (!is.null(training_mask_tiles) && data_type == "training") {
          m_vals <- training_mask_tiles[[tile_name]]
          if (!is.null(m_vals)) {
            if (length(m_vals) == ncell(r_align)) {
              vals_r <- r_align[]
              vals_r[!m_vals] <- NA
              r_align[] <- vals_r
            } else {
              cat(sprintf(
                "WARNING: mask length mismatch for tile %s (mask: %d, raster: %d)\n",
                tile_name, length(m_vals), ncell(r_align)
              ), file = logf, append = TRUE)
            }
          }
        }
        
        names(r_align) <- nm
        r_align
      }, error = function(e) {
        cat(sprintf("Error in %s: %s\n", nm, e$message), file = logf, append = TRUE)
        NULL
      })
    })
    
    
    layer_list <- layer_list[!sapply(layer_list, is.null)]
    if (length(layer_list) == 0) {
      cat("No valid rasters for this tile.\n", file = logf, append = TRUE)
      return(NULL)
    }
    
    # --- Stack and extract all cells (including NA) ---
    r_stack <- stack(layer_list)
    names(r_stack) <- sapply(layer_list, names)
    
    xy   <- xyFromCell(r_stack, 1:ncell(r_stack))
    vals <- as.data.frame(getValues(r_stack))
    combined <- cbind(data.frame(X = xy[,1], Y = xy[,2]), vals)
    
    ## Data-type specific cleanup
    if (data_type == "training") {
      # drop any bt.* columns if present
      combined <- combined %>% dplyr::select(-dplyr::starts_with("bt."))
      
      needed <- c("bathy_2004_filled", "bathy_2006_filled", "bathy_2010_filled",
                  "bathy_2015_filled", "bathy_2022_filled")
      if (all(needed %in% names(combined))) {
        combined <- combined %>%
          mutate(
            delta_bathy_2004_2006 = bathy_2006_filled - bathy_2004_filled,
            delta_bathy_2006_2010 = bathy_2010_filled - bathy_2006_filled,
            delta_bathy_2010_2015 = bathy_2015_filled - bathy_2010_filled,
            delta_bathy_2015_2022 = bathy_2022_filled - bathy_2015_filled
          )
      }
    } else if (data_type == "prediction") {
      pattern   <- "^(bathy|bpi_broad|bpi_fine|slope|terrain_classification|rugosity|curv_plan|curv_profile|curv_total|flowacc|flowdir|gradmag|shearproxy|slope_deg|tci)_\\d{4}"
      drop_cols <- grep(pattern, names(combined), value = TRUE)
      if (length(drop_cols) > 0) {
        combined <- combined %>% dplyr::select(-dplyr::all_of(drop_cols))
      }
    }
    
    combined$FID     <- seq_len(nrow(combined))
    combined$tile_id <- tile_name
    
    out_path <- file.path(tile_dir,
                          sprintf("%s_%s_clipped_data.fst", tile_name, data_type))
    write.fst(combined, out_path)
    
    cat(sprintf("[%s] ✅ Wrote %s (%d rows, %d cols)\n",
                format(Sys.time(), "%H:%M:%S"), basename(out_path),
                nrow(combined), ncol(combined)),
        file = logf, append = TRUE)
    
    return(list(tile_id = tile_name, n_points = nrow(combined)))
  }
  
  ## --- Run tiles (optionally in parallel) ---
  results <- list()
  
  if (parallel_tiles) {
    cores <- max(1, parallel::detectCores() - 1)
    log_message(sprintf("Running tiles in parallel using %d cores.", cores))
    cl <- makeCluster(cores)
    on.exit(stopCluster(cl), add = TRUE)
    
    registerDoParallel(cl)
    
    # make sure needed packages & objects are visible in workers
    clusterEvalQ(cl, {
      library(raster); library(sf); library(sp)
      library(dplyr); library(fst)
      NULL
    })
    
    results <- foreach(i = seq_len(nrow(grid_all)),
                       .packages = c("raster","sf","sp","dplyr","fst","tools"),
                       .export   = c("process_tile", "training_mask_tiles",
                                     "template_r", "all_files")) %dopar% {
                                       process_tile(grid_all[i, ])
                                     }
  } else {
    log_message("Running tiles sequentially (no parallel).")
    for (i in seq_len(nrow(grid_all))) {
      results[[i]] <- process_tile(grid_all[i, ])
      gc()
    }
  }
  
  ## --- Summarise ---
  results <- results[!sapply(results, is.null)]
  if (length(results) > 0) {
    result_df <- do.call(rbind, lapply(results, as.data.frame))
    write.csv(result_df,
              file.path(output_dir, "tile_processing_summary.csv"),
              row.names = FALSE)
    log_message("=== ALL TILES COMPLETE ===")
    log_message(sprintf("Summary written to %s",
                        file.path(output_dir, "tile_processing_summary.csv")))
  } else {
    log_message("No tiles processed successfully.")
  }
  
  invisible(results)
}

grid_out_raster_data <- function(master_grid_gpkg, # V3 Now makes a template, for no values > 0 using bt. bathy, this works well for the prediction data but will need tested for the  training data...  
                                 raster_dir,
                                 output_dir,
                                 mask_path,
                                 template_mask_path,
                                 data_type = c("training", "prediction"),
                                 relevant_tile_ids = NULL,
                                 parallel_tiles = TRUE) {
  
  ## --- Load packages ---
  library(sf); library(raster); library(dplyr)
  library(foreach); library(doParallel); library(fst); library(sp)
  
  data_type <- match.arg(data_type)
  log_message <- function(msg) {
    cat(sprintf("[%s] %s\n",
                format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg))
  }
  
  ## --- Read grid & template mask ---
  log_message("=== INITIAL SETUP & CRS VERIFICATION ===")
  grid_all <- sf::st_read(master_grid_gpkg, quiet = TRUE)
  
  if (!"tile_id" %in% names(grid_all)) {
    stop("master_grid_gpkg must contain a 'tile_id' column.")
  }
  
  # template grid for alignment
  template_r <- raster(template_mask_path)
  mask_r     <- raster(mask_path)  # currently unused, but kept for flexibility
  
  # align grid to template CRS if needed
  if (!compareCRS(grid_all, template_r)) {
    log_message("Grid CRS differs from template; reprojecting grid to template CRS.")
    grid_all <- st_transform(grid_all, crs = crs(template_r))
  }
  
  # subset tiles if requested
  if (!is.null(relevant_tile_ids)) {
    grid_all <- grid_all[grid_all$tile_id %in% relevant_tile_ids, ]
  }
  
  # --- Automatically filter to true training tiles ---
  if (data_type == "training") {
    training_tiles_gpkg <- gsub("master_grid_canonical.gpkg",
                                "Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg",
                                master_grid_gpkg)
    
    if (file.exists(training_tiles_gpkg)) {
      orig_train <- sf::st_read(training_tiles_gpkg, quiet = TRUE)
      true_train_ids <- orig_train$tile_id
      
      log_message(sprintf(
        "Filtering training run to original %d training tiles.",
        length(true_train_ids)
      ))
      
      # restrict grid_all BEFORE processing
      grid_all <- grid_all[grid_all$tile_id %in% true_train_ids, ]
      
      if (nrow(grid_all) == 0)
        stop("Training mode: No matching tiles found in the canonical grid.")
    } else {
      log_message("⚠ Could not locate original training grid — NOT filtering.")
    }
  }
  
  if (nrow(grid_all) == 0) stop("No matching tiles to process.")
  
  ## --- File list ---
  all_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  if (length(all_files) == 0) stop("No .tif files found in raster_dir.")
  log_message(sprintf("Processing %d tile(s) for '%s' using %d rasters.",
                      nrow(grid_all), data_type, length(all_files)))
  
  ## --- Preprocess training mask OUTSIDE parallel (critical for stability) ---
  # training_mask_tiles: named list, one logical vector per tile_id
  training_mask_tiles <- NULL
  
  if (data_type == "training") {
    training_mask_path <- gsub("prediction", "training", mask_path)
    if (file.exists(training_mask_path)) {
      log_message("Preprocessing training mask (once, outside parallel).")
      
      # load and align to template grid
      tm <- raster(training_mask_path)
      if (!compareCRS(tm, template_r)) {
        log_message("Reprojecting training mask to template CRS...")
        tm <- projectRaster(tm, template_r, method = "ngb")
      } else {
        # still resample to guarantee same grid if needed
        tm <- resample(tm, template_r, method = "ngb")
      }
      
      training_mask_tiles <- list()
      
      for (i in seq_len(nrow(grid_all))) {
        tile    <- grid_all[i, ]
        tname   <- as.character(tile$tile_id[1])
        tile_sp <- as(tile, "Spatial")
        
        # tile's template subset
        tile_template <- crop(template_r, extent(tile_sp))
        
        # crop training mask to same tile extent and resample onto tile_template
        tm_crop <- crop(tm, extent(tile_template))
        tm_tile <- resample(tm_crop, tile_template, method = "ngb")
        
        # store logical vector: TRUE for inside-training, FALSE elsewhere
        vals <- tm_tile[]
        training_mask_tiles[[tname]] <- !is.na(vals) & vals != 0
      }
      
      log_message("Training mask preprocessed for all tiles.")
      
    } else {
      log_message("⚠ No training mask raster found; proceeding without it.")
    }
  }
  
  ## --- Per-tile processor ---
  process_tile <- function(tile_row) {
    tile     <- tile_row
    tile_name <- as.character(tile$tile_id[1])
    
    tile_dir <- file.path(output_dir, tile_name)
    dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
    logf <- file.path(tile_dir, paste0("tile_log_", tile_name, ".txt"))
    cat(sprintf("[%s] === PROCESSING TILE %s ===\n",
                format(Sys.time(), "%H:%M:%S"), tile_name),
        file = logf, append = TRUE)
    
    tile_sp      <- as(tile, "Spatial")
    template_crop <- crop(template_r, extent(tile_sp), snap = "out")
    
    bt_bathy_tile <- NULL
    
    
    # --- Load and align rasters for this tile ---
    layer_list <- lapply(all_files, function(rf) {
      nm <- tools::file_path_sans_ext(basename(rf))
      
      tryCatch({
        r <- raster(rf)
        
        if (!compareCRS(r, template_r)) {
          r <- projectRaster(r, template_r, method = "bilinear")
        }
        # ORDER IS IMPORTANT - INTERPOLATION BEFORE CROPPING! 
        # define interp method (only bathy bilinear, everything else ngb)
        interp_method <- if (grepl("^bathy_|^bt\\.bathy$", nm)) "bilinear" else "ngb"
        
        # resample directly onto the tile template grid (no pre-crop)
        r_align <- resample(r, template_crop, method = interp_method)
        
        if (nm == "bt.bathy") {
          bt_bathy_tile <<- r_align
        }
        
        # Apply marine-only mask using bt.bathy
        if (!is.null(bt_bathy_tile)) {
          bathy_vals <- values(bt_bathy_tile)
          
          # Marine = bathy <= 0
          marine_mask <- !is.na(bathy_vals) & bathy_vals <= 0
          
          vals <- values(r_align)
          vals[!marine_mask] <- NA
          r_align[] <- vals
        }
        
        
        # training mask (unchanged)
        if (!is.null(training_mask_tiles) && data_type == "training") {
          m_vals <- training_mask_tiles[[tile_name]]
          if (!is.null(m_vals)) {
            if (length(m_vals) == ncell(r_align)) {
              vals_r <- r_align[]
              vals_r[!m_vals] <- NA
              r_align[] <- vals_r
            } else {
              cat(sprintf(
                "WARNING: mask length mismatch for tile %s (mask: %d, raster: %d)\n",
                tile_name, length(m_vals), ncell(r_align)
              ), file = logf, append = TRUE)
            }
          }
        }
        
        names(r_align) <- nm
        r_align
      }, error = function(e) {
        cat(sprintf("Error in %s: %s\n", nm, e$message), file = logf, append = TRUE)
        NULL
      })
    })
    
    
    layer_list <- layer_list[!sapply(layer_list, is.null)]
    if (length(layer_list) == 0) {
      cat("No valid rasters for this tile.\n", file = logf, append = TRUE)
      return(NULL)
    }
    
    # --- Stack and extract all cells (including NA) ---
    r_stack <- stack(layer_list)
    names(r_stack) <- sapply(layer_list, names)
    
    xy   <- xyFromCell(r_stack, 1:ncell(r_stack))
    vals <- as.data.frame(getValues(r_stack))
    combined <- cbind(data.frame(X = xy[,1], Y = xy[,2]), vals)
    
    ## Data-type specific cleanup
    if (data_type == "training") {
      # drop any bt.* columns if present
      combined <- combined %>% dplyr::select(-dplyr::starts_with("bt."))
      
      needed <- c("bathy_2004_filled", "bathy_2006_filled", "bathy_2010_filled",
                  "bathy_2015_filled", "bathy_2022_filled")
      if (all(needed %in% names(combined))) {
        combined <- combined %>%
          mutate(
            delta_bathy_2004_2006 = bathy_2006_filled - bathy_2004_filled,
            delta_bathy_2006_2010 = bathy_2010_filled - bathy_2006_filled,
            delta_bathy_2010_2015 = bathy_2015_filled - bathy_2010_filled,
            delta_bathy_2015_2022 = bathy_2022_filled - bathy_2015_filled
          )
      }
    } else if (data_type == "prediction") {
      pattern   <- "^(bathy|bpi_broad|bpi_fine|slope|terrain_classification|rugosity|curv_plan|curv_profile|curv_total|flowacc|flowdir|gradmag|shearproxy|slope_deg|tci)_\\d{4}"
      drop_cols <- grep(pattern, names(combined), value = TRUE)
      if (length(drop_cols) > 0) {
        combined <- combined %>% dplyr::select(-dplyr::all_of(drop_cols))
      }
    }
    
    combined$FID     <- seq_len(nrow(combined))
    combined$tile_id <- tile_name
    
    out_path <- file.path(tile_dir,
                          sprintf("%s_%s_clipped_data.fst", tile_name, data_type))
    write.fst(combined, out_path)
    
    cat(sprintf("[%s] ✅ Wrote %s (%d rows, %d cols)\n",
                format(Sys.time(), "%H:%M:%S"), basename(out_path),
                nrow(combined), ncol(combined)),
        file = logf, append = TRUE)
    
    return(list(tile_id = tile_name, n_points = nrow(combined)))
  }
  
  ## --- Run tiles (optionally in parallel) ---
  results <- list()
  
  if (parallel_tiles) {
    cores <- max(1, parallel::detectCores() - 1)
    log_message(sprintf("Running tiles in parallel using %d cores.", cores))
    cl <- makeCluster(cores)
    on.exit(stopCluster(cl), add = TRUE)
    
    registerDoParallel(cl)
    
    # make sure needed packages & objects are visible in workers
    clusterEvalQ(cl, {
      library(raster); library(sf); library(sp)
      library(dplyr); library(fst)
      NULL
    })
    
    results <- foreach(i = seq_len(nrow(grid_all)),
                       .packages = c("raster","sf","sp","dplyr","fst","tools"),
                       .export   = c("process_tile", "training_mask_tiles",
                                     "template_r", "all_files")) %dopar% {
                                       process_tile(grid_all[i, ])
                                     }
  } else {
    log_message("Running tiles sequentially (no parallel).")
    for (i in seq_len(nrow(grid_all))) {
      results[[i]] <- process_tile(grid_all[i, ])
      gc()
    }
  }
  
  ## --- Summarise ---
  results <- results[!sapply(results, is.null)]
  if (length(results) > 0) {
    result_df <- do.call(rbind, lapply(results, as.data.frame))
    write.csv(result_df,
              file.path(output_dir, "tile_processing_summary.csv"),
              row.names = FALSE)
    log_message("=== ALL TILES COMPLETE ===")
    log_message(sprintf("Summary written to %s",
                        file.path(output_dir, "tile_processing_summary.csv")))
  } else {
    log_message("No tiles processed successfully.")
  }
  
  invisible(results)
}



# function call 
grid_out_raster_data(
  master_grid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/master_grid_canonical.gpkg",
  raster_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/processed",
  output_dir  = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles",
  mask_path   = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif",
  template_mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif",  # reference grid
  data_type = "prediction",
  # relevant_tile_ids = "BH4S556X_3", # Example training tile
  parallel_tiles = F
)

grid_out_raster_data(
  master_grid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/master_grid_canonical.gpkg",
  raster_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/training/processed",
  output_dir  = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles",
  mask_path   = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif",
  template_mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif",  # reuse alignment alignment from prediction mask to get same number of rows
  data_type = "training",
  relevant_tile_ids = "BH4S556X_3", # Example training tile BH4S556X_3
  parallel_tiles = F
)



# 
# library(raster)
# library(dplyr)
# 
# count_na_fast <- function(r) {
#   bs <- blockSize(r)
#   
#   na_count <- 0L
#   n_total  <- ncell(r)
#   
#   for (i in seq_len(bs$n)) {
#     v <- getValues(r, row = bs$row[i], nrows = bs$nrows[i])
#     na_count <- na_count + sum(is.na(v))
#   }
#   
#   list(
#     na_count = na_count,
#     na_frac  = na_count / n_total
#   )
# }
# 
# raster_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/processed"
# 
# files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
# 
# na_audit <- lapply(files, function(f) {
#   
#   nm <- tools::file_path_sans_ext(basename(f))
#   r  <- raster(f)
#   
#   na <- count_na_fast(r)
#   
#   tibble(
#     raster   = nm,
#     ncell    = ncell(r),
#     na_count = na$na_count,
#     na_frac  = na$na_frac,
#     res_x    = res(r)[1],
#     res_y    = res(r)[2],
#     crs      = as.character(crs(r))
#   )
# }) %>%
#   bind_rows() %>%
#   arrange(desc(na_frac))
# 
# na_audit




# Check alignment of your test tiles match # same number of rows
train <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/BH4S556X_3_training_clipped_data.fst")
pred  <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles/BH4S556X_3/BH4S556X_3_prediction_clipped_data.fst")
identical(train[,c("X","Y")], pred[,c("X","Y")])   # should be TRUE ✅


#' Check Alignment Between Training and Prediction Tiles
#'
#' Verifies that each pair of training/prediction tile data files
#' have identical coordinate grids (X,Y), row counts, and extents.
#'
#' @param training_dir Path to directory containing training tile folders (each with *_training_clipped_data.fst).
#' @param prediction_dir Path to directory containing prediction tile folders (each with *_prediction_clipped_data.fst).
#' @param output_csv Optional path for a CSV summary output.
#' @param check_templates Logical; if TRUE, also check .tif template extents if present.
#' @return A data.frame summary invisibly.
#' @import fst raster

check_tile_alignment_v2 <- function(
    training_dir,
    prediction_dir,
    output_csv = NULL,
    verbose = TRUE
){
  library(fst)
  library(dplyr)
  
  p <- function(...) if (verbose) cat("[", format(Sys.time(), "%H:%M:%S"), "]", ..., "\n")
  
  p("=== Alignment Check: Training vs Prediction ===")
  
  train_tiles <- list.dirs(training_dir, recursive = FALSE, full.names = TRUE)
  pred_tiles  <- list.dirs(prediction_dir, recursive = FALSE, full.names = TRUE)
  
  train_ids <- basename(train_tiles)
  pred_ids  <- basename(pred_tiles)
  common_tiles <- intersect(train_ids, pred_ids)
  
  if (length(common_tiles) == 0)
    stop("No overlapping tile folders found.")
  
  results <- list()
  
  for (tile in common_tiles) {
    
    train_file <- list.files(file.path(training_dir, tile),
                             pattern = "_training_clipped_data\\.fst$", full.names = TRUE)
    pred_file  <- list.files(file.path(prediction_dir, tile),
                             pattern = "_prediction_clipped_data\\.fst$", full.names = TRUE)
    
    if (length(train_file)==0 || length(pred_file)==0) {
      results[[tile]] <- data.frame(
        tile_id = tile,
        status = "missing_files"
      )
      next
    }
    
    train <- read.fst(train_file)
    pred  <- read.fst(pred_file)
    
    # key checks
    coords_match  <- identical(train[,c("X","Y")], pred[,c("X","Y")])
    nrow_match    <- (nrow(train) == nrow(pred))
    
    # These will ALWAYS be different and should NOT imply misalignment
    na_train <- sum(is.na(train))
    na_pred  <- sum(is.na(pred))
    
    status <- if (coords_match && nrow_match) "aligned" else "mismatch"
    
    results[[tile]] <- data.frame(
      tile_id         = tile,
      coords_match    = coords_match,
      nrow_training   = nrow(train),
      nrow_prediction = nrow(pred),
      nrow_match      = nrow_match,
      na_training_sum = na_train,
      na_prediction_sum = na_pred,
      status = status
    )
  }
  
  final <- bind_rows(results)
  
  if (!is.null(output_csv)) {
    write.csv(final, output_csv, row.names = FALSE)
    p("Summary written to:", output_csv)
  }
  
  return(final)
}


# Directory 
training_dir   <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"


# Function Call
check_tile_alignment_v2(
  training_dir   = training_dir,
  prediction_dir = prediction_dir,
  output_csv     = file.path(dirname(training_dir), "tile_alignment_summary.csv")
)

 

 # Location of data directories 
input_raw_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/raw"
input_for_train_std <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/processed"
output_proc_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Training/processed"
output_proc_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Model_variables/Prediction/processed"
output_mask_pred_utm <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif"
output_mask_train_utm <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/training.mask.UTM17_8m.tif"

master_grid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/master_grid_canonical.gpkg"

prediction_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
training_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
prediction_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"
training_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles"





# ==============================================================================
#
#           Data Transformation for Time-Series Modeling (with Verification)
#
# ==============================================================================
#
# Purpose:
# This script transforms wide-format grid tile data into a long-format structure
# suitable for time-series modeling.
#
# Key Improvement (Robustness):
# Includes a "write-then-verify" step. After each file is written in parallel,
# it is immediately read back to check for corruption. If a file is corrupt,
# it is automatically deleted and a warning is issued, preventing downstream
# errors and application crashes.
#
# ==============================================================================

library(data.table)
library(stringr)
library(dplyr)
library(fst)
library(parallel)
library(foreach)
library(doParallel)


# --- Core Transformation Functions ---

# --- Angle Transformation Helper ---
transform_flowdir_cols <- function(dt) {
  
  # detect any columns that look like flow direction (0–360°)
  flow_cols <- grep("flowdir", names(dt), value = TRUE)
  
  if (length(flow_cols) == 0) return(dt)
  
  for (col in flow_cols) {
    new_sin <- paste0(col, "_sin")
    new_cos <- paste0(col, "_cos")
    
    dt[, (new_sin) := sin(get(col) * pi/180)]
    dt[, (new_cos) := cos(get(col) * pi/180)]
  }
  
  # optional: remove original degree columns
  dt[, (flow_cols) := NULL]
  
  return(dt)
}

#' Process a single training data tile into long-format year-pair dataframes
process_training_df <- function(df) {
  
  if (!is.data.table(df)) {
    setDT(df)
  }
  
  # --- NEW: transform flow direction before doing anything ---
  df <- transform_flowdir_cols(df)
  #---
  
  
  year_pairs <- list(
    c("2004", "2006"), c("2006", "2010"),
    c("2010", "2015"), c("2015", "2022")
  )
  
  all_cols <- names(df)
  potential_state_cols <- all_cols[str_detect(all_cols, "_\\d{4}") & !str_detect(all_cols, "_\\d{4}_\\d{4}$")]
  
  col_meta <- data.table(
    colname = potential_state_cols,
    year = as.integer(str_extract(potential_state_cols, "(?<=_)\\d{4}"))
  )
  col_meta[, var_base := str_replace(colname, paste0("_", year), "")]
  
  year_pair_dfs <- list()
  
  for (pair in year_pairs) {
    y0 <- as.integer(pair[1])
    y1 <- as.integer(pair[2])
    pair_name <- paste(y0, y1, sep = "_")
    
    cols_t_meta <- col_meta[year == y0]
    cols_t1_meta <- col_meta[year == y1]
    
    common_vars <- sort(intersect(cols_t_meta$var_base, cols_t1_meta$var_base))
    
    target_base_var <- "bathy_filled"
    if (!target_base_var %in% common_vars) {
      next
    }
    
    cols_t_exist <- cols_t_meta[match(common_vars, var_base), colname]
    cols_t1_exist <- cols_t1_meta[match(common_vars, var_base), colname]
    
    new_names_t <- paste0(common_vars, "_t")
    new_names_t1 <- paste0(common_vars, "_t1")
    
    forcing_pattern <- paste0("(", y0, "_", y1, ")$")
    forcing_cols <- grep(forcing_pattern, names(df), value = TRUE)
    
    delta_bathy_col <- paste0("delta_bathy_", pair_name)
    forcing_cols <- setdiff(forcing_cols, delta_bathy_col)
    
    
    static_vars <- c("grain_size_layer", "prim_sed_layer", "survey_end_date")
    static_cols <- static_vars[static_vars %in% names(df)]
    id_cols <- c("X", "Y", "FID", "tile_id")
    
    cols_to_grab_from_df <- c(id_cols, cols_t_exist, cols_t1_exist, forcing_cols, static_cols)
    if (delta_bathy_col %in% names(df)) {
      cols_to_grab_from_df <- c(cols_to_grab_from_df, delta_bathy_col)
    }
    
    missing_cols <- setdiff(cols_to_grab_from_df, names(df))
    if(length(missing_cols) > 0) {
      next
    }
    
    pair_df <- df[, unique(cols_to_grab_from_df), with = FALSE]
    
    setnames(pair_df, old = c(cols_t_exist, cols_t1_exist), new = c(new_names_t, new_names_t1))
    
    if (delta_bathy_col %in% names(pair_df)) {
      setnames(pair_df, old = delta_bathy_col, new = "delta_bathy")
    }
    
    pair_df[, `:=`(year_t = y0, year_t1 = y1)]
    
    # Compute only bathymetry delta
    if (all(c("bathy_t", "bathy_t1") %in% names(pair_df))) {
      pair_df[, delta_bathy := bathy_t1 - bathy_t]
    }
    
    
    filled_cols <- names(pair_df)[str_detect(names(pair_df), "_filled")]
    if (length(filled_cols) > 0) {
      clean_names <- str_remove(filled_cols, "_filled")
      setnames(pair_df, old = filled_cols, new = clean_names)
    }
    
    target_col <- "bathy_t1"
    predictor_cols_t <- names(pair_df)[str_ends(names(pair_df), "_t")]
    predictor_cols_delta <- names(pair_df)[str_starts(names(pair_df), "delta_")]
    
    id_cols_final <- c("X", "Y", "FID", "tile_id", "year_t", "year_t1")
    
    final_cols <- c(id_cols_final, target_col, predictor_cols_t, predictor_cols_delta, forcing_cols, static_cols)
    final_pair_df <- pair_df[, intersect(final_cols, names(pair_df)), with = FALSE]
    
    year_pair_dfs[[pair_name]] <- final_pair_df
  }
  
  return(year_pair_dfs)
}

#' Process a single prediction data tile into long-format year-pair dataframes
process_prediction_df <- function(df) {
  
  if (!is.data.table(df)) {
    setDT(df)
  }
  
  
  # --- NEW: transform flow direction before any renaming ---
  df <- transform_flowdir_cols(df)
  #---
  df_copy <- copy(df)
  bt_cols <- grep("^bt\\.", names(df_copy), value = TRUE)
  
  new_t_names <- str_replace(bt_cols, "^bt\\.", "")
  new_t_names <- paste0(new_t_names, "_t")
  
  setnames(df_copy, old = bt_cols, new = new_t_names)
  
  year_pairs <- list(
    c("2004", "2006"), c("2006", "2010"),
    c("2010", "2015"), c("2015", "2022")
  )
  
  year_pair_dfs <- list()
  
  for (pair in year_pairs) {
    y0 <- pair[1]
    y1 <- pair[2]
    pair_name <- paste(y0, y1, sep = "_")
    
    predictor_cols_t <- new_t_names
    
    forcing_pattern <- paste0("(", y0, "_", y1, ")$")
    forcing_cols <- grep(forcing_pattern, names(df_copy), value = TRUE)
    
    static_vars <- c("grain_size_layer", "prim_sed_layer", "survey_end_date")
    static_cols <- static_vars[static_vars %in% names(df_copy)]
    
    id_cols <- c("X", "Y", "FID", "tile_id")
    
    final_cols <- c(id_cols, predictor_cols_t, forcing_cols, static_cols)
    
    pred_df_pair <- df_copy[, unique(final_cols), with = FALSE]
    
    year_pair_dfs[[pair_name]] <- pred_df_pair
  }
  
  return(year_pair_dfs)
}


#' Wrapper to process and save all tiles in a directory with file verification
transform_and_save_tiles <- function(base_dir, 
                                     mode = c("training", "prediction"), 
                                     parallel_run = TRUE,
                                     max_cores = parallel::detectCores() - 1) {
  mode <- match.arg(mode)
  
  cat("\n🚀 Starting data transformation for mode:", mode, "\n")
  
  file_pattern <- paste0("_", mode, "_clipped_data\\.fst$")
  files_to_process <- list.files(base_dir, pattern = file_pattern, full.names = TRUE, recursive = TRUE)
  
  if (length(files_to_process) == 0) {
    cat("⚠️ No files found to process. Exiting.\n")
    return(invisible(NULL))
  }
  
  cat(paste("Found", length(files_to_process), "tiles to process.\n"))
  
  # --- Helper function for file verification ---
  verify_fst_file <- function(file_path) {
    tryCatch({
      # A quick read of metadata or first row is enough to check integrity
      fst::metadata_fst(file_path)
      return(TRUE)
    }, error = function(e) {
      return(FALSE)
    })
  }
  
  # --- PARALLEL EXECUTION LOGIC ---
  if (parallel_run) {
    num_cores <- min(length(files_to_process), max_cores)
    cat(paste("Setting up parallel processing with", num_cores, "cores.\n"))
    cl <- makeCluster(num_cores)
    registerDoParallel(cl)
    
    on.exit(stopCluster(cl), add = TRUE)
    
    foreach(
      f_path = files_to_process, 
      .packages = c("fst", "data.table", "stringr"),
      .export = c("process_training_df", "process_prediction_df", 
                  "transform_flowdir_cols", "verify_fst_file") # Export the flow direction helper too
    ) %dopar% {
      
      tile_name <- stringr::str_extract(basename(f_path), "^[^_]+")
      output_tile_dir <- dirname(f_path)
      
      df <- read_fst(f_path, as.data.table = TRUE)
      
      # Determine which processing function to use
      process_func <- if (mode == "training") process_training_df else process_prediction_df
      processed_list <- process_func(df)
      
      # Loop through and save each generated dataframe
      for (pair_name in names(processed_list)) {
        file_suffix <- if (mode == "training") "_long.fst" else "_prediction_long.fst"
        out_name <- paste0(tile_name, "_", pair_name, file_suffix)
        out_path <- file.path(output_tile_dir, out_name)
        
        # Write the file
        write_fst(processed_list[[pair_name]], out_path)
        
        # --- VERIFICATION STEP ---
        if (!verify_fst_file(out_path)) {
          # Use a thread-safe way to message, or just know it will print to worker logs
          warning_msg <- paste0("CRITICAL WARNING: Corrupt file detected and deleted -> ", out_name, " in tile folder ", tile_name, "\n")
          cat(warning_msg) # This will print in the worker's console/log
          file.remove(out_path)
        }
      }
      return(NULL)
    }
    
  } else {
    # --- SEQUENTIAL EXECUTION LOGIC ---
    cat("Running in sequential mode.\n")
    for (f_path in files_to_process) {
      tile_name <- str_extract(basename(f_path), "^[^_]+")
      cat(paste("Processing tile:", tile_name, "\n"))
      
      output_tile_dir <- dirname(f_path)
      df <- read_fst(f_path, as.data.table = TRUE)
      
      process_func <- if (mode == "training") process_training_df else process_prediction_df
      processed_list <- process_func(df)
      
      for (pair_name in names(processed_list)) {
        file_suffix <- if (mode == "training") "_long.fst" else "_prediction_long.fst"
        out_name <- paste0(tile_name, "_", pair_name, file_suffix)
        out_path <- file.path(output_tile_dir, out_name)
        
        write_fst(processed_list[[pair_name]], out_path)
        
        # --- VERIFICATION STEP ---
        if (!verify_fst_file(out_path)) {
          cat(paste0("  - ❌ CRITICAL WARNING: Corrupt file detected and deleted -> ", out_name, "\n"))
          file.remove(out_path)
        }
      }
    }
  }
  
  cat("\n🎉 Transformation complete. Files saved and verified within their respective tile folders.\n")
}



# TESTING ONLY 
# ==============================================================================
#
#           Test Script for Data Transformation on a Single Tile
#
# ==============================================================================
#
# Purpose:
# This script is designed to test the data transformation functions on a single
# specified grid tile for both training and prediction modes. It provides a
# safe and quick way to verify the output before running the process on all tiles.
#
# Instructions:
# 1. Make sure this script is in the same directory as 'data_transformation.R'.
# 2. Update the `tile_to_test` variable to the ID of the tile you wish to process.
# 3. Update the base directory paths if they differ from the main script.
# 4. Run the entire script. A new '_test_output' folder will be created in
#    each base directory with the transformed files for the single tile.
#
# ==============================================================================



# --- 2. CONFIGURATION ---

# Specify the ID of the single tile you want to process for the test
tile_to_test <- "BH4S556X_3" # Example training tile with good data
# tile_to_test <- "BH4RZ577_2" # Example prediction tile

# Define the base directories for your training and prediction data
training_base_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles"
prediction_base_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"

# Define temporary output directories to keep test files separate
training_output_dir <- file.path(training_base_dir, "_test_output")
prediction_output_dir <- file.path(prediction_base_dir, "_test_output")
dir.create(training_output_dir, showWarnings = FALSE)
dir.create(prediction_output_dir, showWarnings = FALSE)


# --- 3. TEST TRAINING DATA TRANSFORMATION (SINGLE TILE) ---

cat("\n Testing TRAINING data transformation for tile:", tile_to_test, "\n")

# Construct the full path to the input file
training_file_path <- file.path(training_base_dir, tile_to_test, paste0(tile_to_test, "_training_clipped_data.fst"))

if (file.exists(training_file_path)) {
  # Read the single data file
  train_df <- read_fst(training_file_path, as.data.table = TRUE)
  
  # Process the dataframe (this returns a list of dataframes, one per year pair)
  processed_list_train <- process_training_df(train_df)
  
  # Save each resulting dataframe to the test output directory
  for (pair_name in names(processed_list_train)) {
    out_name <- paste0(tile_to_test, "_", pair_name, "_long.fst")
    out_path <- file.path(training_output_dir, out_name)
    write_fst(processed_list_train[[pair_name]], out_path)
    cat("  ✅ Saved training output:", out_name, "\n")
  }
} else {
  cat("  ⚠️ WARNING: Training file not found at:", training_file_path, "\n")
}


# --- 4. TEST PREDICTION DATA TRANSFORMATION ---

cat("\n Testing PREDICTION data transformation for tile:", tile_to_test, "\n")

# Construct the full path to the input file
prediction_file_path <- file.path(prediction_base_dir, tile_to_test, paste0(tile_to_test, "_prediction_clipped_data.fst"))

if (file.exists(prediction_file_path)) {
  # Read the single data file
  pred_df <- read_fst(prediction_file_path, as.data.table = TRUE)
  
  # Process the dataframe (returns a list of dataframes, one for each year pair)
  processed_list_pred <- process_prediction_df(pred_df)
  
  # Save each resulting dataframe to the test output directory
  for (pair_name in names(processed_list_pred)) {
    out_name <- paste0(tile_to_test, "_", pair_name, "_prediction_long.fst")
    out_path <- file.path(prediction_output_dir, out_name)
    write_fst(processed_list_pred[[pair_name]], out_path)
    cat("  ✅ Saved prediction output:", out_name, "\n")
  }
} else {
  cat("  ⚠️ WARNING: Prediction file not found at:", prediction_file_path, "\n")
}

cat("\n🎉 Test complete. Check the '_test_output' folders in your training and prediction directories.\n")


# ==============================================================================
#
#           Execute Full Data Transformation (Training & Prediction) - convert to long
#
# ==============================================================================
#
# Purpose:
# This script runs the full data transformation workflow on all grid tiles for
# both the training and prediction datasets. It uses parallel processing to
# speed up the operation.
#
# Instructions:
# 1. Ensure 'data_transformation.R' is in the same directory as this script.
# 2. Verify the base directory paths below are correct.
# 3. Run the entire script to process all data.
#
# ==============================================================================

# --- 1. Load Libraries & Functions ---

library(parallel)
library(foreach)
library(doParallel)

# Load the core transformation functions
# source("data_transformation.R")

# --- 2. Define File Paths ---

training_base_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles"
prediction_base_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"

# --- 3. Execute Transformation for TRAINING Data ---

# This will find all `_training_clipped_data.fst` files in the subdirectories,
# process them in parallel, and save the long-format output to a new folder:
# .../training_data_grid_tiles/long_format/
transform_and_save_tiles(
  base_dir = training_base_dir,
  mode = "training",
  parallel_run = TRUE  # Use TRUE to run in parallel
)

# --- 4. Execute Transformation for PREDICTION Data ---

# This will find all `_prediction_clipped_data.fst` files, process them,
# and save the long-format output to a new folder:
# .../Prediction_data_grid_tiles/long_format/
transform_and_save_tiles(
  base_dir = prediction_base_dir,
  mode = "prediction",
  parallel_run = TRUE  # Use TRUE to run in parallel
)

# --- 5. Clean Up ---
# Close any stray parallel processing connections if they exist
if (exists("cl") && inherits(cl, "cluster")) {
  stopCluster(cl)
}
closeAllConnections()

cat("\n\n✅ All processing jobs have been launched.\n")





# ==============================================================================
#
#       Boruta Predictor Selection with Simplification & Consistency
#
# ==============================================================================
#
# Purpose:
# This script runs a memory-safe Boruta feature selection with key
# methodological improvements:
#   1. Simplification: It only considers delta variables for core predictors.
#   2. Consistency: It enforces a "parent-child" rule for neighborhood stats.
#   3. Exclusion: It explicitly removes non-driver variables like 'survey_end_date'.
#
# ==============================================================================

# --- Load Libraries ---
library(Boruta)
library(dplyr)
library(data.table)
library(fst)
library(stringr)
library(tibble)
library(parallel)
library(foreach)
library(doParallel)

# Helper Function
# ========================================================================
#  ENFORCE HIERARCHICAL & FLOW-DIRECTION CONSISTENCY
# ========================================================================

enforce_predictor_consistency <- function(confirmed_preds, all_predictors) {
  
  # (A) FLOW DIRECTION: sin/cos pairing ----------------------------
  flow_pairs <- list(
    flowdir_sin_t = "flowdir_cos_t",
    flowdir_cos_t = "flowdir_sin_t"
  )
  
  for (fp in names(flow_pairs)) {
    if (fp %in% confirmed_preds && flow_pairs[[fp]] %in% all_predictors) {
      confirmed_preds <- unique(c(confirmed_preds, flow_pairs[[fp]]))
    }
  }
  
  # (B) HIGH-RES GEOMORPHOLOGY PARENT–CHILD RULES ------------------
  parent_map <- list(
    # curvature → slope
    curv_total_t   = "slope_t",
    curv_plan_t    = "slope_t",
    curv_profile_t = "slope_t",
    
    # gradient magnitude → slope
    gradmag_t = "slope_t",
    
    # BPI fine
    bpi_fine_mean3_t = "bpi_fine_t",
    bpi_fine_sd3_t   = "bpi_fine_t",
    
    # BPI broad
    bpi_broad_mean3_t = "bpi_broad_t",
    bpi_broad_sd3_t   = "bpi_broad_t",
    
    # Slope neighborhoods → slope
    slope_mean3_t = "slope_t",
    slope_sd3_t   = "slope_t",
    
    # Rugosity neighborhoods → rugosity
    rugosity_mean3_t = "rugosity_t",
    rugosity_sd3_t   = "rugosity_t",
    
    # Flow accumulation / TCI → gradient
    tci_t     = "gradmag_t",
    flowacc_t = "gradmag_t"
  )
  
  for (child in names(parent_map)) {
    parent <- parent_map[[child]]
    if (child %in% confirmed_preds && parent %in% all_predictors) {
      confirmed_preds <- unique(c(confirmed_preds, parent))
    }
  }
  
  # (C) General neighborhood parent rule (mean/sd → base) ----------
  neighborhood_stats <- confirmed_preds[stringr::str_detect(confirmed_preds, "_mean\\d|_sd\\d")]
  
  if (length(neighborhood_stats) > 0) {
    parent_vars    <- stringr::str_replace(neighborhood_stats, "_mean\\d|_sd\\d", "")
    parents_to_add <- intersect(parent_vars, all_predictors)
    confirmed_preds <- unique(c(confirmed_preds, parents_to_add))
  }
  
  unique(confirmed_preds)
}

# ==============================================================================
#   MAIN BORUTA SELECTION FUNCTION
# ==============================================================================
run_boruta_selection_by_pair <- function(training_base_dir,
                                         year_pairs,
                                         sample_size = 1000000,
                                         max_runs    = 100,
                                         chunk_size  = 5,
                                         max_cores   = parallel::detectCores() - 1) {
  
  cat("\n🚀 Starting Boruta Selection with Hierarchical Consistency...\n")
  
  num_cores <- min(length(year_pairs), max_cores)
  cat(paste("Setting up parallel cluster with", num_cores, "cores.\n"))
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  on.exit(stopCluster(cl), add = TRUE)
  
  # Export helper(s) explicitly to workers
  clusterExport(cl,
                varlist = c("enforce_predictor_consistency"),
                envir   = environment())
  
  all_results_paths <- foreach(
    pair = year_pairs,
    .packages = c("Boruta", "data.table", "fst", "stringr", "dplyr", "tibble"),
    .combine = 'c'
  ) %dopar% {
    
    log_message_local <- function(msg) {
      cat(paste0(Sys.time(), " [Core ", Sys.getpid(), "] ", msg, "\n"))
    }
    
    log_message_local(paste0("Processing Year Pair: ", pair))
    
    # 1. Find and chunk data files ---------------------------------
    file_pattern <- paste0("_", pair, "_long\\.fst$")
    files_to_process <- list.files(
      path      = training_base_dir,
      pattern   = file_pattern,
      recursive = TRUE,
      full.names = TRUE
    )
    if (!length(files_to_process)) {
      log_message_local(paste0("⚠️ No data files found for pair ", pair, ". Skipping."))
      return(NULL)
    }
    file_chunks <- split(files_to_process,
                         ceiling(seq_along(files_to_process) / chunk_size))
    
    # 2. Sample from chunks (memory-safe) --------------------------
    log_message_local(paste0("Sampling from ", length(files_to_process),
                             " files in ", length(file_chunks), " chunks..."))
    list_of_samples   <- list()
    samples_per_chunk <- max(100, ceiling(sample_size / length(file_chunks)))
    
    for (i in seq_along(file_chunks)) {
      chunk_files <- file_chunks[[i]]
      chunk_dfs <- lapply(
        chunk_files,
        function(f) tryCatch(
          fst::read_fst(f, as.data.table = TRUE),
          error = function(e) NULL
        )
      )
      chunk_dfs <- chunk_dfs[!vapply(chunk_dfs, is.null, logical(1))]
      if (!length(chunk_dfs)) next
      
      chunk_df <- data.table::rbindlist(chunk_dfs, use.names = TRUE, fill = TRUE)
      if (nrow(chunk_df) > 0) {
        sample_n <- min(nrow(chunk_df), samples_per_chunk)
        set.seed(123 + i)
        list_of_samples[[i]] <- chunk_df[sample(.N, sample_n)]
      }
      rm(chunk_dfs, chunk_df); invisible(gc())
    }
    
    if (!length(list_of_samples)) {
      log_message_local(paste0("⚠️ No valid data after sampling for pair ", pair, ". Skipping."))
      return(NULL)
    }
    
    combined_df <- data.table::rbindlist(list_of_samples, use.names = TRUE, fill = TRUE)
    
    # 3. Prepare data for Boruta -----------------------------------
    response_var <- "bathy_t1"
    id_vars      <- c("X", "Y", "FID", "tile_id", "year_t", "year_t1")
    vars_to_exclude <- c(response_var, id_vars, "survey_end_date")
    
    all_potential_predictors <- setdiff(names(combined_df), vars_to_exclude)
    
    deltas_to_remove <- all_potential_predictors[
      stringr::str_detect(all_potential_predictors, "^delta_.*(_mean\\d|_sd\\d)$")
    ]
    if (length(deltas_to_remove) > 0) {
      log_message_local(paste0(
        "Simplifying: Removing ", length(deltas_to_remove),
        " delta variables of neighborhood stats."
      ))
    }
    potential_predictors <- setdiff(all_potential_predictors, deltas_to_remove)
    
    initial_rows <- nrow(combined_df)
    boruta_data  <- na.omit(combined_df[, c(response_var, potential_predictors), with = FALSE])
    rows_after_na <- nrow(boruta_data)
    log_message_local(
      paste0("Data cleaned for Boruta. Retained ", rows_after_na,
             " of ", initial_rows, " rows (",
             round(100 * rows_after_na / initial_rows, 1), "%).")
    )
    
    if (rows_after_na < 100) {
      log_message_local(paste0("⚠️ Insufficient non-NA data for Boruta. Skipping."))
      return(NULL)
    }
    
    x_data <- as.data.frame(boruta_data[, potential_predictors, with = FALSE])
    y_data <- boruta_data[[response_var]]
    rm(combined_df, boruta_data); invisible(gc())
    
    # 4. Run Boruta -------------------------------------------------
    log_message_local(paste0("Running Boruta on ", nrow(x_data), " samples..."))
    boruta_result <- tryCatch(
      Boruta::Boruta(x = x_data, y = y_data, maxRuns = max_runs, doTrace = 0),
      error = function(e) {
        log_message_local(paste0("❌ Boruta failed with error: ", e$message))
        return(NULL)
      }
    )
    if (is.null(boruta_result)) return(NULL)
    
    # 5. Get initial confirmed predictors ---------------------------
    confirmed_preds <- Boruta::getSelectedAttributes(boruta_result, withTentative = FALSE)
    
    # 6. Apply our consistency rules -------------------------------
    log_message_local("Enforcing hierarchical & flow-direction consistency...")
    confirmed_preds <- enforce_predictor_consistency(
      confirmed_preds = confirmed_preds,
      all_predictors  = potential_predictors
    )
    
    # 7. Process and save final results ----------------------------
    boruta_stats <- Boruta::attStats(boruta_result) %>%
      as.data.frame() %>%
      tibble::rownames_to_column(var = "predictor") %>%
      arrange(desc(meanImp))
    
    output_list <- list(
      confirmed_predictors = confirmed_preds,
      boruta_statistics    = boruta_stats
    )
    
    save_path <- file.path(training_base_dir,
                           paste0("boruta_results_", pair, ".rds"))
    saveRDS(output_list, file = save_path)
    
    log_message_local(paste0("✅ Final results saved to ", basename(save_path)))
    save_path
  }
  
  cat("\n🎉 Boruta Predictor Selection Complete!\n")
  invisible(all_results_paths)
}



# ==============================================================================
# FUNCTION CALL
# ==============================================================================
#
# Instructions:
# 1. Update the 'training_dir' to point to your main training data folder.
# 2. 'max_cores' now controls the outer loop (how many year-pairs run at once).
# 3. Run the code block.

# --- Define Parameters ---
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles"
year_intervals <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")

# --- Execute the Function ---
Sys.time()
boruta_results <- run_boruta_selection_by_pair(
  training_base_dir = training_dir,
  year_pairs = year_intervals,
  # --- TUNING PARAMETERS ---
  # Total data points to sample for each year-pair analysis.
  sample_size = 100000,
  # Number of files to load at a time to manage RAM.
  chunk_size = 5,
  # Number of cores to use for running year-pairs in parallel.
  # Since you have 4 year-pairs, a value of 4 is ideal.
  max_cores = 4,
  # --- BORUTA ALGORITHM PARAMETERS ---
  max_runs = 100
)
Sys.time()



# ----- ENSBMLE STRESS TEST TOOL
# Flow direction helper
transform_flowdir_cols <- function(dt) {
  flow_cols <- grep("flowdir", names(dt), value = TRUE)
  if (length(flow_cols) == 0) return(dt)
  
  for (col in flow_cols) {
    sin_name <- paste0(col, "_sin")
    cos_name <- paste0(col, "_cos")
    dt[, (sin_name) := sin(get(col) * pi / 180)]
    dt[, (cos_name) := cos(get(col) * pi / 180)]
  }
  
  dt[, (flow_cols) := NULL]
  dt
}

# Sampling training data per year-pair
sample_training_for_pair <- function(training_base_dir,
                                     year_pair,
                                     sample_size = 100000,
                                     chunk_size  = 5) {
  file_pattern <- paste0("_", year_pair, "_long\\.fst$")
  files_to_process <- list.files(
    path      = training_base_dir,
    pattern   = file_pattern,
    recursive = TRUE,
    full.names = TRUE
  )
  
  if (length(files_to_process) == 0) {
    stop("No long-format training files found for pair: ", year_pair)
  }
  
  file_chunks <- split(files_to_process,
                       ceiling(seq_along(files_to_process) / chunk_size))
  
  list_of_samples <- list()
  samples_per_chunk <- max(100, ceiling(sample_size / length(file_chunks)))
  
  for (i in seq_along(file_chunks)) {
    chunk_files <- file_chunks[[i]]
    chunk_dfs <- lapply(chunk_files, function(f) {
      tryCatch(fst::read_fst(f, as.data.table = TRUE),
               error = function(e) NULL)
    })
    chunk_dfs <- chunk_dfs[!vapply(chunk_dfs, is.null, logical(1))]
    if (!length(chunk_dfs)) next
    
    chunk_dt <- data.table::rbindlist(chunk_dfs, use.names = TRUE, fill = TRUE)
    
    if (nrow(chunk_dt) > 0) {
      n_samp <- min(nrow(chunk_dt), samples_per_chunk)
      set.seed(123 + i)
      list_of_samples[[i]] <- chunk_dt[sample(.N, n_samp)]
    }
    
    rm(chunk_dfs, chunk_dt); invisible(gc())
  }
  
  dt <- data.table::rbindlist(list_of_samples, use.names = TRUE, fill = TRUE)
  
  if (!"tile_id" %in% names(dt)) {
    stop("tile_id column is required for spatial CV.")
  }
  
  # transform flowdir cols here so they’re available
  dt <- transform_flowdir_cols(dt)
  
  dt
}
# Load Boruta confirmed predictors
load_boruta_predictors <- function(training_base_dir,
                                   year_pair,
                                   boruta_prefix = "boruta_results_") {
  path <- file.path(training_base_dir,
                    paste0(boruta_prefix, year_pair, ".rds"))
  if (!file.exists(path)) {
    stop("Boruta result file not found for pair ", year_pair, ": ", path)
  }
  res <- readRDS(path)
  preds <- res$confirmed_predictors
  
  # enforce that bathy_t is always included as a core predictor
  preds <- unique(c(preds, "bathy_t"))
  preds
}

# Group predictors (bathy / geomorph / storms / sediment / delta / other)
categorise_predictors <- function(predictors) {
  data.table::data.table(
    predictor = predictors
  )[, group := dplyr::case_when(
    predictor == "bathy_t" ~ "bathy_base",
    grepl("^bathy_", predictor) ~ "bathy_other",
    grepl("hurr_|tsm_", predictor) ~ "storms",
    grepl("grain_size_layer|prim_sed_layer", predictor) ~ "sediment",
    grepl("bpi_|rugosity|slope_|curv_|gradmag_|flowacc_|tci_|shearproxy|flowdir_.*_(sin|cos)",
          predictor) ~ "geomorph",
    grepl("^delta_", predictor) ~ "delta",
    TRUE ~ "other"
  )]
}

# Build model configurations (which groups to include)
build_model_configs <- function(groups_present) {
  # Only keep configs that actually have those groups present
  has <- function(g) all(g %in% groups_present)
  
  configs <- list()
  
  if (has("bathy_base")) {
    configs[["bathy_only"]] <- c("bathy_base")
  }
  if (has(c("bathy_base", "geomorph"))) {
    configs[["bathy_geomorph"]] <- c("bathy_base", "geomorph")
  }
  if (has(c("bathy_base", "storms"))) {
    configs[["bathy_storms"]] <- c("bathy_base", "storms")
  }
  if (has("geomorph")) {
    configs[["geomorph_only"]] <- c("geomorph")
  }
  if (has(c("storms", "geomorph"))) {
    configs[["storms_geomorph"]] <- c("storms", "geomorph")
  }
  if (has(c("bathy_base", "geomorph", "storms"))) {
    configs[["bathy_geomorph_storms"]] <- c("bathy_base", "geomorph", "storms")
  }
  # Full model: all groups except maybe 'other'
  full_groups <- setdiff(groups_present, c("other"))
  configs[["full_model"]] <- full_groups
  
  configs
}

# Spatial CV runner
run_spatial_cv <- function(dt,
                           predictors,
                           response    = "bathy_t1",
                           k_folds     = 5,
                           seed        = 123,
                           compute_moran = TRUE) {
  if (!all(c(response, "tile_id") %in% names(dt))) {
    stop("Response or tile_id missing from dataset.")
  }
  
  # Drop rows with NA in response or predictors
  cols_needed <- c(response, predictors, "tile_id", "X", "Y")
  cols_needed <- intersect(cols_needed, names(dt))
  dt_use <- dt[, ..cols_needed]
  dt_use <- na.omit(dt_use)
  
  if (nrow(dt_use) < 200) {
    stop("Not enough complete rows for CV.")
  }
  
  # Assign folds by tile_id -> spatial CV
  set.seed(seed)
  tiles <- unique(dt_use$tile_id)
  folds <- sample(rep(1:k_folds, length.out = length(tiles)))
  tile_to_fold <- data.table::data.table(tile_id = tiles,
                                         fold    = folds)
  dt_use <- merge(dt_use, tile_to_fold, by = "tile_id")
  
  preds_all <- numeric(0)
  obs_all   <- numeric(0)
  coords_all <- NULL
  
  for (k in 1:k_folds) {
    train_dt <- dt_use[fold != k]
    val_dt   <- dt_use[fold == k]
    
    if (nrow(val_dt) < 10) next
    
    x_train <- train_dt[, ..predictors]
    x_val   <- val_dt[, ..predictors]
    y_train <- train_dt[[response]]
    y_val   <- val_dt[[response]]
    
    fit <- ranger::ranger(
      x          = x_train,
      y          = y_train,
      num.trees  = 500,
      mtry       = max(1, floor(sqrt(length(predictors)))),
      importance = "permutation"
    )
    
    pred_val <- predict(fit, data = x_val)$predictions
    
    preds_all <- c(preds_all, pred_val)
    obs_all   <- c(obs_all, y_val)
    
    if (!is.null(val_dt$X) && !is.null(val_dt$Y)) {
      coords_all <- rbind(coords_all,
                          as.matrix(val_dt[, .(X, Y)]))
    }
  }
  
  if (!length(preds_all)) stop("No predictions generated in CV.")
  
  residuals <- obs_all - preds_all
  
  rmse <- sqrt(mean((obs_all - preds_all)^2))
  mae  <- mean(abs(obs_all - preds_all))
  r2   <- 1 - sum((obs_all - preds_all)^2) / sum((obs_all - mean(obs_all))^2)
  
  moran_I <- NA_real_
  if (compute_moran &&
      !is.null(coords_all) &&
      requireNamespace("spdep", quietly = TRUE)) {
    
    # sample if too many points for Moran's I
    max_moran_n <- 10000
    if (nrow(coords_all) > max_moran_n) {
      set.seed(seed)
      idx <- sample(seq_len(nrow(coords_all)), max_moran_n)
      coords_m <- coords_all[idx, , drop = FALSE]
      res_m    <- residuals[idx]
    } else {
      coords_m <- coords_all
      res_m    <- residuals
    }
    
    nb <- spdep::knearneigh(coords_m, k = 8)
    nb <- spdep::knn2nb(nb)
    lw <- spdep::nb2listw(nb, style = "W")
    
    mt <- spdep::moran.test(res_m, lw, zero.policy = TRUE)
    moran_I <- as.numeric(mt$estimate[["Moran I statistic"]])
  }
  
  list(
    rmse    = rmse,
    mae     = mae,
    r2      = r2,
    moran_I = moran_I
  )
}

# Main Stress Test Runner
run_predictor_stress_test <- function(training_base_dir,
                                      year_pairs,
                                      sample_size   = 100000,
                                      chunk_size    = 5,
                                      k_folds       = 5,
                                      max_cores     = parallel::detectCores() - 1,
                                      compute_moran = TRUE) {
  library(data.table)
  library(fst)
  library(stringr)
  library(ranger)
  
  cat("\n🚀 Starting Predictor Ensemble Stress Test...\n")
  
  num_cores <- min(length(year_pairs), max_cores)
  cl <- parallel::makeCluster(num_cores)
  doParallel::registerDoParallel(cl)
  on.exit(parallel::stopCluster(cl), add = TRUE)
  
  clusterExport(
    cl,
    varlist = c(
      "sample_training_for_pair",
      "transform_flowdir_cols",
      "load_boruta_predictors",
      "categorise_predictors",
      "build_model_configs",
      "run_spatial_cv"
    ),
    envir = environment()
  )
  
  results_all <- foreach::foreach(
    pair = year_pairs,
    .packages = c("data.table", "fst", "stringr", "ranger", "spdep"),
    .combine = rbind
  ) %dopar% {
    
    message("=== Year pair: ", pair, " ===")
    
    # 1. Sample training data
    dt <- sample_training_for_pair(
      training_base_dir = training_base_dir,
      year_pair         = pair,
      sample_size       = sample_size,
      chunk_size        = chunk_size
    )
    
    # 2. Load Boruta-confirmed predictors and intersect with columns
    boruta_preds <- load_boruta_predictors(
      training_base_dir = training_base_dir,
      year_pair         = pair
    )
    
    # Keep only those actually present in dt
    boruta_preds <- intersect(boruta_preds, names(dt))
    
    # 3. Categorise predictors into groups
    pred_meta <- categorise_predictors(boruta_preds)
    groups_present <- unique(pred_meta$group)
    
    # 4. Build model configs
    model_configs <- build_model_configs(groups_present)
    
    # 5. For each config, collect predictors and run spatial CV
    pair_results <- list()
    
    for (cfg_name in names(model_configs)) {
      cfg_groups <- model_configs[[cfg_name]]
      
      preds_cfg <- pred_meta[group %in% cfg_groups, predictor]
      preds_cfg <- unique(preds_cfg)
      
      if (!length(preds_cfg)) next
      
      # Always ensure bathy_t is present in any model with bathy_base
      if ("bathy_base" %in% cfg_groups && !"bathy_t" %in% preds_cfg) {
        preds_cfg <- c("bathy_t", preds_cfg)
      }
      
      metrics <- tryCatch(
        run_spatial_cv(
          dt            = dt,
          predictors    = preds_cfg,
          response      = "bathy_t1",
          k_folds       = k_folds,
          compute_moran = compute_moran
        ),
        error = function(e) {
          warning("CV failed for pair ", pair, ", config ", cfg_name, ": ", e$message)
          return(NULL)
        }
      )
      
      if (!is.null(metrics)) {
        pair_results[[cfg_name]] <- data.table(
          year_pair   = pair,
          model_name  = cfg_name,
          n_predictors = length(preds_cfg),
          rmse        = metrics$rmse,
          mae         = metrics$mae,
          r2          = metrics$r2,
          moran_I     = metrics$moran_I
        )
      }
    }
    
    if (!length(pair_results)) return(NULL)
    
    res_dt <- rbindlist(pair_results, use.names = TRUE)
    
    # Compute deltas vs baseline bathy_only if available
    if ("bathy_only" %in% res_dt$model_name) {
      base_rmse  <- res_dt[model_name == "bathy_only", rmse][1]
      base_moran <- res_dt[model_name == "bathy_only", moran_I][1]
      
      res_dt[, delta_rmse_vs_bathy :=
               if (!is.na(base_rmse)) rmse - base_rmse else NA_real_]
      res_dt[, delta_moran_vs_bathy :=
               if (!is.na(base_moran)) moran_I - base_moran else NA_real_]
    } else {
      res_dt[, `:=`(
        delta_rmse_vs_bathy  = NA_real_,
        delta_moran_vs_bathy = NA_real_
      )]
    }
    
    # Save per-pair results
    out_path <- file.path(
      training_base_dir,
      paste0("stress_test_results_", pair, ".fst")
    )
    fst::write_fst(res_dt, out_path)
    
    message("Saved stress-test results for ", pair, " to: ",
            basename(out_path))
    
    res_dt
  }
  
  cat("\n🎉 Predictor Ensemble Stress Test complete.\n")
  invisible(results_all)
}


# Function Call
stress_results <- run_predictor_stress_test(
  training_base_dir = training_dir,
  year_pairs        = year_intervals,
  sample_size       = 100000,   # or higher if RAM allows
  chunk_size        = 5,
  k_folds           = 5,
  max_cores         = 4,
  compute_moran     = TRUE      # set FALSE if you don’t want spatial metric
)

glimpse(stress_results)

# ==============================================================================
#
#           Boruta Predictor Selection Summary Report & Master List
#
# ==============================================================================
#
# Purpose:
# This function scans for centralized Boruta results, aggregates them, and
# generates two key outputs:
#   1. A visual summary plot of the top predictors for each year-pair.
#   2. A text file containing a master list of the top predictors aggregated
#      across all time periods, ranked by overall importance.
#
# ==============================================================================

# --- Load All Necessary Libraries ---
library(dplyr)
library(stringr)
library(data.table)
library(tibble)

# ==============================================================================
#   MAIN REPORTING FUNCTION
# ==============================================================================

#' Create a visual report and a master predictor list from Boruta results.
#'
#' @param training_base_dir The directory where `boruta_results_*.rds` files are.
#' @param output_plot_filename The name for the output PNG plot.
#' @param output_list_filename The name for the output TXT file with the master list.
#' @param top_n_plot The number of predictors to show in each panel of the plot.
#' @param top_n_list The number of predictors to include in the final master list.
#'
#' @return None. A PNG and a TXT file are saved to the `training_base_dir`.

create_boruta_summary_report <- function(training_base_dir,
                                         output_plot_filename = "boruta_summary_plot.png",
                                         output_list_filename = "boruta_master_predictor_list.txt",
                                         top_n_plot = 15,
                                         top_n_list = 15) {
  
  # -------------------------------------------------------
  # 1. FIND and LOAD Centralized Result Files
  # -------------------------------------------------------
  cat("🚀 Starting Boruta summary report generation...\n")
  
  selection_files <- list.files(
    path = training_base_dir,
    pattern = "^boruta_results_.*\\.rds$",
    recursive = FALSE,
    full.names = TRUE
  )
  
  if (length(selection_files) == 0) {
    stop("No 'boruta_results_*.rds' files found in the specified directory.")
  }
  cat(paste("Found", length(selection_files), "Boruta result files to summarize.\n"))
  
  # -------------------------------------------------------
  # 2. PROCESS and COMBINE All Results
  # -------------------------------------------------------
  all_results <- lapply(selection_files, function(fp) {
    result_list <- readRDS(fp)
    confirmed <- result_list$confirmed_predictors
    # Filter the stats to only include predictors that were actually confirmed
    stats_df <- result_list$boruta_statistics %>%
      filter(predictor %in% confirmed)
    
    stats_df$year_pair <- str_extract(basename(fp), "\\d{4}_\\d{4}")
    return(stats_df)
  })
  
  combined_df <- bind_rows(all_results)
  cat("Successfully processed all result files.\n")
  
  # -------------------------------------------------------
  # 3. GENERATE AND SAVE THE MASTER PREDICTOR LIST
  # -------------------------------------------------------
  cat("Generating master predictor list...\n")
  
  # --- **FIX**: Normalize forcing variable names for proper aggregation ---
  master_list_df <- combined_df %>%
    mutate(
      # Create a generic name for forcing vars, leave others unchanged
      predictor_generic = if_else(
        str_detect(predictor, "_\\d{4}_\\d{4}$"),
        str_remove(predictor, "_\\d{4}_\\d{4}$"),
        predictor
      )
    )
  
  master_predictors <- master_list_df %>%
    group_by(predictor_generic) %>%
    summarise(
      # More robust ranking: count how many pairs it's confirmed in, then use avg importance
      confirmation_count = n_distinct(year_pair),
      overall_importance = mean(meanImp)
    ) %>%
    arrange(desc(confirmation_count), desc(overall_importance)) %>%
    head(top_n_list)
  
  # Save the master list to a text file
  list_save_path <- file.path(training_base_dir, output_list_filename)
  writeLines(master_predictors$predictor_generic, list_save_path)
  cat(paste("✅ Master predictor list saved to:", list_save_path, "\n"))
  
  # -------------------------------------------------------
  # 4. PREPARE DATA FOR PLOTTING
  # -------------------------------------------------------
  cat("Preparing data for year-pair specific importance plot...\n")
  year_pairs <- sort(unique(combined_df$year_pair))
  
  importance_by_year <- lapply(year_pairs, function(pair) {
    combined_df %>%
      filter(year_pair == pair) %>%
      arrange(desc(meanImp)) %>%
      head(top_n_plot)
  })
  names(importance_by_year) <- year_pairs
  
  # -------------------------------------------------------
  # 5. GENERATE YEAR-PAIR SPECIFIC IMPORTANCE PLOT
  # -------------------------------------------------------
  cat("Generating year-pair specific importance plot...\n")
  png(
    file.path(training_base_dir, output_plot_filename),
    width = 1600, height = 1200, res = 100
  )
  par(mfrow = c(2, 2), mar = c(5, 10, 4, 2), oma = c(0, 0, 3, 0)) # Outer margins for a main title
  
  for (pair in year_pairs) {
    plot_data <- importance_by_year[[pair]]
    if (nrow(plot_data) > 0) {
      barplot(
        height = rev(plot_data$meanImp),
        names.arg = rev(plot_data$predictor),
        horiz = TRUE, las = 1,
        main = paste("Year Pair:", pair),
        xlab = "Mean Importance (Z-Score)",
        col = "darkcyan",
        cex.names = 0.8 # Use smaller font if names are long
      )
    } else {
      # Create a blank plot with a message if no predictors were confirmed
      plot(1, type="n", axes=FALSE, xlab="", ylab="", main = paste("Year Pair:", pair))
      text(1, 1, "No 'Confirmed' predictors found.", cex = 1.2)
    }
  }
  
  mtext(paste("Top", top_n_plot, "Confirmed Predictor Importance by Year Pair"), outer = TRUE, cex = 1.8, font = 2)
  dev.off()
  
  # Reset plotting layout to default
  par(mfrow = c(1, 1), oma = c(0, 0, 0, 0))
  cat(paste("✅ Plot saved to:", file.path(training_base_dir, output_plot_filename), "\n"))
  cat("Process complete.\n")
}


# ==============================================================================
# FUNCTION CALL
# ==============================================================================
# --- Define Parameters ---
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles"

# --- Execute the Function ---
create_boruta_summary_report(
  training_base_dir = training_dir,
  top_n_plot = 20,
  top_n_list = 15 # Creates a text file with the top 10 overall predictors
)

create_predictor_summary_report <- function(
    training_base_dir,
    output_plot_filename = "predictor_summary_plot.png",
    output_master_list   = "final_master_predictor_list.txt",
    output_csv_filename  = "predictor_combined_scores.csv",
    top_n_plot           = 20,
    top_n_list           = 20,
    # ---- WEIGHTS FOR FINAL SCORE ----
    w_confirm    = 2,
    w_boruta_imp = 1,
    w_rmse_gain  = 1,
    w_moran_gain = 0.5
) {
  library(data.table)
  library(dplyr)
  library(stringr)
  library(fst)
  
  cat("\n=========================================\n")
  cat("🚀 STARTING COMBINED BORUTA + STRESS-TEST REPORT\n")
  cat("=========================================\n\n")
  
  # ------------------------------------------------------------
  # 1. Load BORUTA results
  # ------------------------------------------------------------
  boruta_files <- list.files(
    training_base_dir,
    pattern = "^boruta_results_.*\\.rds$",
    full.names = TRUE
  )
  if (length(boruta_files) == 0)
    stop("No Boruta results found.")
  
  boruta_list <- lapply(boruta_files, function(fp) {
    r <- readRDS(fp)
    stats <- r$boruta_statistics
    confirmed <- r$confirmed_predictors
    stats <- stats %>% filter(predictor %in% confirmed)
    stats$year_pair <- str_extract(basename(fp), "\\d{4}_\\d{4}")
    stats
  })
  
  boruta_df <- bind_rows(boruta_list)
  
  cat("📌 Loaded Boruta results for", length(unique(boruta_df$year_pair)), "year-pairs.\n")
  
  boruta_summary <- boruta_df %>%
    group_by(predictor) %>%
    summarise(
      boruta_confirm_count = n_distinct(year_pair),
      boruta_mean_importance = mean(meanImp)
    )
  
  # ------------------------------------------------------------
  # 2. Load ALL individual stress-test results
  # ------------------------------------------------------------
  stress_files <- list.files(
    training_base_dir,
    pattern = "^stress_test_results_.*\\.fst$",
    full.names = TRUE
  )
  if (length(stress_files) == 0)
    stop("No stress-test results found!")
  
  stress_df <- rbindlist(lapply(stress_files, fst::read_fst, as.data.table=TRUE))
  
  cat("📌 Loaded stress-test results for", length(unique(stress_df$year_pair)), "year-pairs.\n")
  
  # ------------------------------------------------------------
  # 3. Assign predictor → group (terrain_class → geomorph)
  # ------------------------------------------------------------
  categorise_predictors <- function(predictors) {
    data.table(predictor = predictors)[, group := case_when(
      predictor == "bathy_t" ~ "bathy_base",
      grepl("^bathy_", predictor) ~ "bathy_other",
      grepl("hurr_|tsm_", predictor) ~ "storms",
      grepl("grain_size_layer|prim_sed_layer", predictor) ~ "sediment",
      grepl("^flowdir_(sin|cos)_t$", predictor) ~ "geomorph",      # flowdir fix
      grepl("bpi_|rugosity|slope_|curv_|gradmag_|flowacc_|tci_|shearproxy", predictor) ~ "geomorph",
      predictor == "terrain_classification_t" ~ "geomorph",       # terrain fix
      grepl("^delta_", predictor) ~ "delta",
      TRUE ~ "other"
    )]
  }
  
  boruta_groups <- categorise_predictors(boruta_summary$predictor)
  
  # ------------------------------------------------------------
  # 4. Compute group-level stress-test improvements
  # ------------------------------------------------------------
  stress_group_scores <- stress_df %>%
    group_by(model_name) %>%
    summarise(
      mean_rmse_gain  = mean(-delta_rmse_vs_bathy, na.rm=TRUE),
      mean_moran_gain = mean(delta_moran_vs_bathy, na.rm=TRUE)
    )
  
  model_group_map <- list(
    bathy_only             = "bathy_base",
    bathy_geomorph         = c("bathy_base", "geomorph"),
    bathy_storms           = c("bathy_base", "storms"),
    geomorph_only          = "geomorph",
    storms_geomorph        = c("storms", "geomorph"),
    bathy_geomorph_storms  = c("bathy_base", "geomorph", "storms"),
    full_model             = c("bathy_base", "geomorph", "storms", "sediment")
  )
  
  group_scores <- data.table()
  for (i in 1:nrow(stress_group_scores)) {
    m <- stress_group_scores$model_name[i]
    if (!m %in% names(model_group_map)) next
    groups <- model_group_map[[m]]
    group_scores <- rbind(
      group_scores,
      data.table(
        group = groups,
        rmse_gain  = stress_group_scores$mean_rmse_gain[i],
        moran_gain = stress_group_scores$mean_moran_gain[i]
      )
    )
  }
  
  group_scores <- group_scores[, .(
    rmse_gain = mean(rmse_gain),
    moran_gain = mean(moran_gain)
  ), by=group]
  
  # ------------------------------------------------------------
  # 5. Merge Boruta + Stress Test and compute scores
  # ------------------------------------------------------------
  merged <- merge(boruta_summary, boruta_groups, by="predictor", all.x=TRUE)
  merged <- merge(merged, group_scores, by="group", all.x=TRUE)
  
  merged <- merged %>%
    mutate(
      rmse_gain  = ifelse(is.na(rmse_gain), 0, rmse_gain),
      moran_gain = ifelse(is.na(moran_gain), 0, moran_gain)
    )
  
  
  merged <- merged %>%
    mutate(
      weighted_rmse_gain  = rmse_gain  * (boruta_confirm_count / max(boruta_confirm_count)),
      weighted_moran_gain = moran_gain * (boruta_confirm_count / max(boruta_confirm_count)),
      final_score =
        w_confirm    * boruta_confirm_count +
        w_boruta_imp * boruta_mean_importance +
        w_rmse_gain  * weighted_rmse_gain +
        w_moran_gain * weighted_moran_gain
    ) %>%
    arrange(desc(final_score))
  
  # ------------------------------------------------------------
  # 6. Collapse storm-year suffixes, flowdir, BPI families
  # ------------------------------------------------------------
  collapse_predictor_name <- function(x) {
    
    # 1 — Remove ONLY year-pair suffixes (not '_t', '_sd3_t', etc.)
    x <- gsub("_[0-9]{4}_[0-9]{4}$", "", x)
    
    # 2 — Storm & climate metrics collapse to base name
    x <- gsub("^hurr_strength.*$", "hurr_strength", x)
    x <- gsub("^hurr_count.*$",    "hurr_count", x)
    x <- gsub("^tsm.*$",           "tsm", x)
    
    # 3 — Terrain classification stays as-is
    x <- gsub("^terrain_classification.*$", "terrain_classification_t", x)
    
    # 4 — Flowdir sin/cos preserved
    x <- gsub("^flowdir_sin_t$", "flowdir_sin_t", x)
    x <- gsub("^flowdir_cos_t$", "flowdir_cos_t", x)
    
    # 5 — Geomorph metrics: keep full variable names
    # (no additional collapsing)
    
    return(x)
  }
  
  
  merged$predictor_collapsed <- collapse_predictor_name(merged$predictor)
  
  
  # ------------------------------------------------------------
  # 7. Compute collapsed ranking
  # ------------------------------------------------------------
  collapsed <- merged %>%
    group_by(predictor_collapsed) %>%
    summarise(
      max_score = max(final_score),
      mean_score = mean(final_score),
      max_confirm = max(boruta_confirm_count)
    ) %>%
    arrange(desc(max_score))
  
  final_preds <- collapsed$predictor_collapsed[1:top_n_list]
  
  # REMOVE TERRAIN CLASSIFICATION FROM FINAL PREDICTOR LIST
  final_preds <- setdiff(final_preds, "terrain_classification")
  
  # ------------------------------------------------------------
  # 8. Save final list
  # ------------------------------------------------------------
  writeLines(final_preds, file.path(training_base_dir, output_master_list))
  
  cat("✅ Final combined predictor list saved:", output_master_list, "\n")
  
  # ------------------------------------------------------------
  # 9. Save CSV
  # ------------------------------------------------------------
  fwrite(merged, file.path(training_base_dir, output_csv_filename))
  cat("📄 Predictor ranking CSV saved to:", output_csv_filename, "\n")
  
  # ------------------------------------------------------------
  # 10. Plot Boruta Importances
  # ------------------------------------------------------------
  png(file.path(training_base_dir, output_plot_filename),
      width=1800, height=1400, res=120)
  par(mfrow=c(2,2), mar=c(5,12,4,2), oma=c(0,0,3,0))
  
  yr_pairs <- sort(unique(boruta_df$year_pair))
  for (pair in yr_pairs) {
    dfp <- boruta_df %>%
      filter(year_pair == pair) %>%
      arrange(desc(meanImp)) %>%
      head(top_n_plot)
    
    barplot(
      rev(dfp$meanImp),
      names.arg=rev(dfp$predictor),
      horiz=TRUE, las=1,
      col="steelblue",
      main=paste("Boruta Importance:", pair)
    )
  }
  
  mtext("Predictor Importance Summary (Boruta + Stress Test)",
        outer=TRUE, cex=1.7, font=2)
  
  dev.off()
  
  cat("🎨 Summary plot saved:", output_plot_filename, "\n\n")
  cat("✨ REPORT COMPLETE ✨\n")
  
  invisible(merged)
}


create_predictor_summary_report(training_dir)

# STEPS:
 # preprocessing all complete ( terrain classification, new Geomorphology layers)
 # Convert the dataframes into long format for analysis 
# run boruta slection - then run additional stat tests to select the strongest predictors, 
# use summary report to make a final predictor list ~10 -16
# Now with the refined predictors, run the XGB parameter search, to find the best combination of hyperparameters to run the model
# update the hyperparameters and now run the main modelling workflow. 
# run the prediction workflow

# 
# ==============================================================================
# XGB PARAMETER SEARCH - AUTOMATED MINI WORKFLOW
#   - Runs a single-model (no bootstrap) XGBoost for each parameter set
#   - Saves rasters: mean_predicted_change, prediction_residual, delta_residual
#   - includes weighted-loss tuning also 
#   - Computes metrics + scores per run
#   - Identifies:
#       * best_for_accuracy
#       * best_for_change
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

#---------------

# Location of data directories 
prediction_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
training_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
prediction_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"
training_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles"
master_grid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/master_grid_canonical.gpkg"

# MODEL AIMS 
# • Independent variable (Y):
#   bathy_t+1 (the ending bathy state).
# • Predictors (X):
#   Bathy_t (starting state), plus derived predictors like Δrugosity, Δslope, Δstorm metrics, sediment type, and optionally Δbathy (past changes) if you keep that history.
# • Model learns “drivers + state_t → state_t+1”.
# At prediction time, you feed it bathy_t and drivers, and it outputs a prediction of bathy_t+1 directly — no adding step needed




# V2 now storing mean boots in an array 


# ==============================================================================
#
#   _XGBoost Model Training Function Set (Refactored for Long-Format Data)
#
# ==============================================================================
#
# This script contains a refactored set of functions to train XGBoost models
# based on a new data structure where each tile/year-pair is a separate file.
# It is designed to:
#   1. Load a master list of predictors from a central text file.
#   2. Iterate through grid tiles and their corresponding long-format year-pair files.
#   3. Run a robust, parallelized spatial cross-validation to find optimal model parameters.
#   4. Train a final model and run multiple bootstrap replicates, storing results
#      in a robust 3D array to ensure spatial integrity.
#   5. Save all necessary outputs for the prediction workflow.

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(ggplot2)
library(foreach)
library(doParallel)
library(blockCV)
library(stringr)

#' -----------------------------------------------------------------------------
#' Helper Function: Generate and Save a Diagnostic Plot for Spatial CV
#' -----------------------------------------------------------------------------
generate_cv_diagnostic_plot <- function(sf_data, block_geom, max_k, tile_id, pair, output_dir_train) {
  diag_plot <- ggplot() +
    geom_sf(data = sf_data, color = "grey50", size = 0.1, alpha = 0.5) +
    geom_sf(data = block_geom, color = "blue", fill = "transparent", linewidth = 0.5) +
    labs(
      title = paste("Diagnostic Plot for Tile:", tile_id, "| Pair:", pair),
      subtitle = paste("Block Size:", st_bbox(block_geom)[3] - st_bbox(block_geom)[1], "m | Max Possible Folds (k):", max_k),
      x = "X (UTM)", y = "Y (UTM)"
    ) +
    theme_minimal()
  
  plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
  if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
  
  ggsave(
    file.path(plot_dir, paste0("spatial_cv_map_", pair, ".png")),
    plot = diag_plot, width = 8, height = 7, dpi = 150
  )
}


# ==============================================================================
# REFACTORED MAIN TRAINING FUNCTION
# ==============================================================================
Model_Train_Full_SpatialCV <- function(training_sub_grids_UTM, output_dir_train, year_pairs,
                                       master_predictor_list_path,
                                       block_size_m, n.boot = 10, n.folds = 5) {
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\nStarting Full XGBoost Model Training with Robust Parallel Spatial CV...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  if (num_cores < 1) num_cores <- 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  on.exit({
    if (exists("cl") && inherits(cl, "cluster")) {
      stopCluster(cl)
    }
    closeAllConnections()
  }, add = TRUE)
  
  master_log_file <- file.path(output_dir_train, "error_log_final.txt")
  cat("Error Log - XGBoost Full Training (run started at", as.character(Sys.time()), ")\n", file = master_log_file, append = FALSE)
  
  # -------------------------------------------------------
  # 2. MAIN PARALLEL PROCESSING LOOP
  # -------------------------------------------------------
  results_list <- foreach(
    i = seq_len(length(tiles_df)),
    .combine = 'c',
    .packages = c("data.table", "fst", "stringr"),
    .export = "generate_cv_diagnostic_plot",
    .errorhandling = "pass"
  ) %dopar% {
    library(xgboost)
    library(dplyr)
    library(tidyr)
    library(sf)
    library(blockCV)
    library(raster)
    library(ggplot2)
    
    tile_id <- tiles_df[[i]]
    worker_log_file <- file.path(output_dir_train, tile_id, paste0("log_worker_", tile_id, ".txt"))
    cat("Worker log for tile:", tile_id, "started at", as.character(Sys.time()), "\n", file = worker_log_file, append = FALSE)
    
    # --- Load the master predictor list once per worker ---
    if (!file.exists(master_predictor_list_path)) {
      stop("Master predictor list file not found at:", master_predictor_list_path)
    }
    master_predictors <- readLines(master_predictor_list_path)
    
    
    # Inner loop over year pairs (sequential within each worker)
    for (pair in year_pairs) {
      tryCatch({
        # --- a. Load Data & Define Variables ---
        cat("\nProcessing Tile:", tile_id, "| Year Pair:", pair, "\n", file = worker_log_file, append = TRUE)
        
        # NEW: Construct file prefix by removing sub-grid suffix (e.g., "_4")
        file_prefix <- sub("_[0-9]+$", "", tile_id)
        
        # MODIFIED: Use the file_prefix to build the correct path
        training_data_path <- file.path(output_dir_train, tile_id, paste0(file_prefix, "_", pair, "_long.fst"))
        if (!file.exists(training_data_path)) {
          cat("DIAGNOSTIC: Missing input file:", basename(training_data_path), ". Skipping.\n", file = worker_log_file, append = TRUE)
          next
        }
        
        full_tile_data_raw <- as.data.frame(read_fst(training_data_path, as.data.table = TRUE))
        
        # MODIFIED: Changed response variable to predict the bathymetry state at time t1.
        response_var <- "bathy_t1"
        
        current_predictors <- sapply(master_predictors, function(p) {
          if (p %in% c("hurr_count", "hurr_strength", "tsm")) {
            return(paste0(p, "_", pair))
          } else {
            return(p)
          }
        }, USE.NAMES = FALSE)
        
        predictors <- intersect(current_predictors, names(full_tile_data_raw))
        
        if (length(predictors) == 0 || !response_var %in% names(full_tile_data_raw)) {
          cat("DIAGNOSTIC: No predictors or response variable found. Skipping.\n", file = worker_log_file, append = TRUE)
          next
        }
        
        # --- b. Define CRS ---
        grid_crs_epsg <- 32617
        grid_crs_proj4 <- sf::st_crs(grid_crs_epsg)$proj4string
        
        # -------------------------------------------------------
        # 3. FILTER & PREPARE TRAINING DATA
        # -------------------------------------------------------
        # `full_tile_data` contains ALL points for prediction
        full_tile_data <- full_tile_data_raw %>%
          dplyr::select(any_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
          mutate(across(all_of(c(predictors, response_var)), as.numeric))
        
        subgrid_data <- full_tile_data %>%
          filter(is.finite(.data[[response_var]]))
        
        if (nrow(subgrid_data) < 100) {
          cat("DIAGNOSTIC: Insufficient training data (<100 rows). Skipping.\n", file = worker_log_file, append = TRUE)
          next
        }
        
        subgrid_sf <- st_as_sf(subgrid_data, coords = c("X", "Y"), remove = FALSE, crs = grid_crs_epsg)
        
        
        # -------------------------------------------------------
        # 4. INITIALIZE METRIC OUTPUT STORAGE ARRAYS
        # -------------------------------------------------------
        deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot); colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
        influence_mat <- array(NA, dim = c(length(predictors), n.boot)); rownames(influence_mat) <- predictors; colnames(influence_mat) <- paste0("Rep_", 1:n.boot)
        
        # MODIFICATION: Initialize a 3D array for storing bootstrap predictions.
        # Dimensions are [rows, 1 column for prediction, number of bootstraps]
        boot_array <- array(NA, dim = c(nrow(full_tile_data), 1, n.boot))
        
        # -------------------------------------------------------
        # 5. SETUP for PARTIAL DEPENDENCE PLOT (PDP) DATA
        # -------------------------------------------------------
        PredMins <- apply(subgrid_data[, predictors, drop = FALSE], 2, min, na.rm = TRUE)
        PredMaxs <- apply(subgrid_data[, predictors, drop = FALSE], 2, max, na.rm = TRUE)
        
        EnvRanges <- expand.grid(Env_Value = seq(0, 1, length.out = 100), Predictor = predictors)
        
        for (pred in predictors) {
          min_val <- PredMins[pred]; max_val <- PredMaxs[pred]
          if (is.finite(min_val) && is.finite(max_val) && min_val != max_val) {
            EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(min_val, max_val, length.out = 100)
          }
        }
        if(nrow(subgrid_data) >= 100){
          EnvRanges$X <- rep(subgrid_data$X[1:100], length(predictors))
          EnvRanges$Y <- rep(subgrid_data$Y[1:100], length(predictors))
          EnvRanges$FID <- rep(subgrid_data$FID[1:100], length(predictors))
        }
        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
        dimnames(PD)[[2]] <- predictors; dimnames(PD)[[3]] <- paste0("Rep_", 1:n.boot)
        all_pdp_long_list <- list()
        
        # -------------------------------------------------------
        # 6. SETUP ADAPTIVE SPATIAL CROSS VALIDATION & MODEL TRAINING
        # -------------------------------------------------------
        best_iteration <- 100 # Default fallback
        cv_results_df <- NULL
        
        grid_blocks <- st_make_grid(subgrid_sf, cellsize = block_size_m)
        points_in_blocks <- st_intersects(grid_blocks, subgrid_sf)
        max_k <- length(which(lengths(points_in_blocks) > 0))
        generate_cv_diagnostic_plot(subgrid_sf, st_sf(grid_blocks), max_k, tile_id, pair, output_dir_train)
        
        k_final <- min(n.folds, max_k)
        
        tryCatch({
          if (k_final < 2) stop("CV not possible.")
          scv <- cv_spatial(x = subgrid_sf, size = block_size_m, k = k_final, iteration = 200)
          if (is.null(scv) || length(scv$folds_list) < k_final) stop("blockCV failed.")
          
          best_nrounds_per_fold <- c(); rmse_per_fold <- c(); mae_per_fold <- c()
          
          for (k in 1:k_final) {
            train_idx <- unlist(scv$folds_list[[k]][1]); test_idx <- unlist(scv$folds_list[[k]][2])
            if (length(unique(subgrid_data[test_idx, response_var])) < 2) next
            
            dtrain_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[train_idx, predictors]), label = subgrid_data[train_idx, response_var], missing = NA)
            dtest_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[test_idx, predictors]), label = subgrid_data[test_idx, response_var], missing = NA)
            
            fold_model <- xgb.train(params = list(max_depth = 4, eta = 0.01, gamma = 1, objective = "reg:squarederror"),
                                    data = dtrain_fold, nrounds = 1000,
                                    watchlist = list(train = dtrain_fold, test = dtest_fold),
                                    early_stopping_rounds = 10, eval_metric = c("rmse", "mae"), verbose = 0)
            
            if (!is.null(fold_model$best_iteration) && fold_model$best_iteration > 0) {
              best_nrounds_per_fold <- c(best_nrounds_per_fold, fold_model$best_iteration)
              rmse_per_fold <- c(rmse_per_fold, fold_model$evaluation_log$test_rmse[fold_model$best_iteration])
              mae_per_fold <- c(mae_per_fold, fold_model$evaluation_log$test_mae[fold_model$best_iteration])
            }
          }
          
          if (length(best_nrounds_per_fold) > 0) {
            best_iteration <- round(mean(best_nrounds_per_fold, na.rm = TRUE))
            cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                        test_rmse_mean = mean(rmse_per_fold, na.rm = TRUE), test_rmse_std = sd(rmse_per_fold, na.rm = TRUE),
                                        test_mae_mean = mean(mae_per_fold, na.rm = TRUE), test_mae_std = sd(mae_per_fold, na.rm = TRUE))
          }
        }, error = function(e) {
          cat("WARNING: CV SKIPPED:", conditionMessage(e), "\n", file = worker_log_file, append = TRUE)
        })
        
        if (is.null(cv_results_df)) {
          cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                      test_rmse_mean = NA, test_rmse_std = NA, test_mae_mean = NA, test_mae_std = NA)
        }
        
        # --- FINAL MODEL SETUP ---
        dtrain_full <- xgb.DMatrix(data = as.matrix(subgrid_data[, predictors]), label = subgrid_data[[response_var]], missing = NA)
        dpredict_full <- xgb.DMatrix(data = as.matrix(full_tile_data[, predictors]), missing = NA)
        xgb_params <- list(max_depth = 4, eta = 0.01, gamma = 1, subsample = 0.7, colsample_bytree = 0.8, objective = "reg:squarederror")
        
        # -------------------------------------------------------
        # 7. BOOTSTRAP LOOP & SAVE MODEL METRICS
        # -------------------------------------------------------
        for (b in seq_len(n.boot)) {
          boot_indices <- sample(seq_len(nrow(subgrid_data)), replace = TRUE)
          dtrain_boot <- xgb.DMatrix(data = as.matrix(subgrid_data[boot_indices, predictors]),
                                     label = subgrid_data[boot_indices, response_var, drop = TRUE], missing = NA)
          
          xgb_model <- xgb.train(params = xgb_params, data = dtrain_boot, nrounds = best_iteration, nthread = 1)
          
          predictions <- predict(xgb_model, newdata = dpredict_full)
          # MODIFICATION: Store predictions in the 3D array slice for this bootstrap.
          boot_array[, 1, b] <- predictions
          
          deviance_mat[b, "Dev.Exp"] <- cor(predictions, full_tile_data[[response_var]], use = "complete.obs")^2
          deviance_mat[b, "RMSE"] <- sqrt(mean((predictions - full_tile_data[[response_var]])^2, na.rm = TRUE))
          deviance_mat[b, "R2"] <- cor(predictions, full_tile_data[[response_var]], use = "complete.obs")^2
          
          importance_matrix <- xgb.importance(model = xgb_model)
          if (nrow(importance_matrix) > 0) {
            imp_vals <- setNames(importance_matrix$Gain, importance_matrix$Feature)
            match_idx <- match(names(imp_vals), rownames(influence_mat))
            influence_mat[match_idx[!is.na(match_idx)], b] <- imp_vals[!is.na(match_idx)]
          }
          
            # --- 8. STORE PARTIAL DEPENDENCE PLOT DATA ---
            PDP_Storage <- list()
            predictor_means <- colMeans(subgrid_data[, predictors, drop = FALSE], na.rm = TRUE)
            
            for (j in seq_along(predictors)) {
              pred_name <- predictors[j]
              pdp_grid <- as.data.frame(matrix(rep(predictor_means, each = 100), nrow = 100))
              colnames(pdp_grid) <- names(predictor_means)
              pdp_grid[[pred_name]] <- EnvRanges$Env_Value[EnvRanges$Predictor == pred_name]
              pdp_grid <- pdp_grid[, predictors, drop = FALSE]
              
              pdp_predictions <- predict(xgb_model, newdata = as.matrix(pdp_grid))
              PD[, j, b] <- pdp_predictions
              
              PDP_Storage[[j]] <- data.frame(Predictor = pred_name, Env_Value = pdp_grid[[pred_name]],
                                             Replicate = paste0("Rep_", b), PDP_Value = PD[, j, b],
                                             X = subgrid_data$X[1:100], Y = subgrid_data$Y[1:100], FID = subgrid_data$FID[1:100])
            }
            all_pdp_long_list[[b]] <- bind_rows(PDP_Storage)
          }
          
          cat("DIAGNOSTIC: Bootstrap loop finished.\n", file = worker_log_file, append = TRUE)
          PDP_Long <- bind_rows(all_pdp_long_list)
         
        
        # -------------------------------------------------------
        # 8.5. PROCESS BOOTSTRAP PREDICTIONS
        # -------------------------------------------------------
        cat("DIAGNOSTIC: Processing bootstrap results...\n", file = worker_log_file, append = TRUE)
        
        # MODIFICATION: Calculate statistics by applying functions over the 3rd dimension (bootstraps).
        Mean_Prediction <- apply(boot_array, 1, mean, na.rm = TRUE)
        Uncertainty_SD <- apply(boot_array, 1, sd, na.rm = TRUE)
        if (n.boot == 1) Uncertainty_SD[is.na(Uncertainty_SD)] <- 0
        
        # MODIFICATION: Construct final data frame ensuring perfect alignment.
        boot_df <- data.table(
          FID = full_tile_data$FID,
          X = full_tile_data$X,
          Y = full_tile_data$Y,
          actual_bathy_t1 = full_tile_data[[response_var]], # Use actual bathy_t1 for comparison
          Mean_Prediction = Mean_Prediction,
          Uncertainty_SD = Uncertainty_SD
        )
        
        # -------------------------------------------------------
        # 9. SAVE OUTPUTS
        # -------------------------------------------------------
        tile_dir <- file.path(output_dir_train, tile_id)
        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE)
        
        write_fst(as.data.table(cv_results_df), file.path(tile_dir, paste0("cv_results_", pair, ".fst")))
        write_fst(as.data.table(deviance_mat), file.path(tile_dir, paste0("deviance_", pair, ".fst")))
        influence_df <- as.data.frame(influence_mat); influence_df$Predictor <- rownames(influence_mat)
        write_fst(as.data.table(influence_df), file.path(tile_dir, paste0("influence_", pair, ".fst")))
        write_fst(predictor_ranges, file.path(tile_dir, paste0("predictor_ranges_", pair, ".fst")))
        write_fst(boot_df, file.path(tile_dir, paste0("bootstraps_", pair, ".fst")))
        
        mean_raster_df <- boot_df[!is.na(Mean_Prediction), .(x = X, y = Y, z = Mean_Prediction)]
        if(nrow(mean_raster_df) > 0) {
          mean_raster <- rasterFromXYZ(mean_raster_df, crs = grid_crs_proj4)
          writeRaster(mean_raster, file.path(tile_dir, paste0("Mean_Boots_Prediction_", pair, ".tif")), format="GTiff", overwrite=TRUE)
        }
        sd_raster_df <- boot_df[!is.na(Uncertainty_SD), .(x = X, y = Y, z = Uncertainty_SD)]
        if(nrow(sd_raster_df) > 0) {
          sd_raster <- rasterFromXYZ(sd_raster_df, crs = grid_crs_proj4)
          writeRaster(sd_raster, file.path(tile_dir, paste0("Uncertainty_SD_", pair, ".tif")), format="GTiff", overwrite=TRUE)
        }
        
        final_model <- xgb.train(params = xgb_params, data = dtrain_full, nrounds = best_iteration, nthread = 1)
        saveRDS(final_model, file.path(tile_dir, paste0("xgb_model_", pair, ".rds")))
        
        write_fst(PDP_Long, file.path(tile_dir, paste0("pdp_data_long_", pair, ".fst")))
        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))
        
        
        # --- DIAGNOSTIC PLOT ---
        plot_data <- boot_df[!is.na(actual_bathy_t1), .(Actual = actual_bathy_t1, Predicted = Mean_Prediction)]
        fit_plot <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
          geom_point(alpha = 0.3, color = "darkblue") +
          geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
          labs(title = paste("Model Fit for Tile:", tile_id, "| Pair:", pair),
               subtitle = paste("Mean R-squared =", round(mean(deviance_mat[,"R2"], na.rm=TRUE), 3), "| Mean RMSE =", round(mean(deviance_mat[,"RMSE"], na.rm=TRUE), 3)),
               x = "Actual Bathymetry (t1)", y = "Mean Predicted Bathymetry (t1)") +
          theme_minimal()
        plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
        ggsave(filename = file.path(plot_dir, paste0("model_fit_", pair, ".png")), plot = fit_plot, width = 7, height = 7, dpi = 150)
        
        cat("DIAGNOSTIC: All outputs saved successfully.\n", file = worker_log_file, append = TRUE)
        
      }, error = function(e) {
        cat("FATAL ERROR in Tile:", tile_id, "| Pair:", pair, "|", conditionMessage(e), "\n", file = worker_log_file, append = TRUE)
        cat("Backtrace:\n", paste(capture.output(traceback()), collapse="\n"), "\n", file = worker_log_file, append = TRUE)
      })
      gc()
    } # End inner for loop
  } # End outer foreach loop
  
  # -------------------------------------------------------
  # 10. CLOSE PARALLEL CLUSTER AND CONSOLIDATE LOGS
  # -------------------------------------------------------
  if (exists("cl") && inherits(cl, "cluster")) {
    stopCluster(cl)
  }
  cat("\nParallel processing complete. Consolidating logs...\n")
  
  worker_log_files <- list.files(output_dir_train, pattern = "^log_worker_.*\\.txt$", recursive = TRUE, full.names = TRUE)
  for (log in worker_log_files) {
    content <- try(readLines(log), silent = TRUE)
    if (!inherits(content, "try-error")) {
      write(content, file = master_log_file, append = TRUE)
      file.remove(log)
    }
  }
  
  cat("\n[SUCCESS] Model Training Complete! Check `error_log_final.txt` for any issues.\n")
  return(results_list)
}


# new output from Chat GPT R Wizard -USE THIS ONE
#' -----------------------------------------------------------------------------

# ==============================================================================
# HYBRID MAIN TRAINING FUNCTION - V5.6----

# ==============================================================================
# ==============================================================================
# HYBRID MAIN TRAINING FUNCTION - V5.6
#   - Added global + per-bootstrap seeding
#   - Added bootstrap array saving (RDS + long FST)
#   - Enforced stable predictor set (no NA-based predictor removal)
#   - Added residual & delta_residual and full environmental fields to summary
#   - Added logging of seeds, feature counts, and basic distribution diagnostics
#   - Same core workflow as V5.5 (SHAP, bootstraps, spatial CV, PDP, etc.)
#   - Updated to use tuned default hyperparameters from XGB param search:
#       max_depth = 4, gamma = 0.5, lambda = 0, alpha = 0
#   - Still compatible with the new master predictor list using collapsed names,
#     e.g. hurr_count / hurr_strength / tsm + year_pair translation.
# ==============================================================================

#' Helper Function: Generate and Save a Diagnostic Plot for Spatial CV
#' (No changes)
#' -----------------------------------------------------------------------------
generate_cv_diagnostic_plot <- function(sf_data, block_geom, max_k, tile_id, pair, output_dir_train) {
  # ... (function content as before) ...
  diag_plot <- ggplot() +
    geom_sf(data = sf_data, color = "grey50", size = 0.1, alpha = 0.5) +
    geom_sf(data = block_geom, color = "blue", fill = "transparent", linewidth = 0.5) +
    labs(
      title = paste("Diagnostic Plot for Tile:", tile_id, "| Pair:", pair),
      subtitle = paste("Block Size:", sf::st_bbox(block_geom)[3] - sf::st_bbox(block_geom)[1], "m | Max Possible Folds (k):", max_k),
      x = "X (UTM)", y = "Y (UTM)"
    ) +
    theme_minimal()
  
  plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
  if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
  
  ggsave(
    file.path(plot_dir, paste0("spatial_cv_map_", pair, ".png")),
    plot = diag_plot, width = 8, height = 7, dpi = 150
  )
}


#' -----------------------------------------------------------------------------
#' Helper Function: Calculate Sample Weights Based on |delta_bathy|
#' (No changes)
#' -----------------------------------------------------------------------------
calculate_sample_weights <- function(delta_bathy, alpha = 1.5, epsilon = 1e-6) {
  # ... (function content as before) ...
  abs_change <- abs(delta_bathy) + epsilon
  weights <- abs_change^alpha
  weights <- weights / sum(weights, na.rm = TRUE) * length(weights)
  weights[!is.finite(weights)] <- epsilon
  return(weights)
}


Model_Train_Full_SpatialCV <- function(training_sub_grids_UTM, output_dir_train, year_pairs,
                                       master_predictor_list_path,
                                       block_size_m, n.boot = 10, n.folds = 5,
                                       use_weighted_loss = T, weight_alpha = 1.5,
                                       # --- TUNED PARAM DEFAULTS (from param search run_005) ---
                                       max_depth = 4, gamma = 0.5, lambda = 0, alpha = 0,
                                       # --- Global seed control ---
                                       global_seed = 12345) {
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  set.seed(global_seed)
  cat("\nStarting Hybrid XGBoost Model Training V5.6 (tuned defaults + SHAP)...\n")
  cat("Global RNG seed set to", global_seed, "\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  if (num_cores < 1) num_cores <- 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  on.exit({
    if (exists("cl") && inherits(cl, "cluster")) stopCluster(cl)
    closeAllConnections()
  }, add = TRUE)
  
  master_log_file <- file.path(output_dir_train, "error_log_final_v5.6.txt")
  cat("Error Log - XGBoost Hybrid Training V5.6 (run started at", as.character(Sys.time()), ")\n",
      file = master_log_file, append = FALSE)
  cat("Global seed used:", global_seed, "\n", file = master_log_file, append = TRUE)
  
  # -------------------------------------------------------
  # 2. MAIN PARALLEL PROCESSING LOOP
  # -------------------------------------------------------
  results_list <- foreach(
    i = seq_len(length(tiles_df)),
    .combine = 'c',
    .packages = c("data.table", "fst", "stringr"),
    .export = c("generate_cv_diagnostic_plot", "calculate_sample_weights", "global_seed"),
    .errorhandling = "pass"
  ) %dopar% {
    library(xgboost); library(dplyr); library(tidyr); library(sf)
    library(blockCV); library(raster); library(ggplot2); library(data.table)
    
    tile_id <- tiles_df[[i]]
    worker_log_file <- file.path(output_dir_train, tile_id,
                                 paste0("log_worker_", tile_id, "_v5.6.txt"))
    cat("Worker log V5.6 for tile:", tile_id, "started at", as.character(Sys.time()), "\n",
        file = worker_log_file, append = FALSE)
    
    if (!file.exists(master_predictor_list_path)) {
      stop("Master predictor list file not found.")
    }
    master_predictors <- readLines(master_predictor_list_path)
    
    # --- deterministic base seed per tile (for consistency) ---
    tile_hash <- sum(utf8ToInt(tile_id))
    worker_seed_base <- global_seed + tile_hash
    set.seed(worker_seed_base)
    cat("  - INFO: Worker base seed set to", worker_seed_base, "\n",
        file = worker_log_file, append = TRUE)
    
    for (pair in year_pairs) {
      tryCatch({
        # --- a. Load Data & Define Variables ---
        cat("\nProcessing Tile:", tile_id, "| Year Pair:", pair, "\n",
            file = worker_log_file, append = TRUE)
        file_prefix <- tile_id
        training_data_path <- file.path(output_dir_train, tile_id,
                                        paste0(tile_id, "_", pair, "_long.fst"))
        
        if (!file.exists(training_data_path)) {
          cat("  - WARN: Training data not found for pair", pair, "- skipping.\n",
              file = worker_log_file, append = TRUE)
          next
        }
        
        full_tile_data_raw <- as.data.frame(read_fst(training_data_path, as.data.table = TRUE))
        
        response_var <- "bathy_t1"
        actual_change_var <- "delta_bathy"
        available_cols <- names(full_tile_data_raw)
        
        if (!actual_change_var %in% available_cols) {
          if (all(c("bathy_t1", "bathy_t") %in% available_cols)) {
            full_tile_data_raw$delta_bathy <- full_tile_data_raw$bathy_t1 - full_tile_data_raw$bathy_t
            actual_change_var <- "delta_bathy"
          } else {
            cat("  - FATAL: Cannot find or calculate 'delta_bathy'. Skipping pair.\n",
                file = worker_log_file, append = TRUE)
            next
          }
        }
        
        # --- Predictor Selection with collapsed storms & year_pair translation ---
        current_predictors <- sapply(master_predictors, function(p) {
          if (p %in% c("hurr_count", "hurr_strength", "tsm")) {
            paste0(p, "_", pair)
          } else {
            p
          }
        }, USE.NAMES = FALSE)
        
        predictors_base <- intersect(current_predictors, available_cols)
        predictors <- setdiff(predictors_base,
                              c(response_var, actual_change_var, "X", "Y", "FID", "tile_id",
                                "year_t", "year_t1"))
        
        if ("bathy_t" %in% available_cols && !("bathy_t" %in% predictors)) {
          predictors <- c(predictors, "bathy_t")
          cat("  - INFO: Explicitly added 'bathy_t' as a predictor.\n",
              file = worker_log_file, append = TRUE)
        } else if (!("bathy_t" %in% available_cols)) {
          cat("  - FATAL: 'bathy_t' column missing. Cannot proceed. Skipping pair.\n",
              file = worker_log_file, append = TRUE)
          next
        }
        
        if (length(predictors) == 0 || !response_var %in% available_cols) {
          cat("  - WARN: Missing predictors or response 'bathy_t1'. Skipping pair.\n",
              file = worker_log_file, append = TRUE)
          next
        }
        
        # --- Log & lock feature set (no removal downstream) ---
        cat("  - INFO: LOCKED predictor set (", length(predictors), " predictors ).\n",
            file = worker_log_file, append = TRUE)
        cat(paste("    *", predictors), sep = "\n",
            file = worker_log_file, append = TRUE)
        cat("  - INFO: Target:", response_var, "| Weighting by:", actual_change_var, "\n",
            file = worker_log_file, append = TRUE)
        
        grid_crs_epsg <- 32617
        grid_crs_proj4 <- sf::st_crs(grid_crs_epsg)$proj4string
        
        # --- 3. FILTER & PREPARE TRAINING WORKFLOW DATA ---
        cols_to_select <- c(predictors, response_var, "bathy_t", actual_change_var,
                            "X", "Y", "FID")
        full_tile_data <- full_tile_data_raw %>%
          dplyr::select(any_of(cols_to_select)) %>%
          mutate(across(
            all_of(c(predictors, response_var, "bathy_t", actual_change_var)),
            ~ suppressWarnings(as.numeric(as.character(.)))
          ))
        
        # --- NA diagnostics (no predictor removal) ---
        na_check <- sapply(full_tile_data[, predictors, drop = FALSE],
                           function(col) sum(is.na(col)))
        if (any(na_check > 0)) {
          na_msg <- paste(names(na_check), na_check, sep = "=", collapse = ", ")
          cat("  - INFO: NA counts per predictor (before row filtering):", na_msg, "\n",
              file = worker_log_file, append = TRUE)
        }
        if (any(na_check == nrow(full_tile_data))) {
          all_na_cols <- names(na_check[na_check == nrow(full_tile_data)])
          cat("  - WARN: Predictors entirely NA across full tile:",
              paste(all_na_cols, collapse = ", "),
              ". Predictors retained for consistency but may reduce usable rows.\n",
              file = worker_log_file, append = TRUE)
        }
        
        subgrid_data <- full_tile_data %>%
          filter(is.finite(.data[[response_var]]) &
                   is.finite(.data[[actual_change_var]])) %>%
          filter(complete.cases(.[, predictors, drop = FALSE]))
        
        if (nrow(subgrid_data) < 100) {
          cat("  - WARN: Insufficient valid (finite + complete) training data (<100 rows). Skipping pair.\n",
              file = worker_log_file, append = TRUE)
          next
        }
        
        # Calculate weights
        if (use_weighted_loss) {
          sample_weights <- calculate_sample_weights(subgrid_data[[actual_change_var]],
                                                     alpha = weight_alpha)
        } else {
          sample_weights <- rep(1, nrow(subgrid_data))
        }
        if (any(!is.finite(sample_weights))) {
          sample_weights[!is.finite(sample_weights)] <- 1
        }
        
        # Basic distribution diagnostics for full training data
        pred_sd_full <- apply(subgrid_data[, predictors, drop = FALSE], 2, sd, na.rm = TRUE)
        n_constant_full <- sum(!is.na(pred_sd_full) & pred_sd_full < 1e-6)
        if (n_constant_full > 0) {
          const_names_full <- names(pred_sd_full)[pred_sd_full < 1e-6]
          cat("  - INFO: Constant predictors in full training data (retained for consistency):",
              paste(const_names_full, collapse = ", "), "\n",
              file = worker_log_file, append = TRUE)
        }
        
        subgrid_sf <- st_as_sf(subgrid_data, coords = c("X", "Y"),
                               remove = FALSE, crs = grid_crs_epsg)
        
       ### 3B. FILTER & PREPARE PREDICTION WORKFLOW DATA ###
        
        # --- 4. Initialize TRAINING WORKFLOW Storage ---
        deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot)
        colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
        
        influence_mat <- array(NA,
                               dim = c(length(predictors), n.boot))
        rownames(influence_mat) <- predictors
        colnames(influence_mat) <- paste0("Rep_", 1:n.boot)
        
        boot_array <- array(NA_real_,
                            dim = c(nrow(full_tile_data), 1, n.boot))
        dimnames(boot_array)[[2]] <- "prediction_t1"
        dimnames(boot_array)[[3]] <- paste0("Rep_", 1:n.boot)
        
        predictor_boot_ranges_list <- vector("list", n.boot)
        shap_summary_list <- vector("list", n.boot)
        
        ### 4B - Initialize PREDICTION STORAGE ###
        
        # --- 5. SETUP PDP DATA ---
        predictor_means <- colMeans(subgrid_data[, predictors, drop = FALSE],
                                    na.rm = TRUE)
        start_bathy_for_pdp <- predictor_means["bathy_t"]
        if (is.na(start_bathy_for_pdp)) {
          cat("  - FATAL (PDP Setup): Mean 'bathy_t' is NA. Skipping pair.\n",
              file = worker_log_file, append = TRUE)
          next
        }
        
        PredMins <- apply(subgrid_data[, predictors, drop = FALSE], 2, min, na.rm = TRUE)
        PredMaxs <- apply(subgrid_data[, predictors, drop = FALSE], 2, max, na.rm = TRUE)
        
        EnvRanges <- expand.grid(Env_Value = seq(0, 1, length.out = 100),
                                 Predictor = predictors)
        for (pred in predictors) {
          min_val <- PredMins[pred]
          max_val <- PredMaxs[pred]
          if (is.finite(min_val) && is.finite(max_val) && min_val != max_val) {
            EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(min_val, max_val,
                                                                    length.out = 100)
          } else {
            EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- predictor_means[pred]
          }
        }
        
        if (nrow(subgrid_data) >= 100) {
          # Seed PDP sampling deterministically for reproducibility
          pair_hash <- sum(utf8ToInt(as.character(pair)))
          pdp_seed <- global_seed + tile_hash + pair_hash + 999L
          set.seed(pdp_seed)
          sample_idx_pdp <- sample(seq_len(nrow(subgrid_data)), 100)
          
          EnvRanges$X <- rep(subgrid_data$X[sample_idx_pdp],
                             length.out = nrow(EnvRanges))
          EnvRanges$Y <- rep(subgrid_data$Y[sample_idx_pdp],
                             length.out = nrow(EnvRanges))
          EnvRanges$FID <- rep(subgrid_data$FID[sample_idx_pdp],
                               length.out = nrow(EnvRanges))
        } else {
          EnvRanges$X <- NA
          EnvRanges$Y <- NA
          EnvRanges$FID <- NA
        }
        
        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
        dimnames(PD)[[2]] <- predictors
        dimnames(PD)[[3]] <- paste0("Rep_", 1:n.boot)
        all_pdp_long_list <- list()
        
        # --- 6. SPATIAL CROSS VALIDATION ---
        best_iteration <- 100
        cv_results_df <- NULL
        
        grid_blocks <- st_make_grid(subgrid_sf, cellsize = block_size_m)
        if (inherits(grid_blocks, "sfc")) {
          grid_blocks <- st_sf(geometry = grid_blocks)
        }
        points_in_blocks <- st_intersects(grid_blocks, subgrid_sf)
        max_k <- length(which(lengths(points_in_blocks) > 0))
        
        if (max_k > 0) {
          valid_block_geom <- grid_blocks[which(lengths(points_in_blocks) > 0), ]
          generate_cv_diagnostic_plot(subgrid_sf, valid_block_geom, max_k,
                                      tile_id, pair, output_dir_train)
        } else {
          cat("  - WARN: No spatial blocks for CV plot.\n",
              file = worker_log_file, append = TRUE)
        }
        
        k_final <- min(n.folds, max_k)
        
        # deterministic CV seeding
        pair_hash <- sum(utf8ToInt(as.character(pair)))
        cv_seed <- global_seed + tile_hash + pair_hash + 1000L
        set.seed(cv_seed)
        cat("  - INFO: CV seed set to", cv_seed, "\n",
            file = worker_log_file, append = TRUE)
        
        tryCatch({
          if (k_final < 2) stop("CV not possible (k < 2).")
          
          scv <- cv_spatial(x = subgrid_sf,
                            size = block_size_m,
                            k = k_final,
                            iteration = 200)
          if (is.null(scv) || length(scv$folds_list) < k_final) {
            stop("blockCV failed.")
          }
          
          best_nrounds_per_fold <- c()
          rmse_per_fold <- c()
          mae_per_fold <- c()
          
          cv_params <- list(
            max_depth = max_depth,
            eta = 0.01,
            gamma = gamma,
            objective = "reg:squarederror",
            lambda = lambda,
            alpha = alpha
          )
          
          for (k in seq_len(k_final)) {
            train_idx <- unlist(scv$folds_list[[k]][1])
            test_idx <- unlist(scv$folds_list[[k]][2])
            
            train_idx <- train_idx[train_idx <= nrow(subgrid_data)]
            test_idx <- test_idx[test_idx <= nrow(subgrid_data)]
            
            if (length(train_idx) < 10 || length(test_idx) < 10 ||
                length(unique(subgrid_data[test_idx, response_var])) < 2) {
              next
            }
            
            dtrain_fold <- xgb.DMatrix(
              data = as.matrix(subgrid_data[train_idx, predictors]),
              label = subgrid_data[train_idx, response_var],
              weight = sample_weights[train_idx],
              missing = NA
            )
            dtest_fold <- xgb.DMatrix(
              data = as.matrix(subgrid_data[test_idx, predictors]),
              label = subgrid_data[test_idx, response_var],
              missing = NA
            )
            
            fold_model <- xgb.train(
              params = cv_params,
              data = dtrain_fold,
              nrounds = 1000,
              watchlist = list(train = dtrain_fold, test = dtest_fold),
              early_stopping_rounds = 15,
              eval_metric = c("rmse", "mae"),
              verbose = 0
            )
            
            if (!is.null(fold_model$best_iteration) &&
                fold_model$best_iteration > 0) {
              best_nrounds_per_fold <- c(best_nrounds_per_fold,
                                         fold_model$best_iteration)
              rmse_per_fold <- c(rmse_per_fold,
                                 fold_model$evaluation_log$test_rmse[fold_model$best_iteration])
              mae_per_fold <- c(mae_per_fold,
                                fold_model$evaluation_log$test_mae[fold_model$best_iteration])
            }
          }
          
          if (length(best_nrounds_per_fold) > 0) {
            best_iteration <- max(50, round(mean(best_nrounds_per_fold, na.rm = TRUE)))
            cv_results_df <- data.frame(
              tile_id = tile_id,
              year_pair = pair,
              best_iteration = best_iteration,
              test_rmse_mean = mean(rmse_per_fold, na.rm = TRUE),
              test_rmse_std = sd(rmse_per_fold, na.rm = TRUE),
              test_mae_mean = mean(mae_per_fold, na.rm = TRUE),
              test_mae_std = sd(mae_per_fold, na.rm = TRUE)
            )
            cat("  - INFO: CV successful. Optimal iteration:", best_iteration, "\n",
                file = worker_log_file, append = TRUE)
          } else {
            stop("CV loop failed.")
          }
        }, error = function(e) {
          cat("  - WARN: CV SKIPPED:", conditionMessage(e),
              ". Using default", best_iteration, "rounds.\n",
              file = worker_log_file, append = TRUE)
        })
        
        if (is.null(cv_results_df)) {
          cv_results_df <- data.frame(
            tile_id = tile_id,
            year_pair = pair,
            best_iteration = best_iteration,
            test_rmse_mean = NA,
            test_rmse_std = NA,
            test_mae_mean = NA,
            test_mae_std = NA
          )
        }
        
        # --- FINAL MODEL SETUP ---
        dtrain_full <- xgb.DMatrix(
          data = as.matrix(subgrid_data[, predictors]),
          label = subgrid_data[[response_var]],
          weight = sample_weights,
          missing = NA
        )
        dpredict_full <- xgb.DMatrix(
          data = as.matrix(full_tile_data[, predictors]),
          missing = NA
        )
        
        xgb_params <- list(
          max_depth = max_depth,
          eta = 0.01,
          gamma = gamma,
          subsample = 0.7,
          colsample_bytree = 0.8,
          objective = "reg:squarederror",
          lambda = lambda,
          alpha = alpha
        )
        
        # --- 7. BOOTSTRAP LOOP ---
        cat("DIAGNOSTIC: Starting bootstrap loop for", n.boot,
            "iterations...\n", file = worker_log_file, append = TRUE)
        
        tile_dir <- file.path(output_dir_train, tile_id)
        model_save_dir <- file.path(tile_dir, "bootstrap_models")
        if (!dir.exists(model_save_dir)) dir.create(model_save_dir, recursive = TRUE)
        
        max_boot_resample <- 10
        
        for (b in seq_len(n.boot)) {
          # deterministic seed per bootstrap iteration
          boot_seed <- global_seed + tile_hash + pair_hash + as.integer(b)
          set.seed(boot_seed)
          cat("  - Bootstrap iteration:", b, "| seed:", boot_seed, "\n",
              file = worker_log_file, append = TRUE)
          
          # resampling to avoid constant predictors where possible
          resample_try <- 1L
          boot_indices <- NULL
          boot_sample_data <- NULL
          n_constant_boot <- NA_integer_
          
          while (resample_try <= max_boot_resample) {
            boot_indices <- sample(seq_len(nrow(subgrid_data)), replace = TRUE)
            boot_sample_data <- subgrid_data[boot_indices, predictors, drop = FALSE]
            
            boot_sd <- apply(boot_sample_data, 2, sd, na.rm = TRUE)
            n_constant_boot <- sum(!is.na(boot_sd) & boot_sd < 1e-6)
            
            if (n_constant_boot == 0) {
              break
            }
            resample_try <- resample_try + 1L
          }
          
          if (!is.null(boot_sample_data) && !is.na(n_constant_boot) &&
              n_constant_boot > 0) {
            const_names_boot <- names(boot_sd)[boot_sd < 1e-6]
            cat("    - WARN: Constant predictors in bootstrap sample after",
                max_boot_resample, "resamples (retained):",
                paste(const_names_boot, collapse = ", "), "\n",
                file = worker_log_file, append = TRUE)
          }
          
          # Basic distribution diagnostics per bootstrap
          resp_range <- range(subgrid_data[[response_var]][boot_indices],
                              na.rm = TRUE)
          cat("    - DIAG: Response range in bootstrap:",
              paste0("min=", round(resp_range[1], 3),
                     ", max=", round(resp_range[2], 3)), "\n",
              file = worker_log_file, append = TRUE)
          
          dtrain_boot <- xgb.DMatrix(
            data = as.matrix(subgrid_data[boot_indices, predictors]),
            label = subgrid_data[boot_indices, response_var, drop = TRUE],
            weight = sample_weights[boot_indices],
            missing = NA
          )
          
          xgb_model <- xgb.train(
            params = xgb_params,
            data = dtrain_boot,
            nrounds = best_iteration,
            nthread = 1,
            verbose = 0
          )
          
          model_save_path <- file.path(model_save_dir,
                                       paste0("model_boot_", b, "_", pair, ".rds"))
          saveRDS(xgb_model, file = model_save_path)
          
          predictions_t1 <- predict(xgb_model, newdata = dpredict_full)
          boot_array[, 1, b] <- predictions_t1
          
          deviance_mat[b, "Dev.Exp"] <- cor(predictions_t1,
                                            full_tile_data[[response_var]],
                                            use = "complete.obs")^2
          deviance_mat[b, "RMSE"] <- sqrt(mean(
            (predictions_t1 - full_tile_data[[response_var]])^2,
            na.rm = TRUE
          ))
          deviance_mat[b, "R2"] <- cor(predictions_t1,
                                       full_tile_data[[response_var]],
                                       use = "complete.obs")^2
          
          importance_matrix <- xgb.importance(model = xgb_model)
          influence_mat[, b] <- 0
          if (nrow(importance_matrix) > 0) {
            imp_vals <- setNames(importance_matrix$Gain,
                                 importance_matrix$Feature)
            match_idx <- match(names(imp_vals), rownames(influence_mat))
            influence_mat[match_idx[!is.na(match_idx)], b] <- imp_vals[!is.na(match_idx)]
          }
          
          if (is.null(boot_sample_data)) {
            boot_sample_data <- subgrid_data[boot_indices, predictors, drop = FALSE]
          }
          
          predictor_boot_ranges_list[[b]] <- data.table(
            Predictor = predictors,
            Min_Value = sapply(boot_sample_data, min, na.rm = TRUE),
            Max_Value = sapply(boot_sample_data, max, na.rm = TRUE),
            Mean_Value = sapply(boot_sample_data, mean, na.rm = TRUE),
            Bootstrap_Iteration = b
          )
          
          # --- 8. STORE PDP DATA (as CHANGE) ---
          PDP_Storage <- list()
          for (j in seq_along(predictors)) {
            pred_name <- predictors[j]
            pdp_grid <- as.data.frame(
              matrix(rep(predictor_means, each = 100), nrow = 100)
            )
            colnames(pdp_grid) <- names(predictor_means)
            
            if (pred_name %in% names(pdp_grid) &&
                pred_name %in% EnvRanges$Predictor) {
              pdp_grid[[pred_name]] <- EnvRanges$Env_Value[EnvRanges$Predictor == pred_name]
            }
            
            pdp_grid_matrix <- as.matrix(pdp_grid[, predictors, drop = FALSE])
            pdp_predictions_t1 <- predict(xgb_model, newdata = pdp_grid_matrix)
            pdp_change <- pdp_predictions_t1 - start_bathy_for_pdp
            PD[, j, b] <- pdp_change
            
            if (pred_name %in% names(pdp_grid)) {
              PDP_Storage[[pred_name]] <- data.frame(
                Predictor = pred_name,
                Env_Value = pdp_grid[[pred_name]],
                Replicate = paste0("Rep_", b),
                PDP_Value = PD[, j, b],
                X = EnvRanges$X[EnvRanges$Predictor == pred_name],
                Y = EnvRanges$Y[EnvRanges$Predictor == pred_name],
                FID = EnvRanges$FID[EnvRanges$Predictor == pred_name]
              )
            }
          }
          
          valid_pdp_storage <- PDP_Storage[!sapply(PDP_Storage, is.null)]
          if (length(valid_pdp_storage) > 0) {
            all_pdp_long_list[[b]] <- bind_rows(valid_pdp_storage)
          } else {
            all_pdp_long_list[[b]] <- data.table()
          }
          
          # --- Calculate SHAP summary ---
          cat("    - SHAP: Preparing data...\n",
              file = worker_log_file, append = TRUE)
          valid_predictors_for_shap <- predictors[
            sapply(subgrid_data[, predictors, drop = FALSE], is.numeric)
          ]
          
          if (length(valid_predictors_for_shap) > 0) {
            shap_sample_size <- min(1000, length(boot_indices))
            shap_sample_idx <- sample(boot_indices, shap_sample_size)
            
            shap_matrix_boot <- as.matrix(
              subgrid_data[shap_sample_idx, valid_predictors_for_shap, drop = FALSE]
            )
            
            col_sds <- apply(shap_matrix_boot, 2, sd, na.rm = TRUE)
            non_constant_preds <- names(col_sds[!is.na(col_sds) & col_sds > 1e-6])
            
            if (length(non_constant_preds) > 0) {
              cat("    - SHAP: Found", length(non_constant_preds),
                  "non-constant predictors for SHAP.\n",
                  file = worker_log_file, append = TRUE)
              tryCatch({
                shap_matrix_boot_clean <- shap_matrix_boot[, non_constant_preds,
                                                           drop = FALSE]
                
                cat("    - SHAP: Running predict(predcontrib=TRUE)...\n",
                    file = worker_log_file, append = TRUE)
                
                shap_values_matrix <- predict(
                  xgb_model,
                  newdata = shap_matrix_boot_clean,
                  predcontrib = TRUE
                )
                cat("    - SHAP: predict() complete, processing results...\n",
                    file = worker_log_file, append = TRUE)
                
                shap_values_df <- as.data.frame(
                  shap_values_matrix[, -ncol(shap_values_matrix)]
                )
                colnames(shap_values_df) <- non_constant_preds
                
                shap_mean_abs <- colMeans(abs(shap_values_df), na.rm = TRUE)
                
                shap_mean_abs_df <- data.table(
                  Predictor = names(shap_mean_abs),
                  Mean_Abs_SHAP = as.numeric(shap_mean_abs)
                )
                
                if (is.null(shap_mean_abs_df) || nrow(shap_mean_abs_df) == 0) {
                  stop("SHAP calculation returned empty or NULL data.")
                }
                
                shap_mean_abs_df[, Bootstrap_Iteration := b]
                shap_summary_list[[b]] <- shap_mean_abs_df[
                  , .(Predictor, Mean_Abs_SHAP, Bootstrap_Iteration)
                ]
                cat("    - SHAP: Summary for bootstrap", b, "saved.\n",
                    file = worker_log_file, append = TRUE)
              }, error = function(e) {
                cat("  - WARN: SHAP calculation failed for bootstrap", b, ":",
                    conditionMessage(e), "\n",
                    file = worker_log_file, append = TRUE)
                shap_summary_list[[b]] <- data.table(
                  Predictor = character(),
                  Mean_Abs_SHAP = numeric(),
                  Bootstrap_Iteration = integer()
                )
              })
            } else {
              cat("  - WARN: No non-constant predictors in boot sample for SHAP (Bootstrap",
                  b, ").\n", file = worker_log_file, append = TRUE)
              shap_summary_list[[b]] <- data.table(
                Predictor = character(),
                Mean_Abs_SHAP = numeric(),
                Bootstrap_Iteration = integer()
              )
            }
          } else {
            cat("  - WARN: No valid numeric predictors for SHAP (Bootstrap", b, ").\n",
                file = worker_log_file, append = TRUE)
            shap_summary_list[[b]] <- data.table(
              Predictor = character(),
              Mean_Abs_SHAP = numeric(),
              Bootstrap_Iteration = integer()
            )
          }
          
          ### 7b - Prediction boostrap/ model loop (pass all the above needed data to the prediction workflow inside the main boostrap loop ###)
        } # End bootstrap loop
        
        cat("DIAGNOSTIC: Bootstrap loop finished.\n",
            file = worker_log_file, append = TRUE)
        
        # --- 8.5 PROCESS BOOTSTRAP ---
        Mean_Prediction_t1 <- apply(boot_array, 1, mean, na.rm = TRUE)
        Uncertainty_SD <- apply(boot_array, 1, sd, na.rm = TRUE)
        if (n.boot <= 1) {
          Uncertainty_SD[!is.na(Uncertainty_SD)] <- 0
        }
        
        # summary residuals & delta residuals
        actual_bathy_t1_vec <- full_tile_data[[response_var]]
        bathy_t_vec <- full_tile_data$bathy_t
        delta_bathy_actual_vec <- full_tile_data[[actual_change_var]]
        
        prediction_residual <- actual_bathy_t1_vec - Mean_Prediction_t1
        mean_predicted_change <- Mean_Prediction_t1 - bathy_t_vec
        delta_residual <- delta_bathy_actual_vec - mean_predicted_change
        
        # attach environmental predictors to summary
        env_predictors_df <- as.data.table(full_tile_data[, predictors, drop = FALSE])
        
        boot_df <- data.table(
          FID = full_tile_data$FID,
          X = full_tile_data$X,
          Y = full_tile_data$Y,
          actual_bathy_t1 = actual_bathy_t1_vec,
          bathy_t = bathy_t_vec,
          delta_bathy_actual = delta_bathy_actual_vec,
          mean_predicted_bathy_t1 = Mean_Prediction_t1,
          mean_prediction_t1 = Mean_Prediction_t1,           # alias name
          uncertainty_sd_bathy_t1 = Uncertainty_SD,
          uncertainty_SD = Uncertainty_SD,                   # alias name
          mean_predicted_change = mean_predicted_change,
          prediction_residual = prediction_residual,
          delta_residual = delta_residual
        )
        
        # Avoid duplicating bathy_t if already in predictors
        if ("bathy_t" %in% names(env_predictors_df)) {
          env_predictors_df[, bathy_t := NULL]
        }
        boot_df <- cbind(boot_df, env_predictors_df)
        
        predictor_boot_ranges_all <- rbindlist(predictor_boot_ranges_list, fill = TRUE)
        if (nrow(predictor_boot_ranges_all) > 0 &&
            "Min_Value" %in% names(predictor_boot_ranges_all)) {
          predictor_boot_summary <- predictor_boot_ranges_all[, .(
            Min_across_boots = suppressWarnings(min(Min_Value, na.rm = TRUE)),
            Max_across_boots = suppressWarnings(max(Max_Value, na.rm = TRUE)),
            Mean_across_boots = mean(Mean_Value, na.rm = TRUE)
          ), by = Predictor]
          predictor_boot_summary[is.infinite(Min_across_boots),
                                 Min_across_boots := NA]
          predictor_boot_summary[is.infinite(Max_across_boots),
                                 Max_across_boots := NA]
        } else {
          predictor_boot_summary <- data.table(
            Predictor = predictors,
            Min_across_boots = NA_real_,
            Max_across_boots = NA_real_,
            Mean_across_boots = NA_real_
          )
        }
        
        valid_shap_summaries <- Filter(function(x) !is.null(x) && nrow(x) > 0,
                                       shap_summary_list)
        if (length(valid_shap_summaries) > 0) {
          shap_summary_all <- rbindlist(valid_shap_summaries, fill = TRUE)
          if (nrow(shap_summary_all) > 0 &&
              "Mean_Abs_SHAP" %in% names(shap_summary_all)) {
            shap_summary_all[, Mean_Abs_SHAP := as.numeric(Mean_Abs_SHAP)]
            shap_final_summary <- shap_summary_all[is.finite(Mean_Abs_SHAP),
                                                   .(Overall_Mean_Abs_SHAP = mean(
                                                     Mean_Abs_SHAP,
                                                     na.rm = TRUE
                                                   )),
                                                   by = Predictor]
            setorder(shap_final_summary, -Overall_Mean_Abs_SHAP)
          } else {
            shap_final_summary <- data.table(
              Predictor = character(),
              Overall_Mean_Abs_SHAP = numeric()
            )
          }
        } else {
          shap_final_summary <- data.table(
            Predictor = character(),
            Overall_Mean_Abs_SHAP = numeric()
          )
        }
        
        # --- Save full bootstrap array (RDS) and long-form predictions (FST) ---
        tile_dir <- file.path(output_dir_train, tile_id)
        saveRDS(boot_array,
                file = file.path(tile_dir, paste0("boot_array_", pair, ".rds")))
        
        n_cells <- nrow(full_tile_data)
        if (n_cells > 0 && n.boot > 0) {
          boot_long <- data.table(
            FID = rep(full_tile_data$FID, times = n.boot),
            X = rep(full_tile_data$X, times = n.boot),
            Y = rep(full_tile_data$Y, times = n.boot),
            bathy_t = rep(bathy_t_vec, times = n.boot),
            actual_bathy_t1 = rep(actual_bathy_t1_vec, times = n.boot),
            delta_bathy_actual = rep(delta_bathy_actual_vec, times = n.boot),
            bootstrap_iter = rep(seq_len(n.boot), each = n_cells),
            prediction_t1 = as.vector(boot_array[, 1, , drop = FALSE])
          )
          
          env_pred_rep <- env_predictors_df[
            rep(seq_len(nrow(env_predictors_df)), times = n.boot)
          ]
          boot_long <- cbind(boot_long, env_pred_rep)
          
          write_fst(
            boot_long,
            file.path(tile_dir, paste0("bootstraps_full_array_long_", pair, ".fst"))
          )
        }
        
        # --- 9. SAVE TRAINING OUTPUTS ---
        write_fst(as.data.table(cv_results_df),
                  file.path(tile_dir, paste0("cv_results_", pair, ".fst")))
        
        write_fst(as.data.table(deviance_mat),
                  file.path(tile_dir, paste0("deviance_", pair, ".fst")))
        
        influence_df <- as.data.frame(influence_mat)
        influence_df$Predictor <- rownames(influence_mat)
        influence_df <- influence_df[, c("Predictor",
                                         setdiff(names(influence_df), "Predictor"))]
        write_fst(as.data.table(influence_df),
                  file.path(tile_dir, paste0("influence_", pair, ".fst")))
        
        write_fst(boot_df,
                  file.path(tile_dir, paste0("bootstraps_summary_df_", pair, ".fst")))
        
        write_fst(predictor_boot_summary,
                  file.path(tile_dir, paste0("predictor_boot_ranges_", pair, ".fst")))
        
        write_fst(shap_final_summary,
                  file.path(tile_dir, paste0("shap_summary_", pair, ".fst")))
        
        mean_change_raster_df <- boot_df[is.finite(mean_predicted_change),
                                         .(x = X, y = Y, z = mean_predicted_change)]
        if (nrow(mean_change_raster_df) > 0) {
          mean_change_raster <- rasterFromXYZ(mean_change_raster_df,
                                              crs = grid_crs_proj4)
          writeRaster(
            mean_change_raster,
            file.path(tile_dir, paste0("Mean_predicted_change_", pair, ".tif")),
            format = "GTiff",
            overwrite = TRUE
          )
        }
        
        sd_raster_df <- boot_df[is.finite(uncertainty_sd_bathy_t1),
                                .(x = X, y = Y, z = uncertainty_sd_bathy_t1)]
        if (nrow(sd_raster_df) > 0) {
          sd_raster <- rasterFromXYZ(sd_raster_df, crs = grid_crs_proj4)
          writeRaster(
            sd_raster,
            file.path(tile_dir, paste0("Uncertainty_SD_bathy_t1_", pair, ".tif")),
            format = "GTiff",
            overwrite = TRUE
          )
        }
        
        final_model <- xgb.train(
          params = xgb_params,
          data = dtrain_full,
          nrounds = best_iteration,
          nthread = 1,
          verbose = 0
        )
        saveRDS(final_model,
                file.path(tile_dir, paste0("xgb_model_final_", pair, ".rds")))
        
        valid_pdp_long_list <- Filter(function(x) !is.null(x) && nrow(x) > 0,
                                      all_pdp_long_list)
        if (length(valid_pdp_long_list) > 0) {
          PDP_Long <- rbindlist(valid_pdp_long_list, fill = TRUE)
          if (nrow(PDP_Long) > 0) {
            write_fst(PDP_Long,
                      file.path(tile_dir, paste0("pdp_data_long_", pair, ".fst")))
          }
        }
        
        write_fst(as.data.table(EnvRanges),
                  file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))
        
        # --- DIAGNOSTIC PLOTS ---
        plot_data_fit <- boot_df[
          is.finite(actual_bathy_t1) & is.finite(mean_predicted_bathy_t1),
          .(Actual = actual_bathy_t1, Predicted = mean_predicted_bathy_t1)
        ]
        if (nrow(plot_data_fit) > 2) {
          fit_plot <- ggplot(plot_data_fit, aes(x = Actual, y = Predicted)) +
            geom_point(alpha = 0.3, color = "darkblue") +
            geom_abline(slope = 1, intercept = 0, color = "red",
                        linetype = "dashed", linewidth = 1) +
            labs(
              title = paste("Model Fit (t1):", tile_id, "|", pair),
              subtitle = paste(
                "Mean R2 =", round(mean(deviance_mat[, "R2"], na.rm = TRUE), 3),
                "| Mean RMSE =",
                round(mean(deviance_mat[, "RMSE"], na.rm = TRUE), 3)
              ),
              x = "Actual Bathy (t1)",
              y = "Mean Predicted Bathy (t1)"
            ) +
            theme_minimal()
          
          plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
          if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
          
          ggsave(
            filename = file.path(plot_dir, paste0("model_fit_t1_", pair, ".png")),
            plot = fit_plot,
            width = 7,
            height = 7,
            dpi = 150
          )
        }
        
        if (!is.null(shap_final_summary) && nrow(shap_final_summary) > 0) {
          plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
          if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
          
          n_shap_plot <- min(20, nrow(shap_final_summary))
          shap_plot_data <- shap_final_summary[1:n_shap_plot]
          
          shap_plot <- ggplot(shap_plot_data,
                              aes(x = reorder(Predictor, Overall_Mean_Abs_SHAP),
                                  y = Overall_Mean_Abs_SHAP)) +
            geom_bar(stat = "identity", fill = "steelblue") +
            coord_flip() +
            labs(
              title = paste("SHAP Importance:", tile_id, "|", pair),
              subtitle = paste("Based on", n.boot, "models"),
              x = "Predictor",
              y = "Overall Mean Absolute SHAP"
            ) +
            theme_minimal() +
            theme(axis.text.y = element_text(size = 8))
          
          ggsave(
            filename = file.path(plot_dir, paste0("shap_importance_", pair, ".png")),
            plot = shap_plot,
            width = 8,
            height = 0.3 * n_shap_plot + 2,
            dpi = 150,
            limitsize = FALSE
          )
          cat("  - INFO: SHAP plot saved.\n",
              file = worker_log_file, append = TRUE)
        } else {
          cat("  - WARN: No SHAP data to plot.\n",
              file = worker_log_file, append = TRUE)
        }
        
      }, error = function(e) {
        cat("FATAL ERROR in Tile:", tile_id, "| Pair:", pair, "|",
            conditionMessage(e), "\n",
            file = worker_log_file, append = TRUE)
        cat("Backtrace:\n",
            paste(capture.output(traceback()), collapse = "\n"), "\n",
            file = worker_log_file, append = TRUE)
      })
      
      ### 9b - SAVE PREDICTION OUTPUTS ###
      gc()
    } # End inner for loop (year_pairs)
  } # End outer foreach loop (tiles)
  
  # --- 10. Cleanup & Log Consolidation ---
  if (exists("cl") && inherits(cl, "cluster")) stopCluster(cl)
  cat("\nParallel processing complete. Consolidating logs...\n")
  worker_log_files <- list.files(
    output_dir_train,
    pattern = "^log_worker_.*_v5\\.6\\.txt$",
    recursive = TRUE,
    full.names = TRUE
  )
  for (log in worker_log_files) {
    content <- try(readLines(log), silent = TRUE)
    if (!inherits(content, "try-error")) {
      write(content, file = master_log_file, append = TRUE)
      try(file.remove(log), silent = TRUE)
    }
  }
  cat("\n[SUCCESS] Model Training V5.6 Complete! Check log file for details.\n")
  return(results_list)
}




#--------------
#FUNCTION CALL 

#
# --- 1. Define Parameters ---
training_subgrid_UTM <- st_read(training_grid_gpkg)
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles"
years <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
block_size <- 200
master_predictor_file <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles/final_master_predictor_list.txt"

# --- 2. Load and (optionally) Subset Your Tile Data ---
# Assuming 'all_tiles' is your full sf object of training grid polygons
all_tiles <- training_subgrid_UTM

# # Example of running a subset for testing
# specific_tile_ids <- c("BH4S556X_3")
# test_tiles <- all_tiles %>%
#   filter(tile_id %in% specific_tile_ids)

cat("Starting test run on", nrow(test_tiles), "tiles...\n")
cat("Starting test run on", nrow(all_tiles), "tiles...\n")
#
# --- 3. Execute the Function Call ---
Sys.time()
Model_Train_Full_SpatialCV(
  training_sub_grids_UTM =  all_tiles, #test_tiles, for a specific tile ID # Use 'all_tiles' for a full run
  output_dir_train = output_dir,
  year_pairs = years,
  master_predictor_list_path = master_predictor_file,
  block_size_m = block_size,
  n.boot = 20,
  n.folds = 5
)
Sys.time()

closeAllConnections()


# ==============================================================================
#   DIAGNOSTIC FUNCTION (REFACTORED)
# ==============================================================================
#' Diagnose XGBoost Model and Input Data (Diagnostic and error checking tool)
#'
#' This function loads all the final outputs for a specific tile and year-pair
#' and compiles a diagnostic summary. It is designed to help troubleshoot
#' issues by providing a snapshot of the data that was fed into the model.
#'
#' @param output_dir_train The base directory where the tile folders are located.
#' @param tile_id The specific tile ID you want to diagnose.
#' @param year_pair The specific year pair you want to diagnose.
#' @param master_predictor_list_path The file path to the master predictor list .txt file.
#'
#' @return A detailed list object containing diagnostic information about the
#'   data and the trained model. This list is also printed to the console.
#'
diagnose_xgb_run <- function(output_dir_train, tile_id, year_pair, master_predictor_list_path) {
  
  cat("--- Starting Diagnosis for Tile:", tile_id, "| Year Pair:", year_pair, "---\n\n")
  
  # -------------------------------------------------------
  # 1. DEFINE FILE PATHS
  # -------------------------------------------------------
  base_path <- file.path(output_dir_train, tile_id)
  if (!dir.exists(base_path)) {
    stop("Directory for the specified tile_id does not exist:", base_path)
  }
  
  file_prefix <- sub("_[0-9]+$", "", tile_id)
  
  paths <- list(
    training_data = file.path(base_path, paste0(file_prefix, "_", year_pair, "_long.fst")),
    xgb_model = file.path(base_path, paste0("xgb_model_", year_pair, ".rds")),
    deviance = file.path(base_path, paste0("deviance_", year_pair, ".fst")),
    influence = file.path(base_path, paste0("influence_", year_pair, ".fst"))
  )
  
  # Check if all necessary files exist
  files_exist <- sapply(paths, file.exists)
  if (!all(files_exist)) {
    stop("One or more required files are missing for this tile/year-pair. Missing: ",
         paste(names(paths[!files_exist]), collapse = ", "))
  }
  if (!file.exists(master_predictor_list_path)) {
    stop("Master predictor list file not found at:", master_predictor_list_path)
  }
  
  
  # -------------------------------------------------------
  # 2. LOAD AND PREPARE DATA (Replicating the training script logic)
  # -------------------------------------------------------
  cat("1. Loading and preparing data...\n")
  training_data <- read_fst(paths$training_data)
  master_predictors <- readLines(master_predictor_list_path)
  
  # Set the new response variable
  response_var <- "bathy_t1"
  
  # Dynamically create the predictor list for the current year_pair
  current_predictors <- sapply(master_predictors, function(p) {
    if (p %in% c("hurr_count", "hurr_strength", "tsm")) {
      return(paste0(p, "_", year_pair))
    } else {
      return(p)
    }
  }, USE.NAMES = FALSE)
  
  predictors <- intersect(current_predictors, names(training_data))
  
  # Perform the same filtering and coercion as the training script
  subgrid_data <- training_data %>%
    dplyr::select(any_of(c(predictors, response_var))) %>%
    dplyr::mutate(across(everything(), .fns = ~as.numeric(as.character(.)))) %>%
    drop_na()
  
  # -------------------------------------------------------
  # 3. PERFORM DATA DIAGNOSTICS
  # -------------------------------------------------------
  cat("2. Performing data diagnostics on the prepared data...\n")
  
  # Check the variance of each predictor
  predictor_variances <- sapply(subgrid_data[predictors], function(x) var(x, na.rm = TRUE))
  
  # Check the number of unique values for each predictor
  unique_value_counts <- sapply(subgrid_data[predictors], function(x) length(unique(x)))
  
  data_summary <- list(
    n_rows = nrow(subgrid_data),
    n_predictors = length(predictors),
    response_variable_summary = summary(subgrid_data[[response_var]]),
    predictor_variances = sort(predictor_variances, decreasing = TRUE),
    predictors_with_zero_variance = names(predictor_variances[predictor_variances == 0]),
    unique_value_counts = sort(unique_value_counts)
  )
  
  
  # -------------------------------------------------------
  # 4. PERFORM MODEL DIAGNOSTICS
  # -------------------------------------------------------
  cat("3. Performing model diagnostics...\n")
  xgb_model <- readRDS(paths$xgb_model)
  deviance_df <- read_fst(paths$deviance)
  influence_df <- read_fst(paths$influence)
  
  # Summarize predictor influence
  mean_influence <- influence_df %>%
    tidyr::pivot_longer(cols = starts_with("Rep_"), names_to = "Replicate", values_to = "Influence") %>%
    group_by(Predictor) %>%
    summarise(Mean_Influence = mean(Influence, na.rm = TRUE)) %>%
    arrange(desc(Mean_Influence))
  
  model_summary <- list(
    model_class = class(xgb_model),
    number_of_trees = xgb_model$niter,
    top_5_influential_predictors = head(mean_influence, 5),
    mean_deviance_explained = mean(deviance_df$Dev.Exp, na.rm = TRUE),
    mean_rmse = mean(deviance_df$RMSE, na.rm = TRUE),
    mean_r_squared = mean(deviance_df$R2, na.rm = TRUE)
  )
  
  # -------------------------------------------------------
  # 5. COMPILE AND RETURN FINAL REPORT
  # -------------------------------------------------------
  cat("4. Compiling final report...\n\n")
  final_report <- list(
    data_diagnostics = data_summary,
    model_diagnostics = model_summary
  )
  
  # Print the report to the console in a readable format
  print(final_report)
  
  cat("\n--- Diagnosis Complete ---\n")
  return(invisible(final_report))
}

# --- EXAMPLE FUNCTION CALL ---
diagnose_xgb_run(
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles",
  tile_id = "BH4S556X_3",
  year_pair = "2004_2006",
  master_predictor_list_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles/boruta_master_predictor_list.txt"
)



# Verify model outputs look good
# Performance 
deviance <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/deviance_2004_2006.fst")
glimpse(deviance)
influence <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/influence_2004_2006.fst")
glimpse(influence)
cv <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/cv_results_2004_2006.fst")
glimpse(cv)

# Data shape / relationships 
envranges <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/pdp_env_ranges_2004_2006.fst")
glimpse(envranges)
pdp <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/pdp_data_long_2004_2006.fst")
glimpse(pdp)
shaps <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/shap_summary_2004_2006.fst")
glimpse(shaps)


# Boostrapped data 
pred_boot_ranges <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/predictor_boot_ranges_2004_2006.fst")
glimpse(pred_boot_ranges)
bootdf <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/bootstraps_summary_df_2004_2006.fst")
glimpse(bootdf)
boots <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/bootstraps_full_array_long_2004_2006.fst")
glimpse(boots)
boot_array <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/boot_array_2004_2006.rds")
glimpse(boot_array)







# ==============================================================================
#  MODEL PERFORMANCE REPORTING FUNCTION (REFACTORED)
# ==============================================================================
#' Create Visual Reports for XGBoost Model Performance
#'
#' This function scans a directory for XGBoost model outputs, including
#' cross-validation (`cv_results_*.fst`) and deviance (`deviance_*.fst`) files.
#' It can process all tiles or a specified subset and generates two separate
#' PNG images with summary plots.
#'
#' @param output_dir_train The base directory where the tile folders are located.
#' @param tile_ids (Optional) A character vector of specific tile IDs to process.
#'   If NULL (default), all tiles in the output directory will be processed.
#' @param cv_report_filename The name of the output PNG file for CV results.
#' @param perf_report_filename The name of the output PNG file for performance results.
#'
#' @return None. Two PNG files are saved to the `output_dir_train`.
#'
create_xgboost_performance_report <- function(output_dir_train,
                                              tile_ids = NULL,
                                              cv_report_filename = "xgboost_cv_summary_report.png",
                                              perf_report_filename = "xgboost_performance_summary_report.png") {
  
  # -------------------------------------------------------
  # 1. LOAD LIBRARIES and FIND FILES
  # -------------------------------------------------------
  cat("Starting XGBoost performance report generation...\n")
  library(dplyr)
  library(stringr)
  library(fst)
  
  all_cv_files <- list.files(
    path = output_dir_train,
    pattern = "^cv_results_.*\\.fst$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  all_deviance_files <- list.files(
    path = output_dir_train,
    pattern = "^deviance_.*\\.fst$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  cv_files <- all_cv_files
  deviance_files <- all_deviance_files
  
  if (!is.null(tile_ids)) {
    cat("Filtering results for", length(tile_ids), "specified tiles...\n")
    # More robustly extract tile_id from path
    get_tile_id_from_path <- function(path) {
      basename(dirname(path))
    }
    cv_files <- all_cv_files[get_tile_id_from_path(all_cv_files) %in% tile_ids]
    deviance_files <- all_deviance_files[get_tile_id_from_path(all_deviance_files) %in% tile_ids]
  } else {
    cat("Processing all available tiles...\n")
  }
  
  
  if (length(cv_files) == 0 && length(deviance_files) == 0) {
    stop("No 'cv_results' or 'deviance' files found for the specified tiles.")
  }
  
  # -------------------------------------------------------
  # 2. PROCESS CROSS-VALIDATION (CV) RESULTS
  # -------------------------------------------------------
  if (length(cv_files) > 0) {
    cat("Processing", length(cv_files), "CV result files...\n")
    
    cv_df_list <- lapply(cv_files, function(f) {
      tryCatch(read_fst(f), error = function(e) NULL)
    })
    cv_df <- bind_rows(cv_df_list)
    
    if (nrow(cv_df) > 0 && "best_iteration" %in% names(cv_df) && is.numeric(cv_df$best_iteration)) {
      cat("Generating CV summary plot...\n")
      png(
        file.path(output_dir_train, cv_report_filename),
        width = 1200, height = 1000, res = 100
      )
      par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))
      
      hist(cv_df$best_iteration,
           main = "Distribution of Best Training Iterations",
           xlab = "Optimal Number of Rounds",
           col = "cornflowerblue", border = "white")
      
      if ("test_rmse_mean" %in% names(cv_df) && !all(is.na(cv_df$test_rmse_mean))) {
        avg_rmse <- cv_df %>%
          group_by(year_pair) %>%
          summarise(mean_rmse = mean(test_rmse_mean, na.rm = TRUE), .groups = 'drop')
        barplot(height = avg_rmse$mean_rmse, names.arg = avg_rmse$year_pair,
                main = "Average Test RMSE by Year Pair",
                ylab = "Mean Test RMSE", col = "darkseagreen")
      } else {
        plot.new(); title("No RMSE data available")
      }
      
      if ("test_mae_mean" %in% names(cv_df) && !all(is.na(cv_df$test_mae_mean))) {
        avg_mae <- cv_df %>%
          group_by(year_pair) %>%
          summarise(mean_mae = mean(test_mae_mean, na.rm = TRUE), .groups = 'drop')
        barplot(height = avg_mae$mean_mae, names.arg = avg_mae$year_pair,
                main = "Average Test MAE by Year Pair",
                ylab = "Mean Test MAE", col = "darkkhaki")
      } else {
        plot.new(); title("No MAE data available")
      }
      
      if ("test_rmse_mean" %in% names(cv_df) && "test_mae_mean" %in% names(cv_df) &&
          !all(is.na(cv_df$test_rmse_mean)) && !all(is.na(cv_df$test_mae_mean))) {
        plot(cv_df$test_rmse_mean, cv_df$test_mae_mean,
             main = "CV Error: RMSE vs. MAE",
             xlab = "Test RMSE", ylab = "Test MAE",
             pch = 19, col = rgb(0.1, 0.1, 0.1, 0.3))
        abline(a = 0, b = 1, col = "red", lty = 2)
      } else {
        plot.new(); title("No RMSE/MAE data for scatterplot")
      }
      
      dev.off()
      cat("CV report saved to:", file.path(output_dir_train, cv_report_filename), "\n")
      
    } else {
      cat("No valid CV data with a 'best_iteration' column found to plot.\n")
    }
    
  } else {
    cat("No CV result files found to process.\n")
  }
  
  # -------------------------------------------------------
  # 3. PROCESS MODEL PERFORMANCE (DEVIANCE) RESULTS
  # -------------------------------------------------------
  if (length(deviance_files) > 0) {
    cat("Processing", length(deviance_files), "deviance result files...\n")
    deviance_list <- lapply(deviance_files, function(fp) {
      df <- read_fst(fp)
      df$year_pair <- str_extract(basename(fp), "\\d{4}_\\d{4}")
      df$tile_id <- basename(dirname(fp))
      return(df)
    })
    deviance_df <- bind_rows(deviance_list)
    
    if(nrow(deviance_df) > 0) {
      cat("Generating model performance summary plot...\n")
      png(
        file.path(output_dir_train, perf_report_filename),
        width = 1200, height = 1000, res = 100
      )
      par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))
      
      hist(deviance_df$Dev.Exp, main = "Distribution of Deviance Explained", xlab = "Deviance Explained", col = "slateblue", border = "white")
      abline(v = mean(deviance_df$Dev.Exp, na.rm = TRUE), col = "red", lwd = 2)
      
      hist(deviance_df$RMSE, main = "Distribution of Final Model RMSE", xlab = "RMSE", col = "tomato", border = "white")
      abline(v = mean(deviance_df$RMSE, na.rm = TRUE), col = "red", lwd = 2)
      
      hist(deviance_df$R2, main = "Distribution of Final Model R-squared", xlab = "R-squared", col = "gold", border = "white")
      abline(v = mean(deviance_df$R2, na.rm = TRUE), col = "red", lwd = 2)
      
      boxplot(R2 ~ year_pair, data = deviance_df, main = "R-squared Performance by Year Pair", xlab = "Year Pair", ylab = "R-squared", col = "lightblue", notch = F)
      
      dev.off()
      cat("Performance report saved to:", file.path(output_dir_train, perf_report_filename), "\n")
    } else {
      cat("No valid deviance data found to plot.\n")
    }
  } else {
    cat("No deviance result files found to process.\n")
  }
  
  par(mfrow = c(1, 1)) # Reset plotting layout
  cat("Process complete.\n")
}

# Define the tile IDs you want to test
# specific_tile_ids <- c("BH4RZ577_3", "BH4RZ577_2")

# Call the function with the correct arguments
create_xgboost_performance_report(
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles",
   tile_ids = "BH4S556X_3"
)


# ==============================================================================
#
#   Partial Dependence Plot (PDP) Reporting Functions
#
# ==============================================================================
# This script contains two functions to generate visual reports from the PDP
# data created during the XGBoost model training.
#
# 1. create_pdp_report_magnitude: Creates a multi-panel plot where all
#    predictors share the same y-axis scale, making it easy to compare the
#    magnitude of effect between them.
#
# 2. create_pdp_report_shape: Creates a multi-panel plot where each predictor
#    has its own y-axis scale, which is useful for examining the specific
#    shape and nuances of each predictor's relationship with the response.
#
# Both functions are updated to work with the latest data format and model outputs.
# --- Load All Necessary Libraries ---
library(dplyr)
library(stringr)
library(fst)
library(ggplot2)
library(tidyr)

#' Create and Save Partial Dependence Plots (Magnitude Comparison)
#'
#' Aggregates PDP results and generates a multi-panel plot where all predictors
#' share a common y-axis, allowing for comparison of effect magnitude.
#'
#' @param output_dir The base directory where the tile folders are located.
#' @param year_pairs A character vector of year pairs to process (e.g., "2004_2006").
#' @param tile_id (Optional) A character string for a single tile_id to process.
#'   If NULL (default), all tiles in the output directory will be processed.
#' @param plot_output_dir The directory where the final PNG plots will be saved.
#' @param n_bins The number of bins to use for summarizing the continuous predictors.
#' @param exclude_predictors (Optional) A character vector of predictor names to
#'   exclude from the plot.
#'
#' @return None. PNG plot files are saved to the `plot_output_dir`.
#'
create_pdp_report_magnitude <- function(output_dir, year_pairs, tile_id = NULL,
                                        plot_output_dir = output_dir, n_bins = 50,
                                        exclude_predictors = NULL) {
  
  # -------------------------------------------------------
  # 1. LOAD LIBRARIES and SETUP
  # -------------------------------------------------------
  cat("Starting PDP Magnitude report generation...\n")
  
  if (!is.null(tile_id)) {
    tile_folders <- file.path(output_dir, tile_id)
    cat("Processing single specified tile:", tile_id, "\n")
  } else {
    tile_folders <- list.dirs(output_dir, recursive = FALSE, full.names = TRUE)
    cat("Processing all", length(tile_folders), "tiles found in the directory.\n")
  }
  
  if (length(tile_folders) == 0 || !any(dir.exists(tile_folders))) {
    stop("No valid tile folders found to process.")
  }
  
  all_pdp_list <- list()
  
  # -------------------------------------------------------
  # 2. PROCESS DATA FOR EACH YEAR PAIR
  # -------------------------------------------------------
  for (pair in year_pairs) {
    cat("\nProcessing year pair:", pair, "...\n")
    
    pdp_files <- unlist(sapply(tile_folders, function(tile) {
      file.path(tile, paste0("pdp_data_long_", pair, ".fst"))
    }))
    
    pdp_files <- pdp_files[file.exists(pdp_files)]
    
    if (length(pdp_files) == 0) {
      cat("  No PDP data files found for this year pair. Skipping.\n")
      next
    }
    
    overall_pdp_data <- bind_rows(lapply(pdp_files, read_fst))
    
    if (nrow(overall_pdp_data) == 0) {
      cat("  No available PDP data for", pair, "- Skipping this year pair.\n")
      next
    }
    
    # --- Binning logic for smooth aggregation ---
    pdp_summary <- overall_pdp_data %>%
      filter(is.finite(Env_Value) & is.finite(PDP_Value)) %>%
      group_by(Predictor) %>%
      mutate(Env_Bin = cut(Env_Value, breaks = n_bins, include.lowest = TRUE)) %>%
      group_by(Predictor, Env_Bin) %>%
      summarise(
        Env_Value_Mid = mean(Env_Value, na.rm = TRUE),
        PDP_Mean = mean(PDP_Value, na.rm = TRUE),
        PDP_Min  = min(PDP_Value, na.rm = TRUE),
        PDP_Max  = max(PDP_Value, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      filter(!is.na(Env_Bin))
    
    all_pdp_list[[pair]] <- pdp_summary
  }
  
  # -------------------------------------------------------
  # 3. CREATE AND SAVE PLOTS
  # -------------------------------------------------------
  for (pair in names(all_pdp_list)) {
    pdp_to_plot <- all_pdp_list[[pair]]
    
    if (!is.null(exclude_predictors)) {
      patterns_to_exclude <- paste0("\\b", exclude_predictors, "\\b", collapse = "|")
      pdp_to_plot <- pdp_to_plot %>%
        filter(!grepl(patterns_to_exclude, Predictor))
      cat("  Excluding predictors matching:", paste(exclude_predictors, collapse=", "), "from the plot.\n")
    }
    
    if (nrow(pdp_to_plot) == 0) {
      cat("  Skipping plot for", pair, "as no valid data remains after exclusion.\n")
      next
    }
    
    cat("  Generating magnitude plot for", pair, "...\n")
    
    y_range <- range(c(pdp_to_plot$PDP_Min, pdp_to_plot$PDP_Max), na.rm = TRUE)
    
    plot_pdp <- ggplot(pdp_to_plot, aes(x = Env_Value_Mid, y = PDP_Mean)) +
      geom_ribbon(aes(ymin = PDP_Min, ymax = PDP_Max), fill = "grey70", alpha = 0.8) +
      geom_line(color = "black", linewidth = 0.8) +
      facet_wrap(~ Predictor, scales = "free_x", ncol = 3) +
      scale_y_continuous(breaks = seq(floor(y_range[1] * 2) / 2,
                                      ceiling(y_range[2] * 2) / 2,
                                      by = 1)) +
      labs(
        x = "Model Predictor Value",
        y = "Mean Predicted Bathymetry (m)" # UPDATED LABEL
      ) +
      theme_minimal(base_size = 14) +
      theme(
        strip.background = element_rect(fill = "lightgray", color = "grey"),
        strip.text = element_text(face = "bold", size = 10),
        axis.text.x = element_text(angle = 45, hjust = 1)
      )
    
    output_filename <- file.path(plot_output_dir, paste0("Overall_PDP_Magnitude_Report_", pair, ".png"))
    ggsave(
      filename = output_filename,
      plot = plot_pdp,
      width = 12, height = 8, dpi = 150,
      units = "in"
    )
    cat("  Plot saved to:", output_filename, "\n")
  }
  
  cat("\nProcess complete.\n")
}


#' Create and Save Partial Dependence Plots (Shape Comparison)
#'
#' Aggregates PDP results and generates a multi-panel plot where each predictor
#' has its own independent y-axis, ideal for viewing the shape of the relationship.
#'
#' @param output_dir The base directory where the tile folders are located.
#' @param year_pairs A character vector of year pairs to process.
#' @param tile_id (Optional) A character string for a single tile_id to process.
#' @param plot_output_dir The directory where the final PNG plots will be saved.
#' @param n_bins The number of bins to use for summarizing.
#' @param exclude_predictors (Optional) A character vector of predictor names to exclude.
#'
#' @return None. PNG plot files are saved to the `plot_output_dir`.
#'
create_pdp_report_shape <- function(output_dir, year_pairs, tile_id = NULL,
                                    plot_output_dir = output_dir, n_bins = 50,
                                    exclude_predictors = NULL) {
  
  # -------------------------------------------------------
  # 1. LOAD LIBRARIES and SETUP
  # -------------------------------------------------------
  cat("Starting PDP Shape report generation...\n")
  library(dplyr)
  library(stringr)
  library(fst)
  library(ggplot2)
  library(tidyr)
  
  if (!is.null(tile_id)) {
    tile_folders <- file.path(output_dir, tile_id)
    cat("Processing single specified tile:", tile_id, "\n")
  } else {
    tile_folders <- list.dirs(output_dir, recursive = FALSE, full.names = TRUE)
    cat("Processing all", length(tile_folders), "tiles found in the directory.\n")
  }
  
  if (length(tile_folders) == 0 || !any(dir.exists(tile_folders))) {
    stop("No valid tile folders found to process.")
  }
  
  all_pdp_list <- list()
  
  # -------------------------------------------------------
  # 2. PROCESS DATA FOR EACH YEAR PAIR
  # -------------------------------------------------------
  for (pair in year_pairs) {
    cat("\nProcessing year pair:", pair, "...\n")
    
    # CORRECTED: The file name is "pdp_data_long_...".
    pdp_files <- unlist(sapply(tile_folders, function(tile) {
      file.path(tile, paste0("pdp_data_long_", pair, ".fst"))
    }))
    
    pdp_files <- pdp_files[file.exists(pdp_files)]
    
    if (length(pdp_files) == 0) {
      cat("  No PDP data files found for this year pair. Skipping.\n")
      next
    }
    
    overall_pdp_data <- bind_rows(lapply(pdp_files, read_fst))
    
    if (nrow(overall_pdp_data) == 0) {
      cat("  No available PDP data for", pair, "- Skipping this year pair.\n")
      next
    }
    
    # --- Binning logic for smooth aggregation ---
    pdp_summary <- overall_pdp_data %>%
      filter(is.finite(Env_Value) & is.finite(PDP_Value)) %>%
      group_by(Predictor) %>%
      mutate(Env_Bin = cut(Env_Value, breaks = n_bins, include.lowest = TRUE)) %>%
      group_by(Predictor, Env_Bin) %>%
      summarise(
        Env_Value_Mid = mean(Env_Value, na.rm = TRUE),
        PDP_Mean = mean(PDP_Value, na.rm = TRUE),
        PDP_Min  = min(PDP_Value, na.rm = TRUE),
        PDP_Max  = max(PDP_Value, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      filter(!is.na(Env_Bin))
    
    all_pdp_list[[pair]] <- pdp_summary
  }
  
  # -------------------------------------------------------
  # 3. CREATE AND SAVE PLOTS
  # -------------------------------------------------------
  for (pair in names(all_pdp_list)) {
    pdp_to_plot <- all_pdp_list[[pair]]
    
    if (!is.null(exclude_predictors)) {
      patterns_to_exclude <- paste0("\\b", exclude_predictors, "\\b", collapse = "|")
      pdp_to_plot <- pdp_to_plot %>%
        filter(!grepl(patterns_to_exclude, Predictor))
      cat("  Excluding predictors matching:", paste(exclude_predictors, collapse=", "), "from the plot.\n")
    }
    
    if (nrow(pdp_to_plot) == 0) {
      cat("  Skipping plot for", pair, "as no valid data remains after exclusion.\n")
      next
    }
    
    cat("  Generating shape plot for", pair, "...\n")
    
    plot_pdp <- ggplot(pdp_to_plot, aes(x = Env_Value_Mid, y = PDP_Mean)) +
      geom_ribbon(aes(ymin = PDP_Min, ymax = PDP_Max), fill = "grey70", alpha = 0.8) +
      geom_line(color = "black", linewidth = 0.8) +
      facet_wrap(~ Predictor, scales = "free", ncol = 3) +
      labs(
        x = "Model Predictor Value",
        y = "Mean Predicted Bathymetry (m)" # UPDATED LABEL
      ) +
      theme_minimal(base_size = 14) +
      theme(
        strip.background = element_rect(fill = "lightgray", color = "grey"),
        strip.text = element_text(face = "bold", size = 10),
        axis.text.x = element_text(angle = 45, hjust = 1)
      )
    
    output_filename <- file.path(plot_output_dir, paste0("Overall_PDP_Report_Shape_", pair, ".png"))
    ggsave(
      filename = output_filename,
      plot = plot_pdp,
      width = 12, height = 8, dpi = 150,
      units = "in"
    )
    cat("  Plot saved to:", output_filename, "\n")
  }
  
  cat("\nProcess complete.\n")
}

create_pdp_report_shape (output_dir = output_dir,
                         tile_id = "BH4S556X_3",
                         year_pairs = years)

create_pdp_report_magnitude (output_dir = output_dir, # magnitude relative to bathymetry 
                             tile_id = "BH4S556X_3",
                             year_pairs = years)




# ==============================================================================
#
#   XGBoost Prediction Function Set (Refactored for Long-Format Data) V1
#
# ==============================================================================
#
# This script contains a complete, refactored set of functions to generate
# predictions from the trained XGBoost models. It is designed to:
#   1. Use a direct model if available for a tile, or find the nearest reference model.
#   2. Load and process long-format data for both prediction and training tiles.
#   3. Intelligently handle missing 'delta_' predictors in static prediction data.
#   4. Implement a clear, three-pronged hybrid adjustment workflow.
#   5. Save final predictions and all component predictions as rasters.
#   6. Run robustly in a parallel environment.
#   7. Automatically generate comprehensive diagnostic plots and logs.

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis)
library(gridExtra)
library(ggplot2)
library(FNN)
library(foreach)
library(doParallel)
library(Metrics)
library(cmocean)
library(stringr)

# ==============================================================================
#   MAIN WORKFLOW FUNCTIONS
# ==============================================================================

#' --- 1. Main Orchestration Function ---
process_tile <- function(tile_id, year_pair, training_dir, prediction_dir, output_dir, all_tile_maps, master_predictor_list_path) {
  # --- a. Setup Logging ---
  pred_log_dir <- file.path(output_dir, tile_id, "prediction_logs")
  if (!dir.exists(pred_log_dir)) dir.create(pred_log_dir, recursive = TRUE)
  pred_log_file <- file.path(pred_log_dir, paste0("pred_log_", year_pair, ".txt"))
  
  sink(pred_log_file, append = FALSE, split = TRUE)
  on.exit({ sink() })
  
  cat("\n--- Starting Tile:", tile_id, "| Year Pair:", year_pair, "---\n")
  
  tryCatch({
    # --- b. Determine Model Source Tile ---
    tile_map_for_year <- all_tile_maps[[year_pair]]
    if (is.null(tile_map_for_year)) stop("No valid models found for year pair: ", year_pair)
    
    model_tile_id <- if (tile_id %in% names(tile_map_for_year)) tile_map_for_year[[tile_id]] else tile_id
    processing_mode <- if (model_tile_id == tile_id) "direct_model" else "reference_model"
    
    cat("  - INFO: Processing mode detected:", processing_mode, "\n")
    if(processing_mode == "reference_model") cat("  - INFO: Using reference model from tile:", model_tile_id, "\n")
    
    # --- c. Generate Initial Predictions ---
    cat("Step 1: Generating initial XGBoost predictions...\n")
    initial_predictions_dt <- predict_elevation_change(tile_id, model_tile_id, year_pair, training_dir, prediction_dir)
    if (is.null(initial_predictions_dt)) stop("Initial prediction failed.")
    
    # --- d. Load Data for Hybrid Adjustments ---
    cat("Step 2: Loading data for hybrid adjustments from source:", model_tile_id, "\n")
    pdp_file <- file.path(training_dir, model_tile_id, paste0("pdp_data_long_", year_pair, ".fst"))
    
    train_file_prefix <- sub("_[0-9]+$", "", model_tile_id)
    training_data_path <- file.path(training_dir, model_tile_id, paste0(train_file_prefix, "_", year_pair, "_long.fst"))
    
    if (!file.exists(pdp_file)) stop("Missing required PDP file: ", pdp_file)
    if (!file.exists(training_data_path)) stop("Missing required Training Data file: ", training_data_path)
    
    pdp_data <- read_fst(pdp_file, as.data.table = TRUE)
    training_data <- read_fst(training_data_path, as.data.table = TRUE)
    
    aligned_training_data <- align_predictors(training_data, year_pair)
    
    # --- e. Apply Hybrid Adjustments ---
    cat("Step 3: Applying hybrid adjustments (Bootstrap, PDP, KNN)...\n")
    boot_enriched <- apply_bootstrap_adjustment(initial_predictions_dt, model_tile_id, year_pair, training_dir)
    pdp_enriched <- match_pdp_conditions(boot_enriched, pdp_data)
    knn_enriched <- apply_trend_adjustments_hybrid(pdp_enriched, aligned_training_data)
    
    # --- f. Compare and Finalize ---
    cat("Step 4: Comparing prediction methods and combining results...\n")
    comparison_results <- compare_prediction_methods(knn_enriched, tile_id, year_pair, processing_mode = processing_mode)
    
    cat("\n--- SUCCESS (Data Processing): Completed Tile:", tile_id, "| Year Pair:", year_pair, "---\n")
    
    return(list(
      data = comparison_results$data,
      comparison_log = comparison_results$comparison_log,
      performance_log = comparison_results$performance_log,
      processing_mode = processing_mode,
      tile_id = tile_id,
      year_pair = year_pair,
      success = TRUE
    ))
    
  }, error = function(e) {
    cat("\n--- FATAL ERROR in process_tile ---\n")
    cat("  - Tile:", tile_id, "| Pair:", year_pair, "\n")
    cat("  - Error Message:", conditionMessage(e), "\n")
    cat("  - Traceback:\n", paste(capture.output(traceback()), collapse="\n"), "\n")
    return(list(success = FALSE, tile_id = tile_id, year_pair = year_pair, error = conditionMessage(e)))
  })
}

# ==============================================================================
#   HELPER FUNCTIONS (DATA PROCESSING)
# ==============================================================================

#' --- 2. Core Prediction Function ---
predict_elevation_change <- function(tile_id, model_tile_id, year_pair, training_dir, prediction_dir) {
  model_path <- file.path(training_dir, model_tile_id, paste0("xgb_model_", year_pair, ".rds"))
  
  pred_file_prefix <- sub("_[0-9]+$", "", tile_id)
  # CORRECTED: Added "_prediction" to the filename string
  prediction_data_path <- file.path(prediction_dir, tile_id, paste0(pred_file_prefix, "_", year_pair, "_prediction_long.fst"))
  
  if (!file.exists(model_path)) stop("Missing required model file: ", model_path)
  if (!file.exists(prediction_data_path)) stop("Missing required prediction data file: ", prediction_data_path)
  
  # --- MODIFICATION START ---
  # Load the model FIRST to get the true list of predictors it was trained on.
  xgb_model <- readRDS(model_path)
  model_feature_names <- xgb_model$feature_names
  cat("  - INFO: Model was trained with", length(model_feature_names), "features.\n")
  
  prediction_data <- read_fst(prediction_data_path, as.data.table = TRUE)
  if (!"tile_id" %in% names(prediction_data)) prediction_data[, tile_id := tile_id]
  
  aligned_pred_data <- align_predictors(prediction_data, year_pair)
  
  # Find which predictors the model expects but are missing from the prediction data (these will be the deltas)
  missing_features <- setdiff(model_feature_names, names(aligned_pred_data))
  
  if (length(missing_features) > 0) {
    cat("  - INFO: Adding", length(missing_features), "missing predictor columns (deltas) and filling with NA.\n")
    # Add the missing columns and fill them with NA
    for (col in missing_features) {
      aligned_pred_data[, (col) := NA_real_]
    }
  }
  
  # Create the matrix using the full set of features the model expects.
  # The order is guaranteed to be correct by using the model's internal feature list.
  pred_matrix <- as.matrix(aligned_pred_data[, ..model_feature_names])
  # --- MODIFICATION END ---
  
  aligned_pred_data[, XGB_predicted_bathy_t1 := predict(xgb_model, newdata = pred_matrix)]
  
  message("  - INFO: Generated initial predictions for ", nrow(aligned_pred_data), " points.")
  return(aligned_pred_data)
}

#' --- 3. Align Predictors ---
align_predictors <- function(data_to_align, year_pair) {
  setDT(data_to_align)
  
  if ("bathy_t" %in% names(data_to_align)) data_to_align[, starting_bathy := bathy_t]
  if ("slope_t" %in% names(data_to_align)) data_to_align[, starting_slope := slope_t]
  if ("rugosity_t" %in% names(data_to_align)) data_to_align[, starting_rugosity := rugosity_t]
  
 
  
  return(data_to_align)
}


#' --- 4. Apply Bootstrap Adjustment ---
apply_bootstrap_adjustment <- function(prediction_data, model_tile_id, year_pair, training_dir) {
  boots_path <- file.path(training_dir, model_tile_id, paste0("bootstraps_", year_pair, ".fst"))
  if (!file.exists(boots_path)) {
    warning("Bootstrap file not found for tile ", model_tile_id, ". Skipping.")
    prediction_data[, `:=`(MeanBoots_predicted_change = NA_real_, MeanBoots_SD = NA_real_)]
    return(prediction_data)
  }
  
  boots_data <- read_fst(boots_path, as.data.table = TRUE)
  boots_to_merge <- boots_data[, .(FID, Mean_Prediction, Uncertainty_SD)]
  
  prediction_data[, temp_order := .I]
  merged_data <- merge(prediction_data, boots_to_merge, by = "FID", all.x = TRUE)
  setorderv(merged_data, "temp_order")[, temp_order := NULL]
  
  merged_data[, MeanBoots_predicted_change := Mean_Prediction - bathy_t]
  setnames(merged_data, "Uncertainty_SD", "MeanBoots_SD", skip_absent=TRUE)
  
  message("  - INFO: Applied Bootstrap Mean adjustment.")
  return(merged_data)
}

#' --- 5. Match PDP Conditions ---
match_pdp_conditions <- function(prediction_data, pdp_data) {
  setDT(prediction_data); setDT(pdp_data)
  env_ranges <- pdp_data[, .(range_width = max(Env_Value, na.rm = TRUE) - min(Env_Value, na.rm = TRUE)), by = Predictor]
  
  pdp_match_vars <- list(
    list(pred_pattern = "bathy_t", start_col = "starting_bathy", weight = 0.4),
    list(pred_pattern = "slope_t", start_col = "starting_slope", weight = 0.2),
    list(pred_pattern = "rugosity_t", start_col = "starting_rugosity", weight = 0.1)
  )
  
  prediction_data[, pdp_adjusted_change := {
    current_point <- .SD
    is_match <- rep(FALSE, nrow(pdp_data))
    
    for (var in pdp_match_vars) {
      predictor_name <- var$pred_pattern
      range_info <- env_ranges[Predictor == predictor_name]
      
      if (nrow(range_info) > 0 && var$start_col %in% names(current_point)) {
        current_val <- current_point[[var$start_col]]
        if (!is.na(current_val)) {
          half_window <- var$weight * range_info$range_width
          is_match <- is_match | (pdp_data$Predictor == predictor_name & 
                                    pdp_data$Env_Value >= (current_val - half_window) & 
                                    pdp_data$Env_Value <= (current_val + half_window))
        }
      }
    }
    
    matches <- pdp_data[is_match]
    if (nrow(matches) > 0) {
      mean_pdp_prediction <- mean(matches$PDP_Value, na.rm = TRUE)
      change <- mean_pdp_prediction - current_point$bathy_t
      change
    } else {
      NA_real_
    }
  }, by = 1:nrow(prediction_data)]
  
  message("  - INFO: PDP Adjusted Change matched on: ", sum(!is.na(prediction_data$pdp_adjusted_change)), " points.")
  return(prediction_data)
}

#' --- 6. Apply KNN Trend Adjustments ---
apply_trend_adjustments_hybrid <- function(prediction_data, training_data, k = 15) {
  setDT(prediction_data); setDT(training_data)
  
  env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity")
  
  pred_clean <- prediction_data[complete.cases(prediction_data[, ..env_vars])]
  train_clean <- training_data[complete.cases(training_data[, ..env_vars])]
  
  if (nrow(pred_clean) == 0) { message("  - WARN: No complete cases for KNN. Skipping."); return(prediction_data) }
  if (nrow(train_clean) < k) { stop("Not enough training data for KNN.") }
  
  response_var <- "b_change"
  if (!response_var %in% names(train_clean)) stop("Missing 'b_change' (from delta_bathy) in training data for KNN.")
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]]
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  weights <- 1 / (knn_result$nn.dist + 1e-6)
  pred_avg <- rowSums(matrix(train_resp[knn_result$nn.index], nrow = nrow(knn_result$nn.index)) * weights, na.rm = TRUE) / rowSums(weights, na.rm = TRUE)
  
  pred_clean[, KNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), KNN_predicted_change := i.KNN_predicted_change]
  
  message("  - INFO: Assigned KNN predictions to ", sum(!is.na(prediction_data$KNN_predicted_change)), " rows.")
  return(prediction_data)
}

#' --- 7. Compare Methods and Create Hybrid Prediction ---
compare_prediction_methods <- function(prediction_data, tile_id, year_pair, processing_mode) {
  message("  - INFO: Combining results into hybrid prediction...")
  
  prediction_data[, XGB_predicted_change := XGB_predicted_bathy_t1 - bathy_t]
  
  prediction_data[, hybrid_change := rowMeans(.SD, na.rm = TRUE), 
                  .SDcols = c("MeanBoots_predicted_change", "pdp_adjusted_change", "KNN_predicted_change")]
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  pdp_knn_df <- prediction_data[!is.na(pdp_adjusted_change) & !is.na(KNN_predicted_change)]
  comparison_log <- data.table(Tile_ID = tile_id, Year_Pair = year_pair, Total_Points = nrow(prediction_data),
                               PDP_KNN_Cor = if (nrow(pdp_knn_df) > 2) cor(pdp_knn_df$pdp_adjusted_change, pdp_knn_df$KNN_predicted_change) else NA)
  
  performance_log <- NULL
  if (processing_mode == 'direct_model' && "b_change" %in% names(prediction_data)) {
    performance_log <- data.table(
      Tile_ID = tile_id, Year_Pair = year_pair,
      RMSE_XGB = rmse(prediction_data$b_change, prediction_data$XGB_predicted_change),
      RMSE_Hybrid = rmse(prediction_data$b_change, prediction_data$hybrid_change)
    )
  }
  
  return(list(data = prediction_data, comparison_log = comparison_log, performance_log = performance_log))
}

# ==============================================================================
#   HELPER FUNCTIONS (FILE I/O & PLOTTING)
# ==============================================================================
mae <- function(actual, predicted) { mean(abs(actual - predicted), na.rm = TRUE) }
rmse <- function(actual, predicted) { sqrt(mean((actual - predicted)^2, na.rm = TRUE)) }

save_final_predictions <- function(prediction_data, output_dir, tile_id, year_pair) {
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  pred_depth_col <- paste0("pred_", end_year, "_depth")
  if (!"starting_bathy" %in% names(prediction_data) || !"hybrid_change" %in% names(prediction_data)) { stop("Cannot save final predictions.") }
  
  prediction_data[, (pred_depth_col) := starting_bathy + hybrid_change]
  
  out_file_dir <- file.path(output_dir, tile_id); if (!dir.exists(out_file_dir)) dir.create(out_file_dir, recursive = TRUE)
  out_file <- file.path(out_file_dir, paste0(tile_id, "_prediction_final_", year_pair, ".fst"))
  
  write_fst(prediction_data, out_file)
  message("  - INFO: Final FST prediction file saved to: ", out_file)
  return(prediction_data)
}

save_component_rasters <- function(prediction_data, output_dir, tile_id, year_pair, crs_obj) {
  out_raster_dir <- file.path(output_dir, tile_id)
  if (!dir.exists(out_raster_dir)) dir.create(out_raster_dir, recursive = TRUE)
  
  save_raster <- function(data, col_name, file_suffix) {
    if (col_name %in% names(data)) {
      raster_df <- data[!is.na(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(raster_df) > 0) {
        r <- raster::rasterFromXYZ(raster_df, crs = crs_obj)
        out_path <- file.path(out_raster_dir, paste0(tile_id, "_", file_suffix, "_", year_pair, ".tif"))
        raster::writeRaster(r, out_path, format = "GTiff", overwrite = TRUE)
      }
    }
  }
  
  save_raster(prediction_data, "hybrid_change", "Hybrid_predicted_change")
  save_raster(prediction_data, "pdp_adjusted_change", "PDP_predicted_change")
  save_raster(prediction_data, "KNN_predicted_change", "KNN_predicted_change")
  save_raster(prediction_data, "XGB_predicted_change", "XGB_predicted_change")
  save_raster(prediction_data, "MeanBoots_predicted_change", "MeanBoots_predicted_change")
}

build_all_tile_maps <- function(training_dir, prediction_dir, year_pairs) {
  message("  - INFO: Building year-specific geographic maps...")
  
  training_footprint_gpkg <- file.path(training_dir, "intersecting_sub_grids_UTM.gpkg")
  prediction_footprint_gpkg <- file.path(prediction_dir, "intersecting_sub_grids_UTM.gpkg")
  
  if(!all(file.exists(training_footprint_gpkg, prediction_footprint_gpkg))) {
    stop("Footprint GPKG files not found at:\n", 
         training_footprint_gpkg, "\n", prediction_footprint_gpkg,
         "\nCannot build tile maps. Please run the pre-processing step to generate these files.")
  }
  
  train_sf_full <- sf::st_read(training_footprint_gpkg, quiet = TRUE)
  pred_sf <- sf::st_read(prediction_footprint_gpkg, quiet = TRUE)
  
  id_col <- "tile_id"
  pred_centroids <- sf::st_centroid(pred_sf)
  
  all_maps <- list()
  for (yp in year_pairs) {
    model_file_name <- paste0("xgb_model_", yp, ".rds")
    valid_training_tiles <- list.dirs(training_dir, full.names = FALSE, recursive = FALSE)
    valid_source_tiles <- character()
    for (tile in valid_training_tiles) {
      if (file.exists(file.path(training_dir, tile, model_file_name))) {
        valid_source_tiles <- c(valid_source_tiles, tile)
      }
    }
    
    if (length(valid_source_tiles) == 0) {
      all_maps[[yp]] <- NULL
      next
    }
    
    train_sf_year_specific <- train_sf_full[train_sf_full[[id_col]] %in% valid_source_tiles, ]
    train_centroids_year_specific <- sf::st_centroid(train_sf_year_specific)
    
    nearest_indices <- sf::st_nearest_feature(pred_centroids, train_centroids_year_specific)
    
    tile_map <- train_sf_year_specific[[id_col]][nearest_indices]
    names(tile_map) <- pred_sf[[id_col]]
    
    all_maps[[as.character(yp)]] <- tile_map
  }
  return(all_maps)
}


run_cross_validation <- function(processed_data, tile_id, year_pair, training_dir) {
  message("  - INFO: Running cross-validation for tile: ", tile_id)
  
  train_file_prefix <- sub("_[0-9]+$", "", tile_id)
  training_data_path <- file.path(training_dir, tile_id, paste0(train_file_prefix, "_", year_pair, "_long.fst"))
  
  if(!file.exists(training_data_path)) {
    message("  - WARN: No training data file for cross-validation. Skipping.")
    return(NULL)
  }
  
  truth_data <- read_fst(training_data_path, as.data.table = TRUE)
  
  validation_data <- merge(processed_data, truth_data, by = "FID", suffixes = c("_pred", "_actual"))
  
  actual_depth_col <- "bathy_t1"
  pred_depth_col <- paste0("pred_", as.numeric(strsplit(year_pair, "_")[[1]][2]), "_depth")
  actual_change_col <- "delta_bathy" 
  
  required_cols <- c(actual_depth_col, pred_depth_col, actual_change_col, "hybrid_change")
  if (!all(required_cols %in% names(validation_data))) {
    message("  - WARN: Missing columns for cross-validation. Skipping.")
    return(NULL)
  }
  
  valid_comp_data <- validation_data[!is.na(get(actual_depth_col)) & !is.na(get(pred_depth_col))]
  if(nrow(valid_comp_data) < 2) return(NULL)
  
  cv_log <- data.table(Tile_ID = tile_id, Year_Pair = year_pair,
                       Depth_MAE = mae(valid_comp_data[[actual_depth_col]], valid_comp_data[[pred_depth_col]]),
                       Change_MAE = mae(valid_comp_data[[actual_change_col]], valid_comp_data$hybrid_change))
  return(cv_log)
}

# ==============================================================================
#   Parallel Prediction Test Wrapper
# ==============================================================================
run_prediction_test <- function(prediction_tiles, year_pairs, crs_obj, training_dir, prediction_dir, output_dir, master_predictor_list_path) {
  
  message("\nStarting parallel prediction test run...")
  
  all_tile_maps <- build_all_tile_maps(training_dir, prediction_dir, year_pairs)
  
  num_cores <- detectCores() - 1; if (num_cores < 1) num_cores <- 1
  cl <- makeCluster(num_cores); registerDoParallel(cl); on.exit({ stopCluster(cl) })
  
  task_grid <- expand.grid(tile_id = prediction_tiles, year_pair = year_pairs, stringsAsFactors = FALSE)
  
  message("\n--- Stage 1: Processing data in parallel... ---")
  
  functions_to_export <- c("process_tile", "predict_elevation_change", "align_predictors", "apply_bootstrap_adjustment", 
                           "match_pdp_conditions", "apply_trend_adjustments_hybrid", "compare_prediction_methods", 
                           "save_final_predictions", "save_component_rasters", "run_cross_validation", "mae", "rmse")
  
  results_list <- foreach(
    i = 1:nrow(task_grid),
    .export = functions_to_export,
    .packages = c("data.table", "dplyr", "fst", "sf", "xgboost", "FNN", "raster", "stringr", "Metrics", "cmocean")
  ) %dopar% {
    current_task <- task_grid[i, ]
    process_tile(
      tile_id        = current_task$tile_id,
      year_pair      = current_task$year_pair,
      training_dir   = training_dir,
      prediction_dir = prediction_dir,
      output_dir     = output_dir,
      all_tile_maps  = all_tile_maps,
      master_predictor_list_path = master_predictor_list_path
    )
  }
  
  message("\n--- Stage 2: Saving results and plots sequentially... ---")
  
  successful_results <- Filter(function(res) is.list(res) && isTRUE(res$success), results_list)
  
  all_cv_logs <- list()
  for (result in successful_results) {
    cat("\n--- Saving outputs for Tile:", result$tile_id, "| Year Pair:", result$year_pair, "---\n")
    processed_data <- save_final_predictions(result$data, output_dir, result$tile_id, result$year_pair)
    if (!is.null(crs_obj)) {
      save_component_rasters(processed_data, output_dir, result$tile_id, result$year_pair, crs_obj)
      
      if (result$processing_mode == 'direct_model') {
        cv_log <- run_cross_validation(processed_data, result$tile_id, result$year_pair, training_dir)
        if(!is.null(cv_log)) all_cv_logs[[length(all_cv_logs) + 1]] <- cv_log
      }
    }
  }
  
  if(length(all_cv_logs) > 0) {
    fwrite(rbindlist(all_cv_logs, fill = TRUE), file.path(output_dir, "cross_validation_log.csv"))
    message("  - INFO: Cross-validation log saved.")
  }
  
  message("\n✅ Parallel prediction test run complete.")
}


# ==============================================================================
#
#   XGBoost Prediction Function Set (Refactored for Long-Format Data - V4)
#
# ==============================================================================
#
# This script contains a complete, refactored set of functions to generate
# predictions from the trained XGBoost models. It is designed to:
#   1. Use a direct model if available for a tile, or find the nearest reference model.
#   2. Load and process long-format data for both prediction and training tiles.
#   3. Intelligently handle missing 'delta_' predictors in static prediction data.
#   4. Implement a clear, three-pronged hybrid adjustment workflow, now using
#      full bootstrap information (via KNN) for the Bootstrap Adjustment step.
#   5. Save final predictions and all component predictions as rasters.
#   6. Run robustly in a parallel environment.
#   7. Automatically generate comprehensive diagnostic plots and logs.

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis)
library(gridExtra)
library(ggplot2)
library(FNN)
library(foreach)
library(doParallel)
library(Metrics)
library(cmocean)
library(stringr)

# ==============================================================================
#   MAIN WORKFLOW FUNCTIONS
# ==============================================================================

#' --- 1. Main Orchestration Function ---
process_tile <- function(tile_id, year_pair, training_dir, prediction_dir, output_dir, all_tile_maps, master_predictor_list_path) {
  # --- a. Setup Logging ---
  pred_log_dir <- file.path(output_dir, tile_id, "prediction_logs")
  if (!dir.exists(pred_log_dir)) dir.create(pred_log_dir, recursive = TRUE)
  pred_log_file <- file.path(pred_log_dir, paste0("pred_log_", year_pair, ".txt"))
  
  sink(pred_log_file, append = FALSE, split = TRUE)
  on.exit({ sink() })
  
  cat("\n--- Starting Tile:", tile_id, "| Year Pair:", year_pair, "---\n")
  
  tryCatch({
    # --- b. Determine Model Source Tile ---
    tile_map_for_year <- all_tile_maps[[year_pair]]
    if (is.null(tile_map_for_year)) stop("No valid models found for year pair: ", year_pair)
    
    model_tile_id <- if (tile_id %in% names(tile_map_for_year)) tile_map_for_year[[tile_id]] else tile_id
    processing_mode <- if (model_tile_id == tile_id) "direct_model" else "reference_model"
    
    cat("  - INFO: Processing mode detected:", processing_mode, "\n")
    if(processing_mode == "reference_model") cat("  - INFO: Using reference model from tile:", model_tile_id, "\n")
    
    # --- c. Generate Initial Predictions ---
    cat("Step 1: Generating initial XGBoost predictions...\n")
    initial_predictions_dt <- predict_elevation_change(tile_id, model_tile_id, year_pair, training_dir, prediction_dir, master_predictor_list_path)
    if (is.null(initial_predictions_dt)) stop("Initial prediction failed.")
    
    # --- d. Load Data for Hybrid Adjustments ---
    cat("Step 2: Loading data for hybrid adjustments from source:", model_tile_id, "\n")
    pdp_file <- file.path(training_dir, model_tile_id, paste0("pdp_data_long_", year_pair, ".fst"))
    
    train_file_prefix <- sub("_[0-9]+$", "", model_tile_id)
    training_data_path <- file.path(training_dir, model_tile_id, paste0(train_file_prefix, "_", year_pair, "_long.fst"))
    
    if (!file.exists(pdp_file)) stop("Missing required PDP file: ", pdp_file)
    if (!file.exists(training_data_path)) stop("Missing required Training Data file: ", training_data_path)
    
    pdp_data <- read_fst(pdp_file, as.data.table = TRUE)
    training_data <- read_fst(training_data_path, as.data.table = TRUE)
    
    aligned_training_data <- align_predictors(training_data, year_pair)
    
    # --- e. Apply Hybrid Adjustments ---
    cat("Step 3: Applying hybrid adjustments (Bootstrap KNN, PDP, Training KNN)...\n")
    # **MODIFICATION**: Call the new apply_bootstrap_knn_adjustment
    boot_knn_enriched <- apply_bootstrap_knn_adjustment(initial_predictions_dt, model_tile_id, year_pair, training_dir)
    pdp_enriched <- match_pdp_conditions(boot_knn_enriched, pdp_data)
    # **MODIFICATION**: Use the renamed function apply_training_knn_adjustment
    train_knn_enriched <- apply_training_knn_adjustment(pdp_enriched, aligned_training_data)
    
    # --- f. Compare and Finalize ---
    cat("Step 4: Comparing prediction methods and combining results...\n")
    comparison_results <- compare_prediction_methods(train_knn_enriched, tile_id, year_pair, processing_mode = processing_mode)
    
    cat("\n--- SUCCESS (Data Processing): Completed Tile:", tile_id, "| Year Pair:", year_pair, "---\n")
    
    return(list(
      data = comparison_results$data,
      comparison_log = comparison_results$comparison_log,
      performance_log = comparison_results$performance_log,
      processing_mode = processing_mode,
      tile_id = tile_id,
      year_pair = year_pair,
      success = TRUE
    ))
    
  }, error = function(e) {
    cat("\n--- FATAL ERROR in process_tile ---\n")
    cat("  - Tile:", tile_id, "| Pair:", year_pair, "\n")
    cat("  - Error Message:", conditionMessage(e), "\n")
    cat("  - Traceback:\n", paste(capture.output(traceback()), collapse="\n"), "\n")
    return(list(success = FALSE, tile_id = tile_id, year_pair = year_pair, error = conditionMessage(e)))
  })
}

# ==============================================================================
#   HELPER FUNCTIONS (DATA PROCESSING)
# ==============================================================================

#' --- 2. Core Prediction Function ---
predict_elevation_change <- function(tile_id, model_tile_id, year_pair, training_dir, prediction_dir, master_predictor_list_path=NULL) {
  model_path <- file.path(training_dir, model_tile_id, paste0("xgb_model_", year_pair, ".rds"))
  
  pred_file_prefix <- sub("_[0-9]+$", "", tile_id)
  prediction_data_path <- file.path(prediction_dir, tile_id, paste0(pred_file_prefix, "_", year_pair, "_prediction_long.fst"))
  
  if (!file.exists(model_path)) stop("Missing required model file: ", model_path)
  if (!file.exists(prediction_data_path)) stop("Missing required prediction data file: ", prediction_data_path)
  
  xgb_model <- readRDS(model_path)
  model_feature_names <- xgb_model$feature_names
  cat("  - INFO: Model was trained with", length(model_feature_names), "features.\n")
  
  prediction_data <- read_fst(prediction_data_path, as.data.table = TRUE)
  if (!"tile_id" %in% names(prediction_data)) prediction_data[, tile_id := tile_id]
  
  aligned_pred_data <- align_predictors(prediction_data, year_pair)
  
  missing_features <- setdiff(model_feature_names, names(aligned_pred_data))
  
  if (length(missing_features) > 0) {
    cat("  - INFO: Adding", length(missing_features), "missing predictor columns (deltas) and filling with NA.\n")
    for (col in missing_features) {
      aligned_pred_data[, (col) := as.numeric(NA)]
    }
  }
  
  pred_matrix <- as.matrix(aligned_pred_data[, ..model_feature_names])
  
  aligned_pred_data[, XGB_predicted_bathy_t1 := predict(xgb_model, newdata = pred_matrix)]
  
  message("  - INFO: Generated initial predictions for ", nrow(aligned_pred_data), " points.")
  return(aligned_pred_data)
}

#' --- 3. Align Predictors ---
align_predictors <- function(data_to_align, year_pair) {
  setDT(data_to_align)
  
  # Create standardized 'starting_' columns for consistent use in downstream functions (PDP, KNN)
  if ("bathy_t" %in% names(data_to_align)) data_to_align[, starting_bathy := bathy_t]
  if ("slope_t" %in% names(data_to_align)) data_to_align[, starting_slope := slope_t]
  if ("rugosity_t" %in% names(data_to_align)) data_to_align[, starting_rugosity := rugosity_t]
  if ("bpi_broad_t" %in% names(data_to_align)) data_to_align[, starting_bpi_broad := bpi_broad_t]
  if ("bpi_fine_t" %in% names(data_to_align)) data_to_align[, starting_bpi_fine := bpi_fine_t]
  if ("terrain_classification_t" %in% names(data_to_align)) data_to_align[, starting_terrain_class := terrain_classification_t]
  if ("grain_size_layer" %in% names(data_to_align)) data_to_align[, starting_grain_size := grain_size_layer]
  if ("prim_sed_layer" %in% names(data_to_align)) data_to_align[, starting_sed_type := prim_sed_layer]
  
  # Add neighborhood stats as starting variables
  nbh_cols <- grep("_(mean3|sd3)_t$", names(data_to_align), value = TRUE)
  for(col in nbh_cols){
    new_col_name <- sub("_t$", "", col)
    new_col_name <- paste0("starting_", new_col_name)
    data_to_align[, (new_col_name) := get(col)]
  }
  
  # Add TSM (year-pair specific)
  tsm_var_name <- paste0("tsm_", year_pair)
  if (tsm_var_name %in% names(data_to_align)) data_to_align[, starting_tsm := get(tsm_var_name)]
  
  
  # Standardize the name for observed change, if it exists
  if ("delta_bathy" %in% names(data_to_align)) {
    setnames(data_to_align, old = "delta_bathy", new = "b_change")
  }
  
  return(data_to_align)
}


#' --- 4. NEW Bootstrap KNN Adjustment ---
apply_bootstrap_knn_adjustment <- function(prediction_data, model_tile_id, year_pair, training_dir, k = 15) {
  cat("  - INFO: Applying Bootstrap KNN adjustment...\n")
  
  boot_array_path <- file.path(training_dir, model_tile_id, paste0("bootstraps_array_", year_pair, ".rds"))
  boot_df_path <- file.path(training_dir, model_tile_id, paste0("bootstraps_summary_df_", year_pair, ".fst"))
  
  if (!file.exists(boot_array_path) || !file.exists(boot_df_path)) {
    warning("Bootstrap array or summary df file not found for tile ", model_tile_id, ". Skipping Bootstrap KNN adjustment.")
    prediction_data[, `:=`(MeanBoots_predicted_change = NA_real_, MeanBoots_SD = NA_real_)]
    return(prediction_data)
  }
  
  boot_array <- readRDS(boot_array_path)
  boot_df_train <- read_fst(boot_df_path, as.data.table = TRUE)
  
  setDT(prediction_data)
  
  # **MODIFICATION**: Expand the list of potential environmental variables for matching
  potential_env_vars_pred <- c("starting_bathy", "starting_slope", "starting_rugosity",
                               "starting_bpi_broad", "starting_bpi_fine",
                               "starting_terrain_class", "starting_grain_size", "starting_sed_type",
                               "starting_tsm",
                               grep("^starting_.*_(mean3|sd3)$", names(prediction_data), value = TRUE)) # Add neighborhood stats
  
  potential_env_vars_train <- c("bathy_t", "slope_t", "rugosity_t",
                                "bpi_broad_t", "bpi_fine_t",
                                "terrain_classification_t", "grain_size_layer", "prim_sed_layer",
                                paste0("tsm_", year_pair),
                                grep("_(mean3|sd3)_t$", names(boot_df_train), value = TRUE)) # Add neighborhood stats
  
  # Find common variables dynamically based on what's available in BOTH datasets
  common_base_vars <- intersect(gsub("^starting_", "", potential_env_vars_pred),
                                gsub("_t$", "", gsub(paste0("_", year_pair), "", potential_env_vars_train)))
  
  # Construct the final variable lists making sure they correspond
  env_vars_pred <- paste0("starting_", common_base_vars[common_base_vars %in% gsub("^starting_","", names(prediction_data))])
  env_vars_train <- sapply(common_base_vars, function(v) {
    if (v %in% c("grain_size", "sed_type")) paste0(v, "_layer") # Handle static names
    else if (v == "tsm") paste0(v, "_", year_pair) # Handle year-specific forcing
    else paste0(v, "_t") # Handle _t state variables
  })
  env_vars_train <- env_vars_train[env_vars_train %in% names(boot_df_train)] # Keep only those actually present
  
  # Ensure we use the corresponding predictor variable names
  env_vars_pred <- env_vars_pred[sapply(common_base_vars, function(v) {
    train_name <- if (v %in% c("grain_size", "sed_type")) paste0(v, "_layer") else if (v == "tsm") paste0(v, "_", year_pair) else paste0(v, "_t")
    train_name %in% env_vars_train
  })]
  
  
  if (length(env_vars_pred) == 0) {
    warning("No common environmental variables found for Bootstrap KNN. Skipping.")
    prediction_data[, `:=`(MeanBoots_predicted_change = NA_real_, MeanBoots_SD = NA_real_)]
    return(prediction_data)
  }
  
  cat("    - Matching using variables:", paste(env_vars_pred, collapse=", "), "\n")
  
  pred_clean <- prediction_data[complete.cases(prediction_data[, ..env_vars_pred])]
  train_clean <- boot_df_train[complete.cases(boot_df_train[, ..env_vars_train])]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases in prediction data for Bootstrap KNN. Skipping."); return(prediction_data) }
  if (nrow(train_clean) < k) { stop("Not enough complete cases in bootstrap training data for KNN.") }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars_pred])
  train_mat <- as.matrix(train_clean[, ..env_vars_train])
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  neighbor_indices <- knn_result$nn.index
  train_clean_indices <- which(complete.cases(boot_df_train[, ..env_vars_train]))
  
  adjusted_preds <- numeric(nrow(pred_clean))
  adjusted_sds <- numeric(nrow(pred_clean))
  n_boot_iters <- dim(boot_array)[3]
  
  for (i in 1:nrow(pred_clean)) {
    nn_idx_in_train_clean <- neighbor_indices[i, ]
    original_nn_indices <- train_clean_indices[nn_idx_in_train_clean]
    
    valid_original_indices <- original_nn_indices[original_nn_indices <= dim(boot_array)[1]]
    if(length(valid_original_indices) == 0) next
    
    all_neighbor_boot_preds <- as.vector(boot_array[valid_original_indices, 1, ])
    
    adjusted_preds[i] <- mean(all_neighbor_boot_preds, na.rm = TRUE)
    adjusted_sds[i] <- sd(all_neighbor_boot_preds, na.rm = TRUE)
    if(is.na(adjusted_sds[i])) adjusted_sds[i] <- 0
  }
  
  pred_clean[, `:=`(Mean_Prediction_KNN = adjusted_preds, MeanBoots_SD = adjusted_sds)]
  # Use starting_bathy available in pred_clean for change calculation
  pred_clean[, MeanBoots_predicted_change := Mean_Prediction_KNN - starting_bathy]
  
  # Merge back using FID
  prediction_data[pred_clean, on = .(FID),
                  `:=`(MeanBoots_predicted_change = i.MeanBoots_predicted_change,
                       MeanBoots_SD = i.MeanBoots_SD)]
  
  
  message("  - INFO: Applied Bootstrap KNN adjustment. ", sum(!is.na(prediction_data$MeanBoots_predicted_change)), " points received a value.")
  return(prediction_data)
}


#' --- 5. Match PDP Conditions ---
match_pdp_conditions <- function(prediction_data, pdp_data) {
  setDT(prediction_data); setDT(pdp_data)
  env_ranges <- pdp_data[, .(range_width = max(Env_Value, na.rm = TRUE) - min(Env_Value, na.rm = TRUE)), by = Predictor]
  
  # **MODIFICATION**: Expand potential variables for PDP matching
  pdp_match_vars_potential <- list(
    list(pred_pattern = "bathy_t", start_col = "starting_bathy", weight = 0.3),
    list(pred_pattern = "slope_t", start_col = "starting_slope", weight = 0.15),
    list(pred_pattern = "rugosity_t", start_col = "starting_rugosity", weight = 0.1),
    list(pred_pattern = "bpi_broad_t", start_col = "starting_bpi_broad", weight = 0.1),
    list(pred_pattern = "bpi_fine_t", start_col = "starting_bpi_fine", weight = 0.1),
    list(pred_pattern = "tsm_", start_col = "starting_tsm", weight = 0.1) # Match TSM
    # Add neighborhood stats if PDPs show strong trends
    # list(pred_pattern = "slope_mean3_t", start_col = "starting_slope_mean3", weight = 0.05),
    # list(pred_pattern = "rugosity_sd3_t", start_col = "starting_rugosity_sd3", weight = 0.05)
  )
  
  # Filter to only use vars present in both prediction data and PDP data
  pdp_match_vars <- list()
  available_pdp_predictors <- unique(pdp_data$Predictor)
  available_pred_cols <- names(prediction_data)
  
  for(var_info in pdp_match_vars_potential){
    # Find the specific predictor name in PDP data (could be bathy_t, tsm_2004_2006 etc.)
    actual_pdp_predictor <- grep(var_info$pred_pattern, available_pdp_predictors, value = TRUE)[1]
    if(!is.na(actual_pdp_predictor) && var_info$start_col %in% available_pred_cols){
      var_info$pred_pattern <- actual_pdp_predictor # Use the exact name found
      pdp_match_vars[[length(pdp_match_vars) + 1]] <- var_info
    }
  }
  
  if(length(pdp_match_vars) == 0){
    warning("No common variables found for PDP matching. Skipping.")
    prediction_data[, pdp_adjusted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching PDP using variables:", paste(sapply(pdp_match_vars, `[[`, "start_col"), collapse=", "), "\n")
  
  prediction_data[, pdp_adjusted_change := {
    current_point <- .SD
    is_match <- rep(FALSE, nrow(pdp_data))
    
    for (var in pdp_match_vars) {
      predictor_name <- var$pred_pattern # Now uses the exact name
      range_info <- env_ranges[Predictor == predictor_name]
      
      if (nrow(range_info) > 0 && var$start_col %in% names(current_point)) {
        current_val <- current_point[[var$start_col]]
        if (!is.na(current_val)) {
          half_window <- var$weight * range_info$range_width
          # Ensure half_window is not NA or zero before proceeding
          if(!is.na(half_window) && half_window > 0){
            is_match <- is_match | (pdp_data$Predictor == predictor_name &
                                      pdp_data$Env_Value >= (current_val - half_window) &
                                      pdp_data$Env_Value <= (current_val + half_window))
          } else if (!is.na(half_window) && half_window == 0){ # Handle constant predictors
            is_match <- is_match | (pdp_data$Predictor == predictor_name & pdp_data$Env_Value == current_val)
          }
        }
      }
    }
    
    matches <- pdp_data[is_match]
    if (nrow(matches) > 0) {
      mean_pdp_prediction <- mean(matches$PDP_Value, na.rm = TRUE)
      start_bathy_val <- current_point$starting_bathy
      change <- if (!is.null(start_bathy_val) && !is.na(start_bathy_val)) mean_pdp_prediction - start_bathy_val else NA_real_
      change
    } else {
      NA_real_
    }
  }, by = 1:nrow(prediction_data)]
  
  message("  - INFO: PDP Adjusted Change matched on: ", sum(!is.na(prediction_data$pdp_adjusted_change)), " points.")
  return(prediction_data)
}


#' --- 6. Apply KNN Trend Adjustments (Based on Training Data Change) ---
apply_training_knn_adjustment <- function(prediction_data, training_data, k = 15) {
  cat("  - INFO: Applying Training KNN adjustment (based on observed change)...\n")
  setDT(prediction_data); setDT(training_data)
  
  # **MODIFICATION**: Expand potential variables for KNN matching
  potential_env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity",
                          "starting_bpi_broad", "starting_bpi_fine",
                          "starting_terrain_class", "starting_grain_size", "starting_sed_type",
                          "starting_tsm",
                          grep("^starting_.*_(mean3|sd3)$", names(prediction_data), value = TRUE)) # NBH stats
  
  # Find common variables dynamically
  env_vars <- intersect(potential_env_vars, names(prediction_data))
  env_vars <- intersect(env_vars, names(training_data)) # Ensure they exist in training data too
  
  if (length(env_vars) == 0) {
    warning("No common environmental variables found for Training KNN. Skipping.")
    prediction_data[, TrainingKNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching using variables:", paste(env_vars, collapse=", "), "\n")
  
  pred_clean <- prediction_data[complete.cases(prediction_data[, ..env_vars])]
  train_clean <- training_data[complete.cases(training_data[, ..env_vars])]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases for Training KNN. Skipping."); return(prediction_data) }
  if (nrow(train_clean) < k) { stop("Not enough training data for Training KNN.") }
  
  response_var <- "b_change"
  if (!response_var %in% names(train_clean)) stop("Missing 'b_change' in training data for Training KNN.")
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]]
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  weights <- 1 / (knn_result$nn.dist + 1e-6)
  # Handle cases where all weights might be zero (or distances Inf) for a point
  sum_weights <- rowSums(weights, na.rm = TRUE)
  pred_avg <- rowSums(matrix(train_resp[knn_result$nn.index], nrow = nrow(knn_result$nn.index)) * weights, na.rm = TRUE) / sum_weights
  pred_avg[sum_weights == 0] <- NA_real_ # Assign NA if sum of weights is zero
  
  pred_clean[, TrainingKNN_predicted_change := pred_avg]
  
  prediction_data[pred_clean, on = .(FID), TrainingKNN_predicted_change := i.TrainingKNN_predicted_change]
  
  message("  - INFO: Assigned Training KNN predictions to ", sum(!is.na(prediction_data$TrainingKNN_predicted_change)), " rows.")
  return(prediction_data)
}


#' --- 7. Compare Methods and Create Hybrid Prediction ---
compare_prediction_methods <- function(prediction_data, tile_id, year_pair, processing_mode) {
  message("  - INFO: Combining results into hybrid prediction...")
  
  start_bathy_col <- if ("bathy_t" %in% names(prediction_data)) "bathy_t" else if ("starting_bathy" %in% names(prediction_data)) "starting_bathy" else NULL
  
  if (is.null(start_bathy_col)) {
    warning("Neither 'bathy_t' nor 'starting_bathy' column found, cannot calculate XGB predicted change.")
    prediction_data[, XGB_predicted_change := NA_real_]
  } else {
    if("XGB_predicted_bathy_t1" %in% names(prediction_data)) {
      prediction_data[, XGB_predicted_change := XGB_predicted_bathy_t1 - get(start_bathy_col)]
    } else {
      warning("XGB_predicted_bathy_t1 column missing. Cannot calculate XGB_predicted_change.")
      prediction_data[, XGB_predicted_change := NA_real_]
    }
  }
  
  hybrid_cols <- c("MeanBoots_predicted_change", "pdp_adjusted_change", "TrainingKNN_predicted_change")
  prediction_data[, hybrid_change := rowMeans(.SD, na.rm = TRUE), .SDcols = intersect(hybrid_cols, names(prediction_data))] # Use intersect
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  pdp_knn_df <- prediction_data[!is.na(pdp_adjusted_change) & !is.na(TrainingKNN_predicted_change)]
  comparison_log <- data.table(Tile_ID = tile_id, Year_Pair = year_pair, Total_Points = nrow(prediction_data),
                               PDP_TrainKNN_Cor = if (nrow(pdp_knn_df) > 2) cor(pdp_knn_df$pdp_adjusted_change, pdp_knn_df$TrainingKNN_predicted_change, use="complete.obs") else NA)
  
  performance_log <- NULL
  if (processing_mode == 'direct_model' && "b_change" %in% names(prediction_data)) {
    performance_log <- data.table(
      Tile_ID = tile_id, Year_Pair = year_pair,
      RMSE_XGB = rmse(prediction_data$b_change, prediction_data$XGB_predicted_change),
      RMSE_Hybrid = rmse(prediction_data$b_change, prediction_data$hybrid_change)
    )
  }
  
  return(list(data = prediction_data, comparison_log = comparison_log, performance_log = performance_log))
}

# ==============================================================================
#   HELPER FUNCTIONS (FILE I/O & PLOTTING)
# ==============================================================================
# ... (mae, rmse, save_final_predictions, save_component_rasters, build_all_tile_maps, run_cross_validation functions remain largely the same as V3) ...
# Ensure these functions are included in the final script.
mae <- function(actual, predicted) { mean(abs(actual - predicted), na.rm = TRUE) }
rmse <- function(actual, predicted) { sqrt(mean((actual - predicted)^2, na.rm = TRUE)) }

save_final_predictions <- function(prediction_data, output_dir, tile_id, year_pair) {
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  pred_depth_col <- paste0("pred_", end_year, "_depth")
  start_bathy_col <- if ("starting_bathy" %in% names(prediction_data)) "starting_bathy" else "bathy_t"
  
  if (!start_bathy_col %in% names(prediction_data) || !"hybrid_change" %in% names(prediction_data)) {
    stop("Cannot save final predictions: starting bathy or hybrid_change column is missing.")
  }
  
  prediction_data[, (pred_depth_col) := get(start_bathy_col) + hybrid_change]
  
  out_file_dir <- file.path(output_dir, tile_id); if (!dir.exists(out_file_dir)) dir.create(out_file_dir, recursive = TRUE)
  out_file <- file.path(out_file_dir, paste0(tile_id, "_prediction_final_", year_pair, ".fst"))
  
  write_fst(prediction_data, out_file)
  message("  - INFO: Final FST prediction file saved to: ", out_file)
  return(prediction_data)
}

save_component_rasters <- function(prediction_data, output_dir, tile_id, year_pair, crs_obj) {
  out_raster_dir <- file.path(output_dir, tile_id)
  if (!dir.exists(out_raster_dir)) dir.create(out_raster_dir, recursive = TRUE)
  
  save_raster <- function(data, col_name, file_suffix) {
    if (col_name %in% names(data)) {
      valid_rows <- data[!is.na(get(col_name)) & is.finite(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(valid_rows) > 0) {
        tryCatch({
          r <- raster::rasterFromXYZ(valid_rows, crs = crs_obj)
          out_path <- file.path(out_raster_dir, paste0(tile_id, "_", file_suffix, "_", year_pair, ".tif"))
          raster::writeRaster(r, out_path, format = "GTiff", overwrite = TRUE)
        }, error = function(e){
          cat("    - ERROR saving raster", file_suffix, ":", conditionMessage(e), "\n")
        })
      } else {
        cat("    - WARN: No valid finite data to save raster for", file_suffix, "\n")
      }
    } else {
      cat("    - WARN: Column", col_name, "not found for saving raster", file_suffix, "\n")
    }
  }
  
  save_raster(prediction_data, "hybrid_change", "Hybrid_predicted_change")
  save_raster(prediction_data, "pdp_adjusted_change", "PDP_predicted_change")
  save_raster(prediction_data, "TrainingKNN_predicted_change", "TrainingKNN_predicted_change") # Use correct name
  save_raster(prediction_data, "XGB_predicted_change", "XGB_predicted_change")
  save_raster(prediction_data, "MeanBoots_predicted_change", "MeanBoots_predicted_change")
}

build_all_tile_maps <- function(training_dir, prediction_dir, year_pairs) {
  message("  - INFO: Building year-specific geographic maps...")
  
  training_footprint_gpkg <- file.path(training_dir, "intersecting_sub_grids_UTM.gpkg")
  prediction_footprint_gpkg <- file.path(prediction_dir, "intersecting_sub_grids_UTM.gpkg")
  
  if(!all(file.exists(training_footprint_gpkg, prediction_footprint_gpkg))) {
    stop("Footprint GPKG files not found...") # Simplified error
  }
  
  train_sf_full <- sf::st_read(training_footprint_gpkg, quiet = TRUE)
  pred_sf <- sf::st_read(prediction_footprint_gpkg, quiet = TRUE)
  
  id_col <- "tile_id"
  suppressWarnings({ pred_centroids <- sf::st_centroid(pred_sf) })
  
  all_maps <- list()
  for (yp in year_pairs) {
    model_file_name <- paste0("xgb_model_", yp, ".rds")
    potential_tile_dirs <- list.dirs(training_dir, full.names = FALSE, recursive = FALSE)
    valid_training_tiles <- potential_tile_dirs[!grepl("diagnostic_plots|prediction_logs|\\.", potential_tile_dirs)]
    
    valid_source_tiles <- character()
    for (tile in valid_training_tiles) {
      tile_path <- file.path(training_dir, tile)
      if (dir.exists(tile_path) && file.exists(file.path(tile_path, model_file_name))) {
        valid_source_tiles <- c(valid_source_tiles, tile)
      }
    }
    
    if (length(valid_source_tiles) == 0) { all_maps[[yp]] <- NULL; next }
    
    train_sf_year_specific <- train_sf_full[train_sf_full[[id_col]] %in% valid_source_tiles, ]
    if(nrow(train_sf_year_specific) == 0) { all_maps[[yp]] <- NULL; next }
    
    suppressWarnings({ train_centroids_year_specific <- sf::st_centroid(train_sf_year_specific) })
    if(nrow(train_centroids_year_specific) == 0){ all_maps[[yp]] <- NULL; next }
    
    nearest_indices <- sf::st_nearest_feature(pred_centroids, train_centroids_year_specific)
    
    tile_map <- train_sf_year_specific[[id_col]][nearest_indices]
    names(tile_map) <- pred_sf[[id_col]]
    
    all_maps[[as.character(yp)]] <- tile_map
  }
  return(all_maps)
}


run_cross_validation <- function(processed_data, tile_id, year_pair, training_dir) {
  message("  - INFO: Running cross-validation for tile: ", tile_id)
  
  train_file_prefix <- sub("_[0-9]+$", "", tile_id)
  training_data_path <- file.path(training_dir, tile_id, paste0(train_file_prefix, "_", year_pair, "_long.fst"))
  
  if(!file.exists(training_data_path)) {
    message("  - WARN: No training data file for cross-validation. Skipping.")
    return(NULL)
  }
  
  truth_data <- read_fst(training_data_path, as.data.table = TRUE)
  
  validation_data <- merge(processed_data, truth_data, by = "FID", suffixes = c("_pred", "_actual"))
  
  actual_depth_col <- "bathy_t1"
  pred_depth_col <- paste0("pred_", as.numeric(strsplit(year_pair, "_")[[1]][2]), "_depth")
  actual_change_col <- "delta_bathy"
  
  required_cols <- c(actual_depth_col, pred_depth_col, actual_change_col, "hybrid_change")
  if (!all(required_cols %in% names(validation_data))) {
    message("  - WARN: Missing columns for cross-validation. Skipping.")
    return(NULL)
  }
  
  valid_comp_data <- validation_data[!is.na(get(actual_depth_col)) & !is.na(get(pred_depth_col))]
  if(nrow(valid_comp_data) < 2) return(NULL)
  
  cv_log <- data.table(Tile_ID = tile_id, Year_Pair = year_pair,
                       Depth_MAE = mae(valid_comp_data[[actual_depth_col]], valid_comp_data[[pred_depth_col]]),
                       Change_MAE = mae(valid_comp_data[[actual_change_col]], valid_comp_data$hybrid_change))
  return(cv_log)
}

# ==============================================================================
#   Parallel Prediction Test Wrapper
# ==============================================================================
run_prediction_test <- function(prediction_tiles, year_pairs, crs_obj, training_dir, prediction_dir, output_dir, master_predictor_list_path) {
  
  message("\nStarting parallel prediction test run...")
  
  all_tile_maps <- build_all_tile_maps(training_dir, prediction_dir, year_pairs)
  
  num_cores <- max(1, floor(detectCores() / 2)) # Use fewer cores
  cat("  - INFO: Setting up parallel cluster with", num_cores, "cores.\n")
  cl <- makeCluster(num_cores); registerDoParallel(cl); on.exit({ stopCluster(cl) }, add = TRUE)
  
  task_grid <- expand.grid(tile_id = prediction_tiles, year_pair = year_pairs, stringsAsFactors = FALSE)
  
  message("\n--- Stage 1: Processing data in parallel... ---")
  
  functions_to_export <- c("process_tile", "predict_elevation_change", "align_predictors",
                           "apply_bootstrap_knn_adjustment", # Use new bootstrap function
                           "match_pdp_conditions",
                           "apply_training_knn_adjustment", # Use renamed KNN function
                           "compare_prediction_methods",
                           "save_final_predictions", "save_component_rasters", "run_cross_validation",
                           "mae", "rmse")
  
  results_list <- foreach(
    i = 1:nrow(task_grid),
    .export = functions_to_export,
    .packages = c("data.table", "fst", "xgboost", "FNN", "stringr", "Metrics") # Reduced packages
  ) %dopar% {
    # Load heavier packages inside worker
    library(dplyr)
    library(sf)
    library(raster)
    
    current_task <- task_grid[i, ]
    process_tile(
      tile_id        = current_task$tile_id,
      year_pair      = current_task$year_pair,
      training_dir   = training_dir,
      prediction_dir = prediction_dir,
      output_dir     = output_dir,
      all_tile_maps  = all_tile_maps,
      master_predictor_list_path = master_predictor_list_path
    )
  }
  
  message("\n--- Stage 2: Saving results and plots sequentially... ---")
  
  successful_results <- Filter(function(res) is.list(res) && isTRUE(res$success), results_list)
  
  all_cv_logs <- list()
  # Load plotting libraries here if needed for sequential plotting steps
  # library(rasterVis); library(gridExtra); library(ggplot2); library(cmocean); library(raster)
  
  for (result in successful_results) {
    cat("\n--- Saving outputs for Tile:", result$tile_id, "| Year Pair:", result$year_pair, "---\n")
    processed_data <- save_final_predictions(result$data, output_dir, result$tile_id, result$year_pair)
    if (!is.null(crs_obj)) {
      # Ensure raster is loaded before calling save_component_rasters
      library(raster)
      save_component_rasters(processed_data, output_dir, result$tile_id, result$year_pair, crs_obj)
      
      if (result$processing_mode == 'direct_model') {
        cv_log <- run_cross_validation(processed_data, result$tile_id, result$year_pair, training_dir)
        if(!is.null(cv_log)) all_cv_logs[[length(all_cv_logs) + 1]] <- cv_log
      }
    }
  }
  
  if(length(all_cv_logs) > 0) {
    fwrite(rbindlist(all_cv_logs, fill = TRUE), file.path(output_dir, "cross_validation_log.csv"))
    message("  - INFO: Cross-validation log saved.")
  }
  
  message("\n✅ Parallel prediction test run complete.")
}

#----




#-----------------




## -------
# ==============================================================================
#
#   XGBoost Prediction Function Set (V6.8 - Scoping Bug Fix)
#
# ==============================================================================
#
# Key Enhancements in V6.8:
#   - FIXED (Fatal Error): Corrected a scoping bug in 'process_tile' where
#     'training_data_path' was defined using 'model_tile_id' *before*
#     'model_tile_id' was assigned.
#   - RETAINS V6.7 FIXES:
#     - Land Filter: Filters 'bathy_t > 0' from prediction data *before* processing.
#     - delta_bathy Plot: Plotting functions load 'delta_bathy' from the
#       original training file to remove 'NA' striations.
#     - Robust KNN: KNN functions use a 5-predictor core to prevent 'NA' dropouts.
#     - Feature Log: Saves 'model_feature_usage_log_v6.8.csv'.
#     - Feature Mismatch: 'predict_elevation_change_ensemble' dynamically
#       reads 'xgb_model$feature_names' for each model.
#     - Weighted KNN: KNN adjustments use 'calculate_sample_weights'.
#     - V4 Logic: Includes 'apply_bootstrap_knn_adjustment'.
#
# ==============================================================================

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis)
library(gridExtra)
library(ggplot2)
library(FNN)
library(foreach)
library(doParallel)
library(Metrics)
library(cmocean)
library(stringr)

# --- Helper: Weighted Loss Function (from training) ---
calculate_sample_weights <- function(delta_bathy, alpha = 1.5, epsilon = 1e-6) {
  abs_change <- abs(delta_bathy) + epsilon
  weights <- abs_change^alpha
  weights <- weights / sum(weights, na.rm = TRUE) * length(weights)
  weights[!is.finite(weights)] <- epsilon
  return(weights)
}

# =Monitoring - V6.8 ---
process_tile <- function(tile_id, year_pair, training_dir, prediction_dir, output_dir, all_tile_maps, master_predictor_list_path, n_boot_models_to_use = 10) {
  # --- a. Setup Logging ---
  pred_log_dir <- file.path(output_dir, tile_id, "prediction_logs")
  if (!dir.exists(pred_log_dir)) dir.create(pred_log_dir, recursive = TRUE)
  pred_log_file <- file.path(pred_log_dir, paste0("pred_log_", year_pair, "_v6.8.txt")) # Updated Suffix
  sink(pred_log_file, append = FALSE, split = TRUE); on.exit({ sink() })
  cat("\n--- Starting Tile:", tile_id, "| Year Pair:", year_pair, "(V6.8 Ensemble) ---\n")
  
  # Initialize the log for model features
  model_feature_log <- list()
  
  # --- SCOPING FIX: 'training_data_path' is defined *inside* the tryCatch block ---
  
  tryCatch({
    # --- b. Determine Model Source Tile ---
    tile_map_for_year <- all_tile_maps[[year_pair]]
    if (is.null(tile_map_for_year)) stop("No valid models found for year pair: ", year_pair)
    model_tile_id <- if (tile_id %in% names(tile_map_for_year)) tile_map_for_year[[tile_id]] else tile_id
    processing_mode <- if (model_tile_id == tile_id) "direct_model" else "reference_model"
    cat("  - INFO: Processing mode detected:", processing_mode, "\n")
    if(processing_mode == "reference_model") cat("  - INFO: Using reference model(s) from tile:", model_tile_id, "\n")
    
    # --- NEW: Define training_data_path *after* model_tile_id is known ---
    train_file_prefix <- sub("_[0-9]+$", "", model_tile_id)
    training_data_path <- file.path(training_dir, model_tile_id, paste0(train_file_prefix, "_", year_pair, "_long.fst"))
    
    # --- c. NEW: Load and Filter Prediction Data (Land Filter) ---
    cat("Step 1: Loading and filtering prediction data...\n")
    pred_file_prefix <- sub("_[0-9]+$", "", tile_id)
    prediction_data_path <- file.path(prediction_dir, tile_id, paste0(pred_file_prefix, "_", year_pair, "_prediction_long.fst"))
    if (!file.exists(prediction_data_path)) stop(paste("Missing prediction data file:", basename(prediction_data_path)))
    
    prediction_data <- read_fst(prediction_data_path, as.data.table = TRUE)
    cat("  - INFO: Initial prediction data has", nrow(prediction_data), "rows.\n")
    
    if ("bathy_t" %in% names(prediction_data)) {
      # Filter: keep only rows where bathy_t is <= 0 (not land)
      # Use !is.na() to also keep rows where bathy_t might be NA
      prediction_data <- prediction_data[!(bathy_t > 0)]
      cat("  - INFO: Filtered to", nrow(prediction_data), "rows (bathy_t <= 0 or NA).\n")
    } else {
      cat("  - WARN: 'bathy_t' column not found. Cannot filter land values.\n")
    }
    if (nrow(prediction_data) == 0) stop("No data remaining after land filter.")
    
    # --- d. Generate Initial Ensemble Predictions ---
    cat("Step 2: Generating initial predictions using bootstrap ensemble...\n")
    initial_predictions_list <- predict_elevation_change_ensemble(
      prediction_data, # Pass the filtered data.table
      model_tile_id, 
      year_pair, 
      training_dir,
      master_predictor_list_path, 
      n_boot_models_to_use = n_boot_models_to_use
    )
    if (is.null(initial_predictions_list)) stop("Initial ensemble prediction failed.")
    initial_predictions_dt <- initial_predictions_list$summary_dt
    model_feature_log <- initial_predictions_list$model_feature_log # Capture the log
    
    # --- e. Load Adjustment Data ---
    cat("Step 3: Loading adjustment data from source:", model_tile_id, "\n")
    pdp_file <- file.path(training_dir, model_tile_id, paste0("pdp_data_long_", year_pair, ".fst"))
    shap_summary_path <- file.path(training_dir, model_tile_id, paste0("shap_summary_", year_pair, ".fst"))
    # Path for V4 Bootstrap method
    boot_summary_path <- file.path(training_dir, model_tile_id, paste0("bootstraps_summary_df_", year_pair, ".fst"))
    
    required_files <- c(pdp_file, training_data_path, shap_summary_path, boot_summary_path)
    missing_files <- required_files[!file.exists(required_files)]
    if(length(missing_files) > 0){
      stop("Missing required adjustment data file(s): ", paste(basename(missing_files), collapse=", "))
    }
    
    pdp_data <- read_fst(pdp_file, as.data.table = TRUE)
    training_data <- read_fst(training_data_path, as.data.table = TRUE)
    shap_summary <- read_fst(shap_summary_path, as.data.table = TRUE)
    boot_summary_data <- read_fst(boot_summary_path, as.data.table = TRUE)
    
    cat("  - INFO: Loaded", nrow(training_data), "raw training data rows for adjustments.\n")
    
    # --- Align training data ---
    aligned_training_data <- align_predictors(training_data, year_pair, data_type = "training")
    # Align bootstrap summary data (for V4 KNN method)
    aligned_boot_summary_data <- align_predictors(boot_summary_data, year_pair, data_type = "training")
    
    
    # --- f. Apply Hybrid Adjustments ---
    cat("Step 4: Applying hybrid adjustments (PDP, Training KNN, SHAP-KNN, Boot-KNN)...\n")
    
    if(nrow(pdp_data) == 0 || all(is.na(pdp_data$PDP_Value))) {
      cat("  - WARN: PDP data is empty or all NA. Skipping PDP adjustment.\n")
      pdp_enriched <- initial_predictions_dt
      pdp_enriched[, pdp_adjusted_change := NA_real_]
    } else {
      pdp_enriched <- match_pdp_conditions(initial_predictions_dt, pdp_data)
    }
    
    # KNN on *actual change* (delta_bathy)
    train_knn_enriched <- apply_training_knn_adjustment(pdp_enriched, aligned_training_data)
    
    # SHAP-Weighted KNN on *actual change* (delta_bathy)
    if(nrow(shap_summary) == 0 || all(is.na(shap_summary$Overall_Mean_Abs_SHAP))) {
      cat("  - WARN: SHAP summary is empty or all NA. Skipping SHAP-KNN adjustment.\n")
      shap_knn_enriched <- train_knn_enriched
      shap_knn_enriched[, SHAP_KNN_predicted_change := NA_real_]
    } else {
      shap_knn_enriched <- apply_shap_weighted_knn_adjustment(train_knn_enriched, aligned_training_data, shap_summary)
    }
    
    # V4-style KNN on *mean predicted change*
    boot_knn_enriched <- apply_bootstrap_knn_adjustment(shap_knn_enriched, aligned_boot_summary_data)
    
    
    # --- g. Merge Actual Change (delta_bathy) for direct mode ---
    final_data_for_comparison <- copy(boot_knn_enriched)
    if (processing_mode == 'direct_model') {
      cat("  - INFO (Main): Direct model. Merging 'delta_bathy' for plots/validation.\n")
      
      if ("delta_bathy" %in% names(aligned_training_data) && any(!is.na(aligned_training_data$delta_bathy))) {
        truth_data <- aligned_training_data[, .(FID, delta_bathy_actual_truth = delta_bathy)]
        final_data_for_comparison[truth_data, on = "FID", delta_bathy := i.delta_bathy_actual_truth]
        
        n_merged <- sum(!is.na(final_data_for_comparison$delta_bathy))
        cat("  - INFO (Main): Merged actual 'delta_bathy' for", n_merged, "points.\n")
      } else {
        cat("  - WARN (Main): 'delta_bathy' column not found or all NA in training data.\n")
        if(!"delta_bathy" %in% names(final_data_for_comparison)) final_data_for_comparison[, delta_bathy := NA_real_]
      }
      
    } else {
      if(!"delta_bathy" %in% names(final_data_for_comparison)) final_data_for_comparison[, delta_bathy := NA_real_]
    }
    
    # --- h. Compare and Finalize ---
    cat("Step 5: Comparing prediction methods and combining results...\n")
    comparison_results <- compare_prediction_methods_v6(final_data_for_comparison, tile_id, year_pair, processing_mode = processing_mode)
    
    cat("\n--- SUCCESS (Data Processing): Completed Tile:", tile_id, "| Year Pair:", year_pair, "---\n")
    
    return(list(
      data = comparison_results$data,
      comparison_log = comparison_results$comparison_log,
      performance_log = comparison_results$performance_log,
      model_feature_log = model_feature_log, # Pass log back
      training_data_path = training_data_path, # Pass path for plotting
      processing_mode = processing_mode,
      tile_id = tile_id,
      year_pair = year_pair,
      success = TRUE
    ))
    
  }, error = function(e) { # Outer tryCatch
    cat("\n--- FATAL ERROR in process_tile (V6.8) ---\n")
    cat("  - Tile:", tile_id, "| Pair:", year_pair, "\n")
    cat("  - Error Message:", conditionMessage(e), "\n")
    cat("  - Traceback:\n", paste(capture.output(traceback()), collapse="\n"), "\n")
    return(list(success = FALSE, tile_id = tile_id, year_pair = year_pair, error = conditionMessage(e), model_feature_log = model_feature_log))
  })
}

#' --- 2. Core Prediction Function (Ensemble) (V6.8 - DYNAMIC FEATURE FIX) ---
predict_elevation_change_ensemble <- function(
    prediction_data, # Now receives a data.table
    model_tile_id, 
    year_pair, 
    training_dir, 
    master_predictor_list_path, 
    n_boot_models_to_use = 10
) {
  
  # Initialize the feature log
  model_feature_log <- list()
  
  # --- 1. Align prediction data (creates 'starting_...' columns and 'delta_bathy=NA') ---
  aligned_pred_data <- align_predictors(prediction_data, year_pair, data_type = "prediction")
  
  if(!("bathy_t" %in% names(aligned_pred_data))){
    if("starting_bathy" %in% names(aligned_pred_data)) {
      aligned_pred_data[, bathy_t := starting_bathy]
    } else {
      stop("'bathy_t' or 'starting_bathy' column missing after alignment in prediction data.")
    }
  }
  
  model_dir <- file.path(training_dir, model_tile_id, "bootstrap_models")
  model_files <- list.files(model_dir, pattern = paste0("^model_boot_\\d+_", year_pair, "\\.rds$"), full.names = TRUE)
  
  if (length(model_files) == 0) stop("No bootstrap model files found in '", model_dir, "' for year pair ", year_pair)
  
  n_models_available <- length(model_files)
  n_models_to_load <- min(n_boot_models_to_use, n_models_available)
  models_to_load <- if (n_models_to_load < n_models_available) sample(model_files, n_models_to_load) else model_files
  cat("  - INFO: Loading", n_models_to_load, "of", n_models_available, "bootstrap models for prediction.\n")
  
  # --- 2. Load master predictor list (for logging/comparison only) ---
  if (!file.exists(master_predictor_list_path)) stop("Master predictor list file not found.")
  master_predictors_list <- readLines(master_predictor_list_path)
  target_model_features_superset <- sapply(master_predictors_list, function(p) {
    if (p %in% c("hurr_count", "hurr_strength", "tsm")) {
      paste0(p, "_", year_pair)
    } else {
      p
    }
  }, USE.NAMES = FALSE)
  cat("  - INFO: Master list has", length(target_model_features_superset), "potential features.\n")
  
  
  # --- 3. Run Prediction Loop ---
  all_predictions_array_t1 <- array(NA, dim = c(nrow(aligned_pred_data), 1, n_models_to_load))
  
  for (b in 1:n_models_to_load) {
    model_path <- models_to_load[b]
    xgb_model <- readRDS(model_path)
    
    # --- START FIX: DYNAMICALLY GET FEATURES FROM *THIS* MODEL ---
    model_features <- xgb_model$feature_names
    
    # --- NEW: Log features for this model ---
    model_feature_log[[length(model_feature_log) + 1]] <- data.table(
      tile_id = model_tile_id, 
      year_pair = year_pair, 
      model_file = basename(model_path), 
      feature = model_features,
      n_features = length(model_features)
    )
    if (b == 1) { # Log this only for the first model
      cat("  - INFO: Model 1 (", basename(model_path), ") uses", length(model_features), "features (e.g.,", paste(head(model_features, 3), collapse=", "), "...).\n")
    }
    
    # Check if this model's features are available in the prediction data
    missing_features <- setdiff(model_features, names(aligned_pred_data))
    if (length(missing_features) > 0) {
      cat("  - FATAL: Model", b, "requires features not found in prediction data:", paste(missing_features, collapse=", "), "\n")
      stop("Missing required predictor columns in prediction data file for model ", b)
    }
    
    # Create the prediction matrix *using only the features this model expects*
    pred_matrix <- as.matrix(aligned_pred_data[, ..model_features])
    dpredict_model_specific <- xgb.DMatrix(data = pred_matrix, missing = NA)
    
    # Predict using this model-specific DMatrix
    all_predictions_array_t1[, 1, b] <- predict(xgb_model, newdata = dpredict_model_specific)
    # --- END FIX ---
  }
  
  Mean_Prediction_t1 <- apply(all_predictions_array_t1, 1, mean, na.rm = TRUE)
  Uncertainty_SD <- apply(all_predictions_array_t1, 1, sd, na.rm = TRUE)
  Uncertainty_SD[is.na(Uncertainty_SD)] <- 0
  
  summary_dt <- data.table(
    FID = aligned_pred_data$FID, X = aligned_pred_data$X, Y = aligned_pred_data$Y,
    tile_id = aligned_pred_data$tile_id
  )
  
  start_cols <- grep("^starting_", names(aligned_pred_data), value = TRUE)
  for (col in start_cols) {
    summary_dt[[col]] <- aligned_pred_data[[col]]
  }
  if (!("bathy_t" %in% names(summary_dt)) && "bathy_t" %in% names(aligned_pred_data)) {
    summary_dt[, bathy_t := aligned_pred_data$bathy_t]
  }
  
  summary_dt[, `:=`(
    XGB_predicted_bathy_t1 = Mean_Prediction_t1,
    Uncertainty_SD = Uncertainty_SD,
    delta_bathy = aligned_pred_data$delta_bathy # This will be all NA from prediction data
  )]
  
  summary_dt[, XGB_predicted_change := XGB_predicted_bathy_t1 - bathy_t]
  
  message("  - INFO: Generated ensemble predictions for ", nrow(summary_dt), " points.")
  return(list(
    summary_dt = summary_dt, 
    all_predictions_array_t1 = all_predictions_array_t1,
    model_feature_log = model_feature_log # Return the log
  ))
}


#' --- 3. Align Predictors (V6.8 - SIMPLIFIED) ---
align_predictors <- function(data_to_align, year_pair, data_type = "prediction") { # Default to prediction
  setDT(data_to_align)
  cat(paste0("  - DIAGNOSTIC (align_predictors): Aligning '", data_type, "' data... "))
  
  # This function now assumes data *already* has '_t' or '_year_pair' suffixes.
  # Its job is to create 'starting_...' columns for KNN/PDP and to define 'delta_bathy'.
  
  # --- 1. Create 'starting_...' columns for KNN/PDP/SHAP ---
  # These are based *directly* on the '..._t' columns
  if ("bathy_t" %in% names(data_to_align)) data_to_align[, starting_bathy := bathy_t]
  if ("slope_t" %in% names(data_to_align)) data_to_align[, starting_slope := slope_t]
  if ("rugosity_t" %in% names(data_to_align)) data_to_align[, starting_rugosity := rugosity_t]
  if ("bpi_broad_t" %in% names(data_to_align)) data_to_align[, starting_bpi_broad := bpi_broad_t]
  if ("bpi_fine_t" %in% names(data_to_align)) data_to_align[, starting_bpi_fine := bpi_fine_t]
  if ("terrain_classification_t" %in% names(data_to_align)) data_to_align[, starting_terrain_class := terrain_classification_t]
  
  # Handle neighborhood stats
  nbh_cols_t <- grep("_(mean3|sd3)_t$", names(data_to_align), value = TRUE)
  for(col in nbh_cols_t){
    new_col_name <- paste0("starting_", sub("_t$", "", col)) # e.g., bathy_sd3_t -> starting_bathy_sd3
    data_to_align[, (new_col_name) := get(col)]
  }
  
  # Handle static layers
  if ("grain_size_layer" %in% names(data_to_align)) data_to_align[, starting_grain_size := grain_size_layer]
  if ("prim_sed_layer" %in% names(data_to_align)) data_to_align[, starting_sed_type := prim_sed_layer]
  if ("survey_end_date" %in% names(data_to_align)) data_to_align[, starting_survey_date := survey_end_date]
  
  # Handle forcing layers (year-pair specific)
  tsm_var_name <- paste0("tsm_", year_pair)
  if (tsm_var_name %in% names(data_to_align)) data_to_align[, starting_tsm := get(tsm_var_name)]
  
  hurr_c_var <- paste0("hurr_count_", year_pair)
  if (hurr_c_var %in% names(data_to_align)) data_to_align[, starting_hurr_count := get(hurr_c_var)]
  
  hurr_s_var <- paste0("hurr_strength_", year_pair)
  if (hurr_s_var %in% names(data_to_align)) data_to_align[, starting_hurr_strength := get(hurr_s_var)]
  
  
  # --- 2. Define 'delta_bathy' column based on data_type ---
  if (data_type == "training") {
    if ("delta_bathy" %in% names(data_to_align)) {
      # 'delta_bathy' already exists, ensure it is the column used
    } else if ("bathy_t1" %in% names(data_to_align) && "bathy_t" %in% names(data_to_align)) {
      # Calculate delta_bathy if it's missing but sources are present
      data_to_align[, delta_bathy := bathy_t1 - bathy_t]
    }
    # Ensure delta_bathy is NA if it couldn't be calculated (should be filtered later)
    if (!"delta_bathy" %in% names(data_to_align)) data_to_align[, delta_bathy := NA_real_]
    
  } else { # data_type == "prediction"
    # Ensure delta_bathy column exists in prediction data, but as NA
    if (!"delta_bathy" %in% names(data_to_align)) data_to_align[, delta_bathy := NA_real_]
  }
  
  cat("Done.\n")
  return(data_to_align)
}


# --- 4. Match PDP Conditions (V6.8 - Robust KNN Vars) ---
match_pdp_conditions <- function(prediction_data, pdp_data) {
  cat("  - INFO: Applying PDP adjustment...\n")
  setDT(prediction_data); setDT(pdp_data)
  
  if(nrow(pdp_data) == 0 || all(is.na(pdp_data$PDP_Value))) {
    cat("    - WARN: PDP data is empty or all NA. Skipping PDP adjustment.\n")
    prediction_data[, pdp_adjusted_change := NA_real_]
    return(prediction_data)
  }
  
  env_ranges <- pdp_data[, .(
    min_val = min(Env_Value, na.rm = TRUE),
    max_val = max(Env_Value, na.rm = TRUE),
    range_width = max(Env_Value, na.rm = TRUE) - min(Env_Value, na.rm = TRUE)
  ), by = Predictor]
  env_ranges <- env_ranges[is.finite(range_width)] # Filter out NA/Inf ranges
  
  # --- Use a core set of variables for matching ---
  pdp_match_vars_potential <- list(
    list(pred_pattern = "bathy_t", start_col = "starting_bathy", weight = 0.3),
    list(pred_pattern = "slope_t", start_col = "starting_slope", weight = 0.15),
    list(pred_pattern = "rugosity_t", start_col = "starting_rugosity", weight = 0.1),
    list(pred_pattern = "bpi_broad_t", start_col = "starting_bpi_broad", weight = 0.1),
    list(pred_pattern = "bpi_fine_t", start_col = "starting_bpi_fine", weight = 0.1),
    list(pred_pattern = "tsm_", start_col = "starting_tsm", weight = 0.1)
  )
  
  pdp_match_vars <- list()
  available_pdp_predictors <- unique(pdp_data$Predictor)
  available_pred_cols <- names(prediction_data)
  
  for(var_info in pdp_match_vars_potential){
    actual_pdp_predictor <- grep(var_info$pred_pattern, available_pdp_predictors, value = TRUE)[1]
    # Also check that the predictor has a valid range
    if(!is.na(actual_pdp_predictor) && var_info$start_col %in% available_pred_cols && actual_pdp_predictor %in% env_ranges$Predictor){
      var_info$pred_pattern <- actual_pdp_predictor
      pdp_match_vars[[length(pdp_match_vars) + 1]] <- var_info
    }
  }
  
  if(length(pdp_match_vars) == 0){
    warning("No common variables with valid ranges found for PDP matching. Skipping.")
    prediction_data[, pdp_adjusted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching PDP using variables:", paste(sapply(pdp_match_vars, `[[`, "start_col"), collapse=", "), "\n")
  
  prediction_data[, pdp_adjusted_change := {
    current_point <- .SD
    is_match <- rep(FALSE, nrow(pdp_data))
    for (var in pdp_match_vars) {
      predictor_name <- var$pred_pattern
      range_info <- env_ranges[Predictor == predictor_name] # Already filtered for finite range_width
      if (nrow(range_info) > 0 && var$start_col %in% names(current_point)) {
        current_val <- current_point[[var$start_col]]
        if (!is.na(current_val)) {
          half_window <- var$weight * range_info$range_width
          if(!is.na(half_window) && half_window > 0){
            is_match <- is_match | (pdp_data$Predictor == predictor_name &
                                      pdp_data$Env_Value >= (current_val - half_window) &
                                      pdp_data$Env_Value <= (current_val + half_window))
          } else if (!is.na(half_window) && half_window == 0){
            is_match <- is_match | (pdp_data$Predictor == predictor_name & pdp_data$Env_Value == current_val)
          }
        }
      }
    }
    matches <- pdp_data[is_match]
    if (nrow(matches) > 0) mean(matches$PDP_Value, na.rm = TRUE) else NA_real_
  }, by = 1:nrow(prediction_data)]
  
  message("  - INFO: PDP Adjusted Change matched on: ", sum(!is.na(prediction_data$pdp_adjusted_change)), " points.")
  return(prediction_data)
}


# --- 5. Apply KNN Trend Adjustments (V6.8 - Robust + Weighted) ---
apply_training_knn_adjustment <- function(prediction_data, training_data, k = 15) {
  cat("  - INFO: Applying Training KNN adjustment (Weighted by delta_bathy)...\n")
  setDT(prediction_data); setDT(training_data)
  
  # --- NEW: Use a core set of variables for robust matching ---
  potential_env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity", 
                          "starting_bpi_broad", "starting_bpi_fine")
  
  env_vars <- intersect(potential_env_vars, names(prediction_data))
  env_vars <- intersect(env_vars, names(training_data)) # Ensure they exist in training data too
  
  if (length(env_vars) == 0) {
    warning("No common env variables for Training KNN. Skipping.")
    prediction_data[, TrainingKNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching using variables:", paste(env_vars, collapse=", "), "\n")
  
  response_var <- "delta_bathy" # Use delta_bathy
  if (!response_var %in% names(training_data) || all(is.na(training_data[[response_var]]))) {
    if (all(c("bathy_t1", "bathy_t") %in% names(training_data))){
      training_data[, delta_bathy := bathy_t1 - bathy_t]
      cat("    - INFO: 'delta_bathy' column calculated for Training KNN.\n")
    } else {
      stop("Missing 'delta_bathy' or source columns in training data.")
    }
  }
  
  cols_for_pred_cases <- env_vars
  for(col in cols_for_pred_cases) {
    if(!is.numeric(prediction_data[[col]])) {
      cat("    - WARN (TrainKNN): Coercing pred col to numeric:", col, "\n")
      prediction_data[, (col) := as.numeric(get(col))]
    }
  }
  pred_numeric_check <- sapply(prediction_data[, ..cols_for_pred_cases], is.numeric)
  if(!all(pred_numeric_check)) { stop("Not all prediction columns for KNN are numeric.") }
  
  keep_rows_pred <- prediction_data[, complete.cases(.SD), .SDcols = cols_for_pred_cases]
  pred_clean <- prediction_data[keep_rows_pred]
  
  cols_for_train_cases <- c(env_vars, response_var)
  for(col in cols_for_train_cases) {
    if(!is.numeric(training_data[[col]])) {
      cat("    - WARN (TrainKNN): Coercing train col to numeric:", col, "\n")
      training_data[, (col) := as.numeric(get(col))]
    }
  }
  train_numeric_check <- sapply(training_data[, ..cols_for_train_cases], is.numeric)
  if(!all(train_numeric_check)) { stop("Not all training columns for KNN are numeric.") }
  
  keep_rows_train <- training_data[, complete.cases(.SD), .SDcols = cols_for_train_cases]
  train_clean <- training_data[keep_rows_train]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases for Training KNN. Skipping."); prediction_data[, TrainingKNN_predicted_change := NA_real_]; return(prediction_data) }
  if (nrow(train_clean) < k) {
    message("    - WARN: Not enough training data (", nrow(train_clean), ") for Training KNN (k=", k, "). Skipping adjustment.");
    prediction_data[, TrainingKNN_predicted_change := NA_real_];
    return(prediction_data)
  }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]] # This is delta_bathy
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  # --- ENHANCEMENT: Use weighted mean based on training weights ---
  pred_avg <- sapply(1:nrow(knn_result$nn.index), function(i) {
    neighbor_indices <- knn_result$nn.index[i, ]
    neighbor_deltas <- train_resp[neighbor_indices]
    
    # Calculate weights *for the neighbors* based on their delta_bathy
    neighbor_weights <- calculate_sample_weights(neighbor_deltas)
    
    # Return the weighted mean
    return(weighted.mean(neighbor_deltas, neighbor_weights, na.rm = TRUE))
  })
  pred_avg[!is.finite(pred_avg)] <- NA_real_
  # --- END ENHANCEMENT ---
  
  pred_clean[, TrainingKNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), TrainingKNN_predicted_change := i.TrainingKNN_predicted_change]
  
  message("  - INFO: Assigned Training KNN predictions to ", sum(!is.na(prediction_data$TrainingKNN_predicted_change)), " rows.")
  return(prediction_data)
}


# --- 6. SHAP-Weighted KNN Adjustment (V6.8 - Robust + Weighted) ---
apply_shap_weighted_knn_adjustment <- function(prediction_data, training_data, shap_summary, k = 15) {
  cat("  - INFO: Applying SHAP-Weighted KNN adjustment (Weighted by delta_bathy)...\n")
  setDT(prediction_data); setDT(training_data); setDT(shap_summary)
  
  if(nrow(shap_summary) == 0 || all(is.na(shap_summary$Overall_Mean_Abs_SHAP))) {
    cat("    - WARN: SHAP summary is empty or invalid. Skipping SHAP-KNN adjustment.\n")
    prediction_data[, SHAP_KNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  
  # --- NEW: Define a core set of variables to consider for SHAP-KNN ---
  core_knn_vars <- c("starting_bathy", "starting_slope", "starting_rugosity", 
                     "starting_bpi_broad", "starting_bpi_fine")
  
  shap_predictors <- shap_summary$Predictor
  # This mapping is CRITICAL and links model names to 'starting_' names
  shap_to_starting_map <- data.table(
    Predictor = shap_predictors,
    starting_col = case_when(
      shap_predictors == "bathy_t" ~ "starting_bathy",
      shap_predictors == "slope_t" ~ "starting_slope",
      shap_predictors == "rugosity_t" ~ "starting_rugosity",
      shap_predictors == "bpi_broad_t" ~ "starting_bpi_broad",
      shap_predictors == "bpi_fine_t" ~ "starting_bpi_fine",
      shap_predictors == "terrain_classification_t" ~ "starting_terrain_class",
      shap_predictors == "grain_size_layer" ~ "starting_grain_size",
      shap_predictors == "prim_sed_layer" ~ "starting_sed_type",
      grepl("^tsm_", shap_predictors) ~ "starting_tsm",
      grepl("^hurr_count_", shap_predictors) ~ "starting_hurr_count",
      grepl("^hurr_strength_", shap_predictors) ~ "starting_hurr_strength",
      grepl("_(mean3|sd3)_t$", shap_predictors) ~ paste0("starting_", sub("_t$", "", shap_predictors)),
      TRUE ~ NA_character_
    )
  )
  
  valid_map <- shap_to_starting_map[!is.na(starting_col)]
  valid_map <- valid_map[starting_col %in% names(prediction_data)]
  valid_map <- valid_map[starting_col %in% names(training_data)]
  
  # --- NEW: Filter the map to ONLY the core_knn_vars ---
  valid_map <- valid_map[starting_col %in% core_knn_vars]
  
  if(nrow(valid_map) == 0) {
    warning("No common core variables found for SHAP-Weighted KNN. Skipping.")
    prediction_data[, SHAP_KNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  
  env_vars <- valid_map$starting_col
  
  shap_weights_dt <- merge(valid_map, shap_summary, by = "Predictor")
  shap_weights <- shap_weights_dt$Overall_Mean_Abs_SHAP
  names(shap_weights) <- shap_weights_dt$starting_col
  shap_weights <- (shap_weights / sum(shap_weights, na.rm=TRUE) * length(shap_weights))
  shap_weights[!is.finite(shap_weights)] <- 1e-6
  
  cat("    - Matching using SHAP-weighted variables:", paste(env_vars, collapse=", "), "\n")
  
  response_var <- "delta_bathy" # Use delta_bathy
  if (!response_var %in% names(training_data) || all(is.na(training_data[[response_var]]))) {
    if (all(c("bathy_t1", "bathy_t") %in% names(training_data))){
      training_data[, delta_bathy := bathy_t1 - bathy_t]
      cat("    - INFO: 'delta_bathy' column calculated for SHAP-KNN.\n")
    } else {
      stop("Missing 'delta_bathy' or source columns in training data for SHAP-KNN.")
    }
  }
  
  cols_for_pred_cases_shap <- env_vars
  for(col in cols_for_pred_cases_shap) {
    if(!is.numeric(prediction_data[[col]])) {
      cat("    - WARN (SHAP-KNN): Coercing pred col to numeric:", col, "\n")
      prediction_data[, (col) := as.numeric(get(col))]
    }
  }
  pred_numeric_check_shap <- sapply(prediction_data[, ..cols_for_pred_cases_shap], is.numeric)
  if(!all(pred_numeric_check_shap)) { stop("Not all prediction columns for SHAP-KNN are numeric.") }
  
  keep_rows_pred_shap <- prediction_data[, complete.cases(.SD), .SDcols = cols_for_pred_cases_shap]
  pred_clean <- prediction_data[keep_rows_pred_shap]
  
  cols_for_train_cases_shap <- c(env_vars, response_var)
  for(col in cols_for_train_cases_shap) {
    if(!is.numeric(training_data[[col]])) {
      cat("    - WARN (SHAP-KNN): Coercing train col to numeric:", col, "\n")
      training_data[, (col) := as.numeric(get(col))]
    }
  }
  train_numeric_check_shap <- sapply(training_data[, ..cols_for_train_cases_shap], is.numeric)
  if(!all(train_numeric_check_shap)) { stop("Not all training columns for SHAP-KNN are numeric.") }
  
  keep_rows_train_shap <- training_data[, complete.cases(.SD), .SDcols = cols_for_train_cases_shap]
  train_clean <- training_data[keep_rows_train_shap]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases for SHAP-KNN. Skipping."); prediction_data[, SHAP_KNN_predicted_change := NA_real_]; return(prediction_data) }
  if (nrow(train_clean) < k) {
    message("    - WARN: Not enough training data (", nrow(train_clean), ") for SHAP-KNN (k=", k, "). Skipping adjustment.");
    prediction_data[, SHAP_KNN_predicted_change := NA_real_];
    return(prediction_data)
  }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]] # This is delta_bathy
  
  all_mat <- rbind(pred_mat, train_mat)
  means <- colMeans(all_mat, na.rm=TRUE)
  sds <- apply(all_mat, 2, sd, na.rm=TRUE)
  sds[sds == 0] <- 1
  
  pred_mat_scaled <- scale(pred_mat, center = means, scale = sds)
  train_mat_scaled <- scale(train_mat, center = means, scale = sds)
  
  col_weights <- shap_weights[env_vars]
  
  pred_mat_weighted <- t(t(pred_mat_scaled) * col_weights)
  train_mat_weighted <- t(t(train_mat_scaled) * col_weights)
  
  knn_result <- FNN::get.knnx(train_mat_weighted, pred_mat_weighted, k = k)
  
  # --- ENHANCEMENT: Use weighted mean based on training weights ---
  pred_avg <- sapply(1:nrow(knn_result$nn.index), function(i) {
    neighbor_indices <- knn_result$nn.index[i, ]
    neighbor_deltas <- train_resp[neighbor_indices]
    
    # Calculate weights *for the neighbors* based on their delta_bathy
    neighbor_weights <- calculate_sample_weights(neighbor_deltas)
    
    # Return the weighted mean
    return(weighted.mean(neighbor_deltas, neighbor_weights, na.rm = TRUE))
  })
  pred_avg[!is.finite(pred_avg)] <- NA_real_
  # --- END ENHANCEMENT ---
  
  pred_clean[, SHAP_KNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), SHAP_KNN_predicted_change := i.SHAP_KNN_predicted_change]
  
  message("  - INFO: Assigned SHAP-Weighted KNN predictions to ", sum(!is.na(prediction_data$SHAP_KNN_predicted_change)), " rows.")
  return(prediction_data)
}


#' --- 7. NEW (from V4): Apply Bootstrap KNN Adjustment (V6.8 - Robust KNN) ---
apply_bootstrap_knn_adjustment <- function(prediction_data, boot_summary_data, k = 15) {
  cat("  - INFO: Applying Bootstrap KNN adjustment (V4 method)...\n")
  setDT(prediction_data); setDT(boot_summary_data)
  
  # --- NEW: Use a core set of variables for robust matching ---
  potential_env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity", 
                          "starting_bpi_broad", "starting_bpi_fine")
  
  env_vars <- intersect(potential_env_vars, names(prediction_data))
  env_vars <- intersect(env_vars, names(boot_summary_data)) # Ensure they exist in boot summary data
  
  if (length(env_vars) == 0) {
    warning("No common env variables for Bootstrap KNN. Skipping.")
    prediction_data[, BootstrapKNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching using variables:", paste(env_vars, collapse=", "), "\n")
  
  response_var <- "mean_predicted_change" # This is the target
  
  cols_for_pred_cases <- env_vars
  for(col in cols_for_pred_cases) {
    if(!is.numeric(prediction_data[[col]])) {
      prediction_data[, (col) := as.numeric(get(col))]
    }
  }
  
  cols_for_train_cases <- c(env_vars, response_var)
  for(col in cols_for_train_cases) {
    if(!is.numeric(boot_summary_data[[col]])) {
      boot_summary_data[, (col) := as.numeric(get(col))]
    }
  }
  
  keep_rows_pred <- prediction_data[, complete.cases(.SD), .SDcols = cols_for_pred_cases]
  pred_clean <- prediction_data[keep_rows_pred]
  
  keep_rows_train <- boot_summary_data[, complete.cases(.SD), .SDcols = cols_for_train_cases]
  train_clean <- boot_summary_data[keep_rows_train]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases for Bootstrap KNN. Skipping."); prediction_data[, BootstrapKNN_predicted_change := NA_real_]; return(prediction_data) }
  if (nrow(train_clean) < k) {
    message("    - WARN: Not enough training data (", nrow(train_clean), ") for Bootstrap KNN (k=", k, "). Skipping.");
    prediction_data[, BootstrapKNN_predicted_change := NA_real_];
    return(prediction_data)
  }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]] # This is 'mean_predicted_change'
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  # Use distance-weighted mean for this method
  dist_weights <- 1 / (knn_result$nn.dist + 1e-6)
  sum_dist_weights <- rowSums(dist_weights, na.rm = TRUE)
  pred_avg <- rowSums(matrix(train_resp[knn_result$nn.index], nrow = nrow(knn_result$nn.index)) * dist_weights, na.rm = TRUE) / sum_dist_weights
  pred_avg[sum_dist_weights == 0] <- NA_real_
  
  pred_clean[, BootstrapKNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), BootstrapKNN_predicted_change := i.BootstrapKNN_predicted_change]
  
  message("  - INFO: Assigned Bootstrap KNN predictions to ", sum(!is.na(prediction_data$BootstrapKNN_predicted_change)), " rows.")
  return(prediction_data)
}


#' --- 8. Compare Methods (V6.8 - 4 methods) ---
compare_prediction_methods_v6 <- function(prediction_data, tile_id, year_pair, processing_mode) {
  message("  - INFO: Comparing V6 prediction methods...")
  setDT(prediction_data)
  
  pred_cols <- c("XGB_predicted_change", "pdp_adjusted_change", "TrainingKNN_predicted_change", 
                 "SHAP_KNN_predicted_change", "BootstrapKNN_predicted_change")
  for(col in pred_cols){
    if(!col %in% names(prediction_data)) prediction_data[, (col) := NA_real_]
  }
  
  # --- NEW Hybrid Logic ---
  # Primary blend: XGB Ensemble and SHAP-KNN
  hybrid_cols_primary <- c("XGB_predicted_change", "SHAP_KNN_predicted_change")
  prediction_data[, hybrid_change := rowMeans(.SD, na.rm = TRUE), .SDcols = hybrid_cols_primary]
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  # Fallback blend: PDP and simple Training KNN
  hybrid_cols_fallback <- c("pdp_adjusted_change", "TrainingKNN_predicted_change")
  prediction_data[is.na(hybrid_change), hybrid_change := rowMeans(.SD, na.rm = TRUE),
                  .SDcols = hybrid_cols_fallback]
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  # Final fallback: V4 Bootstrap KNN
  prediction_data[is.na(hybrid_change), hybrid_change := BootstrapKNN_predicted_change]
  
  cor_pdp_knn <- if(sum(!is.na(prediction_data$pdp_adjusted_change) & !is.na(prediction_data$TrainingKNN_predicted_change)) > 2) {
    cor(prediction_data$pdp_adjusted_change, prediction_data$TrainingKNN_predicted_change, use="complete.obs")
  } else { NA_real_ }
  cor_xgb_shapknn <- if(sum(!is.na(prediction_data$XGB_predicted_change) & !is.na(prediction_data$SHAP_KNN_predicted_change)) > 2) {
    cor(prediction_data$XGB_predicted_change, prediction_data$SHAP_KNN_predicted_change, use="complete.obs")
  } else { NA_real_ }
  
  comparison_log <- data.table(Tile_ID = tile_id, Year_Pair = year_pair, Total_Points = nrow(prediction_data),
                               PDP_TrainKNN_Cor = cor_pdp_knn,
                               XGB_SHAPKNN_Cor = cor_xgb_shapknn)
  
  performance_log <- NULL
  # Use 'delta_bathy' for validation
  if (processing_mode == 'direct_model' && "delta_bathy" %in% names(prediction_data) && any(!is.na(prediction_data$delta_bathy))) {
    performance_log <- data.table(
      Tile_ID = tile_id, Year_Pair = year_pair,
      N_Actual = sum(!is.na(prediction_data$delta_bathy)),
      RMSE_XGB_Ens = rmse(prediction_data$delta_bathy, prediction_data$XGB_predicted_change),
      RMSE_SHAP_KNN = rmse(prediction_data$delta_bathy, prediction_data$SHAP_KNN_predicted_change),
      RMSE_Boot_KNN = rmse(prediction_data$delta_bathy, prediction_data$BootstrapKNN_predicted_change),
      RMSE_Hybrid = rmse(prediction_data$delta_bathy, prediction_data$hybrid_change),
      MAE_XGB_Ens = mae(prediction_data$delta_bathy, prediction_data$XGB_predicted_change),
      MAE_SHAP_KNN = mae(prediction_data$delta_bathy, prediction_data$SHAP_KNN_predicted_change),
      MAE_Boot_KNN = mae(prediction_data$delta_bathy, prediction_data$BootstrapKNN_predicted_change),
      MAE_Hybrid = mae(prediction_data$delta_bathy, prediction_data$hybrid_change)
    )
  }
  
  message("  - INFO: Hybrid change calculated. XGB/SHAP-KNN Cor:", round(cor_xgb_shapknn, 3))
  return(list(data = prediction_data, comparison_log = comparison_log, performance_log = performance_log))
}

# --- 9. save_final_predictions (V6.8 - suffix update) ---
save_final_predictions <- function(prediction_data, output_dir, tile_id, year_pair) {
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  pred_depth_col <- paste0("pred_", end_year, "_depth")
  start_bathy_col <- "starting_bathy"
  if (!start_bathy_col %in% names(prediction_data) || !"hybrid_change" %in% names(prediction_data)) {
    stop("Cannot save final predictions: 'starting_bathy' or 'hybrid_change' column is missing.")
  }
  if (nrow(prediction_data[is.na(get(start_bathy_col))]) > 0) {
    message("  - NOTE: Some 'starting_bathy' values are NA. Final predicted depth may also be NA.")
  }
  prediction_data[, (pred_depth_col) := get(start_bathy_col) + hybrid_change]
  out_file_dir <- file.path(output_dir, tile_id); if (!dir.exists(out_file_dir)) dir.create(out_file_dir, recursive = TRUE)
  out_file <- file.path(out_file_dir, paste0(tile_id, "_prediction_final_", year_pair, "_v6.8.fst")) # Suffix update
  write_fst(prediction_data, out_file)
  message("  - INFO: Final FST prediction file saved to: ", basename(out_file))
  return(prediction_data)
}

# --- 10. save_component_rasters (V6.8 - new method) ---
save_component_rasters <- function(prediction_data, output_dir, tile_id, year_pair, crs_obj) {
  out_raster_dir <- file.path(output_dir, tile_id)
  if (!dir.exists(out_raster_dir)) dir.create(out_raster_dir, recursive = TRUE)
  
  save_raster <- function(data, col_name, file_suffix) {
    if (col_name %in% names(data)) {
      valid_rows <- data[!is.na(get(col_name)) & is.finite(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(valid_rows) > 0) {
        tryCatch({
          r <- raster::rasterFromXYZ(valid_rows, crs = crs_obj)
          out_path <- file.path(out_raster_dir, paste0(tile_id, "_", file_suffix, "_", year_pair, ".tif"))
          raster::writeRaster(r, out_path, format = "GTiff", overwrite = TRUE)
        }, error = function(e){ cat("    - ERROR saving raster", file_suffix, ":", conditionMessage(e), "\n") })
      } else { cat("    - WARN: No valid finite data for raster", file_suffix, "\n") }
    } else { cat("    - WARN: Column missing for raster", file_suffix, "\n") }
  }
  
  save_raster(prediction_data, "hybrid_change", "Hybrid_predicted_change")
  save_raster(prediction_data, "pdp_adjusted_change", "PDP_predicted_change")
  save_raster(prediction_data, "TrainingKNN_predicted_change", "TrainingKNN_predicted_change")
  save_raster(prediction_data, "XGB_predicted_change", "XGB_Ens_predicted_change")
  save_raster(prediction_data, "SHAP_KNN_predicted_change", "SHAP_KNN_predicted_change")
  save_raster(prediction_data, "BootstrapKNN_predicted_change", "BootstrapKNN_predicted_change") # NEW
  save_raster(prediction_data, "Uncertainty_SD", "Uncertainty_SD")
}

# --- 11. build_all_tile_maps (V6.8 - no change) ---
build_all_tile_maps <- function(training_dir, prediction_dir, year_pairs) {
  message("  - INFO: Building year-specific geographic maps (V6.8)...")
  training_footprint_gpkg <- file.path(training_dir, "intersecting_sub_grids_UTM.gpkg")
  prediction_footprint_gpkg <- file.path(prediction_dir, "intersecting_sub_grids_UTM.gpkg")
  if(!all(file.exists(training_footprint_gpkg, prediction_footprint_gpkg))) stop("Footprint GPKG files not found.")
  
  train_sf_full <- sf::st_read(training_footprint_gpkg, quiet = TRUE)
  pred_sf <- sf::st_read(prediction_footprint_gpkg, quiet = TRUE)
  id_col <- "tile_id"
  if(!(id_col %in% names(train_sf_full)) || !(id_col %in% names(pred_sf))) stop("'tile_id' column not found in GPKG files.")
  
  suppressWarnings({ pred_centroids <- sf::st_centroid(pred_sf) })
  
  all_maps <- list()
  for (yp in year_pairs) {
    model_file_pattern <- paste0("^model_boot_\\d+_", yp, "\\.rds$")
    potential_tile_dirs <- list.dirs(training_dir, full.names = FALSE, recursive = FALSE)
    valid_training_tiles <- potential_tile_dirs[!grepl("diagnostic_plots|prediction_logs|\\.", potential_tile_dirs)]
    
    valid_source_tiles <- character()
    for (tile in valid_training_tiles) {
      model_sub_dir <- file.path(training_dir, tile, "bootstrap_models")
      if (dir.exists(model_sub_dir) && length(list.files(model_sub_dir, pattern = model_file_pattern)) > 0) {
        valid_source_tiles <- c(valid_source_tiles, tile)
      }
    }
    
    if (length(valid_source_tiles) == 0) {
      cat("    - WARN: No training tiles with models found for year-pair: ", yp, "\n")
      all_maps[[yp]] <- NULL; next
    }
    
    train_sf_year_specific <- train_sf_full[train_sf_full[[id_col]] %in% valid_source_tiles, ]
    if(nrow(train_sf_year_specific) == 0) { all_maps[[yp]] <- NULL; next }
    
    suppressWarnings({ train_centroids_year_specific <- sf::st_centroid(train_sf_year_specific) })
    if(nrow(train_centroids_year_specific) == 0){ all_maps[[yp]] <- NULL; next }
    
    nearest_indices <- sf::st_nearest_feature(pred_centroids, train_centroids_year_specific)
    
    tile_map <- train_sf_year_specific[[id_col]][nearest_indices]
    names(tile_map) <- pred_sf[[id_col]]
    all_maps[[as.character(yp)]] <- tile_map
  }
  return(all_maps)
}

# --- 12. run_cross_validation (V6.8 - new method) ---
run_cross_validation <- function(processed_data, tile_id, year_pair, training_dir) {
  message("  - INFO: Running cross-validation for tile: ", tile_id)
  
  # 'processed_data' (result$data) should now have 'delta_bathy' merged in
  if (!"delta_bathy" %in% names(processed_data) || all(is.na(processed_data$delta_bathy))) {
    message("  - WARN: No 'delta_bathy' (actual change) data found in merged results. Skipping CV.")
    return(NULL)
  }
  
  valid_comp_data <- processed_data[is.finite(delta_bathy)]
  if(nrow(valid_comp_data) < 2) {
    message("  - WARN: Not enough valid comparison points for CV.")
    return(NULL)
  }
  
  cv_log <- data.table(
    Tile_ID = tile_id, Year_Pair = year_pair,
    N_Valid = nrow(valid_comp_data),
    MAE_Hybrid = mae(valid_comp_data$delta_bathy, valid_comp_data$hybrid_change),
    RMSE_Hybrid = rmse(valid_comp_data$delta_bathy, valid_comp_data$hybrid_change),
    MAE_XGB_Ens = mae(valid_comp_data$delta_bathy, valid_comp_data$XGB_predicted_change),
    RMSE_XGB_Ens = rmse(valid_comp_data$delta_bathy, valid_comp_data$XGB_predicted_change),
    MAE_SHAP_KNN = mae(valid_comp_data$delta_bathy, valid_comp_data$SHAP_KNN_predicted_change),
    RMSE_SHAP_KNN = rmse(valid_comp_data$delta_bathy, valid_comp_data$SHAP_KNN_predicted_change),
    MAE_Train_KNN = mae(valid_comp_data$delta_bathy, valid_comp_data$TrainingKNN_predicted_change),
    RMSE_Train_KNN = rmse(valid_comp_data$delta_bathy, valid_comp_data$TrainingKNN_predicted_change),
    MAE_Boot_KNN = mae(valid_comp_data$delta_bathy, valid_comp_data$BootstrapKNN_predicted_change), # NEW
    RMSE_Boot_KNN = rmse(valid_comp_data$delta_bathy, valid_comp_data$BootstrapKNN_predicted_change), # NEW
    MAE_PDP = mae(valid_comp_data$delta_bathy, valid_comp_data$pdp_adjusted_change),
    RMSE_PDP = rmse(valid_comp_data$delta_bathy, valid_comp_data$pdp_adjusted_change)
  )
  return(cv_log)
}

# --- 13. save_summary_plot (V6.8 - 3x3 layout & delta_bathy fix) ---
save_summary_plot <- function(processed_data, tile_id, year_pair, prediction_dir, crs_obj, training_data_path) {
  message("  - INFO: Generating summary plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  library(rasterVis); library(gridExtra); library(ggplot2); library(cmocean); library(raster)
  
  setorderv(processed_data, c("Y", "X"), c(-1, 1))
  plot_list <- list()
  plot_colors <- cmocean('deep')(100)
  
  dt_to_raster <- function(dt, col_name, crs) {
    if (col_name %in% names(dt) && any(is.finite(dt[[col_name]]))) { # Check for finite
      df <- dt[is.finite(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(df) > 0) return(raster::rasterFromXYZ(df, crs = crs))
    }
    return(NULL)
  }
  
  # --- NEW: Load full training data to plot complete delta_bathy ---
  actual_change_raster <- NULL
  if (file.exists(training_data_path)) {
    training_data_full <- read_fst(training_data_path, as.data.table = TRUE)
    if ("delta_bathy" %in% names(training_data_full)) {
      actual_change_raster <- dt_to_raster(training_data_full, "delta_bathy", crs_obj)
      message("  - INFO: Loaded full 'delta_bathy' from training file for plot.")
    }
  }
  if (is.null(actual_change_raster)) {
    message("  - WARN: Could not load full 'delta_bathy'. Plot will use merged (sparse) data.")
    actual_change_raster <- dt_to_raster(processed_data, "delta_bathy", crs_obj) # Fallback
  }
  # --- END NEW ---
  
  plot_list[['xgb']] <- dt_to_raster(processed_data, "XGB_predicted_change", crs_obj)
  plot_list[['shap_knn']] <- dt_to_raster(processed_data, "SHAP_KNN_predicted_change", crs_obj)
  plot_list[['knn']] <- dt_to_raster(processed_data, "TrainingKNN_predicted_change", crs_obj)
  plot_list[['boot_knn']] <- dt_to_raster(processed_data, "BootstrapKNN_predicted_change", crs_obj) # NEW
  plot_list[['pdp']] <- dt_to_raster(processed_data, "pdp_adjusted_change", crs_obj)
  plot_list[['hybrid']] <- dt_to_raster(processed_data, "hybrid_change", crs_obj)
  plot_list[["start_bathy"]] <- dt_to_raster(processed_data, "starting_bathy", crs_obj)
  plot_list[["survey_date"]] <- dt_to_raster(processed_data, "starting_survey_date", crs_obj)
  plot_list[['actual_change']] <- actual_change_raster # Use the full raster
  
  p1 <- if(!is.null(plot_list$xgb)) levelplot(plot_list$xgb, main="XGB Ens. Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("XGB Ens. (NA/No Data)")
  p2 <- if(!is.null(plot_list$shap_knn)) levelplot(plot_list$shap_knn, main="SHAP-KNN Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("SHAP-KNN (NA/No Data)")
  p3 <- if(!is.null(plot_list$knn)) levelplot(plot_list$knn, main="TrainKNN Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("TrainKNN (NA/No Data)")
  p4 <- if(!is.null(plot_list$boot_knn)) levelplot(plot_list$boot_knn, main="BootKNN Change (V4)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("BootKNN (NA/No Data)") # NEW
  p5 <- if(!is.null(plot_list$pdp)) levelplot(plot_list$pdp, main="PDP Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("PDP (NA/No Data)")
  p6 <- if(!is.null(plot_list$hybrid)) levelplot(plot_list$hybrid, main="Final Hybrid Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Hybrid (NA/No Data)")
  p7 <- if(!is.null(plot_list$start_bathy)) levelplot(plot_list$start_bathy, main="Start Bathy (t)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Start Bathy (NA/No Data)")
  p8 <- if(!is.null(plot_list$survey_date)) levelplot(plot_list$survey_date, main="Start Survey Date", margin=FALSE, col.regions=terrain.colors) else ggplot() + theme_void() + ggtitle("Survey (NA/No Data)")
  p9 <- if(!is.null(plot_list$actual_change)) levelplot(plot_list$actual_change, main="Actual Change (delta_bathy)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Actual Change (NA/No Data)")
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_summary_plot_", year_pair, "_v6.8.pdf")) # Suffix update
  
  pdf(plot_out_file, width = 18, height = 18) # Make plot bigger (3x3)
  grid.arrange(grobs = list(p1, p2, p3, p4, p5, p6, p7, p8, p9), ncol = 3, nrow = 3)
  dev.off()
  
  message("  - INFO: Summary plot saved to: ", basename(plot_out_file))
}


# --- 14. save_summary_plot_scaled (V6.8 - 3x3 layout & delta_bathy fix) ---
save_summary_plot_scaled <- function(processed_data, tile_id, year_pair, prediction_dir, crs_obj, training_data_path) {
  message("  - INFO: Generating SCALED summary plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  library(rasterVis); library(gridExtra); library(ggplot2); library(cmocean); library(raster)
  
  setorderv(processed_data, c("Y", "X"), c(-1, 1))
  
  dt_to_raster <- function(dt, col_name, crs) {
    if (col_name %in% names(dt) && any(is.finite(dt[[col_name]]))) {
      df <- dt[is.finite(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(df) > 0) return(raster::rasterFromXYZ(df, crs = crs))
    }
    return(NULL)
  }
  
  # --- NEW: Load full training data to plot complete delta_bathy AND get scale ---
  actual_change_raster <- NULL
  zlim <- c(-6, 6) # Default fallback
  
  if (file.exists(training_data_path)) {
    training_data_full <- read_fst(training_data_path, as.data.table = TRUE)
    if ("delta_bathy" %in% names(training_data_full)) {
      actual_change_raster <- dt_to_raster(training_data_full, "delta_bathy", crs_obj)
      zlim_data <- training_data_full[is.finite(delta_bathy), delta_bathy]
      if (length(zlim_data) > 1) {
        zlim <- range(zlim_data, na.rm = TRUE)
      }
      message("  - INFO: Scaled plot range set from 'delta_bathy': ", round(zlim[1],2), " to ", round(zlim[2],2))
    }
  }
  if (is.null(actual_change_raster)) {
    message("  - WARN: Could not load full 'delta_bathy' for scale. Using merged data.")
    actual_change_raster <- dt_to_raster(processed_data, "delta_bathy", crs_obj) # Fallback
    zlim_data <- processed_data[is.finite(delta_bathy), delta_bathy]
    if (length(zlim_data) > 1) {
      zlim <- range(zlim_data, na.rm = TRUE)
    }
  }
  if(!all(is.finite(zlim))) zlim <- c(-6, 6)
  # --- END NEW ---
  
  at <- seq(zlim[1], zlim[2], length.out = 99)
  plot_colors <- cmocean('deep')(100)
  
  change_rasters <- list()
  change_cols <- c("XGB_predicted_change", "SHAP_KNN_predicted_change", "TrainingKNN_predicted_change", 
                   "BootstrapKNN_predicted_change", "pdp_adjusted_change", "hybrid_change")
  for(col in change_cols) {
    change_rasters[[col]] <- dt_to_raster(processed_data, col, crs_obj)
  }
  start_bathy_raster <- dt_to_raster(processed_data, "starting_bathy", crs_obj)
  survey_raster <- dt_to_raster(processed_data, "starting_survey_date", crs_obj)
  
  # Add the (now loaded) actual_change_raster
  change_rasters[['actual_change']] <- actual_change_raster
  
  p1 <- if(!is.null(change_rasters$XGB_predicted_change)) levelplot(change_rasters$XGB_predicted_change, main="XGB Ens. Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("XGB Ens. (NA/No Data)")
  p2 <- if(!is.null(change_rasters$SHAP_KNN_predicted_change)) levelplot(change_rasters$SHAP_KNN_predicted_change, main="SHAP-KNN Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("SHAP-KNN (NA/No Data)")
  p3 <- if(!is.null(change_rasters$TrainingKNN_predicted_change)) levelplot(change_rasters$TrainingKNN_predicted_change, main="TrainKNN Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("TrainKNN (NA/No Data)")
  p4 <- if(!is.null(change_rasters$BootstrapKNN_predicted_change)) levelplot(change_rasters$BootstrapKNN_predicted_change, main="BootKNN Change (V4)", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("BootKNN (NA/No Data)") # NEW
  p5 <- if(!is.null(change_rasters$pdp_adjusted_change)) levelplot(change_rasters$pdp_adjusted_change, main="PDP Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("PDP (NA/No Data)")
  p6 <- if(!is.null(change_rasters$hybrid_change)) levelplot(change_rasters$hybrid_change, main="Final Hybrid Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Hybrid (NA/No Data)")
  p7 <- if(!is.null(start_bathy_raster)) levelplot(start_bathy_raster, main="Start Bathy (t)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Start Bathy (NA/No Data)")
  p8 <- if(!is.null(survey_raster)) levelplot(survey_raster, main="Start Survey Date", margin=FALSE, col.regions=terrain.colors) else ggplot() + theme_void()+ ggtitle("Survey (NA/No Data)")
  p9 <- if(!is.null(change_rasters$actual_change)) levelplot(change_rasters$actual_change, main="Actual Change (delta_bathy)", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Actual Change (NA/No Data)")
  
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_summary_plot_SCALED_", year_pair, "_v6.8.pdf")) # Suffix update
  
  pdf(plot_out_file, width = 18, height = 18) # Make plot bigger (3x3)
  grid.arrange(grobs = list(p1, p2, p3, p4, p5, p6, p7, p8, p9), ncol = 3, nrow = 3)
  dev.off()
  
  message("  - INFO: SCALED summary plot saved to: ", basename(plot_out_file))
}


# --- 15. save_density_plot (V6.8 - 3x3 layout & delta_bathy fix) ---
save_density_plot <- function(processed_data, tile_id, year_pair, prediction_dir, training_data_path) {
  message("  - INFO: Generating density plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  library(ggplot2); library(data.table); library(gridExtra)
  
  cols_to_plot <- c("XGB_predicted_change", "pdp_adjusted_change", "TrainingKNN_predicted_change",
                    "SHAP_KNN_predicted_change", "BootstrapKNN_predicted_change", "hybrid_change",
                    "starting_bathy")
  cols_to_plot <- intersect(cols_to_plot, names(processed_data))
  if (length(cols_to_plot) < 2) { message("  - WARN: Not enough columns for density plot."); return(NULL) }
  
  plot_data_pred <- melt(processed_data[, ..cols_to_plot], measure.vars = cols_to_plot, na.rm = TRUE)
  plot_data_pred <- plot_data_pred[is.finite(value)]
  
  # --- NEW: Load full training data to plot complete delta_bathy density ---
  plot_data_actual <- NULL
  if (file.exists(training_data_path)) {
    training_data_full <- read_fst(training_data_path, as.data.table = TRUE, columns = "delta_bathy")
    if ("delta_bathy" %in% names(training_data_full)) {
      plot_data_actual <- training_data_full[is.finite(delta_bathy), .(variable = "delta_bathy", value = delta_bathy)]
      message("  - INFO: Loaded", nrow(plot_data_actual), "finite 'delta_bathy' values for density plot.")
    }
  }
  if (is.null(plot_data_actual)) {
    message("  - WARN: Could not load 'delta_bathy' for density. Using merged (sparse) data.")
    plot_data_actual <- processed_data[is.finite(delta_bathy), .(variable = "delta_bathy", value = delta_bathy)] # Fallback
  }
  
  plot_data <- rbindlist(list(plot_data_pred, plot_data_actual), use.names = TRUE, fill = TRUE)
  if (nrow(plot_data) == 0) { message("  - WARN: No finite data available for density plot."); return(NULL)}
  # --- END NEW ---
  
  plot_data[, variable := factor(variable,
                                 levels = c("XGB_predicted_change", "SHAP_KNN_predicted_change", "TrainingKNN_predicted_change",
                                            "BootstrapKNN_predicted_change", "pdp_adjusted_change", "hybrid_change",
                                            "starting_bathy", "delta_bathy"),
                                 labels = c("XGB Ens Pred Change", "SHAP-KNN Change", "TrainKNN Pred Change",
                                            "BootKNN Pred Change", "PDP Pred Change", "Final Hybrid Change",
                                            "Start Bathy (t)", "Actual Change (delta_bathy)"))]
  
  plot_data <- plot_data[!is.na(variable)]
  
  change_data <- plot_data[!variable %in% c("Start Bathy (t)")]
  
  change_x_range <- if ("Actual Change (delta_bathy)" %in% change_data$variable && nrow(change_data[variable == "Actual Change (delta_bathy)"]) > 0) {
    range(change_data[variable == "Actual Change (delta_bathy)"]$value, na.rm = TRUE)
  } else if (nrow(change_data) > 0) {
    range(change_data$value, na.rm=TRUE)
  } else { c(-6, 6) }
  if(!all(is.finite(change_x_range))) change_x_range <- c(-6, 6)
  
  
  plot_list <- list()
  defined_levels <- levels(plot_data$variable)
  
  for(var_name in defined_levels) {
    if(var_name %in% unique(plot_data$variable)) {
      sub_data <- plot_data[variable == var_name]
      p <- ggplot(sub_data, aes(x = value)) +
        geom_density(fill = "skyblue", alpha = 0.7) +
        labs(title = var_name, x = "Value (m)", y = "Density") +
        theme_minimal() + theme(legend.position = "none", axis.text.x = element_text(angle=45, hjust=1))
      if (var_name != "Start Bathy (t)") {
        p <- p + coord_cartesian(xlim = change_x_range)
      }
      plot_list[[var_name]] <- p
    } else {
      plot_list[[var_name]] <- ggplot() + theme_void() + ggtitle(paste(var_name, "(No Data)"))
    }
  }
  
  plot_list[["placeholder"]] <- ggplot() + theme_void()
  
  final_plot_order <- c("XGB Ens Pred Change", "SHAP-KNN Change", "TrainKNN Pred Change",
                        "BootKNN Pred Change", "PDP Pred Change", "Final Hybrid Change",
                        "Start Bathy (t)", "Actual Change (delta_bathy)", "placeholder")
  
  ordered_grobs <- lapply(final_plot_order, function(name) {
    if (name %in% names(plot_list)) plot_list[[name]] else ggplot() + theme_void() + ggtitle(paste(name, "(NA)"))
  })
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_density_plot_", year_pair, "_v6.8.pdf")) # Suffix update
  
  pdf(plot_out_file, width = 12, height = 12) # 3x3 layout
  grid.arrange(grobs = ordered_grobs, ncol = 3, nrow = 3)
  dev.off()
  
  message("  - INFO: Density plot saved to: ", basename(plot_out_file))
}

# ==============================================================================
#   Parallel Prediction Test Wrapper (V6.8 - Stripped Down)
# ==============================================================================
run_prediction_test <- function(prediction_tiles, year_pairs, crs_obj, training_dir, prediction_dir, output_dir, master_predictor_list_path, n_boot_models_to_use = 10) {
  
  message("\nStarting parallel prediction test run (V6.8 Ensemble)...")
  
  all_tile_maps <- build_all_tile_maps(training_dir, prediction_dir, year_pairs)
  num_cores <- max(1, floor(detectCores() / 2))
  cat("  - INFO: Setting up parallel cluster with", num_cores, "cores.\n")
  cl <- makeCluster(num_cores); registerDoParallel(cl); on.exit({ stopCluster(cl) }, add = TRUE)
  task_grid <- expand.grid(tile_id = prediction_tiles, year_pair = year_pairs, stringsAsFactors = FALSE)
  
  message("\n--- Stage 1: Processing data in parallel... ---")
  
  functions_to_export <- c(
    "process_tile", "predict_elevation_change_ensemble",
    "align_predictors", "match_pdp_conditions",
    "apply_training_knn_adjustment",
    "apply_shap_weighted_knn_adjustment",
    "apply_bootstrap_knn_adjustment", # NEW
    "compare_prediction_methods_v6",
    # "run_cross_validation", # --- REMOVED ---
    "calculate_sample_weights", # NEW
    "mae", "rmse"
  )
  
  results_list <- foreach(
    i = 1:nrow(task_grid),
    .export = functions_to_export,
    .packages = c("data.table", "fst", "xgboost", "FNN", "stringr", "Metrics")
  ) %dopar% {
    library(dplyr); library(sf); library(raster);
    
    current_task <- task_grid[i, ]
    
    process_tile_result <- process_tile(
      tile_id        = current_task$tile_id,
      year_pair      = current_task$year_pair,
      training_dir   = training_dir,
      prediction_dir = prediction_dir,
      output_dir     = output_dir,
      all_tile_maps  = all_tile_maps,
      master_predictor_list_path = master_predictor_list_path,
      n_boot_models_to_use = n_boot_models_to_use
    )
    
    return(process_tile_result)
  }
  
  message("\n--- Stage 2: Saving results and plots sequentially... ---")
  
  successful_results <- Filter(function(res) is.list(res) && isTRUE(res$success), results_list)
  failed_results <- Filter(function(res) is.list(res) && isFALSE(res$success), results_list)
  if(length(failed_results) > 0) {
    cat("\n--- WARNING: Some tiles failed processing. See error log for details. ---\n")
  }
  
  all_comparison_logs <- rbindlist(lapply(successful_results, `[[`, "comparison_log"), fill = TRUE)
  all_performance_logs <- rbindlist(lapply(successful_results, `[[`, "performance_log"), fill = TRUE)
  
  # --- NEW: Save the model feature log (FLATTENED) ---
  all_model_feature_logs_list <- lapply(successful_results, `[[`, "model_feature_log")
  # Flatten the list of lists of data.tables
  all_model_feature_logs_flat <- unlist(all_model_feature_logs_list, recursive = FALSE) 
  all_model_feature_logs_flat <- Filter(function(x) is.data.table(x) && nrow(x) > 0, all_model_feature_logs_flat)
  
  if(length(all_model_feature_logs_flat) > 0) {
    all_model_feature_logs <- rbindlist(all_model_feature_logs_flat, fill = TRUE)
    if(nrow(all_model_feature_logs) > 0) fwrite(all_model_feature_logs, file.path(output_dir, "model_feature_usage_log_v6.8.csv"))
  } else {
    cat("  - INFO: No model feature logs were generated.\n")
  }
  # --- END FIX ---
  
  if(nrow(all_comparison_logs) > 0) fwrite(all_comparison_logs, file.path(output_dir, "prediction_method_comparison_log_v6.8.csv"))
  if(nrow(all_performance_logs) > 0) fwrite(all_performance_logs, file.path(output_dir, "model_performance_log_v6.8.csv"))
  
  # --- REMOVED CV LOG ---
  # all_cv_logs <- list()
  library(rasterVis); library(gridExtra); library(ggplot2); library(cmocean); library(raster)
  
  for (result in successful_results) {
    cat("\n--- Saving outputs for Tile:", result$tile_id, "| Year Pair:", result$year_pair, "---\n")
    
    processed_data <- save_final_predictions(result$data, output_dir, result$tile_id, result$year_pair)
    
    if (!is.null(crs_obj)) {
      save_component_rasters(processed_data, output_dir, result$tile_id, result$year_pair, crs_obj)
      
      # --- NEW: Pass training_data_path to plotting functions ---
      save_summary_plot(processed_data, result$tile_id, result$year_pair, prediction_dir, crs_obj, result$training_data_path)
      save_summary_plot_scaled(processed_data, result$tile_id, result$year_pair, prediction_dir, crs_obj, result$training_data_path)
      
      # --- REMOVED DENSITY PLOT and CV ---
      # save_density_plot(processed_data, result$tile_id, result$year_pair, prediction_dir, result$training_data_path)
      # if (result$processing_mode == 'direct_model') {
      #   cv_log <- run_cross_validation(processed_data, result$tile_id, result$year_pair, training_dir)
      #   if(!is.null(cv_log)) all_cv_logs[[length(all_cv_logs) + 1]] <- cv_log
      # }
    }
  }
  
  # if(length(all_cv_logs) > 0) {
  #   fwrite(rbindlist(all_cv_logs, fill = TRUE), file.path(output_dir, "cross_validation_log_v6.8.csv"))
  #   message("  - INFO: Cross-validation log saved.")
  # }
  
  message("\n✅ Parallel prediction test run V6.8 complete.")
}



# ==============================================================================
#
#   XGBoost Prediction Function Set (V6.9 - Model EXTREMES ORIENTED  -best utilizing large boostrap array data)
#
# ==============================================================================

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis)
library(gridExtra)
library(ggplot2)
library(FNN)
library(foreach)
library(doParallel)
library(Metrics)
library(cmocean)
library(stringr)

# --- Helper: Weighted Loss Function (from training) ---
calculate_sample_weights <- function(delta_bathy, alpha = 1.5, epsilon = 1e-6) {
  abs_change <- abs(delta_bathy) + epsilon
  weights <- abs_change^alpha
  weights <- weights / sum(weights, na.rm = TRUE) * length(weights)
  weights[!is.finite(weights)] <- epsilon
  return(weights)
}

# =Monitoring - V6.8 ---
process_tile <- function(tile_id, year_pair, training_dir, prediction_dir, output_dir, all_tile_maps, master_predictor_list_path, n_boot_models_to_use = 10) {
  # --- a. Setup Logging ---
  pred_log_dir <- file.path(output_dir, tile_id, "prediction_logs")
  if (!dir.exists(pred_log_dir)) dir.create(pred_log_dir, recursive = TRUE)
  pred_log_file <- file.path(pred_log_dir, paste0("pred_log_", year_pair, "_v6.8.txt")) # Updated Suffix
  sink(pred_log_file, append = FALSE, split = TRUE); on.exit({ sink() })
  cat("\n--- Starting Tile:", tile_id, "| Year Pair:", year_pair, "(V6.8 Ensemble) ---\n")
  
  # Initialize the log for model features
  model_feature_log <- list()
  
  # --- SCOPING FIX: 'training_data_path' is defined *inside* the tryCatch block ---
  
  tryCatch({
    # --- b. Determine Model Source Tile ---
    tile_map_for_year <- all_tile_maps[[year_pair]]
    if (is.null(tile_map_for_year)) stop("No valid models found for year pair: ", year_pair)
    model_tile_id <- if (tile_id %in% names(tile_map_for_year)) tile_map_for_year[[tile_id]] else tile_id
    processing_mode <- if (model_tile_id == tile_id) "direct_model" else "reference_model"
    cat("  - INFO: Processing mode detected:", processing_mode, "\n")
    if(processing_mode == "reference_model") cat("  - INFO: Using reference model(s) from tile:", model_tile_id, "\n")
    
    # --- NEW: Define training_data_path *after* model_tile_id is known ---
    # Training files are stored inside the tile folder and have the same prefix
    # as the folder name, e.g. BH4S556X_3_2004_2006_long.fst
    training_data_path <- file.path(
      training_dir,
      model_tile_id,
      paste0(model_tile_id, "_", year_pair, "_long.fst")
    )
    
    
    # --- c. NEW: Load and Filter Prediction Data (Land Filter) ---
    cat("Step 1: Loading and filtering prediction data...\n")
    # tile_base = parent tile (BH4S556X)
    tile_base <- sub("_[0-9]+$", "", tile_id)
    
    # prediction files are named: <tile_base>_<year_pair>_prediction_long.fst
    prediction_data_path <- file.path(
      prediction_dir, 
      tile_id,
      paste0(tile_base, "_", year_pair, "_prediction_long.fst")
    )
    
    if (!file.exists(prediction_data_path)) stop(paste("Missing prediction data file:", basename(prediction_data_path)))
    
    prediction_data <- read_fst(prediction_data_path, as.data.table = TRUE)
    cat("  - INFO: Initial prediction data has", nrow(prediction_data), "rows.\n")
    
    if ("bathy_t" %in% names(prediction_data)) {
      # Filter: keep only rows where bathy_t is <= 0 (not land)
      # Use !is.na() to also keep rows where bathy_t might be NA
      prediction_data <- prediction_data[!(bathy_t > 0)]
      cat("  - INFO: Filtered to", nrow(prediction_data), "rows (bathy_t <= 0 or NA).\n")
    } else {
      cat("  - WARN: 'bathy_t' column not found. Cannot filter land values.\n")
    }
    if (nrow(prediction_data) == 0) stop("No data remaining after land filter.")
    
    # --- d. Generate Initial Ensemble Predictions ---
    cat("Step 2: Generating initial predictions using bootstrap ensemble...\n")
    initial_predictions_list <- predict_elevation_change_ensemble(
      prediction_data, # Pass the filtered data.table
      model_tile_id, 
      year_pair, 
      training_dir,
      master_predictor_list_path, 
      n_boot_models_to_use = n_boot_models_to_use
    )
    if (is.null(initial_predictions_list)) stop("Initial ensemble prediction failed.")
    initial_predictions_dt <- initial_predictions_list$summary_dt
    model_feature_log <- initial_predictions_list$model_feature_log # Capture the log
    
    # --- e. Load Adjustment Data ---
    cat("Step 3: Loading adjustment data from source:", model_tile_id, "\n")
    pdp_file <- file.path(training_dir, model_tile_id, paste0("pdp_data_long_", year_pair, ".fst"))
    shap_summary_path <- file.path(training_dir, model_tile_id, paste0("shap_summary_", year_pair, ".fst"))
    # Path for V4 Bootstrap method
    boot_summary_path <- file.path(training_dir, model_tile_id, paste0("bootstraps_summary_df_", year_pair, ".fst"))
    
    required_files <- c(pdp_file, training_data_path, shap_summary_path, boot_summary_path)
    missing_files <- required_files[!file.exists(required_files)]
    if(length(missing_files) > 0){
      stop("Missing required adjustment data file(s): ", paste(basename(missing_files), collapse=", "))
    }
    
    pdp_data <- read_fst(pdp_file, as.data.table = TRUE)
    training_data <- read_fst(training_data_path, as.data.table = TRUE)
    shap_summary <- read_fst(shap_summary_path, as.data.table = TRUE)
    boot_summary_data <- read_fst(boot_summary_path, as.data.table = TRUE)
    
    cat("  - INFO: Loaded", nrow(training_data), "raw training data rows for adjustments.\n")
    
    # --- Align training data ---
    aligned_training_data <- align_predictors(training_data, year_pair, data_type = "training")
    # Align bootstrap summary data (for V4 KNN method)
    aligned_boot_summary_data <- align_predictors(boot_summary_data, year_pair, data_type = "training")
    
    
    # --- f. Apply Hybrid Adjustments ---
    cat("Step 4: Applying hybrid adjustments (PDP, Training KNN, SHAP-KNN, Boot-KNN)...\n")
    
    if(nrow(pdp_data) == 0 || all(is.na(pdp_data$PDP_Value))) {
      cat("  - WARN: PDP data is empty or all NA. Skipping PDP adjustment.\n")
      pdp_enriched <- initial_predictions_dt
      pdp_enriched[, pdp_adjusted_change := NA_real_]
    } else {
      pdp_enriched <- match_pdp_conditions(initial_predictions_dt, pdp_data)
    }
    
    # KNN on *actual change* (delta_bathy)
    train_knn_enriched <- apply_training_knn_adjustment(pdp_enriched, aligned_training_data)
    
    # SHAP-Weighted KNN on *actual change* (delta_bathy)
    if(nrow(shap_summary) == 0 || all(is.na(shap_summary$Overall_Mean_Abs_SHAP))) {
      cat("  - WARN: SHAP summary is empty or all NA. Skipping SHAP-KNN adjustment.\n")
      shap_knn_enriched <- train_knn_enriched
      shap_knn_enriched[, SHAP_KNN_predicted_change := NA_real_]
    } else {
      shap_knn_enriched <- apply_shap_weighted_knn_adjustment(train_knn_enriched, aligned_training_data, shap_summary)
    }
    
    # V4-style KNN on *mean predicted change*
    boot_knn_enriched <- apply_bootstrap_knn_adjustment(shap_knn_enriched, aligned_boot_summary_data)
    
    
    # --- g. Merge Actual Change (delta_bathy) for direct mode ---
    final_data_for_comparison <- copy(boot_knn_enriched)
    if (processing_mode == 'direct_model') {
      cat("  - INFO (Main): Direct model. Merging 'delta_bathy' for plots/validation.\n")
      
      if ("delta_bathy" %in% names(aligned_training_data) && any(!is.na(aligned_training_data$delta_bathy))) {
        truth_data <- aligned_training_data[, .(FID, delta_bathy_actual_truth = delta_bathy)]
        final_data_for_comparison[truth_data, on = "FID", delta_bathy := i.delta_bathy_actual_truth]
        
        n_merged <- sum(!is.na(final_data_for_comparison$delta_bathy))
        cat("  - INFO (Main): Merged actual 'delta_bathy' for", n_merged, "points.\n")
      } else {
        cat("  - WARN (Main): 'delta_bathy' column not found or all NA in training data.\n")
        if(!"delta_bathy" %in% names(final_data_for_comparison)) final_data_for_comparison[, delta_bathy := NA_real_]
      }
      
    } else {
      if(!"delta_bathy" %in% names(final_data_for_comparison)) final_data_for_comparison[, delta_bathy := NA_real_]
    }
    
    # --- h. Compare and Finalize ---
    cat("Step 5: Comparing prediction methods and combining results...\n")
    comparison_results <- compare_prediction_methods_v6(final_data_for_comparison, tile_id, year_pair, processing_mode = processing_mode)
    
    cat("\n--- SUCCESS (Data Processing): Completed Tile:", tile_id, "| Year Pair:", year_pair, "---\n")
    
    return(list(
      data = comparison_results$data,
      comparison_log = comparison_results$comparison_log,
      performance_log = comparison_results$performance_log,
      model_feature_log = model_feature_log, # Pass log back
      training_data_path = training_data_path, # Pass path for plotting
      processing_mode = processing_mode,
      tile_id = tile_id,
      year_pair = year_pair,
      success = TRUE
    ))
    
  }, error = function(e) { # Outer tryCatch
    cat("\n--- FATAL ERROR in process_tile (V6.8) ---\n")
    cat("  - Tile:", tile_id, "| Pair:", year_pair, "\n")
    cat("  - Error Message:", conditionMessage(e), "\n")
    cat("  - Traceback:\n", paste(capture.output(traceback()), collapse="\n"), "\n")
    return(list(success = FALSE, tile_id = tile_id, year_pair = year_pair, error = conditionMessage(e), model_feature_log = model_feature_log))
  })
}

# --- 2. Core Prediction Function (Ensemble) (V6.9 - EXTREME-AWARE) ---
predict_elevation_change_ensemble <- function(
    prediction_data, # Now receives a data.table
    model_tile_id, 
    year_pair, 
    training_dir, 
    master_predictor_list_path, 
    n_boot_models_to_use = 10
) {
  
  # Initialize the feature log
  model_feature_log <- list()
  
  # --- 1. Align prediction data (creates 'starting_...' columns and 'delta_bathy=NA') ---
  aligned_pred_data <- align_predictors(prediction_data, year_pair, data_type = "prediction")
  
  if(!("bathy_t" %in% names(aligned_pred_data))){
    if("starting_bathy" %in% names(aligned_pred_data)) {
      aligned_pred_data[, bathy_t := starting_bathy]
    } else {
      stop("'bathy_t' or 'starting_bathy' column missing after alignment in prediction data.")
    }
  }
  
  model_dir <- file.path(training_dir, model_tile_id, "bootstrap_models")
  model_files <- list.files(model_dir, pattern = paste0("^model_boot_\\d+_", year_pair, "\\.rds$"), full.names = TRUE)
  
  if (length(model_files) == 0) stop("No bootstrap model files found in '", model_dir, "' for year pair ", year_pair)
  
  n_models_available <- length(model_files)
  n_models_to_load <- min(n_boot_models_to_use, n_models_available)
  models_to_load <- if (n_models_to_load < n_models_available) sample(model_files, n_models_to_load) else model_files
  cat("  - INFO: Loading", n_models_to_load, "of", n_models_available, "bootstrap models for prediction.\n")
  
  # --- 2. Load master predictor list (for logging/comparison only) ---
  if (!file.exists(master_predictor_list_path)) stop("Master predictor list file not found.")
  master_predictors_list <- readLines(master_predictor_list_path)
  target_model_features_superset <- sapply(master_predictors_list, function(p) {
    if (p %in% c("hurr_count", "hurr_strength", "tsm")) {
      paste0(p, "_", year_pair)
    } else {
      p
    }
  }, USE.NAMES = FALSE)
  cat("  - INFO: Master list has", length(target_model_features_superset), "potential features.\n")
  
  
  # --- 3. Run Prediction Loop ---
  n_pts <- nrow(aligned_pred_data)
  all_predictions_array_t1 <- array(NA_real_, dim = c(n_pts, 1, n_models_to_load))
  
  for (b in 1:n_models_to_load) {
    model_path <- models_to_load[b]
    xgb_model <- readRDS(model_path)
    
    # --- DYNAMIC FEATURES PER MODEL ---
    model_features <- xgb_model$feature_names
    
    # Log features for this model
    model_feature_log[[length(model_feature_log) + 1]] <- data.table(
      tile_id    = model_tile_id, 
      year_pair  = year_pair, 
      model_file = basename(model_path), 
      feature    = model_features,
      n_features = length(model_features)
    )
    if (b == 1) { # Log this only for the first model
      cat("  - INFO: Model 1 (", basename(model_path), ") uses", length(model_features),
          "features (e.g.,", paste(head(model_features, 3), collapse=", "), "...).\n")
    }
    
    # Check if this model's features are available in the prediction data
    missing_features <- setdiff(model_features, names(aligned_pred_data))
    if (length(missing_features) > 0) {
      cat("  - FATAL: Model", b, "requires features not found in prediction data:",
          paste(missing_features, collapse=", "), "\n")
      stop("Missing required predictor columns in prediction data file for model ", b)
    }
    
    # Create the prediction matrix *using only the features this model expects*
    pred_matrix <- as.matrix(aligned_pred_data[, ..model_features])
    dpredict_model_specific <- xgb.DMatrix(data = pred_matrix, missing = NA)
    
    # Predict using this model-specific DMatrix
    all_predictions_array_t1[, 1, b] <- predict(xgb_model, newdata = dpredict_model_specific)
  }
  
  # ------------------------------------------------------------------
  # NEW: FULL BOOTSTRAP DISTRIBUTION -> MEAN, SD, QUANTILES, EXTREMES
  # ------------------------------------------------------------------
  # Flatten [n_pts x 1 x n_models] -> [n_pts x n_models]
  boot_mat <- if (n_models_to_load == 1) {
    matrix(all_predictions_array_t1[, 1, 1], nrow = n_pts, ncol = 1)
  } else {
    matrix(all_predictions_array_t1[, 1, ], nrow = n_pts, ncol = n_models_to_load)
  }
  
  Mean_Prediction_t1 <- rowMeans(boot_mat, na.rm = TRUE)
  Uncertainty_SD     <- apply(boot_mat, 1, sd, na.rm = TRUE)
  Uncertainty_SD[is.na(Uncertainty_SD)] <- 0
  
  # Quantiles (depth at t1)
  q_probs <- c(0.05, 0.25, 0.5, 0.75, 0.95)
  q_names <- c("q05", "q25", "q50", "q75", "q95")
  q_mat   <- matrix(NA_real_, nrow = n_pts, ncol = length(q_probs))
  
  for (i in seq_len(n_pts)) {
    row_vals <- boot_mat[i, ]
    if (!all(is.na(row_vals))) {
      q_mat[i, ] <- stats::quantile(row_vals, probs = q_probs,
                                    na.rm = TRUE, names = FALSE, type = 7)
    }
  }
  colnames(q_mat) <- q_names
  
  # ------------------------------------------------------------------
  # Build output DT
  # ------------------------------------------------------------------
  summary_dt <- data.table(
    FID     = aligned_pred_data$FID,
    X       = aligned_pred_data$X,
    Y       = aligned_pred_data$Y,
    tile_id = aligned_pred_data$tile_id
  )
  
  # Carry over all starting_* columns (for KNN, PDP, etc.)
  start_cols <- grep("^starting_", names(aligned_pred_data), value = TRUE)
  for (col in start_cols) {
    summary_dt[[col]] <- aligned_pred_data[[col]]
  }
  if (!("bathy_t" %in% names(summary_dt)) && "bathy_t" %in% names(aligned_pred_data)) {
    summary_dt[, bathy_t := aligned_pred_data$bathy_t]
  }
  
  # Central prediction + distribution in depth space
  summary_dt[, `:=`(
    XGB_predicted_bathy_t1 = Mean_Prediction_t1,
    XGB_q05_bathy_t1       = q_mat[, "q05"],
    XGB_q25_bathy_t1       = q_mat[, "q25"],
    XGB_q50_bathy_t1       = q_mat[, "q50"],
    XGB_q75_bathy_t1       = q_mat[, "q75"],
    XGB_q95_bathy_t1       = q_mat[, "q95"],
    Uncertainty_SD         = Uncertainty_SD,
    delta_bathy            = aligned_pred_data$delta_bathy # NA for prediction data
  )]
  
  # Change space (relative to bathy_t)
  summary_dt[, XGB_predicted_change := XGB_predicted_bathy_t1 - bathy_t]
  summary_dt[, XGB_q05_change       := XGB_q05_bathy_t1       - bathy_t]
  summary_dt[, XGB_q25_change       := XGB_q25_bathy_t1       - bathy_t]
  summary_dt[, XGB_q50_change       := XGB_q50_bathy_t1       - bathy_t]
  summary_dt[, XGB_q75_change       := XGB_q75_bathy_t1       - bathy_t]
  summary_dt[, XGB_q95_change       := XGB_q95_bathy_t1       - bathy_t]
  
  # Erosion (negative) & deposition (positive) extremes
  # By convention: q05 ~ strong erosion, q95 ~ strong deposition
  summary_dt[, `:=`(
    XGB_erosion_extreme_change    = XGB_q05_change,
    XGB_deposition_extreme_change = XGB_q95_change,
    XGB_change_spread_q95_q05     = XGB_q95_change - XGB_q05_change
  )]
  
  message("  - INFO: Generated ensemble predictions (mean, SD, quantiles) for ",
          nrow(summary_dt), " points.")
  
  return(list(
    summary_dt             = summary_dt, 
    all_predictions_array_t1 = all_predictions_array_t1,  # still returned
    model_feature_log      = model_feature_log            # Return the log
  ))
}



#' --- 3. Align Predictors (V6.8 - SIMPLIFIED) ---
align_predictors <- function(data_to_align, year_pair, data_type = "prediction") { # Default to prediction
  setDT(data_to_align)
  cat(paste0("  - DIAGNOSTIC (align_predictors): Aligning '", data_type, "' data... "))
  
  # This function now assumes data *already* has '_t' or '_year_pair' suffixes.
  # Its job is to create 'starting_...' columns for KNN/PDP and to define 'delta_bathy'.
  
  # --- 1. Create 'starting_...' columns for KNN/PDP/SHAP ---
  # These are based *directly* on the '..._t' columns
  if ("bathy_t" %in% names(data_to_align)) data_to_align[, starting_bathy := bathy_t]
  if ("slope_t" %in% names(data_to_align)) data_to_align[, starting_slope := slope_t]
  if ("rugosity_t" %in% names(data_to_align)) data_to_align[, starting_rugosity := rugosity_t]
  if ("bpi_broad_t" %in% names(data_to_align)) data_to_align[, starting_bpi_broad := bpi_broad_t]
  if ("bpi_fine_t" %in% names(data_to_align)) data_to_align[, starting_bpi_fine := bpi_fine_t]
  if ("terrain_classification_t" %in% names(data_to_align)) data_to_align[, starting_terrain_class := terrain_classification_t]
  
  # Handle neighborhood stats
  nbh_cols_t <- grep("_(mean3|sd3)_t$", names(data_to_align), value = TRUE)
  for(col in nbh_cols_t){
    new_col_name <- paste0("starting_", sub("_t$", "", col)) # e.g., bathy_sd3_t -> starting_bathy_sd3
    data_to_align[, (new_col_name) := get(col)]
  }
  
  # Handle static layers
  if ("grain_size_layer" %in% names(data_to_align)) data_to_align[, starting_grain_size := grain_size_layer]
  if ("prim_sed_layer" %in% names(data_to_align)) data_to_align[, starting_sed_type := prim_sed_layer]
  if ("survey_end_date" %in% names(data_to_align)) data_to_align[, starting_survey_date := survey_end_date]
  
  # Handle forcing layers (year-pair specific)
  tsm_var_name <- paste0("tsm_", year_pair)
  if (tsm_var_name %in% names(data_to_align)) data_to_align[, starting_tsm := get(tsm_var_name)]
  
  hurr_c_var <- paste0("hurr_count_", year_pair)
  if (hurr_c_var %in% names(data_to_align)) data_to_align[, starting_hurr_count := get(hurr_c_var)]
  
  hurr_s_var <- paste0("hurr_strength_", year_pair)
  if (hurr_s_var %in% names(data_to_align)) data_to_align[, starting_hurr_strength := get(hurr_s_var)]
  
  
  # --- 2. Define 'delta_bathy' column based on data_type ---
  if (data_type == "training") {
    if ("delta_bathy" %in% names(data_to_align)) {
      # 'delta_bathy' already exists, ensure it is the column used
    } else if ("bathy_t1" %in% names(data_to_align) && "bathy_t" %in% names(data_to_align)) {
      # Calculate delta_bathy if it's missing but sources are present
      data_to_align[, delta_bathy := bathy_t1 - bathy_t]
    }
    # Ensure delta_bathy is NA if it couldn't be calculated (should be filtered later)
    if (!"delta_bathy" %in% names(data_to_align)) data_to_align[, delta_bathy := NA_real_]
    
  } else { # data_type == "prediction"
    # Ensure delta_bathy column exists in prediction data, but as NA
    if (!"delta_bathy" %in% names(data_to_align)) data_to_align[, delta_bathy := NA_real_]
  }
  
  cat("Done.\n")
  return(data_to_align)
}


# --- 4. Match PDP Conditions (V6.8 - Robust KNN Vars) ---
match_pdp_conditions <- function(prediction_data, pdp_data) {
  cat("  - INFO: Applying PDP adjustment...\n")
  setDT(prediction_data); setDT(pdp_data)
  
  if(nrow(pdp_data) == 0 || all(is.na(pdp_data$PDP_Value))) {
    cat("    - WARN: PDP data is empty or all NA. Skipping PDP adjustment.\n")
    prediction_data[, pdp_adjusted_change := NA_real_]
    return(prediction_data)
  }
  
  env_ranges <- pdp_data[, .(
    min_val = min(Env_Value, na.rm = TRUE),
    max_val = max(Env_Value, na.rm = TRUE),
    range_width = max(Env_Value, na.rm = TRUE) - min(Env_Value, na.rm = TRUE)
  ), by = Predictor]
  env_ranges <- env_ranges[is.finite(range_width)] # Filter out NA/Inf ranges
  
  # --- Use a core set of variables for matching ---
  pdp_match_vars_potential <- list(
    list(pred_pattern = "bathy_t", start_col = "starting_bathy", weight = 0.3),
    list(pred_pattern = "slope_t", start_col = "starting_slope", weight = 0.15),
    list(pred_pattern = "rugosity_t", start_col = "starting_rugosity", weight = 0.1),
    list(pred_pattern = "bpi_broad_t", start_col = "starting_bpi_broad", weight = 0.1),
    list(pred_pattern = "bpi_fine_t", start_col = "starting_bpi_fine", weight = 0.1),
    list(pred_pattern = "tsm_", start_col = "starting_tsm", weight = 0.1)
  )
  
  pdp_match_vars <- list()
  available_pdp_predictors <- unique(pdp_data$Predictor)
  available_pred_cols <- names(prediction_data)
  
  for(var_info in pdp_match_vars_potential){
    actual_pdp_predictor <- grep(var_info$pred_pattern, available_pdp_predictors, value = TRUE)[1]
    # Also check that the predictor has a valid range
    if(!is.na(actual_pdp_predictor) && var_info$start_col %in% available_pred_cols && actual_pdp_predictor %in% env_ranges$Predictor){
      var_info$pred_pattern <- actual_pdp_predictor
      pdp_match_vars[[length(pdp_match_vars) + 1]] <- var_info
    }
  }
  
  if(length(pdp_match_vars) == 0){
    warning("No common variables with valid ranges found for PDP matching. Skipping.")
    prediction_data[, pdp_adjusted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching PDP using variables:", paste(sapply(pdp_match_vars, `[[`, "start_col"), collapse=", "), "\n")
  
  prediction_data[, pdp_adjusted_change := {
    current_point <- .SD
    is_match <- rep(FALSE, nrow(pdp_data))
    for (var in pdp_match_vars) {
      predictor_name <- var$pred_pattern
      range_info <- env_ranges[Predictor == predictor_name] # Already filtered for finite range_width
      if (nrow(range_info) > 0 && var$start_col %in% names(current_point)) {
        current_val <- current_point[[var$start_col]]
        if (!is.na(current_val)) {
          half_window <- var$weight * range_info$range_width
          if(!is.na(half_window) && half_window > 0){
            is_match <- is_match | (pdp_data$Predictor == predictor_name &
                                      pdp_data$Env_Value >= (current_val - half_window) &
                                      pdp_data$Env_Value <= (current_val + half_window))
          } else if (!is.na(half_window) && half_window == 0){
            is_match <- is_match | (pdp_data$Predictor == predictor_name & pdp_data$Env_Value == current_val)
          }
        }
      }
    }
    matches <- pdp_data[is_match]
    if (nrow(matches) > 0) mean(matches$PDP_Value, na.rm = TRUE) else NA_real_
  }, by = 1:nrow(prediction_data)]
  
  message("  - INFO: PDP Adjusted Change matched on: ", sum(!is.na(prediction_data$pdp_adjusted_change)), " points.")
  return(prediction_data)
}


# --- 5. Apply KNN Trend Adjustments (V6.8 - Robust + Weighted) ---
apply_training_knn_adjustment <- function(prediction_data, training_data, k = 15) {
  cat("  - INFO: Applying Training KNN adjustment (Weighted by delta_bathy)...\n")
  setDT(prediction_data); setDT(training_data)
  
  # --- NEW: Use a core set of variables for robust matching ---
  potential_env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity", 
                          "starting_bpi_broad", "starting_bpi_fine")
  
  env_vars <- intersect(potential_env_vars, names(prediction_data))
  env_vars <- intersect(env_vars, names(training_data)) # Ensure they exist in training data too
  
  if (length(env_vars) == 0) {
    warning("No common env variables for Training KNN. Skipping.")
    prediction_data[, TrainingKNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching using variables:", paste(env_vars, collapse=", "), "\n")
  
  response_var <- "delta_bathy" # Use delta_bathy
  if (!response_var %in% names(training_data) || all(is.na(training_data[[response_var]]))) {
    if (all(c("bathy_t1", "bathy_t") %in% names(training_data))){
      training_data[, delta_bathy := bathy_t1 - bathy_t]
      cat("    - INFO: 'delta_bathy' column calculated for Training KNN.\n")
    } else {
      stop("Missing 'delta_bathy' or source columns in training data.")
    }
  }
  
  cols_for_pred_cases <- env_vars
  for(col in cols_for_pred_cases) {
    if(!is.numeric(prediction_data[[col]])) {
      cat("    - WARN (TrainKNN): Coercing pred col to numeric:", col, "\n")
      prediction_data[, (col) := as.numeric(get(col))]
    }
  }
  pred_numeric_check <- sapply(prediction_data[, ..cols_for_pred_cases], is.numeric)
  if(!all(pred_numeric_check)) { stop("Not all prediction columns for KNN are numeric.") }
  
  keep_rows_pred <- prediction_data[, complete.cases(.SD), .SDcols = cols_for_pred_cases]
  pred_clean <- prediction_data[keep_rows_pred]
  
  cols_for_train_cases <- c(env_vars, response_var)
  for(col in cols_for_train_cases) {
    if(!is.numeric(training_data[[col]])) {
      cat("    - WARN (TrainKNN): Coercing train col to numeric:", col, "\n")
      training_data[, (col) := as.numeric(get(col))]
    }
  }
  train_numeric_check <- sapply(training_data[, ..cols_for_train_cases], is.numeric)
  if(!all(train_numeric_check)) { stop("Not all training columns for KNN are numeric.") }
  
  keep_rows_train <- training_data[, complete.cases(.SD), .SDcols = cols_for_train_cases]
  train_clean <- training_data[keep_rows_train]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases for Training KNN. Skipping."); prediction_data[, TrainingKNN_predicted_change := NA_real_]; return(prediction_data) }
  if (nrow(train_clean) < k) {
    message("    - WARN: Not enough training data (", nrow(train_clean), ") for Training KNN (k=", k, "). Skipping adjustment.");
    prediction_data[, TrainingKNN_predicted_change := NA_real_];
    return(prediction_data)
  }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]] # This is delta_bathy
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  # --- ENHANCEMENT: Use weighted mean based on training weights ---
  pred_avg <- sapply(1:nrow(knn_result$nn.index), function(i) {
    neighbor_indices <- knn_result$nn.index[i, ]
    neighbor_deltas <- train_resp[neighbor_indices]
    
    # Calculate weights *for the neighbors* based on their delta_bathy
    neighbor_weights <- calculate_sample_weights(neighbor_deltas)
    
    # Return the weighted mean
    return(weighted.mean(neighbor_deltas, neighbor_weights, na.rm = TRUE))
  })
  pred_avg[!is.finite(pred_avg)] <- NA_real_
  # --- END ENHANCEMENT ---
  
  pred_clean[, TrainingKNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), TrainingKNN_predicted_change := i.TrainingKNN_predicted_change]
  
  message("  - INFO: Assigned Training KNN predictions to ", sum(!is.na(prediction_data$TrainingKNN_predicted_change)), " rows.")
  return(prediction_data)
}


# --- 6. SHAP-Weighted KNN Adjustment (V6.8 - Robust + Weighted) ---
apply_shap_weighted_knn_adjustment <- function(prediction_data, training_data, shap_summary, k = 15) {
  cat("  - INFO: Applying SHAP-Weighted KNN adjustment (Weighted by delta_bathy)...\n")
  setDT(prediction_data); setDT(training_data); setDT(shap_summary)
  
  if(nrow(shap_summary) == 0 || all(is.na(shap_summary$Overall_Mean_Abs_SHAP))) {
    cat("    - WARN: SHAP summary is empty or invalid. Skipping SHAP-KNN adjustment.\n")
    prediction_data[, SHAP_KNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  
  # --- NEW: Define a core set of variables to consider for SHAP-KNN ---
  core_knn_vars <- c("starting_bathy", "starting_slope", "starting_rugosity", 
                     "starting_bpi_broad", "starting_bpi_fine")
  
  shap_predictors <- shap_summary$Predictor
  # This mapping is CRITICAL and links model names to 'starting_' names
  shap_to_starting_map <- data.table(
    Predictor = shap_predictors,
    starting_col = case_when(
      shap_predictors == "bathy_t" ~ "starting_bathy",
      shap_predictors == "slope_t" ~ "starting_slope",
      shap_predictors == "rugosity_t" ~ "starting_rugosity",
      shap_predictors == "bpi_broad_t" ~ "starting_bpi_broad",
      shap_predictors == "bpi_fine_t" ~ "starting_bpi_fine",
      shap_predictors == "terrain_classification_t" ~ "starting_terrain_class",
      shap_predictors == "grain_size_layer" ~ "starting_grain_size",
      shap_predictors == "prim_sed_layer" ~ "starting_sed_type",
      grepl("^tsm_", shap_predictors) ~ "starting_tsm",
      grepl("^hurr_count_", shap_predictors) ~ "starting_hurr_count",
      grepl("^hurr_strength_", shap_predictors) ~ "starting_hurr_strength",
      grepl("_(mean3|sd3)_t$", shap_predictors) ~ paste0("starting_", sub("_t$", "", shap_predictors)),
      TRUE ~ NA_character_
    )
  )
  
  valid_map <- shap_to_starting_map[!is.na(starting_col)]
  valid_map <- valid_map[starting_col %in% names(prediction_data)]
  valid_map <- valid_map[starting_col %in% names(training_data)]
  
  # --- NEW: Filter the map to ONLY the core_knn_vars ---
  valid_map <- valid_map[starting_col %in% core_knn_vars]
  
  if(nrow(valid_map) == 0) {
    warning("No common core variables found for SHAP-Weighted KNN. Skipping.")
    prediction_data[, SHAP_KNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  
  env_vars <- valid_map$starting_col
  
  shap_weights_dt <- merge(valid_map, shap_summary, by = "Predictor")
  shap_weights <- shap_weights_dt$Overall_Mean_Abs_SHAP
  names(shap_weights) <- shap_weights_dt$starting_col
  shap_weights <- (shap_weights / sum(shap_weights, na.rm=TRUE) * length(shap_weights))
  shap_weights[!is.finite(shap_weights)] <- 1e-6
  
  cat("    - Matching using SHAP-weighted variables:", paste(env_vars, collapse=", "), "\n")
  
  response_var <- "delta_bathy" # Use delta_bathy
  if (!response_var %in% names(training_data) || all(is.na(training_data[[response_var]]))) {
    if (all(c("bathy_t1", "bathy_t") %in% names(training_data))){
      training_data[, delta_bathy := bathy_t1 - bathy_t]
      cat("    - INFO: 'delta_bathy' column calculated for SHAP-KNN.\n")
    } else {
      stop("Missing 'delta_bathy' or source columns in training data for SHAP-KNN.")
    }
  }
  
  cols_for_pred_cases_shap <- env_vars
  for(col in cols_for_pred_cases_shap) {
    if(!is.numeric(prediction_data[[col]])) {
      cat("    - WARN (SHAP-KNN): Coercing pred col to numeric:", col, "\n")
      prediction_data[, (col) := as.numeric(get(col))]
    }
  }
  pred_numeric_check_shap <- sapply(prediction_data[, ..cols_for_pred_cases_shap], is.numeric)
  if(!all(pred_numeric_check_shap)) { stop("Not all prediction columns for SHAP-KNN are numeric.") }
  
  keep_rows_pred_shap <- prediction_data[, complete.cases(.SD), .SDcols = cols_for_pred_cases_shap]
  pred_clean <- prediction_data[keep_rows_pred_shap]
  
  cols_for_train_cases_shap <- c(env_vars, response_var)
  for(col in cols_for_train_cases_shap) {
    if(!is.numeric(training_data[[col]])) {
      cat("    - WARN (SHAP-KNN): Coercing train col to numeric:", col, "\n")
      training_data[, (col) := as.numeric(get(col))]
    }
  }
  train_numeric_check_shap <- sapply(training_data[, ..cols_for_train_cases_shap], is.numeric)
  if(!all(train_numeric_check_shap)) { stop("Not all training columns for SHAP-KNN are numeric.") }
  
  keep_rows_train_shap <- training_data[, complete.cases(.SD), .SDcols = cols_for_train_cases_shap]
  train_clean <- training_data[keep_rows_train_shap]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases for SHAP-KNN. Skipping."); prediction_data[, SHAP_KNN_predicted_change := NA_real_]; return(prediction_data) }
  if (nrow(train_clean) < k) {
    message("    - WARN: Not enough training data (", nrow(train_clean), ") for SHAP-KNN (k=", k, "). Skipping adjustment.");
    prediction_data[, SHAP_KNN_predicted_change := NA_real_];
    return(prediction_data)
  }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]] # This is delta_bathy
  
  all_mat <- rbind(pred_mat, train_mat)
  means <- colMeans(all_mat, na.rm=TRUE)
  sds <- apply(all_mat, 2, sd, na.rm=TRUE)
  sds[sds == 0] <- 1
  
  pred_mat_scaled <- scale(pred_mat, center = means, scale = sds)
  train_mat_scaled <- scale(train_mat, center = means, scale = sds)
  
  col_weights <- shap_weights[env_vars]
  
  pred_mat_weighted <- t(t(pred_mat_scaled) * col_weights)
  train_mat_weighted <- t(t(train_mat_scaled) * col_weights)
  
  knn_result <- FNN::get.knnx(train_mat_weighted, pred_mat_weighted, k = k)
  
  # --- ENHANCEMENT: Use weighted mean based on training weights ---
  pred_avg <- sapply(1:nrow(knn_result$nn.index), function(i) {
    neighbor_indices <- knn_result$nn.index[i, ]
    neighbor_deltas <- train_resp[neighbor_indices]
    
    # Calculate weights *for the neighbors* based on their delta_bathy
    neighbor_weights <- calculate_sample_weights(neighbor_deltas)
    
    # Return the weighted mean
    return(weighted.mean(neighbor_deltas, neighbor_weights, na.rm = TRUE))
  })
  pred_avg[!is.finite(pred_avg)] <- NA_real_
  # --- END ENHANCEMENT ---
  
  pred_clean[, SHAP_KNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), SHAP_KNN_predicted_change := i.SHAP_KNN_predicted_change]
  
  message("  - INFO: Assigned SHAP-Weighted KNN predictions to ", sum(!is.na(prediction_data$SHAP_KNN_predicted_change)), " rows.")
  return(prediction_data)
}


#' --- 7. NEW (from V4): Apply Bootstrap KNN Adjustment (V6.8 - Robust KNN) ---
apply_bootstrap_knn_adjustment <- function(prediction_data, boot_summary_data, k = 15) {
  cat("  - INFO: Applying Bootstrap KNN adjustment (V4 method)...\n")
  setDT(prediction_data); setDT(boot_summary_data)
  
  # --- NEW: Use a core set of variables for robust matching ---
  potential_env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity", 
                          "starting_bpi_broad", "starting_bpi_fine")
  
  env_vars <- intersect(potential_env_vars, names(prediction_data))
  env_vars <- intersect(env_vars, names(boot_summary_data)) # Ensure they exist in boot summary data
  
  if (length(env_vars) == 0) {
    warning("No common env variables for Bootstrap KNN. Skipping.")
    prediction_data[, BootstrapKNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  cat("    - Matching using variables:", paste(env_vars, collapse=", "), "\n")
  
  response_var <- "mean_predicted_change" # This is the target
  
  cols_for_pred_cases <- env_vars
  for(col in cols_for_pred_cases) {
    if(!is.numeric(prediction_data[[col]])) {
      prediction_data[, (col) := as.numeric(get(col))]
    }
  }
  
  cols_for_train_cases <- c(env_vars, response_var)
  for(col in cols_for_train_cases) {
    if(!is.numeric(boot_summary_data[[col]])) {
      boot_summary_data[, (col) := as.numeric(get(col))]
    }
  }
  
  keep_rows_pred <- prediction_data[, complete.cases(.SD), .SDcols = cols_for_pred_cases]
  pred_clean <- prediction_data[keep_rows_pred]
  
  keep_rows_train <- boot_summary_data[, complete.cases(.SD), .SDcols = cols_for_train_cases]
  train_clean <- boot_summary_data[keep_rows_train]
  
  if (nrow(pred_clean) == 0) { message("    - WARN: No complete cases for Bootstrap KNN. Skipping."); prediction_data[, BootstrapKNN_predicted_change := NA_real_]; return(prediction_data) }
  if (nrow(train_clean) < k) {
    message("    - WARN: Not enough training data (", nrow(train_clean), ") for Bootstrap KNN (k=", k, "). Skipping.");
    prediction_data[, BootstrapKNN_predicted_change := NA_real_];
    return(prediction_data)
  }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]] # This is 'mean_predicted_change'
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  # Use distance-weighted mean for this method
  dist_weights <- 1 / (knn_result$nn.dist + 1e-6)
  sum_dist_weights <- rowSums(dist_weights, na.rm = TRUE)
  pred_avg <- rowSums(matrix(train_resp[knn_result$nn.index], nrow = nrow(knn_result$nn.index)) * dist_weights, na.rm = TRUE) / sum_dist_weights
  pred_avg[sum_dist_weights == 0] <- NA_real_
  
  pred_clean[, BootstrapKNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), BootstrapKNN_predicted_change := i.BootstrapKNN_predicted_change]
  
  message("  - INFO: Assigned Bootstrap KNN predictions to ", sum(!is.na(prediction_data$BootstrapKNN_predicted_change)), " rows.")
  return(prediction_data)
}


#' --- 8. Compare Methods (V6.8 - 4 methods) ---
#' --- 8. Compare Methods (V6.9 - mean + extremes) ---
compare_prediction_methods_v6 <- function(prediction_data, tile_id, year_pair, processing_mode) {
  message("  - INFO: Comparing V6.9 prediction methods (mean + extremes)...")
  setDT(prediction_data)
  
  # Ensure mean-change prediction columns exist
  mean_pred_cols <- c("XGB_predicted_change", "pdp_adjusted_change",
                      "TrainingKNN_predicted_change", 
                      "SHAP_KNN_predicted_change",
                      "BootstrapKNN_predicted_change")
  for(col in mean_pred_cols){
    if(!col %in% names(prediction_data)) prediction_data[, (col) := NA_real_]
  }
  
  # Ensure extreme fields from XGB exist (may be NA if older runs)
  extreme_cols <- c("XGB_q05_change", "XGB_q95_change",
                    "XGB_erosion_extreme_change", "XGB_deposition_extreme_change",
                    "Uncertainty_SD")
  for(col in extreme_cols) {
    if(!col %in% names(prediction_data)) prediction_data[, (col) := NA_real_]
  }
  
  # ------------------------------------------------------------------
  # CENTRAL HYBRID (unchanged logic)
  # ------------------------------------------------------------------
  # Primary blend: XGB Ensemble and SHAP-KNN
  hybrid_cols_primary <- c("XGB_predicted_change", "SHAP_KNN_predicted_change")
  prediction_data[, hybrid_change := rowMeans(.SD, na.rm = TRUE), .SDcols = hybrid_cols_primary]
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  # Fallback blend: PDP and simple Training KNN
  hybrid_cols_fallback <- c("pdp_adjusted_change", "TrainingKNN_predicted_change")
  prediction_data[is.na(hybrid_change), hybrid_change := rowMeans(.SD, na.rm = TRUE),
                  .SDcols = hybrid_cols_fallback]
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  # Final fallback: V4 Bootstrap KNN
  prediction_data[is.na(hybrid_change), hybrid_change := BootstrapKNN_predicted_change]
  
  # ------------------------------------------------------------------
  # EXTREME HYBRID: erosion / deposition
  # ------------------------------------------------------------------
  # If quantiles are missing but mean+SD exist, approximate with normal tails
  missing_q05 <- is.na(prediction_data$XGB_q05_change) &
    is.finite(prediction_data$XGB_predicted_change) &
    is.finite(prediction_data$Uncertainty_SD)
  missing_q95 <- is.na(prediction_data$XGB_q95_change) &
    is.finite(prediction_data$XGB_predicted_change) &
    is.finite(prediction_data$Uncertainty_SD)
  
  if (any(missing_q05)) {
    prediction_data[missing_q05,
                    XGB_q05_change := XGB_predicted_change - 1.645 * Uncertainty_SD]
  }
  if (any(missing_q95)) {
    prediction_data[missing_q95,
                    XGB_q95_change := XGB_predicted_change + 1.645 * Uncertainty_SD]
  }
  
  # Define erosion / deposition extremes from XGB quantiles
  prediction_data[, XGB_erosion_extreme_change    := XGB_q05_change]
  prediction_data[, XGB_deposition_extreme_change := XGB_q95_change]
  
  # For now, the "hybrid" extremes are just aliases to the XGB extremes.
  # Later you could blend with KNN-adjusted means if desired.
  prediction_data[, hybrid_erosion_extreme    := XGB_erosion_extreme_change]
  prediction_data[, hybrid_deposition_extreme := XGB_deposition_extreme_change]
  
  # ------------------------------------------------------------------
  # Diagnostics / logs
  # ------------------------------------------------------------------
  cor_pdp_knn <- if(sum(!is.na(prediction_data$pdp_adjusted_change) &
                        !is.na(prediction_data$TrainingKNN_predicted_change)) > 2) {
    cor(prediction_data$pdp_adjusted_change,
        prediction_data$TrainingKNN_predicted_change, use="complete.obs")
  } else { NA_real_ }
  
  cor_xgb_shapknn <- if(sum(!is.na(prediction_data$XGB_predicted_change) &
                            !is.na(prediction_data$SHAP_KNN_predicted_change)) > 2) {
    cor(prediction_data$XGB_predicted_change,
        prediction_data$SHAP_KNN_predicted_change, use="complete.obs")
  } else { NA_real_ }
  
  comparison_log <- data.table(
    Tile_ID          = tile_id,
    Year_Pair        = year_pair,
    Total_Points     = nrow(prediction_data),
    PDP_TrainKNN_Cor = cor_pdp_knn,
    XGB_SHAPKNN_Cor  = cor_xgb_shapknn
  )
  
  performance_log <- NULL
  # Use 'delta_bathy' for validation of central tendency
  if (processing_mode == 'direct_model' &&
      "delta_bathy" %in% names(prediction_data) &&
      any(!is.na(prediction_data$delta_bathy))) {
    
    performance_log <- data.table(
      Tile_ID = tile_id, Year_Pair = year_pair,
      N_Actual = sum(!is.na(prediction_data$delta_bathy)),
      RMSE_XGB_Ens   = rmse(prediction_data$delta_bathy, prediction_data$XGB_predicted_change),
      RMSE_SHAP_KNN  = rmse(prediction_data$delta_bathy, prediction_data$SHAP_KNN_predicted_change),
      RMSE_Boot_KNN  = rmse(prediction_data$delta_bathy, prediction_data$BootstrapKNN_predicted_change),
      RMSE_Hybrid    = rmse(prediction_data$delta_bathy, prediction_data$hybrid_change),
      MAE_XGB_Ens    = mae(prediction_data$delta_bathy, prediction_data$XGB_predicted_change),
      MAE_SHAP_KNN   = mae(prediction_data$delta_bathy, prediction_data$SHAP_KNN_predicted_change),
      MAE_Boot_KNN   = mae(prediction_data$delta_bathy, prediction_data$BootstrapKNN_predicted_change),
      MAE_Hybrid     = mae(prediction_data$delta_bathy, prediction_data$hybrid_change)
      # (We intentionally do NOT score extremes against single realisation delta_bathy)
    )
  }
  
  message("  - INFO: Hybrid change calculated. XGB/SHAP-KNN Cor:",
          round(cor_xgb_shapknn, 3))
  return(list(data = prediction_data,
              comparison_log = comparison_log,
              performance_log = performance_log))
}


# --- 9. save_final_predictions (V6.8 - suffix update) ---
# --- 9. save_final_predictions (V6.9 - mean + extreme depths) ---
save_final_predictions <- function(prediction_data, output_dir, tile_id, year_pair) {
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  
  # Central depth (hybrid mean)
  pred_depth_col <- paste0("pred_", end_year, "_depth")
  start_bathy_col <- "starting_bathy"
  
  if (!start_bathy_col %in% names(prediction_data) || !"hybrid_change" %in% names(prediction_data)) {
    stop("Cannot save final predictions: 'starting_bathy' or 'hybrid_change' column is missing.")
  }
  if (nrow(prediction_data[is.na(get(start_bathy_col))]) > 0) {
    message("  - NOTE: Some 'starting_bathy' values are NA. Final predicted depth may also be NA.")
  }
  prediction_data[, (pred_depth_col) := get(start_bathy_col) + hybrid_change]
  
  # NEW: XGB extreme depths (erosion/deposition) from quantiles
  # These are purely model/ensemble based; hybrid still used for central.
  if ("XGB_q05_change" %in% names(prediction_data)) {
    pred_depth_q05_col <- paste0("pred_", end_year, "_depth_xgb_q05")
    prediction_data[, (pred_depth_q05_col) := get(start_bathy_col) + XGB_q05_change]
  }
  if ("XGB_q50_change" %in% names(prediction_data)) {
    pred_depth_q50_col <- paste0("pred_", end_year, "_depth_xgb_q50")
    prediction_data[, (pred_depth_q50_col) := get(start_bathy_col) + XGB_q50_change]
  }
  if ("XGB_q95_change" %in% names(prediction_data)) {
    pred_depth_q95_col <- paste0("pred_", end_year, "_depth_xgb_q95")
    prediction_data[, (pred_depth_q95_col) := get(start_bathy_col) + XGB_q95_change]
  }
  
  out_file_dir <- file.path(output_dir, tile_id)
  if (!dir.exists(out_file_dir)) dir.create(out_file_dir, recursive = TRUE)
  out_file <- file.path(out_file_dir, paste0(tile_id, "_prediction_final_", year_pair, "_v6.9.fst"))
  write_fst(prediction_data, out_file)
  message("  - INFO: Final FST prediction file saved to: ", basename(out_file))
  return(prediction_data)
}


# --- 10. save_component_rasters (V6.9 - includes extremes) ---
save_component_rasters <- function(prediction_data, output_dir, tile_id, year_pair, crs_obj) {
  out_raster_dir <- file.path(output_dir, tile_id)
  if (!dir.exists(out_raster_dir)) dir.create(out_raster_dir, recursive = TRUE)
  
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  pred_depth_col       <- paste0("pred_", end_year, "_depth")
  pred_depth_q05_col   <- paste0("pred_", end_year, "_depth_xgb_q05")
  pred_depth_q50_col   <- paste0("pred_", end_year, "_depth_xgb_q50")
  pred_depth_q95_col   <- paste0("pred_", end_year, "_depth_xgb_q95")
  
  save_raster <- function(data, col_name, file_suffix) {
    if (col_name %in% names(data)) {
      valid_rows <- data[!is.na(get(col_name)) & is.finite(get(col_name)),
                         .(x = X, y = Y, z = get(col_name))]
      if (nrow(valid_rows) > 0) {
        tryCatch({
          r <- raster::rasterFromXYZ(valid_rows, crs = crs_obj)
          out_path <- file.path(out_raster_dir,
                                paste0(tile_id, "_", file_suffix, "_", year_pair, ".tif"))
          raster::writeRaster(r, out_path, format = "GTiff", overwrite = TRUE)
        }, error = function(e){
          cat("    - ERROR saving raster", file_suffix, ":", conditionMessage(e), "\n")
        })
      } else {
        cat("    - WARN: No valid finite data for raster", file_suffix, "\n")
      }
    } else {
      cat("    - WARN: Column missing for raster", file_suffix, "\n")
    }
  }
  
  # Central change & components
  save_raster(prediction_data, "hybrid_change",               "Hybrid_predicted_change")
  save_raster(prediction_data, "pdp_adjusted_change",         "PDP_predicted_change")
  save_raster(prediction_data, "TrainingKNN_predicted_change","TrainingKNN_predicted_change")
  save_raster(prediction_data, "XGB_predicted_change",        "XGB_Ens_predicted_change")
  save_raster(prediction_data, "SHAP_KNN_predicted_change",   "SHAP_KNN_predicted_change")
  save_raster(prediction_data, "BootstrapKNN_predicted_change","BootstrapKNN_predicted_change")
  save_raster(prediction_data, "Uncertainty_SD",              "Uncertainty_SD")
  
  # NEW: XGB change quantiles & extreme hybrids
  save_raster(prediction_data, "XGB_q05_change",              "XGB_Ens_q05_change")
  save_raster(prediction_data, "XGB_q50_change",              "XGB_Ens_q50_change")
  save_raster(prediction_data, "XGB_q95_change",              "XGB_Ens_q95_change")
  save_raster(prediction_data, "XGB_change_spread_q95_q05",   "XGB_Ens_change_spread_q95_q05")
  
  save_raster(prediction_data, "hybrid_erosion_extreme",      "Hybrid_erosion_extreme_change")
  save_raster(prediction_data, "hybrid_deposition_extreme",   "Hybrid_deposition_extreme_change")
  
  # Depth rasters
  save_raster(prediction_data, pred_depth_col,     "Hybrid_predicted_depth")
  save_raster(prediction_data, pred_depth_q05_col, "XGB_Ens_pred_depth_q05")
  save_raster(prediction_data, pred_depth_q50_col, "XGB_Ens_pred_depth_q50")
  save_raster(prediction_data, pred_depth_q95_col, "XGB_Ens_pred_depth_q95")
}


# --- 11. build_all_tile_maps (V6.8 - no change) ---
build_all_tile_maps <- function(training_dir, prediction_dir, year_pairs) {
  message("  - INFO: Building year-specific geographic maps (V6.8)...")
  training_footprint_gpkg <- file.path(training_dir, "intersecting_sub_grids_UTM.gpkg")
  prediction_footprint_gpkg <- file.path(prediction_dir, "intersecting_sub_grids_UTM.gpkg")
  if(!all(file.exists(training_footprint_gpkg, prediction_footprint_gpkg))) stop("Footprint GPKG files not found.")
  
  train_sf_full <- sf::st_read(training_footprint_gpkg, quiet = TRUE)
  pred_sf <- sf::st_read(prediction_footprint_gpkg, quiet = TRUE)
  id_col <- "tile_id"
  if(!(id_col %in% names(train_sf_full)) || !(id_col %in% names(pred_sf))) stop("'tile_id' column not found in GPKG files.")
  
  suppressWarnings({ pred_centroids <- sf::st_centroid(pred_sf) })
  
  all_maps <- list()
  for (yp in year_pairs) {
    model_file_pattern <- paste0("^model_boot_\\d+_", yp, "\\.rds$")
    potential_tile_dirs <- list.dirs(training_dir, full.names = FALSE, recursive = FALSE)
    valid_training_tiles <- potential_tile_dirs[!grepl("diagnostic_plots|prediction_logs|\\.", potential_tile_dirs)]
    
    valid_source_tiles <- character()
    for (tile in valid_training_tiles) {
      model_sub_dir <- file.path(training_dir, tile, "bootstrap_models")
      if (dir.exists(model_sub_dir) && length(list.files(model_sub_dir, pattern = model_file_pattern)) > 0) {
        valid_source_tiles <- c(valid_source_tiles, tile)
      }
    }
    
    if (length(valid_source_tiles) == 0) {
      cat("    - WARN: No training tiles with models found for year-pair: ", yp, "\n")
      all_maps[[yp]] <- NULL; next
    }
    
    train_sf_year_specific <- train_sf_full[train_sf_full[[id_col]] %in% valid_source_tiles, ]
    if(nrow(train_sf_year_specific) == 0) { all_maps[[yp]] <- NULL; next }
    
    suppressWarnings({ train_centroids_year_specific <- sf::st_centroid(train_sf_year_specific) })
    if(nrow(train_centroids_year_specific) == 0){ all_maps[[yp]] <- NULL; next }
    
    nearest_indices <- sf::st_nearest_feature(pred_centroids, train_centroids_year_specific)
    
    tile_map <- train_sf_year_specific[[id_col]][nearest_indices]
    names(tile_map) <- pred_sf[[id_col]]
    all_maps[[as.character(yp)]] <- tile_map
  }
  return(all_maps)
}

# --- 12. run_cross_validation (V6.8 - new method) ---
run_cross_validation <- function(processed_data, tile_id, year_pair, training_dir) {
  message("  - INFO: Running cross-validation for tile: ", tile_id)
  
  # 'processed_data' (result$data) should now have 'delta_bathy' merged in
  if (!"delta_bathy" %in% names(processed_data) || all(is.na(processed_data$delta_bathy))) {
    message("  - WARN: No 'delta_bathy' (actual change) data found in merged results. Skipping CV.")
    return(NULL)
  }
  
  valid_comp_data <- processed_data[is.finite(delta_bathy)]
  if(nrow(valid_comp_data) < 2) {
    message("  - WARN: Not enough valid comparison points for CV.")
    return(NULL)
  }
  
  cv_log <- data.table(
    Tile_ID = tile_id, Year_Pair = year_pair,
    N_Valid = nrow(valid_comp_data),
    MAE_Hybrid = mae(valid_comp_data$delta_bathy, valid_comp_data$hybrid_change),
    RMSE_Hybrid = rmse(valid_comp_data$delta_bathy, valid_comp_data$hybrid_change),
    MAE_XGB_Ens = mae(valid_comp_data$delta_bathy, valid_comp_data$XGB_predicted_change),
    RMSE_XGB_Ens = rmse(valid_comp_data$delta_bathy, valid_comp_data$XGB_predicted_change),
    MAE_SHAP_KNN = mae(valid_comp_data$delta_bathy, valid_comp_data$SHAP_KNN_predicted_change),
    RMSE_SHAP_KNN = rmse(valid_comp_data$delta_bathy, valid_comp_data$SHAP_KNN_predicted_change),
    MAE_Train_KNN = mae(valid_comp_data$delta_bathy, valid_comp_data$TrainingKNN_predicted_change),
    RMSE_Train_KNN = rmse(valid_comp_data$delta_bathy, valid_comp_data$TrainingKNN_predicted_change),
    MAE_Boot_KNN = mae(valid_comp_data$delta_bathy, valid_comp_data$BootstrapKNN_predicted_change), # NEW
    RMSE_Boot_KNN = rmse(valid_comp_data$delta_bathy, valid_comp_data$BootstrapKNN_predicted_change), # NEW
    MAE_PDP = mae(valid_comp_data$delta_bathy, valid_comp_data$pdp_adjusted_change),
    RMSE_PDP = rmse(valid_comp_data$delta_bathy, valid_comp_data$pdp_adjusted_change)
  )
  return(cv_log)
}

# --- 13. save_summary_plot (V6.8 - 3x3 layout & delta_bathy fix) ---
save_summary_plot <- function(processed_data, tile_id, year_pair, prediction_dir, crs_obj, training_data_path) {
  message("  - INFO: Generating summary plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  library(rasterVis); library(gridExtra); library(ggplot2); library(cmocean); library(raster)
  
  setorderv(processed_data, c("Y", "X"), c(-1, 1))
  plot_list <- list()
  plot_colors <- cmocean('deep')(100)
  
  dt_to_raster <- function(dt, col_name, crs) {
    if (col_name %in% names(dt) && any(is.finite(dt[[col_name]]))) { # Check for finite
      df <- dt[is.finite(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(df) > 0) return(raster::rasterFromXYZ(df, crs = crs))
    }
    return(NULL)
  }
  
  # --- NEW: Load full training data to plot complete delta_bathy ---
  actual_change_raster <- NULL
  if (file.exists(training_data_path)) {
    training_data_full <- read_fst(training_data_path, as.data.table = TRUE)
    if ("delta_bathy" %in% names(training_data_full)) {
      actual_change_raster <- dt_to_raster(training_data_full, "delta_bathy", crs_obj)
      message("  - INFO: Loaded full 'delta_bathy' from training file for plot.")
    }
  }
  if (is.null(actual_change_raster)) {
    message("  - WARN: Could not load full 'delta_bathy'. Plot will use merged (sparse) data.")
    actual_change_raster <- dt_to_raster(processed_data, "delta_bathy", crs_obj) # Fallback
  }
  # --- END NEW ---
  
  plot_list[['xgb']] <- dt_to_raster(processed_data, "XGB_predicted_change", crs_obj)
  plot_list[['shap_knn']] <- dt_to_raster(processed_data, "SHAP_KNN_predicted_change", crs_obj)
  plot_list[['knn']] <- dt_to_raster(processed_data, "TrainingKNN_predicted_change", crs_obj)
  plot_list[['boot_knn']] <- dt_to_raster(processed_data, "BootstrapKNN_predicted_change", crs_obj) # NEW
  plot_list[['pdp']] <- dt_to_raster(processed_data, "pdp_adjusted_change", crs_obj)
  plot_list[['hybrid']] <- dt_to_raster(processed_data, "hybrid_change", crs_obj)
  plot_list[["start_bathy"]] <- dt_to_raster(processed_data, "starting_bathy", crs_obj)
  plot_list[["survey_date"]] <- dt_to_raster(processed_data, "starting_survey_date", crs_obj)
  plot_list[['actual_change']] <- actual_change_raster # Use the full raster
  
  p1 <- if(!is.null(plot_list$xgb)) levelplot(plot_list$xgb, main="XGB Ens. Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("XGB Ens. (NA/No Data)")
  p2 <- if(!is.null(plot_list$shap_knn)) levelplot(plot_list$shap_knn, main="SHAP-KNN Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("SHAP-KNN (NA/No Data)")
  p3 <- if(!is.null(plot_list$knn)) levelplot(plot_list$knn, main="TrainKNN Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("TrainKNN (NA/No Data)")
  p4 <- if(!is.null(plot_list$boot_knn)) levelplot(plot_list$boot_knn, main="BootKNN Change (V4)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("BootKNN (NA/No Data)") # NEW
  p5 <- if(!is.null(plot_list$pdp)) levelplot(plot_list$pdp, main="PDP Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("PDP (NA/No Data)")
  p6 <- if(!is.null(plot_list$hybrid)) levelplot(plot_list$hybrid, main="Final Hybrid Change", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Hybrid (NA/No Data)")
  p7 <- if(!is.null(plot_list$start_bathy)) levelplot(plot_list$start_bathy, main="Start Bathy (t)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Start Bathy (NA/No Data)")
  p8 <- if(!is.null(plot_list$survey_date)) levelplot(plot_list$survey_date, main="Start Survey Date", margin=FALSE, col.regions=terrain.colors) else ggplot() + theme_void() + ggtitle("Survey (NA/No Data)")
  p9 <- if(!is.null(plot_list$actual_change)) levelplot(plot_list$actual_change, main="Actual Change (delta_bathy)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Actual Change (NA/No Data)")
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_summary_plot_", year_pair, "_v6.8.pdf")) # Suffix update
  
  pdf(plot_out_file, width = 18, height = 18) # Make plot bigger (3x3)
  grid.arrange(grobs = list(p1, p2, p3, p4, p5, p6, p7, p8, p9), ncol = 3, nrow = 3)
  dev.off()
  
  message("  - INFO: Summary plot saved to: ", basename(plot_out_file))
}


# --- 14. save_summary_plot_scaled (V6.8 - 3x3 layout & delta_bathy fix) ---
save_summary_plot_scaled <- function(processed_data, tile_id, year_pair, prediction_dir, crs_obj, training_data_path) {
  message("  - INFO: Generating SCALED summary plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  library(rasterVis); library(gridExtra); library(ggplot2); library(cmocean); library(raster)
  
  setorderv(processed_data, c("Y", "X"), c(-1, 1))
  
  dt_to_raster <- function(dt, col_name, crs) {
    if (col_name %in% names(dt) && any(is.finite(dt[[col_name]]))) {
      df <- dt[is.finite(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(df) > 0) return(raster::rasterFromXYZ(df, crs = crs))
    }
    return(NULL)
  }
  
  # --- NEW: Load full training data to plot complete delta_bathy AND get scale ---
  actual_change_raster <- NULL
  zlim <- c(-6, 6) # Default fallback
  
  if (file.exists(training_data_path)) {
    training_data_full <- read_fst(training_data_path, as.data.table = TRUE)
    if ("delta_bathy" %in% names(training_data_full)) {
      actual_change_raster <- dt_to_raster(training_data_full, "delta_bathy", crs_obj)
      zlim_data <- training_data_full[is.finite(delta_bathy), delta_bathy]
      if (length(zlim_data) > 1) {
        zlim <- range(zlim_data, na.rm = TRUE)
      }
      message("  - INFO: Scaled plot range set from 'delta_bathy': ", round(zlim[1],2), " to ", round(zlim[2],2))
    }
  }
  if (is.null(actual_change_raster)) {
    message("  - WARN: Could not load full 'delta_bathy' for scale. Using merged data.")
    actual_change_raster <- dt_to_raster(processed_data, "delta_bathy", crs_obj) # Fallback
    zlim_data <- processed_data[is.finite(delta_bathy), delta_bathy]
    if (length(zlim_data) > 1) {
      zlim <- range(zlim_data, na.rm = TRUE)
    }
  }
  if(!all(is.finite(zlim))) zlim <- c(-6, 6)
  # --- END NEW ---
  
  at <- seq(zlim[1], zlim[2], length.out = 99)
  plot_colors <- cmocean('deep')(100)
  
  change_rasters <- list()
  change_cols <- c("XGB_predicted_change", "SHAP_KNN_predicted_change", "TrainingKNN_predicted_change", 
                   "BootstrapKNN_predicted_change", "pdp_adjusted_change", "hybrid_change")
  for(col in change_cols) {
    change_rasters[[col]] <- dt_to_raster(processed_data, col, crs_obj)
  }
  start_bathy_raster <- dt_to_raster(processed_data, "starting_bathy", crs_obj)
  survey_raster <- dt_to_raster(processed_data, "starting_survey_date", crs_obj)
  
  # Add the (now loaded) actual_change_raster
  change_rasters[['actual_change']] <- actual_change_raster
  
  p1 <- if(!is.null(change_rasters$XGB_predicted_change)) levelplot(change_rasters$XGB_predicted_change, main="XGB Ens. Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("XGB Ens. (NA/No Data)")
  p2 <- if(!is.null(change_rasters$SHAP_KNN_predicted_change)) levelplot(change_rasters$SHAP_KNN_predicted_change, main="SHAP-KNN Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("SHAP-KNN (NA/No Data)")
  p3 <- if(!is.null(change_rasters$TrainingKNN_predicted_change)) levelplot(change_rasters$TrainingKNN_predicted_change, main="TrainKNN Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("TrainKNN (NA/No Data)")
  p4 <- if(!is.null(change_rasters$BootstrapKNN_predicted_change)) levelplot(change_rasters$BootstrapKNN_predicted_change, main="BootKNN Change (V4)", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("BootKNN (NA/No Data)") # NEW
  p5 <- if(!is.null(change_rasters$pdp_adjusted_change)) levelplot(change_rasters$pdp_adjusted_change, main="PDP Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void() + ggtitle("PDP (NA/No Data)")
  p6 <- if(!is.null(change_rasters$hybrid_change)) levelplot(change_rasters$hybrid_change, main="Final Hybrid Change", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Hybrid (NA/No Data)")
  p7 <- if(!is.null(start_bathy_raster)) levelplot(start_bathy_raster, main="Start Bathy (t)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Start Bathy (NA/No Data)")
  p8 <- if(!is.null(survey_raster)) levelplot(survey_raster, main="Start Survey Date", margin=FALSE, col.regions=terrain.colors) else ggplot() + theme_void()+ ggtitle("Survey (NA/No Data)")
  p9 <- if(!is.null(change_rasters$actual_change)) levelplot(change_rasters$actual_change, main="Actual Change (delta_bathy)", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()+ ggtitle("Actual Change (NA/No Data)")
  
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_summary_plot_SCALED_", year_pair, "_v6.8.pdf")) # Suffix update
  
  pdf(plot_out_file, width = 18, height = 18) # Make plot bigger (3x3)
  grid.arrange(grobs = list(p1, p2, p3, p4, p5, p6, p7, p8, p9), ncol = 3, nrow = 3)
  dev.off()
  
  message("  - INFO: SCALED summary plot saved to: ", basename(plot_out_file))
}


# --- 15. save_density_plot (V6.8 - 3x3 layout & delta_bathy fix) ---
save_density_plot <- function(processed_data, tile_id, year_pair, prediction_dir, training_data_path) {
  message("  - INFO: Generating density plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  library(ggplot2); library(data.table); library(gridExtra)
  
  cols_to_plot <- c("XGB_predicted_change", "pdp_adjusted_change", "TrainingKNN_predicted_change",
                    "SHAP_KNN_predicted_change", "BootstrapKNN_predicted_change", "hybrid_change",
                    "starting_bathy")
  cols_to_plot <- intersect(cols_to_plot, names(processed_data))
  if (length(cols_to_plot) < 2) { message("  - WARN: Not enough columns for density plot."); return(NULL) }
  
  plot_data_pred <- melt(processed_data[, ..cols_to_plot], measure.vars = cols_to_plot, na.rm = TRUE)
  plot_data_pred <- plot_data_pred[is.finite(value)]
  
  # --- NEW: Load full training data to plot complete delta_bathy density ---
  plot_data_actual <- NULL
  if (file.exists(training_data_path)) {
    training_data_full <- read_fst(training_data_path, as.data.table = TRUE, columns = "delta_bathy")
    if ("delta_bathy" %in% names(training_data_full)) {
      plot_data_actual <- training_data_full[is.finite(delta_bathy), .(variable = "delta_bathy", value = delta_bathy)]
      message("  - INFO: Loaded", nrow(plot_data_actual), "finite 'delta_bathy' values for density plot.")
    }
  }
  if (is.null(plot_data_actual)) {
    message("  - WARN: Could not load 'delta_bathy' for density. Using merged (sparse) data.")
    plot_data_actual <- processed_data[is.finite(delta_bathy), .(variable = "delta_bathy", value = delta_bathy)] # Fallback
  }
  
  plot_data <- rbindlist(list(plot_data_pred, plot_data_actual), use.names = TRUE, fill = TRUE)
  if (nrow(plot_data) == 0) { message("  - WARN: No finite data available for density plot."); return(NULL)}
  # --- END NEW ---
  
  plot_data[, variable := factor(variable,
                                 levels = c("XGB_predicted_change", "SHAP_KNN_predicted_change", "TrainingKNN_predicted_change",
                                            "BootstrapKNN_predicted_change", "pdp_adjusted_change", "hybrid_change",
                                            "starting_bathy", "delta_bathy"),
                                 labels = c("XGB Ens Pred Change", "SHAP-KNN Change", "TrainKNN Pred Change",
                                            "BootKNN Pred Change", "PDP Pred Change", "Final Hybrid Change",
                                            "Start Bathy (t)", "Actual Change (delta_bathy)"))]
  
  plot_data <- plot_data[!is.na(variable)]
  
  change_data <- plot_data[!variable %in% c("Start Bathy (t)")]
  
  change_x_range <- if ("Actual Change (delta_bathy)" %in% change_data$variable && nrow(change_data[variable == "Actual Change (delta_bathy)"]) > 0) {
    range(change_data[variable == "Actual Change (delta_bathy)"]$value, na.rm = TRUE)
  } else if (nrow(change_data) > 0) {
    range(change_data$value, na.rm=TRUE)
  } else { c(-6, 6) }
  if(!all(is.finite(change_x_range))) change_x_range <- c(-6, 6)
  
  
  plot_list <- list()
  defined_levels <- levels(plot_data$variable)
  
  for(var_name in defined_levels) {
    if(var_name %in% unique(plot_data$variable)) {
      sub_data <- plot_data[variable == var_name]
      p <- ggplot(sub_data, aes(x = value)) +
        geom_density(fill = "skyblue", alpha = 0.7) +
        labs(title = var_name, x = "Value (m)", y = "Density") +
        theme_minimal() + theme(legend.position = "none", axis.text.x = element_text(angle=45, hjust=1))
      if (var_name != "Start Bathy (t)") {
        p <- p + coord_cartesian(xlim = change_x_range)
      }
      plot_list[[var_name]] <- p
    } else {
      plot_list[[var_name]] <- ggplot() + theme_void() + ggtitle(paste(var_name, "(No Data)"))
    }
  }
  
  plot_list[["placeholder"]] <- ggplot() + theme_void()
  
  final_plot_order <- c("XGB Ens Pred Change", "SHAP-KNN Change", "TrainKNN Pred Change",
                        "BootKNN Pred Change", "PDP Pred Change", "Final Hybrid Change",
                        "Start Bathy (t)", "Actual Change (delta_bathy)", "placeholder")
  
  ordered_grobs <- lapply(final_plot_order, function(name) {
    if (name %in% names(plot_list)) plot_list[[name]] else ggplot() + theme_void() + ggtitle(paste(name, "(NA)"))
  })
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_density_plot_", year_pair, "_v6.8.pdf")) # Suffix update
  
  pdf(plot_out_file, width = 12, height = 12) # 3x3 layout
  grid.arrange(grobs = ordered_grobs, ncol = 3, nrow = 3)
  dev.off()
  
  message("  - INFO: Density plot saved to: ", basename(plot_out_file))
}

# ==============================================================================
#   Parallel Prediction Test Wrapper (V6.8 - Stripped Down)
# ==============================================================================
run_prediction_test <- function(prediction_tiles, year_pairs, crs_obj, training_dir, prediction_dir, output_dir, master_predictor_list_path, n_boot_models_to_use = 10) {
  
  message("\nStarting parallel prediction test run (V6.8 Ensemble)...")
  
  all_tile_maps <- build_all_tile_maps(training_dir, prediction_dir, year_pairs)
  num_cores <- max(1, floor(detectCores() / 2))
  cat("  - INFO: Setting up parallel cluster with", num_cores, "cores.\n")
  cl <- makeCluster(num_cores); registerDoParallel(cl); on.exit({ stopCluster(cl) }, add = TRUE)
  task_grid <- expand.grid(tile_id = prediction_tiles, year_pair = year_pairs, stringsAsFactors = FALSE)
  
  message("\n--- Stage 1: Processing data in parallel... ---")
  
  functions_to_export <- c(
    "process_tile", "predict_elevation_change_ensemble",
    "align_predictors", "match_pdp_conditions",
    "apply_training_knn_adjustment",
    "apply_shap_weighted_knn_adjustment",
    "apply_bootstrap_knn_adjustment", # NEW
    "compare_prediction_methods_v6",
    # "run_cross_validation", # --- REMOVED ---
    "calculate_sample_weights", # NEW
    "mae", "rmse"
  )
  
  results_list <- foreach(
    i = 1:nrow(task_grid),
    .export = functions_to_export,
    .packages = c("data.table", "fst", "xgboost", "FNN", "stringr", "Metrics")
  ) %dopar% {
    library(dplyr); library(sf); library(raster);
    
    current_task <- task_grid[i, ]
    
    process_tile_result <- process_tile(
      tile_id        = current_task$tile_id,
      year_pair      = current_task$year_pair,
      training_dir   = training_dir,
      prediction_dir = prediction_dir,
      output_dir     = output_dir,
      all_tile_maps  = all_tile_maps,
      master_predictor_list_path = master_predictor_list_path,
      n_boot_models_to_use = n_boot_models_to_use
    )
    
    return(process_tile_result)
  }
  
  message("\n--- Stage 2: Saving results and plots sequentially... ---")
  
  successful_results <- Filter(function(res) is.list(res) && isTRUE(res$success), results_list)
  failed_results <- Filter(function(res) is.list(res) && isFALSE(res$success), results_list)
  if(length(failed_results) > 0) {
    cat("\n--- WARNING: Some tiles failed processing. See error log for details. ---\n")
  }
  
  all_comparison_logs <- rbindlist(lapply(successful_results, `[[`, "comparison_log"), fill = TRUE)
  all_performance_logs <- rbindlist(lapply(successful_results, `[[`, "performance_log"), fill = TRUE)
  
  # --- NEW: Save the model feature log (FLATTENED) ---
  all_model_feature_logs_list <- lapply(successful_results, `[[`, "model_feature_log")
  # Flatten the list of lists of data.tables
  all_model_feature_logs_flat <- unlist(all_model_feature_logs_list, recursive = FALSE) 
  all_model_feature_logs_flat <- Filter(function(x) is.data.table(x) && nrow(x) > 0, all_model_feature_logs_flat)
  
  if(length(all_model_feature_logs_flat) > 0) {
    all_model_feature_logs <- rbindlist(all_model_feature_logs_flat, fill = TRUE)
    if(nrow(all_model_feature_logs) > 0) fwrite(all_model_feature_logs, file.path(output_dir, "model_feature_usage_log_v6.8.csv"))
  } else {
    cat("  - INFO: No model feature logs were generated.\n")
  }
  # --- END FIX ---
  
  if(nrow(all_comparison_logs) > 0) fwrite(all_comparison_logs, file.path(output_dir, "prediction_method_comparison_log_v6.8.csv"))
  if(nrow(all_performance_logs) > 0) fwrite(all_performance_logs, file.path(output_dir, "model_performance_log_v6.8.csv"))
  
  # --- REMOVED CV LOG ---
  # all_cv_logs <- list()
  library(rasterVis); library(gridExtra); library(ggplot2); library(cmocean); library(raster)
  
  for (result in successful_results) {
    cat("\n--- Saving outputs for Tile:", result$tile_id, "| Year Pair:", result$year_pair, "---\n")
    
    processed_data <- save_final_predictions(result$data, output_dir, result$tile_id, result$year_pair)
    
    if (!is.null(crs_obj)) {
      save_component_rasters(processed_data, output_dir, result$tile_id, result$year_pair, crs_obj)
      
      # --- NEW: Pass training_data_path to plotting functions ---
      save_summary_plot(processed_data, result$tile_id, result$year_pair, prediction_dir, crs_obj, result$training_data_path)
      save_summary_plot_scaled(processed_data, result$tile_id, result$year_pair, prediction_dir, crs_obj, result$training_data_path)
      
      # --- REMOVED DENSITY PLOT and CV ---
      # save_density_plot(processed_data, result$tile_id, result$year_pair, prediction_dir, result$training_data_path)
      # if (result$processing_mode == 'direct_model') {
      #   cv_log <- run_cross_validation(processed_data, result$tile_id, result$year_pair, training_dir)
      #   if(!is.null(cv_log)) all_cv_logs[[length(all_cv_logs) + 1]] <- cv_log
      # }
    }
  }
  
  # if(length(all_cv_logs) > 0) {
  #   fwrite(rbindlist(all_cv_logs, fill = TRUE), file.path(output_dir, "cross_validation_log_v6.8.csv"))
  #   message("  - INFO: Cross-validation log saved.")
  # }
  
  message("\n✅ Parallel prediction test run V6.8 complete.")
}




# Location of data directories 
prediction_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
training_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
prediction_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"
training_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles"

# Define the directories and parameters
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles"
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles"
years <- c("2004_2006") #, "2006_2010") # "2010_2015", "2015_2022")
mask <- raster::raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif")
master_predictor_list_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles/final_master_predictor_list.txt"
mask_crs <- raster::crs(mask)
crs_obj <- mask_crs
# Define the specific tiles you want to generate predictions for
prediction_tile_ids <- c("BH4S556X_3") #, "BH4S2572_4") # "BH4S2572_1")

Sys.time()
run_prediction_test(
  prediction_tiles = prediction_tile_ids,
  year_pairs = years,
  crs_obj = mask_crs, # Pass the CRS object, not the whole mask
  training_dir = training_dir,
  prediction_dir = prediction_dir,
  master_predictor_list_path = master_predictor_list_path,
  output_dir = prediction_dir
)
Sys.time()


showConnections()
closeAllConnections()
gc()





##### ----- SIMPLE ENSEMBLE PREDICTION TO MATCH TRAINING DATA HANDLING-----
# this will do for now, but next week we need to re-structure the training code to make 'common' data processing / handling functions that can be re-used for prediction

# Recreating the predictor order (temp) from the master predictor list 
# A This preserves the Boruta-determined order, which is good.
master_predictors <- readLines(
  "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/training_data_grid_tiles/final_master_predictor_list.txt"
)
pred.data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/prediction_data_grid_tiles/BH4S556X_3/BH4S556X_2004_2006_prediction_long.fst")

# B. Reconstruct the training-true predictor order for a year pair
build_predictors_for_pair <- function(master_predictors,
                                      available_cols,
                                      year_pair) {
  
  # 1. Translate collapsed temporal predictors
  translated <- vapply(
    master_predictors,
    function(p) {
      if (p %in% c("hurr_count", "hurr_strength", "tsm")) {
        paste0(p, "_", year_pair)
      } else {
        p
      }
    },
    character(1)
  )
  
  # 2. Keep only predictors present in data (ORDER PRESERVED)
  predictors <- translated[translated %in% available_cols]
  
  # 3. Remove non-feature columns (defensive)
  predictors <- setdiff(
    predictors,
    c("bathy_t1", "delta_bathy", "X", "Y", "FID",
      "tile_id", "year_t", "year_t1")
  )
  
  # 4. Enforce bathy_t inclusion (exactly like training)
  if ("bathy_t" %in% available_cols && !"bathy_t" %in% predictors) {
    predictors <- c(predictors, "bathy_t")
  }
  
  predictors
}

# Function Call 
predictors <- build_predictors_for_pair(
  master_predictors = master_predictors,
  available_cols   = names(pred.data),
  year_pair        = "2004_2006"
)

stopifnot("bathy_t" %in% predictors)
length(predictors)

# Now, using the above predictor list / key - run the temp prediction code

# 1. Load required artifacts
library(data.table)
library(xgboost)
library(fst)

tile_id   <- "BH4S556X_3"
year_pair <- "2004_2006"

model_dir <- file.path(
  "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1",
  "Coding_Outputs/Training_data_grid_tiles",
  tile_id
)

# Predictor order EXACTLY as training
# predictors <- readRDS(
#   file.path(model_dir, paste0("predictors_", year_pair, ".rds"))
# )

# Predictor ranges learned during training
pred_ranges <- read.fst(
  file.path(model_dir, paste0("predictor_boot_ranges_", year_pair, ".fst"))
)


# 
# 2. Prepare prediction data to mirror training
# 
# This mirrors your training logic minus response-only fields.

prepare_prediction_data <- function(dt, predictors) {
  
  dt <- as.data.table(dt)
  
  # 1. Ensure predictors exist
  missing_preds <- setdiff(predictors, names(dt))
  if (length(missing_preds) > 0) {
    stop("Missing predictors in prediction data: ",
         paste(missing_preds, collapse = ", "))
  }
  
  # 2. Coerce to numeric (important for mosaics)
  dt[, (predictors) := lapply(.SD, as.numeric), .SDcols = predictors]
  
  # 3. Training-equivalent filtering:
  #    training used complete.cases(predictors)
  keep <- complete.cases(dt[, ..predictors])
  
  dt_clean <- dt[keep]
  
  list(
    data = dt_clean,
    keep_index = keep
  )
}

# 3. (Strongly recommended) Flag extrapolation beyond training domain

# This does not block prediction — it gives you diagnostics.

flag_extrapolation <- function(dt, pred_ranges) {
  
  flags <- lapply(pred_ranges$Predictor, function(p) {
    x <- dt[[p]]
    x < pred_ranges[Predictor == p, Min_across_boots] |
      x > pred_ranges[Predictor == p, Max_across_boots]
  })
  
  names(flags) <- pred_ranges$Predictor
  
  as.data.table(flags)
}

# 4. Build DMatrix exactly like training
prep <- prepare_prediction_data(Pred.data, predictors)

dmat <- xgb.DMatrix(
  data    = as.matrix(prep$data[, ..predictors]),
  missing = NA
)

# 5. Bootstrap ensemble prediction (training-consistent)
boot_files <- list.files(
  file.path(model_dir, "bootstrap_models"),
  pattern = paste0("model_boot_.*_", year_pair, "\\.rds$"),
  full.names = TRUE
)

stopifnot(length(boot_files) > 0)

pred_matrix <- vapply(
  boot_files,
  function(f) {
    model <- readRDS(f)
    predict(model, dmat)
  },
  numeric(nrow(prep$data))
)

# Ensemble summary (matches training)
mean_pred <- rowMeans(pred_matrix)
sd_pred   <- apply(pred_matrix, 1, sd)

# 6. Reassemble full prediction table (training-style)
result <- data.table(
  FID = prep$data$FID,
  X   = prep$data$X,
  Y   = prep$data$Y,
  mean_prediction_t1 = mean_pred,
  uncertainty_SD     = sd_pred
)

# OR YOU CAN MAKE A FULL LIST 
result_full <- cbind(
  prep$data[, c("FID", "X", "Y", predictors), with = FALSE],
  mean_prediction_t1 = mean_pred,
  uncertainty_SD     = sd_pred
)

## COMPARE training data to prediction data:
# Compare against training domain for overlapping rows
   common_fid <- intersect(
         bootdf$FID,
         result_full$FID )

cor(
        bootdf[bootdf$FID %in% common_fid, bootdf$mean_prediction_t1],
        result_full[result_full$FID %in% common_fid, result_full$mean_prediction_t1] )



# If you want row-for-row alignment with original grid:
  
  Pred.data[, mean_prediction_t1 := NA_real_]
Pred.data[, uncertainty_SD     := NA_real_]

Pred.data$mean_prediction_t1[prep$keep_index] <- mean_pred
Pred.data$uncertainty_SD[prep$keep_index]     <- sd_pred

# 7. Optional: Extrapolation diagnostics (highly useful offshore)
extrap_flags <- flag_extrapolation(prep$data, pred_ranges)

prep$data[, extrapolation_fraction :=
            rowMeans(as.matrix(extrap_flags))
]


# -----

# plotting function
library(ggplot2)
library(patchwork)
library(data.table)

plot_model_diagnostics <- function(
    bootdf,
    pred_df,
    tile_id,
    year_pair,
    out_dir,
    device = c("pdf", "tiff")
) {
  
  device <- match.arg(device)
  
  # --- Panel 1: Actual delta bathy ---
  p1 <- ggplot(bootdf, aes(X, Y, fill = delta_bathy_actual)) +
    geom_raster() +
    coord_equal() +
    scale_fill_viridis_c(name = "Δ bathy (actual)") +
    ggtitle("Actual Δ bathymetry") +
    theme_minimal()
  
  # --- Panel 2: Predicted bathy_t1 (training) ---
  p2 <- ggplot(bootdf, aes(X, Y, fill = mean_predicted_bathy_t1)) +
    geom_raster() +
    coord_equal() +
    scale_fill_viridis_c(name = "Pred bathy_t1") +
    ggtitle("Predicted bathy_t1 (training)") +
    theme_minimal()
  
  # --- Panel 3: Predicted delta bathy ---
  p3 <- ggplot(
    bootdf,
    aes(X, Y, fill = mean_predicted_bathy_t1 - bathy_t)
  ) +
    geom_raster() +
    coord_equal() +
    scale_fill_viridis_c(name = "Δ bathy (pred)") +
    ggtitle("Predicted Δ bathymetry") +
    theme_minimal()
  
  # --- Panel 4: Prediction-domain bathy_t1 ---
  p4 <- ggplot(
    pred_df,
    aes(X, Y, fill = mean_prediction_t1)
  ) +
    geom_raster() +
    coord_equal() +
    scale_fill_viridis_c(name = "Pred bathy_t1") +
    ggtitle("Predicted bathy_t1 (offshore)") +
    theme_minimal()
  
  fig <- (p1 | p2) / (p3 | p4)
  
  outfile <- file.path(
    out_dir,
    tile_id,
    paste0("model_diagnostics_", year_pair, ".", device)
  )
  
  if (device == "pdf") {
    ggsave(outfile, fig, width = 12, height = 10)
  } else {
    ggsave(outfile, fig, width = 12, height = 10, dpi = 300)
  }
  
  message("Saved diagnostic plot to: ", outfile)
}

#function Call
plot_model_diagnostics(
  bootdf   = bootdf,
  pred_df = result_full,
  tile_id = "BH4S556X_3",
  year_pair = "2004_2006",
  out_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles",
  device = "pdf"
)




#----
#SPATIAL TEST 


 setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Prediction_data_grid_tiles/BH4S656W_2")
 list.files()
"BH4S656W_2_prediction_clipped_data.fst" 
pd5 <- read.fst("BH4S656W_2_prediction_clipped_data.fst")

output_mask_pred_wgs <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/prediction.mask.UTM17_8m.tif"
mask <- raster(output_mask_pred_wgs)

tile_BH4S556X_3 <- rasterFromXYZ(data.frame(x = result[,"X"],  # LAT
                                       y = result[,"Y"],  # LON
                                       z = result[, "mean_prediction_t1"]), # elevation
                            crs = crs(mask))


plot(tile_BH4S556X_3)

train.bathy_t <- rasterFromXYZ(data.frame(x = bootdf[,"X"],  # LAT
                                  y = bootdf[,"Y"],  # LON
                                  z = bootdf[, "bathy_t"]), # elevation
                       crs = crs(mask))

plot(train.bathy_t)



pred.delta <- tile_BH4S556X_3 - train.bathy_t
plot(pred.delta) 


setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs")
writeRaster(bt.bathy, "bt.bob6.tif", format = "GTiff", overwrite = T)

# Sanity check

library(data.table)
library(raster)
library(xgboost)

#----------------------------------
# User Inputs
#----------------------------------
model_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_outputs/Training_data_grid_tiles/BH4S556X_3/bootstrap_models"
out_dir   <- model_dir   # save outputs to same directory
models_to_load <- list.files(model_dir, pattern = "^model_boot_.*\\.rds$", full.names = TRUE)

#----------------------------------
# Prepare Prediction Data
#----------------------------------
# Remove any positive bathy (land) pixels
pred_data <- copy(pred_pts)[bathy_t <= 0]

cat("Filtered prediction dataset: ", nrow(pred_pts), "→", nrow(pred_data), "rows retained (seafloor only)\n")

#----------------------------------
# Grid Resolution Reference
#----------------------------------
res_x <- mean(unique(diff(sort(unique(pred_data$X)))))
res_y <- mean(unique(diff(sort(unique(pred_data$Y)))))

#----------------------------------
# Prediction Loop over Models
#----------------------------------
pred_matrix_list <- list()   # store predictions for averaging

for (m in seq_along(models_to_load)) {
  model_path <- models_to_load[m]
  cat("\n--- Running model", m, "of", length(models_to_load), ":", basename(model_path), "---\n")
  
  xgb_model <- readRDS(model_path)
  model_features <- xgb_model$feature_names
  missing_feats <- setdiff(model_features, names(pred_data))
  
  if (length(missing_feats) > 0) {
    warning("Skipping model ", basename(model_path), 
            " because missing features: ", paste(missing_feats, collapse = ", "))
    next
  }
  
  # Prepare input matrix
  pred_matrix <- as.matrix(pred_data[, ..model_features])
  
  # Predict
  preds <- predict(xgb_model, xgb.DMatrix(pred_matrix, missing = NA))
  pred_matrix_list[[basename(model_path)]] <- preds
  
  #----------------------------------
  # Save individual model raster
  #----------------------------------
  df_out <- data.table(x = pred_data$X, y = pred_data$Y, z = preds)
  r_out <- rasterFromXYZ(df_out, res = c(res_x, res_y))
  
  out_tif <- file.path(out_dir, paste0(tools::file_path_sans_ext(basename(model_path)), "_pred.tif"))
  writeRaster(r_out, out_tif, format = "GTiff", overwrite = TRUE)
  cat("Saved:", out_tif, "\n")
}

#----------------------------------
# Ensemble Mean Raster
#----------------------------------
if (length(pred_matrix_list) > 0) {
  pred_df <- as.data.frame(pred_matrix_list)
  mean_preds <- rowMeans(pred_df, na.rm = TRUE)
  
  df_mean <- data.table(x = pred_data$X, y = pred_data$Y, z = mean_preds)
  r_mean <- rasterFromXYZ(df_mean, res = c(res_x, res_y))
  
  mean_tif <- file.path(out_dir, "ensemble_mean_prediction.tif")
  writeRaster(r_mean, mean_tif, format = "GTiff", overwrite = TRUE)
  cat("\n✅ Ensemble mean raster saved to:", mean_tif, "\n")
} else {
  warning("No predictions were generated (check feature mismatches).")
}
