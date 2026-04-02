# Script split into 3  stages, pre processing of raw NOW Coast data, then running of the Machine Learning MOdel (ML)
# Raw data downlaoded into a single folder comprising of GeoTiffs and seperate Raster Attribute Tables .xml files.

# WORKFLOW:----

# PRE-PROCESSING (STAGE 1): - (This will be its own engine)
# 1. Extract all survey end date data from xml (Raster Attribute Table) and create indavidaul rasters
#    Merge all individual survey end date rasters into a combined raster (currently mosaic via ArcGIS)
# 2. Standardize all model rasters (created in GIS /other) and clip to both prediction.mask and training.mask
     #ensure all data sets are in the same crs / projection, resolution (5m), and clip to different masks to ensure same size
#    also ensure all bathy elevation values above 0m are removed after clipping, if not filter (0m mark varies on boundary)
# 3. Convert prediction.mask and training.mask in to a Spatial Points Dataframe for sub grid processing
# 4. Load the blue topo grid tile gpkg and create a sub grid (by dividing it into 4 squares) for both the prediction and training mask extent
#    Create subset data frames of all the processed raster data (model variables), into the each sub grid tile folder, over the grid extent for model training

 
# ********** the below code is for stage 2 only**********
# MODEL TRAINING, GENERATING TILE PDPS AND EVALUATION METRICS (STAGE 2): (This will be its own engine)
# 1. Train the model over all sub grids
# 2. Create Partial Dependance plots from the model data - WE HAVE THEM ALL FOR EACH TILE, BUT THE AVERAGE PDP OVER STUDY AREA NEEDs COMPUTED
# 3. Evaluate model performance from Ranger RF summary data - PERFORMANCE METRICS ARE SAVED TO CSV, BUT MODEL AND METRICS NEED INTEROGATED TO INFORM CHANGES BEFORE PREDICTION CODE RUN 


# MODEL PREDICTION (STAGE 3): (This will be its own engine)
# 1. NEEDS UPDATED---  Generate the 10 year average trend of change: from (i) the training /JALBTCX temporal datasets[[actual avg change]], and (ii) the model predictions over the training extent[[predicted avg change]]
# 2. NEEDS UPDATED--- Apply prediction using 10 year average trend, survey age (survey end date), and model varaible relationships to determine change in depth over prediction extent
# 3. Prediction Validation - process needs created. probably best ot compare to how well its predicted over the training extent and compare back. 


# Jan 2025 modifications:
# Even though code was modified to process data on a tile by tile approach, it was still running of a master dataframe datasets created from the raster mosaic. Once 
# at the prediction stage this caused bottlenecks and extremely slow processing. In this iteration all data set creation and functions have been modified to subset and save
# all data within the bounds of each tile, and run solely on the information of each tile. 


# Load Packages
library(raster)
require(xml2)
library(dplyr)
require(sp) 
require(ranger) 
require(sf)
require(mgcv)
require(stringr)
library(progress) # For a progress bar
library(pdp)
library(data.table)
library(ggplot2)
library(tidyr)
library(readr)
library(purrr)
library(future.apply)
library(pbapply)
library(tools)    # For file path functions
library(leaps)
library(doParallel)
library(foreach)
library(RANN)
library(FNN)  # For weighted KNN
library(fst) # faster reading and writing of files than .rds
library(progressr)  # 🔥 Progress bar for parallel jobs
# library(conflicted) # helps overcome conflict issues between packages.

# new STAGE 1 - PRE-PROCESSING MODUL IN PARALLEL----
# ─────────────────────────────────────────────
# PREPROCESSING MODULE: Full Workflow (Steps 1–5)
# ─────────────────────────────────────────────

# ──────────────
# 1. LOAD PACKAGES & GLOBAL PARAMS
# ──────────────
suppressPackageStartupMessages({
  library(raster)
  library(sp)
  library(dplyr)
  library(sf)
  library(pbapply)
  library(foreach)
  library(doParallel)
  library(future)
  library(future.apply)
  library(progressr)
  library(xml2)
  library(doParallel)
})

# Global Settings
log_file <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/preprocessing_log.txt"

# Logging helper
log_message <- function(msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- sprintf("[%s] %s\n", timestamp, msg)
  cat(line)
  write(line, file = log_file, append = TRUE)
}

# ──────────────
# 2. DEFINE DIRECTORIES
# ──────────────
# Data to fill NA values in bathy (partial processed)
shapefile_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Pilot_model_prediction_boundary_Final.shp"
input_partial <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/part_processed"
output_filled <- input_partial
# Raw data to be processed 
input_raw_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # prediction data used to create training data clipped
input_raw_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw"
# Processed outputs
output_proc_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed"
output_proc_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed"
# Training and prediction masks
output_mask_train_utm <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif"
output_mask_train_wgs <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.WGS84_8m.tif"
output_mask_pred_utm <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif"
output_mask_pred_wgs <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.WGS84_8m.tif"
# training and prediction spatial dataframes
output_SPDF - "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data"
# KML / XML survey end date paths
input_dir_survey <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/UTM17"
kml_dir_survey   <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/RATs"
output_dir_survey_dates <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/survey_date_end"
# training and prediction sub grid GeoPackage 
training_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
prediction_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
# Sub grid tile folder directories
training_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
prediction_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"

# ──────────────
# 3. DEFINE FUNCTIONS
# ──────────────

## F1 - Function that will attempt to run in parallel, to fill NA values in raw bathymetry data using focal statistics but if memory limit reached
        # will fallback to sequential focal processing with a single iteration at a time.
# ──────────────
# F1.1 FULL ITERATIVE FOCAL FILL (HEAVY)
# ──────────────
iterative_focal_fill <- function(r, max_iters = 10, w = 3) {
  kernel <- matrix(1, w, w)
  for (i in seq_len(max_iters)) {
    if (sum(is.na(values(r))) == 0) break
    filled <- focal(r, w = kernel, fun = mean, na.rm = TRUE, NAonly = TRUE, pad = TRUE, padValue = NA)
    r <- overlay(r, filled, fun = function(orig, interp) ifelse(is.na(orig), interp, orig))
  }
  return(r)
}
# ──────────────
# F1.2 LIGHTWEIGHT FOCAL FILL (REPEATED SINGLE PASS)
# ──────────────
repeat_disk_focal_fill <- function(input_file, output_final, output_dir, n_repeats = 5, w = 3, layer_name = "unknown") {
  input_raster <- raster(input_file)
  temp_file <- input_file
  
  for (i in seq_len(n_repeats)) {
    log_message(paste("🔁", layer_name, "- Disk-Based Focal Fill Iteration", i, "of", n_repeats))
    
    out_path <- file.path(output_dir, paste0(tools::file_path_sans_ext(basename(input_file)), "_f", i, ".tif"))
    
    # Prevent overwrite issues by deleting the output if it exists
    if (file.exists(out_path)) file.remove(out_path)
    
    tryCatch({
      r <- raster(temp_file)
      filled <- focal(r, w = matrix(1, w, w), fun = mean, na.rm = TRUE, NAonly = TRUE, pad = TRUE, padValue = NA)
      final <- overlay(r, filled, fun = function(orig, interp) ifelse(is.na(orig), interp, orig))
      writeRaster(final, filename = out_path, format = "GTiff", overwrite = TRUE)
      
      # Set current output as input for next round
      temp_file <- out_path
      
      # Clean up memory & temp files
      removeTmpFiles(h = 0)
      gc()
    }, error = function(e) {
      log_message(paste("❌ Focal failed at iteration", i, "-", e$message))
    })
  }
  
  # Rename final result
  if (file.exists(temp_file)) {
    file.rename(temp_file, output_final)
    log_message(paste("✅ Final filled raster saved as:", basename(output_final)))
  } else {
    log_message(paste("❌ Final result file was not created for", layer_name))
  }
}


# ──────────────
# F1.3 FILL WITH FALLBACK STRATEGY
# ──────────────
fill_with_fallback <- function(input_file, output_file, max_iters = 10, fallback_repeats = 5, w = 3) {
  layer_name <- basename(input_file)
  
  tryCatch({
    r <- raster(input_file)
    log_message(paste("💪 Attempting iterative fill for", layer_name))
    r_filled <- iterative_focal_fill(r, max_iters = max_iters, w = w)
    writeRaster(r_filled, filename = output_file, format = "GTiff", overwrite = TRUE)
    log_message(paste("✅ Iterative fill succeeded:", layer_name))
  }, error = function(e) {
    log_message(paste("⚠️ Iterative fill failed for", layer_name, "-", e$message))
    
    # Fallback disk-based approach
    tryCatch({
      log_message(paste("🛠️  Fallback disk-based fill starting for", layer_name))
      repeat_disk_focal_fill(
        input_file = input_file,
        output_final = output_file,
        output_dir = dirname(output_file),
        n_repeats = fallback_repeats,
        w = w,
        layer_name = layer_name
      )
      
    }, error = function(e2) {
      log_message(paste("❌ Fallback fill also failed for", layer_name, "-", e2$message))
    })
  })
  
  removeTmpFiles(h = 0)
  gc()
}

# ──────────────
# F1.4 MAIN GAP FILL FUNCTION (TRY PARALLEL FIRST)
# ──────────────
cores <- 8  # or even 1 to start safely
run_gap_fill <- function(bathy_files,output_dir, cores = 8,max_iters = 10, fallback_repeats = 10, w = 3) {
  
  log_message(" Starting gap fill module...")
  
  # Try PARALLEL
  parallel_success <- TRUE
  results <- tryCatch({
    cl <- parallel::makeCluster(cores)
    doParallel::registerDoParallel(cl)
    
    result <- foreach::foreach(i = seq_along(bathy_files),
                               .packages = c("raster")) %dopar% {
                                 file <- bathy_files[i]
                                 output_file <- file.path(output_dir, paste0(tools::file_path_sans_ext(basename(file)), "_filled.tif"))
                                 # Since log_message not visible in cluster, don't use it here
                                 tryCatch({
                                   r <- raster(file)
                                   r_filled <- iterative_focal_fill(r, max_iters = max_iters, w = w)
                                   writeRaster(r_filled, filename = output_file, format = "GTiff", overwrite = TRUE)
                                   TRUE
                                 }, error = function(e) {
                                   FALSE
                                 })
                               }
    parallel::stopCluster(cl)
    result
  }, error = function(e) {
    log_message(paste(" Parallel processing failed:", e$message))
    parallel_success <<- FALSE
    return(NULL)
  })
  
  # SEQUENTIAL FALLBACK if needed
  if (!parallel_success || any(unlist(results) == FALSE)) {
    log_message("🕳️ Falling back to sequential fill method...")
    for (file in bathy_files) {
      output_file <- file.path(output_dir, paste0(tools::file_path_sans_ext(basename(file)), "_filled.tif"))
      fill_with_fallback(file, output_file, max_iters = max_iters, fallback_repeats = fallback_repeats, w = w)
    }
  }
  
  log_message(" Gap fill process complete.")
}

## F2 - Create Training Mask (and WGS Copy)
create_training_mask <- function(input_dir, output_mask_utm, output_mask_wgs, pattern = "_filled\\.tif$") {
  files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  stopifnot(length(files) > 0)
  
  mask <- calc(stack(files), fun = function(x) if (all(is.na(x))) NA else 1)
  binary_mask <- calc(mask, fun = function(x) ifelse(!is.na(x), 1, 0))
  
  # UTM version
  binary_mask_utm <- projectRaster(binary_mask, crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs", res = 8, method = "ngb")
  writeRaster(binary_mask_utm, output_mask_utm, format = "GTiff", overwrite = TRUE)
  log_message(paste("Training mask saved (UTM):", output_mask_utm))
  
  # WGS copy
  binary_mask_wgs <- projectRaster(binary_mask_utm, crs = "+init=epsg:4326", method = "ngb")
  writeRaster(binary_mask_wgs, output_mask_wgs, format = "GTiff", overwrite = TRUE)
  log_message(paste("Training mask saved (WGS84):", output_mask_wgs))
}

## F3 - Create Prediction Mask from Shapefile
create_prediction_mask <- function(shapefile_path, output_mask_utm, output_mask_wgs) {
  poly <- st_read(shapefile_path, quiet = TRUE)
  poly_utm <- st_transform(poly, crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs")
  ext <- extent(poly_utm)
  template <- raster(ext, res = 8, crs = projection(poly_utm))
  
  mask_ras <- rasterize(poly_utm, template, field = 1, background = NA)
  mask_bin <- calc(mask_ras, fun = function(x) ifelse(is.na(x), 0, 1))
  
  writeRaster(mask_bin, output_mask_utm, overwrite = TRUE)
  writeRaster(projectRaster(mask_bin, crs = "+init=epsg:4326", method = "ngb"), output_mask_wgs, overwrite = TRUE)
  log_message("Prediction mask created (UTM & WGS)")
}

## F4 - Create Spatial DF for Masks (UTM + WGS)
create_spatial_mask_df <- function(mask_utm_path = NULL, mask_wgs_path = NULL, mask_type = "prediction", output_dir = ".") {
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  if (!is.null(mask_utm_path)) {
    r <- raster::raster(mask_utm_path)
    pts <- raster::rasterToPoints(r, spatial = TRUE)
    df <- data.frame(pts@data, X = pts@coords[, 1], Y = pts@coords[, 2])
    df$FID <- raster::cellFromXY(r, df[, c("X", "Y")])
    df <- df[df[, 1] == 1, ]
    
    out_path_utm <- file.path(output_dir, paste0(mask_type, ".mask.df.utm.fst"))
    write.fst(df, out_path_utm)
    log_message(paste("Spatial UTM DF saved to", out_path_utm))
  }
  
  if (!is.null(mask_wgs_path)) {
    r <- raster::raster(mask_wgs_path)
    pts <- raster::rasterToPoints(r, spatial = TRUE)
    df <- data.frame(pts@data, X = pts@coords[, 1], Y = pts@coords[, 2])
    df$FID <- raster::cellFromXY(r, df[, c("X", "Y")])
    df <- df[df[, 1] == 1, ]
    
    out_path_wgs <- file.path(output_dir, paste0(mask_type, ".mask.df.wgs84.fst"))
    write.fst(df, out_path_wgs)
    log_message(paste("Spatial WGS84 DF saved to", out_path_wgs))
  }
}

## F5 - Extract survey end dates from Blue Topo xml files
extract_survey_end_dates <- function(input_dir, kml_dir, output_dir) {
  log_message(" Extracting survey end dates from TIFF + XML...")
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  tiff_files <- list.files(input_dir, pattern = "\\.tiff$", full.names = TRUE)
  
  for (tiff_file in tiff_files) {
    file_base <- basename(tiff_file)
    file_name <- tools::file_path_sans_ext(file_base)
    kml_file <- file.path(kml_dir, paste0(file_name, ".tiff.aux.xml"))
    
    if (!file.exists(kml_file)) {
      log_message(paste(" Missing XML for:", file_base))
      next
    }
    
    tryCatch({
      r <- raster::brick(tiff_file)
      contributor_band <- r[[3]]  # Band 3 = Contributor
      
      xml_data <- xml2::read_xml(kml_file)
      contributor_band_xml <- xml2::xml_find_all(xml_data, "//PAMRasterBand[Description='Contributor']")
      rows <- xml2::xml_find_all(contributor_band_xml, ".//GDALRasterAttributeTable/Row")
      
      table_data <- lapply(rows, function(row) {
        fields <- xml2::xml_find_all(row, ".//F")
        values <- xml2::xml_text(fields)
        list(
          value = as.numeric(values[1]),
          survey_date_end = as.Date(values[18], format = "%Y-%m-%d")
        )
      })
      
      attribute_table_df <- do.call(rbind, lapply(table_data, as.data.frame, stringsAsFactors = FALSE)) %>%
        mutate(
          survey_year_end = as.numeric(format(survey_date_end, "%Y")),
          survey_year_end = ifelse(is.na(survey_year_end), 0, survey_year_end)
        )
      
      reclass_matrix <- as.matrix(attribute_table_df %>% distinct(value, survey_year_end))
      year_raster <- raster::reclassify(contributor_band, rcl = reclass_matrix, right = FALSE)
      
      out_file <- file.path(output_dir, file_base)
      raster::writeRaster(year_raster, out_file, format = "GTiff", overwrite = TRUE)
      log_message(paste("Survey year raster saved:", file_base))
      
    }, error = function(e) {
      log_message(paste("Failed to process:", file_base, "-", e$message))
    })
  }
  
  log_message(" Survey date extraction complete.")
}

## F6 - Split Blue Topo Grid into sub grid (divides grid into 4 subgrids)
split_tile_into_quadrants <- function(tile) {
  bbox <- st_bbox(tile)
  dx <- (bbox["xmax"] - bbox["xmin"]) / 2
  dy <- (bbox["ymax"] - bbox["ymin"]) / 2
  
  coords_list <- list(
    matrix(c(bbox["xmin"], bbox["ymin"], bbox["xmin"] + dx, bbox["ymin"], bbox["xmin"] + dx, bbox["ymin"] + dy,
             bbox["xmin"], bbox["ymin"] + dy, bbox["xmin"], bbox["ymin"]), ncol = 2, byrow = TRUE),
    matrix(c(bbox["xmin"] + dx, bbox["ymin"], bbox["xmax"], bbox["ymin"], bbox["xmax"], bbox["ymin"] + dy,
             bbox["xmin"] + dx, bbox["ymin"] + dy, bbox["xmin"] + dx, bbox["ymin"]), ncol = 2, byrow = TRUE),
    matrix(c(bbox["xmin"] + dx, bbox["ymin"] + dy, bbox["xmax"], bbox["ymin"] + dy, bbox["xmax"], bbox["ymax"],
             bbox["xmin"] + dx, bbox["ymax"], bbox["xmin"] + dx, bbox["ymin"] + dy), ncol = 2, byrow = TRUE),
    matrix(c(bbox["xmin"], bbox["ymin"] + dy, bbox["xmin"] + dx, bbox["ymin"] + dy, bbox["xmin"] + dx, bbox["ymax"],
             bbox["xmin"], bbox["ymax"], bbox["xmin"], bbox["ymin"] + dy), ncol = 2, byrow = TRUE)
  )
  
  sub_polys <- lapply(coords_list, function(coords) st_polygon(list(coords)))
  ids <- paste0(tile$tile, "_", 1:4)
  
  df <- data.frame(tile_id = ids, original_tile = tile$tile)
  st_sf(df, geometry = st_sfc(sub_polys, crs = 4326))
}

## F7- Intersects blue topo grid with training or prediction mask to make new Geopackage (in WGS84)
prepare_subgrids <- function(grid_tiles, mask_df, output_dir) {
  log_message("🧩 Preparing sub-grids from tile scheme...")
  
  mask_sf <- st_as_sf(mask_df, coords = c("X", "Y"), crs = 4326)
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) split_tile_into_quadrants(grid_tiles[i, ])))
  
  log_message(paste("🔹 Total sub-grids generated:", nrow(sub_grids)))
  
  intersecting <- st_filter(sub_grids, st_union(mask_sf))
  log_message(paste("🔸 Sub-grids intersecting mask:", nrow(intersecting)))
  
  out_gpkg <- file.path(output_dir, "intersecting_sub_grids_WGS84.gpkg")
  out_rds  <- file.path(output_dir, "grid_tile_extents_WGS84.rds")
  
  st_write(intersecting, out_gpkg, delete_layer = TRUE, quiet = TRUE)
  saveRDS(intersecting, out_rds)
  
  log_message(paste("✅ Saved sub-grids to:", out_gpkg))
  return(intersecting)
}

## F8 - Re projects sub grid geopackage into desired projection - UTM 
reproject_subgrids_to_utm <- function(input_gpkg, output_gpkg, target_crs) {
  log_message(paste("🌍 Reprojecting sub-grids to:", target_crs))
  
  tryCatch({
    sub_grids <- st_read(input_gpkg, quiet = TRUE)
    sub_grids_utm <- st_transform(sub_grids, crs = target_crs)
    st_write(sub_grids_utm, output_gpkg, delete_layer = TRUE, quiet = TRUE)
    log_message(paste("✅ Reprojected sub-grids saved to:", output_gpkg))
  }, error = function(e) {
    log_message(paste("❌ Failed to reproject sub-grids:", e$message))
  })
}

## F9 - Standardize Rasters (Parallel)
standardize_rasters <- function(mask_path, input_dir_raw, input_dir_partial, output_dir) {
  mask <- raster(mask_path)
  crs_mask <- crs(mask)
  
  non_bathy <- list.files(input_dir_raw, pattern = "^(?!bathy_).*\\.tif$", perl = TRUE, full.names = TRUE)
  bathy <- list.files(input_dir_partial, pattern = "^bathy_.*\\.tif$", full.names = TRUE)
  all_files <- c(non_bathy, bathy)
  
  progressr::with_progress({
    future_pblapply(seq_along(all_files), function(i) {
      f <- all_files[i]
      tryCatch({
        r <- raster(f)
        if (!compareCRS(r, mask)) r <- projectRaster(r, crs = crs_mask, method = "bilinear")
        r <- resample(r, mask, method = "bilinear")
        r <- raster::mask(r, mask)
        if (grepl("^bathy_", basename(f))) r[r > 0] <- NA
        writeRaster(r, filename = file.path(output_dir, basename(f)), overwrite = TRUE)
        log_message(paste("Standardized:", basename(f)))
      }, error = function(e) {
        log_message(paste("Failed to standardize:", basename(f), "-", e$message))
      })
    })
  })
}

## F10 - Parallel Tile Chunking (Spatial Dataframes)
grid_out_raster_data <- function(sub_grid_gpkg, raster_dir, output_dir, data_type) {
  grids <- st_read(sub_grid_gpkg)
  rasters <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  
  foreach(i = seq_len(nrow(grids)), .packages = c("raster", "sf", "dplyr")) %dopar% {
    tile <- grids[i, ]
    name <- tile$tile_id
    extent_poly <- as(extent(st_bbox(tile)), "SpatialPolygons")
    crs(extent_poly) <- st_crs(grids)$proj4string
    
    dir.create(file.path(output_dir, name), showWarnings = FALSE, recursive = TRUE)
    clipped_data <- lapply(rasters, function(rf) {
      r <- raster(rf)
      if (is.null(intersect(extent(r), extent_poly))) return(NULL)
      r_crop <- crop(r, extent_poly)
      df <- as.data.frame(rasterToPoints(r_crop))
      colnames(df) <- c("X", "Y", tools::file_path_sans_ext(basename(rf)))
      df$FID <- cellFromXY(r, df[, 1:2])
      return(df)
    })
    clipped_data <- Filter(Negate(is.null), clipped_data)
    if (length(clipped_data) == 0) return(NULL)
    
    combined <- Reduce(function(x, y) merge(x, y, by = c("X", "Y", "FID"), all = TRUE), clipped_data)
    if (data_type == "training") {
      combined <- combined %>% select(-starts_with("bt")) %>%
        mutate(b.change.2004_2006 = bathy_2006 - bathy_2004,
               b.change.2006_2010 = bathy_2010 - bathy_2006,
               b.change.2010_2015 = bathy_2015 - bathy_2010,
               b.change.2015_2022 = bathy_2022 - bathy_2015)
    }
    write.fst(combined, file.path(output_dir, name, paste0(name, "_", data_type, "_clipped_data.fst")))
    log_message(paste("Chunk processed:", name))
  }
}

# ──────────────
# 4. RUN MODULE FUNCTIONS
# ──────────────
#initiate parallel processing 
cores <- 8  # or even 1 to start safely
cl <- makeCluster(cores)

handlers(global = TRUE)
plan(multisession, workers = cores)
registerDoParallel(cl)

start_time <- Sys.time()
log_message(" Starting preprocessing module...")


# F1 - FOCAL GAP FILL (uses ~6-8GB of RAM and takes 2.5hrs per bathy tiff [pilot model extent])
bathy_files <- list.files(input_raw_pred, pattern = "^bathy_\\d{4}\\.tif$", full.names = TRUE)
run_gap_fill(bathy_files, output_dir = output_filled, cores = 8, max_iters = 5)
log_message(" Final cleanup of temp raster files...")
cleanup_intermediate_rasters <- function(base_name, dir) {
  files <- list.files(dir, pattern = paste0("^", base_name, "_f\\d+\\.tif$"), full.names = TRUE)
  file.remove(files)
}
gc()

# F2 - TRAINING MASK
create_training_mask(input_partial, output_mask_train_utm, output_mask_train_wgs)

# F3 - PREDICTION MASK
create_prediction_mask(shapefile_path, output_mask_pred_utm, output_mask_pred_wgs)

# F4 - SPATIAL DATAFRAMES
create_spatial_mask_df(mask_utm_path = output_mask_pred_utm,mask_wgs_path = output_mask_pred_wgs,mask_type = "training", output_dir = output_SPDF)

create_spatial_mask_df(mask_utm_path = output_mask_pred_utm,mask_wgs_path = output_mask_pred_wgs,mask_type = "prediction", output_dir = output_SPDF)

# F5 - EXTRACT SURVEY END DATES
extract_survey_end_dates(input_dir = input_dir_survey, kml_dir = kml_dir_survey, output_dir = output_dir_survey_dates)

# F6 & F7 - PREPARE SUB-GRIDS (Training & Prediction Masks)
# OG Blue Topo Gpkg
grid_tile_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg"
grid_tiles <- st_read(grid_tile_path, quiet = TRUE)

training_mask_df_wgs <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.wgs84.rds")
prediction_mask_df_wgs <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/prediction.mask.df.wgs84.rds")

training_subgrids_wgs <- prepare_subgrids(grid_tiles = grid_tiles, mask_df = training_mask_df_wgs, output_dir = training_subgrid_out)

prediction_subgrids_wgs <- prepare_subgrids(grid_tiles = grid_tiles, mask_df = prediction_mask_df_wgs, output_dir = prediction_subgrid_out)

# F8 - REPROJECT SUB GRIDS to UTM
utm_crs_str <- "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"

reproject_subgrids_to_utm(
  input_gpkg = file.path(training_subgrid_out, "intersecting_sub_grids_WGS84.gpkg"),
  output_gpkg = file.path(training_subgrid_out, "intersecting_sub_grids_UTM.gpkg"),
  target_crs = utm_crs_str
)

reproject_subgrids_to_utm(
  input_gpkg = file.path(prediction_subgrid_out, "intersecting_sub_grids_WGS84.gpkg"),
  output_gpkg = file.path(prediction_subgrid_out, "intersecting_sub_grids_UTM.gpkg"),
  target_crs = utm_crs_str
)
 
# F9 -  STANDARDIZE RASTERS
standardize_rasters(output_mask_pred_utm, input_raw_pred, input_partial, output_proc_pred)
standardize_rasters(output_mask_train_utm, input_raw_train, input_partial, output_proc_train)

# F10 - RASTER CHUNK TILE DATA
grid_out_raster_data(training_grid_gpkg, output_proc_train, training_subgrid_out, "training")
grid_out_raster_data(prediction_grid_gpkg, output_proc_pred, prediction_subgrid_out, "prediction")

# ──────────────
# 5. FINISH / Close parallel 
# ──────────────
stopCluster(cl)
end_time <- Sys.time()
log_message(sprintf(" All preprocessing completed in %.1f minutes", as.numeric(difftime(end_time, start_time, units = "mins"))))


#---------------------------------------------


# PRE READY NEW STAGE 1 ( my piece parts before in parallel - DELETE AFTERWARD ONE ABOVE IS WORKING)
# Define input and output directories
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/UTM17" # this should point to the folder, where Stephen has downloaded all BT data for the grid automation tool
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/survey_date_end"
kml_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/RATs"

### naming convention of .tiff and .xml files much match ###

# Create a list of TIFF files
tiff_files <- list.files(input_dir, pattern = "\\.tiff$", full.names = TRUE)

# Iterate through TIFF files and process each one
for (tiff_file in tiff_files) {
  # Extract the base name of the file
  file_base <- basename(tiff_file)
  file_name <- tools::file_path_sans_ext(file_base)
  
  # Define the corresponding KML file path
  kml_file <- file.path(kml_dir, paste0(file_name, ".tiff.aux.xml"))
  
  # Read the raster file
  r <- brick(tiff_file)
  
  # Extract the 'contributor' band (Band 3)
  contributor_band <- r[[3]]
  
  # Read the XML file
  xml_data <- xml2::read_xml(kml_file)
  
  # Extract Raster Attribute Table Rows (e.g., survey dates)
  # Find each row in the GDALRasterAttributeTable for 'Contributor' band
  contributor_band_xml <- xml2::xml_find_all(xml_data, "//PAMRasterBand[Description='Contributor']")
  rows <- xml2::xml_find_all(contributor_band_xml, ".//GDALRasterAttributeTable/Row")
  
  # Extract specific fields from each row
  table_data <- lapply(rows, function(row) {
    fields <- xml2::xml_find_all(row, ".//F")
    field_values <- xml2::xml_text(fields)
    list(
      value = as.numeric(field_values[1]),
      survey_date_end = as.Date(field_values[18], format = "%Y-%m-%d")
    )
  })
  
  # Convert extracted data to a data frame for easier analysis
  attribute_table_df <- do.call(rbind, lapply(table_data, as.data.frame, stringsAsFactors = FALSE))
  
  # Extract the year from survey_date_end
  attribute_table_df <- attribute_table_df %>%
    mutate(survey_year_end = as.numeric(format(survey_date_end, "%Y")))
  
  # Ensure no NA values in 'survey_year_end'
  attribute_table_df <- attribute_table_df %>%
    mutate(survey_year_end = ifelse(is.na(survey_year_end), 0, survey_year_end))
  
  
  attribute_table_df$survey_year_end <- round(attribute_table_df$survey_year_end, digits = 2)
  
  
  # Create a lookup table to map raster values to the survey end year
  date_mapping <- attribute_table_df %>%
    select(value, survey_year_end) %>%
    distinct()  # Ensure unique mapping
  
  # Convert to matrix for compatibility with reclassify
  reclass_matrix <- as.matrix(date_mapping[, c("value", "survey_year_end")])
  
  # Reclassify the raster using the cleaned and year mapping
  year_raster <- reclassify(
    contributor_band,
    rcl = reclass_matrix,
    right = FALSE
  )
  
  # Define output file path with the same naming convention
  output_file <- file.path(output_dir, file_base)
  
  # Save the new year raster as a TIFF file
  writeRaster(year_raster, output_file, format = "GTiff", overwrite = TRUE)
}

print("All Survey End Date files processed and saved with the same naming convention.")

# ───────────────────────────────────────────────
#   PREPROCESSING MODULE: Full Workflow
# ───────────────────────────────────────────────

# Load Required Packages
suppressPackageStartupMessages({
  library(raster)
  library(rgdal)
  library(dplyr)
  library(doParallel)
  library(foreach)
  library(pbapply)
  library(sf)
})

# ─────────────────────────────
# 1. Define Global Parameters
# ─────────────────────────────
cores <- 8
log_file <- "gap_fill_errors.log"



# Ensure output dir exists
if (!dir.exists(dirs$output_filled)) dir.create(dirs$output_filled, recursive = TRUE)

#  make sure the link points to the newest dataset----
grid_gpkg <- st_read("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg") # from Blue topo
#
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
output_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"
input_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed" # raster data 
input_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # raster data 
# ─────────────────────────────
# 2. Initialize Parallel Processing
# ─────────────────────────────
cl <- makeCluster(cores)
registerDoParallel(cl)

# ─────────────────────────────
# 3. Define Functions
# ─────────────────────────────

# F1 - Focal Gap-Fill Function (Iterative)
iterative_focal_fill <- function(r, max_iters = 10, w = 3) {
  kernel <- matrix(1, w, w)
  for (i in seq_len(max_iters)) {
    na_count <- sum(is.na(values(r)))
    if (na_count == 0) break
    filled <- focal(r, w = kernel, fun = mean, na.rm = TRUE, NAonly = TRUE, pad = FALSE)
    r <- overlay(r, filled, fun = function(orig, interp) ifelse(is.na(orig), interp, orig))
  }
  return(r)
} (# CAn we add in if NA values present, iter and fill?)

# F2 - Raster Fill Wrapper with Error Logging
fill_internal_na_parallel <- function(input_file, output_file, max_iters = 5, w = 3, log_file = NULL) {
  tryCatch({
    r <- raster(input_file)
    filled <- iterative_focal_fill(r, max_iters = max_iters, w = w)
    writeRaster(filled, filename = output_file, format = "GTiff", overwrite = TRUE)
  }, error = function(e) {
    msg <- sprintf("❌ ERROR in %s: %s\n", basename(input_file), e$message)
    cat(msg)
    if (!is.null(log_file)) write(msg, file = log_file, append = TRUE)
  })
}

# F3 - Training Mask Creation 
create_training_mask <- function(input_dir, 
                                 output_mask_utm, 
                                 output_mask_wgs,
                                 pattern = "_filled\\.tif$",
                                 target_crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs",
                                 target_res = 8) {
  
  # List input rasters
  files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) stop("❌ No raster files found.")
  
  # Stack rasters
  ras_stack <- stack(files)
  
  # Create raw binary mask (presence/absence of data)
  mask <- calc(ras_stack, fun = function(x) if (all(is.na(x))) NA else 1)
  binary_mask <- calc(mask, fun = function(x) ifelse(!is.na(x), 1, 0))
  
  # Project to UTM with 8m resolution
  binary_mask_utm <- projectRaster(
    binary_mask,
    crs = target_crs,
    res = target_res,
    method = "ngb"
  )
  
  # Save UTM version
  writeRaster(binary_mask_utm,
              filename = output_mask_utm,
              format = "GTiff",
              overwrite = TRUE)
  cat("✅ Training mask saved in UTM:", output_mask_utm, "\n")
  
  # Reproject to WGS84 (EPSG:4326)
  binary_mask_wgs <- projectRaster(
    binary_mask_utm,
    crs = CRS("+init=epsg:4326"),
    method = "ngb"
  )
  
  # Save WGS version
  writeRaster(binary_mask_wgs,
              filename = output_mask_wgs,
              format = "GTiff",
              overwrite = TRUE)
  cat("✅ Training mask saved in WGS84:", output_mask_wgs, "\n")
}


# F4 - Prediction Mask Creation 
create_prediction_mask <- function(shapefile_path,
                                   output_mask_utm,
                                   output_mask_wgs,
                                   res = 8,
                                   utm_crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs",
                                   wgs_crs = "+init=epsg:4326") {
  
  # Load the boundary shapefile
  poly <- sf::st_read(shapefile_path, quiet = TRUE)
  
  # Reproject to UTM
  poly_utm <- sf::st_transform(poly, crs = utm_crs)
  
  # Create an empty raster grid covering the polygon extent
  ext <- raster::extent(poly_utm)
  template_ras <- raster::raster(ext, crs = utm_crs,
                                 res = res)
  
  # Rasterize: 1 inside polygon, NA outside
  mask_ras <- rasterize(poly_utm, template_ras, field = 1, background = NA)
  mask_bin <- calc(mask_ras, fun = function(x) ifelse(is.na(x), 0, 1))
  
  # Save UTM binary mask
  writeRaster(mask_bin,
              filename = output_mask_utm,
              format = "GTiff",
              overwrite = TRUE)
  cat("✅ Prediction mask saved (UTM):", output_mask_utm, "\n")
  
  # Reproject to WGS84
  mask_bin_wgs <- projectRaster(mask_bin,
                                crs = CRS(wgs_crs),
                                method = "ngb")
  
  # Save WGS84 binary mask
  writeRaster(mask_bin_wgs,
              filename = output_mask_wgs,
              format = "GTiff",
              overwrite = TRUE)
  cat("✅ Prediction mask saved (WGS84):", output_mask_wgs, "\n")
}

# F5 - Standardise Rasters 
standardize_rasters <- function(mask_path,
                                input_dir_raw,
                                input_dir_partial,
                                output_dir,
                                pattern_nonbathy = "^(?!bathy_).*\\.tif$",
                                pattern_bathy = "^bathy_.*\\.tif$") {
  
  # Load mask raster
  mask <- raster(mask_path)
  crs_mask <- crs(mask)
  
  # Get rasters to process
  nonbathy_files <- list.files(input_dir_raw, pattern = pattern_nonbathy, full.names = TRUE, perl = TRUE)
  bathy_files <- list.files(input_dir_partial, pattern = pattern_bathy, full.names = TRUE)
  
  all_files <- c(nonbathy_files, bathy_files)
  
  if (length(all_files) == 0) {
    stop("❌ No rasters found to process.")
  }
  
  if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  
  total_rasters <- length(all_files)
  start_time <- Sys.time()
  
  # Raster standardization function
  process_raster <- function(i) {
    file <- all_files[i]
    source_name <- tools::file_path_sans_ext(basename(file))
    cat(sprintf("\nProcessing [%d/%d]: %s\n", i, total_rasters, source_name))
    
    tryCatch({
      ras <- raster(file)
      
      # Project if needed
      if (!compareCRS(ras, mask)) {
        ras <- projectRaster(ras, crs = crs_mask, method = "bilinear")
      }
      
      # Resample to match resolution and extent
      ras <- resample(ras, mask, method = "bilinear")
      
      # Mask to training/prediction area
      temp_ras <- raster::mask(ras, mask)
      
      # Apply threshold for bathy layers
      if (grepl("^bathy_", source_name)) {
        temp_ras[temp_ras > 0] <- NA
      }
      
      # Save output
      output_name <- file.path(output_dir, paste0(source_name, ".tif"))
      writeRaster(temp_ras, filename = output_name, format = "GTiff", overwrite = TRUE)
      
      # Estimate and display progress
      elapsed <- difftime(Sys.time(), start_time, units = "mins")
      avg <- as.numeric(elapsed) / i
      remaining <- avg * (total_rasters - i)
      cat(sprintf("✅ [%d/%d]: %s — ETA: ~%.1f min\n", i, total_rasters, source_name, remaining))
      
    }, error = function(e) {
      cat(sprintf("❌ ERROR [%s]: %s\n", source_name, e$message))
    })
  }
  
  # Run processing sequentially (or wrap in parallel if needed)
  pblapply(seq_along(all_files), process_raster, cl = 1)
  
  # Check extent consistency
  processed_files <- list.files(output_dir, pattern = "\\.tif$", full.names = TRUE)
  rasters <- lapply(processed_files, raster)
  ext_ref <- extent(rasters[[1]])
  
  all_same_extent <- all(sapply(rasters, function(r) identical(extent(r), ext_ref)))
  
  if (all_same_extent) {
    cat("✅ All rasters have matching extents.\n")
  } else {
    warning("⚠️ Rasters do not have consistent extents!")
  }
  
  total_time <- difftime(Sys.time(), start_time, units = "mins")
  cat(sprintf("\n🕒 All rasters processed in %.1f minutes.\n", total_time))
}


# Below Functions to prepare and save a new sub-grid gpkg 
#(master blue topo grid tile divided by 4, same tile I.Ds with sub identifier _1,_2_3,_4 in clockwise order)

# F6- Split a single tile into 4 equal sub-grids, starting top left in clockwise order
split_tile <- function(tile) {
  # Get tile bounding box
  bbox <- st_bbox(tile)
  
  # Compute sub-grid size (divide original size by 2)
  sub_width <- (bbox["xmax"] - bbox["xmin"]) / 2
  sub_height <- (bbox["ymax"] - bbox["ymin"]) / 2
  
  # Define 4 sub-grids
  sub_grids <- list(
    st_polygon(list(matrix(c(
      bbox["xmin"], bbox["ymin"],
      bbox["xmin"] + sub_width, bbox["ymin"],
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmin"], bbox["ymin"] + sub_height,
      bbox["xmin"], bbox["ymin"]
    ), ncol = 2, byrow = TRUE))),
    
    st_polygon(list(matrix(c(
      bbox["xmin"] + sub_width, bbox["ymin"],
      bbox["xmax"], bbox["ymin"],
      bbox["xmax"], bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymin"]
    ), ncol = 2, byrow = TRUE))),
    
    st_polygon(list(matrix(c(
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmax"], bbox["ymin"] + sub_height,
      bbox["xmax"], bbox["ymax"],
      bbox["xmin"] + sub_width, bbox["ymax"],
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height
    ), ncol = 2, byrow = TRUE))),
    
    st_polygon(list(matrix(c(
      bbox["xmin"], bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymax"],
      bbox["xmin"], bbox["ymax"],
      bbox["xmin"], bbox["ymin"] + sub_height
    ), ncol = 2, byrow = TRUE)))
  )
  
  # Create sub-grid tile IDs
  sub_grid_ids <- paste0(tile$tile, "_", 1:4)
  
  # Preserve attributes
  df <- data.frame(tile_id = sub_grid_ids, original_tile = tile$tile)
  df_sf <- st_as_sf(df, geometry = st_sfc(sub_grids, crs = 4326))
  
  return(df_sf)
}

# F7 - Prepare and save sub-grids
prepare_subgrids <- function(grid_gpkg, mask.df, output_dir) {
  cat("Preparing grid tiles and sub-grids...\n")
  
  # Read the original grid (Already in WGS 84)
  grid_tiles <- grid_gpkg
  
  # Convert mask to sf object (Already in WGS 84)
  mask_sf <- st_as_sf(mask.df, coords = c("X", "Y"), crs = 4326)
  
  #  Split ALL tiles into sub-grids
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) split_tile(grid_tiles[i, ])))
  
  # Debugging: Check if sub-grids exist before filtering
  cat("Total sub-grids generated:", nrow(sub_grids), "\n")
  
  #  Filter sub-grids that intersect with the mask
  intersecting_sub_grids <- st_filter(sub_grids, st_union(mask_sf))
  
  # Save outputs
  st_write(intersecting_sub_grids, file.path(output_dir, "intersecting_sub_grids_WGS84.gpkg"), delete_layer = TRUE, quiet = FALSE)
  saveRDS(intersecting_sub_grids, file.path(output_dir, "grid_tile_extents_WGS84.rds"))
  
  cat("✅ Sub-grids successfully prepared:", nrow(intersecting_sub_grids), "\n")
  return(intersecting_sub_grids)
}

# F8 - Reproject sub grid and save new GPKG as UTM
reproject_sub_grids <- function(input_gpkg, output_gpkg, target_crs = crs(training.mask.UTM)) {
  cat("Reading:", input_gpkg, "\n")
  
  # Load the original sub-grid (WGS84)
  sub_grids <- st_read(input_gpkg, quiet = TRUE)
  
  # 🚀 Reproject to UTM (same as the mask)
  sub_grids_utm <- st_transform(sub_grids, target_crs)
  
  # 🚀 Save the new UTM version
  cat("Saving:", output_gpkg, "\n")
  st_write(sub_grids_utm, output_gpkg, delete_layer = TRUE, quiet = TRUE)
  
  cat("✅ Reprojection complete:", output_gpkg, "\n\n")
}

# F9 - Process raster data, into a chunk size spatial datasets per tile folder
grid_out_raster_data <- function(sub_grid_gpkg, raster_dir, output_dir, data_type) {
  # Ensure sub_grid_gpkg is a valid path
  if (is.character(sub_grid_gpkg)) {
    sub_grids <- st_read(sub_grid_gpkg)
  } else if (inherits(sub_grid_gpkg, "sf")) {
    sub_grids <- sub_grid_gpkg
  } else {
    stop("`sub_grid_gpkg` must be a valid file path or an sf object.")
  }
  
  # List raster files in the input directory
  raster_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  
  # Process each tile
  for (i in seq_len(nrow(sub_grids))) {
    sub_grid <- sub_grids[i, ]
    tile_name <- sub_grid$tile_id  # Ensure sub-grid has a `tile_id` column
    tile_extent <- as(extent(st_bbox(sub_grid)), "SpatialPolygons")  # Convert extent to SpatialPolygons
    crs(tile_extent) <- st_crs(sub_grids)$proj4string  # Assign CRS
    
    # Create sub-folder for the tile if it doesn't exist
    tile_dir <- file.path(output_dir, tile_name)
    if (!dir.exists(tile_dir)) {
      dir.create(tile_dir, showWarnings = FALSE, recursive = TRUE)
    }
    
    # Path to save the clipped raster data
    clipped_data_path <- file.path(tile_dir, paste0(tile_name, "_", data_type, "_clipped_data.rds"))
    
    # Overwrite existing file if present
    if (file.exists(clipped_data_path)) {
      cat("Overwriting existing file for tile:", tile_name, "\n")
    }
    
    cat("Processing", data_type, "tile:", tile_name, "\n")
    
    # Clip rasters to the tile extent and process
    clipped_data <- lapply(raster_files, function(r_file) {
      r <- raster::raster(r_file)  # Load raster
      
      # 🔍 **Check if raster and tile overlap before cropping**
      if (is.null(raster::intersect(extent(r), tile_extent))) {
        cat("  ⚠️ Skipping raster (no overlap):", basename(r_file), "\n")
        return(NULL)  # Skip this raster
      }
      
      cropped_r <- raster::crop(r, tile_extent)  # Crop to tile extent
      
      # Extract raster values along with X and Y coordinates
      raster_data <- as.data.frame(raster::rasterToPoints(cropped_r, spatial = FALSE))
      raster_name <- tools::file_path_sans_ext(basename(r_file))  # Extract raster name
      colnames(raster_data) <- c("X", "Y", paste0(raster_name))  
      raster_data$FID <- raster::cellFromXY(r, raster_data[, c("X", "Y")])  # Add FID
      return(raster_data)
    })
    
    # Remove NULL elements (rasters that were skipped)
    clipped_data <- Filter(Negate(is.null), clipped_data)
    
    # Combine all rasters into a single data frame
    if (length(clipped_data) > 0) {
      combined_data <- Reduce(function(x, y) merge(x, y, by = c("X", "Y", "FID"), all = TRUE), clipped_data)
      
      # 🚀 **Remove specific columns based on `data_type`**
      if (data_type == "training") {
        combined_data <- combined_data %>% select(-starts_with("bt"))  # Remove "bt" columns
        
        # 🆕 **Create `b.change` columns**
        combined_data <- combined_data %>%
          mutate(
            b.change.2004_2006 = bathy_2006 - bathy_2004, # 2 years
            b.change.2006_2010 = bathy_2010 - bathy_2006, # 4 years
            b.change.2010_2015 = bathy_2015 - bathy_2010, # 5 years 
            b.change.2015_2022 = bathy_2022 - bathy_2015  # 7 years
          )
      } else if (data_type == "prediction") {
        combined_data <- combined_data %>% select(-starts_with("bathy"), -starts_with("slope"), -starts_with("rugosity"))  # Remove specified columns
      }
      
      # Save the combined data as RDS
      saveRDS(combined_data, file = clipped_data_path)
      cat("✅ Saved", data_type, "clipped data for tile:", tile_name, "\n")
    } else {
      cat("⚠️ No overlapping rasters for tile:", tile_name, "- Skipping saving.\n")
    }
  }
  
  cat("✅ Finished processing all", data_type, "tiles in", output_dir, "\n")
}


# ─────────────────────────────
# 4. Run Workflow Steps (Step x... using Function x..)
# ─────────────────────────────

  ## S1 (F1 & F2): GAP FILL PARALLEL
bathy_files <- list.files(dirs$input_raw, pattern = "bathy_\\d{4}\\.tif$", full.names = TRUE)

foreach(i = seq_along(bathy_files), .packages = "raster") %dopar% {
  file <- bathy_files[i]
  year <- tools::file_path_sans_ext(basename(file))
  out_file <- file.path(dirs$output_filled, paste0(year, "_filled.tif"))
  fill_internal_na_parallel(file, out_file, max_iters = 5, w = 3, log_file = log_file)
}

 ## S2 (F3): CREATE TRAINING MASK FROM INTERPOLATED BATHY
create_training_mask(
  input_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/model_varaibles/Prediction/part_processed",
  output_mask_utm = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif",
  output_mask_wgs = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.WGS84_8m.tif"
)


## S3 (F4): CREATE PEDICTION MASK (includes training and prediction extent)
create_prediction_mask(
  shapefile_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/GIS/Pilot_model/Pilot_model_prediction_boundary_Final.shp",
  output_mask_utm = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif",
  output_mask_wgs = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.WGS84_8m.tif"
)

## S4: Extract Survey End Date
# Define input and output directories
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/UTM17" # this should point to the folder, where Stephen has downloaded all BT data for the grid automation tool
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/survey_date_end"
kml_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/RATs"

### naming convention of .tiff and .xml files must match ###

# Create a list of TIFF files
tiff_files <- list.files(input_dir, pattern = "\\.tiff$", full.names = TRUE)

# Iterate through TIFF files and process each one
for (tiff_file in tiff_files) {
  # Extract the base name of the file
  file_base <- basename(tiff_file)
  file_name <- tools::file_path_sans_ext(file_base)
  
  # Define the corresponding KML file path
  kml_file <- file.path(kml_dir, paste0(file_name, ".tiff.aux.xml"))
  
  # Read the raster file
  r <- brick(tiff_file)
  
  # Extract the 'contributor' band (Band 3)
  contributor_band <- r[[3]]
  
  # Read the XML file
  xml_data <- xml2::read_xml(kml_file)
  
  # Extract Raster Attribute Table Rows (e.g., survey dates)
  # Find each row in the GDALRasterAttributeTable for 'Contributor' band
  contributor_band_xml <- xml2::xml_find_all(xml_data, "//PAMRasterBand[Description='Contributor']")
  rows <- xml2::xml_find_all(contributor_band_xml, ".//GDALRasterAttributeTable/Row")
  
  # Extract specific fields from each row
  table_data <- lapply(rows, function(row) {
    fields <- xml2::xml_find_all(row, ".//F")
    field_values <- xml2::xml_text(fields)
    list(
      value = as.numeric(field_values[1]),
      survey_date_end = as.Date(field_values[18], format = "%Y-%m-%d")
    )
  })
  
  # Convert extracted data to a data frame for easier analysis
  attribute_table_df <- do.call(rbind, lapply(table_data, as.data.frame, stringsAsFactors = FALSE))
  
  # Extract the year from survey_date_end
  attribute_table_df <- attribute_table_df %>%
    mutate(survey_year_end = as.numeric(format(survey_date_end, "%Y")))
  
  # Ensure no NA values in 'survey_year_end'
  attribute_table_df <- attribute_table_df %>%
    mutate(survey_year_end = ifelse(is.na(survey_year_end), 0, survey_year_end))
  
  
  attribute_table_df$survey_year_end <- round(attribute_table_df$survey_year_end, digits = 2)
  
  
  # Create a lookup table to map raster values to the survey end year
  date_mapping <- attribute_table_df %>%
    select(value, survey_year_end) %>%
    distinct()  # Ensure unique mapping
  
  # Convert to matrix for compatibility with reclassify
  reclass_matrix <- as.matrix(date_mapping[, c("value", "survey_year_end")])
  
  # Reclassify the raster using the cleaned and year mapping
  year_raster <- reclassify(
    contributor_band,
    rcl = reclass_matrix,
    right = FALSE
  )
  
  # Define output file path with the same naming convention
  output_file <- file.path(output_dir, file_base)
  
  # Save the new year raster as a TIFF file
  writeRaster(year_raster, output_file, format = "GTiff", overwrite = TRUE)
}

print("All Survey End Date files processed and saved with the same naming convention.")

 ## S5.1 (F5): Standardize all rasters (PREDICTION Extent first as its larger)----
#makes all same extent, for processing into spatial points dataframe and removes all land based elevation values > 0 as well

#### NOTE:------ It is critical that BOTH the prediction and training datasets must have the same values of X, Y and FID within each, 
# although they are different extents, the smaller training data must be a direct subset of the prediction datafor variables they have in common,
#even if the datasets vary between the two final datasets, we will divide the pertiant columns afterward.

mask.train <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif")  # Ensure this is a 1/0 mask
plot(mask) # visualize that this is binary 
# Process rasters
standardize_rasters(
  mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif",
  input_dir_raw = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw",
  input_dir_partial = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/model_varaibles/Prediction/part_processed",
  output_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed"
)

## S5.2 (F5)  Standardize all rasters (TRAINING Extent - a direct sub sample of prediction extent, clipped using the training mask)----
# Set working directory and load the training mask
mask.train <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif")  # Ensure this is a 1/0 mask
plot(mask) # visualize that this is binary 
# Process rasters 
standardize_rasters(
  mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif",
  input_dir_raw = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/raw",
  input_dir_partial = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/model_varaibles/Prediction/part_processed",
  output_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed"
)

##S6.1. Create a spatial DF from TRAINING extent mask (IN UTM COORDINATES & WGS84 COORDINATES [for sub grid creation only])----
create_spatial_mask_df(
  mask_utm_path = "N:/.../training.mask.UTM17_8m.tif",
  mask_wgs_path = "N:/.../training.mask.WGS84_8m.tif",
  mask_type = "training"
)

##S6.3 Create a spatial DF from TRAINING extent mask (IN UTM COORDINATES & WGS84 COORDINATES [for sub grid creation only])----
create_spatial_mask_df(
  mask_utm_path = "N:/.../prediction.mask.UTM17_8m.tif",
  mask_wgs_path = "N:/.../prediction.mask.WGS84_8m.tif",
  mask_type = "prediction"
)

##S7.  # Processing Subgrid Functions----
# Run prediction sub grid processing----
prediction_sub_grids <- prepare_subgrids(
  grid_gpkg = grid_gpkg,
  mask.df = prediction.mask.df.wgs84, # WGS 84 DataFrame version of mask DF
  output_dir = output_dir_pred
)

# Run training sub grid processing----
training_sub_grids <- prepare_subgrids(
  grid_gpkg,
  mask.df = training.mask.df.wgs84,  # WGS 84 DataFrame version of mask DF
  output_dir = output_dir_train
)

# Sub grid re-projection to UTM
#training 
reproject_sub_grids(input_gpkg = training_sub_grids_WGS84,
                    output_gpkg = training_sub_grids_UTM)
#Prediction
reproject_sub_grids(input_gpkg = prediction_sub_grids_WGS84,
                    output_gpkg = prediction_sub_grids_UTM)

# load the re-projected Sub_grids now in UTM (a training and prediction version)
training_sub_grids_UTM <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
prediction_sub_grids_UTM <-st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")

# Run the grid_out_raster_data function for raster subset creation
#Training - 1.25 hrs &  10-15GB
Sys.time()
grid_out_raster_data(
  sub_grid_gpkg = training_sub_grids_UTM,
  data_type = "training", 
  output_dir = output_dir_train,
  raster_dir = input_dir_train)
Sys.time()
# Prediction - 1.75hrs & 10GB-15GB
Sys.time()
grid_out_raster_data(
  sub_grid_gpkg = prediction_sub_grids_UTM,
  data_type = "prediction", 
  output_dir = output_dir_pred,
  raster_dir = input_dir_pred)
Sys.time()

# Check after processing what your data looks like, correct columns etc..----
pred.data <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/BH4RZ577_3/BH4RZ577_3_prediction_clipped_data.rds")
training_data <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4RZ577_3/BH4RZ577_3_training_clipped_data.rds")


# ─────────────────────────────
# 5. Shut Down Parallel
# ─────────────────────────────
stopCluster(cl)

cat("🎉 All preprocessing tasks completed.\n")







### OLDER VERSION BELOW _ KEEP UNTILL THE ABOVE IS WORKING 
# STAGE 1 - PREPROCESSING:----
## 1. Survey End Date Extraction MULTIPLE .xml :----

# Define input and output directories
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/UTM17"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/survey_date_end"
kml_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/RATs"

### naming convention of .tiff and .xml files much match ###

# Create a list of TIFF files
tiff_files <- list.files(input_dir, pattern = "\\.tiff$", full.names = TRUE)

# Iterate through TIFF files and process each one
for (tiff_file in tiff_files) {
  # Extract the base name of the file
  file_base <- basename(tiff_file)
  file_name <- tools::file_path_sans_ext(file_base)
  
  # Define the corresponding KML file path
  kml_file <- file.path(kml_dir, paste0(file_name, ".tiff.aux.xml"))
  
  # Read the raster file
  r <- brick(tiff_file)
  
  # Extract the 'contributor' band (Band 3)
  contributor_band <- r[[3]]
  
  # Read the XML file
  xml_data <- xml2::read_xml(kml_file)
  
  # Extract Raster Attribute Table Rows (e.g., survey dates)
  # Find each row in the GDALRasterAttributeTable for 'Contributor' band
  contributor_band_xml <- xml2::xml_find_all(xml_data, "//PAMRasterBand[Description='Contributor']")
  rows <- xml2::xml_find_all(contributor_band_xml, ".//GDALRasterAttributeTable/Row")
  
  # Extract specific fields from each row
  table_data <- lapply(rows, function(row) {
    fields <- xml2::xml_find_all(row, ".//F")
    field_values <- xml2::xml_text(fields)
    list(
      value = as.numeric(field_values[1]),
      survey_date_end = as.Date(field_values[18], format = "%Y-%m-%d")
    )
  })
  
  # Convert extracted data to a data frame for easier analysis
  attribute_table_df <- do.call(rbind, lapply(table_data, as.data.frame, stringsAsFactors = FALSE))
  
  # Extract the year from survey_date_end
  attribute_table_df <- attribute_table_df %>%
    mutate(survey_year_end = as.numeric(format(survey_date_end, "%Y")))
  
  # Ensure no NA values in 'survey_year_end'
  attribute_table_df <- attribute_table_df %>%
    mutate(survey_year_end = ifelse(is.na(survey_year_end), 0, survey_year_end))
  
  
  attribute_table_df$survey_year_end <- round(attribute_table_df$survey_year_end, digits = 2)
  
  
  # Create a lookup table to map raster values to the survey end year
  date_mapping <- attribute_table_df %>%
    select(value, survey_year_end) %>%
    distinct()  # Ensure unique mapping
  
  # Convert to matrix for compatibility with reclassify
  reclass_matrix <- as.matrix(date_mapping[, c("value", "survey_year_end")])
  
  # Reclassify the raster using the cleaned and year mapping
  year_raster <- reclassify(
    contributor_band,
    rcl = reclass_matrix,
    right = FALSE
  )
  
  # Define output file path with the same naming convention
  output_file <- file.path(output_dir, file_base)
  
  # Save the new year raster as a TIFF file
  writeRaster(year_raster, output_file, format = "GTiff", overwrite = TRUE)
}

print("All TIFF files processed and saved with the same naming convention.")

library(raster)
library(dplyr)
#2. Fill NA values in bathy and create binary training mask---- 
#  Fill internal NA values using focal filter
fill_internal_na_focal <- function(input_file, output_file, w = 3) {
  r <- raster(input_file)
  
  # Define a square moving window
  focal_weights <- matrix(1, nrow = w, ncol = w)
  
  # Interpolate NA cells using the mean of surrounding values
  # Preserve non-NA values as-is
  filled <- focal(r,
                  w = focal_weights,
                  fun = mean,
                  na.rm = TRUE,
                  NAonly = TRUE,
                  pad = TRUE,
                  padValue = NA)
  
  # Keep original values where not NA, otherwise use filled
  final <- overlay(r, filled, fun = function(orig, interp) {
    ifelse(is.na(orig), interp, orig)
  })
  
  writeRaster(final, filename = output_file, format = "GTiff", overwrite = TRUE)
}

#  Create binary training mask from all filled rasters 
create_training_mask <- function(input_dir, output_file) {
  filled_files <- list.files(input_dir, pattern = "bathy_\\d{4}\\.tif$", full.names = TRUE)
  ras_stack <- stack(filled_files)
  
  # Combine to detect presence of data
  mask <- calc(ras_stack, fun = function(x) {
    if (all(is.na(x))) return(NA)
    1  # Data exists
  })
  
  # Convert to 1/0 binary mask
  binary_mask <- calc(mask, fun = function(x) ifelse(!is.na(x), 1, 0))
  
  writeRaster(binary_mask, filename = output_file, format = "GTiff", overwrite = TRUE)
}


## 3. Standardize all rasters (PREDICTION Extent first as its larger)----
#makes all same extent, for processing into spatial points dataframe and removes all land based elevation values > 0 as well

#### NOTE:------ It is critical that BOTH the prediction and training datasets must have the same values of X, Y and FID within each, ---###
# although they are different extents, the smaller training data must be a direct subset of the prediction data
# for variables they have in common, even if the datasets vary between the two final datasets, we will divide the pertiant 
#columns afterward.


# Set WD on N drive 
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model")
mask <- raster("prediction.mask.UTM17_8m.tif")  # Load mask
crs_mask <- crs(mask)  # Get CRS of the mask
# plot(mask)

# Get raster list 
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed"
if (!dir.exists(output_dir)) dir.create(output_dir)

f8m <- list.files(input_dir, pattern = "\\.tif$", full.names = TRUE)

# Progress Bar Setup
total_rasters <- length(f8m)
pboptions(type = "txt")  # Set progress bar type
start_time <- Sys.time()  # Start timer

# Process each raster individually
# Processing function
process_raster <- function(i) {
  file <- f8m[i]
  source_name <- file_path_sans_ext(basename(file))
  cat(sprintf("\nProcessing [%d/%d]: %s\n", i, total_rasters, source_name))
  
  tryCatch({
    # Load raster
    ras <- raster(file)
    
    # Ensure CRS matches the mask
    if (!compareCRS(ras, mask)) {
      ras <- projectRaster(ras, crs = crs_mask, method = "bilinear")
    }
    
    # Align extent properly (this ensures snapping to the mask)
    ras <- resample(ras, mask, method = "bilinear")
    
    # Masking step
    temp_ras <- mask(ras, mask)
    
    # Set values > 0 to NA for "_bathy" rasters
    if (grepl("_bathy", source_name)) {
      temp_ras[temp_ras > 0] <- NA
    }
    
    # Save output
    output_name <- file.path(output_dir, paste0(source_name, ".tif"))
    writeRaster(temp_ras, output_name, overwrite = TRUE)
    
    # Memory cleanup
    rm(ras, temp_ras)
    gc()
    
    # Estimate and display remaining time
    elapsed_time <- difftime(Sys.time(), start_time, units = "mins")
    avg_time <- as.numeric(elapsed_time) / i
    remaining_time <- avg_time * (total_rasters - i)
    cat(sprintf("completed [%d/%d]: %s (Estimated Time Left: ~%.1f min)\n",
                i, total_rasters, source_name, remaining_time))
    
  }, error = function(e) {
    cat(sprintf("ERROR: Skipping %s due to error: %s\n", source_name, e$message))
  })
}

# Run sequentially to avoid memory overload
pblapply(seq_along(f8m), process_raster, cl = 1)


# Final time summary
total_time <- difftime(Sys.time(), start_time, units = "mins")
cat(sprintf("\n All rasters processed in %.1f minutes! \n", total_time))


### Check if the rasters achieved the same extent###
raster_files <- list.files("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed", pattern = "\\.tif$", full.names = TRUE)
# Create a list of rasters
raster_list <- lapply(raster_files, raster)
# Function to check if all rasters have the same extents
check_rasters_same_extent <- function(raster_list) {
  ext <- extent(raster_list[[1]])
  
  for (i in 2:length(raster_list)) {
    if (!identical(ext, extent(raster_list[[i]]))) {
      return(FALSE)
    }
  }
  return(TRUE)
}
# Check if all rasters have the same extents
if (check_rasters_same_extent(raster_list)) {
  print("All rasters have the same extents.")
} else { print("Rasters do not have the same extents.")
}


## 3 Standardize all rasters (TRAINING Extent - sub sample of prediction extent)----
#- THIS IS A DIRECT SUBSET OF THE PREDICTION AREA - clipped using the training mask. 

# Set working directory and load the training mask
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model")
mask <- raster("training.mask.UTM17_8m.tif")  # Ensure this is a 1/0 mask
crs_mask <- crs(mask)  # Extract CRS from the mask
plot(mask) # visualize that this is binary 

# Set input and output directories
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed"
if (!dir.exists(output_dir)) dir.create(output_dir)

f8m <- list.files(input_dir, pattern = "\\.tif$", full.names = TRUE)

# Progress Bar Setup
total_rasters <- length(f8m)
pboptions(type = "txt")  # Set progress bar type
start_time <- Sys.time()  # Start timer

# Process each raster individually
# Processing function
process_raster <- function(i) {
  file <- f8m[i]
  source_name <- file_path_sans_ext(basename(file))
  cat(sprintf("\nProcessing [%d/%d]: %s\n", i, total_rasters, source_name))
  
  tryCatch({
    # Load raster
    ras <- raster(file)
    
    # Ensure CRS matches the mask
    if (!compareCRS(ras, mask)) {
      ras <- projectRaster(ras, crs = crs_mask, method = "bilinear")
    }
    
    # Align extent properly (this ensures snapping to the mask)
    ras <- resample(ras, mask, method = "bilinear")
    
    # Masking step
    temp_ras <- mask(ras, mask)
    
    # Set values > 0 to NA for "_bathy" rasters
    if (grepl("_bathy", source_name)) {
      temp_ras[temp_ras > 0] <- NA
    }
    
    # Save output
    output_name <- file.path(output_dir, paste0(source_name, ".tif"))
    writeRaster(temp_ras, output_name, overwrite = TRUE)
    
    # Memory cleanup
    rm(ras, temp_ras)
    gc()
    
    # Estimate and display remaining time
    elapsed_time <- difftime(Sys.time(), start_time, units = "mins")
    avg_time <- as.numeric(elapsed_time) / i
    remaining_time <- avg_time * (total_rasters - i)
    cat(sprintf("completed [%d/%d]: %s (Estimated Time Left: ~%.1f min)\n",
                i, total_rasters, source_name, remaining_time))
    
  }, error = function(e) {
    cat(sprintf("ERROR: Skipping %s due to error: %s\n", source_name, e$message))
  })
}

# Run sequentially to avoid memory overload
pblapply(seq_along(f8m), process_raster, cl = 1)

# Final time summary
total_time <- difftime(Sys.time(), start_time, units = "mins")
cat(sprintf("\n All rasters processed in %.1f minutes! \n", total_time))


### Check if the rasters achieved the same extent###
# Get the list of raster files
raster_files <- list.files("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed", pattern = "\\.tif$", full.names = TRUE)
# Create a list of rasters
raster_list <- lapply(raster_files, raster)
# Function to check if all rasters have the same extents
check_rasters_same_extent <- function(raster_list) {
  ext <- extent(raster_list[[1]])
  
  for (i in 2:length(raster_list)) {
    if (!identical(ext, extent(raster_list[[i]]))) {
      return(FALSE)
    }
  }
  return(TRUE)
}
# Check if all rasters have the same extents
if (check_rasters_same_extent(raster_list)) {
  print("All rasters have the same extents.")
} else { print("Rasters do not have the same extents.")
}



#4.1. Create a spatial DF from TRAINING extent mask (IN UTM COORDINATES)----
training.mask.UTM <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif") # for reference CRS of training grid
# Convert raster to a spatial points dataframe
training.mask.spdf <- rasterToPoints(training.mask.UTM, spatial = TRUE)
# Extract coordinates and bind with raster data
training.mask.df <- data.frame(training.mask.spdf@data, X = training.mask.spdf@coords[, 1], Y = training.mask.spdf@coords[, 2])
# Set unique FID and extract X & Y data from the raster stack
training.mask.df$FID <- cellFromXY(training.mask.UTM, training.mask.df[, c("X", "Y")])
training.mask.df <- training.mask.df[training.mask.df$training.mask == 1, ]
# Save the filtered dataframe
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data")
saveRDS(training.mask.df, file = "training.mask.df.021425.Rds")
training.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.021425.Rds")

#4.2 Create another spatial DF from TRAINING extent mask (IN WGS84 COORDINATES for sub grid creation only)----
training.mask.WGS84 <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.WGS84_8m.tif") # for reference CRS of training grid
# Convert raster to a spatial points dataframe
training.mask.spdf <- rasterToPoints(training.mask.WGS84, spatial = TRUE)
# Extract coordinates and bind with raster data
training.mask.df <- data.frame(training.mask.spdf@data, X = training.mask.spdf@coords[, 1], Y = training.mask.spdf@coords[, 2])
# Set unique FID and extract X & Y data from the raster stack
training.mask.df$FID <- cellFromXY(training.mask.WGS84, training.mask.df[, c("X", "Y")])
training.mask.df.wgs84 <- training.mask.df[training.mask.df$training.mask == 1, ]
# Save the filtered dataframe
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data")
saveRDS(training.mask.df.wgs84, file = "training.mask.df.wgs84.021425.Rds")
training.mask.df.wgs84 <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.wgs84.021425.Rds")

#4.3 Create a spatial DF from PREDICTION extent mask (IN UTM COORDINATES)----
prediction.mask.UTM <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif")
prediction.mask.spdf <- rasterToPoints(prediction.mask.UTM, spatial = T) # turns data into a large spatial points dataframe
prediction.mask.spdf@coords # a command to see the x and y from the raster data
# Extract the coordinates, for the spatial points df, and bind it with the env data from the raster stack,and save as a new dataframe
prediction.mask.df <- data.frame(prediction.mask.spdf@data, X =prediction.mask.spdf@coords[,1],Y = prediction.mask.spdf@coords[,2])
head(prediction.mask.df)
#--SET UNIQUE FID, EXTRACT X & Y(row and column numbers) data from raster stack
prediction.mask.df$FID <- cellFromXY(prediction.mask.UTM, prediction.mask.df[,c("X", "Y")])
prediction.mask.df <- prediction.mask.df[prediction.mask.df$prediction.mask == 1, ]
head(prediction.mask.df)
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data")
saveRDS (prediction.mask.df, file = "prediction.mask.df.021425.Rds")
prediction.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/prediction.mask.df.021425.Rds")

#4.4 Create a spatial DF from PREDICTION extent mask (IN WGS84 COORDINATES for sub grid creation only)----
prediction.mask.WGS84 <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.WGS84_8m.tif")
prediction.mask.spdf <- rasterToPoints(prediction.mask, spatial = T) # turns data into a large spatial points dataframe
prediction.mask.spdf@coords # a command to see the x and y from the raster data
# Extract the coordinates, for the spatial points df, and bind it with the env data from the raster stack,and save as a new dataframe
prediction.mask.df <- data.frame(prediction.mask.spdf@data, X =prediction.mask.spdf@coords[,1],Y = prediction.mask.spdf@coords[,2])
head(prediction.mask.df)
#--SET UNIQUE FID, EXTRACT X & Y(row and column numbers) data from raster stack
prediction.mask.df$FID <- cellFromXY(prediction.mask.WGS84, prediction.mask.df[,c("X", "Y")])
prediction.mask.df.wgs84 <- prediction.mask.df[prediction.mask.df$prediction.mask == 1, ]
head(prediction.mask.df.wgs84)
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data")
saveRDS (prediction.mask.df.wgs84, file = "prediction.mask.df.wgs84.021425.Rds")
prediction.mask.df.wgs84 <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/prediction.mask.df.wgs84.021425.Rds")

#5.  # Processing Subgrid Functions----
# Functions to prepare and save a new sub-grid gpkg (master blue topo grid tile divided by 4, same tile I.Ds with sub identifier _1,_2_3,_4 in clockwise order)
# FUNCTION: Split a single tile into 4 equal sub-grids, starting top left in clockwise order
split_tile <- function(tile) {
  # Get tile bounding box
  bbox <- st_bbox(tile)
  
  # Compute sub-grid size (divide original size by 2)
  sub_width <- (bbox["xmax"] - bbox["xmin"]) / 2
  sub_height <- (bbox["ymax"] - bbox["ymin"]) / 2
  
  # Define 4 sub-grids
  sub_grids <- list(
    st_polygon(list(matrix(c(
      bbox["xmin"], bbox["ymin"],
      bbox["xmin"] + sub_width, bbox["ymin"],
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmin"], bbox["ymin"] + sub_height,
      bbox["xmin"], bbox["ymin"]
    ), ncol = 2, byrow = TRUE))),
    
    st_polygon(list(matrix(c(
      bbox["xmin"] + sub_width, bbox["ymin"],
      bbox["xmax"], bbox["ymin"],
      bbox["xmax"], bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymin"]
    ), ncol = 2, byrow = TRUE))),
    
    st_polygon(list(matrix(c(
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmax"], bbox["ymin"] + sub_height,
      bbox["xmax"], bbox["ymax"],
      bbox["xmin"] + sub_width, bbox["ymax"],
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height
    ), ncol = 2, byrow = TRUE))),
    
    st_polygon(list(matrix(c(
      bbox["xmin"], bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymin"] + sub_height,
      bbox["xmin"] + sub_width, bbox["ymax"],
      bbox["xmin"], bbox["ymax"],
      bbox["xmin"], bbox["ymin"] + sub_height
    ), ncol = 2, byrow = TRUE)))
  )
  
  # Create sub-grid tile IDs
  sub_grid_ids <- paste0(tile$tile, "_", 1:4)
  
  # Preserve attributes
  df <- data.frame(tile_id = sub_grid_ids, original_tile = tile$tile)
  df_sf <- st_as_sf(df, geometry = st_sfc(sub_grids, crs = 4326))
  
  return(df_sf)
}

# FUNCTION: Prepare and save sub-grids
prepare_subgrids <- function(grid_gpkg, mask.df, output_dir) {
  cat("Preparing grid tiles and sub-grids...\n")
  
  # Read the original grid (Already in WGS 84)
  grid_tiles <- grid_gpkg
  
  # Convert mask to sf object (Already in WGS 84)
  mask_sf <- st_as_sf(mask.df, coords = c("X", "Y"), crs = 4326)
  
  #  Split ALL tiles into sub-grids
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) split_tile(grid_tiles[i, ])))
  
  # Debugging: Check if sub-grids exist before filtering
  cat("Total sub-grids generated:", nrow(sub_grids), "\n")
  
  #  Filter sub-grids that intersect with the mask
  intersecting_sub_grids <- st_filter(sub_grids, st_union(mask_sf))
  
  # Save outputs
  st_write(intersecting_sub_grids, file.path(output_dir, "intersecting_sub_grids_WGS84.gpkg"), delete_layer = TRUE, quiet = FALSE)
  saveRDS(intersecting_sub_grids, file.path(output_dir, "grid_tile_extents_WGS84.rds"))
  
  cat("✅ Sub-grids successfully prepared:", nrow(intersecting_sub_grids), "\n")
  return(intersecting_sub_grids)
}

# Function to reproject sub grid and save new GPKG as UTM
reproject_sub_grids <- function(input_gpkg, output_gpkg, target_crs = crs(training.mask.UTM)) {
  cat("Reading:", input_gpkg, "\n")
  
  # Load the original sub-grid (WGS84)
  sub_grids <- st_read(input_gpkg, quiet = TRUE)
  
  # 🚀 Reproject to UTM (same as the mask)
  sub_grids_utm <- st_transform(sub_grids, target_crs)
  
  # 🚀 Save the new UTM version
  cat("Saving:", output_gpkg, "\n")
  st_write(sub_grids_utm, output_gpkg, delete_layer = TRUE, quiet = TRUE)
  
  cat("✅ Reprojection complete:", output_gpkg, "\n\n")
}

# Function to process raster data into a chunk size spatial datasets per tile folder
# the process rasters function creates a sub sample of all the raster data (model variables) from which the model training code will pull from to run the model, 
# so in each tile folder it will create a .rds file for all data in that tile extent e.g., Tile_BH4S656W_4_training_clipped_data.rds

process_rasters <- function(sub_grid_gpkg, raster_dir, output_dir, data_type) {
  # Ensure sub_grid_gpkg is a valid path
  if (is.character(sub_grid_gpkg)) {
    sub_grids <- st_read(sub_grid_gpkg)
  } else if (inherits(sub_grid_gpkg, "sf")) {
    sub_grids <- sub_grid_gpkg
  } else {
    stop("`sub_grid_gpkg` must be a valid file path or an sf object.")
  }
  
  # List raster files in the input directory
  raster_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  
  # Process each tile
  for (i in seq_len(nrow(sub_grids))) {
    sub_grid <- sub_grids[i, ]
    tile_name <- sub_grid$tile_id  # Ensure sub-grid has a `tile_id` column
    tile_extent <- as(extent(st_bbox(sub_grid)), "SpatialPolygons")  # Convert extent to SpatialPolygons
    crs(tile_extent) <- st_crs(sub_grids)$proj4string  # Assign CRS
    
    # Create sub-folder for the tile if it doesn't exist
    tile_dir <- file.path(output_dir, tile_name)
    if (!dir.exists(tile_dir)) {
      dir.create(tile_dir, showWarnings = FALSE, recursive = TRUE)
    }
    
    # Path to save the clipped raster data
    clipped_data_path <- file.path(tile_dir, paste0(tile_name, "_", data_type, "_clipped_data.rds"))
    
    # Overwrite existing file if present
    if (file.exists(clipped_data_path)) {
      cat("Overwriting existing file for tile:", tile_name, "\n")
    }
    
    cat("Processing", data_type, "tile:", tile_name, "\n")
    
    # Clip rasters to the tile extent and process
    clipped_data <- lapply(raster_files, function(r_file) {
      r <- raster::raster(r_file)  # Load raster
      
      # 🔍 **Check if raster and tile overlap before cropping**
      if (is.null(raster::intersect(extent(r), tile_extent))) {
        cat("  ⚠️ Skipping raster (no overlap):", basename(r_file), "\n")
        return(NULL)  # Skip this raster
      }
      
      cropped_r <- raster::crop(r, tile_extent)  # Crop to tile extent
      
      # Extract raster values along with X and Y coordinates
      raster_data <- as.data.frame(raster::rasterToPoints(cropped_r, spatial = FALSE))
      raster_name <- tools::file_path_sans_ext(basename(r_file))  # Extract raster name
      colnames(raster_data) <- c("X", "Y", paste0(raster_name))  
      raster_data$FID <- raster::cellFromXY(r, raster_data[, c("X", "Y")])  # Add FID
      return(raster_data)
    })
    
    # Remove NULL elements (rasters that were skipped)
    clipped_data <- Filter(Negate(is.null), clipped_data)
    
    # Combine all rasters into a single data frame
    if (length(clipped_data) > 0) {
      combined_data <- Reduce(function(x, y) merge(x, y, by = c("X", "Y", "FID"), all = TRUE), clipped_data)
      
      # 🚀 **Remove specific columns based on `data_type`**
      if (data_type == "training") {
        combined_data <- combined_data %>% select(-starts_with("bt"))  # Remove "bt" columns
        
        # 🆕 **Create `b.change` columns**
        combined_data <- combined_data %>%
          mutate(
            b.change.2004_2006 = bathy_2006 - bathy_2004, # 2 years
            b.change.2006_2010 = bathy_2010 - bathy_2006, # 4 years
            b.change.2010_2015 = bathy_2015 - bathy_2010, # 5 years 
            b.change.2015_2022 = bathy_2022 - bathy_2015  # 7 years
          )
      } else if (data_type == "prediction") {
        combined_data <- combined_data %>% select(-starts_with("bathy"), -starts_with("slope"), -starts_with("rugosity"))  # Remove specified columns
      }
      
      # Save the combined data as RDS
      saveRDS(combined_data, file = clipped_data_path)
      cat("✅ Saved", data_type, "clipped data for tile:", tile_name, "\n")
    } else {
      cat("⚠️ No overlapping rasters for tile:", tile_name, "- Skipping saving.\n")
    }
  }
  
  cat("✅ Finished processing all", data_type, "tiles in", output_dir, "\n")
}


# 5. DEFINE FUNCTION PARAMETERS before running functions below- make sure the link points to the newest dataset----
grid_gpkg <- st_read("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg") # from Blue topo
#
training_sub_grids_WGS84 <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_WGS84.gpkg"
prediction_sub_grids_WGS84 <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_WGS84.gpkg"
#
training.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif") # for reference CRS of training grid
training.mask.wgs84 <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.WGS84_8m.tif")
prediction.mask <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif")
prediction.mask.wgs84 <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.WGS84_8m.tif")
#
training.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.021425.Rds")# spatial DF of extent
training.mask.df.wgs84 <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.wgs84.021425.Rds")
prediction.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/prediction.mask.df.021425.Rds")# spatial DF of extent
prediction.mask.df.wgs84 <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/prediction.mask.df.wgs84.021425.Rds")# spatial DF of extent
#
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
output_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"
input_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed" # raster data 
input_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # raster data 

# 6 - RUN Functions----
# Run Fill NA values in bathy and create binary training mask----
# Define paths
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw"
filled_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/part_processed"
mask_output <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/training.mask.tif"

if (!dir.exists(filled_dir)) dir.create(filled_dir)

# Step 1: Apply focal fill to all bathy rasters
bathy_files <- list.files(input_dir, pattern = "bathy_\\d{4}\\.tif$", full.names = TRUE)

for (file in bathy_files) {
  year <- tools::file_path_sans_ext(basename(file))
  out_file <- file.path(filled_dir, paste0(year, "_filled.tif"))
  
  fill_internal_na_focal(input_file = file,
                         output_file = out_file,
                         w = 3)  # Optional: increase for smoother fill
}

# Step 2: Create binary training mask
create_training_mask(input_dir = filled_dir,
                     output_file = mask_output)
# Run prediction sub grid processing----
Sys.time()
prediction_sub_grids <- prepare_subgrids(
  grid_gpkg = grid_gpkg,
  mask.df = prediction.mask.df.wgs84, # WGS 84 DataFrame version of mask DF
  output_dir = output_dir_pred
)
Sys.time() # Takes approx 40 minutes and 25GB of space for this size of area. 

# Run training sub grid processing----
Sys.time()
training_sub_grids <- prepare_subgrids(
  grid_gpkg,
  mask.df = training.mask.df.wgs84,  # WGS 84 DataFrame version of mask DF
  output_dir = output_dir_train
)
Sys.time()


# Run sub grid re-projection for both training and prediction grids to UTM
#training 
reproject_sub_grids(input_gpkg = training_sub_grids_WGS84,
                    output_gpkg = training_sub_grids_UTM)
#Prediction
reproject_sub_grids(input_gpkg = prediction_sub_grids_WGS84,
                    output_gpkg = prediction_sub_grids_UTM)

# load the re-projected Sub_grids now in UTM (a training and prediction version)
training_sub_grids_UTM <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
prediction_sub_grids_UTM <-st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")

# Run the process_rasters function for raster subset creation
#Training - 1.25 hrs &  10-15GB
Sys.time()
process_rasters(
  sub_grid_gpkg = training_sub_grids_UTM,
  data_type = "training", 
  output_dir = output_dir_train,
  raster_dir = input_dir_train)
Sys.time()
# Prediction - 1.75hrs & 10GB-15GB
Sys.time()
process_rasters(
  sub_grid_gpkg = prediction_sub_grids_UTM,
  data_type = "prediction", 
  output_dir = output_dir_pred,
  raster_dir = input_dir_pred)
Sys.time()

# Check after processing what your data looks like, correct columns etc..----
pred.data <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/BH4RZ577_3/BH4RZ577_3_prediction_clipped_data.rds")
training_data <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4RZ577_3/BH4RZ577_3_training_clipped_data.rds")


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





# 6. Model training over all sub grids ----
# This script loops through grid tiles intersecting the training data boundary, runs models for each year pair, and stores results as .rds files.


# UPDATE TO THE ABOVE PARAMETERS (UPDATE AGAIN BEFORE Running model tomorrow!!!!
# load and check the clipped data)
# restart R.
# modify the code to run un paralell!!!!!! 
# add in the X,Y and FID to the predictions saving. 

# Push new code for Aubrey. 
# then run!!! and time / monitor resource. 

# LOAD PARAMETERS FROM CREATED FROM STAGE PREPROCESSING if not already loaded from step 4:----
grid_gpkg <- st_read("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg") # from Blue topo
#
training_sub_grids_UTM <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
prediction_sub_grids_UTM <-st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
#
training.mask.UTM <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif") # for reference CRS of training grid
prediction.mask.UTM <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif")
#
training.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/training.mask.df.021425.Rds")# spatial DF of extent
prediction.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data/prediction.mask.df.021425.Rds")# spatial DF of extent
#
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"


# 🚀 Updated Model Training Function with PDP, Performance Tracking, & Predictions Fixes
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

# 🚀 **New Safe rbind Function**
rbind_safe <- function(...) {
  args <- list(...)
  args <- args[!sapply(args, is.null)]  # Remove NULL results
  if (length(args) == 0) return(data.frame())  # Ensure it returns a dataframe
  return(bind_rows(args))
}


#V1 - works perfectly but only for 1 year pair iteration 
# registerDoParallel(detectCores() - 1)
Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 2) { 
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n🚀 Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  # num_cores <- detectCores() - 1
  # cl <- makeCluster(num_cores)
  # registerDoParallel(cl)  # ✅ Enable parallel processing
  registerDoSEQ()  # Run sequentially for debugging
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
                          .packages = c("xgboost", "dplyr", "data.table", "fst", "tidyr", "foreach")) %do% { 
                            
                            tryCatch({
                              tile_id <- tiles_df[[i]]
                              training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
                              
                              if (!file.exists(training_data_path)) {
                                cat(Sys.time(), "⚠️ Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
                                return(NULL)
                              }
                              
                              training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                              training_data <- as.data.frame(training_data)
                              
                              missing_static <- setdiff(static_predictors, names(training_data))
                              if (length(missing_static) > 0) {
                                cat(Sys.time(), "🚨 ERROR: Static predictors missing for Tile", tile_id, ":", paste(missing_static, collapse = ", "), "\n", file = log_file, append = TRUE)
                                return(NULL)
                              }
                              
                              for (pair in year_pairs) {
                                cat(Sys.time(), "➡️ Starting Year Pair:", pair, "for Tile:", tile_id, "\n", file = log_file, append = TRUE)
                                
                                start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                
                                # Skip this pair if response variable is missing
                                response_var <- paste0("b.change.", pair)
                                if (!response_var %in% names(training_data)) {
                                  cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  next  # Move to next year pair
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
                                response_var <- paste0("b.change.", pair)
                                
                                if (!response_var %in% names(training_data)) {
                                  cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  next
                                }
                                
                                
                                # -------------------------------------------------------
                                # 3. FILTER & PREPARE TRAINING DATA
                                # -------------------------------------------------------
                                cat(Sys.time(), "📌 Tile:", tile_id, "| Year Pair:", pair, "| Available Predictors:", 
                                    paste(predictors, collapse = ", "), "| Response:", response_var, "\n", 
                                    file = log_file, append = TRUE)
                                
                                 subgrid_data <- training_data %>%
                                  dplyr::select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                  drop_na()
                                
                                if (nrow(subgrid_data) == 0) {
                                  cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  next
                                }
                                 
                                 if (!response_var %in% names(training_data)) {
                                   cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                       file = log_file, append = TRUE)
                                   next  # Move to next year pair
                                 }
                                 if (nrow(subgrid_data) == 0) {
                                   cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                  cat(Sys.time(), "🚨 ERROR: No predictors found for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  next
                                }
                                
                                
                                if (any(!is.finite(as.matrix(subgrid_data[, predictors, drop = FALSE])))) {
                                  cat(Sys.time(), "🚨 ERROR: NA/NaN/Inf detected in predictors\n", file = log_file, append = TRUE)
                                  return(NULL)
                                }
                                
                                cat(Sys.time(), "📢 Training XGBoost model for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                
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
                                  
                                  
                                  for (j in seq_along(predictors)) { 
                                    grid <- data.frame(x = EnvRanges[[predictors[j]]])
                                    grid$y <- predict(xgb_model, newdata = as.matrix(grid$x))
                                    
                                    # Remove NAs and check unique values
                                    grid <- grid[complete.cases(grid), ]
                                    if (length(unique(grid$x)) < 5) {
                                      cat(Sys.time(), "⚠️ Skipping PDP for Predictor:", predictors[j], "- Too few unique values\n", file = log_file, append = TRUE)
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
                                # 8. SAVE OUTPUTS
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
                                
                              }  # End for-loop over year_pairs
                              
                            }, error = function(e) {
                              cat(Sys.time(), "❌ ERROR in Tile:", tiles_df[[i]], "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
                              return(NULL)
                            })
                          }
  
# -------------------------------------------------------
# 6. CLOSE PARALLEL PROCESSING
# -------------------------------------------------------
# stopCluster(cl)

cat("\n✅ Model Training Complete! Check `error_log.txt` for any issues.\n")
}

#v2 - swapping the order of the for each loops. 
Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 2) { 
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n🚀 Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  # num_cores <- detectCores() - 1
  # cl <- makeCluster(num_cores)
  # registerDoParallel(cl)  # ✅ Enable parallel processing
  registerDoSEQ()  # Run sequentially for debugging
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
                          .packages = c("xgboost", "dplyr", "data.table", "fst", "tidyr", "foreach")) %do% { 
                            
                            tile_id <- tiles_df[[i]] # MUST START COUNTER
                            
                            foreach(pair = year_pairs, .combine = rbind, 
                                    .packages = c("xgboost", "dplyr", "data.table", "fst", "tidyr", "foreach")) %do% { 
                            
                                      tryCatch({
                                        training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
                                        
                                        if (!file.exists(training_data_path)) {
                                          cat(Sys.time(), "⚠️ Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                              
                              training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                              training_data <- as.data.frame(training_data)
                              
                              # missing_static <- setdiff(static_predictors, names(training_data))
                              # if (length(missing_static) > 0) {
                              #   cat(Sys.time(), "🚨 ERROR: Static predictors missing for Tile", tile_id, ":", paste(missing_static, collapse = ", "), "\n", file = log_file, append = TRUE)
                              #   return(NULL)
                              # }
                              # 
                              # for (pair in year_pairs) {
                              #   cat(Sys.time(), "➡️ Starting Year Pair:", pair, "for Tile:", tile_id, "\n", file = log_file, append = TRUE)
                                
                                start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                response_var <- paste0("b.change.", pair)
                                
                                
                                if (!response_var %in% names(training_data)) {
                                  cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                  cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  next
                                }
                                
                                
                                # -------------------------------------------------------
                                # 3. FILTER & PREPARE TRAINING DATA
                                # -------------------------------------------------------
                                cat(Sys.time(), "📌 Tile:", tile_id, "| Year Pair:", pair, "| Available Predictors:", 
                                    paste(predictors, collapse = ", "), "| Response:", response_var, "\n", 
                                    file = log_file, append = TRUE)
                                
                                subgrid_data <- training_data %>%
                                  dplyr::select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                  drop_na()
                                
                                if (nrow(subgrid_data) == 0) {
                                  cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  next
                                }
                                
                                if (!response_var %in% names(training_data)) {
                                  cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                      file = log_file, append = TRUE)
                                  next  # Move to next year pair
                                }
                                if (nrow(subgrid_data) == 0) {
                                  cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                  cat(Sys.time(), "🚨 ERROR: No predictors found for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                  next
                                }
                                
                                
                                if (any(!is.finite(as.matrix(subgrid_data[, predictors, drop = FALSE])))) {
                                  cat(Sys.time(), "🚨 ERROR: NA/NaN/Inf detected in predictors\n", file = log_file, append = TRUE)
                                  return(NULL)
                                }
                                
                                cat(Sys.time(), "📢 Training XGBoost model for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                
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
                                  
                                  
                                  for (j in seq_along(predictors)) { 
                                    grid <- data.frame(x = EnvRanges[[predictors[j]]])
                                    grid$y <- predict(xgb_model, newdata = as.matrix(grid$x))
                                    
                                    # Remove NAs and check unique values
                                    grid <- grid[complete.cases(grid), ]
                                    if (length(unique(grid$x)) < 5) {
                                      cat(Sys.time(), "⚠️ Skipping PDP for Predictor:", predictors[j], "- Too few unique values\n", file = log_file, append = TRUE)
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
                                # 8. SAVE OUTPUTS
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
                              cat(Sys.time(), "❌ ERROR in Tile:", tiles_df[[i]], "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
                              return(NULL)
                            })
                          }
                       }
  # -------------------------------------------------------
  # 6. CLOSE PARALLEL PROCESSING
  # -------------------------------------------------------
  # stopCluster(cl)
  
  cat("\n✅ Model Training Complete! Check `error_log.txt` for any issues.\n")
}

# V3 the same as the above, but in full parallel %dopar% WORKS Great
registerDoParallel(detectCores() - 1) 
Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 20) { 
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n🚀 Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)  # ✅ Enable parallel processing
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
                                          cat(Sys.time(), "⚠️ Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                                        training_data <- as.data.frame(training_data)
                                        
                                        # missing_static <- setdiff(static_predictors, names(training_data))
                                        # if (length(missing_static) > 0) {
                                        #   cat(Sys.time(), "🚨 ERROR: Static predictors missing for Tile", tile_id, ":", paste(missing_static, collapse = ", "), "\n", file = log_file, append = TRUE)
                                        #   return(NULL)
                                        # }
                                        # 
                                        # for (pair in year_pairs) {
                                        #   cat(Sys.time(), "➡️ Starting Year Pair:", pair, "for Tile:", tile_id, "\n", file = log_file, append = TRUE)
                                        
                                        start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                        end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                        response_var <- paste0("b.change.", pair)
                                        
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        
                                        # -------------------------------------------------------
                                        # 3. FILTER & PREPARE TRAINING DATA
                                        # -------------------------------------------------------
                                        cat(Sys.time(), "📌 Tile:", tile_id, "| Year Pair:", pair, "| Available Predictors:", 
                                            paste(predictors, collapse = ", "), "| Response:", response_var, "\n", 
                                            file = log_file, append = TRUE)
                                        
                                        subgrid_data <- training_data %>%
                                          dplyr::select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                          drop_na()
                                        
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                              file = log_file, append = TRUE)
                                          next  # Move to next year pair
                                        }
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                          cat(Sys.time(), "🚨 ERROR: No predictors found for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        
                                        if (any(!is.finite(as.matrix(subgrid_data[, predictors, drop = FALSE])))) {
                                          cat(Sys.time(), "🚨 ERROR: NA/NaN/Inf detected in predictors\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        cat(Sys.time(), "📢 Training XGBoost model for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                        
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
                                          
                                          
                                          for (j in seq_along(predictors)) { 
                                            grid <- data.frame(x = EnvRanges[[predictors[j]]])
                                            grid$y <- predict(xgb_model, newdata = as.matrix(grid$x))
                                            
                                            # Remove NAs and check unique values
                                            grid <- grid[complete.cases(grid), ]
                                            if (length(unique(grid$x)) < 5) {
                                              cat(Sys.time(), "⚠️ Skipping PDP for Predictor:", predictors[j], "- Too few unique values\n", file = log_file, append = TRUE)
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
                                        # 8. SAVE OUTPUTS
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
                                        cat(Sys.time(), "❌ ERROR in Tile:", tiles_df[[i]], "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
                                        return(NULL)
                                      })
                                    }
                          }
  # -------------------------------------------------------
  # 6. CLOSE PARALLEL PROCESSING
  # -------------------------------------------------------
  # stopCluster(cl)
  
  cat("\n✅ Model Training Complete! Check `error_log.txt` for any issues.\n")
}

# V4the same as the above, but modified PDP and env range data keeping %dopar% WORKS Great
registerDoParallel(detectCores() - 1) 
Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 20) { 
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n🚀 Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)  # ✅ Enable parallel processing
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
                                          cat(Sys.time(), "⚠️ Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                                        training_data <- as.data.frame(training_data)
                                        
                                        # missing_static <- setdiff(static_predictors, names(training_data))
                                        # if (length(missing_static) > 0) {
                                        #   cat(Sys.time(), "🚨 ERROR: Static predictors missing for Tile", tile_id, ":", paste(missing_static, collapse = ", "), "\n", file = log_file, append = TRUE)
                                        #   return(NULL)
                                        # }
                                        # 
                                        # for (pair in year_pairs) {
                                        #   cat(Sys.time(), "➡️ Starting Year Pair:", pair, "for Tile:", tile_id, "\n", file = log_file, append = TRUE)
                                        
                                        start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                        end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                        response_var <- paste0("b.change.", pair)
                                        
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        
                                        # -------------------------------------------------------
                                        # 3. FILTER & PREPARE TRAINING DATA
                                        # -------------------------------------------------------
                                        cat(Sys.time(), "📌 Tile:", tile_id, "| Year Pair:", pair, "| Available Predictors:", 
                                            paste(predictors, collapse = ", "), "| Response:", response_var, "\n", 
                                            file = log_file, append = TRUE)
                                        
                                        subgrid_data <- training_data %>%
                                          dplyr::select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                          drop_na()
                                        
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                              file = log_file, append = TRUE)
                                          next  # Move to next year pair
                                        }
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                        EnvRanges <- expand.grid(
                                          Env_Value = seq(0, 1, length.out = 100),  # Placeholder (Will be replaced)
                                          Predictor = predictors
                                        )
                                        
                                        # Update with Actual Ranges
                                        for (pred in predictors) {
                                          EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(PredMins[pred], PredMaxs[pred], length.out = 100)
                                        }
                                        
                                        # Set up PDP storage: [100 (Env Values) x N Predictors x N Bootstraps]
                                        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
                                        dimnames(PD)[[2]] <- predictors
                                        dimnames(PD)[[3]] <- paste0("Rep_", seq_len(n.boot))
                                        
                                        
                                        # -------------------------------------------------------
                                        # 6. MODEL TRAINING & BOOTSTRAPPING
                                        # -------------------------------------------------------
                                        if (length(predictors) == 0) {
                                          cat(Sys.time(), "🚨 ERROR: No predictors found for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        
                                        if (any(!is.finite(as.matrix(subgrid_data[, predictors, drop = FALSE])))) {
                                          cat(Sys.time(), "🚨 ERROR: NA/NaN/Inf detected in predictors\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        cat(Sys.time(), "📢 Training XGBoost model for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                        
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
                                          
                                          for (j in seq_along(predictors)) { 
                                            grid <- data.frame(Env_Value = EnvRanges$Env_Value[EnvRanges$Predictor == predictors[j]])
                                            grid$PDP_Value <- predict(xgb_model, newdata = as.matrix(grid$Env_Value))
                                            
                                            # Remove NAs and check unique values
                                            grid <- grid[complete.cases(grid), ]
                                            if (length(unique(grid$Env_Value)) < 5) {
                                              cat(Sys.time(), "⚠️ Skipping PDP for Predictor:", predictors[j], "- Too few unique values\n", file = log_file, append = TRUE)
                                              next
                                            }
                                            
                                            # Apply loess smoothing (fallback to raw if needed)
                                            loess_fit <- tryCatch(loess(PDP_Value ~ Env_Value, data = grid, span = 1), error = function(e) NULL)
                                            PD[, j, b] <- if (!is.null(loess_fit)) predict(loess_fit, newdata = grid$Env_Value) else grid$PDP_Value
                                            
                                            # Store results in long format
                                            PDP_Storage[[j]] <- data.frame(
                                              Predictor = predictors[j],
                                              Env_Value = EnvRanges$Env_Value[EnvRanges$Predictor == predictors[j]],
                                              Replicate = paste0("Rep_", b),
                                              PDP_Value = PD[, j, b]
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
                                        
                                        # Save PDP and Env Values Together
                                        write_fst(as.data.table(PDP_Long), file.path(tile_dir, paste0("pdp_data_", pair, ".fst")))  
                                        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))  
                                        
                                        
                                        writeLines(paste(Sys.time(), "📁 Saved Model for Tile:", tile_id, "| Year Pair:", pair), log_file, append = TRUE)
                                        
                                        # }  # End for-loop over year_pairs
                                        
                                      }, error = function(e) {
                                        cat(Sys.time(), "❌ ERROR in Tile:", tiles_df[[i]], "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
                                        return(NULL)
                                      })
                                    }
                          }
  # -------------------------------------------------------
  # 6. CLOSE PARALLEL PROCESSING
  # -------------------------------------------------------
  # stopCluster(cl)
  
  cat("\n✅ Model Training Complete! Check `error_log.txt` for any issues.\n")
}

# V5 same as above but with extra spatial info to be included in PDP data works great

registerDoParallel(detectCores() - 1) 
Model_Train_XGBoost <- function(training_sub_grids_UTM, output_dir_train, year_pairs, n.boot = 20) { 
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\n🚀 Starting XGBoost Model Training with Bootstrapping...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)  # ✅ Enable parallel processing
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
                                          cat(Sys.time(), "⚠️ Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        training_data <- read_fst(training_data_path, as.data.table = TRUE)  
                                        training_data <- as.data.frame(training_data)
                                        
                                        # missing_static <- setdiff(static_predictors, names(training_data))
                                        # if (length(missing_static) > 0) {
                                        #   cat(Sys.time(), "🚨 ERROR: Static predictors missing for Tile", tile_id, ":", paste(missing_static, collapse = ", "), "\n", file = log_file, append = TRUE)
                                        #   return(NULL)
                                        # }
                                        # 
                                        # for (pair in year_pairs) {
                                        #   cat(Sys.time(), "➡️ Starting Year Pair:", pair, "for Tile:", tile_id, "\n", file = log_file, append = TRUE)
                                        
                                        start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
                                        end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
                                        response_var <- paste0("b.change.", pair)
                                        
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        
                                        # -------------------------------------------------------
                                        # 3. FILTER & PREPARE TRAINING DATA
                                        # -------------------------------------------------------
                                        cat(Sys.time(), "📌 Tile:", tile_id, "| Year Pair:", pair, "| Available Predictors:", 
                                            paste(predictors, collapse = ", "), "| Response:", response_var, "\n", 
                                            file = log_file, append = TRUE)
                                        
                                        subgrid_data <- training_data %>%
                                          dplyr::select(all_of(c(predictors, response_var, "X", "Y", "FID"))) %>%
                                          drop_na()
                                        
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        if (!response_var %in% names(training_data)) {
                                          cat(Sys.time(), "🚨 ERROR: Response variable missing for Tile:", tile_id, "| Year Pair:", pair, "\n", 
                                              file = log_file, append = TRUE)
                                          next  # Move to next year pair
                                        }
                                        if (nrow(subgrid_data) == 0) {
                                          cat(Sys.time(), "❌ No valid data after filtering NA for Tile:", tile_id, "| Year Pair:", pair, "\n", 
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
                                            cat(Sys.time(), "⚠️ Skipping predictor:", pred, "- Invalid range\n", file = log_file, append = TRUE)
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
                                          cat(Sys.time(), "🚨 ERROR: No predictors found for Tile:", tile_id, "| Year Pair:", pair, "\n", file = log_file, append = TRUE)
                                          next
                                        }
                                        
                                        
                                        if (any(!is.finite(as.matrix(subgrid_data[, predictors, drop = FALSE])))) {
                                          cat(Sys.time(), "🚨 ERROR: NA/NaN/Inf detected in predictors\n", file = log_file, append = TRUE)
                                          return(NULL)
                                        }
                                        
                                        cat(Sys.time(), "📢 Training XGBoost model for Tile:", tile_id, "| Year Pair:", pair, "\n")
                                        
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
                                              cat(Sys.time(), "⚠️ Skipping PDP for Predictor:", predictors[j], "- No valid values\n", file = log_file, append = TRUE)
                                              next
                                            }
                                            
                                            # Predict PDP Values (Ensure correct matrix shape)
                                            grid$PDP_Value <- predict(xgb_model, newdata = matrix(grid$Env_Value, ncol = 1))
                                            
                                            # Remove NAs and check unique values
                                            grid <- grid[complete.cases(grid), ]
                                            if (length(unique(grid$Env_Value)) < 5) {
                                              cat(Sys.time(), "⚠️ Skipping PDP for Predictor:", predictors[j], "- Too few unique values\n", file = log_file, append = TRUE)
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
                                        
                                        # Save PDP and Env Values Together
                                        write_fst(as.data.table(PDP_Long), file.path(tile_dir, paste0("pdp_data_", pair, ".fst")))  
                                        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))  
                                        write_fst(as.data.table(bind_rows(PDP_Storage)), file.path(tile_dir, paste0("pdp_data_raw_", pair, ".fst")))
                                        
                                        
                                        writeLines(paste(Sys.time(), "📁 Saved Model for Tile:", tile_id, "| Year Pair:", pair), log_file, append = TRUE)
                                        
                                        # }  # End for-loop over year_pairs
                                        
                                      }, error = function(e) {
                                        cat(Sys.time(), "❌ ERROR in Tile:", tiles_df[[i]], "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
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
# ✅ **Run Model Training**
Sys.time()
Model_Train_XGBoost(
  training_sub_grids_UTM,
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles",
  year_pairs = c("2004_2006", "2006_2010", "2010_2015", "2015_2022"),
  n.boot = 20
)
Sys.time()


 showConnections()

 closeAllConnections()

# Verify model outputs look good
deviance <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/deviance_2004_2006.fst")
glimpse(deviance)
influence <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/influence_2004_2006.fst")
glimpse(influence)
boots <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/bootstraps_2004_2006.fst")
glimpse(boots)
envranges <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/pdp_env_ranges_2004_2006.fst")
glimpse(envranges)
pdp <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/pdp_data_2004_2006.fst")
glimpse(pdp)


##3 NEW EXTRACT XGB MODEL OUTPUTS / AVERAGES ----

library(dplyr)
library(readr)
library(fst)
library(ggplot2)

# Directories
tiles_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
# Initialize storage for overall performance metrics
all_tile_metrics <- list()

# Loop through each tile
for (tile in tile_folders) {
  
  tile_id <- basename(tile)  # Extract tile name
  tile_metrics_list <- list()  # Store per tile
  
  # Loop through each year pair
  for (pair in year_pairs) {
    
    # Construct file paths
    boots_file <- file.path(tile, paste0("bootstraps_", pair, ".fst"))
    deviance_file <- file.path(tile, paste0("deviance_", pair, ".fst"))
    
    if (file.exists(boots_file) && file.exists(deviance_file)) {
      
      # Load the data
      boots <- read_fst(boots_file)
      deviance <- read_fst(deviance_file)
      
      # Compute mean prediction, standard deviation, CV, residuals
      boot_mat <- boots %>% select(starts_with("Boot_")) %>% as.matrix()
      boot_mean <- rowMeans(boot_mat, na.rm = TRUE)
      boot_sd <- apply(boot_mat, 1, sd, na.rm = TRUE)
      boot_cv <- boot_sd / boot_mean
      residuals <- abs(boots$Actual_Change - boot_mean)
      
      # Compute model performance metrics (Deviance Exp., RMSE, R²)
      if (nrow(deviance) > 0) {  # ✅ Ensure deviance is not empty
        mean_dev_exp <- mean(deviance$Dev.Exp, na.rm = TRUE)
        mean_rmse <- mean(deviance$RMSE, na.rm = TRUE)
        mean_r2 <- mean(deviance$R2, na.rm = TRUE)
        
        # Store per-tile results
        metrics_df <- data.frame(
          X = boots$X, Y = boots$Y,
          boot_mean = boot_mean,
          residuals = residuals,
          boot_sd = boot_sd,
          boot_cv = boot_cv
        )
        
        # Save per-tile, per-year summary
        write_csv(metrics_df, file.path(tile, paste0("summary_metrics_", pair, ".csv")))
        
        # Store tile-level performance **only if valid data exists**
        tile_metrics_list[[pair]] <- c(mean_dev_exp, mean_rmse, mean_r2)
      }
    }
  }
  
  # Convert tile-level metrics to a dataframe only if there are valid results
  if (length(tile_metrics_list) > 0) {
    tile_metrics_df <- do.call(rbind, tile_metrics_list) %>%
      as.data.frame() %>%
      setNames(c("Deviance_Exp", "RMSE", "R2")) %>%
      mutate(Year_Pair = names(tile_metrics_list), Tile_ID = tile_id)  # Fix rownames issue
    
    # Save the tile-level summary
    write_csv(tile_metrics_df, file.path(tile, "tile_performance_summary.csv"))
    
    # Store for overall computation
    all_tile_metrics[[tile_id]] <- tile_metrics_df
  }
}

# --- Compute Overall Performance Across All Tiles ---
overall_metrics_df <- bind_rows(all_tile_metrics)

# Compute mean performance across all tiles
# Compute mean performance across all tiles
overall_summary <- overall_metrics_df %>%
  group_by(Year_Pair) %>%
  summarise(across(c(Deviance_Exp, RMSE, R2), \(x) mean(x, na.rm = TRUE)))


# Save results
write_csv(overall_metrics_df, file.path(output_dir, "all_tiles_performance_summary.csv"))
write_csv(overall_summary, file.path(output_dir, "overall_performance_summary.csv"))

# --- Visualization part 1  ----
# Boxplot: Predictions across tiles and years
ggplot(overall_metrics_df, aes(x = Year_Pair, y = Deviance_Exp, fill = Year_Pair)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Deviance Explained Across Tiles", x = "Year Pair", y = "Deviance Explained")

library(dplyr)
library(fst)
library(ggplot2)

# Directories
tiles_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"

# Year pairs to loop through
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")

# List all tile folders
tile_folders <- list.dirs(tiles_dir, recursive = FALSE)

# Storage for combined data
all_data <- list()

# Loop through each year pair
for (pair in year_pairs) {
  
  year_pair_data <- list()  # Store per-year results
  
  # Loop through each tile
  for (tile in tile_folders) {
    
    boots_file <- file.path(tile, paste0("bootstraps_", pair, ".fst"))
    
    if (file.exists(boots_file)) {
      
      # Read bootstrap file
      boots <- read_fst(boots_file)
      
      # Compute mean predicted change from bootstraps
      boot_mat <- boots %>% select(starts_with("Boot_")) %>% as.matrix()
      boot_mean <- rowMeans(boot_mat, na.rm = TRUE)
      
      # Store actual vs. predicted data
      df <- data.frame(
        Actual_Change = boots$Actual_Change,
        Predicted_Change = boot_mean,
        Year_Pair = pair
      )
      
      year_pair_data[[tile]] <- df
    }
  }
  
  # Combine all tile data for this year pair
  if (length(year_pair_data) > 0) {
    all_data[[pair]] <- bind_rows(year_pair_data)
  }
}

# Combine all year pairs into one dataset
scatter_data <- bind_rows(all_data)

# Create 4-panel scatter plot
p1 <- ggplot(scatter_data, aes(x = Actual_Change, y = Predicted_Change)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +  # Reference line y = x
  facet_wrap(~Year_Pair, scales = "free") +
  labs(
    title = "Actual vs Predicted Elevation Change",
    x = "Actual Change",
    y = "Mean Predicted Change"
  ) +
  theme_minimal()

ggsave("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/test.plot.tif", plot = p1, width = 5, height = 3, units = "in", dpi = 300)

ggsave("test.plot2.tif", plot = p1, width = 5, height = 3, units = "in", dpi = 300)

# Visualisation part 2 ----
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)  # For side-by-side plots

# Sample Data (replace with your actual dataframe)
df <- overall_metrics_df

# 1️⃣ **Performance Across Tiles (Point Scatter with Lines)**
plot_per_tile <- df %>%
  pivot_longer(cols = c(Deviance_Exp, RMSE, R2), names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = Tile_ID, y = Value, color = Year_Pair, group = Tile_ID)) +
  geom_point(alpha = 0.7, size = 2) +  # Make points more visible
  geom_line(alpha = 0.3) +  # Connect points to show trends
  facet_wrap(~Metric, scales = "free_y") +
  labs(
    x = "Tile ID",
    y = "Performance Metric",
    title = "Model Performance by Tile"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_blank(),  # Too many tiles; remove labels
    axis.ticks.x = element_blank(),
    strip.text = element_text(size = 10, face = "bold"),
    panel.grid.major.x = element_blank(),
    legend.position = "top"
  )


# 2️⃣ **Overall Performance Across All Tiles (Density + Histogram)**
# 2️⃣ **Overall Performance Across All Tiles (Density + Histogram)**
plot_overall <- df %>%
  pivot_longer(cols = c(Deviance_Exp, RMSE, R2), names_to = "Metric", values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Metric)) +
  geom_histogram(aes(y = after_stat(density)), bins = 20, alpha = 0.5, position = "identity") +  # ✅ Updated
  geom_density(alpha = 0.7) +
  facet_wrap(~Metric, scales = "free_x") +
  labs(
    x = "Metric Value",
    y = "Density",
    title = "Overall Model Performance Distribution"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(size = 10, face = "bold"),
    legend.position = "none"
  )


# 3️⃣ **Combine Both Using patchwork**
final_plot <- plot_per_tile / plot_overall  # Stack vertically

# Save it!
ggsave("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/Model_Performance_Visuals.pdf", final_plot, width = 12, height = 8, dpi = 300)

# Model Predictor Influence code and plots ----
# Load Libraries
library(tidyverse)
library(fst)

# Directories
tiles_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
output_dir <- tiles_dir  # Save summaries here

# Year pairs
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")

# List all tile folders
tile_folders <- list.dirs(tiles_dir, recursive = FALSE)

# Storage for summaries
influence_summary_year <- list()
influence_summary_tile <- list()

# Loop through each year pair
for (pair in year_pairs) {
  
  all_tile_influence <- list()  # Store influence per tile
  
  # Loop through each tile folder
  for (tile in tile_folders) {
    
    # Define file path
    influence_file <- file.path(tile, paste0("influence_", pair, ".fst"))
    
    if (file.exists(influence_file)) {
      
      # Load influence data
      influence_data <- read_fst(influence_file)
      
      # Compute mean influence across replicates
      influence_mean <- influence_data %>%
        pivot_longer(cols = starts_with("Rep_"), names_to = "Replicate", values_to = "Influence") %>%
        group_by(Predictor) %>%
        summarise(Mean_Influence = mean(Influence, na.rm = TRUE), .groups = "drop") %>%
        mutate(Tile_ID = basename(tile), Year_Pair = pair)
      
      # Store per tile
      all_tile_influence[[tile]] <- influence_mean
    }
  }
  
  # Merge influence across tiles for this year pair
  if (length(all_tile_influence) > 0) {
    influence_summary_year[[pair]] <- bind_rows(all_tile_influence)
  }
}


# Combine all results
df_influence_year <- bind_rows(influence_summary_year)
df_influence_tile <- df_influence_year %>%
  group_by(Tile_ID, Predictor) %>%
  summarise(Mean_Influence = mean(Mean_Influence, na.rm = TRUE), .groups = "drop")

# # 🏷️ Clean Predictor Names (Remove Year Suffix)
# df_influence_year <- df_influence_year %>%
#   mutate(Predictor = str_remove(Predictor, "_\\d{4}"))

# 🏷️ Standardize Predictor Names (Remove All Year Suffixes)
df_influence_year <- df_influence_year %>%
  mutate(Predictor = str_replace_all(Predictor, "_\\d{4}", "")) %>%  # Remove any year suffix
  group_by(Year_Pair, Predictor) %>%  # Now merge duplicates
  summarise(Mean_Influence = mean(Mean_Influence, na.rm = TRUE), .groups = "drop")



# Save as CSV files
write_csv(df_influence_year, file.path(output_dir, "Influence_Summary_By_Year_Pair.csv"))
write_csv(df_influence_tile, file.path(output_dir, "Influence_Summary_By_Tile.csv"))

# 🎨 **Create the faceted bar plot**
plot_influence <- ggplot(df_influence_year, aes(x = reorder(Predictor, -Mean_Influence), y = Mean_Influence, fill = Year_Pair)) +
  geom_bar(stat = "identity", position = "dodge", alpha = 0.7) +
  facet_wrap(~Year_Pair, scales = "free_y", ncol = 2) +  # Split into subplots
  labs(
    x = "Predictor",
    y = "Mean Influence on elevation change",
    title = "Predictor Influence by Year Pair"
  ) +
  theme_minimal(base_size = 16) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.text = element_text(size = 16, face = "bold"),
    legend.position = "none"  # Remove redundant legend
  )

# Save the figure
ggsave("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/Influence_Summary_plots.pdf", plot_influence, width = 12, height = 8, dpi = 300)


### NEW PARTIAL DEPENDANE PLOTS 
pdp_data_env_ranges <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/pdp_env_ranges_2010_2015.fst")
pdp_data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/pdp_data_2010_2015.fst")
pdp_data2 <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/pdp_data_2015_2022.fst")
pdp_data3 <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_1/pdp_data_2015_2022.fst")

training_data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/BH4S656W_4_training_clipped_data.fst")
glimpse(pdp_data)
glimpse(pdp_data_env_ranges)

# Load required libraries
library(dplyr)
library(fst)
library(ggplot2)

# Directories
tiles_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"

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
      
      # ✅ Load PDP data
      pdp_data <- read_fst(pdp_file)
      
      # 🔍 Debug: Check column names
      if (!"Env_Value" %in% colnames(pdp_data)) {
        cat("🚨 ERROR: `Env_Value` is missing in:", pdp_file, "\n")
        cat("Available columns:", paste(colnames(pdp_data), collapse = ", "), "\n")
        next  # Skip this tile
      }
      
      # ✅ Ensure `Env_Value` is numeric
      pdp_data <- pdp_data %>%
        mutate(Env_Value = as.numeric(Env_Value))
      
      # ✅ Skip invalid PDP files
      if (any(is.na(pdp_data$Env_Value))) {
        cat("🚨 Skipping invalid PDP file (contains NAs):", pdp_file, "\n")
        next
      }
      
      # ✅ Aggregate PDP Data Using Its Own `Env_Value`
      # ✅ Aggregate PDP Data Using Its Own `Env_Value`
      pdp_data_agg <- pdp_data %>%
        select(Predictor, Env_Value, PDP_Value)  # ✅ Keep PDP_Value!
      
      
      
      
      
      
      
      # ✅ Store aggregated data
      pdp_all_tiles[[tile]] <- pdp_data_agg
    } else {
      cat("⚠️ Skipping missing PDP file:", pdp_file, "\n")
    }
  }
  
  # ✅ Merge all available tiles into a single dataframe
  if (length(pdp_all_tiles) > 0) {
    overall_pdp_data <- bind_rows(pdp_all_tiles)
    
    pdp_summary <- overall_pdp_data %>%
      group_by(Predictor, Env_Value) %>%
      summarise(
        PDP_Mean = mean(PDP_Value, na.rm = TRUE),
        PDP_Min  = min(PDP_Value, na.rm = TRUE),
        PDP_Max  = max(PDP_Value, na.rm = TRUE),
        .groups = "drop"
      )
    
    
    
    # # ✅ Compute overall mean but keep min/max as actual min/max
    # pdp_summary <- overall_pdp_data %>%
    #   group_by(Predictor, Env_Value) %>%
    #   summarise(
    #     PDP_Mean = mean(PDP_Value, na.rm = TRUE),  # ✅ Now PDP_Value exists!
    #     PDP_Min = min(PDP_Value, na.rm = TRUE),
    #     PDP_Max = max(PDP_Value, na.rm = TRUE),
    #     .groups = "drop"
    #   )
   
    
    
    
    pdp_summary <- pdp_summary %>%
      mutate(
        PDP_Max = ifelse(PDP_Min == PDP_Max, PDP_Max + 1e-5, PDP_Max) # Add small variation
      )
    
    
    # ✅ Print a Sample to Check for NA Issues
    print(paste0("✅ Processed PDP for ", pair, ":"))
    print(head(pdp_summary))
    
    # ✅ Store for plotting
    all_pdp_list[[pair]] <- pdp_summary
  } else {
    cat("🚨 No available PDP data for", pair, "- Skipping this year pair.\n")
  }
}

# ✅ Create PDP Plots (Only for available year pairs)
for (pair in names(all_pdp_list)) {
  
  # ✅ Ensure `PDP_Min` and `PDP_Max` are valid before plotting
  pdp_filtered <- all_pdp_list[[pair]] %>%
    filter(!is.na(PDP_Min), !is.na(PDP_Max))
  
  
  
  # 🚨 If no valid data remains after filtering, skip
  if (nrow(pdp_filtered) == 0) {
    cat("⚠️ Skipping PDP plot for", pair, "as no valid data remains.\n")
    next
  }
  
  plot_pdp <- ggplot(pdp_filtered, aes(x = Env_Value, y = PDP_Mean)) +
    geom_ribbon(aes(ymin = PDP_Min, ymax = PDP_Max), fill = "grey70") +
    geom_smooth(method = "loess", se = FALSE, color = "black", linewidth = 1) +
    facet_wrap(~Predictor, scales = "free_y", ncol = 3) +  # 🔥 Allow Y-axis to scale naturally
    scale_x_continuous(
      breaks = seq(
        floor(min(pdp_filtered$Env_Value, na.rm = TRUE)),
        ceiling(max(pdp_filtered$Env_Value, na.rm = TRUE)),
        by = 5  
      )
    ) +
    scale_y_continuous(
      breaks = seq(
        floor(min(pdp_filtered$PDP_Min, na.rm = TRUE) * 10) / 10,
        ceiling(max(pdp_filtered$PDP_Max, na.rm = TRUE) * 10) / 10,
        by = 0.2
      )
    ) +
    labs(
      x = "Model Predictor Value",
      y = "Mean elevation change (m)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_blank(),  
      strip.text = element_text(size = 10, face = "bold"),  
      axis.title = element_text(size = 12),                
      axis.text = element_text(size = 8),  
      axis.text.x = element_text(angle = 45, hjust = 1, margin = margin(t = 5)),  # 🔥 Fixed duplicate issue
      strip.background = element_rect(fill = "lightgray"),
      panel.grid.major = element_line(linewidth = 0.2, linetype = "dotted"),
      panel.grid.minor = element_blank(),
      legend.position = "none",
      plot.margin = margin(5, 5, 5, 5)
    )
  
  
  
  
  
  
  
  # ✅ Save the final PDP plot
  ggsave(filename = file.path(output_dir, paste0("Overall_PDP_", pair, ".pdf")), 
         plot = plot_pdp, width = 12, height = 6, dpi = 300) # saved colors in a vectorised format to store detail
  
}


#3. Evaluate performance metrics (TO DO)----
# when the model para-maters are tweeked, the model performance changes. We want to evaluate the model performance, to make changes to
# to the model training stage, before we progress to the model prediction.


# # ## DOUBLE CHECK IT SPATIALLY!----
prediction.rast.actual.avg.change <- rasterFromXYZ(data.frame(x = prediction_data[,"X"],
                                      y = prediction_data[,"b.change"]),
                           crs = crs(mask))

prediction.rast <- rasterFromXYZ(data.frame(x = prediction_data[,"X"],
                                                              y = prediction_data[,"Y"],
                                                              z = prediction_data[, "b.change"]),
                                                   crs = crs(mask))

plot(prediction.rast.actual.avg.change)
plot(prediction.rast.trend.adj.change)




# V5 - WF 2.0 - New prediction function set---- below is the XGB model version
pdp_data <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/pdp_data_2004_2006.fst")
glimpse(pdp_data1)
training_data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S656W_4/BH4S656W_4_training_clipped_data.fst")
prediction_data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/BH4S656W_4/BH4S656W_4_prediction_clipped_data.fst")
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
  message("🔎 Comparing PDP vs KNN for: ", tile_id, " | ", year_pair)
  
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
  
  message("✅ Assigned KNN predictions to ", sum(!is.na(prediction_data$pred_avg_change)), " rows.")
  
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

training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"
# run on all tiles
run_predictions(tile_ids, year_pairs, training_dir, prediction_dir, output_dir, num_cores = 1)


# Testing prediction on a single tile(s)----- 
# Mask reference
mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif")

# Only this training tile will be available for reference
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"
prediction_output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"

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












