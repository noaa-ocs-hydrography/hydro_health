
# PRE-PROCESSING (STAGE 1): - this will be its own script (aka engine)
# 1. Extract all survey end date data from .xml (Raster Attribute Table) and create indavidaul rasters
#    Merge all individual survey end date rasters into a combined raster (currently mosaic via ArcGIS)
# 2. Standardize all model rasters (created in GIS /other) and clip to both prediction.mask and training.mask
     #ensure all data sets are in the same crs / projection, resolution (5m), and clip to different masks to ensure same size
#    also ensure all bathy elevation values above 0m are removed after clipping, if not filter (0m mark varies on boundary)
# 3. Convert prediction.mask and training.mask in to a Spatial Points Dataframe for sub grid processing
# 4. Load the blue topo grid tile gpkg and create a sub grid (by dividing it into 4 squares) for both the prediction and training mask extent
#    Create subset data frames of all the processed raster data (model variables), into the each sub grid tile folder, over the grid extent for model training


# ─────────────────────────────────────────────
# PREPROCESSING MODULE: Full Workflow (Steps 1–5)
# ─────────────────────────────────────────────
# 1. Load packages and global paramaters 
# 2. Define Directories 
# 3. Define Functions
# 4. Run Functions
    # F1 - Fill NA values in raw bathymetry data using Focal statistics iteratively 
    # F2 - Create training mask by merging all part-processed bathymetry data from F1
    # F3 - Create prediction mask from shapefile (* this will later be defined through the automated grid script as there will be multiple grid tiles for prediction)
    # F4 - Create Spatial Dataframes from prediction and training extent masks (also WGS and UTM copies)
    # F5 - Extract Survey End dates from blue topo xml / RAT data 
    # F6 - Function to split blue topo grid into sub grid / divide by 4 
    # F7 - Prepares sub grid by intersecting blue topo geopackage using F5 (In WGS84)
    # F8 - Re-projects the sub grid into UTM for model use
    # F9 -  Standardize all model rasters for prediction and training extent ( projection, resolution [8m], extent) 
    # F10 - Parallel tile chunking **(chunks out data into gridded format if starting from larger format, only required for PILOT MODEL)
# 5. Done / Close Parallel 

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
    log_message(paste(" ", layer_name, "- Disk-Based Focal Fill Iteration", i, "of", n_repeats))
    
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
      log_message(paste(" Focal failed at iteration", i, "-", e$message))
    })
  }
  
  # Rename final result
  if (file.exists(temp_file)) {
    file.rename(temp_file, output_final)
    log_message(paste(" Final filled raster saved as:", basename(output_final)))
  } else {
    log_message(paste(" Final result file was not created for", layer_name))
  }
}


# ──────────────
# F1.3 FILL WITH FALLBACK STRATEGY
# ──────────────
fill_with_fallback <- function(input_file, output_file, max_iters = 10, fallback_repeats = 5, w = 3) {
  layer_name <- basename(input_file)
  
  tryCatch({
    r <- raster(input_file)
    log_message(paste(" Attempting iterative fill for", layer_name))
    r_filled <- iterative_focal_fill(r, max_iters = max_iters, w = w)
    writeRaster(r_filled, filename = output_file, format = "GTiff", overwrite = TRUE)
    log_message(paste(" Iterative fill succeeded:", layer_name))
  }, error = function(e) {
    log_message(paste(" Iterative fill failed for", layer_name, "-", e$message))
    
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
      log_message(paste(" Fallback fill also failed for", layer_name, "-", e2$message))
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
    log_message(" Falling back to sequential fill method...")
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
  log_message(" Preparing sub-grids from tile scheme...")
  
  mask_sf <- st_as_sf(mask_df, coords = c("X", "Y"), crs = 4326)
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) split_tile_into_quadrants(grid_tiles[i, ])))
  
  log_message(paste("🔹 Total sub-grids generated:", nrow(sub_grids)))
  
  intersecting <- st_filter(sub_grids, st_union(mask_sf))
  log_message(paste(" Sub-grids intersecting mask:", nrow(intersecting)))
  
  out_gpkg <- file.path(output_dir, "intersecting_sub_grids_WGS84.gpkg")
  out_rds  <- file.path(output_dir, "grid_tile_extents_WGS84.rds")
  
  st_write(intersecting, out_gpkg, delete_layer = TRUE, quiet = TRUE)
  saveRDS(intersecting, out_rds)
  
  log_message(paste(" Saved sub-grids to:", out_gpkg))
  return(intersecting)
}

## F8 - Re projects sub grid geopackage into desired projection - UTM 
reproject_subgrids_to_utm <- function(input_gpkg, output_gpkg, target_crs) {
  log_message(paste(" Reprojecting sub-grids to:", target_crs))
  
  tryCatch({
    sub_grids <- st_read(input_gpkg, quiet = TRUE)
    sub_grids_utm <- st_transform(sub_grids, crs = target_crs)
    st_write(sub_grids_utm, output_gpkg, delete_layer = TRUE, quiet = TRUE)
    log_message(paste(" Reprojected sub-grids saved to:", output_gpkg))
  }, error = function(e) {
    log_message(paste(" Failed to reproject sub-grids:", e$message))
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
cl <- makeCluster(cores)
cores <- 8  # or even 1 to start safely
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

training_mask_df_wgs <- read.fst("training.mask.df.wgs84.rds")
prediction_mask_df_wgs <- read.fst("prediction.mask.df.wgs84.rds")

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

