# Script split into 4  stages:
# 1 - pre processing of raw NOW Coast / BT data / Our created datasets 
# 2 - then running of the Machine Learning Model (ML)
# 3 - Prediction of training relationships 
# 4 - Post processing 

# WORKFLOW:----

# (STAGE 1) PRE-PROCESSING : - (This will be its own engine)
# 1. Extract all survey end date data from xml (Raster Attribute Table) and create indavidaul rasters
#    Merge all individual survey end date rasters into a combined raster (currently mosaic via ArcGIS)
# 2. Standardize all model rasters (created in GIS /other) and clip to both prediction.mask and training.mask
     #ensure all data sets are in the same crs / projection, resolution (5m), and clip to different masks to ensure same size
#    also ensure all bathy elevation values above 0m are removed after clipping, if not filter (0m mark varies on boundary)
# 3. Convert prediction.mask and training.mask in to a Spatial Points Dataframe for sub grid processing
# 4. Load the blue topo grid tile gpkg and create a sub grid (by dividing it into 4 squares) for both the prediction and training mask extent
#    Create subset data frames of all the processed raster data (model variables), into the each sub grid tile folder, over the grid extent for model training

# (STAGE 2) MODEL TRAINING, GENERATING PDPS AND EVALUATION METRICS : (This will be its own engine)
# 1. Train the model over all sub grids
# 2. Create Partial Dependance plots from the model data - WE HAVE THEM ALL FOR EACH TILE, BUT THE AVERAGE PDP OVER STUDY AREA NEEDs COMPUTED
# 3. Evaluate model performance from Ranger RF summary data - PERFORMANCE METRICS ARE SAVED TO CSV, BUT MODEL AND METRICS NEED INTEROGATED TO INFORM CHANGES BEFORE PREDICTION CODE RUN 


# (STAGE 3) MODEL PREDICTION : (This will be its own engine)
# 1. NEEDS UPDATED---  Generate the 10 year average trend of change: from (i) the training /JALBTCX temporal datasets[[actual avg change]], and (ii) the model predictions over the training extent[[predicted avg change]]
# 2. NEEDS UPDATED--- Apply prediction using 10 year average trend, survey age (survey end date), and model varaible relationships to determine change in depth over prediction extent
# 3. Prediction Validation - process needs created. probably best ot compare to how well its predicted over the training extent and compare back. 

# (STAGE 4) POST PROCESSING:
# taking the raw model predictions and applying a depth attentuation coeficient.
# working out how the area has changed over time by applying a hindcasting approach 

# Load Packages
library(raster)
require(xml2)
library(dplyr)
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
library(foreach)
# library(conflicted) # helps overcome conflict issues between packages.

# new STAGE 1 - PRE-PROCESSING MODULE IN PARALLEL----
# ─────────────────────────────────────────────
# PREPROCESSING MODULE: Full Workflow (Steps 1–5)
# ─────────────────────────────────────────────

# ──────────────
# 1. LOG FILE & GLOBAL PARAMS
# ──────────────
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
shapefile_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/GIS/Pilot_model/Pilot_model_prediction_boundary_Final.shp"
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
output_SPDF <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data"
# KML / XML survey end date paths
input_dir_survey <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/UTM17"
kml_dir_survey   <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/RATs"
output_dir_survey_dates <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Modeling/survey_date_end_new"
# training and prediction sub grid GeoPackage 
training_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
prediction_grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg"
# Sub grid tile folder directories
training_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
prediction_subgrid_out <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"

# ──────────────
# 3. DEFINE FUNCTIONS
# ──────────────

## F1 - Function that will attempt to run in parallel, to fill NA values in raw bathymetry data using focal statistics but if memory limit reached----
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

## F2 - Create Training Mask (and WGS Copy) with Boundary Intersect----
create_training_mask <- function(input_dir,
                                 output_mask_utm,
                                 output_mask_wgs,
                                 shapefile_path,
                                 pattern = "_filled\\.tif$") {
  library(raster)
  library(sf)
  
  utm_wgs84_crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"
  
  # Load bathy or predictor rasters
  files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  stopifnot(length(files) > 0)
  
  stack_rasters <- stack(files)
  crs(stack_rasters) <- utm_wgs84_crs  # enforce correct native CRS
  
  # Create binary mask
  mask <- calc(stack_rasters, fun = function(x) if (all(is.na(x))) NA else 1)
  binary_mask <- calc(mask, fun = function(x) ifelse(!is.na(x), 1, 0))
  
  # Read and transform the prediction boundary
  boundary <- st_read(shapefile_path, quiet = TRUE)
  boundary_utm <- st_transform(boundary, crs = utm_wgs84_crs)
  
  binary_mask <- crop(binary_mask, extent(boundary_utm))
  binary_mask <- mask(binary_mask, as(boundary_utm, "Spatial"))
  
  # Save in native CRS (NAD83 / UTM17)
  writeRaster(binary_mask, output_mask_utm, format = "GTiff", overwrite = TRUE)
  message("✅ Training mask saved (UTM NAD83): ", output_mask_utm)
  
  # Optional: project to WGS84
  binary_mask_wgs <- projectRaster(binary_mask, crs = "+proj=longlat +datum=WGS84", method = "ngb")
  writeRaster(binary_mask_wgs, output_mask_wgs, format = "GTiff", overwrite = TRUE)
  message("✅ Training mask saved (WGS84): ", output_mask_wgs)
}
## F3 - Create Prediction Mask from Shapefile----
create_prediction_mask <- function(shapefile_path,
                                   output_mask_utm,
                                   output_mask_wgs) {
  library(raster)
  library(sf)
  
  utm_wgs84_crs <- "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"
  
  poly <- st_read(shapefile_path, quiet = TRUE)
  poly_utm <- st_transform(poly, crs = utm_wgs84_crs)
  
  ext <- extent(poly_utm)
  template <- raster(ext, res = 8, crs = utm_wgs84_crs)
  
  mask_ras <- rasterize(poly_utm, template, field = 1, background = NA)
  mask_bin <- calc(mask_ras, fun = function(x) ifelse(is.na(x), 0, 1))
  
  writeRaster(mask_bin, output_mask_utm, overwrite = TRUE)
  message("✅ Prediction mask saved (UTM NAD83): ", output_mask_utm)
  
  # Optional: WGS84 projection
  mask_bin_wgs <- projectRaster(mask_bin, crs = "+proj=longlat +datum=WGS84", method = "ngb")
  writeRaster(mask_bin_wgs, output_mask_wgs, overwrite = TRUE)
  message("✅ Prediction mask saved (WGS84): ", output_mask_wgs)
}


## F3 old - Create Spatial DF for Masks (UTM + WGS)
# create_spatial_mask_df <- function(mask_utm_path = NULL, mask_wgs_path = NULL, mask_type = "prediction", output_dir = ".") {
#   if (!dir.exists(output_dir)) {
#     dir.create(output_dir, recursive = TRUE)
#   }
#   
#   if (!is.null(mask_utm_path)) {
#     r <- raster::raster(mask_utm_path)
#     pts <- raster::rasterToPoints(r, spatial = TRUE)
#     df <- data.frame(pts@data, X = pts@coords[, 1], Y = pts@coords[, 2])
#     df$FID <- raster::cellFromXY(r, df[, c("X", "Y")])
#     df <- df[df[, 1] == 1, ]
#     
#     out_path_utm <- file.path(output_dir, paste0(mask_type, ".mask.df.utm.fst"))
#     write.fst(df, out_path_utm)
#     log_message(paste("Spatial UTM DF saved to", out_path_utm))
#   }
#   
#   if (!is.null(mask_wgs_path)) {
#     r <- raster::raster(mask_wgs_path)
#     pts <- raster::rasterToPoints(r, spatial = TRUE)
#     df <- data.frame(pts@data, X = pts@coords[, 1], Y = pts@coords[, 2])
#     df$FID <- raster::cellFromXY(r, df[, c("X", "Y")])
#     df <- df[df[, 1] == 1, ]
#     
#     out_path_wgs <- file.path(output_dir, paste0(mask_type, ".mask.df.wgs84.fst"))
#     write.fst(df, out_path_wgs)
#     log_message(paste("Spatial WGS84 DF saved to", out_path_wgs))
#   }
# }

## NEW F4 - Extract blue topo Uncertainty data----
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/NAVD88_download_Feb_25/BlueTopo/UTM17" # Raw BT data
uncertainty_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/NAVD88_download_Feb_25/BlueTopo/Uncertainty" # folder where UC extracted to 
mosaic_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw/bt.UC.tif" # final mosaic of UC / output dir

extract_band2_to_uncertainty <- function(input_dir, uncertainty_dir) {
  tif_files <- list.files(input_dir, pattern = "\\.tiff?$", full.names = TRUE)
  
  if (length(tif_files) == 0) stop("No TIFF files found in input directory.")
  if (!dir.exists(uncertainty_dir)) dir.create(uncertainty_dir, recursive = TRUE)
  
  plan(multisession, workers = 4)
  
  future_lapply(tif_files, function(tif_path) {
    tryCatch({
      r <- brick(tif_path)  # Load multi-band raster
      band2 <- raster(r, 2)  # Extract Band 2 (uncertainty)
      crs(band2) <- crs(r)   # Assign CRS from original
      
      out_name <- file.path(uncertainty_dir, basename(tif_path))
      writeRaster(band2, filename = out_name, format = "GTiff", overwrite = TRUE)
      message("✅ Extracted Band 2 from: ", basename(tif_path))
      rm(r, band2); gc()
    }, error = function(e) {
      message("❌ Failed on ", basename(tif_path), ": ", e$message)
    })
  })
  
  message("✅ Finished extracting Band 2 from all files.")
}

mosaic_uncertainty_rasters <- function(input_dir, mosaic_path) {
  tif_files <- list.files(input_dir, pattern = "\\.tiff?$", full.names = TRUE)
  if (length(tif_files) == 0) stop("No TIFF files found in input directory.")
  
  rasters <- lapply(tif_files, raster)
  mosaic_raster <- Reduce(function(x, y) mosaic(x, y, fun = mean), rasters)
  
  # Apply CRS from the first raster
  crs(mosaic_raster) <- crs(rasters[[1]])
  
  writeRaster(mosaic_raster, filename = mosaic_path, format = "GTiff",
              options = c("COMPRESS=LZW"), datatype = "FLT4S", overwrite = TRUE)
  
  message("✅ Mosaic saved to: ", mosaic_path)
  return(mosaic_raster)
}


## F5 - Extract survey end dates from Blue Topo xml files----
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

## F6 - Split Blue Topo Grid into sub grid (divides grid into 4 subgrids)----
## Function Set 2 -  Split blue topo Gpkg grid (variable grid) into smaller sub grids (uniform grid) using reference tile
split_tile_by_reference <- function(tile, dx, dy) {
  bbox <- st_bbox(tile)  # Get bounding box of the input tile (min/max coordinates)
  
  x_breaks <- seq(from = bbox["xmin"], to = bbox["xmax"], by = dx)  # Create horizontal cut points based on dx
  y_breaks <- seq(from = bbox["ymin"], to = bbox["ymax"], by = dy)  # Create vertical cut points based on dy
  
  sub_tiles <- list()  # Store subgrid tiles here
  id <- 1              # Unique ID for each sub-tile within the parent tile
  
  for (i in seq_len(length(x_breaks) - 1)) {
    for (j in seq_len(length(y_breaks) - 1)) {
      
      # Construct bounding box corners (clockwise from lower-left)
      coords <- matrix(c(
        x_breaks[i],     y_breaks[j],
        x_breaks[i + 1], y_breaks[j],
        x_breaks[i + 1], y_breaks[j + 1],
        x_breaks[i],     y_breaks[j + 1],
        x_breaks[i],     y_breaks[j]
      ), ncol = 2, byrow = TRUE)
      
      poly <- st_polygon(list(coords)) %>% st_sfc(crs = st_crs(tile))  # Convert coordinates to polygon with inherited CRS
      
      sub_tile <- st_sf(
        tile_id = paste0(tile$tile, "_", id),  # Unique ID per sub-tile
        parent_tile = tile$tile,               # Track which parent tile this came from
        geometry = poly
      )
      
      sub_tiles[[id]] <- sub_tile  # Append to list
      id <- id + 1
    }
  }
  
  do.call(rbind, sub_tiles)  # Combine list into single sf object
} # Splitting function


## F7 Generate full subgrid dataset ----
generate_subgrids_from_mask_wgs84 <- function(
    grid_tiles_gpkg,
    output_subgrid_gpkg,
    reference_tile_id = "BH4S257K", # any of the smaller inner blue topo tiles
    mask_path = NULL,
    layer_name = "prediction_subgrid",
    workers = 4
) {
  message("Reading grid tiles...")
  grid_tiles <- st_read(grid_tiles_gpkg, quiet = TRUE)         # Load full grid tile layer
  ref_tile <- grid_tiles %>% filter(tile == reference_tile_id) # Extract reference tile to determine dx/dy
  
  message("Calculating reference subgrid size...")
  ref_bbox <- st_bbox(ref_tile)  # Get bounding box of reference tile
  dx <- (ref_bbox["xmax"] - ref_bbox["xmin"]) / 2  # Half width of reference tile
  dy <- (ref_bbox["ymax"] - ref_bbox["ymin"]) / 2  # Half height of reference tile
  message(paste("Subgrid dx/dy:", dx, dy))
  
  message("Splitting all tiles into subgrids...")
  plan(multisession, workers = workers)  # Set up parallel plan for `future_lapply`
  
  all_subgrids <- future_lapply(seq_len(nrow(grid_tiles)), function(i) {
    split_tile_by_reference(grid_tiles[i, ], dx, dy)  # Apply splitting function to each tile
  }) %>% bind_rows()  # Combine all results into single sf object
  
  # Optional: mask filter
  if (!is.null(mask_path)) {
    message("Filtering subgrids using raster mask...")
    mask <- terra::rast(mask_path)  # Load binary mask raster
    vals <- terra::extract(mask, terra::vect(all_subgrids), fun = "max", na.rm = TRUE)[, 2]
    all_subgrids <- all_subgrids[!is.na(vals) & vals == 1, ]  # Keep only intersecting subgrids
  }
  
  #clean up geometries
  all_subgrids <- st_make_valid(all_subgrids)  # Ensure geometries are valid (esp. for very small slivers)
  
  message("Writing GeoPackage...")
  if (file.exists(output_subgrid_gpkg)) file.remove(output_subgrid_gpkg)  # Clean overwrite
  
  st_write(
    all_subgrids,
    dsn = output_subgrid_gpkg,
    layer = layer_name,
    delete_layer = TRUE,
    quiet = TRUE
  )
  
  message("Subgrids written to:\n", output_subgrid_gpkg)
  return(all_subgrids)
}


## F8 - Re projects sub grid geopackage into desired projection - UTM ----
reproject_subgrids_to_utm <- function(input_gpkg, output_gpkg, utm_wgs84_crs) {
  log_message(paste("🌍 Reprojecting sub-grids to:", utm_wgs84_crs))
  
  tryCatch({
    sub_grids <- st_read(input_gpkg, quiet = TRUE)
    sub_grids_utm <- st_transform(sub_grids, crs = utm_wgs84_crs)
    st_write(sub_grids_utm, output_gpkg, delete_layer = TRUE, quiet = TRUE)
    log_message(paste("✅ Reprojected sub-grids saved to:", output_gpkg))
  }, error = function(e) {
    log_message(paste("❌ Failed to reproject sub-grids:", e$message))
  })
}

## F9 - Standardize Rasters (Parallel)----
standardize_rasters <- function(mask_path, input_dir_raw, input_dir_partial, output_dir, log_csv = NULL) {
  mask <- raster(mask_path)
  crs_mask <- crs(mask)
  
  all_files_raw <- list.files(input_dir_raw, full.names = TRUE)
  non_bathy <- all_files_raw[grepl("^(?!bathy_).*\\.tif$", basename(all_files_raw), perl = TRUE)]
  
  bathy <- list.files(input_dir_partial, pattern = "^bathy_.*\\.tif$", full.names = TRUE)
  all_files <- c(non_bathy, bathy)
  
  log_records <- list()
  
  progressr::with_progress({
    results <- future_lapply(seq_along(all_files), function(i) {
      f <- all_files[i]
      tryCatch({
        r <- raster(f)
        raw_min <- suppressWarnings(minValue(r))
        raw_max <- suppressWarnings(maxValue(r))
        
        if (!compareCRS(r, mask)) {
          r <- projectRaster(r, crs = crs_mask, method = "bilinear")
        }
        
        r <- resample(r, mask, method = "bilinear")
        r <- raster::mask(r, mask)
        
        note <- "OK"
        post_clip_min <- suppressWarnings(minValue(r))
        post_clip_max <- suppressWarnings(maxValue(r))
        na_count <- sum(is.na(values(r)))
        total_count <- ncell(r)
        na_percent <- round((na_count / total_count) * 100, 2)
        
        if (grepl("^bathy_", basename(f))) {
          # Apply bathymetry-specific filter
          r[r > 0] <- NA
          post_clip_min <- suppressWarnings(minValue(r))
          post_clip_max <- suppressWarnings(maxValue(r))
          na_count <- sum(is.na(values(r)))
          na_percent <- round((na_count / total_count) * 100, 2)
          
          if (all(is.na(values(r)))) {
            note <- "Skipped: all values became NA after threshold"
            log_message(paste("Skipped bathy:", basename(f)))
            return(data.frame(
              file = basename(f),
              min_val = raw_min,
              max_val = raw_max,
              na_percent = 100,
              note = note
            ))
          }
        }
        
        out_path <- file.path(output_dir, basename(f))
        writeRaster(r, filename = out_path, overwrite = TRUE, format = "GTiff")
        raster::removeTmpFiles(h = 0)
        
        log_message(paste("Standardized:", basename(f)))
        
        rm(r)
        gc(verbose = FALSE)
        
        return(data.frame(
          file = basename(f),
          min_val = post_clip_min,
          max_val = post_clip_max,
          na_percent = na_percent,
          note = note
        ))
        
      }, error = function(e) {
        log_message(paste("Failed:", basename(f), "-", e$message))
        return(data.frame(
          file = basename(f),
          min_val = NA,
          max_val = NA,
          na_percent = NA,
          note = paste("Error:", e$message)
        ))
      })
    }, future.seed = TRUE)
  })
  
  results_df <- do.call(rbind, results)
  if (!is.null(log_csv)) {
    write.csv(results_df, log_csv, row.names = FALSE)
    log_message(paste("Log written to:", log_csv))
  }
  
  return(invisible(results_df))
}




## F10 - Parallel Tile Chunking (Spatial Dataframes)----


grid_out_raster_data <- function(sub_grid_gpkg, #v1
                                 raster_dir,
                                 output_dir,
                                 data_type = c("training", "prediction"),
                                 parallel = TRUE,
                                 export_footprints = FALSE) {
  library(sf)
  library(raster)
  library(dplyr)
  library(foreach)
  library(tools)
  library(fst)
  library(sp)
  
  data_type <- match.arg(data_type)
  grids <- st_read(sub_grid_gpkg, quiet = TRUE)
  raster_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  
  if (length(raster_files) == 0) stop("❌ No raster .tif files found in the raster directory.")
  
  if (export_footprints) {
    st_write(grids,
             file.path(output_dir, paste0("subgrid_footprints_", data_type, ".gpkg")),
             delete_dsn = TRUE, quiet = TRUE)
  }
  
  # Main tile processing function
  process_tile <- function(i) {
    tile <- grids[i, ]
    tile_name <- tile$tile_id
    tile_geom <- as(tile, "Spatial")
    
    tile_dir <- file.path(output_dir, tile_name)
    dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
    
    log_file <- file.path(tile_dir, paste0("tile_log_", tile_name, ".txt"))
    log_msg <- function(...) {
      ts <- format(Sys.time(), "[%H:%M:%S]")
      cat(ts, ..., "\n", file = log_file, append = TRUE)
    }
    
    log_msg("📦 Processing tile:", tile_name)
    
    out_path <- file.path(tile_dir, paste0(tile_name, "_", data_type, "_clipped_data.fst"))
    
    clipped_data <- lapply(raster_files, function(r_file) {
      log_msg("📄 Checking raster:", basename(r_file))
      r <- tryCatch(raster(r_file), error = function(e) {
        log_msg("⚠️ Could not load raster:", e$message)
        return(NULL)
      })
      if (is.null(r)) return(NULL)
      
      # Match CRS
      if (!compareCRS(r, tile_geom)) {
        tile_proj <- tryCatch(spTransform(tile_geom, crs(r)), error = function(e) {
          log_msg("⚠️ Failed to reproject tile:", e$message)
          return(NULL)
        })
        if (is.null(tile_proj)) return(NULL)
      } else {
        tile_proj <- tile_geom
      }
      
      # Check for overlap
      if (is.null(raster::intersect(extent(r), extent(tile_proj)))) {
        log_msg("⛔ No overlap for raster:", basename(r_file))
        return(NULL)
      }
      
      # Crop and extract
      cropped <- tryCatch(crop(r, tile_proj), error = function(e) {
        log_msg("⚠️ Crop failed:", e$message)
        return(NULL)
      })
      if (is.null(cropped)) return(NULL)
      
      pts <- rasterToPoints(cropped, spatial = FALSE)
      if (nrow(pts) == 0) {
        log_msg("⚠️ Cropped raster returned 0 rows:", basename(r_file))
        return(NULL)
      }
      
      df <- as.data.frame(pts)
      colnames(df) <- c("X", "Y", file_path_sans_ext(basename(r_file)))
      df$FID <- cellFromXY(cropped, df[, c("X", "Y")])
      return(df)
    })
    
    clipped_data <- Filter(Negate(is.null), clipped_data)
    if (length(clipped_data) == 0) {
      log_msg("⚠️ No raster data retained for tile:", tile_name)
      return(NULL)
    }
    
    combined <- Reduce(function(x, y) merge(x, y, by = c("X", "Y", "FID"), all = TRUE), clipped_data)
    
    if (data_type == "training") {
      required_bathy <- c("bathy_2004", "bathy_2006", "bathy_2010", "bathy_2015", "bathy_2022")
      if (all(required_bathy %in% colnames(combined))) {
        combined <- combined %>%
          select(-starts_with("bt")) %>%
          mutate(
            b.change.2004_2006 = bathy_2006 - bathy_2004,
            b.change.2006_2010 = bathy_2010 - bathy_2006,
            b.change.2010_2015 = bathy_2015 - bathy_2010,
            b.change.2015_2022 = bathy_2022 - bathy_2015
          )
        log_msg("✅ Bathymetry change fields added.")
      } else {
        log_msg("⚠️ Missing bathy layers for b.change calculation.")
      }
    } else if (data_type == "prediction") {
      combined <- combined %>% select(-starts_with("bathy"), -starts_with("slope"), -starts_with("rugosity"))
    }
    
    write.fst(combined, out_path)
    log_msg("✅ Data written to:", out_path)
    return(TRUE)
  }
  
  # Parallel execution
  if (parallel) {
    foreach(i = seq_len(nrow(grids)),
            .packages = c("raster", "sf", "sp", "dplyr", "tools", "fst")) %dopar% {
              process_tile(i)
            }
  } else {
    foreach(i = seq_len(nrow(grids)),
            .packages = c("raster", "sf", "sp", "dplyr", "tools", "fst")) %do% {
              process_tile(i)
            }
  }
}

grid_out_raster_data <- function(sub_grid_gpkg,
                                 raster_dir,
                                 output_dir,
                                 data_type = c("training", "prediction"),
                                 parallel = TRUE,
                                 export_footprints = FALSE) {
  library(sf)
  library(raster)
  library(dplyr)
  library(foreach)
  library(tools)
  library(fst)
  library(sp)
  
  data_type <- match.arg(data_type)
  grids <- st_read(sub_grid_gpkg, quiet = TRUE)
  raster_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  
  if (length(raster_files) == 0) stop("❌ No raster .tif files found in the raster directory.")
  
  if (export_footprints) {
    st_write(grids,
             file.path(output_dir, paste0("subgrid_footprints_", data_type, ".gpkg")),
             delete_dsn = TRUE, quiet = TRUE)
  }
  
  # Main tile processing function
  process_tile <- function(i) {
    tile <- grids[i, ]
    tile_name <- tile$tile_id
    tile_geom <- as(tile, "Spatial")
    
    tile_dir <- file.path(output_dir, tile_name)
    dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
    
    log_file <- file.path(tile_dir, paste0("tile_log_", tile_name, ".txt"))
    log_msg <- function(...) {
      ts <- format(Sys.time(), "[%H:%M:%S]")
      cat(ts, ..., "\n", file = log_file, append = TRUE)
    }
    
    log_msg("📦 Processing tile:", tile_name)
    
    out_path <- file.path(tile_dir, paste0(tile_name, "_", data_type, "_clipped_data.fst"))
    
    clipped_data <- lapply(raster_files, function(r_file) {
      log_msg("📄 Checking raster:", basename(r_file))
      r <- tryCatch(raster(r_file), error = function(e) {
        log_msg("⚠️ Could not load raster:", e$message)
        return(NULL)
      })
      if (is.null(r)) return(NULL)
      
      # Match CRS
      tile_proj <- if (!compareCRS(r, tile_geom)) {
        tryCatch(spTransform(tile_geom, crs(r)), error = function(e) {
          log_msg("⚠️ Failed to reproject tile:", e$message)
          return(NULL)
        })
      } else {
        tile_geom
      }
      if (is.null(tile_proj)) return(NULL)
      
      # Check for overlap
      if (is.null(raster::intersect(extent(r), extent(tile_proj)))) {
        log_msg("⛔ No overlap for raster:", basename(r_file))
        return(NULL)
      }
      
      cropped <- tryCatch(crop(r, tile_proj), error = function(e) {
        log_msg("⚠️ Crop failed:", e$message)
        return(NULL)
      })
      if (is.null(cropped)) return(NULL)
      
      pts <- rasterToPoints(cropped, spatial = FALSE)
      if (nrow(pts) == 0) {
        log_msg("⚠️ Cropped raster returned 0 rows:", basename(r_file))
        return(NULL)
      }
      
      df <- as.data.frame(pts)
      base_name <- file_path_sans_ext(basename(r_file))
      base_name <- sub("(bathy_\\d{4})_filled", "\\1", base_name)  # normalize filled name
      colnames(df) <- c("X", "Y", base_name)
      df$FID <- cellFromXY(cropped, df[, c("X", "Y")])
      return(df)
    })
    
    clipped_data <- Filter(Negate(is.null), clipped_data)
    if (length(clipped_data) == 0) {
      log_msg("⚠️ No raster data retained for tile:", tile_name)
      return(NULL)
    }
    
    combined <- Reduce(function(x, y) merge(x, y, by = c("X", "Y", "FID"), all = TRUE), clipped_data)
    
    if (data_type == "training") {
      required_bathy <- c("bathy_2004", "bathy_2006", "bathy_2010", "bathy_2015", "bathy_2022")
      if (all(required_bathy %in% colnames(combined))) {
        combined <- combined %>%
          select(-starts_with("bt")) %>%
          mutate(
            b.change.2004_2006 = bathy_2006 - bathy_2004,
            b.change.2006_2010 = bathy_2010 - bathy_2006,
            b.change.2010_2015 = bathy_2015 - bathy_2010,
            b.change.2015_2022 = bathy_2022 - bathy_2015
          )
        
        log_msg("✅ Bathymetry change fields added.")
        
        # Export b.change rasters
        bchange_cols <- grep("^b\\.change", colnames(combined), value = TRUE)
        for (col in bchange_cols) {
          r <- rasterFromXYZ(combined[, c("X", "Y", col)])
          crs(r) <- crs(tile_geom)
          out_tif <- file.path(raster_dir, paste0(tile_name, "_", col, ".tif"))
          writeRaster(r, out_tif, format = "GTiff", overwrite = TRUE)
          log_msg("🖼️ b.change raster written:", out_tif)
        }
        
      } else {
        log_msg("⚠️ Missing bathy layers for b.change calculation.")
      }
    } else if (data_type == "prediction") {
      combined <- combined %>% select(-starts_with("bathy"), -starts_with("slope"), -starts_with("rugosity"))
    }
    
    write.fst(combined, out_path)
    log_msg("✅ Data written to:", out_path)
    return(TRUE)
  }
  
  # Parallel execution
  if (parallel) {
    foreach(i = seq_len(nrow(grids)),
            .packages = c("raster", "sf", "sp", "dplyr", "tools", "fst")) %dopar% {
              process_tile(i)
            }
  } else {
    foreach(i = seq_len(nrow(grids)),
            .packages = c("raster", "sf", "sp", "dplyr", "tools", "fst")) %do% {
              process_tile(i)
            }
  }
}

# new version 8/2025 makes standard sizes for both prediction and training data by sub grid size

#Modifying grid_out_raster_data to accept a template_raster as an argument.
#Updating the test function to create one single template from your prediction.mask.UTM17_8m.tif and pass it into both
#the training and prediction calls. This ensures perfect consistency.

#' @title Grid Out Raster Data to Standardized Data Frames
#' @description Processes a set of sub-grid tiles by creating a standardized data 
#' frame for each tile based on a master template raster. It extracts values 
#' from multiple source rasters and handles data-type specific calculations.
#'
#' @param sub_grid_gpkg Path to the geopackage containing the sub-grid polygons.
#' @param raster_dir Directory containing the source predictor raster files (.tif).
#' @param output_dir The root directory where tile-specific subfolders will be created.
#' @param data_type A string, either "training" or "prediction".
#' @param parallel Logical, whether to process tiles in parallel.
#' @param export_footprints Logical, whether to save a copy of the sub-grid footprints.
#' @param template_raster A master RasterLayer object to use as the authoritative grid.
#' @param bchange_raster_dir Path to a directory where new b.change rasters will be saved.
#'   This should be different from `raster_dir`. If NULL, rasters are not saved.
#'
#' @return Invisibly returns TRUE on completion.
grid_out_raster_data <- function(sub_grid_gpkg,
                                 raster_dir,
                                 output_dir,
                                 data_type = c("training", "prediction"),
                                 parallel = TRUE,
                                 export_footprints = FALSE,
                                 template_raster = NULL,
                                 bchange_raster_dir = NULL) { # <-- NEW ARGUMENT
  # Required libraries
  library(sf)
  library(raster)
  library(dplyr)
  library(foreach)
  library(tools)
  library(fst)
  library(sp)
  
  data_type <- match.arg(data_type)
  grids <- st_read(sub_grid_gpkg, quiet = TRUE)
  
  # --- CRITICAL FIX: Filter raster_files to only include core predictors ---
  # This prevents reading previously generated b.change rasters.
  all_files <- list.files(raster_dir, pattern = "\\.tif$", full.names = TRUE)
  raster_files <- all_files[!grepl("_b\\.change\\.", basename(all_files))]
  
  if (length(raster_files) == 0) stop("❌ No core predictor raster .tif files found.")
  
  if (export_footprints) {
    st_write(grids, file.path(output_dir, paste0("subgrid_footprints_", data_type, ".gpkg")), delete_dsn = TRUE, quiet = TRUE)
  }
  
  # Create the b.change output directory if it's specified and doesn't exist
  if (!is.null(bchange_raster_dir)) {
    dir.create(bchange_raster_dir, showWarnings = FALSE, recursive = TRUE)
  }
  
  process_tile <- function(i) {
    tile <- grids[i, ]
    tile_name <- tile$tile_id
    tile_geom <- as(tile, "Spatial")
    tile_dir <- file.path(output_dir, tile_name)
    dir.create(tile_dir, recursive = TRUE, showWarnings = FALSE)
    log_file <- file.path(tile_dir, paste0("tile_log_", tile_name, ".txt"))
    log_msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S]"), ..., "\n", file = log_file, append = TRUE)
    
    log_msg("📦 Processing tile:", tile_name)
    out_path <- file.path(tile_dir, paste0(tile_name, "_", data_type, "_clipped_data.fst"))
    
    if (is.null(template_raster)) stop("❌ A 'template_raster' must be provided.")
    
    log_msg("🛠️ Creating master grid from provided template...")
    template_for_tile <- crop(template_raster, tile_geom)
    template_for_tile <- mask(template_for_tile, tile_geom)
    
    combined <- as.data.frame(template_for_tile, xy = TRUE, na.rm = FALSE)[, c("x", "y")]
    colnames(combined) <- c("X", "Y")
    
    if (nrow(combined) == 0) {
      log_msg("⛔ Tile has no overlapping cells with the template. Skipping.")
      return(NULL)
    }
    log_msg(paste("✅ Master grid created with", nrow(combined), "rows."))
    
    for (r_file in raster_files) {
      r <- raster(r_file)
      base_name <- sub("(bathy_\\d{4})_filled", "\\1", file_path_sans_ext(basename(r_file)))
      combined[[base_name]] <- raster::extract(r, combined[, c("X", "Y")])
    }
    
    combined$FID <- cellFromXY(template_for_tile, combined[, c("X", "Y")])
    
    if (data_type == "training") {
      required_bathy <- c("bathy_2004", "bathy_2006", "bathy_2010", "bathy_2015", "bathy_2022")
      if (all(required_bathy %in% colnames(combined))) {
        combined <- combined %>%
          select(-starts_with("bt")) %>%
          mutate(
            b.change.2004_2006 = bathy_2006 - bathy_2004,
            b.change.2006_2010 = bathy_2010 - bathy_2006,
            b.change.2010_2015 = bathy_2015 - bathy_2010,
            b.change.2015_2022 = bathy_2022 - bathy_2015
          )
        log_msg("✅ Bathymetry change fields added.")
        
        # --- MODIFIED: Save b.change rasters to the new, separate directory ---
        if (!is.null(bchange_raster_dir)) {
          bchange_cols <- grep("^b\\.change", colnames(combined), value = TRUE)
          for (col in bchange_cols) {
            r <- rasterFromXYZ(combined[, c("X", "Y", col)])
            crs(r) <- crs(tile_geom)
            out_tif <- file.path(bchange_raster_dir, paste0(tile_name, "_", col, ".tif"))
            writeRaster(r, out_tif, format = "GTiff", overwrite = TRUE)
            log_msg("🖼️ b.change raster written to:", out_tif)
          }
        }
      } else {
        log_msg("⚠️ Missing bathy layers for b.change calculation.")
      }
    } else if (data_type == "prediction") {
      combined <- combined %>% select(-starts_with("bathy"), -starts_with("slope"), -starts_with("rugosity"))
    }
    
    write.fst(combined, out_path)
    log_msg("✅ Data written to:", out_path)
    return(TRUE)
  }
  
  if (parallel) {
    foreach(i = seq_len(nrow(grids)), .packages = c("raster", "sf", "sp", "dplyr", "tools", "fst")) %dopar% { process_tile(i) }
  } else {
    foreach(i = seq_len(nrow(grids)), .packages = c("raster", "sf", "sp", "dplyr", "tools", "fst")) %do% { process_tile(i) }
  }
}








# ──────────────-----------------------------------
# 4. RUN MODULE FUNCTIONS
# ──────────────-----------------------------------
#initiate parallel processing 
registerDoParallel(cores = parallel::detectCores() - 1)
cl <- makeCluster(cores)
cores <- 8  # or even 1 to start safely
handlers(global = TRUE)
plan(multisession, workers = cores)
registerDoParallel(cl)

start_time <- Sys.time()
log_message(" Starting preprocessing module...")

log_file <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/preprocessing_log.txt"

# Logging helper
log_message <- function(msg) {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  line <- sprintf("[%s] %s\n", timestamp, msg)
  cat(line)
  write(line, file = log_file, append = TRUE)
}





# F1 - FOCAL GAP FILL (uses ~6-8GB of RAM and takes 2.5hrs per bathy tiff [pilot model extent])----
bathy_files <- list.files(input_raw_pred, pattern = "^bathy_\\d{4}\\.tif$", full.names = TRUE)
run_gap_fill(bathy_files, output_dir = output_filled, cores = 8, max_iters = 5)
log_message(" Final cleanup of temp raster files...")
cleanup_intermediate_rasters <- function(base_name, dir) {
  files <- list.files(dir, pattern = paste0("^", base_name, "_f\\d+\\.tif$"), full.names = TRUE)
  file.remove(files)
}
gc()

# F2 - TRAINING MASK-----
create_training_mask(
  input_dir = input_partial,
  output_mask_utm = output_mask_train_utm,
  output_mask_wgs = output_mask_train_wgs,
  shapefile_path = shapefile_path
)

# F3 - PREDICTION MASK----
create_prediction_mask(
  output_mask_utm = output_mask_pred_utm,
  output_mask_wgs = output_mask_pred_wgs,
  shapefile_path = shapefile_path
)
# F4 - SPATIAL DATAFRAMES
# create_spatial_mask_df(mask_utm_path = output_mask_pred_utm, mask_wgs_path = output_mask_train_wgs, mask_type = "training", output_dir = output_SPDF)
# 
# create_spatial_mask_df(mask_utm_path = output_mask_pred_utm,mask_wgs_path = output_mask_pred_wgs,mask_type = "prediction", output_dir = output_SPDF)

# NEW F4 - BT.UNCERTAINTY----
extract_band2_to_uncertainty(input_dir, uncertainty_dir)
mosaic_ras <- mosaic_uncertainty_rasters(uncertainty_dir, mosaic_path)

# F5 - EXTRACT SURVEY END DATES----
extract_survey_end_dates(input_dir = input_dir_survey, kml_dir = kml_dir_survey, output_dir = output_dir_survey_dates)
# F6 & F7 - PREPARE SUB-GRIDS (Training & Prediction Masks)
# OG Blue Topo Gpkg

# Prediction Extent 
generate_subgrids_from_mask_wgs84(
  grid_tiles_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg",
  output_subgrid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_WGS84.gpkg",
  reference_tile_id = "BH4S257K", # any of the smaller inner blue topo tiles
  mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.WGS84_8m.tif",
  workers = 4  # set number of cores
)

# Training Extent 
generate_subgrids_from_mask_wgs84(
  grid_tiles_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg",
  output_subgrid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_WGS84.gpkg",
  reference_tile_id = "BH4S257K", # any of the smaller inner blue topo tiles
  mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.WGS84_8m.tif",
  workers = 4  # set number of cores
)
# F8 - REPROJECT SUB GRIDS to UTM----
reproject_subgrids_to_utm( # training extent 
  input_gpkg = file.path(training_subgrid_out, "intersecting_sub_grids_WGS84.gpkg"),
  output_gpkg = file.path(training_subgrid_out, "intersecting_sub_grids_UTM.gpkg"),
  utm_wgs84_crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"
)

reproject_subgrids_to_utm( # Prediction extent 
  input_gpkg = file.path(prediction_subgrid_out, "intersecting_sub_grids_WGS84.gpkg"),
  output_gpkg = file.path(prediction_subgrid_out, "intersecting_sub_grids_UTM.gpkg"),
  utm_wgs84_crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"
)
 
# F9 -  STANDARDIZE RASTERS----
standardize_rasters( # Prediction
  mask_path = output_mask_pred_utm,
  input_dir_raw = input_raw_pred,   
  input_dir_partial = input_partial,
  output_dir = output_proc_pred,
  log_csv = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/bathy_standardization_pred_log.csv"
)

standardize_rasters( # Training
  mask_path = output_mask_train_utm,
  input_dir_raw = input_raw_train,
  input_dir_partial = input_partial,
  output_dir = output_proc_train,
  log_csv = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/bathy_standardization_train_log.csv"
)

# F10 - RASTER CHUNK TILE DATA----
# Define a new, separate output directory for the b.change rasters
bchange_output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/bchange_rasters"

# Load the authoritative template once
authoritative_template <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.UTM17_8m.tif")

# Call for PREDICTION data (no b.change output needed)
grid_out_raster_data(
  sub_grid_gpkg = prediction_grid_gpkg,
  raster_dir = output_proc_pred,
  output_dir = prediction_subgrid_out,
  data_type = "prediction",
  parallel = TRUE,
  template_raster = authoritative_template
)

# Call for TRAINING data
grid_out_raster_data(
  sub_grid_gpkg = training_grid_gpkg,
  raster_dir = output_proc_train,
  output_dir = training_subgrid_out,
  data_type = "training",
  parallel = TRUE,
  template_raster = authoritative_template,
  bchange_raster_dir = bchange_output_dir # <-- Add this argument
)

# Grid out r

#' @title Run a Definitive Unit Test on a Single Sub-Grid Tile
#' @description This function cleans up previous outputs, isolates a single tile, 
#' processes it for training and prediction using an authoritative master template, 
#' and verifies that the output data frames have identical row counts and that the
#' training data has the correct number of columns.
#'
#' @param tile_id The character string ID of the tile to test (e.g., "BH4S2574_3").
#' @param base_dir The base directory path for your project.
#'
#' @return Prints the results of the test to the console.
run_single_tile_test <- function(tile_id = "BH4S2574_1", base_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model") {
  
  message("--- Starting Definitive Single Tile Test for Tile: ", tile_id, " ---")
  
  # --- 1. Define all necessary file paths ---
  prediction_grid_gpkg <- file.path(base_dir, "Coding_Outputs/Prediction_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
  training_grid_gpkg   <- file.path(base_dir, "Coding_Outputs/Training_data_grid_tiles/intersecting_sub_grids_UTM.gpkg")
  output_proc_pred <- file.path(base_dir, "Model_variables/Prediction/processed")
  output_proc_train <- file.path(base_dir, "Model_variables/Training/processed")
  prediction_subgrid_out <- file.path(base_dir, "Coding_Outputs/Prediction_data_grid_tiles")
  training_subgrid_out   <- file.path(base_dir, "Coding_Outputs/Training_data_grid_tiles")
  prediction_mask_path <- file.path(base_dir, "prediction.mask.UTM17_8m.tif")
  
  # --- NEW: Define a separate directory for b.change raster outputs for the test ---
  bchange_output_dir <- file.path(base_dir, "Coding_Outputs/bchange_rasters_test")
  
  # Temp directory for isolated geopackage
  temp_dir <- file.path(tempdir(), "tile_test")
  dir.create(temp_dir, showWarnings = FALSE, recursive = TRUE)
  
  # --- 2. Pre-run Cleanup ---
  message("\n[Step 1/6] Cleaning up previous test outputs...")
  pred_tile_output_dir <- file.path(prediction_subgrid_out, tile_id)
  train_tile_output_dir <- file.path(training_subgrid_out, tile_id)
  if (dir.exists(pred_tile_output_dir)) unlink(pred_tile_output_dir, recursive = TRUE)
  if (dir.exists(train_tile_output_dir)) unlink(train_tile_output_dir, recursive = TRUE)
  if (dir.exists(bchange_output_dir)) unlink(bchange_output_dir, recursive = TRUE)
  message("✅ Previous outputs cleaned.")
  
  # --- 3. Load Authoritative Template ---
  message("\n[Step 2/6] Loading authoritative template raster...")
  if (!file.exists(prediction_mask_path)) stop("❌ Authoritative mask not found at:", prediction_mask_path)
  authoritative_template <- raster(prediction_mask_path)
  message("✅ Template raster loaded.")
  
  # --- 4. Isolate the single tile ---
  message("\n[Step 3/6] Isolating single tile from source geopackage...")
  master_tile_sf <- st_read(prediction_grid_gpkg, quiet = TRUE) %>% 
    filter(tile_id == !!tile_id)
  if (nrow(master_tile_sf) == 0) stop("❌ Tile ID '", tile_id, "' not found.")
  
  temp_gpkg_path <- file.path(temp_dir, "single_tile.gpkg")
  st_write(master_tile_sf, temp_gpkg_path, delete_dsn = TRUE, quiet = TRUE)
  message("✅ Master tile geopackage created for the test.")
  
  # --- 5. Run grid_out_raster_data for each data type ---
  message("\n[Step 4/6] Processing PREDICTION data for the tile...")
  grid_out_raster_data(
    sub_grid_gpkg = temp_gpkg_path,
    raster_dir = output_proc_pred,
    output_dir = prediction_subgrid_out,
    data_type = "prediction",
    parallel = FALSE,
    template_raster = authoritative_template
  )
  message("✅ Prediction data processed.")
  
  message("\n[Step 5/6] Processing TRAINING data for the tile...")
  grid_out_raster_data(
    sub_grid_gpkg = temp_gpkg_path,
    raster_dir = output_proc_train,
    output_dir = training_subgrid_out,
    data_type = "training",
    parallel = FALSE,
    template_raster = authoritative_template,
    bchange_raster_dir = bchange_output_dir # <-- PASSING NEW ARGUMENT
  )
  message("✅ Training data processed.")
  
  # --- 6. Verify the results ---
  message("\n[Step 6/6] Verifying output files...")
  
  pred_fst_path <- file.path(prediction_subgrid_out, tile_id, paste0(tile_id, "_prediction_clipped_data.fst"))
  train_fst_path <- file.path(training_subgrid_out, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
  
  if (!file.exists(pred_fst_path) || !file.exists(train_fst_path)) {
    stop("❌ ERROR: One or both output .fst files were not created. Check logs.")
  }
  
  # Check row counts
  pred_rows <- fst::metadata_fst(pred_fst_path)$nrOfRows
  train_rows <- fst::metadata_fst(train_fst_path)$nrOfRows
  
  # Check column counts
  pred_cols <- fst::metadata_fst(pred_fst_path)$nrOfCols
  train_cols <- fst::metadata_fst(train_fst_path)$nrOfCols
  
  # Define expected column count for training data (adjust if your base predictors change)
  # Original predictors + X + Y + FID + 4 b.change columns
  # Let's count the non-b.change rasters in the training dir
  base_training_rasters <- list.files(output_proc_train, pattern = "\\.tif$")
  base_training_rasters <- base_training_rasters[!grepl("_b\\.change\\.", base_training_rasters)]
  expected_train_cols <- length(base_training_rasters) + 3 + 4 # predictors + X/Y/FID + b.change
  
  message("\n--- TEST RESULTS ---")
  message("Prediction data dimensions: ", pred_rows, " rows x ", pred_cols, " cols")
  message("Training data dimensions:   ", train_rows, " rows x ", train_cols, " cols")
  message(" (Expected training columns: ~", expected_train_cols, ")")
  
  row_check <- pred_rows == train_rows
  col_check <- train_cols < 100 # A sanity check to ensure it's not hundreds of columns
  
  if (row_check && col_check) {
    message("\n✅ SUCCESS: Row counts are identical and column count is reasonable.")
  } else {
    if (!row_check) message("\n❌ FAILURE: The number of rows is different.")
    if (!col_check) message("\n❌ FAILURE: The training data column count is unexpectedly high.")
  }
  
  unlink(temp_dir, recursive = TRUE)
  unlink(bchange_output_dir, recursive = TRUE)
  message("\n--- Test Complete ---")
}
run_single_tile_test()

setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles/BH4S2574_3")
p.data <- read.fst("BH4S2574_3_prediction_clipped_data.fst")
glimpse(p.data)

setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3")
t.data <- read.fst("BH4S2574_3_training_clipped_data.fst")
glimpse(t.data)

#------
mask <- raster(output_mask_pred_utm)
t.data.rast <- rasterFromXYZ(data.frame(x = t.data[,"X"],  # LAT
                                       y = t.data[,"Y"],  # LON
                                       z = t.data[, "bathy_2004"]), # elevation
                            crs = crs(mask))

p.data.rast <- rasterFromXYZ(data.frame(x = p.data[,"X"],  # LAT
                                        y = p.data[,"Y"],  # LON
                                        z = p.data[, "bt.bathy"]), # elevation
                             crs = crs(mask))


plot (t.data.rast)
plot (p.data.rast)
# ──────────────-----------------------
# 5. FINISH / Close parallel 
# ──────────────----------------------
stopCluster(cl)
end_time <- Sys.time()
log_message(sprintf(" All preprocessing completed in %.1f minutes", as.numeric(difftime(end_time, start_time, units = "mins"))))


# ------------STAGE 2 - MODELLING STEPS--------------------# ----

## APRIL /MAY 25 notes and revisions 

#1 including predictor selection / reduction using leaps package (add in correlation checking, IVF, spatial autocorelation) 
#2 tracking model efficiency metrics / perofrmance with each iteration change / tracker 
#3 inclusion of UC metrics (from BT layers and also from the model)
# 4 Better model validation - in SDM modelling, you would training your models on 75-80% of your training data, withold 20% and see how the predictions 
# compare back to your witheld data as a form of validation - several metrics use this form of data splitting for validation
# 5 - we still need to consider what we want to do for post processing, and how we make it realtable to HHM 1.0. Glen sugested that we could perhaps use Kalman Filters

# THINGS


# 6... Model training over all sub grids ----
# This script loops through grid tiles intersecting the training data boundary, 
#runs models for each year pair, and stores results as .fst files.



# LOAD PARAMETERS CREATED FROM  PREPROCESSING STAGE if not already loaded:----
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
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"


# ==============================================================================
#
#           Boruta Predictor Selection Function Set
#
# ==============================================================================

# This function takes training data and uses the Boruta algorithm to identify
#' important predictors for a given response variable. It saves the results,
#' including a list of confirmed predictors and performance metrics, to be used
#' in downstream modeling.. It is designed to:
#   1. Iterate through all specified tiles and year-pairs in parallel.
#   2. Correctly identify all static and dynamic predictors, including rugosity.
#   3. Run the Boruta algorithm to determine statistically significant predictors.
#   4. Save the results (confirmed predictors and full statistics) for each run.

#' @param training_sub_grids_UTM A data frame or sf object with tile IDs.
#' @param output_dir_train The base directory where training data is stored and results will be saved.
#' @param year_pairs A list of year pairs to process (e.g., "2000_2005").
#' @param max_runs The maximum number of iterations for the Boruta algorithm.
#'
#' @return A list containing the paths to the selection results for each tile and year pair.

# --- Load All Necessary Libraries ---
library(Boruta)
library(dplyr)
library(data.table)
library(fst)
library(tidyr)
library(foreach)
library(doParallel)
library(tibble)

# ==============================================================================
#   MAIN SELECTION FUNCTION
# ==============================================================================

Select_Predictors_Boruta <- function(training_sub_grids_UTM, output_dir_train, year_pairs, max_runs = 100) {
  # -------------------------------------------------------
  # 1. INITIALIZATION & PARALLEL SETUP----
  # -------------------------------------------------------
  cat("\n🚀 Starting Predictor Selection with Boruta...\n")
  
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1 # maximum number of cores detected and then leaves one open for GUI/IDE
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  on.exit({
    if (exists("cl") && inherits(cl, "cluster")) {
      stopCluster(cl)
    }
    closeAllConnections()
  }, add = TRUE)
  
  log_file <- file.path(output_dir_train, "predictor_selection_log.txt")
  cat("Log - Boruta Predictor Selection\n", as.character(Sys.time()), "\n", file = log_file, append = FALSE)
  
  # -------------------------------------------------------
  # 2. ITERATE THROUGH TILES AND YEAR PAIRS----
  # -------------------------------------------------------
  results_paths <- foreach(i = seq_along(tiles_df), .combine = 'c', .packages = c("Boruta", "dplyr", "data.table", "fst", "tidyr", "tibble")) %dopar% {
    tile_id <- tiles_df[i]
    tile_results <- list()
    
    for (pair in year_pairs) {
      tryCatch({
        # --- Data Loading ---
        training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
        if (!file.exists(training_data_path)) {
          cat(Sys.time(), "⚠️ Missing training data for tile:", tile_id, "\n", file = log_file, append = TRUE)
          next
        }
        training_data <- read_fst(training_data_path, as.data.table = TRUE)
        training_data <- as.data.frame(training_data)
        
        # --- Define Predictors and Response ---
        start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
        end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
        response_var <- trimws(paste0("b.change.", pair)) # Use trimws() as a safeguard
        
        if (!response_var %in% names(training_data)) {
          cat(Sys.time(), "🚨 ERROR: Response variable '", response_var, "' missing for Tile:", tile_id, "\n", file = log_file, append = TRUE)
          next
        }
        
        static_predictors <- c("grain_size_layer", "prim_sed_layer")
        
        
        dynamic_predictors <- c(
          paste0("bathy_", start_year), paste0("bathy_", end_year),
          paste0("slope_", start_year), paste0("slope_", end_year),
          paste0("rugosity_", start_year), paste0("rugosity_", end_year),
          # grep(paste0("^Rugosity_", start_year, "_nbh"), names(training_data), value = TRUE),
          # grep(paste0("^Rugosity_", end_year, "_nbh"), names(training_data), value = TRUE),
          paste0("hurr_count_", pair), paste0("hurr_strength_", pair),
          paste0("tsm_", pair)
        )
        all_predictors <- intersect(c(static_predictors, dynamic_predictors), names(training_data))
        
        sub_data <- training_data %>%
          select(all_of(c(all_predictors, response_var))) %>%
          drop_na()
        
        if (nrow(sub_data) < 50 || length(unique(sub_data[[response_var]])) <= 1) {
          cat(Sys.time(), "⚠️ Skipping Boruta for Tile:", tile_id, "| Pair:", pair, "- Insufficient data.\n", file = log_file, append = TRUE)
          next
        }
        
        # --- Run Boruta ---
        cat(Sys.time(), "🏃 Running Boruta for Tile:", tile_id, "| Pair:", pair, "\n")
        boruta_result <- Boruta(
          x = sub_data[, all_predictors, drop = FALSE],
          y = sub_data[[response_var]],
          maxRuns = max_runs,
          doTrace = 0
        )
        
        # --- Store Results ---
        confirmed_preds <- getSelectedAttributes(boruta_result, withTentative = FALSE)
        boruta_stats <- attStats(boruta_result) %>%
          as.data.frame() %>%
          tibble::rownames_to_column(var = "predictor") %>%
          arrange(desc(meanImp))
        
        output_list <- list(
          confirmed_predictors = confirmed_preds,
          boruta_statistics = boruta_stats
        )
        
        tile_dir <- file.path(output_dir_train, tile_id)
        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE)
        
        save_path <- file.path(tile_dir, paste0("boruta_selection_", pair, ".rds"))
        saveRDS(output_list, file = save_path)
        tile_results[[pair]] <- save_path
        
      }, error = function(e) {
        cat(Sys.time(), "❌ ERROR in Boruta selection for Tile:", tile_id, "| Pair:", pair, "|", conditionMessage(e), "\n", file = log_file, append = TRUE)
      })
    }
    unlist(tile_results)
  }
  
  cat("\n✅ Boruta Predictor Selection Complete! Check `predictor_selection_log.txt` for details.\n")
  return(results_paths)
}


# Function Call 
Sys.time() 
Select_Predictors_Boruta(training_sub_grids_UTM,
                         output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles",
                         year_pairs = c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
)
Sys.time() # Takes approx 30 hours to run






# BORUTA PREDICTOR SELECTION SUMMARY REPORT (before running model examine) ----
#' Create Visual Reports for Boruta Predictor Selection
#'
#' This function scans a directory for Boruta selection results (`.rds` files),
#' aggregates them, and generates two separate summary plots using R's base
#' plotting functions:
#' 1. An overall summary of predictor confirmation, rejection, and importance.
#' 2. A breakdown of the most important predictors for each year-pair.
#'
#' @param output_dir_train The base directory where the tile folders are located.
#' @param overall_report_filename The name of the output PNG for the overall summary.
#' @param by_year_report_filename The name of the output PNG for the year-pair breakdown.
#' @param top_n The number of top predictors to display in the plots.
#'
#' @return None. Two PNG files are saved to the `output_dir_train`.

#Function
create_boruta_summary_report <- function(output_dir_train,
                                         overall_report_filename = "boruta_summary_report.png",
                                         by_year_report_filename = "boruta_importance_by_year.png",
                                         top_n = 10) {
  
  # -------------------------------------------------------
  # 1. LOAD LIBRARIES and FIND FILES
  # -------------------------------------------------------
  cat("Starting Boruta summary report generation...\n")
  library(dplyr)
  library(stringr)
  
  # Find all Boruta selection .rds files recursively
  selection_files <- list.files(
    path = output_dir_train,
    pattern = "^boruta_selection_.*\\.rds$",
    recursive = TRUE,
    full.names = TRUE
  )
  
  if (length(selection_files) == 0) {
    stop("No 'boruta_selection_*.rds' files found. Please run the selection script first.")
  }
  cat("Found", length(selection_files), "Boruta result files.\n")
  
  # -------------------------------------------------------
  # 2. LOAD and PROCESS ALL RESULTS
  # -------------------------------------------------------
  all_results <- lapply(selection_files, function(fp) {
    result_list <- readRDS(fp)
    stats_df <- result_list$boruta_statistics
    path_parts <- strsplit(fp, "[/\\\\]")[[1]]
    stats_df$tile_id <- path_parts[length(path_parts) - 1]
    stats_df$year_pair <- str_extract(basename(fp), "\\d{4}_\\d{4}")
    return(stats_df)
  })
  
  combined_df <- bind_rows(all_results)
  cat("Successfully processed all result files.\n")
  
  
  # -------------------------------------------------------
  # 3. PREPARE DATA FOR PLOTTING (OVERALL SUMMARY)
  # -------------------------------------------------------
  confirmed_counts <- combined_df %>%
    filter(decision == "Confirmed") %>%
    count(predictor, sort = TRUE) %>%
    head(top_n)
  
  rejected_counts <- combined_df %>%
    filter(decision == "Rejected") %>%
    count(predictor, sort = TRUE) %>%
    head(top_n)
  
  decision_summary <- combined_df %>%
    count(decision) %>%
    mutate(decision = factor(decision, levels = c("Confirmed", "Tentative", "Rejected"))) %>%
    arrange(decision)
  
  importance_summary <- combined_df %>%
    group_by(predictor) %>%
    summarise(avg_importance = mean(meanImp, na.rm = TRUE)) %>%
    arrange(desc(avg_importance)) %>%
    head(top_n)
  
  
  # -------------------------------------------------------
  # 4. GENERATE OVERALL 4-PANEL PLOT
  # -------------------------------------------------------
  cat("Generating overall summary plot...\n")
  png(
    file.path(output_dir_train, overall_report_filename),
    width = 1200, height = 1000, res = 100
  )
  par(mfrow = c(2, 2), mar = c(5, 8, 4, 2))
  
  # --- PLOT 1: Most Confirmed ---
  barplot(height = rev(confirmed_counts$n), names.arg = rev(confirmed_counts$predictor), horiz = TRUE, las = 1, main = paste("Top", top_n, "Most Confirmed Predictors"), xlab = "Times Confirmed", col = "darkgreen")
  # --- PLOT 2: Most Rejected ---
  barplot(height = rev(rejected_counts$n), names.arg = rev(rejected_counts$predictor), horiz = TRUE, las = 1, main = paste("Top", top_n, "Most Rejected Predictors"), xlab = "Times Rejected", col = "firebrick")
  # --- PLOT 3: Overall Decisions ---
  barplot(height = decision_summary$n, names.arg = decision_summary$decision, main = "Overall Decision Frequency", ylab = "Total Count", col = c("darkgreen", "darkorange", "firebrick"))
  # --- PLOT 4: Top Importance Score ---
  barplot(height = rev(importance_summary$avg_importance), names.arg = rev(importance_summary$predictor), horiz = TRUE, las = 1, main = paste("Top", top_n, "Predictors by Avg. Importance"), xlab = "Mean Importance (Gain)", col = "steelblue")
  
  dev.off()
  cat("Report saved to:", file.path(output_dir_train, overall_report_filename), "\n")
  
  
  # -------------------------------------------------------
  # 5. PREPARE DATA FOR YEAR-PAIR SPECIFIC PLOT
  # -------------------------------------------------------
  cat("Preparing data for year-pair specific importance plot...\n")
  year_pairs <- sort(unique(combined_df$year_pair))
  
  importance_by_year <- lapply(year_pairs, function(pair) {
    combined_df %>%
      filter(year_pair == pair) %>%
      group_by(predictor) %>%
      summarise(avg_importance = mean(meanImp, na.rm = TRUE), .groups = "drop") %>%
      arrange(desc(avg_importance)) %>%
      head(top_n)
  })
  names(importance_by_year) <- year_pairs
  
  
  # -------------------------------------------------------
  # 6. GENERATE YEAR-PAIR SPECIFIC IMPORTANCE PLOT
  # -------------------------------------------------------
  cat("Generating year-pair specific importance plot...\n")
  png(
    file.path(output_dir_train, by_year_report_filename),
    width = 1400, height = 1200, res = 100
  )
  par(mfrow = c(2, 2), mar = c(5, 9, 4, 2), oma = c(0, 0, 2, 0)) # Set outer margins for a main title
  
  for (pair in year_pairs) {
    plot_data <- importance_by_year[[pair]]
    if (nrow(plot_data) > 0) {
      barplot(
        height = rev(plot_data$avg_importance),
        names.arg = rev(plot_data$predictor),
        horiz = TRUE, las = 1,
        main = paste("Year Pair:", pair),
        xlab = "Mean Importance (Gain)",
        col = "darkcyan"
      )
    }
  }
  mtext("Top Predictor Importance by Year Pair", outer = TRUE, cex = 1.5, font = 2)
  dev.off()
  
  # Reset plotting layout
  par(mfrow = c(1, 1), oma = c(0, 0, 0, 0))
  cat("Year-pair report saved to:", file.path(output_dir_train, by_year_report_filename), "\n")
  cat("Process complete.\n")
}

# Function Call 
create_boruta_summary_report(
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
)


# FINAL 
#' -----------------------------------------------------------------------------
#' Helper Function: Generate and Save a Diagnostic Plot for Spatial CV
#' -----------------------------------------------------------------------------
#' Creates a plot to visualize the training data and the spatial blocks.
#'
#' @param sf_data The sf object containing the training data points.
#' @param block_geom The sf object of the grid blocks created for probing.
#' @param max_k The maximum number of folds possible for the geometry.
#' @param tile_id The character ID of the current tile.
#' @param pair The character ID of the current year pair.
#' @param output_dir_train The main output directory.
#' @return Invisibly returns the ggplot object.
#'

  generate_cv_diagnostic_plot <- function(sf_data, block_geom, max_k, tile_id, pair, output_dir_train) {
  diag_plot <- ggplot() +
    # Plot the raw data points
    geom_sf(data = sf_data, color = "grey50", size = 0.1, alpha = 0.5) +
    # Overlay the spatial block boundaries
    geom_sf(data = block_geom, color = "blue", fill = "transparent", linewidth = 0.5) +
    labs(
      title = paste("Diagnostic Plot for Tile:", tile_id, "| Pair:", pair),
      subtitle = paste("Block Size:", st_bbox(block_geom)[3] - st_bbox(block_geom)[1], "m | Max Possible Folds (k):", max_k),
      x = "X (UTM)", y = "Y (UTM)"
    ) +
    theme_minimal()
  
  # Define the output path and save the plot
  plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
  if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
  
  ggsave(
    file.path(plot_dir, paste0("spatial_cv_map_", pair, ".png")),
    plot = diag_plot, width = 8, height = 7, dpi = 150
  )
}


## SEMI -FINAL!
# 7.8 full functionality added back in for testing, PDP works, full CV  
Model_Train_Full_SpatialCV <- function(training_sub_grids_UTM, output_dir_train, year_pairs,
                                       block_size_m, n.boot = 20, n.folds = 5) {
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\nStarting Full XGBoost Model Training with Robust Parallel Spatial CV...\n")
  
  # Load libraries for the main session
  library(foreach)
  library(doParallel)
  
  # --- Setup Parallel Processing ---
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  if (num_cores < 1) num_cores <- 1 # Ensure at least one core is used
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
  # Outer Loop - Using %dopar% to run the outer loop (over tiles) in parallel
  results_list <- foreach(
    i = seq_len(length(tiles_df)),
    .combine = 'c',
    .packages = c("xgboost", "dplyr", "data.table", "fst", "tidyr", "sf", "blockCV", "ggplot2", "foreach"),
    .export = "generate_cv_diagnostic_plot", # Make the helper function available to workers
    .errorhandling = "pass"
  ) %dopar% {
    
    # Inner Loop - over year pairs runs sequentially 
    tile_id <- tiles_df[[i]]
    worker_log_file <- file.path(output_dir_train, tile_id, paste0("log_worker_", tile_id, ".txt"))
    cat("Worker log for tile:", tile_id, "started at", as.character(Sys.time()), "\n", file = worker_log_file, append = FALSE)
    
    foreach(pair = year_pairs, .combine = 'c') %do% {
      
      tryCatch({
        # --- a. Load Data---
        cat("\nProcessing Tile:", tile_id, "| Year Pair:", pair, "\n", file = worker_log_file, append = TRUE)
        
        training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
        boruta_results_path <- file.path(output_dir_train, tile_id, paste0("boruta_selection_", pair, ".rds"))
        if (!file.exists(training_data_path) || !file.exists(boruta_results_path)) { return(NULL) }
        
        training_data <- as.data.frame(read_fst(training_data_path, as.data.table = TRUE))
        boruta_results <- readRDS(boruta_results_path)
        predictors <- boruta_results$confirmed_predictors
        response_var <- paste0("b.change.", pair)
        if (length(predictors) == 0 || !response_var %in% names(training_data)) { return(NULL) }
        
        # -------------------------------------------------------
        # 3. FILTER & PREPARE TRAINING DATA----
        # -------------------------------------------------------
        # This uses XGBoost's native NA handling by only filtering NAs in the response variable.
        cols_to_select <- c(predictors, response_var, "X", "Y", "FID")
        subgrid_data <- training_data %>%
          dplyr::select(any_of(cols_to_select)) %>%
          mutate(across(all_of(c(predictors, response_var)), ~as.numeric(as.character(.)))) %>%
          filter(if_all(all_of(c(predictors, response_var)), is.finite))
        
        if (nrow(subgrid_data) < 100) { return(NULL) }
        
        subgrid_sf <- st_as_sf(subgrid_data, coords = c("X", "Y"), remove = FALSE, crs = 32617)
        
        # -------------------------------------------------------
        # 4. INITIALIZE METRIC OUTPUT STORAGE ARRAYS----
        # -------------------------------------------------------
        
        deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot); colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
        influence_mat <- array(NA, dim = c(length(predictors), n.boot)); rownames(influence_mat) <- predictors; colnames(influence_mat) <- paste0("Rep_", 1:n.boot)
        boot_mat <- matrix(NA, nrow = nrow(subgrid_data), ncol = n.boot)
        cv_results_list <- list()
        
        # -------------------------------------------------------
        # 5. SETUP for PARTIAL DEPENDENCE PLOT (PDP) DATA----
        # -------------------------------------------------------
        PredMins <- apply(subgrid_data[, predictors, drop = FALSE], 2, min, na.rm = TRUE)
        PredMaxs <- apply(subgrid_data[, predictors, drop = FALSE], 2, max, na.rm = TRUE)
        
        EnvRanges <- expand.grid(
          Env_Value = seq(0, 1, length.out = 100),
          Predictor = predictors
        )
        
        for (pred in predictors) {
          min_val <- PredMins[pred]
          max_val <- PredMaxs[pred]
          if (is.finite(min_val) && is.finite(max_val) && min_val != max_val) {
            EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(min_val, max_val, length.out = 100)
          }
        }
        # Store X, Y, FID in EnvRanges
        EnvRanges$X <- NA
        EnvRanges$Y <- NA
        EnvRanges$FID <- NA
        if(nrow(subgrid_data) >= 100){ # Sample 100 spatial points
          EnvRanges$X <- rep(subgrid_data$X[1:100], length(predictors))
          EnvRanges$Y <- rep(subgrid_data$Y[1:100], length(predictors))
          EnvRanges$FID <- rep(subgrid_data$FID[1:100], length(predictors))
        }
        # Set up PDP storage: [100 (Env Values) x N Predictors x N Bootstraps]
        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
        dimnames(PD)[[2]] <- predictors
        dimnames(PD)[[3]] <- paste0("Rep_", 1:n.boot)
        
        all_pdp_long_list <- list() # Initialize the master storage list here (outside of the bootstrap loop)
        
        # -------------------------------------------------------
        # 6. SETUP ADAPTIVE SPATIAL CROSS VALIDATION & MODEL TRAINING 
        # -------------------------------------------------------
        
        # ADAPTIVE SPATIAL BLOCK CV SETUP ---
        best_iteration <- 100 # Default fallback value
        cv_results_df <- NULL # Initialize as NULL
        
        # CV Probe for max spatial blocks within tile and generate plot
        grid_blocks <- st_make_grid(subgrid_sf, cellsize = block_size_m)
        points_in_blocks <- st_intersects(grid_blocks, subgrid_sf)
        max_k <- length(which(lengths(points_in_blocks) > 0))
        generate_cv_diagnostic_plot(subgrid_sf, st_sf(grid_blocks), max_k, tile_id, pair, output_dir_train)
        
        k_final <- min(n.folds, max_k)
        
        # MANUAL CROSS-VALIDATION LOOP SETUP ---[not using built xgboost.cv]
        tryCatch({
          if (k_final < 2) { stop(paste("CV not possible. Only", max_k, "spatial block(s) found.")) }
          
          scv <- cv_spatial(x = subgrid_sf, size = block_size_m, k = k_final, iteration = 200)
          
          if (is.null(scv) || is.null(scv$folds_list) || !is.list(scv$folds_list) || length(scv$folds_list) < k_final) {
            stop("blockCV failed to return a valid set of folds.")
          }
          
          early_stopping_rounds <- 10
          # Initialize empty vectors to store results from each fold
          best_nrounds_per_fold <- c()
          rmse_per_fold <- c()
          mae_per_fold <- c()
          
          
          for (k in 1:k_final) {
            train_idx <- unlist(scv$folds_list[[k]][1])
            test_idx <- unlist(scv$folds_list[[k]][2])
            
            if (length(unique(subgrid_data[test_idx, response_var])) < 2) {
              cat("INFO: Skipping fold", k, "due to zero variance in the test set response.\n", file = worker_log_file, append = TRUE)
              next
            }
            
            # SEPERATE MODEL TRAINING & TESTING DATA FOR CV
            # turn training data into matrix to be compatible with XGboost
            dtrain_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[train_idx, predictors]), label = subgrid_data[train_idx, response_var], missing = NA)
            dtest_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[test_idx, predictors]), label = subgrid_data[test_idx, response_var], missing = NA)
            
            watchlist_fold <- list(train = dtrain_fold, test = dtest_fold)
            # CV / fold model
            fold_model <- xgb.train(params = list(
              max_depth = 4, 
              eta = 0.01, 
              gamma = 1, 
              objective = "reg:squarederror"),
              data = dtrain_fold, nrounds = 1000, # when testing, best models were achieve around 600-1000 rounds
              watchlist = watchlist_fold, 
              early_stopping_rounds = early_stopping_rounds,
              # eval_metric = c("rmse", "mae"),
              eval_metric = "rmse",
              eval_metric = "mae",
              verbose = 0)
            
            if (!is.null(fold_model$best_iteration) && fold_model$best_iteration > 0) {
              best_nrounds_per_fold <- append(best_nrounds_per_fold, fold_model$best_iteration)
              rmse_per_fold <- append(rmse_per_fold, fold_model$evaluation_log$test_rmse[fold_model$best_iteration])
              mae_per_fold <- append(mae_per_fold, fold_model$evaluation_log$test_mae[fold_model$best_iteration])
            }
          }
          
          # --- MODIFIED: Create the final results data frame AFTER the loop ---
          if (length(best_nrounds_per_fold) > 0) {
            best_iteration <- round(mean(best_nrounds_per_fold, na.rm = TRUE))
            cv_results_df <- data.frame(
              tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
              test_rmse_mean = mean(rmse_per_fold, na.rm = TRUE),
              test_rmse_std = sd(rmse_per_fold, na.rm = TRUE),
              test_mae_mean = mean(mae_per_fold, na.rm = TRUE),
              test_mae_std = sd(mae_per_fold, na.rm = TRUE)
            )
            cat("INFO: CV successful. Optimal iteration:", best_iteration, "\n", file = worker_log_file, append = TRUE)
          } else {
            stop("Manual CV loop failed to find any best iterations.")
          }
          
        }, error = function(e) {
          cat("WARNING: CV SKIPPED for Tile:", tile_id, "| Pair:", pair, "with error:", conditionMessage(e), ". Using default 100 rounds.\n", file = worker_log_file, append = TRUE)
        })
        
        # --- If CV failed, create the fallback results dataframe ---
        if (is.null(cv_results_df)) {
          cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                      test_rmse_mean = NA, test_rmse_std = NA,
                                      test_mae_mean = NA, test_mae_std = NA)
        }
        

        
        # FINAL MODEL TRAINING
        dtrain_full <- xgb.DMatrix(data = as.matrix(subgrid_data[, predictors]), label = subgrid_data[[response_var]], missing = NA)
        xgb_params <- list(
          max_depth = 4, # Controls tree depth (higher values capture more interactions but risk overfitting). Common range: 3-10
          eta = 0.01, # Learning rate (lower values prevent overfitting, but require more trees). Common range: 0.001 - 0.3
          gamma = 1, 
          subsample = 0.7, # Fraction of data used per boosting round (lower values prevent overfitting). Common range: 0.5-1
          colsample_bytree = 0.8, # Fraction of predictors used per tree (lower values prevent overfitting). Common range: 0.5-1
          objective = "reg:squarederror") # Specifies regression with squared error loss as the objective function.
        
        
        
        # --- Bootstrap Loop = repeats the above over a number of desired iteration---
        for (b in seq_len(n.boot)) {
          # Use all data for bootstrap model, but train to best # of rounds from CV
          xgb_model <- xgb.train(
            params = xgb_params, # same parameters used in cross validation
            data = dtrain_full,
            nrounds = best_iteration, # Use best # of rounds from CV
            nthread = 1 # Number of CPU threads (set based on available computing resources)
          )
          
          ## ----NOTATION ON HOW CROSS VALIDATION SUPPORTS BETTER MODEL PERFORMANCE---##
          # The Cross-Validation (CV) step runs a preliminary version of the model to find the optimal number of training rounds before the model begins to overfit.
          # It repeatedly trains on subsets of the data (folds) and evaluates on a hold-out set (witheld from training),
          # tracking the performance at each iteration. We then identify the single best iteration number (`best_iteration`) where the error was lowest. 
          # This optimal number is then used to train the final model on all the data, ensuring it is powerful but not overfit.
          
          # -------------------------------------------------------
          # 7. STORE MODEL METRICS ----
          # -------------------------------------------------------
          # store model prediction for every iteration
          boot_mat[, b] <- predict(xgb_model, newdata = dtrain_full)
          
          # Extract importance scores from the model
          importance_matrix <- xgb.importance(model = xgb_model)
          if (nrow(importance_matrix) > 0) {
            # Map importance values to predictor names (some predictors may not appear in importance_matrix)
            importance_values <- setNames(importance_matrix$Gain, importance_matrix$Feature)
            # Ensure alignment: Only assign values where predictor names match
            matching_indices <- match(names(importance_values), rownames(influence_mat))
            valid_indices <- !is.na(matching_indices)
            # Assign values only for existing predictors
            influence_mat[matching_indices[valid_indices], b] <- importance_values[valid_indices]
            # Fill remaining NAs with 0
            influence_mat[is.na(influence_mat)] <- 0
          }   
          
          deviance_mat[b, "Dev.Exp"] <- cor(boot_mat[, b], subgrid_data[[response_var]], use = "complete.obs")^2
          deviance_mat[b, "RMSE"] <- sqrt(mean((boot_mat[, b] - subgrid_data[[response_var]])^2, na.rm = TRUE))
          deviance_mat[b, "R2"] <- cor(boot_mat[, b], subgrid_data[[response_var]], use = "complete.obs")^2
        #}
        
        
        
        
        # -------------------------------------------------------
        # 8. STORE PARTIAL DEPENDENCE PLOT DATA ----
        # -------------------------------------------------------
          # --- CORRECTED PARTIAL DEPENDENCE PLOT (PDP) CALCULATION ---
          # This section correctly isolates the effect of each predictor by holding
          # all other predictors at their mean value during prediction.
          
          # Initialize a list to store the PDP data for this bootstrap iteration
          PDP_Storage <- list()
          
          # First, calculate the mean of each predictor in the current training data.
          # These will be used to hold other predictors constant.
          predictor_means <- colMeans(subgrid_data[, predictors, drop = FALSE], na.rm = TRUE)
          
          # Loop through each predictor to calculate its partial dependence
          # to calculate PDP_Value and create the PDP_Storage[[j]] data frame...
          for (j in seq_along(predictors)) {
            pred_name <- predictors[j]
            
            # Create a temporary data frame for prediction.
            # Start with a grid of 100 rows, where each column is the mean of a predictor.
            pdp_grid <- as.data.frame(matrix(rep(predictor_means, each = 100), nrow = 100))
            colnames(pdp_grid) <- names(predictor_means)
            
            # Now, overwrite the column for the current predictor with its range of values.
            pdp_grid[[pred_name]] <- EnvRanges$Env_Value[EnvRanges$Predictor == pred_name]
            
            # Ensure the column order is correct, just in case.
            pdp_grid <- pdp_grid[, predictors, drop = FALSE]
            
            # Predict using the fully constructed grid.
            # The model now sees all predictors, with only one varying.
            pdp_predictions <- predict(xgb_model, newdata = as.matrix(pdp_grid))
            
            # Create a data frame for smoothing
            grid_for_smoothing <- data.frame(
              Env_Value = pdp_grid[[pred_name]],
              PDP_Value = pdp_predictions
            )
            
            # Apply loess smoothing to the curve, with a fallback to raw values if it fails.
            loess_fit <- tryCatch(loess(PDP_Value ~ Env_Value, data = grid_for_smoothing, span = 1), error = function(e) NULL)
            
            # Store the smoothed (or raw) predictions in the main PD array for this bootstrap.
            PD[, j, b] <- if (!is.null(loess_fit)) {
              predict(loess_fit, newdata = data.frame(Env_Value = grid_for_smoothing$Env_Value))
            } else {
              grid_for_smoothing$PDP_Value
            }
            
            # Store the final, smoothed results in the long-format list
            PDP_Storage[[j]] <- data.frame(
              Predictor = pred_name,
              Env_Value = grid_for_smoothing$Env_Value,
              Replicate = paste0("Rep_", b),
              PDP_Value = PD[, j, b],
              # Replicating the first 100 spatial identifiers for consistency
              X = subgrid_data$X[1:100],
              Y = subgrid_data$Y[1:100],
              FID = subgrid_data$FID[1:100]
            )
          } ## End of partial dependence predictor loop
          
          # Combine the PDP results for this single bootstrap iteration
          PDP_Long_boot <- bind_rows(PDP_Storage) 
          # Add the results for this iteration to the master list
          all_pdp_long_list[[b]] <- PDP_Long_boot

      } ## End bootstrap loop
      
      # Convert all PDP lists from the bootstraps into a single long format dataframe for plotting
      PDP_Long <- bind_rows(all_pdp_long_list)
      
      # -------------------------------------------------------
      # 9. SAVE OUTPUTS----
      # -------------------------------------------------------
      tile_dir <- file.path(output_dir_train, tile_id)
      if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE)
      
      cat(Sys.time(), " Writing outputs for Tile:", tile_id, "| Year Pair:", pair, "\n", file = worker_log_file, append = TRUE)
      
      # Save CV results
      
      write_fst(as.data.table(cv_results_df), file.path(tile_dir, paste0("cv_results_", pair, ".fst")))
      
      
      # Model deviance - goodness of model fit  
      write_fst(as.data.table(deviance_mat), file.path(tile_dir, paste0("deviance_", pair, ".fst")))
      
      # Predictor influence /contribution
      influence_df <- as.data.frame(influence_mat)
      influence_df$Predictor <- rownames(influence_mat)
      influence_df <- influence_df[, c("Predictor", paste0("Rep_", seq_len(n.boot)))] # Reorder columns
      write_fst(as.data.table(influence_df), file.path(tile_dir, paste0("influence_", pair, ".fst")))
      
      # Model predictions from each boostrap
      #retain spatial identifiers, and append actual elevation change (b.change) in addition to predicted change 
      boot_df <- as.data.frame(boot_mat) 
      colnames(boot_df) <- paste0("Boot_", seq_len(n.boot)) # Rename boot iterations
      boot_df <- cbind(subgrid_data[, c("X", "Y", "FID")], b.change = subgrid_data[[response_var]], boot_df) # Append predicted change
      write_fst(as.data.table(boot_df), file.path(tile_dir, paste0("bootstraps_", pair, ".fst")))
      
      # Save model trained on full data with best # of rounds
      final_model <- xgb.train(params = xgb_params, data = dtrain_full, nrounds = best_iteration, nthread = 1)
      saveRDS(final_model, file.path(tile_dir, paste0("xgb_model_", pair, ".rds")))
      
      # Save PDP data and Env Values Together
      write_fst(as.data.table(PDP_Long), file.path(tile_dir, paste0("pdp_data_", pair, ".fst")))  
      write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))  
      write_fst(as.data.table(bind_rows(PDP_Storage)), file.path(tile_dir, paste0("pdp_data_raw_", pair, ".fst")))
      
      
      # -GENERATE DIAGNOSTIC PLOT OF MODEL FIT ---
      plot_data <- data.frame(
        Actual = subgrid_data[[response_var]],
        Predicted = rowMeans(boot_mat, na.rm = TRUE)
      )
      fit_plot <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
        geom_point(alpha = 0.3, color = "darkblue") +
        geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
        labs(
          title = paste("Model Fit for Tile:", tile_id, "| Pair:", pair),
          subtitle = paste("R-squared =", round(mean(deviance_mat[,"R2"], na.rm=TRUE), 3), "| RMSE =", round(mean(deviance_mat[,"RMSE"], na.rm=TRUE), 3)),
          x = "Actual Change (m)",
          y = "Predicted Change (m)"
        ) +
        theme_minimal()
      
      plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
      if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
      ggsave(filename = file.path(plot_dir, paste0("model_fit_", pair, ".png")), plot = fit_plot, width = 7, height = 7, dpi = 150)
      
    }, error = function(e) {
      cat("FATAL ERROR in Tile:", tile_id, "| Pair:", pair, "|", conditionMessage(e), "\n", file = worker_log_file, append = TRUE)
    }) # End master tryCatch
  } # End inner foreach loop
} # End outer foreach loop

# -------------------------------------------------------
# 3. CLOSE PARALLEL CLUSTER AND CONSOLIDATE LOGS
# -------------------------------------------------------
stopCluster(cl)
cat("\nParallel processing complete. Consolidating logs...\n")

worker_log_files <- list.files(output_dir_train, pattern = "^log_worker_.*\\.txt$", recursive = TRUE, full.names = TRUE)
for (log in worker_log_files) {
  content <- try(readLines(log), silent = TRUE)
  if (!inherits(content, "try-error")) {
    write(content, file = master_log_file, append = TRUE)
    file.remove(log) # Clean up individual worker logs
  }
}

cat("\n✅ Model Training Complete! Check `error_log_final.txt` for any issues.\n")
return(results_list)
}

#Final 

# 7.9 full functionality added back in for testing, removed smoothing from PDP, more comprehensive mean bootstrap, Switched to 3D array.  (WITH ALL NOTATION)
# ==============================================================================
#
#               XGBoost Model Training Function Set (Production Version)
#
# ==============================================================================

# This script contains a complete, refactored set of functions to train
# the XGBoost models. It is designed to:
#   1. Run a robust, parallelized spatial cross-validation to find the optimal model parameters.
#   2. Train a final model and run multiple bootstrap replicates to capture uncertainty.
#   3. Save all necessary outputs for the prediction workflow, including:
#      - The final trained model object.
#      - Raw, unsmoothed Partial Dependence Plot (PDP) data.
#      - A complete bootstrap prediction file with Mean and Standard Deviation calculated.
#      - Raster outputs for the Mean Bootstrap Prediction and its uncertainty (SD).

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis) # For advanced plotting
library(gridExtra) # For arranging plots
library(ggplot2)
library(FNN)
library(foreach)
library(doParallel)
library(blockCV)

# ==============================================================================
#   MAIN TRAINING FUNCTION
# ==============================================================================
Model_Train_Full_SpatialCV <- function(training_sub_grids_UTM, output_dir_train, year_pairs,
                                       block_size_m, n.boot = 20, n.folds = 5) {
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\nStarting Full XGBoost Model Training with Robust Parallel Spatial CV...\n")
  
  # Load libraries for the main session
  library(foreach)
  library(doParallel)
  
  # --- Setup Parallel Processing ---
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  if (num_cores < 1) num_cores <- 1 # Ensure at least one core is used
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
  # Outer Loop - Using %dopar% to run the outer loop (over tiles) in parallel
  results_list <- foreach(
    i = seq_len(length(tiles_df)),
    .combine = 'c',
    .packages = c("xgboost", "dplyr", "data.table", "fst", "tidyr", "sf", "blockCV", "raster", "ggplot2", "foreach"),
    .export = "generate_cv_diagnostic_plot", # Make the helper function available to workers
    .errorhandling = "pass"
  ) %dopar% {
    
    # Inner Loop - over year pairs runs sequentially
    tile_id <- tiles_df[[i]]
    worker_log_file <- file.path(output_dir_train, tile_id, paste0("log_worker_", tile_id, ".txt"))
    cat("Worker log for tile:", tile_id, "started at", as.character(Sys.time()), "\n", file = worker_log_file, append = FALSE)
    
    foreach(pair = year_pairs, .combine = 'c') %do% {
      
      tryCatch({
        # --- a. Load Data---
        cat("\nProcessing Tile:", tile_id, "| Year Pair:", pair, "\n", file = worker_log_file, append = TRUE)
        
        training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
        boruta_results_path <- file.path(output_dir_train, tile_id, paste0("boruta_selection_", pair, ".rds"))
        if (!file.exists(training_data_path) || !file.exists(boruta_results_path)) {
          cat("DIAGNOSTIC: Missing input file(s). Skipping.\n", file = worker_log_file, append = TRUE)
          return(NULL)
        }
        
        training_data <- as.data.frame(read_fst(training_data_path, as.data.table = TRUE))
        boruta_results <- readRDS(boruta_results_path)
        predictors <- boruta_results$confirmed_predictors
        response_var <- paste0("b.change.", pair)
        if (length(predictors) == 0 || !response_var %in% names(training_data)) {
          cat("DIAGNOSTIC: No predictors or response variable found. Skipping.\n", file = worker_log_file, append = TRUE)
          return(NULL)
        }
        cat("DIAGNOSTIC: Loaded", nrow(training_data), "total rows from input data.\n", file = worker_log_file, append = TRUE)
        
        # --- b. Define CRS ---
        grid_crs_epsg <- 32617 # Hardcoded UTM Zone 17N - FOR NOW, WILL NEED TO ADAPT SPATIALLY PER REGION
        grid_crs_proj4 <- sf::st_crs(grid_crs_epsg)$proj4string
        
        # -------------------------------------------------------
        # 3. FILTER & PREPARE TRAINING DATA----
        # -------------------------------------------------------
        # This uses XGBoost's native NA handling by only filtering NAs in the response variable.
        cols_to_select <- c(predictors, response_var, "X", "Y", "FID")
        subgrid_data <- training_data %>%
          dplyr::select(any_of(cols_to_select)) %>%
          mutate(across(all_of(c(predictors, response_var)), ~as.numeric(as.character(.)))) %>%
          filter(if_all(all_of(c(predictors, response_var)), is.finite))
        
        cat("DIAGNOSTIC: Filtered data to", nrow(subgrid_data), "rows with finite values for training.\n", file = worker_log_file, append = TRUE)
        if (nrow(subgrid_data) < 100) {
          cat("DIAGNOSTIC: Insufficient data (<100 rows) after filtering. Skipping.\n", file = worker_log_file, append = TRUE)
          return(NULL)
        }
        
        subgrid_sf <- st_as_sf(subgrid_data, coords = c("X", "Y"), remove = FALSE, crs = grid_crs_epsg)
        
        #Modification
        predictor_ranges <- data.table(
          Predictor = predictors,
          Min_Value = sapply(subgrid_data[, predictors], min, na.rm = TRUE),
          Max_Value = sapply(subgrid_data[, predictors], max, na.rm = TRUE)
        )
        predictor_ranges[, Range_Width := Max_Value - Min_Value]

        # -------------------------------------------------------
        # 4. INITIALIZE METRIC OUTPUT STORAGE ARRAYS----
        # -------------------------------------------------------
        deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot); colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
        influence_mat <- array(NA, dim = c(length(predictors), n.boot)); rownames(influence_mat) <- predictors; colnames(influence_mat) <- paste0("Rep_", 1:n.boot)
        
        # MODIFICATION: Use a 3D array for bootstrap predictions [rows, 1, bootstraps]
        boot_array <- array(NA, dim = c(nrow(subgrid_data), 1, n.boot))
        
        cv_results_list <- list()
        
        # -------------------------------------------------------
        # 5. SETUP for PARTIAL DEPENDENCE PLOT (PDP) DATA----
        # -------------------------------------------------------
        PredMins <- apply(subgrid_data[, predictors, drop = FALSE], 2, min, na.rm = TRUE)
        PredMaxs <- apply(subgrid_data[, predictors, drop = FALSE], 2, max, na.rm = TRUE)
        
        EnvRanges <- expand.grid(
          Env_Value = seq(0, 1, length.out = 100),
          Predictor = predictors
        )
        
        for (pred in predictors) {
          min_val <- PredMins[pred]
          max_val <- PredMaxs[pred]
          if (is.finite(min_val) && is.finite(max_val) && min_val != max_val) {
            EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(min_val, max_val, length.out = 100)
          }
        }
        # Store X, Y, FID in EnvRanges (good to save in every output dataset)
        EnvRanges$X <- NA; EnvRanges$Y <- NA; EnvRanges$FID <- NA
        if(nrow(subgrid_data) >= 100){ # Sample 100 spatial points
          EnvRanges$X <- rep(subgrid_data$X[1:100], length(predictors))
          EnvRanges$Y <- rep(subgrid_data$Y[1:100], length(predictors))
          EnvRanges$FID <- rep(subgrid_data$FID[1:100], length(predictors))
        }
        # Set up PDP storage: [100 (Env Values) x N Predictors x N Bootstraps]
        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
        dimnames(PD)[[2]] <- predictors; dimnames(PD)[[3]] <- paste0("Rep_", 1:n.boot)
        
        all_pdp_long_list <- list() # Initialize the master storage list here (outside of the bootstrap loop)
        
        # -------------------------------------------------------
        # 6. SETUP ADAPTIVE SPATIAL CROSS VALIDATION & MODEL TRAINING
        # -------------------------------------------------------
        # ADAPTIVE SPATIAL BLOCK CV SETUP ---
        best_iteration <- 100 # Default fallback value
        cv_results_df <- NULL # Initialize as NULL
        
        # CV Probe for max spatial blocks within tile and generate plot
        grid_blocks <- st_make_grid(subgrid_sf, cellsize = block_size_m)
        points_in_blocks <- st_intersects(grid_blocks, subgrid_sf)
        max_k <- length(which(lengths(points_in_blocks) > 0))
        generate_cv_diagnostic_plot(subgrid_sf, st_sf(grid_blocks), max_k, tile_id, pair, output_dir_train)
        
        k_final <- min(n.folds, max_k)
        
        # MANUAL CROSS-VALIDATION LOOP SETUP ---[not using built xgboost.cv]
        tryCatch({
          if (k_final < 2) { stop(paste("CV not possible. Only", max_k, "spatial block(s) found.")) }
          scv <- cv_spatial(x = subgrid_sf, size = block_size_m, k = k_final, iteration = 200)
          if (is.null(scv) || is.null(scv$folds_list) || !is.list(scv$folds_list) || length(scv$folds_list) < k_final) {
            stop("blockCV failed to return a valid set of folds.")
          }
          early_stopping_rounds <- 10
          # Initialize empty vectors to store results from each fold
          best_nrounds_per_fold <- c()
          rmse_per_fold <- c()
          mae_per_fold <- c()
          
          for (k in 1:k_final) {
            train_idx <- unlist(scv$folds_list[[k]][1]); test_idx <- unlist(scv$folds_list[[k]][2])
            if (length(unique(subgrid_data[test_idx, response_var])) < 2) { next }
            
            # SEPERATE MODEL TRAINING & TESTING DATA FOR CV
            # turn training data into matrix to be compatible with XGboost
            dtrain_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[train_idx, predictors]), label = subgrid_data[train_idx, response_var], missing = NA)
            dtest_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[test_idx, predictors]), label = subgrid_data[test_idx, response_var], missing = NA)
            watchlist_fold <- list(train = dtrain_fold, test = dtest_fold)
            
            # CV/ Fold model 
            fold_model <- xgb.train(params = list
                                    (max_depth = 4,
                                    eta = 0.01, gamma = 1,
                                    objective = "reg:squarederror"),
                                    data = dtrain_fold, nrounds = 1000, # when testing, best models were achieve around 600-1000 rounds
                                    watchlist = watchlist_fold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    eval_metric = c("rmse", "mae"), verbose = 0)
            
            
            if (!is.null(fold_model$best_iteration) && fold_model$best_iteration > 0) {
              best_nrounds_per_fold <- append(best_nrounds_per_fold, fold_model$best_iteration)
              rmse_per_fold <- append(rmse_per_fold, fold_model$evaluation_log$test_rmse[fold_model$best_iteration])
              mae_per_fold <- append(mae_per_fold, fold_model$evaluation_log$test_mae[fold_model$best_iteration])
            }
          }
          
          if (length(best_nrounds_per_fold) > 0) {
            best_iteration <- round(mean(best_nrounds_per_fold, na.rm = TRUE))
            cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                        test_rmse_mean = mean(rmse_per_fold, na.rm = TRUE), test_rmse_std = sd(rmse_per_fold, na.rm = TRUE),
                                        test_mae_mean = mean(mae_per_fold, na.rm = TRUE), test_mae_std = sd(mae_per_fold, na.rm = TRUE))
            cat("DIAGNOSTIC: CV successful. Optimal iteration:", best_iteration, "\n", file = worker_log_file, append = TRUE)
          } else { stop("Manual CV loop failed to find any best iterations.") }
        }, error = function(e) {
          cat("WARNING: CV SKIPPED for Tile:", tile_id, "| Pair:", pair, "with error:", conditionMessage(e), ". Using default 100 rounds.\n", file = worker_log_file, append = TRUE)
        })
        
        if (is.null(cv_results_df)) {
          cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                      test_rmse_mean = NA, test_rmse_std = NA, test_mae_mean = NA, test_mae_std = NA)
        }
        
        # FINAL TRAINING MODEL
        dtrain_full <- xgb.DMatrix(data = as.matrix(subgrid_data[, predictors]), label = subgrid_data[[response_var]], missing = NA)
        xgb_params <- list(
          max_depth = 4, # Controls tree depth (higher values capture more interactions but risk overfitting). Common range: 3-10
          eta = 0.01, # Learning rate (lower values prevent overfitting, but require more trees). Common range: 0.001 - 0.3
          gamma = 1, # Minimum loss reduction required to make a further partition.
          subsample = 0.7, # Fraction of data used per boosting round (lower values prevent overfitting). Common range: 0.5-1
          colsample_bytree = 0.8, # Fraction of predictors used per tree (lower values prevent overfitting). Common range: 0.5-1
          objective = "reg:squarederror") # Specifies regression with squared error loss as the objective function.
        
        cat("DIAGNOSTIC: Starting bootstrap loop for", n.boot, "iterations...\n", file = worker_log_file, append = TRUE)
        
        
        # --- Bootstrap Loop = repeats the above over a number of desired iteration---
        for (b in seq_len(n.boot)) {
          # Use all data for bootstrap model, but train to best # of rounds from CV
          xgb_model <- xgb.train(
            params = xgb_params, # same parameters used in cross validation
            data = dtrain_full,
            nrounds = best_iteration, # Use best # of rounds from CV
            nthread = 1 # Number of CPU threads (set based on available computing resources)
          )
          
          ## ----NOTATION ON HOW CROSS VALIDATION SUPPORTS BETTER MODEL PERFORMANCE---##
          # The Cross-Validation (CV) step runs a preliminary version of the model to find the optimal number of training rounds before the model begins to overfit.
          # It repeatedly trains on subsets of the data (folds) and evaluates on a hold-out set (witheld from training),
          # tracking the performance at each iteration. We then identify the single best iteration number (`best_iteration`) where the error was lowest. 
          # This optimal number is then used to train the final model on all the data, ensuring it is powerful but not overfit.
          
          
          # -------------------------------------------------------
          # 7. STORE MODEL METRICS ----
          # -------------------------------------------------------
          predictions <- predict(xgb_model, newdata = dtrain_full)
          # store model prediction for every iteration
          boot_array[, 1, b] <- predictions
          
          importance_matrix <- xgb.importance(model = xgb_model)
          if (nrow(importance_matrix) > 0) {
            importance_values <- setNames(importance_matrix$Gain, importance_matrix$Feature)
            matching_indices <- match(names(importance_values), rownames(influence_mat))
            valid_indices <- !is.na(matching_indices)
            influence_mat[matching_indices[valid_indices], b] <- importance_values[valid_indices]
            influence_mat[is.na(influence_mat)] <- 0
          }
          
          deviance_mat[b, "Dev.Exp"] <- cor(predictions, subgrid_data[[response_var]], use = "complete.obs")^2
          deviance_mat[b, "RMSE"] <- sqrt(mean((predictions - subgrid_data[[response_var]])^2, na.rm = TRUE))
          deviance_mat[b, "R2"] <- cor(predictions, subgrid_data[[response_var]], use = "complete.obs")^2
          
          # -------------------------------------------------------
          # 8. STORE PARTIAL DEPENDENCE PLOT DATA ----
          # -------------------------------------------------------
          # This section correctly isolates the effect of each predictor by holding
          # all other predictors at their mean value during prediction
          
          # Initialize a list to store the PDP data for this bootstrap iteration
          PDP_Storage <- list()
          # First, calculate the mean of each predictor in the current training data.
          # These will be used to hold other predictors constant
          predictor_means <- colMeans(subgrid_data[, predictors, drop = FALSE], na.rm = TRUE)
         
          # Loop through each predictor to calculate its partial dependence
          # to calculate PDP_Value and create the PDP_Storage[[j]] data frame...
           for (j in seq_along(predictors)) {
            pred_name <- predictors[j]
            
            # Create a temporary data frame for prediction.
            # Start with a grid of 100 rows, where each column is the mean of a predictor.
            pdp_grid <- as.data.frame(matrix(rep(predictor_means, each = 100), nrow = 100))
            colnames(pdp_grid) <- names(predictor_means)
            
            # Now, overwrite the column for the current predictor with its range of values.
            pdp_grid[[pred_name]] <- EnvRanges$Env_Value[EnvRanges$Predictor == pred_name]
           
            # Ensure the column order is correct, just in case.
             pdp_grid <- pdp_grid[, predictors, drop = FALSE]
            
             # Predict using the fully constructed grid.
             # The model now sees all predictors, with only one varying
             pdp_predictions <- predict(xgb_model, newdata = as.matrix(pdp_grid))
            
             # Store the predictions in the main PD array for this bootstrap.
             PD[, j, b] <- pdp_predictions
           
             
             # Store the final, smoothed results in the long-format list
             PDP_Storage[[j]] <- data.frame(
               Predictor = pred_name,
               Env_Value = pdp_grid[[pred_name]],
               Replicate = paste0("Rep_", b),
               PDP_Value = PD[, j, b],
               # Replicating the first 100 spatial identifiers for consistency
               X = subgrid_data$X[1:100],
               Y = subgrid_data$Y[1:100],
               FID = subgrid_data$FID[1:100]
             )
           } ## End of partial dependence predictor loop
          
          # Add the results for this iteration to the master list
          all_pdp_long_list[[b]] <- bind_rows(PDP_Storage)
        }
        cat("DIAGNOSTIC: Bootstrap loop finished.\n", file = worker_log_file, append = TRUE)
        # Convert all PDP lists from the bootstraps into a single long format dataframe for plotting
        PDP_Long <- bind_rows(all_pdp_long_list)
        
        # -------------------------------------------------------
        # 8.5. PROCESS BOOTSTRAP PREDICTIONS & CALCULATE STATISTICS
        # -------------------------------------------------------
        cat("DIAGNOSTIC: Processing bootstrap results. Array dimensions:", paste(dim(boot_array), collapse = "x"), "\n", file = worker_log_file, append = TRUE)
        
        # Calculate Mean and SD directly from the 3D array across the 3rd dimension (bootstraps)
        Mean_Prediction <- apply(boot_array, 1, mean, na.rm = TRUE)
        Uncertainty_SD <- apply(boot_array, 1, sd, na.rm = TRUE)
        if (n.boot == 1) { Uncertainty_SD[is.na(Uncertainty_SD)] <- 0 }
        
        # Create the final, simplified data frame
        boot_df <- data.table(
          FID = subgrid_data$FID,
          X = subgrid_data$X,
          Y = subgrid_data$Y,
          b.change_actual = subgrid_data[[response_var]],
          Mean_Prediction = Mean_Prediction,
          Uncertainty_SD = Uncertainty_SD
        )
        cat("DIAGNOSTIC: Created final bootstrap data frame with", nrow(boot_df), "rows.\n", file = worker_log_file, append = TRUE)
        
        # -------------------------------------------------------
        # 9. SAVE OUTPUTS----
        # -------------------------------------------------------
        tile_dir <- file.path(output_dir_train, tile_id)
        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE)
        cat("DIAGNOSTIC: Writing outputs to", tile_dir, "\n", file = worker_log_file, append = TRUE)
        
        # Save CV results
        write_fst(as.data.table(cv_results_df), file.path(tile_dir, paste0("cv_results_", pair, ".fst")))
        # Model deviance - goodness of model fit 
        write_fst(as.data.table(deviance_mat), file.path(tile_dir, paste0("deviance_", pair, ".fst")))
        # Predictor influence /contribution
        influence_df <- as.data.frame(influence_mat); influence_df$Predictor <- rownames(influence_mat)
        write_fst(as.data.table(influence_df), file.path(tile_dir, paste0("influence_", pair, ".fst")))
        write_fst(predictor_ranges, file.path(tile_dir, paste0("predictor_ranges_", pair, ".fst")))
        
        # # Model predictions from each boostrap
        write_fst(boot_df, file.path(tile_dir, paste0("bootstraps_", pair, ".fst")))
        
        # Save Rasters
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
        # Save model trained on full data with best # of rounds
        final_model <- xgb.train(params = xgb_params, data = dtrain_full, nrounds = best_iteration, nthread = 1)
        saveRDS(final_model, file.path(tile_dir, paste0("xgb_model_", pair, ".rds")))
        # Save PDP data and Env Values Together
        write_fst(PDP_Long, file.path(tile_dir, paste0("pdp_data_long_", pair, ".fst")))
        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))
        
        # GENERATE DIAGNOSTIC PLOT OF MODEL FIT ---
        plot_data <- boot_df[!is.na(Mean_Prediction), .(Actual = b.change_actual, Predicted = Mean_Prediction)]
        fit_plot <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
          geom_point(alpha = 0.3, color = "darkblue") +
          geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
          labs(title = paste("Model Fit for Tile:", tile_id, "| Pair:", pair),
               subtitle = paste("Mean R-squared =", round(mean(deviance_mat[,"R2"], na.rm=TRUE), 3), "| Mean RMSE =", round(mean(deviance_mat[,"RMSE"], na.rm=TRUE), 3)),
               x = "Actual Change (m)", y = "Mean Predicted Change (m)") +
          theme_minimal()
        plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
        if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
        ggsave(filename = file.path(plot_dir, paste0("model_fit_", pair, ".png")), plot = fit_plot, width = 7, height = 7, dpi = 150)
        cat("DIAGNOSTIC: All outputs saved successfully.\n", file = worker_log_file, append = TRUE)
        
      }, error = function(e) {
        cat("FATAL ERROR in Tile:", tile_id, "| Pair:", pair, "|", conditionMessage(e), "\n", file = worker_log_file, append = TRUE)
        cat("Backtrace:\n", paste(capture.output(traceback()), collapse="\n"), "\n", file = worker_log_file, append = TRUE)
      }) # End master tryCatch
    } # End inner foreach loop
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
      file.remove(log) # Clean up individual worker logs
    }
  }
  
  cat("\n[SUCCESS] Model Training Complete! Check `error_log_final.txt` for any issues.\n")
  return(results_list)
}





# 8.0 
# ==============================================================================
#
#               XGBoost Model Training Function Set (Production Version)
#
# ==============================================================================

# This script contains a complete, refactored set of functions to train
# the XGBoost models. It is designed to:
#   1. Run a robust, parallelized spatial cross-validation to find the optimal model parameters.
#   2. Train a final model and run multiple bootstrap replicates to capture uncertainty.
#   3. Save all necessary outputs for the prediction workflow, including:
#      - The final trained model object.
#      - Raw, unsmoothed Partial Dependence Plot (PDP) data.
#      - A complete bootstrap prediction file with Mean and Standard Deviation calculated.
#      - Raster outputs for the Mean Bootstrap Prediction and its uncertainty (SD).

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis) # For advanced plotting
library(gridExtra) # For arranging plots
library(ggplot2)
library(FNN)
library(foreach)
library(doParallel)
library(blockCV)

# ==============================================================================
#MAIN TRAINING FUNCTION = # works well, now uses for loop instead of nested parallel loop. 
#but the mean boots prediction is small (just the overlap of b.change, not the full tile)
# ==============================================================================
Model_Train_Full_SpatialCV <- function(training_sub_grids_UTM, output_dir_train, year_pairs,
                                       block_size_m, n.boot = 20, n.folds = 5) {
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\nStarting Full XGBoost Model Training with Robust Parallel Spatial CV...\n")
  
  # Load libraries for the main session
  library(foreach)
  library(doParallel)
  
  # --- Setup Parallel Processing ---
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  if (num_cores < 1) num_cores <- 1 # Ensure at least one core is used
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
  # Outer Loop - Using %dopar% to run the outer loop (over tiles) in parallel
  results_list <- foreach(
    i = seq_len(length(tiles_df)),
    .combine = 'c',
    .packages = c("data.table", "fst"), # Only load lightweight, essential packages here
    .export = "generate_cv_diagnostic_plot", # Make the helper function available to workers
    .errorhandling = "pass"
  ) %dopar% {
    # Load all necessary libraries within each parallel worker
    # This staggers the memory allocation and prevents initialization crashes.
    library(xgboost)
    library(dplyr)
    library(tidyr)
    library(sf)
    library(blockCV)
    library(raster)
    library(ggplot2)
    
    # Get tile_id for this worker
    tile_id <- tiles_df[[i]]
    worker_log_file <- file.path(output_dir_train, tile_id, paste0("log_worker_", tile_id, ".txt"))
    cat("Worker log for tile:", tile_id, "started at", as.character(Sys.time()), "\n", file = worker_log_file, append = FALSE)
    
    # --- MODIFICATION START ---
    # Changed the inner, sequential loop to a standard 'for' loop.
    # This is more robust and less prone to silent failures within a dopar block.
    for (pair in year_pairs) {
      # --- MODIFICATION END ---
      
      tryCatch({
        # --- a. Load Data---
        cat("\nProcessing Tile:", tile_id, "| Year Pair:", pair, "\n", file = worker_log_file, append = TRUE)
        
        training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
        boruta_results_path <- file.path(output_dir_train, tile_id, paste0("boruta_selection_", pair, ".rds"))
        if (!file.exists(training_data_path) || !file.exists(boruta_results_path)) {
          cat("DIAGNOSTIC: Missing input file(s). Skipping.\n", file = worker_log_file, append = TRUE)
          next # Use 'next' to skip to the next iteration of the for loop
        }
        
        training_data <- as.data.frame(read_fst(training_data_path, as.data.table = TRUE))
        boruta_results <- readRDS(boruta_results_path)
        predictors <- boruta_results$confirmed_predictors
        response_var <- paste0("b.change.", pair)
        if (length(predictors) == 0 || !response_var %in% names(training_data)) {
          cat("DIAGNOSTIC: No predictors or response variable found. Skipping.\n", file = worker_log_file, append = TRUE)
          next
        }
        cat("DIAGNOSTIC: Loaded", nrow(training_data), "total rows from input data.\n", file = worker_log_file, append = TRUE)
        
        # --- b. Define CRS ---
        grid_crs_epsg <- 32617 # Hardcoded UTM Zone 17N - FOR NOW, WILL NEED TO ADAPT SPATIALLY PER REGION
        grid_crs_proj4 <- sf::st_crs(grid_crs_epsg)$proj4string
        
        # -------------------------------------------------------
        # 3. FILTER & PREPARE TRAINING DATA----
        # -------------------------------------------------------
        # This uses XGBoost's native NA handling by only filtering NAs in the response variable.
        cols_to_select <- c(predictors, response_var, "X", "Y", "FID")
        subgrid_data <- training_data %>%
          dplyr::select(any_of(cols_to_select)) %>%
          mutate(across(all_of(c(predictors, response_var)), ~as.numeric(as.character(.)))) %>%
          filter(if_all(all_of(c(predictors, response_var)), is.finite))
        
        cat("DIAGNOSTIC: Filtered data to", nrow(subgrid_data), "rows with finite values for training.\n", file = worker_log_file, append = TRUE)
        if (nrow(subgrid_data) < 100) {
          cat("DIAGNOSTIC: Insufficient data (<100 rows) after filtering. Skipping.\n", file = worker_log_file, append = TRUE)
          next
        }
        
        subgrid_sf <- st_as_sf(subgrid_data, coords = c("X", "Y"), remove = FALSE, crs = grid_crs_epsg)
        
        #Modification
        predictor_ranges <- data.table(
          Predictor = predictors,
          Min_Value = sapply(subgrid_data[, predictors], min, na.rm = TRUE),
          Max_Value = sapply(subgrid_data[, predictors], max, na.rm = TRUE)
        )
        predictor_ranges[, Range_Width := Max_Value - Min_Value]
        
        # -------------------------------------------------------
        # 4. INITIALIZE METRIC OUTPUT STORAGE ARRAYS----
        # -------------------------------------------------------
        deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot); colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
        influence_mat <- array(NA, dim = c(length(predictors), n.boot)); rownames(influence_mat) <- predictors; colnames(influence_mat) <- paste0("Rep_", 1:n.boot)
        
        # Using a standard 2D matrix is more robust for storing bootstrap predictions.
        # This avoids potential errors from handling a 3D array with a singleton dimension.
        # Dimensions: [number of data points, number of bootstraps]
        boot_matrix <- matrix(NA, nrow = nrow(subgrid_data), ncol = n.boot)
        
        cv_results_list <- list()
        
        # -------------------------------------------------------
        # 5. SETUP for PARTIAL DEPENDENCE PLOT (PDP) DATA----
        # -------------------------------------------------------
        PredMins <- apply(subgrid_data[, predictors, drop = FALSE], 2, min, na.rm = TRUE)
        PredMaxs <- apply(subgrid_data[, predictors, drop = FALSE], 2, max, na.rm = TRUE)
        
        EnvRanges <- expand.grid(
          Env_Value = seq(0, 1, length.out = 100),
          Predictor = predictors
        )
        
        for (pred in predictors) {
          min_val <- PredMins[pred]
          max_val <- PredMaxs[pred]
          if (is.finite(min_val) && is.finite(max_val) && min_val != max_val) {
            EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(min_val, max_val, length.out = 100)
          }
        }
        # Store X, Y, FID in EnvRanges (good to save in every output dataset)
        EnvRanges$X <- NA; EnvRanges$Y <- NA; EnvRanges$FID <- NA
        if(nrow(subgrid_data) >= 100){ # Sample 100 spatial points
          EnvRanges$X <- rep(subgrid_data$X[1:100], length(predictors))
          EnvRanges$Y <- rep(subgrid_data$Y[1:100], length(predictors))
          EnvRanges$FID <- rep(subgrid_data$FID[1:100], length(predictors))
        }
        # Set up PDP storage: [100 (Env Values) x N Predictors x N Bootstraps]
        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
        dimnames(PD)[[2]] <- predictors; dimnames(PD)[[3]] <- paste0("Rep_", 1:n.boot)
        
        all_pdp_long_list <- list() # Initialize the master storage list here (outside of the bootstrap loop)
        
        # -------------------------------------------------------
        # 6. SETUP ADAPTIVE SPATIAL CROSS VALIDATION & MODEL TRAINING
        # -------------------------------------------------------
        # ADAPTIVE SPATIAL BLOCK CV SETUP ---
        best_iteration <- 100 # Default fallback value
        cv_results_df <- NULL # Initialize as NULL
        
        # CV Probe for max spatial blocks within tile and generate plot
        grid_blocks <- st_make_grid(subgrid_sf, cellsize = block_size_m)
        points_in_blocks <- st_intersects(grid_blocks, subgrid_sf)
        max_k <- length(which(lengths(points_in_blocks) > 0))
        generate_cv_diagnostic_plot(subgrid_sf, st_sf(grid_blocks), max_k, tile_id, pair, output_dir_train)
        
        k_final <- min(n.folds, max_k)
        
        # MANUAL CROSS-VALIDATION LOOP SETUP ---[not using built xgboost.cv]
        tryCatch({
          if (k_final < 2) { stop(paste("CV not possible. Only", max_k, "spatial block(s) found.")) }
          scv <- cv_spatial(x = subgrid_sf, size = block_size_m, k = k_final, iteration = 200)
          if (is.null(scv) || is.null(scv$folds_list) || !is.list(scv$folds_list) || length(scv$folds_list) < k_final) {
            stop("blockCV failed to return a valid set of folds.")
          }
          early_stopping_rounds <- 10
          # Initialize empty vectors to store results from each fold
          best_nrounds_per_fold <- c()
          rmse_per_fold <- c()
          mae_per_fold <- c()
          
          for (k in 1:k_final) {
            train_idx <- unlist(scv$folds_list[[k]][1]); test_idx <- unlist(scv$folds_list[[k]][2])
            if (length(unique(subgrid_data[test_idx, response_var])) < 2) { next }
            
            # SEPERATE MODEL TRAINING & TESTING DATA FOR CV
            # turn training data into matrix to be compatible with XGboost
            dtrain_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[train_idx, predictors]), label = subgrid_data[train_idx, response_var], missing = NA)
            dtest_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[test_idx, predictors]), label = subgrid_data[test_idx, response_var], missing = NA)
            watchlist_fold <- list(train = dtrain_fold, test = dtest_fold)
            
            # CV/ Fold model
            fold_model <- xgb.train(params = list
                                    (max_depth = 4,
                                      eta = 0.01, gamma = 1,
                                      objective = "reg:squarederror"),
                                    data = dtrain_fold, nrounds = 1000, # when testing, best models were achieve around 600-1000 rounds
                                    watchlist = watchlist_fold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    eval_metric = c("rmse", "mae"), verbose = 0)
            
            
            if (!is.null(fold_model$best_iteration) && fold_model$best_iteration > 0) {
              best_nrounds_per_fold <- append(best_nrounds_per_fold, fold_model$best_iteration)
              rmse_per_fold <- append(rmse_per_fold, fold_model$evaluation_log$test_rmse[fold_model$best_iteration])
              mae_per_fold <- append(mae_per_fold, fold_model$evaluation_log$test_mae[fold_model$best_iteration])
            }
          }
          
          if (length(best_nrounds_per_fold) > 0) {
            best_iteration <- round(mean(best_nrounds_per_fold, na.rm = TRUE))
            cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                        test_rmse_mean = mean(rmse_per_fold, na.rm = TRUE), test_rmse_std = sd(rmse_per_fold, na.rm = TRUE),
                                        test_mae_mean = mean(mae_per_fold, na.rm = TRUE), test_mae_std = sd(mae_per_fold, na.rm = TRUE))
            cat("DIAGNOSTIC: CV successful. Optimal iteration:", best_iteration, "\n", file = worker_log_file, append = TRUE)
          } else { stop("Manual CV loop failed to find any best iterations.") }
        }, error = function(e) {
          cat("WARNING: CV SKIPPED for Tile:", tile_id, "| Pair:", pair, "with error:", conditionMessage(e), ". Using default 100 rounds.\n", file = worker_log_file, append = TRUE)
        })
        
        if (is.null(cv_results_df)) {
          cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                      test_rmse_mean = NA, test_rmse_std = NA, test_mae_mean = NA, test_mae_std = NA)
        }
        
        # FINAL TRAINING MODEL
        dtrain_full <- xgb.DMatrix(data = as.matrix(subgrid_data[, predictors]), label = subgrid_data[[response_var]], missing = NA)
        xgb_params <- list(
          max_depth = 4, # Controls tree depth (higher values capture more interactions but risk overfitting). Common range: 3-10
          eta = 0.01, # Learning rate (lower values prevent overfitting, but require more trees). Common range: 0.001 - 0.3
          gamma = 1, # Minimum loss reduction required to make a further partition.
          subsample = 0.7, # Fraction of data used per boosting round (lower values prevent overfitting). Common range: 0.5-1
          colsample_bytree = 0.8, # Fraction of predictors used per tree (lower values prevent overfitting). Common range: 0.5-1
          objective = "reg:squarederror") # Specifies regression with squared error loss as the objective function.
        
        cat("DIAGNOSTIC: Starting bootstrap loop for", n.boot, "iterations...\n", file = worker_log_file, append = TRUE)
        
        
        # --- Bootstrap Loop = repeats the above over a number of desired iteration---
        for (b in seq_len(n.boot)) {
          # A true bootstrap involves training on a sample of the data drawn with replacement.
          # This correctly captures model uncertainty arising from data variability.
          boot_indices <- sample(seq_len(nrow(subgrid_data)), replace = TRUE)
          dtrain_boot <- xgb.DMatrix(
            data = as.matrix(subgrid_data[boot_indices, predictors]),
            label = subgrid_data[boot_indices, response_var, drop = TRUE], # drop=TRUE ensures it's a vector
            missing = NA
          )
          
          # Train the model on the bootstrap sample, not the full dataset
          xgb_model <- xgb.train(
            params = xgb_params,
            data = dtrain_boot, # Use the resampled data for training
            nrounds = best_iteration,
            nthread = 1
          )
          
          ## ----NOTATION ON HOW CROSS VALIDATION SUPPORTS BETTER MODEL PERFORMANCE---##
          # The Cross-Validation (CV) step runs a preliminary version of the model to find the optimal number of training rounds before the model begins to overfit.
          # It repeatedly trains on subsets of the data (folds) and evaluates on a hold-out set (witheld from training),
          # tracking the performance at each iteration. We then identify the single best iteration number (`best_iteration`) where the error was lowest. 
          # This optimal number is then used to train the final model on all the data, ensuring it is powerful but not overfit.
          
          
          # -------------------------------------------------------
          # 7. STORE MODEL METRICS ----
          # -------------------------------------------------------
          # We predict on the original full dataset to get a prediction for every point
          predictions <- predict(xgb_model, newdata = dtrain_full)
          
          # Store model prediction for every iteration in the 2D matrix
          boot_matrix[, b] <- predictions
          
          importance_matrix <- xgb.importance(model = xgb_model)
          if (nrow(importance_matrix) > 0) {
            importance_values <- setNames(importance_matrix$Gain, importance_matrix$Feature)
            matching_indices <- match(names(importance_values), rownames(influence_mat))
            valid_indices <- !is.na(matching_indices)
            influence_mat[matching_indices[valid_indices], b] <- importance_values[valid_indices]
            influence_mat[is.na(influence_mat)] <- 0
          }
          
          deviance_mat[b, "Dev.Exp"] <- cor(predictions, subgrid_data[[response_var]], use = "complete.obs")^2
          deviance_mat[b, "RMSE"] <- sqrt(mean((predictions - subgrid_data[[response_var]])^2, na.rm = TRUE))
          deviance_mat[b, "R2"] <- cor(predictions, subgrid_data[[response_var]], use = "complete.obs")^2
          
          # -------------------------------------------------------
          # 8. STORE PARTIAL DEPENDENCE PLOT DATA ----
          # -------------------------------------------------------
          # This section correctly isolates the effect of each predictor by holding
          # all other predictors at their mean value during prediction
          
          # Initialize a list to store the PDP data for this bootstrap iteration
          PDP_Storage <- list()
          # First, calculate the mean of each predictor in the current training data.
          # These will be used to hold other predictors constant
          predictor_means <- colMeans(subgrid_data[, predictors, drop = FALSE], na.rm = TRUE)
          
          # Loop through each predictor to calculate its partial dependence
          # to calculate PDP_Value and create the PDP_Storage[[j]] data frame...
          for (j in seq_along(predictors)) {
            pred_name <- predictors[j]
            
            # Create a temporary data frame for prediction.
            # Start with a grid of 100 rows, where each column is the mean of a predictor.
            pdp_grid <- as.data.frame(matrix(rep(predictor_means, each = 100), nrow = 100))
            colnames(pdp_grid) <- names(predictor_means)
            
            # Now, overwrite the column for the current predictor with its range of values.
            pdp_grid[[pred_name]] <- EnvRanges$Env_Value[EnvRanges$Predictor == pred_name]
            
            # Ensure the column order is correct, just in case.
            pdp_grid <- pdp_grid[, predictors, drop = FALSE]
            
            # Predict using the fully constructed grid.
            # The model now sees all predictors, with only one varying
            pdp_predictions <- predict(xgb_model, newdata = as.matrix(pdp_grid))
            
            # Store the predictions in the main PD array for this bootstrap.
            PD[, j, b] <- pdp_predictions
            
            
            # Store the final, smoothed results in the long-format list
            PDP_Storage[[j]] <- data.frame(
              Predictor = pred_name,
              Env_Value = pdp_grid[[pred_name]],
              Replicate = paste0("Rep_", b),
              PDP_Value = PD[, j, b],
              # Replicating the first 100 spatial identifiers for consistency
              X = subgrid_data$X[1:100],
              Y = subgrid_data$Y[1:100],
              FID = subgrid_data$FID[1:100]
            )
          } ## End of partial dependence predictor loop
          
          # Add the results for this iteration to the master list
          all_pdp_long_list[[b]] <- bind_rows(PDP_Storage)
        }
        cat("DIAGNOSTIC: Bootstrap loop finished.\n", file = worker_log_file, append = TRUE)
        # Convert all PDP lists from the bootstraps into a single long format dataframe for plotting
        PDP_Long <- bind_rows(all_pdp_long_list)
        
        # -------------------------------------------------------
        # 8.5. PROCESS BOOTSTRAP PREDICTIONS & CALCULATE STATISTICS
        # -------------------------------------------------------
        cat("DIAGNOSTIC: Processing bootstrap results. Matrix dimensions:", paste(dim(boot_matrix), collapse = "x"), "\n", file = worker_log_file, append = TRUE)
        
        # Calculate Mean and SD from the 2D matrix across the rows.
        # `rowMeans` is an efficient, direct function for this task.
        Mean_Prediction <- rowMeans(boot_matrix, na.rm = TRUE)
        Uncertainty_SD <- apply(boot_matrix, 1, sd, na.rm = TRUE)
        if (n.boot == 1) { Uncertainty_SD[is.na(Uncertainty_SD)] <- 0 }
        
        # Create the final, simplified data frame
        boot_df <- data.table(
          FID = subgrid_data$FID,
          X = subgrid_data$X,
          Y = subgrid_data$Y,
          b.change_actual = subgrid_data[[response_var]],
          Mean_Prediction = Mean_Prediction,
          Uncertainty_SD = Uncertainty_SD
        )
        cat("DIAGNOSTIC: Created final bootstrap data frame with", nrow(boot_df), "rows.\n", file = worker_log_file, append = TRUE)
        
        # Explicitly remove the large matrix to free up memory
        rm(boot_matrix)
        
        # -------------------------------------------------------
        # 9. SAVE OUTPUTS----
        # -------------------------------------------------------
        tile_dir <- file.path(output_dir_train, tile_id)
        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE)
        cat("DIAGNOSTIC: Writing outputs to", tile_dir, "\n", file = worker_log_file, append = TRUE)
        
        # Save CV results
        write_fst(as.data.table(cv_results_df), file.path(tile_dir, paste0("cv_results_", pair, ".fst")))
        # Model deviance - goodness of model fit
        write_fst(as.data.table(deviance_mat), file.path(tile_dir, paste0("deviance_", pair, ".fst")))
        # Predictor influence /contribution
        influence_df <- as.data.frame(influence_mat); influence_df$Predictor <- rownames(influence_mat)
        write_fst(as.data.table(influence_df), file.path(tile_dir, paste0("influence_", pair, ".fst")))
        write_fst(predictor_ranges, file.path(tile_dir, paste0("predictor_ranges_", pair, ".fst")))
        
        # # Model predictions from each boostrap
        write_fst(boot_df, file.path(tile_dir, paste0("bootstraps_", pair, ".fst")))
        
        # Save Rasters
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
        # Save model trained on full data with best # of rounds
        final_model <- xgb.train(params = xgb_params, data = dtrain_full, nrounds = best_iteration, nthread = 1)
        saveRDS(final_model, file.path(tile_dir, paste0("xgb_model_", pair, ".rds")))
        # Save PDP data and Env Values Together
        write_fst(PDP_Long, file.path(tile_dir, paste0("pdp_data_long_", pair, ".fst")))
        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))
        
        # GENERATE DIAGNOSTIC PLOT OF MODEL FIT ---
        plot_data <- boot_df[!is.na(Mean_Prediction), .(Actual = b.change_actual, Predicted = Mean_Prediction)]
        fit_plot <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
          geom_point(alpha = 0.3, color = "darkblue") +
          geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
          labs(title = paste("Model Fit for Tile:", tile_id, "| Pair:", pair),
               subtitle = paste("Mean R-squared =", round(mean(deviance_mat[,"R2"], na.rm=TRUE), 3), "| Mean RMSE =", round(mean(deviance_mat[,"RMSE"], na.rm=TRUE), 3)),
               x = "Actual Change (m)", y = "Mean Predicted Change (m)") +
          theme_minimal()
        plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
        if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
        ggsave(filename = file.path(plot_dir, paste0("model_fit_", pair, ".png")), plot = fit_plot, width = 7, height = 7, dpi = 150)
        cat("DIAGNOSTIC: All outputs saved successfully.\n", file = worker_log_file, append = TRUE)
        
      }, error = function(e) {
        cat("FATAL ERROR in Tile:", tile_id, "| Pair:", pair, "|", conditionMessage(e), "\n", file = worker_log_file, append = TRUE)
        cat("Backtrace:\n", paste(capture.output(traceback()), collapse="\n"), "\n", file = worker_log_file, append = TRUE)
      }) # End master tryCatch
      
      # Call garbage collection at the end of the inner loop to free memory
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
      file.remove(log) # Clean up individual worker logs
    }
  }
  
  cat("\n[SUCCESS] Model Training Complete! Check `error_log_final.txt` for any issues.\n")
  return(results_list)
}

# 8.1
# 
# ==============================================================================
#
#      XGBoost Model Training Function Set (Production Version)
# # now with the option for interpolation / prediction over the whole tile, stored in the form of the boostrap
# matrix, which provided the SD, and mean prediction over the whole tile. Whilst seperately, the XGB boost model runs the training and stores 
# the best model prediction over just the b.change extent (training data - NA values for b.change column). That way when we come to the true prediction
# code we can examine the density distribution of each dataset to test which will give us the best prediction.
# ==============================================================================
# ==============================================================================
#
#  _XGBoost Model Training Function Set (Production Version)
#
# ==============================================================================

# This script contains a complete, refactored set of functions to train
# the XGBoost models. It is designed to:
#   1. Run a robust, parallelized spatial cross-validation to find the optimal model parameters.
#   2. Train a final model and run multiple bootstrap replicates to capture uncertainty.
#   3. Save all necessary outputs for the prediction workflow, including:
#      - The final trained model object.
#      - Raw, unsmoothed Partial Dependence Plot (PDP) data.
#      - A complete bootstrap prediction file with Mean and Standard Deviation calculated.
#      - Raster outputs for the Mean Bootstrap Prediction and its uncertainty (SD).

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis) # For advanced plotting
library(gridExtra) # For arranging plots
library(ggplot2)
library(FNN)
library(foreach)
library(doParallel)
library(blockCV)

# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
Model_Train_Full_SpatialCV <- function(training_sub_grids_UTM, output_dir_train, year_pairs,
                                       block_size_m, n.boot = 10, n.folds = 5) {
  # -------------------------------------------------------
  # 1. MODEL INITIALIZATION & ERROR LOGGING
  # -------------------------------------------------------
  cat("\nStarting Full XGBoost Model Training with Robust Parallel Spatial CV...\n")
  
  # Load libraries for the main session
  library(foreach)
  library(doParallel)
  
  # --- Setup Parallel Processing ---
  tiles_df <- as.character(training_sub_grids_UTM$tile_id)
  num_cores <- detectCores() - 1
  if (num_cores < 1) num_cores <- 1 # Ensure at least one core is used
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
  # Outer Loop - Using %dopar% to run the outer loop (over tiles) in parallel
  results_list <- foreach(
    i = seq_len(length(tiles_df)),
    .combine = 'c',
    .packages = c("data.table", "fst"), # Only load lightweight, essential packages here
    .export = "generate_cv_diagnostic_plot", # Make the helper function available to workers
    .errorhandling = "pass"
  ) %dopar% {
    # Load all necessary libraries within each parallel worker
    # This staggers the memory allocation and prevents initialization crashes.
    library(xgboost)
    library(dplyr)
    library(tidyr)
    library(sf)
    library(blockCV)
    library(raster)
    library(ggplot2)
    
    # Get tile_id for this worker
    tile_id <- tiles_df[[i]]
    worker_log_file <- file.path(output_dir_train, tile_id, paste0("log_worker_", tile_id, ".txt"))
    cat("Worker log for tile:", tile_id, "started at", as.character(Sys.time()), "\n", file = worker_log_file, append = FALSE)
    
    # Changed the inner, sequential loop to a standard 'for' loop.
    # This is more robust and less prone to silent failures within a dopar block.
    for (pair in year_pairs) {
      
      tryCatch({
        # --- a. Load Data---
        cat("\nProcessing Tile:", tile_id, "| Year Pair:", pair, "\n", file = worker_log_file, append = TRUE)
        
        training_data_path <- file.path(output_dir_train, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
        boruta_results_path <- file.path(output_dir_train, tile_id, paste0("boruta_selection_", pair, ".rds"))
        if (!file.exists(training_data_path) || !file.exists(boruta_results_path)) {
          cat("DIAGNOSTIC: Missing input file(s). Skipping.\n", file = worker_log_file, append = TRUE)
          next # Use 'next' to skip to the next iteration of the for loop
        }
        
        training_data <- as.data.frame(read_fst(training_data_path, as.data.table = TRUE))
        boruta_results <- readRDS(boruta_results_path)
        predictors <- boruta_results$confirmed_predictors
        response_var <- paste0("b.change.", pair)
        if (length(predictors) == 0 || !response_var %in% names(training_data)) {
          cat("DIAGNOSTIC: No predictors or response variable found. Skipping.\n", file = worker_log_file, append = TRUE)
          next
        }
        cat("DIAGNOSTIC: Loaded", nrow(training_data), "total rows from input data.\n", file = worker_log_file, append = TRUE)
        
        # --- b. Define CRS ---
        grid_crs_epsg <- 32617 # Hardcoded UTM Zone 17N - FOR NOW, WILL NEED TO ADAPT SPATIALLY PER REGION
        grid_crs_proj4 <- sf::st_crs(grid_crs_epsg)$proj4string
        
        # -------------------------------------------------------
        # 3. FILTER & PREPARE TRAINING DATA----
        # -------------------------------------------------------
        # --- MODIFICATION START ---
        # Create a dataset for PREDICTION (`full_tile_data`) that includes ALL rows.
        # We rely on XGBoost's native ability to handle NAs in predictors.
        cols_to_select <- c(predictors, response_var, "X", "Y", "FID")
        full_tile_data <- training_data %>%
          dplyr::select(any_of(cols_to_select)) %>%
          mutate(across(all_of(c(predictors, response_var)), ~as.numeric(as.character(.))))
        
        # Now, create the TRAINING dataset (`subgrid_data`) by filtering the full data
        # for rows where the response variable is valid. The model can only learn from known outcomes.
        subgrid_data <- full_tile_data %>%
          filter(is.finite(.data[[response_var]]))
        # --- MODIFICATION END ---
        
        cat("DIAGNOSTIC: Data prepared. Total points for prediction:", nrow(full_tile_data), ". Points for training:", nrow(subgrid_data), "\n", file = worker_log_file, append = TRUE)
        if (nrow(subgrid_data) < 100) {
          cat("DIAGNOSTIC: Insufficient training data (<100 rows). Skipping.\n", file = worker_log_file, append = TRUE)
          next
        }
        
        subgrid_sf <- st_as_sf(subgrid_data, coords = c("X", "Y"), remove = FALSE, crs = grid_crs_epsg)
        
        #Modification
        predictor_ranges <- data.table(
          Predictor = predictors,
          Min_Value = sapply(subgrid_data[, predictors], min, na.rm = TRUE),
          Max_Value = sapply(subgrid_data[, predictors], max, na.rm = TRUE)
        )
        predictor_ranges[, Range_Width := Max_Value - Min_Value]
        
        # -------------------------------------------------------
        # 4. INITIALIZE METRIC OUTPUT STORAGE ARRAYS----
        # -------------------------------------------------------
        deviance_mat <- matrix(NA, ncol = 3, nrow = n.boot); colnames(deviance_mat) <- c("Dev.Exp", "RMSE", "R2")
        influence_mat <- array(NA, dim = c(length(predictors), n.boot)); rownames(influence_mat) <- predictors; colnames(influence_mat) <- paste0("Rep_", 1:n.boot)
        
        # Initialize the prediction matrix to match the size of the full prediction dataset.
        boot_matrix <- matrix(NA, nrow = nrow(full_tile_data), ncol = n.boot)
        
        cv_results_list <- list()
        
        # -------------------------------------------------------
        # 5. SETUP for PARTIAL DEPENDENCE PLOT (PDP) DATA----
        # -------------------------------------------------------
        PredMins <- apply(subgrid_data[, predictors, drop = FALSE], 2, min, na.rm = TRUE)
        PredMaxs <- apply(subgrid_data[, predictors, drop = FALSE], 2, max, na.rm = TRUE)
        
        EnvRanges <- expand.grid(
          Env_Value = seq(0, 1, length.out = 100),
          Predictor = predictors
        )
        
        for (pred in predictors) {
          min_val <- PredMins[pred]
          max_val <- PredMaxs[pred]
          if (is.finite(min_val) && is.finite(max_val) && min_val != max_val) {
            EnvRanges$Env_Value[EnvRanges$Predictor == pred] <- seq(min_val, max_val, length.out = 100)
          }
        }
        # Store X, Y, FID in EnvRanges (good to save in every output dataset)
        EnvRanges$X <- NA; EnvRanges$Y <- NA; EnvRanges$FID <- NA
        if(nrow(subgrid_data) >= 100){ # Sample 100 spatial points
          EnvRanges$X <- rep(subgrid_data$X[1:100], length(predictors))
          EnvRanges$Y <- rep(subgrid_data$Y[1:100], length(predictors))
          EnvRanges$FID <- rep(subgrid_data$FID[1:100], length(predictors))
        }
        # Set up PDP storage: [100 (Env Values) x N Predictors x N Bootstraps]
        PD <- array(NA_real_, dim = c(100, length(predictors), n.boot))
        dimnames(PD)[[2]] <- predictors; dimnames(PD)[[3]] <- paste0("Rep_", 1:n.boot)
        
        all_pdp_long_list <- list() # Initialize the master storage list here (outside of the bootstrap loop)
        
        # -------------------------------------------------------
        # 6. SETUP ADAPTIVE SPATIAL CROSS VALIDATION & MODEL TRAINING
        # -------------------------------------------------------
        # ADAPTIVE SPATIAL BLOCK CV SETUP ---
        best_iteration <- 100 # Default fallback value
        cv_results_df <- NULL # Initialize as NULL
        
        # CV Probe for max spatial blocks within tile and generate plot
        grid_blocks <- st_make_grid(subgrid_sf, cellsize = block_size_m)
        points_in_blocks <- st_intersects(grid_blocks, subgrid_sf)
        max_k <- length(which(lengths(points_in_blocks) > 0))
        generate_cv_diagnostic_plot(subgrid_sf, st_sf(grid_blocks), max_k, tile_id, pair, output_dir_train)
        
        k_final <- min(n.folds, max_k)
        
        # MANUAL CROSS-VALIDATION LOOP SETUP ---[not using built xgboost.cv]
        tryCatch({
          if (k_final < 2) { stop(paste("CV not possible. Only", max_k, "spatial block(s) found.")) }
          scv <- cv_spatial(x = subgrid_sf, size = block_size_m, k = k_final, iteration = 200)
          if (is.null(scv) || is.null(scv$folds_list) || !is.list(scv$folds_list) || length(scv$folds_list) < k_final) {
            stop("blockCV failed to return a valid set of folds.")
          }
          early_stopping_rounds <- 10
          # Initialize empty vectors to store results from each fold
          best_nrounds_per_fold <- c()
          rmse_per_fold <- c()
          mae_per_fold <- c()
          
          for (k in 1:k_final) {
            train_idx <- unlist(scv$folds_list[[k]][1]); test_idx <- unlist(scv$folds_list[[k]][2])
            if (length(unique(subgrid_data[test_idx, response_var])) < 2) { next }
            
            # SEPERATE MODEL TRAINING & TESTING DATA FOR CV
            # turn training data into matrix to be compatible with XGboost
            dtrain_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[train_idx, predictors]), label = subgrid_data[train_idx, response_var], missing = NA)
            dtest_fold <- xgb.DMatrix(data = as.matrix(subgrid_data[test_idx, predictors]), label = subgrid_data[test_idx, response_var], missing = NA)
            watchlist_fold <- list(train = dtrain_fold, test = dtest_fold)
            
            # CV/ Fold model
            fold_model <- xgb.train(params = list
                                    (max_depth = 4,
                                      eta = 0.01, gamma = 1,
                                      objective = "reg:squarederror"),
                                    data = dtrain_fold, nrounds = 1000, # when testing, best models were achieve around 600-1000 rounds
                                    watchlist = watchlist_fold,
                                    early_stopping_rounds = early_stopping_rounds,
                                    eval_metric = c("rmse", "mae"), verbose = 0)
            
            
            if (!is.null(fold_model$best_iteration) && fold_model$best_iteration > 0) {
              best_nrounds_per_fold <- append(best_nrounds_per_fold, fold_model$best_iteration)
              rmse_per_fold <- append(rmse_per_fold, fold_model$evaluation_log$test_rmse[fold_model$best_iteration])
              mae_per_fold <- append(mae_per_fold, fold_model$evaluation_log$test_mae[fold_model$best_iteration])
            }
          }
          
          if (length(best_nrounds_per_fold) > 0) {
            best_iteration <- round(mean(best_nrounds_per_fold, na.rm = TRUE))
            cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                        test_rmse_mean = mean(rmse_per_fold, na.rm = TRUE), test_rmse_std = sd(rmse_per_fold, na.rm = TRUE),
                                        test_mae_mean = mean(mae_per_fold, na.rm = TRUE), test_mae_std = sd(mae_per_fold, na.rm = TRUE))
            cat("DIAGNOSTIC: CV successful. Optimal iteration:", best_iteration, "\n", file = worker_log_file, append = TRUE)
          } else { stop("Manual CV loop failed to find any best iterations.") }
        }, error = function(e) {
          cat("WARNING: CV SKIPPED for Tile:", tile_id, "| Pair:", pair, "with error:", conditionMessage(e), ". Using default 100 rounds.\n", file = worker_log_file, append = TRUE)
        })
        
        if (is.null(cv_results_df)) {
          cv_results_df <- data.frame(tile_id = tile_id, year_pair = pair, best_iteration = best_iteration,
                                      test_rmse_mean = NA, test_rmse_std = NA, test_mae_mean = NA, test_mae_std = NA)
        }
        
        # FINAL TRAINING MODEL
        dtrain_full <- xgb.DMatrix(data = as.matrix(subgrid_data[, predictors]), label = subgrid_data[[response_var]], missing = NA)
        xgb_params <- list(
          max_depth = 4, # Controls tree depth (higher values capture more interactions but risk overfitting). Common range: 3-10
          eta = 0.01, # Learning rate (lower values prevent overfitting, but require more trees). Common range: 0.001 - 0.3
          gamma = 1, # Minimum loss reduction required to make a further partition.
          subsample = 0.7, # Fraction of data used per boosting round (lower values prevent overfitting). Common range: 0.5-1
          colsample_bytree = 0.8, # Fraction of predictors used per tree (lower values prevent overfitting). Common range: 0.5-1
          objective = "reg:squarederror") # Specifies regression with squared error loss as the objective function.
        
        cat("DIAGNOSTIC: Starting bootstrap loop for", n.boot, "iterations...\n", file = worker_log_file, append = TRUE)
        
        # Create a DMatrix for the full tile for prediction purposes. This will be used in the loop.
        dpredict_full <- xgb.DMatrix(data = as.matrix(full_tile_data[, predictors]), missing = NA)
        
        # --- Bootstrap Loop = repeats the above over a number of desired iteration---
        for (b in seq_len(n.boot)) {
          # A true bootstrap involves training on a sample of the TRAINING data drawn with replacement.
          boot_indices <- sample(seq_len(nrow(subgrid_data)), replace = TRUE)
          dtrain_boot <- xgb.DMatrix(
            data = as.matrix(subgrid_data[boot_indices, predictors]),
            label = subgrid_data[boot_indices, response_var, drop = TRUE],
            missing = NA
          )
          
          # Train the model on the bootstrap sample.
          xgb_model <- xgb.train(
            params = xgb_params,
            data = dtrain_boot,
            nrounds = best_iteration,
            nthread = 1
          )
          
          ## ----NOTATION ON HOW CROSS VALIDATION SUPPORTS BETTER MODEL PERFORMANCE---##
          # The Cross-Validation (CV) step runs a preliminary version of the model to find the optimal number of training rounds before the model begins to overfit.
          # It repeatedly trains on subsets of the data (folds) and evaluates on a hold-out set (witheld from training),
          # tracking the performance at each iteration. We then identify the single best iteration number (`best_iteration`) where the error was lowest. 
          # This optimal number is then used to train the final model on all the data, ensuring it is powerful but not overfit.
          
          
          # -------------------------------------------------------
          # 7. STORE MODEL METRICS ----
          # -------------------------------------------------------
          # We predict on the FULL TILE DATA to get a prediction for every point.
          predictions <- predict(xgb_model, newdata = dpredict_full)
          
          # Store the full set of predictions in the matrix.
          boot_matrix[, b] <- predictions
          
          # For metrics, compare the full predictions against the full (NA-containing) response column.
          # The 'na.rm' and 'use' arguments will handle the NA's correctly, only comparing where actuals exist.
          deviance_mat[b, "Dev.Exp"] <- cor(predictions, full_tile_data[[response_var]], use = "complete.obs")^2
          deviance_mat[b, "RMSE"] <- sqrt(mean((predictions - full_tile_data[[response_var]])^2, na.rm = TRUE))
          deviance_mat[b, "R2"] <- cor(predictions, full_tile_data[[response_var]], use = "complete.obs")^2
          
          importance_matrix <- xgb.importance(model = xgb_model)
          if (nrow(importance_matrix) > 0) {
            importance_values <- setNames(importance_matrix$Gain, importance_matrix$Feature)
            matching_indices <- match(names(importance_values), rownames(influence_mat))
            valid_indices <- !is.na(matching_indices)
            influence_mat[matching_indices[valid_indices], b] <- importance_values[valid_indices]
            influence_mat[is.na(influence_mat)] <- 0
          }
          
          # -------------------------------------------------------
          # 8. STORE PARTIAL DEPENDENCE PLOT DATA ----
          # -------------------------------------------------------
          # This section correctly isolates the effect of each predictor by holding
          # all other predictors at their mean value during prediction
          
          # Initialize a list to store the PDP data for this bootstrap iteration
          PDP_Storage <- list()
          # First, calculate the mean of each predictor in the current training data.
          # These will be used to hold other predictors constant
          predictor_means <- colMeans(subgrid_data[, predictors, drop = FALSE], na.rm = TRUE)
          
          # Loop through each predictor to calculate its partial dependence
          # to calculate PDP_Value and create the PDP_Storage[[j]] data frame...
          for (j in seq_along(predictors)) {
            pred_name <- predictors[j]
            
            # Create a temporary data frame for prediction.
            # Start with a grid of 100 rows, where each column is the mean of a predictor.
            pdp_grid <- as.data.frame(matrix(rep(predictor_means, each = 100), nrow = 100))
            colnames(pdp_grid) <- names(predictor_means)
            
            # Now, overwrite the column for the current predictor with its range of values.
            pdp_grid[[pred_name]] <- EnvRanges$Env_Value[EnvRanges$Predictor == pred_name]
            
            # Ensure the column order is correct, just in case.
            pdp_grid <- pdp_grid[, predictors, drop = FALSE]
            
            # Predict using the fully constructed grid.
            # The model now sees all predictors, with only one varying
            pdp_predictions <- predict(xgb_model, newdata = as.matrix(pdp_grid))
            
            # Store the predictions in the main PD array for this bootstrap.
            PD[, j, b] <- pdp_predictions
            
            
            # Store the final, smoothed results in the long-format list
            PDP_Storage[[j]] <- data.frame(
              Predictor = pred_name,
              Env_Value = pdp_grid[[pred_name]],
              Replicate = paste0("Rep_", b),
              PDP_Value = PD[, j, b],
              # Replicating the first 100 spatial identifiers for consistency
              X = subgrid_data$X[1:100],
              Y = subgrid_data$Y[1:100],
              FID = subgrid_data$FID[1:100]
            )
          } ## End of partial dependence predictor loop
          
          # Add the results for this iteration to the master list
          all_pdp_long_list[[b]] <- bind_rows(PDP_Storage)
        }
        cat("DIAGNOSTIC: Bootstrap loop finished.\n", file = worker_log_file, append = TRUE)
        # Convert all PDP lists from the bootstraps into a single long format dataframe for plotting
        PDP_Long <- bind_rows(all_pdp_long_list)
        
        # -------------------------------------------------------
        # 8.5. PROCESS BOOTSTRAP PREDICTIONS & CALCULATE STATISTICS
        # -------------------------------------------------------
        cat("DIAGNOSTIC: Processing bootstrap results. Matrix dimensions:", paste(dim(boot_matrix), collapse = "x"), "\n", file = worker_log_file, append = TRUE)
        
        # Calculate Mean and SD from the 2D matrix across the rows.
        Mean_Prediction <- rowMeans(boot_matrix, na.rm = TRUE)
        Uncertainty_SD <- apply(boot_matrix, 1, sd, na.rm = TRUE)
        if (n.boot == 1) { Uncertainty_SD[is.na(Uncertainty_SD)] <- 0 }
        
        # Create the final data frame from the full tile data to include all points.
        boot_df <- data.table(
          FID = full_tile_data$FID,
          X = full_tile_data$X,
          Y = full_tile_data$Y,
          b.change_actual = full_tile_data[[response_var]], # This will correctly have NAs
          Mean_Prediction = Mean_Prediction,
          Uncertainty_SD = Uncertainty_SD
        )
        cat("DIAGNOSTIC: Created final bootstrap data frame with", nrow(boot_df), "rows.\n", file = worker_log_file, append = TRUE)
        
        # Explicitly remove the large matrix to free up memory
        rm(boot_matrix)
        
        # -------------------------------------------------------
        # 9. SAVE OUTPUTS----
        # -------------------------------------------------------
        tile_dir <- file.path(output_dir_train, tile_id)
        if (!dir.exists(tile_dir)) dir.create(tile_dir, recursive = TRUE)
        cat("DIAGNOSTIC: Writing outputs to", tile_dir, "\n", file = worker_log_file, append = TRUE)
        
        # Save CV results
        write_fst(as.data.table(cv_results_df), file.path(tile_dir, paste0("cv_results_", pair, ".fst")))
        # Model deviance - goodness of model fit
        write_fst(as.data.table(deviance_mat), file.path(tile_dir, paste0("deviance_", pair, ".fst")))
        # Predictor influence /contribution
        influence_df <- as.data.frame(influence_mat); influence_df$Predictor <- rownames(influence_mat)
        write_fst(as.data.table(influence_df), file.path(tile_dir, paste0("influence_", pair, ".fst")))
        write_fst(predictor_ranges, file.path(tile_dir, paste0("predictor_ranges_", pair, ".fst")))
        
        # # Model predictions from each boostrap
        write_fst(boot_df, file.path(tile_dir, paste0("bootstraps_", pair, ".fst")))
        
        # Save Rasters
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
        # Save model trained on full data with best # of rounds
        final_model <- xgb.train(params = xgb_params, data = dtrain_full, nrounds = best_iteration, nthread = 1)
        saveRDS(final_model, file.path(tile_dir, paste0("xgb_model_", pair, ".rds")))
        # Save PDP data and Env Values Together
        write_fst(PDP_Long, file.path(tile_dir, paste0("pdp_data_long_", pair, ".fst")))
        write_fst(as.data.table(EnvRanges), file.path(tile_dir, paste0("pdp_env_ranges_", pair, ".fst")))
        
        # GENERATE DIAGNOSTIC PLOT OF MODEL FIT ---
        plot_data <- boot_df[!is.na(Mean_Prediction), .(Actual = b.change_actual, Predicted = Mean_Prediction)]
        fit_plot <- ggplot(plot_data, aes(x = Actual, y = Predicted)) +
          geom_point(alpha = 0.3, color = "darkblue") +
          geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed", linewidth = 1) +
          labs(title = paste("Model Fit for Tile:", tile_id, "| Pair:", pair),
               subtitle = paste("Mean R-squared =", round(mean(deviance_mat[,"R2"], na.rm=TRUE), 3), "| Mean RMSE =", round(mean(deviance_mat[,"RMSE"], na.rm=TRUE), 3)),
               x = "Actual Change (m)", y = "Mean Predicted Change (m)") +
          theme_minimal()
        plot_dir <- file.path(output_dir_train, tile_id, "diagnostic_plots")
        if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
        ggsave(filename = file.path(plot_dir, paste0("model_fit_", pair, ".png")), plot = fit_plot, width = 7, height = 7, dpi = 150)
        cat("DIAGNOSTIC: All outputs saved successfully.\n", file = worker_log_file, append = TRUE)
        
      }, error = function(e) {
        cat("FATAL ERROR in Tile:", tile_id, "| Pair:", pair, "|", conditionMessage(e), "\n", file = worker_log_file, append = TRUE)
        cat("Backtrace:\n", paste(capture.output(traceback()), collapse="\n"), "\n", file = worker_log_file, append = TRUE)
      }) # End master tryCatch
      
      # Call garbage collection at the end of the inner loop to free memory
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
      file.remove(log) # Clean up individual worker logs
    }
  }
  
  cat("\n[SUCCESS] Model Training Complete! Check `error_log_final.txt` for any issues.\n")
  return(results_list)
}

# test function call for 4 top tiles
# --- 1. Define Parameters ---
# Use the same parameters from your last successful run
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
years <- c("2004_2006") #, "2006_2010") #, "2010_2015", "2015_2022")
block_size <- 200 # The 200m block size that worked well

# --- 2. Load and Subset Your Tile Data ---
# Load your main grid file that contains all the tile polygons and IDs
# (Assuming it's an RDS file, adjust if it's a shapefile, etc.)
all_tiles <- training_sub_grids_UTM

# Create a character vector of the specific tile IDs you want to run
specific_tile_ids <- c("BH4S556X_3") #, "BH4RZ577_3") # c("BH4RZ577_2", "BH4RZ577_3", "BH4RZ577_4", "BH4RZ578_1")

# Filter the main tile object to select ONLY the tiles in your list
test_tiles <- all_tiles %>% 
  filter(tile_id %in% specific_tile_ids)

cat("Starting test run on", nrow(test_tiles), "tiles...\n")

# --- 3. Execute the Function Call with ALL or Test Data ---
registerDoParallel(detectCores() - 1) # selects all available cores minus 1 
Sys.time()
Model_Train_Full_SpatialCV(
  training_sub_grids_UTM = test_tiles, #training_sub_grids_UTM, # training_sub_grids_UTM, # test_tiles, you can also Pass a subset of tiles HERE for testing
  output_dir_train = output_dir,
  year_pairs = years,
  block_size_m = block_size, 
  n.boot = 10, # You could reduce n.boot to 5 for an even faster test
  n.folds = 5
)
Sys.time()

showConnections()
closeAllConnections()


##### SPATIAL TEST ####
# do the outputs for the given tile look logical? 

rast <- raster(output_mask_train_utm)
mask <- rast



  b.change.2004_2006 <- rasterFromXYZ(data.frame(x = data[,"X"],  # LAT
                                         y = data[,"Y"],  # LON
                                         z = data[, "b.change.2004_2006"]), # elevation
                              crs = crs(mask))
  
  bathy_2004 <- rasterFromXYZ(data.frame(x = data[,"X"],  # LAT
                                         y = data[,"Y"],  # LON
                                         z = data[, "bathy_2004"]), # elevation
                              crs = crs(mask))
  
  bathy_2006 <- rasterFromXYZ(data.frame(x = data[,"X"],  # LAT
                                         y = data[,"Y"],  # LON
                                         z = data[, "bathy_2006"]), # elevation
                              crs = crs(mask))

mean.pred <-raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4RZ577_2/Mean_Boots_Prediction_2015_2022.tif")
plot(b.change.2004_2006)
plot(bathy_2004)
plot(bathy_2006)
#

#' Diagnose XGBoost Model and Input Data (Diagnostic and error checking tool)
#'
#' This function loads all the final outputs for a specific tile and year-pair
#' and compiles a diagnostic summary. It is designed to help troubleshoot
#' issues, particularly with the cross-validation step, by providing a
#' snapshot of the data that was fed into the model and how the model was built.
#'
#' @param output_dir_train The base directory where the tile folders are located.
#' @param tile_id The specific tile ID you want to diagnose (e.g., "BH4RZ577_2").
#' @param year_pair The specific year pair you want to diagnose (e.g., "2004_2006").
#'
#' @return A detailed list object containing diagnostic information about the
#'   data and the trained model. This list is also printed to the console.
#'

# Function
diagnose_xgb_run <- function(output_dir_train, tile_id, year_pair) {
  
  cat("--- Starting Diagnosis for Tile:", tile_id, "| Year Pair:", year_pair, "---\n\n")
  
  # -------------------------------------------------------
  # 1. DEFINE FILE PATHS
  # -------------------------------------------------------
  base_path <- file.path(output_dir_train, tile_id)
  if (!dir.exists(base_path)) {
    stop("Directory for the specified tile_id does not exist:", base_path)
  }
  
  paths <- list(
    training_data = file.path(base_path, paste0(tile_id, "_training_clipped_data.fst")),
    boruta_results = file.path(base_path, paste0("boruta_selection_", year_pair, ".rds")),
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
  
  # -------------------------------------------------------
  # 2. LOAD AND PREPARE DATA (Replicating the training script logic)
  # -------------------------------------------------------
  cat("1. Loading and preparing data...\n")
  training_data <- read_fst(paths$training_data)
  boruta_results <- readRDS(paths$boruta_results)
  predictors <- boruta_results$confirmed_predictors
  response_var <- paste0("b.change.", year_pair)
  
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

# --- Function Call ---
diagnose_xgb_run(
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles",
  tile_id = "BH4RZ577_2", # BH4RZ577_3
  year_pair = "2006_2010"
)
 
 
# Verify model outputs look good
deviance <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3/deviance_2004_2006.fst")
glimpse(deviance)
influence <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3/influence_2004_2006.fst")
glimpse(influence)
boots <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3/bootstraps_2004_2006.fst")
glimpse(boots)
envranges <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3/pdp_env_ranges_2004_2006.fst")
glimpse(envranges)
pdp <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3/pdp_data_2006_2010.fst")
glimpse(pdp)
cv <- read.fst ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3/cv_results_2006_2010.fst")
glimpse(cv)
predictor_ranges <-read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4S2574_3/predictor_ranges_2006_2010.fst")
glimpse(predictor_ranges)




##7.XGB MODEL PERFORMANCE METRICS ----


#' Create Visual Reports for XGBoost Model Performance
#'
#' This function scans a directory for XGBoost model outputs, including
#' cross-validation (`cv_results_*.fst`) and deviance (`deviance_*.fst`) files.
#' It can process all tiles or a specified subset and generates two separate
#' PNG images with summary plots using R's base plotting functions.
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
  
  # Find all potential result files recursively
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
  
  # Filter files based on the tile_ids argument
  if (!is.null(tile_ids)) {
    cat("Filtering results for", length(tile_ids), "specified tiles...\n")
    tile_pattern <- paste0("/", tile_ids, "/", collapse = "|")
    cv_files <- all_cv_files[grepl(tile_pattern, all_cv_files)]
    deviance_files <- all_deviance_files[grepl(tile_pattern, all_deviance_files)]
  } else {
    cat("Processing all available tiles...\n")
    cv_files <- all_cv_files
    deviance_files <- all_deviance_files
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
      
      # PLOT 1: Distribution of Best Iterations
      hist(cv_df$best_iteration,
           main = "Distribution of Best Training Iterations",
           xlab = "Optimal Number of Rounds",
           col = "cornflowerblue", border = "white")
      
      # PLOT 2: Average Test RMSE per Year Pair
      # --- FIX: Check for non-NA values before plotting ---
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
      
      # PLOT 3: Average Test MAE per Year Pair
      # --- FIX: Check for non-NA values before plotting ---
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
      
      # PLOT 4: RMSE vs MAE
      # --- FIX: Check for non-NA values in both columns ---
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
      df$tile_id <- str_extract(dirname(fp), "[^/\\\\]+$")
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
      
      boxplot(R2 ~ year_pair, data = deviance_df, main = "R-squared Performance by Year Pair", xlab = "Year Pair", ylab = "R-squared", col = "lightblue", notch = TRUE)
      
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
  output_dir_train = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles",
  # tile_ids = specific_tile_ids
)


#' Create and Save Partial Dependence Plots ----
#'
#' This function aggregates PDP results from model outputs and generates a
#' multi-panel plot for each year-pair. It uses a binning strategy to create
#' a smooth, interpretable summary of the PDP trends when aggregating across
#' multiple tiles.
#'
#' @param output_dir The base directory where the tile folders are located.
#' @param year_pairs A character vector of year pairs to process (e.g., "2004_2006").
#' @param tile_id (Optional) A character string for a single tile_id to process.
#'   If NULL (default), all tiles in the output directory will be processed.
#' @param plot_output_dir The directory where the final PNG plots will be saved.
#' @param n_bins The number of bins to use for summarizing the continuous predictors.
#' @param exclude_predictors (Optional) A character vector of predictor names to
#'   exclude from the plot (e.g., c("bathy_2015", "bathy_2022")).
#'
#' @return None. PNG plot files are saved to the `plot_output_dir`.
#'

# The MAGNITUDE PLOT - standardized Y-axis (scales = "free_x"). It accurately reflects the relative importance of the predictors
create_pdp_report_magnitude <- function(output_dir, year_pairs, tile_id = NULL,
                                        plot_output_dir = output_dir, n_bins = 50,
                                        exclude_predictors = NULL) {
  
  # -------------------------------------------------------
  # 1. LOAD LIBRARIES and SETUP
  # -------------------------------------------------------
  cat("Starting PDP Magnitude report generation...\n")
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
    
    # --- FIX: Use word boundaries for more precise exclusion ---
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
    
    # Determine the y-axis range for setting breaks
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
        y = "Mean Elevation Change (m)"
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


# The SHAPE PLOT -  the same plot but with scales = "free" - to understand the nuances of each predictor's effect.
create_pdp_report_shape <- function(output_dir, year_pairs, tile_id = NULL,
                              plot_output_dir = output_dir, n_bins = 50,
                              exclude_predictors = NULL) {
  
  # -------------------------------------------------------
  # 1. LOAD LIBRARIES and SETUP
  # -------------------------------------------------------
  cat("Starting PDP report generation...\n")
  library(dplyr)
  library(stringr)
  library(fst)
  library(ggplot2)
  library(tidyr)
  
  # --- Select tile folders based on the tile_id argument ---
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
  # 2. PROCESS DATA AND GENERATE PLOT FOR EACH YEAR PAIR
  # -------------------------------------------------------
  for (pair in year_pairs) {
    cat("\nProcessing year pair:", pair, "...\n")
    
    pdp_files <- unlist(sapply(tile_folders, function(tile) {
      file.path(tile, paste0("pdp_data_", pair, ".fst"))
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
    
    # --- MODIFIED: Binning logic for smooth aggregation ---
    # We process each predictor separately to handle their different value ranges.
    pdp_summary <- overall_pdp_data %>%
      filter(is.finite(Env_Value) & is.finite(PDP_Value)) %>%
      group_by(Predictor) %>%
      # Create bins for each predictor's Env_Value
      mutate(Env_Bin = cut(Env_Value, breaks = n_bins, include.lowest = TRUE)) %>%
      # Now group by the new bin to summarize
      group_by(Predictor, Env_Bin) %>%
      summarise(
        # For the plot, use the midpoint of each bin on the x-axis
        Env_Value_Mid = mean(Env_Value, na.rm = TRUE),
        PDP_Mean = mean(PDP_Value, na.rm = TRUE),
        PDP_Min  = min(PDP_Value, na.rm = TRUE),
        PDP_Max  = max(PDP_Value, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      filter(!is.na(Env_Bin)) # Remove any rows that couldn't be binned
    
    all_pdp_list[[pair]] <- pdp_summary
  }
  
  # -------------------------------------------------------
  # 3. CREATE AND SAVE PLOTS
  # -------------------------------------------------------
  for (pair in names(all_pdp_list)) {
    pdp_to_plot <- all_pdp_list[[pair]]
    
    # --- FIX: Use word boundaries for more precise exclusion ---
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
    
    # --- MODIFIED: Use the binned Env_Value_Mid for the x-axis ---
    plot_pdp <- ggplot(pdp_to_plot, aes(x = Env_Value_Mid, y = PDP_Mean)) +
      geom_ribbon(aes(ymin = PDP_Min, ymax = PDP_Max), fill = "grey70", alpha = 0.8) +
      geom_line(color = "black", linewidth = 0.8) +
      facet_wrap(~ Predictor, scales = "free", ncol = 3) +
      labs(
        x = "Model Predictor Value",
        y = "Mean Elevation Change (m)"
      ) +
      theme_minimal(base_size = 14) +
      theme(
        strip.background = element_rect(fill = "lightgray", color = "grey"),
        strip.text = element_text(face = "bold", size = 10),
        axis.text.x = element_text(angle = 45, hjust = 1)
      )
    
    output_filename <- file.path(plot_output_dir, paste0("Overall_PDP_Report_Shape", pair, ".png"))
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
                        # tile_id = "BH4RZ577_2",
                       year_pairs = years)

create_pdp_report_magnitude (output_dir = output_dir, # magnitude relative to bathymetry 
                         # tile_id = "BH4RZ577_2",
                         year_pairs = years)





# # ## DOUBLE CHECK IT SPATIALLY!----


bathy.2022 <- rasterFromXYZ(data.frame(x = data[,"X"],  # LAT
                                            y = data[,"Y"],  # LON
                                            z = data[, "bathy_2022"]), # elevation
                                            crs = crs(mask))


mean.pred <-raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4RZ577_2/Mean_Boots_Prediction_2015_2022.tif")
plot(mean.pred)
data <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles/BH4RZ577_2/BH4RZ577_2_training_clipped_data.fst")
plot(bathy.2022)



### NOTE
# - what we really need is to re-run the training model to make a more comprehensive boostrap database with environmental info and ensure the FIDs are consistent across all
#- data, which i think they are 


# PREDICTION FUNCTION SET - V2 - GLOBAL OPTION
# ============================================================================
#   GLOBAL LUT VERSION (Hard Bin + Optional Soft Kernel)
#   Drop-in replacement for apply_global_bootstrap_adjustment/apply_global_knn_adjustment
# ============================================================================

# --- Build Global Bootstrap Summaries (per year_pair) ---
# --- Build Global Bootstrap Summaries with Optional Standardization ---
# --- Build Global Bootstrap Summaries (with optional standardization) ---
build_global_bootstrap_kmeans <- function(global_db_manifest, year_pair,
                                          env_vars   = c("starting_bathy", "starting_slope", "starting_rugosity"),
                                          k_clusters = 200,
                                          standardize = TRUE,
                                          out_dir    = tempdir()) {
  manifest_year <- global_db_manifest[year_pair_col == year_pair]
  if (nrow(manifest_year) == 0) stop("No items in manifest for year pair: ", year_pair)
  
  big_list <- vector("list", nrow(manifest_year))
  for (i in seq_len(nrow(manifest_year))) {
    trn <- read_fst(manifest_year$training_path[i], as.data.table = TRUE)
    bt  <- read_fst(manifest_year$boots_path[i], as.data.table = TRUE)
    keep_cols <- intersect(names(bt), c("FID", "Mean_Prediction", "Uncertainty_SD", "b.change_actual"))
    if (!all(c("FID", "Mean_Prediction") %in% keep_cols)) next
    m <- merge(trn, bt[, ..keep_cols], by = "FID", allow.cartesian = FALSE)
    m <- align_predictors(m, year_pair)
    want_cols <- c(env_vars, "Mean_Prediction", "b.change_actual")
    have_cols <- intersect(want_cols, names(m))
    m <- m[, ..have_cols]
    if ("b.change_actual" %in% names(m)) setnames(m, "b.change_actual", "b_change", skip_absent = TRUE)
    big_list[[i]] <- m
  }
  all_rows <- rbindlist(big_list, use.names = TRUE, fill = TRUE)
  if (nrow(all_rows) == 0) stop("Global bootstrap source empty for ", year_pair)
  
  # Standardize
  mu <- sapply(all_rows[, ..env_vars], mean, na.rm = TRUE)
  sdv <- sapply(all_rows[, ..env_vars], sd, na.rm = TRUE)
  if (standardize) {
    X <- scale(all_rows[, ..env_vars])
  } else {
    X <- as.matrix(all_rows[, ..env_vars])
  }
  
  # K-means clustering
  km <- kmeans(X, centers = k_clusters, nstart = 10)
  all_rows[, cluster_id := km$cluster]
  
  summaries <- all_rows[, .(
    n             = .N,
    mean_boot     = mean(Mean_Prediction, na.rm = TRUE),
    sd_boot       = sd(Mean_Prediction, na.rm = TRUE),
    mean_b_change = mean(b_change, na.rm = TRUE),
    sd_b_change   = sd(b_change, na.rm = TRUE),
    center        = list(colMeans(X[cluster_id == .BY$cluster_id, , drop = FALSE], na.rm = TRUE))
  ), by = cluster_id]
  
  out_obj <- list(
    year_pair = year_pair,
    env_vars  = env_vars,
    mu        = mu,
    sdv       = sdv,
    summaries = summaries
  )
  saveRDS(out_obj, file.path(out_dir, paste0("global_boot_kmeans_", year_pair, ".rds")))
  return(out_obj)
}


# --- Prediction ---
predict_from_global_LUT <- function(prediction_data, lut_obj,
                                    which_mean = c("boot", "b_change"),
                                    kernel = FALSE, bw = 1.0) {
  which_mean <- match.arg(which_mean)
  env_vars   <- lut_obj$raw_env_vars
  std_env_vars <- lut_obj$std_env_vars
  mu <- lut_obj$mu
  sdv <- lut_obj$sdv
  breaks_list <- lut_obj$breaks_list
  summaries   <- copy(lut_obj$summaries)
  setDT(prediction_data)
  
  # Standardize prediction vars
  for (v in env_vars) {
    prediction_data[, (paste0(v, "_std")) := (get(v) - mu[v]) / sdv[v]]
  }
  
  # Bin assignment
  for (j in seq_along(std_env_vars)) {
    brks <- breaks_list[[std_env_vars[j]]]
    prediction_data[, paste0(std_env_vars[j], "_bin") :=
                      cut(.SD[[std_env_vars[j]]], breaks = brks,
                          include.lowest = TRUE, labels = FALSE)]
  }
  bin_cols <- paste0(std_env_vars, "_bin")
  
  if (!kernel) {
    # --- Hard Bin Mode ---
    setkeyv(summaries, bin_cols)
    setkeyv(prediction_data, bin_cols)
    joined <- summaries[prediction_data]
    if (which_mean == "boot") {
      prediction_data[, Global_Boots_predicted_change := joined$mean_boot]
    } else {
      prediction_data[, Global_KNN_predicted_change := joined$mean_b_change]
    }
  } else {
    # --- Soft Kernel Mode ---
    centers_mat <- do.call(rbind, lapply(summaries$centers, unlist))
    for (i in seq_len(nrow(prediction_data))) {
      obs <- as.numeric(prediction_data[i, ..std_env_vars])
      dists <- sqrt(rowSums((centers_mat - matrix(obs, nrow = nrow(centers_mat),
                                                  ncol = length(obs), byrow = TRUE))^2))
      weights <- exp(-0.5 * (dists / bw)^2)
      weights <- weights / sum(weights)
      if (which_mean == "boot") {
        prediction_data[i, Global_Boots_predicted_change := sum(summaries$mean_boot * weights, na.rm = TRUE)]
      } else {
        prediction_data[i, Global_KNN_predicted_change := sum(summaries$mean_b_change * weights, na.rm = TRUE)]
      }
    }
  }
  
  return(prediction_data)
}






#
#           XGBoost Prediction Function Set (Production Version)
#
# ==============================================================================

# This script contains a complete, refactored set of functions to generate
# predictions from the trained XGBoost models. It is designed to:
#   1. Use a direct model if available for a tile.
#   2. For prediction-only tiles, find the nearest training tile that has a valid model for the specific year-pair.
#   3. Intelligently map static prediction predictors (e.g., bt.bathy, bt.rugosity) to all temporal inputs the model expects.
#   4. Implement a clear, four-pronged hybrid adjustment workflow (Local/Global Bootstrap, Local/Global KNN).
#   5. Save final predictions and all component predictions as rasters.
#   6. Run robustly in a parallel environment using a memory-efficient "Index and Query" approach for global methods.
#   7. Automatically generate comprehensive diagnostic plots and logs.

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis) # For advanced plotting
library(gridExtra) # For arranging plots
library(ggplot2)   # For density plots
library(FNN)
library(foreach)
library(doParallel)
library(Metrics)   # For error metrics (MAE, RMSE)
library(cmocean)   # For better color palettes

# ==============================================================================
#   MAIN WORKFLOW FUNCTIONS
# ==============================================================================

# --- 1. Main Orchestration Function ---
process_tile <- function(tile_id, year_pair, training_dir, prediction_dir, output_dir, all_tile_maps, global_db_manifest, lut_map) {
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
    model_tile_id <- if (tile_id %in% names(tile_map_for_year)) tile_map_for_year[[tile_id]] else tile_id
    processing_mode <- if (model_tile_id == tile_id) "direct_model" else "reference_model"
    
    cat("  - INFO: Processing mode detected:", processing_mode, "\n")
    if (processing_mode == "reference_model") cat("  - INFO: Using reference model from tile:", model_tile_id, "\n")
    
    # --- c. Generate Initial Predictions ---
    cat("Step 1: Generating initial XGBoost predictions...\n")
    initial_predictions_dt <- predict_elevation_change(tile_id, model_tile_id, year_pair, training_dir, prediction_dir)
    
    # --- d. Load Local Data for Adjustments ---
    cat("Step 2: Loading local data for hybrid adjustments from source:", model_tile_id, "\n")
    pdp_file <- file.path(training_dir, model_tile_id, paste0("pdp_data_long_", year_pair, ".fst"))
    training_data_path <- file.path(training_dir, model_tile_id, paste0(model_tile_id, "_training_clipped_data.fst"))
    
    pdp_data <- read_fst(pdp_file, as.data.table = TRUE)
    local_training_data <- read_fst(training_data_path, as.data.table = TRUE)
    
    boruta_path <- file.path(training_dir, model_tile_id, paste0("boruta_selection_", year_pair, ".rds"))
    boruta_results <- readRDS(boruta_path)
    all_predictors <- boruta_results$confirmed_predictors
    
    aligned_local_training_data <- align_predictors(local_training_data, year_pair, all_predictors)
    
    # --- e. Apply All Adjustment Methods ---
    cat("Step 3: Applying all local and global adjustment methods...\n")
    
    # --- Local Adjustments ---
    local_boot_enriched <- apply_bootstrap_adjustment(initial_predictions_dt, model_tile_id, year_pair, training_dir)
    local_knn_enriched  <- apply_local_knn_adjustment(local_boot_enriched, aligned_local_training_data)
    
    # --- Global Adjustments (via LUTs) ---
    lut_obj <- lut_map[[year_pair]]
    global_boot_enriched <- predict_from_global_LUT(local_knn_enriched, lut_obj, which_mean = "boot")
    global_knn_enriched  <- predict_from_global_LUT(global_boot_enriched, lut_obj, which_mean = "b_change")
    
    # --- PDP Adjustment (Always Local) ---
    pdp_enriched <- match_pdp_conditions(global_knn_enriched, pdp_data, year_pair)
    
    cat("Step 4: Comparing prediction methods and combining results...\n")
    comparison_results <- compare_prediction_methods(pdp_enriched, tile_id, year_pair, processing_mode = processing_mode)
    
    return(list(
      data = comparison_results$data,
      tile_id = tile_id,
      year_pair = year_pair,
      success = TRUE
    ))
    
  }, error = function(e) {
    cat("\n--- FATAL ERROR in process_tile ---\n")
    cat("  - Tile:", tile_id, "| Pair:", year_pair, "\n")
    cat("  - Error Message:", conditionMessage(e), "\n")
    cat("  - Traceback:\n", paste(capture.output(traceback()), collapse="\n"), "\n")
    return(list(success = FALSE))
  })
}


# ==============================================================================
#   HELPER FUNCTIONS (DATA PROCESSING)
# ==============================================================================

# --- 2. Core Prediction Function ---
predict_elevation_change <- function(tile_id, model_tile_id, year_pair, training_dir, prediction_dir) {
  model_path <- file.path(training_dir, model_tile_id, paste0("xgb_model_", year_pair, ".rds"))
  boruta_path <- file.path(training_dir, model_tile_id, paste0("boruta_selection_", year_pair, ".rds"))
  prediction_data_path <- file.path(prediction_dir, tile_id, paste0(tile_id, "_prediction_clipped_data.fst"))
  
  xgb_model <- readRDS(model_path)
  boruta_results <- readRDS(boruta_path)
  predictors_from_boruta <- boruta_results$confirmed_predictors
  
  prediction_data <- read_fst(prediction_data_path, as.data.table = TRUE)
  if (!"tile_id" %in% names(prediction_data)) prediction_data[, tile_id := tile_id]
  
  aligned_pred_data <- align_predictors(prediction_data, year_pair, predictors_from_boruta)
  
  pred_matrix <- as.matrix(aligned_pred_data[, ..predictors_from_boruta])
  aligned_pred_data[, XGB_predicted_change := predict(xgb_model, newdata = pred_matrix)]
  
  return(aligned_pred_data)
}

# --- 3. Align Predictors ---
align_predictors <- function(data_to_align, year_pair, predictors_from_boruta = NULL) {
  setDT(data_to_align)
  start_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][1])
  
  if (!is.null(predictors_from_boruta)) {
    # Handle bathy
    bathy_req <- predictors_from_boruta[startsWith(predictors_from_boruta, "bathy_")]
    if (length(bathy_req) > 0 && "bt.bathy" %in% names(data_to_align)) {
      for (col in bathy_req) if (!col %in% names(data_to_align)) data_to_align[, (col) := bt.bathy]
    }
    # Handle slope
    slope_req <- predictors_from_boruta[startsWith(predictors_from_boruta, "slope_")]
    if (length(slope_req) > 0 && "bt.slope" %in% names(data_to_align)) {
      for (col in slope_req) if (!col %in% names(data_to_align)) data_to_align[, (col) := bt.slope]
    }
    # Handle rugosity
    rugo_req <- predictors_from_boruta[startsWith(predictors_from_boruta, "rugosity_")]
    if (length(rugo_req) > 0 && "bt.bathy.Rugosity" %in% names(data_to_align)) {
      for (col in rugo_req) {
        if (!col %in% names(data_to_align)) {
          data_to_align[, (col) := bt.bathy.Rugosity]
        }
      }
    }
  }
  
  create_starting_col <- function(dt, prefix, year, static_col) {
    temporal_col <- paste0(prefix, "_", year)
    starting_col_name <- paste0("starting_", prefix)
    if (temporal_col %in% names(dt)) dt[, (starting_col_name) := .SD[[temporal_col]]]
    else if (static_col %in% names(dt)) dt[, (starting_col_name) := .SD[[static_col]]]
  }
  
  create_starting_col(data_to_align, "bathy", start_year, "bt.bathy")
  create_starting_col(data_to_align, "slope", start_year, "bt.slope")
  
  rugo_start_name <- paste0("rugosity_", start_year)
  if (rugo_start_name %in% names(data_to_align)) data_to_align[, starting_rugosity := .SD[[rugo_start_name]]]
  else if ("bt.bathy.Rugosity" %in% names(data_to_align)) data_to_align[, starting_rugosity := bt.bathy.Rugosity]
  
  response_var_original <- paste0("b.change.", year_pair)
  if(response_var_original %in% names(data_to_align)) setnames(data_to_align, old = response_var_original, new = "b_change")
  
  setDT(data_to_align)   # <--- force return type
  return(data_to_align)
}


# --- 4. Local Bootstrap Adjustment ---
apply_bootstrap_adjustment <- function(prediction_data, model_tile_id, year_pair, training_dir) {
  boots_path <- file.path(training_dir, model_tile_id, paste0("bootstraps_", year_pair, ".fst"))
  boots_data <- read_fst(boots_path, as.data.table = TRUE)
  boots_to_merge <- boots_data[, .(FID, Mean_Prediction, Uncertainty_SD, b.change_actual)]
  
  prediction_data[, temp_order := .I]
  merged_data <- merge(prediction_data, boots_to_merge, by = "FID", all.x = TRUE)
  setorderv(merged_data, "temp_order")[, temp_order := NULL]
  
  setnames(merged_data, "Mean_Prediction", "MeanBoots_predicted_change", skip_absent=TRUE)
  setnames(merged_data, "Uncertainty_SD", "MeanBoots_SD", skip_absent=TRUE)
  
  return(merged_data)
}

# --- 5. Local KNN Adjustment ---
apply_local_knn_adjustment <- function(prediction_data, training_data, k = 15) {
  env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity")
  pred_clean <- prediction_data[complete.cases(prediction_data[, ..env_vars])]
  train_clean <- training_data[complete.cases(training_data[, ..env_vars])]
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[["b_change"]]
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  weights <- 1 / (knn_result$nn.dist + 1e-6)
  pred_avg <- rowSums(matrix(train_resp[knn_result$nn.index], nrow = nrow(knn_result$nn.index)) * weights, na.rm = TRUE) / rowSums(weights, na.rm = TRUE)
  
  pred_clean[, Local_KNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), Local_KNN_predicted_change := i.Local_KNN_predicted_change]
  
  return(prediction_data)
}

# --- 6. Global Bootstrap Adjustment (SCALABLE & ROBUST) ---
apply_global_bootstrap_adjustment <- function(prediction_data, global_db_manifest, year_pair, k = 25) {
  message("  - INFO: Starting Scalable Global Bootstrap Adjustment...")
  env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity")
  
  manifest_year <- global_db_manifest[year_pair_col == year_pair]
  
  find_global_boot_neighbors <- function(point_data) {
    top_neighbors <- data.table(dist = rep(Inf, k), val = rep(NA_real_, k))
    
    for (i in 1:nrow(manifest_year)) {
      training_tile_data <- read_fst(manifest_year$training_path[i], as.data.table = TRUE)
      boots_tile_data <- read_fst(manifest_year$boots_path[i], as.data.table = TRUE)
      
      if (!"Mean_Prediction" %in% names(boots_tile_data)) {
        cat("  - WARNING: 'Mean_Prediction' not found in", manifest_year$boots_path[i], ". Skipping.\n")
        next
      }
      
      merged_tile <- merge(training_tile_data, boots_tile_data[, .(FID, Mean_Prediction)], by = "FID")
      aligned_tile <- align_predictors(merged_tile, year_pair)
      clean_tile <- aligned_tile[complete.cases(aligned_tile[, ..env_vars])]
      
      if (nrow(clean_tile) > 0) {
        knn_result <- FNN::get.knnx(as.matrix(clean_tile[, ..env_vars]), as.matrix(point_data[, ..env_vars]), k = k)
        current_best <- data.table(dist = as.vector(knn_result$nn.dist), val = as.vector(clean_tile$Mean_Prediction[knn_result$nn.index]))
        combined <- rbindlist(list(top_neighbors, current_best))
        setorderv(combined, "dist")
        top_neighbors <- combined[1:k, ]
      }
      rm(training_tile_data, boots_tile_data, merged_tile, aligned_tile, clean_tile); gc()
    }
    
    weights <- 1 / (top_neighbors$dist + 1e-6)
    weighted_mean <- sum(top_neighbors$val * weights, na.rm = TRUE) / sum(weights, na.rm = TRUE)
    return(weighted_mean)
  }
  
  prediction_data[, Global_Boots_predicted_change := sapply(1:.N, function(i) find_global_boot_neighbors(.SD[i]))]
  
  message("  - INFO: Completed Scalable Global Bootstrap Adjustment.")
  return(prediction_data)
}

# --- 7. Global KNN Adjustment (SCALABLE & ROBUST) ---
apply_global_knn_adjustment <- function(prediction_data, global_db_manifest, year_pair, k = 25) {
  message("  - INFO: Starting Scalable Global KNN Adjustment...")
  env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity")
  manifest_year <- global_db_manifest[year_pair_col == year_pair]
  
  find_global_knn_neighbors <- function(point_data) {
    top_neighbors <- data.table(dist = rep(Inf, k), val = rep(NA_real_, k))
    
    for (i in 1:nrow(manifest_year)) {
      training_tile_data <- read_fst(manifest_year$training_path[i], as.data.table = TRUE)
      boots_tile_data <- read_fst(manifest_year$boots_path[i], as.data.table = TRUE)
      
      if (!"b.change_actual" %in% names(boots_tile_data)) {
        cat("  - WARNING: 'b.change_actual' not found in", manifest_year$boots_path[i], ". Skipping.\n")
        next
      }
      
      merged_tile <- merge(training_tile_data, boots_tile_data[, .(FID, b.change_actual)], by = "FID")
      setnames(merged_tile, "b.change_actual", "b_change")
      aligned_tile <- align_predictors(merged_tile, year_pair)
      clean_tile <- aligned_tile[complete.cases(aligned_tile[, c(env_vars, "b_change")])]
      
      if (nrow(clean_tile) > 0) {
        knn_result <- FNN::get.knnx(as.matrix(clean_tile[, ..env_vars]), as.matrix(point_data[, ..env_vars]), k = k)
        current_best <- data.table(dist = as.vector(knn_result$nn.dist), val = as.vector(clean_tile$b_change[knn_result$nn.index]))
        combined <- rbindlist(list(top_neighbors, current_best))
        setorderv(combined, "dist")
        top_neighbors <- combined[1:k, ]
      }
      rm(training_tile_data, boots_tile_data, merged_tile, aligned_tile, clean_tile); gc()
    }
    
    weights <- 1 / (top_neighbors$dist + 1e-6)
    weighted_mean <- sum(top_neighbors$val * weights, na.rm = TRUE) / sum(weights, na.rm = TRUE)
    return(weighted_mean)
  }
  
  prediction_data[, Global_KNN_predicted_change := sapply(1:.N, function(i) find_global_knn_neighbors(.SD[i]))]
  
  message("  - INFO: Completed Scalable Global KNN Adjustment.")
  return(prediction_data)
}

# --- 8. Match PDP Conditions ---
match_pdp_conditions <- function(prediction_data, pdp_data, year_pair) {
  setDT(prediction_data); setDT(pdp_data)
  env_ranges <- extract_env_variable_ranges(pdp_data)
  all_pdp_predictors <- unique(pdp_data$Predictor)
  
  pdp_match_vars <- list(
    list(pred_pattern = "bathy_",      start_col = "starting_bathy",      weight = 0.3),
    list(pred_pattern = "slope_",      start_col = "starting_slope",      weight = 0.1),
    list(pred_pattern = "rugosity_",   start_col = "starting_rugosity",   weight = 0.1)
  )
  
  for (i in seq_along(pdp_match_vars)) {
    full_pred_name <- grep(pdp_match_vars[[i]]$pred_pattern, all_pdp_predictors, value = TRUE)[1]
    pdp_match_vars[[i]]$full_pred_name <- if(length(full_pred_name) > 0) full_pred_name else NA_character_
    range_val <- if(!is.na(full_pred_name)) env_ranges[Predictor == full_pred_name, range_width] else NA_real_
    pdp_match_vars[[i]]$range_width <- if(length(range_val) > 0) range_val else NA_real_
  }
  
  prediction_data[, pdp_adjusted_change := {
    current_point <- .SD
    is_match <- rep(FALSE, nrow(pdp_data))
    for (var in pdp_match_vars) {
      if (!is.na(var$full_pred_name) && !is.na(var$range_width) && var$start_col %in% names(current_point)) {
        current_val <- current_point[[var$start_col]]
        if (!is.na(current_val)) {
          half_window <- var$weight * var$range_width
          lower_bound <- current_val - half_window
          upper_bound <- current_val + half_window
          is_match <- is_match | (pdp_data$Predictor == var$full_pred_name & pdp_data$Env_Value >= lower_bound & pdp_data$Env_Value <= upper_bound)
        }
      }
    }
    matches <- pdp_data[is_match]
    if (nrow(matches) > 0) mean(matches$PDP_Value, na.rm = TRUE) else NA_real_
  }, by = 1:nrow(prediction_data)]
  
  return(prediction_data)
}

# --- 9. Compare Methods ---
compare_prediction_methods <- function(prediction_data, tile_id, year_pair, processing_mode) {
  prediction_data[, hybrid_change := rowMeans(.SD, na.rm = TRUE), 
                  .SDcols = c("MeanBoots_predicted_change", "pdp_adjusted_change", "Global_KNN_predicted_change", "Global_Boots_predicted_change")]
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  return(list(data = prediction_data))
}

# ==============================================================================
#   HELPER FUNCTIONS (FILE I/O & PLOTTING)
# ==============================================================================
extract_env_variable_ranges <- function(pdp_data) {
  if (is.null(pdp_data) || nrow(pdp_data) == 0) {
    return(data.table(Predictor = character(), range_width = numeric()))
  }
  pdp_data[, .(range_width = max(Env_Value, na.rm = TRUE) - min(Env_Value, na.rm = TRUE)), by = Predictor]
}

save_final_predictions <- function(prediction_data, output_dir, tile_id, year_pair) {
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  pred_depth_col <- paste0("pred_", end_year, "_depth")
  
  prediction_data[, (pred_depth_col) := starting_bathy + hybrid_change]
  
  out_file_dir <- file.path(output_dir, tile_id); if (!dir.exists(out_file_dir)) dir.create(out_file_dir, recursive = TRUE)
  out_file <- file.path(out_file_dir, paste0(tile_id, "_prediction_final_", year_pair, ".fst"))
  
  write_fst(prediction_data, out_file)
  return(prediction_data)
}

save_component_rasters <- function(prediction_data, output_dir, tile_id, year_pair, crs_obj) {
  out_raster_dir <- file.path(output_dir, tile_id)
  
  save_raster <- function(data, col_name, file_suffix) {
    if (col_name %in% names(data) && any(!is.na(data[[col_name]]))) {
      raster_df <- data[!is.na(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(raster_df) > 0) {
        r <- raster::rasterFromXYZ(raster_df, crs = crs_obj)
        out_path <- file.path(out_raster_dir, paste0(tile_id, "_", file_suffix, "_", year_pair, ".tif"))
        raster::writeRaster(r, out_path, format = "GTiff", overwrite = TRUE)
      }
    }
  }
  
  save_raster(prediction_data, "hybrid_change", "Hybrid_predicted_change")
  save_raster(prediction_data, "Global_KNN_predicted_change", "Global_KNN_predicted_change")
  save_raster(prediction_data, "Local_KNN_predicted_change", "Local_KNN_predicted_change")
  save_raster(prediction_data, "Global_Boots_predicted_change", "Global_Boots_predicted_change")
  save_raster(prediction_data, "MeanBoots_predicted_change", "MeanBoots_predicted_change")
}

build_all_tile_maps <- function(training_dir, prediction_dir, year_pairs) {
  training_footprint_gpkg <- file.path(training_dir, "subgrid_footprints_training.gpkg")
  prediction_footprint_gpkg <- file.path(prediction_dir, "subgrid_footprints_prediction.gpkg")
  
  train_sf_full <- sf::st_read(training_footprint_gpkg, quiet = TRUE)
  pred_sf <- sf::st_read(prediction_footprint_gpkg, quiet = TRUE)
  id_col <- names(train_sf_full)[grepl("tile_id|name|id", names(train_sf_full), ignore.case=TRUE)][1]
  
  suppressWarnings({ pred_centroids <- sf::st_centroid(pred_sf) })
  
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
    suppressWarnings({ train_centroids_year_specific <- sf::st_centroid(train_sf_year_specific) })
    nearest_indices <- sf::st_nearest_feature(pred_centroids, train_centroids_year_specific)
    tile_map <- train_sf_year_specific[[id_col]][nearest_indices]
    names(tile_map) <- pred_sf[[id_col]]
    all_maps[[as.character(yp)]] <- tile_map
  }
  return(all_maps)
}

save_global_comparison_plot <- function(processed_data, tile_id, year_pair, prediction_dir, crs_obj) {
  plot_cols <- c("MeanBoots_predicted_change", "Global_Boots_predicted_change", 
                 "Local_KNN_predicted_change", "Global_KNN_predicted_change")
  plot_titles <- c("Local Boots", "Global Boots", "Local KNN", "Global KNN")
  
  zlim <- range(processed_data[, ..plot_cols], na.rm = TRUE)
  at <- seq(zlim[1], zlim[2], length.out = 99)
  plot_colors <- cmocean('deep')(100)
  
  plot_list <- list()
  for(i in 1:length(plot_cols)) {
    col <- plot_cols[i]
    title <- plot_titles[i]
    if (col %in% names(processed_data) && any(!is.na(processed_data[[col]]))) {
      df <- processed_data[!is.na(get(col)), .(x = X, y = Y, z = get(col))]
      if(nrow(df) > 0) {
        r <- raster::rasterFromXYZ(df, crs = crs_obj)
        plot_list[[col]] <- levelplot(r, main=title, margin=FALSE, at=at, col.regions=plot_colors)
      } else {
        plot_list[[col]] <- ggplot() + theme_void() + ggtitle(title)
      }
    } else {
      plot_list[[col]] <- ggplot() + theme_void() + ggtitle(title)
    }
  }
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_global_comparison_plot_", year_pair, ".pdf"))
  
  pdf(plot_out_file, width = 12, height = 12)
  grid.arrange(grobs = plot_list, ncol = 2, nrow = 2)
  dev.off()
}


# ==============================================================================
#           Parallel Prediction Test Wrapper (Updated for Scalability)
# ==============================================================================

run_prediction_test <- function(prediction_tiles, year_pairs, crs_obj,
                                training_dir, prediction_dir, output_dir) {
  
  message("\nStarting SCALABLE parallel prediction test run...")
  
  # --- 1. Build File Manifest ---
  message("  - INFO: Building file manifest from training directory...")
  all_tile_ids <- list.dirs(training_dir, full.names = FALSE, recursive = FALSE)
  all_tile_ids <- all_tile_ids[!grepl("footprints|diagnostics", all_tile_ids)]
  
  manifest_list <- list()
  for (tile in all_tile_ids) {
    for (yp in year_pairs) {
      training_path <- file.path(training_dir, tile, paste0(tile, "_training_clipped_data.fst"))
      boots_path    <- file.path(training_dir, tile, paste0("bootstraps_", yp, ".fst"))
      if (file.exists(training_path) && file.exists(boots_path)) {
        manifest_list[[length(manifest_list) + 1]] <- data.table(
          tile_id_col   = tile,
          year_pair_col = yp,
          training_path = training_path,
          boots_path    = boots_path
        )
      }
    }
  }
  global_db_manifest <- rbindlist(manifest_list)
  message("  - INFO: Manifest created with ", nrow(global_db_manifest), " valid file pairs.")
  
  # --- 2. Build LUTs for Global Adjustments ---
  message(" - INFO: Building global LUTs (once per year_pair)...")
  lut_map <- list()
  for (yp in year_pairs) {
    lut_map[[yp]] <- build_global_bootstrap_summaries(
      global_db_manifest = global_db_manifest,
      year_pair          = yp,
      env_vars           = c("starting_bathy", "starting_slope", "starting_rugosity"),
      n_bins             = c(24, 16, 16),
      out_dir            = output_dir
    )
  }
  
  # --- 3. Build Geographic Tile Maps ---
  all_tile_maps <- build_all_tile_maps(training_dir, prediction_dir, year_pairs)
  
  # --- 4. Setup Parallel Processing ---
  num_cores <- detectCores() - 1; if (num_cores < 1) num_cores <- 1
  cl <- makeCluster(num_cores)
  registerDoParallel(cl)
  on.exit({ stopCluster(cl) })
  
  task_grid <- expand.grid(tile_id = prediction_tiles, year_pair = year_pairs, stringsAsFactors = FALSE)
  
  results_list <- foreach(
    i = 1:nrow(task_grid),
    .export = c("process_tile", "predict_elevation_change", "align_predictors",
                "apply_bootstrap_adjustment", "apply_local_knn_adjustment",
                "predict_from_global_LUT", "match_pdp_conditions",
                "compare_prediction_methods", "extract_env_variable_ranges")
  ) %dopar% {
    # Load libraries inside worker
    library(data.table); library(fst); library(dplyr); library(FNN); library(raster)
    
    current_task <- task_grid[i, ]
    process_tile(
      tile_id        = current_task$tile_id,
      year_pair      = current_task$year_pair,
      training_dir   = training_dir,
      prediction_dir = prediction_dir,
      output_dir     = output_dir,
      all_tile_maps  = all_tile_maps,
      global_db_manifest = global_db_manifest,
      lut_map        = lut_map
    )
  }
  
  # --- 5. Save Outputs Sequentially ---
  message("\n--- Stage 2: Saving results and plots sequentially... ---")
  successful_results <- Filter(function(res) !is.null(res) && is.list(res) && res$success, results_list)
  
  for (result in successful_results) {
    cat("\n--- Saving outputs for Tile:", result$tile_id, "| Year Pair:", result$year_pair, "---\n")
    processed_data <- save_final_predictions(result$data, output_dir, result$tile_id, result$year_pair)
    if (!is.null(crs_obj)) {
      save_component_rasters(processed_data, output_dir, result$tile_id, result$year_pair, crs_obj)
      save_global_comparison_plot(processed_data, result$tile_id, result$year_pair, prediction_dir, crs_obj)
    }
  }
  
  message("\n✅ Scalable prediction test run complete.")
}

# Define the directories and parameters
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
years <- c("2004_2006")
mask <- raster::raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif")
mask_crs <- raster::crs(mask)
crs_obj <- mask_crs
prediction_tile_ids <- c("BH4S556X_3")



showConnections()
closeAllConnections()
gc()




# PREDICTION FUNTION SET code V1 ----

# --- NON GLOBAL SOLUTION ---

# Geminini update PREDICTION CODE ----
#
#           XGBoost Prediction Function Set (Production Version)
#
# ==============================================================================

# This script contains a complete, refactored set of functions to generate
# predictions from the trained XGBoost models. It is designed to:
#   1. Use a direct model if available for a tile.
#   2. For prediction-only tiles, find the nearest training tile that has a valid model for the specific year-pair.
#   3. Intelligently map static prediction predictors (e.g., bt.bathy, bt.rugosity) to all temporal inputs the model expects.
#   4. Implement a clear, three-pronged hybrid adjustment workflow (Bootstrap Mean, PDP, and KNN).
#   5. Save final predictions and all component predictions as rasters.
#   6. Run robustly in a parallel environment with memory-safe sequential file I/O.
#   7. Automatically generate comprehensive diagnostic plots (summary, density, cross-validation) and logs.

# --- Load All Necessary Libraries ---
library(data.table)
library(dplyr)
library(fst)
library(sf)
library(xgboost)
library(raster)
library(rasterVis) # For advanced plotting
library(gridExtra) # For arranging plots
library(ggplot2)   # For density plots
library(FNN)
library(foreach)
library(doParallel)
library(Metrics)   # For error metrics (MAE, RMSE)
library(cmocean)   # For better color palettes

# ==============================================================================
#   MAIN WORKFLOW FUNCTIONS
# ==============================================================================

# --- 1. Main Orchestration Function ---
# Detects processing mode (Direct vs. Reference) and orchestrates the workflow.
# ------------------------------------------------------------------------------
process_tile <- function(tile_id, year_pair, training_dir, prediction_dir, output_dir, all_tile_maps) {
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
    
    # --- c. Generate Initial Predictions using appropriate model ---
    cat("Step 1: Generating initial XGBoost predictions...\n")
    initial_predictions_dt <- predict_elevation_change(tile_id, model_tile_id, year_pair, training_dir, prediction_dir)
    if (is.null(initial_predictions_dt)) stop("Initial prediction failed.")
    
    # --- d. Load Data for Hybrid Adjustments (always from model_tile_id) ---
    cat("Step 2: Loading data for hybrid adjustments from source:", model_tile_id, "\n")
    pdp_file <- file.path(training_dir, model_tile_id, paste0("pdp_data_long_", year_pair, ".fst")) # Using the long format
    training_data_path <- file.path(training_dir, model_tile_id, paste0(model_tile_id, "_training_clipped_data.fst"))
    
    if (!file.exists(pdp_file)) stop("Missing required PDP file for model source tile: ", model_tile_id)
    if (!file.exists(training_data_path)) stop("Missing required Training Data for model source tile: ", model_tile_id)
    
    pdp_data <- read_fst(pdp_file, as.data.table = TRUE)
    training_data <- read_fst(training_data_path, as.data.table = TRUE)
    
    boruta_path <- file.path(training_dir, model_tile_id, paste0("boruta_selection_", year_pair, ".rds"))
    boruta_results <- readRDS(boruta_path)
    all_predictors <- boruta_results$confirmed_predictors
    
    aligned_training_data <- align_predictors(training_data, year_pair, all_predictors)
    
    # --- e. Apply Hybrid Adjustments ---
    cat("Step 3: Applying hybrid adjustments (Bootstrap, PDP, KNN)...\n")
    boot_enriched <- apply_bootstrap_adjustment(initial_predictions_dt, model_tile_id, year_pair, training_dir)
    pdp_enriched <- match_pdp_conditions(boot_enriched, pdp_data, year_pair)
    knn_enriched <- apply_trend_adjustments_hybrid(pdp_enriched, aligned_training_data, year_pair)
    
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

# --- 2. Core Prediction Function ---
predict_elevation_change <- function(tile_id, model_tile_id, year_pair, training_dir, prediction_dir) {
  model_path <- file.path(training_dir, model_tile_id, paste0("xgb_model_", year_pair, ".rds"))
  boruta_path <- file.path(training_dir, model_tile_id, paste0("boruta_selection_", year_pair, ".rds"))
  prediction_data_path <- file.path(prediction_dir, tile_id, paste0(tile_id, "_prediction_clipped_data.fst"))
  
  if (!file.exists(model_path)) stop("Missing required file: ", model_path)
  if (!file.exists(boruta_path)) stop("Missing required file: ", boruta_path)
  if (!file.exists(prediction_data_path)) stop("Missing required file: ", prediction_data_path)
  
  xgb_model <- readRDS(model_path)
  boruta_results <- readRDS(boruta_path)
  predictors_from_boruta <- boruta_results$confirmed_predictors
  
  prediction_data <- read_fst(prediction_data_path, as.data.table = TRUE)
  if (!"tile_id" %in% names(prediction_data)) prediction_data[, tile_id := tile_id]
  
  aligned_pred_data <- align_predictors(prediction_data, year_pair, predictors_from_boruta)
  
  missing_preds <- setdiff(predictors_from_boruta, names(aligned_pred_data))
  if (length(missing_preds) > 0) {
    stop("The following predictors required by the model are missing after alignment: ", paste(missing_preds, collapse=", "))
  }
  
  pred_matrix <- as.matrix(aligned_pred_data[, ..predictors_from_boruta])
  aligned_pred_data[, XGB_predicted_change := predict(xgb_model, newdata = pred_matrix)]
  
  message("  - INFO: Generated initial predictions for ", nrow(aligned_pred_data), " points.")
  return(aligned_pred_data)
}

# --- 3. Align Predictors (FINAL REVISION with Diagnostics) ---
align_predictors <- function(data_to_align, year_pair, predictors_from_boruta) {
  setDT(data_to_align)
  
  cat("  - DIAGNOSTIC (align_predictors): Available columns for tile '", data_to_align$tile_id[1], "' are:\n",
      "     ", paste(names(data_to_align), collapse = ", "), "\n")
  
  # --- Map Static Predictors to Temporal Versions ---
  # This section ensures that if a model was trained on a specific year's data
  # (e.g., bathy_2004), the prediction data (which has generic 'bt.bathy') can be used.
  bathy_req <- predictors_from_boruta[startsWith(predictors_from_boruta, "bathy_")]
  if (length(bathy_req) > 0 && "bt.bathy" %in% names(data_to_align)) {
    for (col in bathy_req) {
      if (!col %in% names(data_to_align)) { data_to_align[, (col) := bt.bathy] }
    }
  }
  slope_req <- predictors_from_boruta[startsWith(predictors_from_boruta, "slope_")]
  if (length(slope_req) > 0 && "bt.slope" %in% names(data_to_align)) {
    for (col in slope_req) {
      if (!col %in% names(data_to_align)) { data_to_align[, (col) := bt.slope] }
    }
  }
  
  rugo_req <- predictors_from_boruta[startsWith(predictors_from_boruta, "rugosity_")]
  if (length(rugo_req) > 0) {
    all_rugo_cols_in_data <- names(data_to_align)[grepl("rugosity", names(data_to_align), ignore.case = TRUE)]
    source_rugo_col <- setdiff(all_rugo_cols_in_data, rugo_req)
    if (length(source_rugo_col) > 0) {
      source_col_name <- source_rugo_col[1]
      for (col in rugo_req) {
        if (!col %in% names(data_to_align)) {
          data_to_align[, (col) := get(source_col_name)]
        }
      }
    }
  }
  
  # --- Create Generic 'starting_' columns for PDP/KNN matching ---
  # This creates standardized column names (e.g., 'starting_bathy') that later
  # functions can rely on, regardless of the specific year_pair.
  start_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][1])
  
  # Helper function to create the starting columns
  create_starting_col <- function(dt, prefix, year, static_col) {
    temporal_col <- paste0(prefix, "_", year)
    starting_col_name <- paste0("starting_", prefix)
    if (temporal_col %in% names(dt)) {
      dt[, (starting_col_name) := .SD[[temporal_col]]]
    } else if (static_col %in% names(dt)) {
      dt[, (starting_col_name) := .SD[[static_col]]]
    }
  }
  
  create_starting_col(data_to_align, "bathy", start_year, "bt.bathy")
  create_starting_col(data_to_align, "slope", start_year, "bt.slope")
  
  # Handle rugosity separately due to its more complex naming
  rugo_start_name <- paste0("rugosity_", start_year)
  if (rugo_start_name %in% names(data_to_align)) {
    data_to_align[, starting_rugosity := .SD[[rugo_start_name]]]
  } else if (exists("source_rugo_col") && length(source_rugo_col) > 0) {
    data_to_align[, starting_rugosity := get(source_rugo_col[1])]
  }
  
  # --- MAJOR CHANGE: Add generic starting columns for new PDP predictors ---
  create_starting_col(data_to_align, "tsm", year_pair, "tsm_layer") # Assumes generic tsm is named tsm_layer
  create_starting_col(data_to_align, "grain_size", year_pair, "grain_size_layer") # Assumes generic is grain_size_layer
  
  # Standardize response variable name
  response_var_original <- paste0("b.change.", year_pair)
  if(response_var_original %in% names(data_to_align)){
    setnames(data_to_align, old = response_var_original, new = "b_change")
  }
  
  return(data_to_align)
}


# --- 4. Apply Bootstrap Adjustment (FIX for Striations) ---
apply_bootstrap_adjustment <- function(prediction_data, model_tile_id, year_pair, training_dir) {
  boots_path <- file.path(training_dir, model_tile_id, paste0("bootstraps_", year_pair, ".fst"))
  
  if (!file.exists(boots_path)) {
    warning("Bootstrap file not found for tile ", model_tile_id, ". Skipping this adjustment.")
    prediction_data[, `:=`(MeanBoots_predicted_change = NA_real_, MeanBoots_SD = NA_real_, b.change_actual = NA_real_)]
    return(prediction_data)
  }
  
  boots_data <- read_fst(boots_path, as.data.table = TRUE)
  boots_to_merge <- boots_data[, .(FID, Mean_Prediction, Uncertainty_SD, b.change_actual)]
  
  # --- MAJOR CHANGE: Preserve and Restore Original Order ---
  # 1. Create a temporary column to store the original row order.
  prediction_data[, temp_order := .I]
  
  # 2. Perform the merge. The order will be scrambled here.
  merged_data <- merge(prediction_data, boots_to_merge, by = "FID", all.x = TRUE)
  
  # 3. Sort the merged data back to its original order using the temp column.
  setorderv(merged_data, "temp_order")
  
  # 4. Remove the temporary ordering column.
  merged_data[, temp_order := NULL]
  # --- END OF MAJOR CHANGE ---
  
  # Rename columns for consistency
  setnames(merged_data, "Mean_Prediction", "MeanBoots_predicted_change", skip_absent=TRUE)
  setnames(merged_data, "Uncertainty_SD", "MeanBoots_SD", skip_absent=TRUE)
  
  message("  - INFO: Applied Bootstrap Mean adjustment. ", sum(!is.na(merged_data$MeanBoots_predicted_change)), " points received a value.")
  return(merged_data)
}

# --- 5. Match PDP Conditions (ENHANCED with more predictors) ---
match_pdp_conditions <- function(prediction_data, pdp_data, year_pair) {
  setDT(prediction_data); setDT(pdp_data)
  env_ranges <- extract_env_variable_ranges(pdp_data)
  start_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][1])
  all_pdp_predictors <- unique(pdp_data$Predictor)
  
  # --- MAJOR CHANGE: Define a list of key predictors and their properties ---
  # This makes the function more flexible and easier to update.
  # 'pred_pattern': The prefix used in Boruta results (e.g., "bathy_2004").
  # 'start_col': The generic column name created in align_predictors (e.g., "starting_bathy").
  # 'weight': The fraction of the predictor's range to use for matching.
  pdp_match_vars <- list(
    list(pred_pattern = "bathy_",      start_col = "starting_bathy",      weight = 0.4),
    list(pred_pattern = "slope_",      start_col = "starting_slope",      weight = 0.2),
    list(pred_pattern = "rugosity_",   start_col = "starting_rugosity",   weight = 0.1),
    list(pred_pattern = "tsm_",        start_col = "starting_tsm",        weight = 0.1),
    list(pred_pattern = "grain_size_", start_col = "starting_grain_size", weight = 0.1)
  )
  
  # Dynamically find which of these predictors are available in the PDP data
  for (i in seq_along(pdp_match_vars)) {
    # Find the full name of the predictor (e.g., "bathy_2004")
    full_pred_name <- grep(pdp_match_vars[[i]]$pred_pattern, all_pdp_predictors, value = TRUE)[1]
    pdp_match_vars[[i]]$full_pred_name <- if(length(full_pred_name) > 0) full_pred_name else NA_character_
    
    # Get the range width for that predictor
    range_val <- if(!is.na(full_pred_name)) env_ranges[Predictor == full_pred_name, range_width] else NA_real_
    pdp_match_vars[[i]]$range_width <- if(length(range_val) > 0) range_val else NA_real_
  }
  
  # --- Main matching loop using the dynamic list ---
  prediction_data[, pdp_adjusted_change := {
    current_point <- .SD
    # Create a vector to hold logical results for each row in pdp_data
    is_match <- rep(FALSE, nrow(pdp_data))
    
    for (var in pdp_match_vars) {
      # Check if the variable is usable for matching
      if (!is.na(var$full_pred_name) && !is.na(var$range_width) && var$start_col %in% names(current_point)) {
        
        current_val <- current_point[[var$start_col]]
        if (!is.na(current_val)) {
          # Calculate the matching window
          half_window <- var$weight * var$range_width
          lower_bound <- current_val - half_window
          upper_bound <- current_val + half_window
          
          # Update the is_match vector: a point matches if it falls within the window for ANY variable
          is_match <- is_match | (pdp_data$Predictor == var$full_pred_name & pdp_data$Env_Value >= lower_bound & pdp_data$Env_Value <= upper_bound)
        }
      }
    }
    
    matches <- pdp_data[is_match]
    if (nrow(matches) > 0) mean(matches$PDP_Value, na.rm = TRUE) else NA_real_
  }, by = 1:nrow(prediction_data)]
  
  message("  - INFO: PDP Adjusted Change matched on: ", sum(!is.na(prediction_data$pdp_adjusted_change)), "/", nrow(prediction_data))
  return(prediction_data)
}


# --- 6. Apply KNN Trend Adjustments ---
apply_trend_adjustments_hybrid <- function(prediction_data, training_data, year_pair, k = 15) {
  setDT(prediction_data); setDT(training_data)
  
  potential_env_vars <- c("starting_bathy", "starting_slope", "starting_rugosity", "tsm_layer", "grain_size") #, "hurr_count", "hurr_strength")
  env_vars <- intersect(potential_env_vars, intersect(names(prediction_data), names(training_data)))
  
  if (length(env_vars) == 0) {
    warning("No environmental variables available for KNN matching. Skipping.")
    prediction_data[, KNN_predicted_change := NA_real_]
    return(prediction_data)
  }
  message("  - INFO: KNN Matching on variables: ", paste(env_vars, collapse = ", "))
  
  for (var in env_vars) { if (is.character(prediction_data[[var]])) prediction_data[, (var) := as.numeric(get(var))]; if (is.character(training_data[[var]])) training_data[, (var) := as.numeric(get(var))] }
  
  pred_clean <- prediction_data[complete.cases(prediction_data[, ..env_vars])]
  train_clean <- training_data[complete.cases(training_data[, ..env_vars])]
  
  if (nrow(pred_clean) == 0) { message("  - WARN: No complete cases in prediction data for KNN. Skipping."); prediction_data[, KNN_predicted_change := NA_real_]; return(prediction_data) }
  if (nrow(train_clean) < k) { stop("Not enough complete cases in training data for KNN (less than k).") }
  
  response_var <- "b_change"
  if (!response_var %in% names(train_clean)) { stop(paste0("Missing response variable in training_data: ", response_var)) }
  
  pred_mat <- as.matrix(pred_clean[, ..env_vars])
  train_mat <- as.matrix(train_clean[, ..env_vars])
  train_resp <- train_clean[[response_var]]
  
  knn_result <- FNN::get.knnx(train_mat, pred_mat, k = k)
  
  weights <- 1 / (knn_result$nn.dist + 1e-6)
  weight_sums <- rowSums(weights, na.rm = TRUE)
  pred_avg <- rowSums(matrix(train_resp[knn_result$nn.index], nrow = nrow(knn_result$nn.index)) * weights, na.rm = TRUE) / weight_sums
  
  pred_clean[, KNN_predicted_change := pred_avg]
  prediction_data[pred_clean, on = .(FID), KNN_predicted_change := i.KNN_predicted_change]
  
  message("  - INFO: Assigned KNN predictions to ", sum(!is.na(prediction_data$KNN_predicted_change)), " rows.")
  return(prediction_data)
}

# --- 7. Compare Methods and Create Hybrid Prediction (ENHANCED) ---
compare_prediction_methods <- function(prediction_data, tile_id, year_pair, processing_mode) {
  message("  - INFO: Comparing prediction methods and combining results...")
  
  # --- MAJOR CHANGE 1: The 'hybrid_change' is now the average of the three methods ---
  # This provides a more stable, blended prediction.
  prediction_data[, hybrid_change := rowMeans(.SD, na.rm = TRUE), 
                  .SDcols = c("MeanBoots_predicted_change", "pdp_adjusted_change", "KNN_predicted_change")]
  # If all three are NA, the result of rowMeans is NaN; convert this to NA.
  prediction_data[is.nan(hybrid_change), hybrid_change := NA_real_]
  
  pdp_knn_df <- prediction_data[!is.na(pdp_adjusted_change) & !is.na(KNN_predicted_change)]
  comparison_log <- data.table(
    Tile_ID = tile_id,
    Year_Pair = year_pair,
    Total_Points = nrow(prediction_data),
    PDP_KNN_Cor = if (nrow(pdp_knn_df) > 2) cor(pdp_knn_df$pdp_adjusted_change, pdp_knn_df$KNN_predicted_change) else NA,
    PDP_KNN_MeanDiff = if (nrow(pdp_knn_df) > 0) mean(pdp_knn_df$pdp_adjusted_change - pdp_knn_df$KNN_predicted_change, na.rm=TRUE) else NA
  )
  
  performance_log <- NULL
  if (processing_mode == 'direct_model') {
    calculate_diff_stats <- function(col1, col2) {
      diff_vals <- col1 - col2
      return(list(
        MeanDiff = mean(diff_vals, na.rm = TRUE),
        MinDiff = min(diff_vals, na.rm = TRUE),
        MaxDiff = max(diff_vals, na.rm = TRUE)
      ))
    }
    pdp_stats <- calculate_diff_stats(prediction_data$pdp_adjusted_change, prediction_data$XGB_predicted_change)
    knn_stats <- calculate_diff_stats(prediction_data$KNN_predicted_change, prediction_data$XGB_predicted_change)
    hybrid_stats <- calculate_diff_stats(prediction_data$hybrid_change, prediction_data$XGB_predicted_change)
    performance_log <- data.table(
      Tile_ID = tile_id,
      Year_Pair = year_pair,
      PDP_vs_XGB_MeanDiff = pdp_stats$MeanDiff,
      PDP_vs_XGB_MinDiff = pdp_stats$MinDiff,
      PDP_vs_XGB_MaxDiff = pdp_stats$MaxDiff,
      KNN_vs_XGB_MeanDiff = knn_stats$MeanDiff,
      KNN_vs_XGB_MinDiff = knn_stats$MinDiff,
      KNN_vs_XGB_MaxDiff = knn_stats$MaxDiff,
      Hybrid_vs_XGB_MeanDiff = hybrid_stats$MeanDiff,
      Hybrid_vs_XGB_MinDiff = hybrid_stats$MinDiff,
      Hybrid_vs_XGB_MaxDiff = hybrid_stats$MaxDiff
    )
  }
  
  return(list(data = prediction_data, comparison_log = comparison_log, performance_log = performance_log))
}

# ==============================================================================
#   HELPER FUNCTIONS (FILE I/O & PLOTTING)
# ==============================================================================
# --- Helper: Extract Environmental Ranges from PDP data ---
extract_env_variable_ranges <- function(pdp_data) {
  if (is.null(pdp_data) || nrow(pdp_data) == 0) {
    return(data.table(Predictor = character(), range_width = numeric()))
  }
  env_ranges <- pdp_data[, .(range_width = max(Env_Value, na.rm = TRUE) - min(Env_Value, na.rm = TRUE)), by = Predictor]
  return(env_ranges)
}


# --- Helper: Error Metrics ---
mae <- function(actual, predicted) {
  mean(abs(actual - predicted), na.rm = TRUE)
}

rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2, na.rm = TRUE))
}


# --- 8. Save Final FST Output ---
save_final_predictions <- function(prediction_data, output_dir, tile_id, year_pair) {
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  pred_depth_col <- paste0("pred_", end_year, "_depth")
  if (!"starting_bathy" %in% names(prediction_data) || !"hybrid_change" %in% names(prediction_data)) { stop("Cannot save final predictions: 'starting_bathy' or 'hybrid_change' column is missing.") }
  
  prediction_data[, (pred_depth_col) := starting_bathy + hybrid_change]
  
  out_file_dir <- file.path(output_dir, tile_id); if (!dir.exists(out_file_dir)) dir.create(out_file_dir, recursive = TRUE)
  out_file <- file.path(out_file_dir, paste0(tile_id, "_prediction_final_", year_pair, ".fst"))
  
  write_fst(prediction_data, out_file)
  message("  - INFO: Final FST prediction file saved to: ", out_file)
  return(prediction_data)
}

# --- 9. Save Component Raster Outputs ---
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
        message("  - INFO: Component raster saved to: ", out_path)
      }
    }
  }
  
  save_raster(prediction_data, "hybrid_change", "Hybrid_predicted_change")
  save_raster(prediction_data, "pdp_adjusted_change", "PDP_predicted_change")
  save_raster(prediction_data, "KNN_predicted_change", "KNN_predicted_change")
  save_raster(prediction_data, "XGB_predicted_change", "XGB_predicted_change")
  save_raster(prediction_data, "MeanBoots_predicted_change", "MeanBoots_predicted_change")
}


# --- 10. Build Year-Specific Tile Maps ---
build_all_tile_maps <- function(training_dir, prediction_dir, year_pairs) {
  message("  - INFO: Building year-specific geographic maps...")
  
  training_footprint_gpkg <- file.path(training_dir, "subgrid_footprints_training.gpkg")
  prediction_footprint_gpkg <- file.path(prediction_dir, "subgrid_footprints_prediction.gpkg")
  
  if(!all(file.exists(training_footprint_gpkg, prediction_footprint_gpkg))) {
    stop("Missing footprint shapefiles required to build the tile map.")
  }
  
  train_sf_full <- sf::st_read(training_footprint_gpkg, quiet = TRUE)
  pred_sf <- sf::st_read(prediction_footprint_gpkg, quiet = TRUE)
  
  id_col <- names(train_sf_full)[grepl("tile_id|name|id", names(train_sf_full), ignore.case=TRUE)][1]
  if(is.na(id_col)) stop("Could not find a tile ID column in footprint files.")
  
  suppressWarnings({
    pred_centroids <- sf::st_centroid(pred_sf)
  })
  
  all_maps <- list()
  
  for (yp in year_pairs) {
    cat("        - Mapping for year-pair:", yp, "\n")
    
    model_file_name <- paste0("xgb_model_", yp, ".rds")
    valid_training_tiles <- list.dirs(training_dir, full.names = FALSE, recursive = FALSE)
    valid_source_tiles <- c()
    for (tile in valid_training_tiles) {
      if (file.exists(file.path(training_dir, tile, model_file_name))) {
        valid_source_tiles <- c(valid_source_tiles, tile)
      }
    }
    
    if (length(valid_source_tiles) == 0) {
      warning("No training tiles with a model found for year-pair: ", yp, ". Skipping map creation for this year.")
      all_maps[[yp]] <- NULL
      next
    }
    
    cat("          - Found", length(valid_source_tiles), "valid source tiles.\n")
    
    train_sf_year_specific <- train_sf_full[train_sf_full[[id_col]] %in% valid_source_tiles, ]
    
    suppressWarnings({
      train_centroids_year_specific <- sf::st_centroid(train_sf_year_specific)
    })
    
    nearest_indices <- sf::st_nearest_feature(pred_centroids, train_centroids_year_specific)
    
    tile_map <- train_sf_year_specific[[id_col]][nearest_indices]
    names(tile_map) <- pred_sf[[id_col]]
    
    all_maps[[as.character(yp)]] <- tile_map
  }
  
  message("  - INFO: All year-specific geographic maps successfully built.")
  return(all_maps)
}

# --- 11. Save Summary Plot (ENHANCED) ---
save_summary_plot <- function(processed_data, tile_id, year_pair, prediction_dir, crs_obj) {
  message("  - INFO: Generating summary plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  
  setorderv(processed_data, c("Y", "X"), c(-1, 1))
  
  plot_list <- list()
  
  # --- MAJOR CHANGE 2: Define the color palette ---
  # Using cmocean 'deep' which is similar to Python's 'ocean'
  plot_colors <- cmocean('deep')(100)
  
  dt_to_raster <- function(dt, col_name, crs) {
    if (col_name %in% names(dt) && any(!is.na(dt[[col_name]]))) {
      df <- dt[!is.na(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(df) > 0) return(raster::rasterFromXYZ(df, crs = crs))
    }
    return(NULL)
  }
  
  # --- Create all component rasters ---
  plot_list[['pdp']] <- dt_to_raster(processed_data, "pdp_adjusted_change", crs_obj)
  plot_list[['knn']] <- dt_to_raster(processed_data, "KNN_predicted_change", crs_obj)
  plot_list[['boots']] <- dt_to_raster(processed_data, "MeanBoots_predicted_change", crs_obj)
  plot_list[['xgb']] <- dt_to_raster(processed_data, "XGB_predicted_change", crs_obj)
  
  bathy_col_to_use <- if ("bt.bathy" %in% names(processed_data)) "bt.bathy" else "starting_bathy"
  plot_list[["bt_bathy"]] <- dt_to_raster(processed_data, bathy_col_to_use, crs_obj)
  
  plot_list[["survey_date"]] <- dt_to_raster(processed_data, "survey_end_date", crs_obj)
  plot_list[['actual_change']] <- dt_to_raster(processed_data, "b.change_actual", crs_obj)
  plot_list[['hybrid']] <- dt_to_raster(processed_data, "hybrid_change", crs_obj)
  
  # --- Create plot objects from the raster list ---
  p1 <- if(!is.null(plot_list$pdp)) levelplot(plot_list$pdp, main="PDP Prediction", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  p2 <- if(!is.null(plot_list$knn)) levelplot(plot_list$knn, main="KNN Prediction", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  p3 <- if(!is.null(plot_list$boots)) levelplot(plot_list$boots, main="Mean Boots Prediction", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  p4 <- if(!is.null(plot_list$xgb)) levelplot(plot_list$xgb, main="XGB Prediction", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  p5 <- if(!is.null(plot_list$bt_bathy)) levelplot(plot_list$bt_bathy, main="Input Bathy", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  p6 <- if(!is.null(plot_list$survey_date)) levelplot(plot_list$survey_date, main="Survey End Date", margin=FALSE, col.regions=terrain.colors) else ggplot() + theme_void()
  p7 <- if(!is.null(plot_list$actual_change)) levelplot(plot_list$actual_change, main="Actual Change (b.change)", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  p8 <- if(!is.null(plot_list$hybrid)) levelplot(plot_list$hybrid, main="Final Hybrid (Avg) Prediction", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_summary_plot_", year_pair, ".pdf"))
  
  pdf(plot_out_file, width = 20, height = 10)
  grid.arrange(grobs = list(p1, p2, p3, p4, p5, p6, p7, p8), ncol = 4, nrow = 2)
  dev.off()
  
  message("  - INFO: Summary plot saved to: ", plot_out_file)
}

# --- 11.5. NEW FUNCTION: Save SCALED Summary Plot ---
save_summary_plot_scaled <- function(processed_data, tile_id, year_pair, prediction_dir, crs_obj) {
  message("  - INFO: Generating SCALED summary plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  
  setorderv(processed_data, c("Y", "X"), c(-1, 1))
  
  # --- Define the shared color scale for all 'change' plots ---
  change_cols <- c("pdp_adjusted_change", "KNN_predicted_change", "MeanBoots_predicted_change", "XGB_predicted_change", "b.change_actual", "hybrid_change")
  
  # Determine the range: use actual change if available, otherwise default to -6 to 6
  zlim <- if ("b.change_actual" %in% names(processed_data) && any(is.finite(processed_data$b.change_actual))) {
    range(processed_data$b.change_actual, na.rm = TRUE)
  } else {
    c(-6, 6)
  }
  
  # Create color breaks for the legend
  at <- seq(zlim[1], zlim[2], length.out = 99)
  plot_colors <- cmocean('deep')(100)
  
  plot_list <- list()
  
  dt_to_raster <- function(dt, col_name, crs) {
    if (col_name %in% names(dt) && any(!is.na(dt[[col_name]]))) {
      df <- dt[!is.na(get(col_name)), .(x = X, y = Y, z = get(col_name))]
      if (nrow(df) > 0) return(raster::rasterFromXYZ(df, crs = crs))
    }
    return(NULL)
  }
  
  # Create all component rasters
  for(col in change_cols) {
    plot_list[[col]] <- dt_to_raster(processed_data, col, crs_obj)
  }
  bathy_col_to_use <- if ("bt.bathy" %in% names(processed_data)) "bt.bathy" else "starting_bathy"
  plot_list[["bt_bathy"]] <- dt_to_raster(processed_data, bathy_col_to_use, crs_obj)
  plot_list[["survey_date"]] <- dt_to_raster(processed_data, "survey_end_date", crs_obj)
  
  # --- Create plot objects, applying the shared scale to change plots ---
  p1 <- if(!is.null(plot_list$pdp_adjusted_change)) levelplot(plot_list$pdp_adjusted_change, main="PDP Prediction", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()
  p2 <- if(!is.null(plot_list$KNN_predicted_change)) levelplot(plot_list$KNN_predicted_change, main="KNN Prediction", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()
  p3 <- if(!is.null(plot_list$MeanBoots_predicted_change)) levelplot(plot_list$MeanBoots_predicted_change, main="Mean Boots Prediction", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()
  p4 <- if(!is.null(plot_list$XGB_predicted_change)) levelplot(plot_list$XGB_predicted_change, main="XGB Prediction", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()
  p5 <- if(!is.null(plot_list$bt_bathy)) levelplot(plot_list$bt_bathy, main="Input Bathy", margin=FALSE, col.regions=plot_colors) else ggplot() + theme_void()
  p6 <- if(!is.null(plot_list$survey_date)) levelplot(plot_list$survey_date, main="Survey End Date", margin=FALSE, col.regions=terrain.colors) else ggplot() + theme_void()
  p7 <- if(!is.null(plot_list$b.change_actual)) levelplot(plot_list$b.change_actual, main="Actual Change (b.change)", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()
  p8 <- if(!is.null(plot_list$hybrid_change)) levelplot(plot_list$hybrid_change, main="Final Hybrid (Avg) Prediction", margin=FALSE, at=at, col.regions=plot_colors) else ggplot() + theme_void()
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_summary_plot_SCALED_", year_pair, ".pdf"))
  
  pdf(plot_out_file, width = 20, height = 10)
  grid.arrange(grobs = list(p1, p2, p3, p4, p5, p6, p7, p8), ncol = 4, nrow = 2)
  dev.off()
  
  message("  - INFO: SCALED summary plot saved to: ", plot_out_file)
}

# --- 12. Save Density Plot (ENHANCED) ---
save_density_plot <- function(processed_data, tile_id, year_pair, prediction_dir) {
  message("  - INFO: Generating density plot for tile: ", tile_id, " | Year Pair: ", year_pair)
  
  # --- MAJOR CHANGE 2: Define the specific variables for the density plot ---
  cols_to_plot <- c("XGB_predicted_change", "pdp_adjusted_change", "KNN_predicted_change", 
                    "MeanBoots_predicted_change", "b.change_actual", "hybrid_change", 
                    "starting_bathy")
  cols_to_plot <- intersect(cols_to_plot, names(processed_data))
  
  if (length(cols_to_plot) < 2) {
    message("  - WARN: Not enough data columns to generate a density plot.")
    return(NULL)
  }
  
  plot_data <- melt(processed_data[, ..cols_to_plot], measure.vars = cols_to_plot, na.rm = TRUE)
  
  # Create a factor for ordering and labeling the plots to match the summary plot
  plot_data[, variable := factor(variable,
                                 levels = c("pdp_adjusted_change", "KNN_predicted_change", "MeanBoots_predicted_change", "XGB_predicted_change",
                                            "starting_bathy", "b.change_actual", "hybrid_change"),
                                 labels = c("PDP Pred", "KNN Pred", "Mean Boots Pred", "XGB Pred",
                                            "Input Bathy", "Actual Change", "Final Hybrid Pred"))]
  
  # --- MAJOR CHANGE 3: Standardize X-axis for change variables ---
  # Separate the data into two groups for different x-axis scaling
  change_data <- plot_data[!variable %in% c("Input Bathy")]
  bathy_data <- plot_data[variable == "Input Bathy"]
  
  # Calculate a common x-axis range for all 'change' related plots
  change_x_range <- range(change_data$value, na.rm = TRUE, finite = TRUE)
  
  # If actual change exists, use its range; otherwise, use a default or the combined range
  if ("Actual Change" %in% change_data$variable) {
    actual_range <- range(change_data[variable == "Actual Change"]$value, na.rm = TRUE, finite = TRUE)
    if(all(is.finite(actual_range))) {
      change_x_range <- actual_range
    }
  }
  # Fallback to a sensible default if the range is still not finite
  if(!all(is.finite(change_x_range))) {
    change_x_range <- c(-6, 6)
  }
  
  # Create a list to hold the individual plots
  plot_list <- list()
  
  # Generate plots for each variable type
  for(var_name in levels(plot_data$variable)) {
    sub_data <- plot_data[variable == var_name]
    
    p <- ggplot(sub_data, aes(x = value)) +
      geom_density(fill = "skyblue", alpha = 0.7) +
      labs(title = var_name, x = "Depth (m)", y = "Density") +
      theme_minimal() +
      theme(legend.position = "none")
    
    # Apply the standardized x-axis to the change plots
    if (var_name != "Input Bathy") {
      p <- p + scale_x_continuous(limits = change_x_range)
    }
    
    plot_list[[var_name]] <- p
  }
  
  # Add a placeholder for the empty slot in the 4x2 grid
  plot_list[["placeholder"]] <- ggplot() + theme_void()
  
  # Define the final order for the 4x2 grid
  final_plot_order <- c("PDP Pred", "KNN Pred", "Mean Boots Pred", "XGB Pred", 
                        "Input Bathy", "placeholder", "Actual Change", "Final Hybrid Pred")
  
  # Reorder the plot list to match the desired layout
  ordered_grobs <- lapply(final_plot_order, function(name) plot_list[[name]])
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_density_plot_", year_pair, ".pdf"))
  
  pdf(plot_out_file, width = 16, height = 8)
  grid.arrange(grobs = ordered_grobs, ncol = 4, nrow = 2)
  dev.off()
  
  message("  - INFO: Density plot saved to: ", plot_out_file)
}


# --- 13. Run Cross-Validation ---
run_cross_validation <- function(processed_data, tile_id, year_pair, training_dir, prediction_dir) {
  message("  - INFO: Running cross-validation for tile: ", tile_id, " | Year Pair: ", year_pair)
  
  training_data_path <- file.path(training_dir, tile_id, paste0(tile_id, "_training_clipped_data.fst"))
  if(!file.exists(training_data_path)) {
    message("  - WARN: No training data file found for this tile. Skipping cross-validation.")
    return(NULL)
  }
  
  truth_data <- read_fst(training_data_path, as.data.table = TRUE)
  
  validation_data <- merge(processed_data, truth_data, by = "FID", suffixes = c("_pred", "_actual"))
  
  end_year <- as.numeric(strsplit(as.character(year_pair), "_")[[1]][2])
  actual_depth_col <- paste0("bathy_", end_year)
  pred_depth_col <- paste0("pred_", end_year, "_depth")
  actual_change_col <- paste0("b.change.", year_pair)
  
  if (!all(c(actual_depth_col, pred_depth_col, actual_change_col) %in% names(validation_data))) {
    message("  - WARN: Missing required columns for cross-validation. Skipping.")
    return(NULL)
  }
  
  valid_comp_data <- validation_data[!is.na(get(actual_depth_col)) & !is.na(get(pred_depth_col))]
  
  if(nrow(valid_comp_data) < 2) {
    message("  - WARN: Not enough valid overlapping points for cross-validation.")
    return(NULL)
  }
  
  depth_mae <- mae(valid_comp_data[[actual_depth_col]], valid_comp_data[[pred_depth_col]])
  depth_rmse <- rmse(valid_comp_data[[actual_depth_col]], valid_comp_data[[pred_depth_col]])
  change_mae <- mae(valid_comp_data[[actual_change_col]], valid_comp_data$hybrid_change)
  change_rmse <- rmse(valid_comp_data[[actual_change_col]], valid_comp_data$hybrid_change)
  
  cv_log <- data.table(
    Tile_ID = tile_id,
    Year_Pair = year_pair,
    Depth_MAE = depth_mae,
    Depth_RMSE = depth_rmse,
    Change_MAE = change_mae,
    Change_RMSE = change_rmse
  )
  
  depth_plot_data <- melt(valid_comp_data, measure.vars = c(actual_depth_col, pred_depth_col))
  p1 <- ggplot(depth_plot_data, aes(x=value, fill=variable)) +
    geom_density(alpha=0.7) +
    scale_fill_manual(values=c("black", "cyan"), labels=c("Actual", "Predicted")) +
    labs(title="Predicted vs. Actual Depth", x="Depth (m)", fill="Data Type") +
    theme_minimal()
  
  change_plot_data <- melt(valid_comp_data, measure.vars = c(actual_change_col, "hybrid_change"))
  p2 <- ggplot(change_plot_data, aes(x=value, fill=variable)) +
    geom_density(alpha=0.7) +
    scale_fill_manual(values=c("black", "cyan"), labels=c("Actual", "Predicted")) +
    labs(title="Predicted vs. Actual Change", x="Change (m)", fill="Data Type") +
    theme_minimal()
  
  plot_out_dir <- file.path(prediction_dir, tile_id, "diagnostics")
  if (!dir.exists(plot_out_dir)) dir.create(plot_out_dir, recursive = TRUE)
  plot_out_file <- file.path(plot_out_dir, paste0(tile_id, "_cross_validation_plot_", year_pair, ".pdf"))
  
  pdf(plot_out_file, width = 12, height = 6)
  grid.arrange(p1, p2, ncol = 2)
  dev.off()
  
  message("  - INFO: Cross-validation plot saved to: ", plot_out_file)
  
  return(cv_log)
}


# ==============================================================================
#
#           Parallel Prediction Test Wrapper
#
# ==============================================================================

run_prediction_test <- function(prediction_tiles, year_pairs, crs_obj, training_dir, prediction_dir, output_dir) {
  
  message("\nStarting parallel prediction test run for ", length(prediction_tiles), " tiles...")
  
  # --- Pre-computation Step ---
  all_tile_maps <- build_all_tile_maps(training_dir, prediction_dir, year_pairs)
  
  # --- Setup Parallel Processing ---
  num_cores <- detectCores() - 1; if (num_cores < 1) num_cores <- 1
  cl <- makeCluster(num_cores); registerDoParallel(cl); on.exit({ stopCluster(cl) })
  
  task_grid <- expand.grid(tile_id = prediction_tiles, year_pair = year_pairs, stringsAsFactors = FALSE)
  
  message("\n--- Stage 1: Processing data in parallel... ---")
  
  # --- MAJOR CHANGE: Memory Optimization for Parallel Processing ---
  # Remove the .packages argument to prevent high memory usage on startup.
  # Libraries will be loaded inside the loop by each worker as needed.
  results_list <- foreach(
    i = 1:nrow(task_grid),
    .export = c("process_tile", "predict_elevation_change", "align_predictors",
                "extract_env_variable_ranges", "match_pdp_conditions",
                "apply_trend_adjustments_hybrid", "compare_prediction_methods",
                "apply_bootstrap_adjustment", "save_final_predictions", 
                "save_component_rasters", "save_summary_plot", "save_summary_plot_scaled",
                "save_density_plot", "run_cross_validation", "mae", "rmse")
  ) %dopar% {
    
    # --- Load libraries within each worker to stagger memory allocation ---
    library(data.table)
    library(dplyr)
    library(fst)
    library(sf)
    library(xgboost)
    library(FNN)
    library(raster)
    library(rasterVis)
    library(gridExtra)
    library(ggplot2)
    library(Metrics)
    library(cmocean)
    
    current_task <- task_grid[i, ]
    process_tile(
      tile_id = current_task$tile_id,
      year_pair = current_task$year_pair,
      training_dir = training_dir,
      prediction_dir = prediction_dir,
      output_dir = output_dir,
      all_tile_maps = all_tile_maps
    )
  }
  
  # --- Stage 2: Saving results and plots sequentially... ---
  message("\n--- Stage 2: Saving results and plots sequentially... ---")
  
  # --- MAJOR CHANGE 4: Make log saving more robust ---
  # Filter out any results that are not successful lists before trying to bind them.
  # This prevents errors from a single failed tile from stopping the entire process.
  successful_results <- Filter(function(res) !is.null(res) && is.list(res) && res$success, results_list)
  
  # --- Collect and save diagnostic logs ---
  all_comparison_logs <- rbindlist(lapply(successful_results, `[[`, "comparison_log"), fill = TRUE)
  all_performance_logs <- rbindlist(lapply(successful_results, `[[`, "performance_log"), fill = TRUE)
  
  if(nrow(all_comparison_logs) > 0) {
    fwrite(all_comparison_logs, file.path(output_dir, "prediction_method_comparison_log.csv"))
    message("  - INFO: Prediction comparison log saved.")
  }
  if (nrow(all_performance_logs) > 0) {
    fwrite(all_performance_logs, file.path(output_dir, "model_performance_log.csv"))
    message("  - INFO: Model performance log saved.")
  }
  
  # --- Save data and plot outputs ---
  all_cv_logs <- list()
  for (result in successful_results) { # Only loop through successful results
    cat("\n--- Saving outputs for Tile:", result$tile_id, "| Year Pair:", result$year_pair, "---\n")
    processed_data <- save_final_predictions(result$data, output_dir, result$tile_id, result$year_pair)
    if (!is.null(crs_obj)) {
      save_component_rasters(processed_data, output_dir, result$tile_id, result$year_pair, crs_obj)
      save_summary_plot(processed_data, result$tile_id, result$year_pair, prediction_dir, crs_obj)
      # --- MAJOR CHANGE: Call the new scaled summary plot function ---
      save_summary_plot_scaled(processed_data, result$tile_id, result$year_pair, prediction_dir, crs_obj)
      save_density_plot(processed_data, result$tile_id, result$year_pair, prediction_dir)
      
      # --- MAJOR CHANGE 3: Add diagnostic logging for Cross-Validation ---
      if (result$processing_mode == 'direct_model') {
        cv_log <- run_cross_validation(processed_data, result$tile_id, result$year_pair, training_dir, prediction_dir)
        if(!is.null(cv_log)) all_cv_logs[[length(all_cv_logs) + 1]] <- cv_log
      } else {
        # This message will now appear in the main console log.
        cat("  - INFO: Cross-validation skipped for tile", result$tile_id, "(Processing mode:", result$processing_mode, ")\n")
      }
    }
  }
  
  # --- Save Cross-Validation Log ---
  if(length(all_cv_logs) > 0) {
    all_cv_logs_dt <- rbindlist(all_cv_logs, fill = TRUE)
    fwrite(all_cv_logs_dt, file.path(output_dir, "cross_validation_log.csv"))
    message("  - INFO: Cross-validation log saved.")
  }
  
  message("\n✅ Parallel prediction test run complete.")
}


# Define the directories and parameters
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
prediction_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction_data_grid_tiles"
training_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training_data_grid_tiles"
years <- c("2004_2006") #, "2006_2010") # "2010_2015", "2015_2022")
mask <- raster::raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.UTM17_8m.tif")
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
  output_dir = prediction_dir
)
Sys.time()





showConnections()
closeAllConnections()
gc()








# Post Processing functions (still in development)

# Ensemble 
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












