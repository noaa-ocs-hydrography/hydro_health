# Amalgamation of all code 

## Data Pre-processing ----

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


# ─────────────────────────────────────────────
# PREPROCESSING MODULE: Full Workflow (Steps 1–5)
# ─────────────────────────────────────────────

# Functions list first, then function calls below
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
input_partial <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/part_processed1"
output_filled <- input_partial
# Raw data to be processed 
input_raw_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # prediction data used to create training data clipped
input_raw_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw1"
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

#F4 - Extract blue topo Uncertainty data----
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

grid_out_raster_data <- function(master_grid_gpkg, # V3 Now makes a template, for no values > 0 using bt. bathy
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




# ──────────────-----------------------------------
# PREPROCESSING FUNCTION CALLS
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

# F4 - BT.UNCERTAINTY----
extract_band2_to_uncertainty(input_dir, uncertainty_dir)
mosaic_ras <- mosaic_uncertainty_rasters(uncertainty_dir, mosaic_path)

# F5 - EXTRACT SURVEY END DATES----
extract_survey_end_dates(input_dir = input_dir_survey, kml_dir = kml_dir_survey, output_dir = output_dir_survey_dates)

# F6 & F7 - PREPARE SUB-GRIDS (Training & Prediction Masks)
# From OG Blue Topo Gpkg

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


# ──────────────-----------------------
# 5. FINISH / Close parallel 
# ──────────────----------------------
stopCluster(cl)
end_time <- Sys.time()
log_message(sprintf(" All preprocessing completed in %.1f minutes", as.numeric(difftime(end_time, start_time, units = "mins"))))




# CHECK OUTPUTS FOR ALIGNMENT!!!  BEFORE RUNNING THE GRID OUT RASTER FUNCTIONS !!!
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
# if there are discrepancies between the training and prediction gpkgs this function aligns them to make a master grid. gpkg
# Function 
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

# Function call 
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


# # F10 - Grid Out Raster Data----
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
  # relevant_tile_ids = "BH4S556X_3", # Example training tile BH4S556X_3
  parallel_tiles = F
)

# Check Alignment Between Training and Prediction Tiles----
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


# Single Tile fast check 
# After running the above - make sure grid tiles with the same IDs align between training and prediction 
# Check alignment of your test tiles match # same number of rows
train <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Training_data_grid_tiles/BH4S556X_3/BH4S556X_3_training_clipped_data.fst")
pred  <- read.fst("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model1/Coding_Outputs/Prediction_data_grid_tiles/BH4S556X_3/BH4S556X_3_prediction_clipped_data.fst")
identical(train[,c("X","Y")], pred[,c("X","Y")])   # should be TRUE ✅


# All tile extent check 
check_tile_alignment <- function(
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
check_tile_alignment(
  training_dir   = training_dir,
  prediction_dir = prediction_dir,
  output_csv     = file.path(dirname(training_dir), "tile_alignment_summary.csv")
)


# Data transformation into long format for time series modelling ---

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
  if (!data.table::is.data.table(df)) {
    data.table::setDT(df)
  }
  
  df_copy <- data.table::copy(df)
  
  # Transform flow direction first
  df_copy <- transform_flowdir_cols(df_copy)
  
  # Rename bt.* columns to *_t
  bt_cols <- grep("^bt\\.", names(df_copy), value = TRUE)
  
  if (length(bt_cols) > 0) {
    bt_new_names <- bt_cols |>
      stringr::str_replace("^bt\\.", "") |>
      paste0("_t")
    
    data.table::setnames(df_copy, old = bt_cols, new = bt_new_names)
  } else {
    bt_new_names <- character()
  }
  
  year_pairs <- list(
    c("2004", "2006"),
    c("2006", "2010"),
    c("2010", "2015"),
    c("2015", "2022")
  )
  
  static_vars <- c("grain_size_layer", "prim_sed_layer", "survey_end_date")
  id_cols <- c("X", "Y", "FID", "tile_id")
  
  static_cols <- static_vars[static_vars %in% names(df_copy)]
  id_cols <- id_cols[id_cols %in% names(df_copy)]
  
  year_pair_dfs <- list()
  
  for (pair in year_pairs) {
    y0 <- pair[1]
    y1 <- pair[2]
    pair_name <- paste(y0, y1, sep = "_")
    
    # Match both:
    #   xxx_2004_2006
    #   xxx_2004_2006_
    pair_pattern <- paste0("_", y0, "_", y1, "_?$")
    
    pair_specific_cols <- grep(pair_pattern, names(df_copy), value = TRUE)
    
    # Remove delta_bathy if present in prediction data
    pair_specific_cols <- setdiff(
      pair_specific_cols,
      paste0("delta_bathy_", pair_name)
    )
    
    final_cols <- unique(c(
      id_cols,
      bt_new_names,
      pair_specific_cols,
      static_cols
    ))
    
    pred_df_pair <- df_copy[, final_cols, with = FALSE]
    
    # Normalize only malformed pair-based names with trailing underscore
    # Example:
    #   tsm_mean_2004_2006_ -> tsm_mean_2004_2006
    pair_cols_in_df <- intersect(pair_specific_cols, names(pred_df_pair))
    
    if (length(pair_cols_in_df) > 0) {
      cleaned_pair_names <- stringr::str_remove(pair_cols_in_df, "_$")
      
      # Only rename columns that actually change
      cols_to_rename <- pair_cols_in_df[pair_cols_in_df != cleaned_pair_names]
      new_names <- cleaned_pair_names[pair_cols_in_df != cleaned_pair_names]
      
      if (length(cols_to_rename) > 0) {
        data.table::setnames(
          pred_df_pair,
          old = cols_to_rename,
          new = new_names
        )
      }
    }
    
    year_pair_dfs[[pair_name]] <- pred_df_pair
  }
  
  year_pair_dfs
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
      
      tile_name <- stringr::str_remove(
        basename(f_path),
        paste0("_", mode, "_clipped_data\\.fst$")
      )
      
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






