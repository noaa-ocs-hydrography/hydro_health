
# PRE-PROCESSING (STAGE 1): - this will be its own script (aka engine)
# 1. Extract all survey end date data from .xml (Raster Attribute Table) and create indavidaul rasters
#    Merge all individual survey end date rasters into a combined raster (currently mosaic via ArcGIS)
# 2. Standardize all model rasters (created in GIS /other) and clip to both prediction.mask and training.mask
     #ensure all data sets are in the same crs / projection, resolution (5m), and clip to different masks to ensure same size
#    also ensure all bathy elevation values above 0m are removed after clipping, if not filter (0m mark varies on boundary)
# 3. Convert prediction.mask and training.mask in to a Spatial Points Dataframe for sub grid processing
# 4. Load the blue topo grid tile gpkg and create a sub grid (by dividing it into 4 squares) for both the prediction and training mask extent
#    Create subset data frames of all the processed raster data (model variables), into the each sub grid tile folder, over the grid extent for model training


# Jan 2025 modifications:
# Even though code was modified to process data on a tile by tile approach it focussed on the model training and prediction,
# however it was still running off a master dataframe / dataset created from the raster mosaic over the extent of
# the model triaining / prediction extent. Once at the prediction stage this caused bottlenecks and extremely slow processing. In this iteration all data set creation and functions have been modified to subset and save
# all data within the bounds of each tile, and run solely on the information of each tile. 


# Load Packages
require(raster); require(terra);require(xml2)
require(dplyr); require(sp); require(ranger) 
require(sf); require(mgcv); require(stringr)
library(progress) # For a progress bar
library(pdp)
library(data.table)
library(ggplot2)
library(tidyr)
library(readr)
library(purrr)
library(terra)
library(future)
library(pbapply)


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

## 2. Standardize all rasters (PREDICTION Extent first as its larger)----
#makes all same extent, for processing into spatial points dataframe and removes all land based elevation values > 0 as well



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


## 2 Standardize all rasters (TRAINING Extent - sub sample of prediction extent)----
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


#3.1. Create a spatial DF from TRAINING extent mask (IN UTM COORDINATES)----
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

#3.2 Create another spatial DF from TRAINING extent mask (IN WGS84 COORDINATES for sub grid creation only)----
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

#3.3 Create a spatial DF from PREDICTION extent mask (IN UTM COORDINATES)----
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

#3.4 Create a spatial DF from PREDICTION extent mask (IN WGS84 COORDINATES for sub grid creation only)----
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

#4.  # Processing Functions----
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


# 4. DEFINE FUNCTION PARAMETERS before running functions below- make sure the link points to the newest dataset----
grid_gpkg <- st_read("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg") # from Blue topo
#
training_sub_grids_WGS84 <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/intersecting_sub_grids_WGS84.gpkg"
prediction_sub_grids_WGS84 <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/intersecting_sub_grids_WGS84.gpkg"
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
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"
output_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles"
input_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed" # raster data 
input_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # raster data 

# 5 - RUN Functions----
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
training_sub_grids_UTM <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/intersecting_sub_grids_UTM.gpkg")
prediction_sub_grids_UTM <-st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/intersecting_sub_grids_UTM.gpkg")

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
pred.data <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/BH4RZ577_3/BH4RZ577_3_prediction_clipped_data.rds")
train.data <- readRDS ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/BH4RZ577_3/BH4RZ577_3_training_clipped_data.rds")
