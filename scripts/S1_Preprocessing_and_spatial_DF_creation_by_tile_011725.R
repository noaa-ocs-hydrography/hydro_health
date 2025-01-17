
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

#### NOTE:------ It is critical that BOTH the prediction and training datasets must have the same values of X, Y and FID within each, ---###
# although they are different extents, the smaller training data must be a direct subset of the prediction data
# for variables they have in common, even if the datasets vary between the two final datasets, we will divide the pertiant 
#columns afterward.

# Step 1: Load the mask and prepare spatial points dataframe
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model") 
mask <- raster("prediction.mask.tif")
crs_mask <- crs(mask)  # CRS of the mask for projection checks

# Convert mask to a dataframe with precision spatial points
mask_df <- as.data.frame(rasterToPoints(mask))  # Includes X (Easting), Y (Northing), and values
mask_coords <- mask_df[, c("x", "y")]  # Extract X, Y coordinates

# Step 2: Load the list of rasters to be processed
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/raw")
f250m <- list.files(getwd(), pattern = "\\.tif$", full.names = TRUE)
ras250m <- lapply(f250m, raster)  # Load all .tif files as raster objects

# Step 3: Ensure output directory exists
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed"
if (!dir.exists(output_dir)) dir.create(output_dir)

# Step 4: First Loop - Ensure CRS consistency
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Ensure CRS matches the mask
  if (!compareCRS(ras, mask)) {
    ras <- projectRaster(ras, crs = crs_mask)  # Reproject to match mask CRS
  }
  
  # Replace in list
  ras250m[[i]] <- ras
  cat("CRS processed for:", source_name, "\n")
}

# Step 5: Second Loop - Adjust extents to match mask
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Extend raster if extent is smaller than the mask
  if (!all(extent(ras) == extent(mask))) {
    # Crop or extend based on relative extents
    ras <- crop(ras, mask)  # Crop to mask extent
    ras <- extend(ras, mask, value = NA)  # Extend to match mask extent
  }
  
  # Resample to 5m resolution if needed
  if (xres(ras) != 5 || yres(ras) != 5) {
    ras <- resample(ras, mask, method = "bilinear")  # Resample to match mask resolution
  }
  
  # # Save intermediate raster for debugging (optional)
  # temp_file <- file.path(output_dir, paste0(source_name, "_extent.tif"))
  # writeRaster(ras, filename = temp_file, overwrite = TRUE)
  
  # Replace in list
  ras250m[[i]] <- ras
  cat("Extent adjusted for:", source_name, "\n")
}

# Step 6: Third Loop - Mask combination and final processing
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Combine mask spatial points with raster values for precise cropping/extending
  temp_df <- cbind(mask_coords, value = extract(ras, mask_coords))  # Bind X, Y, and raster values
  temp_df[is.na(temp_df)] <- NA  # Ensure consistency in NA handling
  temp_ras <- rasterFromXYZ(temp_df, crs = crs_mask)  # Convert back to raster
  
  # Set values > 0 to NA for "_bathy" rasters
  if (grepl("_bathy", source_name)) {
    temp_ras[temp_ras > 0] <- NA
  }
  
  # Save the final raster
  output_name <- file.path(output_dir, paste0(source_name, ".tif"))
  writeRaster(temp_ras, filename = output_name, overwrite = TRUE)
  
  cat("Processed and saved:", source_name, "\n")
}

### Check if the rasters achieved the same extent###
# Get the list of raster files
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
# Step 1: Load the mask and prepare spatial points dataframe
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model") 
mask <- raster("training.mask.tif")
plot(mask)
crs_mask <- crs(mask)  # CRS of the mask for projection checks

# Convert mask to a dataframe with precision spatial points
mask_df <- as.data.frame(rasterToPoints(mask))  # Includes X (Easting), Y (Northing), and values
mask_coords <- mask_df[, c("x", "y")]  # Extract X, Y coordinates

# Step 2: Load the list of rasters to be processed and Retain only pertinent data
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed")
f250m <- list.files(getwd(), pattern = "\\.tif$", full.names = TRUE)
print(f250m)
# Create a new list that excludes files starting with "blue_topo" 
filtered_files <- f250m[!grepl("^blue_topo", basename(f250m))] # === we dont need the wider prediction blue topo data 
f250m <- filtered_files
# Print the filtered list of files print(filtered_files)
ras250m <- lapply(f250m, raster)  # Load all .tif files as raster objects

# Step 3: Ensure output directory exists
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed"
if (!dir.exists(output_dir)) dir.create(output_dir)

# Step 4: First Loop - Ensure CRS consistency
Sys.time() # print to check processing time
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Ensure CRS matches the mask
  if (!compareCRS(ras, mask)) {
    ras <- projectRaster(ras, crs = crs_mask)  # Reproject to match mask CRS
  }
  
  # Replace in list
  ras250m[[i]] <- ras
  cat("CRS processed for:", source_name, "\n")
}

# Step 5: Second Loop - Adjust extents to match mask
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Extend raster if extent is smaller than the mask
  if (!all(extent(ras) == extent(mask))) {
    # Crop or extend based on relative extents
    ras <- crop(ras, mask)  # Crop to mask extent
    ras <- extend(ras, mask, value = NA)  # Extend to match mask extent
  }
  
  # Resample to 5m resolution if needed
  if (xres(ras) != 5 || yres(ras) != 5) {
    ras <- resample(ras, mask, method = "bilinear")  # Resample to match mask resolution
  }
  
  # # Save intermediate raster for debugging (optional)
  # temp_file <- file.path(output_dir, paste0(source_name, "_extent.tif"))
  # writeRaster(ras, filename = temp_file, overwrite = TRUE)
  
  # Replace in list
  ras250m[[i]] <- ras
  cat("Extent adjusted for:", source_name, "\n")
}

# Step 6: Third Loop - Mask combination and final processing
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Combine mask spatial points with raster values for precise cropping/extending
  temp_df <- cbind(mask_coords, value = extract(ras, mask_coords))  # Bind X, Y, and raster values
  temp_df[is.na(temp_df)] <- NA  # Ensure consistency in NA handling
  temp_ras <- rasterFromXYZ(temp_df, crs = crs_mask)  # Convert back to raster
  
  # Set values > 0 to NA for "_bathy" rasters
  if (grepl("_bathy", source_name)) {
    temp_ras[temp_ras > 0] <- NA
  }
  
  # Save the final raster
  output_name <- file.path(output_dir, paste0(source_name, ".tif"))
  writeRaster(temp_ras, filename = output_name, overwrite = TRUE)
  
  cat("Processed and saved:", source_name, "\n")
  Sys.time()
}

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
} else {
  print("Rasters do not have the same extents.")
}



#3. Create a spatial DF from TRAINING extent mask----
library(dplyr)
training.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.tif") # for reference CRS of training grid
# Convert raster to a spatial points dataframe
training.mask.spdf <- rasterToPoints(training.mask, spatial = TRUE)
# Extract coordinates and bind with raster data
training.mask.df <- data.frame(training.mask.spdf@data, X = training.mask.spdf@coords[, 1], Y = training.mask.spdf@coords[, 2])
# Set unique FID and extract X & Y data from the raster stack
training.mask.df$FID <- cellFromXY(training.mask, training.mask.df[, c("X", "Y")])
training.mask.df <- training.mask.df[training.mask.df$training.mask == 1, ]
# Save the filtered dataframe
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
saveRDS(training.mask.df, file = "training.mask.df.011425.Rds")
training.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/training.mask.df.011425.Rds")

#3. Create a spatial DF from PREDICTION extent mask----
prediction.mask <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.tif")
prediction.mask.spdf <- rasterToPoints(prediction.mask, spatial = T) # turns data into a large spatial points dataframe
prediction.mask.spdf@coords # a command to see the x and y from the raster data
# Extract the coordinates, for the spatial points df, and bind it with the env data from the raster stack,and save as a new dataframe
prediction.mask.df <- data.frame(prediction.mask.spdf@data, X =prediction.mask.spdf@coords[,1],Y = prediction.mask.spdf@coords[,2])
head(prediction.mask.df)
#--SET UNIQUE FID, EXTRACT X & Y(row and column numbers) data from raster stack
prediction.mask.df$FID <- cellFromXY(prediction.mask, prediction.mask.df[,c("X", "Y")])
prediction.mask.df <- prediction.mask.df[prediction.mask.df$prediction.mask == 1, ]
head(prediction.mask.df)
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
saveRDS (prediction.mask.df, file = "prediction.mask.df.011425.Rds")
prediction.mask.df <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/prediction.mask.df.011425.Rds")

#4. Functions to split a tile into sub-grids and create tile sized spatial dataframes in each tile folder and return extents variables----
split_tile <- function(tile) {
  bbox <- st_bbox(tile)
  mid_x <- (bbox["xmin"] + bbox["xmax"]) / 2
  mid_y <- (bbox["ymin"] + bbox["ymax"]) / 2
  
  # Define sub-grid polygons in clockwise order
  sub_grids <- list(
    st_polygon(list(matrix(c(bbox["xmin"], bbox["ymin"], mid_x, bbox["ymin"], 
                             mid_x, mid_y, bbox["xmin"], mid_y, bbox["xmin"], bbox["ymin"]), ncol = 2, byrow = TRUE))),
    st_polygon(list(matrix(c(mid_x, bbox["ymin"], bbox["xmax"], bbox["ymin"], 
                             bbox["xmax"], mid_y, mid_x, mid_y, mid_x, bbox["ymin"]), ncol = 2, byrow = TRUE))),
    st_polygon(list(matrix(c(mid_x, mid_y, bbox["xmax"], mid_y, 
                             bbox["xmax"], bbox["ymax"], mid_x, bbox["ymax"], mid_x, mid_y), ncol = 2, byrow = TRUE))),
    st_polygon(list(matrix(c(bbox["xmin"], mid_y, mid_x, mid_y, 
                             mid_x, bbox["ymax"], bbox["xmin"], bbox["ymax"], bbox["xmin"], mid_y), ncol = 2, byrow = TRUE)))
  )
  
  # Combine extents with IDs into a DataFrame
  sub_grid_ids <- paste0("Tile_", tile$tile, "_", 1:4)
  sub_grid_extents <- lapply(sub_grids, st_bbox)
  
  data.frame(
    tile_id = sub_grid_ids,
    xmin = sapply(sub_grid_extents, `[[`, "xmin"),
    ymin = sapply(sub_grid_extents, `[[`, "ymin"),
    xmax = sapply(sub_grid_extents, `[[`, "xmax"),
    ymax = sapply(sub_grid_extents, `[[`, "ymax"),
    geometry = st_sfc(sub_grids, crs = st_crs(tile))
  )
}# defines the size and order of the sub grid - accompanying function to work inside the prepare_subgrids functions

# Function to prepare and save a new sub-grid gpkg (master blue topo grid tile divided by 4, same tile I.Ds with sub identifier _1,_2_3,_4 in clockwise order)
prepare_subgrids <- function(grid_gpkg, mask.df, mask, output_dir) {
  cat("Preparing grid tiles and sub-grids...\n")
  
  # Read and transform the grid tiles to match the CRS of the mask
  grid_tiles <- st_read(grid_gpkg) %>%
    st_transform(st_crs(mask))
  
  # Convert mask.df to an sf object
  training_sf <- st_as_sf(mask.df, coords = c("X", "Y"), crs = st_crs(mask))
  
  # Split grid tiles into sub-grids
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) split_tile(grid_tiles[i, ])))
  
  # Convert sub_grids to an sf object
  sub_grids_sf <- st_as_sf(sub_grids, coords = c("xmin", "ymin", "xmax", "ymax"), crs = st_crs(grid_tiles))
  
  # Filter sub-grids that intersect with the mask points
  intersecting_sub_grids <- st_filter(sub_grids_sf, st_union(training_sf))
  
  # Save intersecting sub-grids to a GeoPackage
  st_write(intersecting_sub_grids, file.path(output_dir, "intersecting_sub_grids.gpkg"), delete_layer = TRUE)
  
  # Save the training sub-grid extents as RDS
  saveRDS(intersecting_sub_grids, file.path(output_dir, "grid_tile_extents.rds"))
  
  cat("Sub-grids prepared:", nrow(intersecting_sub_grids), "\n")
  return(intersecting_sub_grids)
} # universal grid creation for any spatial DF and mask extent

# Function to process raster data into a chunk size spatial datasets per tile folder
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
    tile_extent <- st_bbox(sub_grid)  # Get spatial extent of the tile
    
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
      r <- raster::raster(r_file)  # Load raster using `raster`
      cropped_r <- raster::crop(r, tile_extent)  # Crop to tile extent
      
      # Extract raster values along with X and Y coordinates
      raster_data <- as.data.frame(raster::rasterToPoints(cropped_r, spatial = FALSE))
      raster_name <- tools::file_path_sans_ext(basename(r_file))  # Extract raster name
      colnames(raster_data) <- c("X", "Y", paste0(raster_name))  
      raster_data$FID <- raster::cellFromXY(r, raster_data[, c("X", "Y")])  # Add FID
      return(raster_data)
    })
    
    # Combine all rasters into a single data frame
    combined_data <- Reduce(function(x, y) merge(x, y, by = c("X", "Y", "FID"), all = TRUE), clipped_data)
    
    # Create new b.change columns here (old code)
    ## Create depth change columns between each pair of survey years as this will be our model response variable (what we want to predict)----
    combined_data <- combined_data %>%
      mutate(b.change.2004_2006 = `bathy_2006` - `bathy_2004`, # 2 years
        b.change.2006_2010 = `bathy_2010` - `bathy_2006`, # 4 years
        b.change.2010_2015 = `bathy_2015` - `bathy_2010`, # 5 years 
        b.change.2015_2022 = `bathy_2022` - `bathy_2015`) # 7 years
    
    print(str(combined_data))  # Log structure to verify changes
    
    # Save the combined data as RDS
    saveRDS(combined_data, file = clipped_data_path)
    cat("Saved", data_type, "clipped data for tile:", tile_name, "\n")
  }
  cat("Finished processing all", data_type, "tiles in", output_dir, "\n")
}


# Define parameters
grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg" # from Blue topo
training.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.tif") # for reference CRS of training grid
prediction.mask <- raster ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.tif")
output_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles"
output_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles"
input_dir_train <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Training/processed" # raster data 
input_dir_pred <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed" # raster data 

# Run prediction sub grid processing----
Sys.time()
prediction_sub_grids <- prepare_subgrids(
  grid_gpkg = grid_gpkg,
  mask = prediction.mask,
  mask.df = prediction.mask.df,
  output_dir = output_dir_pred
)
Sys.time() # Takes approx 40 minutes and 25GB of space for this size of area. 

# Run training sub grid processing----
Sys.time()
training_sub_grids <- prepare_subgrids(
  grid_gpkg = grid_gpkg,
  mask = training.mask,
  mask.df = training.mask.df,
  output_dir = output_dir_train
)
Sys.time()  # Takes approx 15 minutes time and 20GB of space for this size of area. 

# Define new parameters
training_sub_grids <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Training.data.grid.tiles/intersecting_sub_grids.gpkg")
prediction_sub_grids <- st_read ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Coding_Outputs/Prediction.data.grid.tiles/intersecting_sub_grids.gpkg")

# Run the process_rasters function for raster subset creation
#Training - 1.5 hrs & < 10GB
Sys.time()
process_rasters(
  sub_grid_gpkg = training_sub_grids,
  data_type = "training", 
  output_dir = output_dir_train,
  raster_dir = input_dir_train)
Sys.time()
# Prediction - 3.5hrs & < 10GB
Sys.time()
process_rasters(
  sub_grid_gpkg = prediction_sub_grids,
  data_type = "prediction", 
  output_dir = output_dir_pred,
  raster_dir = input_dir_pred)
Sys.time()