# """Sample R script to translate to Python"""
# 1. Model Training
# 2. Trend Extraction
# 3. Prediction




# Script split into two stage, pre processing of raw NOW Coast data, then running of the Machine Learning MOdel (ML)
# Raw data downlaoded into a single folder comprising of GeoTiffs and seperate Raster Attribute Tables .xml files.

# STEPS IN EACH STAGE:----

# PRE-PROCESSING (STAGE 1):
# extract all survey end date data from xml (Raster Attribute Table) and create standalone rasters
# merge all indavidual survey end date rasters into a combined raster (currently mosaic via ArcGIS)
# load all JALBTCX bathymetry mosaic rasters and associated slope rasters
# load in other model variables (sediment type, sediment grain size - other contributers surface current to come)
# standardise rasters and clip to both prediction.mask and training.mask: ensure all datasets are in the same crs / projection, resolution (5m), and clip to different masks to ensure same size
# ensure all bathy elevation values above 0m are removed after clipping, if not filter (0m mark varies on boundary)
# stack all data and convert to Spatial Points Dataframe / for proeccesing / save


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


# STAGE 1 - Pre-Processing:----


## P1 Survey End Date Extraction MULTIPLE .xml :----

# Define input and output directories
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_Data/Modeling/UTM17"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_Data/Modeling/survey_date_end"
kml_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_Data/Modeling/RATs"

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

## P2 Standardize all rasters (Prediction Extent first as its larger)----
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



## P3 Standardize all rasters (Training Extent - sub sample of prediction extent)----

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


## P4 Create Spatial Points Data Frames for modelling and training data sets:----

## ----MASTER DATASET(ALL DATA): ALREADY clipped TO P.mask extent from standardise rasters step-----
Sys.time()
raster_files <- list.files("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Model_variables/Prediction/processed", pattern = "\\.tif$", full.names = TRUE)
# create a list of rasters
print(raster_files) # check to see what are in the list
raster_list <- lapply(raster_files, raster)
# Create stack of rasters 
M1 <- stack(raster_list)
# create a spatial points dataframe
M1.spdf <- rasterToPoints(M1, spatial = T) # turns data into a large spatial points dataframe
M1.spdf@coords # a command to see the x and y from the raster data
# Extract the coordinates, for the spatial points df, and bind it with the env data from the raster stack,and save as a new dataframe
M1.df <- data.frame(M1.spdf@data, X =M1.spdf@coords[,1],Y = M1.spdf@coords[,2])
head(M1.df)
#--SET UNIQUE FID, EXTRACT X & Y(row and column numbers) data from raster stack
M1.df$FID <- cellFromXY(M1, M1.df[,c("X", "Y")])
head(M1.df)
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
save(M1.df, file = "pm.master_dataset.112124.Rdata")
# load("pm.prediction.data.111524.Rdata")

##---------- prediction data -------
# Assuming M1.spdf is your spatial points dataframe # Get column names of 
col_names <- colnames(M1.df) # Filter out column names that start with "20" 
filtered_col_names <- col_names[!grepl("^X20", col_names)] # Subset M1.spdf by selecting only the filtered columns 
P.data <- M1.df[, filtered_col_names] # Print the resulting subset to verify 
head(P.data)
colnames(P.data) <- c("bt.bathy","bt.slope","grain_size","sediment","survey_end_date", "X","Y","FID") # bt = blue topo
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
save(P.data, file = "pm.prediction.dataset.112124.Rdata")

## -----training data------
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model")
t.mask <- raster("training.mask.tif")
# Convert t.mask to a Spatial Points Data Frame
t.mask.spdf <- rasterToPoints(t.mask, spatial = T)
t.mask.spdf@coords # a command to see the x and y from the raster data
#Extract the coordinates, for the spatial points df, and bind it with the env data from the raster stack,and save as a new dataframe
T.data <- data.frame(t.mask.spdf@data, X = t.mask.spdf@coords[,1],Y = t.mask.spdf@coords[,2]) # see if this works, might need to replace X and y with t.mask.spdf

## EXTRACT model variables INFO where MASK cells are (to get exact FID Subset)----
T.data$FID <- cellFromXY(M1, T.data[,c("X", "Y")])# from the raster stack, return the raster cells (row +cols), in which
#the coordinates from T.data (in the form of X and Y) are found within the raster, and place them into newly created  FID column 
T.data <- cbind(T.data ,raster::extract(M1, T.data[,c("X", "Y")])) # extract all env data where T.data occurs in raster stack

# discard columns from master dataset not needed
T.data <- T.data   %>%
  select (-training.mask, -blue_topo_bathy_111924, -blue_topo_slope)
#### COL NAMES MUST ALWAYS BE ALPHA starting....!!!
colnames(T.data) <- c("X","Y","FID","bathy_2004","slope_2004", "bathy_2006","slope_2006",
                      "bathy_2010", "slope_2010", "bathy_2015","slope_2015",
                      "bathy_2022","slope_2022", "grain_size","sediment","survey_end_date")

#3 DOUBLE CHECK that All FID columns in T.data are found in P.data (essential for predicion later)
# Check if all FIDs in T.data are in P.data
all_T_in_P <- all(T.data$FID %in% P.data$FID)
# Print the results
cat("All FIDs in T.data are in P.data:", all_T_in_P, "\n")
# Check for any FIDs in T.data not in P.data
missing_in_P <- T.data$FID[!T.data$FID %in% P.data$FID]

## P5 Create depth change columns between each pair of survey years as this will be our model response variable (what we want to predict)----
T.data <- T.data %>%
  mutate(
    
    b.change.2004_2006 = `bathy_2006` - `bathy_2004`, # 2 years
    b.change.2006_2010 = `bathy_2010` - `bathy_2006`, # 4 years
    b.change.2010_2015 = `bathy_2015` - `bathy_2010`, # 5 years 
    b.change.2015_2022 = `bathy_2022` - `bathy_2015`) # 7 years
# total time span = 16 years

## P6 Assess missing data----
# library(naniar)
# gg_miss_var(S2.spdf)  # Visualize missingness per variable, remove poor coverage data 
# OPTIONAL # 
# # Remove columns with high NA proportion
# threshold <- 0.9  # Define threshold for NA proportion
# cols_to_keep <- colMeans(is.na(T.data)) < threshold
# T.data1 <- T.data[, cols_to_keep]


setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
save(T.data, file = "pm.training.dataset.112124.Rdata")

# # double check spatially
# ## - SPATIAL TEST--
# # ## DOUBLE CHECK IT SPATIALLY! (Does the primary[earliest] raster look like how it was input before we perform analysis??)
# test<- rasterFromXYZ(data.frame(x = T.data[,2],
#                                      y = T.data[,3],
#                                      z = T.data[, "training.mask"]),
#                           crs = crs(M1[[1]]))
# plot(test)
# plot(t.mask)
# test
# t.mask


# STAGE 2 - MODELLING:----

## P1 # LOAD DATA  ----

setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
load("pm.prediction.dataset.112124.Rdata")
load("pm.training.dataset.112124.Rdata")
training.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.tif") # for reference CRS of training grid
prediction.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.tif")

# ------------MODELLING STEPS--------------------# Kalman Filter>?????
# 1 run the models separately for each change year as is --- examine results.
# 2 then create new predictor features (time gap) and (historical change metrics) and see how this changes results 
# 3 Aggregate predicted changes with existing bathy data for all model years (may want to minus prediction from existing bathy to get just change, or add them together.... )
# 4 Fit temproal trend for cumulative change across all years 

# 1. Preparing the Model Script for Each Grid Tile
# This script loops through grid tiles intersecting the training data boundary, runs models for each year pair, and stores results as .rds files.

# MODEL TRAINING FUNCTIONS:---- 
# Function to split a tile into sub-grids
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
  
  # Assign IDs and return sf object
  sub_grid_ids <- paste0("Tile_", tile$tile, "_", 1:4)
  st_sf(tile = sub_grid_ids, geometry = st_sfc(sub_grids, crs = st_crs(tile)))
}

# Function to process grid tiles and prepare sub-grids
prepare_subgrids <- function(grid_gpkg, training_data, training.mask, output_dir) {
  cat("Preparing grid tiles and sub-grids...\n")
  
  grid_tiles <- st_read(grid_gpkg) %>% 
    st_transform(st_crs(training.mask))
  
  training_sf <- st_as_sf(training_data, coords = c("X", "Y"), crs = st_crs(training.mask))
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) split_tile(grid_tiles[i, ])))
  
  intersecting_sub_grids <- st_filter(sub_grids, st_union(training_sf))
  st_write(intersecting_sub_grids, file.path(output_dir, "intersecting_sub_grids.gpkg"))
  
  cat("Sub-grids prepared:", nrow(intersecting_sub_grids), "\n")
  return(intersecting_sub_grids)
}

# Function to train a model for a single sub-grid
train_single_subgrid <- function(sub_grid, training_sf, year_pairs, output_dir, save_training_data = TRUE) {
  sub_grid_id <- sub_grid$tile
  sub_grid_dir <- file.path(output_dir, sub_grid_id)
  dir.create(sub_grid_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("Processing Subgrid:", sub_grid_id, "\n")
  
  for (pair in year_pairs) {
    start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
    end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
    
    subgrid_training_data <- st_filter(training_sf, sub_grid) %>%
      filter(
        !is.na(!!sym(paste0("bathy_", start_year))) &
          !is.na(!!sym(paste0("bathy_", end_year)))
      ) %>%
      st_drop_geometry()
    
    if (nrow(subgrid_training_data) == 0) {
      cat("  No data for Subgrid:", sub_grid_id, "and Year Pair:", pair, "\n")
      next
    }
    
    if (save_training_data) {
      training_data_path <- file.path(sub_grid_dir, paste0("training_data_", pair, ".rds"))
      saveRDS(subgrid_training_data, training_data_path)
    }
    
    formula <- as.formula(paste0("b.change.", pair, " ~ ", paste(c(
      paste0("bathy_", start_year), paste0("bathy_", end_year),
      paste0("slope_", start_year), paste0("slope_", end_year),
      "grain_size", "sediment"
    ), collapse = " + ")))
    
    rf_model <- ranger::ranger(
      formula = formula,
      data = subgrid_training_data,
      num.trees = 500,
      mtry = floor(sqrt(length(all.vars(formula)))),
      importance = "impurity",
      write.forest = TRUE  # Ensure training data structure is saved
    )
    
    saveRDS(rf_model, file.path(sub_grid_dir, paste0("model_", pair, ".rds")))
    cat("  Model saved for Pair:", pair, "\n")
  }
}

# Function to process and train models for all sub-grids
process_all_subgrids <- function(sub_grids, training_sf, year_pairs, output_dir, save_training_data = TRUE) {
  cat("Starting model training for all sub-grids...\n")
  
  pb <- progress_bar$new(
    format = "[:bar] :percent in :elapsed | Subgrid :current of :total",
    total = nrow(sub_grids), clear = FALSE, width = 80
  )
  
  for (i in seq_len(nrow(sub_grids))) {
    tryCatch({
      train_single_subgrid(sub_grids[i, ], training_sf, year_pairs, output_dir, save_training_data)
    }, error = function(e) {
      cat("Error processing Subgrid:", sub_grids$tile[i], "-", e$message, "\n")
    })
    pb$tick()
  }
  
  cat("All sub-grids processed.\n")
}

# Step 1: Load or Define Parameters
grid_gpkg <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Modeling/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/GRID_ID"
training_data <- T.data  # Input dataset
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
training.mask <- training.mask  # Input mask

# Step 2: Process and Split Grid Tiles over training extent
sub_grids <- prepare_subgrids(
  grid_gpkg = grid_gpkg, 
  training_data = training_data, 
  training.mask = training.mask, 
  output_dir = output_dir
)

# Step 3: Train Models for All Subgrid Tiles
process_all_subgrids(
  sub_grids = sub_grids,  # Corrected argument name
  training_sf = st_as_sf(training_data, coords = c("X", "Y"), crs = st_crs(training.mask)),
  year_pairs = year_pairs,
  output_dir = output_dir
)


# Function to generate PDPs for a single tile
generate_fast_pdps_for_tile <- function(tile, year_pairs, output_dir) {
  tile_id <- tile$tile
  cat("Generating fast PDPs for Tile ID:", tile_id, "\n")
  
  tile_dir <- file.path(output_dir, paste0("Tile_", tile_id))
  dir.create(tile_dir, showWarnings = FALSE)  # Ensure the directory exists
  
  for (pair in year_pairs) {
    model_path <- file.path(tile_dir, paste0("model_", pair, ".rds"))
    training_data_path <- file.path(tile_dir, paste0("training_data_", pair, ".rds"))
    plot_path <- file.path(tile_dir, paste0("fast_pdp_", pair, ".png"))
    
    # Check if files exist
    if (!all(file.exists(c(model_path, training_data_path)))) {
      cat("  Skipping year pair:", pair, "due to missing files.\n")
      next
    }
    
    tryCatch({
      rf_model <- readRDS(model_path)
      tile_training_data <- readRDS(training_data_path)
      
      # Determine predictors dynamically
      start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
      end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
      dynamic_predictors <- c(
        paste0("bathy_", start_year), paste0("bathy_", end_year),
        paste0("slope_", start_year), paste0("slope_", end_year)
      )
      static_predictors <- c("grain_size", "sediment")
      predictors <- c(
        dynamic_predictors[dynamic_predictors %in% colnames(tile_training_data)],
        static_predictors[static_predictors %in% colnames(tile_training_data)]
      )
      
      # Skip if no predictors are valid
      if (length(predictors) == 0) {
        cat("  No valid predictors for year pair:", pair, "\n")
        next
      }
      
      # Save PDP plot
      png(plot_path, width = 1000, height = 800)
      par(mfrow = c(2, 3), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0))
      for (pred in predictors) {
        cat("  Plotting predictor:", pred, "\n")
        pred_range <- seq(
          min(tile_training_data[[pred]], na.rm = TRUE),
          max(tile_training_data[[pred]], na.rm = TRUE),
          length.out = 50
        )
        pdp_data <- tile_training_data[1, ][rep(1, 50), ]
        pdp_data[[pred]] <- pred_range
        
        predictions <- predict(rf_model, data = pdp_data)$predictions
        
        plot(
          pred_range, predictions,
          type = "l", col = "blue", lwd = 2,
          main = pred, xlab = pred, ylab = "b.change"
        )
      }
      title(paste("Tile:", tile_id, "Year Pair:", pair), outer = TRUE)
      dev.off()
      cat("  Fast PDP saved to:", plot_path, "\n")
    }, error = function(e) {
      cat("  Error processing year pair:", pair, "\n", "   Message:", e$message, "\n")
    })
  }
}
# Function to geenrate PDPs over all tiles
average_fast_pdps_all_tiles <- function(intersecting_tiles, year_pairs, input_dir, output_dir) {
  cat("Generating averaged PDPs across all intersecting tiles...\n")
  combined_pdp <- data.table()
  
  for (pair in year_pairs) {
    cat("\nProcessing year pair:", pair, "\n")
    temp_pdp <- data.table()
    
    for (i in seq_len(nrow(intersecting_tiles))) {
      tile <- intersecting_tiles[i, ]
      tile_id <- tile$tile
      cat("\n  Processing Tile ID:", tile_id, "\n")
      
      # Define paths for the current tile
      tile_dir <- file.path(input_dir, tile_id)
      model_path <- file.path(tile_dir, paste0("model_", pair, ".rds"))
      training_data_path <- file.path(tile_dir, paste0("training_data_", pair, ".rds"))
      
      # Debugging file paths
      cat("    Checking file paths:\n")
      cat("      Model Path:", model_path, "\n")
      cat("      Training Data Path:", training_data_path, "\n")
      
      if (!file.exists(model_path) || !file.exists(training_data_path)) {
        cat("    WARNING: Missing files for Tile ID:", tile_id, "\n")
        next
      }
      
      # Read files if they exist
      rf_model <- tryCatch(readRDS(model_path), error = function(e) {
        cat("    ERROR: Failed to load model for Tile ID:", tile_id, "\n")
        return(NULL)
      })
      tile_training_data <- tryCatch(readRDS(training_data_path), error = function(e) {
        cat("    ERROR: Failed to load training data for Tile ID:", tile_id, "\n")
        return(NULL)
      })
      
      if (is.null(rf_model) || is.null(tile_training_data)) {
        next
      }
      
      # Extract predictors
      start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
      end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
      dynamic_predictors <- c(
        paste0("bathy_", start_year), paste0("bathy_", end_year),
        paste0("slope_", start_year), paste0("slope_", end_year)
      )
      static_predictors <- c("grain_size", "sediment")
      predictors <- c(dynamic_predictors[dynamic_predictors %in% colnames(tile_training_data)], 
                      static_predictors[static_predictors %in% colnames(tile_training_data)])
      
      cat("    Identified predictors:", paste(predictors, collapse = ", "), "\n")
      if (length(predictors) == 0) {
        cat("    No valid predictors found. Skipping Tile ID:", tile_id, "\n")
        next
      }
      
      # Generate PDPs for each predictor
      for (pred in predictors) {
        pred_range <- seq(
          min(tile_training_data[[pred]], na.rm = TRUE),
          max(tile_training_data[[pred]], na.rm = TRUE),
          length.out = 50
        )
        pdp_data <- tile_training_data[1, ][rep(1, 50), ]
        pdp_data[[pred]] <- pred_range
        
        predictions <- tryCatch({
          predict(rf_model, data = pdp_data)$predictions
        }, error = function(e) {
          cat("    ERROR: Prediction failed for Tile ID:", tile_id, " Predictor:", pred, "\n")
          return(rep(NA, length(pred_range)))
        })
        
        temp_pdp <- rbind(temp_pdp, data.table(
          predictor = pred,
          x = pred_range,
          yhat = predictions,
          tile_id = tile_id
        ), use.names = TRUE, fill = TRUE)
      }
    }
    
    # Summarize PDP results for this year pair
    if (nrow(temp_pdp) > 0) {
      avg_pdp <- temp_pdp[, .(
        mean_yhat = mean(yhat, na.rm = TRUE),
        sd_yhat = sd(yhat, na.rm = TRUE),
        min_yhat = min(yhat, na.rm = TRUE),
        max_yhat = max(yhat, na.rm = TRUE)
      ), by = .(predictor, x)]
      avg_pdp[, year_pair := pair]
      combined_pdp <- rbind(combined_pdp, avg_pdp, use.names = TRUE, fill = TRUE)
    } else {
      cat("  No data for year pair:", pair, "\n")
    }
  }
  
  # Output results
  if (nrow(combined_pdp) > 0) {
    output_csv <- file.path(output_dir, "combined_pdp.csv")
    fwrite(combined_pdp, output_csv)
    cat("\nCombined PDP data saved to:", output_csv, "\n")
    
    for (pair in unique(combined_pdp$year_pair)) {
      pdp_data <- combined_pdp[year_pair == pair]
      gg <- ggplot(pdp_data, aes(x = x, y = mean_yhat)) +
        geom_ribbon(aes(ymin = min_yhat, ymax = max_yhat), fill = "grey80") +
        geom_line(color = "black", size = 1) +
        facet_wrap(~ predictor, scales = "free") +
        theme_minimal() +
        labs(title = paste("Averaged PDPs for Year Pair:", pair),
             x = "Predictor Value", y = "Mean Prediction")
      ggsave(file.path(output_dir, paste0("averaged_pdp_", pair, ".png")), gg, width = 10, height = 8)
      cat("  Averaged PDP plot saved for year pair:", pair, "\n")
    }
  } else {
    cat("No data available to create averaged PDPs.\n")
  }
}

# V2 normalised Y axis
generate_fast_pdps_for_tile <- function(tile, year_pairs, output_dir) {
  tile_id <- tile$tile
  cat("Generating fast PDPs for Tile ID:", tile_id, "\n")
  
  tile_dir <- file.path(output_dir, paste0("Tile_", tile_id))
  dir.create(tile_dir, showWarnings = FALSE)
  
  for (pair in year_pairs) {
    model_path <- file.path(tile_dir, paste0("model_", pair, ".rds"))
    training_data_path <- file.path(tile_dir, paste0("training_data_", pair, ".rds"))
    plot_path <- file.path(tile_dir, paste0("fast_pdp_", pair, ".png"))
    
    if (!all(file.exists(c(model_path, training_data_path)))) {
      cat("  Skipping year pair:", pair, "due to missing files.\n")
      next
    }
    
    tryCatch({
      rf_model <- readRDS(model_path)
      tile_training_data <- readRDS(training_data_path)
      
      start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
      end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
      dynamic_predictors <- c(
        paste0("bathy_", start_year), paste0("bathy_", end_year),
        paste0("slope_", start_year), paste0("slope_", end_year)
      )
      static_predictors <- c("grain_size", "sediment")
      predictors <- c(
        dynamic_predictors[dynamic_predictors %in% colnames(tile_training_data)],
        static_predictors[static_predictors %in% colnames(tile_training_data)]
      )
      
      if (length(predictors) == 0) {
        cat("  No valid predictors for year pair:", pair, "\n")
        next
      }
      
      png(plot_path, width = 1000, height = 800)
      par(mfrow = c(2, 3), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0))
      
      for (pred in predictors) {
        cat("  Plotting predictor:", pred, "\n")
        pred_range <- seq(
          min(tile_training_data[[pred]], na.rm = TRUE),
          max(tile_training_data[[pred]], na.rm = TRUE),
          length.out = 50
        )
        pdp_data <- tile_training_data[1, ][rep(1, 50), ]
        pdp_data[[pred]] <- pred_range
        
        predictions <- predict(rf_model, data = pdp_data)$predictions
        
        plot(
          pred_range, predictions,
          type = "l", col = "blue", lwd = 2,
          ylim = c(-2.5, 2.5),
          main = pred, xlab = pred, ylab = "b.change"
        )
      }
      title(paste("Tile:", tile_id, "Year Pair:", pair), outer = TRUE)
      dev.off()
      cat("  Fast PDP saved to:", plot_path, "\n")
    }, error = function(e) {
      cat("  Error processing year pair:", pair, "\n", "   Message:", e$message, "\n")
    })
  }
}
average_fast_pdps_all_tiles <- function(intersecting_tiles, year_pairs, input_dir, output_dir) {
  cat("Generating averaged PDPs across all intersecting tiles...\n")
  combined_pdp <- data.table()
  
  for (pair in year_pairs) {
    cat("\nProcessing year pair:", pair, "\n")
    temp_pdp <- data.table()
    
    for (i in seq_len(nrow(intersecting_tiles))) {
      tile <- intersecting_tiles[i, ]
      tile_id <- tile$tile
      cat("\n  Processing Tile ID:", tile_id, "\n")
      
      tile_dir <- file.path(input_dir, tile_id)
      model_path <- file.path(tile_dir, paste0("model_", pair, ".rds"))
      training_data_path <- file.path(tile_dir, paste0("training_data_", pair, ".rds"))
      
      cat("    Checking file paths:\n")
      cat("      Model Path:", model_path, "\n")
      cat("      Training Data Path:", training_data_path, "\n")
      
      if (!file.exists(model_path) || !file.exists(training_data_path)) {
        cat("    WARNING: Missing files for Tile ID:", tile_id, "\n")
        next
      }
      
      rf_model <- tryCatch(readRDS(model_path), error = function(e) {
        cat("    ERROR: Failed to load model for Tile ID:", tile_id, "\n")
        return(NULL)
      })
      tile_training_data <- tryCatch(readRDS(training_data_path), error = function(e) {
        cat("    ERROR: Failed to load training data for Tile ID:", tile_id, "\n")
        return(NULL)
      })
      
      if (is.null(rf_model) || is.null(tile_training_data)) {
        next
      }
      
      start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
      end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
      dynamic_predictors <- c(
        paste0("bathy_", start_year), paste0("bathy_", end_year),
        paste0("slope_", start_year), paste0("slope_", end_year)
      )
      static_predictors <- c("grain_size", "sediment")
      predictors <- c(dynamic_predictors[dynamic_predictors %in% colnames(tile_training_data)], 
                      static_predictors[static_predictors %in% colnames(tile_training_data)])
      
      cat("    Identified predictors:", paste(predictors, collapse = ", "), "\n")
      if (length(predictors) == 0) {
        cat("    No valid predictors found. Skipping Tile ID:", tile_id, "\n")
        next
      }
      
      for (pred in predictors) {
        pred_range <- seq(
          min(tile_training_data[[pred]], na.rm = TRUE),
          max(tile_training_data[[pred]], na.rm = TRUE),
          length.out = 50
        )
        pdp_data <- tile_training_data[1, ][rep(1, 50), ]
        pdp_data[[pred]] <- pred_range
        
        predictions <- tryCatch({
          predict(rf_model, data = pdp_data)$predictions
        }, error = function(e) {
          cat("    ERROR: Prediction failed for Tile ID:", tile_id, " Predictor:", pred, "\n")
          return(rep(NA, length(pred_range)))
        })
        
        temp_pdp <- rbind(temp_pdp, data.table(
          predictor = pred,
          x = pred_range,
          yhat = predictions,
          tile_id = tile_id
        ), use.names = TRUE, fill = TRUE)
      }
    }
    
    if (nrow(temp_pdp) > 0) {
      avg_pdp <- temp_pdp[, .(
        mean_yhat = mean(yhat, na.rm = TRUE),
        sd_yhat = sd(yhat, na.rm = TRUE),
        min_yhat = min(yhat, na.rm = TRUE),
        max_yhat = max(yhat, na.rm = TRUE)
      ), by = .(predictor, x)]
      avg_pdp[, year_pair := pair]
      combined_pdp <- rbind(combined_pdp, avg_pdp, use.names = TRUE, fill = TRUE)
    } else {
      cat("  No data for year pair:", pair, "\n")
    }
  }
  
  if (nrow(combined_pdp) > 0) {
    output_csv <- file.path(output_dir, "combined_pdp.csv")
    fwrite(combined_pdp, output_csv)
    cat("\nCombined PDP data saved to:", output_csv, "\n")
    
    for (pair in unique(combined_pdp$year_pair)) {
      pdp_data <- combined_pdp[year_pair == pair]
      gg <- ggplot(pdp_data, aes(x = x, y = mean_yhat)) +
        geom_ribbon(aes(ymin = min_yhat, ymax = max_yhat), fill = "grey80") +
        geom_line(color = "black", size = 1) +
        facet_wrap(~ predictor, scales = "free") +
        theme_minimal() +
        labs(title = paste("Averaged PDPs for Year Pair:", pair),
             x = "Predictor Value", y = "Mean Prediction") +
        ylim(c(-2.5, 2.5))
      ggsave(file.path(output_dir, paste0("averaged_pdp_", pair, ".png")), gg, width = 10, height = 8)
      cat("  Averaged PDP plot saved for year pair:", pair, "\n")
    }
  } else {
    cat("No data available to create averaged PDPs.\n")
  }
}


# PDP Generation Parameters
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID"
setwd(output_dir)
tile_dir <- file.path(output_dir, tile_id)
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
intersecting_tiles <- st_read(file.path(output_dir, "intersecting_sub_grids.gpkg"))
# PROCESS PDPs
print(output_dir)
average_fast_pdps_all_tiles(intersecting_tiles, year_pairs, input_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID",
                            output_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID")

# dat <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID/Tile_BH4S656W_4/model_2004_2006.rds")
# head(dat)

# ### ERROR HANDLING 
# # Verify the Model Files
# model_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID/Tile_BH4S2576_4/model_2015_2022.rds"
# rf_model <- readRDS(model_path)
# print(rf_model)
# 
# training_data_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID/Tile_BH4S2576_4/training_data_2004_2006.rds"
# TD <- readRDS(training_data_path)
# print(TD)
# 
# # generated error when running PDP script, error with predict function
# "Error in predict.ranger.forest(forest, data, predict.all, num.trees, type,  : 
#                                  Error: Invalid forest object."
# 
# ### Two tile test / different predict function test ####
# # Load necessary library
# library(ranger)  # Ensure ranger is loaded if you're using ranger::predict
# 
# # Function to generate PDPs for a single tile
# generate_fast_pdps_for_tile <- function(tile_name, year_pairs, input_dir, output_dir) {
#   cat("Generating fast PDPs for Tile:", tile_name, "\n")
#   
#   tile_dir <- file.path(input_dir, tile_name)  # Path to the tile directory
#   if (!dir.exists(tile_dir)) {
#     cat("Tile directory does not exist:", tile_dir, "\n")
#     return(NULL)
#   }
#   
#   # Create the output directory for this tile
#   tile_output_dir <- file.path(output_dir, tile_name)  # Fixed to avoid redundant 'Tile_' prefix
#   dir.create(tile_output_dir, showWarnings = FALSE)  # Ensure the directory exists
#   
#   for (pair in year_pairs) {
#     model_path <- file.path(tile_dir, paste0("model_", pair, ".rds"))
#     training_data_path <- file.path(tile_dir, paste0("training_data_", pair, ".rds"))
#     plot_path <- file.path(tile_output_dir, paste0("fast_pdp_", pair, ".png"))
#     
#     # Check if files exist
#     if (!all(file.exists(c(model_path, training_data_path)))) {
#       cat("  Skipping year pair:", pair, "due to missing files.\n")
#       next
#     }
#     
#     tryCatch({
#       rf_model <- readRDS(model_path)
#       tile_training_data <- readRDS(training_data_path)
#       
#       # Determine predictors dynamically
#       start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
#       end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
#       dynamic_predictors <- c(
#         paste0("bathy_", start_year), paste0("bathy_", end_year),
#         paste0("slope_", start_year), paste0("slope_", end_year)
#       )
#       static_predictors <- c("grain_size", "sediment")
#       predictors <- c(
#         dynamic_predictors[dynamic_predictors %in% colnames(tile_training_data)],
#         static_predictors[static_predictors %in% colnames(tile_training_data)]
#       )
#       
#       # Skip if no predictors are valid
#       if (length(predictors) == 0) {
#         cat("  No valid predictors for year pair:", pair, "\n")
#         next
#       }
#       
#       # Save PDP plot
#       png(plot_path, width = 1000, height = 800)
#       par(mfrow = c(2, 3), mar = c(4, 4, 2, 1), oma = c(0, 0, 4, 0))
#       for (pred in predictors) {
#         cat("  Plotting predictor:", pred, "\n")
#         pred_range <- seq(
#           min(tile_training_data[[pred]], na.rm = TRUE),
#           max(tile_training_data[[pred]], na.rm = TRUE),
#           length.out = 50
#         )
#         pdp_data <- tile_training_data[1, ][rep(1, 50), ]
#         pdp_data[[pred]] <- pred_range
#         
#         predictions <- predict(rf_model, data = pdp_data)$predictions
#         
#         plot(
#           pred_range, predictions,
#           type = "l", col = "blue", lwd = 2,
#           main = pred, xlab = pred, ylab = "b.change"
#         )
#       }
#       title(paste("Tile:", tile_name, "Year Pair:", pair), outer = TRUE)
#       dev.off()
#       cat("  Fast PDP saved to:", plot_path, "\n")
#     }, error = function(e) {
#       cat("  Error processing year pair:", pair, "\n", "   Message:", e$message, "\n")
#     })
#   }
# }
# 
# # Function to generate PDPs over all tiles
# average_fast_pdps_all_tiles <- function(tile_names, year_pairs, input_dir, output_dir) {
#   cat("Generating averaged PDPs for the specified tiles...\n")
#   combined_pdp <- data.table()
#   
#   for (pair in year_pairs) {
#     cat("\nProcessing year pair:", pair, "\n")
#     temp_pdp <- data.table()
#     
#     for (tile_name in tile_names) {
#       cat("\n  Processing Tile:", tile_name, "\n")
#       
#       tile_dir <- file.path(input_dir, tile_name)
#       model_path <- file.path(tile_dir, paste0("model_", pair, ".rds"))
#       training_data_path <- file.path(tile_dir, paste0("training_data_", pair, ".rds"))
#       
#       # Debugging file paths
#       cat("    Checking file paths:\n")
#       cat("      Model Path:", model_path, "\n")
#       cat("      Training Data Path:", training_data_path, "\n")
#       
#       if (!file.exists(model_path) || !file.exists(training_data_path)) {
#         cat("    WARNING: Missing files for Tile:", tile_name, "\n")
#         next
#       }
#       
#       # Read files if they exist
#       rf_model <- tryCatch(readRDS(model_path), error = function(e) {
#         cat("    ERROR: Failed to load model for Tile:", tile_name, "\n")
#         return(NULL)
#       })
#       tile_training_data <- tryCatch(readRDS(training_data_path), error = function(e) {
#         cat("    ERROR: Failed to load training data for Tile:", tile_name, "\n")
#         return(NULL)
#       })
#       
#       if (is.null(rf_model) || is.null(tile_training_data)) {
#         next
#       }
#       
#       start_year <- as.numeric(strsplit(pair, "_")[[1]][1])
#       end_year <- as.numeric(strsplit(pair, "_")[[1]][2])
#       dynamic_predictors <- c(
#         paste0("bathy_", start_year), paste0("bathy_", end_year),
#         paste0("slope_", start_year), paste0("slope_", end_year)
#       )
#       static_predictors <- c("grain_size", "sediment")
#       predictors <- c(dynamic_predictors[dynamic_predictors %in% colnames(tile_training_data)], 
#                       static_predictors[static_predictors %in% colnames(tile_training_data)])
#       
#       cat("    Identified predictors:", paste(predictors, collapse = ", "), "\n")
#       if (length(predictors) == 0) {
#         cat("    No valid predictors found. Skipping Tile:", tile_name, "\n")
#         next
#       }
#       
#       for (pred in predictors) {
#         pred_range <- seq(
#           min(tile_training_data[[pred]], na.rm = TRUE),
#           max(tile_training_data[[pred]], na.rm = TRUE),
#           length.out = 50
#         )
#         pdp_data <- tile_training_data[1, ][rep(1, 50), ]
#         pdp_data[[pred]] <- pred_range
#         
#         predictions <- tryCatch({
#           predict(rf_model, data = pdp_data)$predictions
#         }, error = function(e) {
#           cat("    ERROR: Prediction failed for Tile:", tile_name, " Predictor:", pred, "\n")
#           return(rep(NA, length(pred_range)))
#         })
#         
#         temp_pdp <- rbind(temp_pdp, data.table(
#           predictor = pred,
#           x = pred_range,
#           yhat = predictions,
#           tile_id = tile_name
#         ), use.names = TRUE, fill = TRUE)
#       }
#     }
#     
#     # Summarize PDP results
#     if (nrow(temp_pdp) > 0) {
#       avg_pdp <- temp_pdp[, .(
#         mean_yhat = mean(yhat, na.rm = TRUE),
#         sd_yhat = sd(yhat, na.rm = TRUE),
#         min_yhat = min(yhat, na.rm = TRUE),
#         max_yhat = max(yhat, na.rm = TRUE)
#       ), by = .(predictor, x)]
#       avg_pdp[, year_pair := pair]
#       combined_pdp <- rbind(combined_pdp, avg_pdp, use.names = TRUE, fill = TRUE)
#     } else {
#       cat("  No data for year pair:", pair, "\n")
#     }
#   }
#   
#   # Output results
#   if (nrow(combined_pdp) > 0) {
#     output_csv <- file.path(output_dir, "combined_pdp.csv")
#     fwrite(combined_pdp, output_csv)
#     cat("\nCombined PDP data saved to:", output_csv, "\n")
#     
#     for (pair in unique(combined_pdp$year_pair)) {
#       pdp_data <- combined_pdp[year_pair == pair]
#       gg <- ggplot(pdp_data, aes(x = x, y = mean_yhat)) +
#         geom_ribbon(aes(ymin = min_yhat, ymax = max_yhat), fill = "grey80") +
#         geom_line(color = "black", size = 1) +
#         facet_wrap(~ predictor, scales = "free") +
#         theme_minimal() +
#         labs(title = paste("Averaged PDPs for Year Pair:", pair),
#              x = "Predictor Value", y = "Mean b.change")
#       ggsave(file.path(output_dir, paste0("averaged_pdp_", pair, ".png")), gg, width = 10, height = 8)
#       cat("  Averaged PDP plot saved for year pair:", pair, "\n")
#     }
#   } else {
#     cat("No data available to create averaged PDPs.\n")
#   }
# }



year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")  # Example year pairs
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID"  # Path to your input directory with the tile subfolders
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID"  # Path to your output directory

# Test the function on the provided tiles
lapply(tile_names, function(tile_name) {
  generate_fast_pdps_for_tile(tile_name = tile_name, year_pairs = year_pairs, input_dir = input_dir, output_dir = output_dir)
})

# Run the averaged PDP generation
average_fast_pdps_all_tiles(tile_names = tile_names, year_pairs = year_pairs, input_dir = input_dir, output_dir = output_dir)


# Evaluate model performance (NEEDS UPDATED INTO sub grid format, performance per tile and average over area like PDP code) ----
# Each ranger model (e.g., b.change.2004_2006, b.change.2006_2010, etc.) has provided:
  # - Out-of-bag (OOB) prediction error and R-squared values, which indicate the quality of the model’s predictions.
  #  -Variable importance results that tell us which covariates were most influential in explaining the bathymetric changes over time.
# direct from model 
summarize_ranger_model(model)
# Validation ----
summary(models[["2004_2006"]])
summary(models[["2006_2010"]])
summary(models[["2010_2015"]])
summary(models[["2015_2022"]])

importance <- ranger::importance(models[["2004_2006"]])
importance <- ranger::importance(models[["2006_2010"]])
importance <- ranger::importance(models[["2010_2015"]])
importance <- ranger::importance(models[["2015_2022"]])

# # Load the saved training data
# training_data_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Modeling/GRID_ID/Tile_BH4S656W/training_data_2004_2006.rds"
# tile_training_data_no_geom <- readRDS(training_data_path)

# # Model summary
# cat("Ranger Model Summary:\n")
# print(rf_model)
# 
# # Variable importance
# cat("\nVariable Importance:\n")
# importance <- ranger::importance(rf_model)
# print(importance)
# 
# # Prediction error
# cat("\nPrediction Error:", rf_model$prediction.error, "\n")
# 
# # Number of trees and mtry
# cat("\nNumber of Trees:", rf_model$num.trees, "\n")
# cat("mtry:", rf_model$mtry, "\n")

#------------------------------------## BELOW CODE NOT FULLY  TESTED ###------------------------------------------#

# MODEL PREDICTION FUNCTIONS:---- 

# NOTES::::
# #2 AND #3 BELOW ARE OG functions, that we will maybe bring back in, and we may use to test functionality of different methods:
# Option 1# 
# need to decide if we are going to apply the prediction of the central grid tile over all year pairs of data, then take the average?
# Option 2 #
# or if we will try and compute the 10 year trend from model results over training data extent then apply that to the central grid tile
# averaged over surround tiles. There may be a slight difference in computational output... Option 1 would be using just data we
# have from all year pairs, option 2 would involve interpolation over missing years of data

### ANALYSIS OPTION 2 TESTING - Average Trends over 10 years ###----
# Step 1 - Generate sub grid tiles over full prediction extent, like we did for training extent
# Step 2 - Calculate 10 year trend from all training data & prediction

# prediction SUB Grid functions
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
  
  # Assign IDs and return sf object
  sub_grid_ids <- paste0("Tile_", tile$tile, "_", 1:4)
  st_sf(tile = sub_grid_ids, geometry = st_sfc(sub_grids, crs = st_crs(tile)))
}
prepare_subgrids <- function(grid_gpkg, P_data, prediction_mask, output_dir) {
  library(sf)
  library(raster)
  library(dplyr)
  
  cat("Preparing grid tiles and sub-grids...\n")
  
  # Read and transform grid tiles
  grid_tiles <- st_read(grid_gpkg) %>% 
    st_transform(crs = st_crs(prediction_mask))
  
  # Convert prediction mask (RasterLayer) to an sf object
  mask_extent <- as(prediction_mask, "SpatialPolygons")
  mask_sf <- st_as_sf(mask_extent) %>%
    st_set_crs(crs(prediction_mask))
  
  # Convert prediction data to an sf object
  prediction_sf <- st_as_sf(P_data, coords = c("X", "Y"), crs = st_crs(mask_sf))
  
  # Split grid tiles into sub-grids
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) {
    split_tile(grid_tiles[i, ], mask_extent = mask_sf)
  }))
  
  # Filter sub-grids that intersect with the mask
  intersecting_sub_grids <- st_filter(sub_grids, mask_sf)
  st_write(intersecting_sub_grids, file.path(output_dir, "intersecting_sub_grids.gpkg"))
  
  cat("Sub-grids prepared:", nrow(intersecting_sub_grids), "\n")
  return(intersecting_sub_grids)
}# inspect output in GIS
# Version 2 - copy of training sub grid function
# Function to process grid tiles and prepare sub-grids
prepare_subgrids <- function(grid_gpkg, prediction_data, prediction_mask, output_dir) {
  cat("Preparing grid tiles and sub-grids...\n")
  
  grid_tiles <- st_read(grid_gpkg) %>% 
    st_transform(st_crs(prediction_mask))
  
  training_sf <- st_as_sf(prediction_data, coords = c("X", "Y"), crs = st_crs(prediction.mask))
  sub_grids <- do.call(rbind, lapply(1:nrow(grid_tiles), function(i) split_tile(grid_tiles[i, ])))
  
  intersecting_sub_grids <- st_filter(sub_grids, st_union(training_sf))
  st_write(intersecting_sub_grids, file.path(output_dir, "intersecting_pred_sub_grids.gpkg"))
  
  cat("Sub-grids prepared:", nrow(intersecting_sub_grids), "\n")
  return(intersecting_sub_grids)
}

# Process Sub Grids----
# Call the updated prepare_subgrids function
prediction_sub_grids <- prepare_subgrids(
  grid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Modeling/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg", # original Blue Topo Grid
  prediction_data = P.data, 
  prediction_mask = prediction.mask, 
  output_dir = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Prediction.GRID_ID",
  append = F)

# Inspect the result
print(prediction_sub_grids) # OR check grid in GIS
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Prediction.GRID_ID")
save(prediction_sub_grids, filename = "prediction_sub_grids.gpkg")


# PREDICTION AND NEW 10 YEAR TREND CODE # 
# --- Define Parameters ---
input_dir <- ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID") # training grid tile folders, with models in each 
output_dir <- ("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Prediction.GRID_ID") #prediction grid tile folders, with predictions in each
training.extent.grid <- st_read(file.path(input_dir, "intersecting_sub_grids.gpkg")) #("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID/intersecting_sub_grids.gpkg")
# prediction.extent.grid <- st_read(file.path(output_dir, "intersecting_pred_sub_grids.gpkg")) #("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Prediction.GRID_ID/intersecting_pred_sub_grids.gpkg")
prediction_sub_grids <- st_read(file.path(output_dir, "intersecting_pred_sub_grids.gpkg"))
year_pairs <- c("2004_2006", "2006_2010", "2010_2015", "2015_2022")
prediction.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/prediction.mask.tif")
training.mask <- raster("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/training.mask.tif") # for reference CRS of training grid
T.data <- load("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/pm.training.dataset.010325.Rdata")
P.data <- load("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/pm.prediction.dataset.010325.Rdata")

# Extra 
# Check model data and structure to confirm analysis approach 
# M1 <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID/Tile_BH4RZ578_2/model_2006_2010.rds")
# M2 <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Grid_ID/Tile_BH4RZ578_2/training_data_2006_2010.rds")
# --- Function Definitions ---
# Compute 10-year trends from paired training data


### Step 1: Compute 10-Year Trends and write Geotiff for each tile (TRAINING DATA EXTENT ONLY)###
compute_10_year_trends <- function(input_dir, year_pairs, output_dir) {
  message("Step 1: Computing 10-Year Trends...")
  trend_data <- list() # Initialize list for trends
  
  # List all tile folders
  tile_folders <- list.dirs(input_dir, recursive = FALSE)
  message(paste("Found", length(tile_folders), "tile folders for processing."))
  
  for (tile_folder in tile_folders) {
    message(paste("Processing folder:", basename(tile_folder)))
    tile_trends <- list() # Trends for the current folder
    
    for (pair in year_pairs) {
      model_file <- file.path(tile_folder, paste0("model_", pair, ".rds"))
      training_file <- file.path(tile_folder, paste0("training_data_", pair, ".rds"))
      message(paste("Processing year pair:", pair, "in folder:", basename(tile_folder)))
      
      if (file.exists(model_file) && file.exists(training_file)) {
        model <- readRDS(model_file)
        training_data <- readRDS(training_file)
        
        # Check for predictions in the model
        if (!"predictions" %in% names(model)) {
          warning(paste("Predictions missing in model file:", model_file))
          next
        }
        
        # Extract predictions
        predictions <- model$predictions
        
        # Ensure FID alignment and include X, Y
        training_data <- training_data %>%
          filter(FID %in% unique(training_data$FID)) %>%
          arrange(FID) %>%
          select(FID, X, Y) # Retain X, Y, and FID columns
        
        # Validate predictions length against training_data rows
        if (length(predictions) != nrow(training_data)) {
          warning(paste("Mismatch in predictions and training data for:", pair))
          next
        }
        
        # Add predictions to training data
        training_data <- training_data %>%
          mutate(predicted_b_change = predictions)
        
        # Calculate trends for the year pair
        trends <- training_data %>%
          group_by(FID, X, Y) %>%
          summarise(
            cumulative_change = sum(predicted_b_change, na.rm = TRUE),
            average_change = mean(predicted_b_change, na.rm = TRUE),
            .groups = "drop"
          )
        
        tile_trends[[pair]] <- trends
      } else {
        warning(paste("Model or training data file not found for year pair:", pair, "in folder:", basename(tile_folder)))
      }
    }
    
    # Combine trends across all year pairs for this tile
    combined_tile_trends <- bind_rows(tile_trends, .id = "year_pair")
    trend_data[[basename(tile_folder)]] <- combined_tile_trends
    
    # Create rasters and save to GeoTIFF
    if (nrow(combined_tile_trends) > 0) {
      # Rasterize cumulative change
      raster_cumulative <- rasterFromXYZ(combined_tile_trends %>% select(X, Y, cumulative_change))
      cumulative_tiff_path <- file.path(tile_folder, "cumulative_change.tiff")
      writeRaster(raster_cumulative, cumulative_tiff_path, format = "GTiff", overwrite = TRUE)
      
      # Rasterize average change
      raster_average <- rasterFromXYZ(combined_tile_trends %>% select(X, Y, average_change))
      average_tiff_path <- file.path(tile_folder, "average_change.tiff")
      writeRaster(raster_average, average_tiff_path, format = "GTiff", overwrite = TRUE)
      
      message(paste("GeoTIFFs saved to:", tile_folder))
    }
  }
  
  # Combine all trends across all tiles
  all_trends <- bind_rows(trend_data, .id = "tile")
  
  # Aggregate trends across tiles
  final_trend <- all_trends %>%
    group_by(FID, X, Y) %>%
    summarise(
      avg_cumulative_change = mean(cumulative_change, na.rm = TRUE),
      avg_average_change = mean(average_change, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Save results
  output_file <- file.path(output_dir, "trend_analysis_10_years.rds")
  saveRDS(final_trend, file = output_file)
  message(paste("10-Year Trends saved to:", output_file))
  
  return(final_trend)
} #older 

compute_10_year_trends <- function(input_dir, year_pairs, output_dir, T_data) {
  message("Step 1: Computing 10-Year Trends...")
  trend_data <- list() # Initialize list for trends
  
  # List all tile folders
  tile_folders <- list.dirs(input_dir, recursive = FALSE)
  message(paste("Found", length(tile_folders), "tile folders for processing."))
  
  for (tile_folder in tile_folders) {
    message(paste("Processing folder:", basename(tile_folder)))
    tile_trends <- list() # Trends for the current folder
    tile_id <- basename(tile_folder)
    
    for (pair in year_pairs) {
      model_file <- file.path(tile_folder, paste0("model_", pair, ".rds"))
      training_file <- file.path(tile_folder, paste0("training_data_", pair, ".rds"))
      message(paste("Processing year pair:", pair, "in folder:", basename(tile_folder)))
      
      if (file.exists(model_file) && file.exists(training_file)) {
        model <- readRDS(model_file)
        training_data <- readRDS(training_file)
        
        # Check for predictions in the model
        if (!"predictions" %in% names(model)) {
          warning(paste("Predictions missing in model file:", model_file))
          next
        }
        
        # Extract predictions
        predictions <- model$predictions
        
        # Ensure FID alignment by joining training_data with T_data to get X, Y
        training_data <- training_data %>%
          inner_join(T_data %>% select(FID, X, Y), by = "FID") %>%
          arrange(FID)
        
        # Validate predictions length against training_data rows
        if (length(predictions) != nrow(training_data)) {
          warning(paste("Mismatch in predictions and training data for:", pair))
          next
        }
        
        # Add predictions to training data
        training_data <- training_data %>%
          mutate(predicted_b_change = predictions)
        
        # Calculate trends for the year pair
        trends <- training_data %>%
          group_by(FID, X, Y) %>%
          summarise(
            cumulative_change = sum(predicted_b_change, na.rm = TRUE),
            average_change = mean(predicted_b_change, na.rm = TRUE),
            .groups = "drop"
          )
        
        trends <- trends %>% mutate(Tile_ID = tile_id) # Add Tile_ID column
        
        tile_trends[[pair]] <- trends
      } else {
        warning(paste("Model or training data file not found for year pair:", pair, "in folder:", basename(tile_folder)))
      }
    }
    
    # Combine trends across all year pairs for this tile
    combined_tile_trends <- bind_rows(tile_trends, .id = "year_pair")
    trend_data[[basename(tile_folder)]] <- combined_tile_trends
    
    # Create rasters and save to GeoTIFF
    if (nrow(combined_tile_trends) > 0) {
      # Rasterize cumulative change
      raster_cumulative <- rasterFromXYZ(combined_tile_trends %>% select(X, Y, cumulative_change))
      cumulative_tiff_path <- file.path(tile_folder, "cumulative_change.tiff")
      writeRaster(raster_cumulative, cumulative_tiff_path, format = "GTiff", overwrite = TRUE)
      
      # Rasterize average change
      raster_average <- rasterFromXYZ(combined_tile_trends %>% select(X, Y, average_change))
      average_tiff_path <- file.path(tile_folder, "average_change.tiff")
      writeRaster(raster_average, average_tiff_path, format = "GTiff", overwrite = TRUE)
      
      message(paste("GeoTIFFs saved to:", tile_folder))
    }
  }
  
  # Combine all trends across all tiles into a final dataframe
  final_trend <- bind_rows(trend_data, .id = "Tile_ID")
  
  # Save results
  output_file <- file.path(output_dir, "trend_analysis_10_years.rds")
  saveRDS(final_trend, file = output_file)
  message(paste("10-Year Trends saved to:", output_file))
  
  return(final_trend)
} # used / worked

compute_10_year_trends <- function(input_dir, year_pairs, output_dir, T.data) {

  message("Step 1: Computing 10-Year Trends...")
  
  trend_data <- list()
  
  # List all tile folders
  tile_folders <- list.dirs(input_dir, recursive = FALSE)
  message(paste("Found", length(tile_folders), "tile folders for processing."))
  
  for (tile_folder in tile_folders) {
    message(paste("Processing folder:", basename(tile_folder)))
    tile_id <- basename(tile_folder)
    all_year_data <- list()
    
    for (pair in year_pairs) {
      model_file <- file.path(tile_folder, paste0("model_", pair, ".rds"))
      training_file <- file.path(tile_folder, paste0("training_data_", pair, ".rds"))
      
      if (file.exists(model_file) && file.exists(training_file)) {
        model <- readRDS(model_file)
        training_data <- readRDS(training_file)
        
        if (!"predictions" %in% names(model)) {
          warning(paste("Predictions missing in model file:", model_file))
          next
        }
        
        predictions <- model$predictions
        actual_change_column <- paste0("b.change.", pair)
        
        # Ensure the actual change column exists in T.data
        if (!(actual_change_column %in% colnames(T.data))) {
          stop(paste("Column", actual_change_column, "not found in T.data."))
        }
        
        # Merge training data with T.data and filter valid rows
        year_data <- training_data %>%
          inner_join(
            T.data %>%
              select(FID, X, Y, actual_change = all_of(actual_change_column)),
            by = "FID"
          ) %>%
          mutate(
            predicted_b_change = predictions,
            actual_b_change = actual_change
          ) %>%
          select(FID, X, Y, predicted_b_change, actual_b_change) %>%
          filter(!is.na(predicted_b_change) | !is.na(actual_b_change))  # Remove rows with no data
        
        all_year_data[[pair]] <- year_data
      } else {
        warning(paste("Model or training data file not found for year pair:", pair, "in folder:", basename(tile_folder)))
      }
    }
    
    # Combine year data and aggregate trends
    combined_data <- bind_rows(all_year_data)
    
    aggregated_trends <- combined_data %>%
      group_by(FID, X, Y) %>%
      summarise(
        pred_avg_change = mean(predicted_b_change, na.rm = TRUE),
        actual_avg_change = mean(actual_b_change, na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(Tile_ID = tile_id)
    
    trend_data[[tile_id]] <- aggregated_trends
    
    # Create and save rasters for this tile
    raster_actual <- rasterFromXYZ(aggregated_trends %>% select(X, Y, actual_avg_change))
    raster_predicted <- rasterFromXYZ(aggregated_trends %>% select(X, Y, pred_avg_change))
    
    writeRaster(raster_actual, file.path(tile_folder, "actual_avg_change.tiff"), format = "GTiff", overwrite = TRUE)
    writeRaster(raster_predicted, file.path(tile_folder, "pred_avg_change.tiff"), format = "GTiff", overwrite = TRUE)
  }
  
  # Combine all tile trends
  final_trend <- bind_rows(trend_data)
  
  # Save the aggregated trend data
  saveRDS(final_trend, file = file.path(output_dir, "trend_analysis_10_years.rds"))
  message(paste("10-Year Trends saved to:", file.path(output_dir, "trend_analysis_10_years.rds")))
  
  # Save rasters for full extent
  raster_actual_all <- rasterFromXYZ(final_trend %>% select(X, Y, actual_avg_change))
  raster_predicted_all <- rasterFromXYZ(final_trend %>% select(X, Y, pred_avg_change))
  
  writeRaster(raster_actual_all, file.path(output_dir, "actual_avg_change_all.tiff"), format = "GTiff", overwrite = TRUE)
  writeRaster(raster_predicted_all, file.path(output_dir, "pred_avg_change_all.tiff"), format = "GTiff", overwrite = TRUE)
  
  return(final_trend)
} #NOICE - keeper

# Newest

### Step 2: Predict Changes - find matches between the trend data and P.data and apply predictions over prediction extent ###
trend_data <- trend_results
predict_changes <- function(P.data, trend_data, current_year, output_dir) {
  message("Step 2: Predicting Changes...")
  print(paste("Input data contains", nrow(P.data), "rows."))
  
  P_data <- P.data %>%
    mutate(survey_age = current_year - survey_end_date) %>%
    left_join(trend_data, by = "FID") %>%
    mutate(predicted_depth_change = (avg_average_change / 10) * (survey_age / 10)) %>%
    mutate(predicted_depth_change = ifelse(is.na(predicted_depth_change), 0, predicted_depth_change))
  
  output_file <- file.path(output_dir, "predicted_depth_changes.rds")
  saveRDS(P_data, file = output_file)
  message(paste("Predictions saved to:", output_file))
  
  return(P_data)
} # old 


predict_changes <- function(P_data, trend_data, current_year, output_dir) {
  library(dplyr)
  
  message("Step 2: Predicting Changes...")
  print(paste("Input data contains", nrow(P_data), "rows."))
  
  # Correctly calculate survey age
  P_data <- P_data %>%
    mutate(survey_age = pmax(0, current_year - survey_end_date))  # Ensure no negative ages
  
  # Predict changes by matching P.data to trend_data
  P_data <- P_data %>%
    rowwise() %>%
    mutate(
      match = list(
        trend_data %>%
          filter(
            between(bt.bathy, bt.bathy - 0.5, bt.bathy + 0.5),  # Allow some tolerance in matching
            between(bt.slope, bt.slope - 0.1, bt.slope + 0.1),
            (grain_size == grain_size | is.na(grain_size)),
            (sediment == sediment | is.na(sediment))
          ) %>%
          summarise(
            pred_avg_change = mean(pred_avg_change, na.rm = TRUE),
            actual_avg_change = mean(actual_avg_change, na.rm = TRUE)
          )
      )
    ) %>%
    unnest(cols = c(match))  # Expand the nested data
  
  # Calculate depth changes
  P_data <- P_data %>%
    mutate(
      predicted_depth_change = (pred_avg_change / 10) * survey_age,
      actual_depth_change = (actual_avg_change / 10) * survey_age
    ) %>%
    mutate(
      predicted_depth_change = ifelse(is.na(predicted_depth_change), 0, predicted_depth_change),
      actual_depth_change = ifelse(is.na(actual_depth_change), 0, actual_depth_change)
    )
  
  # Select relevant columns for output
  P_data <- P_data %>%
    select(
      FID, X, Y, survey_age, pred_avg_change, actual_avg_change,
      predicted_depth_change, actual_depth_change, bt.bathy, bt.slope, grain_size, sediment, survey_end_date
    )
  
  # Save results
  output_file <- file.path(output_dir, "predicted_depth_changes.rds")
  saveRDS(P_data, file = output_file)
  message(paste("Predictions saved to:", output_file))
  
  return(P_data)
}
# newest


### Step 3: Apply Moving Window ###
apply_moving_window <- function(P_data, grid_size, output_dir) {
  message("Step 3: Applying Moving Window Analysis...")
  print(paste("Input data contains", nrow(P_data), "rows."))
  
  grid_tiles <- unique(P_data %>% select(X, Y) %>% mutate(
    grid_X = floor(X / grid_size) * grid_size,
    grid_Y = floor(Y / grid_size) * grid_size
  ))
  
  predictions <- list()
  
  for (tile in seq_len(nrow(grid_tiles))) {
    tile_data <- P_data %>%
      filter(
        X >= grid_tiles$grid_X[tile] & X < (grid_tiles$grid_X[tile] + grid_size),
        Y >= grid_tiles$grid_Y[tile] & Y < (grid_tiles$grid_Y[tile] + grid_size)
      )
    
    if (nrow(tile_data) > 0) {
      tile_prediction <- tile_data %>%
        summarise(
          avg_predicted_change = mean(predicted_depth_change, na.rm = TRUE)
        )
      
      predictions[[tile]] <- tile_prediction
    }
  }
  
  grid_predictions <- bind_rows(predictions)
  output_file <- file.path(output_dir, "grid_predictions.rds")
  saveRDS(grid_predictions, file = output_file)
  message(paste("Grid Predictions saved to:", output_file))
  
  return(grid_predictions)
}

### Workflow Function (Runs all 3 prediction functions sequentially) ###
workflow <- function(input_dir, output_dir, T.data.pairs, year_pairs, P_data, current_year) {
  # Step 1: Compute 10-Year Trends
  trend_data <- compute_10_year_trends(input_dir, T.data.pairs, year_pairs, output_dir)
  
  # Step 2: Predict Changes
  predictions <- predict_changes(P_data, trend_data, current_year, output_dir)
  
  # Step 3: Apply Moving Window
  grid_predictions <- apply_moving_window(predictions, grid_size = 5, output_dir)
  
  return(grid_predictions)
}

### Example Usage ###

current_year <- 2024

# Call each function separately for debugging
trend_results <- compute_10_year_trends(input_dir, year_pairs, output_dir, T.data)
predictions <- predict_changes(P.data, trend_data, current_year, output_dir)
grid_predictions <- apply_moving_window(predictions, grid_size = 5, output_dir)

### RELOAD DATA FROM ABOVE FUNCTIONS AND INSPECT / MAKE SPATIAL (Steps 1 & 2 so far)
trend_results <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Prediction.Grid_ID/trend_analysis_10_years.rds")
predictions <- readRDS("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Grid_Development/Prediction.Grid_ID/predicted_depth_changes.rds")



# # ## DOUBLE CHECK IT SPATIALLY! (Does the primary[earliest] raster look like how it was input before we perform analysis??)
test.rast <- rasterFromXYZ(data.frame(x = predictions[,"X"],
                                     y = predictions[,"Y"],
                                     z = predictions[, "predicted_depth_change"]),
                          crs = crs(prediction.mask))

setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data/model.out")
writeRaster(test.rast, filename = "prediction.test.output010225_v2.tif")



