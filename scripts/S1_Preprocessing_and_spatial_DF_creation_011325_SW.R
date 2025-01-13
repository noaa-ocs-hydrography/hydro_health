# This script is for the pre processing of raw NOW Coast data and other model predictors as spatial rasters
# Raw data downlaoded into a single folder comprising of GeoTiffs and seperate Raster Attribute Tables .xml files.
### NOTE ### in its current form 01-13-25 this processes large rasters or raster mosaics as a whole extent of the Pilot Model 
# Training or Prediction extent - after the completion of the pilot model refinement, this code will need to be modified
# to bring in the data in raw tile format as it is from now coast. This will also mean that other predictors i.e hurricanes, TSM,
# sediment and gravel will need to be split out over the tiles / sub grid. 


# PRE-PROCESSING STEPS:
# extract all survey end date data from xml (Raster Attribute Table) and create standalone rasters
# merge all indavidual survey end date rasters into a combined raster (currently mosaic via ArcGIS)
# load all JALBTCX bathymetry mosaic rasters and associated slope rasters
# load in other model variables (sediment type, sediment grain size - other contributers surface current to come)
# standardise rasters and clip to both prediction.mask and training.mask: ensure all datasets are in the same crs / projection, resolution (5m), and clip to different masks to ensure same size
# ensure all bathy elevation values above 0m are removed after clipping, if not filter (0m mark varies on boundary)
# stack all data and convert to Spatial Points Dataframe / for proeccesing / save

# Load Packages
require(raster); require(terra);require(xml2)
require(dplyr); require(sp); 
library(tidyr)

## P1 Survey End Date Extraction MULTIPLE .xml file, from blue topo data for each grid tile :----

# Define input and output directories
input_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_Data/Modeling/UTM17"
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_Data/Modeling/survey_date_end"
kml_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_Data/Modeling/RATs"

### naming convention of .tiff and .xml files much match!! ###

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

## P2 Standardize all rasters (PREDICTION Extent first as its larger)----
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
# Convert mask coordinates to a SpatialPoints object
mask_points <- sp::SpatialPoints(coords = mask_coords, proj4string = crs(mask))
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Extract values from raster using spatial points
  raster_values <- raster::extract(ras, mask_points)
  
  # Combine mask spatial points with raster values
  temp_df <- cbind(mask_coords, value = raster_values)
  temp_ras <- raster::rasterFromXYZ(temp_df, crs = crs_mask)
  
  # Apply "_bathy" condition
  if (grepl("_bathy", source_name)) {
    temp_ras[temp_ras > 0] <- NA
  }
  
  # Save the final raster
  output_name <- file.path(output_dir, paste0(source_name, ".tif"))
  raster::writeRaster(temp_ras, filename = output_name, overwrite = TRUE)
  
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
# Convert mask coordinates to a SpatialPoints object
mask_points <- sp::SpatialPoints(coords = mask_coords, proj4string = crs(mask))
for (i in 1:length(ras250m)) {
  ras <- ras250m[[i]]
  source_name <- tools::file_path_sans_ext(basename(f250m[i]))
  
  # Extract values from raster using spatial points
  raster_values <- raster::extract(ras, mask_points)
  
  # Combine mask spatial points with raster values
  temp_df <- cbind(mask_coords, value = raster_values)
  temp_ras <- raster::rasterFromXYZ(temp_df, crs = crs_mask)
  
  # Apply "_bathy" condition
  if (grepl("_bathy", source_name)) {
    temp_ras[temp_ras > 0] <- NA
  }
  
  # Save the final raster
  output_name <- file.path(output_dir, paste0(source_name, ".tif"))
  raster::writeRaster(temp_ras, filename = output_name, overwrite = TRUE)
  
  cat("Processed and saved:", source_name, "\n")
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
# We create 3 spatial dataframes here:
                                      #1 M1 (MASTER DATASET) contains all data colomns for both prediction and training datasets 
                                      #2 P.data (Prediction dataset) Contains all the same data as the training datasets except - all temporal bathy and slope data is subsituted with blue topo
                                      # data, which provides the last surveyed depth and slope, and the supporting xml. file provides the survey age for temporal coherance
                                      #3 T.data (Training dataset) All non temporal and temporal predictors for each survey year (i.e., bathy, slope, hurricanes, turbidity)]

## ----MASTER DATASET (ALL DATA): ALREADY clipped TO P.mask extent from standardise rasters step above-----
## Take about 50 minutes to process ##
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
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Data")
saveRDS(M1.df, file = "pm.master_dataset.011325.Rds")
Sys.time()
# load("pm.prediction.data.111524.Rdata")

##---------- prediction data -------
# Assuming M1.spdf is your spatial points dataframe and is loaded # Get column names of 
col_names <- colnames(M1.df) # Filter out column names that start with "20" 
filtered_col_names <- col_names[!grepl("^X20", col_names) | grepl("hurricane|tsm", col_names, ignore.case = TRUE)] 
P.data <- M1.df[, filtered_col_names] # Print the resulting subset to verify 
head(P.data)
P.data <- P.data[, c(18,19,20,13,14,17,15,16,1:12)] # re-shuffle order 
colnames(P.data) <- c("X", "Y", "FID","bt.bathy", "bt.slope","survey_end_date","grain_size","sediment", # bt = blue topo
                      "hurr_count_2004_2006","hurr_strength_2004_2006","tsm_2004_2006",
                      "hurr_count_2006_2010","hurr_strength_2006_2010","tsm_2006_2010",
                      "hurr_count_2010_2015","hurr_strength_2010_2015","tsm_2010_2015",
                      "hurr_count_2015_2022","hurr_strength_2015_2022","tsm_2015_2022")#  X, Y , FID always at the front - Define the new column order and names # bt = blue topo
glimpse(P.data)
setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
saveRDS(P.data, file = "pm.prediction.dataset.011325.Rds")
 

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

# discard columns from master dataset not needed (post ivan data removed as too many NAs)
T.data <- T.data   %>%
  select (-training.mask, -blue_topo_bathy_111924, -blue_topo_slope - X2004_PostIvan_bathy, -X2004_PostIvan_slope)

# Re-shuffle order 
T.data <- T.data[, c(1,2,3,7,8,12,13,17,18,22,23,24,25,26,27,28,4,5,6,9,10,11,14,15,16,19,20,21)]

#### COL NAMES MUST ALWAYS BE ALPHA starting, remove numerics from first section of name....!!!
colnames(T.data) <- c("X","Y","FID","bathy_2004","slope_2004", "bathy_2006","slope_2006",
                      "bathy_2010", "slope_2010", "bathy_2015","slope_2015",
                      "bathy_2022","slope_2022", "grain_size","sediment", "survey_end_date",
                      "hurr_count_2004_2006","hurr_strength_2004_2006","tsm_2004_2006",
                      "hurr_count_2006_2010","hurr_strength_2006_2010","tsm_2006_2010",
                      "hurr_count_2010_2015","hurr_strength_2010_2015","tsm_2010_2015",
                      "hurr_count_2015_2022","hurr_strength_2015_2022","tsm_2015_2022")


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
T.data <- T.data[, c(1:13,29:32,14:28)] # re-shuffle before saving 

setwd("N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/data")
saveRDS(T.data, file = "pm.training.dataset.011325.Rds") # save as .RDS to be compatible with python