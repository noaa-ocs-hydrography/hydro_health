# Stephanie Watson 9/16/24

# Code section to loop through multiple lists of Geotifs and pre-process the data for further analysis
# tifs are re-projected, resampled to new 5 x 5 m resolution and all values greater than 0 removed. 


# ****NOTE - if native data resolution in is feet, use CODE SECTION 1, if its in meters, use CODE SECTION 2 - data, data will need to 
#           be saved in different working directories. 


#### CODE SECTION 1:USE IF RAW DATA ***NEEDS CONVERTED FROM FT to METERS***----
# FUNCTION 1 to process a list of rasters: convert, project, resample, and mask values
process_raster_function <- function(raster_list, output_folder, scaling_factor = 0.3048006096) {
  for (raster in raster_list) {
    # Extract the source path (file name) from the raster object
    raster_path <- sources(raster)[1]  # Get the source file path
    
    # Print initial summary for validation
    message("Processing: ", basename(raster_path))
    print(summary(raster))
    
    # Step 1: Convert units from feet to meters by multiplying values by the scaling factor
    message("Converting raster units from feet to meters...")
    raster_meters <- raster * scaling_factor
    
    # Step 2: Change the projection to NAD83 UTM Zone 17N, assuming raster is in survey feet (check projection)
    target_crs <- "EPSG:26917"  # NAD83 UTM Zone 17N
    message("Reprojecting raster to NAD83 UTM Zone 17N...")
    raster_projected <- project(raster_meters, target_crs)
    
    # Step 3: Resample the raster to 5x5 meter resolution
    message("Resampling raster to 5x5 meter resolution...")
    target_res <- c(5, 5)  # 5x5 meters
    raster_resampled <- resample(raster_projected, rast(res=target_res, extent=raster_projected), method="bilinear")
    
   
    # Step 4: Convert all values greater than 0 to NaN, (keep only negative bathymetric values)
    message("Converting all values greater than 0 to NaN...")
    raster_resampled[raster_resampled > 0] <- NA
    # message("Setting positive values to NaN...")
    # raster_resampled <- classify(raster_resampled, matrix(c(0, Inf, NA), ncol=3, byrow=TRUE))
    
    # Validate by printing summary and checking if the units and resolution have been changed
    print(summary(raster_resampled))
    
    # Step 5: Write the raster to file using LZ77 compression (lossless)
    compressed_output <- file.path(output_folder, paste0(tools::file_path_sans_ext(basename(raster_path)), "_processed.tif"))
    message("Saving processed raster as compressed GeoTIFF: ", compressed_output)
    writeRaster(raster_resampled, filename=compressed_output, overwrite=TRUE)
  }
}

## SET UP WORKING DIRECTORIES 

library(terra)

# Step 1: Function to load rasters for each survey year
load_rasters <- function(year_folder) {
  # Load all rasters in the year folder
  rasters <- list.files(year_folder, pattern = "\\.tif$", full.names = TRUE)
  raster_list <- lapply(rasters, rast)
  
  # Check if raster_list is empty
  if (length(raster_list) == 0) {
    warning(paste("No rasters found in", year_folder))
    return(NULL)
  }
  return(raster_list)
}

# Base directory where the survey year folders are located
base_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/Florida_all_survey_years_raw"

# Output directory for processed rasters
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/Pre_processed_tiffs"

# Step 2: Get the list of survey year folders
survey_year_folders <- list.dirs(base_dir, recursive = FALSE)

### APPLY THE FUNCTION 

# Step 3: Loop through each survey year folder and process its rasters
for (year_folder in survey_year_folders) {
  
  # Load rasters for the current survey year
  raster_list <- load_rasters(year_folder)
  
  # Skip if raster_list is NULL (no rasters found)
  if (is.null(raster_list)) {
    message("Skipping folder: ", year_folder)
    next
  }
  
  # Get survey year from the folder name
  survey_year <- basename(year_folder)
  
  # Generate the output folder for the current survey year
  output_folder <- file.path(output_dir, survey_year)
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }
  
  # Process each raster in the list
  process_raster_function(raster_list, output_folder)
}



#### CODE SECTION 2: USE IF RAW DATA ***IS ALREADY in METERS***----
# FUNCTION 1 to process a list of rasters: convert, project, resample, and mask values--
process_raster_function <- function(raster_list, output_folder) {
  for (raster in raster_list) {
    # Extract the source path (file name) from the raster object
    raster_path <- sources(raster)[1]  # Get the source file path
    
    # Print initial summary for validation
    message("Processing: ", basename(raster_path))
    print(summary(raster))
    
    
    # Step 1: Change the projection to NAD83 UTM Zone 17N, assuming raster is in survey feet (check projection)
    target_crs <- "EPSG:26917"  # NAD83 UTM Zone 17N
    message("Reprojecting raster to NAD83 UTM Zone 17N...")
    raster_projected <- project(raster, target_crs)
    
    # Step 2: Resample the raster to 5x5 meter resolution
    message("Resampling raster to 5x5 meter resolution...")
    target_res <- c(5, 5)  # 5x5 meters
    raster_resampled <- resample(raster_projected, rast(res=target_res, extent=raster_projected), method="bilinear")
    
    
    # Step 3: Convert all values greater than 0 to NaN, (keep only negative bathymetric values)
    message("Converting all values greater than 0 to NaN...")
    raster_resampled[raster_resampled > 0] <- NA
    # message("Setting positive values to NaN...")
    # raster_resampled <- classify(raster_resampled, matrix(c(0, Inf, NA), ncol=3, byrow=TRUE))
    
    # Validate by printing summary and checking if the units and resolution have been changed
    print(summary(raster_resampled))
    
    # Step 4: Write the raster to file using LZ77 compression (lossless)
    compressed_output <- file.path(output_folder, paste0(tools::file_path_sans_ext(basename(raster_path)), "_processed.tif"))
    message("Saving processed raster as compressed GeoTIFF: ", compressed_output)
    writeRaster(raster_resampled, filename=compressed_output, overwrite=TRUE)
  }
}

## SET UP WORKING DIRECTORIES 

library(terra)

# Step 1: Function to load rasters for each survey year
load_rasters <- function(year_folder) {
  # Load all rasters in the year folder
  rasters <- list.files(year_folder, pattern = "\\.tif$", full.names = TRUE)
  raster_list <- lapply(rasters, rast)
  
  # Check if raster_list is empty
  if (length(raster_list) == 0) {
    warning(paste("No rasters found in", year_folder))
    return(NULL)
  }
  return(raster_list)
}

# Base directory where the survey year folders are located (UNITS IN METERS ONLY!)
base_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/Florida_all_survey_years_raw/native_units_m"
# Output directory for processed rasters
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/Pre_processed_tiffs"

# Step 2: Get the list of survey year folders
survey_year_folders <- list.dirs(base_dir, recursive = FALSE)

### APPLY THE FUNCTION 

# Step 3: Loop through each survey year folder and process its rasters
for (year_folder in survey_year_folders) {
  
  # Load rasters for the current survey year
  raster_list <- load_rasters(year_folder)
  
  # Skip if raster_list is NULL (no rasters found)
  if (is.null(raster_list)) {
    message("Skipping folder: ", year_folder)
    next
  }
  
  # Get survey year from the folder name
  survey_year <- basename(year_folder)
  
  # Generate the output folder for the current survey year
  output_folder <- file.path(output_dir, survey_year)
  
  # Create output directory if it doesn't exist
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }
  
  # Process each raster in the list
  process_raster_function(raster_list, output_folder)
}
####--------- END OF PRE PROCESSING SECTION ---###




## MOSAIC creation for all pre-processed rasters----

library(terra)

# Define the directories
data_in_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/in" # Pre_processed_tiffs" # where files are located
output_dir <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/Florida_wc/mosaic"
mask_file <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/JABLTCX_data/masks/Job1051879_data_extent_mask_Florida_wc_1_0.tif"  # Specify the path to your mask file
# Job 1051879 was selected as mask as it was the earliest datasets (e.g., baseline)

# Load the mask raster
mask_raster <- rast(mask_file)
mask_extent <- ext(mask_raster)  # Get extent of the mask
mask_res <- res(mask_raster)  # Get resolution of the mask

# Load the mask raster
mask_raster <- rast(mask_file)
mask_extent <- ext(mask_raster)  # Get extent of the mask
mask_res <- res(mask_raster)  # Get resolution of the mask

# List all the subfolders in the data out directory (each representing a survey year)
survey_year_folders <- list.dirs(data_in_dir, recursive = FALSE)

# Loop through each survey year folder
for (year_folder in survey_year_folders) {
  
  # Get the list of GeoTIFF files in the folder
  tiff_files <- list.files(year_folder, pattern = "\\.tif$", full.names = TRUE)
  
  if (length(tiff_files) > 0) {
    # Create a raster stack from all the GeoTIFFs in the folder
    raster_stack <- lapply(tiff_files, rast)
    
    # Combine rasters into a single object using do.call
    message("Creating mosaic for survey year: ", basename(year_folder))
    combined_rasters <- do.call(terra::mosaic, raster_stack)
    
    # Resample the mosaic to match the mask extent and 5x5m resolution
    message("Resampling mosaic to match mask file extent and resolution...")
    resampled_mosaic <- resample(combined_rasters, mask_raster, method = "bilinear")
    
    # Set the extent of the resampled mosaic to match the mask file
    message("Adjusting mosaic extent to match mask file...")
    resampled_mosaic <- extend(resampled_mosaic, mask_extent, fill = NA)
    
    # Write the resampled mosaic to the output directory
    output_filename <- paste0(basename(year_folder), "_mosaic_resampled.tif")
    output_filepath <- file.path(output_dir, output_filename)
    
    message("Saving resampled mosaic to: ", output_filepath)
    writeRaster(resampled_mosaic, filename = output_filepath, overwrite = TRUE)
    
    cat("Processed mosaic saved for survey year: ", basename(year_folder), "\n")
  } else {
    message("No GeoTIFF files found in folder: ", year_folder)
  }
}

cat("All mosaics processed and saved.\n")
