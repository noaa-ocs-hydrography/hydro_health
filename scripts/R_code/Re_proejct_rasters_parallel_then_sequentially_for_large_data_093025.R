# 1. Load necessary libraries
library(raster)
library(parallel)
library(foreach)
library(doSNOW)

# 2. Setup paths and the target Coordinate Reference System (CRS)
target_crs <- "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs"

# Define input directory
input_dir <- getwd()

# Define and create the output directory
output_dir <- file.path(input_dir, "processed_new")
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# 3. Get a list of all .tif files
raster_files <- list.files(path = input_dir, pattern = "\\.tif$", full.names = FALSE, ignore.case = TRUE)
cat(paste("Found", length(raster_files), "raster files to process.\n\n"))

# 4. Set up and run the parallel processing job
num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)
registerDoSNOW(cl)

# Setup a progress bar
pb <- txtProgressBar(max = length(raster_files), style = 3)
progress <- function(n) setTxtProgressBar(pb, n)
opts <- list(progress = progress)

cat("--- Starting parallel processing phase ---\n")
# Use foreach to loop through files and capture failures
failed_files_list <- foreach(
  file_name = raster_files, 
  .packages = c("raster"), 
  .options.snow = opts, 
  .combine = 'c'
) %dopar% {
  
  result <- NULL # Default return for success
  
  # Use try() to catch any errors during the process
  error_check <- try({
    input_path <- file.path(input_dir, file_name)
    output_path <- file.path(output_dir, file_name)
    
    # Load and re-project the raster
    original_raster <- raster(input_path)
    projectRaster(from = original_raster, 
                  crs = target_crs, 
                  filename = output_path, 
                  overwrite = TRUE)
  })
  
  # If an error occurred, 'error_check' will be class 'try-error'
  # If so, we set the result to be the file name to mark it for retry
  if (inherits(error_check, "try-error")) {
    result <- file_name
  }
  
  return(result)
}

# 5. Clean up the parallel cluster
stopCluster(cl)
close(pb)
cat("\n--- Parallel processing phase complete ---\n")

# 6. Sequentially process any files that failed
# Filter the results list to get only the names of files that failed
failed_files <- failed_files_list[!sapply(failed_files_list, is.null)]

if (length(failed_files) > 0) {
  cat(paste("\n--- Retrying", length(failed_files), "file(s) sequentially due to errors ---\n"))
  
  # Loop through only the failed files one by one
  for (file_name in failed_files) {
    cat(paste("Processing:", file_name, "... "))
    
    # It's good practice to wrap this in a try block as well
    try({
      input_path <- file.path(input_dir, file_name)
      output_path <- file.path(output_dir, file_name)
      
      original_raster <- raster(input_path)
      projectRaster(from = original_raster, 
                    crs = target_crs, 
                    filename = output_path, 
                    overwrite = TRUE)
      
      cat("Success!\n")
    }, silent = TRUE)
  }
} else {
  cat("\nAll files were processed successfully in parallel.\n")
}

cat("\n✨ Reprojection complete! All files have been saved to the 'processed_new' directory.\n")