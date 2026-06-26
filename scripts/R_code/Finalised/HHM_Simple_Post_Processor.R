# HHM "Simple Analysis" (i.e re-run of the 2018 methodology with new data) is split into two parts, and two resolutions

# Part 1 = Creates the spatially variable decay coefficient using "change agents" - marine hazards / marine debris, hurricane data and tidal current data from 2011 by applying 
            # the following equation. Decay Coefficient = ((change hurricanes + change currents + change human debris) - Number of terms) x 0.002 /3

# Part 2 = Takes the decay coefficient, UKC, Survey end date, to create finalized outputs of ISS, DSS, and PSS

# Res 1 - 100m - (Offshore tiles and all tiles )will focus on a single tiff for the whole of the USA (we will run a pass with the full USA extent, and just offshore)
# Res 2 - 20m - (Inshore tiles only) will focus on going through each Eco Region Folder, which has a number of 20m tiffs for the inshore grid tiles only


# HHM simple post-processing workflow

## -  PRE Step for merging rasters - ##

# merging only required to run over full extent of USA if input data is split by Eco Region 
library(terra)
 # make a mosaic of the buffered and non-buffered! 

#P2 - Buffered Hazards
merge_hazard_tifs_resampled <- function(
    input_dir,
    pattern = "P2_buffered_2nm_total_weighted_hazard_score_100m\\.tif$",
    output_tif = "P2_merged_hazard_score_2nm_buffer_100m_EPSG6350.tif",
    target_crs = "EPSG:6350",
    res_m = 100,
    overwrite = TRUE
) {
  library(raster)
  library(sp)
  
  tif_files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  if (length(tif_files) == 0) stop("No matching TIFF files found.")
  
  tif_files <- sort(tif_files)
  
  message("Loading rasters...")
  rasters <- lapply(tif_files, function(f) {
    r <- raster::raster(f)
    raster::crs(r) <- sp::CRS(SRS_string = target_crs)
    r
  })
  
  # Build one shared snapped extent
  xmin_all <- floor(min(sapply(rasters, function(r) extent(r)@xmin)) / res_m) * res_m
  xmax_all <- ceiling(max(sapply(rasters, function(r) extent(r)@xmax)) / res_m) * res_m
  ymin_all <- floor(min(sapply(rasters, function(r) extent(r)@ymin)) / res_m) * res_m
  ymax_all <- ceiling(max(sapply(rasters, function(r) extent(r)@ymax)) / res_m) * res_m
  
  master_template <- raster::raster(
    xmn = xmin_all,
    xmx = xmax_all,
    ymn = ymin_all,
    ymx = ymax_all,
    resolution = res_m,
    crs = sp::CRS(SRS_string = target_crs)
  )
  
  message("Resampling rasters to shared 100 m origin...")
  
  rasters_aligned <- lapply(seq_along(rasters), function(i) {
    r <- rasters[[i]]
    
    target_ext <- raster::alignExtent(
      raster::extent(r),
      master_template,
      snap = "out"
    )
    
    target <- raster::raster(
      target_ext,
      resolution = res_m,
      crs = sp::CRS(SRS_string = target_crs)
    )
    
    message("Aligning: ", basename(tif_files[i]))
    
    raster::resample(
      r,
      target,
      method = "ngb"
    )
  })
  
  message("Checking origins after alignment:")
  print(sapply(rasters_aligned, raster::origin))
  
  message("Mosaicking...")
  merged <- do.call(
    raster::mosaic,
    c(rasters_aligned, fun = max, na.rm = TRUE)
  )
  
  raster::crs(merged) <- sp::CRS(SRS_string = target_crs)
  
  out_path <- file.path(input_dir, output_tif)
  
  message("Writing output...")
  raster::writeRaster(
    merged,
    filename = out_path,
    format = "GTiff",
    overwrite = overwrite,
    options = c(
      "COMPRESS=LZW",
      "TILED=YES",
      "BIGTIFF=YES"
    )
  )
  
  message("Done: ", out_path)
  merged
}

# function call 
merged <- merge_hazard_tifs_resampled(
  input_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards/outputs_by_ecoregion"
)

# P1 - NON Buffered (true polygone extent)
merge_hazard_tifs_resampled <- function(
    input_dir,
    pattern = "P1_true_geometry_total_weighted_hazard_score_100m\\.tif$",
    output_tif = "P1_merged_hazard_score_2nm_geometry_100m_EPSG6350.tif",
    target_crs = "EPSG:6350",
    res_m = 100,
    overwrite = TRUE
) {
  library(raster)
  library(sp)
  
  tif_files <- list.files(input_dir, pattern = pattern, full.names = TRUE)
  if (length(tif_files) == 0) stop("No matching TIFF files found.")
  
  tif_files <- sort(tif_files)
  
  message("Loading rasters...")
  rasters <- lapply(tif_files, function(f) {
    r <- raster::raster(f)
    raster::crs(r) <- sp::CRS(SRS_string = target_crs)
    r
  })
  
  # Build one shared snapped extent
  xmin_all <- floor(min(sapply(rasters, function(r) extent(r)@xmin)) / res_m) * res_m
  xmax_all <- ceiling(max(sapply(rasters, function(r) extent(r)@xmax)) / res_m) * res_m
  ymin_all <- floor(min(sapply(rasters, function(r) extent(r)@ymin)) / res_m) * res_m
  ymax_all <- ceiling(max(sapply(rasters, function(r) extent(r)@ymax)) / res_m) * res_m
  
  master_template <- raster::raster(
    xmn = xmin_all,
    xmx = xmax_all,
    ymn = ymin_all,
    ymx = ymax_all,
    resolution = res_m,
    crs = sp::CRS(SRS_string = target_crs)
  )
  
  message("Resampling rasters to shared 100 m origin...")
  
  rasters_aligned <- lapply(seq_along(rasters), function(i) {
    r <- rasters[[i]]
    
    target_ext <- raster::alignExtent(
      raster::extent(r),
      master_template,
      snap = "out"
    )
    
    target <- raster::raster(
      target_ext,
      resolution = res_m,
      crs = sp::CRS(SRS_string = target_crs)
    )
    
    message("Aligning: ", basename(tif_files[i]))
    
    raster::resample(
      r,
      target,
      method = "ngb"
    )
  })
  
  message("Checking origins after alignment:")
  print(sapply(rasters_aligned, raster::origin))
  
  message("Mosaicking...")
  merged <- do.call(
    raster::mosaic,
    c(rasters_aligned, fun = max, na.rm = TRUE)
  )
  
  raster::crs(merged) <- sp::CRS(SRS_string = target_crs)
  
  out_path <- file.path(input_dir, output_tif)
  
  message("Writing output...")
  raster::writeRaster(
    merged,
    filename = out_path,
    format = "GTiff",
    overwrite = overwrite,
    options = c(
      "COMPRESS=LZW",
      "TILED=YES",
      "BIGTIFF=YES"
    )
  )
  
  message("Done: ", out_path)
  merged
}

# function call 
merged <- merge_hazard_tifs_resampled(
  input_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards/outputs_by_ecoregion"
)



# HHM simple post-processing workflow
#
# Runs the two-stage HHM simple workflow at either 100 m or 20 m resolution.
# Stage 1 creates the spatially variable decay coefficient.
# Stage 2 creates ISS, DSS, and PSS outputs and optional graphics.

library(terra)

# User-facing workflow ---------------------------------------------------------

run_hhm_simple_workflow <- function(
    resolution = c("100m", "20m"),
    stages = c("both", "stage1", "stage2"),
    eco_regions = "all",
    root_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/HHM_simple",
    analysis_year = 2026,
    make_plots = TRUE,
    overwrite = TRUE,
    terra_temp_dir = file.path(root_dir, "terra_temp"),
    terra_memfrac = 0.35
) {
  resolution <- match.arg(resolution)
  stages <- match.arg(stages)
  
  dir.create(terra_temp_dir, recursive = TRUE, showWarnings = FALSE)
  terra::terraOptions(
    tempdir = terra_temp_dir,
    memfrac = terra_memfrac,
    todisk = TRUE,
    progress = 1
  )
  
  config <- get_hhm_config(root_dir, resolution)
  tile_table <- get_tile_table(config, eco_regions = eco_regions)
  
  results <- lapply(seq_len(nrow(tile_table)), function(i) {
    tile <- tile_table[i, ]
    log_step(paste0("Processing ", tile$tile_id, " at ", resolution, "."))
    
    dir.create(tile$output_dir, recursive = TRUE, showWarnings = FALSE)
    
    stage1_result <- NULL
    stage2_result <- NULL
    
    if (stages %in% c("both", "stage1")) {
      stage1_result <- create_decay_coefficient(
        input_dir = tile$input_dir,
        output_dir = tile$output_dir,
        resolution = resolution,
        tile_id = tile$tile_id,
        analysis_year = analysis_year,
        overwrite = overwrite
      )
    }
    
    if (stages %in% c("both", "stage2")) {
      stage2_result <- create_survey_scores(
        input_dir = tile$input_dir,
        output_dir = tile$output_dir,
        resolution = resolution,
        tile_id = tile$tile_id,
        analysis_year = analysis_year,
        make_plots = make_plots,
        overwrite = overwrite
      )
    }
    
    data.frame(
      tile_id = tile$tile_id,
      input_dir = tile$input_dir,
      output_dir = tile$output_dir,
      stage1_complete = !is.null(stage1_result),
      stage2_complete = !is.null(stage2_result),
      stringsAsFactors = FALSE
    )
  })
  
  do.call(rbind, results)
}

# Logging ---------------------------------------------------------------------

log_step <- function(text) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " | ", text)
}

# Directory setup -------------------------------------------------------------

get_hhm_config <- function(root_dir, resolution) {
  if (resolution == "100m") {
    return(list(
      resolution = resolution,
      input_dir = file.path(root_dir, "inputs", "Offshore_tiles_100m" ), #"Eco_Region_All_tiles_100m" #Eco_Region_All_tiles_Buffered_100m
      output_dir = file.path(root_dir, "outputs", "Offshore_tiles_100m")
    ))
  }
  
  list(
    resolution = resolution,
    input_dir = file.path(root_dir, "inputs", "Nearshore_tiles_20m"),
    output_dir = file.path(root_dir, "outputs", "Nearshore_tiles_20m")
  )
}

get_tile_table <- function(config, eco_regions = "all") {
  
  if (config$resolution == "100m") {
    
    stopifnot(length(config$input_dir) == 1)
    stopifnot(length(config$output_dir) == 1)
    
    return(data.frame(
      tile_id = "Eco_region_all_tiles_100m",
      input_dir = as.character(config$input_dir),
      output_dir = as.character(config$output_dir),
      stringsAsFactors = FALSE
    ))
  }
  
  eco_region_dirs <- list.dirs(config$input_dir, recursive = FALSE, full.names = TRUE)
  eco_region_dirs <- eco_region_dirs[grepl("^ER_[0-9]+$", basename(eco_region_dirs))]
  
  if (!identical(eco_regions, "all")) {
    eco_regions <- as.character(eco_regions)
    
    eco_region_dirs <- eco_region_dirs[basename(eco_region_dirs) %in% eco_regions]
    
    if (length(eco_region_dirs) == 0) {
      stop(
        "No matching eco-region folders found for: ",
        paste(eco_regions, collapse = ", "),
        call. = FALSE
      )
    }
  }
  
  data.frame(
    tile_id = basename(eco_region_dirs),
    input_dir = eco_region_dirs,
    output_dir = file.path(config$output_dir, basename(eco_region_dirs)),
    stringsAsFactors = FALSE
  )
}

# config <- get_hhm_config(root_dir, resolution)
# 
# print(config)
# str(config)
  


# Stage 1 ---------------------------------------------------------------------

create_decay_coefficient <- function(
    input_dir,
    output_dir,
    resolution,
    tile_id,
    analysis_year = 2026,
    divisor_terms = 3,
    decay_rate = 0.022,
    overwrite = TRUE
) {
  log_step("Reading stage 1 rasters")
  bathy <- terra::rast(find_raster(input_dir, "bathy.*mosaic"))
  survey_date <- terra::rast(find_raster(input_dir, "survey.*date|survey.*end"))
  sand_mud <- terra::rast(find_raster(input_dir, "sand.*mud.*mask"))
  tidal <- terra::rast(find_raster(input_dir, "marine.*current|tidal|currents"))
  hurricanes <- terra::rast(find_raster(input_dir, "hurricane"))
  debris <- terra::rast(find_raster(input_dir, "marine.*hazard|human.*debris"))
  
  ref <- bathy
  survey_date <- align_to_ref(survey_date, ref, "near")
  sand_mud <- align_to_ref(sand_mud, ref, "near")
  tidal <- align_to_ref(tidal, ref, "bilinear")
  hurricanes <- align_to_ref(hurricanes, ref, "near")
  debris <- align_to_ref(debris, ref, "near")
  
  stage1_stack <- c(bathy, survey_date, sand_mud, tidal, hurricanes, debris)
  names(stage1_stack) <- c("bathy", "survey_date", "sand_mud", "tidal", "hurricanes", "debris")
  
  temp_stage1_path <- file.path(output_dir, paste0("stage1_temp_", tile_id, ".tif"))
  log_step("Computing stage 1 bins and decay coefficient on disk")
  
  stage1_outputs <- terra::lapp(
    stage1_stack,
    fun = function(bathy, survey_date, sand_mud, tidal, hurricanes, debris) {
      
      survey_year <- clean_year_values(survey_date, analysis_year)
      survey_age <- analysis_year - survey_year
      survey_age[!is.na(survey_age) & survey_age < 0] <- NA
      
      depth <- abs(bathy)
      
      # Full model domain should follow bathy + valid survey year
      valid_domain <- !is.na(bathy) & !is.na(survey_year)
      
      sand_mud_positive <- !is.na(sand_mud) & sand_mud > 0
      
      # Hurricanes only apply in sand/mud and <= 40 m
      hurricane_applicable <- valid_domain & sand_mud_positive & depth <= 40
      hurricane_cond <- ifelse(hurricane_applicable, hurricanes, NA)
      hurricane_bin <- bin_hurricane_values(hurricane_cond)
      
      # Tidal only applies in sand/mud and <= 20 m
      tidal_applicable <- valid_domain & sand_mud_positive & depth <= 20
      tidal_exposure <- ifelse(tidal_applicable, tidal * survey_age, NA)
      tidal_bin <- bin_tidal_values(tidal_exposure)
      
      # Human debris / hazard term should exist across full valid model domain.
      # NA debris means no known hazards, not no data.
      debris_clean <- ifelse(valid_domain & is.na(debris), 0, debris)
      debris_bin <- ifelse(valid_domain, bin_debris_values(debris_clean), NA)
      
      n_terms <- 
        (!is.na(hurricane_bin)) +
        (!is.na(tidal_bin)) +
        (!is.na(debris_bin))
      
      change_sum <-
        ifelse(is.na(hurricane_bin), 0, hurricane_bin) +
        ifelse(is.na(tidal_bin), 0, tidal_bin) +
        ifelse(is.na(debris_bin), 0, debris_bin)
      
      decay_coefficient <- ((change_sum - n_terms) * decay_rate) / divisor_terms
      
      # Keep decay as 0 where the model domain is valid but there is no change.
      decay_coefficient[valid_domain & n_terms == 0] <- 0
      decay_coefficient[!valid_domain] <- NA
      
      cbind(
        decay_coefficient,
        hurricane_bin,
        tidal_bin,
        debris_bin,
        tidal_exposure
      )
    },
    filename = temp_stage1_path,
    overwrite = overwrite,
    wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW", "TILED=YES"))
  )
  
  names(stage1_outputs) <- c(
    "decay_coefficient",
    "change_hurricanes_bin",
    "change_tidal_bin",
    "change_human_debris_bin",
    "tidal_exposure"
  )
  terra::crs(stage1_outputs) <- terra::crs(ref)
  
  write_hhm_raster(stage1_outputs[["decay_coefficient"]], file.path(output_dir, paste0("decay_coefficient_", tile_id, ".tif")), overwrite = overwrite, ref = ref)
  write_hhm_raster(stage1_outputs[["change_hurricanes_bin"]], file.path(output_dir, paste0("change_hurricanes_bin_", tile_id, ".tif")), overwrite = overwrite, ref = ref)
  write_hhm_raster(stage1_outputs[["change_tidal_bin"]], file.path(output_dir, paste0("change_tidal_bin_", tile_id, ".tif")), overwrite = overwrite, ref = ref)
  write_hhm_raster(stage1_outputs[["change_human_debris_bin"]], file.path(output_dir, paste0("change_human_debris_bin_", tile_id, ".tif")), overwrite = overwrite, ref = ref)
  write_hhm_raster(stage1_outputs[["tidal_exposure"]], file.path(output_dir, paste0("tidal_exposure_", tile_id, ".tif")), overwrite = overwrite, ref = ref)
  
  unlink(temp_stage1_path)
  terra::tmpFiles(remove = TRUE)
  invisible(gc())
  
  invisible(stage1_outputs)
}

# Stage 2 ---------------------------------------------------------------------

create_survey_scores <- function(
    input_dir,
    output_dir,
    resolution,
    tile_id,
    analysis_year = 2026,
    make_plots = TRUE,
    overwrite = TRUE
) {
  log_step("Reading stage 2 rasters")
  survey_year <- terra::rast(find_raster(input_dir, "survey.*date|survey.*end"))
  iss <- terra::rast(find_raster(input_dir, "(^|_)iss|supersession.*score"))
  ukc <- terra::rast(find_raster(input_dir, "ukc|under.*keel.*clearance"))
  slope <- terra::rast(find_raster(input_dir, "slope"))
  bathy <- terra::rast(find_raster(input_dir, "bathy.*mosaic"))
  
  decay_path <- find_raster(output_dir, "decay.*coefficient")
  decay_coefficient <- terra::rast(decay_path)
  
  # Use bathy as the authoritative spatial reference
  ref <- bathy
  
  if (is.na(terra::crs(decay_coefficient)) || terra::crs(decay_coefficient) == "") {
    terra::crs(decay_coefficient) <- terra::crs(ref)
  }
  
  survey_year <- align_to_ref(survey_year, ref, "near")
  iss <- align_to_ref(iss, ref, "near")
  ukc <- align_to_ref(ukc, ref, "bilinear")
  slope <- align_to_ref(slope, ref, "bilinear")
  bathy <- align_to_ref(bathy, ref, "bilinear")
  decay_coefficient <- align_to_ref(decay_coefficient, ref, "bilinear")
  
  stage2_stack <- c(survey_year, iss, ukc, bathy, slope, decay_coefficient)
  names(stage2_stack) <- c("survey_year", "iss", "ukc", "bathy", "slope", "decay_coefficient")
  
  temp_stage2_path <- file.path(output_dir, paste0("stage2_temp_", tile_id, ".tif"))
  log_step("Computing stage 2 scores on disk")
  
  stage2_outputs <- terra::lapp(
    stage2_stack,
    fun = function(survey_year, iss, ukc, bathy, slope, decay_coefficient) {
      survey_year <- clean_year_values(survey_year, analysis_year)
      survey_age <- analysis_year - survey_year
      survey_age[!is.na(survey_age) & survey_age < 0] <- NA
      survey_age[!is.na(survey_age) & survey_age > 200] <- 200
      
      ukc[!is.na(ukc) & ukc < 0] <- NA
      # iss <- reclassify_iss_values(iss)
      
      pss <- iss * exp(-decay_coefficient * survey_age)
      pss[!is.na(pss) & pss < 0] <- 0
      pss[!is.na(pss) & pss > 110] <- 110
      
      depth <- abs(bathy)
      dss <- make_dss_values(ukc, depth, slope)
      hydrographic_gap <- dss - pss
      
      
      
      cbind(survey_year, iss, dss, pss, hydrographic_gap)
    },
    filename = temp_stage2_path,
    overwrite = overwrite,
    wopt = list(datatype = "FLT4S", gdal = c("COMPRESS=LZW", "TILED=YES"))
  )
  
  names(stage2_outputs) <- c("survey_end_year", "ISS", "DSS", "PSS", "HG")
  
  # Force CRS from decay coefficient reference onto all stage 2 outputs
  terra::crs(stage2_outputs) <- terra::crs(ref)
  
  # Check values 
  message("Stage 2 outputs created. Skipping full-raster global diagnostics to avoid memory crash.")
  
  survey_year_path <- file.path(output_dir, paste0("survey_end_year_", tile_id, ".tif"))
  iss_path <- file.path(output_dir, paste0("ISS_", tile_id, ".tif"))
  dss_path <- file.path(output_dir, paste0("DSS_", tile_id, ".tif"))
  pss_path <- file.path(output_dir, paste0("PSS_", tile_id, ".tif"))
  hg_path <- file.path(output_dir, paste0("HG_", tile_id, ".tif"))
  
  write_hhm_raster(stage2_outputs[["survey_end_year"]], survey_year_path, overwrite = overwrite, ref = ref)
  write_hhm_raster(stage2_outputs[["ISS"]], iss_path, overwrite = overwrite, ref = ref)
  write_hhm_raster(stage2_outputs[["DSS"]], dss_path, overwrite = overwrite, ref = ref)
  write_hhm_raster(stage2_outputs[["PSS"]], pss_path, overwrite = overwrite, ref = ref)
  write_hhm_raster(stage2_outputs[["HG"]], hg_path, overwrite = overwrite, ref = ref)
  
  if (isTRUE(make_plots)) {
    log_step("Creating stage 2 plots")
    make_three_panel_plot(
      stage2_outputs[["survey_end_year"]],
      stage2_outputs[["ISS"]],
      stage2_outputs[["PSS"]],
      file.path(output_dir, paste0("survey_end_year_ISS_PSS_", tile_id, ".png"))
    )
    make_dss_pss_plot(
      stage2_outputs[["DSS"]],
      stage2_outputs[["PSS"]],
      file.path(output_dir, paste0("DSS_PSS_", tile_id, ".png"))
    )
    make_gap_dss_pss_plot(
      stage2_outputs[["HG"]],
      stage2_outputs[["DSS"]],
      stage2_outputs[["PSS"]],
      file.path(output_dir, paste0("HG_DSS_PSS_", tile_id, ".png"))
    )
  }
  
  unlink(temp_stage2_path)
  terra::tmpFiles(remove = TRUE)
  invisible(gc())
  
  invisible(data.frame(
    tile_id = tile_id,
    survey_end_year = survey_year_path,
    iss = iss_path,
    dss = dss_path,
    pss = pss_path,
    hydrographic_gap = hg_path
  ))
}

# reclassify_iss <- function(iss) {
#   terra::classify(
#     iss,
#     rcl = matrix(c(
#       -Inf, 0, 0,
#       0, 30, 30,
#       30, 80, 80,
#       80, 100, 100,
#       100, Inf, 110
#     ), ncol = 3, byrow = TRUE),
#     include.lowest = TRUE,
#     right = TRUE
#   )
# }

# Chunk-safe vector functions -------------------------------------------------

clean_year_values <- function(x, analysis_year) {
  x <- round(x)
  x[is.na(x) | x <= 1] <- NA
  
  date_yyyymmdd <- !is.na(x) & x >= 10000000
  x[date_yyyymmdd] <- floor(x[date_yyyymmdd] / 10000)
  
  date_yyyymm <- !is.na(x) & x >= 10000 & x < 10000000
  x[date_yyyymm] <- floor(x[date_yyyymm] / 100)
  
  x[!is.na(x) & (x < 1800 | x > analysis_year)] <- NA
  x
}

bin_hurricane_values <- function(x) {
  out <- rep(NA_real_, length(x))
  out[!is.na(x) & x <= 0] <- 1
  out[!is.na(x) & x > 0 & x <= 4] <- 2
  out[!is.na(x) & x > 4 & x <= 10] <- 3
  out[!is.na(x) & x > 10 & x <= 20] <- 4
  out[!is.na(x) & x > 20] <- 5
  out
}

bin_tidal_values <- function(x) {
  out <- rep(NA_real_, length(x))
  
  out[!is.na(x) & x < 0.1] <- 1
  out[!is.na(x) & x >= 0.1 & x < 0.2] <- 2
  out[!is.na(x) & x >= 0.2 & x < 0.5] <- 3
  out[!is.na(x) & x >= 0.5 & x < 1] <- 4
  out[!is.na(x) & x >= 1] <- 5
  
  out
}

bin_debris_values <- function(x) {
  out <- rep(NA_real_, length(x))
  out[!is.na(x) & x <= 0] <- 1
  out[!is.na(x) & x > 0 & x <= 1] <- 2
  out[!is.na(x) & x > 1 & x <= 5] <- 3
  out[!is.na(x) & x > 5 & x <= 10] <- 4
  out[!is.na(x) & x > 10] <- 5
  out
}

# reclassify_iss_values <- function(x) {
#   out <- rep(NA_real_, length(x))
#   
#   out[!is.na(x) & x <= 0] <- 0
#   out[!is.na(x) & x > 0 & x <= 30] <- 30
#   out[!is.na(x) & x > 30 & x <= 80] <- 80
#   out[!is.na(x) & x > 80 & x < 100] <- 100
#   out[!is.na(x) & x >= 100] <- 110
#   
#   out
# }

make_dss_values <- function(ukc, depth, slope) {
  ukc[!is.na(ukc) & ukc < 0] <- NA
  
  use_ukc <- !is.na(ukc)
  use_depth <- is.na(ukc) & !is.na(depth)
  
  simple <- !is.na(slope) & slope < 0.5
  moderate <- !is.na(slope) & slope >= 0.5 & slope < 1
  complex <- !is.na(slope) & slope >= 1
  
  out <- rep(NA_real_, length(ukc))
  
  out[use_ukc & simple] <- bin_ukc_values(
    ukc[use_ukc & simple],
    shallow_max = 20,
    mid_max = 50
  )
  
  out[use_ukc & moderate] <- bin_ukc_values(
    ukc[use_ukc & moderate],
    shallow_max = 30,
    mid_max = 75
  )
  
  out[use_ukc & complex] <- bin_ukc_values(
    ukc[use_ukc & complex],
    shallow_max = 40,
    mid_max = 100
  )
  
  out[use_depth & simple] <- bin_depth_values(
    depth[use_depth & simple],
    shallow_max = 20,
    mid_max = 50
  )
  
  out[use_depth & moderate] <- bin_depth_values(
    depth[use_depth & moderate],
    shallow_max = 30,
    mid_max = 75
  )
  
  out[use_depth & complex] <- bin_depth_values(
    depth[use_depth & complex],
    shallow_max = 40,
    mid_max = 100
  )
  
  out
}

bin_ukc_values <- function(ukc, shallow_max, mid_max) {
  out <- rep(NA_real_, length(ukc))
  out[!is.na(ukc) & ukc <= 1] <- 100
  out[!is.na(ukc) & ukc > 1 & ukc <= shallow_max] <- 80
  out[!is.na(ukc) & ukc > shallow_max & ukc <= mid_max] <- 30
  out[!is.na(ukc) & ukc > mid_max] <- 10
  out
}

# Helpers ---------------------------------------------------------------------
bin_depth_values <- function(depth, shallow_max, mid_max) {
  out <- rep(NA_real_, length(depth))
  
  out[!is.na(depth) & depth < 2] <- 100
  out[!is.na(depth) & depth >= 2 & depth <= shallow_max] <- 80
  out[!is.na(depth) & depth > shallow_max & depth <= mid_max] <- 30
  out[!is.na(depth) & depth > mid_max] <- 10
  
  out
}

clean_survey_year <- function(year_raster, analysis_year) {
  terra::app(year_raster, function(x) {
    x <- round(x)
    x[is.na(x) | x <= 1] <- NA
    
    date_yyyymmdd <- !is.na(x) & x >= 10000000
    x[date_yyyymmdd] <- floor(x[date_yyyymmdd] / 10000)
    
    date_yyyymm <- !is.na(x) & x >= 10000 & x < 10000000
    x[date_yyyymm] <- floor(x[date_yyyymm] / 100)
    
    x[!is.na(x) & (x < 1800 | x > analysis_year)] <- NA
    x
  })
}

align_to_ref <- function(raster, ref, method = "near") {
  
  # Case 1: CRS is truly the same
  if (terra::same.crs(raster, ref)) {
    if (!terra::compareGeom(raster, ref, crs = FALSE, stopOnError = FALSE)) {
      raster <- terra::resample(raster, ref, method = method)
    }
    return(raster)
  }
  
  # Case 2: CRS names match even if WKT/projjson differs
  raster_crs_name <- get_crs_name(raster)
  ref_crs_name <- get_crs_name(ref)
  
  if (!is.na(raster_crs_name) &&
      !is.na(ref_crs_name) &&
      identical(raster_crs_name, ref_crs_name)) {
    
    terra::crs(raster) <- terra::crs(ref)
    
    if (!terra::compareGeom(raster, ref, crs = FALSE, stopOnError = FALSE)) {
      raster <- terra::resample(raster, ref, method = method)
    }
    
    return(raster)
  }
  
  # Case 3: extents/resolution are in the same coordinate system but CRS metadata differs
  # This avoids unnecessary PROJ transformations for your already-aligned HHM rasters.
  same_units_and_overlap <- tryCatch({
    er <- terra::ext(raster)
    ef <- terra::ext(ref)
    
    overlap_x <- er$xmin < ef$xmax && er$xmax > ef$xmin
    overlap_y <- er$ymin < ef$ymax && er$ymax > ef$ymin
    
    rx <- terra::res(raster)
    rr <- terra::res(ref)
    
    similar_res <- all(abs(rx - rr) < 1e-6)
    
    overlap_x && overlap_y && similar_res
  }, error = function(e) FALSE)
  
  if (same_units_and_overlap) {
    message("CRS metadata differs, but raster appears to share the same grid units. Forcing CRS to reference and resampling.")
    terra::crs(raster) <- terra::crs(ref)
    
    if (!terra::compareGeom(raster, ref, crs = FALSE, stopOnError = FALSE)) {
      raster <- terra::resample(raster, ref, method = method)
    }
    
    return(raster)
  }
  
  # Last resort: true reprojection
  terra::project(raster, ref, method = method)
}

get_crs_name <- function(raster) {
  crs_text <- tryCatch(
    terra::crs(raster, describe = TRUE),
    error = function(e) NULL
  )
  
  if (is.data.frame(crs_text) && "name" %in% names(crs_text)) {
    return(crs_text$name[1])
  }
  
  NA_character_
}

find_raster <- function(directory, pattern) {
  raster_paths <- list.files(
    directory,
    pattern = "\\.tiff?$|\\.tif$",
    full.names = TRUE,
    recursive = FALSE,
    ignore.case = TRUE
  )
  
  raster_paths <- raster_paths[!grepl("Thumbs\\.db$", raster_paths, ignore.case = TRUE)]
  matches <- raster_paths[grepl(pattern, basename(raster_paths), ignore.case = TRUE)]
  
  if (length(matches) != 1) {
    stop(
      "Expected one raster matching `", pattern, "` in `", directory,
      "`, but found ", length(matches), ".",
      call. = FALSE
    )
  }
  
  matches
}

write_hhm_raster <- function(raster, path, overwrite = TRUE, datatype = "FLT4S", ref = NULL) {
  log_step(paste0("Writing ", basename(path)))
  
  if (!is.null(ref)) {
    terra::crs(raster) <- terra::crs(ref)
  }
  
  if (is.na(terra::crs(raster)) || terra::crs(raster) == "") {
    stop("Output raster has missing CRS before write: ", basename(path))
  }
  
  terra::writeRaster(
    raster,
    path,
    overwrite = overwrite,
    datatype = datatype,
    gdal = c("COMPRESS=LZW", "TILED=YES", "BIGTIFF=YES")
  )
  
  check <- terra::rast(path)
  
  if (is.na(terra::crs(check)) || terra::crs(check) == "") {
    stop("CRS was lost during write: ", basename(path))
  }
  
  invisible(path)
}

make_three_panel_plot <- function(survey_year, iss, pss, output_file) {
  grDevices::png(output_file, width = 1800, height = 750, res = 150)
  
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  })
  
  graphics::par(mfrow = c(1, 3), mar = c(3, 3, 3, 6))
  plot_survey_year(survey_year, main = "Survey end year")
  plot_survey_score(iss, main = "Initial Survey Score (ISS)")
  plot_survey_score(pss, main = "Present Survey Score (PSS)")
  
  invisible(output_file)
}

make_dss_pss_plot <- function(dss, pss, output_file) {
  grDevices::png(output_file, width = 1400, height = 750, res = 150)
  
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  })
  
  graphics::par(mfrow = c(1, 2), mar = c(3, 3, 3, 6))
  plot_dss(dss, main = "Desired Survey Score (DSS)")
  plot_survey_score(pss, main = "Present Survey Score (PSS)")
  
  invisible(output_file)
}

make_gap_dss_pss_plot <- function(hydrographic_gap, dss, pss, output_file) {
  grDevices::png(output_file, width = 1800, height = 750, res = 150)
  
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit({
    graphics::par(old_par)
    grDevices::dev.off()
  })
  
  graphics::par(mfrow = c(1, 3), mar = c(3, 3, 3, 6))
  terra::plot(
    hydrographic_gap,
    main = "Hydrographic Gap (DSS - PSS)",
    col = hcl.colors(20, "RdYlBu", rev = TRUE)
  )
  plot_dss(dss, main = "Desired Survey Score (DSS)")
  plot_survey_score(pss, main = "Present Survey Score (PSS)")
  
  invisible(output_file)
}

plot_survey_score <- function(raster, main) {
  terra::plot(
    raster,
    main = main,
    breaks = c(0, 30, 80, 100, 110),
    col = c("#ff9999", "#ffffcc", "#66ff66", "#00b050"),
    plg = list(title = "Score")
  )
}

plot_dss <- function(raster, main) {
  terra::plot(
    raster,
    main = main,
    breaks = c(0, 10, 30, 80, 100),
    col = c("#ffc7ce", "#fff2cc", "#92d050", "#00b050"),
    plg = list(title = "Score")
  )
}

plot_survey_year <- function(raster, main) {
  breaks <- c(
    1800, 1900, 1950, 1975, 1980, 1985, 1990, 1995,
    2000, 2005, 2010, 2015, 2020, 2024, 2026
  )
  
  terra::plot(
    raster,
    main = main,
    breaks = breaks,
    col = hcl.colors(length(breaks) - 1, "YlGnBu", rev = TRUE),
    plg = list(title = "Survey end year")
  )
}



# Example calls ---------------------------------------------------------------

# Run both stages for the single 100 m USA mosaic.
Sys.time() #12:26
run_hhm_simple_workflow(resolution = "100m", stages = "both")
Sys.time()

# Run both stages for each 20 m eco-region folder.
run_hhm_simple_workflow(resolution = "20m", stages = "both")

# Run only stage 2 after the decay coefficient already exists.
Sys.time()
run_hhm_simple_workflow(resolution = "100m", stages = "stage2")
Sys.time()

# Inshore per eco region folder 
# All 20 m eco regions
run_hhm_simple_workflow(
  resolution = "20m",
  stages = "both",
  eco_regions = "all"
)

# Only ER_6
run_hhm_simple_workflow(
  resolution = "20m",
  stages = "both",
  eco_regions = "ER_6"
)

# Multiple selected ERs
run_hhm_simple_workflow(
  resolution = "20m",
  stages = "both",
  eco_regions = c("ER_2", "ER_6")
)
