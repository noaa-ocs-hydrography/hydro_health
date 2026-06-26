 

# ENC Hazard Extractor 



# # ============================================================
# # NOAA Native S-57 ENC Hazard Workflow
# # Purpose:
# #   Extract charted marine hazard features from native NOAA ENC
# #   S-57 .000 files so QUAPOS and STATUS are preserved.
# #
# # Outputs:
# #   One GeoPackage with separate layers:
# #     01_hazard_features_raw
# #     02_known_hazards
# #     03_unverified_reported_errors
# #     04_known_hazard_buffers_2nm
# #     05_unverified_buffers_2nm
# #     06_hazard_grid_500m_summary
# #
# # Main hazard feature classes:
# #   OBSTRN = Obstruction
# #   WRECKS = Wreck
# #   PILPNT = Pile
# #   UWTROC = Underwater / awash rock
# #
# # Important S-57 metadata acronyms:
# #   QUAPOS = Quality of Position
# #            confidence in horizontal feature location.
# #   STATUS = Status of feature, e.g. reported, not confirmed,
# #            existence doubtful, historic, temporary, etc.
# #   VALSOU = Value of sounding / least depth value.
# #   WATLEV = Water level effect.
# #   QUASOU = Quality of sounding.
# #   SOUACC = Sounding accuracy.
# #   VERACC = Vertical accuracy.
# #   HORACC = Horizontal accuracy.
# #   CATWRK = Category of wreck.
# #   EXPSOU = Exposition of sounding.
# #   SORDAT = Source date.
# #   SORIND = Source indication.
# #   INFORM = Information text.
# #
# # QUAPOS values, IHO S-57:
# #   1  = surveyed
# #   2  = unsurveyed
# #   3  = inadequately surveyed
# #   4  = approximate
# #   5  = position doubtful
# #   6  = unreliable
# #   7  = reported, not surveyed
# #   8  = reported, not confirmed
# #   9  = estimated
# #   10 = precisely known
# #   11 = calculated
# #
# # Grouping logic used here:
# #   Known hazards:
# #     OBSTRN, WRECKS, PILPNT, UWTROC with QUAPOS 1, 10, 11, or NULL.
# #     These are treated as charted hazards with acceptable/known position.
# #
# #   Unverified / reported errors:
# #     OBSTRN, WRECKS, PILPNT, UWTROC with QUAPOS 2:9
# #     OR STATUS suggesting reported, doubtful, approximate, or existence doubtful.
# #     These are treated as uncertainty indicators and receive higher weight.
# #
# # Methodology-style weighting:
# #   known hazard weight = 1
# #   unverified / reported hazard weight = 3
# #
# # Notes:
# #   - Native S-57 preserves attributes that may be omitted by
# #     ENC Direct-to-GIS.
# #   - NOAA updates ENCs frequently; rerun periodically for refreshed data.



# Pass 1: Read normal feature records
# OBSTRN, WRECKS, PILPNT, UWTROC
#
# Pass 2: Read primitive/vector records
# IsolatedNode, ConnectedNode, Edge, Face, etc.
# Keep records where QUAPOS is present.
#
# Then spatially transfer nearest/intersecting primitive QUAPOS back onto hazard features.
# ============================================================
# USER-FACING FUNCTION
# ============================================================

# V5 for speed improvements 
# ============================================================
# NOAA ENC S-57 HAZARD CACHE + ECOREGION SUMMARIES
# ============================================================

library(sf)
library(terra)
library(dplyr)
library(purrr)
library(stringr)
library(readr)
library(xml2)
library(httr2)
library(future)
library(furrr)
library(fs)

sf::sf_use_s2(FALSE)

proj_candidates <- list.files(
  "C:/Users",
  pattern = "proj.db",
  recursive = TRUE,
  full.names = TRUE
)

Sys.setenv(PROJ_LIB = dirname(proj_candidates[1]))

# ----------------------------
# BASIC HELPERS
# ----------------------------

set_s57_feature_env <- function() {
  Sys.setenv(
    OGR_S57_OPTIONS =
      "SPLIT_MULTIPOINT=ON,LIST_AS_STRING=ON,PRESERVE_EMPTY_NUMBERS=ON,ADD_SOUNDG_DEPTH=ON,LNAM_REFS=ON"
  )
}

set_s57_primitive_env <- function() {
  Sys.setenv(
    OGR_S57_OPTIONS =
      "RETURN_PRIMITIVES=ON,LIST_AS_STRING=ON,PRESERVE_EMPTY_NUMBERS=ON"
  )
}

clean_sf_geometries <- function(g, label = "sf_object") {
  if (!inherits(g, "sf")) stop(label, " is not an sf object.")
  
  g <- st_make_valid(g)
  g <- g[!st_is_empty(g), ]
  g <- g[!is.na(st_geometry(g)), ]
  
  good <- vapply(st_geometry(g), function(geom) {
    bb <- tryCatch(st_bbox(geom), error = function(e) rep(NA_real_, 4))
    all(is.finite(bb))
  }, logical(1))
  
  g <- g[good, ]
  
  if (nrow(g) == 0) stop(label, " has zero valid geometries after cleaning.")
  
  g
}

log_error_row <- function(error_log, source_s57, layer, step, message) {
  row <- tibble(
    time = as.character(Sys.time()),
    source_s57 = source_s57,
    layer = layer,
    step = step,
    message = message
  )
  
  write_csv(row, error_log, append = file.exists(error_log))
}

# ============================================================
# BUFFERED DISPLAY LAYER
# ============================================================

make_buffered_display_layer <- function(
    hazards,
    buffer_nm = 2
) {
  
  if (nrow(hazards) == 0) {
    return(hazards)
  }
  
  buffer_m <- buffer_nm * 1852
  
  hazards |>
    sf::st_transform(6350) |>
    sf::st_buffer(buffer_m) |>
    sf::st_make_valid()
}

# ----------------------------
# MASK HELPERS
# ----------------------------

get_mask_table <- function(mask_dir, regions = "all") {
  mask_files <- list.files(
    mask_dir,
    pattern = "\\.tif(f)?$",
    full.names = TRUE
  )
  
  mask_tbl <- tibble(
    mask_path = mask_files,
    region_id = str_extract(basename(mask_files), "ER_[0-9]+")
  ) |>
    filter(!is.na(region_id))
  
  if (!identical(regions, "all")) {
    mask_tbl <- mask_tbl |> filter(region_id %in% regions)
  }
  
  if (nrow(mask_tbl) == 0) stop("No matching masks found.")
  
  mask_tbl
}

make_mask_aoi <- function(mask_path) {
  m <- terra::rast(mask_path)
  m[m != 1] <- NA
  
  poly <- terra::as.polygons(m, dissolve = TRUE, na.rm = TRUE)
  
  sf::st_as_sf(poly) |>
    sf::st_make_valid()
}

make_summary_grid_from_mask <- function(mask_path, region_id) {
  
  m <- terra::rast(mask_path)
  
  m[m != 1] <- NA
  
  grid_poly <- terra::as.polygons(
    m,
    dissolve = FALSE,
    na.rm = TRUE
  )
  
  sf::st_as_sf(grid_poly) |>
    sf::st_make_valid() |>
    dplyr::mutate(
      region_id = region_id,
      grid_id = paste0(region_id, "_grid_", dplyr::row_number())
    )
}

# ----------------------------
# NOAA CATALOG / DOWNLOAD
# ----------------------------

read_enc_catalog_urls <- function() {
  catalog_url <- "https://charts.noaa.gov/ENCs/ENCProdCat.xml"
  
  message("Reading NOAA ENC product catalog...")
  
  doc <- read_xml(catalog_url)
  txt <- as.character(doc)
  
  urls <- str_extract_all(txt, "https?://[^\"'< >]+\\.zip")[[1]] |>
    unique()
  
  tibble(
    download_url = urls,
    enc_zip = basename(download_url),
    cell_name = str_remove(enc_zip, "\\.zip$")
  ) |>
    filter(str_detect(cell_name, regex("^US", ignore_case = TRUE)))
}

download_file_if_missing <- function(url, dest) {
  if (file.exists(dest) && file.size(dest) > 0) return(dest)
  
  message("Downloading: ", basename(dest))
  
  resp <- request(url) |>
    req_timeout(300) |>
    req_perform()
  
  writeBin(resp_body_raw(resp), dest)
  
  dest
}

download_and_unzip_cell <- function(download_url, download_dir, extract_dir, error_log) {
  zip_dest <- file.path(download_dir, basename(download_url))
  
  tryCatch({
    download_file_if_missing(download_url, zip_dest)
    
    cell_extract_dir <- file.path(
      extract_dir,
      tools::file_path_sans_ext(basename(zip_dest))
    )
    
    if (!dir.exists(cell_extract_dir)) {
      dir_create(cell_extract_dir)
      unzip(zip_dest, exdir = cell_extract_dir)
    }
    
    s57_files <- dir_ls(
      cell_extract_dir,
      recurse = TRUE,
      regexp = "\\.000$",
      type = "file"
    )
    
    tibble(
      download_url = download_url,
      zip_path = zip_dest,
      extract_dir = cell_extract_dir,
      s57_path = as.character(s57_files)
    )
  }, error = function(e) {
    log_error_row(error_log, basename(download_url), NA, "download_unzip", e$message)
    
    tibble(
      download_url = download_url,
      zip_path = zip_dest,
      extract_dir = NA_character_,
      s57_path = NA_character_
    )
  })
}

# ----------------------------
# S-57 READERS
# ----------------------------

get_available_layers <- function(s57_path, error_log) {
  set_s57_feature_env()
  
  tryCatch({
    lyr <- st_layers(s57_path)$name
    
    tibble(
      source_s57 = basename(s57_path),
      s57_path = s57_path,
      layer_name = lyr
    )
  }, error = function(e) {
    log_error_row(error_log, basename(s57_path), NA, "st_layers", e$message)
    
    tibble(
      source_s57 = basename(s57_path),
      s57_path = s57_path,
      layer_name = NA_character_
    )
  })
}

read_s57_hazard_layer <- function(s57_path, layer_name, error_log) {
  tryCatch({
    set_s57_feature_env()
    
    g <- suppressWarnings(
      st_read(
        dsn = s57_path,
        layer = layer_name,
        quiet = TRUE,
        options = c("RETURN_PRIMITIVES=NO", "LNAM_REFS=ON")
      )
    )
    
    if (!inherits(g, "sf") || nrow(g) == 0) return(NULL)
    
    geom_col <- attr(g, "sf_column")
    
    g |>
      st_transform(4326) |>
      mutate(
        across(
          .cols = -all_of(geom_col),
          .fns = ~ as.character(.)
        ),
        source_s57 = basename(s57_path),
        source_path = s57_path,
        object_group = layer_name
      )
  }, error = function(e) {
    log_error_row(error_log, basename(s57_path), layer_name, "st_read", e$message)
    NULL
  })
}

read_s57_hazards_for_file <- function(
    s57_path,
    available_layers,
    hazard_layers,
    error_log
) {
  layer_names <- available_layers |>
    filter(.data$s57_path == s57_path) |>
    pull(layer_name) |>
    unique()
  
  layers_to_read <- intersect(hazard_layers, layer_names)
  
  if (length(layers_to_read) == 0) return(NULL)
  
  pieces <- map(
    layers_to_read,
    ~ read_s57_hazard_layer(s57_path, .x, error_log)
  )
  
  pieces <- pieces[!vapply(pieces, is.null, logical(1))]
  
  if (length(pieces) == 0) return(NULL)
  
  bind_rows(pieces)
}

read_s57_quapos_primitives <- function(s57_path, error_log) {
  set_s57_primitive_env()
  
  lyr <- tryCatch(
    st_layers(s57_path)$name,
    error = function(e) {
      log_error_row(error_log, basename(s57_path), NA, "primitive_st_layers", e$message)
      character(0)
    }
  )
  
  if (length(lyr) == 0) return(NULL)
  
  primitive_candidates <- lyr[
    str_detect(lyr, regex("Node|Edge|Face|Primitive|Vector", ignore_case = TRUE))
  ]
  
  if (length(primitive_candidates) == 0) return(NULL)
  
  pieces <- map(primitive_candidates, function(layer_name) {
    tryCatch({
      g <- suppressWarnings(
        st_read(
          dsn = s57_path,
          layer = layer_name,
          quiet = TRUE,
          options = c(
            "RETURN_PRIMITIVES=ON",
            "LIST_AS_STRING=ON",
            "PRESERVE_EMPTY_NUMBERS=ON"
          )
        )
      )
      
      if (!inherits(g, "sf") || nrow(g) == 0) return(NULL)
      if (!"QUAPOS" %in% names(g)) return(NULL)
      
      g <- g |>
        filter(!is.na(QUAPOS), as.character(QUAPOS) != "")
      
      if (nrow(g) == 0) return(NULL)
      
      g <- tryCatch(
        clean_sf_geometries(g, paste0("primitive ", basename(s57_path), " ", layer_name)),
        error = function(e) NULL
      )
      
      if (is.null(g) || nrow(g) == 0) return(NULL)
      
      geom_col <- attr(g, "sf_column")
      
      g |>
        st_transform(4326) |>
        mutate(
          across(
            .cols = -all_of(geom_col),
            .fns = ~ as.character(.)
          ),
          source_s57 = basename(s57_path),
          source_path = s57_path,
          primitive_layer = layer_name,
          primitive_id = paste0(
            basename(s57_path), "_",
            layer_name, "_",
            row_number()
          )
        )
    }, error = function(e) {
      log_error_row(error_log, basename(s57_path), layer_name, "primitive_st_read", e$message)
      NULL
    })
  })
  
  pieces <- pieces[!vapply(pieces, is.null, logical(1))]
  
  if (length(pieces) == 0) return(NULL)
  
  bind_rows(pieces)
}

# ----------------------------
# QUAPOS + CLASSIFICATION
# ----------------------------

attach_quapos_from_primitives <- function(
    hazards,
    quapos_primitives,
    error_log,
    max_join_distance_m = 25
) {
  if (is.null(quapos_primitives) || nrow(quapos_primitives) == 0) {
    hazards$QUAPOS_primitive <- NA_character_
    hazards$QUAPOS_source <- NA_character_
    hazards$QUAPOS_join_distance_m <- NA_real_
    return(hazards)
  }
  
  hazards <- clean_sf_geometries(hazards, "hazards before primitive QUAPOS join")
  quapos_primitives <- clean_sf_geometries(quapos_primitives, "QUAPOS primitives before join")
  
  hazards_m <- st_transform(hazards, 5070)
  prim_m <- st_transform(quapos_primitives, 5070)
  
  nearest_idx <- tryCatch(
    st_nearest_feature(hazards_m, prim_m),
    error = function(e) {
      log_error_row(error_log, "all", "QUAPOS primitives", "st_nearest_feature", e$message)
      rep(NA_integer_, nrow(hazards_m))
    }
  )
  
  good_idx <- !is.na(nearest_idx)
  
  joined_quapos <- rep(NA_character_, nrow(hazards_m))
  d_m <- rep(NA_real_, nrow(hazards_m))
  
  if (any(good_idx)) {
    nearest_geom <- prim_m[nearest_idx[good_idx], ]
    
    d_tmp <- as.numeric(st_distance(
      st_geometry(hazards_m[good_idx, ]),
      st_geometry(nearest_geom),
      by_element = TRUE
    ))
    
    q_tmp <- as.character(nearest_geom$QUAPOS)
    
    q_tmp[d_tmp > max_join_distance_m] <- NA_character_
    d_tmp[d_tmp > max_join_distance_m] <- NA_real_
    
    joined_quapos[good_idx] <- q_tmp
    d_m[good_idx] <- d_tmp
  }
  
  hazards |>
    mutate(
      QUAPOS_primitive = joined_quapos,
      QUAPOS_source = case_when(
        !is.na(QUAPOS_primitive) ~ "S57 primitive/vector record",
        TRUE ~ NA_character_
      ),
      QUAPOS_join_distance_m = d_m
    )
}

add_xy_fields <- function(g) {
  g <- clean_sf_geometries(g, "hazards before XY creation")
  
  g_4326 <- st_transform(g, 4326)
  g_m <- st_transform(g_4326, 5070)
  
  pts_m <- suppressWarnings(st_point_on_surface(g_m))
  pts_m <- clean_sf_geometries(pts_m, "representative hazard points")
  
  pts_4326 <- st_transform(pts_m, 4326)
  xy <- st_coordinates(pts_4326)
  
  g_4326 |>
    mutate(
      lon_wgs84 = xy[, 1],
      lat_wgs84 = xy[, 2],
      geometry_type = as.character(st_geometry_type(g_4326))
    )
}

add_hazard_classification <- function(g) {
  needed <- c(
    "QUAPOS", "QUAPOS_primitive", "STATUS", "VALSOU", "WATLEV",
    "QUASOU", "SOUACC", "VERACC", "HORACC", "CATWRK",
    "EXPSOU", "SORDAT", "SORIND", "INFORM"
  )
  
  for (nm in needed) {
    if (!nm %in% names(g)) g[[nm]] <- NA_character_
  }
  
  g |>
    mutate(
      hazard_id = paste0(object_group, "_", source_s57, "_", row_number()),
      
      hazard_type = case_when(
        object_group == "OBSTRN" ~ "Obstruction",
        object_group == "WRECKS" ~ "Wreck",
        object_group == "PILPNT" ~ "Pile",
        object_group == "UWTROC" ~ "Underwater / awash rock",
        TRUE ~ "Other"
      ),
      
      # QUAPOS = Quality of Position
      # 1  surveyed
      # 2  unsurveyed
      # 3  inadequately surveyed
      # 4  approximate
      # 5  position doubtful
      # 6  unreliable
      # 7  reported, not surveyed
      # 8  reported, not confirmed
      # 9  estimated
      # 10 precisely known
      # 11 calculated
      QUAPOS_final = coalesce(
        as.character(QUAPOS),
        as.character(QUAPOS_primitive)
      ),
      
      QUAPOS_code = str_extract(as.character(QUAPOS_final), "\\d+"),
      QUAPOS_num = suppressWarnings(as.integer(QUAPOS_code)),
      
      QUAPOS_method_note = case_when(
        !is.na(QUAPOS) ~ "Read directly from S-57 feature object",
        is.na(QUAPOS) & !is.na(QUAPOS_primitive) ~
          "Transferred from nearest S-57 primitive/vector record",
        TRUE ~ "No QUAPOS available"
      ),
      
      QUAPOS_label = case_when(
        QUAPOS_num == 1  ~ "surveyed",
        QUAPOS_num == 2  ~ "unsurveyed",
        QUAPOS_num == 3  ~ "inadequately surveyed",
        QUAPOS_num == 4  ~ "approximate",
        QUAPOS_num == 5  ~ "position doubtful",
        QUAPOS_num == 6  ~ "unreliable",
        QUAPOS_num == 7  ~ "reported, not surveyed",
        QUAPOS_num == 8  ~ "reported, not confirmed",
        QUAPOS_num == 9  ~ "estimated",
        QUAPOS_num == 10 ~ "precisely known",
        QUAPOS_num == 11 ~ "calculated",
        TRUE ~ NA_character_
      ),
      
      STATUS_raw = as.character(STATUS),
      INFORM_lc = tolower(coalesce(as.character(INFORM), "")),
      STATUS_lc = tolower(coalesce(as.character(STATUS), "")),
      
      pos_unverified = case_when(
        QUAPOS_num %in% 2:9 ~ TRUE,
        str_detect(
          paste(STATUS_lc, INFORM_lc),
          "approx|doubt|reported|unconfirmed|existence"
        ) ~ TRUE,
        TRUE ~ FALSE
      ),
      
      pos_known_or_null = case_when(
        is.na(QUAPOS_num) ~ TRUE,
        QUAPOS_num %in% c(1, 10, 11) ~ TRUE,
        TRUE ~ FALSE
      ),
      
      methodology_group = case_when(
        object_group %in% c("OBSTRN", "WRECKS", "PILPNT", "UWTROC") &
          pos_unverified ~ "02_unverified_reported_error",
        
        object_group %in% c("OBSTRN", "WRECKS", "PILPNT", "UWTROC") &
          pos_known_or_null ~ "01_known_hazard",
        
        TRUE ~ "other"
      ),
      
      hazard_weight = case_when(
        methodology_group == "02_unverified_reported_error" ~ 3,
        methodology_group == "01_known_hazard" ~ 1,
        TRUE ~ 0
      ),
      
      known_hazard_weight = if_else(methodology_group == "01_known_hazard", 1, 0),
      reported_error_weight = if_else(methodology_group == "02_unverified_reported_error", 3, 0)
    ) |>
    select(-INFORM_lc, -STATUS_lc)
}

# ============================================================
# NEIGHBOURHOOD HAZARD SCORING
# ============================================================

calculate_neighbourhood_scores <- function(
    hazards,
    search_radius_nm = 2
) {
  
  if (nrow(hazards) == 0) {
    hazards$hazard_score <- numeric(0)
    hazards$known_hazard_score <- numeric(0)
    hazards$reported_error_score <- numeric(0)
    hazards$neighbour_hazard_count <- integer(0)
    
    return(hazards)
  }
  
  search_radius_m <- search_radius_nm * 1852
  
  hazards_m <- hazards |>
    sf::st_transform(6350)
  
  hazard_pts <- suppressWarnings(
    sf::st_point_on_surface(hazards_m)
  )
  
  neighbours <- sf::st_is_within_distance(
    hazard_pts,
    hazard_pts,
    dist = search_radius_m
  )
  
  hazards_m$neighbour_hazard_count <-
    purrr::map_int(
      neighbours,
      ~ max(length(.x) - 1, 0)
    )
  
  hazards_m$hazard_score <-
    purrr::map_dbl(
      neighbours,
      ~ sum(
        hazards_m$hazard_weight[.x],
        na.rm = TRUE
      )
    )
  
  hazards_m$known_hazard_score <-
    purrr::map_dbl(
      neighbours,
      ~ sum(
        hazards_m$known_hazard_weight[.x],
        na.rm = TRUE
      )
    )
  
  hazards_m$reported_error_score <-
    purrr::map_dbl(
      neighbours,
      ~ sum(
        hazards_m$reported_error_weight[.x],
        na.rm = TRUE
      )
    )
  
  sf::st_transform(hazards_m, 4326)
}


# ============================================================
# BUFFERED DISPLAY LAYER
# ============================================================

make_buffered_display_layer <- function(
    hazards,
    buffer_nm = 2
) {
  
  if (nrow(hazards) == 0) {
    return(hazards)
  }
  
  buffer_m <- buffer_nm * 1852
  
  hazards |>
    sf::st_transform(6350) |>
    sf::st_buffer(buffer_m) |>
    sf::st_make_valid()
}

# ----------------------------
# STAGE 1: BUILD HAZARD CACHE ONCE
# ----------------------------

build_noaa_s57_hazard_cache <- function(
    work_dir,
    mask_dir = NULL,
    regions = "all",
    force_rebuild = FALSE,
    use_parallel = TRUE,
    n_workers = max(1, parallel::detectCores() - 2)
) {
  dir_create(work_dir)
  
  download_dir <- file.path(work_dir, "enc_downloads")
  extract_dir  <- file.path(work_dir, "enc_extracted")
  log_dir      <- file.path(work_dir, "logs")
  cache_dir    <- file.path(work_dir, "cache")
  
  dir_create(download_dir)
  dir_create(extract_dir)
  dir_create(log_dir)
  dir_create(cache_dir)
  
  error_log <- file.path(log_dir, "s57_hazard_error_log.csv")
  layer_log <- file.path(log_dir, "s57_available_layers_log.csv")
  
  cache_region_tag <- if (identical(regions, "all")) {
    "all"
  } else {
    paste(regions, collapse = "_")
  }
  
  cache_gpkg <- file.path(
    cache_dir,
    paste0("noaa_s57_hazard_feature_cache_", cache_region_tag, ".gpkg")
  )

  cache_layer <- "hazard_features_cache"
  
  
  
  if (file.exists(cache_gpkg) && !force_rebuild) {
    message("Using existing hazard cache: ", cache_gpkg)
    return(list(cache_gpkg = cache_gpkg, cache_layer = cache_layer))
  }
  
  if (use_parallel) {
    future::plan(multisession, workers = n_workers)
  } else {
    future::plan(sequential)
  }
  
  
  hazard_layers <- c("OBSTRN", "WRECKS", "PILPNT", "UWTROC")
  
  enc_catalog <- read_enc_catalog_urls()
  message("Catalog cells found: ", nrow(enc_catalog))
  
  enc_files <- future_map_dfr(
    enc_catalog$download_url,
    download_and_unzip_cell,
    download_dir = download_dir,
    extract_dir = extract_dir,
    error_log = error_log,
    .options = furrr_options(seed = TRUE)
  ) |>
    filter(!is.na(s57_path), file.exists(s57_path))
  
  message("S-57 files available: ", nrow(enc_files))
  
  available_layers <- map_dfr(enc_files$s57_path, get_available_layers, error_log = error_log)
  write_csv(available_layers, layer_log)
  
  hazard_parts <- future_map(
    enc_files$s57_path,
    read_s57_hazards_for_file,
    available_layers = available_layers,
    hazard_layers = hazard_layers,
    error_log = error_log,
    .options = furrr_options(seed = TRUE)
  )
  
  hazard_parts <- hazard_parts[
    vapply(hazard_parts, function(x) inherits(x, "sf") && nrow(x) > 0, logical(1))
  ]
  
  if (length(hazard_parts) == 0) stop("No hazard features read.")
  
  hazards_raw <- bind_rows(hazard_parts) |>
    clean_sf_geometries("hazards_raw full cache")
  
  # Optional early clip to union of masks
  # Optional early clip to union of masks
  if (!is.null(mask_dir)) {
    mask_tbl <- get_mask_table(mask_dir, regions)
    
    # Use the first mask as the shared EPSG:6350 CRS reference
    mask_ref_path <- mask_tbl$mask_path[1]
    mask_crs <- terra::crs(terra::rast(mask_ref_path))
    
    mask_aois <- purrr::map(mask_tbl$mask_path, make_mask_aoi)
    
    mask_union <- dplyr::bind_rows(mask_aois) |>
      sf::st_transform(mask_crs) |>
      sf::st_union() |>
      sf::st_as_sf()
    
    hazards_raw <- hazards_raw |>
      sf::st_transform(mask_crs) |>
      sf::st_filter(mask_union, .predicate = sf::st_intersects) |>
      clean_sf_geometries("hazards_raw mask union clipped") |>
      sf::st_transform(4326)
  }
  
  s57_files_with_hazards <- hazards_raw |>
    st_drop_geometry() |>
    distinct(source_path) |>
    pull(source_path)
  
  message("Reading QUAPOS primitive/vector records sequentially...")
  
  quapos_primitive_parts <- map(
    unique(s57_files_with_hazards),
    read_s57_quapos_primitives,
    error_log = error_log
  )
  
  quapos_primitive_parts <- quapos_primitive_parts[
    vapply(quapos_primitive_parts, function(x) inherits(x, "sf") && nrow(x) > 0, logical(1))
  ]
  
  quapos_primitives <- if (length(quapos_primitive_parts) > 0) {
    bind_rows(quapos_primitive_parts)
  } else {
    NULL
  }
  
  hazards_with_quapos <- attach_quapos_from_primitives(
    hazards = hazards_raw,
    quapos_primitives = quapos_primitives,
    error_log = error_log,
    max_join_distance_m = 25
  )
  
  hazards <- hazards_with_quapos |>
    add_xy_fields() |>
    add_hazard_classification() |>
    calculate_neighbourhood_scores(
      search_radius_nm = 2
    )
  
  
  
  if (file.exists(cache_gpkg)) file.remove(cache_gpkg)
  
  st_write(
    hazards,
    cache_gpkg,
    layer = cache_layer,
    quiet = FALSE,
    append = FALSE
  )
  
  message("Wrote hazard cache: ", cache_gpkg)
  
  list(cache_gpkg = cache_gpkg, cache_layer = cache_layer)
}



# ============================================================
# STAGE 2: ECOREGION GRID SUMMARIES + 100 m TIFF OUTPUTS
# Uses:
#   - Existing hazard cache GPKG
#   - EPSG:6350 100 m binary masks
# Produces:
#   - 500 m polygon grid summary GPKG
#   - 100 m TIFF rasterized from the 500 m grid summary
# ============================================================

get_mask_table <- function(mask_dir, regions = "all") {
  mask_files <- list.files(
    mask_dir,
    pattern = "\\.tif(f)?$",
    full.names = TRUE
  )
  
  mask_tbl <- tibble::tibble(
    mask_path = mask_files,
    region_id = stringr::str_extract(basename(mask_files), "ER_[0-9]+")
  ) |>
    dplyr::filter(!is.na(region_id)) |>
    dplyr::arrange(region_id)
  
  if (!identical(regions, "all")) {
    mask_tbl <- mask_tbl |>
      dplyr::filter(region_id %in% regions)
  }
  
  if (nrow(mask_tbl) == 0) {
    stop("No matching EPSG:6350 100 m mask TIFFs found in: ", mask_dir)
  }
  
  mask_tbl
}


make_mask_aoi <- function(mask_path, aoi_grid_m = 500) {
  m <- terra::rast(mask_path)
  
  fact <- max(1, round(aoi_grid_m / terra::res(m)[1]))
  
  m_aoi <- terra::aggregate(
    m,
    fact = fact,
    fun = "max",
    na.rm = TRUE
  )
  
  m_aoi <- terra::ifel(m_aoi == 1, 1, NA)
  
  poly <- terra::as.polygons(
    m_aoi,
    dissolve = TRUE,
    na.rm = TRUE
  )
  
  sf::st_as_sf(poly) |>
    sf::st_make_valid()
}



make_summary_grid_from_mask <- function(mask_path, grid_m = 500, region_id) {
  m <- terra::rast(mask_path)
  
  fact <- max(1, round(grid_m / terra::res(m)[1]))
  
  m_grid <- terra::aggregate(
    m,
    fact = fact,
    fun = "max",
    na.rm = TRUE
  )
  
  m_grid <- terra::ifel(m_grid == 1, 1, NA)
  
  grid_poly <- terra::as.polygons(
    m_grid,
    dissolve = FALSE,
    na.rm = TRUE
  )
  
  sf::st_as_sf(grid_poly) |>
    sf::st_make_valid() |>
    dplyr::mutate(
      region_id = region_id,
      grid_id = paste0(region_id, "_grid_", dplyr::row_number())
    )
}

# ============================================================
# SAFE MAX
# Avoids max() returning -Inf for empty groups
# ============================================================

# ============================================================
# CLEAN PRODUCT GENERATION SECTION
# ============================================================

safe_max <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(0)
  max(x)
}

make_buffered_product_geometry <- function(hazards, buffer_nm = 2, crs_m) {
  if (is.null(hazards) || nrow(hazards) == 0) return(hazards)
  
  buffer_m <- buffer_nm * 1852
  
  hazards |>
    sf::st_transform(crs_m) |>
    sf::st_buffer(buffer_m) |>
    sf::st_make_valid()
}

summarize_hazards_to_grid <- function(
    grid,
    hazards,
    product_type = c("P1_true_geometry", "P2_buffered_geometry")
) {
  product_type <- match.arg(product_type)
  
  grid_crs <- sf::st_crs(grid)
  
  grid_m <- sf::st_transform(grid, grid_crs)
  hazards_m <- sf::st_transform(hazards, grid_crs)
  
  if (nrow(hazards_m) == 0) {
    return(
      grid_m |>
        dplyr::mutate(
          product_type = product_type,
          total_hazard_count = 0,
          known_hazard_count = 0,
          unverified_reported_error_count = 0,
          obstruction_count = 0,
          wreck_count = 0,
          pile_count = 0,
          rock_count = 0,
          total_weighted_hazard_score = 0,
          known_hazard_score = 0,
          reported_error_score = 0
        )
    )
  }
  
  joined <- sf::st_join(
    grid_m,
    hazards_m |>
      dplyr::select(
        hazard_id,
        object_group,
        methodology_group,
        hazard_score,
        known_hazard_score,
        reported_error_score
      ),
    join = sf::st_intersects,
    left = TRUE
  )
  
  # Important:
  # P1 can SUM because true hazard geometry should not massively overlap.
  # P2 must use MAX because every hazard already carries its full 2 nm score.
  # Summing overlapping buffers double/triple counts the same neighbourhood score.
  score_fun <- safe_max
  
  summary_tbl <- joined |>
    sf::st_drop_geometry() |>
    dplyr::group_by(grid_id) |>
    dplyr::summarise(
      total_hazard_count = sum(!is.na(hazard_id)),
      
      known_hazard_count = sum(
        methodology_group == "01_known_hazard",
        na.rm = TRUE
      ),
      
      unverified_reported_error_count = sum(
        methodology_group == "02_unverified_reported_error",
        na.rm = TRUE
      ),
      
      obstruction_count = sum(object_group == "OBSTRN", na.rm = TRUE),
      wreck_count       = sum(object_group == "WRECKS", na.rm = TRUE),
      pile_count        = sum(object_group == "PILPNT", na.rm = TRUE),
      rock_count        = sum(object_group == "UWTROC", na.rm = TRUE),
      
      total_weighted_hazard_score = score_fun(hazard_score),
      known_hazard_score          = score_fun(known_hazard_score),
      reported_error_score        = score_fun(reported_error_score),
      
      .groups = "drop"
    )
  
  grid_m |>
    dplyr::left_join(summary_tbl, by = "grid_id") |>
    dplyr::mutate(
      product_type = product_type,
      dplyr::across(where(is.numeric), ~ dplyr::coalesce(.x, 0))
    )
}


rasterize_grid_summary_to_mask <- function(
    grid_summary,
    mask_path,
    out_tif,
    value_field = "total_weighted_hazard_score"
) {
  if (!value_field %in% names(grid_summary)) {
    stop("value_field not found in grid_summary: ", value_field)
  }
  
  template <- terra::rast(mask_path)
  template <- terra::ifel(template == 1, 1, NA)
  
  grid_summary_mask_crs <- sf::st_transform(
    grid_summary,
    terra::crs(template)
  )
  
  r <- terra::rasterize(
    terra::vect(grid_summary_mask_crs),
    template,
    field = value_field,
    touches = TRUE,
    background = NA
  )
  
  r <- terra::mask(r, template)
  # IMPORTANT: remove zero-score cells from display/output
  r <- terra::ifel(r <= 0, NA, r)
  
  terra::writeRaster(
    r,
    out_tif,
    overwrite = TRUE,
    gdal = c(
      "COMPRESS=LZW",
      "TILED=YES",
      "BIGTIFF=IF_SAFER"
    )
  )
  
  invisible(r)
}


safe_remove_file <- function(path) {
  if (file.exists(path)) {
    removed <- file.remove(path)
    
    if (!removed) {
      timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
      new_path <- sub("\\.gpkg$", paste0("_", timestamp, ".gpkg"), path)
      message("Existing file locked. Writing to: ", new_path)
      return(new_path)
    }
  }
  
  path
}


run_hazard_summaries_by_ecoregion <- function(
    work_dir,
    mask_dir,
    cache_gpkg,
    cache_layer = "hazard_features_cache",
    regions = "all",
    grid_m = 500,
    value_field = "total_weighted_hazard_score",
    buffer_nm = 2
) {
  
  out_dir <- file.path(work_dir, "outputs_by_ecoregion")
  fs::dir_create(out_dir)
  
  mask_tbl <- get_mask_table(mask_dir, regions)
  
  hazards_all <- sf::st_read(
    cache_gpkg,
    layer = cache_layer,
    quiet = FALSE
  ) |>
    clean_sf_geometries("cached hazards")
  
  results <- list()
  
  for (i in seq_len(nrow(mask_tbl))) {
    
    region_id <- mask_tbl$region_id[i]
    mask_path <- mask_tbl$mask_path[i]
    
    message("===================================")
    message("Processing ", region_id)
    message("===================================")
    
    mask_crs <- terra::crs(terra::rast(mask_path))
    
    mask_aoi <- make_mask_aoi(
      mask_path = mask_path,
      aoi_grid_m = grid_m
    ) |>
      sf::st_transform(mask_crs)
    
    hazards_region <- hazards_all |>
      sf::st_transform(mask_crs) |>
      sf::st_filter(mask_aoi, .predicate = sf::st_intersects)
    
    if (nrow(hazards_region) > 0) {
      hazards_region <- clean_sf_geometries(
        hazards_region,
        paste0(region_id, " hazards")
      )
    }
    
    message(region_id, " true hazards: ", nrow(hazards_region))
    
    summary_grid <- make_summary_grid_from_mask(
      mask_path = mask_path,
      grid_m = grid_m,
      region_id = region_id
    )
    
    # ========================================================
    # P1: TRUE HAZARD GEOMETRY
    # ========================================================
    
    p1_grid_summary <- summarize_hazards_to_grid(
      grid = summary_grid,
      hazards = hazards_region,
      product_type = "P1_true_geometry"
    )
    
    # ========================================================
    # P2: BUFFERED HAZARD GEOMETRY
    # Same scores, buffered geometry.
    # Do NOT recalculate hazard_score here.
    # ========================================================
    
    hazards_region_buffered <- make_buffered_product_geometry(
      hazards = hazards_region,
      buffer_nm = buffer_nm,
      crs_m = mask_crs
    )
    
    p2_grid_summary <- summarize_hazards_to_grid(
      grid = summary_grid,
      hazards = hazards_region_buffered,
      product_type = "P2_buffered_geometry"
    )
    message(region_id, " P1 nonzero cells: ", sum(p1_grid_summary[[value_field]] > 0, na.rm = TRUE))
    message(region_id, " P2 nonzero cells: ", sum(p2_grid_summary[[value_field]] > 0, na.rm = TRUE))
    
    # ========================================================
    # OUTPUTS
    # ========================================================
    
    out_gpkg_p1 <- file.path(
      out_dir,
      paste0(region_id, "_P1_true_geometry_hazard_grid_", grid_m, "m.gpkg")
    )
    
    out_gpkg_p2 <- file.path(
      out_dir,
      paste0(region_id, "_P2_buffered_2nm_hazard_grid_", grid_m, "m.gpkg")
    )
    
    out_tif_p1 <- file.path(
      out_dir,
      paste0(region_id, "_P1_true_geometry_", value_field, "_100m.tif")
    )
    
    
    out_tif_p2 <- file.path(
      out_dir,
      paste0(region_id, "_P2_buffered_2nm_", value_field, "_100m.tif")
    )
    
    sf::st_write(
      p1_grid_summary,
      safe_remove_file(out_gpkg_p1),
      layer = paste0(region_id, "_P1_true_geometry"),
      append = FALSE,
      quiet = FALSE
    )
    
    sf::st_write(
      p2_grid_summary,
      safe_remove_file(out_gpkg_p2),
      layer = paste0(region_id, "_P2_buffered_2nm"),
      append = FALSE,
      quiet = FALSE
    )
    
    rasterize_grid_summary_to_mask(
      grid_summary = p1_grid_summary,
      mask_path = mask_path,
      out_tif = out_tif_p1,
      value_field = value_field
    )
    
    rasterize_grid_summary_to_mask(
      grid_summary = p2_grid_summary,
      mask_path = mask_path,
      out_tif = out_tif_p2,
      value_field = value_field
    )
    
    message(region_id, " P1 score range:")
    print(range(p1_grid_summary[[value_field]], na.rm = TRUE))
    
    message(region_id, " P2 score range:")
    print(range(p2_grid_summary[[value_field]], na.rm = TRUE))
    
    results[[region_id]] <- list(
      region_id = region_id,
      gpkg_p1 = out_gpkg_p1,
      gpkg_p2 = out_gpkg_p2,
      tif_p1 = out_tif_p1,
      tif_p2 = out_tif_p2,
      n_hazards = nrow(hazards_region),
      p1_score_range = range(p1_grid_summary[[value_field]], na.rm = TRUE),
      p2_score_range = range(p2_grid_summary[[value_field]], na.rm = TRUE)
    )
  }
  
  invisible(results)
}







# ----------------------------
# ONE CLEAN CALL FUNCTION
# ----------------------------

run_full_enc_hazard_ecoregion_workflow <- function(
    work_dir,
    mask_dir,
    regions = "all",
    grid_m = 500,
    value_field = "total_weighted_hazard_score",
    force_rebuild_cache = FALSE,
    delete_enc_zips_after_cache = FALSE,
    delete_extracted_enc_after_cache = FALSE,
    use_parallel = TRUE,
    n_workers = 6,
    buffer_nm = 2
) {
  cache <- build_noaa_s57_hazard_cache(
    work_dir = work_dir,
    mask_dir = mask_dir,
    regions = regions,
    force_rebuild = force_rebuild_cache,
    use_parallel = use_parallel,
    n_workers = n_workers
  )
  
  results <- run_hazard_summaries_by_ecoregion(
    work_dir = work_dir,
    mask_dir = mask_dir,
    cache_gpkg = cache$cache_gpkg,
    cache_layer = cache$cache_layer,
    regions = regions,
    grid_m = grid_m,
    value_field = value_field,
    buffer_nm = buffer_nm
    
  )
  
  if (delete_enc_zips_after_cache) {
    unlink(file.path(work_dir, "enc_downloads"), recursive = TRUE, force = TRUE)
  }
  
  if (delete_extracted_enc_after_cache) {
    unlink(file.path(work_dir, "enc_extracted"), recursive = TRUE, force = TRUE)
  }
  
  invisible(
    list(
      cache = cache,
      summaries = results
    )
  )
}



# function calls

# single Region 
result_er6 <- run_full_enc_hazard_ecoregion_workflow(
  work_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards",
  mask_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/Eco_Region_Masks/epsg_6350_100m",
  regions = "ER_6",
  grid_m = 500,
  #output_resolution_m = 100,
  value_field = "total_weighted_hazard_score",
  force_rebuild_cache = FALSE, # leave this on for testing - turn off for the main run
  delete_enc_zips_after_cache = FALSE,
  delete_extracted_enc_after_cache = FALSE,
  use_parallel = TRUE,
  n_workers = 6,
  buffer_nm = 2
)


# All regions
result_all <- run_full_enc_hazard_ecoregion_workflow(
  work_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards",
  mask_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/Eco_Region_Masks/epsg_6350_100m",
  regions = "all",
  grid_m = 500,
  value_field = "total_weighted_hazard_score",
  force_rebuild_cache = TRUE,
  delete_enc_zips_after_cache = FALSE,
  delete_extracted_enc_after_cache = FALSE,
  use_parallel = TRUE,
  n_workers = 6,
  buffer_nm = 2
)





# result_all <- run_full_enc_hazard_ecoregion_workflow(
#   work_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards",
#   mask_dir = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/Eco_Region_Masks/epsg_6350",
#   regions = "all",
#   grid_m = 500,
#   value_field = "total_weighted_hazard_score",
#   force_rebuild_cache = FALSE,
#   delete_enc_zips_after_cache = TRUE,
#   delete_extracted_enc_after_cache = TRUE,
#   use_parallel = TRUE,
#   n_workers = 6
# )

# # Verify everything has run as expected for the test run 
# verify_hazard_cache <- function(
#     cache_gpkg,
#     cache_layer = "hazard_features_cache",
#     error_log = NULL
# ) {
#   library(sf)
#   library(dplyr)
#   library(readr)
#   
#   hazards <- st_read(cache_gpkg, layer = cache_layer, quiet = TRUE)
#   
#   cat("\nFeature counts by object_group:\n")
#   print(
#     hazards |>
#       st_drop_geometry() |>
#       count(object_group, sort = TRUE)
#   )
#   
#   cat("\nFeature counts by source ENC and object_group:\n")
#   print(
#     hazards |>
#       st_drop_geometry() |>
#       count(source_s57, object_group, sort = TRUE)
#   )
#   
#   cat("\nQUAPOS coverage:\n")
#   print(
#     hazards |>
#       st_drop_geometry() |>
#       summarise(
#         total_features = n(),
#         quapos_final_present = sum(!is.na(QUAPOS_final)),
#         quapos_primitive_present = sum(!is.na(QUAPOS_primitive)),
#         quapos_feature_present = sum(!is.na(QUAPOS))
#       )
#   )
#   
#   cat("\nMethodology group counts:\n")
#   print(
#     hazards |>
#       st_drop_geometry() |>
#       count(methodology_group, sort = TRUE)
#   )
#   
#   if (!is.null(error_log) && file.exists(error_log)) {
#     err <- read_csv(error_log, show_col_types = FALSE)
#     
#     cat("\nLogged read issues by layer/step:\n")
#     print(
#       err |>
#         count(step, layer, message, sort = TRUE)
#     )
#   }
#   
#   invisible(hazards)
# }
# 
# 
# verify_hazard_cache(
#   cache_gpkg = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards/cache/noaa_s57_hazard_feature_cache.gpkg",
#   error_log = "N:/CSDL/Projects/Hydro_Health_Model/HHM2025/working/HHM_Run/ENC_hazards/logs/s57_hazard_error_log.csv"
# )