#what this is: 
## 3 function to create a prediction mask extent over the scale of a given Eco Region, then create a new sub grid geopackage using the blue topo tessalation grid tile 
# scheme and the prediction mask, and then convert the sub grid gpkg in WGS84 for UTM for easier modelling in meters. 

library(raster)
library(terra)
library(sp)
library(dplyr)
library(sf)
library(foreach)
library(doParallel)
library(future)
library(future.apply)
library(fst)
library(sf)


# File Paths
# for function 1 
shapefile_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/GIS/Pilot_model/Eco_Region_3_FL.shp"
output_mask_pred_utm <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/HHM_Run/ER_3/prediction_masks/prediction.mask.UTM17_8m.tif"
output_mask_pred_wgs <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/HHM_Run/ER_3/prediction_masks/prediction.mask.WGS84_8m.tif"

# For function 2
grid_tiles_gpkg <-  "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg"
output_subgrid_gpkg <-  "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/HHM_Run/ER_3/processing_grids/testB10.prediction.subgrid.WGS84_8m.gpkg"
mask_path <- "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/HHM_Run/ER_3/prediction_masks/prediction.mask.WGS84_8m.tif"
  
## Function 1 - Create Prediction Mask from Shapefile (both WGS and UTM version)
create_prediction_mask <- function(shapefile_path, output_mask_utm, output_mask_wgs) {
  poly <- st_read(shapefile_path, quiet = TRUE)
  poly_utm <- st_transform(poly, crs = "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs")
  ext <- extent(poly_utm)
  template <- raster(ext, res = 8, crs = projection(poly_utm))
  
  mask_ras <- rasterize(poly_utm, template, field = 1, background = NA)
  mask_bin <- calc(mask_ras, fun = function(x) ifelse(is.na(x), 0, 1))
  
  writeRaster(mask_bin, output_mask_utm, overwrite = TRUE)
  writeRaster(projectRaster(mask_bin, crs = "+init=epsg:4326", method = "ngb"), output_mask_wgs, overwrite = TRUE)
  log_message("Prediction mask created (UTM & WGS)")
}
#run
create_prediction_mask(shapefile_path, output_mask_pred_utm, output_mask_pred_wgs)


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

#--- Generate full subgrid dataset ---
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


# run
generate_subgrids_from_mask_wgs84(
  grid_tiles_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/Pilot_model/Now_Coast_NBS_Data/Tessellation/Modeling_Tile_Scheme_20241205_151018.gpkg",
  output_subgrid_gpkg = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/HHM_Run/ER_3/processing_grids/testB10.prediction.subgrid.WGS84_8m.gpkg",
  reference_tile_id = "BH4S257K", # any of the smaller inner blue topo tiles
  mask_path = "N:/HSD/Projects/HSD_DATA/NHSP_2_0/HH_2024/working/HHM_Run/ER_3/prediction_masks/prediction.mask.WGS84_8m.tif",
  workers = 4  # tset number of cores
)


## Function 3 - Reprojects sub grid Gpkg into desired projection - UTM 
reproject_subgrids_to_utm <- function(input_gpkg, output_gpkg, target_crs) {
  log_message(paste("Reprojecting sub-grids to:", target_crs))  # Log target CRS
  
  tryCatch({
    sub_grids <- st_read(input_gpkg, quiet = TRUE)              # Read input geopackage
    sub_grids_utm <- st_transform(sub_grids, crs = target_crs)  # Project to UTM (or other EPSG)
    st_write(sub_grids_utm, output_gpkg, delete_layer = TRUE, quiet = TRUE)  # Write result
    log_message(paste("Reprojected sub-grids saved to:", output_gpkg))
  }, error = function(e) {
    log_message(paste("Failed to reproject sub-grids:", e$message))  # Catch errors
  })
}
#run
reproject_subgrids_to_utm(
  input_gpkg = output_subgrid_gpkg,
  output_gpkg = sub("WGS84", "UTM17N", output_subgrid_gpkg),
  target_crs = 32617 #UTM Zone 17N, need to think about how we make this variable accross different eco regions....
)
