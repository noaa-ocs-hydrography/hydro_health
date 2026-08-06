import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats

# Paths
gdb_path = r"C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\GIS\HHM2025.gdb"
feature_class_name = "grid_tile_division_256"
raster_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\manipulated_outputs\pm_mean_absolute_annual_change_v1.tif"

# 1. Load vector grid
gdf = gpd.read_file(gdb_path, layer=feature_class_name)

# 2. Get the raster's Coordinate Reference System (CRS)
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

# 3. REPROJECT the vector grid to match the raster!
print(f"Reprojecting vector grid from {gdf.crs} to match raster ({raster_crs})...")
gdf_reprojected = gdf.to_crs(raster_crs)

# 4. Run zonal statistics on the reprojected layer
# (Added all_touched=True so small boundary cells don't get missed)
stats = zonal_stats(
    gdf_reprojected,
    raster_path,
    stats=["mean", "median", "min", "max"],
    all_touched=True,
    geojson_out=False,
)

# 5. Append results back to your dataframe
stats_df = pd.DataFrame(stats)
gdf["mean_val"] = stats_df["mean"]
gdf["median_val"] = stats_df["median"]
gdf["min_val"] = stats_df["min"]
gdf["max_val"] = stats_df["max"]

# Check output in console
print("Results sample:")
print(gdf[["mean_val", "median_val", "min_val", "max_val"]].head())

# 6. Save back to Geodatabase (saved with original vector CRS)
gdf.to_file(
    gdb_path,
    layer="averaged_abs_annual_change",
    driver="OpenFileGDB",
    mode="w",
)