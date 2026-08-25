import gc
import arcpy
import rasterio
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
from shapely import wkb

# ==========================================
# 1. Configuration & File Paths
# ==========================================
gdb_path = r"C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\GIS\HHM2025.gdb"
in_layer_name = "grid_tile_division_4096"
out_layer_name = "averaged_abs_annual_change"

fc_full_path = f"{gdb_path}\\{in_layer_name}"
out_fc_path = f"{gdb_path}\\{out_layer_name}"
raster_path = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\Post_processing_logic\manipulated_outputs\pm_mean_absolute_annual_change_v1.tif"

# Adjust batch size depending on available RAM (10,000 is safe for large tile counts)
BATCH_SIZE = 10000

# ==========================================
# 2. Setup Spatial References & Feature Class
# ==========================================
arcpy.env.overwriteOutput = True

print(f"Reading spatial reference from '{in_layer_name}'...")
spatial_ref = arcpy.Describe(fc_full_path).spatialReference
epsg_code = spatial_ref.factoryCode if spatial_ref.factoryCode else 4326

print(f"Creating output feature class '{out_layer_name}' in GDB...")
arcpy.management.CreateFeatureclass(
    out_path=gdb_path,
    out_name=out_layer_name,
    geometry_type="POLYGON",
    spatial_reference=spatial_ref
)

# Add output fields to target feature class
arcpy.management.AddField(out_fc_path, "Parent_ID", "LONG")
arcpy.management.AddField(out_fc_path, "SubTile_ID", "LONG")
arcpy.management.AddField(out_fc_path, "mean_val", "DOUBLE")
arcpy.management.AddField(out_fc_path, "median_val", "DOUBLE")
arcpy.management.AddField(out_fc_path, "min_val", "DOUBLE")
arcpy.management.AddField(out_fc_path, "max_val", "DOUBLE")

# Get target CRS from raster file
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

# ==========================================
# 3. Batch Reader Generator
# ==========================================
def batch_cursor_reader(fc_path, fields, batch_size):
    """Yields batches of rows from an ArcPy SearchCursor."""
    batch = []
    with arcpy.da.SearchCursor(fc_path, fields) as cursor:
        for row in cursor:
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

# ==========================================
# 4. Process Batches & Compute Zonal Stats
# ==========================================
print("Starting batch processing...")

cursor_fields = ["Parent_ID", "SubTile_ID", "SHAPE@WKB"]
insert_fields = ["Parent_ID", "SubTile_ID", "mean_val", "median_val", "min_val", "max_val", "SHAPE@WKB"]

batch_num = 1
total_processed = 0

with arcpy.da.InsertCursor(out_fc_path, insert_fields) as ins_cursor:
    for row_batch in batch_cursor_reader(fc_full_path, cursor_fields, BATCH_SIZE):
        print(f"Processing Batch {batch_num} ({len(row_batch)} features)...")
        
        geometries = []
        metadata = []
        
        for parent_id, subtile_id, wkb_bytes in row_batch:
            if wkb_bytes:
                geom = wkb.loads(bytes(wkb_bytes))
                geometries.append(geom)
                metadata.append((parent_id, subtile_id))

        if not geometries:
            print(f"Warning: Batch {batch_num} contained no valid geometries. Skipping.")
            continue

        # Create batch GeoDataFrame
        batch_gdf = gpd.GeoDataFrame(
            metadata, 
            columns=["Parent_ID", "SubTile_ID"], 
            geometry=geometries, 
            crs=f"EPSG:{epsg_code}"
        )
        
        # Reproject batch to match raster CRS
        batch_gdf_reprojected = batch_gdf.to_crs(raster_crs)
        
        # Calculate zonal statistics using __geo_interface__ compatibility fix
        stats = zonal_stats(
            batch_gdf_reprojected.__geo_interface__,
            raster_path,
            stats=["mean", "median", "min", "max"],
            all_touched=True,
            geojson_out=False
        )
        
        # Write results directly to GDB feature class via InsertCursor
        for i, stat in enumerate(stats):
            p_id, sub_id = metadata[i]
            wkb_data = geometries[i].wkb
            
            m_val = float(stat["mean"]) if stat["mean"] is not None else None
            med_val = float(stat["median"]) if stat["median"] is not None else None
            mn_val = float(stat["min"]) if stat["min"] is not None else None
            mx_val = float(stat["max"]) if stat["max"] is not None else None
            
            ins_cursor.insertRow([p_id, sub_id, m_val, med_val, mn_val, mx_val, wkb_data])

        total_processed += len(row_batch)
        print(f"Completed Batch {batch_num}. Total processed: {total_processed}")
        batch_num += 1
        
        # Clean up batch variables and call garbage collector to free memory
        del row_batch, geometries, metadata, batch_gdf, batch_gdf_reprojected, stats
        gc.collect()

print(f"\nProcess completed successfully! Total features written to '{out_layer_name}': {total_processed}")