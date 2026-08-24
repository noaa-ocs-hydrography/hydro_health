import arcpy

def subdivide_polygon(extent, levels, spatial_ref):
    """
    Recursively divides a bounding extent into 4 equal quadrants.
    Passes spatial_ref to ensure created polygons are spatially valid.
    """
    xmin, ymin, xmax, ymax = extent.XMin, extent.YMin, extent.XMax, extent.YMax
    
    # Base case: stopped dividing, return geometry as Array of Points
    if levels == 0:
        # Clockwise order: Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Top-Left
        pts = arcpy.Array([
            arcpy.Point(xmin, ymax), # Top-Left
            arcpy.Point(xmax, ymax), # Top-Right
            arcpy.Point(xmax, ymin), # Bottom-Right
            arcpy.Point(xmin, ymin), # Bottom-Left
            arcpy.Point(xmin, ymax)  # Close Polygon
        ])
        # IMPORTANT: Explicitly pass spatial_ref so geometry isn't created as 'Unknown SR'
        return [arcpy.Polygon(pts, spatial_ref)]

    # Calculate midpoints
    xmid = (xmin + xmax) / 2.0
    ymid = (ymin + ymax) / 2.0

    # Define 4 quadrant extents
    quadrants = [
        arcpy.Extent(xmin, ymid, xmid, ymax), # Top-Left (Tile 1)
        arcpy.Extent(xmid, ymid, xmax, ymax), # Top-Right (Tile 2)
        arcpy.Extent(xmin, ymin, xmid, ymid), # Bottom-Left (Tile 4)
        arcpy.Extent(xmid, ymin, xmax, ymid)  # Bottom-Right (Tile 3)
    ]

    # Recursively divide each quadrant
    sub_polygons = []
    for quad in quadrants:
        sub_polygons.extend(subdivide_polygon(quad, levels - 1, spatial_ref))
        
    return sub_polygons


def create_subgrids(input_fc, output_fc, subdivision_factor=4):
    """
    Reads parent grid feature class and outputs subdivided grid feature class.
    """
    division_levels = {
        4: 1,    # 4^1
        16: 2,   # 4^2
        64: 3,   # 4^3
        256: 4,  # 4^4
        1024: 5, # 4^5
        4096: 6  # 4^6
    }

    if subdivision_factor not in division_levels:
        raise ValueError(f"subdivision_factor must be one of {list(division_levels.keys())}")

    levels = division_levels[subdivision_factor]

    # Setup environment
    arcpy.env.overwriteOutput = True
    spatial_ref = arcpy.Describe(input_fc).spatialReference

    # Create new Output Feature Class in GDB
    out_dir, out_name = arcpy.os.path.split(output_fc)
    arcpy.management.CreateFeatureclass(
        out_path=out_dir,
        out_name=out_name,
        geometry_type="POLYGON",
        spatial_reference=spatial_ref
    )

    # Add fields to track parent relationship
    arcpy.management.AddField(output_fc, "Parent_ID", "LONG")
    arcpy.management.AddField(output_fc, "SubTile_ID", "SHORT")

    # Read input geometries and insert subdivided geometries
    search_fields = ["OID@", "SHAPE@"]
    insert_fields = ["SHAPE@", "Parent_ID", "SubTile_ID"]

    with arcpy.da.SearchCursor(input_fc, search_fields) as s_cursor:
        with arcpy.da.InsertCursor(output_fc, insert_fields) as i_cursor:
            for row in s_cursor:
                parent_oid, geom = row[0], row[1]
                if geom is None:
                    continue
                    
                extent = geom.extent
                
                # Generate child geometries passing the spatial reference
                child_polygons = subdivide_polygon(extent, levels, spatial_ref)
                
                # Write to output feature class
                for sub_idx, child_geom in enumerate(child_polygons, start=1):
                    i_cursor.insertRow([child_geom, parent_oid, sub_idx])

    # Re-calculate Spatial Index to ensure feature class renders immediately in Pro
    print(f"Recalculating Spatial Index for '{output_fc}'...")
    arcpy.management.AddSpatialIndex(output_fc)

    print(f"Successfully generated sub-grids at {output_fc}")


# --- EXECUTION EXAMPLE ---
if __name__ == "__main__":
    subdivision_number = 4096
    gdb_path = r"C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\GIS\HHM2025.gdb"
    input_tiles = f"{gdb_path}\\medium_Blue_topo_Grid_Tiles"
    output_tiles = f"{gdb_path}\\grid_tile_division_{subdivision_number}"

    create_subgrids(
        input_fc=input_tiles,
        output_fc=output_tiles,
        subdivision_factor=subdivision_number
    )