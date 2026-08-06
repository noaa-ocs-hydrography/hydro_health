import arcpy

def subdivide_polygon(extent, levels):
    """
    Recursively divides a bounding extent into 4 equal quadrants.
    levels = 1 ->  4 sub-tiles (1/4)
    levels = 2 -> 16 sub-tiles (1/16)
    levels = 3 -> 64 sub-tiles (1/64)
    """
    xmin, ymin, xmax, ymax = extent.XMin, extent.YMin, extent.XMax, extent.YMax
    
    # Base case: stopped dividing, return geometry as Array of Points
    if levels == 0:
        pts = arcpy.Array([
            arcpy.Point(xmin, ymax), # Top-Left
            arcpy.Point(xmax, ymax), # Top-Right
            arcpy.Point(xmax, ymin), # Bottom-Right
            arcpy.Point(xmin, ymin), # Bottom-Left
            arcpy.Point(xmin, ymax)  # Close Polygon
        ])
        return [arcpy.Polygon(pts)]

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
        sub_polygons.extend(subdivide_polygon(quad, levels - 1))
        
    return sub_polygons


def create_subgrids(input_fc, output_fc, subdivision_factor=4):
    """
    Reads parent grid feature class and outputs subdivided grid feature class.
    subdivision_factor: Options are 4 (1/4), 16 (1/16), 64 (1/64)
    """
    # Map division factor to recursion level count
    division_levels = {
        4: 1,   # 4^1
        16: 2,  # 4^2
        64: 3,   # 4^3
        256: 4   # 4^4
    }

    if subdivision_factor not in division_levels:
        raise ValueError("subdivision_factor must be 4, 16, or 64.")

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
                extent = geom.extent
                
                # Generate child geometries for this grid tile
                child_polygons = subdivide_polygon(extent, levels)
                
                # Write to output feature class
                for sub_idx, child_geom in enumerate(child_polygons, start=1):
                    i_cursor.insertRow([child_geom, parent_oid, sub_idx])

    print(f"Successfully generated sub-grids at {output_fc}")


# --- EXECUTION EXAMPLE ---
if __name__ == "__main__":
    subdivision_number = 256
    # Specify your Geodatabase paths
    gdb_path = r"C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\GIS\HHM2025.gdb" # Or .gdb
    input_tiles = f"{gdb_path}\\medium_Blue_topo_Grid_Tiles"
    output_tiles = f"{gdb_path}\\grid_tile_division_{subdivision_number}"

    # Set factor: 4 (for 1/4), 16 (for 1/16), 64 (for 1/64)
    create_subgrids(
        input_fc=input_tiles,
        output_fc=output_tiles,
        subdivision_factor=subdivision_number
    )