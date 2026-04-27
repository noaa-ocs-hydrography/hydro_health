import os
import pathlib
import arcpy
from arcgis.gis import GIS

def vector_upload_via_pro(folder_path, folder_name):
    print("Connecting via ArcGIS Pro active session...")
    try:
        gis = GIS("pro")
        print(f"Connected as: {gis.users.me.username} to {gis.properties.name}")
    except Exception as e:
        print("Failed to connect via 'pro'. Ensure ArcGIS Pro is open and signed in.")
        print(f"Error details: {e}")
        return

    # Set up arcpy workspace for temporary processing
    arcpy.env.overwriteOutput = True

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.tif', '.tiff')):
            file_path = pathlib.Path(folder_path) / file_name
            base_name = file_path.stem
            temp_shp = str(pathlib.Path(folder_path) / f"{base_name}_vector.shp")

            print(f"--- Processing {file_name} ---")
            
            try:
                # 1. Convert Raster to Vector (Polygon)
                # Field "Value" usually contains the raster pixel values
                print(f"Converting {file_name} to vector...")
                arcpy.RasterToPolygon_conversion(
                    in_raster=str(file_path),
                    out_polygon_features=temp_shp,
                    simplify="SIMPLIFY",
                    raster_field="Value"
                )

                # 2. Define Properties for AGO
                properties = {
                    'title': base_name,
                    'description': f'Vectorized Hydro Health 2025 - {file_name}',
                    'tags': ['HydroHealth', 'Vector'],
                    'type': 'Shapefile'
                }

                # 3. Upload the Shapefile and Publish as a Feature Layer
                print(f"Uploading and publishing {base_name} to AGO...")
                
                # We zip the shapefile components because AGO requires .zip for shp uploads
                # Note: arcgis.gis.ContentManager.add handles local files
                shp_item = gis.content.add(item_properties=properties, data=temp_shp, folder=folder_name)
                
                # Publish the shapefile item into a Feature Layer
                published_item = shp_item.publish()
                
                print(f"Successfully published: {published_item.title} (ID: {published_item.id})")

                # 4. Cleanup: Remove the temporary shapefile and the uploaded .zip item
                # (Optional) You can keep the shp_item if you want the source data on AGO
                shp_item.delete() 
                
                # Clean up local temp files created by arcpy
                for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
                    f = temp_shp.replace('.shp', ext)
                    if os.path.exists(f):
                        os.remove(f)

            except Exception as e:
                print(f"Failed to process {file_name}. Error: {e}")

LOCAL_FOLDER = r"C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs"
FOLDER_NAME = "ER3_Model_Outputs"

if __name__ == "__main__":
    vector_upload_via_pro(LOCAL_FOLDER, FOLDER_NAME)
    print('All tasks complete.')