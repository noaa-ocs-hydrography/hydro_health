import os
import pathlib
import arcpy
import zipfile
import re
import json
import time
import gc
from arcgis.gis import GIS

# --- CONFIGURATION ---
BASE_FOLDER = r"C:\Users\aubrey.mccutchan\Documents\AGO_uploads"
OFFSHORE_DIR = pathlib.Path(BASE_FOLDER) / "Offshore_100m"
INSHORE_DIR = pathlib.Path(BASE_FOLDER) / "Inshore_20m"

# The folder in ArcGIS Online where these will be saved
AGO_FOLDER_NAME = "Hydro_Health_Outputs" 

# The exact title of the target Web Map where layers should be added
TARGET_WEBMAP_TITLE = "Hydro Health Model - Simple"

# Dictionary to customize output layer names on AGO
LAYER_NAME_KEY = {
    "HG": "Hydrographic Gap",
    "PSS": "Present Survey Score",
    "DSS": "Desired Survey Score",
    "ISS": "Initial Survey Score",
    "decay_coefficient": "Decay Coefficient"
}

def zip_shapefile(shp_path):
    """Zips a shapefile and all its associated component files."""
    shp_path = str(shp_path)
    zip_path = shp_path.replace('.shp', '.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Standard shapefile extensions
        for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml']:
            file_to_zip = shp_path.replace('.shp', ext)
            if os.path.exists(file_to_zip):
                zipf.write(file_to_zip, arcname=os.path.basename(file_to_zip))
                
    return zip_path

def cleanup_shapefile(shp_path):
    """Deletes a shapefile, its components, and its zip file from the local drive."""
    # Force Python's Garbage Collector to release memory handles on the zip file
    gc.collect()
    
    shp_path = str(shp_path)
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.xml', '.zip']:
        f = shp_path.replace('.shp', ext)
        if os.path.exists(f):
            # Aggressive retry loop to handle stubborn Windows file locks
            for attempt in range(5):
                try:
                    os.remove(f)
                    break # Success!
                except Exception as e:
                    if attempt == 4:
                        print(f"  [Warning] Could not delete {f}: {e}")
                    else:
                        time.sleep(1) # Wait a second and try again

def delete_existing_ago_items(gis, title):
    """Searches for and deletes existing items on AGO with the exact same title."""
    print(f"  Checking AGO for existing items named '{title}'...")
    # Search by exact title and current user
    query = f'title:"{title}" AND owner:"{gis.users.me.username}"'
    existing_items = gis.content.search(query=query, max_items=10)
    
    deleted_count = 0
    for item in existing_items:
        # AGO search can sometimes return fuzzy/partial matches, so we ensure an exact title match
        if item.title == title:
            print(f"    -> Deleting existing {item.type} (ID: {item.id})...")
            item.delete()
            deleted_count += 1
            
    if deleted_count == 0:
        print("    -> No existing items found. Proceeding with fresh upload.")

def add_layer_to_webmap(gis, webmap_title, feature_layer_item):
    """Adds a published Feature Layer to a specific Web Map, replacing older versions if present."""
    print(f"  Adding '{feature_layer_item.title}' to Web Map '{webmap_title}'...")
    try:
        # Removed the 'owner' requirement so it finds maps owned by other team members (like Stephen)
        query = f'title:"{webmap_title}" AND type:"Web Map"'
        wm_items = gis.content.search(query=query, max_items=1)
        
        if not wm_items:
            print(f"    -> [Warning] Web Map '{webmap_title}' not found. Cannot add layer.")
            return

        wm_item = wm_items[0]
        
        # Download the Web Map's underlying JSON data
        wm_data = wm_item.get_data()
        
        if wm_data is None:
            wm_data = {"operationalLayers": []}
        elif "operationalLayers" not in wm_data:
            wm_data["operationalLayers"] = []

        # Check if a layer with the same title already exists and remove it from the list
        original_count = len(wm_data['operationalLayers'])
        wm_data['operationalLayers'] = [
            layer for layer in wm_data['operationalLayers'] 
            if layer.get('title') != feature_layer_item.title
        ]
        
        if len(wm_data['operationalLayers']) < original_count:
            print(f"    -> Removing older version of '{feature_layer_item.title}' from Web Map...")

        # Construct the new layer definition
        new_layer = {
            "id": f"{feature_layer_item.id}",
            "layerType": "ArcGISFeatureLayer",
            "url": feature_layer_item.layers[0].url, # Get the exact URL to the first layer
            "title": feature_layer_item.title,
            "itemId": feature_layer_item.id,
            "visibility": True,
            "opacity": 1
        }

        # Add the newly published layer to the operational layers
        wm_data['operationalLayers'].append(new_layer)

        # Save the updates back to ArcGIS Online
        wm_item.update(item_properties={"text": json.dumps(wm_data)})
        print("    -> Successfully added layer to Web Map!")
        
    except Exception as e:
        if "403" in str(e):
            print(f"    -> [Warning] Cannot edit Web Map '{webmap_title}'. It is likely owned by someone else.")
            print(f"                 To fix this, the owner must put the map in a 'Shared Update' group.")
        else:
            print(f"    -> [Error] Failed to add layer to Web Map: {e}")

def upload_and_publish(gis, zip_path, title, tags, folder_name):
    """Uploads a zipped shapefile to AGO and publishes it."""
    # Step 1: Clean up any existing items first to avoid duplicates
    delete_existing_ago_items(gis, title)
    
    print(f"  Uploading and publishing '{title}' to AGO...")
    properties = {
        'title': title,
        'description': f'Vectorized Hydro Health Data - {title}',
        'tags': tags,
        'type': 'Shapefile'
    }
    
    shp_item = None
    published_item = None
    
    try:
        # Fetch the live Folder object to use Folder.add() and avoid deprecation warnings
        target_folder = None
        
        # Try live API first
        if hasattr(gis.content, 'folders'):
            try:
                for f in gis.content.folders.get():
                    if f.title == folder_name:
                        target_folder = f
                        break
            except Exception:
                pass
                
        # Fallback to cached API if new API fails
        if not target_folder:
            for f in gis.users.me.folders:
                if hasattr(f, 'title') and f.title == folder_name:
                    target_folder = f
                    break
                elif isinstance(f, dict) and f.get('title') == folder_name:
                    target_folder = f
                    break
        
        # Upload
        if target_folder and hasattr(target_folder, 'add'):
            shp_item = target_folder.add(item_properties=properties, data=zip_path)
        else:
            shp_item = gis.content.add(item_properties=properties, data=zip_path, folder=folder_name)

        # Publish it as a Feature Layer
        # We generate a unique system name to avoid AGO cache collisions when overwriting
        safe_name = re.sub(r'[^a-zA-Z0-9]', '_', title)
        unique_service_name = f"{safe_name}_{int(time.time())}"
        
        published_item = shp_item.publish(publish_parameters={'name': unique_service_name})
        print(f"  -> Successfully published: {published_item.title} (ID: {published_item.id})")
        
        # --- Configure Symbology ---
        try:
            print("    -> Configuring default symbology for True_Value...")
            fl = published_item.layers[0]
            
            # 1. Ask AGO for the min and max values of the True_Value field
            stats = fl.query(out_statistics=[
                {"statisticType": "min", "onStatisticField": "True_Value", "outStatisticFieldName": "min_val"},
                {"statisticType": "max", "onStatisticField": "True_Value", "outStatisticFieldName": "max_val"}
            ])
            min_val = stats.features[0].attributes.get('min_val', 0)
            max_val = stats.features[0].attributes.get('max_val', 1)

            # 2. Define a continuous color ramp (Light Yellow to Dark Blue)
            renderer_update = {
                "drawingInfo": {
                    "renderer": {
                        "type": "classBreaks",
                        "field": "True_Value",
                        "minValue": min_val,
                        "classBreakInfos": [{
                            "classMaxValue": max_val,
                            "symbol": {
                                "type": "esriSFS",
                                "style": "esriSFSSolid",
                                "color": [13, 136, 198, 255],
                                "outline": {"type": "esriSLS", "style": "esriSLSSolid", "color": [153, 153, 153, 64], "width": 0.5}
                            }
                        }],
                        "visualVariables": [{
                            "type": "colorInfo",
                            "field": "True_Value",
                            "stops": [
                                {"value": min_val, "color": [255, 255, 178, 255]}, # Light Yellow
                                {"value": max_val, "color": [37, 52, 148, 255]}    # Dark Blue
                            ]
                        }]
                    }
                }
            }
            # 3. Apply the update directly to the layer's settings
            fl.manager.update_definition(renderer_update)
            print("    -> Default symbology successfully set!")
            
        except Exception as e:
            print(f"    -> [Warning] Could not set default symbology: {e}")
            
        # --- Add the styled layer to the Web Map ---
        add_layer_to_webmap(gis, TARGET_WEBMAP_TITLE, published_item)

    except Exception as e:
        print(f"  -> [Error] Failed to upload/publish '{title}'.")
        print(f"     Details: {e}")
        
    finally:
        # Guarantee the underlying zipped shapefile item is always deleted, even on crash
        if shp_item:
            try:
                shp_item.delete()
            except Exception:
                pass
        
        # Explicitly delete python references before calling garbage collection to free file locks
        try:
            del shp_item
        except NameError:
            pass
            
        try:
            del published_item
        except NameError:
            pass
            
        gc.collect()

def convert_raster_to_polygon_safe(tif_path, out_shp):
    """Safely converts floating-point rasters to polygons by scaling."""
    # Multiply by 10,000 to preserve 4 decimal places, convert to Int
    in_ras = arcpy.sa.Raster(str(tif_path))
    scaled_ras = arcpy.sa.Int(in_ras * 10000)
    
    # Now it is an integer, so it will convert successfully
    arcpy.RasterToPolygon_conversion(
        in_raster=scaled_ras,
        out_polygon_features=out_shp,
        simplify="SIMPLIFY",
        raster_field="Value"
    )
    
    # RasterToPolygon stores the raster value in a field called 'gridcode'
    # We create a new float field and divide by 10,000 to get the exact decimals back
    arcpy.management.AddField(out_shp, "True_Value", "DOUBLE")
    arcpy.management.CalculateField(out_shp, "True_Value", "!gridcode! / 10000.0", "PYTHON3")
    
    # Clean up the unnecessary fields
    try:
        arcpy.management.DeleteField(out_shp, ["Id", "gridcode"])
    except:
        pass

def process_offshore(gis):
    """Processes the 100m offshore TIFFs individually."""
    print("\n" + "="*40)
    print("PROCESSING OFFSHORE (100m) FOLDER")
    print("="*40)
    
    if not OFFSHORE_DIR.exists():
        print(f"Directory not found: {OFFSHORE_DIR}")
        return

    tifs = list(OFFSHORE_DIR.glob("*.tif")) + list(OFFSHORE_DIR.glob("*.tiff"))
    
    for tif_path in tifs:
        base_name = tif_path.stem
        print(f"\n--- Processing {base_name} ---")
        
        # Extract the variable shorthand (e.g., 'HG') from 'HG_Eco_region_all_tiles_100m'
        var_key = base_name.replace("_Eco_region_all_tiles_100m", "")
        
        # Look up the friendly name, default to the var_key if not found in dictionary
        friendly_name = LAYER_NAME_KEY.get(var_key, var_key)
        final_title = f"{friendly_name} Offshore 100m"
        
        temp_shp = str(OFFSHORE_DIR / f"{base_name}_vector.shp")
        
        try:
            print(f"  Converting {base_name} to vector (handling decimals)...")
            convert_raster_to_polygon_safe(tif_path, temp_shp)
            
            # Zip, upload, and clean up
            zip_path = zip_shapefile(temp_shp)
            upload_and_publish(gis, zip_path, title=final_title, tags=['HydroHealth', 'Offshore', '100m', var_key], folder_name=AGO_FOLDER_NAME)
            cleanup_shapefile(temp_shp)
            
        except Exception as e:
            print(f"  [Error] Processing failed for {base_name}: {e}")

def process_inshore(gis):
    """Groups the 20m inshore TIFFs by variable, merges them, and uploads."""
    print("\n" + "="*40)
    print("PROCESSING INSHORE (20m) FOLDER")
    print("="*40)
    
    if not INSHORE_DIR.exists():
        print(f"Directory not found: {INSHORE_DIR}")
        return

    tifs = list(INSHORE_DIR.glob("*.tif")) + list(INSHORE_DIR.glob("*.tiff"))
    
    # 1. Group the TIFFs by their variable prefix (e.g., 'HG' from 'HG_ER_1.tif')
    # This regex looks for anything before "_ER_" followed by numbers.
    pattern = re.compile(r"(.+)_ER_\d+", re.IGNORECASE)
    groups = {}
    
    for tif in tifs:
        match = pattern.match(tif.stem)
        if match:
            var_name = match.group(1) # This extracts 'HG', 'PSS', 'decay_coefficient', etc.
            if var_name not in groups:
                groups[var_name] = []
            groups[var_name].append(tif)
        else:
            print(f"  [Warning] File {tif.name} does not match the expected naming convention and will be skipped.")

    # 2. Process each group
    for var_name, tif_list in groups.items():
        print(f"\n--- Processing Variable Group: {var_name} ({len(tif_list)} files) ---")
        
        # Look up the friendly name for the final AGO title
        friendly_name = LAYER_NAME_KEY.get(var_name, var_name)
        final_title = f"{friendly_name} Inshore 20m"
        
        temp_shps_to_merge = []
        
        try:
            # Step A: Convert each individual ER TIFF to a temporary shapefile
            for tif_path in tif_list:
                print(f"  Converting {tif_path.name} to vector (handling decimals)...")
                temp_shp = str(INSHORE_DIR / f"temp_{tif_path.stem}.shp")
                convert_raster_to_polygon_safe(tif_path, temp_shp)
                temp_shps_to_merge.append(temp_shp)
            
            # Step B: Merge all the temporary shapefiles into one master shapefile
            merged_shp = str(INSHORE_DIR / f"{var_name}_Inshore_Merged.shp")
            print(f"  Merging {len(temp_shps_to_merge)} vector layers into {var_name}_Inshore_Merged...")
            arcpy.management.Merge(inputs=temp_shps_to_merge, output=merged_shp)
            
            # Step C: Zip and Upload the merged shapefile
            zip_path = zip_shapefile(merged_shp)
            upload_and_publish(
                gis, 
                zip_path, 
                title=final_title, 
                tags=['HydroHealth', 'Inshore', '20m', var_name], 
                folder_name=AGO_FOLDER_NAME
            )
            
        except Exception as e:
            print(f"  [Error] Failed processing group {var_name}: {e}")
            
        finally:
            # Step D: Clean up ALL temporary files (both the individual ER shps and the merged one)
            print(f"  Cleaning up temporary files for {var_name}...")
            for shp in temp_shps_to_merge:
                cleanup_shapefile(shp)
            if 'merged_shp' in locals():
                cleanup_shapefile(merged_shp)

if __name__ == "__main__":
    print("Connecting via ArcGIS Pro active session...")
    try:
        gis = GIS("pro")
        print(f"Connected as: {gis.users.me.username} to {gis.properties.name}")
    except Exception as e:
        print("Failed to connect via 'pro'. Ensure ArcGIS Pro is open and signed in.")
        print(f"Error details: {e}")
        exit()

    # --- Check if the user has publishing privileges BEFORE running anything (Warning only) ---
    user_privs = [priv.lower() for priv in gis.users.me.privileges]
    has_publish_rights = any('publish' in priv for priv in user_privs)
    
    if gis.users.me.role not in ['org_publisher', 'org_admin'] and not has_publish_rights:
        print("\n" + "!"*60)
        print("PERMISSION WARNING:")
        print("It looks like your account might not have Organization-level 'Publisher' privileges.")
        print("Being a 'Contributor' in a Group is different from being an Org Publisher.")
        print("The script will attempt to publish anyway, but if it fails with Error 400,")
        print("you will still need your ArcGIS Administrator to upgrade your account role.")
        print("!"*60 + "\n")

    # --- Check for and create the output folder if it doesn't exist ---
    print(f"Checking for AGO folder '{AGO_FOLDER_NAME}'...")
    try:
        folder_exists = False
        
        # Try new API first (avoids caching bugs)
        if hasattr(gis.content, 'folders'):
            user_folders = gis.content.folders.get()
            for f in user_folders:
                if f.title == AGO_FOLDER_NAME:
                    folder_exists = True
                    break
            
            if not folder_exists:
                print(f"Creating missing AGO folder '{AGO_FOLDER_NAME}'...")
                gis.content.folders.create(AGO_FOLDER_NAME)
            else:
                print(f"Found AGO folder '{AGO_FOLDER_NAME}'.")
                
        # Fallback to old API
        else:
            for f in gis.users.me.folders:
                if isinstance(f, dict) and f.get('title') == AGO_FOLDER_NAME:
                    folder_exists = True
                    break
            
            if not folder_exists:
                print(f"Creating missing AGO folder '{AGO_FOLDER_NAME}'...")
                gis.content.create_folder(AGO_FOLDER_NAME)
            else:
                print(f"Found AGO folder '{AGO_FOLDER_NAME}'.")
                
    except Exception as e:
        # Catch the "not available" error just in case of race conditions
        if "not available" in str(e).lower():
            print(f"Found AGO folder '{AGO_FOLDER_NAME}' (already exists).")
        else:
            print(f"Warning: Could not verify or create folder '{AGO_FOLDER_NAME}': {e}")

    # Enable Spatial Analyst extension to handle the decimal math
    arcpy.CheckOutExtension("Spatial")

    # Set up arcpy workspace for temporary processing
    arcpy.env.overwriteOutput = True
    
    # Run the processors
    process_offshore(gis)
    process_inshore(gis)
    
    print("\nAll tasks complete.")