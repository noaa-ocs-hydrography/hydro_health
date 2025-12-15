import os
from arcgis.gis import GIS

def upload_tif_files_via_pro(folder_path, folder_name):
    print("Connecting via ArcGIS Pro active session...")
    try:
        gis = GIS("pro")
        print(f"Connected as: {gis.users.me.username} to {gis.properties.name}")
    except Exception as e:
        print("Failed to connect via 'pro'. Ensure ArcGIS Pro is open and signed in.")
        print(f"Error details: {e}")
        return

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.tif', '.tiff')):
            file_path = os.path.join(folder_path, file_name)   
            item_properties = {
                "title": os.path.splitext(file_name)[0],
                "type": "Image",
                "tags": "HydroHealth",
                "description": f"{file_name}"
            }
            
            print(f"Uploading {file_name}...")
            try:
                upload_folder = gis.content.folders.get(folder_name)
                uploaded_item = upload_folder.add(item_properties=item_properties, data=file_path)
                print(f"Successfully uploaded: {uploaded_item.title} (ID: {uploaded_item.id})")
            except Exception as e:
                print(f"Failed to upload {file_name}. Error: {e}")


LOCAL_FOLDER = r"input/folder/of/tif/files"
FOLDER_NAME = "ER3_Input_Data"


if __name__ == "__main__":
    upload_tif_files_via_pro(LOCAL_FOLDER, FOLDER_NAME)
    print('done')