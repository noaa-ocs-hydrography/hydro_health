import os
import pathlib
from arcgis.gis import GIS
from arcgis.raster import utils


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
            file_path = pathlib.Path(folder_path) / file_name   
            item_properties = {
                "title": os.path.splitext(file_name)[0],
                "productType": "Raster Dataset",
                "tags": "HydroHealth",
                "description": f"{file_name}"
            }
            
            print(f"Uploading {file_name}...")
            try:

                properties = {
                    'title': file_path.stem,
                    'description': f'Hydro Health 2025 - Ecoregion 3 - {file_name}',
                    'tags': ['HydroHealth'],
                    'snippet': file_name
                }
                imagery_layer_item = utils.publish_hosted_imagery_layer(
                    input_data=str(file_path),
                    gis=gis,
                    layer_configuration="ONE_IMAGE",
                    tiles_only=False,
                    raster_type_name='Raster Dataset',
                    output_name=file_path.stem,
                    folder=folder_name
                )
                print(f"Successfully uploaded: {imagery_layer_item.title} (ID: {imagery_layer_item.id})")

                hosted_layer_item = gis.content.get(imagery_layer_item.id)
                hosted_layer_item.update(item_properties=properties)
                print(f"Successfully updated item properties")
                
            except Exception as e:
                print(f"Failed to upload {file_name}. Error: {e}")


LOCAL_FOLDER = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\ER3_NOAA_AGO_Files\Pilot_Model_Files\Input_data\problem_files"
FOLDER_NAME = "ER3_Input_Data"


if __name__ == "__main__":
    upload_tif_files_via_pro(LOCAL_FOLDER, FOLDER_NAME)
    print('done')