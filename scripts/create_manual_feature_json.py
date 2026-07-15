import os
import re
import json
import pathlib
import requests
import string
import geopandas as gpd

# The API URL used by the Engine base class
DIGITALCOAST_API_URL = "https://coast.noaa.gov/dataviewer/api/v1/search/missions"

# 1. Rename manual downloads to match website
# 2. try to manually build JSON file to get InPort number
# 3. Update main code to rename providers to match website



def build_manual_feature_json(folder_path, full_match, partial_matches) -> None:
    """Build the manual feature JSON from found features"""

    if full_match:
        attrs = full_match['attributes']
        
        if 'ExternalProviderLink' in attrs:
            try:
                raw_links = attrs['ExternalProviderLink']
                if isinstance(raw_links, str):
                    attrs['ExternalProviderLink'] = json.loads(raw_links).get('links', [])
                elif isinstance(raw_links, dict):
                    attrs['ExternalProviderLink'] = raw_links.get('links', [])
            except Exception as parse_err:
                print(f"    Warning: could not format ExternalProviderLink: {parse_err}")

        # has_provider = attrs.get('provider_results_name')
        # provider_results_name = has_provider.lower() if isinstance(has_provider, str) else ''
        inport_number = attrs['Metalink'].split('/')[-1] if attrs.get('Metalink') else 'No_InPort'
        # provider_folder_name = f"{attrs.get('Year', '')}_{attrs.get('DataType', '')}_{provider_results_name.replace(' ', '_')}_{inport_number}"
        # TODO need to remove special characters and update provider folder
        output_json_path = folder_path / 'manual_feature.json'
        with open(output_json_path, 'w') as writer:
            json.dump(attrs, writer, indent=4)
            
        print(f"    Successfully generated: {output_json_path.name} ({attrs.get('DataType')} - {attrs.get('Year')})")
        # illegal_chars = string.punctuation
        # illegal_chars = illegal_chars.replace('_', '')
        # illegal_chars_translation = str.maketrans("", "", illegal_chars)
        # folder_name = provider_folder_name.translate(illegal_chars_translation)
        
        print(f'    InPort Number: {inport_number}')
    else:
        print(f"    Partial match only")
        for feature in partial_matches:
            attrs = feature['attributes']
            # has_provider = attrs.get('provider_results_name')
            # provider_results_name = has_provider.lower() if isinstance(has_provider, str) else ''
            inport_number = attrs['Metalink'].split('/')[-1] if attrs.get('Metalink') else 'No_InPort'
            # provider_folder_name = f"{attrs.get('Year', '')}_{attrs.get('DataType', '')}_{provider_results_name.replace(' ', '_')}_{inport_number}"
            print(f"    {attrs['Year']} {attrs['DataType']} {attrs['provider_results_name']} {attrs['Name']} {inport_number}")
            # print(f'    New name: {provider_folder_name}')


def clean_folder_name(provider_folder_name) -> str:
    """Remove illegal chars from string"""



def get_wkt_bbox_from_shapefile(shp_path: pathlib.Path) -> str:
    """
    Reads a shapefile, forces/projects it to EPSG:4269, 
    and returns its bounding box envelope as a WKT Polygon.
    """
    try:
        gdf = gpd.read_file(shp_path)
        if gdf.empty:
            return None
        
        # Ensure it's projected to EPSG:4269 (NAD83) for the API
        if gdf.crs is None:
            # Fallback to EPSG:4326 if unassigned, then project
            gdf.set_crs(epsg=4326, inplace=True)
        
        gdf_4269 = gdf.to_crs(epsg=4269)
        
        # Get overall bounding box of all geometries combined
        minx, miny, maxx, maxy = gdf_4269.total_bounds
        
        # Format as WKT Polygon
        wkt_polygon = f"POLYGON(({minx} {miny}, {maxx} {miny}, {maxx} {maxy}, {minx} {maxy}, {minx} {miny}))"
        return wkt_polygon
    except Exception as e:
        print(f"    Error reading shapefile {shp_path.name}: {e}")
        return None


def find_matching_datasets(folder_path: pathlib.Path, shp_path: pathlib.Path, expected_year: str) -> dict|list[dict]:
    """
    Queries the API with the shapefile geometry, finds the correct dataset metadata,
    and writes the feature.json into the folder. Strictly enforces DEM-only selection.
    """
    
    wkt_geom = get_wkt_bbox_from_shapefile(shp_path)
    if not wkt_geom:
        return

    payload = {
        "aoi": f"SRID=4269;{wkt_geom}",
        "published": "true",
        "dataTypes": ["Lidar", "DEM"],
        "dialect": "arcgis",
    }

    try:
        response = requests.post(DIGITALCOAST_API_URL, data=payload, timeout=30)
        if not response.ok:
            print(f"    API Error: {response.status_code} - {response.reason}")
            return
        
        datasets_json = response.json()
        features = datasets_json.get('features', [])
        
        matched_feature = None
        folder_clean_name = folder_path.name.lower()
        partial_features = []        
        provider_folder_name = None
        
        # --- Year Logic Update ---
        # Calculate the previous year as a string
        current_expected_int = int(expected_year)
        previous_year = str(current_expected_int - 1)
        allowed_years = [expected_year, previous_year]
        # -------------------------

        for feature in features:
            attrs = feature.get('attributes', {})
            
            # CRITICAL: Skip non-DEM data types immediately
            data_type = str(attrs.get('DataType', '')).upper()
            # TODO manual downloads seem to be only LIDAR, but 2019_Lidar_NGS_63018 uses the DEM InPort and should be 63017
            # Changed that one to 63017 to use Liidar
            if data_type != 'LIDAR' or 'CUDEM' in attrs['Name']:
                continue

            metalink_filename = attrs['Metalink'].split('/')[-1] if attrs.get('Metalink') else 'No_Metalink'
            # if attrs['Year'] == 0:
            #     print('     ', attrs['Name'])
            # print(f"    [API Diagnostic Sample] {attrs['Year']}_{attrs['provider_results_name'].replace(' ', '_')}_{attrs['DataType']}_{metalink_filename}")
            # print('    [Name Keywords]:', folder_clean_name)
                
            api_year = str(attrs.get('Year', ''))
            has_provider = attrs.get('provider_results_name')
            # Sometimes provider is actually None and needs the double check
            provider_results_name = has_provider.lower() if isinstance(has_provider, str) else ''
            
            # FIXED: Updated regex to strip the year from the BEGINNING of the folder name
            name_keywords = re.sub(r'^\d{4}_', '', folder_clean_name).replace('_', ' ')
            name_keywords_list = [word for word in name_keywords.split() if len(word) > 2]
            
            # UPDATED: Checks if the API year is in our list of allowed years
            year_match = api_year in allowed_years
            name_match = any(word in provider_results_name for word in name_keywords_list) if name_keywords_list else True
            
            if year_match and name_match:
                matched_feature = feature
                print(attrs['DataType'])
                break
            elif year_match:
                # Regarding your TODO comment: Do NOT add a break statement here. 
                # If you break here, the loop stops searching entirely and you might 
                # miss a perfect 'year_match AND name_match' combination further down the list.
                partial_features.append(feature)
            provider_folder_name = f"{attrs.get('Year', '')}_{attrs.get('DataType', '')}_{provider_results_name.replace(' ', '_')}_{metalink_filename}"
        return matched_feature, partial_features
            
    except Exception as e:
        print(f"    Failed to retrieve or write JSON: {e}")
        print(f"    {provider_folder_name}")


def process_manual_downloads(parent_dir_path: str):
    """
    Main loop to process all folders inside the manual downloads directory.
    """
    parent_dir = pathlib.Path(parent_dir_path)
    # valid_folder_regex = re.compile(r'^.+_(\d{4})$')  # end of folder name
    valid_folder_regex = re.compile(r'^(\d{4})_.+$')  # beginning of folder name
    
    print(f"Scanning directory: {parent_dir.resolve()}\n")
    
    for folder in parent_dir.iterdir():
        if not folder.is_dir():
            continue
            
        match = valid_folder_regex.match(folder.name)
        if not match:
            # Safely skips "invalid_datasets" and "laz_issues"
            continue
            
        year = match.group(1)
        print(f"Processing Provider: {folder.name} (Year: {year})")
            
        # Locate the tileindex shapefile inside the folder
        shp_files = list(folder.glob('*index*.shp'))
        if not shp_files:
            # Fallback to any shapefile if "index" isn't in the name
            shp_files = list(folder.glob('*.shp'))
            
        if not shp_files:
            print("    No tileindex shapefile found. Skipping.")
            continue
            
        target_shp = shp_files[0]
        print(f"    Found shapefile: {target_shp.name}")
        
        # Fetch and write metadata
        full, partials = find_matching_datasets(folder, target_shp, year)
        build_manual_feature_json(folder, full, partials)
        print("-" * 50)

if __name__ == "__main__":
    MANUAL_DOWNLOADS_DIR = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\ER_3\model_variables\Prediction\raw\Digital_Coast_Manual_Downloads" 
    process_manual_downloads(MANUAL_DOWNLOADS_DIR)