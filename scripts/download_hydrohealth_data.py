import pathlib
import os
import json
import requests
import zipfile
import geopandas
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'


def get_reef_5k_json(save: bool=False) -> dict:
    """Download and obtain global reef-5km features"""

    url = get_config_item('REEF', '5KM')
    global_reefs = requests.get(url).json()

    polygons = []
    for feature in global_reefs['features']:
        if feature['geometry']['type'] != 'Point':
            polygons.append(feature)
    reef_data = {
        'polygons': polygons
    }

    if save:
        for key, value in reef_data.items():
            print(f'Saving: {key}')
            template = {
                "type": "FeatureCollection", 
                "features": value
            }
            with open(INPUTS / f'global_reef_{key}.geojson', 'w') as features:
                features.write(json.dumps(template))

    return polygons


def convert_json_to_shp():
    """Convert Reef 5k polygons to shapefile using Geopandas"""

    reef_polygons_json = str(INPUTS / 'global_reef_polygons.geojson')
    print(reef_polygons_json)
    if not os.path.exists(reef_polygons_json):
        get_reef_5k_json(save=True)
    gdf = geopandas.read_file(reef_polygons_json)
    global_reef_polygons = str(INPUTS / 'global_reef_polygons_5k.shp')
    gdf.to_file(global_reef_polygons)
    return global_reef_polygons


def get_reef_1km_data() -> dict:
    """Download global reef-1km features"""

    url = get_config_item('REEF', '1KM')
    zip_file = INPUTS / 'global_reefs_1km.zip'
    zip_folder = INPUTS / 'global_reefs_1km'
    global_reefs_download = requests.get(url, stream=True)
    with open(zip_file, 'wb') as data:
        print('Saving 1km Reef zip file')
        for chunk in global_reefs_download.iter_content(chunk_size=128):
            data.write(chunk)
    with zipfile.ZipFile(zip_file, 'r') as reef_zip:
        print('Unzipping 1km Reef zip file')
        reef_zip.extractall(zip_folder)


if __name__ == '__main__':
    get_reef_5k_json(True)
    convert_json_to_shp()
    get_reef_1km_data()