import requests
import pathlib
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'
download_path = INPUTS / 'usace_dredge_data'
# gpkg_path = str(OUTPUTS / 'dredge_outline.gpkg')

start_year = 1990
end_year = 2024
dredge_output = str(OUTPUTS / 'dredge_outline.gpkg')
dredge_layer_name = 'dredge_survey_extents'
borrow_layer_name = 'borrow_sites'
placement_layer_name = 'placement_sites'
epa_disposal_name = 'epa_disposal_sites'
harbour_layer_name = 'HARBOUR_of_Top_60_Ports' 
channel_layer_name = 'channel_network'

channel_path = str(OUTPUTS / 'channel_outline.shp')

dredge_url = 'https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/eHydro_Survey_Data/FeatureServer/0/query'
borrow_url = 'https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Sediment_Management_Framework/FeatureServer/0/query'
placement_url = 'https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Sediment_Management_Framework/FeatureServer/1/query'
epa_disposal_url = 'https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/ODMDSpolygons_wgs84_20201102_v2/FeatureServer/0/query'
channel_url = 'https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/National_Channel_Framework/FeatureServer/1/query'


def get_record_count(url):
    count_params = {
        "where": '1=1',
        "returnCountOnly": True,
        "f": "json"
    }
    response = requests.get(url, params=count_params)
    return response.json()["count"]

def get_all_dredge_data():
    download_gpkg(borrow_url, dredge_output, borrow_layer_name)
    download_gpkg(placement_url, dredge_output, placement_layer_name)
    download_gpkg(epa_disposal_url, dredge_output, epa_disposal_name) # TODO does it need transformed?
    download_gpkg(dredge_url, dredge_output, dredge_layer_name)
    download_gpkg(channel_url, dredge_output, channel_layer_name)


def download_gpkg(url, gpkg_output, layer_name):
    total_records = get_record_count(url)
    print(f'Total records: {total_records}')
    max_records = 2000

    all_features = []
    offset = 0
    
    while offset < total_records:
        query_params = {
            "where": '1=1',
            "outFields": '*',
            "f": "geojson",
            "resultOffset": offset
        }

        response = requests.get(url, params=query_params)
        features = response.json()["features"]
        all_features.extend(features)
        offset += max_records
        print(offset)

    gdf = gpd.GeoDataFrame.from_features(all_features)
    # print(gdf.columns)
    # gdf.drop(['datenotified', 'projectedarea'],  axis='columns', inplace=True)
    # if layer_name == channel_layer_name:
    #     gdf['projectedarea'] = gdf['projectedarea'].round(2)
    

    gdf.set_crs('EPSG:4269', inplace=True)
    # gdf['projectedarea'] = gdf['projectedarea'].round(2)

    gdf.to_file(gpkg_output, layer=layer_name, driver='GPKG')
    gdf.to_file(channel_path, driver='ESRI Shapefile', encoding='utf-8')
    # print(gdf.columns)


def download_usace_data():
    urls = [('https://ndc.ops.usace.army.mil/dis/placement-locations/{year}.json',
            '{year}_placements.json'),
            ('https://ndc.ops.usace.army.mil/dis/dredging-locations/{year}.json',
            '{year}_dredging.json'),
            ('https://ndc.ops.usace.army.mil/dis/jobs/{year}.json', 
            '{year}_jobs.json')]
    
    for year in range(start_year, end_year + 1):
        print(f'Downloading {year} data.')
        for url, filename in urls:
            url = url.format(year=year)
            filename = filename.format(year=year)

            response = requests.get(url)
            if response.status_code == 200:
                filepath = os.path.join(download_path, filename)
                os.makedirs(download_path, exist_ok=True)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            else:
                print('Error downloading USACE data.')  

def json_to_df(file_path):   
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    return df

def filter_jobs_data(df):
    df = df[~((df['total_dredge_quantity'].isin([0, np.nan])) & (df['total_placement_quantity'].isin([0, np.nan])))]
    df = df.reset_index(drop=True)
    return df

def create_job_locations_gpkg():
    locations_df = pd.DataFrame()
    df_list = []
    print(f'Writing location data for GPKG.')

    for year in range(start_year, end_year + 1):
        print(f' - {year}')
        jobs_df = json_to_df(download_path / f'{year}_jobs.json')
        placement_df = json_to_df(download_path / f'{year}_placements.json')
        dredge_df = json_to_df(download_path / f'{year}_dredging.json')

        jobs_df_filtered = filter_jobs_data(jobs_df)        
        
        for id_value in jobs_df_filtered['id']:
            placement_df_filtered = placement_df[placement_df['job_id'] == id_value]
            dredge_df_filtered = dredge_df[dredge_df['job_id'] == id_value]

            combined = pd.concat([placement_df_filtered, dredge_df_filtered], ignore_index=True)
            combined['year'] = year
            df_list.append(combined)

    locations_df = pd.concat(df_list, ignore_index=True)
    locations_df.drop(['id', 'usace_district_code', 'district_idfk', 'area_idpk',
                       'placement_type_id', 'acreage_created', 'placement_subtype_id'], axis=1, inplace=True)
            
    gdf = gpd.GeoDataFrame(locations_df, 
                            geometry=gpd.points_from_xy(locations_df['longitude'], locations_df['latitude']))  
    gdf.set_crs(crs="EPSG:4326", inplace=True)
    gdf_reprojected = gdf.to_crs("EPSG:4269")

    print(gdf_reprojected.columns)
    
    gdf_reprojected.to_file(dredge_output, layer='point_locations', driver="GPKG")    

def caculate_frequency_data():
    print('Calculating frequency metrics.')
    # polygon_gdf = gpd.read_file(dredge_output, layer=harbour_layer_name)  
    polygon_gdf = gpd.read_file(channel_path)  
    polygon_gdf = polygon_gdf.to_crs("EPSG:4269") 

    point_gdf = gpd.read_file(dredge_output, layer='point_locations')   
    dredging_points = point_gdf[point_gdf['location_type'] == 'dredging']

    joined_gdf = gpd.sjoin(dredging_points, polygon_gdf, how='inner', predicate='within')
    joined_gdf = joined_gdf.drop_duplicates(subset=['index_right', 'job_id'])
    aggregate_years = (joined_gdf.groupby('index_right').agg(years=('year', list)).reset_index())

    result_gdf = polygon_gdf.merge(aggregate_years, left_index=True, right_on='index_right', how='left')
    result_gdf = result_gdf[['geometry', 'years']]
    # print(result_gdf.head(5))

    result_gdf['years'] = result_gdf['years'].apply(lambda x: x if isinstance(x, list) else [])
    result_gdf['count_1994_2003'] = result_gdf['years'].apply(lambda x: count_years(x, 1994, 2003))
    result_gdf['count_2004_2013'] = result_gdf['years'].apply(lambda x: count_years(x, 2004, 2013))
    result_gdf['count_2014_2023'] = result_gdf['years'].apply(lambda x: count_years(x, 2014, 2023))
    result_gdf['count_all_years'] = result_gdf['years'].apply(lambda x: count_years(x, 1994, 2023))
    result_gdf.to_file(dredge_output, layer='frequency_data', driver='GPKG')

def dissolve_by_featurename():
    polygon_gdf = gpd.read_file(channel_path)  
    polygon_gdf = polygon_gdf.dissolve(by='sdsfeatu_1')
    polygon_gdf.to_file(channel_path, driver='ESRI Shapefile', encoding='utf-8')

def count_years(years, start_year, end_year):
    return sum(start_year <= year <= end_year for year in years)  

# def clip_harbour_data():
#     gdf_to_clip = gpd.read_file(dredge_output, layer=harbour_layer_name)
#     gdf = gdf_to_clip.to_crs("EPSG:4269") 
#     gdf = gdf[~(gdf['ChartScale'] == 'APPROACH')]

#     gdf['area'] = gdf.geometry.area
#     buffer_distance = 10000 # 10 km
#     gdf['geometry_buffer'] = gdf.geometry.buffer(buffer_distance)

#     neighbors = gpd.sjoin(gdf, gdf[['geometry', 'area']], how='inner', predicate='intersects')

#     size_threshold = 10
#     large_polygons = neighbors[neighbors['area_left'] > size_threshold * neighbors['area_right']]
#     unique_large_polygons = gdf.loc[gdf.index.isin(large_polygons.index_right)]

#     smaller_polygons = gdf.loc[~gdf.index.isin(unique_large_polygons.index)]
#     clipped_large_polygons = gpd.overlay(unique_large_polygons, smaller_polygons, how='difference')
#     clipped_large_polygons.set_geometry('geometry', inplace=True)
#     clipped_large_polygons.drop('geometry_buffer', axis=1, inplace=True)
#     final_polygons = gpd.GeoDataFrame(pd.concat([clipped_large_polygons, smaller_polygons], ignore_index=True),
#                                       crs=gdf.crs)
    
#     final_polygons.drop('geometry_buffer', axis=1, inplace=True)
#     print(final_polygons.columns)

#     final_polygons.to_file(dredge_output, layer='harbour_extents_clipped', driver='GPKG')  

# def clip_survey_with_harbours():
#     gdf_to_clip = gpd.read_file(dredge_output, layer=dredge_layer_name)
#     gdf_clip = gpd.read_file(dredge_output, layer=harbour_layer_name)
#     gdf_clip = gdf_clip.to_crs("EPSG:4269") 

#     clipped_gdf = gpd.clip(gdf_to_clip, gdf_clip)
#     clipped_gdf.to_file(dredge_output, layer='dredge_extent_clipped', driver='GPKG')

# get_all_dredge_data()
# download_usace_data()
create_job_locations_gpkg()
# dissolve_by_featurename()
# caculate_frequency_data()
# clip_survey_with_harbours()
# clip_harbour_data()