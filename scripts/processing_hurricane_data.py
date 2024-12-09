import requests
import pathlib
import geopandas as gpd
import pandas as pd
import numpy as np
import os
from shapely.geometry import LineString, Polygon, GeometryCollection
from shapely.ops import unary_union  # Import unary_union function
from ftplib import FTP
import xarray as xr
import matplotlib.pyplot as plt
import rasterio
from rasterio.features import rasterize

INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'
folder_path = INPUTS / 'hurricane_data'
hurricane_data_path = str(folder_path / 'hurricane_data.gpkg')

#TODO combine these
atlantic_point_layer_name = 'atlantic_hurricane_points'
pacific_point_layer_name = 'pacific_hurricane_points'
atlantic_line_layer_name = 'atlantic_hurricane_lines'
pacific_line_layer_name = 'pacific_hurricane_lines'

def download_tsm_data():
    server = 'ftp.hermes.acri.fr'
    username = 'ftp_hermes'
    password = 'hermes%'
    directory = '824777975'

    download_folder = INPUTS / 'hurricane_data' / 'tsm_data'

    ftp = FTP(server)
    ftp.login(user=username, passwd=password)
    ftp.cwd(directory)

    files = ftp.nlst()
    print(files)

    for file_name in files:
        local_file_path = os.path.join(download_folder, file_name)
        print(f'Downloading {file_name} to {local_file_path}...')
        with open(local_file_path, 'wb') as file:
            ftp.retrbinary(f'RETR {file_name}', file.write)

    ftp.quit()    

def read_tsm_data():
    file_path = INPUTS / 'hurricane_data' / 'tsm_data' / 'L3m_20020429__GLOB_4_AV-MER_TSM_DAY_00.nc'
    ds = xr.open_dataset(file_path)

    print(ds)
    print(f'Variables:', list(ds.data_vars))

    TSM_mean = ds['TSM_mean']

    plt.figure(figsize=(10, 6))
    TSM_mean.plot(cmap='viridis', robust=True)
    plt.title('TSM mean')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

    ds.close


def download_hurricane_data():
    urls = [('atlantic_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt'), 
            ('pacific_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-042624.txt')]
    
    for filename, url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            filepath = os.path.join(folder_path, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
        else:
            print('Error downloading HURDAT2 data.')  

def read_text_data(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    data = []
    current_separator = None

    for line in lines:
        line = line.strip() 
        if line.startswith('AL'):
            current_separator = line
        elif line:
            row = current_separator.split(',') + line.split(',')
            data.append(row)
    return data

def convert_text_to_gpkg():
    column_names = ['area_date',
                'name',
                'num_points',
                'cyclone_num',
                'date',
                'time_utc', 
                'identifier', 
                'type', 
                'latitude', 
                'longitude', 
                'max_wind', 
                'min_pressure',
                'r_ne34', 'r_se34', 'r_sw34', 'r_nw34',
                'r_ne50', 'r_se50', 'r_sw50', 'r_nw50',
                'r_ne64', 'r_se64', 'r_sw64', 'r_nw64',
                'r_max_wind']
                   
    data = read_text_data(folder_path / 'atlantic_hurricane_data.txt')
    df = pd.DataFrame(data)    
    if len(column_names) == df.shape[1]:
        df.columns = column_names
    else:
        raise ValueError('The number of column names does not match.')   

    df = df.applymap(lambda x: x.strip())
    df['cyclone_num'] = df['area_date'].apply(lambda x: x[2:4])
    df['year'] = df['date'].apply(lambda x: x[0:4]).astype(int)
    df['latitude'] = df['latitude'].apply(convert_coordinates)
    df['longitude'] = df['longitude'].apply(convert_coordinates)

    gdf = gpd.GeoDataFrame(df, 
                            geometry=gpd.points_from_xy(df['longitude'], df['latitude']))  
    gdf.set_crs(crs="EPSG:4326", inplace=True)
    gdf = gdf.to_crs("EPSG:4269")
    gdf.to_file(hurricane_data_path, layer=atlantic_point_layer_name, driver='GPKG', overwrite=True)

    return gdf

def convert_coordinates(coord):
    coord = coord.replace('-', '') # TODO double check this isn't causing problems
    if 'N' in coord or 'E' in coord:
        return float(coord[:-1])
    elif 'S' in coord or 'W' in coord:
        return -float(coord[0:-1])
    else:
        raise ValueError ('Invalid coordinate format.')

def create_line_layer():
    gdf = convert_text_to_gpkg()

    lines = (gdf.groupby('area_date').apply(
        lambda points: {
            'name': points['name'].iloc[0],
            'year': points['year'].iloc[0],
            'geometry': LineString(points.geometry.tolist()) if len(points) > 1 else points.geometry.iloc[0]}))
        
    line_gdf = gpd.GeoDataFrame(list(lines), crs=gdf.crs).reset_index()
    line_gdf.to_file(hurricane_data_path, layer=atlantic_line_layer_name, driver='GPKG', overwrite=True)

def create_overlapping_buffers():
    gdf = convert_text_to_gpkg()
    gdf = gdf[gdf['year'] >= 2004]
    gdf = gdf.to_crs('EPSG:32617')  # Change to a UTM zone
    
    quadrants = ['ne', 'nw', 'se', 'sw']
    buffer_data = []

    for _, row in gdf.iterrows():
        for quadrant in quadrants:
            for i in [34, 50, 64]:
                column_name = f'r_{quadrant}{i}'
                buffer_value = int(row[column_name])
                
                if buffer_value > 0:
                    label = f'{quadrant}_{i}'
                    wind_speed = i
                    center = row['geometry']
                    
                    quarter_circle = create_quarter_circle(center, buffer_value * 1852, quadrant)
                    buffer_data.append({
                        'geometry': quarter_circle,
                        'label': label,
                        'id': f"{wind_speed}_{quadrant}_{row['name']}_{row['year']}",
                        'name': row['name'],
                        'year': row['year'],
                        'wind_speed': wind_speed,
                        'longitude': row['longitude'],
                        'latitude': row['latitude'],
                        'buffer_radius_nm': buffer_value
                    })
    
    buffer_data = sorted(buffer_data, key=lambda x: x['id'])

    buffer_gdf = gpd.GeoDataFrame(buffer_data, geometry='geometry', crs=gdf.crs)
    buffer_gdf.to_file(hurricane_data_path, layer='atlantic_point_buffer', driver='GPKG', overwrite=True)
    
    merged_buffer_data = []
    for i in range(len(buffer_data) - 1):
        current_polygon = buffer_data[i]
        if buffer_data[i]['id'] == buffer_data[i + 1]['id']:
            next_polygon = buffer_data[i + 1]
        else:
            continue    

        mega_polygon = create_mega_polygon(current_polygon, next_polygon)

        merged_buffer_data.append({
            'geometry': mega_polygon,
            'id': current_polygon['id'],
            'name': current_polygon['name'],
            'year': current_polygon['year'],
            'wind_speed': current_polygon['wind_speed'],
            'dissolve_id': f"{current_polygon['wind_speed']}_{current_polygon['name']}_{current_polygon['year']}"
        })

    buffer_gdf = gpd.GeoDataFrame(merged_buffer_data, geometry='geometry', crs=gdf.crs)
    # buffer_gdf = buffer_gdf.dissolve(by='id')
    buffer_gdf = buffer_gdf.dissolve(by='dissolve_id')
    buffer_gdf.to_file(hurricane_data_path, layer='atlantic_polygon_buffer', driver='GPKG', overwrite=True)

def get_corner_points(quarter_circle, quadrant):
    coords = list(quarter_circle.exterior.coords)
    # print(coords)
    # TODO this will need to be refined some, maybe map out points and see if there is a pattern
    if quadrant == 'ne':
        return [coords[0], coords[1], coords[-1]]
    elif quadrant == 'nw':
        return [coords[0], coords[2], coords[-2]]
    elif quadrant == 'se':
        return [coords[0], coords[-2], coords[2]]
    elif quadrant == 'sw':
        return [coords[0], coords[-1], coords[1]]

def create_quarter_circle(center, radius, quadrant):
    angles = {
        'ne': np.linspace(0, np.pi / 2, 10),
        'nw': np.linspace(np.pi / 2, np.pi, 10),
        'se': np.linspace(-np.pi / 2, 0, 10),
        'sw': np.linspace(-np.pi, -np.pi / 2, 10)
    }

    angle_array = angles.get(quadrant, np.linspace(0, np.pi / 2, 10))
    coords = [(center.x + radius * np.cos(angle), center.y + radius * np.sin(angle)) for angle in angle_array]
    coords.append((center.x, center.y))

    quarter_circle = Polygon(coords)
    return quarter_circle

def create_mega_polygon(current_polygon, next_polygon):
    current_corners = get_corner_points(current_polygon['geometry'], current_polygon['label'].split('_')[0])
    next_corners = get_corner_points(next_polygon['geometry'], next_polygon['label'].split('_')[0])

    connecting_lines = []
    for current_point, next_point in zip(current_corners, next_corners):
        line = LineString([current_point, next_point])
        connecting_lines.append(line)

    all_geometries = [current_polygon['geometry']] + connecting_lines + [next_polygon['geometry']]
    mega_polygon = all_geometries[0]

    for geom in all_geometries[1:]:
        mega_polygon = mega_polygon.union(geom)
    if isinstance(mega_polygon, GeometryCollection):
        mega_polygon = unary_union(mega_polygon.geoms)

    mega_polygon = mega_polygon.buffer(0)

    if mega_polygon.geom_type == 'MultiPolygon':
        mega_polygon = mega_polygon.convex_hull

    return mega_polygon

def clip_polygons():    
    gdf_to_clip = gpd.read_file(hurricane_data_path, layer='atlantic_polygon_buffer') 
    clip_boundary = gpd.read_file(r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\coastal_boundary_dataset\50m_isobath_polygon\50m_isobath_polygon.shp')

    if gdf_to_clip.crs != clip_boundary.crs:
        clip_boundary = clip_boundary.to_crs(gdf_to_clip.crs)

    clipped_gdf = gpd.clip(gdf_to_clip, clip_boundary)

    layer_name = 'clipped_polygons'
    clipped_gdf.to_file(hurricane_data_path, layer=layer_name, driver='GPKG')

def polygons_to_rasters():
    gpkg = ogr.Open(gpkg_path, 1)
    layer = gpkg.GetLayerByName('sediment_polys_clipped')

    tiff = gdal.Open(tile_path)
    geotransform = tiff.GetGeoTransform()
    proj = tiff.GetProjection()
    cols = tiff.RasterXSize
    rows = tiff.RasterYSize

    driver = gdal.GetDriverByName('GTIFF')
    file_path = INPUTS / "sediment_data" / f"{tile_number}_{field_name}_raster.tif"
    out_raster = driver.Create(file_path, cols, rows, 1, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    out_raster.SetGeoTransform(geotransform)
    out_raster.SetProjection(proj)

    gdal.RasterizeLayer(out_raster, [1], layer, options=['ALL_TOUCHED=FALSE', 'ATTRIBUTE={field_name}'])

    band = out_raster.GetRasterBand(1)
    band.SetNoDataValue(0) 
    out_raster = None
    gpkg = None
    tiff = None    

    # gdf = gpd.read_file(hurricane_data_path, layer='clipped_polygons') 
    # for (year, name), group in gdf.groupby(['year', 'name']):
    #     dissolved_gdf = group.dissolve(by='wind_speed', aggfunc='first')  
    #     dissolved_gdf = dissolved_gdf.reset_index()
    #     print(dissolved_gdf.head(5))

    #     raster_resolution = 100  # in meters
    #     bounds = dissolved_gdf.total_bounds
    #     minx, miny, maxx, maxy = bounds
    #     width = int((maxx - minx) / raster_resolution)
    #     height = int((maxy - miny) / raster_resolution)
    #     transform = rasterio.transform.from_origin(minx, maxy, raster_resolution, raster_resolution)

    #     shapes = ((geom, row['wind_speed']) for geom, row in dissolved_gdf.iterrows())
    #     print(shapes)

    #     raster = np.zeros((height, width), dtype='float32')
    #     raster = rasterize(
    #         shapes,
    #         out_shape=(height, width),
    #         transform=transform,
    #         fill='NaN',  
    #         dtype='float32',
    #         all_touched=True,
    #     )

    #     output_raster_path = str(folder_path / 'hurricane_rasters' / f'wind_speed_raster_{year}_{name}.tif')
    #     with rasterio.open(
    #         output_raster_path,
    #         "w",
    #         driver="GTiff",
    #         height=height,
    #         width=width,
    #         count=1,
    #         dtype='float32',
    #         crs=gdf.crs.to_string(),
    #         transform=transform,
    #     ) as dst:
    #         dst.write(raster, 1)

# download_hurricane_data()    
# create_line_layer()
# download_tsm_data()
# read_tsm_data()
create_overlapping_buffers()
clip_polygons()
# polygons_to_rasters()
