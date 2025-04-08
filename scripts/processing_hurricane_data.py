import os
import requests
import pathlib
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString, Polygon, GeometryCollection
from shapely.ops import unary_union
import rasterio
from rasterio.transform import from_bounds
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from scipy.ndimage import gaussian_filter


INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'
folder_path = INPUTS / 'hurricane_data'
hurricane_data_path = str(folder_path / 'hurricane_data.gpkg')

atlantic_point_layer_name = 'atlantic_hurricane_points'
pacific_point_layer_name = 'pacific_hurricane_points'
atlantic_line_layer_name = 'atlantic_hurricane_lines'
pacific_line_layer_name = 'pacific_hurricane_lines'

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

    # Estimate radius for 34 to 0 knots using a 1.5x factor
    df['r_ne0'] = pd.to_numeric(df['r_ne34']) * 1.5
    df['r_nw0'] = pd.to_numeric(df['r_nw34']) * 1.5
    df['r_se0'] = pd.to_numeric(df['r_se34']) * 1.5
    df['r_sw0'] = pd.to_numeric(df['r_sw34']) * 1.5

    gdf = gpd.GeoDataFrame(df, 
                            geometry=gpd.points_from_xy(df['longitude'], df['latitude']))  
    gdf.set_crs(crs="EPSG:4326", inplace=True)

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
    gdf = gdf.to_crs('EPSG:32617')  # WGS 84 UTM 17N

    
    quadrants = ['ne', 'nw', 'se', 'sw']
    buffer_data = []

    for _, row in gdf.iterrows():
        for quadrant in quadrants:
            for i in [0, 34, 50, 64]:
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
    buffer_gdf = buffer_gdf.dissolve(by='dissolve_id')
    buffer_gdf.to_file(hurricane_data_path, layer='atlantic_polygon_buffer', driver='GPKG', overwrite=True)

def get_corner_points(quarter_circle, quadrant):
    coords = list(quarter_circle.exterior.coords)
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
    clip_boundary = gpd.read_file(r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\masks\50m_isobath_polygon\50m_isobath_polygon.shp')

    if gdf_to_clip.crs is None or clip_boundary.crs is None:
        raise ValueError("One or both GeoDataFrames are missing a CRS.")
    
    if gdf_to_clip.crs != clip_boundary.crs:
        clip_boundary = clip_boundary.to_crs(gdf_to_clip.crs)

    gdf_to_clip = gdf_to_clip[~gdf_to_clip.geometry.is_empty & gdf_to_clip.geometry.is_valid]

    clipped_rows = []
    for (name, year), group in gdf_to_clip.groupby(["name", "year"]):
        group = group.sort_values(by="wind_speed", ascending=False).reset_index(drop=True)
        for _, row in group.iterrows():
            geom = row.geometry
            for clipped_row in clipped_rows:
                if clipped_row["name"] == name and clipped_row["year"] == year:
                    geom = geom.difference(clipped_row.geometry)
            if not geom.is_empty:
                new_row = row.copy()
                new_row.geometry = geom
                clipped_rows.append(new_row)

    clipped_gdf = gpd.GeoDataFrame(clipped_rows, columns=gdf_to_clip.columns, crs=gdf_to_clip.crs)
    clipped_gdf = clipped_gdf[clipped_gdf.geometry.is_valid]

    clipped_gdf['dissolve_id'] = clipped_gdf['dissolve_id'].astype(str)
    clipped_gdf['id'] = clipped_gdf['id'].astype(str)
    clipped_gdf['name'] = clipped_gdf['name'].astype(str)
    clipped_gdf['year'] = pd.to_numeric(clipped_gdf['year'], errors='coerce').astype('Int32')
    clipped_gdf['wind_speed'] = pd.to_numeric(clipped_gdf['wind_speed'], errors='coerce').astype('Int32')

    layer_name = 'trimmed_polygons'
    clipped_gdf.to_file(hurricane_data_path, layer=layer_name, driver='GPKG', overwrite=True)

def polygons_to_raster(resolution=500):

    gdf = gpd.read_file(hurricane_data_path, layer="trimmed_polygons")

    grouped = gdf.groupby(['name', 'year'])

    for (name, year), group in grouped:
        output_folder = rf'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\hurricane_rasters\{year}'
        os.makedirs(output_folder, exist_ok=True)

        minx, miny, maxx, maxy = group.total_bounds
        width = int(np.ceil((maxx - minx) / resolution))
        height = int(np.ceil((maxy - miny) / resolution))
        transform = from_bounds(minx, miny, maxx, maxy, width, height)

        raster_data = np.full((height, width), np.NaN, dtype=np.float32)

        for _, row in group.iterrows():
            shapes = [(row.geometry, row['wind_speed'])]
            rasterio.features.rasterize(
                shapes,
                out=raster_data,
                transform=transform,
                fill=np.NaN, 
            )

        raster_file = os.path.join(output_folder, f"{name}_{year}.tif")

        raster_data = gaussian_filter(raster_data, sigma=3)

        with rasterio.open(
            raster_file,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=raster_data.dtype,
            crs=gdf.crs,
            nodata=np.NaN,
            transform=transform,
        ) as dst:
            dst.write(raster_data, 1)

        print(f"Raster for {name} - {year} saved to {raster_file}.")

def generate_cumulative_rasters(output_folder, value):
    input_raster_folder = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\hurricane_rasters'
    mask_raster_path = r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\masks\prediction.mask.WGS84_8m.tif'
    
    with rasterio.open(mask_raster_path) as mask_src:
        mask_data = mask_src.read(1) 
        mask_transform = mask_src.transform
        mask_crs = mask_src.crs
        mask_height, mask_width = mask_src.height, mask_src.width
        mask_valid = mask_data == 1
    
    year_folders = [f for f in os.listdir(input_raster_folder) if os.path.isdir(os.path.join(input_raster_folder, f))]

    for year_folder in year_folders:
        year_folder_path = os.path.join(input_raster_folder, year_folder)
        raster_files = [f for f in os.listdir(year_folder_path) if f.endswith('.tif')]

        cumulative_count = np.zeros((mask_height, mask_width), dtype=np.float32)
        cumulative_windspeed = np.zeros((mask_height, mask_width), dtype=np.float32)

        for raster_file in raster_files:
            raster_path = os.path.join(year_folder_path, raster_file)
            
            with rasterio.open(raster_path) as src:
                raster_data = src.read(1)  
                transform = src.transform
                crs = src.crs

                raster_data_resampled = np.full((mask_height, mask_width), np.NaN, dtype=np.float32)
                if crs != mask_src.crs or transform != mask_src.transform:
                    reproject(
                        raster_data, raster_data_resampled,
                        src_transform=transform,
                        src_crs=crs,
                        dst_transform=mask_src.transform,
                        dst_crs=mask_src.crs,
                        resampling=Resampling.bilinear,
                        dst_nodata=np.nan 
                    )

                raster_data = raster_data_resampled
                
                if value == "cumulative_count":
                    count_raster = np.where(np.isnan(raster_data), 0, 1)
                    cumulative_count += count_raster

                elif value == "cumulative_windspeed":
                    cumulative_windspeed += np.nan_to_num(raster_data, nan=0)

            cumulative_windspeed[~mask_valid] = np.NaN
            cumulative_count[~mask_valid] = np.NaN
            
            print(f"Processed raster: {raster_file} for year {year_folder}")

        if value == "cumulative_count":
            output_name = f"cumulative_count_{year_folder}.tif"
            output_raster = cumulative_count

        elif value == "cumulative_windspeed":
            output_name = f"cumulative_windspeed_{year_folder}.tif"
            output_raster = cumulative_windspeed
        
        output_path = os.path.join(output_folder, output_name)
        save_raster(output_raster, output_path, mask_height, mask_width, mask_transform, mask_crs)
        print(f"Cumulative raster for year {year_folder} saved to {output_path}.")

def save_raster(data, path, height, width, transform, crs):
    """
    Save a raster to disk with LZW compression.
    """
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=np.NaN,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

def average_rasters(input_folder, start_year, end_year, output_name):
    output_folder = r"C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\year_averages"

    raster_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".tif") and any(str(year) in f for year in range(start_year, end_year + 1))
    ]

    if not raster_files:
        print(f"No rasters found between {start_year} and {end_year}.")
        return

    with rasterio.open(raster_files[0]) as src:
        meta = src.meta.copy()
        meta.update(dtype="float32", nodata=np.nan, compress="lzw")
        raster_shape = src.shape
        mask = src.read_masks(1)

    sum_array = np.zeros(raster_shape, dtype=np.float32)
    count_array = np.zeros(raster_shape, dtype=np.int32)

    for raster_file in raster_files:
        with rasterio.open(raster_file) as src:
            data = src.read(1)
            data[data == 0] = np.nan

            valid_mask = ~np.isnan(data)
            sum_array[valid_mask] += data[valid_mask]
            count_array[valid_mask] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        average_array = np.where(count_array > 0, sum_array / count_array, 0)

    average_array[mask == 0] = np.nan

    if output_name.endswith('hurricane_count_mean.tif'):
        average_array = gaussian_filter(average_array, sigma=10)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{start_year}_{end_year}_{output_name}")
    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(average_array, 1)

    print(f"Averaged raster saved to {output_path}.")

year_ranges = [
    (2004, 2006),
    (2006, 2010),
    (2010, 2015),
    (2015, 2022)]

download_hurricane_data()    
create_line_layer()
create_overlapping_buffers()
clip_polygons()
polygons_to_raster()

generate_cumulative_rasters(
    output_folder=r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\hurricane_count_rasters',
    value="cumulative_count")

generate_cumulative_rasters(
    output_folder=r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\hurricane_cumulative_rasters',
    value="cumulative_windspeed")

for start_year, end_year in year_ranges:
    average_rasters(
        input_folder=r"C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\hurricane_count_rasters",
        start_year=start_year,
        end_year=end_year,
        output_name=f"hurricane_count_mean.tif"
    )

    average_rasters(
        input_folder=r"C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\hurricane_data\hurricane_cumulative_rasters",
        start_year=start_year,
        end_year=end_year,
        output_name=f"hurricane_strength_mean.tif"
    )
