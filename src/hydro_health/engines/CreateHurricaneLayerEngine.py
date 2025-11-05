import os
import pathlib
import requests

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import gaussian_filter
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import unary_union

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


class CreateHurricaneLayerEngine(Engine):
    """Class to hold the logic for processing the Hurricane layer"""

    def __init__(self):
        super().__init__()
        self.year_ranges = [
            (1998, 2004),
            (2004, 2006),
            (2006, 2007),
            (2007, 2010),
            (2010, 2015),
            (2014, 2022),
            (2016, 2017),
            (2017, 2018),
            (2018, 2019),
            (2020, 2022),
            (2022, 2024)
        ]
        self.hurricane_data_path = pathlib.Path(get_config_item('HURRICANE', 'GPKG_PATH'))
        self.sediment_data = None
        self.txt_data_path = pathlib.Path(get_config_item('HURRICANE', 'DATA_PATH'))

    def average_rasters(self, input_folder, start_year, end_year, output_name)-> None:
        """Average rasters for a given year range and save the result.

        :param input_folder: Folder containing the input rasters
        :param start_year: Start year for the averaging
        :param end_year: End year for the averaging
        :param output_name: Name of the output raster file
        :return: None
        """

        output_folder = pathlib.Path(get_config_item('HURRICANE', 'YEAR_PAIR_RASTER_PATH'))
        os.makedirs(output_folder, exist_ok=True)

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
        output_path = os.path.join(output_folder, f"{output_name}_{start_year}_{end_year}.tif")
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(average_array, 1)

        print(f"Year pair raster saved to {output_path}.")    

    def clip_polygons(self)-> None:
        """Clip the hurricane polygons to the coastal boundary and save the result.

        :return: None
        """

        gdf_to_clip = gpd.read_file(self.hurricane_data_path, layer='atlantic_polygon_buffer')
        clip_boundary = gpd.read_file(get_config_item('MASK', 'COAST_BOUNDARY_PATH'))

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
        clipped_gdf.to_file(self.hurricane_data_path, layer=layer_name, driver='GPKG', overwrite=True)

    def convert_text_to_gpkg(self)-> gpd.GeoDataFrame:
        """Convert the hurricane text data to a GeoPackage with point and line layers.

        :return: GeoDataFrame containing hurricane data"""

        atlantic_point_layer_name = 'atlantic_hurricane_points'

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
                    
        data = self.read_text_data(self.txt_data_path / 'atlantic_hurricane_data.txt')
        df = pd.DataFrame(data)    
        
        if len(column_names) == df.shape[1]:
            df.columns = column_names
        else:
            raise ValueError('The number of column names does not match.')   

        df = df.applymap(lambda x: x.strip())
        df['cyclone_num'] = df['area_date'].apply(lambda x: x[2:4])
        df['year'] = df['date'].apply(lambda x: x[0:4]).astype(int)
        df['latitude'] = df['latitude'].apply(self.convert_coordinates)
        df['longitude'] = df['longitude'].apply(self.convert_coordinates)

        # Estimate radius for 34 to 0 knots using a 1.5x factor
        df['r_ne0'] = pd.to_numeric(df['r_ne34']) * 1.5
        df['r_nw0'] = pd.to_numeric(df['r_nw34']) * 1.5
        df['r_se0'] = pd.to_numeric(df['r_se34']) * 1.5
        df['r_sw0'] = pd.to_numeric(df['r_sw34']) * 1.5

        gdf = gpd.GeoDataFrame(df, 
                                geometry=gpd.points_from_xy(df['longitude'], df['latitude']))  
        gdf.set_crs(crs="EPSG:4326", inplace=True)

        gdf.to_file(self.hurricane_data_path, layer=atlantic_point_layer_name, driver='GPKG', overwrite=True)

        return gdf

    def convert_coordinates(self, coord)-> float:
        """Convert coordinates from string format to float.

        :param str coord: coordinate in string format (e.g., '25.0N', '80.0W')
        :return: float representation of the coordinate
        """

        coord = coord.replace('-', '') # TODO double check this isn't causing problems
        if 'N' in coord or 'E' in coord:
            return float(coord[:-1])
        elif 'S' in coord or 'W' in coord:
            return -float(coord[0:-1])
        else:
            raise ValueError ('Invalid coordinate format.')

    def create_line_layer(self)-> None:
        """Create a line layer from the hurricane point data.

        :return: None
        """

        atlantic_line_layer_name = 'atlantic_hurricane_lines'
        gdf = self.convert_text_to_gpkg()

        lines = (gdf.groupby('area_date').apply(
            lambda points: {
                'name': points['name'].iloc[0],
                'year': points['year'].iloc[0],
                'geometry': LineString(points.geometry.tolist()) if len(points) > 1 else points.geometry.iloc[0]}))
            
        line_gdf = gpd.GeoDataFrame(list(lines), crs=gdf.crs).reset_index()
        line_gdf.to_file(self.hurricane_data_path, layer=atlantic_line_layer_name, driver='GPKG', overwrite=True)

    def create_overlapping_buffers(self)-> None:
        """Create overlapping buffers around hurricane points and save them as polygons.

        :return: None
        """

        gdf = self.convert_text_to_gpkg()
        gdf = gdf[gdf['year'] >= 1998]
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
                        
                        quarter_circle = self.create_quarter_circle(center, buffer_value * 1852, quadrant)
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
        buffer_gdf.to_file(self.hurricane_data_path, layer='atlantic_point_buffer', driver='GPKG', overwrite=True)
        
        merged_buffer_data = []
        for i in range(len(buffer_data) - 1):
            current_polygon = buffer_data[i]
            if buffer_data[i]['id'] == buffer_data[i + 1]['id']:
                next_polygon = buffer_data[i + 1]
            else:
                continue    

            mega_polygon = self.create_mega_polygon(current_polygon, next_polygon)

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
        buffer_gdf.to_file(self.hurricane_data_path, layer='atlantic_polygon_buffer', driver='GPKG', overwrite=True)

    def create_quarter_circle(self, center, radius, quadrant)-> Polygon:
        """Create a quarter circle polygon based on the center, radius, and quadrant.

        :param center: Shapely Point representing the center of the quarter circle
        :param radius: Radius of the quarter circle in meters
        :param quadrant: Quadrant of the quarter circle ('ne', 'nw', 'se', 'sw')
        :return: Shapely Polygon representing the quarter circle
        """

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

    def create_mega_polygon(self, current_polygon, next_polygon)-> Polygon:
        """Create a mega polygon by connecting the corner points of two polygons.

        :param current_polygon: Dictionary containing the current polygon data
        :param next_polygon: Dictionary containing the next polygon data
        :return: Shapely Polygon representing the mega polygon
        """
        
        current_corners = self.get_corner_points(current_polygon['geometry'], current_polygon['label'].split('_')[0])
        next_corners = self.get_corner_points(next_polygon['geometry'], next_polygon['label'].split('_')[0])

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

    def download_hurricane_data(self)-> None:
        """Download the HURDAT2 hurricane data"""
        
        urls = [('atlantic_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt'), 
                ('pacific_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-042624.txt')]
        
        for filename, url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                make_dirs = pathlib.Path(self.txt_data_path)
                make_dirs.mkdir(parents=True, exist_ok=True)
                filepath = os.path.join(self.txt_data_path, filename)

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            else:
                print('Error downloading HURDAT2 data.')  

    def generate_cumulative_rasters(self, output_folder, value)-> None:
        """Generate cumulative rasters for hurricane data.

        :param output_folder: Folder to save the cumulative rasters
        :param value: Type of cumulative raster to generate ('cumulative_count' or 'cumulative_windspeed')
        :return: None
        """

        input_raster_folder = get_config_item('HURRICANE', 'RASTER_PATH')
        mask_raster_path = get_config_item('MASK', 'MASK_PRED_PATH')
        
        target_resolution = 100.0

        with rasterio.open(mask_raster_path) as mask_src:
            mask_crs = mask_src.crs
            
            minx, miny, maxx, maxy = mask_src.bounds
            
            new_width = int(np.ceil((maxx - minx) / target_resolution))
            new_height = int(np.ceil((maxy - miny) / target_resolution))
            
            mask_transform = Affine(target_resolution, 0, minx,
                                    0, -target_resolution, maxy)
            
            mask_data_resampled = np.full((new_height, new_width), np.nan, dtype=np.float32)
            reproject(
                source=mask_src.read(1),
                destination=mask_data_resampled,
                src_transform=mask_src.transform,
                src_crs=mask_src.crs,
                dst_transform=mask_transform,
                dst_crs=mask_crs,
                resampling=Resampling.nearest,
                dst_nodata=np.nan
            )
            mask_data = mask_data_resampled
            mask_height, mask_width = new_height, new_width
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

                    raster_data_resampled = np.full((mask_height, mask_width), np.nan, dtype=np.float32)
                    
                    reproject(
                        raster_data, raster_data_resampled,
                        src_transform=transform,
                        src_crs=crs,
                        dst_transform=mask_transform,
                        dst_crs=mask_crs,
                        resampling=Resampling.bilinear,
                        dst_nodata=np.nan 
                    )

                    raster_data = raster_data_resampled
                    
                    if value == "cumulative_count":
                        count_raster = np.where(np.isnan(raster_data), 0, 1)
                        cumulative_count += count_raster

                    elif value == "cumulative_windspeed":
                        cumulative_windspeed += np.nan_to_num(raster_data, nan=0)

                print(f"Processed raster: {raster_file} for year {year_folder}")
            
            cumulative_windspeed[~mask_valid] = np.nan
            cumulative_count[~mask_valid] = np.nan

            if value == "cumulative_count":
                output_name = f"cumulative_count_{year_folder}.tif"
                output_raster = cumulative_count

            elif value == "cumulative_windspeed":
                output_name = f"cumulative_windspeed_{year_folder}.tif"
                output_raster = cumulative_windspeed
            
            output_path = os.path.join(output_folder, output_name)

            self.save_raster(output_raster, output_path, mask_height, mask_width, mask_transform, mask_crs)
            print(f"Cumulative raster for year {year_folder} saved to {output_path}.")

        print("Cumulative raster generation complete.")

    def get_corner_points(self, quarter_circle, quadrant)-> list:
        """Get the corner points of a quarter circle polygon based on the quadrant.

        :param quarter_circle: Shapely Polygon representing the quarter circle
        :param quadrant: Quadrant of the quarter circle ('ne', 'nw', 'se', 'sw')
        :return: List of corner points as tuples
        """

        coords = list(quarter_circle.exterior.coords)
        if quadrant == 'ne':
            return [coords[0], coords[1], coords[-1]]
        elif quadrant == 'nw':
            return [coords[0], coords[2], coords[-2]]
        elif quadrant == 'se':
            return [coords[0], coords[-2], coords[2]]
        elif quadrant == 'sw':
            return [coords[0], coords[-1], coords[1]]

    def polygons_to_raster(self, resolution=500)-> None:
        """Convert hurricane polygons to rasters.
        
        :param resolution: Resolution of the output raster in meters
        :return: None"""

        gdf = gpd.read_file(self.hurricane_data_path, layer="trimmed_polygons")

        grouped = gdf.groupby(['name', 'year'])

        for (name, year), group in grouped:
            output_folder = pathlib.Path(get_config_item('HURRICANE', 'RASTER_PATH')) / str(year)
            os.makedirs(output_folder, exist_ok=True)
            raster_file = output_folder / f"{name}_{year}.tif"
            
            if raster_file.exists():
                print(f"Skipping {raster_file.name}, already exists.")
                continue  

            minx, miny, maxx, maxy = group.total_bounds
            width = int(np.ceil((maxx - minx) / resolution))
            height = int(np.ceil((maxy - miny) / resolution))
            transform = from_bounds(minx, miny, maxx, maxy, width, height)

            raster_data = np.full((height, width), np.nan, dtype=np.float32)

            for _, row in group.iterrows():
                shapes = [(row.geometry, row['wind_speed'])]
                rasterio.features.rasterize(
                    shapes,
                    out=raster_data,
                    transform=transform,
                    fill=np.nan, 
                )

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
                nodata=np.nan,
                transform=transform,
            ) as dst:
                dst.write(raster_data, 1)

            print(f"Raster for {name} - {year} saved to {raster_file}.")

    def read_text_data(self, txt_path)-> list:
        """Read the hurricane data from a text file.
        
        :param str txt_path: path to the text file containing hurricane data
        :return: List of lists containing hurricane data
        """  

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

    def run(self):
        """Entrypoint for processing the Hurricane layer
        HURDAT2 data before 2004 does not have wind radii information.
        Run time is 1.25 hours on the remote desktop.
        """

        self.download_hurricane_data()    
        self.create_line_layer()
        self.create_overlapping_buffers()
        self.clip_polygons()
        self.polygons_to_raster()

        self.generate_cumulative_rasters(
            output_folder=get_config_item('HURRICANE', 'COUNT_RASTER_PATH'),
            value="cumulative_count")

        self.generate_cumulative_rasters(
            output_folder=get_config_item('HURRICANE', 'CUMULATIVE_RASTER_PATH'),
            value="cumulative_windspeed")

        for start_year, end_year in self.year_ranges:
            self.average_rasters(
                input_folder=get_config_item('HURRICANE', 'COUNT_RASTER_PATH'),
                start_year=start_year,
                end_year=end_year,
                output_name=f"hurr_count"
            )

            self.average_rasters(
                input_folder=get_config_item('HURRICANE', 'CUMULATIVE_RASTER_PATH'),
                start_year=start_year,
                end_year=end_year,
                output_name=f"hurr_strength"
            )

    def save_raster(self, data, path, height, width, transform, crs)-> None:
        """Save the raster data to a file.

        :param data: Raster data to save
        :param path: Path to save the raster file
        :param height: Height of the raster
        :param width: Width of the raster
        :param transform: Affine transform for the raster
        :param crs: Coordinate reference system for the raster
        :return: None
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
            nodata=np.nan,
            compress="lzw",
        ) as dst:
            dst.write(data, 1)
