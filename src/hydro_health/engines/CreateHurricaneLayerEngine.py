"""Class to hold the logic for processing the Hurricane layer"""

import os
# Allow GDAL/Rasterio to write GeoTIFFs to S3 by using a local temp file under the hood
os.environ["CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE"] = "YES"
# Bypass GDAL free disk space checks for large rasters
os.environ["CHECK_DISK_FREE_SPACE"] = "FALSE"

import requests
import tempfile
import shutil
from collections import defaultdict
from upath import UPath

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features  # Explicitly imported to resolve AttributeError
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import Polygon
from shapely.ops import unary_union

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_environment


class CreateHurricaneLayerEngine(Engine):
    """Class to hold the logic for processing the Hurricane layer"""

    def __init__(self, overwrite: bool = False):
        super().__init__()
        
        self.is_aws = (get_environment() == 'aws')
        self.overwrite = overwrite
        
        self.sediment_data = None
        self.create_file_paths()

    def create_file_paths(self):
        """Creates Universal Paths based on the environment."""
        if self.is_aws:
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            # Disable directory caching since GDAL writes files outside of Python's fsspec awareness
            kwargs = {"use_listings_cache": False}
            
            # Use lstrip('/') to prevent 's3://bucket//path' double-slash S3 key errors
            self.hurricane_data_path = UPath(f"s3://{bucket}/{str(get_config_item('HURRICANE', 'GPKG_PATH')).lstrip('/')}", **kwargs)
            self.txt_data_path = UPath(f"s3://{bucket}/{str(get_config_item('HURRICANE', 'DATA_PATH')).lstrip('/')}", **kwargs)
            self.year_pair_raster_path = UPath(f"s3://{bucket}/{str(get_config_item('HURRICANE', 'YEAR_PAIR_RASTER_PATH')).lstrip('/')}", **kwargs)
            self.coast_boundary_path = UPath(f"s3://{bucket}/{str(get_config_item('MASK', 'COAST_BOUNDARY_PATH')).lstrip('/')}", **kwargs)
            self.raster_path = UPath(f"s3://{bucket}/{str(get_config_item('HURRICANE', 'RASTER_PATH')).lstrip('/')}", **kwargs)
            self.mask_pred_path = UPath(f"s3://{bucket}/{str(get_config_item('MASK', 'MASK_PRED_PATH')).lstrip('/')}", **kwargs)
            self.count_raster_path = UPath(f"s3://{bucket}/{str(get_config_item('HURRICANE', 'COUNT_RASTER_PATH')).lstrip('/')}", **kwargs)
            self.cumulative_raster_path = UPath(f"s3://{bucket}/{str(get_config_item('HURRICANE', 'CUMULATIVE_RASTER_PATH')).lstrip('/')}", **kwargs)
        else:
            self.hurricane_data_path = UPath(get_config_item('HURRICANE', 'GPKG_PATH'))
            self.txt_data_path = UPath(get_config_item('HURRICANE', 'DATA_PATH'))
            self.year_pair_raster_path = UPath(get_config_item('HURRICANE', 'YEAR_PAIR_RASTER_PATH'))
            self.coast_boundary_path = UPath(get_config_item('MASK', 'COAST_BOUNDARY_PATH'))
            self.raster_path = UPath(get_config_item('HURRICANE', 'RASTER_PATH'))
            self.mask_pred_path = UPath(get_config_item('MASK', 'MASK_PRED_PATH'))
            self.count_raster_path = UPath(get_config_item('HURRICANE', 'COUNT_RASTER_PATH'))
            self.cumulative_raster_path = UPath(get_config_item('HURRICANE', 'CUMULATIVE_RASTER_PATH'))

        print("--- Hurricane Engine File Paths ---")
        print(f"GPKG Data Path: {self.hurricane_data_path}")
        print(f"TXT Data Path: {self.txt_data_path}")
        print(f"Raster Base Path: {self.raster_path}")
        print(f"Overwrite Existing Files: {self.overwrite}")
        print("-----------------------------------")

    def _get_gdal_path(self, path_obj) -> str:
        """Convert UPath to GDAL-compatible virtual file string if on AWS."""
        path_str = str(path_obj)
        if path_str.startswith("s3://"):
            return path_str.replace("s3://", "/vsis3/")
        return path_str

    def _read_gpkg_layer(self, layer_name: str) -> gpd.GeoDataFrame:
        """Safely read a layer from a GeoPackage, avoiding GDAL /vsis3/ cache desyncs."""
        if self.is_aws:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_gpkg = os.path.join(tmpdir, "temp_read.gpkg")
                
                with self.hurricane_data_path.open('rb') as f_in, open(local_gpkg, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
                return gpd.read_file(local_gpkg, layer=layer_name)
        else:
            # Explicitly cast to string to prevent engine errors with custom UPath types
            return gpd.read_file(str(self.hurricane_data_path), layer=layer_name)

    def _save_gpkg_layer(self, gdf: gpd.GeoDataFrame, layer_name: str) -> None:
        """Safely save a layer to a GeoPackage, handling S3 temp files if necessary."""
        print(f"Saving layer '{layer_name}' to {self.hurricane_data_path}...")
        
        if gdf.empty:
            print(f"WARNING: The GeoDataFrame for layer '{layer_name}' is empty. Saving empty layer.")

        write_kwargs = {
            "layer": layer_name,
            "driver": "GPKG",
            "engine": "pyogrio"
        }
        
        file_exists = self.hurricane_data_path.exists()

        if self.is_aws:
            with tempfile.TemporaryDirectory() as tmpdir:
                local_gpkg = os.path.join(tmpdir, "temp.gpkg")
                
                if file_exists:
                    try:
                        with self.hurricane_data_path.open('rb') as f_in, open(local_gpkg, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                        write_kwargs["mode"] = "a"
                        write_kwargs["layer_options"] = {"OVERWRITE": "YES"}
                    except Exception as e:
                        print(f"Warning: Could not read existing GPKG from S3 ({e}). Starting fresh.")
                        file_exists = False
                
                try:
                    gdf.to_file(local_gpkg, **write_kwargs)
                except Exception as e:
                    print(f"Warning: Failed to append to GPKG ({e}). File may be corrupted. Recreating...")
                    if os.path.exists(local_gpkg):
                        os.remove(local_gpkg)
                    # Strip out append options and force a new file creation
                    write_kwargs.pop("mode", None)
                    write_kwargs.pop("layer_options", None)
                    gdf.to_file(local_gpkg, **write_kwargs)
                
                with open(local_gpkg, 'rb') as f_in, self.hurricane_data_path.open('wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            # Ensure the parent directory exists locally before pyogrio attempts creation
            self.hurricane_data_path.parent.mkdir(parents=True, exist_ok=True)
            
            if file_exists:
                write_kwargs["mode"] = "a"
                write_kwargs["layer_options"] = {"OVERWRITE": "YES"}
                
            try:
                # Explicitly cast to string to prevent C-engine errors
                gdf.to_file(str(self.hurricane_data_path), **write_kwargs)
            except Exception as e:
                print(f"Warning: Failed to append to GPKG ({e}). File may be corrupted. Recreating...")
                if self.hurricane_data_path.exists():
                    self.hurricane_data_path.unlink()
                # Strip out append options and force a new file creation
                write_kwargs.pop("mode", None)
                write_kwargs.pop("layer_options", None)
                gdf.to_file(str(self.hurricane_data_path), **write_kwargs)
            
        print(f"Successfully saved layer '{layer_name}'.")

    def average_rasters(self, input_folder, start_year, end_year, output_name) -> None:
        """
        Average and sum rasters for a given year range and save the results.
        
        Note: Despite the function name, this method calculates and outputs BOTH the 
        average (mean) raster and the cumulative (sum) raster for the specified year range.
        """

        output_folder = self.year_pair_raster_path
        output_folder.mkdir(parents=True, exist_ok=True)

        mean_output_path = output_folder / f"{output_name}_mean_{start_year}_{end_year}.tif"
        cumulative_output_path = output_folder / f"{output_name}_cumulative_{start_year}_{end_year}.tif"

        try:
            # Safely grab matching raster files and keep track of which years were found
            raster_files = []
            found_years = set()
            for f in input_folder.rglob("*.tif"):
                for year in range(start_year, end_year + 1):
                    if str(year) in f.name:
                        raster_files.append(f)
                        found_years.add(year)
                        break
        except Exception as e:
            print(f"Error accessing {input_folder}: {e}")
            return

        # Check for missing years before doing any calculations
        expected_years = set(range(start_year, end_year + 1))
        missing_years = expected_years - found_years
        
        if missing_years:
            print(f"Skipping year pair {start_year}-{end_year} for '{output_name}'. Missing data for years: {sorted(list(missing_years))}")
            return

        # Prepare dimensions and masks
        with rasterio.open(self._get_gdal_path(raster_files[0])) as src:
            meta = src.meta.copy()
            meta.update(dtype="float32", nodata=np.nan, compress="lzw")
            raster_shape = src.shape
            mask = src.read_masks(1)

        # Arrays to hold cumulative (sum) data and counts (for averages)
        sum_array = np.zeros(raster_shape, dtype=np.float32)
        count_array = np.zeros(raster_shape, dtype=np.int32)

        for raster_file in raster_files:
            with rasterio.open(self._get_gdal_path(raster_file)) as src:
                data = src.read(1)
                data[data == 0] = np.nan

                valid_mask = ~np.isnan(data)
                sum_array[valid_mask] += data[valid_mask]
                count_array[valid_mask] += 1

        # Calculate averages from the accumulated sum
        with np.errstate(divide='ignore', invalid='ignore'):
            average_array = np.where(count_array > 0, sum_array / count_array, 0)

        # Masking out nodata boundaries
        average_array[mask == 0] = np.nan
        sum_array[mask == 0] = np.nan

        # Save both results using the class's built-in save wrapper 
        self.save_raster(average_array, mean_output_path, raster_shape[0], raster_shape[1], meta['transform'], meta['crs'])
        self.save_raster(sum_array, cumulative_output_path, raster_shape[0], raster_shape[1], meta['transform'], meta['crs'])

        # UPDATED: Print exact output paths
        print(f"Saved mean raster to: {mean_output_path}")
        print(f"Saved cumulative raster to: {cumulative_output_path}")

    def clip_polygons(self) -> None:
        """Subtract higher wind radii from lower wind radii to create distinct rings."""

        print("Subtracting wind radii from hurricane polygons to create rings...")
        gdf_to_clip = self._read_gpkg_layer('atlantic_polygon_buffer')

        # Buffer by 0 to clean up invalid topological intersections before iterating
        gdf_to_clip['geometry'] = gdf_to_clip.geometry.buffer(0)
        gdf_to_clip = gdf_to_clip[~gdf_to_clip.geometry.is_empty & gdf_to_clip.geometry.is_valid]

        clipped_rows = []
        # Group by 'area_date' to prevent completely un-related UNNAMED storms from destroying each other
        for area_date, group in gdf_to_clip.groupby("area_date"):
            group = group.sort_values(by="wind_speed", ascending=False).reset_index(drop=True)
            for _, row in group.iterrows():
                geom = row.geometry
                for clipped_row in clipped_rows:
                    if clipped_row["area_date"] == area_date:
                        geom = geom.difference(clipped_row.geometry)
                if not geom.is_empty:
                    new_row = row.copy()
                    new_row.geometry = geom
                    clipped_rows.append(new_row)

        clipped_gdf = gpd.GeoDataFrame(clipped_rows, columns=gdf_to_clip.columns, crs=gdf_to_clip.crs)
        clipped_gdf = clipped_gdf[~clipped_gdf.geometry.is_empty & clipped_gdf.geometry.is_valid]

        # Ensure correct datatypes are propagated downstream
        clipped_gdf['dissolve_id'] = clipped_gdf['dissolve_id'].astype(str)
        clipped_gdf['id'] = clipped_gdf['id'].astype(str)
        clipped_gdf['area_date'] = clipped_gdf['area_date'].astype(str)
        clipped_gdf['name'] = clipped_gdf['name'].astype(str)
        clipped_gdf['year'] = pd.to_numeric(clipped_gdf['year'], errors='coerce').astype('Int32')
        clipped_gdf['wind_speed'] = pd.to_numeric(clipped_gdf['wind_speed'], errors='coerce').astype('Int32')

        layer_name = 'trimmed_polygons'
        self._save_gpkg_layer(clipped_gdf, layer_name)

    def convert_text_to_gpkg(self) -> gpd.GeoDataFrame:
        """Convert the hurricane text data to a GeoPackage with point and line layers."""

        print("Converting text data to GeoPackage points...")
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
                    
        txt_file = self.txt_data_path / 'atlantic_hurricane_data.txt'
            
        data = self.read_text_data(txt_file)
        df = pd.DataFrame(data)    
        
        if len(column_names) == df.shape[1]:
            df.columns = column_names
        else:
            raise ValueError('The number of column names does not match.')   

        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df['cyclone_num'] = df['area_date'].apply(lambda x: x[2:4])
        df['year'] = df['date'].apply(lambda x: x[0:4]).astype(int)
        
        df['latitude'] = df['latitude'].apply(self.convert_coordinates)
        df['longitude'] = df['longitude'].apply(self.convert_coordinates)

        # Convert wind and radii columns to numeric, replacing missing (-999) with NaN temporarily
        radii_cols = ['r_ne34', 'r_se34', 'r_sw34', 'r_nw34',
                      'r_ne50', 'r_se50', 'r_sw50', 'r_nw50',
                      'r_ne64', 'r_se64', 'r_sw64', 'r_nw64']
        
        for col in radii_cols + ['max_wind']:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-999)

        # --- APPLY PROXY RADII FOR MISSING HISTORICAL DATA ---
        
        # 34-knot proxy (100 nm)
        missing_34 = (df[['r_ne34', 'r_se34', 'r_sw34', 'r_nw34']] <= 0).all(axis=1)
        mask_34 = missing_34 & (df['max_wind'] >= 34)
        df.loc[mask_34, ['r_ne34', 'r_se34', 'r_sw34', 'r_nw34']] = 100

        # 50-knot proxy (50 nm)
        missing_50 = (df[['r_ne50', 'r_se50', 'r_sw50', 'r_nw50']] <= 0).all(axis=1)
        mask_50 = missing_50 & (df['max_wind'] >= 50)
        df.loc[mask_50, ['r_ne50', 'r_se50', 'r_sw50', 'r_nw50']] = 50

        # 64-knot proxy (25 nm)
        missing_64 = (df[['r_ne64', 'r_se64', 'r_sw64', 'r_nw64']] <= 0).all(axis=1)
        mask_64 = missing_64 & (df['max_wind'] >= 64)
        df.loc[mask_64, ['r_ne64', 'r_se64', 'r_sw64', 'r_nw64']] = 25

        # Estimate radius for 34 to 0 knots using a 1.5x factor
        # Safety Check: Use np.maximum(0, ...) so proxy holes don't generate massive negative buffers (-1498)
        df['r_ne0'] = np.maximum(0, pd.to_numeric(df['r_ne34'])) * 1.5
        df['r_nw0'] = np.maximum(0, pd.to_numeric(df['r_nw34'])) * 1.5
        df['r_se0'] = np.maximum(0, pd.to_numeric(df['r_se34'])) * 1.5
        df['r_sw0'] = np.maximum(0, pd.to_numeric(df['r_sw34'])) * 1.5

        gdf = gpd.GeoDataFrame(df, 
                                geometry=gpd.points_from_xy(df['longitude'], df['latitude']))  
        gdf.set_crs(crs="EPSG:4326", inplace=True)

        self._save_gpkg_layer(gdf, atlantic_point_layer_name)

        return gdf

    def convert_coordinates(self, coord) -> float:
        """Convert coordinates from string format to float."""

        coord = coord.replace('-', '') # TODO double check this isnt causing problems
        if 'N' in coord or 'E' in coord:
            return float(coord[:-1])
        elif 'S' in coord or 'W' in coord:
            return -float(coord[0:-1])
        else:
            raise ValueError ('Invalid coordinate format.')

    def create_line_layer(self, gdf: gpd.GeoDataFrame) -> None:
        """Create a line layer from the hurricane point data."""

        print("Creating line layer from points...")
        atlantic_line_layer_name = 'atlantic_hurricane_lines'

        lines = (gdf.groupby('area_date').apply(
            lambda points: {
                'name': points['name'].iloc[0],
                'year': points['year'].iloc[0],
                'geometry': LineString(points.geometry.tolist()) if len(points) > 1 else points.geometry.iloc[0]},
            include_groups=False))
            
        line_gdf = gpd.GeoDataFrame(list(lines), crs=gdf.crs).reset_index()
        self._save_gpkg_layer(line_gdf, atlantic_line_layer_name)

    def create_overlapping_buffers(self, gdf: gpd.GeoDataFrame) -> None:
        """Create overlapping buffers around hurricane points and save them as polygons."""

        print("Creating overlapping buffers...")
        # Maintain EPSG:5070 (NAD83 / Conus Albers) for SAFE geometrical buffering across the entire US
        gdf = gdf.to_crs('EPSG:5070') 
        
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
                            # Include area_date to uniquely identify storms sharing same name/year
                            'id': f"{wind_speed}_{quadrant}_{row['area_date']}",
                            'name': row['name'],
                            'year': row['year'],
                            'area_date': row['area_date'],
                            'wind_speed': wind_speed,
                            'longitude': row['longitude'],
                            'latitude': row['latitude'],
                            'buffer_radius_nm': buffer_value
                        })
        
        buffer_data = sorted(buffer_data, key=lambda x: x['id'])
        
        merged_buffer_data = []
        
        # 1. Pre-load ALL individual point circles first. 
        # This guarantees storms with only 1 historical record point don't vanish.
        for poly in buffer_data:
            merged_buffer_data.append({
                'geometry': poly['geometry'],
                'id': poly['id'],
                'name': poly['name'],
                'year': poly['year'],
                'area_date': poly['area_date'],
                'wind_speed': poly['wind_speed'],
                'dissolve_id': f"{poly['wind_speed']}_{poly['area_date']}"
            })

        # 2. Add sweeping connecting swaths between multi-record points
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
                'area_date': current_polygon['area_date'],
                'wind_speed': current_polygon['wind_speed'],
                'dissolve_id': f"{current_polygon['wind_speed']}_{current_polygon['area_date']}"
            })

        buffer_gdf = gpd.GeoDataFrame(merged_buffer_data, geometry='geometry', crs=gdf.crs)
        buffer_gdf = buffer_gdf.dissolve(by='dissolve_id')
        self._save_gpkg_layer(buffer_gdf, 'atlantic_polygon_buffer')

    def create_quarter_circle(self, center, radius, quadrant) -> Polygon:
        """Create a quarter circle polygon based on the center, radius, and quadrant."""

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

    def create_mega_polygon(self, current_polygon, next_polygon) -> Polygon:
        """Create a mega polygon by connecting the corner points of two polygons."""
        
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

    def download_hurricane_data(self) -> None:
        """Download the HURDAT2 hurricane data"""
        
        urls = [('atlantic_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2023-051124.txt'), 
                ('pacific_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-042624.txt')]
        
        print("Starting hurricane data download...")
        for filename, url in urls:
            response = requests.get(url)
            if response.status_code == 200:
                self.txt_data_path.mkdir(parents=True, exist_ok=True)
                filepath = self.txt_data_path / filename
                print(f"Downloading {filename} to {filepath}")
                with filepath.open('w', encoding='utf-8') as f:
                    f.write(response.text)
            else:
                print(f"Error downloading HURDAT2 data for {filename}.")  
        print("Download complete.")

    def generate_cumulative_rasters(self, output_folder, value) -> None:
        """Generate cumulative rasters for all possible hurricane years."""

        print(f"Generating cumulative rasters for {value}...")
        input_raster_folder = self.raster_path
        
        target_resolution = 100.0

        eco_path = "/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/Master_Grids.gpkg"
        print(f"Reading EcoRegions from {eco_path} for valid area mask...")
        eco_gdf = gpd.read_file(eco_path, layer='EcoRegions')
        
        # Project EcoRegions to match target CRS (EPSG:32617)
        print("Projecting EcoRegions to match target projection (EPSG:32617)...")
        eco_gdf = eco_gdf.to_crs('EPSG:32617')
        
        # FIX for topology twisting: Repair self-intersecting lines caused by out-of-bounds projection distortion
        eco_gdf['geometry'] = eco_gdf.geometry.buffer(0)
        eco_gdf = eco_gdf[eco_gdf.geometry.is_valid & ~eco_gdf.geometry.is_empty]
        
        mask_crs = eco_gdf.crs
        minx, miny, maxx, maxy = eco_gdf.total_bounds
        
        # Recalculate dimensions for exactly 100m resolution based on EcoRegion extent
        mask_width = int(np.ceil((maxx - minx) / target_resolution))
        mask_height = int(np.ceil((maxy - miny) / target_resolution))
        
        mask_transform = Affine(target_resolution, 0, minx,
                                0, -target_resolution, maxy)
        
        print("Rasterizing EcoRegions directly into EPSG:32617 array to avoid GDAL inverse projection failures...")
        shapes = [(geom, 1) for geom in eco_gdf.geometry if geom is not None and geom.is_valid and not geom.is_empty]
        mask_data = rasterio.features.rasterize(
            shapes,
            out_shape=(mask_height, mask_width),
            transform=mask_transform,
            fill=0,
            dtype='uint8'
        )
        
        mask_valid = mask_data == 1

        try:
            # Clear fsspec cache since GDAL may have written files behind its back
            if hasattr(input_raster_folder, "fs"):
                input_raster_folder.fs.invalidate_cache()
            all_rasters = list(input_raster_folder.rglob('*.tif'))
        except Exception as e:
            print(f"Error accessing {input_raster_folder}: {e}")
            all_rasters = []

        # Group the existing rasters by their parent year folder
        year_to_rasters = defaultdict(list)
        for r_path in all_rasters:
            year_to_rasters[r_path.parent.name].append(r_path)

        # Query the base dataset to determine ALL possible years
        print("Querying base dataset to find all possible years...")
        try:
            base_gdf = self._read_gpkg_layer('atlantic_hurricane_points')
            # Extract entire historical range dynamically without clamping to start_year/end_year
            min_year = int(base_gdf['year'].min())
            max_year = int(base_gdf['year'].max())
            all_possible_years = list(range(min_year, max_year + 1))
            print(f"Generating data for years {min_year} to {max_year}.")
        except Exception as e:
            print(f"Warning: Could not query dataset ({e}). Defaulting to existing folders.")
            if year_to_rasters:
                found_years = sorted([int(y) for y in year_to_rasters.keys() if y.isdigit()])
                if not found_years:
                    return
                all_possible_years = list(range(min(found_years), max(found_years) + 1))
            else:
                return

        # Iterate over ALL possible years, even if no rasters exist for that year
        for year in all_possible_years:
            year_folder = str(year)
            raster_files = year_to_rasters.get(year_folder, [])

            if not raster_files:
                print(f"No data for year {year_folder}. Skipping raster creation.")
                continue

            if value == "cumulative_count":
                output_name = f"cumulative_count_{year_folder}.tif"
            elif value == "cumulative_windspeed":
                output_name = f"cumulative_windspeed_{year_folder}.tif"
            
            output_path = output_folder / output_name

            # ----- NEW CHECK ADDED HERE -----
            if not self.overwrite and output_path.exists():
                print(f"Skipping cumulative raster for year {year_folder} ({value}) - already exists at {output_path}.")
                continue
            # --------------------------------

            # Default empty rasters
            cumulative_count = np.zeros((mask_height, mask_width), dtype=np.float32)
            cumulative_windspeed = np.zeros((mask_height, mask_width), dtype=np.float32)

            for raster_path in raster_files:
                with rasterio.open(self._get_gdal_path(raster_path)) as src:
                    raster_data = src.read(1)  
                    transform = src.transform
                    crs = src.crs

                    raster_data_resampled = np.full((mask_height, mask_width), np.nan, dtype=np.float32)
                    
                    # Safe to use reproject here because BOTH src and dst are now EPSG:32617.
                    # Bypasses the PROJ inverse pipeline entirely; acts strictly as an affine aligner.
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

                print(f"Processed raster: {raster_path.name} for year {year_folder}")
            
            # Apply Mask bounds constraint
            cumulative_windspeed[~mask_valid] = np.nan
            cumulative_count[~mask_valid] = np.nan

            if value == "cumulative_count":
                output_raster = cumulative_count
            elif value == "cumulative_windspeed":
                output_raster = cumulative_windspeed
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_raster(output_raster, output_path, mask_height, mask_width, mask_transform, mask_crs)
            print(f"Cumulative raster for year {year_folder} saved to {output_path}.")

        print(f"Cumulative raster generation complete for {value}.")

    def get_corner_points(self, quarter_circle, quadrant) -> list:
        """Get the corner points of a quarter circle polygon based on the quadrant."""

        coords = list(quarter_circle.exterior.coords)
        if quadrant == 'ne':
            return [coords[0], coords[1], coords[-1]]
        elif quadrant == 'nw':
            return [coords[0], coords[2], coords[-2]]
        elif quadrant == 'se':
            return [coords[0], coords[-2], coords[2]]
        elif quadrant == 'sw':
            return [coords[0], coords[-1], coords[1]]

    def polygons_to_raster(self, resolution=100.0) -> None:
        """Convert hurricane polygons to rasters."""

        print("Converting trimmed polygons to rasters...")
        
        try:
            gdf = self._read_gpkg_layer("trimmed_polygons")
        except Exception as e:
            print(f"Error opening 'trimmed_polygons' layer. It may be missing due to empty overlap. Exception: {e}")
            return

        print("Projecting trimmed polygons to target CRS (EPSG:32617) to prevent raster warping failure...")
        gdf = gdf.to_crs('EPSG:32617')
        
        # Repair the vector topology in case the extreme UTM distortion folded any polygons over themselves
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        
        # Load EcoRegions bounds to constrain massive ocean-wide storm rasterization sizes
        eco_path = "/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/Master_Grids.gpkg"
        print(f"Reading EcoRegions from {eco_path} to constrain raster bounding boxes...")
        eco_gdf = gpd.read_file(eco_path, layer='EcoRegions')
        eco_gdf = eco_gdf.to_crs('EPSG:32617')
        eco_minx, eco_miny, eco_maxx, eco_maxy = eco_gdf.total_bounds
            
        # Group by 'area_date' ensures unique files per storm.
        grouped = gdf.groupby('area_date')

        for area_date, group in grouped:
            
            name = group.iloc[0]['name']
            year = group.iloc[0]['year']
            safe_name = str(name).strip().replace(" ", "")
            
            output_folder = self.raster_path / str(year)
            output_folder.mkdir(parents=True, exist_ok=True)
            raster_file = output_folder / f"{safe_name}_{area_date}.tif"
            
            if not self.overwrite and raster_file.exists():
                print(f"Skipping individual raster {safe_name}_{area_date}.tif ({year}) - already exists.")
                continue  

            s_minx, s_miny, s_maxx, s_maxy = group.total_bounds
            
            # Constrain raster size to EcoRegion bounds to prevent massive memory/disk blowouts
            minx = max(s_minx, eco_minx)
            miny = max(s_miny, eco_miny)
            maxx = min(s_maxx, eco_maxx)
            maxy = min(s_maxy, eco_maxy)

            if minx >= maxx or miny >= maxy:
                print(f"Storm {name} - {year} is completely outside the target EcoRegions bounding box. Skipping rasterization.")
                continue

            width = int(np.ceil((maxx - minx) / resolution))
            height = int(np.ceil((maxy - miny) / resolution))
            
            if width <= 0 or height <= 0:
                continue

            transform = from_bounds(minx, miny, maxx, maxy, width, height)

            raster_data = np.full((height, width), np.nan, dtype=np.float32)

            for _, row in group.iterrows():
                shapes = [(row.geometry, row['wind_speed'])]
                rasterio.features.rasterize(
                    shapes,
                    out=raster_data,
                    transform=transform,
                )

            # Safely save the generated raster via local temp wrapper (Native EPSG:32617)
            with tempfile.TemporaryDirectory() as tmpdir:
                local_raster = os.path.join(tmpdir, "temp.tif")
                with rasterio.open(
                    local_raster,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=raster_data.dtype,
                    crs=gdf.crs,
                    nodata=np.nan,
                    compress="lzw",  # Ensures we don't eat all disk space for individual storms
                    transform=transform,
                ) as dst:
                    dst.write(raster_data, 1)

                if self.is_aws:
                    with open(local_raster, 'rb') as f_in, raster_file.open('wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                else:
                    shutil.copy(local_raster, raster_file)

            print(f"Raster for {name} - {year} saved to {raster_file}.")

    def read_text_data(self, txt_path) -> list:
        """Read the hurricane data from a text file."""  

        with UPath(txt_path).open('r', encoding='utf-8') as f:
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
        """Entrypoint for processing the Hurricane layer"""
        print("Starting Hurricane Layer Engine processing...")

        run_vector_processing = True

        if self.overwrite:
            print("Overwrite is enabled. Removing existing GeoPackage and Rasters to start fresh...")
            try:
                if self.hurricane_data_path.exists():
                    self.hurricane_data_path.unlink()
            except Exception as e:
                print(f"Warning: Could not delete existing GPKG: {e}")

            # FIX: Explicitly scrub the base raster path of old run files so tiny old footprints 
            # don't bleed into the new Conus-wide cumulative footprints
            try:
                if self.raster_path.exists():
                    for tif in self.raster_path.rglob("*.tif"):
                        # Only delete files inside year folders (e.g., 1851/storm.tif)
                        if tif.parent.name.isdigit():
                            tif.unlink()
            except Exception as e:
                print(f"Warning: Could not delete existing year rasters: {e}")
        else:
            # Short-circuit: Check if we can skip the heavy vector generation steps
            try:
                if self.hurricane_data_path.exists():
                    print("Checking for existing processed vector data...")
                    _ = self._read_gpkg_layer('trimmed_polygons')
                    print("Processed vector data found! Skipping download and buffering steps.")
                    run_vector_processing = False
            except Exception:
                print("Vector data incomplete or missing. Proceeding with full generation.")

        if run_vector_processing:
            self.download_hurricane_data()    
            
            # Parse points once and pass the Dataframe down the chain
            point_gdf = self.convert_text_to_gpkg()
            self.create_line_layer(point_gdf)
            self.create_overlapping_buffers(point_gdf)
            
            self.clip_polygons()
            
        self.polygons_to_raster()

        self.generate_cumulative_rasters(
            output_folder=self.count_raster_path,
            value="cumulative_count")

        self.generate_cumulative_rasters(
            output_folder=self.cumulative_raster_path,
            value="cumulative_windspeed")

        print("Averaging rasters for defined year ranges...")
        for start_year, end_year in self.year_ranges:
            self.average_rasters(
                input_folder=self.count_raster_path,
                start_year=start_year,
                end_year=end_year,
                output_name=f"hurr_count"
            )

            self.average_rasters(
                input_folder=self.cumulative_raster_path,
                start_year=start_year,
                end_year=end_year,
                output_name=f"hurr_strength"
            )
        
        print("Hurricane Layer Engine processing complete.")

    def save_raster(self, data, path_obj, height, width, transform, crs) -> None:
        """Save the raster data to a file safely using a local temp file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            local_raster = os.path.join(tmpdir, "temp.tif")
            with rasterio.open(
                local_raster,
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

            if self.is_aws:
                with open(local_raster, 'rb') as f_in, path_obj.open('wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            else:
                shutil.copy(local_raster, path_obj)