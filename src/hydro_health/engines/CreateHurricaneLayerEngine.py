"""Class to hold the logic for processing the Hurricane layer"""

import os
import pathlib
import requests
import tempfile
import shutil
from collections import defaultdict
from upath import UPath

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import rasterio.features  # Explicitly imported to resolve AttributeError
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.transform import from_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds as window_from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import GeometryCollection
from shapely.geometry import LineString
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class CreateHurricaneLayerEngine(Engine):
    """Class to hold the logic for processing the Hurricane layer"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix
        self.is_aws = param_lookup.get('env', 'local') == 'aws'
        self.overwrite = True

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""
        self.region = region
        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f'CreateHurricaneLayerEngine resolved outputs_dir for region {region}: {self.outputs_dir}', OUTPUTS)

        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        s3_dir_base = f's3://{bucket}/{region}'

        data_path = get_config_item('HURRICANE', 'DATA_PATH')
        local_data_base = OUTPUTS / self.output_prefix if self.output_prefix else OUTPUTS
        self.txt_data_path = UPath(f's3://{bucket}/{data_path}') if self.is_aws else UPath(local_data_base / data_path)

        gpkg_path = get_config_item('HURRICANE', 'GPKG_PATH')
        self.hurricane_data_path = UPath(f's3://{bucket}/{gpkg_path}') if self.is_aws else UPath(local_data_base / gpkg_path)

        year_pair_raster_path = get_config_item('HURRICANE', 'YEAR_PAIR_RASTER_PATH')
        self.year_pair_raster_path = UPath(f'{s3_dir_base}/{year_pair_raster_path}') if self.is_aws else UPath(self.outputs_dir / year_pair_raster_path)

        raster_path = get_config_item('HURRICANE', 'RASTER_PATH')
        self.raster_path = UPath(f'{s3_dir_base}/{raster_path}') if self.is_aws else UPath(self.outputs_dir / raster_path)

        count_raster_path = get_config_item('HURRICANE', 'COUNT_RASTER_PATH')
        self.count_raster_path = UPath(f'{s3_dir_base}/{count_raster_path}') if self.is_aws else UPath(self.outputs_dir / count_raster_path)

        cumulative_raster_path = get_config_item('HURRICANE', 'CUMULATIVE_RASTER_PATH')
        self.cumulative_raster_path = UPath(f'{s3_dir_base}/{cumulative_raster_path}') if self.is_aws else UPath(self.outputs_dir / cumulative_raster_path)

        prediction_mask_path = get_config_item('MASK', 'MASK_PRED_PATH')
        self.mask_pred_path = UPath(f'{s3_dir_base}/{prediction_mask_path}') if self.is_aws else UPath(self.outputs_dir / prediction_mask_path)

        master_grids_path = str(get_config_item('SHARED', 'MASTER_GRIDS'))
        self.coast_boundary_path = UPath(master_grids_path) if '://' in master_grids_path else UPath(INPUTS / master_grids_path)
        if not self.is_aws:
            self.hurricane_data_path.parent.mkdir(parents=True, exist_ok=True)
            self.txt_data_path.mkdir(parents=True, exist_ok=True)
            self.year_pair_raster_path.mkdir(parents=True, exist_ok=True)
            self.raster_path.mkdir(parents=True, exist_ok=True)
            self.count_raster_path.mkdir(parents=True, exist_ok=True)
            self.cumulative_raster_path.mkdir(parents=True, exist_ok=True)

    def _read_ecoregion(self) -> gpd.GeoDataFrame:
        """Read the selected region using its identifier column in the master grids."""
        eco_gdf = self._read_gpkg_layer('Enhanced_EcoRegions', self.coast_boundary_path)
        region_column = self.param_lookup.get('eco_region_column')
        if region_column is None:
            identifier_names = {'ecoregion', 'ecoregionid', 'ecoregioncode', 'er', 'ercode'}
            candidates = [column for column in eco_gdf.columns
                          if column != eco_gdf.geometry.name
                          and str(column).lower().replace('_', '').replace(' ', '') in identifier_names]
            if len(candidates) != 1:
                raise ValueError("Set param_lookup['eco_region_column'] to the master grids' ecoregion identifier column")
            region_column = candidates[0]
        if region_column not in eco_gdf.columns:
            raise ValueError(f"Ecoregion column {region_column!r} was not found in Enhanced_EcoRegions")
        eco_gdf = eco_gdf.loc[eco_gdf[region_column].astype(str).str.strip().eq(self.region)].copy()
        if eco_gdf.empty:
            raise ValueError(f"Ecoregion {self.region!r} was not found in Enhanced_EcoRegions")
        return eco_gdf.to_crs(self.target_crs)

    def _get_gdal_path(self, path_obj: UPath | str) -> str:
        """Convert UPath to GDAL-compatible virtual file string if on AWS."""
        path_str = str(path_obj)
        if path_str.startswith("s3://"):
            return path_str.replace("s3://", "/vsis3/")
        return path_str

    def _read_gpkg_layer(self, layer_name: str, path_obj: UPath | None = None) -> gpd.GeoDataFrame:
        """Safely read a layer from a GeoPackage, avoiding GDAL /vsis3/ cache desyncs."""
        target_path = path_obj if path_obj is not None else self.hurricane_data_path

        # Only use the tempfile download workaround if the target path is actually in S3
        if self.is_aws and str(target_path).startswith("s3://"):
            with tempfile.TemporaryDirectory() as tmpdir:
                local_gpkg = os.path.join(tmpdir, "temp_read.gpkg")

                with target_path.open('rb') as f_in, open(local_gpkg, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

                return gpd.read_file(local_gpkg, layer=layer_name)
        else:
            # Explicitly cast to string to prevent engine errors with custom UPath types
            return gpd.read_file(str(target_path), layer=layer_name)

    def _save_gpkg_layer(self, gdf: gpd.GeoDataFrame, layer_name: str) -> None:
        """Stage layer updates locally, preserving the existing file if writing fails."""
        self.write_message(f"Saving layer '{layer_name}' to {self.hurricane_data_path}...", OUTPUTS)
        if not self.is_aws:
            self.hurricane_data_path.parent.mkdir(parents=True, exist_ok=True)
        temp_parent = None if self.is_aws else str(self.hurricane_data_path.parent)
        with tempfile.TemporaryDirectory(dir=temp_parent) as tmpdir:
            local_gpkg = os.path.join(tmpdir, 'hurricane.gpkg')
            if self.hurricane_data_path.exists():
                with self.hurricane_data_path.open('rb') as src, open(local_gpkg, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                existing_layers = {name for name, _ in pyogrio.list_layers(local_gpkg)}
                if not self.overwrite and layer_name in existing_layers:
                    self.write_message(f"Skipping existing GeoPackage layer: {layer_name}", OUTPUTS)
                    return
            gdf.to_file(local_gpkg, layer=layer_name, driver='GPKG', engine='pyogrio', mode='w')
            if self.is_aws:
                with open(local_gpkg, 'rb') as src, self.hurricane_data_path.open('wb') as dst:
                    shutil.copyfileobj(src, dst)
            else:
                os.replace(local_gpkg, str(self.hurricane_data_path))
        self.write_message(f"Successfully saved layer '{layer_name}'.", OUTPUTS)

    def year_pair_rasters(self, start_year: int, end_year: int) -> None:
        """Align annual rasters to the first year's grid and average valid values, including zeros."""
        if end_year < start_year:
            raise ValueError('end_year must be greater than or equal to start_year')
        output_path = self.year_pair_raster_path / f'hurr_strength_mean_{start_year}_{end_year}.tif'
        if not self.overwrite and output_path.exists():
            self.write_message(f"Skipping existing year-pair raster: {output_path}", OUTPUTS)
            return
        raster_files = [self.cumulative_raster_path / f'cumulative_windspeed_{year}.tif'
                        for year in range(start_year, end_year + 1)]
        missing_years = [year for year, path in zip(range(start_year, end_year + 1), raster_files)
                         if not path.exists()]
        if missing_years:
            self.write_message(f"Skipping year pair {start_year}-{end_year}; missing annual rasters: {missing_years}", OUTPUTS)
            return
        with rasterio.open(self._get_gdal_path(raster_files[0])) as src:
            raster_shape = src.shape
            transform = src.transform
            crs = src.crs
        sum_array = np.zeros(raster_shape, dtype=np.float32)
        count_array = np.zeros(raster_shape, dtype=np.int32)
        for raster_file in raster_files:
            with rasterio.open(self._get_gdal_path(raster_file)) as src:
                data = src.read(1, masked=True).astype(np.float32).filled(np.nan)
                if src.shape != raster_shape or src.transform != transform or src.crs != crs:
                    if src.crs is None or crs is None:
                        raise ValueError(f"Cannot align annual raster with a missing CRS: {raster_file}")
                    self.write_message(f"Aligning annual raster to the year-pair reference grid: {raster_file}", OUTPUTS)
                    aligned_data = np.full(raster_shape, np.nan, dtype=np.float32)
                    reproject(
                        source=data,
                        destination=aligned_data,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        src_nodata=np.nan,
                        dst_transform=transform,
                        dst_crs=crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.nearest,
                    )
                    data = aligned_data
                valid = np.isfinite(data)
                sum_array[valid] += data[valid]
                count_array[valid] += 1
        average_array = np.full(raster_shape, np.nan, dtype=np.float32)
        np.divide(sum_array, count_array, out=average_array, where=count_array > 0)
        average_array, transform = self._clip_to_prediction_bounds(average_array, transform, crs)
        self.year_pair_raster_path.mkdir(parents=True, exist_ok=True)
        self.save_raster(average_array, output_path, *average_array.shape, transform, crs)
        self.write_message(f"Saved mean raster to: {output_path}", OUTPUTS)

    def _prediction_bounds(self, crs: rasterio.crs.CRS | str) -> tuple[float, float, float, float]:
        """Read prediction mask bounds in the output raster's CRS."""
        with rasterio.open(self._get_gdal_path(self.mask_pred_path)) as mask:
            if mask.crs is None or crs is None:
                raise ValueError('The prediction mask and output raster must both have a CRS')
            bounds = tuple(mask.bounds)
            if mask.crs != rasterio.crs.CRS.from_user_input(crs):
                bounds = transform_bounds(mask.crs, crs, *bounds, densify_pts=21)
        if not np.all(np.isfinite(bounds)):
            raise ValueError(f'Invalid prediction mask bounding box: {bounds}')
        return bounds

    def _clip_to_prediction_bounds(self, data: np.ndarray, transform: Affine, crs: rasterio.crs.CRS | str) -> tuple[np.ndarray, Affine]:
        """Crop to the prediction TIFF's bounding box on the existing pixel grid."""
        bounds = self._prediction_bounds(crs)
        window = window_from_bounds(*bounds, transform=transform)
        # Include intersecting edge pixels while avoiding floating-point off-by-one errors.
        col_start = max(0, int(np.floor(window.col_off + 1e-8)))
        row_start = max(0, int(np.floor(window.row_off + 1e-8)))
        col_end = min(data.shape[1], int(np.ceil(window.col_off + window.width - 1e-8)))
        row_end = min(data.shape[0], int(np.ceil(window.row_off + window.height - 1e-8)))
        if col_start >= col_end or row_start >= row_end:
            raise ValueError(f'Prediction mask bounding box does not overlap the output raster: {self.mask_pred_path}')
        crop_window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
        clipped = data[row_start:row_end, col_start:col_end]
        clipped_transform = window_transform(crop_window, transform)
        self.write_message(f'Cropped raster to prediction mask bounds: {self.mask_pred_path}; shape={clipped.shape}', OUTPUTS)
        return clipped, clipped_transform

    def clip_polygons(self) -> None:
        """Subtract higher wind radii from lower wind radii to create distinct rings."""

        self.write_message("Subtracting wind radii from hurricane polygons to create rings...", OUTPUTS)
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

        self.write_message("Converting text data to GeoPackage points...", OUTPUTS)
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

    def convert_coordinates(self, coord: str) -> float:
        """Convert coordinates from string format to float."""

        coord = coord.replace('-', '') # TODO double check this isnt causing problems
        if 'N' in coord or 'E' in coord:
            return float(coord[:-1])
        elif 'S' in coord or 'W' in coord:
            return -float(coord[0:-1])
        else:
            raise ValueError ('Invalid coordinate format.')

    def create_overlapping_buffers(self, gdf: gpd.GeoDataFrame) -> None:
        """Create overlapping buffers around hurricane points and save them as polygons."""

        self.write_message("Creating overlapping buffers...", OUTPUTS)
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

    def create_quarter_circle(self, center: Point, radius: float, quadrant: str) -> Polygon:
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

    def create_mega_polygon(self, current_polygon: dict, next_polygon: dict) -> Polygon:
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

        urls = [('atlantic_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-02272026.txt'),
                ('pacific_hurricane_data.txt', 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-nepac-1949-2023-042624.txt')]

        self.write_message("Starting hurricane data download...", OUTPUTS)
        for filename, url in urls:
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            self.txt_data_path.mkdir(parents=True, exist_ok=True)
            filepath = self.txt_data_path / filename
            self.write_message(f"Downloading {filename} to {filepath}", OUTPUTS)
            with filepath.open('w', encoding='utf-8') as f:
                f.write(response.text)
        self.write_message("Download complete.", OUTPUTS)

    def generate_cumulative_rasters(self, output_folder: UPath, value: str) -> None:
        """Generate cumulative rasters for all possible hurricane years."""

        if value not in {'cumulative_count', 'cumulative_windspeed'}:
            raise ValueError(f"Unknown cumulative raster type: {value}")
        self.write_message(f"Generating cumulative rasters for {value}...", OUTPUTS)
        input_raster_folder = self.raster_path

        target_resolution = 100.0

        self.write_message(f"Reading EcoRegions from {self.coast_boundary_path} for valid area mask...", OUTPUTS)
        eco_gdf = self._read_ecoregion()

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

        self.write_message("Rasterizing EcoRegions directly into the target CRS array to avoid GDAL inverse projection failures...", OUTPUTS)
        shapes = [(geom, 1) for geom in eco_gdf.geometry if geom is not None and geom.is_valid and not geom.is_empty]
        mask_data = rasterio.features.rasterize(
            shapes,
            out_shape=(mask_height, mask_width),
            transform=mask_transform,
            fill=0,
            dtype='uint8'
        )

        mask_valid = mask_data == 1

        all_rasters = list(input_raster_folder.rglob('*.tif'))

        # Group the existing rasters by their parent year folder
        year_to_rasters = defaultdict(list)
        for r_path in all_rasters:
            year_to_rasters[r_path.parent.name].append(r_path)

        base_gdf = self._read_gpkg_layer('atlantic_hurricane_points')
        min_year = int(base_gdf['year'].min())
        max_year = int(base_gdf['year'].max())
        all_possible_years = range(min_year, max_year + 1)

        # Iterate over ALL possible years, even if no rasters exist for that year
        for year in all_possible_years:
            year_folder = str(year)
            raster_files = year_to_rasters.get(year_folder, [])

            if not raster_files:
                self.write_message(f"No storms intersecting {self.region} in {year_folder}; creating a zero-exposure raster.", OUTPUTS)

            if value == "cumulative_count":
                output_name = f"cumulative_count_{year_folder}.tif"
            elif value == "cumulative_windspeed":
                output_name = f"cumulative_windspeed_{year_folder}.tif"

            output_path = output_folder / output_name

            if not self.overwrite and output_path.exists():
                self.write_message(f"Skipping cumulative raster for year {year_folder} ({value}) - already exists at {output_path}.", OUTPUTS)
                continue

            # Default empty rasters
            output_raster = np.zeros((mask_height, mask_width), dtype=np.float32)

            for raster_path in raster_files:
                with rasterio.open(self._get_gdal_path(raster_path)) as src:
                    raster_data = src.read(1)
                    transform = src.transform
                    crs = src.crs

                    raster_data_resampled = np.full((mask_height, mask_width), np.nan, dtype=np.float32)

                    # Align the individual storm raster with the annual output grid.
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
                        output_raster += count_raster

                    elif value == "cumulative_windspeed":
                        output_raster += np.nan_to_num(raster_data, nan=0)

                self.write_message(f"Processed raster: {raster_path.name} for year {year_folder}", OUTPUTS)

            output_raster[~mask_valid] = np.nan

            output_raster, output_transform = self._clip_to_prediction_bounds(output_raster, mask_transform, mask_crs)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_raster(output_raster, output_path, *output_raster.shape, output_transform, mask_crs)
            self.write_message(f"Cumulative raster for year {year_folder} saved to {output_path}.", OUTPUTS)

        self.write_message(f"Cumulative raster generation complete for {value}.", OUTPUTS)

    def get_corner_points(self, quarter_circle: Polygon, quadrant: str) -> list[tuple[float, float]]:
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

    def polygons_to_raster(self, resolution: float = 100.0) -> None:
        """Convert hurricane polygons to rasters."""

        self.write_message("Converting trimmed polygons to rasters...", OUTPUTS)

        try:
            gdf = self._read_gpkg_layer("trimmed_polygons")
        except Exception as e:
            raise RuntimeError("Could not read the trimmed hurricane polygons") from e

        self.write_message("Projecting trimmed hurricane polygons to the target CRS...", OUTPUTS)
        gdf = gdf.to_crs(self.target_crs)

        # Repair the vector topology in case the extreme UTM distortion folded any polygons over themselves
        gdf['geometry'] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]

        self.write_message(f"Reading EcoRegions from {self.coast_boundary_path} to constrain raster bounding boxes...", OUTPUTS)
        eco_gdf = self._read_ecoregion()
        eco_minx, eco_miny, eco_maxx, eco_maxy = eco_gdf.total_bounds
        pred_minx, pred_miny, pred_maxx, pred_maxy = self._prediction_bounds(gdf.crs)

        # Group by 'area_date' ensures unique files per storm.
        grouped = gdf.groupby('area_date')

        for area_date, group in grouped:

            name = group.iloc[0]['name']
            year = group.iloc[0]['year']
            safe_name = str(name).strip().replace(" ", "")

            output_folder = self.raster_path / str(year)
            output_folder.mkdir(parents=True, exist_ok=True)
            raster_file = output_folder / f"{safe_name}_{area_date}.tif"

            # Prevent overwriting existing individual rasters
            if not self.overwrite and raster_file.exists():
                self.write_message(f"Skipping individual raster {safe_name}_{area_date}.tif ({year}) - already exists.", OUTPUTS)
                continue

            s_minx, s_miny, s_maxx, s_maxy = group.total_bounds

            # Constrain raster size to EcoRegion bounds to prevent massive memory/disk blowouts
            minx = max(s_minx, eco_minx)
            miny = max(s_miny, eco_miny)
            maxx = min(s_maxx, eco_maxx)
            maxy = min(s_maxy, eco_maxy)

            if minx >= maxx or miny >= maxy:
                self.write_message(f"Warning: Storm {name} - {year} is completely outside the target EcoRegions bounding box. Skipping rasterization.", OUTPUTS)
                continue

            if max(minx, pred_minx) >= min(maxx, pred_maxx) or max(miny, pred_miny) >= min(maxy, pred_maxy):
                self.write_message(f'Skipping storm {name} - {year}: outside prediction mask bounds.', OUTPUTS)
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

            raster_data, transform = self._clip_to_prediction_bounds(raster_data, transform, gdf.crs)
            self.save_raster(raster_data, raster_file, *raster_data.shape, transform, gdf.crs)

            self.write_message(f"Raster for {name} - {year} saved to {raster_file}.", OUTPUTS)

    def read_text_data(self, txt_path: UPath | str) -> list[list[str]]:
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

    def save_raster(self, data: np.ndarray, path_obj: UPath, height: int, width: int, transform: Affine, crs: rasterio.crs.CRS | str) -> None:
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

    def run(self) -> None:
        """Prepare shared hurricane data once, then process each configured region."""
        eco_regions = list(self.param_lookup['eco_regions'].value)
        if not eco_regions:
            return
        self._resolve_paths(str(eco_regions[0]))
        self.download_hurricane_data()
        if self.overwrite or not self.hurricane_data_path.exists():
            point_gdf = self.convert_text_to_gpkg()
            self.create_overlapping_buffers(point_gdf)
            self.clip_polygons()
        else:
            # Verify reusable inputs rather than silently proceeding with an incomplete file.
            self._read_gpkg_layer('atlantic_hurricane_points')
            self._read_gpkg_layer('trimmed_polygons')
            self.write_message(f"Reusing shared hurricane GeoPackage: {self.hurricane_data_path}", OUTPUTS)
        for eco_region in eco_regions:
            self._resolve_paths(str(eco_region))
            self.write_message(f"Processing hurricane layers for {eco_region}...", OUTPUTS)
            self.polygons_to_raster()
            self.generate_cumulative_rasters(self.count_raster_path, 'cumulative_count')
            self.generate_cumulative_rasters(self.cumulative_raster_path, 'cumulative_windspeed')
            for start_year, end_year in self.year_ranges:
                self.year_pair_rasters(start_year, end_year)
            self.write_message(f"Hurricane processing complete for {eco_region}.", OUTPUTS)
