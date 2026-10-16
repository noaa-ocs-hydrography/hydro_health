import sys
import pathlib
import requests
import rasterio
import fsspec
import tempfile
import sqlite3
import pandas as pd
import geopandas as gpd
import numpy as np

from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from rasterio.features import rasterize
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from upath import UPath
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class CreateSedimentLayerEngine(Engine):
    """Class to hold the logic for processing the Sediment layer"""

    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.is_aws = param_lookup.get('env', 'local') in ['remote', 'aws']
        self.output_prefix = output_prefix
        self.sediment_types = ['Gravel', 'Sand', 'Mud', 'Clay']
        self.sediment_data = None

    def _resolve_paths(self, region: str) -> None:
        """Resolve paths dynamically for aws or local environments and the given eco region."""

        self.outputs_dir = OUTPUTS / self.output_prefix / region if self.output_prefix else OUTPUTS / region
        self.write_message(f'CreateSedimentLayerEngine resolved outputs_dir for region {region}: {self.outputs_dir}', OUTPUTS)

        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        s3_dir_base = f's3://{bucket}/{region}'

        data_path = get_config_item('SEDIMENT', 'SUBFOLDER')
        local_data_base = OUTPUTS / self.output_prefix if self.output_prefix else OUTPUTS
        self.data_path = UPath(f's3://{bucket}/{data_path}') if self.is_aws else UPath(local_data_base / data_path)
        self.csv_path = str(self.data_path / 'US9_ONE.csv')
        self.data_url = get_config_item('SEDIMENT', 'DATA_URL')

        gpkg_path = get_config_item('SEDIMENT', 'GPKG_PATH')
        self.gpkg_path = UPath(f'{s3_dir_base}/{gpkg_path}') if self.is_aws else UPath(self.outputs_dir / gpkg_path)

        raster_path = get_config_item('SEDIMENT', 'RASTER_PATH')
        self.raster_path = UPath(f'{s3_dir_base}/{raster_path}') if self.is_aws else UPath(self.outputs_dir / raster_path)

        prediction_mask_path = get_config_item('MASK', 'MASK_PRED_PATH')
        self.prediction_mask_path = UPath(f'{s3_dir_base}/{prediction_mask_path}') if self.is_aws else UPath(self.outputs_dir / prediction_mask_path)

        master_grids_path = str(get_config_item('SHARED', 'MASTER_GRIDS'))
        self.master_grids_path = UPath(master_grids_path) if '://' in master_grids_path else UPath(INPUTS / master_grids_path)
        self.filesystem = fsspec.filesystem('s3') if self.is_aws else fsspec.filesystem('file', auto_mkdir=True)
        if not self.is_aws:
            self.gpkg_path.parent.mkdir(parents=True, exist_ok=True)
            self.raster_path.mkdir(parents=True, exist_ok=True)
            self.data_path.mkdir(parents=True, exist_ok=True)

    def _modify_gpkg(self, gdf: gpd.GeoDataFrame, layer_name: str, mode: str = 'w') -> None:
        """Write layers to the local working GeoPackage."""

        gdf.to_file(
            self.working_gpkg_path, layer=layer_name, driver='GPKG',
            engine='pyogrio', mode=mode,
        )

    def _validate_gpkg(self) -> None:
        """Check the completed local database before rasterization and upload."""

        database_uri = self.working_gpkg_path.resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(database_uri, uri=True) as connection:
            results = connection.execute('PRAGMA quick_check').fetchall()
        if results != [('ok',)]:
            details = '; '.join(str(row[0]) for row in results)
            raise RuntimeError(f'Sediment GeoPackage failed SQLite quick_check: {details}')
        self.write_message('Sediment GeoPackage passed SQLite quick_check.', OUTPUTS)

    def add_sed_size_column(self) -> None:
        """
        Adds column for the sediment size in mm
        Renames Grainsze column to Size_phi to clarify size is in phi units
        """

        self.sediment_data['sed_size'] = 2 ** -(self.sediment_data['Grainsze'])
        self.sediment_data = self.sediment_data.rename(columns={'Grainsze': 'Size_phi'})

    def add_sediment_mapping_column(self) -> None:
        """Adds column for the integer value of each sediment time for rasterization"""

        sediment_mapping = {
            'Gravel': 1,
            'Sand': 2,
            'Mud': 3,
            'Clay': 4
        }
        self.sediment_data['sed_int'] = self.sediment_data['sed_type'].map(sediment_mapping)

    def convert_polys_to_raster(self, field_name: str, resolution: float = 100) -> None:
        """Crop to prediction bounds, retaining the sediment CRS and resolution."""

        if resolution <= 0:
            raise ValueError('Raster resolution must be positive')
        self.write_message(f'Creating raster for {field_name} at {resolution} m resolution...', OUTPUTS)
        polygons = gpd.read_file(str(self.working_gpkg_path), layer='sediment_polygons', engine='pyogrio')
        polygons = polygons.to_crs(self.target_crs)
        nodata_val = -9999.0

        with rasterio.open(str(self.prediction_mask_path)) as prediction_mask:
            if prediction_mask.crs is None:
                raise ValueError(f'Prediction mask must have a CRS: {self.prediction_mask_path}')
            left, bottom, right, top = transform_bounds(
                prediction_mask.crs, self.target_crs, *prediction_mask.bounds,
                densify_pts=101,
            )
        if not np.all(np.isfinite([left, bottom, right, top])) or right <= left or top <= bottom:
            raise ValueError('Prediction mask bounds could not be transformed to the sediment CRS')
        # Round outward by at most one sediment pixel; keep the requested resolution.
        width = int(np.ceil((right - left) / resolution))
        height = int(np.ceil((top - bottom) / resolution))
        transform = from_origin(left, top, resolution, resolution)

        self.write_message(f'Reading Enhanced_EcoRegions from {self.master_grids_path}', OUTPUTS)
        eco_gdf = gpd.read_file(str(self.master_grids_path), layer='Enhanced_EcoRegions', engine='pyogrio')
        eco_gdf = eco_gdf.to_crs(self.target_crs)
        eco_shapes = [
            (geometry, 1) for geometry in eco_gdf.geometry
            if geometry is not None and not geometry.is_empty
        ]
        valid_area = rasterize(
            eco_shapes, out_shape=(height, width), transform=transform,
            fill=0, dtype='uint8', all_touched=False,
        ) if eco_shapes else np.zeros((height, width), dtype=np.uint8)

        shapes = [
            (geometry, value)
            for geometry, value in zip(polygons.geometry, polygons[field_name])
            if geometry is not None and not geometry.is_empty and pd.notna(value)
        ]
        rasterized = rasterize(
            shapes=shapes, out_shape=(height, width), transform=transform,
            fill=nodata_val, dtype='float32', all_touched=False,
        ) if shapes else np.full((height, width), nodata_val, dtype=np.float32)
        rasterized[valid_area == 0] = nodata_val
        filenames = {
            'sed_size': 'grain_size_layer.tif',
            'sed_type': 'prim_sed_layer.tif',
            'sand_mud_mask': 'sand_mud_mask.tif',
        }
        filename = filenames.get(field_name, f'{field_name}_raster_{resolution}m.tif')
        destination = f'{self.raster_path}/{filename}'
        self.filesystem.makedirs(str(self.raster_path), exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = pathlib.Path(temporary_dir) / filename
            with rasterio.open(
                temporary_path, 'w', driver='GTiff', height=height, width=width,
                count=1, dtype='float32', crs=self.target_crs, transform=transform,
                nodata=nodata_val, compress='lzw',
            ) as dst:
                dst.write(rasterized, 1)
            self.filesystem.put_file(str(temporary_path), destination)
        self.write_message(f'Raster saved to {destination}', OUTPUTS)

    def correct_sed_type(self, row: gpd.GeoSeries) -> str:
        """Corrects primary sediment type if sediment percerntages do not match grain size"""

        # These size classification ranges are based on the Udden-Wentworth grain size chart
        if 0 < row['sed_size'] < 0.0039:
            return 'Clay'
        elif 0.0039 <= row['sed_size'] < 0.0625:
            return 'Mud'
        elif 0.0625 <= row['sed_size'] < 2:
            return 'Sand'
        elif 2 <= row['sed_size']:
            return 'Gravel'
        else:
            return row['sed_type']

    def create_point_layer(self) -> None:
        """Creates a point layer from the sediment GeoDataFrame"""
        gdf = gpd.GeoDataFrame(
            self.sediment_data,
            geometry=gpd.points_from_xy(self.sediment_data['Longitude'], self.sediment_data['Latitude'])
        )
        gdf.set_crs(crs="EPSG:4326", inplace=True)
        gdf_reprojected = gdf.to_crs(self.target_crs)

        self._modify_gpkg(gdf_reprojected, 'sediment_points', mode='w')
        self.write_message('Created sediment point layer.', OUTPUTS)

    def determine_sed_types(self) -> None:
        """Determines the primary and secondarysediment types at each point based on type percentage"""

        self.write_message('Calculating primary and secondary sediment types', OUTPUTS)
        prim_values = []
        sec_values = []

        for _, row in self.sediment_data.iterrows():
            sediments = row[self.sediment_types]
            sorted_sediments = sediments.sort_values(ascending=False).index
            primary_sed = sorted_sediments[0]
            secondary_sed = sorted_sediments[1]
            prim_values.append(primary_sed)
            sec_values.append(secondary_sed)
        self.sediment_data['sed_type'] = prim_values
        self.sediment_data['sec_sed'] = sec_values

        self.sediment_data['sed_type'] = self.sediment_data.apply(self.correct_sed_type, axis=1)

    def download_sediment_data(self) -> None:
        """Downloads the USGS sediment dataset"""

        csv_path = self.csv_path

        # fsspec works with local and s3 paths
        fs = self.filesystem
        if not fs.exists(csv_path):
            self.write_message(f"Downloading sediment data to {csv_path}...", OUTPUTS)
            try:
                url = self.data_url
                self.filesystem.makedirs(str(self.data_path), exist_ok=True)
                with requests.get(url, stream=True) as r:
                    r.raise_for_status()

                    with self.filesystem.open(csv_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                self.write_message("Sediment data downloaded successfully.", OUTPUTS)
            except Exception as e:
                self.write_message(f'Error downloading sediment CSV: {e}', OUTPUTS)
                sys.exit(1)
        else:
            self.write_message("File already exists. Skipping download.", OUTPUTS)

    def read_sediment_data(self) -> None:
        """Reads and stores the data from the USGS sediment dataset CSV"""

        csv_columns = ['Latitude', 'Longitude', 'Gravel', 'Sand', 'Mud', 'Clay', 'Grainsze']
        sediment_data_path = self.csv_path
        with self.filesystem.open(sediment_data_path, 'rb') as source:
            self.sediment_data = pd.read_csv(source, usecols=csv_columns)

        self.write_message('Filtering out rows with missing data', OUTPUTS)
        self.write_message(f' - Rows before: {self.sediment_data.shape[0]}', OUTPUTS)
        self.sediment_data = self.sediment_data[~((self.sediment_data[self.sediment_types] == -99) | (self.sediment_data[self.sediment_types] == 0)).all(axis=1)]
        self.sediment_data = self.sediment_data[(self.sediment_data['Grainsze'] != -99)].reset_index(drop=True)
        self.write_message(f' - Rows after: {self.sediment_data.shape[0]}', OUTPUTS)

    def transform_points_to_polygons(self) -> None:
        """Polygonize the sediment points and append as a new layer"""
        self.write_message("Transforming sediment points to polygons...", OUTPUTS)

        gdf = gpd.read_file(str(self.working_gpkg_path), layer='sediment_points', engine='pyogrio')

        coordinates_df = gdf.geometry.apply(lambda geom: geom.centroid.coords[0]).apply(pd.Series)
        coordinates_df.columns = ['Longitude', 'Latitude']
        vor = Voronoi(coordinates_df[['Longitude', 'Latitude']].values)
        polygons = []
        for point_idx, region_idx in enumerate(vor.point_region):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0: continue

            sed_int_val = gdf['sed_int'].iloc[point_idx]

            polygons.append({
                'geometry': Polygon([vor.vertices[i] for i in region]),
                'sed_type': sed_int_val,
                'sed_size': gdf['sed_size'].iloc[point_idx],
                'sand_mud_mask': 1 if sed_int_val in [2, 3] else None
            })

        gdf_voronoi = gpd.GeoDataFrame(polygons, crs=self.target_crs)

        self._modify_gpkg(gdf_voronoi, 'sediment_polygons', mode='a')
        self.write_message("Appended sediment polygons layer.", OUTPUTS)

    def run(self) -> None:
        """Entrypoint for processing the Sediment layer"""

        for eco_region in self.param_lookup['eco_regions'].value:
            self._resolve_paths(eco_region)
            self.write_message(f'Processing sediment for {eco_region}', OUTPUTS)
            self.download_sediment_data()
            self.read_sediment_data()
            self.add_sed_size_column()
            self.determine_sed_types()
            self.add_sediment_mapping_column()
            # Use one fresh local database for all reads and writes in this region.
            with tempfile.TemporaryDirectory() as temporary_dir:
                self.working_gpkg_path = pathlib.Path(temporary_dir) / 'sediment.gpkg'
                self.create_point_layer()
                self.transform_points_to_polygons()
                self._validate_gpkg()
                self.convert_polys_to_raster('sed_type')
                self.convert_polys_to_raster('sed_size')
                self.convert_polys_to_raster('sand_mud_mask')
                self.filesystem.put_file(str(self.working_gpkg_path), str(self.gpkg_path))
                self.write_message(f'Sediment GeoPackage saved to {self.gpkg_path}', OUTPUTS)
