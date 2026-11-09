import os
import sys
import pathlib
import numpy as np
import geopandas as gpd
import rasterio
import traceback
import ctypes

from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely.geometry import box
import s3fs
import gc
import shutil
import re
from distributed import as_completed

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, Param

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


def _trim_memory() -> None:
    """Force Linux glibc allocator to release unmanaged C-heap memory back to OS kernel."""

    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _set_gdal_s3_options() -> None:
    """Configure GDAL /vsis3/ driver options for cloud S3 access via process environment variables."""

    os.environ['AWS_NO_SIGN_REQUEST'] = 'NO'
    os.environ['AWS_EC2_METADATA_DISABLED'] = 'FALSE'
    os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
    os.environ['VSI_CACHE'] = 'FALSE'
    os.environ['GDAL_HTTP_MERGE_CONSECUTIVE_RANGES'] = 'YES'
    os.environ['GDAL_HTTP_MULTIPLEX'] = 'YES'
    os.environ['GDAL_CACHEMAX'] = '128'


def _process_tile(param_inputs: list) -> dict:
    """
    Parallel worker function to process a single raster geotiff tile.
    Defined at the module level so Dask can safely pickle and distribute it.
    """

    (geotiff_string, data_type, ecoregion, ecoregion_gdf, 
     temp_path, param_lookup, output_prefix) = param_inputs

    _set_gdal_s3_options()
    engine = DistanceToShoreEngine(param_lookup, data_type)

    water_mask = None
    distance_raster_path = None

    try:
        geotiff_path = pathlib.Path(geotiff_string)
        temp_path = pathlib.Path(temp_path)
        raster_stem = geotiff_path.stem

        output_subfolder = engine.get_output_subfolder(geotiff_path)

        tile_dir = temp_path / output_subfolder
        tile_dir.mkdir(parents=True, exist_ok=True)

        # Generate ocean mask directly inside shared tile_dir
        water_mask, valid_cells = engine.build_ocean_mask(geotiff_path, ecoregion_gdf, tile_dir)

        # Skip if pure land / no valid water
        if valid_cells == 0:
            pathlib.Path(water_mask).unlink(missing_ok=True)
            return {"status": "skipped", "file": str(geotiff_path)}

        print(f"  - Loading precomputed shoreline vertices for: {raster_stem}...")
        shoreline_vertices = engine.get_shoreline_vertices(geotiff_path, ecoregion)

        print(f"  - Calculating Vector-to-Point Distance to Shoreline for {raster_stem}...")
        distance_raster_path = engine.build_distance_from_shoreline(
            shoreline_vertices,
            water_mask,
            tile_dir,
            raster_stem
        )

        del shoreline_vertices
        gc.collect()

        # Clean up the intermediate water mask as soon as the distance raster is generated
        pathlib.Path(water_mask).unlink(missing_ok=True)
        water_mask = None

        if param_lookup['env'] == 'aws':
            engine.upload_file_to_s3(distance_raster_path, ecoregion, str(output_subfolder), output_prefix)
            # Remove the local distance raster only after successful upload to S3
            pathlib.Path(distance_raster_path).unlink(missing_ok=True)
            distance_raster_path = None

        return {"status": "success", "file": str(geotiff_path)}

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing {geotiff_path}:\n{error_trace}")

        # Clean up leftover files on failure
        if water_mask:
            pathlib.Path(water_mask).unlink(missing_ok=True)
        if param_lookup['env'] == 'aws' and distance_raster_path:
            pathlib.Path(distance_raster_path).unlink(missing_ok=True)

        return {"status": "error", "file": str(geotiff_path), "error": error_trace}
    finally:
        gc.collect()
        _trim_memory()


class DistanceToShoreEngine(Engine):
    """
    Engine for generating clean prediction water masks, extracting shoreline vertices,
    and calculating vector-to-point distance-to-shore rasters.
    """

    def __init__(self, param_lookup: dict, data_type: str):
        super().__init__()
        self.param_lookup = param_lookup
        self.data_type = data_type

    def get_output_subfolder(self, raster_path: str | pathlib.Path) -> pathlib.PurePosixPath:
        """
        Determines the output subfolder relative to the dataset root based on self.data_type.
        - BlueTopo:     {tile_id}
        - DigitalCoast: {survey_id}/dem/{tile_id}/distance_to_shore
        """

        p = pathlib.Path(raster_path)
        parent_dir = p.parent

        if self.data_type == 'BlueTopo':
            return pathlib.PurePosixPath(parent_dir.name)
        else:
            idx = parent_dir.parts.index(self.data_type) + 1
            relative_parts = parent_dir.parts[idx:]
            return pathlib.PurePosixPath(*relative_parts) / "distance_to_shore"

    def get_shoreline_vertices(
        self,
        raster_path: pathlib.Path,
        ecoregion: str
    ) -> np.ndarray:
        """Determines the raster's CRS and loads the matching precomputed .npy vertex array."""

        with rasterio.open(str(raster_path)) as ref:
            crs = ref.crs

        epsg_code = crs.to_epsg()
        if not epsg_code:
            try:
                epsg_code = crs.sub_crs_list[0].to_epsg()
            except (AttributeError, IndexError):
                crs_str = getattr(crs, 'name', str(crs))
                epsg_code = re.sub(r'[^a-zA-Z0-9]', '_', crs_str)
                epsg_code = re.sub(r'_+', '_', epsg_code).strip('_')[:50]

        safe_crs_name = f"EPSG_{epsg_code}"
        npy_filename = f"{ecoregion}_shoreline_{safe_crs_name}.npy"

        if self.param_lookup['env'] == 'aws':
            s3 = s3fs.S3FileSystem()
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            s3_path = f"{bucket}/{ecoregion}/support_data/{npy_filename}"

            with s3.open(s3_path, 'rb') as f:
                return np.load(f)
        else:
            npy_path = INPUTS / 'support_data' / npy_filename
            if not npy_path.exists():
                raise FileNotFoundError(f"Precomputed vertices not found: {npy_path}")

            return np.load(npy_path)

    def build_distance_from_shoreline(
        self,
        shoreline_vertices: np.ndarray,
        water_mask: pathlib.Path,
        tile_folder: pathlib.Path,
        raster_filename: str
    ) -> pathlib.Path:
        """Calculates distance from every valid water cell to the nearest shoreline vertex using cKDTree."""

        with rasterio.open(str(water_mask)) as water_src:
            water_data = water_src.read(1)
            crs = water_src.crs
            transform = water_src.transform
            height, width = water_src.height, water_src.width

        dist_in_meters = np.full(water_data.shape, -9999.0, dtype=np.float32)

        if shoreline_vertices.size > 0:
            # Crop vertices to tile extent + 50km buffer to prevent memory bloat
            min_x, min_y, max_x, max_y = rasterio.transform.array_bounds(height, width, transform)
            search_buffer = 50_000
            in_bounds = (
                (shoreline_vertices[:, 0] >= min_x - search_buffer) &
                (shoreline_vertices[:, 0] <= max_x + search_buffer) &
                (shoreline_vertices[:, 1] >= min_y - search_buffer) &
                (shoreline_vertices[:, 1] <= max_y + search_buffer)
            )
            local_vertices = shoreline_vertices[in_bounds]
            target_vertices = local_vertices if local_vertices.size > 0 else shoreline_vertices

            tree = cKDTree(target_vertices)
            num_rows, num_cols = water_data.shape
            step = 500

            for r_start in range(0, num_rows, step):
                r_end = min(r_start + step, num_rows)
                sub_water = water_data[r_start:r_end, :]

                sub_rows, sub_cols = np.where(sub_water == 1)
                if sub_rows.size == 0:
                    continue

                actual_rows = sub_rows + r_start
                xs, ys = rasterio.transform.xy(transform, actual_rows, sub_cols, offset='center')
                water_points = np.column_stack((xs, ys)).astype(np.float32)

                distances, _ = tree.query(water_points)
                dist_in_meters[actual_rows, sub_cols] = distances.astype(np.float32)

                del sub_water, sub_rows, sub_cols, actual_rows, xs, ys, water_points, distances

            del tree
            gc.collect()

        del water_data
        gc.collect()

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": crs,
            "transform": transform,
            "nodata": -9999.0,
            "compress": "deflate",
            "tiled": True
        }

        distance_raster_path = tile_folder / f"{raster_filename}_distance_from_shore.tif"
        distance_raster_path.parent.mkdir(parents=True, exist_ok=True)

        self.write_message(f'- writing distance raster to: {distance_raster_path}', str(OUTPUTS))
        with rasterio.open(distance_raster_path, "w", **profile) as dst:
            dst.write(dist_in_meters, 1)

        del dist_in_meters
        gc.collect()
        _trim_memory()

        return distance_raster_path

    def build_ocean_mask(
        self, 
        raster_path: str | pathlib.Path, 
        current_ecoregion_gdf: gpd.GeoDataFrame,
        output_dir: str | pathlib.Path
    ) -> tuple:
        """Creates a clean binary water mask written directly into output_dir without extra subfolders."""

        raster_path = pathlib.Path(raster_path)
        tile_output_path = pathlib.Path(output_dir)
        tile_output_path.mkdir(parents=True, exist_ok=True)

        output_name = f"{raster_path.stem}_ocean_mask.tif"
        output_file = tile_output_path / output_name

        print(f"  - Creating water mask: {raster_path.name} -> {output_name}")

        with rasterio.open(str(raster_path)) as src:
            bathy_data = src.read(1)
            profile = src.profile.copy()
            src_crs = src.crs

            if src_crs and src_crs.is_geographic:
                raise ValueError(f"- Bathymetry ({raster_path.name}) must be projected, not geographic!")

            # Only select water pixels (negative)
            water_mask = np.where(
                np.isfinite(bathy_data) & (bathy_data != src.nodata) & (bathy_data < 0), 
                1, 
                255
            ).astype(np.uint8)

            del bathy_data

            if current_ecoregion_gdf.crs != src_crs:
                eco_proj = current_ecoregion_gdf.to_crs(src_crs)
            else:
                eco_proj = current_ecoregion_gdf

            boundary_mask = rasterize(
                shapes=eco_proj.geometry,
                out_shape=src.shape,
                transform=src.transform,
                fill=0,
                default_value=1,
                dtype=np.uint8,
            )
            water_mask[boundary_mask == 0] = 255
            del boundary_mask

            profile.update(
                driver="GTiff",
                dtype=rasterio.uint8,
                count=1,
                nodata=255,
                compress="deflate",
                tiled=True,
            )
            self.write_message(f'- writing ocean mask raster to: {output_file}', str(OUTPUTS))
            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(water_mask, 1)

            valid_cells = int(np.sum(water_mask == 1))
            del water_mask
            gc.collect()
            _trim_memory()

        return str(output_file), valid_cells

    def filter_unprocessed_tiles(
        self,
        geotiff_paths: list,
        ecoregion: str,
        output_prefix: str,
        temp_path: pathlib.Path
    ) -> list:
        """Filters out tiles whose distance-to-shore rasters already exist in S3 or locally."""

        if not geotiff_paths:
            return []

        subfolder_key = get_config_item(self.data_type.upper(), 'SUBFOLDER')
        ecoregion_path = f"{output_prefix}/{ecoregion}" if output_prefix else f"{ecoregion}"

        if self.param_lookup['env'] == 'aws':
            s3 = s3fs.S3FileSystem()
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            s3_search_prefix = f"{bucket}/{ecoregion_path}/{subfolder_key}/{self.data_type}/**/*_distance_from_shore.tif"

            print(f" - Checking existing outputs in S3: s3://{s3_search_prefix}...")
            existing_outputs = set(s3.glob(s3_search_prefix))

            unprocessed = []
            for gp in geotiff_paths:
                p = pathlib.Path(gp)
                output_subfolder = self.get_output_subfolder(p)
                expected_s3_key = (
                    f"{bucket}/{ecoregion_path}/{subfolder_key}/"
                    f"{self.data_type}/{output_subfolder}/{p.stem}_distance_from_shore.tif"
                )
                if expected_s3_key not in existing_outputs:
                    unprocessed.append(gp)
        else:
            print(f" - Checking existing outputs locally...")
            unprocessed = []
            for gp in geotiff_paths:
                p = pathlib.Path(gp)
                output_subfolder = self.get_output_subfolder(p)
                expected_local_file = temp_path / output_subfolder / f"{p.stem}_distance_from_shore.tif"
                if not expected_local_file.exists():
                    unprocessed.append(gp)

        skipped_count = len(geotiff_paths) - len(unprocessed)
        print(f" - Found {skipped_count} already completed tile(s). {len(unprocessed)} remaining to process.")
        return unprocessed

    def get_temp_path_and_geotiff_paths(self, output_prefix: str, ecoregion: str) -> tuple:    
        """Set up target folder for local output or AWS intermediate file creation"""

        starting_path = f'{output_prefix}/{ecoregion}' if output_prefix else ecoregion
        subfolder = get_config_item(self.data_type.upper(), 'SUBFOLDER')

        if self.param_lookup['env'] == 'aws':
            geotiff_paths = self.get_s3_geotiff_files(ecoregion)
            temp_path = pathlib.Path.home() / 'local_tmp_dir' / starting_path / subfolder / self.data_type
        else:
            geotiff_paths = self.get_local_geotiff_files(ecoregion, output_prefix)
            temp_path = OUTPUTS / starting_path / subfolder / self.data_type

        return temp_path, geotiff_paths

    def get_local_geotiff_files(self, ecoregion: str, output_prefix: str = "") -> list[pathlib.Path]:
        """Search for local data type geotiffs"""

        if output_prefix:
            data_type_folder = OUTPUTS / output_prefix / ecoregion / get_config_item(
                self.data_type.upper(), 'SUBFOLDER'
            ) / self.data_type
        else:
            data_type_folder = OUTPUTS / ecoregion / get_config_item(self.data_type.upper(), 'SUBFOLDER') / self.data_type

        geotiff_syntax = "*[0-9].tiff" if self.data_type == 'BlueTopo' else "*.tif"
        raster_files = [raster for raster in list(data_type_folder.rglob(geotiff_syntax)) if 'ISS' not in raster.name]
        raster_files = [raster for raster in raster_files if 'original' not in str(raster)]

        print(f"- Discovered {len(raster_files)} local {self.data_type} file(s)")
        return raster_files

    def get_s3_geotiff_files(self, ecoregion: str = "*") -> list[str]:
        """Search for all geotiffs for current data type and ecoregion"""

        _set_gdal_s3_options()
        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

        geotiff_syntax = f"{self.data_type}/**/*[0-9].tiff" if self.data_type == 'BlueTopo' else f"{self.data_type}/**/*.tif"
        s3_search_pattern = f"{bucket}/{ecoregion}/{get_config_item(self.data_type.upper(), 'SUBFOLDER')}/{geotiff_syntax}"

        raw_paths = s3_files.glob(s3_search_pattern)

        if self.data_type == 'BlueTopo':
            geotiffs = [
                gt for gt in raw_paths 
                if not any(pattern in gt.lower() for pattern in [
                    '_unc', '_slope', '_iss', '_survey_end_date', 'decay'])
            ]
        else:
            geotiffs = [raw_path for raw_path in raw_paths if 'original' not in str(raw_path)]

        vsi_geotiff_paths = [
            f"/vsis3/{p}" for p in geotiffs 
            if p.lower().endswith(('.tif', '.tiff'))
        ]

        print(f"- Discovered {len(vsi_geotiff_paths)} S3 {self.data_type} file(s) matching: s3://{s3_search_pattern}")
        return vsi_geotiff_paths

    def upload_file_to_s3(
        self, 
        distance_raster_path: pathlib.Path, 
        ecoregion: str, 
        output_subfolder: str, 
        output_prefix: str = ""
    ) -> None:
        """Upload distance raster file to S3 targeting the exact destination key."""

        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

        ecoregion_path = f"{output_prefix}/{ecoregion}" if output_prefix else f"{ecoregion}"
        subfolder_key = get_config_item(self.data_type.upper(), 'SUBFOLDER')
        s3_key = f"{bucket}/{ecoregion_path}/{subfolder_key}/{self.data_type}/{output_subfolder}/{distance_raster_path.name}"

        print(f"  - Uploading {distance_raster_path.name} to s3://{s3_key}")
        s3.put(str(distance_raster_path), s3_key)

    def run(self, output_prefix: str = '') -> None:
        """Main function for processing Distance-to-Shore across local or S3 environments using Dask."""

        print(f"Starting Distance To Shore engine")

        self.setup_dask(self.param_lookup['env'], n_workers=3, threads_per_worker=1, memory_limit="10GB")

        master_gpkg = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
        ecoregions = self.param_lookup['eco_regions'].value

        for ecoregion in ecoregions:
            temp_path, geotiff_paths = self.get_temp_path_and_geotiff_paths(output_prefix, ecoregion)

            if not geotiff_paths:
                print(f" - No {self.data_type} tiles found for {ecoregion}")
                continue

            # Skip tiles that are already completed
            geotiff_paths = self.filter_unprocessed_tiles(geotiff_paths, ecoregion, output_prefix, temp_path)
            if not geotiff_paths:
                print(f" - All tiles already completed for {ecoregion}!")
                continue

            enhanced_ecoregions = gpd.read_file(str(master_gpkg), layer='Enhanced_EcoRegions_50m')
            ecoregion_gdf = enhanced_ecoregions[enhanced_ecoregions['EcoRegion'] == ecoregion]

            param_inputs = []
            for geotiff_path in geotiff_paths:
                param_inputs.append([
                    geotiff_path,
                    self.data_type,
                    ecoregion,
                    ecoregion_gdf,
                    str(temp_path),
                    self.param_lookup,
                    output_prefix
                ])

            print(f" - Submitting {len(param_inputs)} tiles to Dask cluster...")

            future_tiles = self.client.map(_process_tile, param_inputs)

            results = []
            for future in as_completed(future_tiles):
                res = future.result()
                results.append(res)
                self.client.cancel(future)

            success_count = sum(1 for r in results if r.get('status') == 'success')
            print(f" - Completed {success_count}/{len(results)} tiles successfully for {ecoregion}.")
            
            beginning_prefix = f"{output_prefix}/{ecoregion}" if output_prefix else ecoregion
            manifest_path = pathlib.Path(beginning_prefix) / get_config_item(self.data_type.upper(), 'SUBFOLDER') / self.data_type
            self.write_run_manifest(manifest_path, {'tiles': len(param_inputs)})

        self.close_dask()


if __name__ == "__main__":
    param_lookup = {
        'env': Param('local'),  # 'aws' or 'local'
        'output_directory': Param(OUTPUTS),
        'eco_regions': Param(['ER_3'])
    }
    engine = DistanceToShoreEngine(param_lookup, data_type='DigitalCoast')
    engine.run()