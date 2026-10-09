import os
import sys
import pathlib
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
import traceback
import ctypes  # Added for glibc kernel-level memory trimming

from rasterio.features import rasterize
from scipy.spatial import cKDTree
from shapely.geometry import box
import s3fs
import gc
import shutil
import re  # Added for CRS string parsing
from distributed import as_completed  # Added to stream tasks and free memory dynamically

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
    """Configure GDAL /vsis3/ driver options for cloud S3 access."""

    rasterio.env.Env(
        AWS_NO_SIGN_REQUEST='NO',
        AWS_EC2_METADATA_DISABLED='FALSE',
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        VSI_CACHE='FALSE',
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES',
        GDAL_HTTP_MULTIPLEX='YES',
        GDAL_CACHEMAX='128',  # Cap GDAL internal C block cache to 128MB per worker
    )


def _process_tile(param_inputs: list) -> dict:
    """
    Parallel worker function to process a single BlueTopo tile.
    Defined at the module level so Dask can safely pickle and distribute it.
    """

    (bluetopo_path, ecoregion, ecoregion_gdf, 
     temp_path, param_lookup, output_prefix) = param_inputs

    _set_gdal_s3_options()
    engine = DistanceToShoreEngine(param_lookup)

    try:
        bluetopo_path = pathlib.Path(bluetopo_path)
        temp_path = pathlib.Path(temp_path)
        
        tile_id = bluetopo_path.parents[0].name
        raster_stem = bluetopo_path.stem

        tile_dir = temp_path / tile_id
        tile_dir.mkdir(parents=True, exist_ok=True)

        water_mask, valid_cells = engine.build_ocean_mask(bluetopo_path, ecoregion_gdf, temp_path)

        if valid_cells == 0:
            shutil.rmtree(tile_dir, ignore_errors=True)
            return {"status": "skipped", "file": str(bluetopo_path), "stats": None}

        print(f"  - Loading precomputed shoreline vertices for: {raster_stem}...")
        # Direct .npy array load
        shoreline_vertices = engine.get_shoreline_vertices(bluetopo_path, ecoregion)

        print(f"  - Calculating Vector-to-Point Distance to Shoreline for {raster_stem}...")
        distance_raster_path, stats = engine.build_distance_from_shoreline(
            shoreline_vertices,
            water_mask,
            tile_dir,
            raster_stem
        )
        print(f"  - Successfully generated distance raster: {stats}")

        # Free vertex memory immediately after calculation
        del shoreline_vertices
        gc.collect()

        if param_lookup['env'] == 'aws':
            # TODO should we upload water mask
            engine.upload_file_to_s3(distance_raster_path, ecoregion, tile_id, output_prefix)
            shutil.rmtree(tile_dir, ignore_errors=True)

        return {"status": "success", "file": str(bluetopo_path), "stats": stats}

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error processing {bluetopo_path}:\n{error_trace}")
        return {"status": "error", "file": str(bluetopo_path), "error": error_trace}
    finally:
        gc.collect()
        _trim_memory()  # Release unmanaged glibc memory arenas to OS


class DistanceToShoreEngine(Engine):
    """
    Engine for generating clean prediction water masks, extracting shoreline vertices,
    and calculating vector-to-point distance-to-shore rasters.
    """

    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def get_shoreline_vertices(
        self,
        bluetopo_path: pathlib.Path,
        ecoregion: str
    ) -> np.ndarray:
        """Determines the raster's CRS and loads the matching precomputed .npy vertex array."""
        
        with rasterio.open(str(bluetopo_path)) as ref:
            crs = ref.crs

        # Replicate safe CRS naming logic
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
            # Using the output path structure you defined in prep_shorelines.py
            s3_path = f"{bucket}/{ecoregion}/support_data/{npy_filename}"
            
            with s3.open(s3_path, 'rb') as f:
                return np.load(f)
        else:
            # Need to download from S3 if missing
            npy_path = INPUTS / 'support_data' / npy_filename
            if not npy_path.exists():
                raise FileNotFoundError(f"Precomputed vertices not found: {npy_path}")
            
            return np.load(npy_path)

    def build_distance_from_shoreline(
        self,
        shoreline_vertices: np.ndarray,
        water_mask: pathlib.Path,
        tile_folder: pathlib.Path,
        bluetopo_filename: str
    ) -> tuple:
        """Calculates distance from every valid water cell to the nearest shoreline vertex using cKDTree."""

        with rasterio.open(str(water_mask)) as water_src:
            water_data = water_src.read(1)
            crs = water_src.crs
            transform = water_src.transform
            height, width = water_src.height, water_src.width

        # Initialize the output array fully with NoData
        dist_in_meters = np.full(water_data.shape, -9999.0, dtype=np.float32)

        if shoreline_vertices.size > 0:
            tree = cKDTree(shoreline_vertices)

            # Process raster in 500-row windows to limit array coordinate memory footprint
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

                # Query Euclidean distances for row chunk
                distances, _ = tree.query(water_points)
                dist_in_meters[actual_rows, sub_cols] = distances.astype(np.float32)

                del sub_water, sub_rows, sub_cols, actual_rows, xs, ys, water_points, distances

            del tree
            gc.collect()

        del water_data
        gc.collect()

        # Clean profile dictionary to avoid carrying over incompatible GDAL tags
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

        distance_raster_path = tile_folder / f"{bluetopo_filename}_distance_from_shore.tif"
        distance_raster_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.write_message(f'- writing distance raster to: {distance_raster_path}', str(OUTPUTS))
        with rasterio.open(distance_raster_path, "w", **profile) as dst:
            dst.write(dist_in_meters, 1)

        valid_distances = dist_in_meters[dist_in_meters != -9999.0]
        stats = {
            "min": float(np.min(valid_distances)) if valid_distances.size else None,
            "mean": float(np.mean(valid_distances)) if valid_distances.size else None,
            "max": float(np.max(valid_distances)) if valid_distances.size else None,
        }

        del dist_in_meters, valid_distances
        gc.collect()
        _trim_memory()

        return distance_raster_path, stats

    def build_ocean_mask(
        self, 
        bluetopo_path: str | pathlib.Path,
        current_ecoregion_gdf: gpd.GeoDataFrame,
        temp_path: str | pathlib.Path
    ) -> tuple:
        """Creates a clean binary water mask for a single BlueTopo dataset."""

        bluetopo_path = pathlib.Path(bluetopo_path)
        tile_id = bluetopo_path.parents[0].name
        
        tile_output_path = pathlib.Path(temp_path) / tile_id
        tile_output_path.mkdir(parents=True, exist_ok=True)
        
        output_name = f"{bluetopo_path.stem}_ocean_mask.tif"
        output_file = tile_output_path / output_name

        print(f"  - Creating water mask: {bluetopo_path.name} -> {output_name}")

        with rasterio.open(str(bluetopo_path)) as src:
            bathy_data = src.read(1)
            profile = src.profile.copy()
            src_crs = src.crs

            if src_crs and src_crs.is_geographic:
                raise ValueError(f"- Bathymetry ({bluetopo_path.name}) must be projected, not geographic!")

            # Only select water pixels(negative)
            water_mask = np.where(
                np.isfinite(bathy_data) & (bathy_data != src.nodata) & (bathy_data < 0), 
                1, 
                255
            ).astype(np.uint8)

            del bathy_data  # Free raw bathymetry immediately

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
            del boundary_mask  # Free boundary mask immediately

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
    
    def get_temp_path_and_bluetopo_paths(self, output_prefix: str, ecoregion: str) -> tuple:    
        """Set up target folder for local output or AWS intermediate file creation"""

        starting_path = f'{output_prefix}/{ecoregion}' if output_prefix else ecoregion
        subfolder = get_config_item('BLUETOPO', 'SUBFOLDER')

        if self.param_lookup['env'] == 'aws':
            bluetopo_paths = self.get_s3_bluetopo_files(ecoregion)
            temp_path = pathlib.Path.home() / 'local_tmp_dir' / starting_path / subfolder / 'BlueTopo'
        else:
            bluetopo_paths = self.get_local_bluetopo_files(ecoregion, output_prefix)
            temp_path = OUTPUTS / starting_path / subfolder / 'BlueTopo'
            
        return temp_path, bluetopo_paths

    def get_local_bluetopo_files(self, ecoregion: str, output_prefix: str = "") -> list[pathlib.Path]:
        """Search for local BlueTopo geotiffs"""
        
        if output_prefix:
            bluetopo_folder = OUTPUTS / output_prefix / ecoregion / get_config_item('BLUETOPO', 'SUBFOLDER') / 'BlueTopo'
        else:
            bluetopo_folder = OUTPUTS / ecoregion / get_config_item('BLUETOPO', 'SUBFOLDER') / "BlueTopo"
        
        raster_files = []
        raster_files.extend([raster for raster in list(bluetopo_folder.rglob("*[0-9].tiff")) if 'ISS' not in raster.name])

        print(f"- Discovered {len(raster_files)} local BlueTopo file(s)")
        return raster_files

    def get_s3_bluetopo_files(self, ecoregion: str = "*") -> list[str]:
        """Search for all BlueTopo geotiffs for current ecoregion"""

        _set_gdal_s3_options()
        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

        s3_search_pattern = f"{bucket}/{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo/**/*[0-9].tiff"
        raw_paths = s3_files.glob(s3_search_pattern)
        geotiffs = [
            gt for gt in raw_paths 
            if not any(pattern in gt.lower() for pattern in [
                '_unc', 
                '_slope', 
                '_iss', 
                '_survey_end_date', 
                'decay'])
        ]
        
        vsi_bluetopo_paths = [
            f"/vsis3/{p}" for p in geotiffs 
            if p.lower().endswith(('.tif', '.tiff'))
        ]

        print(f"- Discovered {len(vsi_bluetopo_paths)} S3 BlueTopo file(s) matching: s3://{s3_search_pattern}")
        return vsi_bluetopo_paths
        
    def upload_file_to_s3(self, distance_raster_path: pathlib.Path, ecoregion: str, tile_id: str, output_prefix: str=False) -> None:
        """Upload distance raster file to S3"""

        s3 = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

        ecoregion_path = f"{output_prefix}/{ecoregion}" if output_prefix else f"{ecoregion}"
        s3_path = f"{bucket}/{ecoregion_path}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo/{tile_id}"
        
        print(f"  - Uploading {tile_id} to s3://{s3_path}")
        s3.put(str(distance_raster_path), s3_path, recursive=True)

    def run(self, output_prefix: str='', **kwargs) -> None:
        """Main function for processing Distance-to-Shore across local or S3 environments using Dask."""

        print(f"Starting Distance To Shore engine")

        self.setup_dask(self.param_lookup['env'], n_workers=3, threads_per_worker=1, memory_limit="10GB")

        # Master geopackage is still needed once to isolate the ecoregion boundary for the water mask.
        master_gpkg = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
        ecoregions = self.param_lookup['eco_regions'].value
        
        for ecoregion in ecoregions:
            temp_path, bluetopo_paths = self.get_temp_path_and_bluetopo_paths(output_prefix, ecoregion)

            if not bluetopo_paths:
                print(f" - No BlueTopo tiles found for {ecoregion}")
                continue

            enhanced_ecoregions = gpd.read_file(str(master_gpkg), layer='Enhanced_EcoRegions_50m')
            ecoregion_gdf = enhanced_ecoregions[enhanced_ecoregions['EcoRegion'] == ecoregion]

            param_inputs = []
            for bluetopo_path in bluetopo_paths:
                param_inputs.append([
                    bluetopo_path,
                    ecoregion,
                    ecoregion_gdf,  # Direct parameter passing avoids scattered handle loss if workers restart
                    str(temp_path),
                    self.param_lookup,
                    output_prefix
                ])

            print(f" - Submitting {len(param_inputs)} tiles to Dask cluster...")
            
            future_tiles = self.client.map(_process_tile, param_inputs)

            # Process futures as completed and release references to prevent holding cluster RAM
            results = []
            for future in as_completed(future_tiles):
                res = future.result()
                results.append(res)
                self.client.cancel(future)
            
            success_count = sum(1 for r in results if r.get('status') == 'success')
            print(f" - Completed {success_count}/{len(results)} tiles successfully for {ecoregion}.")

        self.close_dask()


if __name__ == "__main__":
    param_lookup = {
        'env': Param('local'),  # Set to 'aws' or 'local'
        'output_directory': Param(OUTPUTS),
        'eco_regions': Param(['ER_3'])
    }
    engine = DistanceToShoreEngine(param_lookup)
    engine.run(ecoregion="*")