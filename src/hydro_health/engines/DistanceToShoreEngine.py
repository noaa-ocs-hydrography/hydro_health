import os
import sys
import pathlib
import tempfile
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
import s3fs

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, Param


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


def _set_gdal_s3_options() -> None:
    """Configure GDAL /vsis3/ driver options for cloud S3 access."""
    rasterio.env.Env(
        AWS_NO_SIGN_REQUEST='NO',
        AWS_EC2_METADATA_DISABLED='FALSE',
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        VSI_CACHE='FALSE',
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES',
        GDAL_HTTP_MULTIPLEX='YES',
    )


class DistanceToShoreEngine(Engine):
    """
    Engine for generating clean prediction water masks, rasterizing shoreline vectors,
    and calculating distance-to-shore rasters using BlueTopo datasets locally or on AWS S3.
    """

    def __init__(self, param_lookup, pilot_mode=False):
        super().__init__()
        self.param_lookup = param_lookup
        self.pilot_mode = pilot_mode

    def _resolve_vector_inputs(self, base_outputs: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path | None]:
        """Locates the required shoreline and boundary shapefiles within the outputs directory."""
        shoreline_matches = list(base_outputs.rglob("*shoreline*.shp"))
        boundary_matches = list(base_outputs.rglob("*boundary*.shp"))

        if not shoreline_matches:
            raise FileNotFoundError(f"Could not locate a shoreline shapefile (*shoreline*.shp) in {base_outputs}")

        shoreline_path = shoreline_matches[0]
        prediction_boundary_path = boundary_matches[0] if boundary_matches else None

        print(f"- Located Shoreline Vector: {shoreline_path}")
        if prediction_boundary_path:
            print(f"- Located Boundary Vector: {prediction_boundary_path}")

        return shoreline_path, prediction_boundary_path

    def _get_local_vrts(self, outputs: str, output_prefix: str) -> list[pathlib.Path]:
        """Discovers local mosaic_elevation*.vrt files in the outputs directory hierarchy."""
        base_path = pathlib.Path(outputs)
        search_path = base_path / output_prefix if output_prefix else base_path
        
        vrt_files = list(search_path.glob("**/mosaic_elevation*.vrt"))
        print(f"Discovered {len(vrt_files)} local VRT(s) under: {search_path}")
        return vrt_files

    def _get_s3_vrts(self, outputs: str, output_prefix: str) -> list[str]:
        """Discovers S3 mosaic_elevation*.vrt files and formats them into /vsis3/ paths."""
        _set_gdal_s3_options()
        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        prefix_segment = f"{output_prefix}/" if output_prefix else ""
        s3_search_pattern = f"{bucket}/{prefix_segment}**/mosaic_elevation*.vrt"
        
        raw_vrt_paths = s3_files.glob(s3_search_pattern)
        vsi_vrt_paths = [f"/vsis3/{p}" for p in raw_vrt_paths]
        
        print(f"Discovered {len(vsi_vrt_paths)} S3 VRT(s) matching: s3://{s3_search_pattern}")
        return vsi_vrt_paths

    def build_water_mask(
        self, 
        vrt_paths: list[str | pathlib.Path],
        output_dir: pathlib.Path,
        prediction_boundary_path: str | pathlib.Path = None,
        output_prefix: str = "Clean_Prediction_Water_Mask",
    ) -> list[dict]:
        """Creates a clean binary water mask for each discovered BlueTopo VRT dataset."""
        results = []

        for bluetopo_path in vrt_paths:
            path_obj = pathlib.Path(str(bluetopo_path))
            vrt_stem = path_obj.stem
            suffix = vrt_stem.replace("mosaic_elevation", "")
            output_name = f"{output_prefix}{suffix}.tif" if suffix else f"{output_prefix}.tif"
            output_file = output_dir / output_name

            print(f"  - Building water mask from VRT: {path_obj.name} -> {output_name}")

            with rasterio.open(str(bluetopo_path)) as src:
                bathy_data = src.read(1)
                profile = src.profile.copy()
                src_crs = src.crs

                if src_crs and src_crs.is_geographic:
                    raise ValueError(f"Input bathymetry ({path_obj.name}) should be in a projected model CRS.")

                # Binary mask (1 for valid depth pixels, 255/NoData for land or NaNs)
                water_mask = np.where(np.isfinite(bathy_data) & (bathy_data != src.nodata), 1, 255).astype(np.uint8)

                if prediction_boundary_path and pathlib.Path(prediction_boundary_path).exists():
                    boundary_gdf = gpd.read_file(str(prediction_boundary_path))
                    if boundary_gdf.crs != src_crs:
                        boundary_gdf = boundary_gdf.to_crs(src_crs)

                    boundary_mask = rasterize(
                        shapes=boundary_gdf.geometry,
                        out_shape=src.shape,
                        transform=src.transform,
                        fill=0,
                        default_value=1,
                        dtype=np.uint8,
                    )
                    water_mask[boundary_mask == 0] = 255

                profile.update(
                    driver="GTiff",
                    dtype=rasterio.uint8,
                    count=1,
                    nodata=255,
                    compress="deflate",
                    tiled=True,
                )

                with rasterio.open(output_file, "w", **profile) as dst:
                    dst.write(water_mask, 1)

                valid_cells = int(np.sum(water_mask == 1))

            results.append({
                "water_mask": str(output_file),
                "reference_raster": str(bluetopo_path),
                "valid_water_cells": valid_cells,
            })

        return results

    def build_real_shoreline_raster(
        self,
        shoreline_path: str | pathlib.Path,
        bluetopo_path: str | pathlib.Path,
        output_dir: pathlib.Path,
        prediction_boundary_path: str | pathlib.Path = None,
        boundary_clip_buffer_m: float = 100.0,
        output_name: str = "Real_Shoreline_Raster.tif",
    ) -> dict:
        """Rasterizes shoreline vector geometries directly onto the BlueTopo reference grid."""
        output_file = output_dir / output_name

        with rasterio.open(str(bluetopo_path)) as ref:
            ref_profile = ref.profile.copy()
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_shape = ref.shape

        shoreline_gdf = gpd.read_file(str(shoreline_path))
        shoreline_gdf["geometry"] = shoreline_gdf.geometry.make_valid()

        if shoreline_gdf.crs != ref_crs:
            shoreline_gdf = shoreline_gdf.to_crs(ref_crs)

        shoreline_gdf["geometry"] = shoreline_gdf.geometry.apply(
            lambda geom: geom.boundary if geom.geom_type in ["Polygon", "MultiPolygon"] else geom
        )

        if prediction_boundary_path and pathlib.Path(prediction_boundary_path).exists():
            boundary_gdf = gpd.read_file(str(prediction_boundary_path))
            boundary_gdf["geometry"] = boundary_gdf.geometry.make_valid()

            if boundary_gdf.crs != shoreline_gdf.crs:
                boundary_gdf = boundary_gdf.to_crs(shoreline_gdf.crs)

            buffered_boundary = boundary_gdf.buffer(boundary_clip_buffer_m).unary_union
            shoreline_gdf = gpd.clip(shoreline_gdf, buffered_boundary)
            shoreline_gdf = shoreline_gdf[~shoreline_gdf.is_empty]

        if shoreline_gdf.empty:
            raise ValueError("No shoreline features remained after clipping.")

        shoreline_raster = rasterize(
            shapes=shoreline_gdf.geometry,
            out_shape=ref_shape,
            transform=ref_transform,
            fill=255,
            default_value=1,
            dtype=rasterio.uint8,
            all_touched=True,
        )

        shoreline_cells = int(np.sum(shoreline_raster == 1))
        if shoreline_cells == 0:
            raise ValueError("Rasterization produced zero shoreline cells.")

        ref_profile.update(
            driver="GTiff",
            dtype=rasterio.uint8,
            count=1,
            nodata=255,
            compress="deflate",
            tiled=True,
        )

        with rasterio.open(output_file, "w", **ref_profile) as dst:
            dst.write(shoreline_raster, 1)

        return {
            "shoreline_raster": str(output_file),
            "shoreline_cells": shoreline_cells,
        }

    def build_distance_from_real_shoreline(
        self,
        shoreline_raster_path: str | pathlib.Path,
        clean_water_mask_path: str | pathlib.Path,
        output_dir: pathlib.Path,
        output_name: str = "Distance_From_Real_Shore_m.tif",
    ) -> dict:
        """Calculates Euclidean distance to shoreline cells and masks out non-water areas."""
        output_file = output_dir / output_name

        with rasterio.open(str(shoreline_raster_path)) as shore_src, rasterio.open(str(clean_water_mask_path)) as water_src:
            shore_data = shore_src.read(1)
            water_data = water_src.read(1)
            profile = shore_src.profile.copy()
            pixel_size = abs(shore_src.transform.a)

            target_mask = shore_data != 1
            dist_in_pixels = distance_transform_edt(target_mask)
            dist_in_meters = (dist_in_pixels * pixel_size).astype(np.float32)

            dist_in_meters[water_data != 1] = -9999.0

            profile.update(
                driver="GTiff",
                dtype=rasterio.float32,
                count=1,
                nodata=-9999.0,
                compress="deflate",
                predictor=3,
                tiled=True,
            )

            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(dist_in_meters, 1)

            valid_distances = dist_in_meters[dist_in_meters != -9999.0]
            stats = {
                "min": float(np.min(valid_distances)) if valid_distances.size else None,
                "mean": float(np.mean(valid_distances)) if valid_distances.size else None,
                "max": float(np.max(valid_distances)) if valid_distances.size else None,
            }

        return {
            "distance_raster": str(output_file),
            "statistics": stats,
        }

    def process_single_vrt_pipeline(
        self,
        mask_info: dict,
        shoreline_path: pathlib.Path,
        prediction_boundary_path: pathlib.Path | None,
        output_dir: pathlib.Path,
        boundary_clip_buffer_m: float,
    ) -> dict:
        """Executes shoreline rasterization and distance calculation for a single VRT dataset."""
        ref_vrt = mask_info["reference_raster"]
        water_mask_path = mask_info["water_mask"]
        vrt_stem = pathlib.Path(ref_vrt).stem

        print(f"- Rasterizing Shoreline Vector to grid: {vrt_stem}...")
        shoreline = self.build_real_shoreline_raster(
            shoreline_path=shoreline_path,
            bluetopo_path=ref_vrt,
            output_dir=output_dir,
            prediction_boundary_path=prediction_boundary_path,
            boundary_clip_buffer_m=boundary_clip_buffer_m,
            output_name=f"Real_Shoreline_Raster_{vrt_stem}.tif",
        )

        print(f"- Calculating Euclidean Distance to Shoreline for {vrt_stem}...")
        distance = self.build_distance_from_real_shoreline(
            shoreline_raster_path=shoreline["shoreline_raster"],
            clean_water_mask_path=water_mask_path,
            output_dir=output_dir,
            output_name=f"Distance_From_Real_Shore_{vrt_stem}.tif",
        )

        print(f"Successfully generated distance raster: {distance['distance_raster']}")
        print(f"Distance stats: {distance['statistics']}")
        return distance

    def run(
        self,
        output_prefix: str = "",
        boundary_clip_buffer_m: float = 100.0,
    ) -> None:
        """Factory orchestrator executing the distance-to-shore pipeline across local or S3 environments."""
        outputs = self.param_lookup['output_directory'].valueAsText
        base_outputs = pathlib.Path(outputs)
        
        output_dir = base_outputs / "Helpers"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting Distance To Shore workflow (Environment: {self.param_lookup['env']})...")

        # Resolve VRT files dynamically based on execution environment
        if self.param_lookup['env'] in ['local', 'remote']:
            vrt_files = self._get_local_vrts(outputs, output_prefix)
        else:
            vrt_files = self._get_s3_vrts(outputs, output_prefix)

        if not vrt_files:
            print(" - Warning: No VRT files found to process. Exiting run.")
            return

        shoreline_path, prediction_boundary_path = self._resolve_vector_inputs(base_outputs)

        water_masks = self.build_water_mask(
            vrt_paths=vrt_files,
            output_dir=output_dir,
            prediction_boundary_path=prediction_boundary_path,
        )

        for mask_info in water_masks:
            self.process_single_vrt_pipeline(
                mask_info=mask_info,
                shoreline_path=shoreline_path,
                prediction_boundary_path=prediction_boundary_path,
                output_dir=output_dir,
                boundary_clip_buffer_m=boundary_clip_buffer_m,
            )


if __name__ == "__main__":
    param_lookup = {
        'env': Param('local'),  # Set to 'aws' or 'local'
        'output_directory': Param(OUTPUTS)
    }
    engine = DistanceToShoreEngine(param_lookup, pilot_mode=False)
    engine.run(output_prefix="")