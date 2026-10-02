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
    """Configure GDAL /vsis3/ driver options for cloud S3 access. 
    Hehehe, GDAL loves devouring cloud rasters! Nom nom nom!
    """
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

    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def _resolve_vector_inputs(self, base_outputs: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path | None]:
        """Locates the required shoreline and boundary shapefiles within the outputs directory."""

        shoreline_matches = list(base_outputs.rglob("*shoreline*.shp"))
        boundary_matches = list(base_outputs.rglob("*boundary*.shp"))

        if not shoreline_matches:
            raise FileNotFoundError(f"Gizmo hid the shoreline shapefile (*shoreline*.shp) in {base_outputs}!")

        shoreline_path = shoreline_matches[0]
        prediction_boundary_path = boundary_matches[0] if boundary_matches else None

        print(f"- Located Shoreline Vector: {shoreline_path}")
        if prediction_boundary_path:
            print(f"- Located Boundary Vector: {prediction_boundary_path}")

        return shoreline_path, prediction_boundary_path

    def _get_local_bluetopo_files(self, outputs: str, ecoregion: str = "*") -> list[pathlib.Path]:
        """
        Sniffs out local BlueTopo files inside:
        OUTPUTS/{ecoregion}/model_variables/pre_processed/BlueTopo/{tile_folder}
        Looks for VRTs or GeoTIFFs (*.vrt, *.tif, *.tiff).
        """
        base_path = pathlib.Path(outputs)
        
        # Constructs the precise target search path
        target_dir = base_path / ecoregion / "model_variables" / "pre_processed" / "BlueTopo"
        
        # Recursively search tile_folder subdirectories for BlueTopo raster files
        raster_files = []
        for ext in ("*.vrt", "*.tif", "*.tiff"):
            raster_files.extend(list(target_dir.glob(f"**/{ext}")))

        print(f"*Grins* Discovered {len(raster_files)} local BlueTopo file(s) under: {target_dir}")
        return raster_files

    def _get_s3_bluetopo_files(self, outputs: str, ecoregion: str = "*") -> list[str]:
        """
        Discovers S3 BlueTopo files in:
        s3://{bucket}/{ecoregion}/model_variables/pre_processed/BlueTopo/{tile_folder}
        Formats them into GDAL's favorite snack: /vsis3/ paths!
        """
        _set_gdal_s3_options()
        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

        # Clean trailing/leading slashes from outputs if bucket prefix is passed
        s3_search_pattern = f"{bucket}/{ecoregion}/model_variables/pre_processed/BlueTopo/**/*"
        
        raw_paths = s3_files.glob(s3_search_pattern)
        
        # Filter down to rasters and slice into GDAL's /vsis3/ format
        vsi_bluetopo_paths = [
            f"/vsis3/{p}" for p in raw_paths 
            if p.lower().endswith(('.vrt', '.tif', '.tiff'))
        ]

        print(f"*Cackles* Discovered {len(vsi_bluetopo_paths)} S3 BlueTopo file(s) matching: s3://{s3_search_pattern}")
        return vsi_bluetopo_paths

    def build_water_mask(
        self, 
        bluetopo_paths: list[str | pathlib.Path],
        output_dir: pathlib.Path,
        prediction_boundary_path: str | pathlib.Path = None,
        output_prefix: str = "Clean_Prediction_Water_Mask",
    ) -> list[dict]:
        """Creates a clean binary water mask for each discovered BlueTopo dataset."""

        results = []

        for bluetopo_path in bluetopo_paths:
            path_obj = pathlib.Path(str(bluetopo_path))
            stem = path_obj.stem
            
            output_name = f"{output_prefix}_{stem}.tif"
            output_file = output_dir / output_name

            print(f"  - Chewing up raster for water mask: {path_obj.name} -> {output_name}")

            with rasterio.open(str(bluetopo_path)) as src:
                bathy_data = src.read(1)
                profile = src.profile.copy()
                src_crs = src.crs

                if src_crs and src_crs.is_geographic:
                    raise ValueError(f"Aah! Bright light! Bathymetry ({path_obj.name}) must be projected, not geographic!")

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
            raise ValueError("No shoreline features remained after clipping! Stripe stole them all!")

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
            raise ValueError("Rasterization produced zero shoreline cells. GDAL demands a sacrifice!")

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

    def process_single_bluetopo_pipeline(
        self,
        mask_info: dict,
        shoreline_path: pathlib.Path,
        prediction_boundary_path: pathlib.Path | None,
        output_dir: pathlib.Path,
        boundary_clip_buffer_m: float,
    ) -> dict:
        """Executes shoreline rasterization and distance calculation for a single BlueTopo dataset."""

        ref_raster = mask_info["reference_raster"]
        water_mask_path = mask_info["water_mask"]
        raster_stem = pathlib.Path(ref_raster).stem

        print(f"- Rasterizing Shoreline Vector to grid: {raster_stem}...")
        shoreline = self.build_real_shoreline_raster(
            shoreline_path=shoreline_path,
            bluetopo_path=ref_raster,
            output_dir=output_dir,
            prediction_boundary_path=prediction_boundary_path,
            boundary_clip_buffer_m=boundary_clip_buffer_m,
            output_name=f"Real_Shoreline_Raster_{raster_stem}.tif",
        )

        print(f"- Calculating Euclidean Distance to Shoreline for {raster_stem}...")
        distance = self.build_distance_from_real_shoreline(
            shoreline_raster_path=shoreline["shoreline_raster"],
            clean_water_mask_path=water_mask_path,
            output_dir=output_dir,
            output_name=f"Distance_From_Real_Shore_{raster_stem}.tif",
        )

        print(f"- Successfully generated distance raster: {distance['distance_raster']}")
        print(f"Distance stats: {distance['statistics']}")
        return distance

    def run(self, boundary_clip_buffer_m: float = 100.0) -> None:
        """Main function for processing Distance-to-Shore across local or S3 environments."""

        outputs = self.param_lookup['output_directory'].valueAsText
        base_outputs = pathlib.Path(outputs)
        
        output_dir = base_outputs / "Helpers"
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Starting Distance To Shore workflow (Environment: {self.param_lookup['env']})...")

        ecoregions = self.param_lookup['eco_regions'].value
        for ecoregion in ecoregions:
            if self.param_lookup['env'] == 'aws':
                bluetopo_files = self._get_s3_bluetopo_files(outputs, ecoregion=ecoregion)
            else:
                bluetopo_files = self._get_local_bluetopo_files(outputs, ecoregion=ecoregion)

            if not bluetopo_files:
                print(" - No BlueTopo tiles found")
                return

            shoreline_path, prediction_boundary_path = self._resolve_vector_inputs(base_outputs)

            water_masks = self.build_water_mask(
                bluetopo_paths=bluetopo_files,
                output_dir=output_dir,
                prediction_boundary_path=prediction_boundary_path,
            )

            for mask_info in water_masks:
                self.process_single_bluetopo_pipeline(
                    mask_info=mask_info,
                    shoreline_path=shoreline_path,
                    prediction_boundary_path=prediction_boundary_path,
                    output_dir=output_dir,
                    boundary_clip_buffer_m=boundary_clip_buffer_m,
                )


if __name__ == "__main__":
    param_lookup = {
        'env': Param('local'),  # Set to 'aws' or 'local'
        'output_directory': Param(OUTPUTS),
        'eco_regions': Param(['ER_3'])
    }
    engine = DistanceToShoreEngine(param_lookup)
    # Pass a specific ecoregion folder name if you don't want to scan all ('*')
    engine.run(ecoregion="*")