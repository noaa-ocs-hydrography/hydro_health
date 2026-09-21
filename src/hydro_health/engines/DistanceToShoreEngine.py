import pathlib
import sys
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt
from shapely.geometry import box
from upath import UPath

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, Param


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class DistanceToShoreEngine(Engine):
    """
    Engine for generating clean prediction water masks, rasterizing shoreline vectors,
    and calculating distance-to-shore rasters.
    """

    def __init__(self, param_lookup, pilot_mode=False):
        super().__init__()
        self.param_lookup = param_lookup
        self.pilot_mode = pilot_mode

    def build_clean_prediction_water_mask(
        self, 
        prediction_bathy_path: str,
        output_dir: pathlib.Path,
        prediction_boundary_path: str = None,
        output_name: str = "Clean_Prediction_Water_Mask.tif",
    ) -> dict:
        """
        Creates a clean binary water mask (1 = water/valid cell, 255/NoData = land/unavailable)
        from the prediction bathymetry raster and an optional boundary vector.
        """

        output_file = output_dir / output_name

        with rasterio.open(prediction_bathy_path) as src:
            bathy_data = src.read(1)
            profile = src.profile.copy()
            src_crs = src.crs

            # Projected CRS Check
            if src_crs and src_crs.is_geographic:
                raise ValueError("Prediction bathymetry should be in a projected model CRS.")

            # Create binary mask (1 for finite values, NoData for non-finite)
            water_mask = np.where(np.isfinite(bathy_data), 1, 255).astype(np.uint8)

            # Optional boundary clipping
            if prediction_boundary_path:
                boundary_gdf = gpd.read_file(prediction_boundary_path)
                if boundary_gdf.crs != src_crs:
                    boundary_gdf = boundary_gdf.to_crs(src_crs)

                # Rasterize boundary polygon to create a spatial filter mask
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
                dtype=rasterio.uint8,
                count=1,
                nodata=255,
                compress="deflate",
                tiled=True,
            )

            with rasterio.open(output_file, "w", **profile) as dst:
                dst.write(water_mask, 1)

            valid_cells = int(np.sum(water_mask == 1))

        return {
            "water_mask": str(output_file),
            "reference_raster": prediction_bathy_path,
            "valid_water_cells": valid_cells,
        }

    def build_real_shoreline_raster(
        self,
        shoreline_path: str,
        reference_raster_path: str,
        output_dir: pathlib.Path,
        prediction_boundary_path: str = None,
        boundary_clip_buffer_m: float = 100.0,
        output_name: str = "Real_Shoreline_Raster.tif",
    ) -> dict:
        """
        Rasterizes shoreline vector geometries directly onto the reference grid.
        Converts polygons to boundaries and clips using an optional buffered boundary.
        """

        output_file = output_dir / output_name

        with rasterio.open(reference_raster_path) as ref:
            ref_profile = ref.profile.copy()
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_shape = ref.shape

        shoreline_gdf = gpd.read_file(shoreline_path)
        shoreline_gdf["geometry"] = shoreline_gdf.geometry.make_valid()

        if shoreline_gdf.crs != ref_crs:
            shoreline_gdf = shoreline_gdf.to_crs(ref_crs)

        # Extract line boundaries from polygons if present
        shoreline_gdf["geometry"] = shoreline_gdf.geometry.apply(
            lambda geom: geom.boundary if geom.geom_type in ["Polygon", "MultiPolygon"] else geom
        )

        # Optional boundary clipping with buffer
        if prediction_boundary_path:
            boundary_gdf = gpd.read_file(prediction_boundary_path)
            boundary_gdf["geometry"] = boundary_gdf.geometry.make_valid()

            if boundary_gdf.crs != shoreline_gdf.crs:
                boundary_gdf = boundary_gdf.to_crs(shoreline_gdf.crs)

            buffered_boundary = boundary_gdf.buffer(boundary_clip_buffer_m).unary_union
            shoreline_gdf = gpd.clip(shoreline_gdf, buffered_boundary)
            shoreline_gdf = shoreline_gdf[~shoreline_gdf.is_empty]

        if shoreline_gdf.empty:
            raise ValueError("No shoreline features remained after clipping.")

        # Rasterize shoreline vector
        shoreline_raster = rasterize(
            shapes=shoreline_gdf.geometry,
            out_shape=ref_shape,
            transform=ref_transform,
            fill=255,
            default_value=1,
            dtype=rasterio.uint8,
            all_touched=True,  # Matches touches = TRUE in R terra
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
        shoreline_raster_path: str,
        clean_water_mask_path: str,
        output_dir: pathlib.Path,
        output_name: str = "Distance_From_Real_Shore_m.tif",
    ) -> dict:
        """
        Calculates Euclidean distance to the shoreline cells and masks out non-water areas.
        """

        output_file = output_dir / output_name

        with rasterio.open(shoreline_raster_path) as shore_src, rasterio.open(clean_water_mask_path) as water_src:
            shore_data = shore_src.read(1)
            water_data = water_src.read(1)
            profile = shore_src.profile.copy()
            pixel_size = abs(shore_src.transform.a)

            # Euclidean Distance Transform (0 where shoreline exists, calculates distance to nearest 0)
            target_mask = shore_data != 1
            dist_in_pixels = distance_transform_edt(target_mask)
            dist_in_meters = (dist_in_pixels * pixel_size).astype(np.float32)

            # Apply water mask constraint
            dist_in_meters[water_data != 1] = -9999.0

            profile.update(
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

    def run(
        self,
        outputs: str,
        prediction_bathy_path: str,
        reference_raster_path: str,
        shoreline_path: str,
        prediction_boundary_path: str = None,
        boundary_clip_buffer_m: float = 100.0,
    ) -> None:
        """Sequential workflow execution method for building distance to shore raster."""

        output_dir = outputs / "Helpers"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Starting Distance To Shore workflow...")

        print("- Building Clean Prediction Water Mask...")
        clean_mask = self.build_clean_prediction_water_mask(
            prediction_bathy_path=prediction_bathy_path,
            output_dir=output_dir,
            prediction_boundary_path=prediction_boundary_path,
        )

        print("- Rasterizing Shoreline Vector...")
        shoreline = self.build_real_shoreline_raster(
            shoreline_path=shoreline_path,
            reference_raster_path=reference_raster_path,
            output_dir=output_dir,
            prediction_boundary_path=prediction_boundary_path,
            boundary_clip_buffer_m=boundary_clip_buffer_m,
        )

        print("- Calculating Euclidean Distance to Shoreline...")
        distance = self.build_distance_from_real_shoreline(
            shoreline_raster_path=shoreline["shoreline_raster"],
            clean_water_mask_path=clean_mask["water_mask"],
            output_dir=output_dir,
        )

        print(f"Successfully generated distance raster: {distance['distance_raster']}")
        print(f"Distance stats: {distance['statistics']}")

if __name__ == "__main__":
    param_lookup = {
        'env': Param('aws'),
        'output_directory': Param(OUTPUTS)
    }
    engine = DistanceToShoreEngine(param_lookup, pilot_mode=False)
    engine.run(
        OUTPUTS,
        prediction_bathy_path = OUTPUTS / r"ER_3\model_variables\Prediction\processed\bt.bathy.tif",
        reference_raster_path = OUTPUTS / r"ER_3\model_variables\Prediction\processed\LOCAL_Pred_Start_Bathy_t_2004_2006_MOSAIC.vrt",
        shoreline_path = OUTPUTS / r"ER_3\model_variables\Prediction\processed\composite_shoreline_final.shp",
        prediction_boundary_path = OUTPUTS / r"ER_3\model_variables\Prediction\processed\pilot_model_base_boundary.shp",
        boundary_clip_buffer_m = 100.0,
    )