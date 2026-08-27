import pathlib
import sys
import concurrent.futures
import geopandas as gpd
import numpy as np

from typing import List, Tuple
from shapely.geometry import box, Polygon

HH_MODEL = pathlib.Path(__file__).parents[2]
sys.path.append(str(HH_MODEL))

from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / "inputs"
OUTPUTS = pathlib.Path(__file__).parents[3] / "outputs"


class SubgriddingEngine:
    """Subdivides vector polygon tiles in a GeoPackage based on a reference tile size."""

    tile_id_col = "tile"

    @staticmethod
    def calculate_reference_subgrid_size(ref_tile_geom: Polygon,) -> Tuple[float, float]:
        """Calculates dx and dy (half width and height of reference tile)."""

        minx, miny, maxx, maxy = ref_tile_geom.bounds
        dx = (maxx - minx) / 2.0
        dy = (maxy - miny) / 2.0
        return dx, dy

    @staticmethod
    def split_single_tile(row: gpd.GeoSeries, dx: float, dy: float, tile_id_col: str) -> List[dict]:
        """Splits a single tile geometry bottom-to-top per column, starting from the west."""

        orig_geom = row.geometry
        orig_id = row[tile_id_col]
        minx, miny, maxx, maxy = orig_geom.bounds

        x_steps = np.arange(minx, maxx, dx)
        y_steps = np.arange(miny, maxy, dy)

        subgrid_records = []
        grid_counter = 1

        # West-to-East (Columns)
        for x in x_steps:
            # Bottom-to-Top (Rows within column)
            for y in y_steps:
                grid_cell = box(x, y, x + dx, y + dy)
                intersection = orig_geom.intersection(grid_cell)

                if not intersection.is_empty:
                    rec = row.to_dict()
                    rec["geometry"] = intersection
                    rec["subgrid_tile"] = f"{orig_id}_{grid_counter}"
                    subgrid_records.append(rec)
                    grid_counter += 1

        return subgrid_records

    def get_reference_dimensions(self, gdf: gpd.GeoDataFrame, reference_tile_id: str) -> Tuple[float, float]:
        """Extracts the reference geometry and computes subgrid dx and dy."""

        ref_matches = gdf[gdf[self.tile_id_col] == reference_tile_id]
        if ref_matches.empty:
            raise ValueError(
                f"Reference tile '{reference_tile_id}' not found in column '{self.tile_id_col}'."
            )

        ref_geom = ref_matches.geometry.iloc[0]
        return self.calculate_reference_subgrid_size(ref_geom)

    def _parallel_split_tiles(self, gdf: gpd.GeoDataFrame, dx: float, dy: float, workers: int) -> gpd.GeoDataFrame:
        """Executes parallel splitting across all input features."""

        print(f"Subdividing features using {workers} parallel workers...")
        records_list = []
        rows = [row for _, row in gdf.iterrows()]

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    self.split_single_tile, row, dx, dy, self.tile_id_col
                )
                for row in rows
            ]
            for future in concurrent.futures.as_completed(futures):
                records_list.extend(future.result())

        subgrid_gdf = gpd.GeoDataFrame(records_list, crs=gdf.crs)
        subgrid_gdf["geometry"] = subgrid_gdf.geometry.make_valid()
        return subgrid_gdf

    @staticmethod
    def _write_layer(subgrid_gdf: gpd.GeoDataFrame, gpkg_path: pathlib.Path, output_layer: str) -> None:
        """Writes the layer back to the GeoPackage using fiona with explicit layer targeting."""

        print(f"Writing layer '{output_layer}' to GeoPackage...")

        # Remove unnecessary layers
        cols_to_remove = ["GeoTIFF_Link", "RAT_Link", "GeoTIFF_SHA256_Checksum", "RAT_SHA256_Checksum"]
        subgrid_gdf = subgrid_gdf.drop(columns=[c for c in cols_to_remove if c in subgrid_gdf.columns])

        # Explicit layer-write without appending onto existing GDAL metadata pointers
        subgrid_gdf.to_file(
            str(gpkg_path),
            driver="GPKG",
            layer=output_layer,
            overwrite=True
        )

    def run(self, reference_tile_id: str=None, workers: int=6) -> gpd.GeoDataFrame:
        """Subdivides the target layer geometries and appends the result layer back to the GeoPackage."""

        master_grids = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
        master_grids_gpkg = pathlib.Path(master_grids)
        input_layer = get_config_item('SHARED', 'TILES')
        print(f"Reading layer '{input_layer}' from {master_grids_gpkg}...")
        gdf = gpd.read_file(master_grids_gpkg, layer=input_layer)

        reference_tile = reference_tile_id if reference_tile_id else get_config_item('BLUETOPO', 'REFERENCE_TILE')
        dx, dy = self.get_reference_dimensions(gdf, reference_tile)
        print(f"Reference dimensions calculated -> dx: {dx:.4f}, dy: {dy:.4f}")

        subgrid_gdf = self._parallel_split_tiles(gdf, dx, dy, workers)
        output_layer = 'named_subgrid_tiles'
        self._write_layer(subgrid_gdf, master_grids_gpkg, output_layer)

        print(
            f"Done. Processed {len(subgrid_gdf)} subgrid polygons into '{output_layer}'."
        )
        return subgrid_gdf


if __name__ == "__main__":
    print('starting')

    # TODO this writes out to the Master_Grids.gpkg
    # Need to dynamically create a temp layer or write to S3?
    engine = SubgriddingEngine()
    engine.run()
    print('done')