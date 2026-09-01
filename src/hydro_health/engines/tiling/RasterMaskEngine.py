import os
import pathlib
import shutil
import numpy as np
import geopandas as gpd
import rasterio

from pathlib import Path
from shapely.geometry import shape
from shapely.ops import unary_union
from upath import UPath
from rasterio.features import shapes
from osgeo import ogr, osr, gdal

os.environ["GDAL_MEM_ENABLE_OPEN"] = "YES"

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_approved_providers

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


class RasterMaskEngine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def raster_mask_to_parquet(self, ecoregion: str, output_prefix: str, raster_path: Path, process_type: str, outputs: str = None) -> gpd.GeoDataFrame:
        """Convert a raster mask to a GeoDataFrame using memory-safe block processing."""
        self.write_message(f"Creating {process_type} mask GeoDataFrame from: {raster_path}", outputs)

        geometries = []
        pilot_mode = getattr(self, 'pilot_mode', False)

        with rasterio.open(raster_path) as src:
            block_size = 4096

            for y in range(0, src.height, block_size):
                for x in range(0, src.width, block_size):
                    window = rasterio.windows.Window(
                        x, y,
                        min(block_size, src.width - x),
                        min(block_size, src.height - y)
                    )

                    mask_chunk = src.read(1, window=window, out_dtype='uint8')

                    if process_type == 'prediction':
                        valid_mask = mask_chunk == 1
                    elif process_type == 'training':
                        valid_mask = (mask_chunk == 1) if pilot_mode else (mask_chunk == 2)
                    else:
                        raise ValueError(f"Unknown process_type: {process_type}")

                    if not valid_mask.any():
                        continue

                    win_transform = src.window_transform(window)
                    shapes_gen = shapes(mask_chunk, mask=valid_mask, transform=win_transform)
                    chunk_geoms = [shape(geom) for geom, _ in shapes_gen]

                    if chunk_geoms:
                        geometries.append(unary_union(chunk_geoms))

            crs = src.crs

        self.write_message(f" -> Extracted {len(geometries)} unified geometries. Building GeoDataFrame...", outputs)

        gdf = gpd.GeoDataFrame({'geometry': geometries}, crs=crs)
        if hasattr(self, 'target_crs') and self.target_crs:
            gdf = gdf.to_crs(self.target_crs)

        gdf['geometry'] = gdf.geometry.make_valid().buffer(0)

        if process_type == 'prediction':
            sub_path = get_config_item('MASK', 'PREDICTION_MASK_PQ', pilot_mode=pilot_mode)
        else:
            sub_path = get_config_item('MASK', 'TRAINING_MASK_PQ', pilot_mode=pilot_mode)

        suffix = str(sub_path).lstrip('/')

        base_dir = pathlib.Path(self.param_lookup['output_directory'].valueAsText)
        if output_prefix:
            mask_path = base_dir / output_prefix / ecoregion / suffix
        else:
            mask_path = base_dir / ecoregion / suffix

        mask_path.parent.mkdir(parents=True, exist_ok=True)

        self.write_message(f"Saving {process_type} mask GeoDataFrame to: {mask_path}", outputs)
        gdf.to_parquet(str(mask_path))

        return gdf

    def create_subgrids(self, mask_gdf_path: Path, output_path: Path, process_type: str, outputs: str = None) -> None:
        """Create subgrids layer by intersecting Master_Grids tiles with mask geometries."""
        self.write_message(f"Preparing {process_type} sub-grids from: {mask_gdf_path}", outputs)

        mask_gdf_df = gpd.read_parquet(mask_gdf_path)

        # one-line fix for geopandas version issues with union_all
        combined_geometry = getattr(mask_gdf_df, "union_all", lambda: getattr(mask_gdf_df, "unary_union", getattr(mask_gdf_df.geometry, "unary_union", None)))()
        mask_gdf_df = gpd.GeoDataFrame(geometry=[combined_geometry], crs=mask_gdf_df.crs)

        grid_gpkg_path = INPUTS / get_config_item('MODEL', 'SUBGRIDS')
        sub_grids = gpd.read_file(str(grid_gpkg_path), layer='prediction_subgrid').to_crs(mask_gdf_df.crs)

        intersecting_sub_grids = gpd.sjoin(sub_grids, mask_gdf_df, how="inner", predicate='intersects')
        intersecting_sub_grids = intersecting_sub_grids.drop_duplicates(subset="geometry")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        intersecting_sub_grids.to_file(str(output_path), driver="GPKG")

        self.write_message(f"[SUCCESS] Successfully saved {process_type} subgrids to: {output_path}", outputs)

    def run(self, outputs: str, output_prefix: str) -> None:
        """Main execution flow using Dask for rasters followed by local parquet/subgrid creation."""
        base_dir = pathlib.Path(self.param_lookup['output_directory'].valueAsText)
        output_folder = base_dir / output_prefix if output_prefix else base_dir

        ecoregions = [d for d in output_folder.glob('ER_*') if d.is_dir()]
        self.setup_dask(self.param_lookup['env'])

        # 1. Run original raster generation tasks via Dask
        self.client.gather(self.client.map(_create_prediction_mask, [[er, self.param_lookup] for er in ecoregions]))
        training_params = [[er, outputs] for er in ecoregions]
        results = self.client.gather(
            self.client.map(_create_training_mask, training_params)
        )

        for r in results:
            print(r)

        self.close_dask()    

        # 2. Vector Post-Processing (Parquet & Subgrids)
        pilot_mode = getattr(self, 'pilot_mode', False)
        mask_sub = get_config_item('MASK', 'SUBFOLDER')
        subgrids_sub = get_config_item('MODEL', 'MODEL_SUBGRIDS')
        pred_suffix = str(get_config_item('MASK', 'PREDICTION_MASK_PQ', pilot_mode=pilot_mode)).lstrip('/')
        train_suffix = str(get_config_item('MASK', 'TRAINING_MASK_PQ', pilot_mode=pilot_mode)).lstrip('/')

        tasks = [
            ('prediction', pred_suffix),
            ('training', train_suffix)
        ]

        for er_dir in ecoregions:
            er = er_dir.name
            er_mask_dir = er_dir / mask_sub
            er_subgrids_dir = er_dir / subgrids_sub

            for mask_type, suffix in tasks:
                # FIX: Added underscores to match "prediction_mask_ER_xx.tif" and "training_mask_ER_xx.tif"
                tif_path = er_mask_dir / f"{mask_type}_mask_{er}.tif"
                
                if tif_path.exists():
                    self.raster_mask_to_parquet(er, output_prefix, tif_path, mask_type, outputs)
                    
                    mask_pq_path = er_mask_dir / suffix
                    subgrid_out_path = er_subgrids_dir / f"{mask_type}_intersecting_subgrids.gpkg"
                    
                    if mask_pq_path.exists():
                        self.create_subgrids(mask_pq_path, subgrid_out_path, mask_type, outputs)

def _create_prediction_mask(param_inputs: list) -> None:
    """Rasterize the Ecoregion boundary into a tiled, compressed GeoTIFF"""
    ecoregion_path, param_lookup = param_inputs

    gpkg = INPUTS / 'Master_Grids.gpkg'
    gpkg_ds = ogr.Open(str(gpkg))
    ecoregions_layer = gpkg_ds.GetLayerByName('Enhanced_EcoRegions_50m')

    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(32617)

    # Create an in-memory layer for the filtered ecoregion
    mem_driver = ogr.GetDriverByName('Memory')
    tmp_ds = mem_driver.CreateDataSource('mem_ds')
    tmp_layer = tmp_ds.CreateLayer('mask_poly', srs=output_srs, geom_type=ogr.wkbPolygon)

    ecoregions_layer.SetAttributeFilter(f"EcoRegion = '{ecoregion_path.stem}'")
    for feat in ecoregions_layer:
        geom = feat.GetGeometryRef()
        # Ensure transformation matches your get_transformation() logic
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(4326) # Source is usually WGS84
        transform = osr.CoordinateTransformation(target_srs, output_srs)
        geom.Transform(transform)

        new_feat = ogr.Feature(tmp_layer.GetLayerDefn())
        new_feat.SetGeometry(geom)
        tmp_layer.CreateFeature(new_feat)

    xmin, xmax, ymin, ymax = tmp_layer.GetExtent()
    pixel_size = 8
    cols = int((xmax - xmin) / pixel_size)
    rows = int((ymax - ymin) / pixel_size)

    mask_path = ecoregion_path / get_config_item('MASK', 'SUBFOLDER') / f"prediction_mask_{ecoregion_path.stem}.tif"
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    creation_options = [
        "COMPRESS=DEFLATE",
        "TILED=YES",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512",
        "SPARSE_OK=YES"
    ]

    target_ds = gdal.GetDriverByName("GTiff").Create(
        str(mask_path), cols, rows, 1, gdal.GDT_Byte, options=creation_options
    )
    target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
    target_ds.SetProjection(output_srs.ExportToWkt())

    # Burn the polygon into the raster
    gdal.RasterizeLayer(target_ds, [1], tmp_layer, burn_values=[1])
    target_ds.FlushCache()
    target_ds = None

def _create_training_mask(param_inputs: list) -> str:
    """Check actual raster data presence to upgrade prediction mask (1) to training mask (2)"""
    
    ecoregion_path, outputs = param_inputs
    mask_subfolder = ecoregion_path / get_config_item('MASK', 'SUBFOLDER')
    prediction_file = mask_subfolder / f'prediction_mask_{ecoregion_path.stem}.tif'
    training_file = mask_subfolder / f'training_mask_{ecoregion_path.stem}.tif'
    dc_vrt_folder = ecoregion_path / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'

    vrts = list(dc_vrt_folder.glob("mosaic_*.vrt"))
    if not vrts: return f"{ecoregion_path.stem}: No VRTs found."

    # Copy prediction to training to start
    shutil.copy(str(prediction_file), str(training_file))

    ds = gdal.Open(str(training_file), gdal.GA_Update)
    band = ds.GetRasterBand(1)
    geo_t = ds.GetGeoTransform()
    proj = ds.GetProjection()
    cols, rows = ds.RasterXSize, ds.RasterYSize

    # Block processing to keep memory footprint low
    block_size = 4096
    total_burns = 0

    engine = RasterMaskEngine(param_lookup={})
    approved_providers = [provider.lower() for provider in get_approved_providers(ecoregion_path.stem)]

    for y in range(0, rows, block_size):
        num_rows = min(block_size, rows - y)
        for x in range(0, cols, block_size):
            num_cols = min(block_size, cols - x)
            mask_chunk = band.ReadAsArray(x, y, num_cols, num_rows)

            # Skip reading VRTs if this block has no prediction pixels (value 1)
            if not np.any(mask_chunk == 1):
                continue

            presence_chunk = np.zeros((num_rows, num_cols), dtype=np.uint8)

            # Calculate world coordinate bounding box for this chunk: [minX, minY, maxX, maxY]
            chunk_min_x = geo_t[0] + x * geo_t[1]
            chunk_max_y = geo_t[3] + y * geo_t[5]  # geo_t[5] is negative pixel height
            chunk_max_x = chunk_min_x + num_cols * geo_t[1]
            chunk_min_y = chunk_max_y + num_rows * geo_t[5]

            bounds = [chunk_min_x, chunk_min_y, chunk_max_x, chunk_max_y]

            for vrt in vrts:
                vrt_provider = '_'.join(vrt.stem.split('_')[2:])
                if vrt_provider.lower() not in approved_providers:
                    engine.write_message(f'- skipping unapproved provider: {vrt_provider}', outputs)
                    continue

                # construct the MEM dataset with dstAlpha instead of loading
                warp_options = gdal.WarpOptions(
                    format='MEM',
                    outputBounds=bounds,
                    width=num_cols,
                    height=num_rows,
                    dstSRS=proj,
                    dstAlpha=True,
                    resampleAlg=gdal.GRA_NearestNeighbour
                )

                vrt_ds = gdal.Open(str(vrt))
                tmp_ds = gdal.Warp('', vrt_ds, options=warp_options)

                if tmp_ds is not None and tmp_ds.RasterCount >= 2:
                    alpha_chunk = tmp_ds.GetRasterBand(2).ReadAsArray()
                    presence_chunk |= (alpha_chunk > 0).astype(np.uint8)

                tmp_ds = None
                vrt_ds = None

            update_idx = (mask_chunk == 1) & (presence_chunk > 0)
            if np.any(update_idx):
                total_burns += int(np.sum(update_idx))
                mask_chunk[update_idx] = 2
                band.WriteArray(mask_chunk, x, y)

    band.FlushCache()
    ds.BuildOverviews("NONE", [])
    ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
    ds = None

    return f"{ecoregion_path.stem}: {total_burns} training pixels marked."