
import json
import pathlib
import shutil
import pandas as pd
import geopandas as gpd
import yaml

import numpy as np

from osgeo import ogr, osr, gdal

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item



INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'



def _create_prediction_mask(param_inputs: list[list]) -> None:
    """Create prediction mask for current ecoregion"""

    ecoregion, param_lookup = param_inputs
    engine = RasterMaskEngine(param_lookup)

    # Project EcoRegions_50m to UTM
    gpkg = INPUTS / 'Master_Grids.gpkg'
    gpkg_ds = ogr.Open(str(gpkg))
    ecoregions_50m = gpkg_ds.GetLayerByName('EcoRegions_50m')
    in_memory_driver = ogr.GetDriverByName('Memory')

    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(32617)
    output_ds = in_memory_driver.CreateDataSource('output_ecoregion')
    output_layer = output_ds.CreateLayer(f'ecoregions', geom_type=ogr.wkbPolygon, srs=output_srs)

    input_layer_definition = ecoregions_50m.GetLayerDefn()
    for i in range(input_layer_definition.GetFieldCount()):
        fieldDefn = input_layer_definition.GetFieldDefn(i)
        output_layer.CreateField(fieldDefn)

    output_layer_definition = output_layer.GetLayerDefn()
    ecoregions_50m.ResetReading()
    for feature in ecoregions_50m:
        ecoregion_id = json.loads(feature.ExportToJson())['properties']['EcoRegion']
        if ecoregion_id == ecoregion.stem:
            geom = feature.GetGeometryRef()
            geom.Transform(engine.get_transformation())
            output_feature = ogr.Feature(output_layer_definition)
            output_feature.SetGeometry(geom)

            for i in range(input_layer_definition.GetFieldCount()):
                output_feature.SetField(input_layer_definition.GetFieldDefn(i).GetNameRef(), feature.GetField(i))

            output_layer.CreateFeature(output_feature)
            output_feature = None
    
    xmin, xmax, ymin, ymax = output_layer.GetExtent()
    pixel_size = 8
    x_res = int((xmax - xmin) / pixel_size)
    y_res = int((ymax - ymin) / pixel_size)

    # Use the ecoregion path to store the file permanently for now
    prediction_mask_name = f"prediction_mask_{ecoregion.stem}.tif"
    mask_path = ecoregion / get_config_item('MASK', 'SUBFOLDER') / prediction_mask_name
    mask_path.parents[0].mkdir(parents=True, exist_ok=True)

    # Updated creation options for a sparse, tiny file
    creation_options = [
        "COMPRESS=LZW", 
        "TILED=YES", 
        "SPARSE_OK=YES", 
        "NUM_THREADS=ALL_CPUS",
        "INTERLEAVE=BAND"
    ]

    target_ds = gdal.GetDriverByName("GTiff").Create(
        str(mask_path), x_res, y_res, 1, gdal.GDT_Byte,
        options=creation_options
    )
    target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
    target_ds.SetProjection(output_srs.ExportToWkt())
    target_ds.GetRasterBand(1).SetNoDataValue(0)

    gdal.RasterizeLayer(target_ds, [1], output_layer, burn_values=[1])
    target_ds = None # Flush to disk
    gpkg_ds = None


def _create_training_mask(ecoregion: pathlib.Path) -> str:
    """Create training mask from valid raster data in the VRT files"""

    gdal.UseExceptions()
    
    mask_subfolder = (ecoregion / get_config_item('MASK', 'SUBFOLDER')).resolve()
    prediction_file = mask_subfolder / f'prediction_mask_{ecoregion.stem}.tif'
    training_file = mask_subfolder / f'training_mask_{ecoregion.stem}.tif'
    digital_coast_folder = (ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast').resolve()

    provider_vrts = [str(f.absolute()) for f in digital_coast_folder.glob("mosaic_*.vrt")]

    if not provider_vrts:
        return f"{ecoregion.stem}: Skip - No VRTs found."
    if not prediction_file.exists():
        return f"{ecoregion.stem}: Error - Prediction mask missing."

    try:
        # Clone the prediction mask
        if training_file.exists():
            training_file.unlink()
        shutil.copy(str(prediction_file), str(training_file))
        
        train_ds = gdal.Open(str(training_file), gdal.GA_Update)
        train_band = train_ds.GetRasterBand(1)
        
        geo_t = train_ds.GetGeoTransform()
        proj = train_ds.GetProjection()
        cols, rows = train_ds.RasterXSize, train_ds.RasterYSize

        # Find actual data presence across all VRTs
        combined_presence = np.zeros((rows, cols), dtype=np.uint8)

        for vrt_path in provider_vrts:
            src_ds = gdal.Open(vrt_path)
            src_nodata = src_ds.GetRasterBand(1).GetNoDataValue()
            src_ds = None

            # Warp this specific VRT to match our mask's grid, but force an Alpha band
            mem_driver = gdal.GetDriverByName('MEM')
            tmp_ds = mem_driver.Create('', cols, rows, 2, gdal.GDT_Byte)
            tmp_ds.SetGeoTransform(geo_t)
            tmp_ds.SetProjection(proj)

            warp_options = gdal.WarpOptions(
                format='MEM',
                dstSRS=proj,
                dstAlpha=True,  # Band 2 becomes our high-res binary mask
                resampleAlg=gdal.GRA_NearestNeighbour,
                srcNodata=src_nodata if src_nodata is not None else -9999
            )
            
            gdal.Warp(tmp_ds, vrt_path, options=warp_options)
            vrt_presence = tmp_ds.GetRasterBand(2).ReadAsArray()
            combined_presence |= (vrt_presence > 0).astype(np.uint8)
            tmp_ds = None

        block_size = 4096 
        burn_count_total = 0

        for y in range(0, rows, block_size):
            num_rows = min(block_size, rows - y)
            for x in range(0, cols, block_size):
                num_cols = min(block_size, cols - x)

                train_chunk = train_band.ReadAsArray(x, y, num_cols, num_rows)
                presence_chunk = combined_presence[y:y+num_rows, x:x+num_cols]
                burn_idx = (train_chunk == 1) & (presence_chunk > 0)
                
                if np.any(burn_idx):
                    burn_count_total += int(np.sum(burn_idx))
                    train_chunk[burn_idx] = 2
                    train_band.WriteArray(train_chunk, x, y)

        train_band.FlushCache()
        train_ds = None 
        
        return f"{ecoregion.stem}: Success! {burn_count_total} raster pixels captured."

    except Exception as e:
        return f"{ecoregion.stem}: Critical Error - {str(e)}"


class RasterMaskEngine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def delete_intermediate_files(self, outputs) -> None:
        """Delete any intermediate shapefiles"""

        merged_shapefiles = pathlib.Path(outputs).rglob('training_data_outlines*')
        for file in merged_shapefiles:
            file.unlink()

    def dissolve_tile_index_shapefiles(self, approved_files, ecoregions) -> None:
        """Create dissolved single polygons to use as burn rasters, excluding missing TIFs"""

        for ecoregion in ecoregions:
            if ecoregion.stem in approved_files:
                for tile_index_path in approved_files[ecoregion.stem]:
                    print(f' - Checking index: {tile_index_path.name}')

                    dissolved_tile_index_path = tile_index_path.parent / f"{tile_index_path.stem}_dis.shp"
                    if dissolved_tile_index_path.exists():
                        print(f' - Skipping: {dissolved_tile_index_path.name}')
                        continue 

                    tile_index = gpd.read_file(tile_index_path)
                    initial_count = len(tile_index)

                    possible_cols = ['filename', 'location']
                    file_col = next((c for c in possible_cols if c in tile_index.columns), None)

                    if file_col:
                        def check_local_tif(row):
                            val = str(row[file_col])
                            if not val or val == 'None':
                                return False
                            
                            tif_name = pathlib.Path(val).name
                            if not tif_name.lower().endswith('.tif'):
                                tif_name = f"{pathlib.Path(tif_name).stem}.tif"
                            
                            local_tif = tile_index_path.parent / tif_name
                            return local_tif.exists()

                        tile_index = tile_index[tile_index.apply(check_local_tif, axis=1)].copy()
                        
                        final_count = len(tile_index)
                        if final_count < initial_count:
                            print(f' - removed {initial_count - final_count} polygons with missing tiles.')
                    else:
                        print(f' - Warning: No filename/location column found in {tile_index_path.name}')

                    # Dissolve only if we have remaining geometry
                    if not tile_index.empty:
                        print(f' - Dissolving {len(tile_index)} polygons...')
                        dissolved_gdf = tile_index.dissolve().to_crs("EPSG:4326")
                        dissolved_gdf.to_file(dissolved_tile_index_path)
                    else:
                        print(f'   -> Skipping {tile_index_path.name}: No local TIF files found.')

    def get_approved_area_files(self, ecoregions: list[pathlib.Path]):
        """Get list of intersected features from ecoregion tile index shapefile"""

        approved_files = {}
        for ecoregion in ecoregions:
            digital_coast = ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            if digital_coast.is_dir():
                config_path = INPUTS / 'lookups' / 'ER_3_lidar_data_config.yaml'
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                excluded_providers = [
                    key for key, value in config.items() if value.get('use') is False
                ]

                json_files = [
                    folder for folder in digital_coast.rglob('feature.json')
                    if 'unused_providers' not in str(folder) and not any(
                        provider in str(folder) for provider in excluded_providers
                    )
                ]
                for file in json_files:
                    if ecoregion.stem not in approved_files:
                        approved_files[ecoregion.stem] = []
                    parent_project = file.parents[0]
                    tile_index_shps = list(parent_project.rglob('*index*.shp'))
                    if tile_index_shps:
                        approved_files[ecoregion.stem].append(tile_index_shps[0])

        print(f' - Approved area files: {approved_files}')                
        return approved_files

    def get_transformation(self) -> osr.CoordinateTransformation:
        """Transformation object for WGS84 to UTM17"""

        utm17_gdal = osr.SpatialReference()
        utm17_gdal.ImportFromEPSG(32617)
        wgs84_gdal = osr.SpatialReference()
        wgs84_gdal.ImportFromEPSG(4326)
        wgs84_to_utm17_transform = osr.CoordinateTransformation(wgs84_gdal, utm17_gdal)
        return wgs84_to_utm17_transform

    def merge_dissolved_polygons(self, ecoregions):
        """Creating full training outline shapefile for ecoregion"""

        for ecoregion in ecoregions:
            digital_coast = ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            if digital_coast.is_dir():
                dissolved_shapefiles = [folder for folder in digital_coast.rglob('*_dis.shp') if 'unused_providers' not in str(folder)]
                merged_training_ds = pd.concat([gpd.read_file(shp) for shp in dissolved_shapefiles]).dissolve()
                training_datasets = ecoregion / get_config_item('MASK', 'SUBFOLDER') / 'training_data_outlines.shp'
                merged_training_ds.to_file(training_datasets)

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)

    def run(self, outputs: str) -> None:
        ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
        print('Creating prediction masks')
        self.setup_dask(self.param_lookup['env'])
        self.process_prediction_masks(ecoregions, outputs)
        # approved_files = self.get_approved_area_files(ecoregions)
        # print(f' - Found files for ecoregions: {list(approved_files.keys())}')
        # print('Dissolving tile index shapefiles')
        # self.dissolve_tile_index_shapefiles(approved_files, ecoregions)
        # print('Merging all dissolved tile index shapfiles')
        # self.merge_dissolved_polygons(ecoregions)
        print('Creating training masks')
        self.process_training_masks(ecoregions, outputs)
        self.close_dask()
        self.delete_intermediate_files(outputs)
        
    def process_prediction_masks(self, ecoregions: list[pathlib.Path], outputs: str) -> None:
        """Multiprocessing entrypoint for creating prediction masks"""

        future_tiles = self.client.map(_create_prediction_mask, [[ecoregion, self.param_lookup] for ecoregion in ecoregions])
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)

    def process_training_masks(self, ecoregions: list[pathlib.Path], outputs: str) -> None:
        """Multiprocessing entrypoint for creating training masks"""

        future_tiles = self.client.map(_create_training_mask, ecoregions)
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)
