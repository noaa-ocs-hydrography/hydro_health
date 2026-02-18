import os
import sys
import json
import boto3
import pathlib
import shutil
import s3fs
import tempfile
import multiprocessing as mp
import pandas as pd
import geopandas as gpd

from osgeo import ogr, osr, gdal

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


mp.set_executable(os.path.join(sys.exec_prefix, 'pythonw.exe'))

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


def _create_prediction_mask(param_inputs: list[list]) -> None:
    """Create, upload, and retain the prediction mask locally for the training step"""

    temp_folder, ecoregion, param_lookup = param_inputs

    s3_client = boto3.client('s3')
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    engine = RasterMaskS3Engine(param_lookup)

    gpkg = INPUTS / 'Master_Grids.gpkg'
    gpkg_ds = ogr.Open(str(gpkg))
    source_layer = gpkg_ds.GetLayerByName('EcoRegions_50m')
    
    mem_driver = ogr.GetDriverByName('Memory')
    output_srs = osr.SpatialReference()
    output_srs.ImportFromEPSG(32617)
    mem_ds = mem_driver.CreateDataSource('mem_ds')
    mem_layer = mem_ds.CreateLayer('selected_ecoregion', srs=output_srs, geom_type=ogr.wkbPolygon)

    source_layer.SetAttributeFilter(f"EcoRegion = '{ecoregion.stem}'")
    for feature in source_layer:
        geom = feature.GetGeometryRef()
        geom.Transform(engine.get_transformation())
        new_feat = ogr.Feature(mem_layer.GetLayerDefn())
        new_feat.SetGeometry(geom)
        mem_layer.CreateFeature(new_feat)
    
    xmin, xmax, ymin, ymax = mem_layer.GetExtent()
    pixel_size = 8
    x_res = int((xmax - xmin) / pixel_size)
    y_res = int((ymax - ymin) / pixel_size)

    # Use the ecoregion path to store the file permanently for now
    prediction_mask_name = f"prediction_mask_{ecoregion.stem}.tif"
    local_path = temp_folder / ecoregion.stem / get_config_item('MASK', 'SUBFOLDER') / prediction_mask_name
    local_path.parents[0].mkdir(parents=True, exist_ok=True)

    target_ds = gdal.GetDriverByName("GTiff").Create(
        str(local_path), x_res, y_res, 1, gdal.GDT_Byte,
        options=["COMPRESS=LZW", "TILED=YES", "NUM_THREADS=ALL_CPUS"]
    )
    target_ds.SetGeoTransform((xmin, pixel_size, 0, ymax, 0, -pixel_size))
    target_ds.SetProjection(output_srs.ExportToWkt())
    target_ds.GetRasterBand(1).SetNoDataValue(0)

    gdal.RasterizeLayer(target_ds, [1], mem_layer, burn_values=[1])
    target_ds = None # Flush to disk

    # Upload to S3
    s3_key = f"testing/{ecoregion.stem}/{get_config_item('MASK', 'SUBFOLDER')}/{prediction_mask_name}"
    s3_client.upload_file(str(local_path), bucket, s3_key)

    mem_ds = None
    gpkg_ds = None


def _create_training_mask(param_inputs: list[list]):
    """Create training mask, upload, and then cleanup local tif files"""

    temp_folder, ecoregion = param_inputs

    mask_subfolder = get_config_item('MASK', 'SUBFOLDER')
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

    prediction_file = temp_folder / ecoregion.stem / mask_subfolder / f'prediction_mask_{ecoregion.stem}.tif'
    training_file = temp_folder / ecoregion.stem / mask_subfolder / f'training_mask_{ecoregion.stem}.tif'
    training_data_outline = temp_folder / ecoregion.stem / mask_subfolder / 'training_data_outlines.shp'

    if prediction_file.exists() and training_data_outline.exists():
        # Copy prediction file locally to create the training base
        shutil.copy(str(prediction_file), str(training_file))
        
        training_ds = gdal.Open(str(training_file), gdal.GA_Update)
        shp_driver = ogr.GetDriverByName("ESRI Shapefile")
        training_data_outline_ds = shp_driver.Open(str(training_data_outline))
        training_mask_layer = training_data_outline_ds.GetLayer(0)

        gpkg_driver = ogr.GetDriverByName("GPKG")
        geopackage_ds = gpkg_driver.Open(str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')))
        ecoregions_layer = geopackage_ds.GetLayer(get_config_item('SHARED', 'ECOREGIONS'))
        ecoregions_layer.SetAttributeFilter(f"EcoRegion = '{ecoregion.stem}'")

        in_memory_driver = ogr.GetDriverByName("Memory")
        clipped_ds = in_memory_driver.CreateDataSource('clipped_mem')
        clipped_layer = clipped_ds.CreateLayer(
            "clipped_layer",
            geom_type=training_mask_layer.GetGeomType(),
            srs=training_mask_layer.GetSpatialRef(),
        )
        training_mask_layer.Clip(ecoregions_layer, clipped_layer)

        gdal.RasterizeLayer(training_ds, [1], clipped_layer, burn_values=[2])
        
        # Explicitly close everything to release file locks
        training_ds = None
        geopackage_ds = None
        clipped_ds = None
        training_data_outline_ds = None

        # Upload final training mask to S3
        s3_client = boto3.client('s3')
        s3_key = f"testing/{ecoregion.stem}/{mask_subfolder}/training_mask_{ecoregion.stem}.tif"
        s3_client.upload_file(str(training_file), bucket, s3_key)

        # CLEANUP: Delete the local .tif files now that they are safely on S3
        if prediction_file.exists():
            prediction_file.unlink()
        if training_file.exists():
            training_file.unlink()
        
        return f' - Processed: {ecoregion.stem}'
    else:
        return f' - Mask files missing: {ecoregion.stem}'


class RasterMaskS3Engine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def dissolve_tile_index_shapefiles(self, approved_files, ecoregions, temp_folder) -> None:
        """Dissolve tile index shapefiles by buffering through local temp storage"""
        
        s3_files = s3fs.S3FileSystem()
        for ecoregion in ecoregions:
            if ecoregion.stem in approved_files:
                for tile_index_path in approved_files[ecoregion.stem]:
                    s3_folder = "/".join(tile_index_path.split('/')[:-1])
                    digital_coast_path = "/".join(tile_index_path.split('/')[2:-1])
                    file_stem = pathlib.Path(tile_index_path).stem
                    for s3_file in s3_files.ls(s3_folder):
                        if file_stem in s3_file:
                            local_file = temp_folder / ecoregion.stem / digital_coast_path / pathlib.Path(s3_file).name
                            s3_files.get(s3_file, str(local_file))
                    
                    local_shp = temp_folder / ecoregion.stem / digital_coast_path/  f"{file_stem}.shp"
                    if not local_shp.exists():
                        print(f"Error: Could not find {local_shp} after download.")
                        continue
                        
                    tile_index = gpd.read_file(str(local_shp))
                    
                    print(f" - {file_stem}...")
                    dissolved_tile_index = tile_index.dissolve().to_crs("EPSG:4326")
                    
                    dissolved_filename = f"{file_stem}_dis"
                    local_out_path = temp_folder / ecoregion.stem / digital_coast_path / f"{dissolved_filename}.shp"
                    dissolved_tile_index.to_file(local_out_path, driver='ESRI Shapefile')
                    
                    # upload local dissolved file to S3
                    # for local_file in temp_folder.glob(f"{dissolved_filename}.*"):
                    #     s3_target = f"{s3_folder}/{local_file.name}"
                    #     s3.put(str(local_file), s3_target)
                    #     print(f" - uploaded: {s3_target}")

    def get_approved_area_files(self, ecoregions: list[pathlib.Path]):
        """Get list of intersected features from ecoregion tile index shapefile"""

        approved_files = {}
        for ecoregion in ecoregions:
            s3_files = s3fs.S3FileSystem()
            digital_coast_path = f"s3://{get_config_item('SHARED', 'OUTPUT_BUCKET')}/{ecoregion.stem}/{get_config_item('DIGITALCOAST', 'SUBFOLDER')}/DigitalCoast"
            json_files = s3_files.glob(f'{digital_coast_path}/**/feature.json')
            for file in json_files:
                if ecoregion.stem not in approved_files:
                    approved_files[ecoregion.stem] = []
                parent_project = '/'.join(file.split('/')[:-1])
                tile_index_shps = s3_files.glob(f'{parent_project}/**/*index*.shp')
                if tile_index_shps:
                    approved_files[ecoregion.stem].append(tile_index_shps[0])
        return approved_files

    def get_transformation(self) -> osr.CoordinateTransformation:
        """Transformation object for WGS84 to UTM17"""

        utm17_gdal = osr.SpatialReference()
        utm17_gdal.ImportFromEPSG(32617)
        wgs84_gdal = osr.SpatialReference()
        wgs84_gdal.ImportFromEPSG(4326)
        wgs84_to_utm17_transform = osr.CoordinateTransformation(wgs84_gdal, utm17_gdal)
        return wgs84_to_utm17_transform

    def merge_dissolved_polygons(self, ecoregions, temp_folder):
        """Creating full training outline shapefile for ecoregion"""

        for ecoregion in ecoregions:
            digital_coast_folder = temp_folder / ecoregion.stem / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            if digital_coast_folder.is_dir():
                dissolved_shapefiles = digital_coast_folder.rglob('*_dis.shp')
                merged_training_ds = pd.concat([gpd.read_file(shp) for shp in dissolved_shapefiles]).dissolve()
                training_data_outlines = temp_folder / ecoregion.stem / get_config_item('MASK', 'SUBFOLDER') / 'training_data_outlines.shp'
                merged_training_ds.to_file(training_data_outlines)

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)  

    def run(self, outputs: str) -> None:
        ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]
        print('Creating prediction masks')
        self.setup_dask(self.param_lookup['env'])
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_folder = pathlib.Path(tmpdir)
            self.process_prediction_masks(ecoregions, temp_folder, outputs)
            approved_files = self.get_approved_area_files(ecoregions)
            print(f' - Found files for ecoregions: {list(approved_files.keys())}')
            print('Dissolving tile index shapefiles')
            self.dissolve_tile_index_shapefiles(approved_files, ecoregions, temp_folder)
            print('Merging all dissolved tile index shapfiles')
            self.merge_dissolved_polygons(ecoregions, temp_folder)
            print('Creating training masks')
            self.process_training_masks(ecoregions, temp_folder, outputs)
            self.close_dask()
        
    def process_prediction_masks(self, ecoregions: list[pathlib.Path], temp_folder: pathlib.Path, outputs: str) -> None:
        """Multiprocessing entrypoint for creating prediction masks"""

        future_tiles = self.client.map(_create_prediction_mask, [[temp_folder, ecoregion, self.param_lookup] for ecoregion in ecoregions])
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)

    def process_training_masks(self, ecoregions: list[pathlib.Path], temp_folder: pathlib.Path, outputs: str) -> None:
        """Multiprocessing entrypoint for creating training masks"""

        future_tiles = self.client.map(_create_training_mask, [[temp_folder, ecoregion] for ecoregion in ecoregions])
        tile_results = self.client.gather(future_tiles)
        self.print_async_results(tile_results, outputs)
