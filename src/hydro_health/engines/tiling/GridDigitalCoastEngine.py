import os
import json
import pathlib
import tempfile
import s3fs
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from distributed import LocalCluster, Client

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


def _grid_single_vrt_s3(params: list) -> str:
    """Grid a single S3 VRT"""

    # blue_topo_gdf is now a Future or the object itself if scattered correctly
    vrt_s3_path, ecoregion_prefix, bluetopo_grids, blue_topo_gdf, param_lookup = params

    engine = GridDigitalCoastEngine(param_lookup)
    
    s3_files = s3fs.S3FileSystem()
    gdal.SetConfigOption('CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE', 'YES')
    gdal.SetConfigOption('CPL_VSIL_S3_WRITE_SUPPORT', 'YES')
    # Limit GDAL's internal cache to prevent "Unmanaged Memory" bloat
    gdal.SetCacheMax(536870912) # 512MB
    gdal.UseExceptions()

    vrt_ds = None
    vrt_stem = pathlib.Path(vrt_s3_path).stem
    vsi_vrt_path = f"/vsis3/{vrt_s3_path}"
    
    try:
        vrt_ds = gdal.Open(vsi_vrt_path)
        vrt_projection = vrt_ds.GetProjection()
        
        vrt_data_suffix = '_'.join(vrt_stem.split('_')[3:])
        vrt_parent = vrt_s3_path.rsplit('/', 1)[0]
        shp_search_path = f"{vrt_parent}/{vrt_data_suffix}/**/*_dis.shp"
        shp_matches = s3_files.glob(shp_search_path)

        if not shp_matches:
            return f" - Skipped: No shapefile for {vrt_stem}"

        with tempfile.TemporaryDirectory() as tmpdir:
            s3_base = shp_matches[0].rsplit('.', 1)[0]
            local_base = os.path.join(tmpdir, "tileindex")
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                s3_target = f"{s3_base}{ext}"
                if s3_files.exists(s3_target):
                    s3_files.get(s3_target, f"{local_base}{ext}")
            
            dissolve_gdf = gpd.read_file(f"{local_base}.shp")
            if dissolve_gdf.crs != blue_topo_gdf.crs:
                dissolve_gdf = dissolve_gdf.to_crs(blue_topo_gdf.crs)
            
            dissolve_geom = dissolve_gdf.union_all()

        intersecting_tiles = blue_topo_gdf[
            (blue_topo_gdf['tile'].isin(bluetopo_grids)) & 
            (blue_topo_gdf.intersects(dissolve_geom))
        ]

        for _, tile_row in intersecting_tiles.iterrows():
            folder_name = tile_row['tile']
            tiled_sub = get_config_item('DIGITALCOAST', 'TILED_SUBFOLDER')
            output_prefix = f"{ecoregion_prefix}/{tiled_sub}/{folder_name}/{vrt_stem}_{folder_name}.tiff"

            if s3_files.exists(output_prefix):
                continue

            with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp_file:
                local_tmp_path = tmp_file.name

            in_memory_geojson = f"/vsimem/cutline_{folder_name}_{vrt_stem}.json"
            tile_geojson = {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": tile_row.geometry.__geo_interface__}]
            }
            gdal.FileFromMemBuffer(in_memory_geojson, json.dumps(tile_geojson))

            gdal.Warp(
                local_tmp_path, vrt_ds, format='GTiff',
                cutlineDSName=in_memory_geojson, cropToCutline=True,
                dstNodata=-9999, srcSRS=vrt_projection, dstSRS=vrt_projection,
                creationOptions=["COMPRESS=DEFLATE", "TILED=YES"]
            )

            s3_files.put(local_tmp_path, output_prefix)
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)
            gdal.Unlink(in_memory_geojson)
            engine.write_message(f" - Processed S3: {output_prefix}", param_lookup['output_directory'].valueAsText)

        return f" - Processed S3: {vrt_stem}"
    except Exception as e:
        return f" - S3 Error on {vrt_stem}: {str(e)}"
    finally:
        vrt_ds = None


def _grid_single_vrt_local(params: list) -> str:
    """Grid a single local VRT"""

    vrt_path, ecoregion, bluetopo_grids, blue_topo_gdf, param_lookup = params
    engine = GridDigitalCoastEngine(param_lookup)

    vrt_ds = None
    
    try:
        vrt = pathlib.Path(vrt_path)
        vrt_ds = gdal.Open(str(vrt))
        vrt_proj = vrt_ds.GetProjection()
        
        vrt_data_folder = vrt.parent / '_'.join(vrt.stem.split('_')[3:])
        shp_list = list(vrt_data_folder.rglob('*_dis.shp'))
        if not shp_list:
            return f" - Skipped: No local shp for {vrt.name}"
            
        dissolve_gdf = gpd.read_file(shp_list[0])
        if dissolve_gdf.crs != blue_topo_gdf.crs:
            dissolve_gdf = dissolve_gdf.to_crs(blue_topo_gdf.crs)
        
        dissolve_geom = dissolve_gdf.union_all()
        intersecting_tiles = blue_topo_gdf[
            (blue_topo_gdf['tile'].isin(bluetopo_grids)) & 
            (blue_topo_gdf.intersects(dissolve_geom))
        ]

        tiled_sub = get_config_item('DIGITALCOAST', 'TILED_SUBFOLDER')
        for _, tile_row in intersecting_tiles.iterrows():
            folder_name = tile_row['tile']
            out_dir = ecoregion / tiled_sub / folder_name
            out_file = out_dir / f'{vrt.stem}_{folder_name}.tiff'
            
            if out_file.exists():
                continue
            
            out_dir.mkdir(parents=True, exist_ok=True)
            gdal.Warp(
                str(out_file), vrt_ds, format='GTiff',
                cutlineDSName=tile_row.geometry.wkt, cropToCutline=True,
                dstNodata=-9999, cutlineSRS=vrt_proj,
                creationOptions=["COMPRESS=DEFLATE", "TILED=YES"]
            )
            engine.write_message(f" - Processed S3: {out_file}", param_lookup['output_directory'].valueAsText)
        return f" - Processed Local: {vrt.name}"
    except Exception as e:
        return f" - Local Error {vrt.name}: {str(e)}"
    finally:
        vrt_ds = None


class GridDigitalCoastEngine(Engine):
    """Class for gridding DigitalCoast VRT files against BlueTopo polygons"""

    def __init__(self, param_lookup) -> None:
        super().__init__()
        self.param_lookup = param_lookup

    def process_s3_vrt_gridding(self, blue_topo_gdf_future, outputs: str, manual_download:bool) -> None:
        """Processor for gridding S3 VRT files with dask"""

        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        ecoregion_paths = s3_files.glob(f"{bucket}/ER_*")

        for ecoregion_prefix in ecoregion_paths:
            print(f"Gridding S3 ecoregion: {ecoregion_prefix}")
            
            bt_sub = get_config_item('BLUETOPO', 'SUBFOLDER')
            blue_topo_search = f"{ecoregion_prefix}/{bt_sub}/BlueTopo/"
            bluetopo_grids = [p.split('/')[-1] for p in s3_files.ls(blue_topo_search) if s3_files.isdir(p)]
            
            dc_sub = get_config_item('DIGITALCOAST', 'SUBFOLDER')
            digital_coast_folder = 'DigitalCoast_manual_downloads' if manual_download else 'DigitalCoast'
            vrt_files = s3_files.glob(f"{ecoregion_prefix}/{dc_sub}/{digital_coast_folder}/*.vrt")
            print('vrt files:', f"{ecoregion_prefix}/{dc_sub}/{digital_coast_folder}/*.vrt")

            if vrt_files:
                # We pass the Future (blue_topo_gdf_future) instead of the full object
                params = [[vrt, ecoregion_prefix, bluetopo_grids, blue_topo_gdf_future, self.param_lookup] for vrt in vrt_files]
                future_tiles = self.client.map(_grid_single_vrt_s3, params)
                tile_results = self.client.gather(future_tiles)
                self.print_async_results(tile_results, outputs)
            else:
                print(f" - No VRTs found for {ecoregion_prefix} in S3.")

    def process_local_vrt_gridding(self, blue_topo_gdf_future, outputs: str) -> None:
        """Processor for gridding local VRT files with dask"""

        ecoregions = [ecoregion for ecoregion in pathlib.Path(outputs).glob('ER_*') if ecoregion.is_dir()]

        for ecoregion in ecoregions:
            print(f"Gridding local ecoregion: {ecoregion.stem}")
            
            bt_sub_path = ecoregion / get_config_item('BLUETOPO', 'SUBFOLDER') / 'BlueTopo'
            bluetopo_grids = [f.stem for f in bt_sub_path.iterdir() if f.is_dir()] if bt_sub_path.exists() else []
            
            dc_sub_path = ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            vrt_files = list(dc_sub_path.glob('*.vrt'))

            if vrt_files:
                params = [[str(v), ecoregion, bluetopo_grids, blue_topo_gdf_future, self.param_lookup] for v in vrt_files]
                future_tiles = self.client.map(_grid_single_vrt_local, params)
                tile_results = self.client.gather(future_tiles)
                self.print_async_results(tile_results, outputs)
            else:
                print(f" - No VRTs found for {ecoregion.stem} locally.")

    def run(self, manual_download=False) -> None:
        outputs = self.param_lookup['output_directory'].valueAsText

        self.setup_dask(self.param_lookup['env'])
        
        master_grids_path = str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS'))
        tiles_layer_name = get_config_item('SHARED', 'TILES')
        blue_topo_gdf = gpd.read_file(master_grids_path, layer=tiles_layer_name)

        [blue_topo_gdf_future] = self.client.scatter([blue_topo_gdf], broadcast=True)

        if self.param_lookup['env'] in ['local', 'remote']:
            self.process_local_vrt_gridding(blue_topo_gdf_future, outputs)
        else:
            self.process_s3_vrt_gridding(blue_topo_gdf_future, outputs, manual_download)

        self.close_dask()