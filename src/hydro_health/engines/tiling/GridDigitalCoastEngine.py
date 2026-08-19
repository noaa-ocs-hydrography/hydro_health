import os
import json
import pathlib
import tempfile
import s3fs
import pandas as pd
import geopandas as gpd
from osgeo import gdal

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_approved_providers

INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


def _grid_single_vrt_s3(params: list) -> str:
    """Grid a single S3 VRT"""

    vrt_s3_path, ecoregion_prefix, bluetopo_grids, blue_topo_gdf, param_lookup = params

    engine = GridDigitalCoastEngine(param_lookup)
    
    s3_files = s3fs.S3FileSystem()
    gdal.SetConfigOption('CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE', 'YES')
    gdal.SetConfigOption('CPL_VSIL_S3_WRITE_SUPPORT', 'YES')
    gdal.SetCacheMax(536870912) # 512MB
    gdal.UseExceptions()

    vrt_ds = None
    vrt_stem = pathlib.Path(vrt_s3_path).stem
    vsi_vrt_path = f"/vsis3/{vrt_s3_path}"
    
    try:
        vrt_ds = gdal.Open(vsi_vrt_path)
        vrt_projection = vrt_ds.GetProjection()
        
        provider_folder_name = vrt_stem.replace('mosaic_', '')
        digital_coast_folder = pathlib.Path(vrt_s3_path).parents[0]
        
        shp_search_path = f"{digital_coast_folder}/{provider_folder_name}/**/*.shp"
        shp_matches = s3_files.glob(shp_search_path)
        
        tileindex_matches = [f for f in shp_matches if 'tileindex' in pathlib.Path(f).name.lower()]
        if tileindex_matches:
            target_shp = tileindex_matches[0]
        elif shp_matches:
            target_shp = shp_matches[0]
        else:
            target_shp = None

        if target_shp:
            with tempfile.TemporaryDirectory() as tmpdir:
                shp_base_name = target_shp.rsplit('.', 1)[0]
                local_base = os.path.join(tmpdir, "tileindex")
                
                # Download sidecar files (.shp, .shx, .dbf, .prj)
                for ext in ['.shp', '.shx', '.dbf', '.prj']:
                    s3_target = f"{shp_base_name}{ext}"
                    try:
                        if s3_files.exists(s3_target):
                            s3_files.get(s3_target, f"{local_base}{ext}")
                    except Exception as sidecar_err:
                        engine.write_message(f"Warning: Skipped sidecar {s3_target}: {sidecar_err}", param_lookup['output_directory'].valueAsText)
                
                raw_gdf = gpd.read_file(f"{local_base}.shp")
                if raw_gdf.crs != blue_topo_gdf.crs:
                    raw_gdf = raw_gdf.to_crs(blue_topo_gdf.crs)
                
                # Dissolve geometries on the fly
                try:
                    dissolve_geom = raw_gdf.geometry.union_all()
                except AttributeError:
                    dissolve_geom = raw_gdf.geometry.unary_union
        else:
            # Fallback to VRT bounds if no shapefile exists
            import shapely.geometry
            gt = vrt_ds.GetGeoTransform()
            minx = gt[0]
            maxy = gt[3]
            maxx = minx + gt[1] * vrt_ds.RasterXSize
            miny = maxy + gt[5] * vrt_ds.RasterYSize
            
            vrt_box = shapely.geometry.box(minx, miny, maxx, maxy)
            raw_gdf = gpd.GeoDataFrame({'geometry': [vrt_box]}, crs=vrt_projection)
            
            if raw_gdf.crs != blue_topo_gdf.crs:
                raw_gdf = raw_gdf.to_crs(blue_topo_gdf.crs)
            
            dissolve_geom = raw_gdf.geometry.iloc[0]

        intersecting_tiles = blue_topo_gdf[
            (blue_topo_gdf['tile'].isin(bluetopo_grids)) & 
            (blue_topo_gdf.intersects(dissolve_geom))
        ]

        for _, tile_row in intersecting_tiles.iterrows():
            folder_name = tile_row['tile']
            tiled_sub = get_config_item('DIGITALCOAST', 'TILED_SUBFOLDER')
            s3_out_file = f"{ecoregion_prefix}/{tiled_sub}/{folder_name}/{vrt_stem}_{folder_name}.tiff"

            if s3_files.exists(s3_out_file):
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
                cutlineSRS=blue_topo_gdf.crs.to_wkt(),
                creationOptions=[
                    "COMPRESS=DEFLATE", 
                    "PREDICTOR=3", 
                    "TILED=YES", 
                    "BLOCKXSIZE=512", 
                    "BLOCKYSIZE=512"
                ]
            )

            final_ds = gdal.Open(local_tmp_path, gdal.GA_Update)
            final_ds.BuildOverviews("BILINEAR", [2, 4, 8])
            final_ds = None

            s3_files.put(local_tmp_path, s3_out_file)
            if os.path.exists(local_tmp_path):
                os.remove(local_tmp_path)
            gdal.Unlink(in_memory_geojson)
            engine.write_message(f" - Processed S3: {s3_out_file}", param_lookup['output_directory'].valueAsText)

        return f" - Processed S3: {vrt_stem}"
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            engine.write_message(f"Traceback for {vrt_stem}:\n{tb}", param_lookup['output_directory'].valueAsText)
        except:
            pass
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
        
        outputs = param_lookup['output_directory'].valueAsText
        
        provider_folder_name = vrt.stem.replace('mosaic_', '')
        vrt_data_folder = vrt.parent / provider_folder_name
        
        # Look for tileindex shapefiles first, fall back to any shapefile if needed
        shp_list = list(vrt_data_folder.rglob('tileindex*.shp'))

        if not shp_list:
            shp_list = [
                f for f in vrt_data_folder.rglob('*.shp') 
                if not f.name.startswith('.')
            ]
            
        if shp_list:
            # Read the shapefile and dissolve on the fly
            raw_gdf = gpd.read_file(shp_list[0])
            engine.write_message(f'shp:{raw_gdf}', outputs)
            if raw_gdf.crs != blue_topo_gdf.crs:
                raw_gdf = raw_gdf.to_crs(blue_topo_gdf.crs)
            
            try:
                dissolve_geom = raw_gdf.geometry.union_all()
            except AttributeError:
                dissolve_geom = raw_gdf.geometry.unary_union

            engine.write_message(f'dissolved: {dissolve_geom}', outputs)
            engine.write_message(f'shp list:{shp_list}', outputs)
        else:
            # Fallback to VRT bounds if no shapefile exists
            import shapely.geometry
            gt = vrt_ds.GetGeoTransform()
            minx = gt[0]
            maxy = gt[3]
            maxx = minx + gt[1] * vrt_ds.RasterXSize
            miny = maxy + gt[5] * vrt_ds.RasterYSize
            
            vrt_box = shapely.geometry.box(minx, miny, maxx, maxy)
            raw_gdf = gpd.GeoDataFrame({'geometry': [vrt_box]}, crs=vrt_proj)
            
            if raw_gdf.crs != blue_topo_gdf.crs:
                raw_gdf = raw_gdf.to_crs(blue_topo_gdf.crs)
            
            dissolve_geom = raw_gdf.geometry.iloc[0]
            engine.write_message('Fallback to VRT bounding box.', outputs)

        intersecting_tiles = blue_topo_gdf[
            (blue_topo_gdf['tile'].isin(bluetopo_grids)) & 
            (blue_topo_gdf.intersects(dissolve_geom))
        ]

        tiled_sub = get_config_item('DIGITALCOAST', 'TILED_SUBFOLDER')

        for _, tile_row in intersecting_tiles.iterrows():
            folder_name = tile_row['tile']
            out_dir = ecoregion / tiled_sub / folder_name
            engine.write_message(f'output_dir:{out_dir}', outputs)
            out_file = out_dir / f'{vrt.stem}_{folder_name}.tiff'
            engine.write_message(f'out file:{out_file}', outputs)
            
            if out_file.exists():
                continue
            
            out_dir.mkdir(parents=True, exist_ok=True)
            
            in_memory_geojson = f"/vsimem/cutline_{folder_name}_{vrt.stem}.json"
            tile_geojson = {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": tile_row.geometry.__geo_interface__}]
            }
            gdal.FileFromMemBuffer(in_memory_geojson, json.dumps(tile_geojson))

            gdal.Warp(
                str(out_file), vrt_ds, format='GTiff',
                cutlineDSName=in_memory_geojson, cropToCutline=True,
                dstNodata=-9999, srcSRS=vrt_proj, dstSRS=vrt_proj,
                cutlineSRS=blue_topo_gdf.crs.to_wkt(),
                creationOptions=[
                    "COMPRESS=DEFLATE", 
                    "PREDICTOR=3", 
                    "TILED=YES", 
                    "BLOCKXSIZE=512", 
                    "BLOCKYSIZE=512"
                ]
            )

            final_ds = gdal.Open(str(out_file), gdal.GA_Update)
            final_ds.BuildOverviews("BILINEAR", [2, 4, 8])
            final_ds = None
            
            gdal.Unlink(in_memory_geojson)
            engine.write_message(f" - Processed Local: {out_file}", param_lookup['output_directory'].valueAsText)
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

    def process_s3_vrt_gridding(self, blue_topo_gdf_future, outputs: str, manual_download: bool, output_prefix: str) -> None:
        """Processor for gridding S3 VRT files with dask (Approved providers only)"""

        s3_files = s3fs.S3FileSystem()
        bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
        
        if output_prefix:
            ecoregion_paths = s3_files.glob(f"{bucket}/{output_prefix}/ER_*")
        else:
            ecoregion_paths = s3_files.glob(f"{bucket}/ER_*")

        for ecoregion_prefix in ecoregion_paths:
            ecoregion_stem = ecoregion_prefix.rsplit('/', 1)[-1]
            print(f"Gridding S3 ecoregion: {ecoregion_prefix} ({ecoregion_stem})")
            
            bt_sub = get_config_item('BLUETOPO', 'SUBFOLDER')
            blue_topo_search = f"{ecoregion_prefix}/{bt_sub}/BlueTopo/"
            bluetopo_grids = [p.split('/')[-1] for p in s3_files.ls(blue_topo_search) if s3_files.isdir(p)]
            
            dc_sub = get_config_item('DIGITALCOAST', 'SUBFOLDER')
            digital_coast_folder = 'Digital_Coast_Manual_Downloads' if manual_download else 'DigitalCoast'
            vrt_files = s3_files.glob(f"{ecoregion_prefix}/{dc_sub}/{digital_coast_folder}/*.vrt")
            
            if vrt_files:
                approved_providers = [p.lower().strip() for p in get_approved_providers(ecoregion_stem)]
                approved_vrt_files = []
                
                for vrt in vrt_files:
                    vrt_stem = pathlib.Path(vrt).stem
                    vrt_provider = vrt_stem.replace('mosaic_', '')
                    
                    is_approved = any(
                        vrt_provider.lower() in ap or vrt_stem.lower() in ap or ap in vrt_provider.lower() 
                        for ap in approved_providers
                    )
                    
                    if is_approved:
                        approved_vrt_files.append(vrt)
                    else:
                        print(f" - Skipping unapproved S3 provider: {vrt_provider}")

                if approved_vrt_files:
                    params = [[vrt, ecoregion_prefix, bluetopo_grids, blue_topo_gdf_future, self.param_lookup] for vrt in approved_vrt_files]
                    future_tiles = self.client.map(_grid_single_vrt_s3, params)
                    tile_results = self.client.gather(future_tiles)
                    self.print_async_results(tile_results, outputs)
                else:
                    print(f" - No approved VRTs found for {ecoregion_stem} in S3.")
            else:
                print(f" - No VRTs found for {ecoregion_prefix} in S3.")

    def process_local_vrt_gridding(self, blue_topo_gdf_future, outputs: str, output_prefix: str) -> None:
        """Processor for gridding local VRT files with dask"""

        base_path = pathlib.Path(outputs)
        if output_prefix:
            ecoregions = [ecoregion for ecoregion in (base_path / output_prefix).glob('ER_*') if ecoregion.is_dir()]
        else:
            ecoregions = [ecoregion for ecoregion in base_path.glob('ER_*') if ecoregion.is_dir()]

        for ecoregion in ecoregions:
            print(f"Gridding local ecoregion: {ecoregion.stem}")
            
            bt_sub_path = ecoregion / get_config_item('BLUETOPO', 'SUBFOLDER') / 'BlueTopo'
            bluetopo_grids = [f.stem for f in bt_sub_path.iterdir() if f.is_dir()] if bt_sub_path.exists() else []
            
            dc_sub_path = ecoregion / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'
            vrt_files = list(dc_sub_path.glob('*.vrt'))

            if vrt_files:
                approved_providers = [provider.lower().strip() for provider in get_approved_providers(ecoregion.stem)]
                approved_vrt_files = []
                
                for vrt in vrt_files:
                    vrt_provider = vrt.stem.replace('mosaic_', '')
                    
                    is_approved = any(
                        vrt_provider.lower() in ap or vrt.stem.lower() in ap or ap in vrt_provider.lower() 
                        for ap in approved_providers
                    )
                    
                    if is_approved:
                        approved_vrt_files.append(vrt)
                        
                params = [[str(v), ecoregion, bluetopo_grids, blue_topo_gdf_future, self.param_lookup] for v in approved_vrt_files]
                future_tiles = self.client.map(_grid_single_vrt_local, params)
                tile_results = self.client.gather(future_tiles)
                self.print_async_results(tile_results, outputs)
            else:
                print(f" - No VRTs found for {ecoregion.stem} locally.")

    def run(self, output_prefix: str, manual_download=False) -> None:
        """Main execution method routing control using structural parameters"""

        outputs = self.param_lookup['output_directory'].valueAsText

        self.setup_dask(self.param_lookup['env'])
        
        master_grids_path = str(INPUTS / get_config_item('SHARED', 'MASTER_GRIDS'))
        tiles_layer_name = get_config_item('SHARED', 'TILES')
        blue_topo_gdf = gpd.read_file(master_grids_path, layer=tiles_layer_name)

        [blue_topo_gdf_future] = self.client.scatter([blue_topo_gdf], broadcast=True)

        if self.param_lookup['env'] in ['local', 'remote']:
            self.process_local_vrt_gridding(blue_topo_gdf_future, outputs, output_prefix)
        else:
            self.process_s3_vrt_gridding(blue_topo_gdf_future, outputs, manual_download, output_prefix)

        self.close_dask()