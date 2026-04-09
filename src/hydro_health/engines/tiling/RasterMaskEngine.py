import pathlib
import shutil
import numpy as np
from osgeo import ogr, osr, gdal
from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[4] / 'inputs'


def _create_prediction_mask(param_inputs: list) -> None:
    """Rasterize the Ecoregion boundary into a tiled, compressed GeoTIFF"""

    ecoregion_path, param_lookup = param_inputs
    
    gpkg = INPUTS / 'Master_Grids.gpkg'
    gpkg_ds = ogr.Open(str(gpkg))
    ecoregions_layer = gpkg_ds.GetLayerByName('EcoRegions_50m')
    
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


def _create_training_mask(ecoregion_path: pathlib.Path) -> str:
    """Check actual raster data presence to upgrade prediction mask (1) to training mask (2)"""
    
    mask_subfolder = ecoregion_path / get_config_item('MASK', 'SUBFOLDER')
    prediction_file = mask_subfolder / f'prediction_mask_{ecoregion_path.stem}.tif'
    training_file = mask_subfolder / f'training_mask_{ecoregion_path.stem}.tif'
    dc_vrt_folder = ecoregion_path / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast'

    vrts = [str(f) for f in dc_vrt_folder.glob("mosaic_*.vrt")]
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

    for y in range(0, rows, block_size):
        num_rows = min(block_size, rows - y)
        for x in range(0, cols, block_size):
            num_cols = min(block_size, cols - x)
            mask_chunk = band.ReadAsArray(x, y, num_cols, num_rows)
            presence_chunk = np.zeros((num_rows, num_cols), dtype=np.uint8)
            
            chunk_geo_t = (
                geo_t[0] + x * geo_t[1], geo_t[1], 0,
                geo_t[3] + y * geo_t[5], 0, geo_t[5]
            )

            for vrt in vrts:
                vrt_ds = gdal.Open(vrt)
                # Warp the VRT into a small memory chunk matching our block
                tmp_ds = gdal.GetDriverByName('MEM').Create('', num_cols, num_rows, 2, gdal.GDT_Byte)
                tmp_ds.SetGeoTransform(chunk_geo_t)
                tmp_ds.SetProjection(proj)
                
                # dstAlpha creates a mask band (Band 2) showing where data exists
                gdal.Warp(tmp_ds, vrt_ds, dstAlpha=True, resampleAlg=gdal.GRA_NearestNeighbour)
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
    ds.BuildOverviews("NEAREST", [2, 4, 8, 16])
    ds = None
    
    return f"{ecoregion_path.stem}: {total_burns} training pixels marked."


class RasterMaskEngine(Engine):
    def __init__(self, param_lookup):
        super().__init__()
        self.param_lookup = param_lookup

    def run(self, outputs: str) -> None:
        ecoregions = [d for d in pathlib.Path(outputs).glob('ER_*') if d.is_dir()]
        self.setup_dask(self.param_lookup['env'])
        
        self.client.gather(self.client.map(_create_prediction_mask, [[er, self.param_lookup] for er in ecoregions]))
        results = self.client.gather(self.client.map(_create_training_mask, ecoregions))
        
        for r in results: 
            print(r)
        self.close_dask()