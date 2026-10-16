# prep_shorelines.py
import os
import sys
import pathlib
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.env
import s3fs
import re

HH_MODEL = pathlib.Path(__file__).parents[1] / 'src'
print(HH_MODEL)
sys.path.append(str(HH_MODEL))

from hydro_health.helpers.tools import get_config_item

# Adjust these paths as needed for your project structure
INPUTS = pathlib.Path(__file__).parents[1] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'

def _set_gdal_s3_options() -> None:
    """Configure GDAL /vsis3/ driver options for fast cloud S3 header reads."""
    rasterio.env.Env(
        AWS_NO_SIGN_REQUEST='NO',
        AWS_EC2_METADATA_DISABLED='FALSE',
        GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
        VSI_CACHE='FALSE',
        GDAL_HTTP_MERGE_CONSECUTIVE_RANGES='YES',
        GDAL_HTTP_MULTIPLEX='YES',
    )


def upload_file_to_s3(output_file_path: pathlib.Path, ecoregion: str, output_prefix: str=False) -> None:
    """Upload file to S3"""

    s3 = s3fs.S3FileSystem()
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

    ecoregion_path = f"{output_prefix}/{ecoregion}" if output_prefix else f"{ecoregion}"
    s3_path = f"{bucket}/{ecoregion_path}/support_data"
    
    print(f"  - Uploading {output_file_path} to s3://{s3_path}")
    s3.put(str(output_file_path), s3_path, recursive=True)


def precompute_shoreline_vertices(master_gpkg: pathlib.Path, ecoregions: list, boundary_clip_buffer_m: float = 100.0):
    """Scans Bluetopo tiles on S3, determines needed CRSs, and saves local vector arrays per Ecoregion & CRS."""
    
    out_dir = INPUTS / 'precomputed_shorelines'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading master shoreline vector to get native CRS...")
    gpkg_crs = gpd.read_file(str(master_gpkg), layer='composite_shoreline_final', rows=1).crs
    
    enhanced_ecoregions = gpd.read_file(str(master_gpkg), layer='Enhanced_EcoRegions_50m')

    # Initialize S3 File System
    s3 = s3fs.S3FileSystem()
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

    # Activate S3 GDAL configurations globally for this session
    _set_gdal_s3_options()

    for ecoregion in ecoregions:
        print(f"\n--- Processing {ecoregion} ---")
        
        # 1. Discover all BlueTopo tiles for this ecoregion in S3
        s3_search_pattern = f"{bucket}/{ecoregion}/{get_config_item('BLUETOPO', 'SUBFOLDER')}/BlueTopo/**/*"
        print(f"Scanning S3 for tiles: s3://{s3_search_pattern}")
        raw_paths = s3.glob(s3_search_pattern)
        
        vsi_bluetopo_paths = [
            f"/vsis3/{p}" for p in raw_paths 
            if p.lower().endswith(('.vrt', '.tif', '.tiff'))
        ]
        
        if not vsi_bluetopo_paths:
            print(f"No BlueTopo tiles found for {ecoregion} in S3, skipping.")
            continue

        # Quickly read only the headers from S3 to get the CRSs
        unique_crss = set()
        for path in vsi_bluetopo_paths:
            with rasterio.open(path) as src:
                unique_crss.add(src.crs)
                
        print(f"Found {len(unique_crss)} unique CRS zones for {len(vsi_bluetopo_paths)} tiles.")

        # 2. Isolate the specific ecoregion boundary
        ecoregion_gdf = enhanced_ecoregions[enhanced_ecoregions['EcoRegion'] == ecoregion]
        eco_bbox = tuple(ecoregion_gdf.to_crs(gpkg_crs).total_bounds)
        
        print(f"Extracting local shoreline from Master Geopackage...")
        master_shoreline = gpd.read_file(
            str(master_gpkg), 
            layer='composite_shoreline_final',
            bbox=eco_bbox
        )

        if master_shoreline.empty:
            print(f"No shoreline features found for {ecoregion}.")
            continue

        # 3. Project, extract, and save arrays for each needed CRS
        for crs in unique_crss:
            # Safely extract EPSG code
            epsg_code = crs.to_epsg()
            if not epsg_code:
                try:
                    # Fallback for compound CRS if available
                    epsg_code = crs.sub_crs_list[0].to_epsg()
                except (AttributeError, IndexError):
                    # Ultimate fallback: stringify the CRS name and remove bad filepath chars
                    crs_str = getattr(crs, 'name', str(crs))
                    epsg_code = re.sub(r'[^a-zA-Z0-9]', '_', crs_str)
                    # Clean up multiple underscores and trim length so the filename isn't massive
                    epsg_code = re.sub(r'_+', '_', epsg_code).strip('_')[:50]

            safe_crs_name = f"EPSG_{epsg_code}"
            npy_path = out_dir / f"{ecoregion}_shoreline_{safe_crs_name}.npy"
            
            if npy_path.exists():
                print(f" - {npy_path.name} already exists. Skipping.")
                continue
                
            print(f" - Building vertex array for {safe_crs_name}...")
            
            local_eco_gdf = ecoregion_gdf.to_crs(crs) if ecoregion_gdf.crs != crs else ecoregion_gdf
            # Updated from .unary_union to .union_all() to fix the deprecation warning
            buffered_boundary = local_eco_gdf.buffer(boundary_clip_buffer_m).union_all()
            
            projected_shoreline = master_shoreline.to_crs(crs)
            clipped_shoreline = gpd.clip(projected_shoreline, buffered_boundary)

            all_coords = []
            for geom in clipped_shoreline.geometry:
                if geom.is_empty: continue
                if geom.geom_type == 'LineString':
                    all_coords.extend(list(geom.coords))
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        all_coords.extend(list(line.coords))
            
            np.save(npy_path, np.array(all_coords))
            upload_file_to_s3(npy_path, ecoregion)
            print(f"   Saved {len(all_coords)} vertices to {npy_path.name}")

if __name__ == "__main__":
    master_gpkg = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
    ecoregions = ['ER_3'] # Add your full list of ecoregions here
    precompute_shoreline_vertices(master_gpkg, ecoregions)