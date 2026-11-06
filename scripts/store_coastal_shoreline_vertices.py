# prep_shorelines.py
import sys
import time
import pathlib
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.env
import s3fs
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

HH_MODEL = pathlib.Path(__file__).parents[1] / 'src'
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


def get_safe_crs_name(crs: rasterio.crs.CRS) -> str:
    """Derive a normalized string safe for file naming from any rasterio CRS."""
    epsg_code = crs.to_epsg()
    if not epsg_code:
        try:
            epsg_code = crs.sub_crs_list[0].to_epsg()
        except (AttributeError, IndexError):
            crs_str = getattr(crs, 'name', str(crs))
            epsg_code = re.sub(r'[^a-zA-Z0-9]', '_', crs_str)
            epsg_code = re.sub(r'_+', '_', epsg_code).strip('_')[:50]
    return f"EPSG_{epsg_code}"


def _inspect_single_crs(vsi_path: str) -> rasterio.crs.CRS | None:
    """Read only the CRS header from a single raster over /vsis3/."""
    try:
        with rasterio.open(vsi_path) as src:
            return src.crs
    except Exception as e:
        print(f"  [Warning] Could not read CRS for {vsi_path}: {e}")
        return None


def discover_unique_crss_concurrent(
    s3: s3fs.S3FileSystem, 
    bucket: str, 
    ecoregion: str, 
    data_type: str,
    max_workers: int = 32
) -> set[rasterio.crs.CRS]:
    """
    Discover GeoTIFFs on S3 and read their CRSs.
    - BlueTopo: Scans headers across all tiles concurrently.
    - DigitalCoast: Reads only the first tile's header since all tiles share the same CRS.
    """
    subfolder = get_config_item(data_type.upper(), 'SUBFOLDER')
    
    if data_type == 'BlueTopo':
        s3_search_pattern = f"{bucket}/{ecoregion}/{subfolder}/BlueTopo/**/*"
        raw_paths = s3.glob(s3_search_pattern)
        geotiffs = [
            p for p in raw_paths
            if p.lower().endswith(('.tif', '.tiff', '.vrt'))
            and not any(pat in p.lower() for pat in ['_unc', '_slope', '_iss', '_survey_end_date', 'decay'])
        ]

        if not geotiffs:
            print(f"  - No {data_type} tiles found at s3://{s3_search_pattern}")
            return set()

        vsi_paths = [f"/vsis3/{p}" for p in geotiffs]
        print(f"  - Scanning headers of {len(vsi_paths)} {data_type} tiles (concurrency={max_workers})...")

        unique_crss = set()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {executor.submit(_inspect_single_crs, path): path for path in vsi_paths}
            for future in as_completed(future_to_path):
                crs = future.result()
                if crs:
                    unique_crss.add(crs)

        return unique_crss

    else:  # DigitalCoast
        s3_search_pattern = f"{bucket}/{ecoregion}/{subfolder}/DigitalCoast/**/*.tif"
        raw_paths = s3.glob(s3_search_pattern)
        first_tile = next((p for p in raw_paths if p.lower().endswith(('.tif', '.tiff'))), None)

        if not first_tile:
            print(f"  - No {data_type} tiles found at s3://{s3_search_pattern}")
            return set()

        print(f"  - Reading CRS from first {data_type} tile: s3://{first_tile}")
        crs = _inspect_single_crs(f"/vsis3/{first_tile}")
        
        return {crs} if crs else set()


def upload_file_to_s3(output_file_path: pathlib.Path, ecoregion: str, output_prefix: str = "") -> None:
    """Upload precomputed vertex file to S3 support_data folder."""
    s3 = s3fs.S3FileSystem()
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')

    ecoregion_path = f"{output_prefix}/{ecoregion}" if output_prefix else f"{ecoregion}"
    s3_path = f"{bucket}/{ecoregion_path}/support_data/{output_file_path.name}"
    
    print(f"  - Uploading {output_file_path.name} to s3://{s3_path}")
    s3.put(str(output_file_path), s3_path)


def precompute_shoreline_vertices(
    master_gpkg: pathlib.Path,
    ecoregions: list[str],
    data_types: list[str] = ['DigitalCoast'],
    boundary_clip_buffer_m: float = 100.0,
    output_prefix: str = ""
) -> None:
    """
    Scans tiles across specified datasets on S3, determines required projection zones,
    and extracts local shoreline vertex arrays per Ecoregion and CRS.
    """
    out_dir = INPUTS / 'precomputed_shorelines'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading master shoreline vector to get native CRS...")
    gpkg_crs = gpd.read_file(str(master_gpkg), layer='composite_shoreline_final', rows=1).crs
    enhanced_ecoregions = gpd.read_file(str(master_gpkg), layer='Enhanced_EcoRegions_50m')

    s3 = s3fs.S3FileSystem()
    bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
    _set_gdal_s3_options()

    for ecoregion in ecoregions:
        print(f"\n==========================================")
        print(f"Processing Ecoregion: {ecoregion}")
        print(f"==========================================")

        # 1. Discover unique CRSs needed for the chosen data types
        all_unique_crss = set()
        for dtype in data_types:
            dtype_crss = discover_unique_crss_concurrent(s3, bucket, ecoregion, dtype)
            print(f"  -> Discovered {len(dtype_crss)} unique CRS(s) in {dtype}")
            all_unique_crss.update(dtype_crss)

        if not all_unique_crss:
            print(f"No valid rasters found for {ecoregion} across any data source. Skipping.")
            continue

        print(f"\nTotal combined unique CRSs to compute for {ecoregion}: {len(all_unique_crss)}")

        # 2. Check local & S3 caches first before loading heavy master GPKG layers
        ecoregion_path = f"{output_prefix}/{ecoregion}" if output_prefix else f"{ecoregion}"
        crss_to_process = []
        for crs in all_unique_crss:
            safe_crs_name = get_safe_crs_name(crs)
            npy_filename = f"{ecoregion}_shoreline_{safe_crs_name}.npy"
            npy_path = out_dir / npy_filename
            s3_dest_path = f"{bucket}/{ecoregion_path}/support_data/{npy_filename}"

            if npy_path.exists():
                print(f" - [Cached Local] {npy_filename} already exists. Skipping.")
                continue
            if s3.exists(s3_dest_path):
                print(f" - [Cached S3] s3://{s3_dest_path} already exists. Skipping.")
                continue

            crss_to_process.append((crs, safe_crs_name, npy_path, npy_filename))

        if not crss_to_process:
            print(f"All arrays for {ecoregion} already generated and cached.")
            continue

        # 3. Extract bounding box and master shoreline for this ecoregion
        ecoregion_gdf = enhanced_ecoregions[enhanced_ecoregions['EcoRegion'] == ecoregion]
        if ecoregion_gdf.empty:
            print(f"Ecoregion '{ecoregion}' not found in Enhanced_EcoRegions_50m layer. Skipping.")
            continue

        eco_bbox = tuple(ecoregion_gdf.to_crs(gpkg_crs).total_bounds)
        print("Extracting local shoreline subset from Master Geopackage...")
        master_shoreline = gpd.read_file(
            str(master_gpkg), 
            layer='composite_shoreline_final',
            bbox=eco_bbox
        )

        if master_shoreline.empty:
            print(f"No shoreline features found for {ecoregion}. Skipping.")
            continue

        # 4. Project, clip, and store vertex arrays
        for crs, safe_crs_name, npy_path, npy_filename in crss_to_process:
            print(f" - Building vertex array for {safe_crs_name}...")

            local_eco_gdf = ecoregion_gdf.to_crs(crs) if ecoregion_gdf.crs != crs else ecoregion_gdf
            buffered_boundary = local_eco_gdf.buffer(boundary_clip_buffer_m).union_all()

            projected_shoreline = master_shoreline.to_crs(crs)
            clipped_shoreline = gpd.clip(projected_shoreline, buffered_boundary)

            all_coords = []
            for geom in clipped_shoreline.geometry:
                if geom.is_empty:
                    continue
                if geom.geom_type == 'LineString':
                    all_coords.extend(list(geom.coords))
                elif geom.geom_type == 'MultiLineString':
                    for line in geom.geoms:
                        all_coords.extend(list(line.coords))

            vertex_array = np.array(all_coords, dtype=np.float64)
            np.save(npy_path, vertex_array)
            upload_file_to_s3(npy_path, ecoregion, output_prefix=output_prefix)
            print(f"   Saved {len(vertex_array)} vertices to {npy_filename}")


if __name__ == "__main__":
    """
    This script depends on the shoreline layer in the Master_Grid_Layers.gpkg.
    The gpkg is stored in S3 because it is too large for the repository. 
    Use the AWS CLI to copy it to the box temporarily to run this code.
    """
    start = time.time()
    master_gpkg = INPUTS / get_config_item('SHARED', 'MASTER_GRIDS')
    ecoregions = ['ER_3']

    # Runs single-file header lookup for DigitalCoast
    precompute_shoreline_vertices(
        master_gpkg, 
        ecoregions, 
        data_types=['BlueTopo', 'DigitalCoast'],
    )
    end = time.time()
    print(f'Done - {(end - start)/60:.2f} minutes')