import geopandas as gpd
import pandas as pd
import numpy as np
import pyogrio
import fiona
import logging
import re
import os
import s3fs
import boto3
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.merge import merge
from rasterio.features import rasterize
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def is_garbage_text(text):
    """
    Detects likely corrupted/garbage text (like 'X;\P;I0;') 
    by checking for excessive punctuation/symbols.
    """
    if pd.isna(text):
        return False
    text = str(text)
    symbols = len(re.findall(r'[^a-zA-Z0-9\s]', text))
    if len(text) > 0 and (symbols / len(text)) > 0.3:
        return True
    return False

def process_ais_layer(gdf, layer_name):
    """
    Processes a single AIS GeoDataFrame: Flags invalid data, logs percentages, 
    and attempts to correct vessel dimensions.
    """
    initial_count = len(gdf)
    logging.info(f"--- Processing Layer: {layer_name} ({initial_count} records) ---")

    gdf['Data_Flags'] = ""
    
    # 1. IDENTIFY COMPLETELY INVALID TRACKLINES
    geom_invalid = gdf.geometry.isna() | gdf.geometry.is_empty | (gdf.geometry.length < 1e-6)
    
    time_invalid = pd.Series(False, index=gdf.index)
    if 'TrackStartTime' in gdf.columns and 'TrackEndTime' in gdf.columns:
        gdf['TrackStartTime'] = pd.to_datetime(gdf['TrackStartTime'], errors='coerce')
        gdf['TrackEndTime'] = pd.to_datetime(gdf['TrackEndTime'], errors='coerce')
        time_invalid = gdf['TrackStartTime'] >= gdf['TrackEndTime']

    trackline_invalid = geom_invalid | time_invalid
    gdf.loc[trackline_invalid, 'Data_Flags'] += "Invalid_Trackline;"

    # 2. FLAG SPECIFIC FIELD ERRORS
    heading_511 = gdf.get('Heading', pd.Series(0, index=gdf.index)) == 511
    gdf.loc[heading_511, 'Data_Flags'] += "Heading_511;"

    if 'VesselName' in gdf.columns:
        garbage_names = gdf['VesselName'].apply(is_garbage_text)
        gdf.loc[garbage_names, 'Data_Flags'] += "Garbage_VesselName;"
    else:
        garbage_names = pd.Series(False, index=gdf.index)

    # 3. VALIDATE AND CORRECT DIMENSIONS
    MAX_LENGTH, MAX_WIDTH, MAX_DRAFT = 500, 100, 30
    
    def get_invalid_dims(col, max_val):
        if col not in gdf.columns:
            return pd.Series(False, index=gdf.index)
        return gdf[col].isna() | (gdf[col] <= 0) | (gdf[col] > max_val)

    len_invalid = get_invalid_dims('Length', MAX_LENGTH)
    wid_invalid = get_invalid_dims('Width', MAX_WIDTH)
    drf_invalid = get_invalid_dims('Draft', MAX_DRAFT)

    gdf.loc[len_invalid, 'Data_Flags'] += "Invalid_Length;"
    gdf.loc[wid_invalid, 'Data_Flags'] += "Invalid_Width;"
    gdf.loc[drf_invalid, 'Data_Flags'] += "Invalid_Draft;"

    logging.info("Attempting to correct dimensions based on MMSI history...")
    for col, is_invalid in zip(['Length', 'Width', 'Draft'], [len_invalid, wid_invalid, drf_invalid]):
        if col in gdf.columns:
            gdf[f'valid_{col}'] = gdf[col].where(~is_invalid, np.nan)
            mmsi_medians = gdf.groupby('MMSI')[f'valid_{col}'].transform('median')
            corrected_mask = is_invalid & mmsi_medians.notna()
            gdf.loc[corrected_mask, col] = mmsi_medians[corrected_mask]
            gdf.loc[corrected_mask, 'Data_Flags'] += f"Corrected_{col};"
            gdf = gdf.drop(columns=[f'valid_{col}'])

    gdf['Data_Flags'] = gdf['Data_Flags'].str.rstrip(';')

    stats = {
        "Completely Invalid Tracklines (Zero Length/Bad Time)": trackline_invalid.mean(),
        "Heading Defaulted (511)": heading_511.mean(),
        "Corrupted Vessel Names": garbage_names.mean(),
        "Originally Invalid Lengths": len_invalid.mean(),
        "Originally Invalid Widths": wid_invalid.mean(),
        "Originally Invalid Drafts": drf_invalid.mean(),
        "Total Rows Flagged (Any Issue)": (gdf['Data_Flags'] != "").mean()
    }

    logging.info("--- Data Quality Statistics ---")
    for issue, pct in stats.items():
        logging.info(f"  {issue}: {pct:.2%} ({int(pct * initial_count)} records)")
    
    return gdf


def get_bluetopo_s3_files(bucket="ocs-dev-csdl-hydrohealth", prefix="low_res/"):
    """
    Finds all primary BlueTopo TIFFs across all ER folders in S3.
    """
    logging.info("Searching S3 for BlueTopo TIFFs...")
    fs = s3fs.S3FileSystem(anon=False) # Assumes EC2 IAM role / credentials are configured
    
    # Glob pattern to find all tiffs in ER_1 through ER_6 etc.
    pattern = f"{bucket}/{prefix}ER_*/model_variables/Prediction/pre_processed/BlueTopo/*/*.tiff"
    all_files = fs.glob(pattern)
    
    # Filter out secondary variables (catzoc, slope, uncertainty)
    primary_files = [f for f in all_files if not any(x in f for x in ['catzoc', 'slope', 'uncertainty'])]
    
    s3_urls = ["s3://" + f for f in primary_files]
    logging.info(f"Found {len(s3_urls)} primary BlueTopo TIFF tiles.")
    return s3_urls


def build_bathy_mosaic(s3_urls, target_crs="EPSG:32617", resolution=100):
    """
    Creates a seamless in-memory mosaic of S3 TIFFs reprojected to the target CRS and resolution.
    """
    logging.info(f"Building {resolution}m Bathymetry Mosaic in {target_crs} from S3...")
    src_files = []
    vrt_datasets = []
    
    # Open files and wrap them in a WarpedVRT to reproject on-the-fly during merge
    # CPL_DEBUG='OFF' helps quiet the benign GDAL INIT_DEST warnings
    with rasterio.Env(CPL_DEBUG='OFF'):  
        for url in s3_urls:
            try:
                src = rasterio.open(url)
                src_files.append(src)
                # Cast the source to our target CRS dynamically
                vrt = WarpedVRT(src, crs=target_crs, resampling=rasterio.enums.Resampling.bilinear)
                vrt_datasets.append(vrt)
            except Exception as e:
                logging.warning(f"Failed to open/warp {url}: {e}")

        if not vrt_datasets:
            raise ValueError("No valid S3 datasets could be opened.")

        # Merge all VRTs into a single numpy array
        logging.info(f"Merging and reprojecting {len(vrt_datasets)} tiles (this may take significant time/RAM)...")
        mosaic, out_trans = merge(vrt_datasets, res=(resolution, resolution), nodata=np.nan)
        
        # Cleanup open handles
        for vrt in vrt_datasets: vrt.close()
        for src in src_files: src.close()

    logging.info(f"Mosaic created with shape: {mosaic.shape}")
    return mosaic, out_trans


def calculate_underkeel_clearance(gpkg_path, bathy_mosaic, out_trans, target_crs="EPSG:32617"):
    """
    Rasterizes AIS tracklines by maximum draft, buffers by width, and calculates underkeel clearance.
    """
    logging.info("Calculating Underkeel Clearance from AIS data...")
    mosaic_shape = (bathy_mosaic.shape[1], bathy_mosaic.shape[2])
    
    # Initialize an array to hold the MAXIMUM draft seen in any given 100m cell
    max_draft_array = np.full(mosaic_shape, np.nan, dtype=np.float32)

    # list_layers returns [[layer_name, geom_type], ...]
    layers_info = pyogrio.list_layers(gpkg_path)
    layers = [info[0] for info in layers_info]
    
    for layer in layers:
        logging.info(f"Rasterizing Drafts for AIS layer: {layer}")
        gdf = pyogrio.read_dataframe(gpkg_path, layer=layer, use_arrow=True)
        
        # Filter for valid Draft and Width
        gdf = gdf.dropna(subset=['Draft', 'Width'])
        gdf = gdf[(gdf['Draft'] > 0) & (gdf['Width'] > 0)]
        
        if gdf.empty:
            continue
            
        # 1. Reproject to match Bathymetry
        gdf = gdf.to_crs(target_crs)
        
        # 2. Buffer the trackline by half the width of the ship to create a swept path polygon
        gdf['geometry'] = gdf.geometry.buffer(gdf['Width'] / 2.0)
        
        # 3. Sort ascending by draft. When rasterizing, later shapes overwrite earlier ones.
        # This ensures the HIGHEST draft acts as the final value in a cell (Max Draft).
        gdf = gdf.sort_values(by='Draft', ascending=True)
        
        shapes = ((geom, float(draft)) for geom, draft in zip(gdf.geometry, gdf['Draft']))
        
        # Rasterize this month's data
        layer_max_draft = rasterize(
            shapes=shapes,
            out_shape=mosaic_shape,
            transform=out_trans,
            fill=np.nan,
            dtype=np.float32
        )
        
        # Update the global maximum draft array
        max_draft_array = np.fmax(max_draft_array, layer_max_draft)

    # Calculate Clearance: Bathymetry (Depth) - Max Draft
    logging.info("Computing final clearance values (Bathymetry - Max Draft)...")
    
    # bathy_mosaic[0] accesses the 2D array from the 3D (band, height, width) mosaic output
    ukc_array = bathy_mosaic[0] - max_draft_array
    
    return ukc_array


def main():
    # --- CONFIGURATION ---
    # Normalize paths for Windows/Linux interoperability
    base_dir = Path(r"/home/aubrey.mccutchen.lx/Repos/hydro_health/inputs/AIS_data")
    
    # Update these filenames to match your exact GDB names
    input_gdb = base_dir / "AIS_2024.gdb" / "AIS_2024.gdb"
    output_gpkg = base_dir / "processed_ais.gpkg"
    
    # S3 Output configuration
    s3_bucket = "ocs-dev-csdl-hydrohealth"
    s3_ukc_key = "low_res/underkeel_clearance_mosaic_100m.tiff"
    local_ukc_tiff = base_dir / "underkeel_clearance_mosaic_100m.tiff"

    target_crs = "EPSG:32617"
    resolution = 100

    if not input_gdb.exists():
        logging.error(f"Input GDB not found: {input_gdb}")
        # return # Uncomment to strictly enforce local GDB cleaning first

    # ==========================================
    # 1. PROCESS AND CLEAN AIS DATA
    # ==========================================
    if input_gdb.exists():
        try:
            # Use pyogrio to list layers natively (returns array of [layer_name, geom_type])
            layers_info = pyogrio.list_layers(input_gdb)
            layers = [info[0] for info in layers_info]
            logging.info(f"Found {len(layers)} layers in GDB.")
            
            for layer in layers:
                try:
                    # use_arrow=True is highly recommended for stability and speed
                    gdf = pyogrio.read_dataframe(input_gdb, layer=layer, use_arrow=True)
                    processed_gdf = process_ais_layer(gdf, layer)
                    pyogrio.write_dataframe(processed_gdf, output_gpkg, layer=layer, driver="GPKG")
                except Exception as pyo_err:
                    logging.error(f"Failed to process layer '{layer}' with Pyogrio: {pyo_err}")
                    logging.info(f"Attempting fallback read for '{layer}' using standard GeoPandas...")
                    try:
                        # Fallback using Geopandas fiona engine if the GDB has tricky geometry
                        gdf = gpd.read_file(input_gdb, layer=layer, engine="fiona")
                        processed_gdf = process_ais_layer(gdf, layer)
                        processed_gdf.to_file(output_gpkg, layer=layer, driver="GPKG")
                    except Exception as gpd_err:
                        logging.error(f"Fallback also failed for '{layer}'. Skipping. Error: {gpd_err}")
                        
        except Exception as e:
            logging.error(f"Failed to list or process GDB layers: {e}")

    # ==========================================
    # 2. BUILD BATHYMETRY MOSAIC FROM S3
    # ==========================================
    s3_urls = get_bluetopo_s3_files(bucket=s3_bucket, prefix="low_res/")
    if not s3_urls:
        logging.error("No BlueTopo files found. Exiting.")
        return
        
    bathy_mosaic, out_trans = build_bathy_mosaic(s3_urls, target_crs=target_crs, resolution=resolution)

    # ==========================================
    # 3. CALCULATE UNDERKEEL CLEARANCE
    # ==========================================
    # We use the cleaned GeoPackage for the clearance calculations
    if output_gpkg.exists():
        ukc_array = calculate_underkeel_clearance(output_gpkg, bathy_mosaic, out_trans, target_crs=target_crs)
        
        # Save to local GeoTIFF
        logging.info(f"Saving Underkeel Clearance GeoTIFF locally to {local_ukc_tiff}...")
        mosaic_shape = ukc_array.shape
        with rasterio.open(
            local_ukc_tiff, 'w',
            driver='GTiff',
            height=mosaic_shape[0],
            width=mosaic_shape[1],
            count=1,
            dtype=np.float32,
            crs=target_crs,
            transform=out_trans,
            nodata=np.nan
        ) as dst:
            dst.write(ukc_array, 1)
            
        # ==========================================
        # 4. UPLOAD TO S3
        # ==========================================
        logging.info(f"Uploading output to s3://{s3_bucket}/{s3_ukc_key} ...")
        s3_client = boto3.client('s3')
        s3_client.upload_file(str(local_ukc_tiff), s3_bucket, s3_ukc_key)
        
        logging.info("Pipeline Complete!")
    else:
        logging.error("Processed AIS GPKG not found. Cannot calculate underkeel clearance.")

if __name__ == "__main__":
    main()