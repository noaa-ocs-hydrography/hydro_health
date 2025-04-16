import os
import requests
import zipfile
import time
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
import fiona
import patoolib


gpkg_path = 'path/to/your/geopackage.gpkg'
output_folder = 'path/to/output_folder'

base_path = r'C:\Users\aubrey.mccutchan\Documents\HydroHealth\AIS_data'
output_folder = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\ais_data\draft_rasters'
mask_file = r'N:\HSD\Projects\HSD_DATA\NHSP_2_0\HH_2024\working\Pilot_model\prediction.mask.tif'


BASE_URL = 'https://marinecadastre.gov/downloads/data/ais/ais{year}/AISVesselTracks{year}.zip'
DOWNLOAD_DIR = r'C:\Users\aubrey.mccutchan\Repo\hydro_health\hydro_health\inputs\ais_data'

def download_and_extract_ais_files(start_year, end_year):
    """Download and extract AIS Vessel Tracks files for the given range of years."""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    for year in range(start_year, end_year + 1):
        url = BASE_URL.format(year=year)
        zip_file_path = os.path.join(DOWNLOAD_DIR, f"AISVesselTracks{year}.zip")
        extract_dir = os.path.join(DOWNLOAD_DIR, f"AISVesselTracks{year}")

        # Skip if zip already exists
        if os.path.exists(zip_file_path):
            print(f"Skipping {year}: zip file already downloaded at {zip_file_path}")
            continue

        print(f"Downloading {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(zip_file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Successfully downloaded: {zip_file_path}")

            if not os.path.exists(extract_dir):
                os.makedirs(extract_dir)

            print(f"Extracting {zip_file_path} to {extract_dir}...")
            patoolib.extract_archive(zip_file_path, outdir=extract_dir)

            print(f"Successfully extracted: {zip_file_path} to {extract_dir}")

            os.remove(zip_file_path)
            print(f"Deleted ZIP file: {zip_file_path}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")
        except zipfile.BadZipFile as e:
            print(f"Failed to extract {zip_file_path}: {e}")

def combine_year_range_data(base_path, start_year, end_year):
    year_folders = []
    
    for folder_name in os.listdir(base_path):
        if folder_name.startswith("AISVesselTracks"):
            try:
                year = int(folder_name.replace("AISVesselTracks", ""))
                if start_year <= year <= end_year:
                    year_folders.append(os.path.join(base_path, folder_name))
            except ValueError:
                print(f"Skipping folder with invalid year: {folder_name}")
    
    combined_gdf = gpd.GeoDataFrame()
    for folder in year_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".gdb") or file.endswith(".gpkg"):
                    file_path = os.path.join(root, file)
                    layers = fiona.listlayers(file_path)
                    for layer in layers:
                        gdf = gpd.read_file(file_path, layer=layer)
                        
                        if 'Draft' not in gdf.columns:
                            print(f"Skipping layer '{layer}' in file without 'Draft' field: {file_path}")
                            continue
                        
                        gdf = gdf.dropna(subset=['Draft'])
                        
                        combined_gdf = pd.concat([combined_gdf, gdf], ignore_index=True)

    return combined_gdf


def create_rasters_for_year_range(base_path, output_folder, mask_file, start_year, end_year, resolution=5):
    """
    Generates rasters for unique Draft values over a specified year range
    """
    combined_gdf = combine_year_range_data(base_path, start_year, end_year)
    combined_gdf['Draft'] = combined_gdf['Draft'].astype(float)
    combined_gdf['Width'] = combined_gdf['Width'].astype(float)

    with rasterio.open(mask_file) as src:
        mask = src.read(1)
        transform = src.transform
        bounds = src.bounds
        crs = src.crs

    width = int((bounds.right - bounds.left) / resolution)
    height = int((bounds.top - bounds.bottom) / resolution)

    unique_drafts = combined_gdf['Draft'].unique()
    
    for draft_value in unique_drafts:
        draft_gdf = combined_gdf[combined_gdf['Draft'] == draft_value]
        
        shapes = (
            (row.geometry.buffer(row.Width / 2), 1) 
            for _, row in draft_gdf.iterrows()
        )
        
        raster = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,  
            dtype=np.uint8,
        )
        
        raster[mask == 0] = 0
        
        output_raster = f"{output_folder}/draft_{draft_value}_years_{start_year}_{end_year}.tif"
        with rasterio.open(
            output_raster, 'w',
            driver='GTiff',
            height=raster.shape[0],
            width=raster.shape[1],
            count=1,
            dtype=np.uint8,
            crs=crs,
            transform=transform
        ) as dst:
            dst.write(raster, 1)
        
        print(f"Saved raster for Draft {draft_value} (Years {start_year}-{end_year}): {output_raster}")

if __name__ == "__main__":

    start_time = time.time()

    start_year = 2023
    end_year = 2023
    download_and_extract_ais_files(start_year, end_year)

    start_year = 2023
    end_year = 2023
    create_rasters_for_year_range(base_path, output_folder, mask_file, start_year, end_year)

    end_time = time.time()
    print(f'Execution time: {(end_time - start_time)/60} minutes')
