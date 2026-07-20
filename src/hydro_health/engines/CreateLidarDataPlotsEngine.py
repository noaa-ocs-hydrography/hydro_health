import os
import re
import math
import warnings
import datetime
import logging
import numpy as np
import geopandas as gpd
import yaml
import pathlib
import io
import glob
import s3fs
from itertools import combinations

from shapely.geometry import shape
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.features import shapes
from rasterio.warp import calculate_default_transform, reproject

import dask
from dask.distributed import Client

from osgeo import gdal, osr

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item, get_environment

# Set up global logging config to write to both a log file and the console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("lidar_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mute noisy third-party loggers from spamming the INFO level
logging.getLogger('botocore').setLevel(logging.WARNING)
logging.getLogger('aiobotocore').setLevel(logging.WARNING)
logging.getLogger('s3fs').setLevel(logging.WARNING)
logging.getLogger('fsspec').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('rasterio').setLevel(logging.WARNING)

# Suppress benign Matplotlib colormap overflow warnings caused by hidden masked array values
warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib.colors")

# Ensure GDAL env vars are set
os.environ['GDAL_MEM_ENABLE_OPEN'] = 'YES'
os.environ["GDAL_CACHEMAX"] = "1024" # Reduced to 1GB to prevent OOM with multiple workers
# Required to allow GDAL to write directly to S3 via /vsis3/
os.environ['CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE'] = 'YES'
# Add robust HTTP retry for S3 reads (helps with transient TIFFReadEncodedTile errors)
os.environ['GDAL_HTTP_MAX_RETRY'] = '5'
os.environ['GDAL_HTTP_RETRY_DELAY'] = '3'

# Ensure GDAL throws Python exceptions so we can catch them in our try/except blocks
gdal.UseExceptions()

INPUTS = pathlib.Path(__file__).parents[3] / 'inputs' / 'lookups'


def _process_all_vrts(params):
    """Processes a single VRT file. This function will be executed by a Dask worker."""

    vrt_file, output_dir, target_resolution_m, resampling_method, creation_options = params
        
    # Skip resampling for NCEI datasets
    if "NCEI" in str(vrt_file).upper():
        logger.info(f"Skipping resampling for NCEI dataset: {vrt_file}")
        return

    engine = CreateLidarDataPlotsEngine()

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if engine.is_aws:
                # Use standard string manipulation for S3 paths to prevent os.path from adding backslashes
                base_name = str(vrt_file).replace('\\', '/').split('/')[-1]
                resampled_base_name = base_name.replace('.vrt', '_resampled.tif')
                output_dir_clean = str(output_dir).replace('\\', '/')
                output_filename = f"{output_dir_clean.rstrip('/')}/{resampled_base_name}"
                if not output_filename.startswith("s3://"): output_filename = f"s3://{output_filename}"
                
                # Convert to GDAL VSI paths
                gdal_input = engine._get_gdal_path(vrt_file)

                # Use local EC2 temporary file instead of direct vsis3 write to bypasses the S3 5GB upload limit!
                # Avoid /tmp/ as it is often a tmpfs (RAM disk) on Linux which causes OOM crashes.
                gdal_output = os.path.join(os.getcwd(), resampled_base_name)
                
                # Check existence using s3fs
                exists = engine.fs.exists(output_filename.replace("s3://", ""))
            else:
                base_name = os.path.basename(vrt_file)
                output_filename = os.path.join(output_dir, base_name.replace(".vrt", "_resampled.tif"))
                gdal_input = vrt_file
                gdal_output = output_filename
                exists = os.path.exists(output_filename)

            if exists:
                skip_message = f"Output file {output_filename} already exists. Skipping {vrt_file}."
                logger.info(skip_message)
                return

            logger.info(f"Resampling: {vrt_file} (Attempt {attempt + 1}/{max_retries})")

            # --- DYNAMIC RESOLUTION FIX ---
            # Set default resolution in degrees (approximate conversion)
            degree_res = target_resolution_m / 111320.0
            actual_x_res = degree_res
            actual_y_res = degree_res
            try:
                open_path = engine._get_s3_path(vrt_file)
                with rasterio.open(open_path) as src:
                    if src.crs and src.crs.is_projected:
                        actual_x_res = float(target_resolution_m)
                        actual_y_res = float(target_resolution_m)
                        logger.info(f"[{base_name}] Projected CRS detected! Swapping degree resolution for {target_resolution_m} meter resolution to prevent memory exhaustion.")
            except Exception:
                pass # Failsafe: if rasterio can't read it, fall back to default behavior
            # ------------------------------

            gdal.Warp(
                gdal_output,
                gdal_input,
                xRes=actual_x_res,
                yRes=actual_y_res,
                resampleAlg=resampling_method,
                creationOptions=creation_options,
                warpOptions=[],   
                multithread=False, # Disable internal GDAL multithreading to avoid thread-explosion with Dask
                warpMemoryLimit=1024 # Reduced to 1GB to safely accommodate workers on a 30GB system
            )

            if engine.is_aws:
                logger.info(f"Uploading resampled file to S3: {output_filename}...")
                engine.fs.put(gdal_output, output_filename.replace("s3://", ""))
                os.remove(gdal_output)

            logger.info(f"Successfully resampled {vrt_file} to {output_filename}")
            return # Exit successfully

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} of {max_retries} failed for {vrt_file}: {e}")
            # Ensure the local temporary file is deleted even if the upload or GDAL process fails
            if engine.is_aws and 'gdal_output' in locals() and os.path.exists(gdal_output):
                os.remove(gdal_output)
            
            if attempt == max_retries - 1:
                logger.error(f"Final failure for {vrt_file}. The underlying TIFF may be physically corrupted or truncated on S3.")


class CreateLidarDataPlotsEngine(Engine):
    """Class to hold the logic for processing the Lidar Data Plots"""

    def __init__(self):
        super().__init__()
        self.config = None
        self.client = None
        self.target_resolution_m = 100.0  # Changeable target resolution for resampling in meters
        self.year_datasets = None
        self.dataset_report_data = {}
        self.invalid_format_files = [] # TRACK REGEX FAILURES
        self.corrupted_files = []      # TRACK CORRUPTED/EMPTY TIFS
        self.missing_config_datasets = [] # TRACK MISSING TIFS CONFIGURED IN YAML
        self.fs = s3fs.S3FileSystem(anon=False)
        self.is_aws = (get_environment() == 'aws')

    def _get_s3_path(self, path: pathlib.Path) -> str:
        """Helper to ensure path has s3:// prefix if on AWS"""
        
        if self.is_aws:
            path_str = str(path).replace('\\', '/')
            if not path_str.startswith('s3://'):
                return f"s3://{path_str.lstrip('/')}"
            return path_str
        return str(path)

    def _get_gdal_path(self, path: pathlib.Path) -> str:
        """Helper to convert path to GDAL VSI format if on AWS"""

        path_str = str(path)
        if self.is_aws:
            path_str = path_str.replace('\\', '/')
            if path_str.startswith('s3://'):
                path_str = path_str.replace('s3://', '')
            if not path_str.startswith('/vsis3/'):
                path_str = f"/vsis3/{path_str.lstrip('/')}"
        return path_str

    def calculateExtent(self, paths: list[pathlib.Path], target_crs: str) -> tuple:
        """
        Calculates a common extent for a list of raster files.
        param list paths: List of paths to raster files.
        param str target_crs: Target coordinate reference system (e.g., 'EPSG:3857').
        return: Affine transform, extent (left, right, bottom, top), and shape (height, width) of the common grid.
        """

        local_left, local_bottom, local_right, local_top = np.inf, np.inf, -np.inf, -np.inf
        reference_res = None

        for path in paths:
            # Rasterio handles s3:// paths automatically if environment is set up
            # Ensure path has s3:// prefix if AWS
            open_path = self._get_s3_path(path)
            
            try:
                with rasterio.open(open_path) as src:
                    transform, width, height = calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds)
                    
                    left, top = transform.c, transform.f
                    right = left + width * transform.a
                    bottom = top + height * transform.e

                    local_left = min(local_left, left)
                    local_bottom = min(local_bottom, bottom)
                    local_right = max(local_right, right)
                    local_top = max(local_top, top)

                    if reference_res is None:
                        reference_res = (abs(transform.a), abs(transform.e))
            except Exception as e:
                logger.error(f"Error calculating extent for {open_path}: {e}")

        if reference_res is None:
            return None, None, None

        xres, yres = reference_res
        dst_width = int((local_right - local_left) / xres)
        dst_height = int((local_top - local_bottom) / yres)
        common_transform = from_bounds(local_left, local_bottom, local_right, local_top, dst_width, dst_height)
        common_extent = (local_left, local_right, local_bottom, local_top)
        
        return common_transform, common_extent, (dst_height, dst_width)

    def extract_date_from_metadata(self, metadata_path: str) -> list[str]:
        """
        Extracts the first date in YYYY-MM format from a metadata file.
        param str metadata_path: Path to the metadata file.
        return list: Date(s) found in the metadata file.
        """

        try:
            content = ""
            if self.is_aws:
                # Use s3fs to read from S3
                s3_path = metadata_path.replace("s3://", "").replace('\\', '/')
                with self.fs.open(s3_path, 'r') as f:
                    content = f.read()
            else:
                with open(metadata_path, 'r') as f:
                    content = f.read()

            date_matches_ym = re.findall(r'(?:18|19|20)\d{2}-\d{2}', content)
            if date_matches_ym:
                return sorted(list(set(date_matches_ym)))

            date_matches_y = re.findall(r'(?:18|19|20)\d{2}', content)
            if date_matches_y:
                return sorted(list(set(date_matches_y)))

            return 'Missing'
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Error reading metadata file {metadata_path}: {e}")
        return None  

    def finalizeFigure(self, fig, axes, im, cbar_label, suptitle, pdf, dpi, show_cbar=False, show_overlap_legend=False) -> None:
        """
        Adds final touches to a figure and saves it to the PDF document.
        param object fig: Matplotlib figure object.
        param object axes: Matplotlib axes object.
        param object im: Matplotlib image object.
        param str cbar_label: Label for the colorbar.
        param str suptitle: Super title for the figure.
        param object pdf: Matplotlib PdfPages object.
        param int dpi: DPI for saving the figure.
        param bool show_cbar: Whether to display the colorbar.
        param bool show_overlap_legend: Whether to include the overlap area in the legend.
        """

        if show_cbar and im is not None and im.get_array() is not None and im.get_array().count() > 0:
            cbar = fig.colorbar(im, ax=axes.ravel().tolist())
            cbar.set_label(cbar_label, fontsize=8, rotation=270, labelpad=20)

        legend_lines = [
            Line2D([0], [0], color='gray', lw=1.5, label='50m Isobath')
        ]
        if show_overlap_legend:
            legend_lines.append(Line2D([0], [0], color='red', lw=2.0, label='Overlap Area'))
            
        fig.legend(handles=legend_lines, loc='lower center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=len(legend_lines), fontsize=10)
        
        plt.suptitle(suptitle, fontsize=16, fontweight='bold')
        
        pdf.savefig(fig, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Appended plot to PDF: {suptitle}")

    def getYearDatasets(self, input_folder: pathlib.Path, config: dict) -> dict:
        """Obtain dictionary of datasets for all years"""

        year_datasets = {}
        self.dataset_report_data = {}
        self.invalid_format_files = [] # Reset trackers
        self.corrupted_files = []      # Reset trackers
        self.missing_config_datasets = [] # Reset trackers
        
        found_config_keys = set()
        
        # Matplotlib MathText compatible bold formatting tags
        bullet_missing = r"$\mathbf{Missing}$"
        bullet_manual = r"$\mathbf{Manual\ Download}$"
        
        # Determine base folder for DigitalCoast metadata
        if self.is_aws:
            # Assuming structure: bucket/path/to/Resampled -> bucket/path/to/DigitalCoast
            input_folder_str = str(input_folder).replace('\\', '/').rstrip('/')
            parent_folder = input_folder_str.replace("s3://", "").rsplit('/', 1)[0]
            base_folders = [
                f"{parent_folder}/DigitalCoast", 
                f"{parent_folder}/Digital_Coast_Manual_Downloads",
                f"{parent_folder}/DigitalCoast_manual_downloads"
            ]
            files_list = self.fs.ls(input_folder_str.replace("s3://", ""))
            filenames = [f.replace('\\', '/').split('/')[-1] for f in files_list]
        else:
            parent_folder = os.path.dirname(input_folder)
            base_folders = [
                os.path.join(parent_folder, 'DigitalCoast'), 
                os.path.join(parent_folder, 'Digital_Coast_Manual_Downloads'),
                os.path.join(parent_folder, 'DigitalCoast_manual_downloads')
            ]
            filenames = [f for f in os.listdir(input_folder) if f.endswith('.tif')]

        # --- Resolve VRT input directories for orphan checking ---
        raw_vrt_dir = get_config_item("LIDAR_PLOTS", "INPUT_VRTS")
        raw_vrt_dir_str = str(raw_vrt_dir)
        parent_vrt = "/".join(raw_vrt_dir_str.replace('\\', '/').rstrip('/').split('/')[:-1])
        
        manual_vrt_dir_str_1 = f"{parent_vrt}/Digital_Coast_Manual_Downloads"
        manual_vrt_dir_str_2 = f"{parent_vrt}/DigitalCoast_manual_downloads"

        vrt_dirs_clean = []
        if self.is_aws:
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            for d in [raw_vrt_dir_str, manual_vrt_dir_str_1, manual_vrt_dir_str_2]:
                d_clean = d.replace('\\', '/').replace("s3://", "")
                if not d_clean.startswith(bucket):
                    vrt_dirs_clean.append(f"{bucket}/{d_clean.lstrip('/')}")
                else:
                    vrt_dirs_clean.append(d_clean)
        else:
            vrt_dirs_clean = [raw_vrt_dir_str, manual_vrt_dir_str_1, manual_vrt_dir_str_2]
        # ---------------------------------------------------------

        for filename in filenames:
            if not filename.endswith('.tif'):
                continue

            # Flag all found keys IMMEDIATELY so corrupted files aren't double-counted as "Missing" later
            for config_key in config.keys():
                if config_key in filename:
                    found_config_keys.add(config_key)

            # Construct full path EARLY so we can test the file
            if self.is_aws:
                input_folder_clean = str(input_folder).replace('\\', '/')
                full_path = f"{input_folder_clean.rstrip('/')}/{filename}"
                # Ensure s3 prefix
                if not full_path.startswith("s3://"): full_path = f"s3://{full_path}"
            else:
                full_path = os.path.join(input_folder, filename)

            # --- Orphaned TIF check (delete if no VRT exists) ---
            vrt_filename = filename.replace('_resampled.tif', '.vrt')
            is_orphaned = True
            
            if self.is_aws:
                for d in vrt_dirs_clean:
                    if self.fs.exists(f"{d.rstrip('/')}/{vrt_filename}"):
                        is_orphaned = False
                        break
            else:
                for d in vrt_dirs_clean:
                    if os.path.exists(os.path.join(d, vrt_filename)):
                        is_orphaned = False
                        break
                        
            if is_orphaned:
                logger.info(f"Deleting orphaned mosaic (no corresponding VRT found): {filename}")
                try:
                    if self.is_aws:
                        self.fs.rm(full_path.replace("s3://", ""))
                    else:
                        os.remove(full_path)
                except Exception as e:
                    logger.warning(f"Could not delete orphaned file {filename}: {e}")
                continue
            # ----------------------------------------------------

            # 1. Regex Validation & Data Extraction
            if not filename.lower().endswith('_resampled.tif'):
                self.invalid_format_files.append(filename)
                continue
                
            # Extract year safely (e.g. _2015_)
            year_match = re.search(r'_((?:18|19|20)\d{2})[_\.]', filename)
            year_from_filename = year_match.group(1) if year_match else None
            
            # Extract 5-digit code/ID safely at the end before _resampled.tif
            id_match = re.search(r'_(\d+)_resampled\.tif$', filename, re.IGNORECASE)
            unique_id = id_match.group(1) if id_match else None
            
            dataset_info = re.sub(r'_resampled\.tif$', '', filename, flags=re.IGNORECASE)

            # 2. Corruption / Empty Data Check
            is_corrupted = False
            try:
                # Do NOT use pathlib.Path() here on AWS strings as it strips the double slashes from URIs
                # e.g., pathlib converts "s3://bucket" -> "s3:/bucket", which instantly fails the rasterio check.
                with rasterio.open(full_path) as src:
                    # Check if it has dimensions and bands
                    if src.width == 0 or src.height == 0 or src.count == 0:
                        is_corrupted = True
            except Exception as e:
                # Added print statement so we don't swallow silent failures anymore
                logger.warning(f"Validation check failed for {full_path}: {e}")
                is_corrupted = True

            if is_corrupted:
                self.corrupted_files.append(filename)
                continue # File is unreadable or empty, log it and skip

            # Extract Dataset Name safely
            clean_info = dataset_info
            if year_from_filename:
                clean_info = clean_info.replace(f'_{year_from_filename}', '')
            if unique_id:
                clean_info = re.sub(f'_{unique_id}$', '', clean_info)

            if "USACE" in dataset_info.upper() and "NCMP" in dataset_info.upper():
                dataset_name = "USACE"
            else:
                dataset_name = clean_info.split('_')[-1]

            # --- GLOBAL SKIP FOR NCEI DATASETS ---
            # Intercept and ignore any NCEI dataset so they never enter the logs, unused lists, or processing logic
            is_ncei = "NCEI" in dataset_name.upper() or "NCEI" in filename.upper()
            if is_ncei:
                continue
            # ------------------------------------

            grouping_year = year_from_filename if year_from_filename else 'No Year Found'
            
            # Using dictionaries to group filenames natively instead of deduping strings
            if grouping_year not in self.dataset_report_data:
                self.dataset_report_data[grouping_year] = {'used': {}, 'unused': {}}

            is_unspecified = True
            should_skip = False
            is_manual = False
            is_use_true = False
            matched_key = None
            dataset_note = None
            
            for config_key, dataset_settings in config.items():
                if config_key in filename:
                    is_unspecified = False
                    matched_key = config_key
                    
                    if dataset_settings:
                        # Robust string and bool casting for YAML parsing quirks
                        use_val = dataset_settings.get('use')
                        if use_val is not None:
                            if str(use_val).strip().lower() in ['false', 'no', '0']:
                                logger.info(f"Skipping '{filename}' due to config for component '{config_key}'.")
                                should_skip = True
                            elif str(use_val).strip().lower() in ['true', 'yes', '1']:
                                is_use_true = True
                        
                        manual_val = dataset_settings.get('manual')
                        if manual_val is not None and str(manual_val).strip().lower() in ['true', 'yes', '1']:
                            is_manual = True
                            
                        dataset_note = dataset_settings.get('note')
                            
            if is_unspecified:
                display_name = f"{dataset_name} (Unlisted in Config - Defaulting to Used)"
            else:
                display_name = f"{dataset_name} (Key: {matched_key})"
            
            # Append flags so they render cleanly in the PDF table
            if is_manual:
                display_name += f"\n• {bullet_manual}"
            
            if should_skip:
                if dataset_note:
                    display_name += f"\n• Note: {dataset_note}"
                if display_name not in self.dataset_report_data[grouping_year]['unused']:
                    self.dataset_report_data[grouping_year]['unused'][display_name] = []
                self.dataset_report_data[grouping_year]['unused'][display_name].append(filename)
                continue

            if display_name not in self.dataset_report_data[grouping_year]['used']:
                self.dataset_report_data[grouping_year]['used'][display_name] = []
            self.dataset_report_data[grouping_year]['used'][display_name].append(filename)
            
            acquisition_date = None
            
            # Metadata search logic
            if self.is_aws:
                for base_folder in base_folders:
                    if self.fs.exists(base_folder):
                        subfolders = self.fs.ls(base_folder)
                        for folder_path in subfolders:
                            folder_name = folder_path.replace('\\', '/').split('/')[-1]
                            
                            # Safely prevent 'NoneType' string errors by checking if variables populated
                            if unique_id and f'_{unique_id}' in folder_name:
                                if not year_from_filename or year_from_filename in folder_name:
                                    metadata_path = f"s3://{folder_path}/metadata.txt"
                                    acquisition_date = self.extract_date_from_metadata(metadata_path)
                                    if acquisition_date:
                                        break
                        if acquisition_date:
                            break
                    else:
                        pass # Silently skip missing manual folders to avoid clutter
            else:
                for base_folder in base_folders:
                    if os.path.isdir(base_folder):
                        for folder_name in os.listdir(base_folder):
                            potential_path = os.path.join(base_folder, folder_name)
                            if os.path.isdir(potential_path) and unique_id and f'_{unique_id}' in folder_name:
                                if not year_from_filename or year_from_filename in folder_name:
                                    metadata_path = os.path.join(potential_path, 'metadata.txt')
                                    acquisition_date = self.extract_date_from_metadata(metadata_path)
                                    if acquisition_date:
                                        break
                        if acquisition_date:
                            break
                    else:
                        pass # Silently skip missing manual folders to avoid clutter
            
            grouping_year = year_from_filename if year_from_filename else 'No Year Found'
            title_str = year_from_filename if year_from_filename else 'Date Not in Filename'

            if grouping_year not in year_datasets:
                year_datasets[grouping_year] = []

            year_datasets[grouping_year].append({
                'path': full_path,
                'dataset_name': dataset_name,
                'date': acquisition_date,
                'title': title_str,
                'is_use_true': is_use_true
            })
            
        # 3. Identify missing items configured to be used, but not found in the input folder
        for config_key, dataset_settings in config.items():
            if config_key not in found_config_keys:
                
                if dataset_settings:
                    use_val = dataset_settings.get('use')
                    # If 'use' is explicitly set to True
                    is_use_true = str(use_val).strip().lower() in ['true', 'yes', '1'] if use_val is not None else False
                    
                    if is_use_true:
                        if "USACE" in config_key and "NCMP" in config_key:
                            dataset_name = "USACE"
                        else:
                            # Clean up config key string if possible to approximate dataset name
                            dataset_name = re.sub(r'_(?:18|19|20)\d{2}.*', '', config_key)
                            
                        is_ncei = "NCEI" in dataset_name.upper() or "NCEI" in config_key.upper()
                        if is_ncei:
                            continue
                            
                        self.missing_config_datasets.append(config_key)
                        
                        # Try to extract year from the config key to group it in the right table row
                        year_match = re.search(r'((?:18|19|20)\d{2})', config_key)
                        grouping_year = year_match.group(1) if year_match else 'Unknown Year'
                        
                        if grouping_year not in self.dataset_report_data:
                            self.dataset_report_data[grouping_year] = {'used': {}, 'unused': {}}
                            
                        display_name = f"{dataset_name} (Key: {config_key})\n• {bullet_missing}"
                        
                        manual_val = dataset_settings.get('manual')
                        is_manual_missing = str(manual_val).strip().lower() in ['true', 'yes', '1'] if manual_val is not None else False
                        if is_manual_missing:
                            display_name += f"\n• {bullet_manual}"
                            
                        dataset_note = dataset_settings.get('note')
                        if dataset_note:
                            display_name += f"\n• Note: {dataset_note}"
                            
                        if display_name not in self.dataset_report_data[grouping_year]['used']:
                            self.dataset_report_data[grouping_year]['used'][display_name] = []
            
        return year_datasets
    
    def generate_dataset_report(self, pdf, ecoregion_name) -> None:
        """Generates a condensed PDF report detailing which datasets were used and skipped per year."""
        if not self.dataset_report_data and not self.invalid_format_files and not self.corrupted_files and not self.missing_config_datasets:
            logger.warning("No dataset data available to generate report.")
            return

        # Calculate absolute file totals by traversing the stored lists
        total_used_count = sum(sum(len(f_list) for f_list in self.dataset_report_data[y]['used'].values()) for y in self.dataset_report_data)
        total_unused_count = sum(sum(len(f_list) for f_list in self.dataset_report_data[y]['unused'].values()) for y in self.dataset_report_data)
        total_err_regex = len(self.invalid_format_files)
        total_err_corrupt = len(self.corrupted_files)
        total_missing_configs = len(self.missing_config_datasets)
        
        # This will perfectly match the count of .tif files in the directory
        total_count = total_used_count + total_unused_count + total_err_regex + total_err_corrupt

        def _format_cell_text(d_name, f_list):
            parts = d_name.split('\n', 1)
            f_count_str = f"[{len(f_list)} file{'s' if len(f_list) != 1 else ''}]"
            res = f"{parts[0]} {f_count_str}"
            if len(parts) > 1:
                res += f"\n{parts[1]}"
            return res

        cell_text = []
        
        for year in sorted(self.dataset_report_data.keys()):
            used_list = [_format_cell_text(d_name, f_list) for d_name, f_list in sorted(self.dataset_report_data[year]['used'].items())]
            unused_list = [_format_cell_text(d_name, f_list) for d_name, f_list in sorted(self.dataset_report_data[year]['unused'].items())]
            
            max_len = max(len(used_list), len(unused_list))
            if max_len == 0:
                cell_text.append([year, "None", "None"])
                continue
                
            # Chunk large years so they don't break Matplotlib's single-page cell rendering limit
            # Increased chunk size from 6 to 15 so years with lots of data (like 2016) don't get arbitrarily split into two boxes
            chunk_size = 15 
            for i in range(0, max_len, chunk_size):
                u_chunk = used_list[i:i+chunk_size]
                un_chunk = unused_list[i:i+chunk_size]
                
                used_str = "\n".join(u_chunk) if u_chunk else ("None" if i == 0 else "")
                unused_str = "\n".join(un_chunk) if un_chunk else ("None" if i == 0 else "")
                
                display_year = year if i == 0 else f"{year} (cont.)"
                cell_text.append([display_year, used_str, unused_str])
            
        col_labels = ["Year", "Used Datasets", "Unused / Skipped Datasets"]
        
        # --- DYNAMIC PAGINATION LOGIC ---
        # Matplotlib tables break down when rendering too much text on one axis.
        # We must paginate based on the total number of lines, not just a fixed number of rows.
        max_lines_per_page = 35 
        pages_data = []
        current_page = []
        current_lines = 0
        
        for row in cell_text:
            # Calculate maximum vertical lines this specific row will occupy
            lines_in_row = max(len(str(row[1]).split('\n')), len(str(row[2]).split('\n')), 1)
            
            # If adding this row exceeds max lines and we already have rows on this page, start a new page
            if current_lines + lines_in_row > max_lines_per_page and current_page:
                pages_data.append(current_page)
                current_page = [row]
                current_lines = lines_in_row
            else:
                current_page.append(row)
                current_lines += lines_in_row
                
        if current_page:
            pages_data.append(current_page)
            
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        
        for i, page_data in enumerate(pages_data):
            # Portrait orientation (8.5 x 11) using exact axes dimensions to control layout
            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_axes([0.05, 0.05, 0.9, 0.81]) # Leave top 19% for title padding
            ax.axis('off')
            
            title_suffix = f" (Page {i + 1})" if len(pages_data) > 1 else ""
            
            # Formatted to break resolution onto its own line to prevent overflow clipping
            report_title = (
                f"{ecoregion_name} Lidar Datasets Report{title_suffix} | Generated: {current_date}\n"
                f"Resolution: {self.target_resolution_m}m\n"
                f"Total .tif Files: {total_count} ({total_used_count} Used, {total_unused_count} Skipped, {total_err_regex} Regex Failed, {total_err_corrupt} Corrupted) | {total_missing_configs} Missing\n"
                f"* Note: NCEI providers are not used *"
            )
            
            # Use fig.suptitle to place the text reliably at the top (97% height)
            fig.suptitle(report_title, fontsize=11, fontweight='bold', y=0.97)
            
            # Anchor table to upper center instead of bbox. This prevents it from stretching
            # out to fill the entire page when there are only a few rows.
            table = ax.table(cellText=page_data, colLabels=col_labels, loc='upper center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(6.5) 
            
            # Dynamically calculate heights based on number of lines per row to prevent text overflow
            row_heights = {0: 0.04} # Header height
            for row_idx, row_data in enumerate(page_data):
                max_lines = max(len(str(row_data[1]).split('\n')), len(str(row_data[2]).split('\n')), 1)
                # Assign height dynamically: minimum 0.04, adding space for each newline
                row_heights[row_idx + 1] = max(0.04, (0.02 * max_lines) + 0.01)
            
            # Explicitly set all column widths and row heights
            for (row, col), cell in table.get_celld().items():
                cell.PAD = 0.01 # Reduce the internal horizontal padding dramatically (default is ~0.1 or 10%)
                
                if col == 0:
                    cell.set_width(0.15) # 15% width for the Year
                elif col == 1 or col == 2:
                    cell.set_width(0.425) # Equal 42.5% width for Used and Unused
                
                if row in row_heights:
                    cell.set_height(row_heights[row])
                    
                if row == 0: # Header styling
                    cell.set_text_props(weight='bold', ha='center', va='center')
                    cell.set_facecolor('#d3d3d3')
                else:
                    cell.set_text_props(va='center') # Center vertically for clean multi-line stacking padding
                        
            pdf.savefig(fig)
            plt.close(fig)

        # --- Append Error Logs Page ---
        if self.invalid_format_files or self.corrupted_files or self.missing_config_datasets:
            fig_err = plt.figure(figsize=(8.5, 11))
            ax_err = fig_err.add_axes([0.1, 0.1, 0.8, 0.8])
            ax_err.axis('off')
            
            fig_err.suptitle("Processing Exceptions & Errors", fontsize=14, fontweight='bold', y=0.92)
            
            error_text = ""
            
            if self.missing_config_datasets:
                error_text += "Missing Configured Datasets (Configured to 'use', but no matching .tif found):\n"
                for f in self.missing_config_datasets:
                    error_text += f"  - {f}\n"
                error_text += "\n"
            
            if self.invalid_format_files:
                error_text += "Files failing expected Regex format (Skipped):\n"
                for f in self.invalid_format_files:
                    error_text += f"  - {f}\n"
                error_text += "\n"
                
            if self.corrupted_files:
                error_text += "Files corrupted, missing data, or unable to be opened (Skipped):\n"
                for f in self.corrupted_files:
                    error_text += f"  - {f}\n"
            
            # Add text to the page, wrapping nicely
            ax_err.text(0, 0.95, error_text, fontsize=9, va='top', ha='left', wrap=True, family='monospace')
            
            pdf.savefig(fig_err)
            plt.close(fig_err)

    def load_config(self, ecoregion_key: str) -> dict:
        """Load lidar config file"""

        config_path = INPUTS / 'ER_3_lidar_data_config.yaml'
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at '{config_path}'. No filtering will be applied.")
            return {}

        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            if full_config is None:
                return {}
            return full_config.get(ecoregion_key, {})

    def _get_dynamic_colors(self) -> dict:
        """
        Generates a dynamic color palette based on all unique dataset names across all years.
        This guarantees each dataset gets a distinct, consistent color. 
        Scales to support any number of datasets without throwing errors.
        """
        if not self.year_datasets:
            return {}
            
        unique_dataset_names = sorted(list(set([
            ds['dataset_name']
            for year in self.year_datasets
            for ds in self.year_datasets[year]
        ])))
        num_unique = len(unique_dataset_names)
        
        # Use colormaps that fit the number of datasets to ensure visual distinction
        if num_unique <= 10:
            cmap = plt.colormaps['tab10']
            dataset_colors = {name: cmap(i) for i, name in enumerate(unique_dataset_names)}
        elif num_unique <= 20:
            cmap = plt.colormaps['tab20']
            dataset_colors = {name: cmap(i) for i, name in enumerate(unique_dataset_names)}
        else:
            cmap = plt.colormaps['turbo']
            dataset_colors = {name: cmap(i / max(1, num_unique - 1)) for i, name in enumerate(unique_dataset_names)}
            
        return dataset_colors

    def plot_rasters_by_year(self, pdf, shp_path) -> None:
        """
        Plots individual raster datasets by year using a global extent.
        param object pdf: PdfPages object to save the plots into.
        param str shp_path: Path to the shapefile for coastline plotting.
        """

        if not self.year_datasets:
            logger.warning("No raster files found.")
            return

        target_crs = 'EPSG:3857'
        
        shp_read_path = self._get_s3_path(shp_path)
        shp_gdf = gpd.read_file(shp_read_path).to_crs(target_crs)
        
        all_raster_paths = [ds['path'] for year in self.year_datasets for ds in self.year_datasets[year]]
        
        common_transform, common_extent, common_shape = self.calculateExtent(all_raster_paths, target_crs)
        
        # ADDED CHECK: Prevent crash if calculateExtent fails to read any file
        if common_shape is None:
            logger.error("Could not calculate global extent. Check paths, files existences, or AWS/S3 permissions.")
            return

        global_min = np.inf
        for path in all_raster_paths:
            reprojected_data, nodata = self.reprojectToGrid(path, common_transform, common_shape, target_crs)
            combined_mask = np.logical_or.reduce((
                    reprojected_data == nodata, 
                    reprojected_data < -10000, # Catch interpolated nodata artifacts
                    reprojected_data >= 0
                ))
            masked_data = np.ma.array(reprojected_data, mask=combined_mask)
            if masked_data.count() > 0:
                global_min = min(global_min, masked_data.min())
                
        if global_min == np.inf:
            global_min = -10  # Fallback to prevent crash if no valid data

        cmap = plt.colormaps['ocean'].copy()
        cmap.set_bad(color='white')
        
        dataset_colors = self._get_dynamic_colors()

        sorted_years = sorted(self.year_datasets.keys())
        num_years = len(sorted_years)
        cols = math.ceil(math.sqrt(num_years))
        rows = math.ceil(num_years / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True)
        axes = np.atleast_1d(axes).flatten()
        im = None
        
        for i, year in enumerate(sorted_years):
            ax = axes[i]
            final_area_mask = np.zeros(common_shape, dtype=bool)
            legend_handles = []

            for j, dataset in enumerate(self.year_datasets[year]):
                destination, nodata = self.reprojectToGrid(dataset['path'], common_transform, common_shape, target_crs)

                current_area_mask = np.logical_and(destination != nodata, destination < 0)
                final_area_mask = np.logical_or(final_area_mask, current_area_mask)
                
                ds_color = dataset_colors[dataset['dataset_name']]

                data_geometries = [shape(geom) for geom, val in shapes(current_area_mask.astype(np.uint8), transform=common_transform) if val > 0]
                for geom in data_geometries:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color=ds_color, linewidth=0.5, zorder=12)
                
                date_str = f" ({dataset['date']})" if dataset.get('date') else ""
                label_text = f"{dataset['dataset_name']}{date_str}"
                legend_handles.append(Line2D([0], [0], color=ds_color, lw=2, label=label_text))
                
                display_mask = np.logical_or.reduce((
                    destination == nodata, 
                    destination < -10000, # Catch interpolated nodata artifacts
                    destination >= 0
                ))
                masked_destination = np.ma.array(destination, mask=display_mask)
                im = ax.imshow(masked_destination, extent=common_extent, cmap=cmap, vmin=global_min, vmax=0)

            self.setupSubplot(ax, shp_gdf, common_extent)
            
            valid_pixels = np.sum(final_area_mask)
            pixel_area_m2 = abs(common_transform.a * common_transform.e)
            total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
            
            subplot_title = self.year_datasets[year][0]['title']
            ax.set_title(
                f"{subplot_title}\nTotal Area: {total_area_km2:.0f} km$^2$",
                fontsize=10
            )
            
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.3), 
                    fancybox=True, shadow=False, ncol=1, fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        self.finalizeFigure(fig, axes, im, 'Depth (meters)', "Rasterized Datasets by Year (Global Extent)", pdf, 1200)

    def plot_rasters_by_year_individual(self, pdf, shp_path) -> None:
        """
        Plots individual raster datasets by year using individual extents.
        param object pdf: PdfPages object to save the plots into.
        param str shp_path: Path to the shapefile for coastline plotting.
        """

        if not self.year_datasets:
            logger.warning("No raster files found.")
            return

        target_crs = 'EPSG:3857'
        shp_read_path = self._get_s3_path(shp_path)
        shp_gdf = gpd.read_file(shp_read_path).to_crs(target_crs)
        
        all_raster_paths = [ds['path'] for year in self.year_datasets for ds in self.year_datasets[year]]
        global_transform, _, global_shape = self.calculateExtent(all_raster_paths, target_crs)

        # ADDED CHECK: Prevent crash if calculateExtent fails to read any file
        if global_shape is None:
            logger.error("Could not calculate global extent for individual plots. Skipping.")
            return
        
        global_min = np.inf
        for path in all_raster_paths:
            reprojected_data, nodata = self.reprojectToGrid(path, global_transform, global_shape, target_crs)
            combined_mask = np.logical_or.reduce((
                    reprojected_data == nodata, 
                    reprojected_data < -10000, # Catch interpolated nodata artifacts
                    reprojected_data >= 0
                ))
            masked_data = np.ma.array(reprojected_data, mask=combined_mask)
            if masked_data.count() > 0:
                global_min = min(global_min, masked_data.min())

        if global_min == np.inf:
            global_min = -10  # Fallback to prevent crash if no valid data

        cmap = plt.colormaps['ocean'].copy()
        cmap.set_bad(color='white')
        
        dataset_colors = self._get_dynamic_colors()

        sorted_years = sorted(self.year_datasets.keys())
        num_years = len(sorted_years)
        cols = math.ceil(math.sqrt(num_years))
        rows = math.ceil(num_years / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True, sharex=False, sharey=False)
        axes = np.atleast_1d(axes).flatten()
        im = None

        for i, year in enumerate(sorted_years):
            ax = axes[i]
            year_paths = [ds['path'] for ds in self.year_datasets[year]]
            local_transform, local_extent, local_shape = self.calculateExtent(year_paths, target_crs)
            
            # ADDED CHECK: Prevent crash if year-specific datasets cannot be calculated
            if local_shape is None:
                logger.warning(f"Could not calculate extent for year {year}. Skipping this year.")
                continue
            
            final_area_mask = np.zeros(local_shape, dtype=bool)
            legend_handles = []

            for j, dataset in enumerate(self.year_datasets[year]):
                destination, nodata = self.reprojectToGrid(dataset['path'], local_transform, local_shape, target_crs)
                
                current_area_mask = np.logical_and(destination != nodata, destination < 0)
                final_area_mask = np.logical_or(final_area_mask, current_area_mask)
                
                ds_color = dataset_colors[dataset['dataset_name']]

                data_geometries = [shape(geom) for geom, val in shapes(current_area_mask.astype(np.uint8), transform=local_transform) if val > 0]
                for geom in data_geometries:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color=ds_color, linewidth=0.5, zorder=12)

                date_str = f" ({dataset['date']})" if dataset.get('date') else ""
                label_text = f"{dataset['dataset_name']}{date_str}"
                legend_handles.append(Line2D([0], [0], color=ds_color, lw=2, label=label_text))
                
                display_mask = np.logical_or.reduce((
                    destination == nodata, 
                    destination < -10000, # Catch interpolated nodata artifacts
                    destination >= 0
                ))
                masked_destination = np.ma.array(destination, mask=display_mask)
                im = ax.imshow(masked_destination, extent=local_extent, cmap=cmap, vmin=global_min, vmax=0)

            self.setupSubplot(ax, shp_gdf, local_extent)
            
            valid_pixels = np.sum(final_area_mask)
            pixel_area_m2 = abs(local_transform.a * local_transform.e)
            total_area_km2 = (valid_pixels * pixel_area_m2) / 1e6
            
            subplot_title = self.year_datasets[year][0]['title']
            ax.set_title(f"{subplot_title}\nTotal Area: {total_area_km2:.0f} km$^2$", fontsize=8)
            
            ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.2), 
                    fancybox=True, shadow=False, ncol=1, fontsize=2)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        self.finalizeFigure(fig, axes, im, 'Depth (meters)', "Rasterized Datasets by Year (Individual Extent)", pdf, 600)
    
    def reprojectToGrid(self, path, transform, shape, target_crs, resampling_method=Resampling.bilinear) -> tuple[np.zeros, int]:
        """
        Reprojects a raster to a specified grid.
        param str path: Path to the input raster file.
        param object transform: Affine transform for the target grid.
        param tuple shape: (height, width) of the target grid.
        param str target_crs: Target coordinate reference system (e.g., 'EPSG:3857').
        param object resampling_method: Resampling method from rasterio.enums.Resampling.
        """

        open_path = self._get_s3_path(path)
        with rasterio.open(open_path) as src:
            destination = np.zeros(shape, dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=src.nodata,
                resampling=resampling_method
            )
            return destination, src.nodata

    def resample_vrt_files(self, resampling_method='nearest') -> None:
        """
        Resample all .vrt files in the input directory in parallel using Dask
        and save them to the output directory.
        param resampling_method: Resampling method to use (e.g., 'bilinear', 'nearest')
        """

        input_dir = get_config_item("LIDAR_PLOTS", "INPUT_VRTS")
        output_dir = get_config_item("LIDAR_PLOTS", "RESAMPLED_VRTS")
        
        input_dir_str = str(input_dir)
        parent = "/".join(input_dir_str.replace('\\', '/').rstrip('/').split('/')[:-1])
        
        manual_input_dir_1 = f"{parent}/Digital_Coast_Manual_Downloads"
        manual_input_dir_2 = f"{parent}/DigitalCoast_manual_downloads"
            
        input_dirs = [input_dir_str, manual_input_dir_1, manual_input_dir_2]
        vrt_files = []

        if self.is_aws:
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
                
            output_dir_clean = str(output_dir).replace('\\', '/').replace("s3://", "")
            if not output_dir_clean.startswith(bucket):
                output_dir = f"s3://{bucket}/{output_dir_clean.lstrip('/')}"
            else:
                output_dir = f"s3://{output_dir_clean}"

            for i_dir in input_dirs:
                # Prepend bucket to directories for s3fs globbing and saving
                i_dir_clean = i_dir.replace('\\', '/').replace("s3://", "")
                if not i_dir_clean.startswith(bucket):
                    i_dir_s3 = f"{bucket}/{i_dir_clean.lstrip('/')}"
                else:
                    i_dir_s3 = i_dir_clean
                    
                # glob returns list of bucket/path/file.vrt
                vrt_files.extend([f"s3://{f}" for f in self.fs.glob(f"{i_dir_s3}/*.vrt")])
        else:
            os.makedirs(output_dir, exist_ok=True)
            for i_dir in input_dirs:
                vrt_files.extend(glob.glob(os.path.join(i_dir, "*.vrt")))

        if not vrt_files:
            logger.warning(f"No .vrt files found in {input_dirs}")
            return

        logger.info(f"Found {len(set(vrt_files))} VRT files to process.")

        # ADDED BLOCKXSIZE and BLOCKYSIZE to prevent the GDAL 2GB Offset Array error 
        creation_options = [
            "COMPRESS=LZW",
            "TILED=YES",
            "BIGTIFF=YES",
            "BLOCKXSIZE=2048", # Lowered from 8192 to 2048 to drastically reduce memory spikes per tile
            "BLOCKYSIZE=2048", # Lowered from 8192 to 2048 to drastically reduce memory spikes per tile
            "NUM_THREADS=1" # Changed from ALL_CPUS to prevent thread explosion when used with Dask
        ]

        vrt_params = [[vrt_file,
                output_dir,
                self.target_resolution_m,
                resampling_method,
                creation_options, 
            ] for vrt_file in vrt_files ]
        
        future_tiles = self.client.map(_process_all_vrts, vrt_params)
        _ = self.client.gather(future_tiles)

        logger.info("All vrts have been resampled.")

    def plot_difference(self, pdf, shp_path, mode, use_individual_extent=False) -> None:
        """
        Calculates and plots raster differences for consecutive or all year pairs.
        param object pdf: PdfPages object to save the plots into.
        param str shp_path: Path to the shapefile for coastline plotting.
        param str mode: 'consecutive' for consecutive year differences, 'all' for all year pairs with >=5% overlap.
        param bool use_individual_extent: Whether to use individual extents for each subplot.
        """

        if not self.year_datasets or len(self.year_datasets) < 2:
            logger.warning("Not enough years of data to calculate differences.")
            return
        
        year_datasets_map = {}
        for year, data in self.year_datasets.items():
            if any(ds.get('is_use_true', False) for ds in data):
                year_datasets_map[year] = data[0]['path']
            else:
                logger.info(f"Excluding year {year} from differences because no dataset is explicitly set to 'use: true'.")

        if len(year_datasets_map) < 2:
            logger.warning(f"Not enough years of explicitly used data to calculate '{mode}' differences.")
            return

        target_crs = 'EPSG:3857'
        shp_read_path = self._get_s3_path(shp_path)
        shp_gdf = gpd.read_file(shp_read_path).to_crs(target_crs)
        
        sorted_years_unfiltered = sorted(year_datasets_map.keys())
        
        all_paths = list(year_datasets_map.values())
        global_transform, global_extent, global_shape = self.calculateExtent(all_paths, target_crs)

        # ADDED CHECK: Prevent crash if calculateExtent fails to read any file
        if global_shape is None:
            logger.error("Could not calculate global extent for diff plots. Skipping.")
            return

        reprojected_global = {}
        valid_years = []
        for year in sorted_years_unfiltered:
            data, nodata = self.reprojectToGrid(year_datasets_map[year], global_transform, global_shape, target_crs)
            # Add the 'data >= 0' condition to the mask
            mask = np.logical_or.reduce((
                data == nodata, 
                data < -10000, # Catch interpolated nodata artifacts
                data >= 0
            ))
            reprojected_data = np.ma.array(data, mask=mask)
            
            # STRICT CHECK: Check if there is any valid data (< 0) before keeping this year
            if reprojected_data.count() > 0:
                reprojected_global[year] = reprojected_data
                valid_years.append(year)
            else:
                logger.info(f"Year {year} has no valid underwater data (< 0m). Excluding from differences.")

        # Override sorted_years so empty datasets are completely removed from combinations
        sorted_years = valid_years

        if len(sorted_years) < 2:
            logger.warning("Not enough years with valid data to calculate differences.")
            return

        global_diff_min, global_diff_max = np.inf, -np.inf
        year_pairs_for_minmax = list(combinations(sorted_years, 2))
        for year1, year2 in year_pairs_for_minmax:
            diff = reprojected_global[year2] - reprojected_global[year1]
            if diff.count() > 0:
                global_diff_min = min(global_diff_min, diff.min())
                global_diff_max = max(global_diff_max, diff.max())

        if global_diff_min == np.inf:
            global_diff_min, global_diff_max = -10, 10

        year_pairs = list(combinations(sorted_years, 2)) if mode == 'all' else list(zip(sorted_years, sorted_years[1:]))
        
        diff_data = []
        for year1, year2 in year_pairs:
            path1, path2 = year_datasets_map[year1], year_datasets_map[year2]
            
            data1_global, data2_global = reprojected_global[year1], reprojected_global[year2]
            combined_mask_global = np.ma.mask_or(data1_global.mask, data2_global.mask)
            overlapping_pixels = np.sum(~combined_mask_global)
            smaller_year_pixels = min(np.sum(~data1_global.mask), np.sum(~data2_global.mask))
            overlap_percentage = (overlapping_pixels / smaller_year_pixels * 100) if smaller_year_pixels > 0 else 0

            pixel_area_m2 = abs(global_transform.a * global_transform.e)
            overlap_area_km2 = (overlapping_pixels * pixel_area_m2) / 1e6
            
            if mode == 'all' and overlap_area_km2 < 10:
                continue

            overlap_text = f"Overlap: {overlap_area_km2:.0f} km$^2$ ({overlap_percentage:.1f}%)"
            title = f'Difference: {year1} to {year2}\n{overlap_text}'
            
            diff_data.append({'title': title, 'path1': path1, 'path2': path2, 'year1': year1, 'year2': year2})
            
        if not diff_data:
            logger.warning(f"No year pairs found for '{mode}' mode with sufficient overlap.")
            return

        num_diffs = len(diff_data)
        cols = math.ceil(math.sqrt(num_diffs))
        rows = math.ceil(num_diffs / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15), constrained_layout=True, sharex=not use_individual_extent, sharey=not use_individual_extent)
        axes = np.atleast_1d(axes).flatten()
        im_diff = None
        cmap = plt.colormaps['ocean'].copy()
        cmap.set_bad(color='white')

        for i, data_dict in enumerate(diff_data):
            ax = axes[i]
            
            if use_individual_extent:
                local_transform, local_extent, local_shape = self.calculateExtent([data_dict['path1'], data_dict['path2']], target_crs)
                
                # ADDED CHECK: Prevent crash if specific year datasets cannot be resolved
                if local_shape is None:
                    logger.warning(f"Could not calculate local extent for {data_dict['year1']} to {data_dict['year2']}. Skipping.")
                    continue

                data1, nodata1 = self.reprojectToGrid(data_dict['path1'], local_transform, local_shape, target_crs)
                data2, nodata2 = self.reprojectToGrid(data_dict['path2'], local_transform, local_shape, target_crs)
                
                # Add conditions to mask pixels where either dataset has a value >= 0 or nodata leakage
                combined_mask = np.logical_or.reduce((
                    data1 == nodata1, 
                    data1 < -10000,
                    data2 == nodata2, 
                    data2 < -10000,
                    data1 >= 0,
                    data2 >= 0
                ))
                diff = np.ma.array(data2 - data1, mask=combined_mask)
                
                im_diff = ax.imshow(diff, extent=local_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
                self.setupSubplot(ax, shp_gdf, local_extent)
            else:
                diff = reprojected_global[data_dict['year2']] - reprojected_global[data_dict['year1']]
                im_diff = ax.imshow(diff, extent=global_extent, cmap=cmap, vmin=global_diff_min, vmax=global_diff_max)
                
                # Create a mask of the valid overlap area
                overlap_mask = ~np.ma.mask_or(reprojected_global[data_dict['year1']].mask, reprojected_global[data_dict['year2']].mask)

                # Generate shapes from the overlap mask
                overlap_geometries = [
                    shape(geom) for geom, val in shapes(overlap_mask.astype(np.uint8), transform=global_transform) if val > 0
                ]

                # Plot the overlap geometries in red
                for geom in overlap_geometries:
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color='red', linewidth=1.5, zorder=12, label='Overlap Area')
                
                self.setupSubplot(ax, shp_gdf, global_extent)

            ax.set_title(data_dict['title'], loc='left', fontsize=8)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        extent_type = "Individual" if use_individual_extent else "Global"
        suptitle = f"{mode.capitalize().replace('_', ' ')} Year Differences ({extent_type} Extent)"
        if mode == 'all':
            suptitle = f"All Year Differences >= 10 km2 ({extent_type} Extent)"
        
        dpi = 600 if use_individual_extent else 1200

        self.finalizeFigure(fig, axes, im_diff, 'Difference (m)', suptitle, pdf, dpi, show_overlap_legend=True)

    def run(self) -> None:
        """Entrypoint for processing the Lidar Data Plots"""

        self.setup_dask(get_environment(), memory_limit="28GB") # Adjusted to match available 30GB system memory

        self.resample_vrt_files()
        self.close_dask()

        if self.is_aws:
            bucket = get_config_item('SHARED', 'OUTPUT_BUCKET')
            raster_folder = f"s3://{bucket}/{get_config_item('LIDAR_PLOTS', 'RESAMPLED_VRTS')}".replace('\\', '/')
            plot_output_folder = f"s3://{bucket}/{get_config_item('LIDAR_PLOTS', 'PLOT_OUTPUTS')}".replace('\\', '/')
            shp_path = f"s3://{bucket}/{get_config_item('MASK', 'COAST_BOUNDARY_PATH')}".replace('\\', '/')
        else:
            raster_folder = get_config_item("LIDAR_PLOTS", "RESAMPLED_VRTS")
            plot_output_folder = get_config_item("LIDAR_PLOTS", "PLOT_OUTPUTS") 
            shp_path = get_config_item("MASK", "COAST_BOUNDARY_PATH")

        ecoregion_name = 'Ecoregion 3'
        self.config = self.load_config('EcoRegion-3') 
        self.year_datasets = self.getYearDatasets(raster_folder, self.config)

        # Handle master PDF setup
        if self.is_aws:
            plot_output_folder_clean = str(plot_output_folder).replace('\\', '/')
            pdf_path_str = f"{plot_output_folder_clean.rstrip('/')}/Ecoregion_3_Lidar_Datasets_Report.pdf"
            if not pdf_path_str.startswith("s3://"): pdf_path_str = f"s3://{pdf_path_str}"
            buf = io.BytesIO()
            pdf_target = buf
        else:
            pdf_path_str = os.path.join(plot_output_folder, 'Ecoregion_3_Lidar_Datasets_Report.pdf')
            os.makedirs(os.path.dirname(pdf_path_str), exist_ok=True)
            pdf_target = pdf_path_str

        # Generate and save all outputs onto the same PDF object
        with PdfPages(pdf_target) as pdf:
            self.generate_dataset_report(pdf, ecoregion_name)
            self.plot_rasters_by_year(pdf, shp_path)
            # self.plot_rasters_by_year_individual(pdf, shp_path)
            self.plot_difference(pdf, shp_path, 'consecutive', use_individual_extent=False)
            # self.plot_difference(pdf, shp_path, 'consecutive', use_individual_extent=True)
            self.plot_difference(pdf, shp_path, 'all', use_individual_extent=False)
            # self.plot_difference(pdf, shp_path, 'all', use_individual_extent=True)

        if self.is_aws:
            buf.seek(0)
            s3_out = pdf_path_str.replace("s3://", "").replace('\\', '/')
            with self.fs.open(s3_out, 'wb') as f:
                f.write(buf.read())
            buf.close()

        logger.info(f"Combined Lidar Data Report and Plots saved to: {pdf_path_str}")
        logger.info("Lidar Data Plots processing complete.")

    def setupSubplot(self, ax, shp_gdf, extent) -> None:
        """
        Configures the appearance of a subplot.
        param object ax: Matplotlib axes object.
        param object shp_gdf: GeoDataFrame containing the shapefile data.
        param tuple extent: (left, right, bottom, top) for setting axis limits.
        """

        shp_gdf.plot(ax=ax, color='none', edgecolor='gray', linewidth=0.5, zorder=11)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        ax.tick_params(left=True, right=True, labelleft=True, labelbottom=True, bottom=True)
        ax.set_xlabel('X distance (km)', fontsize=4)
        ax.set_ylabel('Y distance (km)', fontsize=4)
        
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        
        # Explicitly set ticks before assigning tick labels to avoid Matplotlib warning
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
        ax.set_xticklabels([f'{val/1000:.1f}' for val in xticks], fontsize=4)
        ax.set_yticklabels([f'{val/1000:.1f}' for val in yticks], fontsize=4)