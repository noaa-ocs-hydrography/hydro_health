import pathlib
import sys
import json
import os
import re
import time
import datetime
import requests
import boto3
import geopandas as gpd
import dask
import math

from datetime import date
from osgeo import osr, gdal
from dask.distributed import Client, LocalCluster

from hydro_health.helpers.tools import get_config_item


gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'


class Engine:
    """Base class for all Engines"""

    def __init__(self):
        self.approved_size = 200000000  # 2015 USACE polygon was 107,987,252 sq. meters
        self.cluster = None
        self.client = None
        
        self.year_ranges = [
            (1998, 2004),
            (2004, 2006),
            (2006, 2007),
            (2006, 2010),
            (2007, 2010),
            (2010, 2015),
            (2014, 2022),
            (2015, 2022),
            (2016, 2017),
            (2017, 2018),
            (2018, 2019),
            (2020, 2022),
            (2022, 2024)
        ]

    # def __init__(self):
    #     # set up logging
    #     # TODO do we need daily logs?
    #     self.log_path = str(OUTPUTS / 'hh_log.txt')
    #     self.logger = logging.getLogger(__name__)
    #     logging.basicConfig(filename=self.log_path,
    #                 format='%(levelname)s:%(asctime)s %(message)s',
    #                 level=logging.INFO)
    #     self.logged = False

        self.start_time = time.time()

    def approved_dataset(self, feature_json: dict[dict]) -> bool:
        """Only allow certain provider types"""

        provider_list_text = ['USACE', 'NCMP', 'NGS', 'USGS']  # CUDEM: NOAA NCEI
        for text in provider_list_text:
            if text in feature_json['attributes']['provider_results_name']:
                return True
        return False

    def cleansed_url(self, url: str) -> str:
        """Remove found illegal characters from URLs"""

        illegal_chars = ['{', '}']
        for char in illegal_chars:
            url = url.replace(char, '')
        return url.strip()
    
    def close_dask(self) -> None:
        """Shut down Dask objects"""

        self.client.close()
        self.cluster.close()
    
    def get_available_datasets(self, geometry_coords: str, ecoregion_id: str, outputs: str) -> None:
        """Query NOWCoast REST API for available datasets"""

        payload = {
            "aoi": f"SRID=4269;{geometry_coords}",
            "published": "true",
            "dataTypes": ["Lidar", "DEM"],
            "dialect": "arcgis",
        }
        response = requests.post(get_config_item('DIGITALCOAST', 'API'), data=payload)
        datasets_json = response.json()

        if response.status_code == 404:
            raise Exception(f"Digital Coast Error: {response.reason}")

        tile_index_links = []
        for feature in datasets_json['features']:
            # print(feature['attributes']['DataType'], feature['attributes']['Year'], feature['attributes']['provider_results_name'])
            if not self.approved_dataset(feature):
                continue
            folder_name = re.sub('\W+',' ', feature['attributes']['provider_results_name']).strip().replace(' ', '_') + '_' + str(feature['attributes']['Year'])  # remove illegal chars
            output_folder_path = pathlib.Path(outputs) / ecoregion_id / get_config_item('DIGITALCOAST', 'SUBFOLDER') / 'DigitalCoast' / f"{folder_name}_{feature['attributes']['ID']}"
            output_folder_path.mkdir(parents=True, exist_ok=True)

            # Write out JSON
            output_json = output_folder_path / 'feature.json'
            external_provider_links = json.loads(feature['attributes']['ExternalProviderLink'])['links']
            feature['attributes']['ExternalProviderLink'] = external_provider_links
            with open(output_json, 'w') as writer:
                writer.write(json.dumps(feature['attributes'], indent=4))

            for external_data in external_provider_links:
                if external_data['label'] == 'Bulk Download':
                    tile_index_links.append({'label': 'Bulk Download', 'data_type': feature['attributes']['DataType'], 'link': external_data['link'], 'provider_path': output_folder_path}) 
        return tile_index_links

    def get_ecoregion_geometry_strings(self, tile_gdf: gpd.GeoDataFrame, ecoregion: str) -> str:
        """Build bbox string dictionary of tiles in web mercator projection"""

        geometry_coords = []
        ecoregion_groups = tile_gdf.groupby('EcoRegion')
        for er_id, ecoregion_group in ecoregion_groups:
            if er_id == ecoregion:
                ecoregion_group_web_mercator = ecoregion_group.to_crs(4269)  # POST request only allows this EPSG
                ecoregion_group_web_mercator['geom_type'] = 'Polygon'
                tile_geometries = ecoregion_group_web_mercator[['geom_type', 'geometry']]
                tile_boundary = tile_geometries.dissolve(by='geom_type')
                tile_wkt = tile_boundary.iloc[0].geometry
                geometry_coords.append(tile_wkt)

        return geometry_coords

    def make_esri_projection(self, file_name, epsg=4326):
        """Create an Esri .prj file for a shapefile"""

        output_projection = osr.SpatialReference()
        output_projection.ImportFromEPSG(epsg)
        output_projection.MorphToESRI()
        file = open(OUTPUTS / f'{file_name}.prj', 'w')
        file.write(output_projection.ExportToWkt())
        file.close()

    def print_async_results(self, results: list[str], output_folder: str) -> None:
        """Consolidate result printing"""

        for result in results:
            if result:
                self.write_message(f'Result: {result}', output_folder)  

    def setup_dask(self, env, processes=True, n_workers=4, threads_per_worker=2, memory_limit="8GB") -> None:
        """Create Dask objects outside of init"""

        if env == 'aws':
            dask.config.set({"distributed.worker.multiprocessing-method": "fork"})
            self.set_proj_path()
        self.cluster = LocalCluster(processes=processes, n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
        self.client = Client(self.cluster)
        print(self.client.dashboard_link)

    def set_proj_path(self):
        """Load proj.db path to resolve mismatch"""

        conda_prefix = sys.prefix
        proj_path = os.path.join(conda_prefix, 'share', 'proj')
        os.environ['PROJ_LIB'] = proj_path
        # For newer PROJ versions, also set PROJ_DATA
        os.environ['PROJ_DATA'] = proj_path

    def write_message(self, message: str, output_folder: str) -> None:
        """Write a message to the main logfile in the output folder"""

        with open(pathlib.Path(output_folder) / 'log_prints.txt', 'a') as writer:
            writer.write(message + '\n')

    def write_run_manifest(self, subfolder: str, extra_info: dict = None):
        """Writes a single manifest for the entire Engine execution."""

        end_time = time.time()
        duration = round(end_time - self.start_time, 2) if self.start_time else 0
        
        manifest = {
            "engine": self.__class__.__name__,
            "run_date": datetime.datetime.now().isoformat(),
            "duration_seconds": duration,
            "status": "completed"
        }
        if extra_info:
            manifest.update(extra_info)

        if self.param_lookup['env'] == 'aws':
            s3 = boto3.client('s3')
            s3.put_object(
                Bucket=get_config_item('SHARED', 'OUTPUT_BUCKET'),
                Key=f"{subfolder}/_manifest.json",
                Body=json.dumps(manifest, indent=4)
            )
        else:
            manifest_prefix = OUTPUTS / subfolder / '_manifest.json'
            with open(manifest_prefix, 'w') as manifest_writer:
                manifest_writer.write(json.dumps(manifest, indent=4))


# CATZOC score.py
def catzoc(metadata: dict) -> int:
    """
    Return an enumeration representing the catzoc assocaited with the provided
    metrics.

    The enumeration (catzoc : value) is as follows:
        A1 : 1
        A2 : 2
        B  : 3
        C  : 4
        D  : 5
        U  : 6

    The provided metadata is expected to contain the following metadata values:

    """
    s = supersession(metadata)
    if s > 80:
        return 1
    elif s > 60:
        return 2
    elif s > 40:
        return 3
    elif s > 20:
        return 4
    else:
        return 5


def supersession(metadata: dict) -> float:
    """
    Return the superssion score as defined in Wyllie 2017 at US Hydro for the
    catzoc score.
    """

    required_entries = ['feat_detect', 'complete_coverage', 'horiz_uncert_fixed', 'vert_uncert_fixed',
                        'horiz_uncert_vari', 'vert_uncert_vari']

    for required_entry in required_entries:
        if required_entry not in metadata:
            survey_name = metadata['from_filename']
            raise ValueError(
                f'Metadata for survey "{survey_name}" does not contain an entry for "{required_entry}" and is thus not available to score')

    feat_score = _get_feature_detection(metadata)
    cov_score = _get_coverage(metadata)
    horz_score = _get_horizontal_uncertainty(metadata)
    vert_score = _get_vertical_uncertainty(metadata)
    score = min(feat_score, cov_score, horz_score, vert_score)
    if metadata['interpolated']:
        score -= 0.01
    return score


def _get_feature_detection(metadata: dict) -> float:
    """
    Determine the feature detection capability from the ability to detect
    features, detect the least depth, and the size of the feature.
    """
    least_depth = metadata['feat_detect'] and metadata['feat_least_depth']
    # size_okay = 'feat_size' in metadata and float(metadata['feat_size']) <= 2
    if metadata['feat_detect'] and least_depth:  # and size_okay:
        return 100
    else:
        return 60


def _get_coverage(metadata: dict) -> float:
    """
    Determine the coverage score and return.
    """
    if metadata['complete_coverage']:
        return 100
    else:
        return 60


def _get_horizontal_uncertainty(metadata: dict) -> float:
    """
    Determine the horizontal uncertainty score and return.
    """
    h_fix = float(metadata['horiz_uncert_fixed'])
    h_var = float(metadata['horiz_uncert_vari'])
    if h_fix <= 5 and h_var <= 0.05:
        s = 100
    elif h_fix <= 20:
        s = 80
    elif h_fix <= 50:
        s = 60
    elif h_fix <= 500:
        s = 40
    else:
        s = 20
    return s


def _get_vertical_uncertainty(metadata: dict) -> float:
    """
    Determine the vertical uncertainty score and return.
    """
    v_fix = float(metadata['vert_uncert_fixed'])
    v_var = float(metadata['vert_uncert_vari'])
    if v_fix <= 0.5 and v_var <= 0.01:
        s = 100
    elif v_fix <= 1 and v_var <= 0.02:
        s = 80
    elif v_fix <= 2 and v_var <= 0.05:
        s = 40
    else:
        s = 20
    return s


def decay(metadata: dict, date: date, alpha: float = 0.022) -> float:
    """
    Return the decayed supersession_score.
    """
    sd = metadata['end_date' if 'end_date' in metadata else 'start_date']
    ss = float(metadata['supersession_score'])
    dt = date - sd
    days = dt.days + dt.seconds / (24 * 60 * 60)
    years = days / 365
    ds = ss * math.exp(-alpha * years)
    if ds < 1:
        raise ValueError(f"Decay Score less than 1: end_date {sd}; supersession_score {ss}; date_delta {dt}, days {days}; years {years}; constant_alpha {alpha}")
    else:
        return ds