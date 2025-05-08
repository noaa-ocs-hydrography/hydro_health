import os
import time
import pathlib
import boto3
import xarray as xr  # pip install cfgrib required for cfgrib engine
import geopandas as gpd
import s3fs  # require conda install h5netcdf
import shapely
import thalassa
import numpy as np

from datetime import datetime
from botocore.client import Config
from botocore import UNSIGNED
from concurrent.futures import ThreadPoolExecutor


OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


class SurgeTideForecastProcessor:
    """Download and convert any STOFS data"""
    
    def __init__(self) -> None:
        self.bucket = 'noaa-nos-stofs3d-pds'
        self.weeks = {}

    def build_bucket_lookup(self) -> None:
        """Create lookup dictionary of netcdf files by calendar week"""

        stofs_bucket = self.get_bucket()
        for obj_summary in stofs_bucket.objects.filter(Prefix=f"STOFS-3D-Atl/stofs_3d_atl."):
            if 'n001_024.field2d' in obj_summary.key:
                folder_name = pathlib.Path(obj_summary.key).parents[0].name
                folder_date = datetime.strptime(folder_name[-8:], '%Y%m%d')
                calendar = folder_date.isocalendar()
                week_key = f'Y:{calendar.year}_M:{folder_date.month}_W:{calendar.week}'
                if week_key not in self.weeks:
                    self.weeks[week_key] = []
                self.weeks[week_key].append(obj_summary.key)

    def download_s3_file(self, param_inputs) -> None:
        stofs_bucket, outputs, obj_summary = param_inputs
        output_tile_path = outputs / obj_summary.key
        if output_tile_path.exists():
            return
        output_tile_path.parents[0].mkdir(parents=True, exist_ok=True)  
        stofs_bucket.download_file(obj_summary.key, output_tile_path)

    def download_water_velocity_netcdf(self, outputs) -> None:
        # 12GB file takes 16 minutes to download(949 seconds)

        stofs_bucket = self.get_bucket()
        start = 0
        velocity_files = []
        for obj_summary in stofs_bucket.objects.filter(Prefix=f"STOFS-3D-Atl/stofs_3d_atl."):
            # This gets us into the main folder
            # TODO need to secondary filter obj_summary for 
            if 'horizontalVelX_nowcast' in obj_summary.key or 'horizontalVelY_nowcast' in obj_summary.key:
                print('vel:', obj_summary.key)
                # ds = xr.open_dataset(obj_summary.get()['Body'].read())
                velocity_files.append(obj_summary)
                start += 1
            if start == 4:
                break

        param_inputs = [[stofs_bucket, outputs, file] for file in velocity_files]  # rows out of ER will be nan
        with ThreadPoolExecutor(int(os.cpu_count() - 2)) as pool:
            pool.map(self.download_s3_file, param_inputs)
            
            # ds = xr.open_dataset(output_tile_path)
            # variable - horizontalVelX
            # print(ds)

        # TODO cleaner to use s3fs library? 
        # Sample code uses xarray to directly load 5GB .nc file from s3
        # need to test how slow that is compared to downloading
        # s3 = s3fs.S3FileSystem(anon=True)  # Enable anonymous access to the S3 bucket
        # url = f"s3://{bucket_name}/{key}"
        # ds = xr.open_dataset(s3.open(url, 'rb'), drop_variables=['nvel'])

        # TODO thalassa can spatial filter based on shapely bbox
        # import thalassa
        # import shapely
        # ds = thalassa.open_dataset("some_netcdf.nc")
        # bbox = shapely.box(0, 0, 1, 1)
        # ds = thalassa.crop(ds, bbox)

        # TODO other open-source options for spatial filter netcdf using numpy or xarray
        # https://stackoverflow.com/questions/29135885/netcdf4-extract-for-subset-of-lat-lon
        # https://github.com/Deltares/xugrid/issues/107

    def get_averages(self) -> None:
        # get all year keys
        # for each year
        #   get all monthly keys for year
        #   for each month
        #       get weekly averages for weeks in month
        #     get monthly average from weekly averages in current month
        #     store monthly average
        #   compute annual average
        s3 = self.get_s3_object()
        
        # first 4 files for testing
        first_week = {"Y:2023_M:1_W:2": [
            "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.fields.zCoordinates_nowcast.nc",
            "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.fields.horizontalVelY_nowcast.nc",
            
            # "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.n001_024.field2d.nc",
            # "STOFS-3D-Atl/stofs_3d_atl.20230113/stofs_3d_atl.t12z.n001_024.field2d.nc",
            # "STOFS-3D-Atl/stofs_3d_atl.20230114/stofs_3d_atl.t12z.n001_024.field2d.nc",
            # "STOFS-3D-Atl/stofs_3d_atl.20230115/stofs_3d_atl.t12z.n001_024.field2d.nc",
        ]}
        # for week, files in self.weeks.items():
        for week, files in first_week.items():
            for file in files:
                url = f"s3://{self.bucket}/{file}"
                with xr.open_dataset(s3.open(url, model='rb')) as ds: #, drop_variables=['nvel'])
                    print(week)
                    print(ds.variables)
                    print(ds['zCoordinates'].values)
                    print(ds['zCoordinates'].isel(time=0))
                    # print(ds.variables.keys())
                    # SCHISM_hgrid_node_x      (nSCHISM_hgrid_node) float64 ...
                    # SCHISM_hgrid_node_y      (nSCHISM_hgrid_node) float64 ...
                    # SCHISM_hgrid_face_nodes  (nSCHISM_hgrid_face, nMaxSCHISM_hgrid_face_nodes) int32 ...
                    # depth                    (nSCHISM_hgrid_node) float32 ...
                    # elev                     (time, nSCHISM_hgrid_node) float64 ...
                    # temp_surface             (time, nSCHISM_hgrid_node) float64 ...
                    # temp_bottom              (time, nSCHISM_hgrid_node) float64 ...
                    # salt_surface             (time, nSCHISM_hgrid_node) float64 ...
                    # salt_bottom              (time, nSCHISM_hgrid_node) float64 ...
                    # uvel_surface             (time, nSCHISM_hgrid_node) float64 ...
                    # vvel_surface             (time, nSCHISM_hgrid_node) float64 ...
                    # uvel_bottom              (time, nSCHISM_hgrid_node) float64 ...
                    # vvel_bottom              (time, nSCHISM_hgrid_node) float64 ...
                    # uvel4.5                  (time, nSCHISM_hgrid_node) float64 ...
                    # vvel4.5

                    normalized_ds = thalassa.normalize(ds)  # TODO this updates STOFS to expected variables
                    box = shapely.box(-86.775, -86.7, 30.3750000000001, 30.45)
                    subset = thalassa.crop(normalized_ds, box)
                    print(subset)
                    break


    def get_bucket(self) -> boto3.resource:
        """Connect to anonymous OCS S3 Bucket"""

        bucket = "noaa-nos-stofs3d-pds"
        creds = {
            "aws_access_key_id": "",
            "aws_secret_access_key": "",
            "config": Config(signature_version=UNSIGNED),
        }
        s3 = boto3.resource('s3', **creds)
        nbs_bucket = s3.Bucket(bucket)
        return nbs_bucket
    
    def get_s3_object(self) -> s3fs.S3FileSystem:
        s3 = s3fs.S3FileSystem(anon=True)
        return s3
    
    def process(self, outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        # self.build_bucket_lookup()  # store s3 objects instead of folders
        self.get_averages()  # load each s3 object while looping, store weekly, monthly, and annual results on class
        # self.average_datasets()  # get_averages() could build lists as inputs to this function.  just average a list of inputs


if __name__ == "__main__":
    start = time.time()
    processor = SurgeTideForecastProcessor()
    processor.process(OUTPUTS)
    print(f'Finished: {time.time() - start}')