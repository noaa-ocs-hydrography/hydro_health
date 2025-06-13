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
        s3 = self.get_s3_filesystem()
        
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
    
    def get_s3_filesystem(self) -> s3fs.S3FileSystem:
        s3 = s3fs.S3FileSystem(anon=True)
        return s3
    
    def get_zcoord_third_index(self, grid_dataset: xr.Dataset) -> int|None:
        """Read third non-nan index from first week Z Coordinates file"""

        grid_array = grid_dataset.zCoordinates[1, 1,:].values
        # print(grid_array)
        non_nan_indices = np.where(np.isfinite(grid_array))[0]
        if non_nan_indices.size > 0:
            return non_nan_indices[0] + 2
        else:
            return None
    
    def get_dataset_third_value(self, grid_dataset: xr.Dataset, grid_index: int) -> int|None:
        """Read third non-nan value from first week Z Coordinates file"""

        grid_array = grid_dataset.horizontalVelX[1, 1,:].values
        # print(grid_array)
        if np.isnan(grid_array).all():
            return None
        else:
            return grid_array[grid_index]
        
    def load_zcord_dataset(self, local=False) -> xr.Dataset:
        """Get the z-coordinate dataset"""

        if local:
            z_ds = xr.open_dataset(r'C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data\stofs_3d_atl.t12z.fields.zCoordinates_nowcast.nc')
        else:
            s3 = self.get_s3_filesystem()
            first_week_z_coords_dataset = "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.fields.zCoordinates_nowcast.nc"
            url = f"s3://{self.bucket}/{first_week_z_coords_dataset}"
            z_ds = xr.open_dataset(s3.open(url, 'rb'))
        return z_ds
    
    def load_x_vel_dataset(self, local=False) -> xr.Dataset:
        "Get the x-velocity dataset"

        if local:
            ds = xr.open_dataset(r'C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data\stofs_3d_atl.t12z.fields.horizontalVelX_nowcast.nc')
        else:
            s3 = self.get_s3_filesystem()
            first_week_z_coords_dataset = "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.fields.horizontalVelX_nowcast.nc"
            url = f"s3://{self.bucket}/{first_week_z_coords_dataset}"
            ds = xr.open_dataset(s3.open(url, 'rb'))
        return ds
    
    def process(self, outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        z_coord_ds = self.load_zcord_dataset(local=True)
        z_coord_index = self.get_zcoord_third_index(z_coord_ds)
        print('index:', z_coord_index)

        x_velocity_ds = self.load_x_vel_dataset(local=True)
        x_velocity_value = self.get_dataset_third_value(x_velocity_ds, z_coord_index)
        print('value:', x_velocity_value)
        

if __name__ == "__main__":
    start = time.time()
    # files stored: C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data
    processor = SurgeTideForecastProcessor()
    processor.process(OUTPUTS)
    print(f'Finished: {time.time() - start}')