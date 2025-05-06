import os
import time
import pathlib
import boto3
import xarray as xr  # pip install cfgrib required for cfgrib engine
import geopandas as gpd

from datetime import datetime
from botocore.client import Config
from botocore import UNSIGNED
from concurrent.futures import ThreadPoolExecutor


OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


class SurgeTideForecastProcessor:
    """Download and convert any STOFS data"""

    def average_datasets(self, outputs) -> None:
        """Compute weekly, monthly, and yearly averages"""

        stofs_folder = outputs / 'STOFS-3D-Atl'
        daily_folders = [path for path in stofs_folder.glob('stofs_3d_atl.*') if path.is_dir()]
        weeks = {}
        for folder in daily_folders:
            folder_name = folder.name
            folder_date = datetime.strptime(folder_name[-8:], '%Y%m%d')
            calendar = folder_date.isocalendar()
            week_key = f'Y:{calendar.year}_M:{folder_date.month}_W:{calendar.week}'
            if week_key not in weeks:
                weeks[week_key] = []
            weeks[week_key].append(folder)

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
    
    def process(self, outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""


        self.download_water_velocity_netcdf(outputs)
        self.average_datasets(outputs)
        # Access S3
        # determine years for year pairs we care about(ex: 23-24)
            # folders are for each day starting 1/12/2023
        # download main 12z file for each day
            # loop through folders starting with stofs_3d_atl.YYYYMMDD
            # download base model grib2 files: stofs_3d_atl.t12z.conus.east.cwl.grib2, t12z, no fXXX, starts wiith stofs_3d_atl, has east in it
            # Convert grib2 files to 24 rasters for each day found, this will be alot of files, single grib2 current size is 58mb
            # build daily mean raster from 24 files
            # build monthly mean raster from all daily mean rasters 
            # build annual mean raster from all monthly rasters

    def export_hourly_rasters(self, daily_netcdf: pathlib.Path) -> None:
        # open daily_netcdf
        with xr.open_dataset(daily_netcdf, engine='netcdf4') as file:
            # dimensions = dict(file.dims) 
            # {'time': 24, 'nSCHISM_hgrid_node': 2654153, 'nSCHISM_hgrid_face': 5039151, 'nMaxSCHISM_hgrid_face_nodes': 3}
            # variables = file.variables  # forecast hour timestamp
            # for variable in variables:
            #     print(variable, '---\n', variables[variable])
            # attributes = file.attrs
            # {'title': 'SCHISM Model output', 'source': 'SCHISM model output version v10', 'references': 'http://ccrm.vims.edu/schismweb/'} 

            # variables we care about
            values = ['depth', 'elev', 'temp_surface', 'temp_bottom', 'salt_surface', 'salt_bottom', 
                    'uvel_surface', 'vvel_surface', 'uvel_bottom', 'vvel_bottom']
            # for value in values:
            #     print(file[value])
            print(file.depth)
            print(file.temp_surface)
            fh_1_temp_surface = file.temp_surface.isel(time=0)
            print(fh_1_temp_surface.values)

if __name__ == "__main__":
    start = time.time()
    processor = SurgeTideForecastProcessor()
    processor.process(OUTPUTS)
    print(f'Finished: {time.time() - start}')