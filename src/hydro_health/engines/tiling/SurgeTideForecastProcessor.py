import os
import pathlib
import boto3
import xarray as xr  # pip install cfgrib required for cfgrib engine
import geopandas as gpd

from botocore.client import Config
from botocore import UNSIGNED


OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


class SurgeTideForecastProcessor:
    """Download and convert any STOFS data"""

    def download_water_velocity_netcdf(self) -> None:
        stofs_bucket = self.get_bucket()
        start = 0
        for obj_summary in stofs_bucket.objects.filter(Prefix=f"STOFS-3D-Atl/stofs_3d_atl."):
            # This gets us into the main folder
            # TODO need to secondary filter obj_summary for 
            if 'horizontalVelX_forecast' in obj_summary.key:
                print('X:', obj_summary.key)
                start += 1
            if 'horizontalVelY_forecast' in obj_summary.key:
                print('Y:', obj_summary.key)
                start += 1
            if start == 10:
                break

        # TODO cleaner to use s3fs library? 
        # Sample code uses xarray to directly load 5GB .nc file from s3
        # need to test how slow that is compared to downloading
        # s3 = s3fs.S3FileSystem(anon=True)  # Enable anonymous access to the S3 bucket
        # url = f"s3://{bucket_name}/{key}"
        # ds = xr.open_dataset(s3.open(url, 'rb'), drop_variables=['nvel'])

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


        self.download_water_velocity_netcdf()

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

        # daily_grib = pathlib.Path(r"C:\Users\Stephen.Patterson\Downloads\stofs_3d_atl.t12z.conus.east.f000.grib2")
        # daily_grib = pathlib.Path(r"C:\Users\Stephen.Patterson\Downloads\stofs_3d_atl.t12z.conus.east.cwl.grib2")
        # daily_netcdf = pathlib.Path(r"C:\Users\Stephen.Patterson\Downloads\stofs_3d_atl.t12z.n001_024.field2d.nc")
        # self.export_hourly_rasters(daily_netcdf)

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
    processor = SurgeTideForecastProcessor()
    processor.process(OUTPUTS)