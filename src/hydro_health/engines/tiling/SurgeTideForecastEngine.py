import os
import time
import pathlib
import boto3
import xarray as xr  # pip install cfgrib required for cfgrib engine
import geopandas as gpd
import s3fs  # require conda install h5netcdf
import shapely
import numpy as np
import geopandas as gpd

from osgeo import gdal, osr
from shapely.geometry import Point
from shapely import wkt
from datetime import datetime
from botocore.client import Config
from botocore import UNSIGNED
from concurrent.futures import ThreadPoolExecutor


INPUTS = pathlib.Path(r'C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data')
OUTPUTS = pathlib.Path(__file__).parents[4] / 'outputs'


class SurgeTideForecastEngine:
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

    def add_zcoord_values(self, tile_indices: dict[list[int|float]]) -> None:
        """Read third non-nan index from first week Z Coordinates file"""

        grid_dataset = self.load_zcord_dataset()
        nowcast_hour = 0  # TODO should this be 0 or 1?
        chosen_level = 2
        for tile_id, info in tile_indices.items():
            node_indices = info['nodes']
            # TODO multiprocessing
            for node_index in node_indices:
                # levels change for each node, so you have to calc each iteration
                levels_array = grid_dataset.zCoordinates[nowcast_hour, node_index['node'], :].values
                non_nan_index = np.where(np.isfinite(levels_array))[0][0] + chosen_level
                # node_index['z_coord_level': levels_array[non_nan_index]]  # TODO we probably need the actual level right?
                node_index['z_coord_index'] = non_nan_index

    def add_dataset_third_value(self, tile_indices: dict[list[int|float]]) -> None:
        """Read third non-nan value from first week Z Coordinates file"""

        grid_dataset = self.load_x_vel_dataset(local=True)
        nowcast_hour = 0  # TODO should this be 0 or 1?

        for tile_id, info in tile_indices.items():
            node_indices = info['nodes']
            # TODO multiprocessing
            for node_index in node_indices:
                print(node_index['node'])
                x_vel_array = grid_dataset.horizontalVelX[nowcast_hour, node_index['node'], :].values
                node_index['x_velocity'] = x_vel_array[node_index['z_coord_index']]

    def load_fields_dataset(self, local=False) -> xr.Dataset:
        if local:
            fields_ds = xr.open_dataset(r"C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data\stofs_3d_atl_20230112\stofs_3d_atl.t12z.n001_024.field2d.nc")
        else:
            s3 = self.get_s3_filesystem()
            first_week_z_coords_dataset = "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.n001_024.field2d.nc"
            url = f"s3://{self.bucket}/{first_week_z_coords_dataset}"
            fields_ds = xr.open_dataset(s3.open(url, 'rb'))
        return fields_ds

    def load_zcord_dataset(self, local=False) -> xr.Dataset:
        """Get the z-coordinate dataset"""

        if local:
            z_ds = xr.open_dataset(r'C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data\stofs_3d_atl_20230112\stofs_3d_atl.t12z.fields.zCoordinates_nowcast.nc')
        else:
            s3 = self.get_s3_filesystem()
            first_week_z_coords_dataset = "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.fields.zCoordinates_nowcast.nc"
            url = f"s3://{self.bucket}/{first_week_z_coords_dataset}"
            z_ds = xr.open_dataset(s3.open(url, 'rb'))
        return z_ds

    def load_x_vel_dataset(self, local=False) -> xr.Dataset:
        "Get the x-velocity dataset"

        if local:
            ds = xr.open_dataset(r'C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data\stofs_3d_atl_20230112\stofs_3d_atl.t12z.fields.horizontalVelX_nowcast.nc')
        else:
            s3 = self.get_s3_filesystem()
            first_week_z_coords_dataset = "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.fields.horizontalVelX_nowcast.nc"
            url = f"s3://{self.bucket}/{first_week_z_coords_dataset}"
            ds = xr.open_dataset(s3.open(url, 'rb'))
        return ds
    
    def load_y_vel_dataset(self, local=False) -> xr.Dataset:
        "Get the y-velocity dataset"

        if local:
            ds = xr.open_dataset(r'C:\Users\Stephen.Patterson\Data\Projects\HydroHealth\STOFS_data\stofs_3d_atl_20230112\stofs_3d_atl.t12z.fields.horizontalVelY_nowcast.nc')
        else:
            s3 = self.get_s3_filesystem()
            first_week_z_coords_dataset = "STOFS-3D-Atl/stofs_3d_atl.20230112/stofs_3d_atl.t12z.fields.horizontalVelY_nowcast.nc"
            url = f"s3://{self.bucket}/{first_week_z_coords_dataset}"
            ds = xr.open_dataset(s3.open(url, 'rb'))
        return ds

    def process(self, tile_gdf: gpd.GeoDataFrame, outputs: str = False) -> None:
        """Main entry point for downloading Digital Coast data"""

        tile_indices = self.get_tile_indices(tile_gdf)
        self.add_zcoord_values(tile_indices)  # this updates tile_indices dict
        self.add_dataset_third_value(tile_indices)
        print(tile_indices['BF2H62K6'])
        self.build_tile_raster(tile_indices)

    def get_tile_indices(self, tile_gdf: gpd.GeoDataFrame):
        grid_dataset = self.load_fields_dataset(local=True)
        lons = grid_dataset.SCHISM_hgrid_node_x.values   # 2d array of lons, min to max
        lats = grid_dataset.SCHISM_hgrid_node_y.values
        tile_indices = {}
        for _, row in tile_gdf.iterrows():
            indices = []
            print(row.geometry)
            if isinstance(row[1], str):
                print(row.tile)
                for i, (lon, lat) in enumerate(zip(lons, lats)):
                    if row.geometry.contains(Point(lon, lat)):
                        indices.append({'node': i})
                tile_indices[row.tile] = {
                    "wkt": row.geometry,
                    "nodes": indices,
                    "latitudes": [lats[index['node']] for index in indices],
                    "longitudes": [lons[index['node']] for index in indices]
                }
            if _ == 2:
                break
        return tile_indices
    
    def build_tile_raster(self, tile_indeces):
        pixel = 0.015 # which resolution to use for STOFS?
        no_data = -9999
        
        for tile_id, tile_data in tile_indeces.items():
            polygon_bounds = wkt.loads(tile_data['wkt']).bounds
            # min_lat, min_lon, max_lat, max_lon = min(tile_data['latitudes']), min(tile_data['longitudes']), max(tile_data['latitudes']), max(tile_data['longitudes'])
            min_lon, min_lat, max_lon, max_lat = polygon_bounds
            print(polygon_bounds)
            cols = int((max_lon - min_lon) / pixel)
            rows = int((max_lat - min_lat) / pixel) + 1
            print(cols, rows)

            driver = gdal.GetDriverByName("MEM")  # You can choose a different format like "GTiff"
            dataset = driver.Create('', cols, rows, 1, gdal.GDT_Float32) #
            dataset.SetGeoTransform((min_lon, pixel, 0, max_lat, 0, -pixel))
            
            srs = osr.SpatialReference()
            srs.SetWellKnownGeogCS("EPSG:4326")
            dataset.SetProjection(srs.ExportToWkt())

            band = dataset.GetRasterBand(1)
            band.SetNoDataValue(no_data)
            band.Fill(no_data)

            geotransform = dataset.GetGeoTransform()
            inv_geotransform = gdal.InvGeoTransform(geotransform)

            for i in range(len(tile_data['latitudes'])):
                lon = tile_data['longitudes'][i]
                lat = tile_data['latitudes'][i]
                value = tile_data['nodes'][i]['x_velocity']

                # Convert geographic coordinates to pixel coordinates
                col, row = gdal.ApplyGeoTransform(inv_geotransform, lon, lat) #
                col = int(col)
                row = int(row)

                # Write the value to the raster band
                if 0 <= row < rows and 0 <= col < cols:
                    band.WriteArray(np.array([[value]]), col, row)

            driver = gdal.GetDriverByName("GTiff")
            output_tiff_path = str(OUTPUTS / 'stofs_tile.tif')
            output_dataset = driver.CreateCopy(output_tiff_path, dataset)
            output_dataset = None  # Close the output dataset
            dataset = None
            
            return output_tiff_path
        

    def interpolate_raster(input_raster):
        output_driver = gdal.GetDriverByName('GTiff')
        src_ds = gdal.Open(input_raster)
        output_path = str(OUTPUTS / 'continous_stofs_tile.tif')
        output_ds = output_driver.CreateCopy(output_path, src_ds)

        stofs_band = output_ds.GetRasterBand(1)

        search_distance = 8  # 8 is minimum pixels that filled one tile.  Might be different with other scales
        gdal.FillNodata(stofs_band, maskBand=None, maxSearchDist=search_distance, smoothingIterations=1)
        
        output_ds = None
        src_ds = None


if __name__ == "__main__":
    start = time.time()
    engine = SurgeTideForecastEngine()
    engine.run(OUTPUTS)
    print(f'Finished: {time.time() - start}')
