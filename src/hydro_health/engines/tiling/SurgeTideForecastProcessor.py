import os
import pathlib
import xarray as xr


class SurgeTideForecastProcessor:
    """Download and convert any STOFS data"""

    def process(self):
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

        daily_grib = pathlib.Path(r"C:\Users\Stephen.Patterson\Downloads\stofs_3d_atl.t12z.conus.east.cwl.grib2")
        self.export_hourly_rasters(daily_grib)

    def export_hourly_rasters(self, daily_grib: pathlib.Path) -> None:
        # open daily_grib
        grib_ds = xr.open_dataset(daily_grib)   # ['netcdf4', 'scipy', 'rasterio', 'store']
        # find datasets for all hours
        print(grib_ds.keys())
        print(grib_ds)

if __name__ == "__main__":
    processor = SurgeTideForecastProcessor()
    processor.process()