from __future__ import annotations

import re
import pathlib
import fsspec
import geopandas as gpd
import fnmatch
import datetime
import xarray as xr
import numpy as np
import s3fs
import tempfile
import time

from collections import deque
from collections.abc import Iterator

from ftplib import FTP
from osgeo import gdal, osr

from hydro_health.engines.Engine import Engine
from hydro_health.helpers.tools import get_config_item


INPUTS = pathlib.Path(__file__).parents[3] / 'inputs'
OUTPUTS = pathlib.Path(__file__).parents[3] / 'outputs'

class HydroHealthCredentialsError(Exception):
    pass

class HydroHealthConfig:
    def __init__(self) -> None:
        super().__init__()
        self.config_path = INPUTS / 'lookups' / 'glob_tsm.config'
        self.username = None
        self.password = None
        self._load_credentials()

    def _load_credentials(self) -> None:
        """Read credentials from config file"""

        # you need the inputs/lookups/glob_tsm.config file which is not stored in the repo for security reasons
        if not self.config_path.exists():
            raise HydroHealthCredentialsError(f"Config file not found: {self.config_path}")

        creds = {}
        with open(self.config_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if '=' in line:
                    key, value = line.split("=", 1)
                    creds[key.strip()] = value.strip()

        try:
            self.username = creds["username"]
            self.password = creds["password"]
        except KeyError as e:
            raise HydroHealthCredentialsError(f"Missing value in config: {e}")

class CreateTSMLayerEngine(Engine):
    def __init__(self, param_lookup: dict, output_prefix: str | bool = False) -> None:
        super().__init__()
        self.param_lookup = param_lookup
        self.output_prefix = output_prefix

        creds = HydroHealthConfig()
        self.username = creds.username
        self.password = creds.password
        self.server = 'ftp.hermes.acri.fr'
        self.downloaded_files = []
        self.ftp_file_paths = {}
        self.ftp_directory_cache = {}
        self.output_folder = OUTPUTS
        
        # Flag to control whether to overwrite existing NC files or just check for new ones
        self.force_download = False
        # Independently control replacement of annual and year-pair rasters.
        self.overwrite_existing = True
        # Window size bounds raster read/write and year-pair averaging allocations.
        self.chunk_size = 512
        
    def _resolve_paths(self, region: str) -> None:
        """Configure storage and paths for the selected eco region."""

        self.is_aws = self.param_lookup['env'] in ('remote', 'aws')
        tsm_data_path = str(get_config_item('TSM', 'DATA_PATH')).strip('/')
        subfolder = str(get_config_item('TSM', 'SUBFOLDER')).strip('/')
        mask_path = str(get_config_item('MASK', 'MASK_PRED_PATH')).strip('/')
        prefix = str(self.output_prefix).strip('/') if self.output_prefix else ''
        self.output_folder = OUTPUTS / prefix if prefix else OUTPUTS
        self.outputs_dir = self.output_folder / region
        self.download_log_path = self.output_folder / 'downloaded_files.txt'

        if self.is_aws:
            bucket = str(get_config_item('SHARED', 'OUTPUT_BUCKET'))
            bucket = bucket.removeprefix('s3://').strip('/')
            shared_base = f's3://{bucket}/{prefix}' if prefix else f's3://{bucket}'
            base_raster = f'{shared_base}/{region}/{subfolder}'
            self.prediction_mask_path = f'{shared_base}/{region}/{mask_path}'
            self.nc_files_path = f'{shared_base}/{tsm_data_path}'
            self.filesystem = s3fs.S3FileSystem()
        else:
            base_raster = self.outputs_dir / subfolder
            self.prediction_mask_path = self.outputs_dir / mask_path
            self.nc_files_path = self.output_folder / tsm_data_path
            self.filesystem = fsspec.filesystem('file', auto_mkdir=True)

        self.raster_path = f'{base_raster}/mean_rasters'
        self.year_pair_path = f'{base_raster}/TSM_year_pair_rasters'

    def _open_raster(self, path: str | pathlib.Path) -> gdal.Dataset:
        """Open local or S3 rasters with GDAL."""

        path = str(path)
        if path.startswith('s3://'):
            path = path.replace('s3://', '/vsis3/', 1)
        dataset = gdal.Open(path, gdal.GA_ReadOnly)
        if dataset is None:
            raise OSError(f'Cannot open raster: {path}')
        return dataset

    def _prepare_region_mask(self) -> None:
        """Read only the prediction raster's extent and CRS; ignore pixel values."""

        src = self._open_raster(self.prediction_mask_path)
        try:
            if not src.GetProjection():
                raise ValueError('Prediction raster must have a CRS')
            transform = src.GetGeoTransform()
            if transform[2] != 0 or transform[4] != 0 or transform[1] <= 0 or transform[5] >= 0:
                raise ValueError('Prediction raster must use a north-up, unrotated grid')
            self.region_transform = transform
            self.region_projection = src.GetProjection()
            self.region_shape = (src.RasterYSize, src.RasterXSize)
        finally:
            src = None

    def _native_crop(self, shape: tuple[int, int], transform: tuple[float, ...], crs: str) -> tuple[slice, slice, tuple[float, ...]]:
        """Transform densified mask bounds and snap outward to native pixels."""

        source_crs = osr.SpatialReference()
        source_crs.ImportFromWkt(self.region_projection)
        native_crs = osr.SpatialReference()
        if native_crs.SetFromUserInput(str(crs)) != 0:
            raise ValueError(f'Invalid native CRS: {crs}')
        source_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        native_crs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        conversion = osr.CoordinateTransformation(source_crs, native_crs)
        t = self.region_transform
        height, width = self.region_shape
        left, top = t[0], t[3]
        right, bottom = left + width * t[1], top + height * t[5]
        points = []
        for fraction in np.linspace(0, 1, 101):
            x = left + fraction * (right - left)
            y = bottom + fraction * (top - bottom)
            points.extend([(x, bottom), (x, top), (left, y), (right, y)])
        projected = np.asarray(conversion.TransformPoints(points))[:, :2]
        if not np.all(np.isfinite(projected)):
            raise ValueError('Could not transform prediction mask bounds to the native CRS')
        xmin, ymin = projected.min(axis=0)
        xmax, ymax = projected.max(axis=0)
        col_start = max(0, int(np.floor((xmin - transform[0]) / transform[1])))
        col_stop = min(shape[1], int(np.ceil((xmax - transform[0]) / transform[1])))
        row_start = max(0, int(np.floor((ymax - transform[3]) / transform[5])))
        row_stop = min(shape[0], int(np.ceil((ymin - transform[3]) / transform[5])))
        if col_start >= col_stop or row_start >= row_stop:
            raise ValueError('Prediction mask bounds do not overlap the native TSM grid')
        cropped_transform = (
            transform[0] + col_start * transform[1], transform[1], 0.0,
            transform[3] + row_start * transform[5], 0.0, transform[5],
        )
        return slice(row_start, row_stop), slice(col_start, col_stop), cropped_transform

    def _windows(self, height: int, width: int) -> Iterator[tuple[int, int, int, int]]:
        """Yield GDAL x/y offsets and widths/heights."""

        for row in range(0, height, self.chunk_size):
            for col in range(0, width, self.chunk_size):
                yield col, row, min(self.chunk_size, width - col), min(self.chunk_size, height - row)

    def _read_window(self, dataset: gdal.Dataset, window: tuple[int, int, int, int]) -> np.ndarray:
        """Read a single-band chunk and honor its NoData and validity mask."""

        band = dataset.GetRasterBand(1)
        values = band.ReadAsArray(*window)
        if values is None:
            raise OSError('GDAL failed to read raster window')
        values = values.astype(np.float32, copy=False)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            values[values == nodata] = np.nan
        if not (band.GetMaskFlags() & gdal.GMF_ALL_VALID):
            valid = band.GetMaskBand().ReadAsArray(*window)
            if valid is None:
                raise OSError('GDAL failed to read validity mask')
            values[valid == 0] = np.nan
        return values

    def _create_raster(self, path: str | pathlib.Path, shape: tuple[int, int], transform: tuple[float, ...], projection: str, nodata: float = np.nan) -> gdal.Dataset:
        """Create a tiled on-disk Float32 raster."""

        dst = gdal.GetDriverByName('GTiff').Create(
            str(path), shape[1], shape[0], 1, gdal.GDT_Float32,
            options=['TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512',
                     'COMPRESS=DEFLATE', 'BIGTIFF=IF_SAFER'],
        )
        if dst is None:
            raise OSError(f'Cannot create raster: {path}')
        dst.SetGeoTransform(transform)
        dst.SetProjection(projection)
        dst.GetRasterBand(1).SetNoDataValue(float(nodata))
        return dst

    def _list_files(self, directory: str | pathlib.Path, suffix: str) -> list[str]:
        """List files using the configured filesystem and base directory."""

        if not self.filesystem.exists(str(directory)):
            return []
        return [
            f'{directory}/{pathlib.PurePosixPath(path).name}'
            for path in self.filesystem.ls(str(directory), detail=False)
            if path.endswith(suffix)
        ]

    def _store_raster_data(self, data: np.ndarray, directory: str | pathlib.Path, filename: str, transform: tuple[float, ...], crs: str, nodata_val: float) -> None:
        """Crop annual means to transformed mask bounds without resampling or masking."""

        rows, cols, transform = self._native_crop(data.shape, transform, crs)
        data = data[rows, cols]

        destination = f'{directory}/{filename}'
        self.filesystem.makedirs(str(directory), exist_ok=True)
        source_crs = osr.SpatialReference()
        if source_crs.SetFromUserInput(str(crs)) != 0:
            raise ValueError(f'Invalid source CRS: {crs}')
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = pathlib.Path(temporary_dir) / filename
            dst = None
            try:
                dst = self._create_raster(
                    output_path, data.shape, transform, source_crs.ExportToWkt(), nodata_val
                )
                for col, row, width, height in self._windows(*data.shape):
                    if dst.GetRasterBand(1).WriteArray(
                        data[row:row + height, col:col + width], col, row
                    ) != 0:
                        raise OSError('GDAL failed to write annual mean window')
                dst.FlushCache()
            finally:
                dst = None
            self.filesystem.put_file(str(output_path), destination)

    def extract_start_date(self, filename: str) -> str:
        """Parse file name for start date"""

        match = re.search(r'(\d{8})-\d{8}', filename)
        return match.group(1) if match else '99999999'

    def write_download_log(self) -> None:
        """Helper function to log downloaded NC files"""

        self.write_message("Writing download log...", OUTPUTS)
        sorted_files = sorted(self.downloaded_files, key=self.extract_start_date)
        log_path = self.download_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'w') as f:
            for name in sorted_files:
                f.write(name + '\n')
        self.write_message(f"\nDownload log written to: {log_path}", OUTPUTS)

    def download_recursive(self, ftp: FTP, current_dir: str) -> None:
        """Recursive method to download all NetCDF files"""

        try:
            ftp.cwd(current_dir)
            normalized = current_dir.lower().strip('/')
            
            skip_patterns = (
                '*meris/*', '*viirsn*', '*seawifs*', '*modis*',
                '*viirsj1*', '*meris4rp*', '*olcib*',
                '*/month/*', '*/day/*', '*/track/*',
                '*/olcia/*/2019/*', '*/olcia/*/2020/*', '*/olcia/*/2021/*',
                '*/olcia/*/2022/*', '*/olcia/*/2023/*', '*/olcia/*/2024/*', '*/olcia/*/2025/*'
            )

            if any(fnmatch.fnmatch(normalized, pattern.strip('/')) for pattern in skip_patterns):
                return

            items = list(ftp.mlsd())
            for name, facts in items:
                item_path = f"{current_dir}/{name}"
                if facts.get("type") == "dir":
                    self.download_recursive(ftp, item_path)
                elif name.endswith('.nc') and 'TSM_8D' in name and '_4' in name and 'L3m' in name:
                    # Date filtering for OLCIA
                    if 'AV-OLA' in name:
                        match = re.search(r'L3m_(\d{8})-', name)
                        if match:
                            file_date = datetime.datetime.strptime(match.group(1), "%Y%m%d")
                            if file_date > datetime.datetime(2018, 5, 1):
                                continue

                    self.ftp_file_paths[name] = item_path
                    destination = f'{self.nc_files_path}/{name}'
                    self.filesystem.makedirs(str(self.nc_files_path), exist_ok=True)
                    if not self.force_download and self.filesystem.exists(str(destination)):
                        self.write_message(f"File {name} already exists at {destination}. Skipping download.", OUTPUTS)
                        continue

                    self.write_message(f"Downloading {name} to: {destination}...", OUTPUTS)
                    self._download_verified_nc(ftp, item_path, destination)
                    self.downloaded_files.append(name)
        except Exception as e:
            self.write_message(f"Error accessing {current_dir}: {e}", OUTPUTS)

    def _download_verified_nc(self, ftp: FTP, remote_path: str, destination: str) -> None:
        """Stage on disk and validate all TSM chunks before replacing the destination."""

        with tempfile.TemporaryDirectory() as temporary_dir:
            local_path = pathlib.Path(temporary_dir) / pathlib.PurePosixPath(destination).name
            ftp.voidcmd('TYPE I')
            expected_size = ftp.size(remote_path)
            size_label = f'{expected_size / 1024**2:.1f} MiB' if expected_size is not None else 'size unknown'
            self.write_message(f'  Downloading {remote_path} ({size_label})', OUTPUTS)
            received = 0
            last_report = time.monotonic()
            with local_path.open('wb') as writer:
                def write_block(block: bytes) -> None:
                    nonlocal received, last_report
                    writer.write(block)
                    received += len(block)
                    now = time.monotonic()
                    if now - last_report >= 2:
                        progress = f'{received / 1024**2:.1f} MiB'
                        if expected_size:
                            progress += f' / {expected_size / 1024**2:.1f} MiB ({100 * received / expected_size:.0f}%)'
                        self.write_message(f'    Downloaded {progress}', OUTPUTS)
                        last_report = now
                ftp.retrbinary(f'RETR {remote_path}', write_block, blocksize=256 * 1024)
            self.write_message(f'  Download complete: {received / 1024**2:.1f} MiB', OUTPUTS)
            if expected_size is not None and local_path.stat().st_size != expected_size:
                raise IOError(f'Incomplete FTP download: {remote_path}')
            self.write_message('  Validating downloaded NetCDF...', OUTPUTS)
            with xr.open_dataset(local_path, engine='h5netcdf', cache=False) as ds:
                variable = ds['TSM_mean'].squeeze()
                if variable.shape != (4320, 8640):
                    raise ValueError(f'Unexpected NetCDF shape: {variable.shape}')
                # Read small row blocks so validation does not load a global array.
                for row in range(0, variable.shape[0], 128):
                    variable.isel({variable.dims[0]: slice(row, row + 128)}).values
            self.write_message(f'  Validation passed. Uploading replacement to {destination}...', OUTPUTS)
            self.filesystem.put_file(str(local_path), str(destination))
            self.filesystem.invalidate_cache(str(destination))
            self.write_message('  Replacement complete.', OUTPUTS)

    def _find_ftp_file(self, ftp: FTP, name: str) -> str:
        """Search compatible sensor/year branches, caching paths and listings."""

        if name in self.ftp_file_paths:
            self.write_message(f'  Using cached FTP path: {self.ftp_file_paths[name]}', OUTPUTS)
            return self.ftp_file_paths[name]
        date_match = re.search(r'L3m_(\d{4})', name)
        year = date_match.group(1) if date_match else None
        sensor_match = re.search(r'AV-([A-Z0-9]+)_', name)
        sensor = sensor_match.group(1) if sensor_match else None
        aliases = {
            'MER': ('mer', 'meris', 'meris4rp'),
            'OLA': ('ola', 'olcia'), 'OLB': ('olb', 'olcib'),
        }
        sensor_names = aliases.get(sensor, ())
        known_sensors = ('meris', 'olcia', 'olcib', 'modis', 'viirs', 'seawifs')
        directories = deque(['/GLOB'])
        started = time.monotonic()
        self.write_message(f'  Finding FTP file (sensor={sensor}, year={year})...', OUTPUTS)
        while directories:
            if time.monotonic() - started > 180:
                raise TimeoutError(f'FTP file search exceeded 180 seconds: {name}')
            directory = directories.popleft()
            self.write_message(f'    Searching {directory}', OUTPUTS)
            if directory not in self.ftp_directory_cache:
                self.ftp_directory_cache[directory] = list(ftp.mlsd(directory))
            children = []
            for item_name, facts in self.ftp_directory_cache[directory]:
                remote_path = f'{directory}/{item_name}'
                if facts.get('type') != 'dir':
                    if item_name.endswith('.nc'):
                        self.ftp_file_paths[item_name] = remote_path
                    if item_name == name:
                        self.write_message(f'  Found FTP file: {remote_path}', OUTPUTS)
                        return remote_path
                    continue
                lower = item_name.lower()
                if year and re.fullmatch(r'(?:19|20)\d{2}', lower) and lower != year:
                    continue
                if lower in ('day', 'daily', 'month', 'monthly', 'track'):
                    continue
                if sensor_names and any(label in lower for label in known_sensors):
                    if not any(label in lower for label in sensor_names):
                        continue
                priority = 0 if lower == year or any(label in lower for label in sensor_names) else 1
                children.append((priority, remote_path))
            # Prefer matching branches immediately over generic sibling directories.
            for _, child in reversed(sorted(children)):
                directories.appendleft(child)
        raise FileNotFoundError(f'NetCDF not found in compatible FTP branches: {name}')

    def _redownload_corrupted_nc(self, destination: str) -> None:
        """Find and replace only the damaged file, with visible progress."""

        name = pathlib.PurePosixPath(destination).name
        self.write_message(f'  Repairing corrupted NetCDF: {name}', OUTPUTS)
        self.write_message(f'  Connecting to FTP server {self.server}...', OUTPUTS)
        with FTP(self.server, timeout=30) as ftp:
            ftp.login(user=self.username, passwd=self.password)
            remote_path = self._find_ftp_file(ftp, name)
            self._download_verified_nc(ftp, remote_path, destination)
            self.downloaded_files.append(name)

    def _read_tsm_array(self, path: str | pathlib.Path) -> np.ndarray:
        """Retry once after repairing an HDF5 corruption/read failure."""

        for attempt in range(2):
            try:
                with self.filesystem.open(str(path), 'rb') as source:
                    with xr.open_dataset(source, engine='h5netcdf', cache=False) as ds:
                        arr = ds['TSM_mean'].values.squeeze().astype(np.float32)
                if arr.shape != (4320, 8640):
                    raise ValueError(f'Unexpected NetCDF shape {arr.shape}; expected (4320, 8640)')
                arr[arr == 0] = np.nan
                return arr
            except (OSError, ValueError) as error:
                message = str(error).lower()
                corruption = any(token in message for token in (
                    'truncated file', 'file signature not found', 'bad object header',
                    'inflate() failed', 'filter returned failure', 'checksum',
                    'unable to synchronously read', 'address overflow',
                ))
                if attempt or not corruption:
                    raise
                self._redownload_corrupted_nc(str(path))

    def download_tsm_data(self) -> None:
        """FTP startup method to download all TSM data"""

        self.write_message('Connecting to FTP server...', OUTPUTS)
        ftp = FTP(self.server)
        ftp.login(user=self.username, passwd=self.password)
        try:
            self.download_recursive(ftp, '/GLOB')
        finally:
            ftp.quit()
        self.write_download_log()

    def create_mean_year_rasters(self) -> None:
        """Create mean year raster files from all NetCDF files"""

        grid_shape = (4320, 8640)
        transform = (-180.0, 360.0 / grid_shape[1], 0.0,
                     90.0, 0.0, -180.0 / grid_shape[0])
        crs = "EPSG:4326" # the nc files use this crs
        nodata_val = np.nan

        for year in range(1998, 2025):
            filename = f'TSM_mean_{year}.tif'
            destination = f'{self.raster_path}/{filename}'
            if not self.overwrite_existing and self.filesystem.exists(destination):
                self.write_message(f'Annual raster already exists: {destination}. Skipping.', OUTPUTS)
                continue
            self.write_message(f'Processing year: {year}', OUTPUTS)
            nc_files = [
                path for path in self._list_files(self.nc_files_path, '.nc')
                if f"L3m_{year}" in pathlib.PurePosixPath(str(path)).name
            ]

            if not nc_files:
                self.write_message(f'  No NetCDF files found for {year}', OUTPUTS)
                continue

            running_sum, valid_count = None, None
            annual_mean = None
            for fname in nc_files:
                try:
                    arr = self._read_tsm_array(fname)
                    if running_sum is None:
                        running_sum = np.zeros_like(arr)
                        valid_count = np.zeros_like(arr, dtype=np.uint16)
                    valid = np.isfinite(arr)
                    np.add(running_sum, arr, out=running_sum, where=valid)
                    np.add(valid_count, 1, out=valid_count, where=valid)
                    del arr, valid
                except MemoryError:
                    raise
                except Exception as e:
                    self.write_message(f"  Failed: {fname}: {e}", OUTPUTS)

            if running_sum is None:
                continue

            # Reuse the sum buffer instead of allocating another global float array.
            has_data = valid_count > 0
            np.divide(running_sum, valid_count, out=running_sum, where=has_data)
            running_sum[~has_data] = nodata_val
            annual_mean = running_sum
            del valid_count, has_data

            self._store_raster_data(
                annual_mean, self.raster_path, filename, transform, crs, nodata_val
            )

    def year_pair_rasters(self, start_year: int, end_year: int) -> None:
        """Average native annual rasters in chunks, cropped to prediction bounds."""

        out_name_mean = f'tsm_mean_{start_year}_{end_year}.tif'
        destination = f'{self.year_pair_path}/{out_name_mean}'
        if not self.overwrite_existing and self.filesystem.exists(destination):
            self.write_message(f'Year-pair raster already exists: {destination}. Skipping.', OUTPUTS)
            return

        raster_files = [
            path for path in self._list_files(self.raster_path, '.tif')
            if any(str(year) in pathlib.PurePosixPath(str(path)).name
                   for year in range(start_year, end_year + 1))
        ]

        if not raster_files: return

        self.filesystem.makedirs(str(self.year_pair_path), exist_ok=True)
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = pathlib.Path(temporary_dir) / out_name_mean
            sources = []
            first = source = dst = None
            try:
                for path in raster_files:
                    source = self._open_raster(path)
                    sources.append(source)
                    actual_crs = osr.SpatialReference()
                    actual_crs.ImportFromWkt(source.GetProjection())
                    expected_crs = osr.SpatialReference()
                    expected_crs.SetFromUserInput('EPSG:4326')
                    native_transform = source.GetGeoTransform()
                    if (not actual_crs.IsSame(expected_crs)
                            or not np.allclose(
                                [native_transform[1], native_transform[5]],
                                [360.0 / 8640, -180.0 / 4320], rtol=0, atol=1e-12
                            ) or native_transform[2] != 0 or native_transform[4] != 0):
                        raise ValueError(f'Annual mean must use the native CRS and resolution: {path}')
                first = sources[0]
                for source in sources[1:]:
                    if (source.RasterXSize != first.RasterXSize
                            or source.RasterYSize != first.RasterYSize
                            or not np.allclose(source.GetGeoTransform(), first.GetGeoTransform(), rtol=0, atol=1e-9)):
                        raise ValueError('Annual mean grids differ; rebuild annual means consistently')
                rows, cols, output_transform = self._native_crop(
                    (first.RasterYSize, first.RasterXSize),
                    first.GetGeoTransform(), first.GetProjection(),
                )
                output_shape = (rows.stop - rows.start, cols.stop - cols.start)
                dst = self._create_raster(
                    output_path, output_shape, output_transform, first.GetProjection()
                )
                for window in self._windows(*output_shape):
                    source_window = (
                        window[0] + cols.start, window[1] + rows.start, window[2], window[3]
                    )
                    shape = (window[3], window[2])
                    total = np.zeros(shape, dtype=np.float32)
                    count = np.zeros(shape, dtype=np.uint16)
                    for source in sources:
                        values = self._read_window(source, source_window)
                        valid = np.isfinite(values)
                        np.add(total, values, out=total, where=valid)
                        np.add(count, 1, out=count, where=valid)
                    has_data = count > 0
                    np.divide(total, count, out=total, where=has_data)
                    total[~has_data] = np.nan
                    if dst.GetRasterBand(1).WriteArray(total, window[0], window[1]) != 0:
                        raise OSError('GDAL failed to write year-pair window')
                dst.FlushCache()
            finally:
                first = source = dst = None
                sources.clear()
            self.filesystem.put_file(str(output_path), destination)

    def run(self) -> None:
        """Main entry point for the engine; orchestrates downloading, processing, and raster creation."""
    
        for eco_region in self.param_lookup['eco_regions'].value:
            self._resolve_paths(eco_region)
            self._prepare_region_mask()
            self.create_mean_year_rasters()
            for start_year, end_year in self.year_ranges:
                self.year_pair_rasters(start_year, end_year)
