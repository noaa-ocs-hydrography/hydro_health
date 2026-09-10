import pathlib
import pandas as pd
from osgeo import gdal, osr

# Enable GDAL exceptions for explicit error handling
gdal.UseExceptions()

tiled_path = pathlib.Path(r'C:\Users\Stephen.Patterson\Data\Repos\hydro_health\outputs\ER_3\model_variables\Prediction\pre_processed\tiled')

# Collect .tif and .tiff files, excluding 'unused_providers'
tif_files = [f for f in tiled_path.rglob('*') if f.suffix.lower() in ('.tif', '.tiff') and 'unused_providers' not in str(f)]

def get_safe_units(srs):
    """Safely extract linear or angular units, supporting compound CRSs."""
    try:
        unit_name = srs.GetLinearUnitName()
        if unit_name:
            return unit_name
    except Exception:
        pass
    
    try:
        unit_name = srs.GetAttrValue('UNIT', 0)
        if unit_name:
            return unit_name
    except Exception:
        pass

    try:
        unit_name = srs.GetAngularUnitName()
        if unit_name:
            return unit_name
    except Exception:
        pass

    return 'Unknown'


report_data = []

for i, file_path in enumerate(tif_files, 1):
    file_info = {
        'File': file_path.name,
        'Path': str(file_path),
        'Status': 'OK',
        'Dimensions': None,
        'Bands': 0,
        'CRS_Name': None,
        'Units': None,
        'Res_X': None,
        'Res_Y': None,
        'NoData_Val': None,
        'Min': None,
        'Max': None,
        'Is_Empty': False
    }

    try:
        ds = gdal.Open(str(file_path), gdal.GA_ReadOnly)
        
        if ds is None:
            file_info['Status'] = 'Failed to Open'
            file_info['Is_Empty'] = True
            report_data.append(file_info)
            continue

        x_size, y_size = ds.RasterXSize, ds.RasterYSize
        file_info['Dimensions'] = f'{x_size}x{y_size}'
        file_info['Bands'] = ds.RasterCount

        # Flag zero or placeholder dimensions (<= 1px)
        if x_size <= 1 or y_size <= 1 or ds.RasterCount == 0:
            file_info['Status'] = 'Empty/Placeholder Dimension (<=1px)'
            file_info['Is_Empty'] = True

        # Extract Projection & Units
        srs_wkt = ds.GetProjection()
        if srs_wkt:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(srs_wkt)
            file_info['CRS_Name'] = srs.GetName() or 'Unknown CRS'
            file_info['Units'] = get_safe_units(srs)
        else:
            file_info['CRS_Name'] = 'Unassigned'

        # Extract Pixel Resolution
        gt = ds.GetGeoTransform()
        if gt:
            file_info['Res_X'] = abs(gt[1])
            file_info['Res_Y'] = abs(gt[5])

        # Read Stats from Band 1
        if ds.RasterCount > 0:
            band = ds.GetRasterBand(1)
            nodata = band.GetNoDataValue()
            file_info['NoData_Val'] = nodata

            try:
                # Use ReadAsArray for 1x1 or tiny rasters to prevent GDAL stat crashes
                if x_size <= 2 and y_size <= 2:
                    arr = band.ReadAsArray()
                    file_info['Min'] = float(arr.min())
                    file_info['Max'] = float(arr.max())
                else:
                    stats = band.ComputeRasterMinMax(False)
                    file_info['Min'] = stats[0]
                    file_info['Max'] = stats[1]

                # Check if entire raster contains only NoData
                if nodata is not None and file_info['Min'] == nodata and file_info['Max'] == nodata:
                    file_info['Status'] = 'Only NoData Values'
                    file_info['Is_Empty'] = True

            except Exception as e:
                file_info['Status'] = 'Band Read Failure'
                file_info['Is_Empty'] = True

        ds = None

    except Exception as e:
        file_info['Status'] = f'Error: {str(e)}'
        file_info['Is_Empty'] = True

    report_data.append(file_info)
    print(f'Processed {i}/{len(tif_files)}: {file_path.name}')

# Convert results into a DataFrame
df = pd.DataFrame(report_data)

# Save audit report to CSV
csv_out = 'geotiff_audit_report.csv'
df.to_csv(csv_out, index=False)

# Print summary to console
empty_count = df['Is_Empty'].sum()
print(f"\n--- Audit Complete ---")
print(f"Total files checked: {len(df)}")
print(f"Empty/Problematic files: {empty_count}")
print(f"Report saved to: {csv_out}")

# Print summary of unique units and resolutions found in valid files
valid_df = df[~df['Is_Empty']]
if not valid_df.empty:
    print("\n--- Valid Datasets Overview ---")
    print("Units found:", valid_df['Units'].unique().tolist())
    print("Unique Resolutions (X, Y):")
    print(valid_df[['Res_X', 'Res_Y']].drop_duplicates().to_string(index=False))