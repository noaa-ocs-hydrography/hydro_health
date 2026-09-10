from pathlib import Path
from typing import List, Union
import rasterio
from rasterio.merge import merge


def merge_geotiffs(
    input_paths: List[Union[str, Path]], 
    output_path: Union[str, Path]
) -> None:
    """
    Merges multiple GeoTIFF files into a single optimized output GeoTIFF.

    :param input_paths: List of file paths to the input GeoTIFFs.
    :param output_path: Path where the merged GeoTIFF will be saved.
    """
    if not input_paths:
        raise ValueError("The input paths list cannot be empty.")

    # Open all input raster files
    src_files_to_mosaic = [rasterio.open(path) for path in input_paths]

    try:
        # Get nodata value from the first file
        nodata_val = src_files_to_mosaic[0].nodata

        # Merge rasters specifying nodata handling
        mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata_val)

        # Copy metadata from the first raster
        out_meta = src_files_to_mosaic[0].meta.copy()

        # Update metadata to enable LZW compression, tiling, and correct dimensions
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": src_files_to_mosaic[0].crs,
                "compress": "lzw",       # Keeps the merged file size small
                "tiled": True,            # Stores raster in 256x256 tiles for speed & compression
                "blockxsize": 256,
                "blockysize": 256,
                "nodata": nodata_val
            }
        )

        # Write the merged mosaic to the output path
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)

        print(f"Successfully merged {len(input_paths)} GeoTIFFs into: {output_path}")

    finally:
        # Close all opened dataset handles
        for src in src_files_to_mosaic:
            src.close()


if __name__ == "__main__":
    # Your updated paths:
    geotiff_files = [
        r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\HHM_simple\outputs\Files_upload_to_ArcGIS_AGO\Inshore_20m\PSS_ER_1.tif",
        r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\HHM_simple\outputs\Files_upload_to_ArcGIS_AGO\Inshore_20m\PSS_ER_2.tif",
        r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\HHM_simple\outputs\Files_upload_to_ArcGIS_AGO\Inshore_20m\PSS_ER_3.tif",
        r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\HHM_simple\outputs\Files_upload_to_ArcGIS_AGO\Inshore_20m\PSS_ER_4.tif",
        r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\HHM_simple\outputs\Files_upload_to_ArcGIS_AGO\Inshore_20m\PSS_ER_5.tif",
        r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\HHM_simple\outputs\Files_upload_to_ArcGIS_AGO\Inshore_20m\PSS_ER_6.tif",
    ]
    
    destination = r"N:\CSDL\Projects\Hydro_Health_Model\HHM2025\working\HHM_Run\HHM_simple\outputs\Files_upload_to_ArcGIS_AGO\Inshore_20m\PSS_Eco_region_all_tiles_20m.tif"

    merge_geotiffs(geotiff_files, destination)