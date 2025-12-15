import tempfile
import boto3
import os
import pathlib
# ... (keep your other imports)

def _process_tile(param_inputs: list[list]) -> None:
    """
    Static function that handles processing of a single tile.
    Refactored for S3-to-S3 workflow using ephemeral storage.
    """
    # Note: 'output_folder' now represents the S3 KEY PREFIX, not a local path
    s3_output_bucket, s3_prefix, tile_id, ecoregion_id = param_inputs

    # Create a temporary directory. This is physically located on the EC2 disk.
    # When the 'with' block ends, this folder and all files are auto-deleted.
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = pathlib.Path(temp_dir)
        
        engine = BlueTopoEngine()

        # 1. DOWNLOAD (Updated to save to temp_path)
        tiff_file_path = engine.download_nbs_tile(temp_path, tile_id, ecoregion_id)
        
        if tiff_file_path:
            # 2. PROCESS (Your existing logic runs locally on the temp drive)
            # No changes needed to these methods!
            engine.create_survey_end_date_tiff(tiff_file_path)
            mb_tiff_file = engine.rename_multiband(tiff_file_path)
            engine.multiband_to_singleband(mb_tiff_file, band=1)
            engine.multiband_to_singleband(mb_tiff_file, band=2)
            mb_tiff_file.unlink() 
            engine.set_ground_to_nodata(tiff_file_path)
            engine.create_slope(tiff_file_path)
            engine.create_rugosity(tiff_file_path)

            # 3. UPLOAD (Sync the temp folder results back to S3)
            # We assume anything left in the temp folder is a result we want.
            engine.upload_directory_to_s3(
                local_dir=temp_path, 
                bucket_name=s3_output_bucket, 
                s3_prefix=f"{s3_prefix}/{ecoregion_id}/{tile_id}"
            )

class BlueTopoEngine(Engine):
    # ... (Keep your __init__ and processing methods exactly as they are) ...

    def download_nbs_tile(self, output_folder: pathlib.Path, tile_id: str, ecoregion_id: str) -> pathlib.Path:
        """
        Modified to accept a pathlib.Path object for output_folder 
        and download to that specific local path.
        """
        nbs_bucket = self.get_bucket()
        output_tile_path = None
        
        # Determine the destination subfolder structure inside the temp dir
        # e.g., /tmp/xyz/US4NC/BlueTopo/...
        
        for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
            # We flatten the path structure slightly for the temp dir processing
            # or keep it if your processing logic relies on specific parent folder structures.
            file_name = pathlib.Path(obj_summary.key).name
            local_file = output_folder / file_name
            
            if local_file.suffix == '.tiff':
                output_tile_path = local_file
            
            # Download to the temp directory
            # No need to check if exists, because temp dir is always empty on start
            nbs_bucket.download_file(obj_summary.key, str(local_file))
            
        return output_tile_path

    def upload_directory_to_s3(self, local_dir: pathlib.Path, bucket_name: str, s3_prefix: str) -> None:
        """
        Walks the temporary directory and uploads all generated files to S3.
        """
        s3_client = boto3.client('s3') # Use standard client for uploads
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                
                # Construct the S3 key
                # This puts the file at: s3://bucket/prefix/filename.tiff
                s3_key = f"{s3_prefix}/{file}"
                
                print(f"Uploading {file} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(local_path, bucket_name, s3_key)

    # ... (Keep create_rugosity, create_slope, etc. exactly the same) ...