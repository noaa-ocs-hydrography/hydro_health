import boto3

# def delete_s3_contents_excluding(bucket_name, exclude_prefix):
#     s3 = boto3.resource('s3')
#     bucket = s3.Bucket(bucket_name)
    
#     # Track items for summary
#     deleted_count = 0
#     skipped_count = 0

#     print(f"Starting deletion in bucket: {bucket_name}")
#     print(f"Excluding prefix: {exclude_prefix}\n")

#     # Iterate through all objects in the bucket
#     for obj in bucket.objects.all():
#         if obj.key.startswith(exclude_prefix):
#             print(f"Skipping: {obj.key}")
#             skipped_count += 1
#         else:
#             # Uncomment the line below to actually perform deletion
#             obj.delete() 
#             print(f"DELETED (Simulated): {obj.key}")
#             deleted_count += 1

#     print(f"\nFinished. Deleted: {deleted_count} | Skipped: {skipped_count}")

# # Configuration
# BUCKET = 'ocs-dev-csdl-hydrohealth'
# EXCLUDE = 'ER_3/' 

# delete_s3_contents_excluding(BUCKET, EXCLUDE)



import s3fs

fs = s3fs.S3FileSystem()

def move_s3_up_two_levels():
    """
    Moves a file from: s3://bucket/dir1/dir2/dir3/file.txt
    To:               s3://bucket/dir1/file.txt
    """

    folder = f's3://ocs-dev-csdl-hydrohealth/ER_3/model_variables/Prediction/raw/DigitalCoast/NOAA_NCEI_2022_8483/dem/NCEI_ninth_Topobathy_2014_8483'
    move_files = fs.glob(f'{folder}/**/*.tif')
    for file in move_files:
        # Remove 's3://' if present for easier splitting
        path_parts = file.split('/')

        # Extract filename and bucket/path components
        filename = path_parts[-1]
        # path_parts[:-3] removes the filename and the two immediate parent folders
        
        # Construct the new path
        new_path = f'ocs-dev-csdl-hydrohealth/ER_3/model_variables/Prediction/raw/DigitalCoast/NOAA_NCEI_0_8483/dem/NCEI_ninth_Topobathy_2014_8483/{filename}'
        
        new_path = f"s3://{new_path}"
        
        print(f"Moving: {file}")
        print(f"To:     {new_path}")
        
        # In S3, Move = Copy + Delete
        fs.cp(file, new_path)
        fs.rm(file)

# Example Usage:
BUCKET = 'ocs-dev-csdl-hydrohealth'
move_s3_up_two_levels()
