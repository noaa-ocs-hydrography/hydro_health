import boto3
import pathlib

from botocore.client import Config
from botocore import UNSIGNED


OUTPUTS = pathlib.Path(__file__).parents[1] / 'outputs'

# 1. using build_full_result and filter by Prefix
# bucket = "noaa-ocs-nationalbathymetry-pds"
# prefix = "BlueTopo"  # can also build prefix path with tile: BlueTopo/BH55Z4Z8
# creds = {
#     "aws_access_key_id": "",
#     "aws_secret_access_key": "",
#     "config": Config(signature_version=UNSIGNED),
# }
# client = boto3.client("s3",**creds)
# pageinator = client.get_paginator("list_objects_v2")
# objs = pageinator.paginate(Bucket=bucket, Prefix=prefix).build_full_result()
# for ob in objs['Contents']:
#     print(ob)

# print(len(objs))
# # print(objs.keys()) # 'Contents', 'RequestCharged', 'Prefix'



# 2. Using boto3.resource with filter.  This will only return 1000 rows
# each tile folder has 1 tiff and 1 xml file
bucket = "noaa-ocs-nationalbathymetry-pds"
creds = {
    "aws_access_key_id": "",
    "aws_secret_access_key": "",
    "config": Config(signature_version=UNSIGNED),
}
s3 = boto3.resource('s3', **creds)
nbs_bucket = s3.Bucket(bucket)
tile_id = 'BH4S2574'
for obj_summary in nbs_bucket.objects.filter(Prefix=f"BlueTopo/{tile_id}"):
    print(f'downloading: {obj_summary.key}')
    output_tile = OUTPUTS / obj_summary.key
    output_folder = output_tile.parents[0]
    output_folder.mkdir(parents=True, exist_ok=True)
    nbs_bucket.download_file(obj_summary.key, output_tile)
