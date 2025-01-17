import boto3
from botocore.client import Config
from botocore import UNSIGNED


bucket = 'noaa-ocs-nationalbathymetry-pds'
prefix = ''
cred = {
    "aws_access_key_id": "",
    "aws_secret_access_key": "",
    "config": Config(signature_version=UNSIGNED),
}

client = boto3.client("s3", **cred)
pageinator = client.get_paginator("list_objects_v2")
objs = pageinator.paginate(Bucket=bucket, Prefix=prefix).build_full_result()
print(objs)