import boto3
from config import S3_BUCKET_NAME

s3 = boto3.client("s3")
response = s3.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="bronze/source=lastfm/subgenre=deathcore/")
for obj in response["Contents"]:
    print(f"{obj['Key']}  ({obj['Size']:,} bytes)")
