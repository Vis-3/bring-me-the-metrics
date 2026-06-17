"""Inspect a single Spotify Bronze record to debug popularity field."""
import gzip
import json
import boto3
from config import S3_BUCKET_NAME, AWS_REGION

s3 = boto3.client("s3", region_name=AWS_REGION)
key = "bronze/source=spotify/subgenre=deathcore/date=2026-06-17/artist_popularity.json.gz"
raw = gzip.decompress(s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)["Body"].read())
records = json.loads(raw)

# Show first record structure
bmth = next((r for r in records if "Bring Me" in r.get("lastfm_name", "")), records[0])
print("lastfm_name:", bmth.get("lastfm_name"))
print("spotify_match keys:", list(bmth.get("spotify_match", {}).keys()))
print("popularity:", bmth.get("spotify_match", {}).get("popularity"))
print("followers:", bmth.get("spotify_match", {}).get("followers"))
