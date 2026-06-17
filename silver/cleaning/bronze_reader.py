"""
Reads raw Bronze JSON files from S3 and returns parsed Python objects.
All Bronze reads go through here — Silver never imports boto3 directly.
"""
import gzip
import json
import boto3
from datetime import date
from config import S3_BUCKET_NAME, AWS_REGION, bronze_prefix

_s3 = boto3.client("s3", region_name=AWS_REGION)


def _read_bronze_file(key: str) -> list[dict]:
    """Download and decompress a single Bronze file."""
    try:
        response = _s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        raw = gzip.decompress(response["Body"].read())
        return json.loads(raw)
    except _s3.exceptions.NoSuchKey:
        return []


def read_lastfm_tag_artists(subgenre: str, run_date: date) -> list[dict]:
    prefix = bronze_prefix("lastfm", subgenre, run_date.isoformat())
    return _read_bronze_file(f"{prefix}tag_artists.json.gz")


def read_lastfm_artist_info(subgenre: str, run_date: date) -> list[dict]:
    prefix = bronze_prefix("lastfm", subgenre, run_date.isoformat())
    return _read_bronze_file(f"{prefix}artist_info.json.gz")


def read_lastfm_weekly_charts(subgenre: str, run_date: date) -> list[dict]:
    prefix = bronze_prefix("lastfm", subgenre, run_date.isoformat())
    return _read_bronze_file(f"{prefix}weekly_charts.json.gz")


def read_musicbrainz_metadata(subgenre: str, run_date: date) -> list[dict]:
    prefix = bronze_prefix("musicbrainz", subgenre, run_date.isoformat())
    return _read_bronze_file(f"{prefix}artist_metadata.json.gz")


def read_musicbrainz_albums(run_date: date) -> list[dict]:
    prefix = bronze_prefix("musicbrainz", "betrayal_tracker", run_date.isoformat())
    return _read_bronze_file(f"{prefix}artist_albums.json.gz")
