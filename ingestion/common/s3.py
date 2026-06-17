"""
S3 utilities — all Bronze writes go through here.
Centralising S3 logic means ingestion scripts never import boto3 directly;
if we change compression or serialisation format, we fix it in one place.
"""
import gzip
import json
import boto3
from datetime import date
from config import S3_BUCKET_NAME, AWS_REGION, bronze_prefix


# Module-level client — boto3 clients are thread-safe and expensive to initialise,
# so we create one per process rather than one per request.
_s3 = boto3.client("s3", region_name=AWS_REGION)


def write_bronze(data: list[dict], source: str, subgenre: str, filename: str, run_date: date = None) -> str:
    """
    Gzip-compress and write raw API response to the Bronze partition.

    Returns the full S3 key so the caller can log exactly where data landed.
    data     — list of raw API response dicts, written as-is (no transformation)
    source   — 'spotify' or 'lastfm'
    subgenre — e.g. 'deathcore'
    filename — e.g. 'artists.json.gz' or 'tracks_page_1.json.gz'
    run_date — defaults to today; pass explicitly for backfill runs
    """
    date_str = (run_date or date.today()).isoformat()
    prefix = bronze_prefix(source, subgenre, date_str)
    key = f"{prefix}{filename}"

    # Serialise to JSON bytes then gzip — content stays valid JSON inside,
    # just compressed. Athena can read .json.gz natively without decompression step.
    raw_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
    compressed = gzip.compress(raw_bytes)

    _s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=compressed,
        ContentEncoding="gzip",
        ContentType="application/json",
    )

    print(f"  ✓ wrote s3://{S3_BUCKET_NAME}/{key} ({len(data)} records, {len(compressed):,} bytes)")
    return key
