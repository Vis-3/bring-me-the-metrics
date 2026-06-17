"""
Writes unified artist records to S3 Silver as Parquet.
All Silver writes go through here.
"""
import io
import boto3
import pandas as pd
from datetime import date
from config import S3_BUCKET_NAME, AWS_REGION, silver_prefix

_s3 = boto3.client("s3", region_name=AWS_REGION)


def write_silver_artists(records: list[dict], subgenre: str, run_date: date) -> str:
    """
    Serialize unified artist records to Parquet and write to S3 Silver.

    subgenre_tags is a list of dicts — Parquet handles nested structs natively
    via PyArrow, which pandas uses under the hood with engine='pyarrow'.
    """
    if not records:
        print(f"  ⚠ no records to write for {subgenre}")
        return ""

    df = pd.DataFrame(records)
    prefix = silver_prefix("artists", subgenre)
    key = f"{prefix}date={run_date.isoformat()}/artists.parquet"

    # Write to in-memory buffer then upload — avoids temp files on disk
    buffer = io.BytesIO()
    df.to_parquet(buffer, engine="pyarrow", index=False, compression="snappy")
    parquet_bytes = buffer.getvalue()
    buffer.seek(0)

    _s3.put_object(
        Bucket=S3_BUCKET_NAME,
        Key=key,
        Body=parquet_bytes,
        ContentType="application/octet-stream",
    )

    print(f"  ✓ wrote s3://{S3_BUCKET_NAME}/{key} ({len(records)} records, {len(parquet_bytes):,} bytes)")
    return key
