"""
Creates the Glue database and Athena external tables for the Silver layer.
Run once before dbt run. Safe to re-run — drops and recreates tables.

    uv run python -m scripts.setup_athena
"""
import boto3
import time
from config import AWS_REGION, S3_BUCKET_NAME, ATHENA_S3_OUTPUT

GLUE_DATABASE = "metal_intelligence"
ATHENA_OUTPUT = ATHENA_S3_OUTPUT

athena = boto3.client("athena", region_name=AWS_REGION)
glue = boto3.client("glue", region_name=AWS_REGION)


def run_query(sql: str, description: str) -> None:
    print(f"  {description}...")
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": GLUE_DATABASE, "Catalog": "AwsDataCatalog"},
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
    )
    query_id = response["QueryExecutionId"]

    # Poll until complete
    while True:
        status = athena.get_query_execution(QueryExecutionId=query_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)

    if state != "SUCCEEDED":
        reason = status["QueryExecution"]["Status"].get("StateChangeReason", "unknown")
        raise RuntimeError(f"Query failed ({state}): {reason}\nSQL: {sql}")

    print(f"  ✓ {description}")


def create_database() -> None:
    print("\n── Creating Glue database ──")
    try:
        glue.create_database(
            DatabaseInput={"Name": GLUE_DATABASE, "Description": "Metal Intelligence Pipeline — Gold layer"}
        )
        print(f"  ✓ Created database: {GLUE_DATABASE}")
    except glue.exceptions.AlreadyExistsException:
        print(f"  ✓ Database already exists: {GLUE_DATABASE}")


def create_artists_table() -> None:
    print("\n── Creating external table: artists ──")
    run_query("DROP TABLE IF EXISTS artists", "drop existing")
    run_query(f"""
        CREATE EXTERNAL TABLE artists (
            lastfm_name         STRING,
            mbid                STRING,
            mb_name             STRING,
            listeners           BIGINT,
            play_count          BIGINT,
            source_tag          STRING,
            subgenre_tags       ARRAY<STRUCT<tag:STRING, valid_from:STRING, valid_to:STRING>>,
            country             STRING,
            formed_year         DOUBLE,
            bio_summary         STRING,
            mb_resolution_score DOUBLE
        )
        PARTITIONED BY (subgenre STRING, date STRING)
        STORED AS PARQUET
        LOCATION 's3://{S3_BUCKET_NAME}/silver/table=artists/'
        TBLPROPERTIES ('parquet.compress'='SNAPPY')
    """, "create artists table")
    run_query("MSCK REPAIR TABLE artists", "discover partitions")


def create_albums_table() -> None:
    print("\n── Creating external table: albums ──")
    run_query("DROP TABLE IF EXISTS albums", "drop existing")
    run_query(f"""
        CREATE EXTERNAL TABLE albums (
            artist_name         STRING,
            artist_mbid         STRING,
            album_id            STRING,
            title               STRING,
            type                STRING,
            first_release_date  STRING,
            release_year        BIGINT
        )
        PARTITIONED BY (subgenre STRING, date STRING)
        STORED AS PARQUET
        LOCATION 's3://{S3_BUCKET_NAME}/silver/table=albums/'
        TBLPROPERTIES ('parquet.compress'='SNAPPY')
    """, "create albums table")
    run_query("MSCK REPAIR TABLE albums", "discover partitions")


def main():
    print("Setting up Athena external tables for Silver layer...")
    create_database()
    create_artists_table()
    create_albums_table()
    print("\n✓ Athena setup complete. Run: uv run dbt run --profiles-dir ..")


if __name__ == "__main__":
    main()
