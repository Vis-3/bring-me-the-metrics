"""
Exports Gold mart tables from Athena to local CSV files for visualization.

    uv run python -m scripts.export_marts
"""
import boto3
import pandas as pd
import time
from config import AWS_REGION, ATHENA_S3_OUTPUT

MARTS = {
    "mart_subgenre_health":  "metal_intelligence_marts",
    "mart_album_legacy":     "metal_intelligence_marts",
    "mart_artist_features":  "metal_intelligence_marts",
}


def run_query(sql: str, database: str) -> pd.DataFrame:
    athena = boto3.client("athena", region_name=AWS_REGION)
    response = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database, "Catalog": "AwsDataCatalog"},
        ResultConfiguration={"OutputLocation": ATHENA_S3_OUTPUT},
    )
    query_id = response["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=query_id)
        state = status["QueryExecution"]["Status"]["State"]
        if state == "SUCCEEDED":
            break
        if state in ("FAILED", "CANCELLED"):
            raise RuntimeError(status["QueryExecution"]["Status"].get("StateChangeReason"))
        time.sleep(2)

    paginator = athena.get_paginator("get_query_results")
    rows, columns = [], None
    for page in paginator.paginate(QueryExecutionId=query_id):
        result = page["ResultSet"]
        if columns is None:
            columns = [c["Label"] for c in result["ResultSetMetadata"]["ColumnInfo"]]
        for row in result["Rows"][1 if columns and not rows else 0:]:
            rows.append([d.get("VarCharValue", None) for d in row["Data"]])

    return pd.DataFrame(rows, columns=columns)


def main():
    for table, database in MARTS.items():
        print(f"Exporting {table}...")
        df = run_query(f"SELECT * FROM {table}", database)
        path = f"data/{table}.csv"
        df.to_csv(path, index=False)
        print(f"  ✓ {len(df)} rows → {path}")


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    main()
