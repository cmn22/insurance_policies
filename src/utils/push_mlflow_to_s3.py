# src/utils/push_mlflow_to_s3.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

load_dotenv()  # Load AWS_* and MLFLOW_* from .env

# ---- Config (env-driven, with sensible defaults) ----
BUCKET = os.getenv("MLFLOW_S3_BUCKET") or os.getenv("DATA_BUCKET")
PREFIX = os.getenv("MLFLOW_S3_PREFIX", "mlflow/mlruns").strip("/")
LOCAL_MLRUNS = Path(os.getenv("MLFLOW_LOCAL_DIR", "mlruns"))
LOCAL_DB = Path(os.getenv("MLFLOW_DB_PATH", "mlflow.db"))

# where to keep the DB in S3 (separate from artifacts)
DB_PREFIX = os.getenv("MLFLOW_S3_DB_PREFIX", "mlflow/meta").strip("/")
DB_KEY = f"{DB_PREFIX}/mlflow.db"

def require(cond: bool, msg: str):
    if not cond:
        raise SystemExit(f"[push-mlflow] {msg}")

def iter_files(base: Path) -> Iterable[Path]:
    for p in base.rglob("*"):
        if p.is_file():
            yield p

def delete_prefix(s3, bucket: str, prefix: str):
    print(f"[push-mlflow] Deleting s3://{bucket}/{prefix}/ ...")
    to_delete = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            to_delete.append({"Key": obj["Key"]})
            if len(to_delete) >= 1000:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
                to_delete = []
    if to_delete:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": to_delete})
    print(f"[push-mlflow] Deleted existing objects under {prefix}/")

def upload_dir(s3, bucket: str, local_dir: Path, dest_prefix: str):
    base = local_dir.resolve()
    print(f"[push-mlflow] Uploading {base} -> s3://{bucket}/{dest_prefix}/")
    for f in iter_files(base):
        rel = f.relative_to(base).as_posix()
        key = f"{dest_prefix}/{rel}"
        s3.upload_file(str(f), bucket, key)
    print(f"[push-mlflow] Uploaded {local_dir} to s3://{bucket}/{dest_prefix}/")

def upload_db(s3, bucket: str, local_db: Path, key: str):
    print(f"[push-mlflow] Uploading DB {local_db} -> s3://{bucket}/{key}")
    s3.upload_file(str(local_db), bucket, key)
    print("[push-mlflow] DB uploaded.")

def main():
    require(BUCKET, "Set MLFLOW_S3_BUCKET (or DATA_BUCKET) in .env")
    require(LOCAL_MLRUNS.exists(), f"Local mlruns folder not found: {LOCAL_MLRUNS}")
    require(LOCAL_DB.exists(), f"Local mlflow db not found: {LOCAL_DB}")

    try:
        session = boto3.session.Session()
        s3_client = session.client("s3")
    except (BotoCoreError, NoCredentialsError) as e:
        raise SystemExit(f"[push-mlflow] AWS credentials/region error: {e}")

    # 1) Clear artifacts prefix, then upload mlruns/
    try:
        delete_prefix(s3_client, BUCKET, PREFIX)
        upload_dir(s3_client, BUCKET, LOCAL_MLRUNS, PREFIX)
    except ClientError as e:
        raise SystemExit(f"[push-mlflow] S3 error while uploading artifacts: {e}")

    # 2) Upload SQLite DB as a single object
    try:
        # Ensure metadata prefix exists by uploading directly (S3 is flat)
        upload_db(s3_client, BUCKET, LOCAL_DB, DB_KEY)
    except ClientError as e:
        raise SystemExit(f"[push-mlflow] S3 error while uploading DB: {e}")

    print("[push-mlflow] Done.")

if __name__ == "__main__":
    main()