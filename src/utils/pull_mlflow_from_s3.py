# src/utils/pull_mlflow_from_s3.py
from __future__ import annotations
import shutil
import os
from pathlib import Path

from dotenv import load_dotenv
import boto3
from botocore.exceptions import BotoCoreError, NoCredentialsError, ClientError

load_dotenv()

BUCKET = os.getenv("MLFLOW_S3_BUCKET") or os.getenv("DATA_BUCKET")
PREFIX = os.getenv("MLFLOW_S3_PREFIX", "mlflow/mlruns").strip("/")
LOCAL_MLRUNS = Path(os.getenv("MLFLOW_LOCAL_DIR", "mlruns"))
LOCAL_DB = Path(os.getenv("MLFLOW_DB_PATH", "mlflow.db"))

DB_PREFIX = os.getenv("MLFLOW_S3_DB_PREFIX", "mlflow/meta").strip("/")
DB_KEY = f"{DB_PREFIX}/mlflow.db"

def require(cond: bool, msg: str):
    if not cond:
        raise SystemExit(f"[pull-mlflow] {msg}")

def clear_local(path: Path):
    if path.exists():
        print(f"[pull-mlflow] Deleting local: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def download_prefix(s3, bucket: str, prefix: str, dest_dir: Path):
    print(f"[pull-mlflow] Downloading s3://{bucket}/{prefix}/ -> {dest_dir.resolve()}")
    ensure_dir(dest_dir)
    paginator = s3.get_paginator("list_objects_v2")
    any_obj = False
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            any_obj = True
            rel = key[len(prefix) + 1 :]  # remove "prefix/"
            local_path = dest_dir / rel
            ensure_dir(local_path.parent)
            s3.download_file(bucket, key, str(local_path))
    if not any_obj:
        print(f"[pull-mlflow] No objects found under s3://{bucket}/{prefix}/ (nothing to download).")
    else:
        print("[pull-mlflow] Artifacts downloaded.")

def download_db(s3, bucket: str, key: str, dest: Path):
    print(f"[pull-mlflow] Downloading DB s3://{bucket}/{key} -> {dest.resolve()}")
    s3.download_file(bucket, key, str(dest))
    print("[pull-mlflow] DB downloaded.")

def main():
    require(BUCKET, "Set MLFLOW_S3_BUCKET (or DATA_BUCKET) in .env")

    try:
        session = boto3.session.Session()
        s3_client = session.client("s3")
    except (BotoCoreError, NoCredentialsError) as e:
        raise SystemExit(f"[pull-mlflow] AWS credentials/region error: {e}")

    # 1) Replace local mlruns with S3 snapshot
    clear_local(LOCAL_MLRUNS)
    try:
        download_prefix(s3_client, BUCKET, PREFIX, LOCAL_MLRUNS)
    except ClientError as e:
        raise SystemExit(f"[pull-mlflow] S3 error while downloading artifacts: {e}")

    # 2) Replace local DB with S3 copy
    clear_local(LOCAL_DB)
    try:
        download_db(s3_client, BUCKET, DB_KEY, LOCAL_DB)
    except ClientError as e:
        raise SystemExit(f"[pull-mlflow] S3 error while downloading DB: {e}")

    print("[pull-mlflow] Done.")

if __name__ == "__main__":
    main()