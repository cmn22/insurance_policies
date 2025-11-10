# src/fetch_data.py
import os
import pathlib
from dotenv import load_dotenv

def main():
    # Load .env in runtime so `dvc repro` picks vars without extra wrappers
    load_dotenv()

    source = os.getenv("DATA_SOURCE", "s3")
    out = pathlib.Path("data/raw/insurance_dataset.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    if source == "s3":
        import boto3
        bucket = os.environ["DATA_BUCKET"]
        key = os.environ["DATA_KEY"]
        s3 = boto3.client("s3")
        s3.download_file(bucket, key, str(out))
        print(f"Saved: {out}")
    else:
        raise ValueError(f"Unsupported DATA_SOURCE={source}")

if __name__ == "__main__":
    main()