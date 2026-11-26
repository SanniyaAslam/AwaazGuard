"""Audio storage: S3 bucket in production (S3_BUCKET set), local folder otherwise."""
from __future__ import annotations

import os
from pathlib import Path

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_PREFIX = os.getenv("S3_PREFIX", "uploads")
LOCAL_DIR = Path(os.getenv("AUDIO_STORE_DIR", "./audio_store"))


class Storage:
    def __init__(self) -> None:
        self.s3 = None
        if S3_BUCKET:
            import boto3

            self.s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION"))
        else:
            LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes) -> str:
        if self.s3:
            self.s3.put_object(Bucket=S3_BUCKET, Key=f"{S3_PREFIX}/{key}", Body=data,
                               ContentType="audio/wav")
            return f"s3://{S3_BUCKET}/{S3_PREFIX}/{key}"
        path = LOCAL_DIR / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def get(self, uri: str) -> bytes:
        if uri.startswith("s3://"):
            bucket, key = uri[5:].split("/", 1)
            return self.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return Path(uri).read_bytes()
