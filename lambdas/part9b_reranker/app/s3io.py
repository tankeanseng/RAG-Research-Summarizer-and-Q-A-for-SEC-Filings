import json
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3")


def s3_get_json(bucket: str, key: str):
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except ClientError as e:
        raise RuntimeError(f"S3 get_object failed: s3://{bucket}/{key} :: {e}")


def s3_put_json(bucket: str, key: str, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            ContentType="application/json; charset=utf-8",
        )
    except ClientError as e:
        raise RuntimeError(f"S3 put_object failed: s3://{bucket}/{key} :: {e}")


def s3_list_keys(bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                k = obj.get("Key")
                if k:
                    keys.append(k)
        return keys
    except ClientError as e:
        raise RuntimeError(f"S3 list_objects_v2 failed: s3://{bucket}/{prefix} :: {e}")