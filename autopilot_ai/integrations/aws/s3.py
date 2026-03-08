"""integrations/aws/s3.py — S3 client for document storage using aws_api.

This implementation routes all boto3 usage through `autopilot_ai.integrations.aws.tool`.
The API surface mirrors the previous S3Client but avoids creating persistent
boto3 clients inside multiple modules.
"""

from __future__ import annotations

import hashlib
from typing import Any

from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import S3Error, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.integrations.aws.tool import aws_api

logger = get_logger(__name__)


class S3Client:
    def __init__(self, bucket: str | None = None) -> None:
        self._bucket = bucket or settings.s3_bucket_name

    def _require_bucket(self) -> str:
        if not self._bucket:
            raise S3Error("S3 bucket not configured. Set S3_BUCKET_NAME in .env")
        return self._bucket

    @with_retry(retry_on=(ThrottlingError,))
    async def upload_text(
        self, key: str, content: str, metadata: dict[str, str] | None = None
    ) -> str:
        body = content.encode("utf-8")
        checksum = hashlib.sha256(body).hexdigest()
        meta = {"checksum-sha256": checksum, **(metadata or {})}
        bucket = self._require_bucket()
        params = {
            "Bucket": bucket,
            "Key": key,
            "Body": body,
            "ContentType": "text/plain; charset=utf-8",
            "Metadata": meta,
            "ServerSideEncryption": "AES256",
        }
        try:
            await aws_api("s3", "put_object", params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on put: {e}", key=key) from e
            raise S3Error(f"S3 put_object failed: {e}", key=key, bucket=bucket) from e

        logger.debug("s3_upload_text", key=key, size_bytes=len(body))
        return f"s3://{bucket}/{key}"

    @with_retry(retry_on=(ThrottlingError,))
    async def download_text(self, key: str) -> str:
        bucket = self._require_bucket()
        try:
            resp = await aws_api("s3", "get_object", {"Bucket": bucket, "Key": key})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "NoSuchKey":
                raise S3Error(f"S3 key not found: {key}", key=key, bucket=bucket) from e
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on get: {e}", key=key) from e
            raise S3Error(f"S3 get_object failed: {e}", key=key, bucket=bucket) from e

        body = resp.get("Body")
        if isinstance(body, (bytes, bytearray)):
            return body.decode("utf-8")
        raise S3Error("Unexpected S3 get_object body type", key=key, bucket=bucket)

    @with_retry(retry_on=(ThrottlingError,))
    async def upload_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        bucket = self._require_bucket()
        params = {
            "Bucket": bucket,
            "Key": key,
            "Body": data,
            "ContentType": content_type,
            "Metadata": metadata or {},
            "ServerSideEncryption": "AES256",
        }
        try:
            await aws_api("s3", "put_object", params)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on put: {e}", key=key) from e
            raise S3Error(f"S3 put_object failed: {e}", key=key, bucket=bucket) from e
        return f"s3://{bucket}/{key}"

    @with_retry(retry_on=(ThrottlingError,))
    async def download_bytes(self, key: str) -> bytes:
        bucket = self._require_bucket()
        try:
            resp = await aws_api("s3", "get_object", {"Bucket": bucket, "Key": key})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "NoSuchKey":
                raise S3Error(f"S3 key not found: {key}", key=key, bucket=bucket) from e
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on get: {e}", key=key) from e
            raise S3Error(f"S3 get_object failed: {e}", key=key, bucket=bucket) from e

        body = resp.get("Body")
        if isinstance(body, (bytes, bytearray)):
            return bytes(body)
        raise S3Error("Unexpected S3 get_object body type", key=key, bucket=bucket)

    @with_retry(retry_on=(ThrottlingError,))
    async def list_objects(self, prefix: str = "") -> list[str]:
        bucket = self._require_bucket()
        keys: list[str] = []
        params = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        try:
            while True:
                resp = await aws_api("s3", "list_objects_v2", params)
                for obj in resp.get("Contents", []):
                    keys.append(obj["Key"])
                token = resp.get("NextContinuationToken")
                if not token:
                    break
                params["ContinuationToken"] = token
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on list: {e}", prefix=prefix) from e
            raise S3Error(f"S3 list_objects failed: {e}", prefix=prefix) from e
        return keys

    @with_retry(retry_on=(ThrottlingError,))
    async def delete_object(self, key: str) -> None:
        bucket = self._require_bucket()
        try:
            await aws_api("s3", "delete_object", {"Bucket": bucket, "Key": key})
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "SlowDown" in code:
                raise ThrottlingError(f"S3 throttled on delete: {e}", key=key) from e
            raise S3Error(f"S3 delete_object failed: {e}", key=key, bucket=bucket) from e
        logger.debug("s3_deleted", key=key)


# Module-level singleton
s3_client = S3Client()
