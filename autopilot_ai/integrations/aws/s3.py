"""
integrations/aws/s3.py — S3 client for document storage used by Knowledge Base.

Provides:
  upload_text(key, content, metadata)   -> str  (S3 URI)
  download_text(key)                    -> str
  upload_bytes(key, data, content_type) -> str  (S3 URI)
  download_bytes(key)                   -> bytes
  list_objects(prefix)                  -> list[str]  (keys)
  delete_object(key)                    -> None

Raises S3Error on API failures, ThrottlingError on throttling.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
from functools import partial
from typing import Any

import boto3
from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import S3Error, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry

logger = get_logger(__name__)


def _make_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client("s3", **kwargs)
    return boto3.client("s3", **kwargs)


class S3Client:
    """
    Async wrapper around boto3 S3 client.

    All operations target settings.s3_bucket_name unless an explicit
    bucket is provided.  Objects are stored with server-side encryption
    (AES-256) by default.
    """

    def __init__(self, bucket: str | None = None) -> None:
        self._client = _make_client()
        self._bucket = bucket or settings.s3_bucket_name

    def _require_bucket(self) -> str:
        if not self._bucket:
            raise S3Error(
                "S3 bucket not configured. Set S3_BUCKET_NAME in .env",
            )
        return self._bucket

    # ── Sync helpers ───────────────────────────────────────────────────────

    def _put_object_sync(
        self,
        key: str,
        body: bytes,
        content_type: str,
        metadata: dict[str, str],
    ) -> None:
        bucket = self._require_bucket()
        try:
            self._client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=metadata,
                ServerSideEncryption="AES256",
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on put: {e}", key=key) from e
            raise S3Error(f"S3 put_object failed: {e}", key=key, bucket=bucket) from e

    def _get_object_sync(self, key: str) -> bytes:
        bucket = self._require_bucket()
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            return response["Body"].read()
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code == "NoSuchKey":
                raise S3Error(
                    f"S3 key not found: {key}", key=key, bucket=bucket
                ) from e
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on get: {e}", key=key) from e
            raise S3Error(f"S3 get_object failed: {e}", key=key, bucket=bucket) from e

    def _delete_object_sync(self, key: str) -> None:
        bucket = self._require_bucket()
        try:
            self._client.delete_object(Bucket=bucket, Key=key)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "SlowDown" in code:
                raise ThrottlingError(f"S3 throttled on delete: {e}", key=key) from e
            raise S3Error(f"S3 delete_object failed: {e}", key=key, bucket=bucket) from e

    def _list_objects_sync(self, prefix: str) -> list[str]:
        bucket = self._require_bucket()
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "SlowDown" in code or "TooMany" in code:
                raise ThrottlingError(f"S3 throttled on list: {e}", prefix=prefix) from e
            raise S3Error(f"S3 list_objects failed: {e}", prefix=prefix) from e
        return keys

    # ── Public async API ───────────────────────────────────────────────────

    @with_retry(retry_on=(ThrottlingError,))
    async def upload_text(
        self,
        key: str,
        content: str,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """
        Upload a text document to S3.

        A SHA-256 checksum is computed and stored in object metadata for
        round-trip consistency validation (Property 20).

        Args:
            key:      S3 object key (e.g. "kb/configs/dockerfile-abc123.txt")
            content:  UTF-8 text content.
            metadata: Optional string key-value metadata stored with the object.

        Returns:
            S3 URI: "s3://{bucket}/{key}"
        """
        body = content.encode("utf-8")
        checksum = hashlib.sha256(body).hexdigest()
        meta = {"checksum-sha256": checksum, **(metadata or {})}

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._put_object_sync, key, body, "text/plain; charset=utf-8", meta),
        )
        bucket = self._require_bucket()
        logger.debug("s3_upload_text", key=key, size_bytes=len(body))
        return f"s3://{bucket}/{key}"

    @with_retry(retry_on=(ThrottlingError,))
    async def download_text(self, key: str) -> str:
        """
        Download a text document from S3.

        Returns:
            UTF-8 decoded string content.
        """
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            None, partial(self._get_object_sync, key)
        )
        return raw.decode("utf-8")

    @with_retry(retry_on=(ThrottlingError,))
    async def upload_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Upload raw bytes (e.g. CSV, binary data) to S3. Returns S3 URI."""
        meta = metadata or {}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            partial(self._put_object_sync, key, data, content_type, meta),
        )
        bucket = self._require_bucket()
        return f"s3://{bucket}/{key}"

    @with_retry(retry_on=(ThrottlingError,))
    async def download_bytes(self, key: str) -> bytes:
        """Download raw bytes from S3."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._get_object_sync, key)
        )

    @with_retry(retry_on=(ThrottlingError,))
    async def list_objects(self, prefix: str = "") -> list[str]:
        """
        List all object keys under a prefix (paginated).

        Returns:
            List of S3 keys.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, partial(self._list_objects_sync, prefix)
        )

    @with_retry(retry_on=(ThrottlingError,))
    async def delete_object(self, key: str) -> None:
        """Delete an object from S3."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, partial(self._delete_object_sync, key)
        )
        logger.debug("s3_deleted", key=key)


# Module-level singleton
s3_client = S3Client()
