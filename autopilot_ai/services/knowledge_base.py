"""
services/knowledge_base.py — Bedrock Knowledge Base (RAG) wrapper.

Responsibilities:
  store_configuration(config)         Store a Configuration doc in S3 + kick off KB ingestion
  query_context(query, config_types)  Retrieve relevant docs (similarity > 0.6, Property 21)
  index_metrics(metric_data)          Store metric/billing data for RAG
  get_document(document_id)           Retrieve a stored document by ID (Property 20)

Architecture:
  - Raw documents are first written to S3  (integrations/aws/s3.py)
  - Bedrock Knowledge Base syncs from S3 via a DataSource + ingestion job
  - Queries go through bedrock-agent-runtime Retrieve API
  - similarity_threshold = Settings.kb_similarity_threshold (default 0.6)

AWS clients used:
  - bedrock-agent-runtime  (query / retrieve)
  - bedrock-agent          (ingestion job trigger)
  Both are separate from bedrock-runtime (used for model invocations).

Raises IntegrationError subclasses on failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from datetime import datetime
from functools import partial
from typing import Any

import boto3
from botocore.exceptions import ClientError

from autopilot_ai.core.config import settings
from autopilot_ai.core.exceptions import IntegrationError, ThrottlingError
from autopilot_ai.core.logging import get_logger
from autopilot_ai.core.retry import with_retry
from autopilot_ai.integrations.aws.s3 import s3_client
from autopilot_ai.models.domain import Configuration, ConfigType
from autopilot_ai.models.metrics import MetricData

logger = get_logger(__name__)

# Bedrock KB IDs are short alphanumeric strings (e.g. "AB12CD34EF").
_KB_ID_PATTERN = re.compile(r"^[0-9A-Za-z]{1,10}$")

# S3 key prefixes — KB data source must be configured to crawl these prefixes
_S3_PREFIX_CONFIGS = "kb/configurations/"
_S3_PREFIX_METRICS = "kb/metrics/"


class KnowledgeBaseRetrievalResult:
    """A single result returned by the KB Retrieve API."""

    def __init__(
        self,
        content: str,
        score: float,
        source_uri: str | None,
        metadata: dict[str, Any],
    ) -> None:
        self.content = content
        self.score = score
        self.source_uri = source_uri
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"KBResult(score={self.score:.3f}, uri={self.source_uri!r})"


def _make_agent_runtime_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client(
            "bedrock-agent-runtime", **kwargs
        )
    return boto3.client("bedrock-agent-runtime", **kwargs)


def _make_agent_client() -> Any:
    kwargs: dict[str, Any] = {"region_name": settings.aws_region}
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        kwargs["aws_access_key_id"] = settings.aws_access_key_id
        kwargs["aws_secret_access_key"] = settings.aws_secret_access_key
    elif settings.aws_profile:
        return boto3.Session(profile_name=settings.aws_profile).client(
            "bedrock-agent", **kwargs
        )
    return boto3.client("bedrock-agent", **kwargs)


class KnowledgeBaseService:
    """
    RAG service backed by Amazon Bedrock Knowledge Bases.

    The KB must already be provisioned (KB ID set in .env as KNOWLEDGE_BASE_ID)
    with an S3 data source pointed at settings.s3_bucket_name.

    If KNOWLEDGE_BASE_ID is not configured the service degrades gracefully:
      - store_configuration / index_metrics still write to S3
      - query_context returns an empty list with a warning
    """

    def __init__(self) -> None:
        self._runtime = _make_agent_runtime_client()
        self._agent = _make_agent_client()

    # ── Private sync helpers ───────────────────────────────────────────────

    def _retrieve_sync(
        self,
        query: str,
        kb_id: str,
        max_results: int,
        similarity_threshold: float,
    ) -> list[KnowledgeBaseRetrievalResult]:
        """
        Call Bedrock KB Retrieve API and filter by similarity score.
        Validates Property 21: only results with score > threshold are returned.
        """
        try:
            response = self._runtime.retrieve(
                knowledgeBaseId=kb_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {
                        "numberOfResults": max_results,
                        "overrideSearchType": "HYBRID",
                    }
                },
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(
                    f"Bedrock KB Retrieve throttled: {e}", kb_id=kb_id
                ) from e
            raise IntegrationError(
                f"Bedrock KB Retrieve failed: {e}", kb_id=kb_id
            ) from e

        results: list[KnowledgeBaseRetrievalResult] = []
        for item in response.get("retrievalResults", []):
            score = float(item.get("score", 0.0))
            if score < similarity_threshold:
                continue  # enforce Property 21

            content_obj = item.get("content", {})
            text = content_obj.get("text", "")

            location = item.get("location", {})
            s3_loc = location.get("s3Location", {})
            uri = s3_loc.get("uri")

            meta = item.get("metadata", {})

            results.append(
                KnowledgeBaseRetrievalResult(
                    content=text,
                    score=score,
                    source_uri=uri,
                    metadata=meta,
                )
            )

        # Sort by descending score
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _start_ingestion_job_sync(self, kb_id: str, data_source_id: str) -> str:
        """Trigger a KB ingestion job to sync newly uploaded S3 documents."""
        try:
            response = self._agent.start_ingestion_job(
                knowledgeBaseId=kb_id,
                dataSourceId=data_source_id,
            )
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if "Throttl" in code or "TooMany" in code:
                raise ThrottlingError(
                    f"Bedrock KB ingestion throttled: {e}", kb_id=kb_id
                ) from e
            # Ingestion jobs are best-effort — don't raise hard, log warning
            logger.warning(
                "kb_ingestion_job_failed",
                kb_id=kb_id,
                error=str(e),
            )
            return "failed"
        return response.get("ingestionJob", {}).get("ingestionJobId", "unknown")

    # ── Public async API ───────────────────────────────────────────────────

    async def store_configuration(
        self,
        config: Configuration,
        data_source_id: str | None = None,
    ) -> str:
        """
        Store a Configuration document in S3 and trigger KB ingestion.

        A SHA-256 checksum is embedded in S3 metadata to enable round-trip
        consistency validation (Property 20).

        Args:
            config:          The Configuration object to store.
            data_source_id:  KB DataSource ID; needed to trigger ingestion.
                             If omitted the doc is stored in S3 only.

        Returns:
            S3 URI of the stored document (e.g. "s3://bucket/kb/configs/...")
        """
        # Build a stable, deterministic S3 key
        content_hash = hashlib.sha256(config.content.encode()).hexdigest()[:12]
        key = (
            f"{_S3_PREFIX_CONFIGS}"
            f"{config.config_type.value}/"
            f"{content_hash}.txt"
        )

        metadata: dict[str, str] = {
            "config-type": config.config_type.value,
            "source-path": config.source_path[:512],  # S3 metadata max key length
            "stored-at": config.stored_at.isoformat(),
        }
        if config.commit_sha:
            metadata["commit-sha"] = config.commit_sha

        s3_uri = await s3_client.upload_text(key, config.content, metadata)

        logger.info(
            "kb_configuration_stored",
            config_type=config.config_type.value,
            s3_key=key,
            size=len(config.content),
        )

        # Trigger ingestion if KB is configured
        kb_id = settings.knowledge_base_id
        if kb_id and data_source_id:
            loop = asyncio.get_running_loop()
            job_id = await loop.run_in_executor(
                None,
                partial(self._start_ingestion_job_sync, kb_id, data_source_id),
            )
            logger.info("kb_ingestion_job_started", job_id=job_id, kb_id=kb_id)

        return s3_uri

    @with_retry(retry_on=(ThrottlingError,))
    async def query_context(
        self,
        query: str,
        config_types: list[ConfigType] | None = None,
        max_results: int = 5,
    ) -> list[KnowledgeBaseRetrievalResult]:
        """
        Retrieve relevant context documents for an agent query.

        Only results with similarity > settings.kb_similarity_threshold (0.6)
        are returned — validates Property 21.

        Args:
            query:        Natural-language query from the agent.
            config_types: Optional filter — only return docs of these types.
                          Filtering is done client-side after retrieval.
            max_results:  How many top results to fetch before filtering.

        Returns:
            List of KnowledgeBaseRetrievalResult sorted by score descending.
            Empty list if KB is not configured or no relevant docs exist.
        """
        kb_id = settings.knowledge_base_id
        if not kb_id:
            logger.warning("kb_not_configured", query=query[:80])
            return []
        if not _KB_ID_PATTERN.fullmatch(kb_id):
            logger.warning("kb_invalid_id", kb_id=kb_id, query=query[:80])
            return []

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                partial(
                    self._retrieve_sync,
                    query,
                    kb_id,
                    max_results,
                    settings.kb_similarity_threshold,
                ),
            )
        except IntegrationError as e:
            # KB retrieval should never block core agent analysis.
            logger.warning("kb_query_failed", kb_id=kb_id, error=str(e), query=query[:80])
            return []

        # Client-side config_type filter if requested
        if config_types:
            type_values = {ct.value for ct in config_types}
            results = [
                r
                for r in results
                if r.metadata.get("config-type") in type_values
            ]

        logger.debug(
            "kb_query_context",
            query=query[:80],
            results_count=len(results),
            threshold=settings.kb_similarity_threshold,
        )
        return results

    async def index_metrics(
        self,
        metric_data: list[MetricData],
        label: str = "",
    ) -> str:
        """
        Serialise metric data to JSON and store in S3 for KB indexing.

        Used to build up historical metric context that agents can retrieve
        (e.g. "what was CPU utilisation last week?").

        Args:
            metric_data: List of MetricData objects to store.
            label:       Optional human-readable label (e.g. "prod-ec2-cpu-2025-01").

        Returns:
            S3 URI of the stored metrics document.
        """
        payload = {
            "label": label,
            "stored_at": datetime.utcnow().isoformat(),
            "metrics": [
                {
                    "metric_name": m.metric_name,
                    "namespace": m.namespace,
                    "dimensions": m.dimensions,
                    "statistic": m.statistic,
                    "period_seconds": m.period_seconds,
                    "datapoints": [
                        {"timestamp": dp.timestamp.isoformat(), "value": dp.value}
                        for dp in m.datapoints
                    ],
                }
                for m in metric_data
            ],
        }

        content = json.dumps(payload, indent=2)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        uid = str(uuid.uuid4())[:8]
        key = f"{_S3_PREFIX_METRICS}{ts}-{uid}.json"

        s3_uri = await s3_client.upload_text(
            key,
            content,
            metadata={
                "data-type": "metrics",
                "label": label[:256] if label else "unlabelled",
                "metric-count": str(len(metric_data)),
            },
        )

        logger.info(
            "kb_metrics_indexed",
            metric_count=len(metric_data),
            s3_key=key,
        )
        return s3_uri

    async def get_document(self, s3_uri: str) -> str:
        """
        Retrieve a stored document by its S3 URI.

        Supports round-trip consistency validation (Property 20).
        The caller can re-compute the SHA-256 of the returned content and
        compare it against the checksum stored in S3 metadata by upload_text.

        Args:
            s3_uri: "s3://{bucket}/{key}" as returned by store_configuration.

        Returns:
            Raw text content of the document.
        """
        # Parse s3://bucket/key
        if not s3_uri.startswith("s3://"):
            raise IntegrationError(
                f"Invalid S3 URI: {s3_uri!r} — must start with 's3://'"
            )
        without_scheme = s3_uri[5:]
        bucket_end = without_scheme.find("/")
        if bucket_end == -1:
            raise IntegrationError(f"Cannot parse bucket from S3 URI: {s3_uri!r}")

        key = without_scheme[bucket_end + 1:]
        return await s3_client.download_text(key)

    async def list_stored_configs(
        self, config_type: ConfigType | None = None
    ) -> list[str]:
        """
        List all S3 keys stored under the KB configuration prefix.

        Args:
            config_type: Optional filter by configuration type.

        Returns:
            List of S3 keys.
        """
        prefix = _S3_PREFIX_CONFIGS
        if config_type:
            prefix = f"{_S3_PREFIX_CONFIGS}{config_type.value}/"
        return await s3_client.list_objects(prefix)


# Module-level singleton
knowledge_base = KnowledgeBaseService()
