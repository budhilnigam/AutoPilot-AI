"""
core/config.py — Centralised settings via pydantic-settings.

All values are read from environment variables (or .env file).
Access anywhere with: from autopilot_ai.core.config import settings
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore unknown env vars
    )

    # ── AWS ────────────────────────────────────────────────────────────────
    aws_region: str = Field(default="us-east-1")
    aws_access_key_id: str | None = Field(default=None)
    aws_secret_access_key: str | None = Field(default=None)
    aws_profile: str | None = Field(default=None)

    # ── Bedrock ────────────────────────────────────────────────────────────
    # Primary model: planning, reasoning, complex analysis
    bedrock_model_id: str = Field(
        default="anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    # Secondary model: fast, cheap tasks (data extraction, simple classification)
    bedrock_fast_model_id: str = Field(
        default="deepseek.v3-v1:0",
        alias="bedrock_haiku_model_id",
    )
    # Optional per-agent model overrides. If unset, falls back to BEDROCK_MODEL_ID.
    bedrock_model_observability_id: str | None = Field(default=None)
    bedrock_model_infra_id: str | None = Field(default=None)
    bedrock_model_db_id: str | None = Field(default=None)
    bedrock_model_cost_id: str | None = Field(default=None)
    bedrock_model_cicd_id: str | None = Field(default=None)
    # Tool generator disabled — use AWS SDK directly instead
    bedrock_model_planner_id: str | None = Field(default=None)
    bedrock_max_tokens: int = Field(default=4096)
    # Fast-path planner override (routing/direct answers). Falls back to BEDROCK_HAIKU_MODEL_ID.
    bedrock_model_planner_fast_id: str | None = Field(default=None)
    bedrock_max_tokens: int = Field(default=4096)
    bedrock_temperature: float = Field(default=0.1)  # Low for deterministic SRE analysis

    # ── Knowledge Base ─────────────────────────────────────────────────────
    knowledge_base_id: str | None = Field(default=None)
    s3_bucket_name: str | None = Field(default=None)
    kb_similarity_threshold: float = Field(default=0.6)

    # ── GitHub ─────────────────────────────────────────────────────────────
    github_token: str | None = Field(default=None)
    github_poll_interval_seconds: int = Field(default=300)  # 5 minutes

    # ── Cost / Currency ────────────────────────────────────────────────────
    currency: str = Field(default="INR")
    usd_to_inr_rate: float = Field(default=84.5)

    # ── Alerting ───────────────────────────────────────────────────────────
    alert_delivery_sla_seconds: int = Field(default=60)

    # ── Analysis Thresholds ────────────────────────────────────────────────
    anomaly_sigma_threshold: float = Field(default=2.0)
    build_regression_multiplier: float = Field(default=1.5)
    cost_overprovisioning_threshold: float = Field(default=0.30)
    redis_memory_pressure_threshold: float = Field(default=0.80)
    ci_failure_rate_threshold: float = Field(default=0.10)
    build_history_days: int = Field(default=30)

    # ── Retry / Circuit Breaker ────────────────────────────────────────────
    retry_max_attempts: int = Field(default=3)
    retry_min_wait_seconds: float = Field(default=2.0)
    retry_max_wait_seconds: float = Field(default=10.0)
    circuit_breaker_failure_threshold: int = Field(default=3)
    circuit_breaker_timeout_seconds: float = Field(default=60.0)

    # ── API Server ─────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    # ── Logging ────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")


    # ── Validators ─────────────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}, got {v!r}")
        return upper

    @field_validator("usd_to_inr_rate")
    @classmethod
    def validate_inr_rate(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("usd_to_inr_rate must be positive")
        return v

    @field_validator("kb_similarity_threshold", "anomaly_sigma_threshold")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Threshold must be positive")
        return v

    # ── Helpers ────────────────────────────────────────────────────────────
    def usd_to_inr(self, usd: float) -> float:
        """Convert USD amount to INR using configured rate."""
        return round(usd * self.usd_to_inr_rate, 2)

    def get_agent_model_id(self, agent_type: str, use_fast_path: bool = False) -> str:
        """
        Resolve the model ID for an agent.

        - Standard path: per-agent override -> BEDROCK_MODEL_ID
        - Planner fast path: BEDROCK_MODEL_PLANNER_FAST_ID -> BEDROCK_HAIKU_MODEL_ID
        """
        agent = agent_type.lower().strip()

        if agent == "planner" and use_fast_path:
            return self.bedrock_model_planner_fast_id or self.bedrock_fast_model_id

        overrides: dict[str, str | None] = {
            "observability": self.bedrock_model_observability_id,
            "infra": self.bedrock_model_infra_id,
            "db": self.bedrock_model_db_id,
            "cost": self.bedrock_model_cost_id,
            "cicd": self.bedrock_model_cicd_id,
            "planner": self.bedrock_model_planner_id,
        }
        return overrides.get(agent) or self.bedrock_model_id

    @property
    def bedrock_client_kwargs(self) -> dict:
        """boto3 client kwargs for Bedrock — uses profile or explicit keys."""
        kwargs: dict = {"region_name": self.aws_region}
        if self.aws_access_key_id and self.aws_secret_access_key:
            kwargs["aws_access_key_id"] = self.aws_access_key_id
            kwargs["aws_secret_access_key"] = self.aws_secret_access_key
        return kwargs

    @property
    def is_kb_enabled(self) -> bool:
        """True when both KB ID and S3 bucket are configured."""
        return bool(self.knowledge_base_id and self.s3_bucket_name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton. Called by all modules."""
    return Settings()


# Convenience: import this directly
settings: Settings = get_settings()
