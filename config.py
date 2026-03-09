"""
Centralized Configuration for AutoPilot AI

All configuration values should be defined here.
Load from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


# Load environment variables from project .env before reading any config values.
_ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(_ROOT_DIR / '.env')


def _parse_cors_origins(raw_origins: str) -> list[str]:
    """Parse comma-separated origins and normalize trailing slashes."""
    origins: list[str] = []
    for origin in raw_origins.split(','):
        normalized = origin.strip().rstrip('/')
        if normalized:
            origins.append(normalized)
    return origins


class Config:
    """Centralized configuration class"""
    
    # ========== General Settings ==========
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    ENVIRONMENT: str = os.getenv('ENVIRONMENT', 'development')  # development, staging, production
    
    # ========== AWS Settings ==========
    AWS_REGION: str = os.getenv('AWS_REGION', 'us-east-1')
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    # ========== Bedrock Settings ==========
    # Main orchestrator model
    BEDROCK_MODEL_ID: str = os.getenv('BEDROCK_MODEL_ID', 'openai.gpt-oss-20b-1:0')
    
    # Individual agent models (can be different for cost optimization)
    PLANNER_AGENT_MODEL_ID: str = os.getenv('PLANNER_AGENT_MODEL_ID', BEDROCK_MODEL_ID)
    OBSERVABILITY_AGENT_MODEL_ID: str = os.getenv('OBSERVABILITY_AGENT_MODEL_ID', BEDROCK_MODEL_ID)
    INFRA_AGENT_MODEL_ID: str = os.getenv('INFRA_AGENT_MODEL_ID', BEDROCK_MODEL_ID)
    DB_AGENT_MODEL_ID: str = os.getenv('DB_AGENT_MODEL_ID', BEDROCK_MODEL_ID)
    COST_AGENT_MODEL_ID: str = os.getenv('COST_AGENT_MODEL_ID', BEDROCK_MODEL_ID)
    CICD_AGENT_MODEL_ID: str = os.getenv('CICD_AGENT_MODEL_ID', BEDROCK_MODEL_ID)
    
    # Haiku for faster/cheaper operations
    BEDROCK_HAIKU_MODEL_ID: str = os.getenv('BEDROCK_HAIKU_MODEL_ID', 'openai.gpt-oss-20b-1:0')
    
    # ========== Knowledge Base Settings ==========
    USE_LOCAL_KB: bool = os.getenv('USE_LOCAL_KB', 'true').lower() == 'true'
    LOCAL_KB_PATH: str = os.getenv('LOCAL_KB_PATH', './local_kb')
    
    # AWS Knowledge Base (for production)
    S3_BUCKET_NAME: Optional[str] = os.getenv('S3_BUCKET_NAME')
    KNOWLEDGE_BASE_ID: Optional[str] = os.getenv('KNOWLEDGE_BASE_ID')
    
    # ========== GitHub Settings ==========
    GITHUB_TOKEN: Optional[str] = os.getenv('GITHUB_TOKEN')
    GITHUB_REPO_OWNER: Optional[str] = os.getenv('GITHUB_REPO_OWNER')
    GITHUB_REPO_NAME: Optional[str] = os.getenv('GITHUB_REPO_NAME')
    GITHUB_OAUTH_CLIENT_ID: Optional[str] = os.getenv('GITHUB_OAUTH_CLIENT_ID')
    GITHUB_OAUTH_CLIENT_SECRET: Optional[str] = os.getenv('GITHUB_OAUTH_CLIENT_SECRET')
    GITHUB_OAUTH_REDIRECT_URI: Optional[str] = os.getenv('GITHUB_OAUTH_REDIRECT_URI')
    
    # ========== CloudWatch Settings ==========
    CLOUDWATCH_LOG_GROUP: Optional[str] = os.getenv('CLOUDWATCH_LOG_GROUP')
    CLOUDWATCH_METRICS_NAMESPACE: str = os.getenv('CLOUDWATCH_METRICS_NAMESPACE', 'AutoPilotAI')
    
    # ========== Notification Settings ==========
    SNS_TOPIC_ARN: Optional[str] = os.getenv('SNS_TOPIC_ARN')
    ENABLE_NOTIFICATIONS: bool = os.getenv('ENABLE_NOTIFICATIONS', 'false').lower() == 'true'
    
    # ========== Cost Settings ==========
    USD_TO_INR_RATE: float = float(os.getenv('USD_TO_INR_RATE', '83.0'))
    
    # ========== Health Check Settings ==========
    HEALTH_CHECK_INTERVAL_MINUTES: int = int(os.getenv('HEALTH_CHECK_INTERVAL_MINUTES', '5'))
    HEALTH_CHECK_TIMEOUT_SECONDS: int = int(os.getenv('HEALTH_CHECK_TIMEOUT_SECONDS', '30'))
    
    # ========== Scheduler Settings ==========
    METRIC_CHECK_INTERVAL_MINUTES: int = int(os.getenv('METRIC_CHECK_INTERVAL_MINUTES', '5'))
    COST_CHECK_INTERVAL_HOURS: int = int(os.getenv('COST_CHECK_INTERVAL_HOURS', '24'))
    BUILD_CHECK_INTERVAL_MINUTES: int = int(os.getenv('BUILD_CHECK_INTERVAL_MINUTES', '15'))
    
    # ========== Alert Settings ==========
    ALERT_LATENCY_SECONDS: int = int(os.getenv('ALERT_LATENCY_SECONDS', '60'))
    ANOMALY_SIGMA_THRESHOLD: float = float(os.getenv('ANOMALY_SIGMA_THRESHOLD', '2.0'))
    
    # ========== API Settings ==========
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    ENABLE_CORS: bool = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
    CORS_ORIGINS: list[str] = _parse_cors_origins(
        os.getenv(
            'CORS_ORIGINS',
            'http://localhost:3000,http://localhost:5173,https://autopilot-ai-team-ragnar.netlify.app'
        )
    )
    # Allows all Netlify deploy URLs (production + preview) for development workflows.
    CORS_ORIGIN_REGEX: str = os.getenv('CORS_ORIGIN_REGEX', r'^https://.*\.netlify\.app$')
    
    # ========== Frontend Settings ==========
    FRONTEND_URL: str = os.getenv('FRONTEND_URL', 'http://localhost:5173,https://autopilot-ai-team-ragnar.netlify.app/')
    
    # ========== AWS SDK for Bedrock Agents ==========
    BEDROCK_AGENT_ID: Optional[str] = os.getenv('BEDROCK_AGENT_ID')
    BEDROCK_AGENT_ALIAS_ID: Optional[str] = os.getenv('BEDROCK_AGENT_ALIAS_ID')
    
    # ========== Retry and Timeout Settings ==========
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_MIN_WAIT_SECONDS: int = int(os.getenv('RETRY_MIN_WAIT_SECONDS', '2'))
    RETRY_MAX_WAIT_SECONDS: int = int(os.getenv('RETRY_MAX_WAIT_SECONDS', '10'))
    REQUEST_TIMEOUT_SECONDS: int = int(os.getenv('REQUEST_TIMEOUT_SECONDS', '300'))
    
    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration.
        
        Returns:
            True if configuration is valid
        """
        errors = []
        
        # Check required AWS credentials for production
        if cls.ENVIRONMENT == 'production':
            if not cls.AWS_ACCESS_KEY_ID:
                errors.append("AWS_ACCESS_KEY_ID is required in production")
            if not cls.AWS_SECRET_ACCESS_KEY:
                errors.append("AWS_SECRET_ACCESS_KEY is required in production")
            if not cls.USE_LOCAL_KB and not cls.KNOWLEDGE_BASE_ID:
                errors.append("KNOWLEDGE_BASE_ID is required when not using local KB")
        
        # Check model IDs are not empty
        if not cls.BEDROCK_MODEL_ID:
            errors.append("BEDROCK_MODEL_ID cannot be empty")
        
        # Check numeric values are sensible
        if cls.HEALTH_CHECK_INTERVAL_MINUTES < 1:
            errors.append("HEALTH_CHECK_INTERVAL_MINUTES must be >= 1")
        
        if errors:
            for error in errors:
                print(f"Configuration Error: {error}")
            return False
        
        return True
    
    @classmethod
    def get_summary(cls) -> dict:
        """
        Get configuration summary (safe for logging - no secrets).
        
        Returns:
            Dictionary with configuration summary
        """
        return {
            'environment': cls.ENVIRONMENT,
            'aws_region': cls.AWS_REGION,
            'use_local_kb': cls.USE_LOCAL_KB,
            'bedrock_model': cls.BEDROCK_MODEL_ID,
            'health_check_interval': cls.HEALTH_CHECK_INTERVAL_MINUTES,
            'api_port': cls.API_PORT,
            'cors_enabled': cls.ENABLE_CORS,
        }


# Create a singleton instance
config = Config()
